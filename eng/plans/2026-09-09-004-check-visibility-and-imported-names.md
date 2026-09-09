# Check visibility and imported names

## Sources

- `docs/spec.md#public-declarations` — `pub` accessibility and assignment to a
  public `const`
- `docs/spec.md#use-declarations` — file-local imports, the collision rule, the
  must-be-referenced rule, and shadowing
- `docs/spec.md#module-directories` — one namespace per module directory
- `docs/spec.md#functions` — entry-point selection and a dependency's `main`
- `eng/roadmap.md#modules-and-imports` — fourth task and its boundary
- `src/semantic.rs`, `src/module.rs` — checking and the loaded module graph

## Goal

Check every loaded module, resolving qualified and imported names against the
importing file's `use` declarations and rejecting private access, colliding
imported names, and unreferenced imports. Starts from the completed
import-resolution task and deletes its two guards in `semantic::check`. No
`ir.rs` or `backend.rs` change belongs to this task: lowering already keys
functions and bindings by arena index and emits index-based symbols. Cross-module
IR verification, initialization-order tests, the `examples/` entry, and the
milestone fixture stay with the following two tasks.

## Implementation

- `src/module.rs` — record what each import resolved to, so checking does not
  re-resolve paths.
  - Add `Program.imports: Vec<Vec<usize>>`: for each source file, the module
    index each of its `use` declarations resolves to, in declaration order.
  - Give `ImportPath` a `file: usize`, push an empty resolution vector per file
    in `Loader::open`, and append the resolved module index in `Loader::run` for
    every branch that resolves an import (cached, cycle-free, and newly opened).
  - Add `#[cfg(test)] pub(crate) fn single(syntax: &Syntax) -> (Vec<Module>,
    Vec<Vec<usize>>)`: one root module over every file, with no imports.
  - Drop the `expect(dead_code)` attributes that now have readers:
    `Module::path`, `Module::files`, and `Program::modules`.

- `src/frontend.rs` — drop the `expect(dead_code)` attributes on
  `TopLevelItem::public` and `Import::selection`.

- `src/semantic.rs` — `check<'a>(syntax: &'a Syntax, modules: &[Module],
  imports: &[Vec<usize>]) -> Result<CheckedProgram<'a>, Diagnostic>`. Delete the
  `use` and qualified-name guards and `first_qualified_name`. Add
  `#[cfg(test)] pub(crate) fn check_root(syntax: &Syntax)` wrapping
  `module::single`, and update the six `crate::semantic::check` call sites in
  `frontend`, `ir`, and `backend` tests to it.

- `src/semantic.rs` — new resolution state, all copies of the specification's
  file-local rule in one place:
  - `struct Declaration { public: bool, kind: DeclarationKind }` and
    `enum DeclarationKind { Binding(Idx<Binding>), Function(Idx<Function>) }`,
    both `Copy`. A module namespace is `HashMap<Spur, Declaration>`.
  - `enum Imported { Module(HashMap<Spur, Declaration>),
    Declaration(DeclarationKind) }` and
    `struct FileImports { names: HashMap<Spur, Imported>, introduced: Vec<(Spur,
    Range<usize>)>, used: HashSet<Spur> }`.
  - On `CheckedProgram`, add non-`pub` checking state: `imports:
    Vec<FileImports>` indexed by file and `file: usize`, the file whose
    declarations are being checked. Document both as checking state, like
    `function_names`.

- `src/semantic.rs` — turn the body of `check` into a loop over `modules` in
  order, so dependencies are checked before dependents and `module_bindings`
  accumulates dependency-first. Per module, over the items of its own
  `Module::files` range only:
  1. Collect module-level names, rejecting duplicates; collect
     `function_names`, replacing the previous module's; select `main` and apply
     the no-parameter and `void` result checks only when `module.path` is empty,
     so a dependency's `main` is an ordinary function.
  2. Allocate module bindings into `module_scope` and collect signatures into
     `checked.functions`, as today.
  3. Build the module's namespace from its items and each item's `public` flag,
     keeping it in a `Vec` indexed by module index for later importers.
  4. For each file in the range, build its `FileImports` from
     `syntax.files[file].imports` and `imports[file]`. A whole-module import
     introduces the path's last component bound to a clone of that module's
     namespace; a selective import introduces each selected name bound to its
     declaration, rejecting a name the module does not declare and one that is
     not `pub`. Reject an introduced name already in the module's namespace or
     already in this file's `names`.
  5. Check initializers and function bodies as today, setting
     `checked.file` before each item so resolution sees that file's imports.
     Module-level bindings need their statement's file, so carry
     `(file, Idx<Statement>)` and look the file up when checking the ordered
     initializers.
  6. Report the first name in each file's `introduced` that is not in `used`,
     in declaration order, at its introduced span.
  - `main` stays an `Option` until every module is checked; the missing-`main`
    diagnostic is unchanged.

- `src/semantic.rs` — extend resolution. `resolve`, `resolve_call`, and
  `assignment_target` take `&mut self` so they can record a used import.
  - Unqualified: the existing scope chain first, then `self.imports[self.file]`.
    An `Imported::Declaration` of the wanted kind resolves and marks its name
    used; an `Imported::Module` is an error (``module `text` is not a value``,
    or ``module `text` is not a function`` at a call target); nothing found
    keeps today's `unknown binding` and `unknown function` diagnostics.
  - Qualified: a qualifier held by a lexical scope is an error (``cannot use
    binding `text` as a module``, at the qualifier's span). Otherwise the
    qualifier must be an `Imported::Module` in this file, marking it used;
    an `Imported::Declaration` gives ```step` is not a module`` and a missing
    entry ``unknown module `text```. Inside that namespace, a missing name gives
    ``module `text` has no declaration named `doubled```, a non-`pub` one
    ``declaration `origin` is private to module `counter```, and a kind mismatch
    ``cannot call non-function declaration `value``` or ``cannot use function
    `doubled` as a value``. Assignment keeps the existing immutability
    diagnostic, which is what rejects assigning to a public `const`.

- `src/semantic.rs` — in `collect_references`, skip a `QualifiedName` that has a
  qualifier. A qualified name never refers to the module's own declarations, so
  matching it against them would invent an initializer dependency.

- `src/lib.rs` — pass `program.modules` and `program.imports` to
  `semantic::check`.

## Tests

Multi-module semantic tests load a real tree: reuse `module.rs`'s `tree` helper
by making it `#[cfg(test)] pub(crate)`, call `module::load`, then `check` its
`syntax`, `modules`, and `imports`. Assert diagnostic messages, and spans where
the offending text is unambiguous.

- A public function and a public `var` are reachable through a whole-module
  import and through a selective import; the same names in a private
  declaration are rejected from another module and still work between two files
  of one module.
- Assigning to an imported public `var` succeeds; assigning to an imported
  public `const` reports the immutability diagnostic.
- A module-level initializer reads an imported public `const`.
- An introduced name colliding with a module-level declaration, with another
  selective import, and with a repeat of the same whole-module import are each
  rejected.
- An unreferenced whole-module import and an unreferenced selective import are
  each rejected; a name imported in one file and used only in another file of
  the same module reports the unreferenced import.
- A local binding shadows a selectively imported name, and a local binding named
  like an imported module rejects a qualified use inside its scope.
- An unknown qualifier, a selective import used as a qualifier, an imported
  module used as a value and as a call target, and a qualified function used as
  a value are each rejected.
- The root module's `main` is the entry point while a dependency's `main`, with
  parameters and a value result, checks as an ordinary function; a root module
  with no `main` is rejected even when a dependency has one.
- Replace the guard test `imports_and_qualified_names_are_rejected_until_they_resolve`.
- `tests/compiler.rs` — rewrite the two `IMPORTING_MODULE` tests: make
  `app/main.fern` exit with `text::width` and assert the exit status showing
  which root resolved the import, and keep the `FERNPATH`-replacement case
  asserting the unresolved-import failure.

## Decisions

- A local binding shadows an imported module name, so a qualified use of that
  name inside the binding's scope is rejected rather than reaching the module.
  The specification's shadowing sentence covers every imported name, and a
  whole-module import introduces the module name.

- Each file's `FileImports` clones the namespace of every module it imports
  whole, rather than holding a module index and threading the namespace list
  through resolution. Namespaces are small, and resolution then reads one
  self-contained value.

- Modules are checked one at a time in `Program::modules` order rather than
  collecting every module's signatures first. The dependency graph is acyclic
  and dependencies come first, so an imported declaration is always already
  checked, and `function_names` and `module_scope` stay one module's namespace.
