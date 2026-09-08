# Fern Roadmap

Fern is a compiler written in Rust. The initial stack is Logos for lexing,
la-arena for syntax storage, lasso for identifier interning, Ariadne for
diagnostic rendering, and Insta for syntax and IR snapshots. The parser is
handwritten, and native compilation uses a Fern-owned IR and QBE.

Every milestone implements the corresponding sections of
[the specification](../docs/spec.md). The milestone gates below prove its
integrated outcome. Task checkboxes record progress; milestones remain in
implementation order.

Do not add incremental queries, lossless syntax, control-flow infrastructure, or
alternate backends before a milestone needs them.

---

## Module-level declarations

Support `var` and `const` declarations at the top level of a source file against
[Module-level declarations](../docs/spec.md#module-level-declarations),
[Scope and shadowing](../docs/spec.md#scope-and-shadowing),
[Integer constant expressions](../docs/spec.md#integer-constant-expressions),
and [Assignment](../docs/spec.md#assignment).

The scope is file-scope bindings whose initializers are constant expressions,
their visibility independent of declaration order, and their initialization
before `main` runs. Multiple source files, imports, and non-constant
initializers remain outside this milestone.

### Example

```fern
var counter = start;
const start: int = 40;
const step = 2;

fn main() -> void {
    counter = counter + step;
    exit(counter); // reports 42
}
```

Expected exit status: 42. Future file:
`examples/module_level_declarations.fern`.

### Tasks

- [ ] **Parse top-level bindings**

  - Accept `var` and `const` declarations interleaved with `fn` declarations,
    preserving spans, and reject statements at the top level.
  - Cover ordering, annotations, and malformed top-level source in parser tests
    and snapshots.

- [ ] **Resolve and check module-level bindings**

  - Resolve references to module-level bindings from any declaration in the
    file, regardless of order, and from every function body.
  - Diagnose duplicate module-level names, initializer cycles, and initializers
    that are not constant expressions.
  - Preserve local shadowing of module-level bindings and permit assignment to a
    module-level `var` from a function body.

- [ ] **Lower and execute module-level bindings**

  - Initialize module-level bindings before `main` runs and preserve their
    values and mutations through IR verification, snapshots, and native
    execution.
  - Add the milestone example and validate the integrated milestone.

### Completion gates

- A module-level binding referenced before its own declaration compiles and
  executes; a duplicate name, an initializer cycle, and a non-constant
  initializer each produce a source diagnostic.
- A local binding shadows a module-level binding without modifying it, and an
  assignment in a function body updates the module-level `var`.
- The example compiles and exits 42.

---

## Branches and loops

Add branches, loops, comparisons, and boolean values after their control-flow
rules are specified. This milestone owns the implementation of
[Comparison](../docs/spec.md#comparison). Define this milestone's tasks with
kroadmap once those language decisions are settled.

---

## Parameters, calls, and return values

Add parameters, calls, and return values after the remaining function rules
are specified. Define this milestone's tasks with kroadmap once those language
decisions are settled.

---

## Modules and imports

Add module directories and `use` declarations after the open module questions in
[the specification](../docs/spec.md#open-module-questions) are settled. Define
this milestone's tasks with kroadmap once those language decisions are settled.
