# Compile and execute multi-module programs

## Sources

- `eng/roadmap.md#modules-and-imports` — the sixth task, the milestone program,
  and the completion gates
- `docs/spec.md#module-resolution-and-dependencies` — one initialization per
  dependency module, dependencies before dependents, all before the root
  module's `main`
- `docs/spec.md#functions` — the root module's sole `main` is the entry point; a
  dependency's `main` is an ordinary function
- `src/backend.rs` — `function_symbol`, `place`, `emit_conditional_trap`
- `src/ir.rs` — `lowered_tree` test helper to mirror
- `tests/compiler.rs` — `tree`, `module`, `documented_examples_compile`

## Goal

Prove that a multi-module program compiles to a native executable and runs, add
the milestone's `examples/` entry, and add the completion fixture. This is the
milestone's completion boundary, so it ends with the repository's broad gates
and a pass over the roadmap's gates.

Emission needs no source change. `FunctionId` and `GlobalId` are already
program-wide, so `fn{id}`, `$global{id}`, and
`fern_function{index}_operation{id}_{cause}_message` cannot collide across
modules; only the root `main` is exported; and module-level initializers are
constant expressions, so every global is static data and no run-time
initialization order exists to get wrong. Confirm each of those in tests rather
than changing `src/backend.rs`.

## Implementation

- `examples/modules_and_imports/` — a directory example, the first one in the
  tour. Root module `app`, with dependency modules beside it so the default
  search root resolves them.
  - `app/main.fern` — whole-module `use`, a qualified call, assignment to
    another module's `pub var`, and `exit` with 42.
  - `app/report.fern` — a second file of the root module: a selective `use` and
    a reference to a root-module declaration from the other file.
  - `counter/counter.fern` — `pub var`, `pub fn`, a private `const`, and a
    nested-path `use` of `text::format`.
  - `counter/limits.fern` — a second file of `counter`: a `pub const` whose
    initializer reads the private `const` in `counter.fern`.
  - `text/format/format.fern` — a nested module with one `pub fn`.
  - Follow the existing examples' style: no comments, exit status 42.

- `README.md` — add a `Modules and imports` bullet to the tour list, linking
  `examples/modules_and_imports/`, naming module directories, `pub`,
  whole-module, nested-path, and selective imports, and qualified references.

- `tests/fixtures/programs/modules_and_imports/` — the roadmap's milestone
  program verbatim, with `app` as its root module. It is compiled in place from
  `CARGO_MANIFEST_DIR`, not copied, with the output in a temporary directory.

- `src/backend.rs` — in `tests`, add `lowered_tree` beside `lowered`, mirroring
  the helper in `src/ir.rs`: `module::tree`, `module::load` with the temporary
  directory as the only search root and `app` as the root module, then
  `semantic::check`, `lower`, and `verify`.

- `tests/compiler.rs` — extend `documented_examples_compile` to also compile
  `examples/modules_and_imports/app`, keeping the existing flat-file loop.

## Tests

In `src/backend.rs`, over `emit` text for a module tree:

- A module tree emits exactly one `export function`, and it is `$main`. A
  dependency module's `main` is emitted as a non-exported `$fn{id}`.
- Same-named declarations in two modules get distinct symbols: two `pub fn`s
  named alike emit distinct `$fn{id}` definitions, two `pub var`s named alike
  emit distinct `$global{id}` data definitions, and a trap in each of two
  same-named functions emits distinct message symbols.

In `tests/compiler.rs`, executing programs built with `tree`:

- A qualified cross-module call and a selectively imported one both return the
  callee's value, and assigning to another module's `pub var` from the root
  module is observable in a later cross-module read.
- Same-named `pub fn`s and `pub var`s in two dependency modules are callable and
  mutable independently, and the exit status distinguishes them.
- A dependency's `main` does not run: the root module's `main` exits with a
  value a dependency `main` would have overwritten.
- Every module's bindings hold their initial values in `main`'s first statement,
  including a dependency binding a second dependency also reads.
- `tests/fixtures/programs/modules_and_imports/app` compiles and exits 42.

## Extra validation

- Run `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and
  `cargo test`; this task integrates the milestone.
- Walk the roadmap's completion gates and confirm each has a test. Earlier tasks
  cover namespace sharing, visibility, resolution, `FERNPATH`, cycles,
  file-local imports, and IR initialization order; this task covers native
  symbols, execution, and the example's exit status.
