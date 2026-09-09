# Represent multi-module programs in Fern IR

## Sources

- `docs/spec.md#module-resolution-and-dependencies` — one inclusion and one
  initialization per dependency module, dependencies before dependents, all
  before the root module's `main`
- `docs/spec.md#module-level-declarations` — module-level initializers are
  constant expressions
- `eng/roadmap.md#modules-and-imports` — fifth task and its boundary
- `src/ir.rs` — `lower`, `function_id`, `Program`, `Global`
- `src/module.rs` — `Program::modules`, dependencies before dependents

## Goal

Make Fern IR's handling of a multi-module program a stated, tested property
rather than an accident. Starts from the completed visibility task, where
checking already produces one `CheckedProgram` over every loaded module. The IR
gains no module concept and no new lowering path: identity, cross-module
lowering, and initialization already fall out of the one-arena syntax and the
dependency-first module order. This task proves that in IR tests and snapshots
and removes the loader field that was reserved for it. Native emission, the
`examples/` entry, and the milestone fixture stay with the last task.

## Implementation

- `src/ir.rs` — state the identity rules where they are relied on. Nothing
  changes in `Program`, `Global`, `verify`, or any lowering function.
  - `function_id`: say that `syntax.functions` holds every function of every
    loaded module, so a function's arena index is its program-wide `FunctionId`
    and two modules cannot claim one ID.
  - `lower`: say that `checked.module_bindings` covers every module's bindings
    in dependency-first order, so each module-level `var` becomes exactly one
    `Global` and a dependency's globals precede its dependents'. Note that
    constant initializers make globals static data, so nothing initializes them
    at run time and no ordering code is needed.
  - `Global`: name the whole program, not one module, as the source of globals.

- `src/module.rs` — delete `Module::dependencies`, `Frame::dependencies`, and
  `Loader::depend_on`, with the two `depend_on` calls in `Loader::run` and
  `Loader::complete_top` and the `dependencies: Vec::new()` in `single`. Drop
  the assertions on the field in the loader tests; `labels` already asserts the
  module order those tests care about.

- `src/module.rs` — correct `Module::label`'s `expect(dead_code)` reason: its
  reader is the loader tests, not cycle diagnostics, which use the free `label`.

## Tests

In `src/ir.rs`, add a `lowered_tree(files)` test helper beside `lowered`:
`module::tree`, then `module::load` with the temporary directory as the only
search root and `app` as the root module, then `semantic::check`, `lower`, and
`verify`. Assert on IR shape, not on rendered output.

- A qualified call, a selectively imported call, a cross-module read, and a
  cross-module assignment lower to the same `Call`, `Load`, and `Store` forms as
  their single-module equivalents, against the callee's `FunctionId` and the
  binding's `GlobalId`.
- Every function of every module is lowered once with a distinct `FunctionId`,
  including a private function in a dependency that nothing calls.
- `Program::main` is the root module's `main`, and a dependency module's `main`
  is lowered as an ordinary function under a different `FunctionId`.
- A `pub var` in a module that two other modules both read becomes one `Global`
  that both reads and writes address.
- A dependency's globals come before the root module's, and each holds its
  constant initial value; a dependency's `pub const` produces no global and
  folds into its use site in another module.
- Snapshot the milestone example's module tree as `modules`, so cross-module
  IDs, globals, and spans are reviewable in one place.

## Decisions

- `Module::dependencies` is deleted rather than read here. Lowering needs
  dependency-first order, which `Program::modules` order already gives it, not
  the edges; keeping the edge lists would be data with no reader.
