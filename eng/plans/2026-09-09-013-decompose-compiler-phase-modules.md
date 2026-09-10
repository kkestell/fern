# Decompose compiler phase modules

## Sources

- `AGENTS.md#project-priorities` — one implementation path per behavior and
  shared ownership for duplicated rules
- `docs/spec.md` — behavior must remain unchanged
- `src/lib.rs` — the existing frontend, semantic, IR, and backend phase
  boundaries remain the compiler pipeline

## Goal

Replace the four oversized phase files with small phase facades and
responsibility-based submodules. Preserve every crate-visible API, diagnostic,
snapshot content, emitted QBE program, and native result; this task changes
ownership and visibility only.

## Implementation

- Move one phase at a time in pipeline order. After each phase compiles and its
  focused tests pass, remove the moved definitions from the old file. Do not
  retain forwarding implementations or duplicate helpers. Keep each existing
  phase name as the facade used by `src/lib.rs` and downstream phases.

- `src/frontend.rs` — keep only submodule declarations and the current
  crate-visible exports. Move token recognition, nested-comment scanning, and
  integer-token validation to `src/frontend/lexer.rs`; syntax nodes, type
  annotations, operators, arenas, and source spans to
  `src/frontend/syntax.rs`; and `Parser`, nesting enforcement, `parse_file`,
  and the test-only `parse` entry point to `src/frontend/parser.rs`. Keep
  precedence and syntax construction in the parser rather than copying operator
  knowledge into the lexer or syntax model.

- `src/semantic.rs` — make this the facade for `check`, `check_root`, and the
  checked-program model. Move checked values, bindings, signatures, and program
  storage to `src/semantic/model.rs`; module namespaces, imports, declaration
  ordering, and entry selection to `src/semantic/namespaces.rs`; annotation
  resolution, array lengths, and element-type rules to
  `src/semantic/annotations.rs`; statement, scope, control-flow, call, and
  assignment checking to `src/semantic/statements.rs`; expression inference,
  contextualization, indexing, lengths, and array literals to
  `src/semantic/expressions.rs`; and constant folding, integer range
  operations, shifts, conversions, and comparisons to
  `src/semantic/constants.rs`. Do not introduce a second context object or
  split `CheckedProgram` state between owners.

- `src/ir.rs` — keep the `lower` entry point and IR type exports in the facade.
  Move IR IDs, operands, places, values, instructions, blocks, functions, and
  programs to `src/ir/model.rs`; all structural, typing, reachability, and
  definite-initialization verification to `src/ir/verify.rs`; and lowering,
  `FlowBuilder`, held operands/places, and loop targets to `src/ir/lower.rs`.
  Verification must remain the sole gate that constructs `VerifiedProgram`.

- `src/backend.rs` — keep `emit` and `build` as the facade API. Move QBE
  program, function, block, instruction, place, aggregate, and data emission to
  `src/backend/emitter.rs`; integer operation, conversion, overflow, shift, and
  trap emission to `src/backend/integer.rs`; and temporary-file/native-tool
  orchestration to `src/backend/toolchain.rs`. Keep one `Emitter` shared by the
  two emission modules and keep native process execution out of them.

- Widen field visibility only where a submodule genuinely reads it. The
  checked-program model, IR model, and `Emitter` are defined in one submodule
  and used from its siblings, so their fields become `pub(super)` while the
  types keep their current crate visibility. Cross-submodule helper functions
  and methods also take `pub(super)`. Nothing gains `pub(crate)` that is not
  crate-visible today.

- Move each phase's unit tests into its module directory. Use a `tests/mod.rs`
  for shared fixture helpers and group cases by the production owner above.
  Do not move unit behavior into `tests/compiler.rs`; that file remains the
  end-to-end compiler and CLI boundary.

- Snapshot files follow their tests. Insta derives both the file name and the
  directory from the test's module path, so moving `frontend::tests` to
  `frontend::parser::tests` renames
  `src/snapshots/fern__frontend__tests__*.snap` to
  `src/frontend/snapshots/fern__frontend__parser__tests__*.snap`, and the IR
  snapshots move to `src/ir/snapshots/` the same way. Rename the files with
  `git mv` in the same commit as the test move, keeping their contents byte for
  byte, and update the `src/snapshots/` entry in the `AGENTS.md` codebase map
  to name the per-phase snapshot directories.

## Tests

- Run each moved phase's unit-test target immediately after that phase moves.
  Snapshot contents must be unchanged: after the `git mv`, `cargo insta test`
  reports no pending snapshots, and `git diff -M --stat` shows the snapshot
  files as pure renames.
- Compare QBE emitted for the existing backend fixtures before and after the
  backend move, including trap messages and symbol names.
- Run the complete compiler and CLI integration suite after all facades are in
  place, including the array fixture under `tests/fixtures/programs/`.

## Decisions

- Module boundaries follow compiler responsibilities, not a line-count quota.
  Tests move with their owner, so extracting tests alone cannot satisfy the
  task.
- Facades preserve current paths such as `frontend::Syntax`,
  `semantic::CheckedProgram`, and `ir::Program`; downstream callers do not learn
  the internal file layout.
- Annotation resolution owns its own submodule rather than sitting with module
  namespaces. It resolves element types and evaluates array lengths through
  constant folding, so it belongs between the namespace and constant owners
  instead of inside either.
- The semantic namespace submodule is `namespaces.rs`, because `src/module.rs`
  already owns module discovery, import loading, and the dependency graph over
  source files.
- Snapshots keep insta's default layout next to their tests, so no
  `snapshot_path` configuration is added.
