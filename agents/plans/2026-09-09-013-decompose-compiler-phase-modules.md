# Decompose compiler phase modules

## Sources

- `AGENTS.md#project-priorities` — one implementation path per behavior and
  shared ownership for duplicated rules
- `docs/spec.md` — behavior must remain unchanged
- `src/lib.rs` — the existing frontend, semantic, IR, and backend phase
  boundaries remain the compiler pipeline

## Goal

Replace the four oversized phase files with module roots and
responsibility-based submodules. Preserve every diagnostic, snapshot content,
emitted QBE program, and native result; this task changes ownership and
visibility only.

## Implementation

- Move one phase at a time in pipeline order. After each phase compiles and its
  focused tests pass, remove the moved definitions from the old file. Do not
  retain forwarding implementations, re-exports, or duplicate helpers. Keep
  each existing phase name as a module root, and make callers name the owning
  submodule directly.

- `src/frontend/mod.rs` — keep only submodule declarations. Move token
  recognition, nested-comment scanning, and integer-token validation to
  `src/frontend/lexer.rs`; syntax nodes, type annotations, arenas, source
  spans, and expression traversal to `src/frontend/syntax.rs`; and `Parser`,
  nesting enforcement, `parse_file`, and the test-only `parse` entry point to
  `src/frontend/parser.rs`. Keep precedence, infix classification, and syntax
  construction in the parser rather than copying operator knowledge into the
  lexer or syntax model. The operator enums live in `src/types.rs` beside the
  type enum, because every phase through the backend names them.

- `src/semantic/mod.rs` — keep only submodule declarations. Move checked
  values, bindings, signatures, and program storage to
  `src/semantic/model.rs`; module namespaces, imports, declaration ordering,
  and entry selection to `src/semantic/namespaces.rs`; annotation
  resolution, array lengths, and element-type rules to
  `src/semantic/annotations.rs`; statement, scope, control-flow, call, and
  assignment checking, including the structural termination and break
  analysis a function result requires, to `src/semantic/statements.rs`;
  expression inference, contextualization, indexing, lengths, and array
  literals to
  `src/semantic/expressions.rs`; and constant folding, integer range
  operations, shifts, conversions, and comparisons to
  `src/semantic/constants.rs`. Do not introduce a second context object or
  split `CheckedProgram` state between owners.

- `src/ir/mod.rs` — keep only submodule declarations. Move IR IDs, operands,
  places, values, instructions, blocks, functions, and programs to
  `src/ir/model.rs`; all structural, typing, reachability, and
  definite-initialization verification to `src/ir/verify.rs`; and lowering,
  `FlowBuilder`, held operands/places, and loop targets to `src/ir/lower.rs`.
  `VerifiedProgram` lives in `src/ir/verify.rs` with a private field, so
  verification is the sole gate that constructs one.

- `src/backend/mod.rs` — keep only submodule declarations. Move the shared
  `Emitter` state and the operand and type spellings to `src/backend/qbe.rs`;
  QBE program, function, block, instruction, place, aggregate, and data
  emission to `src/backend/emitter.rs`; integer operation, conversion,
  overflow, shift, and trap emission to `src/backend/integer.rs`; and
  temporary-file/native-tool orchestration to `src/backend/toolchain.rs`. Both
  emission modules depend on `qbe.rs` and never on each other, and native
  process execution stays out of them.

- Expose only responsibility submodules reached across a phase boundary so
  callers use the owning path directly. Widen field visibility only where a
  sibling genuinely reads it. The
  checked-program model, IR model, and `Emitter` are defined in one submodule
  and used from its siblings, so their fields become `pub(super)` while the
  types keep their current crate visibility. Cross-submodule helper functions
  and methods also take `pub(super)`.

- Move each phase's unit tests into its module directory. Use a `tests/mod.rs`
  for shared fixture helpers and group cases by the production owner above.
  Do not move unit behavior into `tests/compiler.rs`; that file remains the
  end-to-end compiler and CLI boundary. Every production submodule with unit
  tests has one test file named for it, so lowering and verification each own
  their own.

- Snapshot files follow their tests. Insta derives both the file name and the
  directory from the test's module path, so the parser cases in
  `frontend::tests::parser` own
  `src/frontend/tests/snapshots/fern__frontend__tests__parser__*.snap` and the
  lowering cases in `ir::tests::lower` own
  `src/ir/tests/snapshots/fern__ir__tests__lower__*.snap`. Rename the files
  with `git mv` in the same commit as the test move, keeping their contents
  byte for byte, and update the `src/snapshots/` entry in the `AGENTS.md`
  codebase map to name the per-phase snapshot directories.

## Tests

- Run each moved phase's unit-test target immediately after that phase moves.
  Snapshot contents must be unchanged: after the `git mv`, `cargo insta test`
  reports no pending snapshots, and `git diff -M --stat` shows the snapshot
  files as pure renames.
- Compare QBE emitted for the existing backend fixtures before and after the
  backend move, including trap messages and symbol names.
- Run the complete compiler and CLI integration suite after the module split,
  including the array fixture under `tests/fixtures/programs/`.
