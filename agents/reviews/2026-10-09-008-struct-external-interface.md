# Review: pass structs by value across the external interface

## Scope and coverage

Reviewed the staged changes for
`agents/plans/2026-10-09-007-struct-external-interface.md`: the
`docs/spec.md` external-interface rule, `Type::crosses_external_interface` in
`src/types.rs`, the semantic check in `src/semantic/namespaces.rs`, the
`StructFields` implementation in `src/ir/model.rs` and its use in
`src/ir/verify.rs`, the aggregate type definitions and external call emission
in `src/backend/emitter.rs` and `src/backend/qbe.rs`, and the semantic, IR,
backend, and native tests. Traced `emit_call` through the destination,
discarded-result, `%v{id}_storage`, and `%result` cases.

The native tests run on arm64 Apple only. The emitted QBE was also assembled
for `amd64_apple` and run under Rosetta. No riscv64 target was checked.

## Fixed

None.

## Findings

No open findings.

## Checks run

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed, including `c_abi_structs`.
- Compiled and ran Fern programs against C helpers covering: a discarded
  memory-class struct result (`Big`, 48 bytes); a million external struct calls
  in a loop, to confirm QBE's result storage does not grow the stack; an
  argument whose storage a later argument mutates, matching a Fern callee; a
  struct assigned from a call that takes it; float and double homogeneous
  aggregates with array fields, a three-`double` aggregate, a mixed
  `float`/`double` struct, and a `u8`/`[3]u8`/`u16` struct; a 300-field
  struct. All matched C.
- Emitted QBE for those programs and for `c_abi_structs`, assembled it with
  `qbe -t amd64_apple` and `qbe -t arm64_apple`, linked with the C helpers,
  and ran it; every program exited 0 on both targets.
- Checked the struct rule against a struct declared after the external
  function, a nested `bool` field, and a struct imported from another module.
- Made the shared rule accept every struct: the IR verifier test fails.

## Verdict

The change meets the plan. No findings.
