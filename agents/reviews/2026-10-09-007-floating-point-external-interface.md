# Review: pass floating-point values across the external interface

## Scope and coverage

Reviewed the staged changes for
`agents/plans/2026-10-09-006-floating-point-external-interface.md`: the
`docs/spec.md` external-interface rule, the semantic check in
`src/semantic/namespaces.rs`, the IR verifier in `src/ir/verify.rs`, the
semantic and native tests, and `agents/todo.md`. Traced external call emission
through `abi_class`, `value_class`, and `qbe_type` in `src/backend/qbe.rs` and
`src/backend/emitter.rs`, and the existing IR verifier tests in
`src/ir/tests/mod.rs`.

## Fixed

- **IR verifier test no longer covers the external type rule**
  (`src/ir/tests/mod.rs:43`): The test set an external result to `f64` and
  expected rejection. With `f64` now allowed, it still passed only because the
  call site's `i32` result no longer matched, so a regression in the verifier's
  external type rule would go unnoticed. The test now drops the call, accepts an
  `f64` result, and rejects a `bool` result; disabling the rule makes it fail.

## Findings

No open findings.

## Checks run

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed, including `c_abi_floating_point`.
- Probed the original IR test case: it failed with the call result type
  mismatch (`i32` against `f64`), not the external declaration error.
- Disabled the verifier's numeric check: the fixed test fails.

## Verdict

The change meets the plan. One test-coverage gap was fixed; nothing remains
open.
