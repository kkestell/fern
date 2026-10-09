# Pass floating-point values across the external interface

## Plan

`agents/plans/2026-10-09-006-floating-point-external-interface.md`

## Summary

External functions accept `f32` and `f64` parameters and results, matching C
`float` and `double`. The spec, semantic check, and IR verifier changed; the
backend needed nothing. The plan's goal is met.

## Decisions

- `sum10` weights each argument by its position, so a misordered or misplaced
  stack argument changes the result.
- Checked off the todo item in `agents/todo.md`.

## Checks run

- `cargo fmt --check` — Passed.
- `cargo clippy --all-targets -- -D warnings` — Passed.
- `cargo test` — Passed, including `c_abi_floating_point`.

## Manual verification

1. The native float test links a C helper and libm and exits successfully.

   ```sh
   cargo test --test core c_abi_floating_point
   ```

   `test c_abi_floating_point ... ok`.
