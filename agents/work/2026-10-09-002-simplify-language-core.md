# Simplify Fern's language core

## Plan

`agents/plans/2026-10-09-002-simplify-language-core.md`

## Summary

Implemented the plan in the existing frontend, semantic, IR, and QBE pipeline.
The language contract is in `docs/spec.md`. Removed obsolete representations,
checks, runtime support, dependencies, fixtures, and snapshots. Updated the
example tour and replaced obsolete tests with focused phase and native tests.

Literal parsing still uses bounded big integers; each integer operation folds
at its concrete machine width. The arbitrary rational dependency is removed.

## Checks run

- `cargo fmt` and `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — 86 tests passed: 46 unit tests, 26 compiler integration tests,
  and 14 core integration tests. Native tests invoke QBE and the host C toolchain.
  The complete example tour executes successfully.

Native coverage includes integer widths, IEEE values and conversions, raw
pointers, aggregate copying and equality, evaluation order, mutable parameter
copies, external libc calls, small-integer C ABI and C struct memory layout.
