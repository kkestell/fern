# Parse pointer syntax

## Plan

`agents/plans/2026-10-08-004-parse-pointer-syntax.md`

## Summary

Every pointer form in `docs/spec.md#pointers-and-heap-memory` now parses into
frontend syntax. `nullable`, `new`, `free`, and `null` are reserved, and
semantic checking rejects each form with `pointers are not yet implemented`.
The plan's goal is met.

## Departures from the plan

- `free` without `(` reports `expected `(`` at the next token, the same as
  `exit`, rather than a reserved-word error at `free`. `free` is a statement
  keyword like `exit`, and only expression keywords (`len`, `new`) report the
  reserved-word error.

## Decisions

- `docs/spec.md#nullable-pointers` now states the `*T(p)` parsing rule the
  plan depends on, written through `kspec` before planning.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 303 unit and 59 integration tests pass.

## Manual verification

1. The CLI reports the temporary guard at the pointer expression.

   ```sh
   mkdir -p target/manual
   printf 'fn main() -> void {\n    const p = new(1);\n    exit(0);\n}\n' > target/manual/pointer.fern
   cargo run -q -- target/manual/pointer.fern -o target/manual/pointer
   ```

   Observed `pointers are not yet implemented` at `2:15`, underlining
   `new(1)`, and exit status 1.

## Follow-up work

- The todo item "Represent pointer types across the compiler" lists rejecting
  `*void`. The parser already rejects it with `expected a type`, so that task
  needs no separate check.
