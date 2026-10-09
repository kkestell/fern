# Check `new`, `free`, nullable pointers, and pointer comparison

## Plan

`agents/plans/2026-10-08-007-check-new-free-nullable-pointers-and-pointer-comparison.md`

## Summary

Semantic checking now accepts every pointer form in the spec. `null` takes
its type from context, `*T` widens to `nullable *T`, and `new`, `free`,
`*T(p)`, and pointer equality are checked. No pointer syntax reports a
temporary diagnostic during checking. The driver guard also rejects `new`.
The plan's goal is met.

## Departures from the plan

- `AnnotationKind`'s `Free::operand` lost its
  `cfg_attr(not(test), expect(dead_code, …))` attribute, because checking now
  reads it.
- A shared `null_without_context` builds the no-context diagnostic, used by
  `infer_expression`, `check_null`, and `common_element_type`.
- `common_element_type` reports the no-context diagnostic itself when every
  element is `null`, at the first element. That is the same diagnostic and
  span the plan expects from `check_null`.

## Decisions

- The nullable-pointer tests share `NULLABLE_PRELUDE` and `nullable_body`,
  which declare `p: *int`, `q: nullable *int`, and `r: *u8`.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 333 unit and 59 integration tests pass.

## Manual verification

1. `null` without context is rejected.

   ```sh
   printf 'fn main() -> void { var p = null; }\n' > /tmp/null.fern
   cargo run -q -- /tmp/null.fern -o /tmp/null
   ```

   Reports `` `null` requires a nullable pointer type from context `` at
   `null`.

2. A program using `new` and pointer equality checks and stops at the driver
   guard.

   ```sh
   printf 'fn main() -> void { var a = new(1); const c = a; if a == c { exit(0); } }\n' > /tmp/new.fern
   cargo run -q -- /tmp/new.fern -o /tmp/new
   ```

   Reports `pointers are not yet implemented` at `new(1)`.

## Follow-up work

- Lowering receives pointer expressions whose checked type is the
  `nullable *T` their context expects while the binding, field, or call they
  read holds a `*T`. The IR must treat the two as one representation.
