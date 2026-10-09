# Call variadic external functions

## Plan

`agents/plans/2026-10-09-008-variadic-external-calls.md`

## Summary

An external declaration can end its parameter list with `...`, and calls pass
extra numeric and pointer arguments with the host C ABI's variadic
conventions. Programs can call `printf` and `snprintf`. The plan's goal is met.

## Departures from the plan

- The syntax snapshot is unchanged. It records statements, expressions, and
  annotations, not functions, so the new `variadic` field never reaches it.
  `variadic_parameter_lists` in `src/frontend/tests/mod.rs` checks the field
  instead.
- `typed_names` now returns the span of a `...` that ends a parameter list
  after a comma, alongside the names. When it only stopped there, `parameters`
  could not tell `f(x:int, ...)` from `f(x:int ...)`.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — all pass.

## Manual verification

1. A Fern program calls `printf` with a mix of argument types.

   ```fern
   extern fn printf(format: *u8, ...) -> i32;
   fn main() -> void {
       printf("%d %s %.1f %ld %c\n", i32(42), "apples", 2.5, 7, 'z');
       printf("none\n");
       var small: i8 = -3;
       printf("%d\n", small);
   }
   ```

   ```sh
   cargo run -- main.fern -o prog && ./prog
   ```

   It printed `42 apples 2.5 7 z`, `none`, and `-3`.

2. `c_variadic_calls` detects a wrong value. I changed the C helper's expected
   `u8` value from `200` to `201` and reverted it afterward.

   ```sh
   cargo test --test core c_variadic_calls
   ```

   The test failed with `exit status: 12`.

## Follow-up work

- Struct extra arguments stay outside the core until QBE passes them
  correctly after `...` on arm64 Apple.
