# Evaluate the `len` operand without a copy

## Plan

`agents/plans/2026-10-09-003-evaluate-the-len-operand-without-a-copy.md`

## Summary

`len` now lowers its operand through `lower_aggregate_effects`, which runs the
operand's effects without loading the array. The program from the bug exits 0,
and the bug entry is removed from `agents/bugs.md`. The plan's goal is met.

## Departures from the plan

- `len_does_not_load_its_operand` takes the array as a parameter of `count`
  instead of a local initialized from a literal: the literal's initialization
  loads the constant array, which is an array-typed `Load` unrelated to `len`.
- `len_runs_the_effects_of_its_operand` writes the dereference as `*(at())`:
  `*at()` parses as a pointer conversion to a type named `at`.

## Checks run

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed.
- With `src/ir/lower.rs` reverted, `len_does_not_load_its_operand` and
  `len_of_an_array_larger_than_the_stack_executes` fail;
  `len_runs_the_effects_of_its_operand` passes both before and after, since it
  guards effects the old lowering already ran.
