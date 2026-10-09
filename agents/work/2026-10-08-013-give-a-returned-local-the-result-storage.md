# Give a returned local the result storage

## Plan

`agents/plans/2026-10-08-013-give-a-returned-local-the-result-storage.md`

## Summary

`returned_local` finds the aggregate local every `return` returns, and
`binding_place` gives it `Place::Result` in place of a fresh local. Each
`return` of it is a `Load(Result)`, which the emitter already returns without
a copy. The plan's goal is met.

## Departures from the plan

- A helper, `binding_place`, chooses the declared binding's storage for both
  the zeroed and the initialized path of `lower_flow_binding`.
- The statement walk is split into `collect_returns` and
  `collect_statement_returns`, and the returned reference is resolved by
  `returned_binding`.

## Decisions

- Three existing expectations changed because their returned locals now live
  in the result storage:
  - the `arrays` IR snapshot, where `totals`'s `out` lives in `Result` (diff
    reviewed);
  - `stack_storage_excludes_borrowed_parameters_and_forwarded_results`, where
    `build` now needs 8 bytes instead of 16008;
  - `a_returned_literal_is_built_in_the_result_storage`, whose non-literal
    case now returns a parameter, since a returned local no longer stays a
    local.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 366 unit and 62 integration tests pass.

## Manual verification

1. A returned local's QBE writes through `%result` and copies nothing:
   `a_returned_local_is_built_where_the_caller_reads_it`.

   ```sh
   cargo test --lib a_returned_local_is_built_where_the_caller_reads_it
   ```

   Passes.
