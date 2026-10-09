# Build returned aggregate literals in the result storage

## Plan

`agents/plans/2026-10-08-012-build-returned-aggregate-literals-in-the-result-storage.md`

## Summary

`Place::Result` names a function's aggregate result storage. A `return` of an
unfolded array or struct literal stores the literal there and returns
`Load(Result)`. The emitter returns that load without storage or a copy,
through `returned_in_place`. The plan's goal is met.

## Departures from the plan

- A small `store_literal` helper chooses between `store_elements` and
  `store_fields` for the returned literal.
- `Places::borrowable` treats a `Result` root as not borrowable, and
  `hold_place` holds it as a plain root. Both are exhaustive matches the new
  place reaches, though lowering never borrows or holds result storage today.

## Decisions

- The `copied` case in the lowering test still builds `[x, x]` in a temporary
  before initializing `local`. That copy belongs to the later todo items.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 362 unit and 62 integration tests pass. The `structs` IR
  snapshot changed to store `shifted`'s literal into `Result` with no
  temporary local. The diff was reviewed before it was accepted.

## Manual verification

1. A returned literal's QBE writes through `%result` and copies nothing:
   `returned_literals_are_built_where_the_caller_reads_them`.

   ```sh
   cargo test --lib returned_literals_are_built_where_the_caller_reads_them
   ```

   Passes.
