# Pass structs by value across the external interface

## Plan

`agents/plans/2026-10-09-007-struct-external-interface.md`

## Summary

External functions now take and return structs by value with the host C ABI,
through QBE aggregate types. A struct that stores `bool` anywhere is rejected.
The plan's goal is met.

## Departures from the plan

- `emit_call` also leaves the leading `l {storage}` argument off external
  calls. Fern callees take their result storage as a hidden first parameter;
  a C callee does not.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — all pass.

## Manual verification

1. `c_abi_structs` detects a wrong result. I changed the expected `spill`
   result from `-258.0` to `-257.0` and reverted it afterward.

   ```sh
   cargo test --test core c_abi_structs
   ```

   The test failed with `exit status: 7`.

2. Before implementing, I compiled hand-written QBE against a C helper on
   arm64 Apple to confirm that QBE passes and returns aggregates the way C
   does: a mixed `{ b, d }` result, a float pair `{ s, s }` argument and result,
   a 160-byte `{ w 40 }` argument and result, and a nested
   `{ b, :in 2, s }` argument and result. All matched C.

## Follow-up work

- The native tests run only on the host target, so this machine checked
  arm64 Apple only. No other QBE target was checked against C.
