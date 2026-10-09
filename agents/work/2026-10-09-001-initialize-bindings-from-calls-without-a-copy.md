# Initialize bindings from calls without a copy

## Plan

`agents/plans/2026-10-09-001-initialize-bindings-from-calls-without-a-copy.md`

## Summary

`Instruction::Call` has a `destination`. `lower_flow_binding` sends an
unaddressed aggregate binding initialized from a call to `lower_call_into`,
which writes the result straight into the binding's storage. That storage is
a local, or `Place::Result` for a returned local. Verification checks
destinations, and the emitter passes the destination's address as the hidden
result argument. The plan's goal is met.

## Departures from the plan

- Destination checks live in a separate `Program::verify_call_destination`,
  called from `verify_call` when a destination is present.
- `emit_call` takes the caller `function` so it can resolve the destination
  with `place_address`.
- The verify test does not build a separate scalar-returning callee. A
  destination whose type differs from the callee's result goes through the
  same check, which the `[2]int`-into-`int` case covers.

## Decisions

- Six IR snapshots gained `destination: None` on every call. The only real
  change is in `structs`, where `var moved = shifted(cell);` writes into its
  local with no `CallResult` and no store. The diffs were reviewed before
  they were accepted.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 370 unit and 62 integration tests pass.

## Manual verification

1. `const b = make(20);` passes `%local0` as the result address and copies
   nothing: `a_binding_initialized_from_a_call_is_written_by_the_callee`.

   ```sh
   cargo test --lib a_binding_initialized_from_a_call_is_written_by_the_callee
   ```

   Passes.
