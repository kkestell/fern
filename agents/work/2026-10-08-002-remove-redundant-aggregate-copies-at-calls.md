# Remove redundant aggregate copies at calls

## Plan

`agents/plans/2026-10-08-002-remove-redundant-aggregate-copies-at-calls.md`

## Summary

A callee reads aggregate parameters through `%paramN`. A caller borrows an
aggregate argument rooted in a local or a constant and still loads one rooted in
a module-level `var`. An aggregate call is its own operand, and `return f(...)`
passes `%result` through to `f`. The plan's goal is met.

## Departures from the plan

- The bounds-check ordering test uses `native_program_status` instead of
  `assert_native_failure`, because the latter takes only a `main` body and the
  test needs helper functions. A trap aborts, so the test expects no exit
  status, where running `quit` first would give status 7.

## Decisions

- `src/backend/emitter.rs` gained `is_aggregate`, used by `place_address`,
  `returned_calls`, and `emit_control_flow`.
- The verification test also accepts a `Borrow` stored into a local, which is
  what a `HeldOperand` spill emits.

## Checks run

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — 295 unit and 58 integration tests passed. The `arrays`,
  `floating_functions`, and `structs` IR snapshots changed: argument loads
  became borrows, and `structs` lost the local and store that held a call
  result before it was passed on.

## Manual verification

1. The emitted QBE has no parameter entry copy, no copy for a borrowed
   argument, and no copy for a returned call.

   ```sh
   mkdir -p /tmp/fernq
   printf '#!/bin/sh\nfor a in "$@"; do last="$a"; done\ncp "$last" /tmp/fernq/out.ssa\nexec qbe "$@"\n' > /tmp/fernq/qbe.sh
   chmod +x /tmp/fernq/qbe.sh
   cat > /tmp/fernq/c.fern <<'EOF'
   type P struct { a: int, b: int, c: int, d: int }
   fn take(p: P) -> int { return p.a + p.d; }
   fn make(x: int) -> P { return P { a = x, b = x, c = x, d = x }; }
   fn relay(x: int) -> P { return make(x); }
   fn main() -> void {
       var p = make(20);
       exit(take(p) + relay(1).a + 1);
   }
   EOF
   QBE=/tmp/fernq/qbe.sh cargo run -q -- /tmp/fernq/c.fern -o /tmp/fernq/c
   /tmp/fernq/c; echo $?
   cat /tmp/fernq/out.ssa
   ```

   The program exits 42. `$fn0` (`take`) reads `%param0` and allocates no
   local. `$fn2` (`relay`) emits `call $fn1(l %result, ...)` and then a bare
   `ret`. `main` passes `take` the address `%local0` without a `blit`.

## Follow-up work

- An element or field of a call result, such as `relay(1).a`, still stores the
  whole result into a fresh local through the `Call` arm of
  `lower_aggregate_place` before reading it.
