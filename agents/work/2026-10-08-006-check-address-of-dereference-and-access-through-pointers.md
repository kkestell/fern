# Check address-of, dereference, and access through pointers

## Plan

`agents/plans/2026-10-08-006-check-address-of-dereference-and-access-through-pointers.md`

## Summary

Semantic checking now types `&location` as `*T` for mutable locations and
types `*p` as a mutable location and as a value. Field selection, indexing,
and `len` look through non-nullable pointers in values and locations. The
driver guard, now `reject_pointer_syntax`, also rejects `&`. The plan's goal
is met.

## Departures from the plan

- A new `pointer_target` helper gives the `` cannot dereference `{ty}` ``
  check one home, shared by the value and location forms of `*p`.
- The fields of `CheckedLocationKind::Pointer` and `Dereference` carry
  `dead_code` expectations until lowering reads them. `Dereference` uses the
  `cfg_attr(not(test), …)` form because the test helper
  `checked_target_steps` walks its operand.
- `checked_target_steps` reports the new kinds as `pointer` and
  `dereference` steps, and `src/semantic/tests/model.rs` lists the new values
  in its exhaustive match.

## Decisions

- In the guard test, the cases that now check moved to the new tests. The
  cases that reach `new` report the guard at `new(…)`, because
  `check_pointer_operand` infers the call-like operand.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 324 unit and 59 integration tests pass.

## Manual verification

1. A program that assigns through a pointer checks and then stops at the
   driver guard.

   ```sh
   printf 'type S struct { x: int }\nfn main() -> void { var s = S { x = 1 }; const p = &s; p.x = 5; exit(p.x); }\n' > /tmp/addr.fern
   cargo run -q -- /tmp/addr.fern -o /tmp/addr
   ```

   Reports `pointers are not yet implemented` at `&s`.

2. The address of a `const` binding is rejected.

   ```sh
   printf 'fn main() -> void { const q = 1; const p = &q; }\n' > /tmp/const.fern
   cargo run -q -- /tmp/const.fern -o /tmp/const
   ```

   Reports `` cannot take the address of immutable binding `q` `` at `q`.

## Follow-up work

- Lowering must derive the number of pointers a `Field`, `Index`, or `Length`
  step follows from its operand's type, and must evaluate the operand of a
  folded `len(*p)` or `len(p)`.
