# Emit address-of, dereference, and pointer comparison in native code

## Plan

`agents/plans/2026-10-08-009-emit-address-of-dereference-and-pointer-comparison.md`

## Summary

The QBE backend now emits:

- pointer loads, stores, parameters, and results as `uint`-width words;
- `Place::Indirect`, `ValueKind::Address`, and `Literal::Null`, in code and
  in static data;
- pointer equality, including fields of aggregates compared for equality.

Programs that take addresses and read, write, and compare through pointers
compile and run. The driver guard, now `reject_heap_syntax`, rejects only
`new`, `*T(p)`, and `free`. The plan's goal is met.

## Departures from the plan

- `Layout::class` returns `l` for an aggregate and `value_class` for anything
  else, rather than matching each type.
- `pointers_not_implemented` became private to `src/semantic/annotations.rs`,
  since only the guard uses it.

## Decisions

- The emitter module doc comment needed no change. It describes aggregate
  passing, which pointers do not alter.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 352 unit and 59 integration tests pass. The new
  `src/backend/tests/pointers.rs` builds and runs six native programs.

## Manual verification

1. The spec's address-of example runs natively.

   ```sh
   printf 'type Point struct { x: int, y: int }\nfn main() -> void {\n    var p = Point { x = 1, y = 2 };\n    const a = &p;\n    a.x = 5;\n    exit(p.x);\n}\n' > /tmp/addr.fern
   cargo run -q -- /tmp/addr.fern -o /tmp/addr && /tmp/addr; echo $?
   ```

   Prints `5`.

## Follow-up work

- The next todo item emits `Allocate`, `CheckNull`, and `Free` and removes
  `reject_heap_syntax`.
