# Represent pointer types across the compiler

## Plan

`agents/plans/2026-10-08-005-represent-pointer-types.md`

## Summary

Pointer annotations resolve to `Type::Pointer`. Pointers have the `uint`
width in the shared layout. Nullable pointers zero to `Constant::Null`, and
declarations and fills of types without a zero value are rejected. The driver
rejects pointer annotations before lowering. The plan's goal is met.

## Departures from the plan

- `AnnotationKind::Pointer::nullable` lost its
  `cfg_attr(not(test), expect(dead_code, …))` attribute, because resolution
  now reads it and the expectation went unfulfilled.
- Zero-value rejection goes through a new helper,
  `CheckedProgram::require_zero_value`, so local and module-level
  declarations share one diagnostic.

## Decisions

- `CheckedStruct::has_zero_value` starts `false` and is only read for a
  `Resolved` struct, where `resolve_struct_fields` sets it.
- The ordering test's last case declares a valid `nullable **int` before the
  invalid `*nullable *int`, which shows that only the type without a zero
  value is rejected.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 317 unit and 59 integration tests pass.

## Manual verification

1. A self-referential struct checks and then hits the driver guard.

   ```sh
   printf 'type Node struct { value: int, next: nullable *Node }\nfn main() -> void { var n: Node; exit(n.value); }\n' > /tmp/node.fern
   cargo run -q -- /tmp/node.fern -o /tmp/node
   ```

   Reports `pointers are not yet implemented` at `nullable *Node` and exits 1.

2. A non-nullable pointer declaration needs an initializer.

   ```sh
   printf 'fn main() -> void { var p: *int; }\n' > /tmp/zero.fern
   cargo run -q -- /tmp/zero.fern -o /tmp/zero
   ```

   Reports `` `*int` has no zero value `` at `var p: *int;` and exits 1.

## Follow-up work

- The IR task replaces the `unreachable!` pointer arms in `src/ir/` and
  `src/backend/`, and removes `reject_pointer_annotations` from `compile`.
