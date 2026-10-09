# Represent pointers in Fern IR

## Plan

`agents/plans/2026-10-08-008-represent-pointers-in-fern-ir.md`

## Summary

Fern IR now has `Place::Indirect`, `Literal::Null`, `ValueKind::Address`,
`Allocate`, and `CheckNull`, and `Instruction::Free`. Lowering translates
every checked pointer form, and verification types each construct. A `*T`
fits wherever a `nullable *T` is expected. Pointer programs lower and verify
during compilation. The driver guard runs after verification, and the backend
treats the new IR as unreachable. The plan's goal is met.

## Departures from the plan

- `new(e)` does not hold its operand: the allocation is defined in the block
  where the operand's lowering ends, so the operand is already readable
  there.
- `LocalId` derives `Hash`. `Places` keeps the `LocalId`s of addressed
  bindings in `addressed_locals`, filled by `Places::insert`, and
  `Places::borrowable` decides the borrow rule from a place's root.
- `describe_operand` in `src/ir/tests/mod.rs` spells index, pointer, and
  `free` operands for `describe` and `stored_places`.
- The verifier checks `ValueKind::Address` in `verify_instruction`, against
  `destination_type`. Its arm in `valid_value` is `unreachable!`.
- The `Zero` diagnostic now reads `IR zeroes non-aggregate`, and the existing
  verify test follows it.

## Decisions

- The guard's doc comment now says native emission, not IR lowering, is what
  lacks pointer support.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 346 unit and 59 integration tests pass, including the new
  `pointers` IR snapshot.

## Manual verification

1. The spec's linked-list program lowers and verifies, then stops at the
   driver guard.

   ```sh
   printf 'type Node struct { value: int, next: nullable *Node }\nfn sum(list: nullable *Node) -> int { var total = 0; var cursor = list; for cursor != null { const node = *Node(cursor); total += node.value; cursor = node.next; } return total; }\nfn main() -> void { const tail = new(Node { value = 2, next = null }); exit(sum(new(Node { value = 1, next = tail }))); }\n' > /tmp/list.fern
   cargo run -q -- /tmp/list.fern -o /tmp/list
   ```

   Reports `pointers are not yet implemented` at `nullable *Node`, with no
   internal compiler error.

## Follow-up work

- Native emission must give `Place::Indirect`, `Literal::Null`, `Address`,
  `Allocate`, `CheckNull`, and `Free` real code. It must also give pointer
  slots in `emit_store`, loads, `class`, `collect_scalar_slots`, and
  `emit_aggregate_equal` a word representation. The emission tasks in
  `agents/todo.md` cover this.
