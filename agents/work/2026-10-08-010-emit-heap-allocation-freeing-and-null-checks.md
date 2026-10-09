# Emit heap allocation, freeing, and null checks in native code

## Plan

`agents/plans/2026-10-08-010-emit-heap-allocation-freeing-and-null-checks.md`

## Summary

`Allocate` calls `malloc` and traps with `out of memory`. `CheckNull` traps
with `null pointer`. `Free` calls `free`. The driver guard is removed, so
every pointer program compiles and runs. The plan's goal is met.

## Departures from the plan

- `tests/compiler.rs` checks the `null pointer` trap in its own test,
  `a_null_pointer_conversion_reports_its_source_location`, which also asserts
  the reported source location. The new and free program joined
  `native_programs_exit_zero` as `new(1); free(p);`.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 357 unit and 60 integration tests pass.

## Manual verification

1. The spec's linked-list program runs natively.

   ```sh
   printf 'type Node struct { value: int, next: nullable *Node }\nfn sum(list: nullable *Node) -> int { var total = 0; var cursor = list; for cursor != null { const node = *Node(cursor); total += node.value; cursor = node.next; } return total; }\nfn main() -> void { const tail = new(Node { value = 2, next = null }); exit(sum(new(Node { value = 1, next = tail }))); }\n' > /tmp/list.fern
   cargo run -q -- /tmp/list.fern -o /tmp/list && /tmp/list; echo $?
   ```

   Prints `3`.

## Follow-up work

- The `out of memory` trap is covered only by the emitted QBE text. No test
  program exhausts memory.
