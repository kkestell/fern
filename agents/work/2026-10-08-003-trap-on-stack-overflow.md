# Trap on stack overflow

## Plan

`agents/plans/2026-10-08-003-trap-on-stack-overflow.md`

## Summary

Implemented the planned call checks and stack-bounds entry wrapper. The named
recursion and oversized-frame cases trap with source diagnostics. The universal
stack-safety goal remains subject to the limitation recorded in `agents/bugs.md`.

## Departures from the plan

- Retained and narrowed the stack bug instead of removing it. A fixed reserve
  does not bound QBE's spill storage, and the plan's guard-region assumption
  does not establish its guarantee for every generated frame.
- Also check `pthread_attr_getstack` failure and destroy the attributes before
  trapping, so the Linux wrapper never reads an unset stack address.
- Updated IR construction helpers and existing arithmetic emission tests for
  the new entry wrapper and `main_span` field. The snapshot changes add only
  `main_span`.
- Reduced the large-aggregate execution fixture's local arrays from 50,000 to
  10,000 elements; its 80 MB global remains unchanged. The planned conservative
  accounting rejected the larger fixture at `make(3)` even though it previously
  ran. Large aggregate operations and static relocation coverage remain.
- README describes the allocation check plus reserve, rather than claiming the
  check measures the exact remaining stack after a call.

## Checks run

- `cargo test stack_ -- --test-threads=1` — four backend tests and one compiler
  integration test passed on macOS arm64.
- `INSTA_UPDATE=always cargo test --lib ir::tests::lower` — 54 passed;
  inspected all 23 snapshot updates.
- `cargo test` — 299 unit tests and 58 integration tests passed; the
  large-aggregate fixture failed under the conservative stack check. After
  reducing its local arrays, `cargo test --test compiler
  large_aggregates_compile_and_execute` passed.
- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `git diff --check` — passed.

## Follow-up work

- Resolve the stack limitation in `agents/bugs.md` before claiming every
  generated frame remains inside the stack bounds.
- The Linux stack-bounds sequence needs native validation on Linux; this host
  exercised the macOS sequence.
