# Review: emit heap allocation, freeing, and null checks

## Scope and coverage

Reviewed the staged changes for
`agents/plans/2026-10-08-010-emit-heap-allocation-freeing-and-null-checks.md`:
the `Allocate`, `CheckNull`, and `Free` emission in `src/backend/emitter.rs`,
the removal of the driver guard from `src/semantic/annotations.rs`,
`src/semantic/mod.rs`, and `src/lib.rs`, the new tests in
`src/backend/tests/pointers.rs`, and the updated cases in `tests/compiler.rs`.
Traced the code they depend on: the lowering of `new`, `free`, and `*T(p)` in
`src/ir/lower.rs`, the verifier rules for the three constructs in
`src/ir/verify.rs`, `emit_conditional_trap` and `operation_message` in
`src/backend/qbe.rs`, and the heap allocation, freeing, nullable pointer, and
pointer comparison rules in `docs/spec.md`.

The `out of memory` trap was checked through the emitted QBE text only. No
program exhausts memory at run time.

## Fixed

None.

## Findings

No open findings.

## Checks run

- `cargo fmt --check`: clean.
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo test`: 357 unit and 60 integration tests pass.
- Native programs compiled with the built compiler: a `nullable *[3]int`
  converted with `*[3]int(q)` and indexed, with `len`; `new(new(4))` read
  through `**p`; a non-null `*int(q)` holding `&x`; a struct with a
  `[100]int` field allocated and read back; and 100000 allocate-and-free
  iterations of an `[8]int`. All produced the expected exit status.
- A `null` conversion inside a called function aborts with `null pointer` at
  the conversion's source location.
- `free(null)`, `*int(null)`, and a module-level `new` are rejected at compile
  time, so no backend path receives a literal `null` operand or a module-level
  allocation.
- Confirmed that values are emitted in instruction order, so `new(e)`
  evaluates `e` before `malloc` and stores it after the null check, as the
  specification orders them.

## Verdict

The change meets its plan. No findings.
