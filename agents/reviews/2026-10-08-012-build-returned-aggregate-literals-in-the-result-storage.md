# Review: build returned aggregate literals in the result storage

## Scope and coverage

Reviewed the staged changes for
`agents/plans/2026-10-08-012-build-returned-aggregate-literals-in-the-result-storage.md`:
`Place::Result` in `src/ir/model.rs`, its typing and checks in
`src/ir/verify.rs`, `returned_literal` and `store_literal` in
`src/ir/lower.rs`, `returned_in_place`, `allocations`, `emit_call`,
`emit_terminator`, `place_address`, and the `Load` arm of `emit_value` in
`src/backend/emitter.rs`, and the new and changed tests and snapshot.

The review traced every way the result storage can be reached. A caller passes
either a fresh `%v{id}_storage` or, for a returned call, its own `%result`. No
lowered assignment or initialization passes a destination as result storage.
An aggregate argument is either a copy or a borrowed constant or unaddressed
local, never the result storage. `&` applies only to `var` bindings, pointer
targets, and their fields and elements, so no pointer reaches a call's storage.
A returned literal therefore cannot observe its own earlier writes, whether
it reads through parameters, globals, pointers, or calls.

## Fixed

- **The verifier accepts IR that borrows or addresses the result storage**
  (`src/ir/verify.rs:79`, `src/ir/verify.rs:470`): `Borrow(Result)` and
  `Address(Result)` passed verification. Lowering never produces either, but
  the verifier is the guard the backend relies on. Passing such a value to a
  returned call would hand the callee the same storage as both an argument and
  its result storage. A callee that builds a returned literal in place would
  then read fields it had already overwritten. Hand-built IR with
  `Borrow(Result)` verified before the fix. The verifier now excludes
  `Place::Result` as a borrow root and as an address target, and
  `verification_rejects_borrowing_or_addressing_result_storage` covers both.

## Findings

No open findings.

## Checks run

- `cargo fmt --check`: clean.
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo test`: 363 unit and 62 integration tests pass.
- The new verifier test fails without the verifier fix.
- Native programs, all exiting 42:
  - literals that swap fields read through a pointer to the caller's own
    destination (`a = swapp(&a)`, `v = arr(&v)`, `*h = swapp(h)` on a heap
    target);
  - a literal that swaps a global assigned back to that global;
  - a literal whose first field's call writes a global the second field reads;
  - a returned call that forwards to a callee building a literal in place;
  - nested literals;
  - a returned fill;
  - a grouped struct literal with omitted array and scalar fields.

## Verdict

The change meets the plan. Building a returned literal in the result storage
cannot be observed from Fern source. One verifier gap was fixed, and no
findings remain open.
