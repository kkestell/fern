# Repair pointer location review findings

## Sources

- `docs/spec.md#assignment` — location, mutability, and evaluation order
- `docs/spec.md#implicit-dereference` — pointer field and index locations
- `eng/architecture.md#storage-and-identity` — checked facts preserved through
  verified IR
- `eng/reviews/2026-09-11-001-pointer-milestone.md` — confirmed defects

## Goal

Make every valid implicit pointer field or index location, including one reached
through a pointer-returning call, compile and execute. Preserve the precise
pointer-operand span for each indirect assignment access.

## Implementation

- `src/frontend/parser.rs` — select the call-statement form only when the full
  statement head is not an assignment; retain normal call statements.
- `src/semantic/model.rs` and `src/semantic/expressions.rs` — let a checked
  implicit pointer location retain the pointer expression and its source span
  without requiring that expression itself to be a location.
- `src/ir/lower.rs` — lower that retained pointer once before an assignment
  value, rebuild it across split blocks, and attach its original operand span
  to the indirect place.
- `src/frontend/tests/parser.rs`, `src/semantic/tests/statements.rs`,
  `src/ir/tests/lower.rs`, `src/backend/tests/pointers.rs`, and
  `tests/compiler.rs` — cover pointer-returning-call field and index stores,
  compound stores, single evaluation, and the exact nested null diagnostic
  location through the public compiler boundary.

## Tests

- A pointer-returning call followed by implicit field or index assignment and
  compound assignment updates the reached storage once.
- A null nested pointer assignment identifies the pointer operand being
  dereferenced rather than the root binding.
