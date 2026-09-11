# Represent pointers and indirect access in Fern IR

## Sources

- `docs/spec.md#pointers` and `#assignment` — pointer values, locations,
  access traps, conversions, comparison, and evaluation order.
- `eng/todo.md#pointers` — the fourth Pointers subtask; native pointer support
  remains next.
- `eng/architecture.md#storage-and-identity` — only verified IR reaches the
  backend and carries resolved locations forward.
- `eng/plans/2026-09-11-001-check-pointer-values-and-indirect-assignment.md`
  — checked pointer expressions and locations this lowering consumes.
- `src/ir/model.rs`, `src/ir/lower.rs`, and `src/ir/verify.rs` — existing IR
  values, places, lowering order, and verification boundaries.

## Goal

Lower every semantically valid pointer value and indirect location into
verified Fern IR, including pointer-bearing globals and aggregates. Native
emission remains staged after IR verification, so this does not complete the
Pointers parent task.

## Implementation

- `src/ir/model.rs` — represent typed null values, address-of values, and an
  indirect place whose pointer operand and source span are retained for the
  later native null check. Let pointer values participate in literals and
  globals without classifying them as numeric scalars; update the IR's
  immediate-value ownership accordingly.

- `src/ir/lower.rs` — lower `null`, address-of, explicit dereference, and the
  one permitted implicit dereference into those IR forms. Replace the
  binding-rooted target reconstruction with recursive checked-location
  lowering, holding index and pointer operands before an assignment value and
  reusing them for compound assignment. Materialize pointer zero values in
  globals, arrays, structs, and declarations, and preserve the `*T` to
  `*const T` value conversion at every contextual copy boundary. Keep pointer
  equality as an IR comparison and `uint(p)` as the already-checked numeric
  conversion path.

- `src/ir/verify.rs` — accept recursive pointer types without treating a
  pointer edge as inline struct containment. Verify pointer literals and
  pointer slots in globals, address-of values, and indirect places (including
  pointer operand type, reached target type, and use-before-definition).
  Extend store, call, return, and equality validation for the one-way
  mutable-to-const pointer conversion and same-target pointer equality while
  retaining exact type checks elsewhere. Do not mark an indirect place's
  referenced local initialized or require it to be locally initialized.

- `src/semantic/mod.rs` and `src/lib.rs` — retain the temporary
  `pointers are not yet implemented` diagnostic, but compute it before moving
  the checked program and return it only after lowering and IR verification.
  This makes the backend, rather than semantic checking, the remaining
  boundary for valid pointer programs.

- `src/ir/tests/mod.rs`, `src/ir/tests/lower.rs`, and
  `src/ir/tests/verify.rs` — add helpers and focused lowering and malformed-IR
  cases for typed null, address-of, explicit and implicit indirect access,
  pointer-bearing aggregate/global storage, pointer comparison and conversion,
  and single evaluation of indexed or indirect compound targets. Update the
  affected debug snapshots for the new IR forms.

- `tests/compiler.rs` — keep the output-preservation check for a valid pointer
  program, asserting that it reaches the post-verification staging diagnostic
  rather than the backend.

## Tests

- IR lowering covers addressable locals, fields, elements, globals, nested
  dereferences, `p.field`, `p[index]`, and `len(p)` while preserving the
  source-order evaluation and single-evaluation guarantees.
- IR verification rejects a null with a non-pointer slot, an address-of value
  whose target differs from its place, indirect access through a non-pointer
  or with a mismatched result type, invalid pointer conversions, and invalid
  pointer comparisons.
- Pointer-containing recursive structs and pointer zero values verify through
  the IR; native compilation still fails before emitting or replacing output.

## Decisions

- The IR carries pointer types and accesses even though it cannot yet emit
  them. The existing staging diagnostic moves after verification so the next
  native slice receives a complete, verified pointer IR contract.

- An indirect place retains the pointer operand rather than rewriting it into
  an address arithmetic expression. That preserves the dereference operation
  and its source span for the native slice's null trap.
