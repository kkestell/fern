# Repair pointer milestone review findings

## Sources

- `docs/spec.md#typing-by-context` — contextual type acquisition
- `docs/spec.md#length` — evaluate the operand while reading length from its type
- `docs/spec.md#the-null-pointer` — pointer context for `null`
- `docs/spec.md#address-of` — addressability diagnostic rule
- `docs/spec.md#implicit-dereference` — null-trapping pointer access
- `eng/architecture.md#storage-and-identity` — shared type facts and verified-IR boundary
- `eng/reviews/2026-09-11-002-pointer-milestone.md` — confirmed defects and coverage gaps

## Goal

Make grouped `null` contextualize like other untyped constants, avoid copying an
array solely to evaluate `len` through a pointer, and consolidate pointer value
compatibility. Improve location diagnostics and coverage around address-taking
and held compound-assignment targets. This completes the repair of the latest
pointer milestone review findings.

## Implementation

- `src/types.rs`, `src/semantic/expressions.rs`, and `src/ir/verify.rs` — own
  pointer mutability-withdrawal compatibility on `Type` and use it at both
  semantic and IR boundaries.
- `src/semantic/expressions.rs` — pass a contextual destination through a
  grouping, recognize grouped `null` when contextualizing comparison operands,
  distinguish address-of from assignment location diagnostics, and factor the
  common field/index location path without changing their evaluation facts.
- `src/semantic/statements.rs` — make the compound-assignment placeholder
  expression self-documenting.
- `src/ir/model.rs`, `src/ir/lower.rs`, and `src/backend/emitter.rs` — model
  and emit a null-check-only indirect access for implicit-pointer `len`, without
  materializing or copying the reached aggregate.
- `src/semantic/tests/expressions.rs`, `src/semantic/tests/statements.rs`,
  `src/ir/tests/lower.rs`, and `src/backend/tests/pointers.rs` — cover grouped
  `null` in each contextual position, address-of non-locations and null
  indirect locations, one evaluation of a compound target, and no aggregate
  load for pointer `len`.

## Tests

- Grouping does not prevent `null` from taking the type of an annotated
  binding, assignment target, parameter, or pointer comparison operand.
- `len(pointer_to_array)` traps when null but its emitted form contains no
  aggregate load or copy.
- Address-of preserves ordinary location rules and reports its own
  non-location diagnostic; address-of through null field, index, and
  dereference paths traps.
- A compound assignment through a call-result pointer evaluates that call once.
