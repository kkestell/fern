# Represent pointer types across the compiler

## Sources

- `docs/spec.md#pointer-types` — pointer spelling, mutability, structural
  identity, value semantics, and fixed target-independent size.
- `docs/spec.md#type-declarations` — pointer fields break recursive value-type
  cycles.
- `eng/todo.md#pointers` — the second Pointers subtask; pointer values,
  operations, IR, and native execution remain later work.
- `eng/architecture.md#storage-and-identity` — checked types and shared layout
  derivation cross the semantic, IR, and backend boundaries.
- `eng/plans/2026-09-10-013-parse-pointer-types-address-of-and-dereference.md`
  — the parsed pointer annotation forms and temporary diagnostics this work
  replaces at the type boundary.

## Goal

Resolve parsed pointer annotations into a recursive compiler type and permit
pointer fields to break struct containment cycles. Pointer values and their
operations remain rejected before IR lowering, so this does not complete the
Pointers task.

## Implementation

- `src/types.rs` — add `Type::Pointer { constant, target: Box<Type> }`, with
  derived structural equality, source spelling as `*T` or `*const T`, and no
  scalar classification. Cover constness and recursively different targets in
  the type tests.

- `src/layout.rs` — give a pointer the target platform's pointer-sized integer
  storage size and alignment without visiting its target. This lets shared
  semantic layout derive a self-referential struct through a pointer while
  retaining the existing inline array/struct traversal.

- `src/semantic/annotations.rs` — replace the pointer-annotation rejection
  with recursive pointer resolution. Resolve a pointer target's names and
  array lengths, but do not resolve or derive the layout of a target struct:
  the pointer edge is not inline containment. Keep ordinary named and array
  annotations on the existing layout-validating path, so direct or
  array-mediated value recursion is still diagnosed. Ensure each module's
  independent struct-resolution pass continues to validate otherwise-unused
  declarations.

- `src/semantic/model.rs` and every exhaustive `Type` consumer in
  `src/semantic/`, `src/ir/`, and `src/backend/` — add the pointer case. Treat
  it as one non-aggregate storage slot for type and layout bookkeeping, and
  mark paths that would lower, verify, or emit a pointer value unreachable
  behind the temporary pre-IR guard. Do not add pointer constants, operands,
  places, QBE classes, or native load/store logic.

- `src/semantic/mod.rs`, `src/lib.rs` — retain expression-level pointer
  diagnostics, but move the annotation staging boundary to after successful
  semantic checking and before IR lowering. Find pointer types recursively in
  bindings, signatures, and checked struct fields; report the written
  annotation span with `pointers are not yet implemented`. This allows checked
  type and layout construction while preventing a pointer value from entering
  the current IR and backend.

- `src/semantic/tests/annotations.rs` and the affected semantic test helpers
  — replace annotation-time rejection expectations with checks of resolved
  pointer types and the pre-IR staging diagnostic. Add direct and indirect
  self-pointer struct fields, pointer targets containing arrays and nested
  pointers, and preserve diagnostics for inline struct recursion.

## Tests

- `src/types.rs` — pointer equality, display spelling, and scalar exclusion
  for mutable and const pointers with nested targets.

- `src/semantic/tests/annotations.rs` — resolve pointers in bindings,
  parameters, results, arrays, and struct fields; accept recursive structs
  whose only recursive edges are pointers; reject the corresponding direct and
  array-mediated inline cycles.

- `tests/compiler.rs` — a semantically valid pointer annotation and a
  self-referential struct reach the single pre-lowering pointer diagnostic,
  preserving any existing output file. Existing `null`, address-of,
  dereference, and indirect-assignment tests continue to reach their temporary
  expression diagnostics.

## Decisions

- Pointer target traversal resolves the written type but deliberately skips
  target layout and field resolution. A pointer's representation is independent
  of its target, while `resolve_module_structs` still checks every declared
  struct in its own right.

- The pre-IR guard is temporary staging. The next pointer-checking slice
  removes only the semantic restrictions it implements; the later IR and
  backend slices own pointer values and native representation.
