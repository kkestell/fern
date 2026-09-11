# Lay out, copy, compare, and access structs in native code

## Sources

- `docs/spec.md#type-declarations`, `#struct-literals` — recursive value
  copying, field selection, zero values, and structural equality
- `eng/roadmap.md#structs` — the fourth task and the later integration
  boundary
- `eng/architecture.md#storage-and-identity` — program-owned struct shapes,
  field ordinals, and the verified-IR-to-backend boundary
- `eng/plans/2026-09-10-009-represent-structs-in-fern-ir.md` — existing struct
  IR and the guard that remains until the final struct task
- `src/backend/emitter.rs`, `src/backend/qbe.rs` — existing aggregate
  signatures, storage, access, copying, static data, and comparison emission

## Goal

Emit verified struct IR as QBE with correct layout, field addresses, copying,
and structural equality. The compiler-level struct guard remains in place, so
the next roadmap task owns public compilation, fixtures, and integration
failures.

## Implementation

- `src/backend/layout.rs` — add one program-aware aggregate layout helper over
  the IR struct table. It must derive a type's QBE class, alignment, byte size,
  field offset, scalar storage slots, and prerequisite QBE aggregate
  definitions. Define structs after their field-type dependencies and preserve
  the existing scalar-array type spelling; arrays whose elements are structs
  use a repeated struct aggregate member. Account for QBE padding in layouts
  and static data rather than treating a struct as a flat homogeneous array.

- `src/backend/mod.rs`, `src/backend/qbe.rs` — expose the shared layout helper
  to emission. Move aggregate size and scalar-slot traversal behind it, and add
  one aggregate equality emitter that compares every scalar slot at its layout
  offset with the scalar's Fern equality operation. This preserves IEEE
  floating-point equality for nested float fields, including signed zero and
  NaN, instead of using bytewise comparison for an aggregate containing
  floats.

- `src/backend/emitter.rs` — make the emitter carry the program-aware layout.
  Emit required struct and struct-containing-array type definitions before QBE
  code; allocate, load, store, copy, pass, return, and address aggregates via
  the shared layout. Extend `Place::Field` by adding the checked field's layout
  offset to its base address. Emit padded global data in layout order, while
  continuing to use the existing flattened IR literals as its values.

- `src/backend/integer.rs`, `src/backend/floating.rs` — route non-scalar
  equality through the shared aggregate comparator and leave scalar arithmetic,
  conversion, and comparison emission unchanged. Remove the homogeneous-array
  comparison paths so arrays and structs have one recursive aggregate equality
  implementation.

- `src/backend/tests/mod.rs`, `src/backend/tests/emitter.rs`,
  `src/backend/tests/structs.rs` — add backend-level coverage for emitted
  struct definitions and padded globals, nested field addresses, whole-value
  copies, struct parameters and results, and equality across nested structs and
  arrays. Run native programs directly from checked and lowered IR to prove
  integer and floating-point struct equality, including signed-zero equality
  and NaN inequality. Keep the public compiler guard test unchanged for the
  following integration task.

## Tests

- QBE type definitions appear before every signature that names a struct or an
  array of structs, include nested dependencies once, and preserve existing
  scalar-array definitions.
- Field loads and stores use the layout-derived offset, including a field after
  a word-sized field and a nested field reached through an array element.
- Locals, globals, assignments, calls, and returns copy structs independently;
  globals include layout padding where QBE requires it.
- `==` and `!=` compare all nested scalar fields with Fern scalar semantics,
  not raw bytes, for integer, array, and floating-point fields.
