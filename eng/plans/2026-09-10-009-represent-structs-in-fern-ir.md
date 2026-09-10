# Represent structs and field access in Fern IR

## Sources

- `docs/spec.md#type-declarations`, `#struct-literals` — nominal struct values,
  recursive copying, field selection, zero fill, and structural equality
- `docs/spec.md#assignment` — mixed element/field targets evaluate their steps
  left to right and compound assignment evaluates the target once
- `eng/roadmap.md#structs` — the third task and the later native-integration
  boundary
- `eng/architecture.md#storage-and-identity` — `StructId`, checked field
  ordinals, IR ownership, and the verified-program boundary
- `eng/plans/2026-09-10-008-check-struct-values-fields-and-equality.md` —
  `CheckedStruct`, `Constant::Struct`, checked struct expressions, and mixed
  assignment steps
- `eng/plans/2026-09-09-011-represent-arrays-in-fern-ir.md` — aggregate loads,
  stores, calls, results, constants, and element places

## Goal

Lower struct declarations, values, literals, field access, mixed assignment
targets, and equality into verified Fern IR. Native emission remains unfinished:
the `semantic::model::reject_uncompiled_structs` guard stays in `src/lib.rs`, and
backend changes are mechanical exhaustiveness updates only.

## Implementation

- `src/ir/model.rs` — add a program-owned struct definition containing its
  field types in ordinal order, and add the definitions to `Program` at the
  same indices as their `StructId`s. Add `Place::Field { base, ordinal }`; make
  `Place::root_local` recurse through both field and element places. Keep whole
  structs on the existing typed `Load`, `Store`, call argument/result, and
  `Return` path used by arrays.

- `src/ir/lower.rs` — copy every checked struct definition into the IR program
  before lowering globals and functions. Flatten global constants by walking a
  `Constant` together with its `Type`: arrays repeat their element type and
  structs pair declaration-order values with the IR definition's field types,
  so heterogeneous and nested aggregates produce typed literals in field
  order without using `Type::leaf` or `Type::element_count`.

- `src/ir/lower.rs` — generalize the array-only place materialization and
  recursive constant store into one aggregate path for arrays and structs.
  Lower a runtime struct literal into a fresh local by evaluating its written
  initializers once in source order and storing each through its recorded field
  ordinal, then recursively storing the zero constants recorded for omitted
  fields. A folded struct follows the same recursive constant-store path as a
  folded array. Extend folded-effect traversal through written struct
  initializers so a folded field expression cannot suppress an owed trap.

- `src/ir/lower.rs` — lower `value.field` to a `Place::Field` over the
  materialized struct operand, then load it through the same scalar-or-aggregate
  operand path as indexing. Struct references, groupings, calls, and fields of
  aggregate type all use the shared aggregate-place helper; do not add a
  struct-specific load or copy instruction.

- `src/ir/lower.rs` — replace `HeldPlace`'s index-only list with ordered held
  steps. An index step lowers and holds its operand before the assignment value;
  a field step records only its ordinal. Rebuild the same mixed field/element
  place after block-splitting expressions, and reuse it for both the read and
  write of a compound assignment.

- `src/ir/verify.rs` — validate the program's struct table before functions:
  every struct reference names a definition, field types are valid, and field
  and array edges cannot form a recursive value type. Apply type validation to
  globals, function parameters/results/locals, and values so an unknown struct
  cannot reach the backend through an otherwise unused type.

- `src/ir/verify.rs` — extend `place_type` through `Place::Field`, requiring a
  struct base and an in-range ordinal and returning the declared field type.
  Keep equality on identical aggregate types and keep arithmetic, conversion,
  ordering, and logical operations scalar-only. A store through a field keeps
  initializing its root local through `root_local`.

- `src/ir/verify.rs` — verify a global by recursively deriving its expected
  scalar type sequence from the global type and the program's struct
  definitions, then compare that sequence with its flattened literals and run
  the existing literal range checks. This replaces the homogeneous
  `element_count`/`leaf` check only at the global boundary; those helpers remain
  array-specific.

- `src/semantic/model.rs` — remove the temporary dead-code annotation from
  `CheckedStep::Field`; lowering now consumes the ordinal. Keep the temporary
  struct rejection and its diagnostic unchanged until native layout and access
  are implemented.

- `src/backend/emitter.rs`, `src/backend/tests/mod.rs`, `src/ir/tests/mod.rs`,
  `src/ir/tests/verify.rs` — add empty struct tables to hand-built programs and
  mark field places unreachable behind the existing compiler guard. Do not add
  layout, address calculation, copying, or comparison emission in this task.

## Tests

- `src/ir/tests/lower.rs` — cover IR definitions for nested and array-containing
  structs; heterogeneous module globals flattened in field order; complete,
  reordered, and zero-filled literals; recursive constant stores; scalar and
  aggregate field selections; independent whole-struct copies; struct
  parameters, results, and arguments; and runtime `==`/`!=` values.

- `src/ir/tests/lower.rs` — prove written field initializers lower in source
  order, including calls; mixed targets such as `grid[i].point.x` and
  `value.rows[i].count` build field and element places in order; and a compound
  assignment evaluates each runtime index once even when its value splits
  control flow.

- `src/ir/tests/verify.rs` — accept valid nested field places and heterogeneous
  struct globals. Reject an unknown struct ID, a recursive struct table, a field
  place over a non-struct, an out-of-range field ordinal, a store with the wrong
  field type, and globals with missing, extra, out-of-order, or out-of-range
  scalar values.

- `src/ir/tests/lower.rs` and `src/ir/tests/snapshots/` — add a struct lowering
  snapshot containing nested literals, field access and assignment, equality,
  and a struct-valued call. Regenerate existing IR snapshots for the new empty
  `Program::structs` field rather than editing them by hand.

## Decisions

- Struct definitions belong to `ir::model::Program`, indexed by the existing
  program-local `StructId`. `Type` keeps nominal identity and diagnostic
  spelling; field shape lives once in the program representation that lowering,
  verification, and the next task's backend layout consume.

- `Place::Field` carries an ordinal but no span. Semantic checking has already
  resolved and validated the field, and field access has no runtime failure to
  diagnose.

- Global data remains a flat sequence of typed scalar literals. Walking the
  program-owned field definitions supports heterogeneous structs while keeping
  one static-data representation for scalars, arrays, and structs.
