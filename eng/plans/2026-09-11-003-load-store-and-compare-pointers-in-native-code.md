# Load, store, and compare pointers in native code

## Sources

- `docs/spec.md#pointers` and `#assignment` — pointer representation, copying,
  dereference traps, implicit access, conversion, comparison, and assignment.
- `eng/todo.md#pointers` — the fourth Pointers subtask; compiler integration
  and pointer failure coverage remain the final subtask.
- `eng/architecture.md#pipeline` and `#storage-and-identity` — verified IR is
  the sole backend input and retains the resolved indirect-place span.
- `eng/plans/2026-09-11-002-represent-pointers-and-indirect-access-in-fern-ir.md`
  — typed nulls, address-of values, indirect places, and the temporary guard
  this emission consumes.
- `src/backend/emitter.rs`, `src/backend/layout.rs`, and `src/backend/qbe.rs`
  — the existing QBE storage, layout, static-data, and comparison paths.

## Goal

Emit verified pointer IR as native QBE, including nulls, addresses, indirect
loads and stores, pointer copies, comparisons, and pointer-to-`uint`
conversion. The compiler-level pointer staging diagnostic remains until the
next subtask, so this does not complete the Pointers parent task.

## Implementation

- `src/ir/verify.rs` — permit the one nonnumeric `Convert` form the language
  defines: a non-truncating pointer-to-`uint` conversion. Keep every other
  numeric-operation restriction unchanged so malformed IR cannot use pointers
  in arithmetic or unrelated conversions.

- `src/backend/layout.rs` — treat a pointer as QBE's word-sized `l` storage
  class wherever a Fern value is stored or passed. Extend the flattened global
  storage-slot traversal to include pointer leaves at their layout offsets,
  while pointer targets contribute neither inline storage nor QBE aggregate
  definitions. This must preserve recursive structs such as `Node { next:
  *Node }` without recursive QBE definitions.

- `src/backend/qbe.rs` — spell a null operand and null static-data leaf as a
  zero `l` value. Extend the recursive aggregate equality emitter to load and
  compare pointer leaves as `l` values, preserving field-by-field Fern
  equality rather than reading padding. Keep `*T` and `*const T` as the same
  runtime representation.

- `src/backend/emitter.rs` — emit pointer locals, parameters, results, calls,
  returns, loads, stores, and address-of values as `l` operands rather than
  aggregate copies. Resolve `Place::Indirect` by evaluating its already-held
  pointer operand, branch to the existing diagnostic trap machinery when it
  is zero, and otherwise use the pointer itself as the reached address. Give
  every indirect access its retained source span and a distinct emission ID so
  explicit dereference and implicit field, index, and `len` access report a
  source diagnostic and never perform the load or store on null. Emit
  pointer-to-`uint` conversion as a same-width `l` copy, with no runtime
  check; pointer mutability conversion remains representation-free.

- `src/backend/tests/mod.rs` and `src/backend/tests/pointers.rs` — add a
  pointer-focused native-emission test module using the existing lowered-IR
  helpers. Keep it below `lib::compile` so the staging diagnostic continues
  to cover the public boundary.

## Tests

- Native programs prove local, parameter, result, and aggregate pointer copies
  alias their original locations; address-of, explicit dereference, and
  indirect assignment load and store the reached value.
- Pointer equality and inequality cover identical addresses, distinct
  addresses, null, and mutable-to-const copies; `uint(p)` preserves the
  non-null address and converts null to zero.
- Struct and array fields containing pointers, including a self-referential
  struct and zero-initialized pointer globals, emit valid QBE with correct
  layout and equality.
- Explicit dereference and implicit pointer field, element, and `len` access
  abort on null and render the corresponding Fern source location. Valid
  indirect access continues after the inserted check.
