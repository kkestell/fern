# Represent slice types across the compiler

## Sources

- `docs/spec.md#slice-types` — `[]T` and `[]const T` as distinct value types,
  the empty slice as the zero value, and the `[]T` to `[]const T` conversion.
- `docs/spec.md#type-declarations` — a slice field breaks a struct's size
  cycle, as a pointer field does.
- `eng/todo.md#slices` — the second Slices subtask; slice values, bounds,
  length, comparison, iteration, IR, and native execution remain later work.
- `eng/architecture.md#storage-and-identity` — `layout` is the one derivation
  semantic validation, IR verification, and the backend all read.
- `eng/plans/2026-09-11-007-parse-slice-types-and-slicing-expressions.md` — the
  parsed annotation and expression forms, and the temporary annotation-time
  diagnostic this task replaces.

## Goal

Resolve parsed `[]T` annotations into a compiler type with a fixed two-word
layout, and let a slice field break a recursive struct declaration. Slice
values and operations stay rejected, now by a pre-IR guard rather than at
annotation resolution, so this does not complete the Slices task.

## Implementation

- `src/types.rs` — add `Type::Slice { constant: bool, element: Box<Type> }`
  beside `Pointer`, with derived structural equality. Spell it `[]T` and
  `[]const T` in `Display`, exclude it from `scalar()`, and extend
  `value_compatible` so a `[]T` value can initialize a `[]const T` destination
  with the same element type, and only in that direction. Reword the method's
  doc comment: both non-identical conversions withdraw mutability, one through
  a pointer and one through a slice.

- `src/layout.rs` — `size` gives a slice `2 * scalar_bytes(Scalar::Uint)` and
  `alignment` gives it `scalar_bytes(Scalar::Uint)`, neither visiting the
  element. This is what lets a struct reach itself through a slice field and
  still derive a finite layout.

- `src/semantic/annotations.rs` — rename `resolve_pointer_target` to
  `resolve_referenced_type` and say in its doc comment that a pointer target
  and a slice element are both reached through the value rather than stored in
  it. Give it an `AnnotationKind::Slice { constant, element }` arm that recurses
  on the element, keeping its existing `[_]` rejection and its refusal to
  resolve a target struct's fields.

- `src/semantic/annotations.rs` — in `resolve_annotation`, replace the
  `AnnotationKind::Slice` rejection with
  `Type::Slice { constant: *constant, element: Box::new(self.resolve_referenced_type(*element, scopes)?) }`,
  and extend the `validate_aggregate_layout` skip to
  `AnnotationKind::Pointer { .. } | AnnotationKind::Slice { .. }`. A slice's own
  size is fixed, so validating it would only risk deriving and caching a layout
  for a struct whose fields are still resolving.

- `src/semantic/annotations.rs` — add the temporary staging guard
  `pub(crate) fn reject_slice_annotations(syntax: &Syntax) -> Result<(), Diagnostic>`.
  It scans `syntax.annotations` for `AnnotationKind::Slice { .. }` and reports
  `slices are not yet implemented` at the earliest one, ordering by `span.start`
  and breaking a tie by the larger `span.end` so `[][]int` reports the whole
  written annotation rather than its element.

- `src/semantic/mod.rs` — make the module `pub(crate) mod annotations;`.

- `src/lib.rs` — call `semantic::annotations::reject_slice_annotations(&syntax)`
  between `semantic::namespaces::check` and `ir::lower::lower`, rendering its
  diagnostic through `sources` the way the checking diagnostic is rendered. Real
  semantic errors are reported first.

- `src/semantic/model.rs` — add `Constant::EmptySlice` as the slice zero value,
  with arms in `integer`, `rational`, and `float`, which list every variant.
  `zero_value_unchecked` returns it for a slice type, and
  `aggregate_value_count` counts a slice as `Some(1)`, matching `Constant::Null`
  for a pointer.

- `src/ir/verify.rs` — `contained_struct` returns `None` for a slice, which is
  the intended end state rather than staging: a slice stores no struct inline.
  `verify_type` returns
  `Err(CompileError::new("internal compiler error: IR uses a slice type"))`,
  and `scalar_types` and `scalar_count` take
  `unreachable!("verified IR has no slice type")`, since both run only after
  `verify_type` on the same type.

- `src/backend/layout.rs` (`class`, `array_class`, `collect_scalar_slots`, and
  `TypeDefinitions::define`), `src/backend/qbe.rs`
  (`emit_aggregate_equal`), and `src/backend/emitter.rs` (the three exhaustive
  `Type` matches around the store, load, and copy paths) — each takes
  `unreachable!("verified IR has no slice type")`. Add no QBE class, scalar
  slot, or load/store logic for a slice.

- `src/ir/lower.rs` — unchanged. Its type matches have fallback arms and the
  guard keeps a slice type out of lowering.

## Tests

- `src/types.rs` — slice equality across constness, element type, and nesting;
  `[]int`, `[3]int`, and `*int` are three types; spellings for `[]int`,
  `[]const int`, `[][]const int`, `[]*int`, and `[]const [3]int`; `scalar()` is
  `None`; `[]T` is value-compatible with `[]const T` and with nothing else,
  including not the reverse, not across element types, and not with `*T`.

- `src/semantic/tests/mod.rs` — a `slice_type(constant, element)` helper beside
  `pointer_type`, and a helper mirroring `rejects_root` that checks a program
  successfully and then asserts the message and `«»`-marked span the staging
  guard reports.

- `src/semantic/tests/annotations.rs` — replace
  `slice_annotations_reach_the_temporary_implementation_guard` with resolution
  tests reading back the checked types for `[]int`, `[]const int`, `[][]int`,
  `[]*const int`, `*[]int`, `[3][]int`, and `[]Point`; a struct whose only
  recursive edge is a `[]Node` field resolves and lays out, while the inline and
  array-mediated cycles are still rejected; `[][_]int` still reports
  ``[_]` requires an array-literal initializer``; `[]void` still fails in the
  parser.

- `src/semantic/tests/annotations.rs` — a slice is exactly two pointer-width
  words: `[70368744177664][]int` is accepted at the 1 PiB limit and
  `[70368744177665][]int` reports
  `aggregate layout exceeds compiler limit of 1 PiB`.

- `src/semantic/tests/annotations.rs` — a slice annotation on a module binding,
  a local binding, a parameter, a result, a struct field, and an array element
  each passes checking and then reaches the staging guard at the annotation's
  span.

- `src/semantic/tests/expressions.rs` and `src/semantic/tests/statements.rs` —
  the existing slicing-expression guards are unchanged; a slicing expression is
  still rejected during checking, ahead of the staging guard.

- `tests/compiler.rs` — a program with a slice annotation fails with
  `slices are not yet implemented` and preserves an existing output file.

## Decisions

- A slice occupies two pointer-width words, the element address and the length.
  Only its size and alignment matter now, so that struct layout, field offsets,
  and the aggregate limit are already correct; slot order, QBE class, and the
  length's storage type belong to the native slice task.

- The staging guard scans written annotations rather than checked types. Every
  slice type in a checked program comes from a written `[]T`, and the annotation
  is what carries the span to report.
