# Remove pointers, slices, and unsettled features

## Goal

Cut Fern back to the parts that are settled and correct: integers, `bool`,
floating-point numbers, bindings and scopes, control flow, functions, modules
and imports, arrays, and structs. Pointers, slices, comparability, and the
specified-but-unimplemented rune, string, character-literal, and non-struct
named-type features leave the language, the compiler, the examples, and the
documents. The IR verifier drops the checks that only re-prove facts semantic
checking already established. `eng/todo.md` is deleted so that planning starts
over.

When this is done, the specification describes exactly what the compiler
accepts. Every remaining type is comparable, every location is rooted in a
binding, and the IR has no pointer, slice, or null-check constructs.

## Related code

- `src/types.rs` — `Type::Pointer`, `Type::Slice`, and `Type::value_compatible`,
  which exists only to let a mutable pointer or slice convert to its `const`
  form.
- `src/comparability.rs` — decides `==`/`!=` on types. Slices are the only
  type it rejects.
- `src/frontend/{lexer,syntax,parser}.rs` — the `null` token; the
  `AnnotationKind::Pointer`/`Slice` annotations; the
  `ExpressionKind::Null`/`AddressOf`/`Dereference`/`Slice` expressions; the
  `PrefixOperator::AddressOf`/`Dereference` operators; `[low:high]` parsing in
  `postfix_expression`; and the leading-`*` loop in `token_after_target`.
- `src/semantic/expressions.rs`:
  - `check_location_with_facts`, `location_step_operand`, `location_operand`,
    `LocationUse`, and `LocationStep` exist so that address-of, slicing, and
    pointer or slice call results can act as locations.
  - `infer_address_of`, `infer_dereference`, `infer_slice`,
    `is_null_expression`, and `check_grouping`.
  - The pointer arms in `infer_field`, `infer_index`, `infer_length`,
    `infer_comparison`, `unify_comparison`, and `infer_conversion`
    (pointer to `uint`).
- `src/semantic/model.rs`:
  - `Constant::Null` and `EmptySlice`.
  - `ExpressionValue::Null`, `AddressOf`, `Dereference`, and `Slice`, and every
    `implicit_dereference` field.
  - `CheckedLocationKind::SliceValue` and `Dereference`.
  - The pointer and slice arms of the zero value and scalar count.
- `src/semantic/statements.rs`:
  - `assignment_target`, `location_binding`, and `location_uses_reference`.
  - The slice arm of `for … in`.
- `src/semantic/annotations.rs` — `resolve_referenced_type` and the
  pointer/slice exemption from `validate_aggregate_layout`.
- `src/semantic/namespaces.rs` — `collect_referenced_lengths` and the
  pointer/slice arms of `collect_annotation_references`.
- `src/semantic/constants.rs` — the `Null`/`EmptySlice` arms.
- `src/ir/model.rs`:
  - `Place::Indirect` and `SliceElement`.
  - `Literal::Null` and `EmptySlice`.
  - `ValueKind::AddressOf`, `WholeSlice`, `SliceRange`, and `SliceLength`.
  - `Instruction::Check`, whose only producer is the null check in
    `lower_length`.
- `src/ir/lower.rs`:
  - `lower_location`, `lower_location_slice`, `HeldPlace`, and `HeldStep`.
  - `lower_slicing`, `lower_expression_location`, `pointer_literal`, and
    `slice_literal`.
  - The pointer and slice branches of `lower_flow_operand`,
    `lower_aggregate_place`, `element_place`, `field_place`, `lower_length`,
    and `lower_folded_effects`.
  - `IterationSource` and `capture`.
- `src/ir/verify.rs`:
  - `verify_structs`, `verify_acyclic`, `Resolution`, `contained_struct`,
    `verify_type`, and `verify_aggregate_layout`.
  - `impl StructFields for Program`, and the comparability call in
    `valid_value`.
  - The definite-initialization analysis: `LocalSet`, `initialized_locals`,
    `block_predecessors`, `local_stores`, and the `initialized` threading
    through `verify_block`/`verify_instruction`.
- `src/backend/emitter.rs`, `src/backend/qbe.rs`, `src/backend/layout.rs`,
  `src/layout.rs` — the pointer and slice emission, the `$emptyslice` data, the
  `:slice` QBE type, `slice_length_offset`, the null and slice-bounds traps, and
  the pointer and slice sizes.
- `docs/spec.md`, `README.md`, `AGENTS.md`, `eng/architecture.md` — documents
  that describe the removed features.

## Decisions

- **Every remaining type is comparable.** `src/comparability.rs` is deleted,
  and both semantic checking and the IR verifier drop their comparability
  checks. The spec says that `==` and `!=` are defined on every type.
- **`value_compatible` is deleted.** Its callers in `semantic/expressions.rs`
  and `ir/verify.rs` compare types with `==`.
- **Every location is rooted in a binding.**
  - `CheckedLocationKind` keeps `Binding`, `Index`, and `Field`, without
    `implicit_dereference`.
  - `check_location` recurses directly through groupings, indexes, and fields.
    It has no `record` flag and no fallback to inferring a non-location operand.
  - `location_binding` returns `Idx<Binding>`, and `location_uses_reference` is
    deleted.
  - An immutable target always reports
    "cannot assign to immutable binding `x`".
  - `HeldPlace.root` becomes a `Place`, and `HeldStep` keeps `Index` and
    `Field`.
- **`for … in` walks arrays only.** `IterationSource` collapses into
  `Iteration`, which holds the captured array place and its length.
- **Comparison and logical-not IR values carry no span.** Neither can trap.
  Lowering passes `None`, and `valid_value` requires `span.is_none()` for both.
- **These stay:**
  - The verifier's literal range and type check (`verify_literal`). Lowering
    creates its own literals through `integer`, for loop counters, lengths, and
    increments.
  - `BinaryForm`. The spec requires a trapping compound assignment to name the
    compound operator.
  - The long-line diagnostic renderer.
    `oversized_integer_diagnostics_are_bounded` relies on it to keep a
    200,000-byte line from flooding stderr and trap messages.
- **`null`, `rune`, and `string` are no longer reserved**, matching the lexer
  test that already treats `rune` as an identifier.

## Test plan

- Every remaining unit, snapshot, and integration test passes, and every
  remaining example compiles and runs with its documented exit status.
- Delete the tests that exercise removed behavior:
  - `src/backend/tests/pointers.rs` and `src/backend/tests/slices.rs`.
  - The pointer, slice, null, address-of, dereference, and comparability tests
    in `src/frontend/tests/`, `src/semantic/tests/`, `src/ir/tests/lower.rs`,
    `src/types.rs`, and `tests/compiler.rs`.
  - The parser snapshots
    `fern__frontend__tests__parser__pointer_types_address_of_and_dereference_snapshot.snap`
    and
    `fern__frontend__tests__parser__slice_types_and_slicing_expressions_snapshot.snap`.
- In `src/ir/tests/verify.rs`, delete:
  - `verification_checks_slice_literals_places_and_values`
  - `verification_rejects_malformed_slice_values_and_places`
  - `verification_rejects_equality_on_aggregates_that_contain_slices`
  - `verification_accepts_pointers_to_non_comparable_types`
  - `verification_rejects_non_pointer_nulls_and_indirect_places`
  - `verification_admits_only_nontruncating_pointer_to_uint_conversion`
  - `verification_rejects_an_aggregate_too_large_for_backend_layout`
  - `verification_rejects_loads_before_definite_initialization`
  - `arguments_initialize_the_parameters_before_the_entry_block`
  - `verification_rejects_a_type_naming_a_struct_the_program_does_not_define`
  - `verification_rejects_a_struct_table_whose_fields_form_a_cycle`
  - `verification_limits_deep_acyclic_struct_tables`

  Remove the pointer and slice cases from the tests that remain.
- In `src/ir/tests/verify.rs`, add one case to
  `verification_checks_operation_shapes_and_accepts_independent_shift_counts`:
  a `Comparison` or `LogicalNot` value that carries a span is rejected.
- IR lowering snapshots change only by dropping the comparison and logical-not
  spans. Review those diffs and accept them.
- Integer, float, array, and struct behavior and diagnostics are otherwise
  unchanged.

## Implementation plan

1. **Frontend.**
   - Remove `Token::Null` and the `"null"` entry in `reserved`.
   - Remove `AnnotationKind::Pointer`/`Slice` and
     `ExpressionKind::Null`/`AddressOf`/`Dereference`/`Slice`, with their arms
     in `syntax.rs`'s walkers.
   - Remove `PrefixOperator::AddressOf`/`Dereference`.
   - Remove the `*`/`*const` and `[]`/`[]const` branches of `type_annotation`.
   - Remove the `:` slicing branch of `postfix_expression`, so `[ … ]` is
     always an index.
   - Remove the leading-`*` loop in `token_after_target`.
2. **Types.**
   - Remove `Type::Pointer`, `Type::Slice`, and `value_compatible`, and update
     `scalar`, `Display`, and the `types.rs` tests.
   - Fix the `Type` doc comment so it no longer mentions slices.
   - Delete `src/comparability.rs` and its `mod` line in `src/lib.rs`.
   - Remove the pointer and slice arms in `src/layout.rs`.
3. **Semantic checking.**
   - Remove the constructs listed under Related code.
   - Simplify locations as described under Decisions.
   - Restrict `check_index_step`, `infer_length`, and `for … in` to arrays, and
     reword their diagnostics: "`len` requires an array operand", and
     "`for … in` requires an array".
   - Remove the ", which contains a slice" diagnostic details and the
     comparability branches in `unify_comparison` and `infer_comparison`.
   - Delete `resolve_referenced_type` and `collect_referenced_lengths`.
4. **IR model and lowering.**
   - Remove the IR constructs listed under Related code, including
     `Instruction::Check` and `FlowBuilder::check`.
   - Simplify `HeldPlace`, `HeldStep`, `lower_location`, `element_place`,
     `field_place`, `lower_aggregate_place`, `lower_flow_operand`,
     `lower_length`, `lower_folded_effects`, `capture`, and `Iteration`.
   - Pass `None` as the span of comparison and logical-not values.
   - Update the `Global.values` doc comment to "One entry per scalar".
5. **IR verification.**
   - Remove the pointer, slice, `Check`, and comparability handling.
   - Remove the struct-table, layout, and definite-initialization checks named
     under Related code. `verify` and `verify_function` no longer build
     `Layouts`.
   - Require no span on `Comparison` and `LogicalNot`.
6. **Backend.**
   - Remove pointer and slice handling from `emitter.rs`, `qbe.rs`, and
     `backend/layout.rs`. That covers literals, loads, stores, aggregate
     equality arms, `AddressOf`, pointer conversion, pointer comparison, the
     slice values, `Place::Indirect`/`SliceElement` addressing,
     `$emptyslice`, `:slice`, and `slice_length_offset`.
7. **Tests, examples, and fixtures.**
   - Apply the Test plan's deletions and update the remaining snapshots.
   - Delete `examples/pointers.fern`, `examples/slices.fern`,
     `tests/fixtures/programs/pointers.fern`, and
     `tests/fixtures/programs/slices.fern`, along with their tests in
     `tests/compiler.rs`.
8. **Documents.** Apply the Documentation updates. Delete `eng/todo.md`,
   `docs/integers.md`, and `eng/string-design-options.md`.
9. **Superseded plans and reviews.**
   - Delete `eng/plans/2026-09-10-013-*`, `eng/plans/2026-09-10-014-*`, and
     `eng/plans/2026-09-11-001-*` through `eng/plans/2026-09-11-013-*`.
   - Delete `eng/reviews/2026-09-11-001-*` through
     `eng/reviews/2026-09-11-005-*`.

## Documentation updates

### `docs/spec.md`

**Introduction**
- Make every invalid program rejected during compilation, with no exceptions,
  by removing the Pointer validity clause.
- Delete the TODO sentence.

**Lexical Structure**
- Keywords:
  - Drop `null` from the reserved words.
  - Drop `rune` and `string` from the reserved type names.
- Delete the Escape sequences, Character literals, and String literals
  sections.

**Declarations**
- Variable declarations:
  - Remove the pointer-conversion exception.
  - Untyped constants default to `int`, `f64`, and `bool`.
  - Zero values are `0`, `false`, and recursive zero values for arrays and
    structs.
- Type declarations:
  - Remove `type Name T;`, its paragraph, and "Other type declarations end with
    `;`".
  - Remove the sentence saying a pointer or slice field breaks a cycle.
  - Remove the struct comparability sentences, so structs compare field by
    field with no comparability condition.

**Integer Semantics**
- Integer types: replace "`uint` is the type of allocation sizes and of
  integers converted from pointers" with "`uint` is the unsigned integer of the
  same width".
- Indexing and lengths:
  - The opening refers only to [Arrays](#arrays).
  - Lengths and indices are `int`.
  - Drop the slice-bounds wording and the allocation-size sentence.
- Comparison: replace the comparability paragraphs with "`==` and `!=` are
  defined on every type", and refer to Array comparison and Struct literals for
  aggregates.
- Assignment:
  - A location is a binding, a field of a location, or an element of an array
    location.
  - A mutable location is a `var` binding or a field or element of one.
  - Every index and field-selection expression is evaluated left to right, then
    `e`.
  - Remove `*p = e;`, the slice-binding sentence, and "addressable".
- Summary of trapping behavior: remove the constant slice-bound,
  integer-to-rune (both rows), slice-bounds, and null-dereference rows.

**Boolean Semantics**
- Say that Comparison produces `bool`, without "comparable operands".

**Removed sections**
- Delete Rune Semantics, Pointers, Slices, and Strings.

**Arrays**
- Remove the "does not implicitly become a slice" sentence and the
  implicit-dereference clauses in Indexing and Length.
- `len` takes an array operand.
- Remove the array comparability paragraph.

**Loops**
- `for … in` takes an array.
- Remove the empty-slice sentence, the slice-copy sentence, and the string
  paragraph and its example.

### `README.md`

- Remove the TODO and "features ahead of the compiler" sentences.
- Remove the Pointers and Slices entries.

### `AGENTS.md`

- Remove the `eng/string-design-options.md` rows from Project Documents and
  the Codebase Map.
- Add `src/layout.rs` to the Codebase Map as the shared value layout.

### `eng/architecture.md`

- Delete the `comparability` paragraph.
- In the `layout` paragraph, say that semantic validation and the backend read
  the derivation; IR verification no longer does.
