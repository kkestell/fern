# Restrict equality to comparable types across the compiler

## Sources

- `docs/spec.md#comparison` — `==` and `!=` require comparable operand types.
- `docs/spec.md#slice-comparison` — a slice is not comparable, and the loop
  that replaces `==` on slices.
- `docs/spec.md#array-comparison` — an array is comparable when its element
  type is comparable.
- `docs/spec.md#struct-literals` — a struct is comparable when every field is.
- `eng/todo.md#comparability` — the single subtask and its parent task.
- `eng/architecture.md#storage-and-identity` — `layout` derives one answer per
  phase-supplied struct table; comparability follows that shape.
- `src/layout.rs` — the `StructFields` trait every phase already implements
  (`src/semantic/model.rs:369`, `src/ir/verify.rs:240`,
  `src/backend/layout.rs:23`).

## Goal

Make comparability a checked rule: the semantic phase rejects `==` and `!=` on
a slice, on an array of slices, and on a struct that reaches a slice through
its fields; IR verification rejects the same operands; and the backend drops
the slice equality helper it can no longer reach. Starting state is the
current tree, where every type compares. This completes the Comparability
parent task.

## Implementation

### The shared predicate

- `src/comparability.rs` — new module with one item:

  ```rust
  pub(crate) fn comparable(table: &dyn StructFields, ty: &Type) -> bool
  ```

  `Scalar` and `Pointer` are comparable; `Slice` is not; `Array` is comparable
  when its element is; `Struct` is comparable when every field, read by
  ordinal from `table`, is. Do not descend through a pointer's target, so the
  walk terminates. No depth guard: struct value containment is already
  acyclic and bounded by `resolve_struct_fields`
  (`src/semantic/annotations.rs:151`) and `verify_acyclic`
  (`src/ir/verify.rs:387`), which both run before any caller here.
- `src/lib.rs` — declare `mod comparability;` in the existing alphabetical
  module list.
- `eng/architecture.md` — in Storage and identity, after the `layout`
  paragraph, add one sentence: `comparability` answers whether `==` and `!=`
  are defined on a type from the same phase-supplied struct table, so semantic
  checking and IR verification enforce one rule.

### Semantic checking

- `src/semantic/expressions.rs`, `unify_comparison`:
  - Before any other work, when `operator.is_equality()`, reject the first
    operand type that is not comparable, left then right. The message is
    ``` `==` and `!=` are not defined on `[]int` ``` when the operand type is
    itself a `Type::Slice`, and
    ``` `==` and `!=` are not defined on `Holder`, which contains a slice ```
    otherwise. Both carry `operator_span`.
  - Delete the whole `Type::Slice` arm. Slices reaching the general path keeps
    the existing messages: two slices of different element types or constness
    report `comparison operands have different types ...`, and `<` on two
    identical slice types reports ``` only `==` and `!=` are defined on ... ```
    from the trailing aggregate check.
  - Leave the `Type::Pointer` arm as it is; pointers stay comparable across
    constness.
- `src/semantic/expressions.rs`, `infer_comparison` — narrow
  `reference_comparison` to `matches!(checked_left.ty, Type::Pointer { .. })`
  and reword its binding or comment accordingly; a slice can no longer reach
  constant folding.
- `src/semantic/constants.rs:628` — the doc comment on the constant comparison
  helper says equality reaches every value type. Correct it to say equality
  reaches every comparable type.
- `src/types.rs`, `ComparisonOperator::is_equality` — the doc comment says
  equality is defined on every value type. Correct it to name comparable
  types, referring to the specification's rule rather than restating it.

### IR verification

- `src/ir/verify.rs`, `valid_value` — add a `table: &dyn StructFields`
  parameter and pass `self` at the call site (`src/ir/verify.rs:601`). In the
  `ValueKind::Comparison` equality branch, delete the `Type::Slice` pair
  allowance and require `comparability::comparable(table, &left)` in addition
  to the existing type agreement. Pointer operands keep their constness-free
  pair rule. A rejected value reports the existing `invalid IR value {id}`
  error; no new message.

### Native emission

- `src/backend/qbe.rs`:
  - `emit_aggregate_equal` — replace the `Type::Slice` arm with
    `unreachable!("a slice is not comparable")`, in the style of `scalar` in
    the same file.
  - Delete `slice_equality` entirely.
  - Delete the `helpers` and `slice_equalities` fields from `Emitter`.
- `src/backend/emitter.rs` — delete the `helpers` and `slice_equalities`
  initializers (lines 224-225) and the `+ &emitter.helpers` term from the
  assembly `emit` returns (line 302).

### Programs that compared slices

- `tests/fixtures/programs/slices.fern` — replace the slice comparisons on
  line 67 with a module-level helper in the shape of the specification's
  `#slice-comparison` example, `fn equal(a: []const int, b: []const int) ->
  bool`, comparing lengths then elements. Keep the same cases the line covered
  — equal contents, differing contents, two empty slices, and a `[]const int`
  operand — and keep `exit(6)` as the failing status and `exit(44)` on
  success. The `Holder` and `Node` declarations stay; they now also stand for
  structs that hold a slice and so are not comparable.
- `examples/slices.fern` — line 30 compares slices. Add the same `equal`
  helper above `main`, use it for the copied-slice check, and note in a
  comment that `==` is not defined on a slice because its length is part of
  its value.
- `README.md:31` — the Slices bullet ends with `and comparison`. Replace that
  with elementwise comparison written as a loop.

## Tests

- `src/semantic/tests/expressions.rs`:
  - In `slices_index_measure_and_compare_without_folding`, drop the accepted
    `s == t` case and change the `[]int` versus `[]i64` rejection to the new
    non-comparable message; the `<` rejection stays as it is.
  - Reject `==` and `!=` on: two `[]int`, `[]int` against `[]const int`, two
    values of a struct with a slice field, two values of a struct whose field
    is a struct with a slice field, and two `[2][]int` arrays. Each reports
    the operand's own type in the message.
  - Accept `==` on a struct of scalars, arrays, pointers, and nested structs,
    and on an array of such structs, so the walk is shown not to over-reject.
    `structs_compare_only_for_equality_and_only_with_their_own_type` and
    `arrays_compare_for_equality_only` already cover the simple shapes.
- `src/ir/tests/verify.rs`:
  - `verification_checks_slice_literals_places_and_values` — remove the slice
    equality value and its comment from the valid program, and add slice
    equality to the rejected values beside the existing mismatched-element
    case.
  - Reject equality on an array of slices and on a struct with a slice field,
    using the struct table these tests already build.
  - Keep the existing array and pointer equality acceptances unchanged.
- `src/backend/tests/slices.rs` — delete
  `native_slice_equality_compares_elements_and_emits_one_helper_per_element_type`.
- `src/backend/tests/structs.rs` — in an existing struct or array equality
  emission test, assert the emitted QBE contains no `$sliceequal`.
- `tests/compiler.rs` — `slice_programs_compile_and_execute` and
  `documented_examples_compile` cover the rewritten fixture and example with
  no change to those tests.

## Decisions

- The predicate lives in its own module rather than in `src/types.rs`, because
  it needs a struct table and `src/types.rs` is what `src/layout.rs` and its
  `StructFields` trait are built on. It reuses `StructFields` rather than
  introducing a second table trait.
- The semantic check runs before operand types are unified, so a program that
  compares two slices is told that slices are not comparable rather than that
  `[]int` and `[]const int` are different types.

## Extra validation

This completes the Comparability task. Run `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, and `cargo test`, and confirm no
`sliceequal`, `slice_equalities`, or `helpers` emitter state remains.
