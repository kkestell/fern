# Represent slices in Fern IR

## Sources

- `docs/spec.md#slices` — slice values and their copying, slicing expressions
  with their operand forms, evaluation order, and trapping bounds, slice
  indexing, `len`, and comparison.
- `docs/spec.md#loops` — `for … in` over a slice.
- `eng/todo.md#slices` — the fourth Slices subtask. Native emission and the
  integrated slice programs remain after it.
- `eng/architecture.md#storage-and-identity` — only verified IR reaches the
  backend, and `layout` is the one derivation of memory. It already sizes a
  slice as two `uint` words.
- `eng/plans/2026-09-11-009-check-slice-values-bounds-length-comparison-and-iteration.md`
  — the checked forms this lowering consumes: `ExpressionValue::Slice`,
  `CheckedLocationKind::SliceValue`, `Constant::EmptySlice`, and slice
  `Index`, `Length`, `Comparison`, and iteration.
- `eng/plans/2026-09-11-002-represent-pointers-and-indirect-access-in-fern-ir.md`
  — the pattern this follows for a value the IR carries before the backend
  emits it, and for moving the staging diagnostic after verification.
- `src/ir/model.rs`, `src/ir/lower.rs`, `src/ir/verify.rs` — the existing
  literals, places, value kinds, lowering order, and verification boundaries.

## Goal

Lower and verify every semantically valid slice value, slice location, slice
`len`, slice comparison, and slice iteration into Fern IR, including
slice-bearing globals and aggregates. The `slices are not yet implemented`
diagnostic moves after IR verification, so the backend becomes the only
remaining boundary and this does not complete the Slices parent task.

## Implementation

### `src/ir/model.rs`

- `Literal::EmptySlice(Type)` — the empty slice, carrying its concrete slice
  type as `Null` carries its pointer type. Reword the enum's doc comment: a
  literal is an immediate value no instruction computes, and this one occupies
  the two words a slice occupies rather than one scalar. `Literal::ty` returns
  the carried type.

- `Place::SliceElement { slice: Operand, index: Operand, span }` — the element
  of the slice value `slice` at `index`. A slice's elements are not inside the
  slice, so this indexes a value instead of descending into a base place, the
  way `Indirect` dereferences a pointer operand. The span is the one the
  bounds-check trap reports; the length it checks against is part of the
  slice's value. `Place::root_local` returns `None` for it, so a store through
  it initializes no local and a load through it requires none.

- `ValueKind::WholeSlice(Place)` — the slice covering every element of an
  array place. Its base is that place's address and its length comes from the
  array's type, so it cannot trap and its `Value::span` stays `None`.

- `ValueKind::SliceRange { slice: Operand, low: Operand, high: Operand }` —
  the slice of `slice`'s elements from `low` up to `high`, trapping when the
  bounds are out of range. `Value::span` is the slicing expression's span,
  which that trap reports.

- `ValueKind::SliceLength { slice: Operand }` — the length part of a slice
  value, of type `int`. It cannot trap, so its `Value::span` stays `None`.

- `Global::values` doc comment: one entry per scalar, pointer, or slice.

### `src/ir/verify.rs`

- `verify_literal` — an `EmptySlice` whose type is not a slice type is an
  internal error, as a non-pointer `Null` is.

- `verify_type` — replace the slice rejection with recursion into the element
  type. Verified IR now carries slice types.

- `scalar_types` and `scalar_count` — a slice contributes one entry holding
  its own type, as a pointer does. Replace both `unreachable!`s.

- `place_type` — `SliceElement` requires a slice-typed `slice` operand and an
  `int` index, and reaches the element type. Route both operands through
  `operand_type`, which also enforces that each is defined earlier in the
  block.

- `valid_value`:
  - `WholeSlice(place)` — the place must be `Type::Array`, the value's type
    must be a slice of that array's element type, and `value.span` must be
    `None`.
  - `SliceRange { slice, low, high }` — `low` and `high` must be `int`, the
    slice operand's type must be `value_compatible` with the value's type
    (which is a slice type), and `value.span` must be `Some`. Using
    `value_compatible` is what stops a `[]const T` source from producing a
    `[]T` result while allowing the reverse.
  - `SliceLength { slice }` — the operand must be a slice type, the value's
    type must be `int`, and `value.span` must be `None`.
  - `Comparison` — the equality branch also accepts two slice types whose
    element types are equal, whatever their constness, beside the existing
    exact-type and pointer-target cases. The ordering branch still requires a
    scalar, which rejects `<` on slices.

- Nothing else changes: `contained_struct` already reports no containment
  through a slice, `Instruction::Store`, `verify_call`, and `verify_return`
  already compare with `value_compatible`, and `layout` already sizes and
  aligns a slice.

### `src/ir/lower.rs` — values

- `slice_literal(constant, ty) -> Literal` beside `pointer_literal`, mapping
  `Constant::EmptySlice` to `Literal::EmptySlice(ty)` and anything else to
  `unreachable!`. Reword `scalar_literal`'s `Constant::EmptySlice` arm to say
  an empty-slice constant has a slice type.

- `flatten` and `store_constant` — a `(Type::Slice { .. }, constant)` arm
  pushing or storing `slice_literal`, beside the existing pointer arms. This is
  what fills a module-level slice binding, a declaration without an
  initializer, and a slice field or element of an aggregate zero value.

- `lower_slicing(checked, id, bindings, builder) -> Operand` — lowers one
  `ExpressionValue::Slice { operand, low, high, implicit_dereference }`:
  - Read the operand's checked type and, when `implicit_dereference`, its
    pointer target, to learn whether the operand is an array or a slice.
  - Build the source slice operand:
    - array operand — the array's place, which is
      `Place::Indirect { pointer, span }` over the lowered operand when
      `implicit_dereference` and `lower_aggregate_place(operand)` otherwise,
      wrapped in a `WholeSlice` value.
    - slice operand — `load_place(Place::Indirect { .. })` when
      `implicit_dereference`, and `lower_flow_operand(operand)` otherwise.
      `lower_flow_operand` already covers a reference, an element, a field, a
      call result, and a nested slicing expression.
  - Hold the source operand, then lower `low` and `high` in that order, holding
    `low` across `high`, so a call in either bound cannot leave an earlier
    operand unreadable. Source order is evaluation order.
  - An omitted `low` is `integer(0, Scalar::Int)`. An omitted `high` is the
    array's length as an `int` literal for an array operand, and a
    `SliceLength` of the source slice for a slice operand.
  - Return the `SliceRange` value, typed with the checked expression's type and
    spanned with the slicing expression's span.

- `lower_flow_operand` — a slice branch beside the pointer branch, before the
  `expression.ty.scalar()` fall-through:
  - a folded constant lowers its effects and returns
    `Operand::Literal(slice_literal(..))`, which is how a `const` slice
    binding's use sites and `null`-free zero values arrive;
  - `ExpressionValue::Slice { .. }` returns `lower_slicing`;
  - every other slice value has a place, so fall through to the existing
    aggregate path, which loads it.

- `lower_aggregate_place` — an `ExpressionValue::Slice { .. }` arm storing
  `lower_slicing` into a fresh slice local and returning that place. A
  slicing expression has no storage of its own, and this is the path a grouped
  or iterated slicing expression reaches.

- `element_place` — when the operand's reached type is a slice, return
  `Place::SliceElement { slice, index, span }` instead of `Place::Element`,
  where `slice` is the loaded slice value (through `Place::Indirect` when
  `implicit_dereference`) and `span` stays the index expression's span. The
  operand is still lowered before the index.

- `lower_length` — when the operand's reached type is a slice, yield a
  `SliceLength` value over the slice operand instead of the type's length. With
  an implicit dereference the slice comes from a load through
  `Place::Indirect`, which is also that access's null check, so no separate
  `Instruction::Check` is emitted. The array branch is unchanged.

- `lower_folded_effects` — replace the staging `unreachable!` on
  `ExpressionValue::Slice` with `unreachable!("a slicing expression is never
  constant")`. A folded `len` reaches a slicing operand through
  `lower_flow_operand`, not through this walk.

### `src/ir/lower.rs` — locations

- `HeldStep::SliceIndex { slice: HeldOperand, index: HeldOperand, span }` —
  `HeldPlace::read` rebuilds it as `Place::SliceElement`. It carries everything
  the place needs, so it appears as a held place's only step.

- `lower_location`'s `Index` arm — compute the reached type from `operand.ty`,
  dereferenced through `implicit_dereference`. When it is a slice, obtain the
  slice value, hold it, lower and hold the index, and return a fresh
  `HeldPlace { root: None, steps: vec![HeldStep::SliceIndex { .. }] }`. The
  earlier steps have already been evaluated by the time the slice value is
  loaded, so discarding them keeps evaluation order and single evaluation. The
  slice value comes from:
  - `lower_location_slice(operand)` when there is no implicit dereference — a
    new helper that lowers `CheckedLocationKind::SliceValue { operand }` with
    `lower_flow_operand`, and every other location by reading its held place
    and loading it. This is the only place a checked slice-value location is
    consumed;
  - otherwise, the operand's held place read and loaded for its pointer, then
    loaded again through `Place::Indirect { pointer, span }`.
  Leave the array path as it is.

- `lower_location`'s `SliceValue` arm becomes
  `unreachable!("a slice value is only the operand of an index step")`, which
  `lower_location_slice` is what makes true.

- Assignment and compound assignment need no change: they already hold the
  target before the value and read it back, and `checked.assignments[..].ty`
  already gives the element type a compound operator works on.

### `src/ir/lower.rs` — iteration

- Replace `Iteration::array` and `Iteration::length` with a source that
  distinguishes the two operand kinds, keeping `element`, `counter`, `value`,
  and `span`:
  - an array place with the static length from its type;
  - a local holding the captured slice value, with its slice type.
- `Iteration::condition` compares the counter against that static length for an
  array and against a `SliceLength` of the captured slice for a slice. Reading
  the length from the captured local each iteration is equivalent to reading it
  once, because nothing writes that local after the capture; the elements a
  slice refers to may change through it, but its base and length cannot.
- `Iteration::bind` loads `Place::Element` for an array and
  `Place::SliceElement` over the captured slice for a slice.
- `capture` lowers a slice operand with `lower_flow_operand` and stores it into
  a fresh slice local, rather than taking a place and loading it. The copy is
  the slice value, so assigning through the slice inside the body is visible to
  the remaining iterations while assigning to the original binding is not.

### `src/lib.rs`

- Move the `semantic::expressions::reject_slice_values(&syntax)` call after
  `ir::lower::lower(checked).verify()?` and before
  `backend::toolchain::build`. Lowering and verification then run on every
  valid slice program, and the next task receives a verified slice IR contract.

### `src/backend/`

- The new IR forms reach the backend's exhaustive matches, which must keep
  compiling without emitting them: `Place::SliceElement` in `place_address`,
  the three new value kinds in the value emitter, and `Literal::EmptySlice`
  wherever `Literal::Null` already appears in `src/backend/emitter.rs` and
  `src/backend/qbe.rs`. Each is `unreachable!("native slice emission is the
  next Slices task")`.
- Reword the eight existing `Type::Slice` arms in `src/backend/layout.rs`,
  `src/backend/emitter.rs`, and `src/backend/qbe.rs`: verified IR now carries
  slice types, and what keeps them out of the backend is the staging
  diagnostic. Name that in each message.

## Tests

`src/ir/tests/mod.rs` — a `slice_type(constant, element)` builder, an
`EmptySlice` literal helper, a `Place::SliceElement` arm in `describe`, and an
`EmptySlice` arm in `describe`'s index match.

`src/ir/tests/lower.rs`:

- `a[1:3]` over an array local lowers to a `WholeSlice` of that local and a
  `SliceRange` over it; `a[:]` uses `0` and the array's length; `a[1:]` fills
  only the high bound; the result stores into a slice local.
- Slicing a slice binding lowers to a `SliceRange` over a load of that local
  with no `WholeSlice`; an omitted high bound becomes a `SliceLength`.
- `p[1:2]` on `*[4]int` builds a `WholeSlice` over an `Indirect` place, and on
  `*[]int` a `SliceRange` over a load through one.
- Slicing a call result of slice type, and `a[1:3][0:1]`, lower without
  materializing a place for the inner slice.
- With a call in the operand, the low bound, and the high bound, the calls are
  emitted in that order and each earlier operand is still readable in the block
  the `SliceRange` lands in.
- `s[i]` loads through a `SliceElement` whose slice operand is a load of the
  slice local and whose span is the index expression's.
- `s[i] = f();` and `s[i] += f();` lower the slice and the index before the
  call and read both back in the block the call leaves; `f()[0] = 1;` and
  `a[1:3][0] = 1;` store through a `SliceElement` over the lowered slice value;
  `p[0] = 1;` on `*[]int` stores through one over a load of an `Indirect`
  place.
- `&s[0]` is an `AddressOf` of a `SliceElement`.
- `len(s)` is a `SliceLength`; `len(p)` on `*[]int` reads it through an
  `Indirect` load; `len(a[1:3])` reads it from the `SliceRange`;
  `len(a[1:3][0])` on `[4][3]int` folds to `3` and still emits the slicing
  bound check; `len(a)` still folds with no instructions.
- `s == t` and `s != t` lower to a `Comparison` of two slice operands,
  including `[]int` against `[]const int`.
- `for v in s` copies the slice into a local once before the loop, reads its
  length in the condition block, binds the element through a `SliceElement` at
  the counter, and advances in the post block even when the body `continue`s;
  `for v, i in s` binds the index to the counter; `for v in a[1:3]` captures
  the constructed slice; `for v in *p` over `*[]int` captures through an
  `Indirect` load.
- A module-level `var s: []int;` becomes a global holding one `EmptySlice`
  literal; a global struct with a slice field and a global `[2][]int` hold one
  per slot; a local `var s: []int;` stores the literal; a module-level
  `const s: []int;` folds to the literal at its use sites.
- A slice parameter, a slice result, and a slice argument lower to slice-typed
  locals, operands, and call results, with `[]T` passed where `[]const T` is
  expected; a struct holding a slice field copies as one whole value.
- A `slices` snapshot fixture covering a slicing expression, element
  assignment, `len`, comparison, and `for … in` over a slice.

`src/ir/tests/verify.rs` — malformed IR is rejected: a `WholeSlice` over a
non-array place, over an array whose element type differs from the result's,
and one carrying a span; a `SliceRange` whose source is not a slice, whose
element type differs, that widens `[]const T` to `[]T`, whose bound is not
`int`, or that carries no span; a `SliceLength` of a non-slice or with a
non-`int` result; a `SliceElement` whose slice operand is not a slice, whose
index is not `int`, whose loaded type is not the element type, or whose slice
operand is defined later in the block; an `EmptySlice` literal with a
non-slice type; a global whose slice slot holds another literal or whose value
count disagrees with a struct-with-slice-field type; `<` on two slices;
equality between slices with different element types and between a slice and an
array. A store of `[]T` into a `[]const T` place verifies and the reverse does
not, and a struct that reaches itself through a slice field verifies and lays
out.

`tests/compiler.rs` — the two slice staging tests keep their assertions and
gain names naming native emission as what remains, since the program they
compile now lowers and verifies before the diagnostic is returned.

## Decisions

- **A slice is a two-word value the IR carries, not a desugared pair.**
  `layout` already sizes and aligns `Type::Slice`, so copying, passing,
  returning, and storing a slice need no new instruction: the existing whole
  value `Load` and `Store` carry it as they carry an array or a struct. Only
  construction, projection, and element access are new.

- **Every slicing expression goes through one bound check.** An array operand
  becomes a `WholeSlice` first, so `SliceRange` always checks
  `0 <= low <= high <= length` against a length that is part of a slice value.
  The redundant check on `a[:]` matches the checked element read inside
  `for … in`: a second unchecked path would be a flag for one caller.

- **The bound trap reports the slicing expression's span, not the offending
  bound's.** One check covers both bounds, so one span keeps one trap and one
  message naming the slice type. The compile-time diagnostics, which can name
  the offending bound, already cover the constant cases the checker rejects.

- **`Place::SliceElement` indexes a value rather than descending a place.** A
  slice's elements live outside it, so an element place needs the slice's base
  and length, not the address of the storage holding the slice. That also makes
  a slice value with no location — a call result or a slicing expression —
  indexable without materializing it, which is what the checked
  `SliceValue` location asks for.
