# Check slice values, bounds, length, comparison, and iteration

## Sources

- `docs/spec.md#slicing-expressions` — operand forms, result constness, bound
  typing and range, evaluation order, and the non-constant rule.
- `docs/spec.md#slice-indexing`, `#slice-length`, `#slice-comparison` — element
  access and its mutability, `len`, and equality across constness.
- `docs/spec.md#assignment` — a slice element is a location whose mutability
  comes from the slice's type alone.
- `docs/spec.md#loops` — `for v in a` and `for v, i in a` over a slice.
- `docs/spec.md#indexing-and-lengths`, `#implicit-dereference` — bound and index
  typing, and the one pointer indirection slicing, indexing, and `len` accept.
- `eng/todo.md#slices` — the third Slices subtask; IR and native support remain
  later work.
- `eng/plans/2026-09-11-008-represent-slice-types-across-the-compiler.md` — the
  resolved `Type::Slice`, `Constant::EmptySlice`, and the staging guard this
  task widens.

## Goal

Check slicing expressions, slice element access and assignment, `len`,
equality, and iteration against the resolved slice types. Slice IR and native
execution remain unfinished, so compilation still stops after a successful
semantic check with one staging diagnostic, now covering slicing expressions as
well as slice annotations.

## Implementation

### Slicing expressions

- `src/semantic/model.rs` — add
  `ExpressionValue::Slice { operand: Idx<Expression>, low: Option<Idx<Expression>>,
  high: Option<Idx<Expression>>, implicit_dereference: bool }`, matching
  `Index`'s shape. The operand's checked type is recorded, so lowering reads the
  operand kind and an array operand's length back from it.

- `src/semantic/expressions.rs` — add `LocationUse::Slice` with the message
  `cannot slice an expression that is not a location`. An array that is not a
  location cannot be sliced, so a slicing operand is checked through the same
  location path as an index target.

- `src/semantic/expressions.rs` — add `infer_slice(operand, low, high, scopes)`
  and dispatch `ExpressionKind::Slice` to it from `infer_expression`, replacing
  the temporary diagnostic:
  - Call `self.location_step_operand(operand, scopes, true, LocationUse::Slice)`.
    It already dereferences one pointer level and reports the operand's own
    diagnostics, so `p[1:2]` on `*[3]int` and on `*[]int` both arrive here
    normalized.
  - From the step's reached type take the element type, the result's constness,
    and the operand length the compile-time bound check uses:
    - `Type::Array { length, element }` — element, `constant: !mutable`,
      length `Some(length)`.
    - `Type::Slice { constant, element }` — element, that same `constant`,
      length `None`.
    - anything else — `Err` on the operand's span,
      ``format!("cannot slice `{ty}`")``, naming the reached type as
      `check_index_step` names it.
  - Check `low` then `high`, each present bound with
    `check_expression(bound, scopes, Some(Scalar::Int.into()))`, and insert each
    into `self.expressions`. Source order is evaluation order.
  - When the length is known, reject a folded bound outside `0..=length` on that
    bound's span with
    ``format!("slice bound {value} is out of range for `{ty}`")``, and, when both
    bounds folded and `low > high`, reject on the high bound's span with
    ``format!("slice lower bound {low} exceeds upper bound {high}")``.
  - Result: `Type::Slice { constant, element }`, `untyped: false`,
    `constant: None`, and the new `ExpressionValue::Slice`.

- `src/semantic/expressions.rs` — remove the `ExpressionKind::Slice` arm of
  `check_location_with_facts`. A slicing expression is not a location, so it
  falls to the existing `_` arm and reports the non-location message for its
  use.

### Slice element locations

- `src/semantic/model.rs` — add
  `CheckedLocationKind::SliceValue { operand: Idx<Expression> }`: a slice value
  that is not itself a location, such as a call result or a slicing expression.
  Its elements are locations even though it is not, so an index step may start
  from it.

- `src/semantic/expressions.rs` — in `location_operand`, accept a slice-typed
  non-location the way the pointer fallback already accepts a pointer-typed one:
  build `CheckedLocationKind::SliceValue { operand: id }` with the slice type and
  `mutable: false`, which the index step below replaces, and insert the checked
  expression regardless of `record`, since the location tree is what lowering
  walks for an assignment target.

- `src/semantic/expressions.rs` — give `check_index_step` a
  `Type::Slice { element, .. }` arm yielding the element type. It still checks
  the index against `int`, and it performs no constant range check: a slice's
  length is part of its value, not its type. Say that in its doc comment.

- `src/semantic/expressions.rs` — in the `ExpressionKind::Index` arm of
  `check_location_with_facts`, when the step's reached type is
  `Type::Slice { constant, .. }`, replace the step's mutability with `!constant`.
  A slice element's mutability comes from the slice's type alone, so a `const`
  slice binding still permits `s[i] = e;` and a `var` binding of `[]const T`
  does not.

- `src/semantic/statements.rs` — rename `location_uses_pointer` to
  `location_uses_reference` and make it true for a step whose operand location
  has a slice type, and for `CheckedLocationKind::SliceValue`. That keeps
  `cannot assign to immutable binding` for bindings whose own storage is the
  target and leaves `cannot assign to an immutable location` for a write through
  a `[]const T`. Add a `SliceValue` arm returning `None` from
  `location_binding`.

### Length, comparison, and iteration

- `src/semantic/expressions.rs` — `infer_length` accepts `Type::Slice` beside
  `Type::Array`, yielding `int`. A slice operand never folds, so only the array
  branch computes a constant. Reword its diagnostic to
  ``` `len` requires an array or slice operand, found `{ty}` ```.

- `src/semantic/expressions.rs` — in `unify_comparison`, add a slice branch
  beside the pointer branch: two slices unify when their element types are
  equal, whatever their constness, and only `==` and `!=` are defined on them.
  Reuse the two existing messages. An array against a slice keeps falling
  through to the different-types message.

- `src/semantic/expressions.rs` — in `infer_comparison`, widen the
  `pointer_comparison` guard that suppresses folding to cover slice operands and
  rename it accordingly. A comparison of two slices is never constant, even when
  both operands are the empty-slice zero value.

- `src/semantic/statements.rs` — `check_iteration_header` accepts
  `Type::Slice { element, .. }` beside `Type::Array`, binding `v` to the element
  type and `i` to `int` as it does today. Do not extend implicit dereference to
  iteration. Reword its diagnostic to
  ``` `for … in` requires an array or slice, found `{ty}` ```.

### Staging guard

- `src/semantic/expressions.rs` — move the guard here from
  `src/semantic/annotations.rs` as
  `pub(crate) fn reject_slice_values(syntax: &Syntax) -> Result<(), Diagnostic>`,
  and widen it to scan `syntax.expressions` for `ExpressionKind::Slice` as well
  as `syntax.annotations` for `AnnotationKind::Slice`, reporting
  `slices are not yet implemented` at the earliest span across both, ordering by
  `span.start` and breaking a tie by the larger `span.end`. Every slice value in
  a checked program comes from a written `[]T` or a slicing expression, so the
  two forms are the whole guard.

- `src/semantic/mod.rs` and `src/lib.rs` — make `expressions` `pub(crate)`,
  restore `annotations` to private, and call
  `semantic::expressions::reject_slice_values` where the annotation guard was
  called, between `semantic::namespaces::check` and `ir::lower::lower`.

- `src/ir/lower.rs` — add arms to the exhaustive checked-model matches:
  `ExpressionValue::Slice` in `lower_folded_effects`, and
  `CheckedLocationKind::SliceValue` in the place lowering at `src/ir/lower.rs:519`,
  each `unreachable!("slices stop before IR lowering")`. The guard keeps both out
  of lowering.

## Tests

- `src/semantic/tests/expressions.rs` — slicing result types: `a[1:3]`,
  `a[1:]`, `a[:3]`, and `a[:]` over `var a: [4]int` give `[]int`; over
  `const a: [4]int` and over an array parameter they give `[]const int`;
  `p[1:2]` on `*[4]int` gives `[]int` and on `*const [4]int` gives
  `[]const int`; slicing a `[]int` gives `[]int`, a `[]const int` gives
  `[]const int`, and `*[]const int` gives `[]const int`; `s[1:3][0:1]`,
  `grid[0][1:2]` on `[2][3]int`, and a slice of `[2][3]int` elements give the
  expected element and slice types.

- `src/semantic/tests/expressions.rs` — rejections: slicing a call result of
  array type, an array literal, and an arithmetic result each report
  `cannot slice an expression that is not a location`; slicing an `int` binding
  and a struct binding report ``cannot slice `int` `` and the struct's
  spelling; slicing a call result of slice type and of `*[3]int` type is
  accepted.

- `src/semantic/tests/expressions.rs` — bounds: `a[u:2]` with a `u8` binding
  reports the implicit-conversion diagnostic; `a[1:true]` is rejected;
  `a[4:]`, `a[:5]`, and `a[-1:]` on `[4]int` report the out-of-range bound with
  its span; `a[4:4]` and `a[0:0]` are accepted; `a[3:1]` reports the
  out-of-order bounds; `a[i:j]` with runtime bounds is accepted; the same
  out-of-range bounds on `*[4]int` are rejected, and on a `[]int` operand every
  one of them is accepted.

- `src/semantic/tests/expressions.rs` — indexing and `len`: `s[0]` on `[]int` is
  `int` and on `[][3]int` is `[3]int`; `s[9]` is accepted with no compile-time
  range check; `s[u]` with a `u8` binding is rejected; `len(s)` is `int`;
  `len(s)` is not constant, shown by a module-level `const n = len(s);` over a
  module-level slice binding reporting the non-constant initializer;
  `[len(s)]int` reports `array length must be a constant expression`;
  `len(1)` and `for v in 1` report the reworded diagnostics.

- `src/semantic/tests/expressions.rs` — comparison: `s == t` and `s != t` on
  `[]int` and `[]const int` in either order are `bool`; `[]int` against
  `[]i64`, against `[3]int`, and against `int` report the different-types
  message; `s < t` reports ``only `==` and `!=` are defined on `[]int` ``; a
  module-level `const b = s == t;` over two module-level slice bindings reports
  the non-constant initializer.

- `src/semantic/tests/statements.rs` — element assignment: `s[0] = 1;` checks
  for both `var s: []int` and `const s: []int`; `t[0] = 1;` on `[]const int`
  reports `cannot assign to an immutable location` for both `var` and `const`
  bindings; `f()[0] = 1;` and `a[1:3][0] = 1;` check through the slice-value
  location; `p[0] = 1;` on `*[]int` checks and on `*[]const int` does not;
  `s[0] += 1;` checks and `s[0] += true;` is rejected; `&s[0]` is `*int` and
  `&t[0]` is `*const int`; `s = t;` assigns `[]int` into `[]const int` and not
  the reverse; `a[:] = s;` reports the non-location assignment message and
  `&a[:]` the non-location address-of message.

- `src/semantic/tests/statements.rs` — iteration: `for v in s` binds `v` to the
  element type, `for v, i in s` also binds `i` to `int`, `for v in a[1:3]` and
  `for v in *p` over `*[]int` check, `v = 1;` in the body reports the
  immutable-binding diagnostic, and `v` is not resolvable after the loop.

- `src/semantic/tests/annotations.rs` and `src/semantic/tests/mod.rs` — replace
  the annotation-only staging helper with one that reaches
  `reject_slice_values`, and keep the existing zero-value and layout
  coverage. Add: a module-level `var s: []int;` and `const s: []int;` hold the
  empty-slice zero value, a struct or array containing a slice field fills with
  it, and a module-level `const s: []int = a[:];` reports the non-constant
  initializer.

- `src/semantic/tests/mod.rs` — the staging helper covers a program whose only
  slice syntax is a slicing expression, one whose only slice syntax is an
  annotation, and one with both, asserting the earliest span is reported.

- `tests/compiler.rs` — a program that slices an array, indexes and assigns
  through the slice, and iterates it fails with `slices are not yet implemented`
  and preserves an existing output file.

## Decisions

- The compile-time bound check runs only when the operand's length is part of
  its type, so `s[3:1]` on a slice traps at runtime while `a[3:1]` on an array
  is rejected. Both bounds are constant in the out-of-order case, so it is a
  constant bound violation like the out-of-range case.

- Slicing reuses the location path rather than plain inference, because the
  result's constness comes from the operand location's mutability and because
  an array that is not a location cannot be sliced at all. A slice operand
  needs no location, which is what `CheckedLocationKind::SliceValue` records.

- Slice element access reuses `ExpressionValue::Index` and
  `CheckedLocationKind::Index`. The operand's checked type already tells
  lowering whether to take an array's address or read a slice's base, so no
  second index representation is needed.
