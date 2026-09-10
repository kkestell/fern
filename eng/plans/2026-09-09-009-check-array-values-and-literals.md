# Check array values and literals

## Sources

- `docs/spec.md#array-literals` — context typing, common element type, element
  count, and when a literal is a constant expression
- `docs/spec.md#fill` — `...` repeats the last element and needs a length from
  context
- `docs/spec.md#array-types` — arrays are value types; initialization,
  assignment, argument passing, and return copy every element
- `eng/roadmap.md#arrays` — third task
- `eng/plans/2026-09-09-008-represent-array-types-across-the-compiler.md` — the
  `Type`/`Scalar` split and annotation resolution this builds on
- `src/semantic.rs:1581` — `check_expression`, where the destination arrives

## Goal

Check array values: array literals, whole-array copying, and array parameters
and results. Indexing, `len`, array comparison, and `for … in` stay rejected for
the next task, and nothing array-shaped is lowered yet, so a new guard between
checking and Fern IR keeps `ir::lower` from seeing an array.

## Implementation

### `src/semantic.rs` — a constant may be an array

`CheckedExpression::constant` and `Binding::constant` are `Option<BigInt>`, so
they cannot hold an array literal's value, which the next two tasks need for
constant array comparison and for constant aggregate globals. Widen them once:

```rust
/// The value a constant expression folds to. An array literal folds when
/// every element does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Constant {
    Integer(BigInt),
    Array(Vec<Constant>),
}
```

- Add `Constant::integer(&self) -> Option<&BigInt>` and
  `impl From<BigInt> for Constant`.
- Add `CheckedExpression::integer(&self) -> Option<&BigInt>`, returning
  `self.constant.as_ref().and_then(Constant::integer)`. Every existing site that
  reads a folded scalar — `array_length`, `evaluate_unary`, `evaluate_binary`,
  `convert_constant`, `infer_comparison`, `infer_logical`, `infer_logical_not`,
  `concretize_value` — becomes a call to it. Sites that only ask *whether* the
  expression folded keep using `constant.is_some()`/`is_none()`.
- The folding helpers keep returning `Option<BigInt>`; the `infer_*` functions
  that store the result add `.map(Constant::Integer)`.
- In the tests module add `fn folded(value: i128) -> Option<Constant>` beside
  `big`, and use it where tests compare a recorded constant.

### `src/semantic.rs` — check array literals

Add `ExpressionValue::Array { elements: Vec<Idx<Expression>>, fill: bool }`.

In `check_expression`, dispatch an `ExpressionKind::ArrayLiteral` to a new
`check_array_literal(id, scopes, destination)` before the existing
infer-then-concretize path, and replace the `ArrayLiteral` arm of
`infer_expression` with the same call passing `None`. An array literal is the
one expression whose checking needs the destination, and this gives it one home
for both cases. `Index` and `Length` keep their `unimplemented_arrays` arm.

`check_array_literal`:

- `Some(Type::Array { length, element })` supplies the length and element type.
- `Some(Type::Scalar(_))` is an error on the literal's span:
  ``format!("cannot implicitly convert an array literal to `{destination}`")``.
- `None` means no context. A `fill` is then an error on the `...` span,
  `"a fill requires a length from context"`. Otherwise the length is
  `elements.len()` and the element type comes from a new
  `common_element_type`: infer each element in turn, take the type of the first
  element that is not `untyped` and stop, otherwise keep the first element's
  own type as the untyped default. It discards those `CheckedExpression`s; the
  shared pass below re-checks and re-records every element, which is what keeps
  one checking path for both cases.
- Check the count against the length on the literal's span, naming the array
  type: without a fill ``"expected {length} elements for `{ty}`, found {n}"``,
  with a fill ``"expected at most {length} elements for `{ty}`, found {n}"``.
  The parser already guarantees at least one element.
- Check every element with `check_expression(element, scopes, Some(element
  type))`, collecting the results. Pushing the element type down is what makes
  `[2][3]u8 = [[1, 2, 3], [4, 5, 6]]` and `[2][3]u8 = [[1...]...]` work.
- Fold: when every element folded, the constant is `Constant::Array` with the
  fill expanded, so it holds `length` values — the first `n - 1` elements, then
  the last repeated `length - (n - 1)` times. Record each element with
  `record_operand(.., folded)` so a folded literal owns the value, as
  `infer_binary` does.
- Return `untyped: false` and `ty` the array type.

### `src/semantic.rs` — let array values through

- Delete the array rejection in `signature_type`, so it becomes a plain call to
  `resolve_annotation` with no initializer. `[_]` in a signature then reports
  ``"`[_]` requires an array-literal initializer"`` from `inferred_length`.
- Add `pub(crate) fn reject_array_values(checked: &CheckedProgram<'_>) ->
  Result<(), Diagnostic>`, the temporary guard the last task of the milestone
  deletes. It reports `"arrays are not yet compiled"` on the first array it
  finds: each function's array-typed parameter or result, on the annotation's
  span, then each array-typed entry of `checked.expressions`, on the
  expression's span. Signatures are covered separately because a function with
  an array parameter that is never called has no array-typed expression.

`check_expression`'s existing `checked.ty != destination` mismatch already
enforces identical types for whole-array initialization, assignment, argument
passing, and return, because `Type` compares structurally. No change there.

### `src/ir.rs` and `src/lib.rs`

- `lib.rs::compile` calls `semantic::reject_array_values` on the checked program
  before `ir::lower`, rendering the diagnostic the same way `semantic::check`'s
  is rendered.
- `ir.rs` reads `.constant` in `lower` (the global's value), `lower_binding`,
  and `lower_flow_operand`. The two that read a value go through
  `Constant::integer` with
  `.expect("array values are rejected until arrays are lowered")`, matching
  `scalar_type`. Add an `ExpressionValue::Array` arm to `lower_flow_operand`
  with the same `unreachable!`.

## Tests

`src/semantic.rs`:

- An annotated binding, a parameter, a result, and an assignment all give a
  literal its type: `[3]int`, `[2]u8` from `[0, 255]`, `[2]bool`, and
  `[2][3]int` from a nested literal.
- Without context, `[1, 2, 3]` is `[3]int`, `[1, x]` with `x: u8` is `[2]u8`,
  `[x, 1]` with `x: u8` is `[2]u8`, and `[[1, 2], [3, 4]]` is `[2][2]int`.
  `[x, y]` with `x: u8` and `y: int` is rejected.
- Element count: too few and too many are rejected against an annotation and
  against a parameter type.
- Fill: `[2]int = [7...]` and `[8]int = [1, 2, 3, 0...]` fold to the expanded
  values, `[3]int = [1, 2, 3...]` repeats nothing, `[2]int = [1, 2, 3...]` is
  rejected, and `var bad = [0...]` reports the no-length diagnostic on `...`.
- `var a: [_]int = [1, 2, 3];` now checks and gives `a` type `[3]int` — replace
  the first assertion of
  `an_underscore_length_comes_from_an_array_literal_initializer`.
- An out-of-range element (`[2]u8 = [0, 256]`) and an array literal against a
  scalar annotation are rejected.
- A module-level `var` and `const` each hold an array literal and record a
  `Constant::Array`; a module-level array literal with a non-constant element is
  rejected as a non-constant initializer.
- Whole-array copy: `var b = a;` gives `b` the same type, `a = b` between
  identical types checks, and assigning `[2]int` to `[3]int` is rejected.
- An array argument and an array result check against the signature, a
  mismatched length is rejected, and `[_]` as a parameter type reports the
  `[_]` diagnostic.
- Arithmetic on an array operand is rejected by the existing integer-operand
  diagnostic.
- Trim `array_values_parse_but_are_not_yet_checked` to the forms the next task
  owns: indexing, `len`, `for … in`, and an indexed assignment target.

`tests/compiler.rs`:

- Add an array program to `source_failures_preserve_output` expecting
  `"arrays are not yet compiled"`.

## Decisions

- Context reaches an array literal only where the specification says it does: a
  binding with an annotation, a call argument, and `return`. A literal inside a
  grouping or any other operand is checked with no context, so
  `var a: [2]u8 = ([0, 255]);` is rejected. Threading a destination through
  every expression form would add a parameter almost no arm uses.
- An array literal is never `untyped`. The `untyped` flag drives `concretize`,
  which is defined on scalars; array literals reach their type through the
  destination instead.
