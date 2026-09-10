# Check indexing, length, comparison, and iteration

## Sources

- `docs/spec.md#indexing` — `a[i]` needs an array operand, yields the element
  type, and is never constant
- `docs/spec.md#indexing-and-lengths` — an index is `int` or an untyped
  constant that fits; `0 <= i < len`
- `docs/spec.md#length` — `len` takes an array operand, yields `int`, still
  evaluates the operand, and is constant when the operand contains no call
- `docs/spec.md#array-comparison` — `==` and `!=` on identical array types,
  constant when both operands are
- `docs/spec.md#assignment` — element targets, `const` elements, and index
  evaluation order
- `docs/spec.md#loops` — both `for … in` forms
- `eng/roadmap.md#arrays` — fourth task
- `eng/plans/2026-09-09-009-check-array-values-and-literals.md` — the
  `Constant` type, `check_array_literal`, and the `reject_array_values` guard

## Goal

Finish semantic checking for arrays. After this task the only unimplemented
array work is Fern IR and the backend, so `reject_array_values` is the sole
remaining guard and `unimplemented_arrays` is deleted along with its four
call sites.

## Implementation

### `src/semantic.rs` — index expressions

Add `ExpressionValue::Index { operand, index }` and
`ExpressionValue::Length { operand }`, and give `infer_expression` an arm for
each.

`infer_index(operand, index, scopes)`:

- Infer the operand. A non-array operand is an error on the operand's span:
  ``format!("cannot index `{ty}`")``.
- Check the index with `check_expression(index, scopes, Some(Scalar::Int.into()))`.
  That already gives the specified rule: an untyped constant concretizes to
  `int`, and another integer type reports the existing implicit-conversion
  diagnostic.
- When the checked index folded, reject a value outside `0..length` on the
  index's span: ``format!("index {value} is out of range for `{ty}`")``,
  naming the array type.
- Result: the element type, `untyped: false`, `constant: None`.
- Record both sub-expressions with `record_operand(.., false)`; both are still
  evaluated.

`infer_length(operand, scopes)`:

- Infer the operand and require an array type, error on the operand's span:
  ``format!("`len` requires an array operand, found `{ty}`")``.
- Result type `int`, `untyped: false`. The constant is the length from the
  operand's type when `find_call(self.syntax, operand)` is `None`, and `None`
  otherwise.
- Record the operand with `record_operand(.., false)` even when the length
  folded, because the operand is still evaluated and may trap.

### `src/semantic.rs` — element assignment

Change `assignment_target` to return `(Idx<Binding>, Type)`, the binding and
the type of the element being assigned. Keep the existing mutability check
before walking indices, so an element of a `const` binding reports the existing
`"cannot assign to immutable binding \`x\`"`. Then fold over `target.indices` in
source order, reusing the two checks `infer_index` performs — extract them into
a helper both call, so the operand-is-an-array error, the index typing, and the
constant range check have one home. Each step descends to the element type, and
each checked index is inserted into `self.expressions`.

Both callers use the returned element type: `StatementKind::Assignment` as the
value's destination, and `check_compound_assignment` as the type of its
synthetic left operand.

### `src/semantic.rs` — array comparison

In `unify_comparison`, keep the scalar path as it is and add an array path when
either operand's `scalar()` is `None`: reject unequal `Type`s with the existing
"comparison operands have different types" wording spelled with `Type`, then
reject any operator but `Equal` and `NotEqual` on the operator's span with
``format!("only `==` and `!=` are defined on `{ty}`")``. This needs `operator`
threaded into `unify_comparison`.

Widen `compare_constants` to take `&Constant` operands: `Equal` and `NotEqual`
compare the `Constant`s directly, which is unchanged for integers and correct
elementwise for arrays; the ordering operators keep reading `.integer()`, which
the array path above has already ruled out. `infer_comparison` then folds from
`checked_left.constant`/`checked_right.constant` instead of `.integer()`.

### `src/semantic.rs` — `for … in`

Add to `CheckedProgram`:

```rust
/// The bindings a `for … in` statement introduces.
pub(crate) struct IterationBindings {
    pub value: Idx<Binding>,
    pub index: Option<Idx<Binding>>,
}
```

with `pub iterations: ArenaMap<Idx<Statement>, IterationBindings>`. IR has no
other way to reach these bindings, and `declarations` holds one binding per
statement.

Pass the statement id from `check_statement` through `check_for` into
`check_for_header`, and replace its `Iteration` arm with
`check_iteration_header`:

- Infer the operand and require an array type, error on the operand's span:
  ``format!("`for … in` requires an array, found `{ty}`")``. Insert the checked
  operand into `self.expressions`.
- Reject an index name equal to the value name on the index's span:
  `"`for` value and index bindings must have different names"`.
- Push a scope, allocate the value binding with the element type and the index
  binding with `int`, both `mutable: false` and `constant: None`, insert both
  names into that scope, record `IterationBindings`, and return `Ok(true)` so
  `check_for` pops the scope after the body. The body's own scope makes
  shadowing work, and the bindings are gone after the loop.

Remove the `expect(dead_code)` attributes on `ForHeader::Iteration`'s `value`,
`index`, and `operand` fields in `src/frontend.rs`.

### `src/semantic.rs` and `src/ir.rs` — the remaining guard

- Delete `unimplemented_arrays` and its four call sites.
- `reject_array_values` also walks `checked.assignments`: an array-typed target
  binding reports on the statement's span. `a[0] = 1;` has no array-typed
  expression, because the target's base is a name rather than an expression.
- Add `ExpressionValue::Index` and `ExpressionValue::Length` to the existing
  `Array { .. }` arm of `lower_flow_operand` in `src/ir.rs:1315`.

## Tests

`src/semantic.rs`, deleting `array_values_parse_but_are_not_yet_checked`:

- Indexing: `a[0]` on `[3]int` is `int`, `grid[0]` on `[2][3]int` is `[3]int`,
  `grid[0][1]` is `int`. Indexing an `int` and indexing with a `u8` binding are
  rejected. `a[3]` and `a[-1]` report the out-of-range diagnostic; `a[i]` with a
  runtime `i` is accepted.
- `a[0]` is never constant: a module-level `const y = a[0];` over a constant
  module-level array is rejected as a non-constant initializer.
- Element assignment: `a[0] = 1;` and `grid[0][1] = 1;` check, `grid[0] = [1, 2,
  3];` checks an array literal against the element type, `a[0] = true;` is
  rejected, `a[0][0] = 1;` on `[3]int` is rejected, and an element of a `const`
  binding reports the immutable-binding diagnostic. `a[0] += 1;` checks and
  `a[0] += true;` is rejected.
- `len`: `const n = len(a);` records `folded(3)`, `[len(a)]int` resolves to
  `[3]int`, `len(a)` on a `var` array is still constant, `len(a[0])` on
  `[2][3]int` is 3, and `len(1)` is rejected.
- Comparison: `a == b` and `a != b` on identical `[3]int` are `bool`; two
  constant arrays fold to `folded(1)` and `folded(0)`; `a < b`, `[2]int` against
  `[3]int`, and an array against an `int` are each rejected.
- Iteration: `for v in a` binds `v` to the element type, `for v, i in grid`
  binds `v` to `[3]int` and `i` to `int`, nested `for row in grid { for v in row
  {} }` checks, `for v, v in a` is rejected, `for v in 1` is rejected, `v = 1;`
  inside the body reports the immutable-binding diagnostic, `v` is not resolvable
  after the loop, and the body may declare its own `v`.

`tests/compiler.rs`: extend the array entry of `source_failures_preserve_output`
with an indexed assignment, which reaches the guard through the target rather
than through an expression.
