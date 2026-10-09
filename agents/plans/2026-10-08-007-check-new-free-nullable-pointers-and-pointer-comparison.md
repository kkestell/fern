# Check `new`, `free`, nullable pointers, and pointer comparison

## Goal

Semantic checking accepts the remaining pointer forms in
`docs/spec.md#pointers-and-heap-memory`:

- `new(e)` and `free(p);`;
- `null` wherever context supplies a nullable pointer type;
- the implicit conversion from `*T` to `nullable *T`;
- the checked conversion `*T(p)`;
- `==` and `!=` between pointers.

No pointer syntax reports a temporary diagnostic during checking any more.
The driver guard also rejects `new`, so no pointer value reaches IR lowering.

## Related code

- `src/types.rs` — `Type`.
- `src/semantic/model.rs` — `ExpressionValue` and `Constant::Null`.
- `src/semantic/expressions.rs`:
  - `check_expression` — applies a destination and reports
    `cannot implicitly convert`.
  - `infer_expression` — the guard arm for `Null`, `New`, and
    `PointerConversion`.
  - `common_element_type` — the element type of an array literal with no
    context.
  - `infer_comparison` and `unify_comparison`.
  - `infer_grouping`, `infer_length`, and `infer_conversion` — patterns the new
    checks follow.
  - `convert_constant` in `src/semantic/constants.rs` — the
    `constant conversion to … would trap` wording.
- `src/semantic/statements.rs` `check_statement` — the `Free` guard arm.
  `check_call` and `check_return` already pass parameter and result types as
  destinations, and struct-literal fields and array elements do too.
- `src/semantic/annotations.rs` — `pointers_not_implemented` and
  `reject_pointer_syntax`.
- `src/ir/lower.rs` — the exhaustive `ExpressionValue` matches.
- `src/semantic/tests/expressions.rs`
  `pointer_syntax_reports_that_pointers_are_not_yet_implemented` — every case
  now checks.

## Decisions

**`null` takes its type from context.** `check_expression` sends an
expression that is `null`, possibly in groupings, to a new
`check_null(id, destination: Option<&Type>)` before inferring anything.
`is_null(id)` makes that test by looking through `Grouping`.

`check_null`:

- On `Null` with a `nullable *T` destination, it gives
  `ExpressionValue::Null` of that type with constant `Constant::Null`.
- With any other destination, it reports
  `` cannot implicitly convert `null` to `{destination}` ``.
- With no destination, it reports
  `` `null` requires a nullable pointer type from context ``.
- On `Grouping`, it checks the inner expression the same way and records it
  as `infer_grouping` does, so `(null)` takes context like `null`.

All diagnostics are at the `null` span. `infer_expression` reports the
no-context diagnostic for `Null`, which covers `var p = null;`, `-null`,
`*null`, and `len(null)`.

Every context the spec lists already reaches `check_expression` with a
destination, except a comparison operand and an array literal with no
context:

- `common_element_type` skips elements for which `is_null` holds, so
  `[null, p]` with `p: nullable *T` has element type `nullable *T`. A literal
  of only `null` elements falls through to `check_null` and reports the
  no-context diagnostic.
- `infer_comparison` infers the operand that is not `null` and checks a
  `null` operand with `check_null` against the other operand's type. With
  `null` on both sides, both are inferred, and the left `null` reports the
  no-context diagnostic.

A constant `null` folds like any constant. A struct or array literal holding
`null` and other constants is a constant expression, and comparing two
`Constant::Null` values folds through `compare_constants`.

**Implicit `*T` to `nullable *T`.** `Type::widens_to(&self, destination)`
holds when `self` is `*T` and `destination` is `nullable *T` for the same
`T`. When `check_expression` would report a mismatch and the checked type
widens to the destination, the expression's checked type becomes the
destination. Both pointer types share one representation, so a pointer
expression may carry the nullable type of its context. `unify_comparison`
does the same: when one operand's type widens to the other's, that operand
takes the other's type. Other pointer pairs still report
`comparison operands have different types`, and the existing
`` only `==` and `!=` are defined on `…` `` covers ordering. Arithmetic,
bitwise, unary operators, and numeric conversions already require scalar
operands, so they reject pointers with their current diagnostics.

**`new(e)`.** `infer_new` checks the operand with
`check_expression(operand, scopes, None)`, which gives an untyped constant its
default type and rejects `null` and an unsized fill as today. The result is
`ExpressionValue::New { operand }` of type `*T` for the operand's type `T`,
never constant. The operand keeps its constant through `record_operand` with
`folded: false`.

**`free(p);`.** `check_statement` infers the operand and requires a pointer
of either nullability. Anything else reports
`` `free` requires a pointer operand, found `{ty}` `` at the operand. The
checked operand is recorded in `expressions`. `free(null)` reports the
no-context diagnostic, because `free` does not supply a type.

**`*T(p)`.** `infer_pointer_conversion` resolves the destination annotation
with `resolve_annotation`. The parser makes it a non-nullable pointer, so a
different type is `unreachable!`. Call the destination `*T`.

- A `null` operand is checked with `check_null` against `nullable *T`.
- Any other operand is inferred, and its type must be `*T` or `nullable *T`.
  Otherwise the check reports `` cannot convert `{ty}` to `*T` `` at the
  conversion span, as `infer_conversion` does.
- An operand whose constant is `Constant::Null` reports
  `` constant conversion to `*T` would trap `` at the conversion span, since
  a checked conversion that would trap at run time is a compile-time error.

The result is `ExpressionValue::PointerConversion { operand }` of type `*T`,
never constant.

**Nullable access.** `*p`, `p.f`, `p[i]`, and `len(p)` on a nullable pointer
are already rejected by `pointer_target` and `Type::through_pointers`. This
change adds tests for indexing and `len`.

**Guards.** `pointers_not_implemented` is now used only by
`reject_pointer_syntax`. That function also reports the first `New`
expression after annotations and `&`. A `null` needs a nullable pointer type
from an annotation, and `*T(p)` writes one, so annotations, `&`, and `new`
cover every pointer value. Update its doc comment. In `src/ir/lower.rs`, the
`Null`, `New`, and `PointerConversion` values join the existing
`unreachable!("pointer types are rejected before lowering")` arms.

## Implementation plan

1. `src/types.rs` — add `Type::widens_to` with unit tests.
2. `src/semantic/model.rs` — add `ExpressionValue::Null`,
   `ExpressionValue::New { operand }`, and
   `ExpressionValue::PointerConversion { operand }`.
3. `src/semantic/expressions.rs`:
   - add `is_null`, `check_null`, `infer_new`, and `infer_pointer_conversion`;
   - route `null` in `check_expression` and apply `widens_to`;
   - replace the guard arm in `infer_expression`;
   - update `common_element_type`, `infer_comparison`, and
     `unify_comparison`.
4. `src/semantic/statements.rs` — check `Free`.
5. `src/semantic/annotations.rs` — extend `reject_pointer_syntax`.
6. `src/ir/lower.rs` — add the `unreachable!` arms.

## Test plan

- `src/types.rs` — `*int` widens to `nullable *int`. It does not widen to
  `nullable *u8`, `*int`, or `nullable *nullable *int`, and `nullable *int`
  does not widen to `*int`.
- `src/semantic/tests/expressions.rs`, `new`:
  - binding types `*int` for `new(1)`, `*f64` for `new(1.5)`, `*bool` for
    `new(true)`, `*u8` for `new(x)` with `x: u8`, `*S` for a struct literal,
    `*[2]int` for `new([1, 2])`, `**int` for `new(new(0))`, and `*int` for
    `new(1,)`;
  - `var p: *u8 = new(1);` reports
    `` cannot implicitly convert `*int` to `*u8` ``;
  - `new(null)` reports the no-context diagnostic, and `new([0...])` reports
    `a fill requires a length from context`.
- `null`:
  - accepted as an annotated initializer, `(null)` initializer, assignment
    value, call argument, `return` value, struct-literal field, element of an
    annotated array literal, and either operand of `==` and `!=` against a
    nullable pointer;
  - `[q, null]` and `[null, q]` with `q: nullable *int` have type
    `[2]nullable *int`;
  - `const q: nullable *int = null;` records `Constant::Null`; a module-level
    `const n = Node { value = 1, next = null };` records a struct constant
    holding `Null`; `q == null` folds to `true`;
  - each at the `null` span:
    - `var p = null;` and `[null, null]` with no annotation report the
      no-context diagnostic;
    - `var p: *int = null;` reports
      `` cannot implicitly convert `null` to `*int` ``;
    - `var x: int = null;` reports
      `` cannot implicitly convert `null` to `int` ``;
    - `null == null` reports the no-context diagnostic at the left `null`;
    - `p == null` with `p: *int` reports
      `` cannot implicitly convert `null` to `*int` ``;
    - `-null` reports the no-context diagnostic.
- Widening:
  - `*int` is accepted where `nullable *int` is expected in an initializer,
    an assignment, an argument, a `return`, a struct field, and an array
    element, and the annotated binding records `nullable *int`;
  - `var r: *int = q;` reports
    `` cannot implicitly convert `nullable *int` to `*int` ``;
  - `var b: [1]nullable *int = a;` with `a: [1]*int` reports
    `` cannot implicitly convert `[1]*int` to `[1]nullable *int` ``.
- Comparison:
  - `p == q`, `q == p`, `p == p`, and `q != q` are accepted;
  - `p == r` with `r: *u8` reports
    `` comparison operands have different types `*int` and `*u8` ``;
  - `p < p` reports `` only `==` and `!=` are defined on `*int` ``;
  - `p + p`, `p & p`, `-p`, `!p`, and `int(p)` report their existing
    diagnostics.
- `*T(p)`:
  - `*int(q)`, `*int(p)`, and `**int(qq)` with `qq: nullable **int` are
    accepted with types `*int`, `*int`, and `**int`;
  - `*u8(q)` reports `` cannot convert `nullable *int` to `*u8` ``;
  - `*int(5)` reports `` cannot convert `int` to `*int` ``;
  - `*int(null)` and `*int(c)` with `const c: nullable *int;` report
    `` constant conversion to `*int` would trap ``.
- Nullable access: `q[0]` with `q: nullable *[2]int` reports
  `` cannot index `nullable *[2]int` ``, and `len(q)` reports
  `` `len` requires an array operand, found `nullable *[2]int` ``.
- `src/semantic/tests/statements.rs`:
  - `free(p);`, `free(q,);`, and `free(new(1));` are accepted;
  - `free(1);` reports
    `` `free` requires a pointer operand, found `int` `` at `1`;
  - `free(null);` reports the no-context diagnostic.
- The spec examples under Heap allocation, Access through a pointer, Nullable
  pointers, and Pointer comparison pass `check_root`.
- Remove `pointer_syntax_reports_that_pointers_are_not_yet_implemented`.
- `tests/compiler.rs` `source_failures_preserve_output` —
  `fn main() -> void { const p = new(1); }` reports
  `pointers are not yet implemented`.
