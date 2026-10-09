# Check address-of, dereference, and access through pointers

## Goal

Semantic checking accepts `&location` and `*p` as
`docs/spec.md#address-of-a-location` and
`docs/spec.md#access-through-a-pointer` define them. `&location` is a `*T` of
a mutable location. `*p` is a mutable location of the target type, whatever
the mutability of the binding that holds `p`. Field selection, indexing, and
`len` look through any number of non-nullable pointers, both in values and in
assignment and `&` targets.

`null`, `new`, `*T(p)`, and `free` keep their temporary guard. The driver
guard also rejects `&`, so no pointer value reaches IR lowering.

## Related code

- `src/types.rs` — `Type`; gains the look-through helper.
- `src/semantic/model.rs`:
  - `ExpressionValue` — `Field`, `Index`, and `Length` keep their shape.
  - `CheckedLocation` and `CheckedLocationKind` — currently always rooted in
    a binding.
  - `CheckedProgram::assignments` — the pattern for storing a checked
    location by syntax id.
- `src/semantic/expressions.rs`:
  - `infer_expression` — the `AddressOf` and `Dereference` guard arm.
  - `check_index_step` and `check_field_step` — shared by values and
    locations.
  - `infer_length`.
  - `check_location` — its `Dereference` guard arm and its
    `cannot assign to an expression that is not a location` message.
- `src/semantic/statements.rs`:
  - `assignment_target` — the mutability check.
  - `location_binding` and `target_name` — find the root binding.
- `src/semantic/annotations.rs` `reject_pointer_annotations`,
  `src/semantic/mod.rs`, and `src/lib.rs` `compile` — the driver guard.
- `src/ir/lower.rs` — `lower_location`, and the exhaustive `ExpressionValue`
  matches.
- `src/semantic/tests/expressions.rs`
  `pointer_syntax_reports_that_pointers_are_not_yet_implemented` — its `&`
  and `*` cases change.

## Decisions

**Looking through pointers.** `Type::through_pointers(&self) -> &Type`
returns the type reached after following every non-nullable pointer: `*S`,
`**S`, and `S` all reach `S`, and a `nullable *S` reaches itself.
`check_field_step`, `check_index_step`, and `infer_length` apply it to the
operand type before requiring a struct or an array. Diagnostics name the
operand's written type, so `p.x` on a `*int` reports
`` cannot select a field of `*int` `` and `len` on a `*int` reports
`` `len` requires an array operand, found `*int` ``. `no_field` and the
out-of-range message name the struct or array reached. A nullable pointer
fails the same steps with its own type in the message.

`ExpressionValue::Field`, `Index`, and `Length` gain no fields. The number of
pointers a step follows is the number `through_pointers` strips from the
operand's checked type, so lowering derives it from that type.

**Locations.** `CheckedLocationKind` gains two kinds:

- `Pointer(Idx<Expression>)` — a pointer computed by an expression that is
  not itself a location, such as a call. It only appears as the operand of a
  `Field`, `Index`, or `Dereference`.
- `Dereference { operand: Box<CheckedLocation> }` — `*p`, the target of the
  pointer the operand holds.

`Field` and `Index` operands may now have pointer type; the step looks
through them as above.

`check_location(id, scopes, use_: LocationUse)` handles `Dereference` by
checking its operand with a new `check_pointer_operand`, requiring a
non-nullable pointer (`` cannot dereference `{ty}` `` at the operand span),
and giving the target type. The location's span is the whole `*p`
expression. `Field` and `Index` also check their operand with
`check_pointer_operand`.

`check_pointer_operand(id, scopes, use_)`:

- `Reference`, `Index`, `Field`, and `Dereference` go to `check_location`;
- `Grouping` recurses into its inner expression;
- anything else is inferred as a value. A pointer-typed value is recorded in
  `expressions` and becomes `Pointer(id)`, spanning the expression. Any other
  value reports the not-a-location diagnostic at its span, as today.

So `f().x = 1;` is valid when `f` returns a `*S`, `(f()).x` likewise, and
`f() = 1;` and `(f()) = 1;` are still not locations.

**`LocationUse`.** A `pub(super) enum LocationUse { Assign, Address }` in
`src/semantic/expressions.rs` with `verb()` returning `assign to` or
`take the address of`. It fills the two diagnostics that differ between
assignment and `&`:
`cannot {verb} an expression that is not a location` and
`` cannot {verb} immutable binding `{name}` ``.

**Mutability.** `location_binding` becomes
`immutable_root(&self, location) -> Option<Idx<Binding>>`: the root binding
when it decides the location's mutability and is immutable, and `None`
otherwise. A `Binding` root decides by its own `mutable`. A `Dereference`, or
a `Field` or `Index` whose operand has pointer type, is reached through a
pointer and is mutable, so the walk stops with `None`. `Pointer` never
reaches this walk on its own.

`assignment_target` keeps its signature and builds on a new
`mutable_location(target, scopes, use_)`, which calls `check_location` and
reports the immutable-binding diagnostic at `location.span` through
`target_name`. A `const` binding of pointer type therefore cannot be
reassigned, while `*p = e;`, `p.x = e;`, and `p[i] = e;` through it are
valid.

**Address-of.** `infer_expression` checks `AddressOf { operand }` with
`mutable_location(operand, scopes, LocationUse::Address)`. The checked
location goes in a new `CheckedProgram::addresses:
ArenaMap<Idx<Expression>, CheckedLocation>`, keyed by the `&` expression.
The value is `ExpressionValue::AddressOf`, with type `*T` for the location's
type `T`, never untyped and never constant. Index and field steps in the
operand are checked and recorded as they are for an assignment target.

**Dereference values.** `infer_expression` checks `Dereference { operand }`
by inferring the operand. A non-nullable `*T` gives
`ExpressionValue::Dereference { operand }` of type `T`, never constant. Any
other type reports `` cannot dereference `{ty}` `` at the operand span.

**Driver guard.** `reject_pointer_annotations` becomes
`reject_pointer_syntax`. It reports `pointers are not yet implemented` at the
first pointer annotation, and then at the first `AddressOf` expression.
Every pointer value starts at one of those, because `null`, `new`, and
`*T(p)` are still rejected during checking.

**Lowering.** `lower_location`'s `Pointer` and `Dereference` arms and the
`ExpressionValue::AddressOf` and `Dereference` arms in `src/ir/lower.rs` are
`unreachable!("pointer types are rejected before lowering")`.

## Implementation plan

1. `src/types.rs` — add `Type::through_pointers` with unit tests.
2. `src/semantic/model.rs` — add `ExpressionValue::AddressOf`,
   `ExpressionValue::Dereference { operand }`, the two
   `CheckedLocationKind` variants, and `CheckedProgram::addresses`. Initialize
   `addresses` in `src/semantic/namespaces.rs`. Update the doc comments on
   `CheckedLocation` and `CheckedTarget`: a location may start at a pointer.
3. `src/semantic/expressions.rs` — add `LocationUse`; apply
   `through_pointers` in `check_field_step`, `check_index_step`, and
   `infer_length`; add `check_pointer_operand`; extend `check_location`;
   replace the `AddressOf` and `Dereference` guard with `infer_address_of`
   and `infer_dereference`. Add arms for the new values wherever the
   compiler requires them, such as `concretize_children`.
4. `src/semantic/statements.rs` — add `mutable_location` and
   `immutable_root`, remove `location_binding`, and pass
   `LocationUse::Assign` from `assignment_target`.
5. `src/semantic/annotations.rs`, `src/semantic/mod.rs`, and `src/lib.rs` —
   rename and extend the driver guard.
6. `src/ir/lower.rs` — add the `unreachable!` arms.

## Test plan

- `src/types.rs` — `through_pointers` reaches `S` from `S`, `*S`, and `**S`;
  stops at `nullable *S`; and reaches `nullable *S` from `*nullable *S`.
- `src/semantic/tests/expressions.rs`, address-of:
  - binding types: `&x` is `*int`, `&a[1]` is `*int`, `&s.x` is `*int`,
    `&s` is `*S`, `&p` for a `*int` binding `p` is `**int`, and `&*p` is
    `*int`;
  - `&p.x` through a `const p = &s` and `&get().x` with `get` returning
    `*S` are accepted;
  - `` cannot take the address of immutable binding `q` `` at the binding
    for a `const` binding, a parameter, a field of a `const` struct, and an
    element of a parameter array;
  - `cannot take the address of an expression that is not a location` at the
    operand for `&1`, `&f()`, `&(x + 1)`, and `&S { x = 1 }`;
  - `&a[5]` on a `[2]int` reports `index 5 is out of range for `[2]int``.
- Dereference and look-through values:
  - `exit(*p)`, `exit(**pp)`, `exit(p.x + p.a[0] + len(p.a))` for a `*S`,
    and `exit(q[1] + len(q))` for a `*[3]int`; `len(q)` folds to `3`;
  - `pp.x` for a `**S`, and `get(p).x` from a call;
  - `` cannot dereference `int` `` and
    `` cannot dereference `nullable *int` `` at the operand;
  - `` cannot select a field of `*int` ``, `` cannot index `*S` ``,
    `` struct `S` has no field `y` `` through a `*S`,
    `` index 5 is out of range for `[3]int` `` through a `*[3]int`,
    `` `len` requires an array operand, found `*int` ``, and
    `` cannot select a field of `nullable *S` ``.
- `src/semantic/tests/statements.rs`, assignment through pointers:
  - with `const p = &x`, `*p = 2;`, `*p += 1;`, `(*p) = 3;`, and `**pp = 4;`
    are accepted;
  - with `const p = &s`, `p.x = 1;`, `p.a[2] = 3;`, `(*p).x = 4;`, and
    `pp.x = 5;` are accepted, as is `p.x = 1;` on a `*S` parameter and
    `get(p).x = 4;`;
  - `*p = true;` reports `` cannot implicitly convert `bool` to `int` ``;
  - `p = &x;` on the `const p` reports
    `` cannot assign to immutable binding `p` ``;
  - `const s = S { … }; s.x = 1;` still reports the immutable-binding error,
    and `get(p) = q;` still reports the not-a-location error;
  - a `for` header whose initializer and post clause assign through `*p`.
- `pointer_syntax_reports_that_pointers_are_not_yet_implemented` — drop the
  `&x`, `*x`, and `*&x` cases, which now check or report
  `cannot dereference`; keep `null`, `new`, `*int(x)`, and `free`.
- `tests/compiler.rs` `source_failures_preserve_output` —
  `fn main() -> void { var x = 1; const p = &x; }` reports
  `pointers are not yet implemented`.
