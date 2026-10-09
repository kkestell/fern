# Represent pointers in Fern IR

## Goal

Fern IR represents pointer values and the operations on them, and lowering
translates every checked pointer form:

- `&location`, `*p`, and field selection, indexing, and `len` through
  pointers, in values and in assignment targets;
- `new(e)`, `free(p);`, `null`, and `*T(p)`;
- pointer equality.

Verification types every new construct and requires an indirect place to be
rooted in a non-nullable pointer value. Pointer programs now lower and verify
during compilation. The driver guard moves to just before native emission,
and the backend treats the new IR as unreachable until the emission tasks.

## Related code

- `src/ir/model.rs` — `Place`, `Place::root`, `Literal`, `ValueKind`,
  `Instruction`.
- `src/ir/verify.rs`:
  - `verify_literal`, `valid_value`, `place_type`, and `destination_type`;
  - `scalar_types` and `scalar_count`, which have `unreachable!` pointer
    arms;
  - `verify_instruction`, `verify_call`, `verify_return`, and
    `record_definition`.
- `src/ir/lower.rs`:
  - `lower`, `Places`, and `lower_function`;
  - `scalar_literal`, `static_values`, `flatten`, and `zero_literal`;
  - `FlowBuilder::store_zero`;
  - `lower_flow_statement` (`Free`), `lower_location`, and `lower_argument`
    (the `Borrow` rule);
  - `lower_aggregate_place`, `element_place`, `field_place`, `lower_length`,
    `lower_folded_effects`, and `lower_flow_operand`;
  - `HeldOperand`, `HeldPlace`, `HeldStep`, and `hold_place`.
- `src/semantic/model.rs` — `CheckedProgram::addresses`,
  `CheckedLocationKind::{Pointer, Dereference}`, whose `dead_code`
  expectations this change removes, and the `ExpressionValue` pointer forms.
- `src/types.rs` — `Type::through_pointers` and `Type::widens_to`.
- `src/lib.rs` `compile` — calls `reject_pointer_syntax` before lowering.
- `src/backend/` — exhaustive matches on `Type`, `Place`, `Literal`,
  `ValueKind`, and `Instruction`.
- `src/ir/tests/mod.rs` (`describe`, `stored_places`, `lowered`),
  `src/ir/tests/lower.rs`, and `src/ir/tests/verify.rs`.

## Decisions

**IR constructs.**

- `Place::Indirect { pointer: Operand }` — the target of a pointer value.
  `Place::root` treats it as a root, beside `Local`, `Global`, and
  `Constant`.
- `Literal::Null(Type)` — `null` of a nullable pointer type. `Literal::ty`
  returns that type.
- `ValueKind::Address(Place)` — the address of a place, of type `*T` for the
  place's type `T`.
- `ValueKind::Allocate` — a pointer to fresh heap storage for the target of
  its type. Its span is the `new` expression, which the `out of memory` trap
  reports.
- `ValueKind::CheckNull { operand }` — the `*T` a `nullable *T` holds. Its
  span is the `*T(p)` expression, which the `null pointer` trap reports.
- `Instruction::Free { operand }` — frees a pointer of either nullability.

Pointer equality is the existing `ValueKind::Comparison`.

**Pointers are word values, not aggregates.** `Type::is_aggregate`, in
`src/types.rs`, holds for arrays and structs. Decisions that today ask
`ty.scalar().is_some()` to mean "carried by value" ask `!ty.is_aggregate()`
instead:

- `lower_flow_operand`'s aggregate dispatch;
- `lower_argument`;
- `FlowBuilder::store_zero`;
- verification of `Borrow`, which requires an aggregate;
- verification of `Zero`, which rejects a non-aggregate.

Numeric operations still require `scalar()`. `lower_flow_operand` keeps the
expression's checked `Type` for loads and literals. Its numeric arms read the
scalar from it.

**`*T` and `nullable *T` share one representation.** Semantic checking gives
a widened expression its context's `nullable *T` while the value it reads is
a `*T` (`Type::widens_to`). Verification accepts that through one helper,
`fits(source, expected)`: the types are equal, or `source` widens to
`expected`. It applies to:

- a `Load`, from place type to value type;
- a `Store`;
- a `Fill` element;
- a call argument;
- a `Return`;
- a `Comparison`, where either operand may fit the other.

Lowering builds each new value with its own type: `*T` for `Address`,
`Allocate`, and `CheckNull`. A load takes the expression's checked type, as
it does today.

**Verification.**

- `verify_literal` requires `Literal::Null` to have a nullable pointer type.
- `place_type` of `Indirect` requires the pointer operand to have type `*T`
  and gives `T`. A nullable or non-pointer operand is an internal error:
  `IR reads through `…``.
- `Address(place)`:
  - span `None`;
  - the value's type is `*T` for the place's type;
  - the place resolves through `destination_type`, so a constant or a
    parameter cannot be addressed.
- `Allocate`: a non-nullable pointer type and a span.
- `CheckNull`: a span, a `*T` value, and a `nullable *T` operand.
- `Free`: a pointer operand.
- `Borrow`: rejects an `Indirect` root as it rejects a `Global` one.
- `scalar_types` and `scalar_count` count a pointer as one slot of its own
  type, so a global's pointer slot holds `Literal::Null`.

**Lowering values.**

- A constant `null` lowers through the constant path. `scalar_literal`
  becomes `literal(constant, ty: &Type)`, which gives `Literal::Null(ty)` for
  `Constant::Null`. `flatten` gives `Literal::Null` for a pointer slot.
  `static_values` treats `Null` as zero bytes. `store_zero` stores
  `Literal::Null` into a nullable pointer place.
- `&location` reads `lower_location(checked.addresses[id])` and defines
  `Address` of type `*T`, where `T` is the location's type.
- `*p`:
  - a non-aggregate target loads from `Indirect { pointer }`;
  - an aggregate target is that place in `lower_aggregate_place`.
- Field, index, and `len` steps through pointers:
  - A new `lower_step_base(operand)` gives the place a step acts on. An
    aggregate operand is `lower_aggregate_place`. A pointer operand lowers
    to its value, and `follow_pointers` reaches the place.
  - `follow_pointers(builder, pointer, ty)` is shared with locations. It
    returns `Indirect { pointer }` and loads each intermediate pointer when
    `ty` has several non-nullable pointer layers.
  - `element_place` and `field_place` use `lower_step_base`.
  - `lower_length` reads the length from `through_pointers()` of the operand
    type.
- `new(e)` lowers the operand first, then defines `Allocate` of type `*T`.
  It stores the operand into `Indirect` of the new pointer, holding the
  operand across any block split.
- `*T(p)` lowers the operand. When the operand's checked type is nullable, it
  defines `CheckNull` of type `*T`. Otherwise the operand is already a `*T`
  and is the result.
- `free(p);` lowers the operand and emits `Instruction::Free`.
- `lower_folded_effects` lists the pointer values with the forms that never
  fold.

**Lowering locations.**

- `HeldPlace::root` becomes a `HeldRoot`:
  - `HeldRoot::Place(Place)` for `Local`, `Global`, and `Constant`;
  - `HeldRoot::Indirect(HeldOperand)` for a pointer lowered before whatever
    follows it.
- `hold_place` holds an `Indirect` pointer operand. It takes the type from a
  new `FlowBuilder::operand_type`.
- In `lower_location`:
  - A `Field` or `Index` whose operand location has pointer type gets the
    pointer from `lower_location_pointer(operand)`. `follow_pointers` and
    `hold_place` then root the step there.
  - `Dereference` does the same for exactly one layer.
  - `lower_location_pointer` lowers a `Pointer(id)` location as the
    expression `id`. Any other location is read and loaded with its type.
  - A bare `Pointer` location is `unreachable!`, because it only appears as
    an operand.
- The pointer is loaded before the assigned value is lowered, keeping the
  target-before-value order the spec requires.

**Borrowing.** A callee can now reach a caller's local through a pointer.
`lower` collects the bindings whose address any `&` takes: the root binding
of each `checked.addresses` location, where the walk stops at a step through
a pointer. `Places` carries that set. `lower_argument` loads (copies) an
aggregate rooted in an addressed local, or in an `Indirect` place, instead
of borrowing it. The `Borrow` doc comment states the rule.

**Driver guard and backend.** `compile` lowers and verifies first and calls
`reject_pointer_syntax` just before `backend::toolchain::build`. Every
backend arm for `Type::Pointer`, `Place::Indirect`, `Literal::Null`,
`Address`, `Allocate`, `CheckNull`, and `Free` is
`unreachable!("pointers are rejected before native emission")`.

## Implementation plan

1. `src/types.rs` — add `Type::is_aggregate` with a unit test.
2. `src/ir/model.rs` — add the constructs above and extend `Place::root` and
   `Literal::ty`.
3. `src/ir/verify.rs` — add `fits` and the verification rules above. Count
   pointer slots in `scalar_types` and `scalar_count`.
4. `src/ir/lower.rs` — add `addressed` to `Places`, `literal`,
   `FlowBuilder::operand_type`, `follow_pointers`, `lower_step_base`,
   `lower_location_pointer`, and `HeldRoot`. Lower every pointer form.
5. `src/semantic/model.rs` — remove the `dead_code` expectations on
   `CheckedLocationKind::Pointer` and `Dereference`.
6. `src/backend/` — add the `unreachable!` arms and update the existing
   pointer arms' message.
7. `src/lib.rs` — move the guard after verification.
8. `src/ir/tests/mod.rs` — spell `Indirect` as `*(…)` in `describe`, and list
   `free` in `stored_places`.

## Test plan

- `src/types.rs` — arrays and structs are aggregates; scalars and pointers
  are not.
- `src/ir/tests/lower.rs`:
  - `&x` defines `Address(Local)`. `&a[i]` and `&s.f` address element and
    field places, and `&p.x` through a `*S` addresses a field of an
    `Indirect`.
  - `*p = 2;` stores to `Indirect` of a load of `p`. `p.x = 1;` stores to a
    field of it. `pp.x = 1;` loads twice. `get(p).x = 4;` stores through the
    call result. In `*p = f();` the pointer load precedes the call.
  - `exit(*p)`, `exit(p.x)`, `exit(q[1])`, and `exit(len(q))` load through
    `Indirect`; `len(q)` folds to the length while `q` is still read.
  - `const c = *ps;` for a struct target loads an aggregate from `Indirect`.
  - `new(1)` defines `Allocate` with the expression span, then stores `1`
    through it. `new(S { … })` stores the literal through it. In
    `new(f())`, the call precedes the allocation.
  - `*int(q)` defines `CheckNull` with the conversion span, and `*int(p)` on
    a `*int` does not.
  - `free(p);` emits `Instruction::Free`.
  - `var q: nullable *int;` stores `Literal::Null`. `var n: Node;` zeroes.
    `q = null;` stores `Literal::Null`. A module-level
    `var n = Node { value = 1, next = null };` is a global holding
    `[1, null]`, and an all-`null` one has no values.
  - `var q: nullable *int = p;`, a widened argument, a widened return, and
    `p == q` verify.
  - An aggregate local whose address is taken is passed by `Load`. One whose
    address is never taken is still passed by `Borrow`. An aggregate read
    through a pointer is passed by `Load`.
  - A `pointers` snapshot of the spec's linked-list program.
- `src/ir/tests/verify.rs`, each rejected with an internal error:
  - `Indirect` through an `int`, and through a `nullable *int`;
  - `Address` of a constant or a parameter, and `Address` whose type does not
    match its place;
  - `Allocate` without a span or with a nullable type;
  - `CheckNull` without a span or on a `*int` operand;
  - `Free` of an `int`;
  - `Literal::Null` of type `*int`;
  - storing a `nullable *int` into a `*int` place;
  - loading a `nullable *int` place as `*int`;
  - `Borrow` of an `Indirect` place.

  Accepted:
  - a `*int` stored into a `nullable *int` place;
  - a `*int` place loaded as `nullable *int`;
  - comparing a `*int` with a `nullable *int`.
- The spec's pointer examples lower and verify through `lowered`.
- `tests/compiler.rs` — the existing pointer cases still report
  `pointers are not yet implemented`, now after verification.
