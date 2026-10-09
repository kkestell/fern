# Emit address-of, dereference, and pointer comparison in native code

## Goal

The QBE backend emits every pointer construct except heap allocation, freeing,
and the null check:

- `Place::Indirect`;
- `ValueKind::Address`;
- `Literal::Null`;
- pointer loads, stores, parameters, results, and static data;
- pointer equality, including inside aggregate equality.

Programs that take addresses, read and write through pointers, and compare
pointers compile and run. The driver guard narrows to `new`, `free`, and
`*T(p)`, which the next task emits.

## Related code

- `src/backend/qbe.rs`:
  - `operand` and `data_item` — the `Literal::Null` arms.
  - `qbe_type`.
  - `emit_aggregate_equal` — its scalar leaf and its pointer arm.
- `src/backend/layout.rs` — `Layout::class` and `collect_scalar_slots`.
- `src/backend/emitter.rs`:
  - `place_address` — the `Place::Indirect` arm;
  - `emit` — the `qbe_result` computation, which reads `Type::scalar`;
  - `emit_control_flow` — the parameter stores, which match `Type::Scalar`;
  - `emit_store`;
  - the `Load`, `Comparison`, and `Address` arms of `emit_value`.
  - `allocations` already gives every non-aggregate local its own slot.
- `src/backend/integer.rs` `emit_comparison`.
- `src/semantic/annotations.rs` `reject_pointer_syntax` and
  `pointers_not_implemented`, and `src/lib.rs` `compile`.
- `src/backend/tests/mod.rs` and `src/backend/tests/structs.rs` — the
  `lowered`, `emit`, and `build_text` test pattern.
- `tests/compiler.rs` `source_failures_preserve_output` — the pointer guard
  cases.

## Decisions

**A pointer is a `uint`-width word.** A pointer has the size of `uint`
(`src/layout.rs` `pointer_bytes`), so its QBE class is
`qbe_type(Scalar::Uint)`. A new `value_class(ty: &Type) -> char` in
`src/backend/qbe.rs` gives the class of any non-aggregate type: a scalar's
`qbe_type`, and that word class for a pointer. An aggregate is
`unreachable!`. Every site that loads, stores, passes, or returns a value by
value uses it:

- `Layout::class` uses it for non-aggregates and keeps `l` for aggregates;
- `emit_store`, and `emit_value`'s `Load` for a non-aggregate;
- the parameter stores in `emit_control_flow`, which now cover every
  non-aggregate parameter;
- `qbe_result` in `emit`, through `!ty.is_aggregate()`.

**Static data.** `collect_scalar_slots` records a pointer slot with its own
type. `data_item(Literal::Null(ty))` writes a zero word of the pointer's
class, and `operand(Literal::Null(_))` spells `0`. `null` is the zero
address, so a `Zero` of an aggregate holding nullable pointers already
stores `null`.

**Places and addresses.**

- `place_address` of `Indirect { pointer }` is the pointer operand itself,
  with the target type read from the operand's pointer type through
  `operand_type`.
- `ValueKind::Address(place)` emits `%v{id} =l copy {address}` from
  `place_address`.

Locals, globals, elements, and fields all have addresses already, and an
element place keeps its bounds check whether its base is a local or a pointer
target. A local whose address is taken stays in memory, because QBE only
promotes a slot whose address never escapes.

**Comparison.** `emit_value`'s `Comparison` sends pointer operands to
`emit_comparison` with `Scalar::Uint`, which spells `ceql` and `cnel` on a
64-bit target. The operands' nullability does not matter, since both forms
are the same word. In `emit_aggregate_equal`, the scalar leaf becomes a
leaf for any non-aggregate type and uses `value_class`.

**Driver guard.** `reject_pointer_syntax` becomes `reject_heap_syntax`. It
reports the first `New` or `PointerConversion` expression, then the first
`Free` statement. `pointers_not_implemented` reports
`heap allocation and null checks are not yet implemented`. The remaining
backend arms for `Allocate`, `CheckNull`, and `Free` keep their
`unreachable!`, and their message says these are rejected before native
emission.

## Implementation plan

1. `src/backend/qbe.rs` — add `value_class`; handle `Literal::Null` in
   `operand` and `data_item`; extend `emit_aggregate_equal`'s leaf.
2. `src/backend/layout.rs` — the pointer arms of `class` and
   `collect_scalar_slots`.
3. `src/backend/emitter.rs` — `place_address`, `emit_store`, the `Load`,
   `Comparison`, and `Address` arms of `emit_value`, the parameter stores, and
   `qbe_result`. Update the remaining `unreachable!` messages and the module
   doc comment.
4. `src/semantic/annotations.rs`, `src/semantic/mod.rs`, and `src/lib.rs` —
   rename and narrow the guard.
5. `src/backend/tests/pointers.rs` (new, registered in
   `src/backend/tests/mod.rs`) — the tests below.
6. `tests/compiler.rs` — update the guard cases.

## Test plan

- `src/backend/tests/pointers.rs`, each building and running a native
  program and checking its exit status:
  - Reading and writing through pointers. In one program:
    - `&x`, `*p = …`, and `*p += …`;
    - `p.x` and `p.a[i]` through a `*S`;
    - `pp.x` through a `**S`;
    - a `const` pointer whose target is assigned;
    - a pointer to an array element, and one to a module-level `var`;
    - a `*S` parameter written by the callee;
    - a function returning one of two `*S` arguments, whose result is
      assigned through;
    - a pointer stored in a struct field and in an array element.

    The exit status is the sum the writes produce.
  - A runtime index through a `*[3]int` past its end traps with the bounds
    message.
  - Pointer equality: `&x == &x`, `&x != &y`, `p == q` with `q` a widened
    `nullable *int`, `q != null`, and a zero `nullable *int` equal to `null`.
    Two structs holding equal and unequal pointer fields compare through
    `emit_aggregate_equal`.
  - A module-level `var n = Node { value = 1, next = null };` emits a zero
    pointer word, and `n.next == null` holds at run time.
  - An argument aliased through a pointer is copied: with
    `fn f(a: [2]int, p: *[2]int) -> int { p[0] = 9; return a[0]; }`,
    `f(arr, &arr)` returns `1`.
  - The QBE text gives a `*S` parameter and a `*S` result the `l` class.
- `tests/compiler.rs`:
  - The `fn f(p: *int)` and `&x` programs move from
    `source_failures_preserve_output` to `native_programs_exit_zero`.
  - `new(1)`, `free(p);`, and `*int(q)` each report
    `heap allocation and null checks are not yet implemented`.
