# Remove redundant aggregate copies at calls

## Goal

An aggregate argument is copied twice today: the caller's `Load` copies it into
the value's storage, and the callee copies `%paramN` into `%localN` on entry. An
aggregate call used as a value is stored into a fresh local and loaded again,
and `return f(...)` then copies that result into `%result`.

When this is done:

- The callee reads an aggregate parameter through `%paramN` and never copies
  it.
- The caller passes an aggregate argument's own storage when its place is
  rooted in a local or a constant. A place rooted in a module-level `var` is
  still copied, because a called function can write it.
- An aggregate call used as a value is the call's own result, with no extra
  store and load.
- `return f(...)` passes the function's own `%result` to `f`, and the `return`
  copies nothing.

## Related code

- `src/ir/model.rs` — `Place`, `ValueKind`, and `Function::parameters`.
- `src/ir/lower.rs:579` `lower_call` — lowers each argument through
  `lower_flow_operand` and holds it.
- `src/ir/lower.rs:1130` `lower_flow_operand` — an aggregate always goes through
  `lower_aggregate_place` and `load_place`.
- `src/ir/lower.rs:990` `lower_aggregate_place` — its `Call` arm stores the call
  result into a fresh local, which is still needed when a call is the base of
  an element or field place.
- `src/ir/lower.rs:782` `capture` — the `for … in` copy, which stays.
- `src/ir/verify.rs` — `valid_value` and `destination_type`, which already
  walks a place to its root to reject writes to constants.
- `src/backend/emitter.rs` — `place_address`, `emit_control_flow` (local and
  value storage and the parameter entry copies), `emit_call`, the `Return` arm
  of `emit_terminator`, and the aggregate `Load` arm of `emit_value`.
- `src/backend/qbe.rs` `Emitter` — the per-function emission state.

## Decisions

**A non-copying aggregate read is its own value kind.** Add
`ValueKind::Borrow(Place)`: an aggregate read from its place's storage instead
of from a copy. The value is defined where the argument is evaluated, so the
bounds checks in an element place still run in argument order, before any
later argument. A `Load` keeps its meaning: a copy taken when it is defined.

**Lowering emits a `Borrow` only for a call argument whose place is rooted in a
local or a constant.** No Fern expression writes a local; only statements do,
and none runs between an argument's evaluation and the end of its call. A
called function cannot reach its caller's locals, and nothing writes a
constant. A module-level `var` can be written by any function the call or a
later argument runs, so its argument stays a `Load`. Restricting `Borrow` to
call arguments keeps an assignment such as `a[i] = a[j]` from copying a place
onto itself.

**A `Borrow` held across a block split is copied.** `HeldOperand::read` spills a
value through a local when a later argument splits blocks. The spill stores the
`Borrow` at the end of its defining block, which copies the place before
anything can write it. This case stays as it is.

**Parameters are never written.** Parameters are immutable bindings
(`docs/spec.md`, Functions), and lowering already writes none. Verification
rejects a write to a place rooted in a parameter local, which is what lets the
backend read an aggregate parameter straight through `%paramN`. A scalar
parameter keeps its entry store into `%localN`.

**An aggregate call is its own operand.** `lower_flow_operand` returns the call
result for an aggregate call, looking through parentheses, instead of storing
it into a fresh local and loading it. A call result already owns its storage,
so every use of an aggregate operand reads it correctly.

**The backend chooses `%result` for a returned call.** A call result that a
`Return` returns is written straight into `%result`, and that `Return` emits
only `ret`. Nothing reads `%result` before the function returns, so writing it
early is unobservable. A value is used only in its defining block, so the call
and the `return` share one block. `%result` is the caller's storage for this
call's result, which never holds any of the arguments, so the callee can read
its parameters while it writes `%result`.

**`Place::root` names a place's outermost place.** Verification and lowering
both need it.

## Implementation plan

- `src/ir/model.rs`: add `Place::root(&self) -> &Place`, which follows
  `Element` and `Field` bases to the `Local`, `Global`, or `Constant` beneath.
  Add `ValueKind::Borrow(Place)` after `Load`, with the doc comment from the
  first decision.
- `src/ir/verify.rs`: use `Place::root` in `destination_type`, and also reject
  a write whose root is a `Local` below `function.parameters`
  (`internal compiler error: IR writes parameter {id}`). In `valid_value`,
  accept a `Borrow` whose type is an aggregate, equals its place's type, whose
  root is not a `Global`, and which has no span.
- `src/ir/lower.rs`:
  - Add `unwrapped_call(checked, id) -> Option<Idx<Expression>>`: the
    unfolded call expression beneath any parentheses around `id`.
  - Add `lower_aggregate_operand`: an aggregate call is the result of
    `lower_call_expression` for the call `unwrapped_call` names; anything else
    is a `Load` of `lower_aggregate_place`. Use it in the aggregate branch of
    `lower_flow_operand`.
  - Add `lower_argument`: a scalar or a call goes through `lower_flow_operand`;
    any other aggregate is a `Borrow` of `lower_aggregate_place` unless the
    place's root is a `Global`, which stays a `Load`. Use it in `lower_call`
    in place of `lower_flow_operand`, and update `lower_call`'s doc comment.
- `src/backend/qbe.rs`: add `Emitter::returned_calls: Vec<ValueId>`, the
  aggregate call results a `Return` of the function being emitted returns.
- `src/backend/emitter.rs`:
  - Set `returned_calls` beside `qbe_result` for each function.
  - In `place_address`, a `Local` below `function.parameters` with an aggregate
    type is `%param{index}`.
  - In `emit_control_flow`, allocate no `%localN` for an aggregate parameter
    and drop its entry copy. Allocate no `%v{id}_storage` for a value in
    `returned_calls`.
  - In `emit_call`, pass `%result` as the result address of a value in
    `returned_calls` and define `%v{id}` as `%result`.
  - In the `Return` arm of `emit_terminator`, emit only `ret` when the value is
    in `returned_calls`.
  - In `emit_value`, a `Borrow` defines `%v{id}` as the address
    `place_address` computes, with no copy.
  - Update the module doc comment and the comment on the parameter loop to
    say when an aggregate is copied.

## Test plan

- `src/ir/tests/lower.rs`:
  - Replace the last assertion of
    `arrays_pass_through_parameters_arguments_and_results`: `first` now takes
    `swapped`'s call result directly, so `main` has one local.
  - A local, an element of a local array with a runtime index, a parameter, a
    constant aggregate, and an array literal passed as arguments each lower to
    a `Borrow`. A module-level `var` and an element of one each lower to a
    `Load`.
  - `return make();` and `return (make());` return the call result value with
    no local between the call and the `return`.
- `src/ir/tests/verify.rs`: rejecting a store into a parameter, a `Borrow` of a
  global, and a `Borrow` of a scalar.
- `src/backend/tests/emitter.rs`:
  - In `aggregates_pass_and_return_through_addresses`, replace the
    `blit %param0, %local0, 24` assertion: `$fn1` contains no `blit` and no
    `%local0` allocation.
  - A function whose body is `return inner(p);` passes `%result` as `inner`'s
    first argument and contains no `blit` or `memcpy`.
- `src/backend/tests/structs.rs`, native:
  - A callee that writes a module-level `var` after receiving it as an
    argument still reads the original value through its parameter.
  - A later argument that calls a function writing a module-level `var` does
    not change an earlier argument read from that `var`.
  - A local passed as an argument and then assigned by the caller after the
    call keeps the callee's result independent of the caller's later write.
  - `return f(...)` chained through two functions returns the right struct,
    and recursion passing a parameter on to the next call still works.
- `src/backend/tests/emitter.rs`, native failure: in `f(rows[i], quit())` with
  `i` out of range, the bounds-check trap fires before `quit` exits.
- Review and accept the changed IR snapshots under `src/ir/tests/snapshots/`.
