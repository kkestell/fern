# Initialize bindings from calls without a copy

## Goal

`const b = make();` with an aggregate result currently takes two steps: the
callee writes into the call value's own storage, and a store copies that
storage into `b`. Afterwards, the call writes straight into `b`'s storage, so
no copy is made.

## Related code

- `src/ir/model.rs` — `Instruction::Call`.
- `src/ir/verify.rs`:
  - `verify_instruction`'s `Call` arm and `verify_call`, which rejects a
    value-returning call with no `result`;
  - `destination_type`.
- `src/ir/lower.rs`:
  - `lower_flow_binding`, `binding_place`, and `Places::addressed` /
    `returned`;
  - `lower_call`, `lower_call_expression`, and `unwrapped_call`;
  - `FlowBuilder::call`.
- `src/backend/emitter.rs`:
  - `emit_instruction`'s `Call` arm and `emit_call`, which pick the result
    storage: `%v{id}_storage`, or `%result` for a returned call;
  - `allocations`;
  - `place_address`.
- Every `Instruction::Call` built in `src/ir/tests/` (`mod.rs`, `lower.rs`,
  `verify.rs`).

## Decisions

**`Instruction::Call` gains `destination: Option<Place>`.** When it is set,
the callee writes its aggregate result straight into that place, and the
call defines no `result` value. Verification checks a destination with:

- `result` must be `None`, and the callee must return an aggregate whose type
  equals the destination's type;
- `destination_type` checks the place, so no constant and no parameter;
- the place's root must be a `Local` or `Place::Result`. The callee could
  read a global or a pointer target while building its result in place, so
  a `Global` or `Indirect` root is rejected:
  `IR call writes its result into `…``.

The existing `(Some(_), None)` "discards the result" error still applies when
there is no destination.

**Lowering.** `lower_flow_binding` sends an initializer to a new
`lower_call_into` when all of these hold:

- the initializer is an unfolded call, through `unwrapped_call`;
- the binding's type is an aggregate;
- the binding has no folded constant;
- the binding is not addressed.

`lower_call_into` lowers the arguments as `lower_call` does, then emits the
call with `destination` set to `binding_place(…)`. That place is a fresh local
or, for the function's returned local, `Place::Result`. The binding then maps
to that place, with no `CallResult` value and no store. `lower_call` and
`lower_call_into` share argument lowering through a helper,
`lower_arguments`. `FlowBuilder::call` takes the destination.

**Why this cannot be observed.** A declared binding is not in scope in its
own initializer, and an unaddressed binding has no pointer to it. The
arguments therefore cannot read the destination while the callee writes it.
A destination of `Place::Result` aliases nothing either: the callee writes
the caller's result storage, as a returned call does today. Across loop
iterations, the previous iteration's value of the binding is dead before the
call runs.

**Backend.** `emit_call` takes the destination. When it is set, the hidden
result argument is the destination's address from `place_address`, and no
`%v{id}` is defined. `allocations` already gives storage only to
`CallResult` values, so a call with a destination needs none.

## Implementation plan

1. `src/ir/model.rs` — add `destination` to `Instruction::Call` and document
   it.
2. `src/ir/verify.rs` — verify destinations.
3. `src/ir/lower.rs` — `lower_arguments`, `lower_call_into`, the destination
   on `FlowBuilder::call`, and the binding path.
4. `src/backend/emitter.rs` — emit a destination call.
5. `src/ir/tests/` — add `destination: None` where tests build calls, and
   the tests below. Review any IR snapshot that changes before accepting it.

## Test plan

- `src/ir/tests/lower.rs`:
  - `const b = make();` and `var b = (make());` emit one call with
    `destination: Some(Local(b))`, no `CallResult`, and no store into `b`;
  - a returned local initialized from a call gets
    `destination: Some(Result)`;
  - an addressed binding, a scalar call, and an aggregate initialized from a
    non-call keep today's lowering.
- `src/ir/tests/verify.rs`, each rejected with an internal error:
  - a destination together with a `result`;
  - a destination whose type differs from the callee's result;
  - a destination for a callee returning a scalar;
  - a destination rooted in a global or in a constant.

  Accepted: a destination in a local, and a destination in `Place::Result`
  of an aggregate-returning caller.
- `src/backend/tests/structs.rs`:
  - The QBE for `const b = make();` passes `%local{n}` as the result address
    and copies nothing after the call.
  - A native program exits `42`. It covers:
    - a swap of a struct into a new binding;
    - a binding redeclared from a call in each loop iteration;
    - a returned local initialized from a call;
    - recursion that initializes bindings from calls.
