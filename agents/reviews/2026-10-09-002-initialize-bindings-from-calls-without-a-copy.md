# Review: initialize bindings from calls without a copy

## Scope and coverage

Reviewed the staged changes for
`agents/plans/2026-10-09-001-initialize-bindings-from-calls-without-a-copy.md`:
`Instruction::Call::destination` in `src/ir/model.rs`; `verify_call` and
`verify_call_destination` in `src/ir/verify.rs`; `FlowBuilder::call`,
`lower_flow_binding`, `lower_call_into`, and `lower_arguments` in
`src/ir/lower.rs`; `emit_call` in `src/backend/emitter.rs`; and the new and
changed tests and snapshots. Also read the code they depend on:
`binding_place`, `returned_local`, `Places::borrowable`, `lower_argument`,
`unwrapped_call`, `addressed_binding`, `destination_type`, `place_type`,
`returned_in_place`, `allocations`, and `place_address`.

The review traced every way a callee writing straight into a binding's storage
could be observed:

- Arguments aliasing the binding: the binding becomes visible only after its
  initializer (`docs/spec.md`, Scope and shadowing), so no argument names it.
  An initializer that names an outer binding of the same name reads a
  different binding with its own local, borrowed or loaded as before.
- Pointers: an addressed binding keeps the old lowering, and
  `addressed_binding` follows fields and elements to their root, so no pointer
  reaches the destination.
- The returned local in `Place::Result`: the caller's result storage is a
  fresh `%v{id}_storage`, the caller's own `%result`, or now a fresh
  unaddressed binding in the caller. Arguments never borrow `Place::Result`,
  so none shares it, through any depth of calls.
- Loop redeclaration: the previous iteration's value is out of scope and
  unaddressed before the call runs. A three-clause `for` initializer runs
  once, and its `post` is an assignment, which is unchanged.
- `returned_in_place` still holds: a `Load(Place::Result)` value is defined in
  the block of its `return`, and the initializing call precedes it.

## Fixed

- **Verifier tests miss the scalar-callee and `Place::Result` cases**
  (`src/ir/tests/verify.rs:1869`): the plan called for rejecting a destination
  for a callee returning a scalar and accepting a destination in the result
  storage of an aggregate-returning caller. The `[2]int`-into-`int` case does
  not reach the `is_aggregate` check, since an `int` callee into an `int`
  local has matching types, so a regression in either path would go
  unnoticed. `verification_checks_call_destinations` now rejects an `int`
  callee writing an `int` local and accepts a call inside the `[2]int` callee
  that writes its own result storage.

## Findings

No open findings.

## Checks run

- `cargo fmt --check`: clean.
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo test`: 370 unit and 62 integration tests pass.
- Native programs, all exiting 42:
  - `{ const s = swap(s); }` and `const t = swap2(s, s);` with an outer `s`;
  - a call taking a pointer to another binding, which writes through it while
    the result goes to a new binding;
  - a binding redeclared from a call on each loop iteration, and a
    three-clause `for` whose initializer is a call;
  - a returned local shadowing an outer local initialized from a call that
    reads the outer one;
  - a three-deep chain where a callee builds its returned local in the
    caller's binding while reading a parameter borrowed from the caller's
    shadowed outer binding of the same name.

## Verdict

The change meets the plan. A callee writing its result straight into a
binding's storage cannot be observed. One test gap was fixed, and no findings
remain open.
