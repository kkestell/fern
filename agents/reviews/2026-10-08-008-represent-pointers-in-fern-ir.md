# Review: Represent pointers in Fern IR

## Scope and coverage

Reviewed the staged changes for
`agents/plans/2026-10-08-008-represent-pointers-in-fern-ir.md`: the new IR
constructs in `src/ir/model.rs`, their verification in `src/ir/verify.rs`,
pointer lowering and the borrow rule in `src/ir/lower.rs`, `Type::is_aggregate`
in `src/types.rs`, the moved driver guard in `src/lib.rs`, the backend
`unreachable!` arms, and the new IR tests and snapshot. Behavior was checked
against the pointer section of `docs/spec.md`, the assignment-order rules, and
the call-argument rule.

Native emission of the new IR is out of scope; the backend arms are
intentionally unreachable behind the driver guard. The emission tasks in
`agents/todo.md` also need to give pointers a word class where the emitter
picks a QBE result type from `Type::scalar` (`emit`) and dispatches
comparisons on `Type::scalar` (`emit_value`).

## Fixed

- **Emitter kept its own aggregate predicate** (`src/backend/emitter.rs:320`):
  a private `is_aggregate` and four inline `ty.scalar().is_none()` checks
  classified any non-scalar type, pointers included, as an aggregate. This
  duplicated the new `Type::is_aggregate` with a different answer for
  pointers, so the emission tasks would have passed and returned pointers
  through result storage. Removed the private function and used
  `Type::is_aggregate` at every site. Behavior for current programs is
  unchanged.

## Findings

No open findings.

## Checks run

- `cargo fmt --check` — clean after formatting the fix.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 346 unit and 59 integration tests pass.
- Compiled hand-written pointer programs through the driver covering
  assignment through `*p`, `p.f`, `pp.a[f()]`, and call results; compound
  assignment through pointers; `&` of elements, fields, globals, and loop
  locals; `new` of scalars, aggregates, and block-splitting `&&` operands;
  `*T(p)` in a `for` loop; `free`; widening in initializers, arguments,
  returns, array literals, and comparisons; module-level `null` constants,
  pointer-holding globals, and borrowed constant aggregates. Each lowered and
  verified, then stopped at `pointers are not yet implemented`. `*int(null)`
  and `free(null)` are rejected by checking.

## Verdict

The IR representation, lowering, and verification match the plan and the
specification. One low-severity duplication was fixed; nothing remains open.
