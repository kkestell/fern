# Review address-of, dereference, and access through pointers

## Scope and coverage

Reviewed all staged changes for
`agents/plans/2026-10-08-006-check-address-of-dereference-and-access-through-pointers.md`
against `docs/spec.md#address-of-a-location`,
`#access-through-a-pointer`, `#indexing`, `#length`, `#immutability`, and
`#module-level-declarations`. Traced `Type::through_pointers` through the
field, index, and `len` checks; `check_location`, `check_pointer_operand`,
and `pointer_target` through assignment, compound assignment, `for` headers,
and `&`; `immutable_root` and `target_name` for the immutable-binding
diagnostic; the `addresses` map; the renamed driver guard; and the
`unreachable!` arms in `src/ir/lower.rs`.

Compiled probe programs with `target/debug/fern` covering `&` on groupings,
nested `&`, `*&x`, `&*&x`, steps on `(&s)`, compound assignment through a
call's pointer, `len` through `*p` and `p.a`, `for … in` over `p.a` and `*p`,
reassignment of a `const` pointer, aggregate assignment through `*p`, nullable
operands of `*` and `.`, arrays of pointers, and `&` in module-level
initializers, including a self-referential one. None panicked; each reported
the spec's diagnostic or reached the driver guard. Lowering of pointer values
belongs to a later task and was not reviewed.

## Fixed

- **A non-pointer operand of `*` in a location reports the wrong error**
  (`src/semantic/expressions.rs:633`, low, error-handling): `*(x + 1) = 2;`
  and `&*1` reported `cannot assign to an expression that is not a location`
  and `cannot take the address of an expression that is not a location`,
  while `*x = 2;` and the value `*(x + 1)` report `` cannot dereference `int` ``.
  The target is a location whenever the operand is a pointer, so the
  not-a-location message misdirected the reader. `check_pointer_operand` now
  returns a computed value as a `Pointer` location whatever its type; the
  `Dereference` arm rejects a non-pointer through `pointer_target`, and the
  new `check_step_operand` keeps the not-a-location diagnostic for index and
  field steps. Added the two cases and a grouped non-pointer call step to
  `a_pointer_binding_is_immutable_while_its_target_is_not`.

## Findings

None remain open.

## Checks run

- `cargo fmt --check` — passed after the fix.
- `cargo clippy --all-targets -- -D warnings` — passed after the fix.
- `cargo test` — 324 unit and 59 integration tests passed after the fix.
- 22 probe programs compiled with `target/debug/fern`, as described under
  Scope and coverage — no panics; diagnostics matched the spec or the
  temporary guard.

## Verdict

Address-of, dereference, look-through steps, and pointer mutability match the
plan and the specification. One low error-handling finding fixed; no open
findings. Ready to commit.
