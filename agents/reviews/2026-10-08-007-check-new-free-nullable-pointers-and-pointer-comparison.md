# Review `new`, `free`, nullable pointers, and pointer comparison

## Scope and coverage

Reviewed all staged changes for
`agents/plans/2026-10-08-007-check-new-free-nullable-pointers-and-pointer-comparison.md`
against `docs/spec.md#pointers-and-heap-memory` and `#array-literals`. Traced
`is_null` and `check_null` through `check_expression`, `infer_comparison`,
`common_element_type`, groupings, and `free`; `Type::widens_to` through
`check_expression` and `unify_comparison`; `infer_new` and
`infer_pointer_conversion`, including constant `null` operands; constant
folding of `Constant::Null` through `compare_constants`; the extended driver
guard in `reject_pointer_syntax` and its call in `src/lib.rs`; and the
`unreachable!` arms in `src/ir/lower.rs`.

Compiled 25 probe programs with `target/debug/fern` covering `return p` and
`return null`, `exit(null)`, `if null`, `null == 1`, `(null) == q`,
`*int((null))`, `*int(p)` with `p: **int`, `*[_]int(q)`, `*Nope(q)`,
`new(g())` for a void `g`, `free(c)` for a constant `null`, a fill of `null`,
`[_]` with `null` elements, array comparison of `[1]*int` with
`[1]nullable *int`, struct constants holding `null`, and literals mixing `*T`,
`nullable *T`, and `null` in both orders. None panicked. Lowering of pointer
values belongs to a later task and was not reviewed.

## Fixed

- **An all-`null` array literal reports at the grouping**
  (`src/semantic/expressions.rs:439`, low, error-handling):
  `var a = [(null), null];` reported the no-context diagnostic at `(null)`,
  while `var a = (null);` and `(null) == null` report it at the inner `null`.
  `common_element_type` now returns the error `check_null` gives for the first
  element, so every no-context `null` reports at the `null` itself. Added the
  grouped case to `null_without_a_nullable_pointer_context_is_rejected`.
- **The nullable-pointer test prelude describes bindings it does not declare**
  (`src/semantic/tests/expressions.rs:1562`, low, comments): the doc comment
  on `NULLABLE_PRELUDE` listed `p`, `q`, and `r`, which `nullable_body`
  declares. Each item now documents what it declares.

## Findings

### medium

#### correctness

- **A literal mixing `*T` and `nullable *T` checks only in one order**
  (`src/semantic/expressions.rs:422`): with `p: *int` and
  `q: nullable *int`, `var a = [q, p];` checks as `[2]nullable *int`, while
  `var a = [p, q];` reports
  `` cannot implicitly convert `nullable *int` to `*int` `` at `q`.
  `common_element_type` takes the first typed element's type, and the
  remaining elements are checked against it, so widening applies only when the
  nullable element comes first. Confirmed with both probe programs. The spec
  requires every element of a literal with no context to "have one common
  type" and does not say whether `*T` and `nullable *T` share one. Suggested
  fix: settle that rule in `docs/spec.md` with `kspec`, then make
  `common_element_type` choose `nullable *T` when any typed element has it, or
  reject both orders. Needs a language decision.

## Checks run

- `cargo fmt --check` — passed after the fixes.
- `cargo clippy --all-targets -- -D warnings` — passed after the fixes.
- `cargo test` — 333 unit and 59 integration tests passed after the fixes.
- 25 probe programs compiled with `target/debug/fern`, as described under
  Scope and coverage — no panics; the diagnostics matched the spec or the
  driver guard, except the open finding.

## Verdict

`new`, `free`, `null`, widening, `*T(p)`, and pointer equality match the plan
and the specification. Two low findings fixed. One medium correctness finding
remains open pending a decision on the common type of `*T` and
`nullable *T` in an array literal with no context; it is recorded in
`agents/bugs.md`.
