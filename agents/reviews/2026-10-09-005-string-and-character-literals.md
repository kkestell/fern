# String and character literals

## Scope and coverage

Reviewed the staged string and character literal implementation against
`agents/plans/2026-10-09-004-string-and-character-literals.md` and `docs/spec.md`,
tracing parsing, semantic checking, IR lowering and verification, QBE emission,
and native tests. Native execution was checked on the current host only.

## Fixed

- **Mixed character and integer literal comparisons reject valid expressions**
  (`src/semantic/expressions.rs:1146`): `97 == 'a'` reached comparison typing
  after both literals had taken their defaults, so it failed instead of
  selecting `u8`. Comparisons containing a character literal now infer both
  operands before unifying them when no concrete numeric type supplies context.
  Regression checks cover both operand orders, floating precedence, and an
  integer that does not fit `u8`.

## Findings

None open.

## Checks run

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed (unit, compiler, and native integration tests).
- `git diff --check` and `git diff --cached --check` — passed.

## Verdict

One finding fixed; no open findings. The staged changes are ready to commit.
