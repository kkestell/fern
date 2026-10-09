# Review pointer syntax parsing

## Scope and coverage

Reviewed all staged changes for
`agents/plans/2026-10-08-004-parse-pointer-syntax.md`, against the pointer and
loop rules in `docs/spec.md`. Traced lexer reservation, annotations, prefix
precedence, conversion lookahead, assignment dispatch, syntax traversal,
module dependency collection, semantic rejection, and the lowering guard.
Inspected the parser snapshot and diagnostic, reservation, nesting, and
semantic tests. Pointer execution remains outside this frontend task.

## Fixed

- **Pointer expression targets fail in three-clause loops**
  (`src/frontend/parser.rs:726`, medium, correctness): Assignment lookahead
  skipped leading dereferences but required a name or parenthesized operand.
  Valid targets such as `*new(0)`, `*&x`, and `*[&x][0]` therefore failed in
  loop initializers or post assignments before semantic checking. Confirmed
  with a failing parser regression: `for *new(0) = 0; true; *new(0) = 1 {}`
  reported `expected `{`` at the initializer's `=`. Extended target lookahead
  to recognize address-of prefixes, allocation operands, and array literals;
  consolidated balanced-delimiter scanning. Added parser coverage for both
  clauses and semantic coverage confirming the temporary pointer diagnostic.

## Findings

None remain open.

## Checks run

- `cargo test` before review fixes — 303 unit and 59 integration tests passed.
- `cargo test pointer_expression_targets_parse_in_three_clause_loops` —
  reproduced the finding before the fix.
- `cargo test frontend::tests` after fixes — all 53 tests passed.
- `cargo test pointer_syntax_reports_that_pointers_are_not_yet_implemented`
  after fixes — passed.
- `cargo test semantic::tests` after the final regression adjustment — all
  102 tests passed.
- `cargo fmt --check` — passed after formatting the fix.
- `cargo clippy --all-targets -- -D warnings` after fixes — passed.
- `git diff --check` — passed.

## Verdict

One medium correctness finding fixed; no open findings. Ready to commit.
