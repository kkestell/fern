# Evaluate the `len` operand without a copy

## Scope and coverage

The staged changes for
`agents/plans/2026-10-09-003-evaluate-the-len-operand-without-a-copy.md`:
`lower_aggregate_effects` and its two callers in `src/ir/lower.rs`, the new
tests in `src/ir/tests/mod.rs` and `tests/core.rs`, and the removal of the bug
from `agents/bugs.md`. Traced every operand shape `lower_aggregate_place`
accepts (reference, grouping, index, field, array and struct literal, call,
dereference) and the folded and unfolded `len` paths against the `len` rule in
`docs/spec.md`.

## Fixed

- **Doc comment overstates what `len` skips** (`src/ir/lower.rs:1154`): The
  comment said nothing is loaded, copied, or emitted as constant data, but
  `len(table[i])` on a `const` table still emits the table through
  `element_place`, and a non-constant array literal is still built in a local.
  A reader could rely on a guarantee the code does not give. The comment now
  says the value is not loaded.

## Findings

No open findings.

## Checks run

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed.
- With the `src/ir/lower.rs` change reverted, `len_does_not_load_its_operand`
  and `len_of_an_array_larger_than_the_stack_executes` fail.

## Verdict

The change fixes the bug and keeps the operand's effects. One comment was
corrected; nothing remains open.
