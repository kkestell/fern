# Remove redundant aggregate copies at calls

## Scope and coverage

The staged changes for
`agents/plans/2026-10-08-002-remove-redundant-aggregate-copies-at-calls.md`:
`ValueKind::Borrow` and `Place::root` in `src/ir/model.rs`, argument and
aggregate-operand lowering in `src/ir/lower.rs`, the parameter-write and
`Borrow` rules in `src/ir/verify.rs`, parameter, borrow, and returned-call
emission in `src/backend/emitter.rs` and `src/backend/qbe.rs`, their tests and
snapshots, and the staged `agents/todo.md` and `AGENTS.md` entry. Traced the
`HeldOperand` spill of a `Borrow`, the aliasing between `%result` and
borrowed arguments through chains of returned calls, and every backend use of
a local's or parameter's address.

## Fixed

- **Function-text helper can match a call site** (`src/backend/tests/emitter.rs:38`):
  `function_text` took the first `{symbol}(` in the QBE, so a function that
  calls a later-defined one would make the helper return part of the caller,
  and the no-`blit` and no-`%local0` assertions would pass without looking at
  the intended function. It now starts at the line that defines the function.

## Findings

No open findings.

## Checks run

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — 295 unit and 58 integration tests passed.
- A native program exercising `x = bump(x)`, a borrowed argument spilled
  across `&&` and `||` with a later argument writing a global, borrowed
  elements of an array parameter, returned calls inside `if` branches, a loop,
  and recursion — exited 42 as expected.

## Verdict

The change meets the plan. One test-helper weakness was fixed; nothing remains
open.
