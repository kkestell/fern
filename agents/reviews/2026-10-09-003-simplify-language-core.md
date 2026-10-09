# Review Fern's simplified core

## Scope and coverage

Reviewed the simplification against `docs/spec.md` and the implementation plan,
tracing parsing, concrete constant folding, checking, lowering, IR verification,
QBE operations, memory layout, external calls, and executable publication.
Checked documentation and replacement tests against the retained contracts.

Native validation covers the current host and installed QBE. Other QBE targets
were not executed. Programs with undefined behavior were not executed.

## Fixed

- **External symbol collisions** (`src/backend/emitter.rs`): generated symbols
  could collide with valid external names. Generated functions, globals, and
  constants now use names outside Fern's identifier syntax, with a native test.
- **Annotation dependencies in sizeof** (`src/semantic/namespaces.rs`): forward
  constant array lengths inside sizeof could be checked before their dependency.
  Its annotation now participates in dependency collection; regression added.
- **Unreachable breaks after continue** (`src/semantic/statements.rs`): nested
  blocks and branching continues could make a terminating infinite loop appear
  to need a return. Reachability scanning now stops after these statements;
  regression added.
- **Incomplete sizeof spans** (`src/frontend/parser.rs`): expressions covered
  only the keyword. They now cover the type and closing parenthesis and account
  for nesting; regression added. Removed stale aggregate-fill comments.

## Findings

No open findings.

## Checks run

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — all 86 tests passed, including 14 native core integration tests
  and execution of the rewritten examples.
- `git diff --cached --check` — passed.

## Verdict

The approved simplification is implemented through QBE, including floating-point
support. No open findings remain; commit the plan, implementation, work log,
and review together.
