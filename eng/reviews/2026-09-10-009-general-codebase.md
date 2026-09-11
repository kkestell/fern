# General codebase review

## Scope and mode

Exhaustive general review of the entire codebase across all thirteen topics.
Corpus: `src/` (frontend, semantic, IR, backend), `src/lib.rs`, `src/main.rs`,
`tests/`, fixtures, examples, `docs/spec.md`, `eng/architecture.md`,
`Cargo.toml`.

## Findings

### Low 1: dead zero-field branch in aggregate comparison emission

`src/backend/qbe.rs:169-174`: `emit_aggregate_equal` special-cases a struct
with zero fields, but the frontend rejects fieldless structs
(`src/frontend/parser.rs:356-358`, "expected a field declaration"), semantic
checking requires at least one field per the specification, and IR
verification carries the table through unchanged. The branch is unreachable.
Consequence is readability-only: a reader must prove unreachability across
three phases. Suggested fix: remove the branch, or replace it with an
`unreachable!` stating the invariant and its owner (the frontend minimum-one-
field rule).

## Suspicions

None.

## Topic verdicts

- Ownership: nothing to report.
- Error handling: nothing to report.
- API design: nothing to report.
- Performance: nothing to report.
- Testing: nothing to report.
- Readability: Low 1.
- Concurrency: nothing to report.
- Security: nothing to report.
- Correctness: nothing to report.
- Unsafe: nothing to report.
- Architecture: nothing to report.
- Dependencies: nothing to report.
- Documentation: nothing to report.

## Checks

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 283 unit, 56 integration pass.
- `cargo tree --duplicates` — none.
- `rg unsafe`, thread/spawn/Mutex/await sweep — none in production code.
