# Ownership review: entire codebase

Scope: the full Fern repository, with an ownership-focused review of the Rust
implementation and tests. There was no pending diff, so this review examines
the current codebase.

Mode: ownership

## Findings

### Low: array-lowering spans are cloned only to be borrowed

`src/ir/lower.rs:796` and `src/ir/lower.rs:827` clone an expression span and
immediately take a reference to that clone. Both uses stay within the current
function, and `checked` is only shared-borrowed while the span is used.

This adds needless temporary ownership and obscures that the span remains owned
by syntax. Borrow it directly with
`&checked.syntax.expressions[id].span` at both sites.

## Unresolved suspicions

None.

## Checks

- Inspected the empty working-tree diff and the complete Rust file list.
- Searched all source and test Rust files for clone, conversion, collection,
  shared-ownership, interior-mutability, `Drop`, and lifetime patterns.
- Traced the `Program`/`SourceMap`/`Syntax`, `CheckedProgram`, and
  `Program`/`VerifiedProgram` ownership boundaries against
  `eng/architecture.md`.
- `cargo clippy --all-targets -- -D warnings` passed.
- `cargo test --quiet` passed: 193 tests.
