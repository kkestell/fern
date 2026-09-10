# Ownership review: entire codebase

Scope: the current Fern codebase, including the compiler, unit tests, and
compiler integration tests.

Mode: ownership

## Findings

No findings.

The phase boundaries match `eng/architecture.md`: `Program` owns sources and
syntax; `CheckedProgram` borrows syntax while owning semantic facts; IR then
owns its model and is moved into `VerifiedProgram`. The named lifetimes express
those actual relationships. There are no reference-counted or interior-mutable
owners, manual cleanup paths, or surprising `Copy` implementations.

The examined clones have a concrete boundary: recursive `Type`, `Place`, and
source-span values are retained in IR or diagnostics; per-function lowering
needs an independent binding map; and parser lexer clones implement bounded
lookahead. In particular, the two array-lowering span uses borrow syntax
directly in the current tree.

## Unresolved suspicions

None.

## Checks

- Inventoried all Rust production and test files, then searched construction
  sites and uses of ownership-sensitive types, `clone`-like conversions,
  collection, shared ownership, interior mutability, `Drop`, and named
  lifetimes.
- Traced the source/syntax, semantic, IR-verification, diagnostic-renderer,
  module-loader, and backend-emitter ownership boundaries against
  `eng/architecture.md`.
- `cargo clippy --all-targets -- -D warnings` passed.
- `cargo test -q` passed: 193 unit tests and 50 integration tests.
