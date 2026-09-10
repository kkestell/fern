# Documentation review: maintained docs and Rust comments

Scope: `README.md`, `docs/spec.md`, `eng/architecture.md`,
`eng/roadmap.md`, and every Rust documentation and line comment in `src/` and
`tests/`.

Mode: documentation

## Findings

### Medium: Empty `FERNPATH` behavior is documented inconsistently

- `src/lib.rs:35` says that setting `FERNPATH` replaces the default import
  roots. `docs/spec.md:1206` says the same.
- `src/module.rs:87` instead ignores empty entries and retains the default root
  when no non-empty entry remains. `README.md:50` correctly documents that
  behavior.
- A library caller that sets `FERNPATH` to an empty value can therefore expect
  no search roots while compilation still resolves imports beside the root
  module.
- State in the `compile` documentation and specification that `FERNPATH`
  replaces the default only when it supplies at least one non-empty,
  platform-separated entry. Alternatively, make the implementation use an
  empty root list. Keep the three descriptions aligned.

## Unresolved suspicions

None.

## Checks

- Read the maintained repository documents and inventoried all 612 Rust
  documentation and line-comment lines in production, unit-test, and
  integration-test sources. Checked the non-obvious implementation comments
  against their adjacent code and `eng/architecture.md`.
- Traced module-root selection through `compile`, `search_roots`, and the
  documented module-resolution contract. The README and `search_roots` agree;
  the public API documentation and specification do not.
- `cargo doc --no-deps` passed.
- `cargo test --doc` passed; the crate has no Rust doctests.
- `cargo test -q` passed: 193 unit tests and 50 integration tests. The
  integration suite compiles the documented Fern examples.
