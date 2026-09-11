# Compile and execute struct programs and failures

## Sources

- `eng/roadmap.md#structs` — final task, completion fixture, and milestone
  gates
- `docs/spec.md#type-declarations` and `#struct-literals` — struct values,
  field mutability, zero values, and structural equality
- `eng/architecture.md#pipeline` — public compiler pipeline boundary
- `eng/plans/2026-09-10-010-lay-out-copy-compare-and-access-structs-in-native-code.md`
  — completed native struct support and the temporary compiler guard
- `tests/compiler.rs` — compiler and CLI integration-test boundary

## Goal

Finish the structs milestone from the existing parsed, checked, lowered, and
native-emitted support. Remove the temporary pre-lowering guard, add the
roadmap fixture, and prove the public compiler executes struct programs while
invalid struct source leaves an existing output intact.

## Implementation

- `src/semantic/model.rs`, `src/lib.rs` — remove
  `reject_uncompiled_structs` and its pipeline call so every successfully
  checked struct program reaches IR verification and the backend. Remove the
  now-unused diagnostic dependency with the guard.

- `tests/fixtures/programs/structs.fern` — add the roadmap completion program
  verbatim, including a mutable field update and structural comparison that
  exits 43.

- `tests/compiler.rs` — replace the temporary guard test with a full-pipeline
  fixture test that compiles and runs the new program. Add focused compiler
  boundary failures for assigning through a `const` struct field and for an
  invalid struct literal or field selection; assert their diagnostics identify
  the source and preserve a pre-existing output. Keep lower-level parser,
  semantic, IR, and backend tests as the detailed coverage for those phases.

## Tests

- The structs completion fixture compiles through the public API and exits 43.
- Valid nested struct values reached through a module-level binding, function
  argument or result, and field assignment execute through the compiler
  boundary without reviving the guard.
- Invalid field mutation and invalid literal or field access fail before native
  output publication, with their source diagnostic retained.

## Extra validation

This task completes the milestone. Run `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, and `cargo test`; then review the
completed milestone for a remaining struct guard or a second layout, copying,
or equality path.
