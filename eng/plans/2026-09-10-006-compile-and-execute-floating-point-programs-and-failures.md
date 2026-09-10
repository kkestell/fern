# Compile and execute floating-point programs and failures

## Sources

- `eng/roadmap.md#floating-point-numbers` — final task, completion fixture,
  and milestone gates
- `docs/spec.md#floating-point-semantics` — runtime arithmetic and comparison
  behavior
- `docs/spec.md#numeric-conversions` — checked-conversion failures
- `docs/spec.md#floating-point-constant-expressions` — compile-time failure
  behavior
- `eng/architecture.md#pipeline` — the public compiler pipeline and native
  boundary
- `eng/plans/2026-09-10-005-lower-and-emit-floating-point-values.md` — the
  completed IR and backend slice
- `tests/compiler.rs` — compiler and CLI integration-test boundary

## Goal

Finish the floating-point milestone from the existing checked, lowered, and
emitted support. Add its roadmap fixture and prove the public compiler both
executes valid floating-point programs and preserves the specified distinction
between compile-time rejection and runtime conversion traps.

## Implementation

- `tests/fixtures/programs/floating_point.fern` — add the roadmap completion
  program verbatim, including the `f32` binding, arithmetic, `f64` conversion,
  and comparison that exits 42.

- `tests/compiler.rs` — compile and run that fixture through `fern::compile`.
  Add integration failures at the same boundary: a non-finite floating
  constant expression must fail compilation without replacing an existing
  output, while compiled programs with lossy floating checked conversions must
  abort with the existing source-aware conversion diagnostic. Cover the
  fractional, non-finite, and precision-loss conversion categories without
  duplicating backend-format tests.

## Tests

- The floating-point fixture exits with status 42 through the full source,
  semantic, IR, QBE, and host-tool pipeline.
- A non-finite floating constant is rejected before native output publication.
- Runtime floating conversion failures leave no normal exit status and report
  their source location and `checked conversion failed` diagnostic.

## Extra validation

This task completes the milestone. Run `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, and `cargo test`, then review the
completed milestone once for cross-phase correctness and a single path for
floating values and conversions.
