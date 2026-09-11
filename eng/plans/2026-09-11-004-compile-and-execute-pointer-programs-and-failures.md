# Compile and execute pointer programs and failures

## Sources

- `docs/spec.md#pointers` and `#assignment` — pointer values, aliasing,
  mutability, conversion, comparisons, and null-dereference traps.
- `eng/todo.md#pointers` — the final Pointers subtask and its completion
  boundary.
- `eng/architecture.md#pipeline` and `#diagnostics` — the public compiler
  boundary and source-mapped runtime diagnostics.
- `eng/plans/2026-09-11-003-load-store-and-compare-pointers-in-native-code.md`
  — completed native pointer support and the staging diagnostic it leaves for
  this slice.
- `tests/compiler.rs` — compiler and CLI integration-test conventions.

## Goal

Remove the temporary pointer staging boundary so valid pointer programs reach
the native backend through `compile`. Add public-boundary execution and failure
coverage, completing the Pointers parent task.

## Implementation

- `src/semantic/mod.rs` and `src/lib.rs` — remove the temporary
  `unimplemented_pointer` diagnostic and the pipeline call that stops verified
  pointer IR before native emission. Remove its now-unused diagnostic support.

- `tests/fixtures/programs/pointers.fern` — add a reusable end-to-end pointer
  program that exercises mutable and const addresses, explicit and implicit
  dereference, indirect assignment, pointer copies through aggregates and a
  call or return, `null`, equality, and pointer-to-`uint` conversion; exit with
  a distinct success status.

- `tests/compiler.rs` — replace the staging-guard assertion with compiler
  integration coverage that compiles and executes the pointer fixture. Add
  runtime null-dereference cases for explicit dereference and the implicit
  field, index, and `len` forms; assert abnormal termination, the null-pointer
  diagnostic, and the original source location. Keep the source-file-removal
  case where needed to prove diagnostics are embedded in the executable.

## Tests

- The pointer fixture compiles through the public API and exits with its
  expected status, including pointer-bearing recursive aggregates and zero
  pointer globals.
- Each explicit or implicit null pointer access compiles, aborts at runtime,
  and identifies the corresponding Fern source location on standard error.

## Extra validation

This task completes the Pointers milestone. Run `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, and `cargo test`; confirm no
pointer staging diagnostic or alternate pointer-emission path remains.
