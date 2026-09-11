# Compile and execute slice programs and failures

## Sources

- `docs/spec.md#slices` — slice values, copying, slicing expressions and their
  bounds, indexing, `len`, and comparison.
- `eng/todo.md#slices` — the final Slices subtask and its completion boundary.
- `eng/architecture.md#pipeline` and `#diagnostics` — the public compiler
  boundary and source-mapped runtime diagnostics.
- `eng/plans/2026-09-11-011-lay-out-slice-and-access-slices-in-native-code.md`
  — completed native slice emission and the staging diagnostic it leaves for
  this slice.
- `eng/plans/2026-09-11-004-compile-and-execute-pointer-programs-and-failures.md`
  — the shape this task follows at the same boundary one milestone earlier.
- `tests/compiler.rs` — compiler and CLI integration-test conventions.
- `README.md` — the ordered example tour.

## Goal

Remove the temporary slice staging boundary so valid slice programs reach the
native backend through `compile`. Add public-boundary execution and failure
coverage and the example tour entry, completing the Slices parent task.

## Implementation

### Remove the staging guard

- `src/semantic/expressions.rs` — delete `reject_slice_values` and its doc
  comment. Drop the now-unused `std::cmp::Reverse` import; keep
  `AnnotationKind` only if another use remains in the file.
- `src/lib.rs` — delete the `semantic::expressions::reject_slice_values(&syntax)`
  call in `compile`, leaving `ir::lower::lower(checked).verify()?` followed
  directly by `backend::toolchain::build`.
- `src/semantic/tests/mod.rs` — delete the `rejects_slice_value` helper, the
  `slice_staging_guard_reports_the_earliest_slice_syntax` test, and the
  `expressions::reject_slice_values` import.
- `src/semantic/tests/expressions.rs` — delete
  `slicing_in_value_position_reaches_the_temporary_implementation_guard`;
  `slicing_checks_types_locations_and_bounds` already covers slicing in value
  position.
- `src/semantic/tests/annotations.rs` — replace
  `slice_annotations_resolve_before_the_temporary_implementation_guard` with an
  acceptance test over the same annotation positions, using `accepts_source` on
  each: a module-level `var`, a local `var`, a parameter, a function result, a
  struct field that names its own struct, an array element, and a nested `[][]int`.
  Keep the position list intact; only the assertion changes from the guard's
  message and span to successful checking.

### Fixture and end-to-end coverage

- `tests/fixtures/programs/slices.fern` — a reusable end-to-end slice program
  that exercises, in one `main` plus a few helpers:
  - all four slicing forms over a mutable array local, an immutable array
    local, a `*[N]T`, and a slice;
  - reading and writing elements through a slice and observing the write in the
    backing array and in a second slice of it;
  - `len` of a slice binding, a slicing expression, a parameter, and the empty
    slice;
  - copying a slice through a binding, a call argument, a result, a struct
    field, and an array element, and passing a `[]T` where `[]const T` is
    expected;
  - a module-level `var` of slice type observed to have length `0`;
  - `for v in s` and `for v, i in s`;
  - equality of slices over different arrays, of differing lengths, of two
    empty slices, and across `[]T` and `[]const T`;
  - a struct that reaches itself through a slice field.

  Exit `44` on success and a distinct small status from each check that fails,
  matching the style of `tests/fixtures/programs/pointers.fern`.

- `tests/compiler.rs`:
  - Delete `slice_annotations_preserve_output_at_the_compiler_staging_diagnostic`
    and `slice_values_preserve_output_at_the_compiler_staging_diagnostic`.
  - Add `slice_programs_compile_and_execute`, compiling
    `include_str!("fixtures/programs/slices.fern")` through `fern::compile` and
    asserting exit code `44`.
  - Add a runtime-failure test in the shape of
    `null_pointer_dereferences_abort_with_the_original_source_location`: compile
    each source, remove the input file, run, and assert abnormal termination,
    the expected message, and `input.fern:` on standard error. Cover an index
    past the length, a negative index, an index into the empty slice,
    `lo > hi`, `hi > len`, a negative `lo`, and a slicing expression through a
    null `*[N]T`, which reports `null pointer dereference`. The runtime
    messages are `slice index out of range for `[]int`` and
    `slice bounds out of range for `[]int``.
  - Add slice cases to the compile-time failures that preserve an existing
    output, following `typed_integer_diagnostics_precede_emission_and_preserve_output`:
    `var a: [4]int = [1, 2, 3, 4]; var e = a[5:];` reports
    `slice bound 5 is out of range for `[4]int``, and
    `var s: []const int; s[0] = 1;` reports
    `cannot assign to an immutable location`.

### Example tour

- `examples/slices.fern` — a topic example in the style of `examples/arrays.fern`
  and `examples/pointers.fern`: comments that teach, no diagnostics, and an
  `exit` of a computed value. Cover slice types and their constness, the four
  slicing forms, writing through a slice, `len`, the empty zero value, copying,
  iteration, and comparison.
- `README.md` — add a `Slices` bullet directly after the `Pointers` bullet,
  naming what the example covers in the style of its neighbours.

## Tests

- The slice fixture compiles through the public API and exits `44`.
- Each runtime slice index and bounds violation compiles, aborts, prints its
  message, and identifies the original Fern source location after the source
  file is gone.
- A constant out-of-range slice bound and a write through a `[]const T` element
  fail before emission and leave an existing output untouched.
- `documented_examples_compile` compiles `examples/slices.fern` with no change
  to that test.

## Extra validation

This task completes the Slices milestone. Run `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, and `cargo test`; confirm no
`slices are not yet implemented` diagnostic, guard helper, or guard-named test
remains anywhere in the repository.
