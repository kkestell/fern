# Compile and execute pointer programs and failures

## Plan

`agents/plans/2026-10-08-011-compile-and-execute-pointer-programs-and-failures.md`

## Summary

`tests/fixtures/programs/pointers.fern` runs every example in the spec's
pointer section, and `pointer_programs_execute` runs it with two smaller
programs. `pointer_programs_report_compile_time_errors` covers the section's
compile-time rejections. A failing stand-in allocator drives the
`out of memory` trap natively. `examples/pointers.fern` joins the tour and
`README.md`. The plan's goal is met.

## Departures from the plan

- `pointer_programs_report_compile_time_errors` calls `fern::compile` and
  checks that the existing output survives, the pattern
  `struct_source_failures_preserve_output` uses, rather than invoking the CLI
  binary.

## Decisions

- `examples/pointers.fern` frees every target it allocates, including the
  second point its comparison uses, so the tour does not leak.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 358 unit and 62 integration tests pass.

## Manual verification

1. The tour example runs.

   ```sh
   cargo run -q -- examples/pointers.fern -o target/pointers && target/pointers; echo $?
   ```

   Prints `0`.
