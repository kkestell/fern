# Compile and execute pointer programs and failures

## Scope and coverage

The staged changes for
`agents/plans/2026-10-08-011-compile-and-execute-pointer-programs-and-failures.md`:

- `tests/fixtures/programs/pointers.fern`;
- `pointer_programs_execute` and `pointer_programs_report_compile_time_errors`
  in `tests/compiler.rs`;
- `native_allocation_failure_traps_with_out_of_memory` in
  `src/backend/tests/pointers.rs`;
- `examples/pointers.fern` and its `README.md` tour entry;
- the `agents/todo.md` item.

The review checked each fixture function against its example in
`docs/spec.md#pointers-and-heap-memory` and each rejection case against the
rule the section states. It also confirmed that every rejection fails on the
intended diagnostic, not on an earlier one, and that the out-of-memory test
cannot pass unless the allocator substitution takes effect. The `null pointer`
trap is covered by the existing
`a_null_pointer_conversion_reports_its_source_location`.

## Fixed

None.

## Findings

No open findings.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 358 unit and 62 integration tests pass.
- `cargo run -q -- examples/pointers.fern -o target/pointers && target/pointers`
  — exits `0`.
- Compiled the `*void`, `*f(x)`, and `*q` rejection sources through the CLI;
  each reports the expected diagnostic at the expected operand.

## Verdict

The changes meet the plan and the todo item. No findings.
