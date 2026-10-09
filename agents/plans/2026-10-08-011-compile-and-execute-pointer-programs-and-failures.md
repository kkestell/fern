# Compile and execute pointer programs and failures

## Goal

Pointers get the end-to-end coverage every other language area has:

- a program fixture that runs every example in
  `docs/spec.md#pointers-and-heap-memory`;
- native tests of the `out of memory` and `null pointer` traps;
- CLI tests of the section's compile-time rejections;
- a pointers stop on the `examples/` tour, linked from `README.md`.

## Related code

- `tests/fixtures/programs/structs.fern` and `tests/compiler.rs`
  `struct_programs_execute` — the fixture pattern: one program whose exit
  status encodes the checks.
- `tests/compiler.rs`:
  - `a_null_pointer_conversion_reports_its_source_location` — already covers
    the `null pointer` trap through the CLI;
  - `source_failures_preserve_output` — the CLI rejection pattern;
  - `documented_examples_compile` — compiles every file in `examples/`.
- `src/backend/tests/pointers.rs` — native pointer tests, including
  `heap_operations_call_the_c_library`, the only `out of memory` coverage
  today.
- `src/backend/toolchain.rs` `build_text` — builds QBE text into an
  executable.
- `examples/structs.fern` and the tour list in `README.md`.

## Decisions

**`out of memory` through a failing allocator.** No portable program
exhausts memory, so the native test edits the emitted QBE text. Every
`call $malloc(` becomes `call $fern_test_failing_malloc(`, and the text gets
`function l $fern_test_failing_malloc(l %size) { @start ret 0 }`. The program
runs the real trap path and must fail with `out of memory` on stderr. The
edit names a new symbol instead of defining `malloc`, so the C library's own
allocations are untouched.

**One fixture for the section's examples.**
`tests/fixtures/programs/pointers.fern` holds the examples from:

- Heap allocation;
- Address of a location, without its rejected `&q` line;
- Access through a pointer;
- Nullable pointers (the linked-list sum);
- Pointer comparison.

Each is a function, and `main` checks its result. It exits `42` when every
check passes, and with a distinct small status for the first one that fails.
`pointer_programs_execute` in `tests/compiler.rs` runs it. Two more
`main`-only sources run alongside it: freeing a target and `null`, and a
`*T(p)` on a non-null pointer.

**Rejections.** A new `pointer_programs_report_compile_time_errors` in
`tests/compiler.rs` compiles each case through the CLI. It checks the message
and that no output is written, as `source_failures_preserve_output` does.
The cases are the rules the section states:

| Source | Message |
| --- | --- |
| `const b = &q;` with `q` a `const` | `` cannot take the address of immutable binding `q` `` |
| `var p = null;` | `` `null` requires a nullable pointer type from context `` |
| `*f(x)` with `f` a function | `` cannot use function `f` as a type `` |
| `var x: *void;` | `expected a type` |
| `*q` on a `nullable *int` | `` cannot dereference `nullable *int` `` |
| `p < p` | `` only `==` and `!=` are defined on `*int` `` |
| `int(p)` | `` cannot convert `*int` to `int` `` |
| `var p: *int;` | `` `*int` has no zero value `` |

**Tour stop.** `examples/pointers.fern` covers:

- `new` of a struct and an array;
- `&` of a local, `*p` reads and writes, and field and index access through
  pointers;
- a linked list of `nullable *Node` walked with `*Node(cursor)`;
- pointer equality;
- `free`.

It exits `0`. `README.md` lists it after Structs:
"[Pointers](examples/pointers.fern) covers heap allocation with `new` and
`free`, addresses, access through pointers, nullable pointers and `*T(p)`,
and pointer comparison." `documented_examples_compile` picks it up without
changes.

## Implementation plan

1. `tests/fixtures/programs/pointers.fern` — the fixture.
2. `tests/compiler.rs` — `pointer_programs_execute` and
   `pointer_programs_report_compile_time_errors`.
3. `src/backend/tests/pointers.rs` —
   `native_allocation_failure_traps_with_out_of_memory`.
4. `examples/pointers.fern` and `README.md`.

## Test plan

- `pointer_programs_execute`:
  - the fixture exits `42`;
  - the `free` program exits `0`;
  - the non-null conversion exits with its target's value.
- `pointer_programs_report_compile_time_errors` — each case in the table.
- `native_allocation_failure_traps_with_out_of_memory` — the program fails,
  and stderr contains `out of memory`.
- `documented_examples_compile` — compiles `examples/pointers.fern`. Run it
  by hand as well, expecting exit status `0`.
