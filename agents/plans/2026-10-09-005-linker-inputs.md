# Linker inputs

## Goal

Let a Fern executable link object files, static archives, and named C libraries
through the CLI. A program can declare functions from those inputs with the
existing `extern fn` syntax and call them without a custom `CC` wrapper.

## Related code

- `src/main.rs` — parses the fixed `fern <root> -o <output>` CLI.
- `src/lib.rs` — exposes `compile` and passes the verified program to the native
  toolchain.
- `src/backend/toolchain.rs` — invokes `qbe` and links its assembly with `cc`.
- `tests/compiler.rs` — checks CLI arguments, tool failures, and output
  preservation.
- `tests/core.rs` — exercises the C ABI; its object-file test currently adds
  the object through a `CC` wrapper.
- `README.md` — owns the CLI, native tool prerequisites, and linking behavior.

## Decisions

- Accept `fern <root> -o <output> [--link <file> | -L <directory> | -l <name>]*`.
  `--link` accepts an object file or archive path. Repeated inputs retain their
  written order, after Fern's generated assembly, so static library resolution
  follows the command line. Reject missing or empty values as usage errors.
- Expose `LinkInput::{File(PathBuf), SearchPath(PathBuf), Library(OsString)}` and
  `compile_with_links(root, output, &[LinkInput])` in the library. Keep `compile`
  as the zero-input convenience call used by existing clients.
- Pass each link input to `cc` as a distinct process argument: a file path,
  `-L` plus its directory, or `-l` plus its name. Let `cc` report missing files,
  libraries, and incompatible inputs. Keep the existing temporary-output and
  failure behavior.

## Test plan

- Replace the `CC` wrapper in `c_abi_small_integers_and_struct_memory` with
  `--link <helper.o>`; compile, run, and verify the C helper is called.
- Build a small C object and archive it as `libhelper.a` with the host archiver.
  Compile a Fern caller with `-L <temp-directory> -l helper`, then run it and
  verify the helper's result. Check that repeated link inputs reach `cc` in
  order.
- Compile and run a Fern program with `-l m` that calls `fegetround()` and
  `fesetround()` from the system math library and `puts()` from the system C
  library. Check the return values and printed output; the extern signatures
  use only supported integer and pointer types.
- Extend CLI error cases for missing option values and unknown options. A
  missing named library must report the `cc` failure and preserve an existing
  output executable, with temporary files removed.
- Existing `fern::compile` callers and ordinary CLI invocations still build
  without extra link inputs.

## Implementation plan

1. In `src/lib.rs`, add `LinkInput` and `compile_with_links`; route `compile`
   through it with an empty input list.
2. In `src/main.rs`, parse repeated `--link`, `-L`, and `-l` options after the
   output path, retaining their order and passing them to `compile_with_links`.
3. In `src/backend/toolchain.rs`, append link inputs to the `cc` command after
   generated assembly and before publishing the executable.
4. In `tests/core.rs` and `tests/compiler.rs`, add the native linking and CLI
   coverage from the test plan.
5. Update `README.md` with the new invocation, link-input behavior, examples,
   and the archiver prerequisite for native tests.

## Documentation updates

- `README.md` — document the CLI link options and replace the statement that
  no additional library selection interface exists.
