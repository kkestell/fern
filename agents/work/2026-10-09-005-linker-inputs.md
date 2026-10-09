# Linker inputs

## Plan

`agents/plans/2026-10-09-005-linker-inputs.md`

## Summary

The CLI accepts repeated `--link`, `-L`, and `-l` options after the output and
passes them to `cc`, in order, after the generated assembly. The library
exposes `LinkInput` and `compile_with_links`. The plan's goal is met.

## Departures from the plan

- The archive test also passes a second object with `--link` after the
  archive, so one CC-wrapper log checks the order of all three input kinds. On
  macOS, `ld` resolves archive members regardless of position, so the test
  checks argument order rather than order-dependent symbol resolution.
- The archiver prerequisite is listed under Development in `README.md`, beside
  the existing native-test requirement, rather than under Build and run.

## Decisions

- A link option's value that begins with `-` is a usage error, matching the
  root and output operands, so `--link -l m` reports a missing value rather
  than linking a file named `-l`. The README's `./` prefix rule covers these
  values too.
- Joined forms such as `-lm` and `-Ldir` are rejected; each option takes a
  separate value.
- `c_abi_small_integers_and_struct_memory` no longer needs a shell wrapper, so
  it drops `#[cfg(unix)]`.
- `system_libraries_link_by_name` uses `FE_UPWARD` by target architecture
  (`0x400000` on aarch64, `0x800` on x86_64); `FE_TONEAREST` is 0 on both.

## Checks run

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed.

## Manual verification

1. A Fern program calls a C function from an object file given with `--link`.

   ```sh
   d=$(mktemp -d)
   printf 'int helper(int x){return x+1;}' > $d/h.c && cc -c $d/h.c -o $d/h.o
   printf 'extern fn helper(x:i32)->i32;\nextern fn exit(s:i32)->void;\nfn main()->void { exit(helper(41)); }\n' > $d/main.fern
   cargo run -q -- $d/main.fern -o $d/program --link $d/h.o && $d/program; echo $?
   ```

   Exited 42.

2. A link option without a value prints the usage.

   ```sh
   cargo run -q -- $d/main.fern -o $d/program -l
   ```

   Printed the usage line and exited 1.
