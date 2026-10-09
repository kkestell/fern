# Zero-initialize a `var` without an initializer

## Plan

`agents/plans/2026-10-09-010-zero-initialized-vars.md`

## Summary

A `var` with a type annotation may omit its initializer, at module level and
in a function. A module-level one is a `Global` with no values, emitted as
zero-fill storage; a local is zeroed in place by `Instruction::Zero`, which
emits typed stores, word stores, or one `memset` call. No semantic `Constant`
or static data holds a zero value. The plan's goal is met.

## Departures from the plan

- `tests/core.rs` `zero_values` has no `nullable *int`: the language has no
  nullable pointer type. Every `*T` may be null, so `*int` covers pointers.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — all pass.

## Manual verification

1. A 1,000,000-element zero array, local and module-level, compiles in
   constant memory and adds no data to the binary.

   ```sh
   cargo build --release
   printf 'var g: [1000000]i32;\nfn main() -> void {\n    var a: [1000000]i32;\n    a[999999] = 1;\n    g[0] = a[999999];\n}\n' > /tmp/big.fern
   /usr/bin/time -l target/release/fern /tmp/big.fern -o /tmp/big
   ls -l /tmp/big && size -m /tmp/big && /tmp/big
   ```

   Peak resident memory 38.6 MB, the same as compiling
   `fn main() -> void { var a: [4]i32; }` (38.6 MB). The binary is 33,480
   bytes; `g` is a 4,000,000-byte `__bss` zerofill section. It exits 0.

2. The ray tracer renders the same image as the committed version.

   ```sh
   git show HEAD:examples/raytrace.fern > target/old.fern
   target/release/fern target/old.fern -o target/old_rt
   target/release/fern examples/raytrace.fern -o target/new_rt
   target/old_rt > target/old.ppm && target/new_rt > target/new.ppm
   cmp target/old.ppm target/new.ppm
   ```

   `cmp` reports no difference. User time 13.3 s before, 12.9 s after.

## Follow-up work

- `emit_copy` calls `value_slots` before checking the copy's size, so copying
  a large array builds one slot per element at compile time. Compiling
  `var a: [1000000]i32; var b = a; var c = b;` peaks at 167 MB rather than
  39 MB. Testing the size first, as `emit_zero` does, avoids it.
