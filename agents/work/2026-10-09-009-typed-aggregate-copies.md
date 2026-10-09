# Copy small aggregates field by field

## Plan

`agents/plans/2026-10-09-009-typed-aggregate-copies.md`

## Summary

`emit_copy` copies an aggregate of at most eight scalars and pointers with one
typed load and store each, and keeps `blit` and `memcpy` for larger ones. The
raytracer renders in about 13.4 CPU seconds instead of 16.6, with
byte-identical output. The plan's goal is met.

## Departures from the plan

- The `field_wise_copies_keep_every_width` struct has no standalone `i32`
  field: with it the struct would hold nine slots and take the `blit` path.
  The pointer's `i32` target covers that width.
- The backend test passes its structs through pointer parameters because Fern
  has no array fill syntax to build `[16]u8` and `[9]f64` values.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — all pass.

## Manual verification

1. The integration test program uses only field-wise copies.

   ```sh
   QBE=<wrapper that saves its input> cargo run -- <test program> -o fw && ./fw
   ```

   Exit status 0; the saved QBE has no `blit` or `memcpy`.

2. Raytracer timing against `main`, two runs each, release builds.

   ```sh
   target/release/fern examples/raytrace.fern -o rt && /usr/bin/time -p ./rt > out.ppm
   ```

   `main`: 16.51 and 16.69 user seconds. This change: 13.51 and 13.34. The
   two outputs compare equal with `cmp`.
