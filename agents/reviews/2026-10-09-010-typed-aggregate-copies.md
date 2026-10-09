# Typed aggregate copies

## Scope and coverage

Reviewed the staged changes for
`agents/plans/2026-10-09-009-typed-aggregate-copies.md`: `emit_copy` and
`MAX_FIELD_COPY_SLOTS` in `src/backend/emitter.rs`, its four callers, the
module comment, the `agents/bugs.md` entry, and the new backend and integration
tests. Also traced the code the change depends on outside the diff:
`Layout::value_slots` in `src/backend/layout.rs`, `value_class`,
`memory_class`, and `load_operation` in `src/backend/qbe.rs`, and the
`accesses` temporaries in `place_address`. Native behavior was exercised on
arm64 Apple only; the raytracer timing was taken from the work log, not
re-measured.

## Fixed

- **Missing blank line before the new backend test**
  (`src/backend/tests/mod.rs:57`): the test sat directly against the previous
  one, unlike every other test in the file. Added the blank line.

## Findings

No open findings.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — all pass.
- Compiled and ran a probe program covering self-assignment of a struct,
  assignment of one nested field from another, a copy through a pointer into
  the same struct, and passing and returning a padded nested struct; every
  field compared equal and the program exited 0.

## Verdict

The field-wise copy is correct for every scalar width and pointer, skips only
padding, and keeps `blit` and `memcpy` for larger aggregates. Ready to commit.
