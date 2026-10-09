# Review pointer type representation

## Scope and coverage

Reviewed all staged changes for
`agents/plans/2026-10-08-005-represent-pointer-types.md` against
`docs/spec.md#pointer-types`, `#variable-declarations`, `#struct-literals`, and
`#nullable-pointers`. Traced the `Type::Pointer` and `Constant::Null`
additions through `src/types.rs`, the shared layout, annotation resolution
with its `Placement` split, deferred target-layout validation through
`pending_targets`, the zero-value rules in local and module-level declarations
and struct-literal fills, the constant-folding arms, the driver guard, and the
`unreachable!` arms in `src/ir/` and `src/backend/`.

Because pointer-typed values now reach semantic checking through parameters,
results, and fields, compiled probe programs that apply every operator,
conversion, `exit`, `len`, indexing, field selection, `for … in`, call
arguments, returns, array literals and fills, struct literals and fills,
constant folding of `null` values, and `==` on pointers and on aggregates that
hold them. Each reported a diagnostic or reached the driver guard;
none panicked. Probed self-referential, mutually recursive, and
pointer-to-array-of-self structs, zero-value propagation through nested
structs and arrays, and deferred target-layout validation in module
initializers. Pointer expressions, comparisons, and lowering belong to later
tasks and were not reviewed.

## Fixed

- **Struct resolution comments describe naming instead of by-value use**
  (`src/semantic/model.rs:265`, `src/semantic/annotations.rs:40`,
  `src/semantic/annotations.rs:179`, low, comments): `FieldState` and
  `resolve_struct_fields` said fields resolve the first time a program names
  the type, but a pointer target now names a struct without resolving it. The
  `Placement` comment said a target never has its layout validated, while the
  `Pointer` arm validates it once final. A reader could conclude that pointer
  targets resolve fields or skip layout limits. Reworded the three comments to
  state that fields resolve on by-value use and that the enclosing pointer
  validates the target's layout.

## Findings

None remain open.

## Checks run

- `cargo fmt --check` — passed before and after the fix.
- `cargo clippy --all-targets -- -D warnings` — passed before and after the
  fix.
- `cargo test` — 317 unit and 59 integration tests passed.
- `cargo test --lib semantic` after the fix — 111 tests passed.
- 48 probe programs compiled with `target/debug/fern`, as described under
  Scope and coverage — no panics; diagnostics matched the spec or the
  temporary guard.

## Verdict

The pointer type, layout, resolution, and zero-value rules match the plan and
the specification. One low comments finding fixed; no open findings. Ready to
commit.
