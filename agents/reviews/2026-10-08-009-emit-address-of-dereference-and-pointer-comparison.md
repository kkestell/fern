# Review: emit address-of, dereference, and pointer comparison

## Scope and coverage

Reviewed the staged changes for
`agents/plans/2026-10-08-009-emit-address-of-dereference-and-pointer-comparison.md`:
`src/backend/emitter.rs`, `src/backend/layout.rs`, `src/backend/qbe.rs`, the
new `src/backend/tests/pointers.rs`, the narrowed driver guard in
`src/semantic/annotations.rs`, `src/semantic/mod.rs`, and `src/lib.rs`, and the
guard cases in `tests/compiler.rs`. Traced the code they depend on: aggregate
argument borrowing and the addressed-binding set in `src/ir/lower.rs`, the
lowering of `new`, `free`, and `*T(p)`, static data emission, aggregate call
results, and the pointer rules in `docs/spec.md`.

Heap allocation, `free`, and the `*T(p)` null check stay behind
`reject_heap_syntax` by design; their emission belongs to the next todo item
and was not reviewed.

## Fixed

- **Backend leaf vocabulary omits pointers** (`src/backend/layout.rs:85`,
  `src/backend/qbe.rs:71`, `src/backend/emitter.rs:62`, `src/ir/model.rs:241`):
  `scalar_slots` and the comments on static data, aggregate equality, and
  `Global::values` described scalar leaves only, while pointers are now leaves
  too, and the aggregate comparison comment credited `scalar_slots` with work
  it does not do. A reader extending static data or equality would miss the
  pointer case. Renamed `scalar_slots` and `collect_scalar_slots` to
  `value_slots` and `collect_value_slots`, matching `value_class`, and updated
  the comments to name scalars and pointers.
- **Guard diagnostic helper keeps its old name**
  (`src/semantic/annotations.rs:19`): `pointers_not_implemented` now reports
  only heap allocation and null checks. Renamed it `heap_not_implemented`.

## Findings

No open findings.

## Checks run

- `cargo fmt --check`: clean.
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo test`: 352 unit and 59 integration tests pass, before and after the
  fixes.
- Native programs compiled with the built compiler: arrays of pointers built
  with a fill and written through; `len` and whole-array assignment through a
  `*[3]int`; an aggregate result loaded through a pointer; `null` on the left
  of `==`; a module-level `[4]nullable *int` filled with `null` and later
  assigned; a module-level struct array whose pointer field is assigned;
  equality of nested arrays of nullable pointers; and a pointer result
  compared with `&x`. All produced the expected exit status.
- Confirmed that Fern has no zero-sized types (`[0]int` and empty structs are
  rejected), so distinct live targets always have distinct addresses, and that
  a module-level `&g` initializer is rejected as non-constant.

## Verdict

The change meets its plan. Two naming and comment findings were fixed; none
remain open.
