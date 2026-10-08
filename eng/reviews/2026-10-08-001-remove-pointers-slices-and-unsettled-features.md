# Remove pointers, slices, and unsettled features

## Scope and coverage

The staged changes for
`eng/plans/2026-10-08-001-remove-pointers-slices-and-unsettled-features.md`:
the frontend, `types.rs`, `layout.rs`, semantic checking, IR model, lowering,
and verification, the backend, the tests, examples, and fixtures, and
`docs/spec.md`, `README.md`, `AGENTS.md`, and `eng/architecture.md`. The review
traced the location, lowering, and verification paths the change simplified,
and the places that relied on checks the IR verifier dropped. It covered
deleted plans and reviews only by confirming that nothing still links to them.

## Fixed

- **Indexing an element place across a block split is an internal compiler
  error** (`src/ir/lower.rs:1036`): `grid[i][pick(b && b)]` lowered `grid[i]`,
  including the load of `i`, before the short-circuiting index split blocks,
  then built the element place in the index's final block. Verification
  rejected the program with "IR value 1 is not defined earlier in block 3".
  The same failure exists before this change. `element_place` now holds its
  base with the new `hold_place`, so the base is rebuilt in whichever block
  the index lowering ends in, the way assignment targets already are.
  `an_element_read_rebuilds_its_base_after_a_split_index` covers it and fails
  without the fix.
- **Ragged doc comment** (`src/semantic/constants.rs:620`): the edit to
  `compare_constants` left the comment wrapped mid-sentence. Rewrapped.

## Findings

None.

## Checks run

- `cargo fmt --check`: passed.
- `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo test`: 284 unit tests and 57 integration tests passed.
- Every example compiles and runs. `control_flow` exits 6, and every other
  example exits 42.
- `grid[i][pick(b && b)]` failed with the internal compiler error before the
  fix, including at `HEAD`, and exits 4 after it.
- Diagnostics for `len` and `for … in` on an `int`, an assignment through a
  grouped field of a `const` struct, and `null` and `string` used as binding
  names behave as the plan states.
- No source, test, example, or document outside `eng/` still mentions
  pointers, slices, `null`, runes, strings, or comparability as Fern features.
  Every `docs/spec.md` anchor link resolves.

## Verdict

The change does what the plan describes. One internal compiler error that
predates it and one ragged comment were fixed. No findings remain open.
