# Slice comparison removal review

Scope: the current comparability implementation and the repository-wide removal
of slice comparison behavior, excluding historical mentions in `eng/todo.md`
and `eng/plans/` as requested.

Mode: general.

Selected topics: correctness (the comparability rule and diagnostics), error
handling (the user-facing rejection), performance (the recursive type walk),
testing (semantic and IR enforcement and stale regression material),
architecture (the shared phase-independent predicate), readability (removed
special-case residue), and documentation (superseded behavior outside the
allowed historical files).

## Findings

### High: comparability takes exponential time on repeated struct types

`src/comparability.rs:6-13` recursively expands every occurrence of a struct
without memoizing its `StructId`. A compact type graph can therefore make one
equality check take exponential time. A 29-line invalid program whose `T26`
has two `[1]T25` fields took 5.56 seconds to reject; the same reproduction at
depths 20, 22, and 24 took 0.13, 0.39, and 1.43 seconds. The source remains far
below Fern's containment and aggregate-layout limits, so slightly deeper input
can hold the compiler for minutes before it emits a diagnostic.

Memoize struct comparability by `StructId` within the query, as `Layouts` does
for layout derivation. Unifying obviously different operand types before the
walk would also avoid this cost on the reproduced invalid comparison, but does
not replace memoization for valid repeated types.

### Medium: slice ordering reports the removed equality behavior

`src/semantic/expressions.rs:1230-1242` still sends every non-scalar ordering
comparison through the diagnostic ``only `==` and `!=` are defined on ...``.
Consequently `a < b` for two `[]int` values is correctly rejected but tells the
user that slice equality exists, directly contradicting `docs/spec.md` and the
purpose of this change. `src/semantic/tests/expressions.rs:1477-1481` preserves
that stale message as its expected result.

Distinguish non-comparable aggregate types before emitting the aggregate
ordering diagnostic, and make the slice test expect a message that does not
claim any comparison operator is defined.

### Medium: old elementwise slice equality remains outside the allowed history

The removal still leaves instructional material and a regression assertion for
the former implementation:

- `docs/spec.md:1499-1520`, `README.md:30-32`, and
  `examples/slices.fern:11-22,42-43` teach the old elementwise result through an
  `equal` loop.
- `tests/fixtures/programs/slices.fern:19-30,76` retains the same behavior as
  integration coverage.
- `src/backend/tests/structs.rs:241` preserves the deleted helper name in an
  absence assertion.
- `eng/reviews/2026-09-11-004-slices-milestone.md:19-43,98,107,118` still says
  slice equality is elementwise and records the deleted helper's behavior.

These are traces of the superseded behavior, while the requested exceptions
cover only the TODO and plans. Keep the normative statement that slices are not
comparable, but remove the replacement recipe, fixture cases, old helper-name
assertion, and obsolete portions of the earlier review. The fixture can retain
ordinary slice coverage through indexing and length.

### Low: the pointer cutoff in comparability has no focused coverage

`src/comparability.rs:8` deliberately makes every pointer comparable without
examining its target. The new accepted semantic case at
`src/semantic/tests/expressions.rs:1533-1542` covers only `*int`, so no test
proves that `*[]int` and a struct containing `*[]int` remain comparable. A
future change that descends through pointer targets would over-reject both and
could recurse through pointer cycles without an existing test failing for the
intended reason.

Add focused semantic and verified-IR acceptance coverage for a pointer to a
non-comparable target, without retaining the old slice-value comparison.

## Topic verdicts

- Correctness: the predicate enforces the specified scalar, pointer, array,
  struct, and slice rule, but slice ordering emits a false contract.
- Error handling: the equality diagnostics carry the operator span; the stale
  ordering diagnostic is the finding above.
- Performance: the recursive predicate has the measured exponential path
  above.
- Testing: rejection reaches semantic checking and IR verification, but an old
  backend-symbol regression remains and the pointer cutoff is uncovered.
- Architecture: one predicate shared by semantic checking and IR verification
  fits the recorded phase boundary; no second implementation was found.
- Readability: the implementation is locally clear; the stale slice test name
  and diagnostic are covered by the findings above.
- Documentation: the normative non-comparability rule is consistent, but the
  old elementwise recipe and earlier review remain outside the allowed history.

## Checks run

- `git diff --check` — passed.
- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed (333 unit and 61 integration tests).
- `cargo test --doc` — passed (0 doc tests).
- `cargo doc --no-deps` — passed.
- A direct compiler reproduction confirmed that slice ordering reports
  ``only `==` and `!=` are defined on `[]int` ``.
- Timed compiler reproductions at repeated-struct depths 20, 22, 24, and 26
  confirmed approximately fourfold growth for every two added declarations.
- Repository-wide searches for slice comparison and the deleted `$sliceequal`
  helper found the stale artifacts listed above; `eng/todo.md` and
  `eng/plans/` were excluded as requested.

## Unresolved suspicions

None.
