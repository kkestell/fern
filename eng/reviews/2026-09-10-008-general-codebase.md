# General codebase review

## Scope and mode

Exhaustive general review of the entire codebase across all thirteen topics:
`src/` (frontend, semantic, IR, backend), `tests/`, fixtures, examples,
`docs/spec.md`, `eng/architecture.md`, and the repository documents.

The review read every production file and the owning documents, traced the
structs and aggregate paths end to end, and verified every suspicion with a
build, test, or reproduction. `cargo fmt --check`, `cargo clippy --all-targets
-- -D warnings`, and the full suite (`283` unit, `56` integration tests) are
clean.

## Findings

### High 1: declarations without initializers are rejected, against the specification

`src/frontend/parser.rs:794-808` (`binding`) unconditionally requires `=`
after an optional annotation. The specification's Variable declarations and
Module-level declarations sections define `var name: T;` with no initializer as
valid, receiving the type's zero value:

```text
var name: T [= e];
```

Reproduction:

```fern
var counter: int;
fn main() -> void {
    counter = counter + 2;
    exit(counter);
}
```

The compiler rejects it with `expected `=``. The specification is authoritative
where explicit, so the compiler rejects a valid program and the zero-value rule
is unreachable.

Suggested fix: make the initializer optional in `binding`, require an
annotation when it is absent (both locally and at module level), and thread the
resolved zero value through checking and lowering the way a struct-literal fill
already does. Add parser, semantic, IR, and native tests for a zero-valued
scalar, array, and struct binding.

### High 2: the fill-expansion limit is per literal, so nested fills multiply past it

`src/semantic/expressions.rs:181-188` bounds a fill by the literal's own
`length` against `MAX_AGGREGATE_INITIALIZER_VALUES` (1,000,000), and
`README.md:83-85` documents the limit as "Repeated array fills and struct
zero fills expand at most 1,000,000 scalar values". Each nested literal is
checked separately, so the total expansion multiplies across literal levels.

Reproduction:

```fern
var x: [4][4][300000]u8 = [[[0...]...]...];   // accepted
var y: [4800000]u8 = [0...];                  // rejected: limit exceeded
```

Both expand 4,800,000 scalars. The flat literal is rejected; the nested one
compiled at ~860 MB peak RSS and produced a 19 MB executable. The checker's
zero-value path composes correctly — `zero_value` uses the total-aware
`aggregate_value_count` (`src/semantic/model.rs:320-369`) — so the array-literal
fill check is the outlier. Fix: bound the folded constant by the total scalar
count of the literal's type (reuse `aggregate_value_count`), which also makes
the README's claim true.

### High 3: layout recomputation is exponential; a valid ~30-deep struct tree hangs the compiler

`src/backend/layout.rs:102-125` recomputes `struct_layout` and `struct_offsets`
on every call, recursing into each field's layout with no memoization, and
`CheckedProgram::aggregate_layout` (`src/semantic/annotations.rs:98-121`)
repeats the same derivation. A struct tree with two sub-structs per level costs
2^depth layout computations.

Measured (`cargo run --release`, programs with one binary struct tree and no
aggregate initializer, so only layout paths run):

| depth | compile time |
| ----- | ------------ |
| 24    | 0.41 s       |
| 28    | 3.4 s        |
| 30    | 12.9 s       |
| 32    | 54.6 s       |

Four-fold cost per two extra levels is pure 2^depth growth. The byte limit does
not stop this: a depth-49 binary tree of `int` fields is 2^50 bytes, exactly at
`MAX_AGGREGATE_LAYOUT_BYTES`, so a source-controlled, layout-valid program of
moderate depth hangs the compiler for hours.

Suggested fix: derive each struct's size, alignment, and field offsets once per
`StructId` (a memoized, depth-first pass over the struct table, iterative or
depth-bounded together with `MAX_STRUCT_CONTAINMENT_DEPTH`), and have semantic
validation and the backend call that one derivation. This one change also
resolves the stack-overflow finding on linear chains from
`2026-09-10-007-structs-milestone.md`, and removes the duplicated derivation
(finding 4).

### Medium 4: aggregate layout is derived in two places

The architecture document gives `backend::layout` one description of memory;
`CheckedProgram::aggregate_layout`
(`src/semantic/annotations.rs:98-121`) duplicates the backend's size,
alignment, offset, and padding rules, plus a second `align` helper. The two
agree today, but under this repository's rules a second implementation of one
rule is the drift that later alignment or layout changes turn into a real
mismatch: the compiler would accept a shape the backend lays out differently.
Fix with finding 3: one phase-neutral derivation over the checked struct table,
called by both semantic validation and the backend.

### Low: the README's fill limit is documented more tightly than it is enforced

`README.md:83-85` states the 1,000,000-scalar expansion limit that finding 2
shows is per literal. State the bound per literal, or fix finding 2 and keep
the claim.

## Suspicions (unconfirmed or unreachable)

- `SourceMap::index_at` (`src/source.rs:56-60`) computes
  `partition_point(...) - 1`, which underflows on an empty map. Every
  diagnostic in the pipeline renders after at least one file has loaded, so it
  is unreachable today; a debug assert or an early return would make the
  invariant local.
- `resolve_struct_fields` (`src/semantic/annotations.rs:148-157`) leaves
  `self.file` swapped and the struct's state `Resolving` on the duplicate-field
  error path. Compilation aborts, so nothing observes it, but a scope-guarded
  restore would remove the wrinkle.
- Native ABI coverage runs only on this host (carried from
  `2026-09-10-007-structs-milestone.md`); `int`/`uint` width comes from the
  compile host's `usize::BITS`, consistent with QBE and `cc` targeting the
  host, but a 32-bit-target pass remains untested.
- A 4-byte `blit` was verified against the installed QBE; other QBE versions'
  `blit` size divisibility rules were not.

## Topic verdicts

- Ownership: `Place`, `HeldOperand`, and layout borrows have clear owners; the
  clones serve copy-semantics boundaries the specification defines. Nothing to
  report.
- Error handling: diagnostics render once at the compile boundary with spans
  and file identity; internal invariants use stated `expect`/`unreachable!`
  and the IR verifier covers lowering bugs. Nothing to report beyond the
  `index_at` suspicion above.
- API design: the crate's surface is `compile` plus `CompileError`; internal
  types keep invalid states unrepresentable (`StructId`-based nominal
  equality, checked field ordinals). Nothing to report.
- Performance: findings 2 and 3; aggregate-equality code growth remains a
  pending task in `eng/todo.md`.
- Testing: coverage is broad; snapshots are checked in and minimal; the
  resource-limit paths findings 2 and 3 exercise have tests only in their
  narrow single-literal forms. Nothing further to report.
- Readability: phases are decomposed, names are domain-accurate, and no
  function needs "and" to describe. Nothing to report.
- Concurrency: no threads, locks, atomics, or async anywhere in the corpus.
  Nothing to report.
- Security: the resource-limit bypass in finding 2 is the security-relevant
  case (untrusted source driving compiler memory); paths, tool invocation, and
  message-data escaping behave as documented. No other finding.
- Correctness: finding 1; traced paths (overflow checks, shift rules,
  conversions, bounds checks, exit status masking, aggregate equality
  including NaN and signed zero) agree with the specification.
- Unsafe: no `unsafe` Rust or FFI in the codebase. Nothing to report.
- Architecture: finding 4 (duplicated layout derivation); otherwise phase
  separation, identity, and diagnostics boundaries hold, including the
  `backend::layout` ownership rule.
- Dependencies: no new dependency; `cargo tree --duplicates` is empty;
  `tempfile` is a production dependency (toolchain intermediates) and `insta`
  a dev-dependency, correctly placed. Nothing to report.
- Documentation: finding 1's contract gap and the README claim above.

## Checks

- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — 283 unit and 56 integration tests pass.
- `cargo tree --duplicates` — none.
- `grep unsafe` — none.
- Reproductions: no-initializer declaration rejected (finding 1); flat vs
  nested fill limit contrast with peak-RSS measurement (finding 2); timed
  deep-struct compiles at four depths (finding 3); `Huge { ... }` limit
  diagnostic through the binary; 4-byte struct copy through the native
  toolchain (blit sanity).
- `cargo-audit` is not installed; no advisory check ran.
