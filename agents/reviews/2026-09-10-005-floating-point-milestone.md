# Floating-point milestone review

## Scope and mode

General review of the completed `Floating-point numbers` milestone: its
uncommitted frontend, semantic, IR, backend, test, fixture, example,
dependency, roadmap, plan, and README changes. The unrelated untracked
`docs/integers.md` is outside this review.

## Findings

No findings.

## Topic verdicts

- Ownership: The new exact and concrete constant forms have clear ownership;
  the required clones preserve checked constants across binding and array
  boundaries.
- Error handling: Invalid source and compile-time constant failures remain
  diagnostics. Runtime conversion guards report through the existing trap
  path.
- API design: `Float`, `Literal`, and the numeric predicates make format and
  immediate-value distinctions explicit without a parallel pipeline.
- Performance: Exact constant work is bounded. The added checks and
  elementwise float-array equality are proportional to the specified work.
- Testing: Parser, semantic, IR, QBE, native backend, fixture, example, and
  compiler integration coverage exercise the milestone's main paths.
- Readability: The integer and floating backend paths share common QBE state
  while keeping their distinct operations explicit.
- Concurrency: Nothing in the change adds shared state or concurrent work.
- Security: Source validation stays before parsing and generated trap data is
  encoded through the existing QBE data helper.
- Correctness: The checked program retains exact untyped values until one
  rounding step, IR literals retain concrete bits, and native tests cover
  arithmetic, comparisons, conversions, signed zero, NaN, arrays, calls, and
  globals.
- Unsafe: No unsafe Rust or new FFI boundary was added.
- Architecture: The milestone uses the established frontend, semantic, IR,
  verification, and backend boundaries with one literal representation shared
  by operands and globals.
- Dependencies: `num-rational` is the narrow dependency needed for exact
  decimal constant evaluation.
- Documentation: The roadmap completion state, plans, public example, and
  README describe the implemented feature and point to the specification.

## Unresolved suspicions

No confirmed issue. The native suite runs on the current host only, so it does
not establish QBE conversion behavior on other supported targets or 32-bit
hosts. A future portability review should cover that boundary.

## Checks

- `git diff --check`
- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test`
- `cargo test --test compiler` — 53 passed

## Recommendation

No deep follow-up is required before accepting this milestone. If portability
becomes a near-term concern, review correctness on additional QBE targets.
