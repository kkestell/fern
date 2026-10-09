# Trap on stack overflow

## Scope and coverage

Reviewed all staged changes against
`agents/plans/2026-10-08-003-trap-on-stack-overflow.md`, including allocation
accounting, caller checks, entry wrapping, function naming, aggregate return
storage, source spans, tests, documentation, and the IR snapshots. Traced the
native toolchain and shared layout calculation. Native verification used macOS
arm64; the Linux bounds-discovery branch was inspected but not executed.

## Fixed

None.

## Findings

### Medium

#### Correctness

- **Unbounded QBE spills can bypass stack checks**
  (`src/backend/emitter.rs:350`): Allocation accounting excludes QBE spill
  slots. A fixed reserve cannot guarantee room for a callee's complete frame,
  so the entry check can allow a frame that faults before its first Fern call
  check. The limitation is already acknowledged in `agents/bugs.md` and the
  work log, but remains contrary to the plan's universal frame-safety goal and
  the specified stack-overflow trap. Confirmed with a program containing a
  zero-initialized global `data: [18000]u64`, an empty `sink` with 18,000 `u64`
  parameters, and `main` calling `sink(data[0], ..., data[17999])`. Compiled
  through the normal CLI, its executable exits with SIGSEGV and empty stderr
  under a 128 KiB stack limit. With 256 KiB it aborts with the source-located
  `stack overflow` diagnostic; with 8 MiB it exits zero. `main` has only its
  eight-byte rounded marker allocation in the checked storage total, while
  its many simultaneously live argument values require spill storage. Fix by
  checking a proven upper bound on the complete generated frame before entry,
  or by integrating native frame sizing/probing into compilation. This needs
  a new implementation plan because the existing plan explicitly relies on
  allocation totals and a fixed reserve; raising the reserve does not bound
  spills. No language-design decision is needed.

## Checks run

- `cargo test stack_ -- --test-threads=1`: four backend tests and one compiler
  integration test passed.
- `cargo test`: all 299 unit tests and 59 integration tests passed.
- Generated and compiled the 18,000-argument reproducer using `cargo run --
  /tmp/fern-stack-review/spill.fern -o /tmp/fern-stack-review/spill`.
- Ran that executable in subprocesses with `RLIMIT_STACK` set to 128 KiB,
  256 KiB, and 8 MiB: SIGSEGV without stderr, SIGABRT with the stack diagnostic,
  and successful exit respectively.
- Inspected the staged diff and snapshot changes; the snapshots add only
  `main_span`.

## Verdict

No findings fixed; one medium finding remains open. The planned recursion and
large-allocation cases work, but complete stack-overflow safety remains
unfinished. The finding is recorded in `agents/bugs.md`. Commit the reviewed
work and review together as required by `AGENTS.md`.
