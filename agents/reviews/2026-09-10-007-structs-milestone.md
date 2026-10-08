# Structs milestone review

## Scope and mode

General review of the complete structs milestone: syntax, semantic checking,
IR, verification, native layout and emission, tests, completion fixture, public
examples, plans, roadmap, and documentation. The corpus includes the committed
frontend-through-IR work and the current backend and integration changes. This
report supersedes `2026-09-10-006-structs-milestone.md`, whose review was not
complete enough to support its conclusions.

## Findings

### High: source-controlled aggregate sizes abort the compiler

`src/semantic/model.rs:308-323` recursively materializes every scalar of a zero
value in `Vec`s. It converts a source-controlled array length to `usize` and
allocates that many constants without a checked aggregate-size limit.
`src/backend/layout.rs:34-39` also computes array byte sizes with unchecked
`u64` multiplication; the struct layout at `src/backend/layout.rs:102-110` uses
the same unchecked size arithmetic for fields and padding.

Two small valid inputs reproduce compiler panics. A struct with an
`[9223372036854775807]int` field and a `Huge { ... }` literal panics with
`capacity overflow` in `alloc::raw_vec`. A function parameter of that array type
panics at `src/backend/layout.rs:37` with `attempt to multiply with overflow`.
In release builds, unchecked arithmetic can instead wrap and describe the wrong
layout. The specification permits resource limits, but requires a compilation
diagnostic rather than a compiler crash or changed value.

Validate aggregate element counts and byte sizes with checked arithmetic before
they reach constant construction or the backend. Make zero-value construction
fallible or keep repeated zeroes compact, and report an aggregate-size limit at
the source annotation or literal.

### High: deep acyclic struct graphs overflow the compiler stack

`src/semantic/annotations.rs:42-55` resolves a named field by calling
`resolve_struct_fields`, and `src/semantic/annotations.rs:77-119` recursively
does the same for each contained struct. The parser's nesting guard does not
bound this declaration graph. A generated legal program containing 20,001
acyclic structs, each holding the next, makes the compiler abort with `stack
overflow` while resolving fields. The recursive IR verification and layout
walks at `src/ir/verify.rs:273-290` and `src/backend/layout.rs:34-47` would also
need to respect the same bound.

Resolve the containment graph iteratively, or enforce a documented maximum
containment depth with a source diagnostic before recursion. Add a child-process
regression test so a compiler abort is observed as a test failure instead of
terminating the test process.

### High: the completed milestone has no public structs example

`examples/` has no structs program, and `README.md:9-32` omits structs from the
language tour. The only complete program is
`tests/fixtures/programs/structs.fern`, which is a regression fixture rather
than public documentation. This violates the repository's separate public
example contract and lets `documented_examples_compile` pass without testing a
documented structs program.

Add `examples/structs.fern` covering declarations, literals and fill, field
selection and assignment, copying, and equality. Add it to the README tour; the
existing documented-example test will then compile it.

### Medium: aggregate equality emits code proportional to every scalar slot

`src/backend/layout.rs:61-64,127-145` first allocates a `Vec` containing every
scalar slot in an aggregate. `src/backend/qbe.rs:74-110` then emits six QBE
instructions per slot. Array lengths are source-controlled, so one comparison
causes linear compiler memory, generated IL, optimization work, and native code
size even though the comparison could use a fixed-size runtime loop.

On this host, compiling an otherwise empty function that compares two
`[10]int` parameters took 0.33 seconds and produced a 16 KiB executable. The
same source with `[10000]int` took 14.80 seconds and produced a 2.1 MiB
executable. This new common aggregate path also regresses integer-array
comparison, which previously did not emit one instruction sequence per element.

Emit aggregate equality recursively: compare struct fields through one helper
and generate a runtime loop for array elements. That preserves scalar floating
semantics without unrolling source-declared lengths into QBE instructions.

## Topic verdicts

- Ownership: aggregate storage and copies have clear owners; no separate
  ownership finding.
- Error handling: the two compiler-abort findings bypass Fern diagnostics.
- API design: `StructId`, field ordinals, and one layout owner are coherent; no
  separate API finding.
- Performance: aggregate equality has the measured code-size and compile-time
  finding above.
- Testing: ordinary behavior is broad, but resource-bound cases and the public
  example are absent.
- Readability: the phase-specific code is locally clear; no separate finding.
- Concurrency: the change adds no concurrent or shared-state boundary.
- Security: small untrusted inputs can panic or abort the compiler through
  unchecked aggregate size and depth.
- Correctness: ordinary struct layout, copying, access, and equality pass; legal
  extreme inputs do not receive the required diagnostic.
- Unsafe: no unsafe Rust or new FFI boundary was introduced.
- Architecture: layout has one backend owner, but recursive aggregate walks need
  one enforced resource contract across phases.
- Dependencies: no dependency or feature change was introduced.
- Documentation: the supported public language tour omits structs.

## Unresolved suspicions

Native aggregate ABI coverage is limited to the current host. A portability
pass should verify layouts, calls, and returns on a 32-bit target after aggregate
size handling is fixed.

## Checks

- `git diff --check`
- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test` — 277 unit tests and 55 integration tests passed
- `cargo tree --duplicates` — no duplicate dependency versions
- huge struct-fill probe — compiler panicked with `capacity overflow`
- maximum-length array parameter probe — compiler panicked on layout overflow
- 20,001-type acyclic struct-chain probe — compiler aborted on stack overflow
- aggregate equality scaling probe — 0.33 s/16 KiB at 10 elements and
  14.80 s/2.1 MiB at 10,000 elements

## Recommendation

Do not accept the milestone until the three High findings are fixed. Then run a
focused correctness and security review of aggregate resource bounds and a
performance review of the revised equality emission.
