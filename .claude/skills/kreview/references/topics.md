# Review topics

Each section below serves both modes. In general mode, select the sections that
fit the corpus after inventorying it. In specific-topic mode, work only the
explicitly selected sections. In either mode, the relevant checklist is the
review and each applicable item deserves an answer.

For a diff, commit, or branch review, “the diff” and “changed” below mean that
change. For a directory or whole-codebase review, they mean the requested
corpus. Inventory that corpus first. In specific-topic mode, read every
production file in it and the relevant tests; then apply each selected checklist
to the matching code. A search identifies candidates to inspect. It does not
complete an exhaustive review.

## General

After inventorying the corpus and its contracts, select the topics that its
changed behavior, boundaries, and risks make material. For example, select
`unsafe` only when unsafe Rust or FFI is involved; select `api-design` and
`documentation` for public or shared contract changes; and select
`dependencies` for dependency configuration changes. Do not select a topic
solely because it exists.

Apply every relevant checklist in each selected section to the requested corpus,
follow its call paths and invariants, and verify suspicions with focused
evidence. Record the selection and its reasons, then either confirmed findings
or a one-line "nothing to report" for each selected topic. Finish with the
selected-topic verdict list and any follow-up requiring evidence outside the
corpus.

## Ownership

Review ownership boundaries, borrowing, and lifetime relationships: unnecessary
clones or allocations, values retained longer than needed, and lifetimes that
make callers or invariants hard to understand.

Read the changed types' definitions, every construction and drop site, and each
caller that passes data across the boundary.

- For each `clone`, `to_owned`, `to_vec`, `to_string`, and `collect` in the
  diff, name the ownership problem it solves. Decide whether a borrow, a move,
  `Cow`, an index, an id, or restructuring the caller removes it. A clone that
  expresses the simplest correct boundary is correct; say so and move on.
- Check each function signature for the weakest sufficient form: `&T` over `T`,
  `&str` over `&String`, `&[T]` over `&Vec<T>`, `impl AsRef` or `impl Into`
  only where callers actually vary. Check that returned owned data is not
  immediately borrowed by every caller, and that returned borrows are not
  immediately cloned by every caller.
- Trace each named lifetime to the values it ties together. Confirm the
  relationship is real rather than a way to silence the borrow checker, and that
  a caller cannot satisfy it by keeping something alive longer than intended.
- Look for `Rc`, `Arc`, `RefCell`, and `Mutex` introduced to escape a borrow
  conflict. Ask what single owner would work instead, and whether shared
  mutability now spans a wider region than the rule it protects.
- Check who owns each buffer, handle, arena, index, or interned id. Confirm
  indices and ids cannot outlive or be used against the wrong container, and
  that the container's lifetime is documented where it is not obvious.
- Look for values held across a long computation, an early `return`, a loop, or
  an `await` purely because they were bound too early.
- Check `Drop` implementations and manual cleanup for double release, leaks on
  the error path, and ordering assumptions between fields.
- Check derived and manual `Clone`, `Copy`, `Default`, `Deref`, and `Borrow` for
  semantics that surprise a caller, and for a `Copy` type large enough that
  copying it is a silent cost.

Verify with `cargo clippy --all-targets -- -D warnings`, a `grep` for each
type's construction sites, and, where a clone looks removable, an actual attempt
to remove it.

## Error handling

Review whether fallible paths use a consistent `Result` strategy, preserve
useful context, and distinguish recoverable errors from violated invariants.

Read the crate's error type, its conversions, the boundary where errors reach
the user, and every caller of the changed fallible functions.

- Enumerate every `unwrap`, `expect`, `panic!`, `unreachable!`, `todo!`,
  `unimplemented!`, `assert!`, indexing, slicing, and integer division in the
  diff. For each, state the reason it cannot fire, or report it. Reachable
  panics on production paths are findings; asserted infallible invariants with a
  stated reason are not.
- Check every `?` for the conversion it performs and the context it loses. Ask
  whether the message that reaches the user names the file, span, symbol, or
  input that caused it.
- Look for discarded errors: `let _ =`, `ok()`, `unwrap_or_default`,
  `unwrap_or(...)`, empty `Err(_) =>` arms, and ignored results from writes,
  flushes, and cleanup. Decide whether the fallback hides a real failure.
- Check that the error type distinguishes cases the caller must act on
  differently, and does not force callers to match on strings.
- Check error construction for correct classification: a user-input problem
  reported as an internal error, or an internal bug reported as a user
  diagnostic, is a finding.
- Trace failure atomicity. When a fallible step runs after a mutation, confirm
  the aborted operation leaves no partial state, half-written file, or dangling
  registration.
- Check that a failure reports once, at one place, with one exit status, rather
  than being logged deep and re-reported shallow.
- Check the panic policy against the repository's rules and against what the
  surrounding code already does.

Verify by writing or running a test that forces each failure path, and by
reading the message it actually produces.

## API design

Review public and shared interfaces for clear names, predictable behavior,
minimal visibility, ergonomic signatures, and types that make invalid states
hard to express.

Read the crate root's exports, the module's existing API, and the `krust`
guidelines when public surface changes.

- Check every new `pub` item. Ask whether `pub(crate)`, `pub(super)`, or private
  suffices. Check that nothing public leaks a private or unstable type, and that
  nothing intended as internal escaped through a `pub use`.
- Check names against the crate's existing vocabulary and Rust conventions:
  `as_`/`to_`/`into_` cost prefixes, `iter`/`iter_mut`/`into_iter`, `get` versus
  panicking indexing, `new` versus a builder, no stutter with the module path.
  Do not invent a second name for a concept the codebase already names.
- Check argument and return types for the encoding of invalid states. Prefer an
  enum over a `bool` pair, a newtype over a bare `usize`, a non-empty type over
  a `Vec` the callee must check, and `Option<T>` over a sentinel.
- Check parameter order, count, and any `bool` flag that a caller reads as a
  mystery at the call site. Check that a function with several optional inputs
  has a shape callers can use.
- Check the standard trait implementations a caller expects: `Debug`, `Clone`,
  `PartialEq`, `Display`, `Error`, `From`, `Default`, `Hash`, `Ord`, iterator
  traits. Check that derived semantics are the intended ones.
- Check for `#[must_use]` on values that are pointless to drop, and for
  `#[non_exhaustive]` only where the crate's stability policy calls for it.
- Confirm generics, trait bounds, and extension points each have a present
  caller. An abstraction with one implementor and no second in sight is a
  finding under this repository's rules.
- Check documentation on each public item for the contract a caller needs, and
  check the change against any stability promise the crate has made.

Verify by writing the call site a real caller would write, and by checking every
existing caller still reads clearly.

## Performance

Review algorithmic complexity, repeated work, allocation and copying, hot loops,
I/O patterns, and data layout where they plausibly matter.

Read the call path from the entry point to the changed code, and establish
whether this code runs once, per file, per node, or per byte.

- State the input size and call frequency before judging anything. A finding
  without that context is speculation.
- Compute the complexity of each changed loop and recursion, including the cost
  of the operations inside it. Look for a linear scan inside a loop, a
  quadratic containment check that a map or set makes linear, and repeated
  sorting or repeated traversal of the same structure.
- Look for work that is invariant across a loop, recomputed on each call, or
  computed eagerly and then discarded on most paths.
- Find allocations on the hot path: per-iteration `Vec`, `String`, `format!`,
  `collect`, boxing, and intermediate collections that an iterator chain
  removes. Check whether a reused buffer or a `with_capacity` reserve fits.
- Check copies of large values through arguments, returns, matches, and `Copy`
  types. Check whether a large enum variant should be boxed.
- Check I/O for unbuffered reads and writes, per-item syscalls, repeated `stat`
  or directory walks, and files read more than once.
- Check the data layout the change implies: pointer chasing where a flat vector
  works, a hash map where a dense index works, and per-element indirection in a
  structure the compiler walks repeatedly.
- Weigh each candidate against this repository's stated priority of clarity over
  speed. Report a slowdown that matters; do not trade readable code for a
  micro-optimization without evidence.

Verify with a measurement: time the compiler on a representative input, count
allocations or iterations with a counter or `dbg!`, or write a small benchmark.
Report the measured effect, or state plainly that the finding is unmeasured.

## Testing

Review whether tests cover boundary values, failure paths, state transitions,
feature interactions, and regressions introduced by the change.

Read the existing tests for the changed modules, the fixtures they use, and the
repository's rules about where behavioral coverage belongs.

- Map every behavior the diff adds or changes to the test that proves it. Name
  the untested ones. An unmapped behavior is the finding, not a missing test
  count.
- Check boundary values concretely: zero, one, many, empty input, maximum and
  minimum integers, off-by-one indices, first and last element, and the exact
  threshold in each comparison the diff introduces.
- Check that every error and diagnostic the change can produce has a test that
  triggers it and asserts the message the user sees.
- Check interactions, not only features in isolation: the new construct nested
  inside another, used twice, used in a loop, combined with each earlier
  milestone's features.
- Check assertions prove behavior. A test asserting a debug format, a field the
  implementation happens to set, or merely that the code did not panic is a weak
  test; say what it should assert instead.
- Check that snapshot fixtures are checked in, minimal, and readable, and that a
  snapshot update in the diff reflects an intended change rather than accepted
  drift.
- Check fixtures live where the repository says they live, and that public
  examples are not being used as regression tests.
- Check tests fail for the right reason: an assertion that passes on both old
  and new behavior proves nothing.

Verify by running the suite, and by temporarily breaking the new code to confirm
a test catches it. Report which tests you ran and what they showed.

## Readability

Review whether names, control flow, and decomposition make the code easy to
reason about locally.

Read each changed function end to end and try to state its rule in one
sentence.

- For each changed function, name its single job. A function you cannot describe
  without "and" is a finding; name the extraction.
- Count nesting levels and early exits. Look for conditions that invert into a
  guard clause, `else` branches that a `return` removes, and `match` arms that
  collapse.
- Check every name against the thing it holds: no `data`, `info`, `tmp`, `res`,
  `helper`, `handle_x`, or `do_y` where the domain has a word. Check that the
  same concept uses the same word everywhere in the change.
- Check that a reader can verify each block without holding distant state in
  mind. Flag invariants enforced far from where they are relied on, mutation
  through a wide-open `&mut` across many lines, and flags read much later than
  they are set.
- Check comments. Remove ones that narrate the code, and require one where a
  constraint, a reason, or a non-obvious choice is invisible. Under this
  repository's rules, comments never record history, corrections, or
  alternatives rejected.
- Check control flow for hidden side effects: a getter that mutates, a
  predicate that logs, an `impl Display` that computes.
- Check that similar cases read similarly. Two branches doing the same thing in
  two shapes cost the reader twice.
- Suggest restructuring only when it makes the rule clearer, and show the
  clearer version rather than describing it.

Verify by reading each proposed rewrite back in place. If the rewrite is not
plainly easier to follow, drop the finding.

## Concurrency

Review shared-state ownership, synchronization, atomic ordering, lock scope and
ordering, blocking work in async code, cancellation, shutdown, and task
lifetime.

Read every producer and consumer of the shared state, not only the changed side.

- Enumerate the shared state the change touches and name its protection for
  each field: a lock, an atomic, a channel, or single ownership. Unprotected
  shared mutability is a finding.
- Check each critical section for the invariant it maintains, and for a
  check-then-act split across two acquisitions.
- Check lock scope: work, allocation, I/O, logging, or a callback held under a
  lock, and a guard living to the end of a function that needed it for one line.
- Establish a lock order for every path that takes two locks and confirm all
  paths agree. Check for re-entrant acquisition through a callback or a nested
  helper.
- Check each atomic's ordering against what it synchronizes. `Relaxed` on a flag
  that publishes data is a finding; state the required `Acquire`/`Release` pair.
- In async code, find blocking calls: file and network I/O, `std::sync::Mutex`
  held across `await`, `std::thread::sleep`, and CPU-bound loops on the runtime.
- Check cancellation at every `await`: what state is left behind if the future
  is dropped there, and whether a partially applied mutation survives.
- Check task and thread lifetime: detached work, joins that never happen,
  channels whose sender or receiver drop unnoticed, and shutdown that leaves a
  task running.
- Check `Send`, `Sync`, and `'static` assumptions the types do not enforce, and
  any manual `unsafe impl` of them.
- Look for missed wakeups: a condition variable without a loop, a notification
  sent before the waiter registers, a `Waker` not stored.

Verify with a test that exercises both orders, a stress loop, or a run under
`RUSTFLAGS="-Z sanitizer=thread"` or `loom` where available. Say which you ran.

## Security

Identify the change's trust boundaries before judging it, then review validation
and normalization of untrusted input, authorization, injection and path
handling, resource limits, sensitive logging, secret storage, and disclosure.

Read where each input enters the process and what it is trusted to be.

- Name every trust boundary the change crosses: command-line arguments, source
  files, imported modules, environment variables, file contents, network data,
  subprocess output. State what the code assumes about each.
- Trace each untrusted value to its use. Check validation happens before use,
  once, on the value actually used rather than a copy checked earlier.
- Check every path built from input for traversal, symlink following, absolute
  path injection, and escape from an intended root. Check canonicalization
  happens before the check, not after.
- Check every command, query, template, or format string built from input for
  injection, and prefer an argument vector over a shell string.
- Check resource limits: unbounded recursion, unbounded read into memory,
  unbounded allocation from an attacker-chosen size, decompression ratios, and
  loops driven by an input-controlled count.
- Check arithmetic on sizes, lengths, offsets, and capacities for overflow that
  turns into an out-of-bounds or under-allocation.
- Check what is logged, printed in diagnostics, embedded in artifacts, or left
  in temporary files: secrets, tokens, absolute paths, environment contents.
- Check secret handling: hardcoded values, storage in a `String` that lingers,
  comparison without constant time where it matters, and secrets in process
  arguments.
- Check file creation for permissions, predictable temporary names, and
  time-of-check-to-time-of-use races.
- Tie each finding to a realistic threat and a concrete attacker input. Drop the
  checklist items that do not apply to this program's threat model.

Verify by constructing the malicious input and running it. Report what happened.

## Correctness

Review stated and implicit invariants, boundary conditions, arithmetic and
conversion behavior, state transitions, ordering, cleanup, and failure
atomicity.

Read the contract that owns the behavior first: the specification section, the
architecture document, or the type's own documentation.

- State the rule the change is supposed to implement, quoting the owning
  document, then check the code against it clause by clause. Contract drift is
  the finding this topic exists to catch.
- Trace at least one success path, one boundary path, and one failure path with
  concrete values. Write the values down.
- Check every comparison for its boundary: `<` versus `<=`, inclusive versus
  exclusive ranges, the empty case, the single-element case, and the last
  iteration.
- Check every arithmetic operation for overflow, underflow, division by zero,
  and truncation, in both debug and release semantics. Check every `as` cast for
  a value it silently changes, and prefer `try_into`, `checked_`, `saturating_`,
  or `wrapping_` chosen deliberately.
- Check every `match` for arms that fall to a catch-all which should be an
  error, and for a `_ =>` that will silently absorb a future variant.
- Check state machines for reachable illegal transitions, states that never
  clear, and two fields that must agree but are set separately.
- Check ordering assumptions: iteration order of a map, evaluation order,
  short-circuit dependence, and sequence dependence between two calls.
- Check `Option` and `Result` handling for a `None` that means something
  different from the default substituted for it.
- Check `PartialEq`, `Ord`, `Hash`, and comparison logic for consistency, and
  any float comparison for `NaN` and precision.
- Check cleanup and failure atomicity: an aborted operation must not leave
  partial state, and the same failure must not be handled twice.
- Check the diff for behavior it changes without meaning to: a moved
  early-return, a widened condition, a removed check that another path relied
  on.

Verify by running the case you traced as a real test, and by re-reading the
owning document rather than trusting your memory of it.

## Unsafe

Review every relevant `unsafe` block and FFI boundary for a documented,
maintained safety invariant.

Read the type's full module, since soundness is a property of the whole
abstraction rather than one block.

- For each `unsafe` block, write out the invariant it requires and the reason it
  holds here. An undocumented `unsafe` block is a finding regardless of whether
  it is currently sound.
- Check that safe code cannot violate the invariant. If any safe public path can
  reach an unsound state, the abstraction is broken, not the block.
- Check pointer validity at each use: non-null, aligned, dereferenceable for the
  full size, and derived from a live allocation with a provenance the code
  actually has.
- Check aliasing: no `&mut` coexisting with another reference to the same data,
  no reference derived from a raw pointer outliving the data, and no
  `&mut` created from a shared pointer.
- Check initialization: `MaybeUninit` read only after full initialization,
  `assume_init` justified, padding not read, and no reference to uninitialized
  memory even transiently.
- Check layout assumptions: `repr` attributes present where the code depends on
  them, `size_of` and `align_of` assumptions verified, no transmute between
  types whose layout is not guaranteed, and enum discriminants valid.
- Check every FFI signature against the foreign declaration: parameter types,
  return type, calling convention, nullability, ownership transfer, who frees,
  string encoding and NUL termination, and errno or error-out conventions.
- Check unwinding: no Rust panic crossing an `extern "C"` boundary, and no
  foreign call that can unwind into Rust.
- Check thread safety of any manual `Send` or `Sync` implementation against what
  the type actually permits.
- Check for undefined behavior the compiler is free to exploit: out-of-bounds
  arithmetic on pointers, invalid enum or `bool` values, misaligned reads,
  aliasing violations, and data races.

Verify under Miri where the code can run there, plus a sanitizer build and a
test that exercises the boundary. State what you were unable to check.

## Architecture

Review whether responsibilities sit at clear boundaries and each behavior has
one implementation path.

Read `eng/architecture.md` when it exists, and locate the change against the
codebase map before judging its placement.

- Name the fact or rule the change implements and the module that owns it.
  Logic placed outside its owner, or split across two owners, is a finding.
- Search for a second implementation of the same rule. Under this repository's
  rules, a fast path, legacy path, fallback path, or representation selector for
  one language rule is a correctness bug. Name both paths and say which one
  should survive.
- Look for duplicated logic that should become one small helper, including cases
  where only two callers exist today. Prefer the abstraction that names the
  shared contract.
- Check coupling: a module reaching into another's internals, a phase depending
  on a later phase, a type known in more layers than it should be, and
  bidirectional dependencies.
- Look for hidden global state, statics, thread-locals, environment reads, and
  caches that make behavior depend on history.
- Check every new trait, generic parameter, mode flag, option, and layer for a
  present caller. Speculative flexibility is a finding here.
- Check whether a removed special case left behind structure that is now
  redundant: newly identical branches, a one-variant enum, a wrapper that only
  forwards, a helper with one caller that no longer earns its name.
- Check that data representations are singular: one AST shape, one storage
  lifetime story, one identity scheme, unless the owning document states why two
  exist.
- Check that the change respects phase separation and the storage and identity
  rules the architecture document sets, and that any new boundary is recorded
  there.
- Check facts are not restated across documents or across modules. One home per
  fact.

Verify by grepping for the rule's other implementations and by reading the
owning document rather than inferring the intended structure.

## Dependencies

Review whether each added or changed crate is necessary and narrowly
configured.

Read the `Cargo.toml` diff, the lockfile diff, and what the crate is actually
used for.

- For each added dependency, name the code that uses it and how much of it. A
  crate pulled in for one function that the standard library or twenty lines
  already covers is a finding.
- Check `default-features = false` and the exact feature list. Look for features
  enabled transitively that the crate does not need, and features that pull in a
  runtime, a TLS stack, or a proc-macro chain.
- Check the lockfile diff for the size of the transitive addition, duplicated
  versions of the same crate, and any crate you would not want in this build.
- Check the version requirement against the repository's policy, and whether it
  permits a range the project has not tested.
- Check maintenance signals: last release, open advisories, maintainer count,
  whether the crate is a thin wrapper, and whether it contains `unsafe` or a
  build script that runs arbitrary code.
- Check build scripts, proc macros, and code generation for what they execute at
  build time and what they require to be installed.
- Check target and toolchain support: platforms the project builds for,
  minimum supported Rust version, and any `cfg` the dependency forces callers to
  write.
- Check whether the dependency's types appear in this crate's public API, which
  makes its version part of this crate's contract.
- Check dev-dependencies and build-dependencies are not in the normal
  dependency section, and that optional dependencies are gated as intended.

Verify with `cargo tree`, `cargo tree --duplicates`, `cargo audit` where
available, and a clean build.

## Documentation

Review public API documentation and explanations of non-obvious invariants.

Read the doc comments in the diff, the items they describe, and the repository
documents that own the same facts.

- Check every public item for a doc comment that states its contract: what it
  does, what the caller must guarantee, and what the return value means.
- Check that `# Errors`, `# Panics`, `# Safety`, and platform behavior are
  documented wherever they apply. A public fallible function whose failure cases
  are undocumented is a finding.
- Check non-obvious invariants are recorded where a maintainer will see them:
  on the type or field that must uphold them, not in a distant module.
- Check examples compile, run, and show the intended use. Verify with
  `cargo test --doc`.
- Check documentation against the code it describes. A comment that was true
  before the diff and is now wrong is worse than no comment.
- Check that a fact documented here is not also owned by `docs/spec.md`,
  `eng/architecture.md`, or `eng/roadmap.md`. Reference the owner rather than
  restating it, and confirm the owning document actually says what the reference
  claims.
- Check comments explain constraints and reasons rather than narrating the code,
  and remove narration the diff added.
- Check that no comment or document records a correction, a migration, a
  superseded design, or history. Under this repository's rules, the text reads
  as if the right way were known from the start.
- Check spelling, terminology, and the project's vocabulary. The same concept
  gets the same word in code and prose.

Verify by reading each doc comment as a caller who has not seen the
implementation, and by running `cargo doc` and `cargo test --doc`.
