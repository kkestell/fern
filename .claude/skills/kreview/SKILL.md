---
name: kreview
description: "Review a Rust code change through one focused quality lens in depth, or sweep every lens at survey depth. Use when the user asks for a Rust code review of a diff, branch, commit, or set of files."
argument-hint: "[general|ownership|error-handling|api-design|performance|testing|readability|concurrency|security|correctness|unsafe|architecture|dependencies|documentation] [review scope]"
---

## Workflow

Review the requested corpus directly. Do not edit implementation files or
delegate the review. Write its report under `eng/reviews/`.

1. Resolve the topic and review scope from
   `<input_document> $ARGUMENTS </input_document>`. The first argument may be
   one topic from the list below. Default to `general` when none is given.
2. Read the repository instructions, then identify the review corpus.
   - For a diff, branch, or commit, inspect its diff and changed-file list.
   - For explicit files, directories, or the whole codebase, inventory that
     scope. Do not reduce it to the current diff.
   Ask one focused question only when the review scope remains ambiguous.
3. Read the chosen topic's section in
   [references/topics.md](references/topics.md), then run the mode below.
4. Review intentional `clone`, `unwrap`, `unsafe`, allocation, and dependency
   choices in context rather than treating them as automatic defects.
5. Write `eng/reviews/YYYY-MM-DD-NNN-slug.md`, using the next sequence for the
   day. Name the scope and mode, then record findings ordered by severity with
   path, line, consequence, and suggested fix. Record unresolved suspicions and
   checks run. When there are no findings, say so plainly and note any material
   validation gap.
6. Report the review concisely and link its document. Do not implement findings
   in the review turn. A later implementation commit includes this report, and
   includes its plan too when the finding becomes a planned task.

### General mode

A survey across every topic. Read the `General` section of
`references/topics.md`, then work through all thirteen topics in the order
listed, spending a bounded pass on each.

- Cover every topic, including ones the change does not obviously touch. A
  topic with nothing to report is a normal outcome; say so in one line.
- Judge each topic from the requested corpus plus the immediate surrounding
  code. Do not open the wider call graph, trace whole execution paths, or read
  a topic's detailed checklist.
- Report a finding when the diff plainly shows the problem. Note a suspicion
  worth a deep pass as a follow-up rather than investigating it now.
- Close with a one-line verdict per topic and a recommendation of which topics
  deserve their own deep review.

### Topic mode

An exhaustive review of one lens. Read that topic's section in
`references/topics.md` and apply every check it lists.

- For a change review, examine every changed line the topic touches, plus the
  code it calls, the code that calls it, and the invariants it depends on.
  For a whole-codebase or directory review, read every production file in the
  corpus and the relevant tests before following the topic's call paths.
  Do not present a pattern search or a pass over central types as an exhaustive
  whole-codebase review.
- Trace representative success, boundary, and failure paths concretely, naming
  the values that reach each branch.
- Verify each suspicion with a focused search, a test, or a build rather than
  reasoning alone. Prefer `cargo test`, `cargo clippy`, a targeted `grep`, or a
  small reproduction over speculation.
- Check the change against the contract that owns the behavior, and against
  the repository's own rules for that topic.
- Report confirmed findings, then suspicions you could not settle and what
  would settle them, then the checks you ran and what they showed.
- Depth is the point. Stay within the one topic and do not drift into others.

## Topics

- `general` — one survey pass over every topic below.
- `ownership` — ownership, borrowing, clones, and lifetimes.
- `error-handling` — `Result`, propagation, context, and panic policy.
- `api-design` — naming, visibility, signatures, and ergonomics.
- `performance` — algorithms, allocations, copies, and hot loops.
- `testing` — edge cases, failure paths, interactions, and assertions.
- `readability` — local reasoning, function length, nesting, and naming.
- `concurrency` — races, locks, atomics, async behavior, and cancellation.
- `security` — trust boundaries, validation, authorization, and secrets.
- `correctness` — invariants, boundaries, arithmetic, and state transitions.
- `unsafe` — unsafe Rust, FFI, layout, aliasing, and soundness.
- `architecture` — boundaries, coupling, duplication, and needless machinery.
- `dependencies` — crates, features, portability, and supply-chain exposure.
- `documentation` — public contracts and non-obvious invariants.
