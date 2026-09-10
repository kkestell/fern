---
name: kreview
description: "Review a Rust code change through one focused quality lens in depth, or sweep every lens at survey depth. Use when the user asks for a Rust code review of a diff, branch, commit, or set of files."
argument-hint: "[general|ownership|error-handling|api-design|performance|testing|readability|concurrency|security|correctness|unsafe|architecture|dependencies|documentation] [review scope]"
---

## Workflow

Review the change directly. Do not edit files or delegate the review.

1. Resolve the topic and review scope from
   `<input_document> $ARGUMENTS </input_document>`. The first argument may be
   one topic from the list below. Default to `general` when none is given.
2. Read the repository instructions, then inspect the diff and changed-file
   list. Ask one focused question only when the review scope remains ambiguous.
3. Read the chosen topic's section in
   [references/topics.md](references/topics.md), then run the mode below.
4. Review intentional `clone`, `unwrap`, `unsafe`, allocation, and dependency
   choices in context rather than treating them as automatic defects.
5. Report actionable findings ordered by severity. Give each finding a path,
   line, consequence, and suggested fix. Name the mode and topics reviewed. If
   there are no findings, say so plainly and mention any material validation
   gap.

### General mode

A survey across every topic. Read the `General` section of
`references/topics.md`, then work through all thirteen topics in the order
listed, spending a bounded pass on each.

- Cover every topic, including ones the change does not obviously touch. A
  topic with nothing to report is a normal outcome; say so in one line.
- Judge each topic from the diff plus the immediate surrounding code. Do not
  open the wider call graph, trace whole execution paths, or read a topic's
  detailed checklist.
- Report a finding when the diff plainly shows the problem. Note a suspicion
  worth a deep pass as a follow-up rather than investigating it now.
- Close with a one-line verdict per topic and a recommendation of which topics
  deserve their own deep review.

### Topic mode

An exhaustive review of one lens. Read that topic's section in
`references/topics.md` and apply every check it lists.

- Examine every changed line the topic touches, plus the code it calls, the
  code that calls it, and the invariants it depends on. Follow the call graph
  out of the diff until the topic's questions are answered.
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
