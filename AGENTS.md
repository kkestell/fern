# AGENTS.md

THIS FILE MUST BE KEPT UP TO DATE AT ALL TIMES

Fern is a low-level programming language in the spirit of C, without C's legacy
baggage, undefined behavior, or unsafe default semantics.

## Project Documents

Choose a document by the fact it owns, not by whether the change is broadly
described as "behavior" or a "specification."

| Document              | Owns                                                                                                                                                                                   |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docs/spec.md`        | The Fern language: source syntax, types, semantic rules, built-ins, and program execution.                                                                                             |
| `eng/architecture.md` | Durable implementation design and boundaries: phase separation, storage lifetimes, node identity, diagnostics, inspection, and verification. Internal AST representation belongs here. |
| `eng/roadmap.md`      | Milestones, their tasks, scope, order, and completion gates. Reference behavior contracts rather than defining them here.                                                              |
| `eng/plans/`          | Implementation plans for individual roadmap tasks: concrete file changes, focused tests, and decisions the owning documents leave open.                                                |
| `AGENTS.md`           | Repository workflow, document ownership, and instructions for agents.                                                                                                                  |

Use `kspec` when creating or modifying `docs/spec.md`. Use `kroadmap` to
add or revise roadmap milestones and tasks; `kwork` only checks off completed
tasks in `eng/roadmap.md`. Reading either document does not require its skill.
Generic skill wording about "product behavior" does not expand the language
specification's scope.

Plans in `eng/plans/` are produced by `kplan` and executed by `kwork`.

The specification and roadmap are created when the project needs them. `kplan`
requires both and will direct the user to the appropriate skill when either is
missing. The architecture document is not required to begin planning.

## Codebase Map

- `src/source.rs`, `src/diagnostic.rs` — source loading and diagnostics.
- `src/types.rs` — the Fern type enum and its spellings, shared by every phase.
- `src/frontend.rs`, `src/semantic.rs` — parsing and semantic checking.
- `src/ir.rs`, `src/backend.rs` — Fern IR and native compilation.
- `src/snapshots/` — checked-in frontend and IR snapshot fixtures.
- `src/lib.rs`, `src/main.rs` — compiler library and command-line driver.
- `tests/` — compiler and CLI integration tests.
- `examples/` — Fern programs demonstrating supported syntax and semantics;
  build and run instructions are documented in `README.md`.

## Development Commands

- `cargo fmt --check` — check Rust formatting.
- `cargo clippy --all-targets -- -D warnings` — lint the compiler and tests.
- `cargo test` — run unit, snapshot, and native integration tests.
- `cargo build` — build the compiler.
- `cargo run -- examples/empty.fern -o target/empty && target/empty` — compile
  and execute the empty example.

Native tests require the tools documented in [README.md](README.md).

## Project Rules

### Answering

Be short. Say the thing and stop.

A few sentences is the normal length of a reply. Most questions need one or two.
Never write five paragraphs where one would do. If a reply is running long, cut
whole points rather than compressing them into denser sentences.

Write plainly. Ordinary words, ordinary sentences, one idea each. Say it the way
you would say it out loud to someone sitting next to you. No throat-clearing
before the answer, no summary of what you just did after it, no restating the
question, no listing the options you considered and rejected.

Do not be clever or cryptic. Do not stack clauses onto a sentence with dashes
and semicolons; start a new sentence. Do not invent names for things that
already have names. Prefer the concrete: name the file, the function, the value.

When you need a decision, ask one plain question.

When asking about a language-design decision, briefly explain what Odin, Hare,
and C do for that same decision. Distinguish specified behavior from
implementation-specific behavior and verify uncertain details in primary sources.

This governs replies. Files you write follow the repository's documentation
rules.

### Project priorities

This project is optimized for clarity, correctness, and ease of reasoning rather
than execution speed. Treat simplicity as a maintained project invariant, not a
cleanup activity.

Keep one implementation path for each behavior. Do not add a fast path, legacy
path, fallback path, or representation selector when the same pipeline can
handle every case. Parallel lowering, checking, verification, or emission paths
for the same language rule are a correctness bug unless their semantics or
ownership genuinely differ. When they do differ, make the boundary explicit in
the owning document.

Treat duplicated logic as shared behavior waiting to drift. Extract a small
helper or abstraction when it gives one home to a real rule, even when two
callers are the only current users. Prefer an abstraction that names the shared
contract over two locally simple copies. Do not use this as permission for
speculative frameworks, extension points, or layers without a present caller.

When removing a special case, enum variant, or separate code path, inspect the
surrounding code for structure it made redundant. Collapse newly identical paths
and use existing helpers in the same change.

Before completing a milestone, search for mode flags, optional representations,
path-selection predicates, duplicated phase logic, and stale suppressions added
during its tasks. Test combinations of the milestone's features, not only each
feature in isolation. Include cases where one new expression or statement is
nested inside another.

Describe Fern and roadmap work in terms of the intended end state. Do not record
superseded syntax, migration steps, compatibility behavior, or design history.
Fern has no source-compatibility constraints until real users and programs create
them.

Read the owning document before making a behavioral decision. For language
changes, use `kspec` to resolve missing rules in `docs/spec.md` before
implementation depends on them.

When `eng/architecture.md` exists, read it before changing the program's
structure. Do not invent architectural constraints when it does not exist.

Finish one roadmap task at a time. A task may leave the feature partially
implemented across compiler phases. Preserve existing supported behavior and
state unfinished integration clearly. Add temporary guards only when needed to
prevent incorrect execution, not to make each task a standalone deliverable.

Use focused checks during implementation. Run broad validation when the feature
or milestone is integrated, or earlier when a concrete risk warrants it. Review
the completed milestone once for correctness and architectural simplicity. A
feature is complete only when its affected phases and tests agree.

Do not reserve names, add extension points, or build infrastructure for
hypothetical future features.

Use `kplan` to write an implementation plan for one roadmap task, then `kwork`
to execute that plan and validate it. Both `docs/spec.md` and `eng/roadmap.md`
must exist before planning begins.

The roadmap is a list of named milestones in implementation order, with task
checkboxes recording progress. Preserve completed tasks and milestone details.

Each milestone must have a complete Fern example in the roadmap when its scope
is planned. Demonstrate its new capabilities, include the expected result, and
name its future `examples/` file. When the milestone is implemented, create
that file. `kroadmap` maintains the roadmap's example snippets and links. For
milestones awaiting language decisions, write the example when those decisions
are settled.

Preserve unrelated working-tree changes. Never commit unless the user asks for a
commit explicitly.

### One home for every fact

Every fact lives in exactly one place, as assigned under Project Documents.
Reference a fact owned by another document instead of restating or summarizing
it. When moving a contract, remove the old copy and update its references.
