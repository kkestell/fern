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
| `eng/todo.md`         | Ordered implementation tasks and their completion state. A task may have subtasks; mark the task complete when all its subtasks are complete.                                           |
| `eng/plans/`          | Implementation plans for a single TODO task or a defect repair: concrete file changes, focused tests, and decisions the owning documents leave open.                                    |
| `eng/reviews/`        | Code-review reports: scope, findings, unresolved suspicions, and checks run.                                                                                                           |
| `AGENTS.md`           | Repository workflow, document ownership, and instructions for agents.                                                                                                                  |

Use `kspec` when creating or modifying `docs/spec.md`. `kplan` plans either the
next incomplete TODO task or a confirmed defect; `eng/todo.md` tracks feature
scope, so defect repair needs no entry there. `kwork` only checks off completed
tasks in `eng/todo.md`, marking a parent task when all of its subtasks are
complete. Reading any of these documents does not require its skill. Generic
skill wording about "product behavior" does not expand the language
specification's scope.

Document ownership governs decisions, not obvious mistakes. When any document
is plainly wrong in a way nobody chose — an example that does not compile, a
stale name, a typo — correct it where you find it and say so. Reach for the
owning skill when the fix changes what the project intends.

Plans in `eng/plans/` are produced by `kplan` and executed by `kwork`.

The specification and TODO are created when the project needs them. `kplan`
requires both and will direct the user to `kspec` or to add `eng/todo.md` when
either is missing. The architecture document is not required to begin planning.

## Codebase Map

- `src/source.rs`, `src/diagnostic.rs` — source loading and diagnostics.
- `src/types.rs` — the Fern type and operator enums, their spellings, and the
  representation of a concrete floating-point value, shared by every phase.
- `src/frontend/`, `src/semantic/` — parsing and semantic implementations,
  tests, and snapshots.
- `src/module.rs` — module discovery, import resolution, and the dependency
  graph.
- `src/ir/`, `src/backend/` — Fern IR and native compilation implementations.
- `eng/architecture.md` — durable compiler phase, storage, identity,
  diagnostic, and verification boundaries.
- `eng/reviews/` — checked-in code-review reports.
- `src/frontend/tests/snapshots/`, `src/ir/tests/snapshots/` — checked-in
  frontend and IR snapshot fixtures.
- `src/lib.rs`, `src/main.rs` — compiler library and command-line driver.
- `tests/` — compiler and CLI integration tests; reusable program fixtures live
  under `tests/fixtures/programs/`.
- `examples/` — an ordered, topic-based tour of supported Fern syntax and
  semantics; build and run instructions are documented in `README.md`.

## Development Commands

- `cargo fmt --check` — check Rust formatting.
- `cargo clippy --all-targets -- -D warnings` — lint the compiler and tests.
- `cargo test` — run unit, snapshot, and native integration tests.
- `cargo build` — build the compiler.
- `cargo run -- examples/literals.fern -o target/literals && target/literals` —
  compile and execute the first example.

Native tests require the tools documented in [README.md](README.md).

## Project Rules

### Answering

Be short. Say the thing and stop.

When you need a decision, ask one plain question. Write it as prose in your
reply. Never ask through a multiple-choice or option-picker tool.

When asking about a language-design decision, first say what Go, Odin, and Hare
do for that same decision, then recommend one answer and say why in a sentence.
Verify each language's behavior in its primary sources before describing it, and
say so plainly when a source does not settle it. Distinguish specified behavior
from implementation-specific behavior.

This governs replies. Files you write follow the repository's documentation
rules.

### Implementation principles

This project is optimized for clarity, correctness, and ease of reasoning.

### Design decisions

Describe Fern work in terms of the intended end state. Do not record superseded
syntax, migration steps, compatibility behavior, or design history. Fern has no
source-compatibility constraints until real users and programs create them.

Read the owning document before making a behavioral decision. For language
changes, use `kspec` to resolve missing rules in `docs/spec.md` before
implementation depends on them.

Read `eng/architecture.md` before changing the program's structure.

### One home for every fact

Every fact lives in exactly one place, as assigned under Project Documents.
Reference a fact owned by another document instead of restating or summarizing
it. When moving a contract, remove the old copy and update its references.
