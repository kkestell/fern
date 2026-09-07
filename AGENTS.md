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
| `eng/roadmap.md`      | Build order, current scope, and completion gates. Reference behavior contracts rather than defining them here.                                                                         |
| `eng/plans/`          | Bounded implementation steps and validation for a slice, produced by `kplan` and executed by `kwork`.                                                                                  |
| `AGENTS.md`           | Repository workflow, document ownership, and instructions for agents.                                                                                                                  |

Use `kspec` when creating or modifying `docs/spec.md`, and `kroadmap` when
creating or modifying `eng/roadmap.md`. Reading either does not require its
skill. Generic skill wording about "product behavior" does not expand the
language specification's scope.

The specification and roadmap are created when the project needs them. `kplan`
requires both and will direct the user to the appropriate skill when either is
missing. The architecture document is not required to begin planning.

## Codebase Map

- `src/source.rs`, `src/diagnostic.rs` — source loading and diagnostics.
- `src/frontend.rs`, `src/semantic.rs` — parsing and semantic checking.
- `src/ir.rs`, `src/backend.rs` — verified Fern IR and native compilation.
- `src/lib.rs`, `src/main.rs` — compiler library and command-line driver.
- `tests/` — compiler and CLI integration tests.
- `examples/` — runnable Fern programs.

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

This governs replies. Files you write follow the repository's documentation
rules.

### Project priorities

This project is optimized for clarity, correctness, and ease of reasoning rather
than execution speed. Treat simplicity as a maintained project invariant, not a
cleanup activity.

Read the owning document before making a behavioral decision. For language
changes, use `kspec` to resolve missing rules in `docs/spec.md` before
implementation depends on them.

When `eng/architecture.md` exists, read it before changing the program's
structure. Do not invent architectural constraints when it does not exist.

Build software in narrow, end-to-end vertical slices. A feature is not
implemented until every part of the codebase it touches, and their tests, agree
on it. Do not reserve names, add extension points, or build infrastructure for
hypothetical future features.

Use `kplan` to plan substantial work and `kwork` to execute an implementation
plan. Both `docs/spec.md` and `eng/roadmap.md` must exist before planning
begins.

Preserve unrelated working-tree changes. Never commit unless the user asks for a
commit explicitly.

### One home for every fact

Every fact lives in exactly one place, as assigned under Project Documents.
Reference a fact owned by another document instead of restating or summarizing
it. When moving a contract, remove the old copy and update its references.
