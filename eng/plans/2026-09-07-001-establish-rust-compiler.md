# Establish the Rust Compiler

## Sources

- [Roadmap: Establish the Rust compiler](../roadmap.md#establish-the-rust-compiler)
  — slice scope, dependency choices, and completion gates.
- [Specification: Lexical Structure](../../docs/spec.md#lexical-structure) —
  source encoding, token boundaries, comments, and identifiers.
- [Specification: Functions](../../docs/spec.md#functions) — entry-point syntax
  and normal completion.

## Goal

Create the first working compiler path, from an empty Fern `main` to a native
executable. The repository currently contains only project documents; this is
one standalone bootstrap slice.

## Implementation

- `Cargo.toml`, `Cargo.lock`, `.gitignore` — create one Rust package with a
  `fern` binary and a library for compiler tests. Add the roadmap dependencies
  and ignore build outputs.
- `src/source.rs`, `src/diagnostic.rs` — own the input path and validated source
  text; represent source locations as byte ranges. Render source diagnostics
  through Ariadne. Report file and tool errors without fabricating source spans;
  invalid UTF-8 must report the input path and offending byte offset without
  passing invalid text to the renderer.
- `src/frontend.rs` — use Logos and a handwritten parser to accept empty
  parameterless function declarations with `void` return annotations. Intern
  names with lasso and store declarations in la-arena with source spans. Support
  the whitespace and nested comments needed around these tokens. Consume the
  whole file and diagnose unexpected tokens, nonempty bodies, and incomplete
  declarations. Preserve enough declaration structure for entry-point checking.
- `src/semantic.rs` — validate the parsed declarations and produce a checked
  entry point. Reject missing or duplicate `main` and other functions before
  lowering; do not accept an arbitrary function merely because its body is
  empty.
- `src/ir.rs` — lower the checked entry point to a small owned representation of
  an entry function completing with status zero. Verify that representation
  before emission. Keep it independent of source storage; add only the nodes and
  checks that this executable path uses.
- `src/backend.rs` — emit QBE for the verified entry function, invoke QBE, and
  assemble/link through the host C toolchain. Capture tool failures and stderr.
  Use temporary intermediate files and publish the requested executable only
  after every backend step succeeds; clean up intermediates on failure.
- `src/lib.rs`, `src/main.rs` — connect the phases and expose
  `fern <input.fern> -o <output>`. Return a nonzero compiler status on failure.
  Reject malformed arguments and an output path that would overwrite the input.
- `examples/empty.fern`, `README.md`, `AGENTS.md` — add the runnable example,
  document the CLI and native tool prerequisites, and update the codebase map
  and actual development commands.

## Tests

- Compile and execute the empty example; assert status zero. Exercise ordinary
  whitespace and nested comments through the same path.
- Check source byte spans for invalid tokens, incomplete declarations, and
  unexpected trailing input, including a Unicode comment before the error.
- Cover missing input, invalid UTF-8, invalid CLI arguments, empty source,
  duplicate entry points, additional functions, and unsupported body syntax.
- Use Insta for a deterministic empty-function syntax snapshot, excluding
  machine-specific paths and interner allocation details.
- Exercise malformed internal IR and prove verification blocks emission.
- Test missing and failing backend tools and an unwritable output destination. A
  failed compilation must not publish a new executable or overwrite an existing
  output. Native success tests require real QBE and C tools; document
  prerequisites rather than silently skipping the completion gate.

## Decisions

- Target the development host using QBE and the host C toolchain. Record the
  tested host and tool setup in the README; cross-compilation is not part of
  this slice.
- Use separate Rust modules within one crate for the phase boundaries. No
  workspace split or general control-flow representation is needed for this
  entry function.
