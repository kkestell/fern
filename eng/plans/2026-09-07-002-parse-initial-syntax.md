# Parse the Initial Syntax

## Sources

- [Roadmap: Parse the initial syntax](../roadmap.md#parse-the-initial-syntax) —
  scope and completion gates.
- [Specification: Lexical Structure](../../docs/spec.md#lexical-structure) —
  token boundaries, reserved names, comments, literals, and trailing commas.
- [Specification: Functions](../../docs/spec.md#functions),
  [Variable declarations](../../docs/spec.md#variable-declarations), and
  [Process exit](../../docs/spec.md#process-exit) — supported declaration and
  statement forms.
- [Specification: Flexible Types](../../docs/spec.md#flexible-types) — literal
  spelling must survive parsing for subsequent type and range checks.

## Goal

Extend the frontend to represent the roadmap's initial syntax. Preserve the
working empty-program compilation path while rejecting newly parsed bodies at
the semantic boundary until checking and lowering support them.

## Implementation

- `src/frontend.rs` — extend Logos tokens for declarations, `int`, `exit`,
  integer literals, and required punctuation. Centralize the existing reserved
  name check so function names, binding names, and binding references agree.
  Preserve keyword-prefix identifiers such as `exit_code`.
- `src/frontend.rs` — recognize integer spellings without converting them to a
  machine integer. Validate base digits and suffix spelling, retaining valid
  suffixes for subsequent semantic rejection when outside the subset. Diagnose
  malformed numeric candidates as a whole rather than accepting a valid prefix
  followed by a name. Keep very large but lexically valid integers parseable.
- `src/frontend.rs` — introduce arena-backed statements and expressions linked
  by typed arena indices. Give each function an ordered body; represent binding
  declarations with mutability, interned name, optional `int` annotation, and
  initializer, and exit statements with their argument. Expressions need only
  integer literals and interned binding references. Retain node spans, name and
  annotation spans, and owned literal spellings.
- `src/frontend.rs` — refactor the handwritten parser to consume a body through
  its closing brace, with enough lookahead for declarations and the optional
  trailing call comma. Keep first-error diagnostics and byte-based EOF spans.
  Consume the whole input and continue parsing statements after an exit.
- `src/semantic.rs` — explicitly reject a nonempty body before producing
  `CheckedEntry`. Otherwise the existing lowering would silently discard parsed
  statements and produce a successful executable. Keep entry-point checks in
  their current phase.
- `src/frontend.rs` tests — extend the deterministic snapshot projection to show
  ordered bodies, resolved names, literal spellings, and spans without exposing
  interner internals. Keep frontend success tests independent of native
  compilation.
- `tests/compiler.rs` — update expectations affected by newly recognized tokens
  and the semantic body rejection. Preserve the existing output-preservation
  assertions for unsupported programs.

## Tests

- Parse inferred and annotated declarations, both binding kinds, reference
  initializers, literal and reference exit arguments, and trailing call commas.
  Include several functions and statements after exit to prove parsing does not
  perform entry-point, name-resolution, or reachability checks.
- Cover every integer base, leading zeroes, hexadecimal letter digits, optional
  `i`, other specified integer suffixes, and a magnitude beyond machine range.
- Reject missing base digits, invalid base digits, digit separators, unknown
  suffixes, and the invalid suffixes `int` and `uint`. Check exact source spans.
- Cover missing names, initializers, annotations, delimiters, and semicolons;
  missing or extra exit arguments; reserved names; and unsupported assignment,
  nested blocks, and operators.
- Exercise all specified whitespace, nested and unterminated comments, and
  Unicode comments preceding errors. Snapshot a representative mixed body and
  retain the empty-function snapshot.
- Prove a syntactically valid nonempty program fails compilation without
  creating or replacing an executable; retain native empty-main success
  coverage.
