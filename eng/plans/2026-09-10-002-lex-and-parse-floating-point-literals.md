# Lex and parse floating-point literals

## Sources

- `docs/spec.md#floating-point-literals` — accepted spellings and invalid
  forms
- `docs/spec.md#keywords` — `f32` and `f64` are reserved
- `docs/spec.md#floating-point-constant-expressions` — literals remain
  untyped until the semantic work
- `eng/roadmap.md#floating-point-numbers` — first task and its parser boundary
- `eng/architecture.md#pipeline` — frontend syntax precedes semantic checking
- `src/frontend/lexer.rs`, `src/frontend/parser.rs`, `src/frontend/syntax.rs`
  — existing literal pipeline and syntax traversal

## Goal

Accept every specified floating-point literal and retain its source spelling in
parsed syntax. This is a frontend-only slice; floating-point types, conversions,
constant evaluation, and runtime behavior remain unimplemented and are rejected
by the semantic phase.

## Implementation

- `src/frontend/lexer.rs` — lex both numeric literals as one broad `Number`
  candidate, so a malformed decimal point, exponent, or suffix stays a single
  token instead of becoming several valid ones. A logos callback extends an
  integer-shaped match over a decimal point and an exponent sign, which its
  regex cannot reach; a point directly before another point is left alone so
  the array fill marker in `[0...]` still lexes as an integer and an ellipsis.
  `is_floating` says which literal a candidate spells, from a decimal point or
  an exponent marker following the decimal digit run, and `valid_number`
  dispatches to the existing integer validation or to a new floating one that
  accepts only the specified spellings.

- `src/frontend/syntax.rs` — add `ExpressionKind::Floating(String)` alongside
  `Integer`, retaining the original spelling for the later exact-constant
  implementation. Treat it as a leaf in `walk_expression`.

- `src/frontend/parser.rs` — validate a numeric candidate in `advance`, build
  a floating candidate as the new expression variant, and reserve `f32` and
  `f64` in every name position. Do not add them to `Scalar` or accept them as
  conversion heads yet: those changes belong to the following type and
  conversion task.

- `src/frontend/tests/mod.rs` — project a floating expression distinctly in
  parser snapshots.

- `src/semantic/expressions.rs` — reject a floating syntax expression at the
  frontend-to-semantic boundary with the same temporary-feature guard pattern
  used for syntax that awaits its later milestone task. Do not add a floating
  constant or type representation in this slice.

## Tests

- `src/frontend/tests/lexer.rs` — prove that `1.0`, `.5`, `2.`, `1e3`, and
  `6.02e-23` retain their complete spelling, while malformed exponents,
  suffixes, digit separators, and hexadecimal forms are rejected with the
  candidate span. Cover the candidate classification directly, so an
  otherwise-integer spelling and a hexadecimal `e` digit stay integers, and
  cover an integer and a floating fill element beside the `...` marker.

- `src/frontend/tests/parser.rs` and its snapshot — cover the valid literal
  forms in bindings and ordinary unary, binary, grouping, and comparison
  contexts, proving that the existing precedence parser treats them as primary
  expressions. Add `f32` and `f64` to the reserved-name coverage.

- `src/semantic/tests/expressions.rs` — prove a parsed floating literal reaches
  the temporary semantic diagnostic rather than an integer parsing path.
