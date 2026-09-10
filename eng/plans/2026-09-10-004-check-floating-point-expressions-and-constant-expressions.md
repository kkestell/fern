# Check floating-point expressions and constant expressions

## Sources

- `docs/spec.md#numeric-conversions` — checked numeric-conversion rules
- `docs/spec.md#operand-types` — numeric operand unification
- `docs/spec.md#comparison` — comparison typing and constant results
- `docs/spec.md#floating-point-constant-expressions` — exact untyped values,
  rounding, and constant failures
- `docs/spec.md#floating-point-operators` — allowed floating operators and
  runtime behavior
- `docs/spec.md#floating-point-comparison` — IEEE comparison behavior
- `eng/roadmap.md#floating-point-numbers` — third task and unfinished lowering
  boundary
- `eng/architecture.md#pipeline` — semantic checking precedes IR lowering
- `src/semantic/constants.rs`, `src/semantic/expressions.rs` — existing exact
  integer constants, contextual typing, and expression checking

## Goal

Check `f32` and `f64` expressions, comparisons, conversions, and constant
expressions using the specified exact and IEEE rounding rules. This completes
semantic support only; the compiler must still reject every checked floating
value before integer-only IR lowering.

## Implementation

- `Cargo.toml`, `Cargo.lock` — add the arbitrary-precision rational dependency
  needed to retain decimal literals and untyped floating-point expressions as
  exact finite mathematical values. Do not parse source literals through host
  `f32` or `f64`.

- `src/semantic/model.rs` — extend checked literal and `Constant` forms to
  distinguish exact untyped floating values from concrete `f32` and `f64`
  values. Store concrete values by their IEEE bit pattern so constants retain
  their type and signed zero for the later IR task; keep arrays recursively
  composed from the same one constant representation.

- `src/semantic/constants.rs` — parse a validated decimal spelling into an
  exact rational, and add the shared numeric constant operations used by
  contextual typing, arithmetic, comparison, and conversion. Round exact
  rationals directly to binary32 or binary64 with round-to-nearest,
  ties-to-even, rejecting a non-finite rounded result. Evaluate concrete float
  operations at their operand format and reject non-finite constant results;
  retain their IEEE bit result otherwise. Check integer-to-float exactness,
  float-to-integer integrality and range, and float narrowing exactness for
  checked constant conversions. Make recursive constant comparison use the
  checked type so floating equality treats signed zero as equal rather than
  comparing raw bits.

- `src/semantic/expressions.rs` — remove the temporary floating-literal,
  conversion, and contextual-typing guards. Generalize the existing
  integer-only binary helper into the single numeric operand-unification path
  used by ordinary and compound assignments: it must let untyped integers and
  untyped floats form an exact untyped float, contextualize either untyped form
  against a concrete numeric scalar, and reject mixed concrete numeric types.
  Permit only `+`, `-`, `*`, `/`, and unary `-` on floats, while preserving the
  existing integer-only checks for remainder, wrapping operations, bits, and
  shifts. Route comparisons and checked conversions through the new constant
  helpers, including defaulting an unconstrained untyped float to `f64` and
  retaining the existing `truncate` integer-only rule.

- `src/semantic/mod.rs`, `src/lib.rs` — replace the signature-only staging
  check with one semantic-to-IR boundary that rejects any checked floating
  expression as well as floating function signatures. Keep it after successful
  semantic checking and before lowering. The next roadmap task removes this
  boundary while adding floating IR and backend support.

- `src/semantic/tests/model.rs` and other exhaustive checked-expression test
  matches — recognize the new floating literal form without weakening the
  existing integer and boolean assertions.

## Tests

- `src/semantic/tests/constants.rs` — cover every literal spelling as an exact
  rational, default `f64` typing, `f32`/`f64` round-to-nearest ties-to-even,
  exact untyped arithmetic, typed per-operation rounding, copied constants,
  and constant failures for zero division, invalid operations, and non-finite
  results, including unreachable code.

- `src/semantic/tests/expressions.rs` and `src/semantic/tests/statements.rs`
  — cover floating arithmetic, unary minus, comparisons, array elements and
  equality, contextual initialization, calls, returns, assignments, and
  compound assignments. Reject integer-only operators on floats, concrete
  `f32`/`f64` mixing, and implicit cross-category conversion.

- `src/semantic/tests/constants.rs` — cover checked constant conversions in
  both directions: rounded untyped float contexts, exact typed float widening
  and narrowing, exact and lossy integer-to-float conversions, and whole versus
  fractional or out-of-range float-to-integer conversions.

- `tests/compiler.rs` — prove a program with otherwise-valid floating values
  reaches the intentional pre-lowering diagnostic, rather than an obsolete
  semantic staging diagnostic or integer-only IR code.

## Decisions

- Exact decimal rationals serve only untyped floating constants. Once a value
  gains `f32` or `f64`, its stored IEEE bits are the single source for later
  constant operations, comparisons, and lowering.

- Float support joins the existing expression and conversion pipeline. It does
  not add a float-specific checking or lowering path; integer-only rules remain
  explicit at their existing operation boundaries.
