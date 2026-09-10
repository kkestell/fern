# Lower and emit floating-point values

## Sources

- `docs/spec.md#numeric-conversions` — checked conversion behavior and traps
- `docs/spec.md#floating-point-semantics` — formats, rounding, runtime
  arithmetic, and comparisons
- `docs/spec.md#array-comparison` — array equality follows its elements'
  equality
- `eng/roadmap.md#floating-point-numbers` — fourth task and remaining
  integration boundary
- `eng/architecture.md#pipeline` — verified IR is the only backend input
- `eng/plans/2026-09-10-004-check-floating-point-expressions-and-constant-expressions.md`
  — checked constants retain concrete IEEE bit patterns
- `src/ir/lower.rs`, `src/ir/verify.rs`, `src/backend/emitter.rs` — current
  integer-only lowering, verification, and QBE emission seams

## Goal

Carry checked `f32` and `f64` values through Fern IR and QBE, including numeric
conversions, arithmetic, and comparisons. This removes the pre-lowering float
guard, but leaves the milestone's compiler integration, completion fixture, and
broad native failure coverage to the next task.

## Implementation

- `src/ir/model.rs` — introduce one typed scalar-literal representation for
  integer/boolean and floating values. Use it in `Operand` and in flattened
  `Global` data, with a floating variant that stores the exact binary32 or
  binary64 bits; do not represent a float as an integer of the same width.

- `src/ir/lower.rs` — replace integer-only constant lowering and global
  flattening with the shared literal conversion. It must recursively retain
  concrete float bits in arrays and turn every concrete scalar constant into
  its typed IR literal. Keep untyped rationals unreachable: semantic checking
  has contextualized them before IR lowering.

- `src/ir/verify.rs` — validate the new literal and global forms against their
  scalar types. Generalize value validation from integer-only operations to
  numeric operations: retain the current integer and truncating-conversion
  restrictions, admit only ordinary negation and `+`, `-`, `*`, `/` for same
  format floating operands, and admit checked conversions between numeric
  scalars. Preserve the existing comparison, store, call, return, and
  initialization checks as their common paths.

- `src/semantic/mod.rs`, `src/lib.rs`, and
  `src/semantic/tests/namespaces.rs` — delete `reject_floating_values` and its
  staging tests. Successful semantic checking now proceeds directly to
  lowering for all supported value types.

- `src/backend/qbe.rs`, `src/backend/emitter.rs`, and a focused floating
  emission submodule under `src/backend/` — give `f32` and `f64` their QBE
  scalar classes and load, store, literal, data, signature, and allocation
  spellings, separate from integer word width. Render literals and static data
  from their stored bits so signed zero is retained. Dispatch scalar value
  emission by numeric category while leaving the existing checked integer
  operation path intact.

  - Emit floating unary negation, `+`, `-`, `*`, `/`, and every comparison with
    QBE's binary32/binary64 operations. These operations have no trap blocks.
  - Emit checked conversions in the same `ValueKind::Convert` path: keep the
    integer truncating path; directly widen `f32` to `f64`; check `f64` to
    `f32` by reconstructing and comparing while accepting NaN; guard
    float-to-integer conversions for NaN, infinity, range, and fractional or
    otherwise lossy results before an unsafe QBE cast; and reject
    information-losing integer-to-float conversions with an exactness check
    that never casts an out-of-range float back to an integer. Report all
    failing checked conversions through the existing source-span trap helper.
  - Keep bytewise array comparison for integer and boolean leaves. For a
    floating leaf, compare each stored element with the floating equality
    operation and combine the results, then negate that result for `!=`; this
    makes `-0.0 == 0.0` true and any NaN element make equality false.

## Tests

- `src/ir/tests/lower.rs` and its snapshots — cover concrete `f32`/`f64`
  operands and module globals by bit pattern, nonconstant arithmetic and
  comparisons, numeric conversions, and float parameters, returns, and array
  elements reaching the existing common IR forms.

- `src/ir/tests/verify.rs` — reject malformed float literal types, integer-only
  operations on floats, floating-only operations with mixed formats, and
  invalid numeric conversion or global-literal combinations while retaining
  valid float comparison and conversion IR.

- `src/backend/tests/` — add focused QBE and native tests for both formats:
  literals and signed zero, locals/globals/calls/returns, runtime arithmetic
  and IEEE comparison behavior including NaN, exact and trapping checked
  conversions in both directions, and array equality with signed zero and NaN.
  Assert emitted QBE uses floating classes and operations rather than integer
  words.

- `tests/compiler.rs` — remove the obsolete assertion that checked floating
  programs stop before lowering. The next task owns end-to-end compiler success
  and failure fixtures.

## Decisions

- The typed scalar literal is the only IR carrier of an immediate numeric
  value. Globals and operands use the same representation, so lowering,
  verification, and emission cannot disagree about a float's format or bits.

- Float conversions use one guarded checked-conversion path. Native casts run
  only after their validity predicate is established, avoiding target-defined
  behavior for NaN, infinity, and out-of-range values.
