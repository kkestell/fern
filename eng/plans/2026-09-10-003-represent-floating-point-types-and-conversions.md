# Represent floating-point types and numeric conversions

## Sources

- `docs/spec.md#floating-point-types` — `f32` and `f64` are distinct,
  fixed-width value types
- `docs/spec.md#numeric-conversions` — numeric conversion heads and the
  integer-only `truncate` form
- `docs/spec.md#floating-point-constant-expressions` — float values and their
  evaluation belong to the following task
- `eng/roadmap.md#floating-point-numbers` — second task and milestone order
- `eng/architecture.md#pipeline` — checked meaning precedes IR and native
  compilation
- `src/types.rs`, `src/frontend/parser.rs`, `src/semantic/expressions.rs` —
  existing scalar, conversion, and temporary-feature boundaries

## Goal

Make `f32` and `f64` usable as written types and conversion destinations
through the frontend and checked-program model. Floating values, expression
checking, IR, and emission remain unfinished; no floating value may reach the
integer-only IR or backend.

## Implementation

- `src/types.rs` — add `Scalar::F32` and `Scalar::F64` to the canonical named
  type table, with their fixed widths. Keep integer-only facts explicit:
  `is_integer` must exclude floats, and range, signedness, and
  all-values-fit helpers must only be usable for integer scalars. Add the
  small floating-type predicate needed by the temporary boundary, including
  recursive detection in `Type` for an array whose element type is floating.
  Keep `ALL_INTEGERS` integer-only.

- `src/frontend/parser.rs` — let both floating type names enter the existing
  named-annotation path and recognize either floating type as a checked
  conversion head. Remove the now-redundant special reservation of their
  spellings. Reject `f32.truncate` and `f64.truncate` at the conversion head;
  only an integer destination has that syntax. Preserve the existing parse
  shape for `ExpressionKind::Conversion`, so later checking and lowering use
  the same conversion representation for every numeric type.

- `src/semantic/expressions.rs` — retain the floating-literal staging
  diagnostic from the preceding parser task, and add one shared temporary
  guard before an integer-only concretization or conversion receives a
  floating destination or operand. This must cover an integer initializer,
  argument, return, or assignment contextualized to `f32`/`f64`, as well as a
  floating conversion head, without calling integer range helpers on a float.

- `src/semantic/mod.rs` and `src/lib.rs` — add a semantic-to-IR boundary check
  and call it after successful semantic checking. It finds floating parameter
  and result types (including array element types) and reports their written
  annotation span before lowering. The later IR/lowering task removes this
  guard; do not add float operands, constants, QBE type spellings, or a second
  lowering path in this slice.

## Tests

- `src/types.rs` — cover type-name lookup and display, fixed widths, and the
  distinction between integer-only helpers and `f32`/`f64`.

- `src/frontend/tests/parser.rs` and its snapshot — cover floating annotations
  in parameters, results, bindings, and arrays; `f32(e)` and `f64(e)`; and a
  diagnostic for floating `.truncate`. Remove `f32` and `f64` from the
  malformed-annotation cases.

- `src/semantic/tests/expressions.rs` and
  `src/semantic/tests/namespaces.rs` — prove signatures retain floating types,
  while a floating literal, an integer contextualized to a floating annotation,
  and an integer-to-floating conversion stop at the temporary semantic guard
  rather than an integer range path.

- `tests/compiler.rs` — compile a program that uses a floating function
  signature but no floating expression, and assert that the pre-lowering guard
  fails without replacing an existing output.

## Decisions

- `Scalar` remains the one type-name representation. Float storage and
  arithmetic do not need a parallel numeric or conversion enum: the existing
  conversion syntax already records its destination scalar and `truncating`
  flag.

- The compiler rejects unsupported float values at the earliest boundary that
  would otherwise enter integer-only logic, and again before lowering for a
  signature with no value expression. This is temporary staging, not a Fern
  language rule.
