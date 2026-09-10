# Represent array types across the compiler

## Sources

- `docs/spec.md#array-types` — length is a constant expression of type `int`,
  at least 1; element type is any value type; `[N]T` types compare structurally
- `docs/spec.md#array-literals` — `[_]T` takes its length from an array-literal
  initializer and is not allowed as a parameter or result type
- `docs/spec.md#module-level-declarations` — a module-level initializer must be
  a constant expression
- `eng/roadmap.md#arrays` — second task
- `src/types.rs` — the type enum this task splits
- `src/semantic.rs:20` — `annotation_type`, the stub this task replaces

## Goal

Give the compiler a type representation that names array types, and resolve
written annotations into it. Array *values* stay unimplemented: the parser-task
guards on array literals, indexing, `len`, and iteration remain, so no array
value can be built and nothing array-shaped reaches Fern IR.

## Implementation

### `src/types.rs`

- Rename today's enum to `Scalar`, unchanged. Every method (`width`, `signed`,
  `name`, `min`, `max`, `all_values_fit`, `ALL_INTEGERS`) stays on `Scalar`.
  These are total on a scalar and meaningless on an array, so a single
  recursive `Type` would make all of them partial and put an unwrap or a panic
  at every arithmetic site in `semantic.rs` and `backend.rs`. `Scalar` is how
  the code states that arithmetic, conversions, overflow, and shifts are
  defined only on scalars. Use a word-boundary rename; `sed` on macOS does not
  support `\b`.

- Add the value-type enum:

  ```rust
  #[derive(Debug, Clone, PartialEq, Eq)]
  pub(crate) enum Type {
      Scalar(Scalar),
      Array { length: u64, element: Box<Type> },
  }
  ```

  Derived `PartialEq` is the structural comparison. Add `impl Display` for both:
  `Scalar` writes `name()`, `Type` writes the scalar spelling or `[{length}]`
  followed by the element. Add `fn scalar(&self) -> Option<Scalar>`.

### `src/semantic.rs` — widen to `Type`

Only the places that hold the type of a value widen; every arithmetic and
constant-evaluation helper keeps `Scalar`. The sites are `Binding::ty`,
`CheckedExpression::ty`, `FunctionSignature::result`, `check_expression`'s
`destination: Option<Type>`, and the `result: Option<Type>` threaded through the
statement checkers. Where those feed integer logic, extract with `scalar()`.
Diagnostics that print a value's type switch from `ty.name()` to `{ty}`.

### `src/semantic.rs` — resolve annotations

Replace `annotation_type` with a resolver that takes the annotation, the scope
list, and the declaration's initializer when there is one. Recursive over
`AnnotationKind`:

- `Named(scalar)` → `Type::Scalar(scalar)`. A `void` element type needs no check
  here: `void` is its own token, so `named_type` already rejects `[3]void` with
  `expected a type`. Cover it with a parser test rather than new code.
- `Array { length: Some(e), element }` → check `e` with destination
  `Some(Type::Scalar(Scalar::Int))`, require `constant`, convert to `u64`, and
  require at least 1. Diagnose on `e`'s span: a non-constant length, a length
  below 1, and a length above `Scalar::Int.max()`.
- `Array { length: None, element }` → `[_]`. Valid only when the initializer is
  an `ExpressionKind::ArrayLiteral`; the length is its element count. A literal
  with a fill, or no array-literal initializer at all, is an error on the
  annotation's span.

### `src/semantic.rs` — resolve lengths where module constants exist

A length may name a module-level `const`, so resolution must run after that
constant has a value. Reorder `check_module`:

1. `declare_module_bindings` stops resolving the annotation and always allocates
   the `Scalar::Int` stand-in it already documents.
2. Split `declare_functions` into `declare_function_names`, which only fills the
   namespace and runs where it does today, and a new `resolve_signatures`, which
   allocates parameter bindings and the signature and runs after
   `check_module_initializers` with `module_scope`.
3. `check_module_initializer` already resolves the binding's own annotation, in
   dependency order. Extend `module_dependencies` to walk the annotation's
   length expressions as well as the initializer, so `var a: [N]int = …;`
   before `const N = 3;` orders correctly and the existing cycle diagnostic
   covers a cyclic length.

`resolve_signatures` running late means `self.functions` is empty while module
initializers are checked. A call there is not a constant expression, so reject
it up front: in `check_module_initializer`, walk the initializer for
`ExpressionKind::Call` before checking it and report the existing
`module-level initializer must be a constant expression` on the call's span. An
existing test asserts that message; its span moves.

### Guard

In `resolve_signatures`, a parameter or result of array type reports
`arrays are not yet implemented` on the annotation's span. This is the only
remaining way an array value could reach lowering, and the next tasks delete it.

### `src/ir.rs`, `src/backend.rs`

Both keep `Scalar` for now. At the semantic-to-IR boundary, extract with
`scalar().expect("array values are rejected until arrays are lowered")`.

This is a staging decision, not the end state. `Represent arrays in Fern IR`
widens IR locals, places, globals, and function results to `Type`, because a
local can be an array. What keeps `Scalar` permanently is the arithmetic:
`qbe_type`, the shift, multiply, and comparison emitters, and the conversion
logic never see an array. Widening IR storage now would add cases with no
reachable input.

## Tests

- `types.rs` unit tests: `[2][3]int` equals itself and differs from `[3][2]int`
  and `[2][3]i64`; `Display` gives `[2][3]int`.
- Resolution unit tests calling the resolver directly: `[3]int`, `[2][3]int`,
  `[_]int = [1, 2, 3]` → `[3]int`, and a length that names a module `const`.
- Semantic diagnostics with spans: `[0]int`, `[-1]int`, `[i]int` for a `var`
  binding `i`, `[true]int`, `[1 + 1]bool` succeeding but `[N]int` with a cyclic
  `N`, `[_]int` as a parameter and as a result, `[_]int = [0...]`, and
  `var a: [3]void`.
- A module-level `const N = 3;` declared after `var a: [N]int = …;` resolves.
- A call in a module-level initializer still reports the constant-expression
  diagnostic.

## Extra validation

- `scripts/code-health.sh check` — the new resolver and the reordered
  `check_module` are the functions at risk.
