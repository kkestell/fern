# Check struct values, fields, and equality

## Sources

- `docs/spec.md#type-declarations`, `#struct-literals` — nominal struct types,
  field validity, finite inline types, literals, copying, zero fill, and equality
- `docs/spec.md#module-level-declarations`, `#public-declarations` — the shared
  module namespace, declaration-order independence, and imported public types
- `docs/spec.md#assignment`, `#comparison` — mutable field targets and operator
  typing
- `eng/roadmap.md#structs` — the semantic-checking task and later integration
  boundary
- `eng/architecture.md#storage-and-identity`, `#diagnostics`, `#testing` —
  checked-program ownership, syntax identity, and semantic test placement
- `eng/plans/2026-09-10-007-parse-struct-syntax.md` — the syntax nodes and
  temporary semantic guards this task starts from

## Goal

Resolve named structs and check their fields, values, selections, assignments,
and equality. The parser work from the preceding task is the starting state;
structs remain blocked before Fern IR until the next roadmap task represents
them there.

## Implementation

- `src/types.rs` — add a program-local `StructId` and a nominal struct variant
  to `Type`. Keep the declaration name with the ID so existing type diagnostics
  can display `Point`, while equality uses the program-unique ID. Update type
  classification and exhaustive matches without teaching the scalar/array
  `leaf` and `element_count` helpers a false homogeneous representation for
  structs; the pre-IR guard below keeps structs away from those helpers.

- `src/semantic/model.rs` — replace `structs_unimplemented` with checked struct
  storage owned by `CheckedProgram`: each struct records its syntax declaration,
  ordered checked fields, a field-name-to-ordinal map, and unresolved/resolving/
  resolved state for cycle detection. Add `Constant::Struct` in declaration
  field order. Add checked expression forms for struct literals and field
  selections, recording field ordinals rather than resolving names again in a
  later phase. Extend `CheckedTarget` with ordered checked index/field steps so
  assignment checking and later lowering share the same resolved target.

- `src/semantic/namespaces.rs` — claim struct names alongside functions and
  bindings, allocate every module struct ID before resolving any field, and add
  structs to `DeclarationKind` and module namespaces with their visibility.
  Extend qualified and selective import resolution with a type-position lookup:
  it marks imports used, accepts only struct declarations, and reports bindings,
  functions, and modules with type-specific diagnostics. Preserve the existing
  common module namespace and lexical shadowing behavior.

- `src/semantic/namespaces.rs`, `src/semantic/annotations.rs` — resolve struct
  fields lazily after all names and imports exist. A named annotation resolves
  to its struct ID, then ensures that declaration's fields are resolved. Resolve
  field annotations in source order, reject duplicate field names on the second
  name, and diagnose a resolving struct reached again through a struct field or
  array element as an invalid recursive struct type. Finish resolving every
  otherwise-unused struct before function signatures and bodies are checked, so
  invalid declarations cannot hide behind lack of use.

- `src/semantic/namespaces.rs` — extend module-binding dependency collection to
  follow local struct types named by annotations and struct literals. Recursively
  collect module binding references from their field annotations' array lengths,
  with a visited-struct set. This keeps declaration-order independence for cases
  such as a struct field `[count]int` and ensures `count` is checked before a
  module-level value needs that struct. Keep the existing binding initializer
  order and cycle diagnostic path.

- `src/semantic/expressions.rs` — check a struct literal against the type named
  by the literal itself. Resolve each initializer once in source order, reject
  unknown or repeated fields, require every field without `...`, and check each
  expression against the declared field type. With `...`, synthesize recursive
  zero constants for omitted fields. Fold a literal to `Constant::Struct` only
  when all written initializers fold, while retaining source-order expression
  IDs plus field ordinals for runtime evaluation.

- `src/semantic/expressions.rs` — infer a field selection by checking its
  operand, requiring a struct, and looking up the selected field once. Record
  the operand and resolved ordinal and return the field type; do not mark a
  selection constant because the specification does not include it among
  constant-expression forms. Let the existing comparison unification accept
  identical nominal struct types only for `==` and `!=`; `Constant` equality
  then folds nested struct/array constants structurally.

- `src/semantic/statements.rs` — walk mixed assignment target steps left to
  right. Reuse `check_index_step` for indices and the same resolved-field helper
  used by expression selection for fields, descending through types after every
  step and recording the checked steps. Keep the root binding mutability check
  first, so fields and elements of a `const` value remain immutable. Ordinary
  assignment uses the final field type as its destination; compound assignment
  continues through the shared numeric-operator checker.

- `src/semantic/model.rs`, `src/lib.rs`, `src/ir/lower.rs` — add a temporary
  `structs are not yet compiled` check between semantic checking and IR lowering,
  anchored to the first struct declaration. Update IR exhaustive matches for
  the checked struct expression and target forms as unreachable behind that
  guard. Do not add a partial struct representation or lowering path in this
  task.

## Tests

- `src/semantic/tests/namespaces.rs`, `annotations.rs` — cover duplicate struct
  names against every module declaration kind; private/public and selective/
  qualified imported types; wrong-kind and unknown type diagnostics; same-module
  forward references; fields containing arrays and other structs; duplicate
  fields; a field array length using a later module `const`; and direct,
  indirect, and array-mediated recursive struct cycles.

- `src/semantic/tests/expressions.rs` — cover complete literals and `...` alone
  or after fields; initializer source order; missing, duplicate, and unknown
  fields; scalar and aggregate field type mismatches; module-level constant
  structs with recursive zero fill; nominal type mismatch; selections through
  nested structs and arrays; selection from a non-struct; and `==`/`!=` across
  runtime and nested constant structs while ordering is rejected.

- `src/semantic/tests/statements.rs` — cover simple and compound assignments to
  mutable fields, immutable roots, wrong value types, unknown fields, and mixed
  targets such as `grid[i].point.x` and `value.rows[i].count`.

- `tests/compiler.rs` — prove a semantically valid struct program reaches the
  temporary pre-IR diagnostic and preserves an existing output file.

## Decisions

- Struct IDs are allocated by semantic checking rather than reusing arena raw
  indices as an undocumented cross-phase identity. A checked struct retains its
  syntax declaration ID as the explicit relation back to source.

- Field ordinals are the one checked identity used by literals, selections, and
  assignment targets. Names remain syntax and diagnostic data only after
  resolution.
