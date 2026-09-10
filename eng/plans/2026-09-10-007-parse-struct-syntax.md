# Parse struct syntax

## Sources

- `docs/spec.md#type-declarations`, `#struct-literals`, `#assignment` — the
  declaration, literal, selection, and assignment forms this task must parse
- `docs/spec.md#precedence-associativity-and-evaluation-order`, `#conditionals`
  — postfix precedence and the parenthesized-literal rule in control conditions
- `docs/spec.md#public-declarations` — public struct declarations and imported
  type names
- `eng/roadmap.md#structs` — this parser-only roadmap task and its milestone
  boundary
- `eng/architecture.md#pipeline`, `#storage-and-identity`, `#testing` — syntax
  ownership, program-wide arenas, and frontend test placement
- `src/frontend/syntax.rs`, `src/frontend/parser.rs` — the existing annotation,
  postfix-expression, assignment-target, nesting, and arena patterns

## Goal

Parse named struct declarations, struct literals, named type annotations, and
mixed field/index selection chains into frontend syntax. This task starts from
the scalar-and-array frontend and leaves struct name resolution, checking, IR,
and native compilation to the remaining milestone tasks, so temporary semantic
guards reject the new syntax after parsing.

## Implementation

- `src/frontend/lexer.rs`, `src/frontend/parser.rs` — add `type` and `struct`
  tokens and reserve both words in every identifier position. The top-level
  `type` branch accepts the roadmap's `type Name struct { ... }` form, requires
  at least one `name: T` field, permits comma separators and a trailing comma,
  and ends at `}` without a semicolon.

- `src/frontend/syntax.rs` — add a `StructDeclaration` arena node containing
  the declaration name and span, ordered fields with their name spans and
  annotation IDs, and the full declaration span. Add a public-aware struct arm
  to `TopLevelItem` and a struct arena to `Syntax`.

- `src/frontend/syntax.rs`, `src/frontend/parser.rs` — split written scalar
  annotations from declared names: keep built-ins as
  `AnnotationKind::Scalar(Scalar)` and represent a plain or `module::Name`
  declared type as `AnnotationKind::Named(QualifiedName)`. Reuse this annotation
  parser recursively for array elements and struct fields. Numeric conversion
  heads continue through the scalar-only parser.

- `src/frontend/syntax.rs`, `src/frontend/parser.rs` — represent a struct
  literal with its qualified type name, ordered field initializers carrying
  field-name spans and expression IDs, and an optional `...` span. Parse the
  exact literal forms from the specification: at least one field or `...`, `=`
  between a field and its value, comma-separated fields, and a final optional
  `...` that may appear alone. Guard nested literal parsing with the existing
  source-nesting limit.

- `src/frontend/parser.rs` — add a condition-expression entry that does not
  recognize a named struct literal directly before an `if` or condition-form
  `for` body. Thread that permission through the precedence and unary parsers;
  grouping and other delimited subexpressions call the ordinary expression
  entry, so parenthesizing either the literal or the whole condition permits
  it. Iteration operands and three-clause initializer/post expressions remain
  ordinary expressions; only the `for` condition uses the restricted entry.

- `src/frontend/syntax.rs`, `src/frontend/parser.rs` — add a field-selection
  expression carrying its operand, field name, and field-name span. Extend the
  existing postfix loop to parse both `.` and `[ ... ]` in source order, so
  calls, literals, fields, and indices compose as `make().rows[i].value` and
  continue to bind more tightly than unary and binary operators. Apply the same
  depth accounting used by indexing.

- `src/frontend/syntax.rs`, `src/frontend/parser.rs` — replace an assignment
  target's index-only list with one ordered list of target steps, whose variants
  are an index expression and a field name/span. Extend both target parsing and
  `token_after_target` to recognize mixed chains such as
  `grid[row].point.x`; call statements still require a direct qualified name
  with no target steps.

- `src/frontend/syntax.rs`, `src/frontend/tests/mod.rs` — visit struct literal
  initializers and field operands in source order. Project struct declarations,
  scalar and declared annotations, struct literals, field selections, and mixed
  assignment-target steps while preserving existing array snapshot wording.

- `src/semantic/annotations.rs`, `src/semantic/expressions.rs`,
  `src/semantic/statements.rs`, `src/semantic/namespaces.rs`, `src/ir/lower.rs`
  — update exhaustive matches for the new syntax. Keep existing scalar, array,
  and index behavior on their current paths; reject struct declarations, named
  annotations, struct literals, field expressions, and field target steps with
  one temporary `structs are not yet implemented` diagnostic before lowering.
  Continue traversing nested expressions where shared syntax utilities require
  it. Mark the IR field-target arm unreachable behind that semantic guard.

## Tests

- `src/frontend/tests/parser.rs` and a new snapshot — cover private and public
  declarations; scalar, array, local named, and qualified field annotations;
  literals with fields in source order and with `...` alone or last; literals
  nested in fields, arrays, calls, and selections; and mixed field/index chains
  in expressions, simple assignments, and compound assignments.

- Prove ordinary bare-name `if` and `for` conditions still parse, direct named
  struct literals in those conditions are rejected, and parentheses around the
  literal or whole condition admit them.

- Add exact-span diagnostics for an empty struct declaration, missing field
  names, colons, types, commas, and braces; a semicolon after a struct
  declaration; empty and malformed literals; misplaced `...`; and missing or
  reserved field names after `.`. Include the unsupported non-struct
  `type Name T;` form so this task cannot silently expand beyond the roadmap.

- Extend reserved-name agreement tests for `type` and `struct`. Exercise the
  nesting limit with nested struct literals and long mixed postfix chains.

- Add focused semantic tests showing that each new syntax boundary reaches the
  temporary struct diagnostic while existing scalar annotations, arrays,
  indexing, assignments, and control conditions retain their current behavior.
