# Parse array syntax

## Sources

- `docs/spec.md#array-types`, `#array-literals`, `#fill`, `#indexing`,
  `#length`, `#assignment`, `#loops` — the forms this task must accept
- `docs/spec.md#keywords` — `in` and `len` are reserved
- `docs/spec.md#precedence-associativity-and-evaluation-order` — indexing,
  calls, conversions, and `len` bind more tightly than any operator
- `eng/roadmap.md#arrays` — first task and its parser boundary
- `src/frontend.rs` — lexer, syntax arenas, parser, and syntax snapshots

## Goal

Parse every array form into `Syntax`: array type annotations, array literals
with fill, indexing, `len`, element assignment targets, and both `for … in`
forms. This is a parser-only slice. Array types, checking, IR, and codegen
arrive in the following tasks, so semantic checking rejects the new forms for
now.

## Implementation

- `src/frontend.rs` — add `[`, `]`, `...`, `in`, and `len` tokens, and add
  `"in"` and `"len"` to `reserved`.

- `src/frontend.rs` — make a type annotation a node. Add
  `Syntax::annotations: Arena<TypeAnnotation>` and give `TypeAnnotation` a
  `kind` of `Named(Type)` or `Array { length: Option<Idx<Expression>>, element:
  Idx<TypeAnnotation> }`, where `None` is `[_]`. `Parameter::annotation`,
  `FunctionResult::Value`, and `StatementKind::Binding::annotation` hold
  `Idx<TypeAnnotation>`.

- `src/frontend.rs` — split the current `type_annotation` into `named_type`,
  returning a `Type` and its span, and a recursive `type_annotation` that reads
  `[` then either `_` or an expression, then `]`, then the element annotation.
  Guard the recursion with `enter_nesting`. `ExpressionKind::Conversion` keeps
  a plain `destination: Type` with its span and calls `named_type`: a
  conversion head is always an integer type name, so routing it through the
  annotation node would create an arm the checker can never reach.

- `src/frontend.rs` — add `ExpressionKind::ArrayLiteral { elements, fill:
  Option<Range<usize>> }`, `Index { operand, index, bracket_span }`, and
  `Length { operand }`. `primary_expression` gains a `[` branch for the literal
  and a `Len` branch; `unary_expression` falls through to a new
  `postfix_expression`, which parses a primary and then loops while the current
  token is `[`, so `-a[0]` negates the element and `f(x)[0]` and `grid[r][c]`
  chain. Compute `depth` and bracket the recursion with `enter_nesting` the way
  `call_parts` does.

- `src/frontend.rs` — extract the `exit` argument list into a helper that
  parses `( expression [,] )` and returns the operand and the `)` span, and use
  it for both `exit` and `len`. Keep the existing `exit` diagnostic strings so
  current tests do not move. A `len` not followed by `(` reports
  `reserved word cannot be used as an identifier` on the `len` span, matching
  `conversion_expression`.

- `src/frontend.rs` — add `AssignmentTarget { name: QualifiedName, indices:
  Vec<Idx<Expression>> }` and use it in `StatementKind::Assignment` and
  `CompoundAssignment`. Replace `token_after_qualified_name` with one scanner
  that also skips balanced `[ … ]` groups and reports whether it skipped any:
  `starts_call` requires no indices and a following `(`, `starts_assignment`
  accepts indices.

- `src/frontend.rs` — add `ForHeader::Iteration { value: Spur, value_span,
  index: Option<(Spur, Range<usize>)>, operand }`. In `for_statement`, select it
  when the current token is an unreserved `Name` followed by `in` or `,`; no
  condition can contain either at that position. Move the three-clause clause
  parsing into its own function first — `for_statement` is at cognitive 13 of
  15 and a fourth header form pushes it over.

- `src/frontend.rs` — add `project_annotation`, printing `named` on one line or
  `array` with a nested length (via `project_expression`, or `inferred` for
  `[_]`) and a nested element annotation, and call it from the parameter,
  result, and binding projections. Project the new expression kinds, target
  indices, and the iteration header. Every existing frontend snapshot that
  shows an annotation moves; review the updates.

- `src/semantic.rs` — resolve a `Named` annotation as today and report
  `arrays are not yet implemented` for an `Array` one, for the new
  `infer_expression` arms, for a `ForHeader::Iteration` header, and for an
  assignment target with indices. The first four are forced by exhaustive
  matches; each is deleted by the task that implements it. `annotation_type`
  becomes fallible and needs the annotation arena.

- `src/ir.rs` — `lower_for` gets an `Iteration` arm that is `unreachable!`,
  removed in the IR task.

## Tests

- A frontend snapshot covering `[3]int`, `[2][3]int`, and `[_]int` annotations,
  an array parameter and result, `[1, 2, 3,]`, `[0...]`, `[1, 2, 3, 0...]`,
  `[[1...]...]`, `grid[r][c]`, `f(x)[0]`, `-a[0]`, `len(a)`, `len(a,)`,
  `grid[r][c] = e;`, `a[i] += 1;`, and both `for … in` forms with and without a
  label.

- Parser diagnostics with exact offending spans for `[]`, `[1,,2]`, `[1 2]`,
  `[...]`, `var x: [3] = …`, `var x: [_ + 1]int = …`, `a[]`, `a[0 = 1;`,
  `len`, `len()`, `len(a, b)`, `for v, in a {}`, `for v i in a {}`,
  `for in a {}`, and `for v in {}`.

- `len` and `in` added to both reserved-name lists in
  `reserved_names_agree_in_all_name_positions`.

- Nesting: a deep chain of index expressions, array literals, and nested array
  annotations each report the `MAX_NESTING` diagnostic.

- Semantic: one program per new form fails with the temporary guard.

## Extra validation

- `scripts/code-health.sh check` — `for_statement` and `primary_expression`
  both gain branches in this task.
