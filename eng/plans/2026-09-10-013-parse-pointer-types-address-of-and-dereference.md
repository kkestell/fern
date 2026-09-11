# Parse pointer types, address-of, and dereference

## Sources

- `docs/spec.md#keywords` — `null` joins the reserved words.
- `docs/spec.md#pointer-types`, `#the-null-pointer`, `#address-of`,
  `#dereference`, `#implicit-dereference` — the written forms this task parses.
- `docs/spec.md#assignment` — which expressions are locations, and the
  `*p = e;` and `(*pp).x = e;` target forms.
- `docs/spec.md#precedence-associativity-and-evaluation-order` — unary `*` and
  `&` bind like the other unary operators and looser than every postfix step.
- `eng/architecture.md#storage-and-identity` — syntax arenas and node identity.
- `src/frontend/parser.rs` — the existing annotation, unary, postfix, target,
  and lookahead patterns this change extends.

## Goal

Parse pointer type annotations, `null`, address-of, dereference, and
dereference assignment targets into frontend syntax, starting from the
struct-complete frontend. Pointer checking, IR, and native compilation remain
for the later subtasks, so a temporary semantic guard rejects the new syntax
after parsing. This does not complete the Pointers task.

## Implementation

- `src/frontend/lexer.rs` — add `#[token("null")] Null` beside `True` and
  `False`. `*`, `**`, and `&` already lex from the existing `Star` and
  `Ampersand` tokens; no other token is needed.

- `src/frontend/parser.rs` — add `"null"` to `reserved`.

- `src/frontend/syntax.rs`, `src/frontend/parser.rs` — add
  `AnnotationKind::Pointer { constant: bool, target: Idx<TypeAnnotation> }`.
  In `type_annotation`, a leading `Star` takes the nesting guard, consumes an
  optional `Token::Const`, parses the target annotation recursively, releases
  the guard, and spans from the `*` to the target's end. `named_type`, which
  serves conversion heads, stays scalar-only. `*void` needs no special case:
  `void` is not a name the annotation parser accepts.

- `src/frontend/syntax.rs`, `src/frontend/parser.rs` — add three expression
  kinds: `Null`, `AddressOf { operator_span, operand }`, and
  `Dereference { operator_span, operand }`. Keep them out of `UnaryOperator`,
  which spells value operators; follow the `LogicalNot` shape instead.
  `primary_expression` returns `ExpressionKind::Null` for `Token::Null` with
  depth 0. `unary_expression` recognizes `Star` and `Ampersand` alongside the
  existing prefix operators, threading `literals` through and keeping the same
  depth and nesting bookkeeping.

- `src/frontend/syntax.rs` — give `walk_expression` arms for the new kinds:
  `Null` is a leaf, and the two prefix forms join the single-operand group.

- `src/frontend/syntax.rs`, `src/frontend/parser.rs`,
  `src/frontend/tests/mod.rs` — delete `AssignmentTarget` and `TargetStep`, and
  make both `StatementKind::Assignment` and
  `StatementKind::CompoundAssignment` hold `target: Idx<Expression>`. A target
  is now a unary expression, which covers `*p`, `**p`, `(*pp).x`, `*a[i].next`,
  and `*f()` without a second grammar. `assignment` parses the target with
  `self.unary_expression(StructLiterals::Restricted)` so a `{` after a name
  cannot be taken for a struct literal, then reads the assignment operator and
  parses the value under the caller's `literals` mode. Project the target with
  `project_expression`; the assignment snapshots change shape accordingly.

- `src/frontend/parser.rs` — extend `token_after_target` so its lookahead skips
  leading `Star` tokens and a balanced parenthesized group before the qualified
  name, marking either as a step so `starts_call` still requires a bare
  `name(`. `statement` routes `Token::Star` and `Token::LeftParen` to
  `assignment`. `for_header` needs no change: the widened
  `starts_assignment` lookahead already separates `for *p = 0; …` from
  `for *flag { … }`.

- Implicit dereference, `uint(p)`, and `for v in *p` need no new syntax: they
  reuse the existing field, index, `len`, conversion, and iteration forms.

- `src/semantic/annotations.rs`, `src/semantic/expressions.rs` — reject
  `AnnotationKind::Pointer` at the annotation's span and `Null`, `AddressOf`,
  and `Dereference` at the expression's span with one temporary
  `pointers are not yet implemented` diagnostic.

- `src/semantic/namespaces.rs` — give `collect_annotation_references` a
  `AnnotationKind::Pointer { .. } => {}` arm that does not descend into the
  target. A pointer field breaks the size cycle, so this arm is the intended
  end state rather than a guard.

- `src/semantic/statements.rs` — rewrite `assignment_target` to walk the target
  expression instead of a step list. Descend through `Grouping`, `Index`, and
  `Field` to the innermost `Reference`, collecting steps outermost-first and
  reversing them so they stay in source order, then run the existing binding
  resolution, mutability check, and per-step checks unchanged against the
  root name's span. `Dereference` and `AddressOf` reach the temporary pointer
  diagnostic; any other expression is rejected at its own span with
  `cannot assign to an expression that is not a location`. `CheckedTarget`,
  `CheckedStep`, and IR lowering are unchanged.

## Tests

- `src/frontend/tests/parser.rs` and a new snapshot — pointer annotations on
  bindings, parameters, results, struct fields, and array elements, including
  `*const T`, `**T`, `*const *T`, `*[3]int`, and `[3]*int`; `null` in every
  position that supplies a type from context; `&x`, `&a[i].f`, `*p`, `**pp`,
  and `(*pp).x`; `*` and `&` composing with postfix steps and binary operators
  so `*p.x` is `*(p.x)`, `&a[0]` takes the element's address, and `a & b`
  still parses as a binary operation with `&b` never read as a prefix.

- Assignment targets: `*p = e;`, `**pp = e;`, `(*pp).x = e;`, `*a[i].next = e;`,
  `*f() = e;`, and compound forms such as `*p += 1;`; plus `for *p = 0; …` as a
  three-clause initializer and post statement against `for *flag { … }` as a
  condition.

- Exact-span diagnostics for a missing target type after `*` or `*const`, a
  missing operand after `&` or `*`, `null` used as an identifier, and
  `int(x) = 1;` reaching the non-location message.

- Add `null` to the name-position list in
  `reserved_names_agree_in_all_name_positions`, not the expression-position
  list, since `null` is an expression.

- Exercise the nesting limit with a deep pointer annotation and a deep prefix
  `*` chain.

- Focused semantic tests showing that a pointer annotation, `null`, `&x`, `*p`,
  and `*p = e;` each reach the temporary pointer diagnostic, while existing
  scalar, array, struct, index, field, and assignment behavior is unchanged.
