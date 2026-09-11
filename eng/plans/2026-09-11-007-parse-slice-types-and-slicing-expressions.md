# Parse slice types and slicing expressions

## Sources

- `docs/spec.md#slice-types` — `[]T` and `[]const T`, the written forms this
  task parses, and the positions a slice type may appear in.
- `docs/spec.md#slicing-expressions` — `a[lo:hi]`, `a[lo:]`, `a[:hi]`, `a[:]`,
  and the operand forms.
- `docs/spec.md#slice-indexing`, `#slice-length`, `#slice-comparison` — reuse
  the existing index, `len`, and comparison forms and need no new syntax.
- `docs/spec.md#implicit-dereference` — slicing joins field selection, indexing,
  and `len` as a form that accepts a pointer operand, so it is a postfix step
  like indexing.
- `docs/spec.md#type-declarations` — a slice field, like a pointer field, breaks
  a struct's size cycle.
- `eng/architecture.md#storage-and-identity` — syntax arenas and node identity.
- `src/frontend/parser.rs` — the existing `type_annotation`,
  `postfix_expression`, and nesting/depth patterns this change extends.

## Goal

Parse slice type annotations and slicing expressions into frontend syntax,
starting from the pointer-complete frontend. Slice types, checking, IR, and
native compilation remain for the later subtasks, so a temporary semantic guard
rejects the new syntax after parsing. This does not complete the Slices task.

## Implementation

No new token or reserved word is needed: `[`, `]`, `:`, and `const` all lex
today, and `::` already wins maximal munch over two `:` tokens, so a qualified
name inside brackets is unaffected.

- `src/frontend/syntax.rs` — add
  `AnnotationKind::Slice { constant: bool, element: Idx<TypeAnnotation> }`
  beside `Pointer`.

- `src/frontend/parser.rs` — in `type_annotation`, after the existing
  `enter_nesting` and the `advance` past `[`, take the slice branch when the
  current token is `Token::RightBracket`: consume `]`, consume an optional
  `Token::Const`, parse the element annotation recursively, release the nesting
  guard, and span from the `[` to the element's end. The array branch keeps its
  current `[_]`/length handling and its ``expected `]` after array length``
  message, so `[3]const int` still fails at `const` with `expected a type`.

- `src/frontend/syntax.rs` — add
  `ExpressionKind::Slice { operand: Idx<Expression>, low: Option<Idx<Expression>>,
  high: Option<Idx<Expression>> }`.
  `low` and `high` are the spec's `lo` and `hi`; `None` is the omitted bound,
  not a synthesized `0` or `len`, so the later subtasks can tell `a[:]` from
  `a[0:len(a)]` in diagnostics.

- `src/frontend/parser.rs` — extend the `Token::LeftBracket` arm of
  `postfix_expression` to produce either an index or a slice from one bracket
  group:
  - After `enter_nesting` and the `advance` past `[`, parse `low` as
    `Some(self.expression()?)` unless the current token is `Token::Colon`, in
    which case `low` is `None`.
  - If the current token is `Token::Colon`, consume it, then parse `high` as
    `Some(self.expression()?)` unless the current token is
    `Token::RightBracket`, in which case `high` is `None`; expect `]` with
    ``expected `]` after slice bounds`` and build `ExpressionKind::Slice`.
  - Otherwise expect `]` with the existing ``expected `]` after index`` message
    and build `ExpressionKind::Index` as today.
  - `a[]` keeps today's behavior: neither `:` nor an expression starts there, so
    `self.expression()` reports at the `]`.
  - `step_depth` is the maximum `depth` of the bounds that are present and `0`
    when both are omitted; the surrounding `depth`, `MAX_NESTING`, and
    `self.nesting -= 1` bookkeeping is unchanged from the index case.
  - Bounds are parsed with `self.expression()`, not the caller's `literals`
    mode, matching the index case: the brackets delimit the bound, so a `{`
    inside them cannot be a statement body.

- `src/frontend/parser.rs` — `token_after_target` needs no change. Its
  bracket-skipping loop counts `[` and `]` and ignores everything between them,
  so `s[1:2] = e;` and `s[:] = e;` already reach `assignment`, where the
  semantic phase rejects the non-location target.

- `src/frontend/syntax.rs` — give `walk_expression` a `Slice` arm that visits
  `operand`, then `low`, then `high`, skipping the omitted bounds, so traversal
  order matches the spec's evaluation order.

- `src/frontend/tests/mod.rs` — project the two new nodes. `project_annotation`
  prints `slice constant={constant} span=…` followed by an `element` line and
  the element annotation, mirroring the `Pointer` arm. `project_expression`
  prints `slice span=…` followed by the operand and, for each bound, either a
  labeled subtree or a line marking it omitted, so `a[:]` and `a[0:len(a)]`
  project differently.

- `src/semantic/annotations.rs` — add an `AnnotationKind::Slice { .. }` arm to
  both `resolve_annotation` and `resolve_pointer_target` returning
  `Err(Diagnostic::new(written.span.clone(), "slices are not yet implemented"))`.
  No `Type` variant is constructed, so `validate_aggregate_layout` is untouched.

- `src/semantic/expressions.rs` — add an `ExpressionKind::Slice { .. }` arm to
  `infer_expression` returning the same temporary diagnostic at the expression's
  span, and an explicit `ExpressionKind::Slice { .. }` arm ahead of the `_`
  fallback in `check_location_with_facts` returning it as well, so `s[1:2] = e;`
  reports the temporary message rather than the non-location message it would
  otherwise inherit.

- `src/semantic/namespaces.rs` — give `collect_annotation_references` an
  `AnnotationKind::Slice { .. } => {}` arm that does not descend into the
  element. A slice field breaks the size cycle, so this arm is the intended end
  state rather than a guard, matching the existing `Pointer` arm.

## Tests

- `src/frontend/tests/parser.rs` and a new snapshot — slice annotations on
  bindings, parameters, results, struct fields, and array elements, including
  `[]int`, `[]const int`, `[][]int`, `[]const []int`, `[]*int`,
  `[]const *const
  int`, `[3][]int`, `[]([3]int)` written as `[][3]int`, and
  `*[]int`; a struct that reaches itself through a `[]Node` field; and all four
  slicing forms `a[1:3]`, `a[1:]`, `a[:3]`, `a[:]`.

- Slicing composes as a postfix step: `m[0][1:2]`, `s[1:3][0]`, `s[1:3].f`,
  `p[1:2]` through a pointer operand, `f()[1:]` on a call result, `*p[1:2]`
  parsing as `*(p[1:2])`, `&s[1:2]` parsing as `&(s[1:2])`, `len(s[1:3])`,
  `s[1:3] == t[:]`, and `for v in s[1:3] {}`.

- Bounds are full expressions: `a[i+1:len(a)-1]`, `a[m::n:]` taking a qualified
  name as `lo`, `a[:m::n]` taking one as `hi`, and `a[b[0]:b[1]]`.

- Exact-span diagnostics for a missing element type after `[]` or `[]const`, a
  missing bound expression where one must appear, `a[1:2:3]` reaching
  ``expected `]` after slice bounds``, `a[1 2]` still reaching
  ``expected `]` after index``, and `[]void` reaching `expected a type`.

- Exercise the nesting limit with a deeply nested slice annotation and a deep
  chain of slicing steps.

- `src/semantic/tests/` — focused tests showing that a slice annotation in each
  position, a slicing expression in value position, and `s[1:2] = e;` each reach
  the temporary `slices are not yet implemented` diagnostic, while existing
  array indexing, pointer, struct, and assignment behavior is unchanged.
