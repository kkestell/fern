# Parse pointer syntax

## Goal

Lex and parse the pointer forms in `docs/spec.md#pointers-and-heap-memory`
into frontend syntax: `*T` and `nullable *T` annotations, `null`, prefix `&`
and `*`, `new(e)`, the `free(p);` statement, and the `*T(p)` conversion.
`nullable`, `new`, `free`, and `null` become reserved in every identifier
position. Semantic checking rejects every new form with one temporary
`pointers are not yet implemented` diagnostic, so nothing reaches lowering
until the checking tasks land.

## Related code

- `src/frontend/lexer.rs` — `Token`; `null` lexes as a keyword token, the same
  way `true` and `false` do.
- `src/frontend/parser.rs`:
  - `reserved` — the reserved-word list.
  - `statement` — dispatches on the first token. `*p = e;` must reach
    `assignment`.
  - `token_after_target` and `starts_assignment` — target lookahead used by
    `statement` and `for_header`.
  - `single_argument` — the `( e [,] )` form `exit` and `len` share.
  - `type_annotation`, `unary_expression`, `primary_expression`,
    `length_expression`, and `conversion_expression` — the patterns the new
    forms follow.
- `src/frontend/syntax.rs` — `AnnotationKind`, `ExpressionKind`,
  `StatementKind`, and `walk_expression`.
- `src/frontend/tests/mod.rs` — `project_annotation`, `project_expression`,
  and the statement projection.
- `src/frontend/tests/lexer.rs` `reserved_names_agree_in_all_name_positions` —
  currently asserts that `free` and `null` are not reserved.
- `src/semantic/annotations.rs` `resolve_annotation`,
  `src/semantic/expressions.rs` `infer_expression` and `check_location`,
  `src/semantic/statements.rs` `check_statement` — the exhaustive matches that
  get the temporary guard.
- `src/semantic/namespaces.rs` `collect_annotation_references` and
  `collect_references` — the walks that order module-level initializers.
- `src/ir/lower.rs` `lower_flow_statement` — the exhaustive `StatementKind`
  match.
- Commit `36e6182` removed an earlier pointer frontend. Its `AddressOf`,
  `Dereference`, `Null`, pointer annotation, leading-`*` target lookahead, and
  their snapshot projections are a starting point. The `*const T` form is not
  part of the language.

## Decisions

**`*` followed by a type and `(` is a conversion.** The spec rule
(`docs/spec.md#nullable-pointers`) says that a `*` followed by a type and then
`(` always begins `*T(p)`, and that the type extends as far as a type can. The
parser decides this with a lookahead, `starts_pointer_conversion`, that scans
from the current `*` on a cloned lexer:

1. Accept any sequence of `*`, `nullable` followed by `*`, and a balanced
   `[ … ]`.
2. Then require a type name: a numeric scalar name, `bool`, or a
   non-reserved name optionally followed by `::` and a name.
3. Report a conversion only when the next token is `(`.

So `**Node(p)` converts to `**Node`, `*f(x)` converts to `*f`, `*(f(x))`
dereferences a call, `**p` dereferences twice, and `*[1, 2][0]` dereferences an
indexed literal, because no type name follows its bracket groups. A conversion
is a primary expression, so `*Node(c).value` selects a field of the conversion
result. Like `T(x)`, it takes exactly one operand and no trailing comma.

**Syntax nodes.**

- `AnnotationKind::Pointer { nullable: bool, target: Idx<TypeAnnotation> }`.
  Its span runs from `nullable`, or from `*` when there is no `nullable`, to the
  end of the target. `nullable` must be followed by `*` (`expected `*` after
  `nullable``).
- `ExpressionKind::Null`.
- `ExpressionKind::AddressOf { operator_span, operand }` and
  `ExpressionKind::Dereference { operator_span, operand }`, parsed in
  `unary_expression` as prefix operators at the same level as `-` and `!`. The
  operand is a unary expression, so `*p.x` is `*(p.x)`, as the spec's
  precedence section requires.
- `ExpressionKind::New { operand }` — parsed like `len` through
  `single_argument("new")`, so a trailing comma is accepted. `new` without `(`
  reports `reserved word cannot be used as an identifier` at `new`.
- `ExpressionKind::PointerConversion { destination: Idx<TypeAnnotation>,
  operand }` — `destination` is the whole `*T` annotation. The existing
  `Conversion` stays scalar-only.
- `StatementKind::Free { operand }` — `free` followed by `single_argument("free")`
  and `;`, the same as `exit`.

The `null` token and `&&` stay as they are: by the longest-match rule `&&x`
lexes as `&&`, which never begins an expression.

**`*void` is a parse error.** `void` is not an annotation, so `*void` reports
`expected a type` at `void` through the existing `named_type` path.

**Assignment through a pointer.** `statement` sends a leading `*` to
`assignment`. `token_after_target` skips leading `*` tokens before the target
and counts them as a step, so `*p = 1;`, `**pp += 1;`, and `*p.x = 1;` are
assignments, and `for` headers detect `*p = 0` as a three-clause initializer.
A target that is not a location, such as `&x = 1;`, still fails in semantic
checking.

**Temporary semantic guard.** A helper
`pointers_not_implemented(span) -> Diagnostic` in `src/semantic/annotations.rs`
returns `pointers are not yet implemented`. It is reported:

- in `resolve_annotation` for `Pointer`, at the annotation span;
- in `infer_expression` for `Null`, `AddressOf`, `Dereference`, `New`, and
  `PointerConversion`, at the expression span;
- in `check_location` for `Dereference`, so that `*p = e;` and `(*p).x = e;`
  report the guard and not the not-a-location error;
- in `check_statement` for `Free`, at the statement span.

`lower_flow_statement` marks `Free` as `unreachable!` behind the guard. The
namespace walks traverse the new forms without the guard:
`collect_annotation_references` recurses into a pointer's target, and
`collect_references` also follows a `PointerConversion` destination through
`collect_annotation_references`. Syntax fields that only snapshots read take
the existing `cfg_attr(not(test), expect(dead_code, …))` attribute.

## Implementation plan

1. `src/frontend/lexer.rs` — add `#[token("null")] Null`, `#[token("new")]
   New`, `#[token("free")] Free`, and `#[token("nullable")] Nullable`.
2. `src/frontend/parser.rs` `reserved` — add `null`, `new`, `free`, and
   `nullable`.
3. `src/frontend/syntax.rs` — add the nodes listed under Decisions, and
   extend `walk_expression`: `Null` is a leaf; `AddressOf`, `Dereference`,
   `New`, and `PointerConversion` visit their operand.
4. `src/frontend/parser.rs` `type_annotation` — parse `*T` and
   `nullable *T` recursively under `enter_nesting`. Update the doc comment.
5. `src/frontend/parser.rs` — add `starts_pointer_conversion` and
   `pointer_conversion_expression`. `unary_expression` sends a `*` that starts
   a conversion to `postfix_expression`, and `primary_expression` dispatches it
   to `pointer_conversion_expression`. That function parses the annotation with
   `type_annotation`, then `(`, one expression, and `)`, under `enter_nesting`,
   with depth one more than the operand.
6. `src/frontend/parser.rs` `unary_expression` — add `AddressOf` and
   `Dereference` to `PrefixOperator` for `&` and `*`.
7. `src/frontend/parser.rs` `primary_expression` — parse `null`, and parse
   `new` in a `new_expression` that mirrors `length_expression`.
8. `src/frontend/parser.rs` `statement` — parse `free(e);`, send `Token::Star`
   to `assignment`, and add `free` to the "expected a declaration, …"
   message. In `token_after_target`, skip leading `*` tokens and count them as
   a step.
9. `src/frontend/tests/mod.rs` — project `pointer nullable=… span=…` with its
   target, `null`, `address-of operator=…`, `dereference operator=…`, `new`,
   `pointer-conversion` with its destination annotation and operand, and the
   `free` statement.
10. `src/semantic/annotations.rs`, `expressions.rs`, `statements.rs`,
    `namespaces.rs`, and `src/ir/lower.rs` — add the guard and the traversal
    arms described under Decisions.

## Test plan

- `src/frontend/tests/parser.rs`, `pointer_syntax_snapshot` with a new
  snapshot. Cover:
  - annotations `*int`, `nullable *Node`, `**int`, `*nullable *int`,
    `*[3]int`, `[2]*int`, and `nullable *m::T`, on bindings, parameters,
    results, and struct fields;
  - `null` as an initializer, argument, field value, and comparison operand;
  - `&x`, `&a[i].f`, `*p`, `**pp`, and `*p.x`, whose operand is the field
    selection;
  - `(*p).x`;
  - `new(1)`, `new(Point { x = 1, y = 2 },)`, and `new(new(0))`;
  - `*Node(cursor)`, `**Node(p)`, `*int(p)`, `*[2]int(p)`, `*m::T(p)`, and
    `*Node(c).value`;
  - `*(f(x))` as a dereference of a call;
  - `free(p);` and `free(p,);`;
  - the assignments `*p = 1;`, `**pp += 2;`, and `*p.x = 3;`, and a `for`
    loop whose initializer and post clause assign through `*p`;
  - `a * *p` and `a & &b` beside binary `*` and `&`.
- Exact-span diagnostics:
  - `nullable int` → `expected `*` after `nullable`` at `int`;
  - `*void` → `expected a type` at `void`;
  - `new` with no `(` → reserved word at `new`;
  - `new()` → `new requires one argument`;
  - `new(1, 2)` → `new takes one argument`;
  - the same three errors for `free`;
  - `*Node(a, b)` and `*Node(p,)` → `expected `)`` at the `,`, because a
    conversion takes one operand and no trailing comma.
- `src/frontend/tests/lexer.rs` — add `null`, `new`, `free`, and `nullable` to
  both reserved lists, except `null` in the expression-position list, where it
  is a literal. Remove `free` and `null` from the not-reserved list.
- The nesting limit with deeply nested `*`, `&`, pointer annotations, and
  `new(…)`.
- `src/semantic/tests/` — a pointer annotation, each new expression form, `*p`
  and `(*p).x` as assignment targets, and `free(p);` each report
  `pointers are not yet implemented` at the span the guard names.
  Programs without pointer syntax keep their existing results.
