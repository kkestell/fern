# Check returns and function reachability

## Sources

- `docs/spec.md#function-return` — return forms, result typing, and the
  structural end-of-body reachability rule
- `docs/spec.md#loop-control-and-labels` — which loop a `break` targets
- `eng/roadmap.md#parameters-calls-and-return-values` — task boundary
- `src/semantic.rs` — body checking, scopes, loop tracking, and the
  annotated-initializer conversion path used for arguments

## Goal

Replace the temporary rejection of `return` in `check_statement` with real
checking of both return forms against each function's declared result, and add
the structural check that rejects a value-returning function whose body end is
reachable. This stays a semantic-only slice: Fern IR still lowers only `main`'s
body, so a program that calls a function or returns a value still has no
lowering path.

## Implementation

- `src/semantic.rs` — thread the enclosing function's declared result
  (`Option<Type>` from its `FunctionSignature`) through `check_body` and
  `check_statement` as a parameter, alongside `scopes` and `loops`. The
  per-function loop in `check` already has the signature available.

- `src/semantic.rs` — check `StatementKind::Return`. With a `void` result,
  `return;` is accepted and `return e;` is rejected at the expression's span.
  With a value result, `return;` is rejected at the statement span and
  `return e;` checks the expression with the declared result as its
  destination type, so untyped constants are contextualized exactly as an
  annotated initializer is. Insert the checked expression into
  `checked.expressions`.

- `src/semantic.rs` — add a structural reachability check that runs after a
  function body is checked, only when the signature has a value result. It
  reads syntax alone, needs no type facts, and reports at the function's
  `name_span` when the end of the body is reachable.

- `src/semantic.rs` — implement the check as two small recursive helpers over
  statement lists:
  - a body terminates when any statement in it terminates;
  - `return` and `exit` terminate; a nested block terminates when its body
    does; an `if` chain terminates only when it ends in an `else` and every
    branch terminates; `for` with `ForHeader::Infinite` terminates when no
    reachable `break` targets it; `ForHeader::Condition` and
    `ForHeader::ThreeClause` never terminate;
  - the break search walks a loop body in source order, stops at the first
    terminating statement, descends into blocks, `if` branches, and nested
    loop bodies, and counts a `break` whose label names this loop, or an
    unlabeled `break` that is not inside a nested loop.

- `src/ir.rs` — correct the two stale messages that say semantic checking
  rejects calls and returns. Lowering support arrives with the next roadmap
  task, which replaces the entry-only IR shape; do not add a second lowering
  path here.

## Tests

- `src/semantic.rs` — accept `return;` and end-of-body fallthrough in a `void`
  function, `return e;` with a matching value type, an untyped constant
  contextualized to the declared result, a `bool` result, and a `return` in a
  branch of an `if` chain.

- `src/semantic.rs` — accept value functions whose end is unreachable through:
  an `if`/`else` where both branches return; a nested block ending in `return`;
  `exit(0);` as the last statement; a `for { ... }` whose only `break` is
  inside a nested loop that the break does not target; and `for { ... }` with
  no `break` at all.

- `src/semantic.rs` — reject: `return e;` in a `void` function; `return;` in a
  value function; a returned value whose type does not match the result; an
  untyped constant that does not fit; an `if` chain with no `else` as the last
  statement of a value function; `for c { ... }` and the three-clause form as
  the last statement; a `for { ... }` containing a reachable unlabeled `break`;
  and a nested loop whose `break :outer;` targets the enclosing infinite loop.

- `src/semantic.rs` — prove that statements after a `return` are still checked
  by asserting an unknown-binding diagnostic from a statement that follows one.

## Decisions

- Only `return` and `exit` end a path. The specification lists them
  exhaustively, so statements after `break;` or `continue;` still count as
  reachable for this check, and a `break` that follows a `return` in the same
  block does not keep its loop from terminating the path.

- Constant conditions are ignored, as the specification requires: the check
  looks only at statement structure, so it runs on syntax and does not consult
  `CheckedExpression::constant`.

- The reachability check is a separate pass rather than state carried through
  `check_statement`, because it must look ahead across a whole body and reads
  no information that body checking produces.
