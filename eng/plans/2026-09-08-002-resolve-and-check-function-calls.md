# Resolve and check function calls

## Sources

- `docs/spec.md#functions` — function namespace, parameters, and entry-point
  requirements
- `docs/spec.md#calls` — direct-call resolution, argument rules, value contexts,
  and constant-expression restriction
- `docs/spec.md#scope-and-shadowing` — lexical binding precedence over function
  names
- `eng/roadmap.md#parameters-calls-and-return-values` — semantic call-checking
  task boundary
- `src/semantic.rs` — module collection, lexical scopes, expression typing, and
  binding identity

## Goal

After the parsed-signature slice, collect every function signature before any
body is checked and type-check direct calls and parameter bindings. This remains
a semantic-only slice: returns, reachability, Fern IR calls, and native call
execution are still unfinished.

## Implementation

- `src/semantic.rs` — replace the current rejection of parameters and value
  results with a module-wide function-signature collection pass. Keep one
  namespace for functions and module bindings, preserve existing duplicate-name
  diagnostics, reject duplicate parameter names, and validate that `main` has
  no parameters and a `void` result. Retain checked function signatures and
  parameter binding identities in `CheckedProgram` for the later return and IR
  slices.

- `src/semantic.rs` — check each function body with a fresh immutable parameter
  scope outside its body scope. Parameters use their declared types, have no
  constant value, may be shadowed by local declarations, and flow through the
  existing assignment-target check so assignment to one is rejected.

- `src/semantic.rs` — add checked-call data to `ExpressionValue` and a shared
  call-checking helper. Resolve a call first through visible bindings, which
  makes a local or module binding of the same name a non-function target; only
  an unshadowed module function name is callable. Record the resolved function
  identity so later lowering does not redo name lookup.

- `src/semantic.rs` — check exact arity and each argument against its matching
  parameter type using the existing annotated-initializer conversion path, in
  source order. Calls have no constant value. Permit either result kind in a
  call statement, but require a value result for a call expression and diagnose
  a `void` call used where a value is needed. Continue to reject functions as
  ordinary binding references and diagnose calls in module-level initializers
  as non-constant expressions.

- `src/ir.rs` — retain the existing entry-only lowering boundary and make its
  unsupported call paths explicit if the semantic representation makes an
  accidental lowering path reachable. Do not introduce a second function or
  call IR shape in this task.

- `tests/compiler.rs` — update parser-era source-failure expectations whose
  rejection now belongs to entry-signature validation rather than the removed
  temporary “not supported yet” diagnostic.

## Tests

- `src/semantic.rs` — cover forward calls, self-recursion, and mutual recursion
  with value and `void` signatures, including calls nested in existing
  expressions and discarded value results.

- `src/semantic.rs` — prove parameter typing, untyped-constant contextualization,
  parameter immutability, local parameter shadowing, and left-to-right argument
  checking preserve the existing binding and expression facts.

- `src/semantic.rs` — assert target-span diagnostics for unknown and shadowed
  call targets, duplicate parameter names, wrong arity, incompatible arguments,
  `void` calls in value contexts, invalid `main` signatures, and calls in
  module-level constant initializers.

## Decisions

- Model a function result as `Option<Type>` in checked signatures: `None` is
  `void`, which is not a value type. A call statement uses the shared checker
  without constructing a value expression for `void`; an expression call must
  have `Some(Type)`.

- Function declarations stay outside lexical binding scopes. This preserves the
  specified distinction that function names are not values while allowing the
  call resolver to give a visible binding priority over a same-named function.
