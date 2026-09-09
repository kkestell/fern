# Parse function signatures, calls, and returns

## Sources

- `docs/spec.md#functions` — function signatures, parameter requirements, and
  the entry-point shape
- `docs/spec.md#calls` — direct-call syntax and argument-list rules
- `docs/spec.md#function-return` — the two return forms
- `eng/roadmap.md#parameters-calls-and-return-values` — first task and its
  parser coverage boundary
- `src/frontend.rs` — lexer, syntax arenas, parser, and syntax snapshots

## Goal

Extend the frontend from void no-parameter functions to the full syntax needed
for this milestone: typed parameters, value or void results, direct calls, and
returns. This is a parser-only slice. The existing semantic checker, entry-only
IR, and native emitter do not yet resolve, type-check, lower, or execute these
new forms.

## Implementation

- `src/frontend.rs` — add `return` to the token and reserved-word sets. Extend
  `Function` with parameter declarations and an explicit result representation
  that can distinguish `void` from a value `TypeAnnotation`; retain names and
  spans for later semantic diagnostics.

- `src/frontend.rs` — parse comma-separated, explicitly typed parameter lists,
  accepting empty lists and exactly one trailing comma. Parse `void` or a value
  type after `->`; keep `void` out of ordinary binding and parameter type
  annotations.

- `src/frontend.rs` — add syntax forms for a direct call expression, a call
  statement, and a return statement with an optional expression. Parse calls
  from a name followed by `(` so nested calls compose with the existing
  precedence parser. Share one argument-list parser between expression and
  statement handling, preserving target, argument, delimiter, and whole-node
  spans as the existing AST does for conversions and control flow.

- `src/frontend.rs` — route a name-led statement to assignment only when an
  assignment operator follows it, and to a call statement when `(` follows it;
  retain the current diagnostic for any other name-led statement. Continue to
  require `;` for calls and returns.

- `src/frontend.rs` — update the test projection so signatures, parameters,
  calls, and returns appear in readable snapshots. Update exhaustive statement
  and expression matches made non-exhaustive by the added syntax variants.

## Tests

- Add a frontend snapshot covering typed integer and boolean parameters,
  `void` and value results, empty and trailing-comma parameter and argument
  lists, calls as statements and nested expressions, and both return forms.

- Add parser diagnostics with exact offending spans for missing parameter names,
  colons, types, commas, parentheses, result types, call arguments, and return
  expressions or semicolons.

- Prove that `return` cannot be used as an identifier, calls can nest within
  existing arithmetic and logical expressions, and name-led assignments retain
  their current parse path.

## Decisions

- Model function results separately from `Type`, since `void` is a function
  result but not a value type. Reuse `TypeAnnotation` for value result and
  parameter annotations so later phases receive the same type-plus-span data as
  bindings.

- Keep calls as syntax without parser-level target validation. Function lookup,
  arity, value-context restrictions, and entry-point validation belong to the
  following semantic tasks.
