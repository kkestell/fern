# Check Initialized Integer Bindings

## Sources

- [Roadmap: Check initialized integer bindings](../roadmap.md#check-initialized-integer-bindings)
  — scope and completion gates.
- [Specification: Integer types](../../docs/spec.md#integer-types) and
  [Flexible Types](../../docs/spec.md#flexible-types) — concrete typing and
  range requirements.
- [Specification: Functions](../../docs/spec.md#functions),
  [Variable declarations](../../docs/spec.md#variable-declarations),
  [Scope and shadowing](../../docs/spec.md#scope-and-shadowing), and
  [Process exit](../../docs/spec.md#process-exit) — semantic contracts.
- `src/frontend.rs` — syntax arenas, interned names, source spans, and integer
  spelling validation.

## Goal

Replace the semantic rejection of all nonempty bodies with checked binding and
expression annotations. Keep the existing empty-program compilation path working
and reject unsupported lowering explicitly.

## Implementation

- `src/semantic.rs` — extend `CheckedEntry` to retain the selected function and
  access to its syntax, with semantic tables keyed by syntax indices. Record a
  concrete type for each expression, parsed integer values for literals, and
  binding identities for references and declarations. Store binding metadata in
  an arena with distinct IDs for distinct declarations, including shadowed
  names.
- `src/semantic.rs` — retain entry-point checks, then walk statements in source
  order. Use an interned-name map for the current block. Check each initializer
  before inserting its new binding. Resolve exit arguments through the same
  expression checker and continue checking subsequent statements. Keep the
  existing first-error diagnostic convention and report the offending reference
  or literal span.
- `src/frontend.rs` — factor integer spelling decomposition into a shared helper
  returning base, digit text, and suffix, so parsing and checking agree. Keep
  spelling validation in the frontend and value conversion in semantic checking.
  Remove temporary dead-code allowances where the new checker consumes nodes.
- `src/semantic.rs` — reject suffixes outside this slice before converting the
  magnitude. Parse with checked arithmetic into the chosen integer
  representation; arbitrarily long values must produce a range diagnostic
  without panicking.
- `src/ir.rs`, `src/lib.rs` — make lowering fallible and reject a checked
  nonempty body at its first statement span until lowering is implemented. Run
  the complete semantic check before this guard, so source errors take
  precedence over the unsupported-lowering diagnostic. Do not let the current
  zero-status lowering silently discard checked statements.
- `README.md` — document the compiler's chosen `int` width and distinguish
  semantic checking support from executable compilation support.
- `tests/compiler.rs` — replace expectations for the old semantic body rejection
  with lowering rejection for valid bodies and specific diagnostics for invalid
  bodies. Retain output creation and replacement protections.

## Tests

- Check inferred and annotated bindings, both mutabilities, reference copies,
  and literal or reference exit arguments. Assert concrete types and resolved
  IDs.
- Prove `const x = 1; var x = x; exit(x);` resolves the initializer to the first
  binding and the exit to the second. Cover repeated shadowing, self-reference
  without a predecessor, forward references, and unknown names.
- Cover zero, leading zeroes, the maximum supported value, and maximum plus one
  in every base, with and without `i`. Include very long overflowing digits and
  long leading-zero spellings that still fit. Reject every other recognized
  suffix.
- Check unresolved names and overflowing literals after an exit. Retain missing,
  duplicate, and additional-function coverage and parser rejection of invalid
  entry signatures.
- Assert exact diagnostic spans, including after Unicode comments. Prove valid
  checked nonempty bodies cannot reach backend emission, while empty main still
  compiles and executes successfully.

## Decisions

- Use a signed 32-bit `int`, independent of host pointer width, represented by
  Rust `i32`. Record that implementation choice in `README.md`.
- Borrow the immutable syntax from `CheckedEntry` and attach semantic facts in
  side tables; do not duplicate the syntax tree or introduce another IR here.
