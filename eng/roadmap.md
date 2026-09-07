# Fern Roadmap

Fern is a Rust compiler. The initial stack is Logos for lexing, la-arena for
syntax storage, lasso for identifier interning, Ariadne for diagnostic
rendering, and Insta for syntax and IR snapshots. The parser is handwritten, and
native compilation uses a Fern-owned IR and QBE.

Every milestone compares its implemented behavior against the corresponding
sections of [the specification](../docs/spec.md). The gates listed under each
task are the behavior that task must prove on top of that.

## Current milestone: Compile the executable integer subset

Build a Rust compiler that accepts the smallest useful Fern program, checks
initialized `int` bindings, and produces a native executable.

```text
fn main() -> void {
    const status = 42;
    exit(status);
}
```

The resulting program exits with status 42. This milestone includes only one
parameterless `main`, initialized `var` and `const` bindings, `int`, integer
literals, binding references, and `exit`. Assignment, nested statement blocks,
other integer types, operators, casts, user functions, and other value types
remain outside this milestone.

### Establish the Rust compiler

**Build**

- Create the Rust compiler and its source-loading, diagnostic, frontend,
  semantic, IR, backend, and driver boundaries.
- Add Logos, la-arena, lasso, Ariadne, and Insta for their stated roles.
- Accept valid UTF-8 Fern source and reject invalid source text.

**Gates**

- An empty `main` compiles to a native executable that exits with status 0.
- Invalid UTF-8, unreadable input, invalid compiler input, and unavailable
  backend tools fail without producing a successful executable.
- Diagnostics identify the source and relevant byte span through Ariadne.

### Parse the initial syntax

**Build**

- Recognize parameterless function declarations, initialized `var` and `const`
  declarations with optional `int` annotations, integer literals, binding
  references, and `exit` calls.
- Preserve source spans and literal spellings in an arena-backed syntax tree.

**Gates**

- Valid programs cover whitespace, nested comments, every integer base, leading
  zeroes, optional `i` suffixes, and trailing call commas.
- Invalid tokens, malformed literals, unterminated comments, and malformed
  declarations or calls are rejected at their source spans.
- Insta snapshots cover representative syntax trees and remain deterministic.

### Check initialized integer bindings

**Build**

- Require exactly one parameterless `main` returning `void` and reject other
  top-level functions in this subset.
- Resolve interned binding names in source order and annotate syntax with
  binding identity and concrete types.
- Check inferred and explicit `int` initializers, references, literal suffixes,
  and literal ranges.

**Gates**

- Same-block shadowing and initializer visibility follow
  [Scope and shadowing](../docs/spec.md#scope-and-shadowing).
- Missing or duplicate entry points, additional functions, unresolved names,
  unsupported suffixes, and out-of-range literals are rejected.
- Checking continues after an `exit` call so later source errors are not hidden
  by runtime reachability.

### Lower and execute checked programs

**Build**

- Lower checked functions, constants, and binding copies into an owned, verified
  Fern IR.
- Emit QBE from verified IR and produce a native executable.

**Gates**

- Native execution covers literal initialization, copying bindings, and exit
  through a binding.
- Exit statuses cover normal completion and the modulo-256 behavior in
  [Process exit](../docs/spec.md#process-exit).
- An `exit` prevents later statements from executing while later statements
  still participate in static checking.
- Insta snapshots cover deterministic Fern IR, and malformed internal IR is
  rejected before backend emission.

## Next milestone: Assignment and nested scopes

Add assignment to mutable local bindings and nested statement blocks. Native
execution must observe reassignment, enclosing-scope access, nested shadowing,
restoration of outer bindings, and rejection of immutable or out-of-scope
targets.

## Later work

Each item is its own milestone, implemented in dependency order. Each begins by
confirming that `docs/spec.md` fully specifies its behavior, then extends the
frontend, semantic checking, Fern IR, backend, diagnostics, and tests together.

- Support every fixed-width integer type plus `int` and `uint`, including
  annotations, suffixes, range checking, inference, and specified implicit
  conversions.
- Add integer expressions after the specification settles operators, precedence,
  evaluation order, flexible operands, overflow, division by zero, and shifts.
- Add branches and loops after their control-flow rules are specified.
- Add parameters, calls, and return values after the remaining function rules
  are specified.

Do not add incremental queries, lossless syntax, control-flow infrastructure, or
alternate backends before a milestone needs them.
