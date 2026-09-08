# Fern Roadmap

Fern is a compiler written in Rust. The initial stack is Logos for lexing,
la-arena for syntax storage, lasso for identifier interning, Ariadne for
diagnostic rendering, and Insta for syntax and IR snapshots. The parser is
handwritten, and native compilation uses a Fern-owned IR and QBE.

Every milestone implements the corresponding sections of
[the specification](../docs/spec.md). The milestone gates below prove its
integrated outcome. Task checkboxes record progress; milestones remain in
implementation order.

Do not add incremental queries, lossless syntax, control-flow infrastructure, or
alternate backends before a milestone needs them.

---

## Executable integer subset

Build a native compiler for parameterless `main`, initialized `int` bindings,
binding references, and process exit against
[Functions](../docs/spec.md#functions),
[Variable declarations](../docs/spec.md#variable-declarations), and
[Process exit](../docs/spec.md#process-exit).

### Examples

- [Empty entry](../examples/empty.fern), expected exit status 0.
- [Integer literals](../examples/integer_literals.fern), expected exit status 42.
- [Shadowing](../examples/shadowing.fern), expected exit status 42.

### Tasks

- [x] **Establish the Rust compiler**

  - Load UTF-8 source, report source diagnostics, and establish the frontend,
    semantic, IR, backend, and driver boundaries.
  - Compile an empty entry point through QBE and the host C toolchain.

- [x] **Parse and check entry syntax**

  - Parse the entry point, initialized bindings, integer literals, references,
    and `exit` while preserving source spans.
  - Resolve bindings in source order and reject invalid entry points, names,
    literals, declarations, and calls.

- [x] **Lower and execute integer programs**

  - Lower checked constants and binding copies into verified Fern IR.
  - Preserve shadowing, source checking after `exit`, and observable exit
    statuses through native execution.

### Completion gates

- The three examples compile and produce their expected statuses.
- Invalid source text, syntax, names, literals, and entry points produce source
  diagnostics without publishing an executable.
- Binding identity, initializer visibility, same-block shadowing, and checking
  after `exit` agree across semantic checking, IR snapshots, and native tests.
- File and native-tool failures preserve an existing output, and malformed IR
  is rejected before emission.

---

## Assignment and nested scopes

Support assignment to mutable local bindings and nested statement blocks against
[Assignment](../docs/spec.md#assignment),
[Scope and shadowing](../docs/spec.md#scope-and-shadowing), and
[Immutability](../docs/spec.md#immutability). Field, element, and pointer
assignment remain outside this milestone.

### Example

See [the example](../examples/assignment_and_scopes.fern).

Expected exit status: 42.

### Tasks

- [x] **Parse local assignments and nested statement blocks**

  - Represent assignment targets, values, and nested bodies with source spans.
  - Cover empty and nested blocks, mixed statements, and malformed assignments
    and block delimiters in parser tests and snapshots.

- [x] **Check assignment targets and lexical scopes**

  - Resolve targets and values by binding identity, with diagnostics for unknown
    targets and assignment to immutable bindings.
  - Check enclosing-scope access, initializer visibility, same-block and nested
    shadowing, and restoration of outer bindings.
  - Continue checking statements after exits, including inside nested blocks.

- [x] **Lower and execute assignments and nested scopes**

  - Preserve reassignment results and saved copies through IR and native
    execution.
  - Preserve outer assignments across nested scopes and stop execution at nested
    exits.
  - Cover the integrated behavior with IR snapshots, native execution tests, and
    the milestone example.

### Completion gates

- Parser snapshots cover assignments and nested blocks; malformed statements
  produce source diagnostics.
- Unknown and immutable assignment targets are rejected, including after exits.
- Native execution preserves saved copies, lexical shadowing, and enclosing
  assignments when nested blocks end.
- Nested exits terminate execution while later source remains checked.
- Deterministic IR snapshots and native tests cover the milestone example and
  its scope boundaries.

---

## Integer types

Support every fixed-width integer type plus `int` and `uint`, including
annotations, range checking, inference, and contextual literals. Extend
initialization, assignment, and `exit` argument checking and native execution
against [Integer types](../docs/spec.md#integer-types),
[Untyped constants](../docs/spec.md#untyped-constants),
[Assignment](../docs/spec.md#assignment), and
[Process exit](../docs/spec.md#process-exit). Operators, explicit conversions,
`size`, `uintptr`, and rune values remain outside this milestone.

### Example

See [the example](../examples/integer_types.fern).

Expected exit status: 42.

### Tasks

- [x] **Check integer types**

  - Check literal ranges, contextual typing, and inferred binding types against
    the specification sections linked above.
  - Apply the host pointer width consistently to `int`, `uint`, and range
    checks.
  - Require matching concrete types for initialization, assignment, binding
    copies, and `exit` arguments, while allowing contextual untyped literals.
  - Cover typing across binding copies, shadowing, nested scopes, and source
    after exits. Lowering and native execution remain unfinished at this
    boundary.

- [x] **Lower typed integers**

  - Carry checked integer values through Fern IR.
  - Preserve full-width values through initialization, assignment, binding
    copies, shadowing, and nested exits.
  - Extend IR verification and snapshots to cover value types, ranges, copies,
    and exits. Reject malformed typed values before emission. Native emission
    belongs to the following task.
  - Reject programs the backend cannot emit without replacing an output file.

- [x] **Execute typed integer programs**

  - Extend native emission to support every checked integer type.
  - Verify full-width signed and unsigned values at the backend boundary,
    alongside native execution of source programs.
  - Add execution coverage and move the milestone example into
    `examples/integer_types.fern`, then validate the integrated milestone
    against the completion gates below.

### Completion gates

- Annotations, inference, and contextual literals cover every integer type in
  scope, with range boundaries exercised across literal bases.
- Source tests cover zero, positive maxima, and out-of-range magnitudes;
  negative signed boundaries are covered through internal IR and backend tests
  until integer expressions add source syntax for negative values.
- Initialization, assignment, and exit checking require identical concrete
  types, including for distinct equal-width types.
- Coverage distinguishes contextual literals from binding references.
- Large unsigned values retain their full range through checking, IR, and native
  emission. Coverage checks high-bit preservation beyond observable exit bytes.
- Invalid literals and type combinations produce source diagnostics, including
  in nested scopes and after exits.
- Native execution covers reassignment, saved copies, shadowing, and nested
  exits.
- Syntax and IR snapshots are deterministic and generated IR passes
  verification.

---

## Integer conversions

Support integer literal validation, reserved names, and explicit integer
conversions across parsing, checking, Fern IR, and native execution. The
contracts are
[Integer literals](../docs/spec.md#integer-literals),
[Keywords](../docs/spec.md#keywords),
[Integer types](../docs/spec.md#integer-types),
[Integer constant expressions](../docs/spec.md#integer-constant-expressions),
[Integer conversions](../docs/spec.md#integer-conversions),
[Assignment](../docs/spec.md#assignment), and
[Process exit](../docs/spec.md#process-exit).

This milestone covers literals, references, and nested conversion expressions.
Unary and binary operators, comparisons, compound assignments, modules, and
collection or pointer operations remain outside its scope. Constant evaluation
here covers literals, conversions, and constant binding references; the following
milestone extends it to arithmetic and bitwise expressions. Native execution
continues to target the development host; this work does not add cross-compilation.

### Example

See [the example](../examples/integer_conversions.fern).

Expected exit status: 42.

### Tasks

- [x] **Validate literal syntax and reserved names**

  - Accept the specified literal bases, reject suffixes, and preserve useful
    source spans for malformed integer tokens.
  - Apply the lexical contract consistently to reserved-name checks.
  - Cover declarations, assignments, blocks, and exits in parser fixtures and
    snapshots.

- [x] **Check integer compatibility**

  - Apply the integer type contract consistently in semantic checking, Fern IR
    verification, and native emission for the supported host target.
  - Require explicit conversions between concrete integer types in
    initialization, assignment, binding copies, and exit arguments.
  - Cover contextual literals, type identity, range checks, native values, and
    exit statuses.

- [x] **Parse and check explicit integer conversions**

  - Accept checked and truncating conversion expressions, including nesting,
    wherever integer expressions are accepted.
  - Evaluate constant conversions using the specified precision and range rules.
    Distinguish constant binding references from runtime binding references
    through copies and shadowing.
  - Check runtime conversion operands and destination types, retaining the
    information needed for lowering and source diagnostics.
  - Cover malformed conversion syntax, large literal inputs, constant failures,
    and source after exits. Native execution is completed in the next task.

- [x] **Lower and execute explicit integer conversions**

  - Carry checked and truncating conversions through Fern IR and its verifier,
    and emit their specified values and runtime failures.
  - Cover all integer type pairs, full-width signed and unsigned boundaries,
    narrowing, widening, equal-width distinct types, and nested conversions.
  - Preserve assignment, saved-copy, shadowing, and exit behavior, including
    termination before subsequent runtime work after a failed conversion.

- [x] **Add examples and validate integer conversions**

  - Keep source examples, README descriptions, tests, and snapshots consistent
    with the linked contracts.
  - Create the example above and link to the file.
  - Validate the integrated compiler against the completion gates and review
    the milestone once for correctness and simplicity.

### Completion gates

- The parser accepts the specified literal bases and conversion forms, rejects
  suffixes, and agrees with the specification's reserved names.
- All ten integer types agree across checking, IR verification, and native
  execution. Width-sensitive checks cover both specified pointer widths without
  claiming native execution on targets the toolchain does not support.
- Contextual literal typing and every source/destination type pair are covered
  in initialization, assignment, and exit checking, including equal-width
  distinct types.
- Constant conversion coverage includes the specified precision, values beyond
  the machine integer range, nested conversions, constant binding references,
  runtime-initialized immutable bindings, and unreachable source.
- Native and internal full-width checks cover checked-conversion success and
  failure and truncating-conversion results at signed and unsigned boundaries.
  Runtime failures identify the operation and occur in every build configuration.
- Scope, mutability, copies, and process exit retain their specified behavior.
  Host-width exit arguments produce the specified externally reported statuses.
- Invalid source and runtime conversion failures remain distinct. Failed
  compilation preserves existing output files, and malformed IR is rejected
  before emission.
- The examples compile and produce their documented results. Tests and snapshots
  agree with the linked contracts, and the integrated milestone receives one
  correctness and simplicity review.

---

## Integer expressions

Support unary and binary integer expressions in local initializers, assignments,
and `exit` arguments across parsing, checking, Fern IR, and native execution.
The behavior contracts are [Integer operators](../docs/spec.md#integer-operators),
[Integer arithmetic](../docs/spec.md#integer-arithmetic),
[Integer shifts](../docs/spec.md#integer-shifts),
[Untyped constants](../docs/spec.md#untyped-constants),
[Integer constant expressions](../docs/spec.md#integer-constant-expressions),
and [Integer conversions](../docs/spec.md#integer-conversions).

The scope is unary and binary arithmetic, wrapping, bitwise, and shift
operations on the ten integer types. It builds on the preceding milestone's
literal and conversion behavior. Comparisons and compound assignments remain
outside this milestone, as do branches, loops, modules, and collection or
pointer operations.

### Example

See [the example](../examples/integer_expressions.fern).

Expected exit status: 42.

### Tasks

- [x] **Parse integer operators and grouping**

  - Accept the specified unary and binary operators and parentheses wherever
    integer expressions are accepted.
  - Preserve operator and operand source locations for later diagnostics.
  - Cover precedence, associativity, unary chains, negative literal syntax,
    comments adjacent to operators, and malformed expressions in parser tests
    and snapshots.

- [x] **Check integer expression types**

  - Check untyped and concrete operand combinations, unary operations, shift
    operands, result types, and destination conversions against the linked
    contracts.
  - Support source-level signed minimum values and preserve contextual and
    explicit-conversion range checks.
  - Preserve binding identity and concrete binding types through expressions,
    assignments, and lexical scopes, including source after exits.
  - Extend constant-expression classification from literals, conversions, and
    binding references to the new operators; evaluation follows in the next task.

- [x] **Evaluate integer constant expressions**

  - Evaluate the required constant expressions and subexpressions, preserving
    untyped values until their specified conversion boundary.
  - Diagnose invalid literal and result conversions, concrete arithmetic
    overflow, invalid divisors, and negative shift counts at source locations.
  - Cover the distinction between untyped expressions, explicitly typed
    constant operations, and binding references, including constant expressions
    nested within runtime expressions and unreachable source.
  - Distinguish constant shifts from runtime shifts, and ordinary arithmetic
    from explicitly wrapping operations during constant evaluation. Runtime lowering remains unfinished at this boundary.

- [x] **Check shift and wrapping boundaries**

  - Apply the precedence and typed wrapping-operand contracts across parsing
    and checking, with focused boundary coverage.
  - Make typed constant shifts agree with runtime shifts while retaining exact
    untyped constant shifts and their final range checks.
  - Preserve constant-expression classification, source diagnostics, and
    unreachable-source checking across these boundaries.

- [x] **Lower checked integer operations**

  - Carry checked operations, conversions, and constant results through Fern IR
    while preserving operand evaluation order and required runtime failures.
  - Extend IR verification and snapshots to cover the supported operations,
    operand and result types, and failure behavior. Malformed IR must be
    rejected before native emission.
  - Reject unsupported emission without replacing an output file while native
    support is incomplete.

- [x] **Execute integer expressions and failures**

  - Emit the integer operations and runtime diagnostics for all types in scope,
    preserving Fern's results independently of host instruction behavior.
  - Cover operation results at full width alongside native source execution,
    including arithmetic boundaries, signed division, and shift counts.
  - Verify first-failure order and preserve expression behavior through
    assignment, saved copies, shadowing, and nested exits.
  - Create the example file above, replace its snippet with a link, and validate
    the integrated milestone against its completion gates.

### Completion gates

- Parser coverage distinguishes every precedence level and associativity rule;
  malformed expressions produce source diagnostics.
- Type coverage includes all integer types in scope, mixed widths, equal-width
  distinct types, signedness mismatches, contextual untyped operands, and
  independent shift-count types.
- Constant evaluation covers untyped intermediate values, final range checks,
  explicitly typed intermediate overflow, constant subexpressions, and source
  after exits. Coverage distinguishes constant binding references from
  runtime values while checking identical initializers consistently.
- Source and native tests cover signed minima and maxima, unsigned maxima,
  trapping and wrapping arithmetic, negation, division and remainder signs, and
  minimum signed values divided by or reduced modulo `-1`.
- Bitwise and shift coverage includes high-bit preservation, signed right
  shifts, discarded bits, zero counts, width boundaries, very large counts,
  negative counts, and untyped versus concrete constant operations.
- Compile-time diagnostics and runtime termination cover invalid divisors and
  counts according to the specification. Native tests observe which failing
  operand executes first.
- Constant folding preserves the specified distinction between required
  compile-time evaluation and runtime operations involving binding references.
- Deterministic syntax and IR snapshots, IR verification, native tests, and the
  example agree with the linked contracts, and the completed milestone receives
  one correctness and simplicity review.

---

## Harden integer compilation

Compile adversarial integer expressions within documented resource bounds while
following the integer contracts linked by the preceding milestone. Keep the
compiler paths that enforce those bounds direct and shared.

### Example

See [the example](../examples/hardened_integers.fern).

Expected exit status: 42.

### Tasks

- [x] **Bound and correct semantic constant handling**

  - Type non-constant untyped shift counts according to the specification and
    diagnose out-of-range counts in reachable and unreachable source.
  - Bound literal parsing and untyped constant folding with documented compiler
    limits, including checks before predictably large allocations.
  - Store each folded value once and keep constant lowering direct.

- [x] **Bound source diagnostic rendering**

  - Render bounded excerpts for very long source lines while preserving the
    source path and useful byte locations.
  - Cover oversized tokens and long-line errors through the compiler boundary.

- [x] **Compact and share backend trap emission**

  - Reuse one indexed source during emission and one path for operation and
    conversion diagnostics.
  - Emit compact QBE message data and share conditional trap bodies without
    changing first-failure behavior.

- [x] **Consolidate integer compiler metadata and lowering**

  - Share integer type ranges, integer type iteration, operator spelling, and
    reserved-name classification.
  - Remove redundant lowering matches and dead QBE copies where focused native
    checks prove them unnecessary.

- [x] **Document compiler limits and validate integer compilation**

  - Document compiler limits, keep the repository map and test comments
    accurate, and keep behavior claims in their owning document.
  - Add the milestone example and validate the integrated compiler and its
    resource bounds.

### Completion gates

- Reported nested shift counts produce source diagnostics instead of panics or
  internal errors, including after `exit`.
- Oversized literals, folded constants, and diagnostics stay within documented
  resource bounds and fail with source diagnostics.
- Generated trap data grows proportionally to message text and every trap family
  retains its source location and first-failure behavior.
- Shared type and operator metadata agrees across parsing, checking, IR, backend
  emission, and tests without broadening the supported language.
- The example exits with status 42, snapshots are deterministic, and all compiler
  checks pass.

---

## Operator spelling and precedence

Support the operator spellings and precedence defined by
[Precedence, associativity, and evaluation order](../docs/spec.md#precedence-associativity-and-evaluation-order),
[Integer operators](../docs/spec.md#integer-operators), and
[Bitwise operations](../docs/spec.md#bitwise-operations).

The scope is lexing, parsing, and the operator spellings and precedence used by
checking, IR, and backend emission. Comparisons appear in the precedence table,
but their implementation belongs to the control-flow milestone.

### Example

See [the example](../examples/wrapping_operators.fern).

Expected exit status: 42.

### Tasks

- [x] **Lex and parse operator spellings**

  - Accept `+%`, `-%`, and `*%` as binary operators and `-%` as a unary
    operator.
  - Preserve operator spans and cover programs that mix `&` with a following
    unary `-` or `^` in parser tests and snapshots.

- [x] **Apply operator precedence levels**

  - Parse `* / % *% << >> &` at the highest binary level, `+ - +% -% | ^` at the
    next, and preserve left associativity and grouping.
  - Cover each level and its boundaries in parser tests and snapshots,
    including `a + b << c`, `a | b + c`, and `a & ^b`.

- [x] **Check and execute the operators**

  - Use the specified spellings in diagnostics and shared operator metadata.
  - Cover the operators in both examples and validate the integrated compiler.

### Completion gates

- The operator spellings execute with the values their contracts specify.
- Parser snapshots distinguish every precedence level, including `&` followed
  by unary `-` or `^`.
- Both examples compile and exit 42, and IR snapshots stay deterministic.

---

## Module-level declarations

Support `var` and `const` declarations at the top level of a source file against
[Module-level declarations](../docs/spec.md#module-level-declarations),
[Scope and shadowing](../docs/spec.md#scope-and-shadowing),
[Integer constant expressions](../docs/spec.md#integer-constant-expressions),
and [Assignment](../docs/spec.md#assignment).

The scope is file-scope bindings whose initializers are constant expressions,
their visibility independent of declaration order, and their initialization
before `main` runs. Multiple source files, imports, and non-constant
initializers remain outside this milestone.

### Example

```fern
var counter = start;
const start: int = 40;
const step = 2;

fn main() -> void {
    counter = counter + step;
    exit(counter); // reports 42
}
```

Expected exit status: 42. Future file:
`examples/module_level_declarations.fern`.

### Tasks

- [ ] **Parse top-level bindings**

  - Accept `var` and `const` declarations interleaved with `fn` declarations,
    preserving spans, and reject statements at the top level.
  - Cover ordering, annotations, and malformed top-level source in parser tests
    and snapshots.

- [ ] **Resolve and check module-level bindings**

  - Resolve references to module-level bindings from any declaration in the
    file, regardless of order, and from every function body.
  - Diagnose duplicate module-level names, initializer cycles, and initializers
    that are not constant expressions.
  - Preserve local shadowing of module-level bindings and permit assignment to a
    module-level `var` from a function body.

- [ ] **Lower and execute module-level bindings**

  - Initialize module-level bindings before `main` runs and preserve their
    values and mutations through IR verification, snapshots, and native
    execution.
  - Add the milestone example and validate the integrated milestone.

### Completion gates

- A module-level binding referenced before its own declaration compiles and
  executes; a duplicate name, an initializer cycle, and a non-constant
  initializer each produce a source diagnostic.
- A local binding shadows a module-level binding without modifying it, and an
  assignment in a function body updates the module-level `var`.
- The example compiles and exits 42.

---

## Branches and loops

Add branches, loops, comparisons, and boolean values after their control-flow
rules are specified. This milestone owns the implementation of
[Comparison](../docs/spec.md#comparison). Define this milestone's tasks with
kroadmap once those language decisions are settled.

---

## Parameters, calls, and return values

Add parameters, calls, and return values after the remaining function rules
are specified. Define this milestone's tasks with kroadmap once those language
decisions are settled.

---

## Modules and imports

Add module directories and `use` declarations after the open module questions in
[the specification](../docs/spec.md#open-module-questions) are settled. Define
this milestone's tasks with kroadmap once those language decisions are settled.
