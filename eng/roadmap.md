# Fern Roadmap

Fern is a compiler written in Rust. The initial stack is Logos for lexing,
la-arena for syntax storage, lasso for identifier interning, Ariadne for
diagnostic rendering, and Insta for syntax and IR snapshots. The parser is
handwritten, and native compilation uses a Fern-owned IR and QBE.

Every milestone compares its implemented behavior against the corresponding
sections of [the specification](../docs/spec.md). The milestone gates below
prove its integrated outcome. Task checkboxes record progress; milestones
remain in implementation order.

Do not add incremental queries, lossless syntax, control-flow infrastructure, or
alternate backends before a milestone needs them.

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

## Integer types and implicit conversions

This milestone records work completed against the earlier specification. Its
suffix and implicit-conversion details are historical; Align existing compiler
behavior with the specification below owns their replacement.

Support every fixed-width integer type plus `int` and `uint`, including
annotations, suffixes, range checking, inference, and specified implicit
conversions. Extend initialization, assignment, and `exit` argument checking and
native execution against [Integer types](../docs/spec.md#integer-types),
[Integer conversions](../docs/spec.md#integer-conversions), and
[Untyped constants](../docs/spec.md#untyped-constants). Operators, explicit casts,
`size`, `uintptr`, and rune values remain outside this milestone.

### Example

See [the example](../examples/integer_types.fern).

Expected exit status: 42.

### Tasks

- [x] **Check integer types and implicit conversions**

  - Check literal ranges, suffix types, contextual typing, and inferred binding
    types against the specification sections linked above.
  - Preserve the existing implementation's `int` width and apply it consistently
    to `uint`, range checks, and conversions.
  - Apply the specified conversions to initialization, assignment, and `exit`
    arguments, with source diagnostics for invalid values and conversions.
  - Retain the source and destination typing needed to lower each checked
    conversion, including conversions of binding references.
  - Cover typing across binding copies, shadowing, nested scopes, and source
    after exits. Lowering and native execution remain unfinished at this
    boundary.

- [x] **Lower typed integers and conversions**

  - Carry checked integer values and conversions through Fern IR.
  - Preserve full-width values and conversion results through initialization,
    assignment, binding copies, shadowing, and nested exits.
  - Extend IR verification and snapshots to cover value types, ranges, copies,
    conversions, and exits. Reject malformed typed values and conversions before
    emission. Native emission belongs to the following task.
  - Preserve existing executable behavior and reject programs the backend cannot
    yet emit without replacing an existing output file.

- [x] **Execute typed integer programs**

  - Extend native emission to support the checked integer types and conversions.
  - Verify full-width values and signed and unsigned widening at the backend
    boundary, alongside native execution of source programs.
  - Add execution coverage and move the milestone example into
    `examples/integer_types.fern`, then validate the integrated milestone
    against the completion gates below.

### Completion gates

- Annotations, suffixes, inference, and contextual literals cover every integer
  type in scope, with range boundaries exercised across literal bases.
- Source tests cover zero, positive maxima, and out-of-range magnitudes;
  negative signed boundaries are covered through internal IR and backend tests
  until integer expressions add source syntax for negative values.
- Initialization, assignment, and exit checking accept and reject conversions
  according to the linked specification, including distinct equal-width types.
- Coverage distinguishes contextual literals from suffixed literals and binding
  references, including a literal invalid for its suffix despite fitting the
  destination type.
- Large unsigned values retain their full range through checking, IR, and native
  emission. Coverage checks high-bit preservation beyond observable exit bytes.
- Invalid literals and conversions produce source diagnostics, including in
  nested scopes and after exits.
- Native execution covers conversions combined with reassignment, saved copies,
  shadowing, and nested exits. Existing executable behavior remains covered.
- Syntax and IR snapshots are deterministic and generated IR passes
  verification.

---

## Align existing compiler behavior with the specification

Bring the existing integer declarations, binding copies, assignments, scopes,
and process exit into agreement with the revised specification. Add explicit
integer conversions so existing programs can express the transfers previously
performed implicitly. The contracts are
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

Create `examples/integer_conversions.fern` when this milestone is implemented.

```fern
fn main() -> void {
    const literal: u8 = 42;
    const limit: u64 = 18446744073709551615;
    const reduced = u8.truncate(340282366920938463463374607431768211498);
    const signed = i8.truncate(255);

    var small: u8 = literal;
    var wide: u64 = u64(small);
    var status = int(wide);
    const saved = status;
    {
        var status: u8 = u8(saved);
        status = reduced;
        small = status;
        wide = u64(small);
    }
    status = int(wide);

    // Uncomment individually to try compile-time diagnostics.
    // const suffix = 42u8;
    // const implicit: u64 = small;
    // const wrong_exit: i64 = status;
    // const out_of_range = u8(256);
    // const constant_failure = u8(limit);

    // Uncomment to try a runtime conversion failure.
    // var large: u64 = limit;
    // status = int(u8(large));

    exit(status);
}
```

Expected exit status: 42.

### Tasks

- [ ] **Align literal syntax and reserved names**

  - Reject removed literal suffixes and retain base validation and useful source
    spans for malformed integer tokens.
  - Align reserved-name checks with the remaining lexical contract.
  - Update affected parser fixtures and snapshots without changing declaration,
    assignment, block, or exit syntax.

- [ ] **Align integer widths and assignment compatibility**

  - Apply the integer type contract consistently in semantic checking, Fern IR
    verification, and native emission for the supported host target.
  - Replace implicit conversion acceptance with the specified compatibility
    checks for initialization, assignment, binding copies, and exit arguments.
  - Update affected tests for contextual literals, type identity, range checks,
    native values, and exit statuses. Explicit conversions are added next.

- [ ] **Parse and check explicit integer conversions**

  - Accept checked and truncating conversion expressions, including nesting,
    wherever the existing expressions are accepted.
  - Evaluate constant conversions using the specified precision and range rules.
    Distinguish constant binding references from runtime binding references
    through copies and shadowing.
  - Check runtime conversion operands and destination types, retaining the
    information needed for lowering and source diagnostics.
  - Cover malformed conversion syntax, large literal inputs, constant failures,
    and source after exits. Native execution is completed in the next task.

- [ ] **Lower and execute explicit integer conversions**

  - Carry checked and truncating conversions through Fern IR and its verifier,
    and emit their specified values and runtime failures.
  - Cover all integer type pairs, full-width signed and unsigned boundaries,
    narrowing, widening, equal-width distinct types, and nested conversions.
  - Preserve assignment, saved-copy, shadowing, and exit behavior, including
    termination before subsequent runtime work after a failed conversion.

- [ ] **Migrate examples and validate compiler alignment**

  - Migrate existing source examples, README descriptions, tests, and snapshots
    from removed syntax and implicit conversions to the revised contracts.
  - Create the example above and replace its snippet with a link to the file.
  - Validate the integrated compiler against the completion gates and review
    the milestone once for correctness and simplicity.

### Completion gates

- The parser accepts the retained literal bases and conversion forms, rejects
  removed suffixes, and agrees with the specification's reserved names.
- All ten integer types agree across checking, IR verification, and native
  execution. Width-sensitive checks cover both specified pointer widths without
  claiming native execution on targets the toolchain does not support.
- Contextual literal typing and every source/destination type pair are covered
  in initialization, assignment, and exit checking. Removed implicit conversion
  behavior has regression coverage, including equal-width distinct types.
- Constant conversion coverage includes the specified precision, values beyond
  the machine integer range, nested conversions, constant binding references,
  runtime-initialized immutable bindings, and unreachable source.
- Native and internal full-width checks cover checked-conversion success and
  failure and truncating-conversion results at signed and unsigned boundaries.
  Runtime failures identify the operation and occur in every build configuration.
- Scope, mutability, copies, and process exit retain their specified behavior.
  Host-width exit arguments preserve the existing externally reported statuses.
- Invalid source and runtime conversion failures remain distinct. Failed
  compilation preserves existing output files, and malformed IR is rejected
  before emission.
- Existing examples and the new example compile and produce their documented
  results. Tests and snapshots agree with the revised contracts, and the
  integrated milestone receives one correctness and simplicity review.

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

Before implementation begins, use kspec to settle operator precedence,
associativity, operand evaluation order, and wrapping expressions with only
untyped operands. Refresh this milestone's tasks and example with kroadmap if
those decisions change its scope. The example below uses explicit grouping and
typed wrapping operands so it does not depend on those open decisions.

### Example

Create `examples/integer_expressions.fern` when this milestone is implemented.

```fern
fn main() -> void {
    const exact: u8 = (250 + 10) / 2;
    var y: u8 = 10;
    const wrapped: u8 = (250 &+ y) / 2;
    const widened: u16 = u16(y) + 1000;

    const bits = (((^u8(0)) ^ 0x0F) & 0xF0) | 0x02;
    const shifted = bits >> 4;
    var high: u8 = 128;
    const discarded = high << 1;
    var one: u8 = 1;
    const large_shift = one << 8;
    var negative: i8 = -7;
    const sign_fill = negative >> 8;
    const quotient = negative / 3;
    const remainder = negative % 3;
    const negated = -negative;
    const minimum: i8 = -128;
    const large: i64 = 1 << 40;

    var status: int = 40;
    {
        const saved = status;
        status = saved + int(wrapped);
        y = y &* 30;
    }

    // Uncomment individually to try compile-time diagnostics.
    // const out_of_range: u8 = 250 + 10;
    // const typed_overflow = u8(250) + 10;
    // const constant_shift = u8(128) << 1;
    // const zero_divisor = status / 0;
    // const negative_shift = status << -1;

    exit(status);
}
```

Expected exit status: 42.

### Tasks

- [ ] **Parse integer operators and grouping**

  - Accept the specified unary and binary operators and parentheses wherever
    the existing integer expressions are accepted.
  - Preserve operator and operand source locations for later diagnostics.
  - Cover precedence, associativity, unary chains, negative literal syntax,
    comments adjacent to operators, and malformed expressions in parser tests
    and snapshots.

- [ ] **Check integer expression types**

  - Check untyped and concrete operand combinations, unary operations, shift
    operands, result types, and destination conversions against the linked
    contracts.
  - Support source-level signed minimum values and preserve contextual and
    explicit-conversion range checks.
  - Preserve binding identity and concrete binding types through expressions,
    assignments, and lexical scopes, including source after exits.
  - Extend constant-expression classification from literals, conversions, and
    binding references to the new operators; evaluation follows in the next task.

- [ ] **Evaluate integer constant expressions**

  - Evaluate the required constant expressions and subexpressions, preserving
    untyped values until their specified conversion boundary.
  - Diagnose invalid literal and result conversions, concrete arithmetic
    overflow, invalid divisors, and negative shift counts at source locations.
  - Cover the distinction between untyped expressions, explicitly typed
    constant operations, and binding references, including constant expressions
    nested within runtime expressions and unreachable source.
  - Distinguish constant shifts from runtime shifts, and ordinary arithmetic
    from explicitly wrapping operations during constant evaluation. Runtime lowering remains unfinished at this boundary.

- [ ] **Lower checked integer operations**

  - Carry checked operations, conversions, and constant results through Fern IR
    while preserving operand evaluation order and required runtime failures.
  - Extend IR verification and snapshots to cover the supported operations,
    operand and result types, and failure behavior. Malformed IR must be
    rejected before native emission.
  - Preserve existing executable behavior while native support is incomplete;
    reject unsupported emission without replacing an existing output file.

- [ ] **Execute integer expressions and failures**

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
  example agree with the linked contracts. Existing supported programs remain
  covered, and the completed milestone receives one correctness and simplicity
  review.

---

## Branches and loops

Add branches and loops after their control-flow rules are specified. Define
this milestone's tasks with kroadmap once those language decisions are settled.

---

## Parameters, calls, and return values

Add parameters, calls, and return values after the remaining function rules
are specified. Define this milestone's tasks with kroadmap once those language
decisions are settled.
