# Fern Roadmap

Fern is a compiler written in Rust. The initial stack is Logos for lexing,
la-arena for syntax storage, lasso for identifier interning, Ariadne for
diagnostic rendering, and Insta for syntax and IR snapshots. The parser is
handwritten, and native compilation uses a Fern-owned IR and QBE.

Every milestone implements the corresponding sections of
[the specification](../docs/spec.md). The milestone gates below prove its
integrated outcome. Only the current milestone carries its scope, example,
tasks, and gates. Finished milestones and their tasks stay recorded, in
implementation order, under Past Milestones.

Do not add incremental queries, lossless syntax, control-flow infrastructure, or
alternate backends before a milestone needs them.

---

## Past Milestones

- [x] Executable integer subset
  - [x] Establish the Rust compiler
  - [x] Parse and check entry syntax
  - [x] Lower and execute integer programs
- [x] Assignment and nested scopes
  - [x] Parse local assignments and nested statement blocks
  - [x] Check assignment targets and lexical scopes
  - [x] Lower and execute assignments and nested scopes
- [x] Integer types
  - [x] Check integer types
  - [x] Lower typed integers
  - [x] Execute typed integer programs
- [x] Integer conversions
  - [x] Validate literal syntax and reserved names
  - [x] Check integer compatibility
  - [x] Parse and check explicit integer conversions
  - [x] Lower and execute explicit integer conversions
  - [x] Add examples and validate integer conversions
- [x] Integer expressions
  - [x] Parse integer operators and grouping
  - [x] Check integer expression types
  - [x] Evaluate integer constant expressions
  - [x] Check shift and wrapping boundaries
  - [x] Lower checked integer operations
  - [x] Execute integer expressions and failures
- [x] Operator spelling and precedence
  - [x] Lex and parse operator spellings
  - [x] Apply operator precedence levels
  - [x] Check and execute the operators
- [x] Module-level declarations
  - [x] Parse top-level bindings
  - [x] Resolve and check module-level bindings
  - [x] Lower and execute module-level bindings
- [x] Branches and loops
  - [x] Parse boolean expressions and control-flow statements
  - [x] Check boolean expressions
  - [x] Check structured control flow
  - [x] Represent and verify control flow in Fern IR
  - [x] Compile and execute branches and loops
- [x] Parameters, calls, and return values
  - [x] Parse function signatures, calls, and returns
  - [x] Resolve and check function calls
  - [x] Check returns and function reachability
  - [x] Represent and verify functions in Fern IR
  - [x] Compile and execute calls and returns
- [x] Modules and imports
  - [x] Parse `pub`, `use`, and qualified names
  - [x] Compile a module from multiple source files
  - [x] Resolve imports to modules
  - [x] Check visibility and imported names
  - [x] Represent multi-module programs in Fern IR
  - [x] Compile and execute multi-module programs
- [x] Arrays
  - [x] Parse array syntax
  - [x] Represent array types across the compiler
  - [x] Check array values and literals
  - [x] Check indexing, length, comparison, and iteration
  - [x] Represent arrays in Fern IR
  - [x] Compile and execute array programs

---

## Compiler phase decomposition

Split the frontend, semantic, IR, and backend phases into responsibility-based
submodules with ownership explicit at each call site. Fern behavior,
diagnostics, snapshots, emitted QBE, and native results remain unchanged.

### Example

The completion fixture is
[`tests/fixtures/programs/arrays.fern`](../tests/fixtures/programs/arrays.fern).
It continues to exit with status 46.

### Tasks

- [x] Decompose compiler phase modules

### Completion gates

- Cross-phase calls name the responsible submodule directly, with one
  implementation path for each behavior.
- Unit tests and snapshots reside with their owning phase submodules, with
  snapshot contents unchanged.
- Existing diagnostics, emitted QBE, and native compiler and CLI results remain
  unchanged, including the array completion fixture.

---

## Compiler architecture hardening

Make program file and module identity explicit, and record the compiler's
durable phase, storage, diagnostic, and verification boundaries. Fern behavior
and generated native programs remain unchanged.

### Example

The completion fixture is
[`tests/fixtures/programs/arrays.fern`](../tests/fixtures/programs/arrays.fern).
It continues to exit with status 46.

### Tasks

- [x] Make program identity explicit and record the compiler architecture

### Completion gates

- A program has one typed identity for files and modules, without parallel
  file-indexed stores that can fall out of sync.
- `eng/architecture.md` owns the compiler's durable implementation boundaries
  and identity rules.
- Existing diagnostics, emitted QBE, and native compiler and CLI results remain
  unchanged, including the array completion fixture.

---

## Floating-point numbers

Implement `f32` and `f64` values, floating-point literals, numeric conversions,
arithmetic, and comparisons as specified in
[the floating-point sections](../docs/spec.md#floating-point-semantics).

### Example

The completion fixture will be
[`tests/fixtures/programs/floating_point.fern`](../tests/fixtures/programs/floating_point.fern).

```fern
fn main() -> void {
    const half: f32 = .5;
    var value: f32 = half + half;

    if f64(value) / 2.0 == 0.5 {
        exit(42);
    }
    exit(255);
}
```

It exits with status 42.

### Tasks

- [x] Lex and parse floating-point literals
- [x] Represent floating-point types and numeric conversions across the compiler
- [x] Check floating-point expressions and constant expressions
- [x] Lower and emit floating-point values, operations, and comparisons
- [x] Compile and execute floating-point programs and failures

### Completion gates

- `f32` and `f64` literals, bindings, arithmetic, comparisons, and conversions
  agree with the specification, including exact constant evaluation and
  round-to-nearest, ties-to-even conversion.
- Runtime arithmetic has the specified IEEE 754 results, while invalid or
  non-finite constant expressions and information-losing conversions are
  rejected or trap as specified.
- The floating-point completion fixture exits with status 42.

---

## Structs

Implement named struct declarations, literals, field selection, copying, and
struct equality as specified in [Type declarations](../docs/spec.md#type-declarations)
and [Struct literals](../docs/spec.md#struct-literals).

### Example

The completion fixture will be
[`tests/fixtures/programs/structs.fern`](../tests/fixtures/programs/structs.fern).

```fern
type Point struct {
    x: int,
    y: int,
}

fn main() -> void {
    var point = Point { x = 19, y = 23 };
    point.x = point.x + 1;
    const expected = Point { x = 20, y = 23 };

    if point == expected {
        exit(point.x + point.y);
    }
    exit(255);
}
```

It exits with status 43.

### Tasks

- [ ] Parse named struct declarations, literals, and field selection
- [ ] Resolve struct fields and check struct values, assignments, and equality
- [ ] Represent structs and field access in Fern IR
- [ ] Lay out, copy, compare, and access structs in native code
- [ ] Compile and execute struct programs and failures

### Completion gates

- Struct declarations, literals, zero-value fields, field selection, and
  recursive value copying agree with the specification.
- Field mutability and structural equality reject invalid programs and handle
  nested comparable values as specified.
- The struct completion fixture exits with status 43.
