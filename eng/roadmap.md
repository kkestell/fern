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

---

## Arrays

Add fixed-length arrays as value types against
[Arrays](../docs/spec.md#arrays),
[Indexing and lengths](../docs/spec.md#indexing-and-lengths),
[Assignment](../docs/spec.md#assignment),
[Comparison](../docs/spec.md#comparison), and
[Loops](../docs/spec.md#loops).

The scope is array types including nested ones, array literals with fill and
`[_]`, indexing and element assignment, `len`, array equality, arrays as
parameters, results, and module-level declarations, whole-array copying on
initialization, assignment, argument passing, and return, and both `for … in`
forms. Slices, strings, and iteration over them remain outside this milestone.

### Example

```fern
const weights: [_]int = [1, 2, 3];

fn weighted(row: [3]int) -> int {
    var total = 0;
    for v, i in row {
        total = total + v * weights[i];
    }
    return total;
}

fn totals(grid: [2][3]int) -> [2]int {
    var out: [2]int = [0...];
    for var r = 0; r < len(grid); r = r + 1 {
        out[r] = weighted(grid[r]);
    }
    return out;
}

fn main() -> void {
    var grid: [2][3]int = [[1, 2, 3], [4, 5, 6]];

    var copy = grid;
    copy[0][0] = 99;
    if copy == grid {
        exit(255);
    }

    var sum = 0;
    for t in totals(grid) {
        sum = sum + t;
    }
    exit(sum); // reports 46
}
```

Expected exit status: 46. Completion fixture:
`tests/fixtures/programs/arrays.fern`.

### Tasks

- [ ] **Parse array syntax**

  - Make a type annotation a syntax node so `[N]T`, `[_]T`, and nested array
    types carry an element type and an unevaluated length expression.
  - Accept array literals with a trailing comma and a trailing `...` fill, index
    expressions, `len`, and both `for … in` forms with and without a label.
  - Bind indexing, calls, conversions, and `len` more tightly than any operator,
    and cover valid forms and malformed source in parser tests and snapshots.

- [ ] **Represent array types across the compiler**

  - Replace the scalar `Type` with a representation that names array types,
    compares them structurally, and spells them in diagnostics, and carry it
    through the semantic, IR, and backend phases.
  - Resolve type annotations to types in the semantic phase, evaluating array
    lengths as constant expressions and rejecting a length below 1, a
    non-constant length, a `void` element type, and `[_]` outside a declaration
    with an array-literal initializer.

- [ ] **Check array values and literals**

  - Type array literals from context and, without context, from a single common
    element type, checking element count against the length and diagnosing fill
    in a literal that has no length from context.
  - Treat an array literal of constant elements as a constant expression so a
    module-level `var` or `const` can hold one.
  - Check that initialization, assignment, argument passing, and return of a
    whole array require identical types and copy the value.

- [ ] **Check indexing, length, comparison, and iteration**

  - Check that an index has type `int` or is an untyped constant that fits,
    reject an out-of-range constant index, and treat `a[i]` as never constant.
  - Allow assignment to an element of a mutable array, reject assignment to an
    element of a `const` binding, and evaluate a target's indices left to right
    before the value and once for a compound assignment.
  - Check `len` on an array operand, including when its result is constant, and
    `==` and `!=` on identical array types while rejecting the other operators.
  - Check both `for … in` forms: an array operand, distinct immutable bindings
    scoped to the body and rebound each iteration, and an `int` index binding.

- [ ] **Represent arrays in Fern IR**

  - Give arrays addressable storage for locals, parameters, results, and
    globals, with element places, whole-array copies, and constant aggregate
    initializers.
  - Lower indexing with a bounds check that traps, `len` from the operand's
    type while still evaluating the operand, elementwise array comparison, and
    both `for … in` forms over a value captured once before the first
    iteration.
  - Verify element types, index types, place types, and copy sizes, with IR
    tests and snapshots covering nested arrays and aggregate calls.

- [ ] **Compile and execute array programs**

  - Emit native aggregate storage, element addressing, whole-array copies,
    aggregate arguments and results, and bounds-check traps for the verified IR.
  - Add an `examples/arrays.fern` topic file, the milestone completion fixture,
    and validate the integrated milestone.

### Completion gates

- Array declarations, literals, fill, `[_]`, and nested arrays compile with the
  specified types; a wrong element count, a fill without a length, a length
  below 1, a non-constant length, and a mismatched element type each produce a
  source diagnostic.
- Whole-array initialization, assignment, argument passing, and return copy the
  value, so mutating one array leaves the other unchanged.
- Indexing reads and writes elements of nested arrays; an out-of-range constant
  index is a compile error, an out-of-range runtime index traps, and an index of
  another integer type is rejected without an explicit conversion.
- Assignment to an element of a `const` binding is rejected, and element
  assignment evaluates its indices left to right before the value.
- `len` yields the length from the operand's type, is usable as a constant
  expression, and still evaluates an operand that traps.
- `==` and `!=` compare arrays of identical type elementwise, while `<`,
  arithmetic, and bitwise operators on arrays are rejected.
- Both `for … in` forms iterate in index order over a value captured once,
  with the bindings immutable and invisible after the loop.
- A module-level `var` and `const` array initialize before `main` runs.
- The example compiles and exits 46.
