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
