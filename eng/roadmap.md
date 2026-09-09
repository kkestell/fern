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

- [x] **Parse top-level bindings**

  - Accept `var` and `const` declarations interleaved with `fn` declarations,
    preserving spans, and reject statements at the top level.
  - Cover ordering, annotations, and malformed top-level source in parser tests
    and snapshots.

- [x] **Resolve and check module-level bindings**

  - Resolve references to module-level bindings from any declaration in the
    file, regardless of order, and from every function body.
  - Diagnose duplicate module-level names, initializer cycles, and initializers
    that are not constant expressions.
  - Preserve local shadowing of module-level bindings and permit assignment to a
    module-level `var` from a function body.

- [x] **Lower and execute module-level bindings**

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

Add integer comparisons, boolean values and logical expressions, conditionals,
the three `for` forms, and labeled and unlabeled loop control against
[Comparison](../docs/spec.md#comparison),
[Boolean semantics](../docs/spec.md#boolean-semantics), and
[Statements and execution](../docs/spec.md#statements-and-execution). Iteration
over arrays, slices, and strings remains outside this milestone.

### Example

```fern
fn main() -> void {
    var found = false;
    var total = 0;

    for :rows var r = 0; r < 4; r = r + 1 {
        for var c = 0; c < 4; c = c + 1 {
            if c == 0 || r == c {
                continue;
            }
            if r * 4 + c == 11 {
                found = true;
                break :rows;
            }
            total = total + 1;
        }
    }

    if found && total == 6 {
        exit(total);
    } else {
        exit(255);
    }
}
```

Expected exit status: 6. Future file: `examples/branches_and_loops.fern`.

### Tasks

- [x] **Parse boolean expressions and control-flow statements**

  - Accept boolean literals and annotations, comparison and logical operators
    with their specified precedence, `if` chains, all three `for` forms, and
    labeled and unlabeled `break` and `continue` statements.
  - Cover valid forms, malformed source, and source nesting limits in parser
    tests and snapshots.

- [x] **Check boolean expressions**

  - Check `bool` bindings, integer and boolean comparisons, logical operators,
    and boolean conditions, including untyped constants and constant
    expressions.
  - Preserve short-circuit runtime behavior while diagnosing invalid constant
    subexpressions and operand types during compilation.

- [x] **Check structured control flow**

  - Enforce branch and loop scopes, three-clause loop restrictions, and the
    placement and targets of labeled and unlabeled loop control.
  - Cover shadowing, loop-initializer lifetimes, nested labels, and invalid
    control-flow statements in semantic tests.

- [x] **Represent and verify control flow in Fern IR**

  - Lower comparisons, logical expressions, branches, loops, and loop control
    without losing source locations or binding mutations across control-flow
    paths.
  - Verify control-flow targets, value types, and termination, with IR tests and
    snapshots covering nested and short-circuit paths.

- [x] **Compile and execute branches and loops**

  - Emit native control flow for the verified IR, including short-circuit
    evaluation, loop backedges, and labeled `break` and `continue` targets.
  - Add the milestone example and validate the integrated milestone.

### Completion gates

- Integer and boolean comparisons, boolean constants, and logical expressions
  compile and evaluate as specified across constant and runtime cases; skipped
  short-circuit operands do not execute.
- `if` chains and all three `for` forms execute with the specified scopes and
  iteration order, including the three-clause loop's `continue` behavior.
- Labeled and unlabeled `break` and `continue` reach the specified enclosing
  loop, while invalid placement, missing targets, and duplicate enclosing labels
  produce source diagnostics.
- Fern IR rejects invalid control flow and preserves values and mutations at
  branch joins and loop backedges.
- The example compiles and exits 6.

---

## Parameters, calls, and return values

Add fixed positional parameters, direct function calls, and zero-or-one return
values against [Functions](../docs/spec.md#functions),
[Calls](../docs/spec.md#calls),
[Scope and shadowing](../docs/spec.md#scope-and-shadowing), and
[Function return](../docs/spec.md#function-return).

The scope is explicitly typed immutable parameters, module-wide function
resolution without overloading, left-to-right argument evaluation, calls as
expressions or statements, explicit returns, and compile-time rejection of a
value-returning function whose end is reachable.

### Example

```fern
var trace = 0;

fn mark(digit: int) -> int {
    trace = trace * 10 + digit;
    return digit;
}

fn difference(left: int, right: int) -> int {
    return left - right;
}

fn sum_to(value: int) -> int {
    if value == 0 {
        return 0;
    }
    return value + sum_to(value - 1);
}

fn positive(value: int) -> bool {
    return value > 0;
}

fn remember_zero(value: int) -> void {
    if value == 0 {
        return;
    }
    trace = 255;
}

fn main() -> void {
    difference(mark(4), mark(2));
    const total = sum_to(3);
    remember_zero(0);

    if positive(total) {
        exit(trace);
    }
    exit(255);
}
```

Expected exit status: 42. Future file:
`examples/parameters_calls_and_returns.fern`.

### Tasks

- [x] **Parse function signatures, calls, and returns**

  - Accept explicitly typed parameter lists, integer, boolean, and `void`
    result types, direct call expressions and statements, and both valid
    `return` forms while preserving source spans.
  - Cover empty and trailing-comma lists, nested calls, malformed signatures,
    malformed arguments, and malformed returns in parser tests and snapshots.

- [x] **Resolve and check function calls**

  - Collect module-wide function signatures before checking bodies so forward,
    recursive, and mutually recursive calls resolve independently of declaration
    order without overloading.
  - Enforce the shared module-level namespace, lexical shadowing at call sites,
    exact arity, immutable parameter bindings, and the specified argument type
    rules.
  - Diagnose non-function call targets, calls used as constant expressions,
    and `void` calls used as values.

- [x] **Check returns and function reachability**

  - Enforce the return form and value type required by each function's declared
    result, including contextual typing of untyped constants.
  - Reject a value-returning function whose end is reachable, accounting for
    branches, nested blocks, unconditional loops, and targeted `break`
    statements while continuing to check unreachable source.

- [x] **Represent and verify functions in Fern IR**

  - Represent every function once with its signature, parameters, body, calls,
    and return terminators, replacing the entry-only IR shape without creating
    a parallel lowering path.
  - Verify call targets, argument and result types, parameter locals, return
    terminators, and control-flow reachability in IR tests and snapshots.

- [x] **Compile and execute calls and returns**

  - Emit native functions, parameters, left-to-right calls, and `void` and value
    returns, preserving module-level binding access and recursive execution.
  - Cover nested calls, mutation-observable argument order, discarded results,
    forward calls, recursion, and return paths in native integration tests.
  - Add the milestone example and validate the integrated milestone.

### Completion gates

- Forward, recursive, and mutually recursive calls resolve independent of
  declaration order; duplicate module-level names and shadowed or otherwise
  non-function call targets produce source diagnostics.
- Arguments execute completely from left to right and obey exact arity and the
  annotated-initializer type rules; assigning to a parameter is rejected.
- Calls compose inside expressions, `void` and value calls work as statements,
  and discarded return values do not change execution.
- Every valid return reaches its caller with the declared type; invalid return
  forms, mismatched values, `void` values, and reachable value-function ends are
  rejected during compilation.
- Fern IR rejects invalid calls, signatures, returns, and cross-function
  control flow.
- The example compiles and exits 42.

---

## Modules and imports

Add module directories, public declarations, and `use` declarations as defined
by [the specification](../docs/spec.md#modules-and-imports). Define this
milestone's tasks with kroadmap before implementation begins.
