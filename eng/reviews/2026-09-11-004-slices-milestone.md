# Slices milestone review

Scope: the completed Slices milestone, including its six plans, language
contract, TODO and README entries, frontend, semantic checker, Fern IR,
backend, unit and integration tests, fixture, and example.

Mode: general.

Selected topics: correctness (slice bounds, recursive values, and module
ordering), error handling (diagnostic propagation and runtime failure policy),
architecture (phase visibility and dependency ownership), testing (boundary,
failure, and interaction coverage), performance (slice traversal and helper
generation), readability, ownership, API design for shared compiler models,
and documentation. Dependency, concurrency, security, and unsafe topics do
not materially apply to this change.

## Findings

### High: recursive slice equality can segfault on a valid cyclic value

`src/backend/qbe.rs:237-309` emits one recursively called equality helper per
slice element type. A valid program can make a finite cyclic value by storing a
slice of an array back into a slice field of one of its elements, then compare
that value. The helper repeatedly calls itself for the same element pair until
the native stack overflows; with a 512 KiB stack the reproducer exits 139
without a Fern diagnostic.

The specification allows structs to reach themselves through slice fields and
defines slice equality element-by-element; it does not permit a valid equality
operation to turn into an unreported native crash. Add cycle handling (or a
specified, checked failure) to the equality representation, and add an
integration test with a self-referential slice value. The current test only
uses a recursive type whose slice is empty, so it never enters the recursive
case.

Reproduction:

```fern
type Node struct { children: []Node }
fn main() -> void {
    var nodes: [1]Node;
    nodes[0].children = nodes[:];
    if nodes[:] == nodes[:] { exit(0); }
}
```

### Medium: slice element annotations lose forward module dependencies

`src/semantic/namespaces.rs:815-827` treats `AnnotationKind::Slice` as having
no references at all. That correctly avoids inline struct-size dependencies,
but it also skips array-length expressions nested in the slice element. Since
module bindings are visible regardless of declaration order (`docs/spec.md`,
Module-level declarations), this valid program is rejected when `n` follows
the slice declaration:

```fern
var rows: [][n]int;
const n = 2;
fn main() -> void {}
```

It reports `array length must be a constant expression`; reversing the two
declarations succeeds. Recurse through pointer/slice wrappers for array-length
references without following named struct fields, and add both declaration
orders to namespace tests.

### Medium: invalid call diagnostics are replaced by “not a location”

`src/semantic/expressions.rs:952-982` calls `infer_expression` in
`location_operand` but discards its error (`Err(_) => return Err(location_error)`).
As a result, a slice of a call result with an invalid argument reports the
location fallback instead of the actual call error. For example,
`make(true)[:]`, where `make` expects `int` and returns `[]int`, reports
`cannot slice an expression that is not a location` rather than
`cannot implicitly convert `bool` to `int``. This violates the normal
operand-first diagnostic path and makes the source of the failure misleading.

Preserve and return the inference error when the operand is a call (while still
using the location fallback for a successfully inferred non-location), and add
a checker test for an invalid call argument in a slicing operand.

### Low: semantic expression module visibility is left wider than its owner

`src/semantic/mod.rs:3` keeps `expressions` as `pub(crate)`, a visibility added
for the temporary slice staging guard. The final subtask removes that guard,
and `rg` finds no remaining cross-module caller. Restore private visibility so
the phase root exposes only the boundaries recorded by the architecture.

## Topic verdicts

- Ownership: no finding; boxes, held operands, and slice copies express their
  required storage and lifetime boundaries.
- Error handling: the invalid-call diagnostic finding above; runtime traps
  otherwise preserve source mapping.
- API design: the stale module visibility above; no externally public API was
  added.
- Performance: no measured regression beyond the unbounded recursive equality
  path; ordinary slice equality is linear in the compared elements and helper
  registration is bounded by distinct element types.
- Testing: the cyclic equality and forward-reference cases are untested; the
  checked-IR tests cover only one valid slice shape and do not exercise the
  malformed verifier branches described by the slice plan.
- Readability: no additional finding.
- Architecture: the forward-reference dependency ownership and stale phase
  visibility findings above.
- Documentation: the contract, TODO, plans, README, fixture, and example are
  consistent, but the specification does not settle behavior for cyclic slice
  equality; that decision is needed with the High finding.
- Correctness: the cyclic equality and forward-reference findings above.

## Checks run

- `git diff --check` — passed.
- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed (330 unit, 61 integration, 0 doc tests).
- Manual compiler reproductions confirmed the forward-reference rejection and
  invalid-call diagnostic; a cyclic slice equality program compiled and then
  exited 139 under a 512 KiB stack.

## Unresolved suspicions

None.
