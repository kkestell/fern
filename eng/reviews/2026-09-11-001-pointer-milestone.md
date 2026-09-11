# Pointer milestone review

Scope: the completed Pointers milestone, including its specification, plans,
frontend, semantic checker, IR, backend, tests, fixture, example, and TODO.

Mode: general.

## Findings

### High: pointer-returning calls cannot be implicit-dereference assignment targets

`src/frontend/parser.rs:433` classifies any name immediately followed by `(` as
a call statement before it considers an assignment. Consequently, a valid
pointer location such as `get(&point).x = 1;` is parsed as the call statement
`get(&point)` and fails at `.` with `expected \`;\``. `get(&array)[0] = 1;` has
the same failure.

The specification defines `p.x` and `p[i]` as implicit dereferences and says
their results are locations. A call result is a valid pointer operand; the
milestone plan also requires call-result pointer targets. This leaves a
specified class of mutable pointer locations unusable.

Resolve statement-head ambiguity by recognizing the full postfix assignment
target before selecting the call-statement form. Represent the implicit
dereference in `CheckedLocation` and lowering without requiring its pointer
operand itself to be a location. Add parser, semantic, IR, native, and
integration cases for a pointer-returning call followed by field and index
assignment, including compound assignment.

Reproduction:

```fern
type Point struct { x: int }
fn get(pointer: *Point) -> *Point { return pointer; }
fn main() -> void {
    var point: Point;
    get(&point).x = 1;
}
```

The compiler reports `expected \`;\`` at the `.` rather than compiling it.

### Medium: nested implicit-dereference stores retain the wrong trap span

`src/semantic/expressions.rs:675-731` propagates the root location's span
through every index and field. `src/ir/lower.rs:521-550` attaches that propagated
span to a `HeldStep::Indirect`; `src/backend/emitter.rs:126-148` renders it at
the runtime null check. Thus `node.next.value = 1;` where `node.next` is null
reports the span of `node`, not the failing implicit dereference `node.next`.

The specification requires a trap diagnostic to identify the failing operation
and its source location. Preserve the source span of each pointer operand when
constructing the implicit indirect step, and assert the exact highlighted span
for nested field and index assignment targets.

Reproduction: executing `node.next.value = 1;` for a zero-initialized
`Node { next: *Node }` reports a diagnostic starting at `node`; the operation
that dereferences null is `node.next`.

## Topic verdicts

- Ownership: no finding; new `Box<Type>` and held operands express the required
  ownership boundaries.
- Error handling: no finding; compiler and runtime failures retain their normal
  diagnostic paths.
- API design: no finding; pointer types fit the existing internal type model.
- Performance: no measured regression; pointer storage uses one word and held
  operands avoid re-evaluating compound-assignment targets.
- Testing: the two findings are untested; existing tests cover direct pointer
  locations and only assert that runtime diagnostics name a file.
- Readability: no finding.
- Concurrency: not applicable; the changed compiler path has no shared or
  asynchronous state.
- Security: no new trust boundary or injection path found.
- Correctness: the two findings above violate implicit-location behavior and
  runtime source mapping.
- Unsafe: no changed unsafe Rust or FFI boundary.
- Architecture: no finding; verified IR remains the sole backend input.
- Dependencies: no dependency or lockfile change.
- Documentation: no finding; the specification, TODO, README, plans, fixture,
  and example describe the completed milestone consistently.

## Checks run

- `git diff --check` — passed.
- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed (302 tests).
- `cargo test --doc` — passed (0 tests).
- `cargo tree --duplicates` — no duplicate dependencies.
- Manual compiler reproductions confirmed both findings.

## Unresolved suspicions

None.
