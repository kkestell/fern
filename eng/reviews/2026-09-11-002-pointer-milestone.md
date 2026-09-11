# Pointer milestone review

Scope: the uncommitted Pointers milestone and its repair, comprising the
specification sections, TODO, README, the six plans in `eng/plans/`, the
frontend, semantic checker, Fern IR, backend, unit and integration tests, the
`tests/fixtures/programs/pointers.fern` fixture, `examples/pointers.fern`, and
`eng/reviews/2026-09-11-001-pointer-milestone.md` with its repair plan
`eng/plans/2026-09-11-005-repair-pointer-location-review-findings.md`.

Mode: general.

## Findings

### Medium: `null` loses its context type inside a grouped operand

`src/semantic/expressions.rs:39` infers an ungrouped `null` against the
destination the caller passes, and `src/semantic/expressions.rs:907-990` types a
comparison operand `null` from the other side's type. Both dispatch on the
expression's syntax kind, so the context does not reach a `null` inside a
grouping: `src/semantic/expressions.rs:560-574` (`infer_grouping`) infers its
inner expression without a destination, and `ExpressionKind::Null` with no
destination is rejected unconditionally. The same holds for an assignment value
and a call argument.

```fern
fn main() -> void {
    var p: *int = (null); // rejected; `p: *int = null` compiles
    var q: *int;
    if (null) == q { // rejected; `null == q` compiles
        exit(1);
    }
}
```

The specification types `null` from context "under the rules of Typing by
context" (`docs/spec.md`, The null pointer), which name the position — an
annotated binding, an assignment target, a call's parameter type, the other
operand of a comparison — not the spelling. Grouped untyped integers already
contextualize in exactly these positions (`(1) == f` with `f: f64` compiles;
verified), so the two untyped constants disagree through one grouping.

Thread the destination through `infer_grouping`, or give the grouped `null` its
context in `infer_grouping` and in the comparison's operand handling, and add
parser-independent checker cases for a grouped `null` in a binding initializer,
an assignment value, a call argument, and a comparison operand.

Reproduction: both rejected programs above; the compiler reports
"`null` requires a pointer type from context" at the grouping.

### Medium: `len` through a pointer copies the whole array on every evaluation

`src/ir/lower.rs:1168-1190` (`lower_length`, implicit-dereference arm) evaluates
its operand by loading the array through a null-checked indirect place; the
emitter renders that load as a full aggregate copy (`blit`), and the loaded
value is discarded because the length comes from the static type. Each
evaluation of `len(p)` therefore copies the entire array the pointer reaches,
in memory proportional to the array's length, where a plain `len(a)` emits
nothing. Confirmed in the emitted QBE for `len(p)` over `[4000]P`:
`blit %v4, %v5_storage, 64000`, once per evaluation. A 200,000-iteration loop
measured the same as the direct-array version at that size (about 1.35 s), so
the effect is unmeasured at scale, but the copy is present in the emitted code
and its cost is provably O(length) per evaluation.

The specification requires `len(p)` to trap on a `null` pointer, which needs
only the null check, not the copy. Emit the indirect null-check trap without
materializing the array — for example a trap-only place-address form — and
assert the emitted IR for `len(pointer)` contains no aggregate load.

### Low: the pointer-withdraw conversion rule has two implementations

`src/semantic/expressions.rs:86` (`value_compatible`) and `src/ir/verify.rs:53`
(`value_compatible`) carry byte-identical definitions of the one non-identical
value conversion (`*T` accepted where `*const T` is expected). Under this
repository's one-home rule the shared fact belongs in `src/types.rs`, where
`Type` already lives and both phases read it. Name the shared function once and
call it from both.

### Low: address-of a non-location is diagnosed as an assignment problem

`src/semantic/expressions.rs:776-780` (`check_location_with_facts`, fallthrough
arm) rejects every non-location operand with "cannot assign to an expression
that is not a location", but the same check serves address-of, where the
specification says "Taking the address of an expression that is not a location
is invalid". Verified: `&f(x)` reports "cannot assign to an expression that is
not a location". Give the address-of path its own message — for example
"cannot take the address of an expression that is not a location" — by passing
the position to `check_location_with_facts`, and add a checker case for
`&f(x)` and `&(x + 1)`.

### Low: untested runtime paths around address-of and single evaluation

- No test traps on address-of through a `null` pointer: `&p.x`, `&p[0]`, and
  `&*p` with `p: *Point = null` reach the null check the emitter emits for
  `Place::Indirect` under `ValueKind::AddressOf`
  (`src/backend/emitter.rs:126-149`), and only happy paths are exercised.
- The "a compound assignment evaluates the target once" rule is structurally
  guaranteed by `HeldPlace`, but no test observes it: the native case uses the
  side-effect-free `echo`, and no IR test counts the call instructions behind a
  call-result compound target. An IR test asserting one `Call` per
  `echo(&a)[0] += 1` would pin the rule.
### Low: location checking's near-duplicate arms

`src/semantic/expressions.rs:675-810` (`check_location_with_facts`) contains two
roughly forty-line arms, `Index` and `Field` that differ only in the step
check they delegate to (`check_index_step` versus `check_field_step`) and the
kind they build. A reader must diff them to confirm they agree. Extracting the
shared operand dereference-and-record shape would leave each arm with its
single distinguishing rule. The `ExpressionValue::Integer` placeholder in
`src/semantic/statements.rs:185-190` (`check_compound_assignment`) similarly
requires the reader to know the value is never consumed; a dedicated unit-like
variant or a comment naming the placeholder's role would say so where it is
read.

## Topic verdicts

- Ownership: no finding; `Box<CheckedLocation>`, `Box<Type>`, and the
  `Clone`d `Literal`/`Operand` bounds express required recursion and spill
  ownership, and held operands exist precisely to move no value twice.
- Error handling: the address-of message finding above; otherwise every new
  `unreachable!`/`expect` (null constants, dereference checking, `HeldPlace`
  roots) is structurally guarded by checker or verifier invariants with stated
  reasons.
- API design: the duplicated `value_compatible` above; otherwise pointer types
  extend the existing internal type model with no new public surface.
- Performance: the `len` copy finding above; everything else keeps the
  one-word representation, emits one null check per indirect access, and adds
  no per-element overhead.
- Testing: the address-of and single-evaluation gaps above; the milestone maps
  every specified behavior to at least one test, including the two repaired
findings (parser/semantic/native coverage for pointer-returning-call targets,
and the exact `node.next` trap span at `src/ir/tests/lower.rs:37-54`).
- Readability: the near-duplicate location arms above; assignment targets
  becoming one expression grammar is a net simplification.
- Concurrency: not applicable; the change adds no shared, atomic, or
  asynchronous state.
- Security: no new trust boundary; trap diagnostics embed source locations
  only, aggregate-size limits are not bypassed because pointer targets are
  never laid out inline.
- Correctness: the grouped-`null` finding above. Verified otherwise against
  `docs/spec.md`: evaluation order (target operands, then value, then held
  rebuild), one-level implicit dereference, `*const` withdrawal in both
  directions rejected, pointer comparison restricted to `==`/`!=` with equal
  targets and never folded, `uint(pointer)` untrapping with `null` to `0`,
  pointer zero values, self-referential structs through pointer fields, and
  `for … in` correctly refusing a bare pointer operand.
- Unsafe: no changed unsafe Rust or FFI boundary; the codebase contains none.
- Architecture: verified IR remains the sole backend input, `Place::Indirect`
  and the null trap live with their owners, and the repaired
  call-result-target rule has exactly one implementation through
  `CheckedLocation`.
- Dependencies: no dependency or lockfile change; `cargo tree --duplicates`
  reports none.
- Documentation: the specification, TODO, README, plans, fixture, and example
  describe the milestone consistently and compile as written; the spec's
  precedence rules match the parser's `*(p.x)` binding.

## Repair-plan follow-through

`eng/plans/2026-09-11-005-repair-pointer-location-review-findings.md` is
delivered: pointer-returning-call field and index targets parse, check, lower,
and execute (parser, semantic, native, and integration tests all assert the
call-result head), and the nested implicit-dereference trap carries the pointer
operand's span, asserted exactly at the IR boundary. The plan's request for the
exact diagnostic location "through the public compiler boundary" is met only at
file granularity (`tests/compiler.rs` asserts `input.fern:`); the exact span is
pinned one phase earlier in the IR test. The remaining gap is the address-of
trap case noted under Testing, which the plan did not enumerate.

## Checks run

- `git diff --check` — passed.
- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed (304 unit, 59 integration).
- `cargo test --doc` — passed (0 doc tests).
- `cargo tree --duplicates` — no duplicates.
- Manual compiler reproductions confirmed the grouped-`null`, `len` copy
  (via a QBE-tee wrapper), address-of message, and `(null)` initializer
  findings; the spec's pointer-conversion example and `examples/pointers.fern`
  both compile and report their expected statuses.
- `rg` confirmed no `unsafe`, no `Cargo.toml`/`Cargo.lock` change, and
  `emit_aggregate_equal`'s pointer arm as the single word-comparison
  implementation for aggregate equality.

## Unresolved suspicions

None.
