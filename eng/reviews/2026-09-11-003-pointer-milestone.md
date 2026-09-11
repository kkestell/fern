# Pointer milestone review

Scope: the uncommitted Pointers milestone as architecture: `docs/spec.md`
pointer sections, `eng/todo.md` Pointers task, the six pointer plans in
`eng/plans/`, frontend syntax, semantic checking, Fern IR, backend emission,
unit and integration tests, `tests/fixtures/programs/pointers.fern`,
`examples/pointers.fern`, and the two prior pointer-milestone reviews.

Mode: architecture.

## Findings

### Medium: pointer annotation targets drop module-ordering dependencies

`src/semantic/namespaces.rs:827` ignores every pointer target when
collecting module-level ordering dependencies:

```rust
AnnotationKind::Pointer { .. } => {}
```

`src/semantic/namespaces.rs:809-829` otherwise collects array-length
expression references so a later `const` is initialized before the
declaration that names it. A length inside a pointer target is still a
value dependency: `src/semantic/annotations.rs:102-113`
(`resolve_pointer_target`, Array arm) evaluates it with `array_length`.
With the dependency dropped, checking falls back to source order and a
valid program is rejected when the length is declared later.

Consequence: declaration-order dependence for a case the direct-array
form accepts in either order, against the specification's
module-level visibility rule and the dependency-graph ownership in
`src/module.rs` / `eng/architecture.md` (semantic namespaces consume
the program; they do not discover files, but they do own ordering).

Reproduction (current tree, `cargo run`):

```fern
var p: *[n]int = null;
const n = 2;
fn main() -> void {}
```

fails with `array length must be a constant expression` at `*[n]int`.
Reversed (`const n = 2;` first) compiles, as does the direct form
`var a: [n]int;` in either order. A null-trapping `len(p)` through the
forward-declared form otherwise works once the order is fixed.

Suggested fix: recurse into the pointer target for ordering
references instead of ignoring it, e.g. collect the target
annotation's length references. Following a target struct's fields
transitively only adds harmless extra edges (a length constant cannot
depend back on a null-initialized pointer), so plain recursion is the
small change; if the owner wants zero spurious edges, collect lengths
under the pointer without expanding `collect_type_references`.

## Examined with no finding

- Phase separation holds: syntax owns parse facts, semantic owns
  `CheckedLocation` / `ExpressionValue` meaning, `Program::verify` stays
  the sole `VerifiedProgram` constructor, and only verified IR reaches
  `backend::toolchain::build` via `src/lib.rs:46-54`.
- `Place::Indirect`, `Instruction::Check`, and `ValueKind::AddressOf`
  live with their owners (IR model, IR lowering/verification, backend
  emission). `Check` has exactly one caller (`len` through a pointer)
  and exists to avoid materializing the reached array; not speculative
  machinery.
- Pointer layout is singular: `src/layout.rs` sizes/aligns pointers as
  one word, `src/backend/layout.rs` owns only the QBE class (`l`) and
  pointer leaf slots, and IR verification repeats the same derivation.
  Skipping inline layout validation under a pointer target is intended;
  the pointer itself stores one word.
- `Type::value_compatible` (`src/types.rs:224`) is now the single home
  for mutability withdrawal; the prior semantic/IR duplication noted in
  `eng/reviews/2026-09-11-002-pointer-milestone.md` is gone from the
  current tree.
- `contained_struct` (`src/ir/verify.rs:188-195`) returning `None`
  through pointers correctly breaks containment cycles, matching the
  semantic pointer-target cycle break.
- `resolve_pointer_target` duplicating part of `resolve_annotation`
  dispatch is deliberate context specialization (no field resolution, no
  `[_]` inference inside a target); the remaining shape difference is
  documented where it is read. No second runtime path for one language
  rule was found: store (`HeldPlace`), load (`element_place` /
  `field_place`), `len`, and address-of paths each construct the same
  `Place::Indirect` for a different required context.
- Diagnostics follow the owned path: trap spans travel with
  `Place::Indirect` to the backend's source-location rendering; no later
  phase reads source files.

## Unresolved suspicions

None. The one ordering question above was settled by reproduction, not
left open.

## Checks run

- `git diff --check` — passed.
- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed (308 unit, 59 integration, per quiet run tail).
- Manual `cargo run` reproductions: pointer-to-array forward reference
  fails (`array length must be a constant expression`) while the
  reversed order and the direct-array form in either order compile;
  fixed-order `len(p)` traps `null pointer dereference` at the operand
  span as specified.
- `grep` over `src/`: `value_compatible` now defined once in
  `src/types.rs` and used by semantic and IR verification;
  `Place::Indirect` constructed in the four expected lowering sites;
  `Instruction::Check` / `ValueKind::AddressOf` each have their single
  intended producer and consumer.
