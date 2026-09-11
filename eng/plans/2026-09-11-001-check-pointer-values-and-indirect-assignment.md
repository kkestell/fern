# Check pointer values and indirect assignment

## Sources

- `docs/spec.md#pointer-types`, `#the-null-pointer`, `#address-of`, and
  `#dereference` — pointer values, their contexts, and location rules.
- `docs/spec.md#implicit-dereference` and `#pointer-conversion-and-comparison`
  — implicit access, pointer compatibility, comparison, and `uint` conversion.
- `docs/spec.md#assignment` — locations, mutability, indirect targets, and
  compound-assignment evaluation.
- `eng/todo.md#pointers` — the third Pointers subtask; IR and native support
  remain later work.
- `eng/architecture.md#storage-and-identity` — checked facts cross the semantic
  to IR boundary.
- `eng/plans/2026-09-10-014-represent-pointer-types-across-the-compiler.md` —
  the resolved pointer type and current pre-IR staging boundary.

## Goal

Check Fern pointer values, addressable locations, dereferences, implicit pointer
access, and indirect assignment against the resolved pointer types. Pointer IR
and native execution remain unfinished; compilation continues to stop after a
successful semantic check with one staging diagnostic.

## Implementation

- `src/semantic/model.rs` — add checked expression facts for `null`,
  address-of, and dereference, retaining each operand expression ID for later
  lowering. Add a pointer-null constant so typed `null`, pointer zero values,
  and aggregate zero filling can remain semantic constants without pretending
  they have an existing IR literal. Replace the binding-rooted assignment-path
  representation with a checked location representation that can preserve an
  arbitrary dereference operand (including a call result) and index, field, and
  dereference steps in evaluation order.

- `src/semantic/expressions.rs` — teach contextual checking that `null` needs a
  pointer destination, remains untyped only until that destination is supplied,
  and is otherwise rejected. Centralize value compatibility so exact types and
  the one permitted `*T` to `*const T` conversion work uniformly for bindings,
  arguments, returns, assignments, struct fields, and array elements; reject a
  const-to-mutable or different-target conversion.

- `src/semantic/expressions.rs` — infer `&` only from a checked location,
  choosing `*T` or `*const T` from that location's mutability; infer `*` only
  from a pointer and produce its target type and dereference fact. Reuse the
  location checker for explicit dereference, address-of, and assignment rather
  than resolving a target name separately. It must preserve the source-order
  checking of every operand in an indexed, selected, or dereferenced location.

- `src/semantic/expressions.rs` — make field selection, indexing, and `len`
  accept exactly one pointer indirection, record that fact for later lowering,
  and keep their ordinary result types and diagnostics after dereferencing.
  Do not extend implicit dereference to iteration. Extend comparisons so only
  `==` and `!=` accept pointers with the same target type regardless of
  pointer constness, or one pointer and contextual `null`; mark every pointer
  comparison nonconstant. Permit non-truncating `uint(p)` for a concrete
  pointer and continue to reject every integer-to-pointer and truncating
  pointer conversion.

- `src/semantic/statements.rs` — use the checked location for simple and
  compound assignments. A mutable pointer target permits an indirect store
  regardless of whether the pointer binding itself is `var` or `const`; a
  `*const T` dereference and every location reached through it is immutable.
  Check the compound operation using the location's reached type, without
  evaluating the target a second time.

- `src/semantic/model.rs`, `src/semantic/statements.rs`, and
  `src/semantic/namespaces.rs` — materialize pointer zero values through the
  new null constant, including module-level declarations and filled struct or
  array values, and remove the current pointer-containing-type bypasses.
  Preserve the rule that module-level address-of is nonconstant and therefore
  invalid.

- `src/semantic/mod.rs` and `src/lib.rs` — remove the expression-level
  `pointers are not yet implemented` errors and broaden the one pre-IR staging
  check to recognize every checked pointer type or pointer expression, not only
  a written pointer annotation. Keep it after semantic checking and before IR
  lowering so valid pointer programs neither reach the current IR nor replace
  an existing compiler output.

- `src/ir/lower.rs` and its exhaustive checked-model matches — adapt the
  existing non-pointer assignment lowering to the new checked-location shape.
  Keep all pointer expression and location cases unreachable behind the
  pre-IR staging check; pointer operands, IR places, and instructions belong to
  the next Pointers subtask.

## Tests

- `src/semantic/tests/expressions.rs` — typed `null` in every contextual value
  position; pointer compatibility and its rejected reverse/target-mismatch
  cases; equality and inequality with another pointer or `null`; `uint(p)`;
  and rejection of contextless `null`, ordering, arithmetic, logical, and
  integer-to-pointer forms.

- `src/semantic/tests/expressions.rs` and `src/semantic/tests/statements.rs`
  — address mutable and immutable bindings, fields, elements, and dereferences;
  reject address-of of calls, literals, arithmetic, and other non-locations.
  Cover explicit and one-level implicit dereference for fields, indices, and
  `len`, including non-pointer operands and the preserved explicit form for
  iteration. Cover `*p`, `*const_p`, chained dereferences, `(*pp).field`, and
  call-result pointer targets under simple and compound assignment, asserting
  mutability and target/value diagnostic spans.

- `src/semantic/tests/annotations.rs` and `src/semantic/tests/statements.rs`
  — declarations without initializers, module-level typed `null`, and filled
  arrays or structs containing pointer fields produce recursive null zero
  values; module-level address-of still fails the constant-expression rule.

- `tests/compiler.rs` — a pointer program exercising values and indirect
  access reaches the sole post-semantic staging diagnostic and preserves an
  existing output. Keep a non-pointer compiler case proving the adapted target
  representation still lowers and runs.
