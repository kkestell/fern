# Repair the general codebase review findings

## Sources

- `eng/reviews/2026-09-10-008-general-codebase.md` — the findings this plan
  repairs.
- `docs/spec.md` "Variable declarations", "Module-level declarations" — a
  declaration without an initializer requires an annotation and takes the
  type's zero value.
- `eng/architecture.md` "IR lowering" paragraph and the `backend::layout`
  paragraph that follows it — the layout ownership rule this plan moves.

## Goal

Repair the four confirmed findings and the two latent invariants the review
records as suspicions. Three defects are independent; do them in the order
below, because the aggregate-bound work and the zero-value work both call the
layout derivation that part 1 introduces. Nothing here is a feature slice, so
`eng/todo.md` is untouched.

Two review claims need correcting as you go:

- Finding 4 names two duplicate layout derivations. There are three:
  `src/semantic/annotations.rs:98`, `src/ir/verify.rs:351`, and
  `src/backend/layout.rs:102-125`, each with its own `align` helper.
- Finding 2 reports the struct path as correct. It is not. `zero_value` is
  total-aware per call, but `check_struct_literal`
  (`src/semantic/expressions.rs:335-352`) calls it once per omitted field, so
  sibling fields multiply. A struct of four `[900000]u8` fields filled with
  `...` expands 3,600,000 scalars and is accepted today.
- Finding 3 claims it also resolves the linear-chain stack overflow from
  `eng/reviews/2026-09-10-007-structs-milestone.md`. That is already fixed:
  `MAX_STRUCT_CONTAINMENT_DEPTH` rejects a 20,001-struct chain with a source
  diagnostic. Do not plan work for it.

## Implementation

### 1. One memoized layout derivation (findings 3 and 4)

- `src/layout.rs` — new crate-level module; add `mod layout;` to `src/lib.rs`
  beside the existing phase modules. It owns:
  - `pub(crate) fn scalar_bytes(scalar: Scalar) -> u64` — `8` when
    `scalar.width() == 64`, otherwise `4`. This is the single definition;
    `backend::qbe::bytes` becomes a call to it.
  - `pub(crate) struct StructLayout { pub size: u64, pub alignment: u64, pub
    offsets: Vec<u64> }`.
  - `pub(crate) trait StructFields { fn field_count(&self, id: StructId) ->
    usize; fn field_type(&self, id: StructId, ordinal: usize) -> &Type; }` —
    each phase stores fields differently, so the table is read by ordinal
    rather than as a slice.
  - `pub(crate) struct Layouts` holding `RefCell<HashMap<StructId,
    Option<Rc<StructLayout>>>>`, with `Default`. Methods take `&self` so the
    backend's existing `&self` call sites stay unchanged:
    - `size(&self, table: &dyn StructFields, ty: &Type) -> Option<u64>`
    - `alignment(&self, table: &dyn StructFields, ty: &Type) -> Option<u64>`
    - `struct_layout(&self, table: &dyn StructFields, id: StructId) ->
      Option<Rc<StructLayout>>`
  - A private `align(offset: u64, alignment: u64) -> Option<u64>` using
    checked arithmetic, replacing the three existing copies.
- Derivation rules, matching what the three copies agree on today: a scalar's
  size and alignment are both `scalar_bytes`; an array's size is
  `length.checked_mul(element_size)` and its alignment is the element's; a
  struct starts each field at the next offset its own alignment allows, takes
  the maximum field alignment (minimum `1`), and pads the total out to that
  alignment. Every step uses `checked_add`/`checked_mul`; `None` propagates
  and means the layout overflows `u64`.
- `struct_layout` returns the cached entry when present, otherwise derives it
  once from the table and caches the result, including a cached `None`. Field
  lookups go back through `size`/`alignment`, so a struct field hits the cache
  and total work is linear in the number of fields in the program. Recursion
  over struct containment stays bounded by `MAX_STRUCT_CONTAINMENT_DEPTH`.
- Caching a struct is only sound once its fields are final. In the semantic
  phase that means a struct in `FieldState::Resolved`; `resolve_annotation`
  already calls `resolve_struct_fields` before `validate_aggregate_layout`, so
  no call site changes. State this invariant in the module doc comment.
- `src/semantic/annotations.rs` — delete
  `CheckedProgram::aggregate_layout` and the local `align`.
  `validate_aggregate_layout` calls `self.layouts.size(self, ty)` and keeps
  its existing `is_none_or(|size| size > MAX_AGGREGATE_LAYOUT_BYTES)`
  diagnostic. Add a `layouts: Layouts` field to
  `CheckedProgram` (`src/semantic/model.rs`) and implement `StructFields` for
  it over `self.structs[id.0].fields[ordinal].ty`.
- `src/ir/verify.rs` — delete the local `aggregate_layout` and `align`. Hold a
  `Layouts` on the verifier and implement `StructFields` over
  `self.structs[id.0].fields[ordinal]`. `verify_aggregate_layout` keeps both
  existing internal-compiler-error messages.
- `src/backend/layout.rs` — delete `struct_layout`, `struct_offsets`, and the
  local `align`. `Layout` gains a `Layouts` field, implements `StructFields`
  over `self.structs[id.0].fields[ordinal]`, and its `size`, `alignment`, and
  `field` methods read the shared derivation. Verification has already
  rejected overflowing layouts, so the backend may `expect` on `None`; use the
  message `"verified IR has a representable layout"`.
- `eng/architecture.md` — the `backend::layout` paragraph now describes a
  crate-level derivation. Rewrite it to say that `layout` derives byte size,
  alignment, and field offsets once per struct from a phase's struct table,
  and that semantic validation, IR verification, and the backend all read that
  one derivation. Keep QBE class and scalar-slot naming with
  `backend::layout`, which still owns them.

### 2. Bound aggregate expansion by total scalar count (finding 2, low finding)

- `src/semantic/model.rs` — make `aggregate_value_count` `pub(super)`.
- `src/semantic/expressions.rs:181-188` — replace the check against
  `length` with a total-aware one over the
  literal's whole type: reject when `self.aggregate_value_count(&ty)` is
  `None` or exceeds the limit. Keep the existing message and span. This check
  already runs before the element loop, so an outer literal is rejected before
  any inner fill expands.
- `src/semantic/expressions.rs:335-352` — the omitted-field loop must bound
  the fill once across all omitted fields rather than once per field. Sum
  `aggregate_value_count` over the omitted fields with `checked_add`, reject
  against `MAX_AGGREGATE_INITIALIZER_VALUES` with the existing message and the
  `name.span` span before constructing any zero value, then build the zeroes
  with `zero_value_unchecked`. Leave `zero_value` as it is; other callers
  still need it.
- `README.md:83-85` — the sentence now describes the enforced bound. Keep the
  wording as a total across one initializer.

### 3. Declarations without an initializer (finding 1)

- `src/frontend/syntax.rs:104` — `initializer: Option<Idx<Expression>>`.
- `src/frontend/parser.rs:794-816` (`binding`) — consume `=` and an expression
  only when `Token::Equals` follows. When it does not, require the annotation
  to be present and report `"a declaration without an initializer requires a
  type annotation"` at the name span when it is absent. Keep the existing
  ``expected `=` `` message for no other case; the statement's
  trailing `;` is already consumed by the caller.
- `src/semantic/statements.rs:116-140` (`check_binding`) — take
  `Option<Idx<Expression>>`. With `None`: the annotation is present, so
  resolve it with `self.resolve_annotation(annotation, scopes, None)?`,
  compute `self.zero_value(&ty, span)?` over the statement's span, allocate
  the `Binding` with `constant: Some(zero.clone())` when the declaration is
  `const` and `None` when it is `var`, and record the zero value for lowering
  as described below.
- `src/semantic/model.rs` — add `zero_declarations: HashMap<Idx<Statement>,
  Constant>` to `CheckedProgram`, holding the zero value of every declaration
  written without an initializer. A `var` cannot carry its value in
  `Binding::constant`, which means "folds into its use sites", so lowering
  needs this map.
- `src/semantic/namespaces.rs:608-656` (`check_module_initializer`) — with no
  initializer, skip the `find_call` probe and the constant-expression check,
  resolve the annotation, and fill `bindings[binding].ty`,
  `bindings[binding].constant` (only for `const`), and `zero_declarations` the
  same way. `src/semantic/namespaces.rs:575-588` (`module_dependencies`) must
  collect references from the annotation only when the initializer is absent.
- `src/ir/lower.rs:52-68` — a module-level `var` without an initializer takes
  its constant from `checked.zero_declarations[&statement]` instead of
  `checked.expressions[initializer].constant`; `flatten` then produces the
  global's values unchanged.
- `src/ir/lower.rs:333-351` (`lower_flow_binding`) — with no initializer,
  allocate the local and call the existing `store_constant` with the local's
  place, the binding's type, `checked.zero_declarations[&statement]`, and the
  statement's span. `store_constant` already handles scalars, arrays, and
  structs.
- `src/ir/lower.rs:339` and `src/ir/lower.rs:673-682` — update the remaining
  `StatementKind::Binding { initializer, .. }` destructurings for the new
  `Option`. A `for` header's three-clause initializer statement may itself be
  a declaration without an initializer and needs no special case.

### 4. Local invariants (the review's first two suspicions)

- `src/source.rs:56-60` — `index_at` returns `FileId(0)` when `self.files` is
  empty rather than underflowing, or asserts the map is non-empty. Keep the
  existing comment's invariant.
- `src/semantic/annotations.rs:148-177` (`resolve_struct_fields`) — every `?`
  path leaves `self.file` swapped, `struct_containment_depth` incremented, and
  the struct in `FieldState::Resolving`, not just the duplicate-field path the
  review names. Restore all three on the error paths, either with a guard type
  or by capturing the loop's result and restoring before returning it.

## Tests

- Parser: `var x: int;` and `const x: int;` parse with no initializer; `var
  x;` reports the missing-annotation diagnostic; `var x: [_]int;` still
  reports `` `[_]` requires an array-literal initializer ``.
- Semantic: a local and a module-level declaration without an initializer take
  the zero value for a scalar, an array, and a struct; a `const` without an
  initializer folds its zero into use sites; assigning to a `const` declared
  without an initializer is still rejected.
- IR: snapshots for a module-level `var` without an initializer (a global of
  zeroes) and a local aggregate declaration without an initializer (stores of
  zero through `store_constant`).
- Native: the review's finding 1 reproduction — a module-level `var counter:
  int;`, incremented in `main`, exits `2` — plus a struct-typed local
  declared without an initializer, compared for equality against an
  explicitly zeroed literal.
- Aggregate bounds: `var x: [4][4][300000]u8 = [[[0...]...]...];` is rejected
  with the initializer-limit diagnostic; a struct of four `[900000]u8` fields
  written `S { ... }` is rejected with the same diagnostic; a nested literal
  whose total stays under the limit is still accepted.
- Layout: one test that a binary struct tree 40 levels deep compiles without
  timing out, which fails by hanging before the memoization and passes after.
  Keep the struct shape small enough that only layout runs — declare the tree
  and use it as a function parameter type, with no value of that type.

## Decisions

- The shared derivation is a crate-level `src/layout.rs`, not a member of any
  phase, because three phases need it and `eng/architecture.md` requires one
  description of memory. `backend::layout` keeps QBE class naming and scalar
  slots, which are genuinely backend concerns.
- `Layouts` memoizes behind a `RefCell` so its accessors take `&self`. The
  alternative, threading `&mut` through the backend emitter, would touch every
  call site for no behavioral gain.
- Zero values for declarations without an initializer live in a
  `zero_declarations` map keyed by the declaration statement rather than in
  `Binding::constant`, whose `Some` already means "this binding folds into its
  use sites" and must stay `None` for a `var`.

## Extra validation

- Time the deep-struct compile before and after part 1. The current cost is
  25.3 s at depth 30 in the semantic phase alone and quadruples every two
  levels; after the change it should be flat.
