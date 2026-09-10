# Represent arrays in Fern IR

## Sources

- `docs/spec.md#arrays` — array values copy on initialization, assignment,
  argument passing, and return
- `docs/spec.md#indexing`, `docs/spec.md#indexing-and-lengths` — element
  access and the trapping bounds rule
- `docs/spec.md#length` — the length comes from the type, the operand is still
  evaluated
- `docs/spec.md#array-comparison` — elementwise `==` and `!=` only
- `docs/spec.md#loops` — `for … in` captures the array once and rebinds `v`
  and `i` each iteration
- `eng/roadmap.md#arrays` — fifth task
- `eng/plans/2026-09-09-010-check-indexing-length-comparison-and-iteration.md`
  — `ExpressionValue::Index`/`Length`, `IterationBindings`, `Constant::Array`

## Goal

Lower and verify arrays in Fern IR. Native emission stays unfinished: the
`semantic::reject_array_values` call in `src/lib.rs` remains, so no array
reaches the backend, and `src/backend.rs` changes only enough to keep
compiling against the new IR types.

## Implementation

### `src/types.rs`

- `Type::leaf(&self) -> Scalar` — the scalar under any nesting.
- `Type::element_count(&self) -> u64` — the product of the lengths, `1` for a
  scalar. Both have callers in IR verification below.

### `src/ir.rs` — representation

- Replace `Scalar` with `Type` in `Value::ty`, `ControlFlow::locals`,
  `Function::result`, and `Global::ty`. `Operand::Integer` keeps its `Scalar`.
  `Place` and `ValueKind` stop being `Copy`; take them by reference where the
  compiler now requires it.
- `Global` becomes `{ ty: Type, values: Vec<i128> }` holding the elements in
  memory order, one entry for a scalar. Flatten `Constant::Array` recursively
  in `lower`.
- Add `Place::Element { base: Box<Place>, index: Operand, span: Range<usize> }`.
  The span is the one the bounds-check trap reports, so a place is enough to
  emit the check and the access together. Add
  `Place::root_local(&self) -> Option<LocalId>` and use it in `local_stores`
  and the uninitialized-load check in `verify_instruction`: a store through an
  element place initializes its root local, since an array local is
  initialized elementwise.
- A whole-array value is `Load(place)`, and `Instruction::Store` of an
  array-typed operand is the whole-array copy. No new instruction: load, store,
  call arguments, call results, and `Return` all carry array types the same way
  they carry scalars.

### `src/ir.rs` — verification

- `place_type` recurses and takes the block's `operand_type` closure: an
  `Element` base must be `Type::Array`, its index must be `Scalar::Int`, and
  the result is the element type. Routing indices through `operand_type` also
  enforces that an index value is defined earlier in the same block.
- `valid_value`: `Comparison` accepts array operands only for
  `ComparisonOperator::Equal` and `NotEqual`. `Convert`, `Unary`, `Binary`, and
  `LogicalNot` require scalars — add one
  `fn scalar(ty: &Type) -> Result<Scalar, CompileError>` helper reporting an
  array reaching integer work, rather than a check per arm.
- Global verification checks `values.len() == ty.element_count()` and each
  value against `ty.leaf()`.
- Store, call argument, call result, and return checks compare `Type`s, which
  is also the copy-size check.

### `src/ir.rs` — lowering

- `scalar_type` goes away; lowering carries `Type` and only reaches for a
  scalar where the IR still requires one.
- `lower_array_place(checked, id, bindings, builder) -> Place` gives the place
  of an array-typed expression: a constant expression materializes into a fresh
  local (see Decisions); an array literal stores its elements into a fresh
  local; `Reference` uses `bindings`; `Index` builds an `Element` place over
  its operand's place; `Grouping` recurses; a `Call` stores its result into a
  fresh local. Nested literals recurse into the element places.
- `lower_flow_operand` handles an array-typed expression by loading
  `lower_array_place`, handles `Index` by loading the element place, and
  handles `Length` **before** the constant shortcut: lower the operand for its
  effects, then yield the length as an `int` operand. The shortcut would
  otherwise skip a bounds-check trap inside `len(a[i])`, which the checker
  folds.
- Change `HeldOperand::read` to `&mut self` and memoize the local it spills
  into, so one held operand can be read in more than one block. Add
  `HeldPlace` — a root `Place` plus held index operands with their spans —
  whose `read` rebuilds the place in the current block.
- Assignment and compound assignment build a `HeldPlace` from the target's
  indices *before* lowering the value, in source order, and read it back
  afterwards. The compound form's place type is the binding's type descended
  through the indices.
- `lower_for` keeps one set of condition/body/post/after blocks for every
  header. Introduce a `LoopKind` computed before the blocks — the existing
  clause form, or an iteration form holding the captured place, length,
  counter local, value binding, element type, and operand span — and branch on
  it in three places: the setup before the loop, the condition block, and the
  post block, plus one store at the top of the body. Iteration always has a
  post block, so `continue` increments.
- Iteration setup copies the operand into a fresh local and initializes an
  `int` counter to 0. The condition compares the counter against the length.
  The body's first instruction stores `Load(Element{captured, counter})` into
  the value binding's local. The post block stores counter + 1. The synthesized
  comparison and addition carry the operand's span, which the verifier
  requires.

### `src/semantic.rs`

- Delete the `expect(dead_code)` attributes on `IterationBindings::value` and
  `index`; lowering reads both.

### `src/backend.rs`

- Mechanical only: `qbe_type` takes a `&Type` and expects a scalar,
  `Place::Element` is unreachable, and the data emission writes `Global::values`.
  Name the next roadmap task in each expect. `reject_array_values` is what keeps
  these unreachable.

## Tests

`src/ir.rs`, extending `lowered_programs` and adding focused tests:

- A module-level `var` array global holds its flattened elements; a nested
  array global flattens in memory order.
- A `const` array used twice materializes into a local at each use.
- `var b = a; b[0] = 99;` lowers to a store of a load and then an element
  store, leaving `a`'s place untouched.
- An array literal with a fill stores the last element into every remaining
  slot; a nested literal stores through nested element places.
- `a[i][j]` builds nested element places carrying the source spans.
- `a[i] = f();` and `a[i] += f();` lower the index before the call, and the
  place is read back in the block the call leaves.
- `len(a)` folds to the length with no instructions; `len(f())` still emits the
  call; `len(a[i])` still emits the bounds-checked access.
- `for v in a` copies the operand once before the loop and reads elements from
  the copy; `for v, i in grid` binds the index to the counter; a `continue`
  inside it still increments.
- An array parameter, an array result, and an array argument each lower to
  array-typed locals, operands, and a call result.
- Verification rejects: an element place over a scalar base, a non-`int` index,
  a store between different array types, arithmetic on an array value, `<` on
  arrays, and a global whose values do not match its type.
- Snapshot fixtures for the milestone example's shapes: nested arrays, an
  aggregate call, and both `for … in` forms.

Every `fern__ir__tests__*.snap` file changes, because locals, results, and
globals now print `Type`. Regenerate them rather than editing by hand.

## Decisions

- **Bounds checks live in the place.** `Place::Element` carries the span, so
  every read and write through it emits one check. The element read inside
  `for … in` is in range by construction but still carries a check; a second
  unchecked path for it would be a path-selection flag for one caller.
- **Constant arrays materialize into a local at each use.** A folded array
  literal clears its elements' constants during checking, so lowering an array
  literal reads the folded `Constant` when there is one and the element
  expressions otherwise. Only a module-level `var` becomes static data.
- **A fill evaluates its last element once** and stores that value into every
  remaining slot.

## Extra validation

- `scripts/code-health.sh check` — `lower_for`, `valid_value`, `place_type`,
  and `lower_flow_operand` all gain branches, and `initialized_locals` is
  already at 13.
