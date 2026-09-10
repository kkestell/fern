# Compile and execute array programs

## Sources

- `eng/roadmap.md#arrays` — the sixth task, the milestone program, and the
  completion gates
- `docs/spec.md#arrays` — copying on initialization, assignment, argument
  passing, and return
- `docs/spec.md#indexing-and-lengths` — `0 <= i < len`, an out-of-range index
  traps, indices are signed
- `docs/spec.md#array-comparison` — `==` and `!=` compare elementwise in any
  order
- `eng/plans/2026-09-09-011-represent-arrays-in-fern-ir.md` —
  `Place::Element`, whole-array `Store`, flattened `Global::values`
- `src/backend.rs` — `place`, `scalar`, `emit_conditional_trap`,
  `operation_message`, `assert_native_failure`

## Goal

Emit native code for the verified array IR and finish the milestone: locals,
globals, parameters, results, element access with a trapping bounds check,
whole-array copies, and array equality. Removing `semantic::reject_array_values`
lets arrays reach the backend for the first time, so this is the milestone's
completion boundary.

## Implementation

### Layout

- Every scalar occupies its QBE word, so an array of `n` leaf scalars occupies
  `n * 8` bytes when `qbe_type(ty.leaf())` is `l` and `n * 4` bytes otherwise.
  Add `fn size(ty: &Type) -> u64` and `fn allocation(ty: &Type) -> &str`
  (`alloc8`/`alloc4`) beside `qbe_type`, and use them for locals so scalars and
  arrays allocate through one path.
- Add `fn class(ty: &Type) -> String`: the base type for a scalar, and
  `:array{word}{count}` for an array, for example `:arrayl6` for `[2][3]int`.
  Signatures, call arguments, and call results all name a type through it.
- `emit` collects the distinct array types appearing as a parameter local or a
  result across every function and writes `type :arrayl6 = { l 6 }` for each,
  ahead of the data and text sections.
- `Emitter.qbe_result` becomes `Option<String>`; `main` keeps `"w"`.

### Places and access

- Replace `fn place` with
  `fn place_address(emitter, flow, place) -> (String, Type)`, returning the
  address and the type stored there. `Local` and `Global` return `%local{n}` /
  `$global{n}` with the local's or global's type and emit nothing. `Element`
  materializes its base, then emits the bounds check and
  `%access{n}_address =l add {base}, %access{n}_offset` where the offset is the
  index times `size(element)`.
- The bounds check is `%access{n}_out =w cugel {index}, {length}`, using the
  base type's length. An unsigned comparison rejects a negative index in the
  same instruction. Pass it to `emit_conditional_trap` with cause `"bounds"` and
  the message `` array index out of range for `[3]int` ``, rendered from the
  place's span. `Emitter` gains an `accesses: usize` counter naming the temps
  and the trap; the cause distinguishes its symbols from the value-numbered
  ones.
- `operation_message` takes `Option<&Range<usize>>` instead of `&Value`, since a
  place has a span but no value.

### Values and instructions

- An array-typed value is a pointer to its storage. The IR only produces two:
  `ValueKind::Load` and `ValueKind::CallResult`.
  - `emit_control_flow` allocates `%v{id}_storage` for every array-typed `Load`
    in `function.values`, beside the locals in `@start`. The load emits
    `blit {address}, %v{id}_storage, {size}` and `%v{id} =l copy %v{id}_storage`,
    so an array value is a copy that a later call cannot mutate.
  - A `CallResult` needs no storage: `%v{id} =:arrayl3 call $fn1(...)` gives a
    pointer to a caller slot QBE owns.
- `Instruction::Store` of an array operand is
  `blit {operand}, {address}, {size}`; a scalar keeps `store{w}`. Both take the
  address and type from `place_address`.
- A parameter local of array type is initialized with
  `blit %param{index}, %local{index}, {size}` rather than a store.
- An array-typed `Return` operand needs no change: `ret` of the pointer copies
  the aggregate.
- `emit_value` and `emit_comparison` branch on an array before reaching
  `scalar`. An array `Comparison` emits
  `%v{id}_diff =w call $memcmp(l {left}, l {right}, l {size})` followed by
  `ceqw`/`cnew` against 0.
- A global's data emits one item per element,
  `data $global0 = { l 1, l 2, l 3 }`, with the leaf's base type on each item.
- Update the doc comment on `fn scalar` to say arithmetic and conversions are
  scalar-only, rather than naming this task.

### Removing the guard

- Delete `semantic::reject_array_values`, its call in `src/lib.rs`, and the
  `"arrays are not yet compiled"` case in `source_failures_preserve_output`.

### Documentation and fixtures

- `examples/arrays.fern` — a topic file in the style of the other examples: no
  comments, exit status 42. Cover array and nested array types, `[_]`, a fill
  literal, indexing and element assignment, `len`, `==`, a whole-array copy, an
  array parameter and result, a module-level array, and both `for … in` forms.
- `README.md` — an `Arrays` bullet in the tour list, after `Functions`.
- `tests/fixtures/programs/arrays.fern` — the roadmap's milestone program
  verbatim.

## Tests

In `src/backend.rs`, over emitted text:

- One `type :` definition per distinct aggregate layout, emitted before the
  functions that name it, and none for a program without array signatures.
- An array global emits one data item per element in memory order.
- Every `alloc` in a function appears in `@start`, including for a program whose
  loop body copies an array and calls an array-returning function.

With `assert_native_failure`:

- A runtime index at the length and a negative runtime index each report
  `` array index out of range for `[3]int` ``; two failing accesses in one
  function link separately.
- `len(a[i])` with an out-of-range `i` still traps.

In `tests/compiler.rs`, executing programs:

- Element reads and writes on nested arrays, and on an array reached through a
  parameter, a result, and a module-level `var`.
- Whole-array copies stay independent for initialization, assignment, argument
  passing, and return: mutating one leaves the other unchanged, including
  mutating a parameter inside the callee.
- `==` and `!=` on nested arrays, on `i8` arrays holding negative elements, and
  on `bool` arrays, comparing values built by arithmetic against literal ones.
- An expression that mutates a module-level array after the array is passed:
  `f(g, mutate())` passes the value `g` held before `mutate()` ran.
- Both `for … in` forms, including over a call result and with `continue`, and a
  loop of 100_000 whole-array copies that returns rather than exhausting the
  stack.
- `len` of a `var` array as an exit status and as a constant.
- `tests/fixtures/programs/arrays.fern` compiles and exits 46.

## Decisions

- **A scalar occupies a full QBE word inside an array.** `[3]u8` is 12 bytes.
  Packing would need width-specific loads and stores at every element access,
  which is a second representation for the same scalars the locals already
  store word-wide. Nothing observable in the language depends on the layout.
- **Equality is `memcmp` over the flattened storage.** Every stored scalar is
  normalized to a canonical word by `emit_truncation_operand`, so equal arrays
  hold identical bytes and there is no padding. The spec leaves element
  comparison order unobservable.
- **Every allocation is emitted in `@start`.** A QBE `alloc` in a loop body
  allocates once per iteration; a million-iteration loop segfaults. This is why
  array load storage is allocated up front by value ID.
- **The bounds check is emitted where the place is materialized**, so a store
  checks its index after the value is computed. The spec fixes the evaluation
  order of the target's indices, which lowering already established, and both
  outcomes are traps.

## Extra validation

- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, and `scripts/code-health.sh check`; this task integrates the
  milestone.
- Walk the roadmap's completion gates and confirm each has a test.
- Review the milestone once for correctness and simplicity: look for
  path-selection between scalars and arrays that one type-driven helper could
  own, and for guards left behind by earlier tasks.
