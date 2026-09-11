# Lay out, slice, and access slices in native code

## Sources

- `docs/spec.md#slices` — slice values and their copying, slicing expressions
  and their trapping bounds, slice indexing, `len`, and comparison.
- `docs/spec.md#loops` — `for … in` over a slice.
- `docs/spec.md#type-declarations` — a struct may reach itself through a slice
  field, and a struct is comparable when every field is.
- `eng/todo.md#slices` — the fifth Slices subtask. The staging diagnostic and
  the integrated slice programs remain after it.
- `eng/architecture.md#storage-and-identity` — `crate::layout` is the one
  derivation of size, alignment, and offsets; `backend::layout` owns QBE
  classes and scalar storage slots.
- `eng/plans/2026-09-11-010-represent-slices-in-fern-ir.md` — the verified IR
  forms this emission consumes: `Literal::EmptySlice`, `Place::SliceElement`,
  `ValueKind::WholeSlice`, `ValueKind::SliceRange`, and
  `ValueKind::SliceLength`.
- `eng/plans/2026-09-11-003-load-store-and-compare-pointers-in-native-code.md`
  — the shape this task follows: emission only, with the staging diagnostic
  left in place for the next subtask.
- `src/backend/layout.rs`, `src/backend/emitter.rs`, `src/backend/qbe.rs` — the
  existing QBE storage, class, static-data, trap, and comparison paths.

## Goal

Emit every verified slice form as native QBE: slice storage and classes, the
empty slice, whole-array and range construction with their bound traps, element
access with its index trap, `len`, copies through locals, parameters, results,
globals and aggregates, and slice equality. The compiler-level staging
diagnostic stays, so this does not complete the Slices parent task.

## Representation

A slice value occupies two words, both spelled `l` as a pointer already is: the
address of its first element at offset `0`, and its length at offset
`crate::layout::scalar_bytes(Scalar::Uint)`. That is the size and alignment
`crate::layout` already derives, so every path that treats an array or a struct
as an address to `blit` treats a slice the same way, and only construction,
projection, element access, and equality are new.

## Implementation

### `src/backend/layout.rs`

- `class` — a slice is `":slice"`.
- `array_class` — a slice element spells `format!(":arrayslice{count}")`. Every
  slice has the same layout, so the element type does not distinguish it.
- `collect_scalar_slots` — a slice is one leaf, `(base, ty.clone())`, as a
  pointer is. Do not descend into the element type.
- `TypeDefinitions` — add a `slice: bool`. `define` on a slice writes
  `type :slice = { l, l }` once and does not define the element type, because a
  slice stores its elements elsewhere. That is what keeps the definitions finite
  for a struct that reaches itself through a slice field.
- Add `pub(super) fn slice_length_offset(&self) -> u64`, returning
  `crate::layout::scalar_bytes(Scalar::Uint)`, so emission reads the one
  derivation of memory rather than restating the offset.

### `src/backend/qbe.rs`

- `operand` — `Literal::EmptySlice(_)` is `"$emptyslice"`, the one shared pair
  of zero words every empty slice reads.
- `data_item` — `Literal::EmptySlice(_)` is `"l 0, l 0"`. The global emitter
  already advances its written count by `layout.size`, so the two words land
  where the layout puts them.
- `Emitter` gains two fields: `helpers: String`, holding the slice-equality
  functions, and `slice_equalities: Vec<Type>`, the element type of each one by
  symbol number.
- `emit_aggregate_equal` — the `Type::Slice` arm calls the element type's
  equality function instead of inlining a loop:

  ```
  %aggregate{c}_equal =w call $sliceequal{n}(l {left}, l {right})
  ```

  and returns `%aggregate{c}_equal`. `{left}` and `{right}` are the addresses of
  the two slice values, as they already are for a struct or an array.

- Add `fn slice_equality(emitter: &mut Emitter<'_>, element: &Type) -> usize`:
  - Return the position of an existing entry in `emitter.slice_equalities`
    equal to `element`.
  - Otherwise push `element.clone()`, take its position as `n`, and emit the
    body. Register before emitting, so an element type that reaches the same
    slice type again finds the entry and the emission terminates.
  - Emit into a taken `emitter.text`, then restore the outer text and append the
    emitted body to `emitter.helpers`, so a helper can be produced while another
    function is being emitted and can itself need a further helper.
  - The body, with `{offset}` the length offset and `{stride}` the element size:

    ```
    function w $sliceequal{n}(l %left, l %right) {
    @start
        %leftbase =l loadl %left
        %rightbase =l loadl %right
        %leftlengthaddress =l add %left, {offset}
        %rightlengthaddress =l add %right, {offset}
        %leftlength =l loadl %leftlengthaddress
        %rightlength =l loadl %rightlengthaddress
        %samelength =w ceql %leftlength, %rightlength
        jnz %samelength, @loop, @unequal
    @loop
        %index =l phi @start 0, @next %nextindex
        %more =w csltl %index, %leftlength
        jnz %more, @body, @equal
    @body
        %offset =l mul %index, {stride}
        %leftelement =l add %leftbase, %offset
        %rightelement =l add %rightbase, %offset
        <element equality over %leftelement and %rightelement>
        jnz <element equality>, @next, @unequal
    @next
        %nextindex =l add %index, 1
        jmp @loop
    @equal
        ret 1
    @unequal
        ret 0
    }
    ```

    The element equality is `emit_aggregate_equal` for a slice, an array, or a
    struct element, and the existing scalar and pointer leaves otherwise. Its
    labels and temporaries all carry the `aggregate{c}` prefix, so they cannot
    collide with the helper's own names.

### `src/backend/emitter.rs`

- `emit` — write `data $emptyslice = { l 0, l 0 }` into `emitter.data` before
  the globals, and append `emitter.helpers` after `emitter.text` in the returned
  program.
- `place_address`, `Place::SliceElement { slice, index, span }` — take an
  `access` id, then with `spelling` the slice operand's type and `element` its
  element type:
  - load the base from the slice address and the length from
    `slice + slice_length_offset()`;
  - compare the index against that length with
    `comparison_operation("ge", false, Scalar::Int)`, which rejects a negative
    index and one at or past the length in one unsigned comparison, and trap
    through `emit_conditional_trap` with cause `"bounds"` and the message
    ``slice index out of range for `{spelling}` ``;
  - multiply the index by `layout.size(element)`, add it to the base, and return
    that address with the element type.

  Unlike an array element, a constant index is checked too: a slice's length is
  part of its value, not its type.
- `Instruction::Store`, the parameter stores in `emit_control_flow`, and the
  `ValueKind::Load` arm of `emit_value` — a slice joins the `blit` arm beside
  `Type::Array` and `Type::Struct`.
- `emit_control_flow`'s value-storage loop — allocate `%v{id}_storage` for a
  `WholeSlice` or a `SliceRange` value as well as for an aggregate `Load`. Both
  construct a slice value that needs storage of its own.
- `emit_value`, `ValueKind::WholeSlice(place)` — resolve the place, which yields
  an address and a `Type::Array { length, .. }`; store that address at
  `%v{id}_storage` and `length` at `%v{id}_storage + slice_length_offset()`;
  `%v{id} =l copy %v{id}_storage`. This cannot trap.
- `emit_value`, `ValueKind::SliceRange { slice, low, high }` — load the source
  base and length; emit two unsigned comparisons with
  `comparison_operation("gt", false, Scalar::Int)`, `low > high` and
  `high > length`, combine them with `or`, and trap through
  `emit_conditional_trap` with the value id, cause `"slicebounds"`, and the
  message ``slice bounds out of range for `{spelling}` `` rendered from
  `value.span`, where `spelling` is `value.ty`. Then store
  `base + low * layout.size(element)` and `high - low` into the two words of
  `%v{id}_storage` and copy the storage into `%v{id}`.
- `emit_value`, `ValueKind::SliceLength { slice }` — add the length offset to
  the slice address and `loadl` from it into `%v{id}`.
- Remove every `native slice emission is the next Slices task` arm in
  `src/backend/layout.rs`, `src/backend/emitter.rs`, and `src/backend/qbe.rs`;
  none remains after this task.

### `tests/compiler.rs`

The two slice staging tests keep their assertions. Rename them so they name the
boundary that actually remains — the compiler-level staging diagnostic, not
native emission.

## Tests

`src/backend/tests/slices.rs`, registered in `src/backend/tests/mod.rs`, built
on the existing `native_program_status`, `native_status`,
`assert_native_failure`, `lowered`, and `emit` helpers:

- A whole program slices an array local through `a[lo:hi]`, `a[lo:]`, `a[:hi]`,
  and `a[:]`, reads and writes elements through the result, and proves a write
  through the slice is visible in the array and in a second slice of it.
- `len` over a slice local, a slicing expression, a slice parameter, and a slice
  through `*[]int`; `len` of the empty slice is `0`.
- Slices are copied, not aliased as storage: assigning one slice binding to
  another, passing one as an argument, returning one, and storing one in a
  struct field and an array element all carry the same base and length, and
  `[]T` passes where `[]const T` is expected.
- A pointer operand: `p[lo:hi]` on `*[2]int`, slicing and indexing through
  `*[]int`, and a slicing expression over a call result of slice type.
- Nested slicing, `a[1:3][0:1]`, and a slice of an array of arrays reach the
  same elements as the equivalent direct slice.
- `for v in s`, `for v, i in s`, and `for v in a[1:]` visit each element once in
  index order; a body that assigns through the slice changes what a later
  iteration reads, and a body that reassigns the slice binding does not.
- Equality: two slices of different arrays holding the same elements are equal,
  differing lengths are unequal, two empty slices are equal, `[]int` compares
  with `[]const int`, and slices of structs, of arrays, and of floating-point
  elements compare element by element rather than by their stored bytes.
- A struct that reaches itself through a slice field compiles, compares, and
  runs, which is what the equality helper's memoization has to make terminate.
- A module-level `var s: []int;` emits `data $global0 = { l 0, l 0 }`, and a
  global struct with a slice field and a global `[2][]int` place their zero
  words at the layout's offsets; each program runs and observes `len` of `0`.
- The emitted QBE defines `type :slice = { l, l }` exactly once, before every
  definition that names it, when a signature takes, returns, or contains a
  slice; an array of slices spells `type :arrayslice{n} = { :slice {n} }`.
- One `$sliceequal` function is emitted per distinct element type, whatever the
  number of comparison sites and however deeply slices nest.
- Failures abort and report their message and source location: an index at the
  length, past it, negative, and into the empty slice; bounds with `lo > hi`,
  `hi > len`, and a negative `lo` on both an array operand and a slice operand;
  and a slicing expression or index through a null `*[2]int`, which still
  reports the null-pointer diagnostic.

## Decisions

- **A slice is a two-word aggregate in memory, not a pair of scalars.** An
  operand is one QBE temporary, so a slice that reached a call, a return, or a
  store as two values would need a second operand channel. Carrying it as the
  address of the two words it already occupies makes every copy the existing
  `blit`, and makes a slice field or element fall out of the struct and array
  layouts unchanged.

- **The empty slice is one shared `$emptyslice` data symbol.** An `EmptySlice`
  literal reaches a store, a call argument, a return, and every slice operand
  position, and `operand` spells a literal without the emitter in hand. Nothing
  writes through a slice value's own storage — assignment writes through places,
  and an element place resolves to storage the slice refers to — so one shared
  pair of zero words serves every use. Indexing it traps on its zero length,
  which is what indexing an empty slice must do.

- **Slice equality is a function per element type, not inlined.** A struct may
  reach itself through a slice field, and every such struct is comparable, so
  inlining the element comparison the way an array's is inlined would not
  terminate at emission. One function per element type, registered before its
  body is emitted, turns that into a run-time recursion over finite data.

- **Every slice index and both slicing bounds are checked at run time.** A
  slice's length is part of its value, so the constant-index elision an array
  place uses has nothing to read. `0 <= lo <= hi <= len` is two unsigned
  comparisons, because a negative bound is a huge unsigned value and the length
  is never negative.
