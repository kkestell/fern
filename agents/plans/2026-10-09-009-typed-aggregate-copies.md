# Copy small aggregates field by field

## Goal

Every struct parameter, local, and loaded aggregate value lives in a stack
slot, and every copy into or out of one is a `blit`. QBE expands a `blit` into
integer word loads and stores, so a later `loadd` of a field does not see the
`storel` that wrote it, and the slot and every field read stay in memory. QBE
does forward a typed store to a typed load at a constant offset from an
`alloc`, and drops the slot when nothing else reads it.

Copy a small aggregate with one typed load and store per scalar and pointer
in its layout, so QBE keeps its fields in registers. A prototype of this
change took `examples/raytrace.fern` from 17.1 to 13.5 CPU seconds with
byte-identical output.

## Related code

- `src/backend/emitter.rs:28` — `MAX_INLINE_COPY_BYTES` and `emit_copy`, which
  emits `blit` up to 64 bytes and `memcpy` above. Its four callers are
  `emit_store`, the external aggregate result copy in `emit_call`, the
  aggregate `return` in `emit_terminator`, and the aggregate `Load` in
  `emit_value`; each has the copied value's type.
- `src/backend/emitter.rs:1` — the module comment says aggregates are copied
  with blit or memcpy.
- `src/backend/layout.rs:85` — `value_slots`, the offset and type of every
  scalar and pointer in a type, in memory order.
- `src/backend/qbe.rs:55` — `value_class`; `src/backend/qbe.rs:292` —
  `memory_class` and `load_operation`, which already pick the load and store
  for each scalar width.
- `src/backend/qbe.rs:97` — `emit_aggregate_equal` compares field by field, so
  padding never affects `==`.
- `docs/spec.md:96` — copies cover all fields and elements of aggregates; the
  spec says nothing about padding bytes, so a copy that skips them conforms.

## Decisions

- `emit_copy` takes the `Emitter` and the copied `Type` instead of a `String`
  and a size. It needs the layout for `value_slots` and the `accesses` counter
  for fresh temporaries.
- A copy is field-wise when its type has at most `MAX_FIELD_COPY_SLOTS = 8`
  slots. Eight slots of at most eight bytes each is never more code than the
  eight word copies of the largest `blit`, so the field-wise copy does not grow
  code beyond what `blit` emits today. A type with more slots keeps the current
  rule: `blit` up to `MAX_INLINE_COPY_BYTES`, `memcpy` above. This keeps byte
  arrays such as `[64]u8` on `blit` rather than 64 byte copies.
- Each slot copy names its temporaries `%access{n}_from`, `%access{n}_to`, and
  `%access{n}_value`, drawing `n` from `emitter.accesses` like `place_address`.
- Padding bytes are not copied. Nothing in Fern reads padding except through a
  pointer cast, and the spec does not promise its contents.

## Test plan

- `src/backend/tests/mod.rs`: `small_aggregates_copy_field_by_field` — a
  function taking and returning a `type V struct {x:f64,y:f64,z:f64}` emits
  `loadd` and `stored` for the parameter copy and no `blit`; a struct with
  more than eight slots but at most 64 bytes, such as `{b:[16]u8}`, still
  emits `blit`; a struct larger than 64 bytes still calls `memcpy`.
- `tests/core.rs`: `field_wise_copies_keep_every_width` — a struct holding
  `bool`, `i8`, `u16`, `i32`, `f32`, `f64`, a pointer, and a nested
  `[2]i16` array, with padding between fields, survives parameter passing,
  return, assignment, and a load into a local; every field compares equal
  after each copy, including negative narrow integers and the pointer's
  target.
- The existing `c_abi_structs` test covers the external result copy, and
  `examples_execute` covers `examples/raytrace.fern`.

## Implementation plan

1. In `src/backend/emitter.rs`, add `MAX_FIELD_COPY_SLOTS` beside
   `MAX_INLINE_COPY_BYTES` and change `emit_copy` to
   `emit_copy(emitter: &mut Emitter<'_>, source: &str, destination: &str, ty: &Type)`.
   When `value_slots(ty)` has at most `MAX_FIELD_COPY_SLOTS` entries, emit for
   each slot an `add` of its offset to `source` and to `destination`, a
   `load_operation` load with `value_class`, and a `store` with
   `memory_class`. Otherwise emit `blit` or `memcpy` by size as today.
2. Update the four callers to pass the emitter and the type.
3. Update the module comment and the comment on `MAX_INLINE_COPY_BYTES` to say
   small aggregates are copied field by field so QBE can keep their fields in
   registers.
4. Add the tests above.
5. Rebuild in release, compile `examples/raytrace.fern`, and time it against
   the current `main` build with `/usr/bin/time -p`. Record both CPU times in
   the work log.

## Documentation updates

- `agents/bugs.md`: rewrite the "Small functions over structs are slow" entry
  to describe what remains after step 5's measurement: no call is inlined,
  aggregate results return through memory, and aggregates over the field-wise
  limit, such as `Hit`, are still copied with `blit` or `memcpy`. State the
  new render time. Remove the sentences about every copy being a `blit`.
