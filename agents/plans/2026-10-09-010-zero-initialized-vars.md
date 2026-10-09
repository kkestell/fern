# Zero-initialize a `var` without an initializer

## Goal

`docs/spec.md#declarations-and-scope` lets a `var` with a type annotation omit
its initializer and hold its type's zero value; a local is zeroed each time its
declaration executes, and a `const` still requires an initializer. The parser
rejects every binding without `=` today.

Accept the form for module-level and local `var`s. Zeroing costs no compile
memory or static data proportional to the value's size: a module-level zero
`var` is all-zero static data, and a local is zeroed in place by stores or one
`memset` call.

## Related code

- `src/frontend/parser.rs:791` — `binding` requires `=` and an initializer.
- `src/frontend/syntax.rs:103` — `StatementKind::Binding` holds
  `initializer: Idx<Expression>`.
- `src/semantic/statements.rs:100` — `check_binding` checks a local binding's
  initializer and records its type.
- `src/semantic/namespaces.rs:623` — `module_dependencies` collects the
  references of a module-level binding's annotation and initializer;
  `src/semantic/namespaces.rs:667` — `check_module_initializer` checks one
  module-level binding.
- `src/ir/lower.rs:48` — `lower` turns each module-level `var` into a `Global`
  whose `values` come from `static_values` of the initializer's constant.
- `src/ir/lower.rs:520` — `lower_flow_binding` stores a local's initializer
  into its place; `binding_place` picks the result storage for the returned
  local.
- `src/ir/model.rs:161` — `Instruction`; `Global.values == None` already means
  every byte is zero.
- `src/ir/verify.rs:482` — `Store` verification through `destination_type`,
  which rejects writes to a constant; `src/ir/verify.rs:725` —
  `record_definition`.
- `src/backend/emitter.rs:29` — `MAX_INLINE_COPY_BYTES`,
  `MAX_FIELD_COPY_SLOTS`, and `emit_copy`, whose tiers zeroing mirrors;
  `src/backend/emitter.rs:96` — `emit_data` writes a `Global` with no values
  as one `z` item; `src/backend/emitter.rs:427` — `emit_instruction` and
  `emit_store`.
- `src/backend/layout.rs:85` — `value_slots`, which builds one entry per
  scalar, so it must not run on a large type.
- `examples/raytrace.fern:640` — `render_rows` mallocs its row and scratch
  buffers; `examples/raytrace.fern:662` — `main` mallocs `readers` and spells
  `var ends: [2]i32 = [0, 0];`.

## Decisions

- `StatementKind::Binding.initializer` becomes `Option<Idx<Expression>>`. The
  parser accepts `var` or `const` with or without `= e`; semantic checking
  rejects a `const` without one ("a `const` requires an initializer") and a
  `var` without one or an annotation ("a `var` without an initializer requires
  a type annotation"), at the statement span, for locals and module-level
  bindings alike.
- No semantic `Constant` represents a zero value. A zero `var`'s binding has its
  annotated type and `constant: None`, and no expression is recorded.
- A module-level zero `var` lowers to `Global { ty, values: None }`.
- One IR instruction, `Instruction::Zero { place: Place }`, sets every value
  in a place to its zero value. A local zero `var` lowers to `Zero` on its
  `binding_place`, so the returned local zeroes the result storage. `Zero` is
  emitted where the declaration executes, so a loop body re-zeroes it.
  Verification checks the place with `destination_type`, as for `Store`.
- `emit_zero(emitter, address, ty)` in `src/backend/emitter.rs`, mirroring
  `emit_copy`:
  - a scalar or pointer is stored with `emit_store` from `zero_operand(ty)`:
    `{class}_0` for a float (`s_0`, `d_0`) and `0` otherwise;
  - an aggregate larger than `MAX_INLINE_COPY_BYTES` calls
    `$memset(l address, w 0, l size)`. This test runs before `value_slots`, so
    a large array never builds its slot list. A type with at most
    `MAX_FIELD_COPY_SLOTS` slots is never larger than `MAX_INLINE_COPY_BYTES`,
    since each slot and the padding after it fit in eight bytes;
  - an aggregate with at most `MAX_FIELD_COPY_SLOTS` slots stores
    `zero_operand` into each slot with its own `memory_class`, so QBE can
    forward the stores to typed loads, as a field-wise copy does;
  - any other aggregate stores zero over its whole size in the widest pieces
    that fit, `storel` for each eight bytes and then `storew`, `storeh`, and
    `storeb` for the rest. Stores at an offset name their address
    `%access{n}_to`, drawn from `emitter.accesses`.

## Test plan

- `src/frontend/tests/mod.rs`: drop `"var x:int;"` from
  `removed_syntax_is_rejected`; new `bindings_may_omit_initializers` parses
  `var x:int;`, `var x;`, and `const x:int;` inside a function and
  `var g:[4]u8;` at module level, and rejects `var x:int` with no `;` and
  `var x:int=;`. Accept the `core_syntax` snapshot change from
  `initializer: Some(...)`.
- `src/semantic/tests/mod.rs`: new `bindings_without_initializers` checks the
  two diagnostics for `var x;` and `const x:int;` both inside a function and at
  module level.
- `src/ir/tests/mod.rs`: new `zero_locals_need_no_data` lowers
  `fn main()->void {var a:[100000]i32;var x:f64;}` and asserts `main` has no
  constants and no `Store`, exactly two `Zero` instructions, and the program
  verifies; `var g:[100000]i32;fn main()->void {}` lowers to one global with
  `values == None`. A `Zero` of a constant place fails verification.
- `src/backend/tests/mod.rs`: new `zero_locals_are_stored_in_place` — a local
  `[100000]i32` emits `call $memset` and no `storel` loop or data; a local
  `type V struct {x:f64,y:f64,z:f64}` emits three `stored d_0` and no
  `memset`; a local `[64]u8` emits eight `storel 0` and no `memset`.
- `tests/core.rs`: drop the `var x:int;` row of
  `invalid_programs_preserve_outputs` and add `const x:int;` expecting
  `initializer` and `var x;` expecting `type annotation`. New
  `zero_values` runs a program with zero locals of `int`, `u8`, `f32`, `f64`
  (compared to `0.0` and with `1.0/x` positive infinity, so the zero is
  `+0.0`), `bool`, `*int` and `nullable *int` (each `== null`), a struct with
  mixed fields and padding, a `[3][5]i16`, a `[200]u8` (memset path), and a
  `[16]u8` (word-store path); module-level zero `var`s of the same kinds; a
  function returning a zero-initialized aggregate local; and a loop whose body
  declares a zero `[100]int` and a zero struct, checks every value is zero,
  then writes to them before the next iteration.
- Release check: compile `var g:[1000000]i32;` and
  `fn main()->void {var a:[1000000]i32;a[999999]=1;g[0]=a[999999];}` with the
  release compiler under `/usr/bin/time -l`. Record the compiler's peak memory
  and the binary's size in the work log; the binary must not hold 4 MB of
  data.

## Implementation plan

1. `src/frontend/syntax.rs`, `src/frontend/parser.rs`: make `initializer` an
   `Option`; `binding` parses `= e` only when the next token is `=`. Remove the
   "every binding requires an initializer" error.
2. `src/semantic/statements.rs`: `check_binding` takes the optional
   initializer; without one, report the `const` or missing-annotation error,
   else record the binding with the resolved annotation type and no constant.
3. `src/semantic/namespaces.rs`: `module_dependencies` collects initializer
   references only when one exists; `check_module_initializer` reports the
   same two errors and otherwise sets the binding's type from the annotation
   with no constant. Keep the `find_call` and constant checks for an
   initializer that exists.
4. `src/ir/model.rs`: add `Instruction::Zero { place }` with a doc comment.
   `src/ir/lower.rs`: in `lower`, a module-level `var` with no initializer gets
   `values: None`; in `lower_flow_binding`, a binding with no initializer pushes
   `Zero` on its `binding_place` and inserts the place. Add a `FlowBuilder::zero`
   beside `store`.
5. `src/ir/verify.rs`: verify `Zero` with `destination_type`; it defines no
   value in `record_definition`.
6. `src/backend/emitter.rs`: add `emit_zero` and `zero_operand` beside
   `emit_copy`, emit `Zero` in `emit_instruction`, and extend the module
   comment to say aggregates are zeroed the same way, with stores or `memset`.
7. Update and add the tests above, and run the release check.
8. `examples/raytrace.fern`: declare `var ends: [2]i32;`. In `render_rows`,
   declare `var bytes: [row_bytes]u8;`, `var light_cells: [samples_per_pixel]int;`,
   and `var bounce_cells: [samples_per_pixel]int;`, pass `&bytes[0]`,
   `&light_cells[0]`, and `&bounce_cells[0]`, and remove their `free` calls. In
   `main`, declare `var readers: [workers]i32;` and remove its `free`. Keep
   `pixels` on the heap. Render the image with the committed example
   (`git show HEAD:examples/raytrace.fern`) and the new one, both built with the
   new compiler, and `cmp` the two `.ppm` files.

## Documentation updates

- `agents/todo.md`: mark "Zero-initialize a `var` without an initializer"
  done.
