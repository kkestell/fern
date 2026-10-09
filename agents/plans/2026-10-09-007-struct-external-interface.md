# Pass structs by value across the external interface

## Goal

External functions accept only numeric and pointer parameters and results, so a
Fern program cannot call a C API that takes or returns a small struct by value.
Allow struct parameters and results, passed with the host C ABI for a C struct
with the same fields in the same order.

## Related code

- `docs/spec.md#functions-and-external-calls` — states the types allowed across
  the external interface and excludes aggregates.
- `src/semantic/namespaces.rs:513` — `resolve_signatures` rejects external
  parameter and result types other than numeric scalars and pointers. Every
  struct a signature can name has resolved fields by then: the module's own
  structs resolve in `resolve_module_structs` first, and imported modules are
  already checked.
- `src/semantic/model.rs:306` — `CheckedProgram` implements `StructFields`.
- `src/ir/verify.rs:381` — `verify_function` checks the same restriction on
  lowered external declarations with a local `external_type` closure.
- `src/layout.rs:40` — the `StructFields` trait, a phase's struct table read by
  ordinal.
- `src/backend/qbe.rs:271` — `abi_class` names an external argument or result
  type in a QBE call; it falls back to `value_class`, which panics on
  aggregates.
- `src/backend/emitter.rs:384` — `emit_call` writes an aggregate result of a
  Fern function into `result_storage` by passing its address as a leading
  argument. An external call cannot do that.
- `src/backend/emitter.rs:188` — `emit` writes `emitter.data` before
  `emitter.text`.
- `tests/core.rs:235` and `tests/core.rs:278` — native tests that link a C
  helper object; the pattern for the new test.

## Decisions

- A struct crosses the interface when nothing it stores by value is `bool`:
  each field is numeric, a pointer, an array of such field types, or a struct
  that itself crosses. This follows the scalar rule, which already excludes
  `bool`. Arrays still cannot be parameters or results on their own, because C
  cannot pass them by value; as fields they correspond to C arrays of the same
  length.
- QBE implements the C ABI for aggregate types declared with `type`. A struct
  that crosses gets a QBE type named `:fern.struct{id}`, whose items follow its
  fields: `memory_class` for scalars, `l` for pointers, `:fern.struct{id}` for
  nested structs, and an array as its innermost element's item repeated by the
  product of the lengths (`[2][3]u8` is `b 6`). QBE pads and aligns these items
  the same way `src/layout.rs` does. Only structs reachable from external
  signatures get a definition, written into `emitter.data` before the globals,
  nested structs first, each once.
- An external struct argument is the address of its storage, named with
  `:fern.struct{id}`; QBE copies it at the call, so the callee's copy is taken
  after every argument has been evaluated, as with Fern calls.
- An external struct result is returned by QBE as the address of storage QBE
  owns. `emit_call` names it `%call{n}` from a new `calls` counter on `Emitter`,
  copies it with `emit_copy` into the same `result_storage` a Fern call would
  write, and, when the call defines a value, copies that storage address into
  the value as today. So the destination, `%v{id}_storage`, and `%result`
  (`returned_in_place`) cases need no other change.
- The rule lives once, as `Type::crosses_external_interface(&self, table: &dyn
  StructFields) -> bool` in `src/types.rs`. The semantic phase and IR
  verification both call it. `ir::model::Program` implements `StructFields`
  over its `structs`.
- Diagnostics: a struct that does not cross reports
  `a struct passed across the external interface cannot contain bool`; any
  other rejected type keeps
  `external functions accept only numeric, pointer, and struct types`.

## Test plan

- Native test `c_abi_structs` in `tests/core.rs`. The C helper defines
  `Vec2 { float x, y; }`, `Mixed { uint8_t tag; int16_t small; double value; }`,
  `Big { int32_t values[10]; void *next; }`, and
  `Box { Vec2 corners[2]; uint8_t id; }`, and functions
  `Vec2 vec2_scale(Vec2, float)`, `Mixed mixed_make(uint8_t, int16_t, double)`,
  `int64_t mixed_sum(Mixed)`, `Big big_make(int32_t start)`,
  `int32_t big_clobber(Big)` that writes every element of its copy and returns
  the original sum, `Box box_flip(Box)`, and
  `double spill(Vec2, Vec2, Vec2, Vec2, Vec2, Mixed)` so some struct arguments
  pass on the stack. The Fern program declares matching structs and calls each
  function: results initialize a `var`, are assigned to a struct field, are
  discarded in a call statement, and are returned from a Fern function that
  returns `vec2_scale(...)` directly; arguments come from a local, a nested
  field, and a dereferenced pointer. It checks that `big_clobber` leaves the
  Fern argument unchanged and that `big_make`'s `next` is `null`. Each
  mismatch exits with a distinct status.
- `src/semantic/tests/mod.rs` `external_signatures_and_returns`: reject
  `type S struct {b:bool} extern fn foo(s:S)->void;`,
  `type S struct {a:[2]bool} extern fn foo()->S;`, and
  `type I struct {b:bool} type S struct {i:I} extern fn foo(s:S)->void;`;
  accept `type N struct {x:f32} type S struct {x:f32,a:[3]u8,p:*S,n:[2]N}
  extern fn foo(s:S)->S;`. `extern fn foo()->[2]int;` stays rejected.
- `tests/core.rs` diagnostic cases: add
  `type S struct {b:bool} extern fn x(s:S)->void;fn main()->void {}` expecting
  `cannot contain bool`, and change the `numeric and pointer` expectation to
  `numeric, pointer, and struct`.
- `src/ir/tests/mod.rs` `rejects_invalid_control_flow_and_types`: a lowered
  `type S struct {x:i32} extern fn f(s:S)->S;` verifies, and fails to verify
  after its struct's field is changed to `bool`.
- `src/backend/tests/mod.rs`: a test `external_structs_use_qbe_aggregate_types`
  that emits a program with nested structs crossing the interface and checks
  that the inner type is defined before the outer one, that an array field is
  a repeated item, and that the call names `:fern.struct{id}` for its argument
  and result.

## Implementation plan

1. Update `docs/spec.md#functions-and-external-calls` through `/kspec`:
   parameters and results are numeric, pointer, or struct types; a struct
   corresponds to a C struct with corresponding fields in the same order, an
   array field to a C array of the same length; a struct crossing the
   interface cannot contain `bool` in any field or element; arrays, rather
   than aggregates, are what cannot pass directly.
2. Add `Type::crosses_external_interface` to `src/types.rs`.
3. In `src/semantic/namespaces.rs`, call it in `resolve_signatures` and report
   the two diagnostics.
4. Implement `StructFields` for `ir::model::Program` and replace the
   `external_type` closure in `src/ir/verify.rs` with the shared rule.
5. In `src/backend/qbe.rs`, add `struct_type(id: StructId) -> String` returning
   `:fern.struct{id}`, and make `abi_class` return it for a struct.
6. In `src/backend/emitter.rs`, add `emit_external_types`, called from `emit`
   before the globals, and handle an external struct result in `emit_call`.
7. Add the tests above.
