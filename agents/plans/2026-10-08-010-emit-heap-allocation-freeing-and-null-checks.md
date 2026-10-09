# Emit heap allocation, freeing, and null checks in native code

## Goal

The QBE backend emits the last three pointer constructs:

- `ValueKind::Allocate` calls `malloc` and traps with `out of memory` on a
  null result;
- `ValueKind::CheckNull` traps with `null pointer` when its operand is
  `null`;
- `Instruction::Free` calls `free`.

Lowering already stores the `new` operand through the allocated pointer. The
driver guard goes away, so every pointer program compiles.

## Related code

- `src/backend/emitter.rs`:
  - the `Allocate` and `CheckNull` arm of `emit_value`, and the `Free` arm of
    `emit_instruction`, both `unreachable!` today;
  - `Layout::size`;
  - the bounds check in `place_address`, the pattern for a trap with a span.
- `src/backend/qbe.rs` `emit_conditional_trap` and `operation_message` — the
  existing failure reporting.
- `src/semantic/annotations.rs` `heap_not_implemented` and
  `reject_heap_syntax`, `src/semantic/mod.rs`, and `src/lib.rs` `compile` —
  the guard to remove.
- `src/backend/tests/pointers.rs` — native pointer tests.
- `tests/compiler.rs` `source_failures_preserve_output` — the three guard
  cases.

## Decisions

**Allocation.** `Allocate` of type `*T` emits
`%v{id} =l call $malloc(l {size})`, with the size from
`emitter.layout.size(T)`. Every Fern type has a nonzero size, so a null
result means allocation failed. It then emits `%v{id}_failed =w ceql %v{id},
0` and an `emit_conditional_trap` with cause `allocation` and the value's
`id`. The message is `operation_message(value.span, …, "out of memory")`.
`malloc` aligns for any Fern value.

**Null check.** `CheckNull` emits `%v{id}_null =w ceql {operand}, 0` and an
`emit_conditional_trap` with cause `null` and the message
`operation_message(value.span, …, "null pointer")`. It then emits
`%v{id} =l copy {operand}`.

**Free.** `Instruction::Free` emits `call $free(l {operand})`. `free` ignores
a null pointer, as the spec requires of freeing `null`. These calls go to the
C library, as `memcpy` and `memset` do, so no Fern stack check precedes them.

**Guard.** Remove `reject_heap_syntax` and `heap_not_implemented` from
`src/semantic/annotations.rs`, the re-export from `src/semantic/mod.rs`, and
the call from `compile`. No backend arm for a pointer construct stays
`unreachable!`.

## Implementation plan

1. `src/backend/emitter.rs` — emit `Allocate`, `CheckNull`, and `Free`.
2. `src/semantic/annotations.rs`, `src/semantic/mod.rs`, and `src/lib.rs` —
   remove the guard.
3. `src/backend/tests/pointers.rs` — add the tests below.
4. `tests/compiler.rs` — replace the guard cases.

## Test plan

- `src/backend/tests/pointers.rs`, building and running native programs:
  - The spec examples exit as the spec says:
    - Heap allocation: `5`;
    - Access through a pointer: `35`;
    - Nullable pointers (the linked-list sum): `3`;
    - Pointer comparison: `0`.
  - `const p = new(1); free(p); var q: nullable *int; free(q); free(q,);`
    exits `0`.
  - `var q: nullable *int; const p = *int(q); exit(*p);` fails with
    `null pointer`. A non-null `*int(q)` exits with its target.
  - A `new` of a struct with a call in its operand stores the operand before
    the program reads it back.
  - The QBE text of `new(1)` holds `call $malloc(l 8)` and an
    `out of memory` message. The text of `free(p)` holds a `call $free`.
- `tests/compiler.rs`:
  - The `new(1)` program moves to `native_programs_exit_zero`.
  - The `free(&x)` case is removed, since freeing storage `new` did not
    allocate is an unchecked error.
  - The `*int(q)` program, with `q` a `null` nullable pointer, compiles, and
    running it fails with `null pointer`.
