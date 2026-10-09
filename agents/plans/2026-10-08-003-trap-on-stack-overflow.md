# Trap on stack overflow

## Goal

Compiled code never checks how much stack is left. Deep recursion or a large
frame runs past the end of the stack, and a frame larger than the guard region
below the stack can write into other mapped memory (`agents/bugs.md`, "Stack
overflow is unchecked").

When this is done, every Fern call checks that the stack has room for the
callee before it calls, and traps with a `stack overflow` diagnostic at the
call's source location when it does not. Entering `main` is checked the same
way and reports `main`'s name. No Fern frame is ever placed below the end of
the stack.

## Related code

- `src/backend/emitter.rs`:
  - `emit` — writes each function's header; Fern `main` is the exported
    `$main` today.
  - `function_symbol` — names `main` `$main` and every other function
    `$fn{id}`.
  - `emit_control_flow` — the `alloc`s at `@start`: every local except an
    aggregate parameter, and the storage of each aggregate `Load` and
    `CallResult` not in `returned_calls`.
  - `emit_instruction` and `emit_call` — the `Call` arm drops the call's span.
- `src/backend/qbe.rs` — `Emitter`, `emit_conditional_trap`, and
  `operation_message`.
- `src/ir/model.rs` — `Program` and `Instruction::Call { span, .. }`.
- `src/ir/lower.rs` `lower_program` — builds `Program`; `checked.main` is the
  `main` function's syntax ID.
- `src/frontend/syntax.rs` `Function::name_span`.
- `src/backend/tests/mod.rs` — `assert_native_values` finds `main` as the first
  `export function`.

## Decisions

**The check runs in the caller, before the call.** QBE emits no stack probes,
and its prologue subtracts the whole frame from the stack pointer and then
stores the frame pointer and saved registers at the new bottom of the frame
before any instruction of the function body runs. A check inside the callee
would run after those stores, so the caller checks for the callee instead.

**Each function's storage size is known before emission.** The `alloc`s a
function emits are its storage; `storage_bytes` is their total, each size
rounded up to eight bytes and the sum saturating at `i64::MAX`. QBE's frame
adds spill slots, saved registers, and alignment padding to that storage.
`emit_control_flow` and `storage_bytes` read the same `allocations` list, so
the two cannot disagree.

**A stack position in every frame.** Each function allocates
`%stack_position =l alloc4 4`. The stack pointer lies at most the function's
whole frame below that address.

**The check.** Before `call $fnN(...)`, the caller computes the bytes the call
needs: the caller's `storage_bytes`, the callee's `storage_bytes`, and eight
bytes per argument. It traps when `%stack_position` is below
`$fern_stack_limit`, or when `%stack_position - $fern_stack_limit` is less
than those bytes. The two unsigned comparisons (`cultl`) never wrap.

**The limit keeps a reserve.** `$fern_stack_limit` is the lowest stack address
plus `STACK_RESERVE_BYTES` (64 KiB). The reserve holds what the check does not
count: the spill slots, saved registers, and padding of the caller's and the
callee's frames, and the C library calls a function makes (`memcpy`,
`memset`, `write`, `abort`, and `exit`), including the trap's own `write` and
`abort`. A frame whose spill area exceeds the reserve still stays within the
guard region below the stack (56 MiB on macOS, 1 MiB on Linux), so it faults
there rather than writing other memory.

**A generated `$main` finds the stack and enters Fern `main`.** Every Fern
function, `main` included, is `$fn{id}`. The exported `$main` wrapper:

1. Finds the main thread's lowest stack address. On macOS it is
   `pthread_get_stackaddr_np(pthread_self())` minus
   `pthread_get_stacksize_np(pthread_self())`. On Linux,
   `pthread_getattr_np(pthread_self(), attr)` fills a 128-byte `attr` buffer,
   `pthread_attr_getstack` reads the address, and `pthread_attr_destroy`
   releases `attr`. A nonzero status from `pthread_getattr_np` traps with
   `cannot find the stack bounds`.
2. Stores that address plus `STACK_RESERVE_BYTES` in `$fern_stack_limit`.
3. Checks room for Fern `main` the same way a call does, reporting `main`'s
   name span.
4. Calls `$fn{main}` and returns its status.

The backend chooses the macOS or Linux sequence with `cfg!(target_os)`, since
QBE compiles for the host. Any other host is a `compile_error!` in
`src/backend/emitter.rs`.

**`main`'s name span reaches the backend through `Program::main_span`.** The
IR carries no function spans, and only the entry check needs one.

**The message is `stack overflow`.** `operation_message` places it at the call
span, or at `main`'s name for the entry check.

## Implementation plan

1. Run `/kspec` to add stack overflow to `docs/spec.md`: a call traps when the
   stack has no room for the called function, including the entry into
   `main`; how much stack a program has is platform-defined. Add a row to the
   summary of trapping behavior and an item to the portability notes.
2. `src/ir/model.rs`: add `Program::main_span: std::ops::Range<usize>`, the
   span of `main`'s name.
3. `src/ir/lower.rs` `lower_program`: set `main_span` from
   `checked.syntax.functions[checked.main].name_span`.
4. `src/backend/qbe.rs` `Emitter`: add `storage_bytes: Vec<u64>`, indexed by
   function, and `stack_checks: usize`, which names each check's trap blocks
   and message symbol. Pass `stack_checks` as `emit_conditional_trap`'s `id`
   with the cause `stack`.
5. `src/backend/emitter.rs`:
   - Add `const STACK_RESERVE_BYTES: u64 = 64 * 1024` with the reserve's doc
     comment from the decisions.
   - Add `allocations(layout, function) -> Vec<(String, Type)>`: the
     `%stack_position` marker, then the locals and value storage
     `emit_control_flow` allocates today, each with its QBE name. Make
     `emit_control_flow` emit exactly this list.
   - Add `storage_bytes(layout, function) -> u64` over `allocations`, and fill
     `Emitter::storage_bytes` for every function before emitting any.
   - `function_symbol(id)`: always `fn{id}`. Drop its `main` parameter.
   - `emit`: no Fern function is `export`. Emit `data $fern_stack_limit = { l
     0 }` and the `$main` wrapper described in the decisions.
   - Add `emit_stack_check(emitter, caller_bytes, callee_bytes, arguments,
     message)`, which emits the comparisons and the trap. Use it in
     `emit_call`, which now takes the call's span from `emit_instruction`, and
     in the wrapper.
   - Update the module doc comment to say every call checks the stack first.
6. `src/backend/tests/mod.rs`: `assert_native_values` finds Fern `main` by its
   `$fn{id}` symbol instead of the first `export function`. Update other tests
   that read Fern `main` as `$main`.
7. `README.md`: after the source nesting paragraph, state that a compiled
   program runs on the main thread's stack and traps with `stack overflow`
   when a call would leave less than 64 KiB of it.
8. `agents/bugs.md`: remove "Stack overflow is unchecked".

## Test plan

- `src/backend/tests/emitter.rs`:
  - A program calling a helper defines exactly one `export function`, `$main`,
    which calls Fern `main`'s `$fn{id}`, and defines `$fern_stack_limit`.
  - The call to the helper is preceded by `cultl` comparisons against
    `$fern_stack_limit`, and its byte count includes a `[1000]u64` local of
    the callee (at least 8000).
  - `storage_bytes` of a function with an aggregate parameter excludes that
    parameter, and one with a `return f(...)` call excludes that call's
    storage.
- `src/backend/tests/emitter.rs`, native, each asserting the process ended by
  `SIGABRT` (`ExitStatusExt::signal() == Some(6)`) and that stderr contains
  `stack overflow`:
  - Unbounded recursion, `fn down(n: int) -> int { return down(n + 1); }`.
  - Recursion whose frame holds a `[1000]u64` local.
  - A call to a function with a `[16000000]u64` local, a 128 MB frame larger
    than macOS's 56 MiB guard region.
  - `main` holding a `[16000000]u64` local.
- `src/backend/tests/emitter.rs`, native: recursion 10,000 calls deep returns
  the expected sum.
- `tests/compiler.rs`: compiled through `fern::compile`, unbounded recursion
  reports `stack overflow` with the source path and the line of the recursive
  call, and an oversized `main` local reports `main`'s line.
- Review and accept any IR snapshot changes under `src/ir/tests/snapshots/`.
