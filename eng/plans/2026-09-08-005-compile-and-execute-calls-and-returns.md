# Compile and execute calls and returns

## Sources

- `docs/spec.md#functions`, `docs/spec.md#calls`,
  `docs/spec.md#function-return` — parameters hold copies of their arguments,
  calls are direct, and a function returns zero or one value
- `docs/spec.md#process-exit` — `exit` terminates the program and does not
  return, from any function
- `eng/roadmap.md` "Parameters, calls, and return values" — task scope, the
  milestone program, and the completion gates
- `src/ir.rs:140-165` — `Global`, `Function` (parameters are the first
  `flow.locals`), and `Program`
- `src/backend.rs:25-157` — `Emitter`, `emit`, `emit_control_flow`,
  `emit_terminator`; `src/backend.rs:821-843` — `emit_conditional_trap`

## Goal

Emit native code for every IR function — signature, parameters, calls, and
returns — replacing the `Instruction::Call` guard in `emit_control_flow`. The
IR already lowers and verifies functions, so this slice only emits them. It
completes the milestone: add the parameters, calls, and returns completion
fixture, run broad validation, and review the integrated milestone.

## Implementation

- `src/backend.rs` — add a `function_symbol(program, id)` helper returning
  `"main"` for `program.main` and `format!("fn{}", id.0)` otherwise, so the
  definition header and every call site share one naming rule. Source names are
  never used, which keeps a Fern function named `write` or `exit` from clashing
  with libc.

- `src/backend.rs::emit` — after emitting the globals, loop over
  `program.functions` and emit each one. `main` gets
  `export function w $main()`; every other function gets
  `function {qbe_type(result)} $fn{id}(...)`, with the return type omitted when
  `result` is `None`. The parameter list is
  `{qbe_type(flow.locals[i])} %param{i}` for `i` in `0..parameters`, joined with
  `", "`. Write `{\n@start\n`, call `emit_control_flow`, then `}\n`.

- `src/backend.rs` — `Emitter` gains `functions: &'a [Function]`,
  `main: FunctionId`, `function: usize` (the index being emitted), and
  `qbe_result: Option<char>` (the emitted signature's return type: `Some('w')`
  for `main`). Set the last two per function in `emit`.

- `src/backend.rs::emit_control_flow` — after the existing `alloc` loop and
  before `jmp @block{entry}`, store each argument into its parameter local:
  `store{width} %param{i}, %local{i}`, width from `qbe_type(flow.locals[i])`.
  Parameters are already allocated by the locals loop.

- `src/backend.rs::emit_control_flow` — replace the `Instruction::Call` guard
  with emission. Read the callee from `emitter.functions[function.0]`; type each
  argument with `qbe_type(callee.flow.locals[i])`. With a `result`, emit
  `    %v{id} ={qbe_type(callee.result)} call ${symbol}({args})`; without one,
  emit `    call ${symbol}({args})`. `ValueKind::CallResult` stays a no-op in
  `emit_value`.

- `src/backend.rs::emit_terminator` — `Terminator::Return { value: Some(v) }`
  emits `ret {operand}`. `Return { value: None }` emits `ret 0` when
  `qbe_result` is `Some` (only `main`, whose end exits zero) and a bare `ret`
  otherwise; `ret 0` is invalid QBE in a function with no return type.

- `src/backend.rs::emit_terminator` — `Terminator::Exit` keeps the
  `and {status}, 255` mask, then emits `call $exit(w %block{n}_status)` followed
  by `hlt`. `ret` only terminated the program from `main`.

- `src/backend.rs::emit_conditional_trap` — emit `hlt` instead of `ret 1` after
  `call $abort()`, which is valid whatever the enclosing signature returns.

- `src/backend.rs` — the trap message symbol `fern_operation{id}_{cause}_message`
  is module-global and now collides when two functions share a value id and
  cause. Include the function index: pass `emitter.function` into
  `emit_conditional_trap` and `emit_checked_conversion` (they take `text`/`data`
  rather than the emitter) and name the symbol
  `fern_function{f}_operation{id}_{cause}_message`. QBE temporaries and block
  labels are function-scoped and need no change.

- `src/backend.rs`, `src/lib.rs` — with the guard gone no emission path fails.
  Make `emit_control_flow` infallible, have `emit` return `String`, and drop the
  `?` in `build`.

- `tests/fixtures/programs/parameters_calls_and_returns.fern` — the roadmap
  completion program verbatim.

## Tests

- `tests/compiler.rs` — a `parameters_calls_and_returns_execute` test in the
  style of `branches_and_loops_execute`, running the fixture (exit 42) plus
  focused sources covering: a nested call as an argument; two argument calls
  that mutate a module `var` proving left-to-right order; a discarded value
  result; a forward call to a function declared after `main`; direct and mutual
  recursion; a `void` early `return`; and a `bool` result used as a condition.

- `tests/compiler.rs` — `exit` inside a called function terminates the program
  with that status rather than returning to its caller.

- `tests/compiler.rs` — a trap inside a called function reports its diagnostic
  and fails, proving the per-function message symbols link.

- `src/backend.rs` — a program whose `main` and a called function both trap at
  the same value id assembles and links.

## Extra validation

- Run the milestone completion gates end to end, then review the integrated
  milestone with `kreview`.
