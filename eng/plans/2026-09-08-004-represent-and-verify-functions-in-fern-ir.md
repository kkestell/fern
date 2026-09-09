# Represent and verify functions in Fern IR

## Sources

- `docs/spec.md#functions`, `docs/spec.md#calls`, `docs/spec.md#function-return`
  — parameter copying, left-to-right argument evaluation, call results, and
  return forms
- `docs/spec.md#module-level-declarations` — module bindings are initialized
  before `main` runs and are visible to every function
- `eng/roadmap.md#parameters-calls-and-return-values` — task boundary and the
  milestone program
- `src/ir.rs` — the entry-only `Entry`/`VerifiedEntry` shape, `FlowBuilder`, and
  operand lowering
- `src/semantic.rs` — `CheckedProgram::functions`, `calls`, and `main`

## Goal

Replace the entry-only IR with a program of functions: lower and verify every
function in the module with its signature, parameters, body, calls, and return
terminators, and move module-level `var` bindings out of `main`'s locals into
program globals. Native emission of calls and of functions other than `main`
belongs to the next roadmap task, so this slice guards that gap rather than
filling it.

## Implementation

- `src/ir.rs` — replace `Entry`/`VerifiedEntry` with `Program`/`VerifiedProgram`.
  A `Program` holds `globals: Vec<Global>`, `functions: Vec<Function>`, and
  `main: FunctionId`. A `Function` holds `parameters: usize`, `result:
  Option<Type>`, its own `values: Vec<Value>`, and its `flow`. The first
  `parameters` locals are the parameters in source order; their types come from
  `flow.locals`. Function ids follow `syntax.functions` order so forward and
  mutually recursive calls need no second pass.

- `src/ir.rs` — `Global { ty, value: i128 }`. Every module-level `var` becomes a
  global whose initial value is its checked constant; module-level `const`
  keeps folding at its use sites. Add `enum Place { Local(LocalId),
  Global(GlobalId) }` and use it in `ValueKind::Load` and `Instruction::Store`,
  so one load and one store path serve both.

- `src/ir.rs` — represent a call as
  `Instruction::Call { result: Option<ValueId>, function: FunctionId,
  arguments: Vec<Operand>, span }`, with a unit `ValueKind::CallResult` for the
  value it defines. One instruction covers `void` and value calls; the result's
  type and span live on its `Value` as for every other value.

- `src/ir.rs` — `Terminator::Return { value: Option<Operand> }`. After lowering
  a function body that did not terminate, emit `Return { value: None }` only
  when the result is `void`; leave a value function's last block unterminated so
  verification reports the lowering bug.

- `src/ir.rs` — lower call arguments left to right and hold each lowered
  operand, so an argument containing `&&` or `||` splits blocks without leaving
  an earlier argument unreadable in the call's block. A held value read from
  another block goes through a local; this is the same rule that keeps the left
  operand of a binary operation, comparison, and compound assignment readable
  across their right operand.

- `src/ir.rs` — lower every function in the module, including unreferenced ones.
  Do not add reachability pruning.

- `src/ir.rs` — verify per function: the entry block starts with the parameters
  initialized; a call names a known function, supplies one argument per
  parameter with the parameter's type, and defines a result exactly when the
  callee returns a value, of that type; a `CallResult` value is defined by a
  call instruction and by nothing else; `Return` carries a value exactly when
  the signature does, of the declared type; `main` has no parameters and no
  result; a global's initial value fits its type. Reachable blocks must still be
  terminated, which is what rejects a value function that can fall off its end.

- `src/backend.rs` — follow the new shape: emit globals into the existing data
  section, emit loads and stores through `Place`, emit `$main` from
  `program.functions[program.main]`, and emit `Return { value: None }` as
  `ret 0`. Return a `CompileError` when an emitted function contains a call
  instruction; that guard and the remaining functions go away in the next task.

## Tests

- `src/ir.rs` — regenerate every `fern__ir__tests__*.snap` fixture for the
  program shape.

- `src/ir.rs` — snapshot the milestone program's shape: a module `var` mutated
  through a called function, a value call used inside an expression, a `void`
  call statement, a discarded result, and direct recursion.

- `src/ir.rs` — a call whose argument short-circuits (`f(c, a && b)`) lowers and
  verifies, proving the hold keeps every argument operand available in the
  call's block and in source order, and that a call nested in a binary
  operation, a comparison, and a compound assignment executes.

- `src/ir.rs` — replace
  `control_flow_in_an_unreferenced_function_does_not_change_main_ir`: an
  unreferenced function is now lowered as its own IR function and `main` is
  unchanged.

- `src/ir.rs` — verification rejects hand-built programs with: an unknown call
  target; too few and too many arguments; an argument whose type is not the
  parameter's; a call to a `void` function that defines a result; a value call
  that defines none; a returned value in a `void` function and a missing one in
  a value function; a returned value of the wrong type; a value function with a
  reachable unterminated block; a store to an unknown global; and a global whose
  initial value does not fit its type.

## Decisions

- Module-level `var` bindings become globals rather than locals initialized in
  `main`'s prologue, because a called function must see and mutate them. Their
  initializers are constant expressions, so a global needs only a constant
  initial value and no initialization code.

- A void call gets no `Value` because `Type` has no `void`. Making the result
  optional on the call instruction keeps one representation for both call forms.

- Parameters are the function's first locals rather than a separate list, so
  loads, stores, and the definite-initialization analysis keep working
  unchanged; only the entry block's initial set differs.
