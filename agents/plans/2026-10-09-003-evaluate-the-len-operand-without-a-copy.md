# Evaluate the `len` operand without a copy

## Goal

`len(a)` lowers its array operand as a value and discards it, so the emitter
turns the aggregate `Load` into stack storage and a copy of the whole array. An
array larger than the stack crashes the process. When this work is done, `len`
evaluates its operand only for the effects it owes (calls, index operands, the
pointer under a dereference) and never loads, copies, or emits constant data for
the array itself. The program in `agents/bugs.md` exits 0, and its entry is
removed.

## Related code

- `src/ir/lower.rs` `lower_length` — lowers a `len` whose operand holds a call;
  calls `lower_flow_operand` on the operand.
- `src/ir/lower.rs` `lower_folded_effects` — its `Length` arm lowers the operand
  of a folded `len` through `lower_flow_operand`. This is the path the bug's
  `len(*big)` takes.
- `src/ir/lower.rs` `lower_flow_operand` / `lower_aggregate_operand` — an
  aggregate operand becomes a `Load` of its place, which is the copy.
- `src/ir/lower.rs` `lower_aggregate_place` — yields an aggregate's place
  without loading it: no instructions for a reference, only the pointer for a
  dereference, the index operands for an element. For a folded aggregate it
  also emits a constant `Global`.
- `src/ir/lower.rs` `unwrapped_call` / `lower_call_expression` — lower a call
  whose aggregate result is used directly, without a local.

## Decisions

- A new function, `lower_aggregate_effects`, lowers an aggregate operand for
  its effects alone. A folded operand goes to `lower_folded_effects`, so no
  constant `Global` is emitted. An operand that `unwrapped_call` resolves to a
  call goes to `lower_call_expression` and the result is discarded, as a call
  statement discards it. Every other operand goes to `lower_aggregate_place`
  and the place is discarded.
- A non-constant array literal operand, such as `len([f(), 2])`, still builds
  its elements in a local through `lower_aggregate_place`. Its size is bounded
  by the source text, and the calls inside it must run in order.
- Skipping the load under a dereference changes no defined behavior: the spec
  says the operand "executes normally", and reading memory has no observable
  effect in a program whose behavior is defined.

## Test plan

- `src/ir/tests/mod.rs`: `len_does_not_load_its_operand` lowers
  `fn main()->void {var a:[4]i64=[1,2,3,4];var p=&a;var n=len(*p);var m=len(a);}`
  and asserts no value in `main` has kind `ValueKind::Load` with an array type,
  then verifies the program.
- `tests/core.rs`: `len_of_an_array_larger_than_the_stack_executes` runs the
  program from `agents/bugs.md` (without its own `exit` declaration, which `run`
  prepends) and expects exit status 0.
- `tests/core.rs`: `len_runs_the_effects_of_its_operand` runs a program whose
  `len` operands are a call returning an array, an index whose index operand
  is a call, and a dereference whose pointer comes from a call. Each call
  increments a global counter, and the program exits non-zero unless every
  `len` yields the right length and the counter equals the number of calls.

## Implementation plan

1. Add `lower_aggregate_effects` to `src/ir/lower.rs`, next to `lower_length`,
   as described under Decisions.
2. In `lower_length`, replace the `lower_flow_operand` call with
   `lower_aggregate_effects`.
3. In the `Length` arm of `lower_folded_effects`, replace the
   `lower_flow_operand` call with `lower_aggregate_effects`.
4. Add the tests from the test plan.
5. Remove the "`len` copies its array operand" entry from `agents/bugs.md`.
