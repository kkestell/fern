# Build returned aggregate literals in the result storage

## Goal

`return [a, b];` and `return S { … };` with a literal that does not fold
currently take three steps:

1. build the literal in a temporary local;
2. load it, copying into the load's own storage;
3. copy that storage into the caller's result storage.

Afterwards, lowering stores the literal's elements or fields straight into
the result storage, and the backend returns it without a copy.

## Related code

- `src/ir/model.rs` — `Place` and `Place::root`.
- `src/ir/verify.rs` — `place_type` and `destination_type`.
- `src/ir/lower.rs`:
  - `lower_flow_statement`'s `Return` arm;
  - `store_elements` and `store_fields`;
  - `lower_aggregate_place`, which builds a literal in a fresh local;
  - `unwrapped_call`, the pattern for looking through groupings.
- `src/backend/emitter.rs`:
  - `returned_calls` and `Emitter::returned_calls`, through which a returned
    aggregate call writes into `%result` with no storage and no copy;
  - `allocations`, `emit_call`'s `storage`, and the `Return` arm of
    `emit_terminator`;
  - `place_address` and the `Load` arm of `emit_value`.
- `src/ir/tests/mod.rs` `describe`; `src/ir/tests/lower.rs`;
  `src/ir/tests/verify.rs`; `src/backend/tests/structs.rs`
  `native_returned_calls_chain_their_results`.

## Decisions

**`Place::Result`.** It names the storage a function returning an aggregate
writes its result into, which the caller provides. `Place::root` treats it as
a root. `place_type` gives the function's result type and rejects it in a
function without an aggregate result:
`IR uses result storage in a function without an aggregate result`. It can
be written, so `destination_type` accepts it.

**Lowering.** In the `Return` arm, a value that is an unfolded array or
struct literal, looking through groupings, is built in place:

- A new `returned_literal(checked, id)` finds it. It returns the literal's
  id when the expression has no constant and its value is
  `ExpressionValue::Array` or `ExpressionValue::Struct`.
- `store_elements` or `store_fields` writes into `Place::Result`.
- The returned value is `Load(Place::Result)`, defined in the same block as
  the `Return`.

A folded literal still returns from its constant. Building in place is
sound because the result storage aliases nothing the literal reads. A caller
passes either a fresh call storage or its own result storage, a parameter
never shares storage with the result, and `&` cannot name the result
storage. So no element can observe an earlier element's write.

**Backend.** `returned_calls` becomes `returned_in_place`: the aggregate
values a `return` returns that already sit in `%result`. These are call
results, as today, and loads of `Place::Result`. Such a load emits
`%v{id} =l copy %result` with no storage and no copy. `allocations` skips its
storage, and the `Return` terminator skips the copy, as for a returned call.
`place_address(Place::Result)` is `%result` with the function's result type.
`Emitter::returned_calls` is renamed to match.

## Implementation plan

1. `src/ir/model.rs` — add `Place::Result` and extend `Place::root`.
2. `src/ir/verify.rs` — type `Place::Result` in `place_type`.
3. `src/ir/lower.rs` — add `returned_literal` and build returned literals in
   place.
4. `src/backend/emitter.rs` and `src/backend/qbe.rs` — rename to
   `returned_in_place`, include returned loads of `Place::Result`, and handle
   `Place::Result` in `place_address`.
5. `src/ir/tests/mod.rs` — spell `Place::Result` as `result` in `describe`.

## Test plan

- `src/ir/tests/lower.rs`:
  - `fn pair(x: int) -> [2]int { return [x, x + 1]; }` stores `result[0]`
    and `result[1]` and returns `Load(Result)`, with no other local;
  - `return (S { a = x, ... });` stores `result.0` and zeroes the omitted
    field in `result`;
  - `return [x...];` fills `result`;
  - a folded literal such as `return [1, 2];` still returns from a constant;
  - a non-literal return such as `return local;` is unchanged.
- `src/ir/tests/verify.rs` — `Place::Result` in a function returning `int`,
  and in `main`, is rejected with the internal error above.
- `src/backend/tests/structs.rs`:
  - The emitted QBE for `pair` writes through `%result` and copies nothing
    into it.
  - A native program exits with the expected status. It returns literals
    whose elements call functions, including the same function
    recursively, and a literal that swaps two fields of a parameter.
- Existing IR snapshots that return non-constant literals change. Review
  each diff for the in-place stores before accepting it.
