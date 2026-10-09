# Give a returned local the result storage

## Goal

A function that returns the same aggregate local on every `return` copies
that local into the caller's result storage at each `return`. Instead,
lowering places the local in `Place::Result`. Every read and write of the
local then goes to the result storage, and `return local;` becomes the
`Load(Result)` that the emitter already returns without a copy
(`returned_in_place`).

## Related code

- `src/ir/lower.rs`:
  - `lower_function` and `Places`, which map a binding to its place;
  - `lower_flow_binding`, which allocates a declared binding's local;
  - `lower_flow_statement`'s `Return` arm and `returned_literal`;
  - `Places::borrowable`, which already refuses a `Result` root, so an
    argument read from the local is copied;
  - `addressed_binding` and `Places::addressed`.
- `src/frontend/syntax.rs` — `StatementKind`, and `ForHeader`, whose
  three-clause initializer and post clause are statements.
- `src/semantic/model.rs` — `CheckedProgram::declarations` and
  `Binding::constant`.
- `src/backend/emitter.rs` — `returned_in_place`, `place_address`, `Zero` and
  `Fill` through `place_address`.
- `src/ir/tests/lower.rs` and `src/backend/tests/structs.rs`.

## Decisions

**Which local qualifies.** A new `returned_local(checked, function) ->
Option<Idx<Binding>>` walks the function body: blocks, both branches of an
`if`, `for` bodies, and three-clause initializers and post clauses. It
collects every `return` value and every binding a `Binding` statement
declares. It returns binding `B` when all of these hold:

- the function's result is an aggregate, and it has at least one `return`;
- every `return` value is a reference to `B`, looking through groupings;
- a `Binding` statement in this function declares `B`, so it is neither a
  parameter, a module binding, nor a `for … in` binding;
- `B` has no folded constant, since a folded `const` has no storage;
- no `&` takes `B`'s address, which is not in the `addressed` set.

Otherwise it returns `None`. A function with a different value on some
`return`, such as a literal on one path, keeps today's lowering.

**Lowering.** `Places` gains `returned: Option<Idx<Binding>>`, set by
`lower_function` from `returned_local`. `lower_flow_binding` gives that
binding `Place::Result` in place of a fresh local, for both a zeroed and an
initialized declaration. The `Return` arm needs no change: a reference to
`B` loads from `Place::Result`. Because each `return` value is a reference,
lowering produces a `Load(Place::Result)` in the `return`'s block, and
`returned_in_place` recognizes it.

**Soundness.** A local placed in the result storage is read and written only
through `B`:

- the result storage aliases nothing else the function reads, as decided for
  returned literals;
- `B` is never addressed;
- an argument read from it is copied, because `Places::borrowable` refuses a
  `Result` root;
- no call writes into this function's result storage, because a returned
  call would be a `return` whose value is not `B`.

Writing `B` before the function returns is therefore invisible to everything
except later reads of `B`, which is what a local's semantics require.

## Implementation plan

1. `src/ir/lower.rs` — add `returned_local` and its statement walk, add
   `Places::returned`, and place the returned binding in `Place::Result` in
   `lower_flow_binding`.
2. `src/ir/tests/lower.rs` and `src/backend/tests/structs.rs` — the tests
   below. Review any IR snapshot that changes before accepting it.

## Test plan

- `src/ir/tests/lower.rs`:
  - `fn build(n: int) -> [3]int { var out: [3]int; out[0] = n; return out; }`
    zeroes `result`, stores `result[0]`, returns `Load(Result)`, and
    allocates no local for `out`;
  - a local returned from both branches of an `if` and from inside a `for`
    body is placed in `result`;
  - each of the following keeps a local and returns a load of it:
    - a function returning the local on one path and a literal on another;
    - a local whose address is taken;
    - a folded `const` local, which returns its constant;
    - a returned parameter;
    - two different locals returned on two paths;
  - `var r = [1, 2]; r[0] = sum(r); return r;` passes `r` to `sum` by `Load`
    of `result`, not `Borrow`.
- `src/backend/tests/structs.rs`:
  - A native program exits `42`. It builds arrays and structs in returned
    locals through loops, branches, and recursion, and passes the local to a
    function while building it.
  - The QBE for `build` copies nothing into `%result`.
