# Review: give a returned local the result storage

## Scope and coverage

Reviewed the staged changes for
`agents/plans/2026-10-08-013-give-a-returned-local-the-result-storage.md`:
`returned_local`, `collect_returns`, `collect_statement_returns`,
`returned_binding`, `Places::returned`, and `binding_place` in
`src/ir/lower.rs`, and the new and changed tests and snapshot. Also read the
code they depend on: `Places::borrowable`, `lower_argument`, `capture`,
`addressed_binding`, the `Return` arm, the IR verifier's `Borrow` and
`Address` rules, and `returned_in_place`, `allocations`, `emit_call`, the
`Load` arm of `emit_value`, and the `Return` terminator in
`src/backend/emitter.rs`.

The review traced every way a local in the result storage could be observed
differently from a private local:

- The caller's result storage is a fresh `%v{id}_storage`, or for a returned
  call the caller's own `%result`. A function with a returned call has no
  returned local, and no argument is borrowed from the result storage, so the
  storage never aliases an argument, a global, or a pointer target.
- `addressed_binding` follows fields and elements to their root binding, so
  `&r.x` and `&r[0]` disqualify `r`.
- A `Load` of an aggregate copies it, so an argument read from the local, a
  `for … in` capture, and `r = [r[1], r[0]]` read a snapshot.
- A call whose result is assigned to the local writes a fresh
  `%v{id}_storage` that is then copied into `%result`.
- A redeclaration in a loop zeroes or initializes `%result` again, as it
  would a local.
- A function with a returned literal on any path has no returned local, so the
  literal's in-place build still reads nothing from the result storage.

## Fixed

- **Comments still describe the result storage as written only by a returned
  call or literal** (`src/backend/emitter.rs:312`, `src/backend/emitter.rs:727`,
  `src/ir/verify.rs:465`): `returned_in_place` claimed nothing reads
  `%result` before the function returns, which a returned local now does, and
  two comments listed only calls and literals as what sits in `%result`. A
  reader relying on them would misjudge what may touch the result storage. The
  comments now name the returned local and state the actual invariant: nothing
  writes `%result` between a returned value's definition and its `return`.
- **Native tests omit caller aliasing and loop redeclaration**
  (`src/backend/tests/structs.rs:347`): no test assigned a returned local's
  result back to the argument it was built from, forwarded it through a
  returned call, or redeclared the local in a loop. A regression in any of
  those paths would go unnoticed. `native_returned_locals_keep_local_semantics`
  now covers `x = swapped(x)`, `x = forwarded(x)`, and a local zeroed on each
  iteration of a `for` loop before being returned.

## Findings

No open findings.

## Checks run

- `cargo fmt --check`: clean.
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo test`: 366 unit and 62 integration tests pass.
- Native programs, all exiting 42:
  - a returned local that reads its parameter after writing itself, assigned
    back to the argument and forwarded through a returned call;
  - a returned local redeclared and zeroed in an infinite loop;
  - `r = [r[1], r[0]]` and `r = swap(r)` on a returned local;
  - a returned local that reads a global, assigned to that global;
  - a nested struct local with a field assigned from its own fields, an array
    field passed to a call, and a `for … in` over that field that writes it;
  - a returned local declared in a three-clause `for` initializer;
  - a returned local built from a pointer to the caller's destination, which
    then writes through the pointer.

## Verdict

The change meets the plan. A returned local living in the result storage
cannot be observed differently from a private local. Two comment and test gaps
were fixed, and no findings remain open.
