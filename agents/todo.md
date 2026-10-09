# Todo

- [x] Parse pointer syntax

  Lex and parse the pointer forms in `docs/spec.md#pointers-and-heap-memory`:
  the `*T` and `nullable *T` annotations, the `null` literal, prefix `&` and
  `*`, `new(e)` with an optional trailing comma, the `free(p);` statement, and
  `*T(p)` as a conversion head. Reserve `nullable`, `new`, `free`, and `null`
  in every identifier position. Represent each form in frontend syntax and
  cover it with parser snapshots; temporary semantic guards reject the syntax
  until the checking tasks land.

- [x] Represent pointer types across the compiler

  Add a pointer type with a target type and nullability to `src/types.rs`,
  with its spelling and sameness rule. Resolve pointer annotations to that
  type, including nested and recursive uses such as `next: nullable *Node`.
  Give pointers the platform pointer width in `src/layout.rs` and a scalar
  slot in the semantic model. Reject `*void`.

- [x] Check address-of, dereference, and access through pointers

  Type `&location` as a `*T` and reject it on `const` bindings, parameters,
  their fields and elements, and non-locations. Treat `*p` as a mutable
  location of the target type regardless of the binding's mutability, and
  make field selection, indexing, and `len` look through any number of
  non-nullable pointers. Index and field steps in an `&` operand check as an
  assignment target does.

- [x] Check `new`, `free`, nullable pointers, and pointer comparison

  Type `new(e)` as a `*T`, giving an untyped constant its default type. Check
  `free(p);` against pointer operands. Type `null` only where context supplies
  a nullable pointer type, convert `*T` to `nullable *T` implicitly, and type
  `*T(p)` as a checked conversion. Allow `==` and `!=` between pointers of the
  same type or a `*T` and a `nullable *T`; reject ordering, arithmetic,
  bitwise operators, and integer conversions on pointers. Reject `*p`, `p.f`,
  `p[i]`, and `len(p)` on nullable pointers.

- [x] Represent pointers in Fern IR

  Add pointer-typed values, an address-of value for binding, field, and
  element places, an indirect place for loads and stores through a pointer,
  the `null` literal, allocation and free instructions, a null check that
  traps, and pointer equality. Lower the checked forms from the previous tasks
  and verify that indirect places are rooted in pointer-typed values.

- [x] Emit address-of, dereference, and pointer comparison in native code

  Take the address of stack and module-level storage in QBE, load and store
  scalars and aggregates through pointer values, copy pointers as word-sized
  scalars in bindings, fields, elements, parameters, and results, and emit
  pointer equality.

- [x] Emit heap allocation, freeing, and null checks in native code

  Call `malloc` and `free` for `new` and `free`, copy the operand into the new
  target, trap with `out of memory` on a null return, and trap with
  `null pointer` on `*T(p)` when `p` is `null`. Route both traps through the
  existing failure reporting.

- [x] Compile and execute pointer programs and failures

  Add program fixtures and native integration tests for every example in the
  pointer section of the specification, including the linked-list sum, the
  `out of memory` and `null pointer` traps, and the compile-time rejections.
  Add `examples/pointers.fern` to the tour and document it in `README.md`.

- [ ] Build returned aggregate literals in the result storage

  A `return` whose value is an array or struct literal builds the literal in a
  temporary local, copies it into the load's storage, and copies that into the
  caller's result storage. Build it in the result storage directly.

- [ ] Give a returned local the result storage

  A function that returns the same aggregate local on every `return` copies that
  local into the caller's result storage. Allocate the local in the result
  storage instead, so the `return` needs no copy.

- [ ] Initialize bindings from calls without a copy

  `const b = make();` writes the result into the call's own storage and then
  copies it into `b`. Pass `b`'s storage as the result address instead.
