# Values, References, Memory, and C Interop

Proposed design for how Fern passes data, manages memory, and calls C. Nothing
here is part of the language until it is adopted into `docs/spec.md` through
`kspec`, which owns the current rules for arrays, structs, parameters, and
`const`.

## Goals

- Scalars (`int`, floats, `bool`, …) are passed by value.
- Structs are the workhorse type. They have value semantics, are cheap to pass,
  and can be passed by reference when a callee must modify them.
- Arrays are a low-level building block. Generic collections will serve most
  user code.
- Memory is garbage collected with libgc (the Boehm–Demers–Weiser collector).
- Calls into C have little runtime overhead and little programmer friction.
- Pointers exist for C interop and are never null. Unsafe operations are
  confined to `unsafe { }` blocks.

## Recommended Design

### Every type is a value

Scalars, structs, and fixed arrays `[N]T` are all value types: initialization,
assignment, argument passing, and return copy the whole value. Passing by
reference is a property of a parameter, not of a type.

Arrays follow the same rules as structs. A reference-typed array would make an
array field ambiguous (copied with its struct, or shared) and would need a home
for its storage.

### Parameter modes

| Mode | Meaning | Argument |
| --- | --- | --- |
| `p: T` | Read-only value | Any expression of type `T` |
| `p: mut T` | Borrowed mutable reference | `mut place`, where `place` is assignable |

A `p: T` parameter is immutable, so the compiler may pass it by hidden pointer
whenever that is faster. Odin's default calling convention does this.

A `mut` argument names an assignable place: a `var` binding, or a field or
element of one. A `const` binding is not assignable and cannot be passed `mut`.
The call site spells `mut` so mutation is visible to the reader.

`mut` references are second-class. A callee may read and assign through one
and pass it on as a `mut` argument. It cannot store it, return it, or place it
in a struct or array. A `mut` reference therefore never outlives the
place it refers to, which lets it point at stack storage safely without
lifetime annotations or escape analysis.

A call may not pass a place as a `mut` argument and also pass that place, or
any place overlapping it, as another argument. This exclusivity rule keeps
hidden-pointer passing of `p: T` parameters unobservable: no write through the
`mut` reference can change a read-only parameter during the call. Swift
enforces the same rule as its law of exclusivity.

### Heap references

`new(e)` allocates a copy of `e` on the garbage-collected heap and yields a
`ref T`. A `ref T` is a first-class value: it may be stored, returned, and
placed in aggregates, and it is never null. Copying a `ref T` copies the reference, and both copies
refer to the same object. Field and element access through a `ref T` uses the
same `.` and `[]` syntax as on the value.

Heap allocation is explicit. Every allocation is visible in the source, and the
QBE backend needs no escape analysis to keep locals on the stack.

### `const`

`const` stays a binding property and is not a type qualifier. Because every
type is a value, an immutable binding is deeply immutable: no field or element
of a `const` struct or array can change. `const` bindings cannot be passed as
`mut` arguments. A `const` initialized by a constant expression remains usable
in constant expressions, so one keyword covers both immutable runtime bindings
and compile-time constants.

### Garbage collection with libgc

libgc scans stacks, registers, and static data conservatively and needs no
stack maps, which QBE cannot emit. Fern links libgc and allocates `new`
objects with `GC_MALLOC`. Consequences:

- libgc becomes a native toolchain requirement listed in `README.md`.
- Threads must be registered with the collector, through its pthread wrappers,
  once Fern has threads.
- Conservative scanning can retain unreachable objects when an integer looks
  like a pointer. Objects never move.

### C interop

Fern code runs on the native stack with no scheduler, so a C call is a plain
machine call with no stack switch. QBE implements the C ABI in full.

**Declarations.** An `extern fn` declares a C function with no body. Calls to
it use the C calling convention. Fern functions keep their own convention,
which is free to pass read-only aggregates by hidden pointer.

**Layout.** Fern struct layout is C struct layout for the target: fields in
declaration order, each at its natural alignment, with padding as C inserts it.
Fixed arrays inside structs match C arrays inside structs.

**Type mapping.**

| Fern | C |
| --- | --- |
| `i8` … `u64`, `f32`, `f64`, `bool` | `int8_t` … `uint64_t`, `float`, `double`, `_Bool` |
| `int`, `uint` | `ptrdiff_t`, `size_t` |
| `c_char`, `c_int`, `c_long`, … | `char`, `int`, `long`, … |
| struct `T` by value | struct `T` by value |
| `p: mut T` | `T *`, borrowed for the call |
| read-only borrow of `T` | `const T *`, borrowed for the call |
| `*T` | `T *` that is never null |
| nullable `*T` | `T *` that may be null |

The `c_` aliases are platform-dependent integer types matching the target's C
types, as in Hare's `types::c` and Odin's `core:c`.

With this mapping, the common C pattern of a pointer borrowed for one call
binds without conversion or `unsafe`:

```fern
extern fn get_size(w: Window, out: mut Size) -> void;
```

**Pointers.** `*T` is a pointer with the representation of C's `T *` that is
never null. It can be stored and returned. A separate nullable pointer type
represents a C pointer that may be null; it must be tested against null before
it can be used as a `*T`, as Hare requires for `nullable *T`. An `extern`
declaration uses the nullable type wherever the C function may accept or return
null. Dereferencing, pointer arithmetic, and conversion between pointers and
references or integers are `unsafe` operations: a non-null pointer can still
dangle.

**Safety boundary.** The `extern` declaration is where the programmer vouches
for a C function. A call to an `extern fn` whose parameters are values and
borrows is an ordinary call. Only pointer operations require `unsafe`.

**GC memory and C.** A garbage-collected object passed to C stays alive for the
duration of the call, because libgc scans the C frames on the same stack. C
code that retains a pointer to a garbage-collected object in memory C allocated
needs that object allocated uncollectable (`GC_MALLOC_UNCOLLECTABLE`) or
registered as a root. Memory C allocates arrives as `*T` and is freed
explicitly.

**Strings.** When Fern gains strings, string literal data is followed by a NUL
byte so literals pass to C without copying.

**Traps.** A trap aborts the process. Fern never unwinds through C frames.

**Bindings.** `extern` declarations are written by hand. Generating them from C
headers is a later step.

## Comparison

| Topic | Go | Odin | Hare |
| --- | --- | --- | --- |
| Arrays | Values; each array holds a complete copy of its data | Fixed arrays are "equivalent to a struct with a field for each element"; copy semantics not stated in the overview | Not settled by the spec sections checked |
| Parameters | Arguments are assigned to parameters, which are variables | Immutable; parameters over 16 bytes pass by reference, others promote to a pointer when more efficient | Not settled by the spec sections checked |
| Pointers | Explicit `*T`; may be `nil` | Explicit `^T`; zero value is `nil` | Explicit `*T`, never null; `nullable *T` must be tested before use |
| Constants | `const` is compile-time only: boolean, rune, numeric, string | `::` constants must be compile-time | `def` is compile-time with no storage; `const` binding is immutable; `const` type flag prohibits writes |
| Memory | Garbage collected; escaping locals move to the heap by escape analysis | Manual, through allocators | Manual, `alloc` and `free` |
| Foreign calls | `C.f(...)` is an ordinary call; raw conversions use `unsafe.Pointer` | `foreign` blocks; foreign procedures default to the C convention | `@symbol` declarations |
| GC pointers in C | Pinned implicitly for the call; retained pointers need `runtime.Pinner` | No GC | No GC |
| C strings | `C.CString` copies into the C heap; caller frees | `cstring` type | `nulstr` borrows a NUL-terminated string; `fromstr` allocates |

Go's cgo call overhead comes from its implementation (goroutines on
runtime-managed stacks), not from its specification.

## Open Questions

1. **Explicit or implicit heap allocation.** Recommended: explicit `new` and
   `ref T`. The alternative is Go's model, where any variable's address can be
   taken and escaping locals move to the heap; it is simpler to write but hides
   allocation and needs escape analysis to perform well.
2. **`unsafe` and `extern` calls.** Recommended: ordinary calls, with `unsafe`
   only for pointer operations. Go, Odin, and Hare all make foreign calls
   ordinary calls; Odin and Hare have no `unsafe` blocks.
3. **Read-only borrow mode.** C's `const T *` needs a parameter form distinct
   from `p: T`, because a by-value struct is also a valid C parameter. Its
   spelling, and whether it is available outside `extern` declarations, are
   undecided.
4. **Nullable pointer spelling.** Whether the nullable form is a pointer-only
   type, like Hare's `nullable *T`, or an instance of a general optional type
   is undecided.
5. **Names of the C integer aliases.**

## Sources

- Go specification: https://go.dev/ref/spec
- Go FAQ, stack and heap allocation: https://go.dev/doc/faq
- cgo: https://pkg.go.dev/cmd/cgo
- Odin overview: https://odin-lang.org/docs/overview/
- Hare specification sources: https://git.sr.ht/~sircmpwn/hare-specification
- Hare introduction: https://harelang.org/tutorials/introduction/
- Hare `types::c`: https://docs.harelang.org/types/c
- QBE: https://c9x.me/compile/
