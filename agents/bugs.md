# Bugs

## `len` copies its array operand

`len(a)` lowers its operand as a value and then discards it. The array type
`len` accepts makes that value a full copy, so `len` emits a stack copy of the
whole array whose bytes the result never reads. A small array wastes work; an
array larger than the stack ends the process with a bare SIGSEGV.

```fern
extern fn exit(status:i32)->void;
extern fn malloc(bytes:uint)->*u8;
fn main()->void {
    var raw = malloc(sizeof([1200000]i64));
    if raw == null { exit(2); }
    var big = *[1200000]i64(raw);
    if len(*big) != 1200000 { exit(1); }
}
```

The 9.6 MB array does not fit the 8 MiB stack, and the copy overflows it. The
length is a compile-time constant, so a variable operand with no side effects
needs no evaluation at all: `agents/plans/2026-09-09-011-represent-arrays-in-fern-ir.md`
records that `len(a)` "folds to the length with no instructions". `lower_length`
and the `Length` arm of `lower_folded_effects` in `src/ir/lower.rs` instead
lower the operand as a value, and the emitter turns the aggregate `Load` into
stack storage plus a copy.
