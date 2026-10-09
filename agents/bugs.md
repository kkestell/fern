# Bugs

## Stack checks do not bound QBE spill storage

The backend checks Fern calls against caller and callee allocation storage and
outgoing argument space before entering the callee. It leaves a fixed 64 KiB
reserve for QBE spills, saved registers, alignment, and C library calls.

QBE adds spill storage after Fern emits these checks. The backend does not
bound that storage, so unusually large spill frames can exceed the reserve and
exhaust the stack before the next check can report `stack overflow`. The checks
therefore do not establish that every Fern frame stays within the stack bounds.

Confirmed on macOS arm64: `main` passing 18,000 distinct elements of a global
`[18000]u64` to a function with 18,000 `u64` parameters ends with SIGSEGV and
empty stderr on a 128 KiB stack. The same executable reports `stack overflow`
on a 256 KiB stack and exits successfully on an 8 MiB stack. See
`agents/reviews/2026-10-08-003-trap-on-stack-overflow.md` for the review.

## A literal mixing `*T` and `nullable *T` checks only in one order

With `p: *int` and `q: nullable *int`, an array literal with no context checks
as `[2]nullable *int` when written `[q, p]` but reports
`` cannot implicitly convert `nullable *int` to `*int` `` when written
`[p, q]`. `common_element_type` in `src/semantic/expressions.rs` takes the
first typed element's type, and the other elements are checked against it, so
widening applies only when the nullable element comes first.

`docs/spec.md#array-literals` requires every element of a literal with no
context to have one common type and does not say whether `*T` and
`nullable *T` share one. The fix needs that rule settled first. See
`agents/reviews/2026-10-08-007-check-new-free-nullable-pointers-and-pointer-comparison.md`
for the review.
