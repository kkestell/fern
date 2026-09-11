# Integers, Bits, and C Weirdness

A computer does not store the number `five`. It stores patterns of bits:

```text
0 1 0 1 1 0 0 1
```

A bit is a tiny switch with two possible states:

```text
0 = off
1 = on
```

## Counting With Bits

Each position represents a power of two:

```text
  1   0   1   1
  |   |   |   |
  8   4   2   1
```

So:

```text
1011₂ = 8 + 2 + 1 = 11
```

With `N` bits, an unsigned integer can represent:

```text
0 through 2^N - 1
```

For 8 bits:

```text
00000000 =   0
11111111 = 255
```

There are only 256 possible patterns. If you need a larger number, you need more bits.

## What Hardware Does

A CPU has circuits that operate on fixed-size groups of bits, such as 8, 16, 32, or 64 bits.

This document follows the usual AArch64 model; x86-64 is the same in all the ways that matter here.

An 8-bit addition looks like this:

```text
  11111111   255
+ 00000001     1
-----------
1 00000000   256
```

The mathematical result is too large for eight bits, so an overflow bit is produced and discarded:

```text
00000000 = 0
```

For an unsigned integer, this is called wrapping: the result is taken modulo `2^8`, or 256.

The CPU may also set status flags:

```text
carry    = an extra bit left the integer
overflow = the signed result was too large
zero     = the result was zero
negative = the top bit was set
```

These flags are information for later instructions. They do not necessarily stop the program.

An integer operation usually does not trap on overflow. The CPU often just keeps the low bits. Division by zero, invalid instructions, and memory faults are examples of operations that commonly trap.

A trap is the hardware saying:

```text
"This operation cannot continue normally."
```

The operating system usually turns that into a signal or process failure.

## Signed Integers

Unsigned integers use every bit for magnitude. Signed integers need to represent negative values too.

Modern machines use **two's complement** for signed integers.

For 8 bits:

```text
00000000 =   0
00000001 =   1
01111111 = 127

10000000 = -128
11111111 =  -1
```

The top bit acts as a sign indicator, but the whole pattern participates in the value.

The range is asymmetric:

```text
-128 through 127
```

There are 128 negative values, but only 127 positive values, plus zero.

This creates an important surprise:

```text
01111111   127
+00000001     1
-----------
10000000  -128
```

At the hardware level, this is usually just bit addition. The CPU may set an overflow flag, but it may not trap.

## Wrapping

Unsigned arithmetic wraps around because there are only finitely many patterns:

```c
#include <stdint.h>

uint8_t x = 255;
x = x + 1;       // 0
```

Conceptually:

```text
255 + 1 = 256
256 modulo 256 = 0
```

The same bit operation can produce a surprising result for signed integers:

```c
int8_t x = 127;
x = x + 1;
```

On common machines this appears to produce `-128`. But that is not a portable assumption in C. The language gives signed overflow special treatment.

## C's First Weirdness: Signed Overflow Is Undefined

In C:

```c
int x = 2147483647;
x = x + 1;
```

If `int` cannot represent the result, the behavior is **undefined**.

Undefined behavior does not mean "the program wraps." It means the C standard places no requirements on what happens. A compiler may assume this situation never occurs and optimize based on that assumption.

For example:

```c
int f(int x) {
    if (x + 1 > x) {
        return 1;
    }
    return 0;
}
```

A compiler can treat this as always returning `1`, because signed overflow is assumed not to happen.

If you want defined wrapping arithmetic, use an unsigned type:

```c
unsigned x = UINT_MAX;
x = x + 1;       // 0, by definition
```

## C's Second Weirdness: Integer Types Have Rules, Not Just Sizes

C has types such as:

```c
char
short
int
long
long long
```

Their exact sizes depend on the platform. C guarantees minimum ranges, not one universal layout.

For exact-width integers, use `<stdint.h>`:

```c
#include <stdint.h>

int32_t  signed_value;
uint32_t unsigned_value;
```

These types exist only when the platform provides a suitable representation.

On common systems, `int` is 32 bits, but the sizes of types such as `long` depend on the platform and its C ABI. Do not infer a C type's size from the CPU's name alone.

## C's Third Weirdness: Small Integers Become `int`

Expectation from the box model: `a` and `b` are 8-bit boxes, so
`a + b` is 8-bit addition that wraps modulo 256. Its value always fits
in `0` through `255`.

Abstract integer math says `255 + 1 = 256`, so 8-bit wrapping would
reduce that to `0` immediately.

C does the addition in a wider type:

```c
uint8_t a = 255;
uint8_t b = 1;

int c = a + b;   // 256, not 0
```

Before the addition, `a` and `b` are usually promoted to `int`. The
calculation is commonly:

```text
255 + 1 = 256   // int, no wrapping
```

So `c` holds `256`, a value no `uint8_t` can hold. Adding two 8-bit
values produced a value outside `0` through `255`.

Truncation happens only when the result is stored back into an 8-bit
type:

```c
uint8_t d = a + b;   // 0
```

```text
256 modulo 256 = 0
```

This is called an **integer promotion**: small integer types such as
`char`, `signed char`, `unsigned char`, `short`, and `uint8_t` are
promoted to `int` (or to `unsigned int` when `int` cannot hold every
value) before most arithmetic. The arithmetic no longer happens in the
original 8- or 16-bit box.

## C's Fourth Weirdness: Signed and Unsigned Mix Badly

Consider:

```c
int a = -1;
unsigned int b = 1;

if (a < b) {
    puts("less");
}
```

Many programmers expect `-1 < 1` to be true.

But C may convert `a` to `unsigned int` first:

```text
-1 becomes UINT_MAX
```

The comparison becomes approximately:

```text
UINT_MAX < 1
```

which is false.

This is one reason to avoid mixing signed and unsigned values casually.

## C's Fifth Weirdness: `char` Is Not Clearly Signed

Plain `char` may be either signed or unsigned:

```c
char c = 200;
```

Whether `c` is positive or negative depends on the implementation.

Use an explicit type when the sign matters:

```c
signed char temperature;
unsigned char byte;
```

Use `char` for text characters, not as a portable numeric byte type.

## C's Sixth Weirdness: Shifts Have Sharp Edges

This is usually safe:

```c
unsigned x = 1;
unsigned y = x << 3;   // 8
```

But shifts involving signed values have restrictions.

```c
int x = 1;
int y = x << 31;       // potentially undefined
```

Problems include:

- Shifting by a negative amount
- Shifting by an amount greater than or equal to the type width
- Left-shifting a signed value into an unrepresentable result
- Left-shifting a negative signed value

Right-shifting a negative signed value is also not fully portable. Some systems fill with `1` bits, others may behave differently.

Use unsigned integers when treating values as bit patterns:

```c
uint32_t flags = 0;
flags |= 1u << 5;
```

## C's Seventh Weirdness: Bytes Are About Storage

A C byte is the size of a `char`, not necessarily eight bits. Most modern computers use eight-bit bytes, but C does not require it.

The macro `CHAR_BIT` tells you how many bits are in a byte:

```c
#include <limits.h>

printf("%d\n", CHAR_BIT);
```

Memory can be viewed as a sequence of bytes:

```text
address:  1000  1001  1002  1003
bytes:      78    56    34    12
```

A multi-byte integer can store those bytes in different orders.

Little-endian:

```text
lowest address
      |
      v
78 56 34 12
```

Big-endian:

```text
lowest address
      |
      v
12 34 56 78
```

The numeric value is the same only if the reader knows the chosen byte order.

## The Useful Mental Model

Think of an integer as:

```text
a fixed-size box of bits
```

The hardware performs operations on the box. The programming language decides:

- Which bit patterns mean which values
- What happens when the result does not fit
- Whether conversions are allowed
- Whether the compiler may assume something never happens
- Whether an invalid representation traps or causes undefined behavior

For portable C:

- Use unsigned types for intentional modulo arithmetic and bit manipulation.
- Use signed types for ordinary quantities that should not overflow.
- Check before signed arithmetic can overflow.
- Avoid mixing signed and unsigned values.
- Do not assume integer sizes, `char` signedness, byte order, or overflow behavior.
- Remember that C describes an abstract machine, not merely the behavior of the CPU underneath.
