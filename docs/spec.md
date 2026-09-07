# Fern Language Specification

This specification defines Fern's language behavior. It is authoritative
wherever it is explicit. An invalid program must be rejected during compilation.
A trap halts execution with a diagnostic on standard error identifying the
failing operation and its source location.

Implementations may impose documented limits on source nesting and compile-time
resource use. Source exceeding such a limit must be rejected with a compilation
diagnostic, rather than crashing the compiler or silently changing a value.

The [roadmap](../eng/roadmap.md) records implementation status.

Fern is a low-level programming language in the spirit of C. Its goals are
clarity, defined behavior, and safe default semantics.

## Lexical Structure

### Source text

Source text is encoded in UTF-8. Source text that is not valid UTF-8 is invalid.
Unicode characters are permitted in comments.

Outside comments, ASCII space, horizontal tab, line feed, carriage return,
vertical tab, and form feed are whitespace. Whitespace and comments separate
tokens but do not otherwise affect syntax. Newlines do not insert semicolons.
Tokens use the longest matching spelling; a keyword is recognized only when it
is the entire identifier, so `exit_code` is an identifier.

### Identifiers

An identifier begins with an ASCII letter (`A`–`Z` or `a`–`z`) or `_`. Each
subsequent character is an ASCII letter, an ASCII digit (`0`–`9`), or `_`.
Non-ASCII characters are not permitted in identifiers.

```text
var player_count = 10;
var _count2 = 20;
var café = 30; // invalid identifier
```

### Comments

A line comment begins with `//` and extends to the end of the line. A block
comment begins with `/*` and ends with its matching `*/`. Block comments nest:
each `/*` within a block comment must have a matching `*/`. An unterminated
block comment is invalid.

```text
var x = 10; // int
/* outer /* inner */ outer */
```

### Keywords

`fn`, `void`, `var`, `const`, `exit`, `true`, and `false` are reserved. The ten
integer type names listed under Integer types and `bool` are also reserved.
Reserved words cannot be identifiers.

### Integer literals

An integer literal contains digits in one of four bases: decimal with no prefix,
hexadecimal with `0x`, binary with `0b`, or octal with `0o`. A prefix must be
followed by at least one digit valid in that base. Hexadecimal digits include
`a`–`f` and `A`–`F`. Digit separators are not allowed.

Leading zeros do not change the base of a decimal literal: `00052` is decimal
52. Octal requires the `0o` prefix, so `0o52` is decimal 42.

Integer literals are untyped and do not accept type suffixes. Use a binding
annotation or an explicit conversion to select an integer type.

```fern
42
0x2A
0b101010
0o52

var a: u8 = 42;
var b = u8(42);
```

### Trailing commas

A trailing comma is permitted in the argument list of `exit`:

```fern
exit(42,);
```

## Declarations

### Functions

The program entry point is a top-level function named `main`, with no parameters
and the explicit return annotation `void`:

```fern
fn main() -> void {
}
```

The body is a brace-delimited sequence of statements and may be empty. No
semicolon follows the closing brace. A program must have exactly one entry
point. Reaching the end of `main` terminates the program with exit status zero.
Explicit termination is defined under Process exit.

### Variable declarations

```text
var name = e;
const name = e;
var name: T = e;
const name: T = e;
```

A declaration introduces a local binding initialized to the value of `e`. A type
annotation determines the binding's type. Without an annotation, the binding
takes the initializer's type, with untyped integer constants defaulting to
`int`. A binding reference has the binding's type. Initialization copies the
integer value; later assignment to the source binding does not change the copy.

### Scope and shadowing

Bindings use lexical block scope. Each brace-delimited statement block,
including a function body, introduces a scope. A binding declared in a block is
not visible outside that block. Nested blocks can access bindings in enclosing
scopes unless those bindings are shadowed.

A local binding becomes visible only after its initializer. The initializer
resolves names using the bindings already in scope, including any earlier
binding with the same name. A reference to a name with no visible declaration is
invalid.

```text
const x = 10;
const x = x; // initializes the new x to 10 using the earlier x
const y = y; // invalid: no earlier y is visible
```

A declaration may reuse the name of an existing binding, whether that binding
was declared with `var` or `const`. The new declaration creates a new binding.
Shadowing is permitted in the same block or a nested block, and the new binding
may have a different type or mutability. The earlier binding is not modified.
When a nested block ends, any outer binding it shadowed becomes visible again.

```text
const x = 10;
const x = 20;
```

```text
const x: int = 10;
{
    var x: u8 = 20;
    x = 30;
}
exit(x); // reports 10: the outer binding is visible again
```

### Immutability

A `var` binding may be reassigned. A `const` binding cannot be reassigned. A
`const` binding may be initialized from a runtime value; immutability does not
require compile-time evaluation. Integer constant expressions defines when a
binding can be used in a constant expression.

```fern
var value = 7;
const saved = value;
value = 42;
exit(saved); // reports 7
```

## Integer Semantics

The integer model uses untyped exact constants, explicit conversions, and
trapping arithmetic.

Design principles:

1. Every integer operation has exactly one meaning, given its operand types.
2. No implicit conversions between integer types, in either direction.
3. Ordinary typed arithmetic traps on out-of-range results. Wrapping operators,
   truncating conversions, and runtime shifts follow their specified bit rules.
4. Exactly two types, `int` and `uint`, have a platform-dependent width. They are distinct from every fixed-width type, so code that mixes them with fixed-width types fails to compile on every platform, not just some.

### Integer types

| Type | Width | Range |
|------|-------|-------|
| `i8` | 8 | −2⁷ … 2⁷−1 |
| `i16` | 16 | −2¹⁵ … 2¹⁵−1 |
| `i32` | 32 | −2³¹ … 2³¹−1 |
| `i64` | 64 | −2⁶³ … 2⁶³−1 |
| `u8` | 8 | 0 … 2⁸−1 |
| `u16` | 16 | 0 … 2¹⁶−1 |
| `u32` | 32 | 0 … 2³²−1 |
| `u64` | 64 | 0 … 2⁶⁴−1 |
| `int` | pointer width | −2ʷ⁻¹ … 2ʷ⁻¹−1 |
| `uint` | pointer width | 0 … 2ʷ−1 |

`int` and `uint` have the width of a pointer on the target platform: 32 or 64
bits. They are always the same width as each other. A conforming implementation
must not choose a width narrower than 32 bits.

All ten types are distinct. `int` is a distinct type from `i32` and `i64` even
when its width matches one of them; the same holds for `uint`. There are no
`byte`, `char`, `short`, or `long` types.

`int` is the general-purpose integer: the default type of constants, the type of
lengths and indices, and the type most code should use. `uint` is the type of
allocation sizes and of integers converted from pointers.

Signed types use two's complement representation.

### Integer constant expressions

#### Untyped constants

Integer literals and ordinary arithmetic and bitwise expressions built entirely
from untyped constants are *untyped*. Constant shifts follow the typing rules
under [Constant shifts](#constant-shifts). Parentheses preserve the enclosed
expression's type and value. An untyped constant is an exact mathematical integer
with no fixed width; the compiler must represent at least 256 bits of precision.
This is a minimum precision guarantee, not a 256-bit integer type. Larger
supported values remain exact, subject to the compiler resource limits described
above. Exceeding a resource limit is distinct from arithmetic overflow.

Untyped constant expressions are evaluated at compile time with exact
arithmetic. Overflow cannot occur in an untyped constant expression. Their
intermediate values need not fit the eventual type; the final value must fit
when it acquires a type. For example, `var x: u8 = (250 + 10) / 2;` initializes
`x` to 130.

Integer literals, integer conversions of constant expressions, integer
operations on constant expressions, and parentheses around constant expressions
are constant expressions. A reference to a `const` binding initialized by a
constant expression is a typed constant expression. References to `var` bindings
and to `const` bindings initialized from runtime values are not constant
expressions.

Constant expressions and constant subexpressions are checked during compilation,
including in unreachable statements. Typed operations obey their type's range
and operation rules; exact intermediate arithmetic applies to untyped
expressions, with the additional rule for typed constant shifts below. Constant
evaluation must reject an operation that would trap, even if a later operation
would bring its result back into range. Optimization of a non-constant
expression must preserve its runtime value and failure behavior.

#### Typing by context

An untyped constant acquires a type from the context in which it is used:

```fern
var a: u8 = 200; // 200 becomes u8
var b: i16 = 1000; // 1000 becomes i16
var c = x + 1; // 1 becomes the type of x
```

It is a compile-time error if the constant's value is not representable in the
target type:

```fern
var a: u8 = 256; // error: 256 does not fit u8
var b: u32 = -1; // error: negative value in unsigned type
var c: i8 = 127 + 1; // error: 128 does not fit i8
```

For `int` and `uint`, representability is checked against the width of the
target platform being compiled for. A constant that fits a 64-bit `int` but not
a 32-bit `int` compiles on 64-bit targets and is rejected on 32-bit targets.
Code intended to be portable to 32-bit targets should use `i64` for such values.

#### Default type

When an untyped integer constant needs a concrete type and no context determines
that type, it becomes `int`:

```fern
var n = 42; // n: int
var big = 1 << 40; // int; compile error on a 32-bit target
```

#### Typed constants

A constant may be declared with an explicit type. Its value must be
representable in that type. It follows the same operand type rules as a
variable of that type, with evaluation governed by Integer constant expressions:

```fern
const Limit: u16 = 65535;
const Bad: u16 = 65536; // error
```

### Integer conversions

There are no implicit conversions between integer types. Widening, narrowing,
and changing signedness all require an explicit conversion. This includes
conversions between `int` and `i64`, or `uint` and `u64`, even on a platform
where the widths coincide.

#### Checked conversion

`T(x)` converts `x` to type `T`. If the value of `x` is not representable in
`T`, the program traps at runtime.

```fern
var a: i64 = 300;
var b = u8(a); // trap: 300 does not fit u8
var c = u16(a); // 300
var d = i8(-1); // -1
var e = u8(i8(-1)); // compile error: constant conversion would trap
var f = int(a); // 300 on every platform
```

Converting an untyped constant follows the rules of Typing by context: the value
must fit or the program does not compile.

#### Truncating conversion

`T.truncate(x)` converts `x` to type `T` by reinterpreting its two's complement
bit pattern at the width of `T`: high bits are discarded on narrowing, and the
value is sign-extended (from a signed source) or zero-extended (from an unsigned
source) on widening. This conversion never traps. An untyped constant is reduced
modulo 2 to the power of the destination width, then interpreted using the
destination signedness; it need not first fit `int` or the destination type.

```fern
var a: i64 = 300;
var b = u8.truncate(a); // 44
var c = u8.truncate(-1); // 255
var d = i8.truncate(200); // -56
```

### Integer arithmetic

#### Operand types

Binary arithmetic operators require both operands to have the identical type.
There is no promotion: `u8 + u8` is `u8`. Untyped constants adopt the type of
the other operand. It is a compile-time error to combine two differently typed
integers.

```fern
var a: u8 = 10;
var b: u16 = 20;
var c = a + b; // error: u8 and u16
var d = u16(a) + b; // ok
var e = a + 5; // ok, 5 becomes u8
var n: int = 3;
var m: i64 = 4;
var p = n + m; // error: int and i64, on every platform
```

#### Integer operators

| Operator | Meaning |
|----------|---------|
| `+ - *` | Addition, subtraction, multiplication. Trap on overflow. |
| `/` | Division, truncating toward zero. Traps on division by zero and on `MIN / -1`. |
| `%` | Remainder. Result has the sign of the dividend. Traps when `/` would. |
| `&+ &- &*` | Wrapping addition, subtraction, multiplication (two's complement). Never trap. |
| unary `-` | Negation. Traps on `-MIN`. Not permitted on unsigned types. |
| unary `&-` | Wrapping negation. Permitted on all types. |

As a mathematical identity, division and remainder satisfy
`(a / b) * b + a % b == a` whenever both are defined.

Division or remainder by a constant zero is a compile-time error, including when
the dividend is not constant. Any constant expression that would trap is also a
compile-time error.

#### Overflow

An arithmetic result that is not representable in the operand type is an error.
The program traps: execution halts with a diagnostic identifying the operation.
This is the defined behavior of the language, not a debug-mode check; a
conforming implementation traps in every build configuration.

An implementation may elide an overflow check when it can prove the result is in
range.

Wrapping operators explicitly request modular arithmetic at the operand type's
width. The retained bits are interpreted using that type's signedness.

```fern
var x: u8 = 255;
var y = x + 1; // trap
var z = x &+ 1; // 0

var h: u64 = 14695981039346656037;
var h2 = h &* 1099511628211; // FNV hashing, wraps by design
```

Because `int` overflow traps rather than wraps, code that overflows a 32-bit
`int` but not a 64-bit one traps on 32-bit targets rather than silently
computing a different result.

### Indexing and lengths

The length of any array, slice, or string has type `int`. Indices and slice
bounds have type `int`.

An index is valid if `0 <= i < len`. Accessing an out-of-range index traps.
Because indices are signed, an expression such as `i - 1` when `i == 0` produces
`-1` and fails the bounds check rather than wrapping to a huge value.

The maximum length of an object is `2ʷ⁻¹ − 1` elements, where `w` is the pointer
width. Allocation sizes use `uint`; converting between lengths and allocation
sizes follows Integer conversions.

### Bitwise operations

| Operator | Meaning |
|----------|---------|
| `&` | And |
| `\|` | Or |
| `^` | Exclusive or |
| `&^` | And-not (bit clear) |
| unary `^` | Complement |

Binary bitwise operators require identical operand types, with untyped constants
adopting the other operand's type when it fits. Typed operations use that type's
two's-complement bit representation and cannot overflow. Untyped bitwise
operations use an unbounded two's-complement interpretation; unary `^x` is
`-x - 1`.

### Integer shifts

`x << n` and `x >> n` shift `x` by `n` bit positions. The count may have any
integer type or be an untyped constant. Its type does not determine the left
operand's type or the result type. An untyped count remains an exact integer in
both constant and non-constant shifts; it need not first fit `int` or any other
concrete type.

Non-constant shifts follow these execution rules; constant shifts are specified
separately below.

- The result type is the type of the left operand.
- A negative shift count traps.
- A shift count greater than or equal to the width of the left operand's type is defined: `<<` yields `0`; `>>` yields `0` for unsigned and for non-negative signed values, and `-1` for negative signed values. It does not trap.
- `>>` is arithmetic (sign-filling) for signed types and logical (zero-filling) for unsigned types.
- For a non-negative count, `<<` discards bits shifted out and never traps. It is a bit operation, not multiplication.

#### Constant shifts

If both operands are constant expressions and the left operand is untyped, the
shift is evaluated exactly and its result remains untyped. This applies whether
the count is typed or untyped. `1 << 40` is the untyped integer 2⁴⁰;
`1 << u8(100)` is the untyped integer 2¹⁰⁰. The result then acquires a type from
context in the ordinary way (Typing by context) and must be representable in it:

```fern
var a: i64 = 1 << 40; // ok
var b = 1 << 40; // int; ok on 64-bit, compile error on 32-bit
var c: i32 = 1 << 40; // compile error on every platform
var reduced: u8 = 256 >> u8(8); // 1; only the final result must fit u8
```

If both operands are constant expressions and the left operand is typed, the
shift is still evaluated exactly, and the result must be representable in that
type; a `<<` that would discard bits is a compile-time error, not a silent
truncation.

```fern
const one: i32 = 1;
var d = one << 40; // compile error: 2⁴⁰ does not fit i32
```

Constant left shifts multiply exactly by two to the power of the count. Constant
right shifts divide by that power, rounding toward negative infinity. The
runtime rules above (zero on overshift, discarded high bits) do not apply to
constant shifts. A negative constant count is a compile-time error, including
when the left operand is not constant.

#### Non-constant shifts

If either operand is not a constant expression, the shift is evaluated at
runtime under the rules above. An untyped constant on the left takes its type
from context first, defaulting to `int`:

```fern
var n = 40; // a variable reference is not a constant expression
var e = 1 << n; // int(1) shifted by n; 0 if n >= width
var f: u64 = 1 << n; // u64(1) shifted by n
```

A typed left operand retains its type regardless of the destination.

### Comparison

`bool` is a distinct type whose values are `true` and `false`. The comparisons
`== != < <= > >=` require identical operand types and yield `bool`. Untyped
constants adopt the type of the other operand when their values fit.

A well-typed comparison remains valid when the operand types make its result
always true or always false. For unsigned `x`, `x < 0` and `0 > x` yield `false`,
and `x >= 0` yields `true`. For `x: u8`, `x <= 255` yields `true`. A constant
that does not fit the other operand's type still makes the comparison invalid;
for example, `x < -1` is invalid when `x` is unsigned.

### Assignment

`x = e;` stores the value of `e` in the mutable local binding `x`. It requires
`e` to have the type of `x`, or to be an untyped constant that fits. Assignment
is a statement and ends with a semicolon.

Compound assignments `+= -= *= /= %= &= |= ^= &^= <<= >>=` and their wrapping
forms `&+= &-= &*=` are equivalent to the corresponding binary operation
followed by assignment, and trap under the same conditions.

There are no `++` or `--` operators.

### Portability notes

The only platform-dependent behavior in this specification is the width of `int`
and `uint`. Its consequences are:

- Untyped constants assigned to `int`/`uint` may fit on one platform and be rejected at compile time on another (Typing by context).
- Checked conversions to `int`/`uint` may succeed on one platform and trap on another (Checked conversion).
- Arithmetic on `int`/`uint` may succeed on one platform and trap on another (Overflow).
- Truncating conversions to or from `int`/`uint` produce different values on different platforms (Truncating conversion).
- Shifts on `int`/`uint` with counts between 32 and 63 produce different results on different platforms (Integer shifts).

Checked arithmetic and checked conversions preserve mathematical values when
they succeed. Explicit wrapping, truncating conversions, and runtime shifts can
produce different in-range results at different widths. Code that needs a
specific width should use fixed-width types; their distinctness from `int` and
`uint` prevents accidental mixing.

### Summary of trapping behavior

| Situation | Behavior |
|-----------|----------|
| Constant does not fit target type | Compile error |
| Constant expression would trap | Compile error |
| `+ - *` overflow | Trap |
| `/` or `%` by zero | Trap |
| `MIN / -1`, `MIN % -1`, `-MIN` | Trap |
| Checked conversion out of range | Trap |
| Negative shift count | Trap |
| Out-of-range index | Trap |
| Runtime shift count ≥ width | Defined, no trap |
| Constant shift result does not fit its type | Compile error |
| Wrapping operators | Defined, no trap |
| Truncating conversion | Defined, no trap |
| Bitwise operators | Defined, no trap |

## Statements and Execution

A function body and each nested brace-delimited block execute statements in
source order. Declarations, assignments, compound assignments, and `exit`
statements end with `;`. A nested block needs no trailing semicolon.
Declarations and references must be valid even after a statement that terminates
execution.

### Process exit

```text
exit(e);
```

The built-in `exit` takes one argument of type `int` and terminates the program.
An untyped constant argument must fit `int`; a differently typed integer
requires an explicit checked or truncating conversion. The reported exit status
is the argument modulo 256, in the range zero through 255, on every host.
`exit(256);` reports zero and `exit(-1);` reports 255. It does not return.

```fern
fn main() -> void {
    exit(42);
}
```

## Modules and Imports

### Module directories

A module is a directory of `.fern` source files. Its directory path relative to
a module search root is its import path, with `::` separating path components.
All source files in the directory share one module namespace and can access each
other's declarations, including private declarations. Each source file has its
own imports.

The directory defines the namespace; source filenames do not introduce
namespaces. For example, `example/print.fern` belongs to module `example`. A
function named `println` declared in that file has the qualified name
`example::println`.

### Use declarations

Imports are file-local `use` declarations terminated by `;`. Importing a module
introduces its name in that file; access to its members is qualified with `::`.
A selective import introduces the selected submodule or member for unqualified
use in that file.

```text
use fmt;        // import a standard-library module
use fs::{flag}; // introduce flag for unqualified use
use example;   // import a module found through the module search path

example::println(...);
```

A `use` declaration in one file does not introduce imported names in other
files, even when those files belong to the same module.

### Module resolution and dependencies

Module resolution is path-based. `FERNPATH` is a colon-delimited list of module
search roots, searched in order. The default search order is the current
directory, the standard library, and then third-party modules. Typical locations
for the latter two roots are `/usr/src/fern/stdlib` and
`/usr/src/fern/third-party`.

Fern has no manifest, package manager, or lockfile. Installing a dependency
means placing its source tree under a module search root. Fern does not provide
a package registry or perform dependency version resolution. Projects manage
their own dependency source trees, for example through copies or Git submodules.

### Open module questions

The specification does not yet settle public-declaration syntax, import name
conflicts, dependency cycles, missing-module diagnostics, or whether setting
`FERNPATH` replaces or extends the default search list.
