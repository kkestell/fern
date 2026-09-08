# Fern Language Specification

This specification defines Fern's language behavior. It is authoritative
wherever it is explicit. An invalid program must be rejected during compilation.
A trap halts execution with a diagnostic on standard error identifying the
failing operation and its source location. It terminates the process abnormally
through the platform's abort mechanism. The resulting termination status is
platform-defined and is never a status that `exit` can report.

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

`fn`, `void`, `var`, `const`, `if`, `else`, `for`, `break`, `continue`, `exit`,
`true`, and `false` are reserved. The ten integer type names listed under
Integer types and `bool` are also reserved. Reserved words cannot be
identifiers.

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

A declaration introduces a binding initialized to the value of `e`. A type
annotation determines the binding's type. Without an annotation, the binding
takes the initializer's type, with untyped integer constants defaulting to
`int`. A binding reference has the binding's type. Initialization copies the
integer value; later assignment to the source binding does not change the copy.

### Module-level declarations

A source file contains function declarations and module-level `var` and `const`
declarations, in any order. Statements appear only in function bodies.

A module-level declaration uses the declaration syntax above, and its
initializer must be a constant expression. Module-level bindings are initialized
before `main` runs.

A module-level binding is visible throughout its file, including in declarations
that appear before it. Two module-level declarations in a file may not use the
same name. A module-level initializer that refers to the binding it initializes,
directly or through other module-level bindings, is a compile-time error.

```fern
var counter = base;
const base = 40;

fn main() -> void {
    counter = counter + 2;
    exit(counter); // reports 42
}
```

### Scope and shadowing

Bindings use lexical block scope. Each brace-delimited statement block,
including a function body, introduces a scope. A binding declared in a block is
not visible outside that block. Nested blocks can access bindings in enclosing
scopes unless those bindings are shadowed. The bindings of a file's module-level
declarations enclose every function body in that file, and a local binding may
shadow one of them.

A local binding becomes visible only after its initializer. The initializer
resolves names using the bindings already in scope, including any earlier
binding with the same name. A reference to a name with no visible declaration is
invalid.

```text
fn main() -> void {
    const x = 10;
    const x = x; // initializes the new x to 10 using the earlier x
    const y = y; // invalid: no earlier y is visible
}
```

A declaration may reuse the name of an existing binding, whether that binding
was declared with `var` or `const`. The new declaration creates a new binding.
Shadowing is permitted in the same block or a nested block, and the new binding
may have a different type or mutability. The earlier binding is not modified.
When a nested block ends, any outer binding it shadowed becomes visible again.

```text
fn main() -> void {
    const x = 10;
    const x = 20;
}
```

```text
fn main() -> void {
    const x: int = 10;
    {
        var x: u8 = 20;
        x = 30;
    }
    exit(x); // reports 10: the outer binding is visible again
}
```

### Immutability

A `var` binding may be reassigned. A `const` binding cannot be reassigned. A
local `const` binding may be initialized from a runtime value; immutability does
not require compile-time evaluation. Integer constant expressions defines when a
binding can be used in a constant expression.

```fern
fn main() -> void {
    var value = 7;
    const saved = value;
    value = 42;
    exit(saved); // reports 7
}
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
under [Constant shifts](#constant-shifts): a constant shift with an untyped left
operand stays untyped even when its count is a typed constant. Parentheses
preserve the enclosed expression's type and value. An untyped constant is an
exact mathematical integer with no fixed width; the compiler must represent at
least 256 bits of precision. This is a minimum precision guarantee, not a
256-bit integer type. Larger supported values remain exact, subject to the
compiler resource limits described above. Exceeding a resource limit is distinct
from arithmetic overflow.

Untyped constant expressions are evaluated at compile time with exact
arithmetic. Overflow cannot occur in an untyped constant expression. Their
intermediate values need not fit the eventual type; the final value must fit
when it acquires a type. For example, `var x: u8 = (250 + 10) / 2;` initializes
`x` to 130.

Integer literals, integer conversions of constant expressions, and parentheses
around constant expressions are constant expressions. So is an arithmetic,
wrapping, bitwise, shift, or unary operation whose operands are all constant
expressions. A reference to a `const` binding initialized by a constant
expression is a typed constant expression. References to `var` bindings and to
`const` bindings initialized from runtime values are not constant expressions.

Constant expressions and constant subexpressions are checked during compilation,
including in unreachable statements. Typed operations obey their type's range
and operation rules; exact intermediate arithmetic applies to untyped
expressions. Constant evaluation must reject an operation that would trap, even
if a later operation would bring its result back into range. Optimization of a
non-constant expression must preserve its runtime value and failure behavior.

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
const limit: u16 = 65535;
const bad: u16 = 65536; // error
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

#### Precedence, associativity, and evaluation order

Unary operators have the highest precedence. Binary operators have five
precedence levels, from highest to lowest:

1. `* / % *% << >> &`
2. `+ - +% -% | ^`
3. `== != < <= > >=`
4. `&&`
5. `||`

Binary operators at the same precedence level associate from left to right.
Parentheses override precedence and associativity.

Every comparison binds less tightly than every arithmetic and bitwise operator,
so `x & 1 == 0` is `(x & 1) == 0`. Shifts bind as tightly as multiplication, so
`a + b << c` is `a + (b << c)`. Bitwise or and exclusive or bind as loosely as
addition, so `a | b + c` is `(a | b) + c`. `&&` binds more tightly than `||`, and both
bind less tightly than every comparison, so `a < b && c < d || e` is
`((a < b) && (c < d)) || e`.

The left operand of a binary expression is evaluated before the right operand.
Each operand is fully evaluated, including any runtime failure, before evaluation
of the next operand begins. `&&` and `||` do not always evaluate their right
operand; see [Logical operators](#logical-operators).

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
| `+% -% *%` | Wrapping addition, subtraction, multiplication (two's complement). Never trap. |
| unary `-` | Negation. Traps on `-MIN`. Not permitted on unsigned types. |
| unary `-%` | Wrapping negation. Permitted on all types. |

As a mathematical identity, division and remainder satisfy
`(a / b) * b + a % b == a` whenever both are defined.

Division or remainder by a constant zero is a compile-time error, including when
the dividend is not constant. That rejection is deliberate and does not depend
on the whole expression being a constant expression. Separately, any constant
expression that would trap is also a compile-time error.

#### Overflow

An arithmetic result that is not representable in the operand type is an error.
The program traps: execution halts with a diagnostic identifying the operation.
This is the defined behavior of the language, not a debug-mode check; a
conforming implementation traps in every build configuration.

An implementation may elide an overflow check when it can prove the result is in
range.

Wrapping operators explicitly request modular arithmetic at the operand type's
width. The retained bits are interpreted using that type's signedness. A binary
wrapping operation requires at least one typed operand; an untyped operand adopts
the other operand's type and must fit it before the operation. In
`var x: u8 = 250 +% 10;` both operands are untyped, and a destination type does
not supply the operation's width, so the declaration is invalid.
Unary wrapping negation likewise requires a typed operand. `-%u8(1)` is
`u8(255)`, while `-%1` is invalid.

```fern
var x: u8 = 255;
var y = x + 1; // trap
var z = x +% 1; // 0

var h: u64 = 14695981039346656037;
var h2 = h *% 1099511628211; // FNV hashing, wraps by design
```

Because `int` overflow traps rather than wraps, code that overflows a 32-bit
`int` but not a 64-bit one traps on 32-bit targets rather than silently
computing a different result.

### Indexing and lengths

Arrays, slices, and strings are specified ahead of their implementation; the
[roadmap](../eng/roadmap.md) records implementation status.

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
| unary `^` | Complement |

Binary bitwise operators require identical operand types, with untyped constants
adopting the other operand's type when it fits. Typed operations use that type's
two's-complement bit representation and cannot overflow. Untyped bitwise
operations use an unbounded two's-complement interpretation; unary `^x` is
`-x - 1`. There is no and-not operator; `a & ^b` clears the bits set in `b`.

### Integer shifts

`x << n` and `x >> n` shift `x` by `n` bit positions. The count may have any
integer type or be an untyped constant. Its type does not determine the left
operand's type or the result type. In a constant shift, an untyped count remains
an exact integer and need not fit a concrete type. In a non-constant shift, an
untyped count acquires type `int`.

Typed shifts and non-constant shifts follow these execution rules; untyped
constant shifts are specified separately below.

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

If the left operand is typed, a constant shift uses the same bit-discarding,
sign-filling, and overshift rules as a non-constant shift. Changing a binding
from `const` to `var` does not change the result of a typed shift. Only a typed
left operand is invariant this way. With an untyped left operand, a `var`
reference makes the shift non-constant, and
[Non-constant shifts](#non-constant-shifts) apply.

```fern
const high: u8 = 128;
const discarded = high << 1; // u8(0)
const negative: i8 = -1;
const sign_fill = negative >> 8; // i8(-1)
```

Untyped constant left shifts multiply exactly by two to the power of the count.
Untyped constant right shifts divide by that power, rounding toward negative
infinity. A negative constant count is a compile-time error, including when the
left operand is not constant.

#### Non-constant shifts

If either operand is not a constant expression, the shift is evaluated at
runtime under the rules above. An untyped constant on the left takes its type
from context first, defaulting to `int`. An untyped expression used as the count
also defaults to `int` and must be representable in that type:

```fern
var n = 40; // a variable reference is not a constant expression
var e = 1 << n; // int(1) shifted by n; 0 if n >= width
var f: u64 = 1 << n; // u64(1) shifted by n
```

A typed left operand retains its type regardless of the destination.

### Comparison

The comparisons `== != < <= > >=` require identical operand types and yield
`bool`, the type specified under [The bool type](#the-bool-type). Untyped
constants adopt the type of the other operand when their values fit. Comparison
precedence is given under Precedence, associativity, and evaluation order.

A well-typed comparison remains valid when the operand types make its result
always true or always false. For unsigned `x`, `x < 0` and `0 > x` yield `false`,
and `x >= 0` yields `true`. For `x: u8`, `x <= 255` yields `true`. A constant
that does not fit the other operand's type still makes the comparison invalid;
for example, `x < -1` is invalid when `x` is unsigned.

A comparison is a constant expression when both of its operands are constant
expressions, and its result is then an untyped boolean constant. A comparison
with an operand that is not a constant expression is evaluated at runtime, even
when its result is always true or always false.

### Assignment

`x = e;` stores the value of `e` in the mutable binding `x`. It requires
`e` to have the type of `x`, or to be an untyped constant that fits. Assignment
is a statement and ends with a semicolon.

Compound assignments `+= -= *= /= %= &= |= ^= <<= >>=` and their wrapping forms
`+%= -%= *%=` are equivalent to the corresponding binary operation
followed by assignment, and trap under the same conditions. The diagnostic for a
trapping compound assignment identifies the compound operator.

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
| Untyped constant shift result does not fit its target type | Compile error |
| Wrapping operators | Defined, no trap |
| Truncating conversion | Defined, no trap |
| Bitwise operators | Defined, no trap |

## Boolean Semantics

### The bool type

`bool` is a distinct type whose values are `true` and `false`. It is not an
integer type, and there is no conversion between `bool` and any integer type.
There is no truthiness: an integer cannot stand in for a condition, and a `bool`
cannot stand in for an integer.

`true` and `false` are untyped boolean constants. An untyped boolean constant
acquires type `bool` from the context in which it is used, and `bool` is its
default type. A `var` or `const` binding may have type `bool`.
[Comparison](#comparison) specifies the operators that produce `bool` from
integer operands.

```fern
var ready = true; // ready: bool
const done: bool = false;
```

### Logical operators

| Operator | Meaning |
|----------|---------|
| `&&` | Conjunction. Short-circuits. |
| `\|\|` | Disjunction. Short-circuits. |
| unary `!` | Negation |

Each operand of `&&` and `||`, and the operand of `!`, must have type `bool` or
be an untyped boolean constant. The result has type `bool`, or is an untyped
boolean constant when every operand is one. There is no logical exclusive or;
`a != b` compares two `bool` values. Their precedence is given under
Precedence, associativity, and evaluation order.

`&&` evaluates its left operand first. If that operand is `false`, the result is
`false` and the right operand is not evaluated. `||` evaluates its left operand
first; if that operand is `true`, the result is `true` and the right operand is
not evaluated. An operand that is not evaluated cannot trap.

```fern
fn main() -> void {
    var d = 0;
    var n = 10;
    if d != 0 && n / d > 1 { // n / d is not evaluated when d is 0
        exit(1);
    }
    exit(0);
}
```

### Boolean constant expressions

`true` and `false` are constant expressions. A logical operation is a constant
expression when all of its operands are constant expressions, and its result is
then an untyped boolean constant. A reference to a `const` binding of type
`bool` initialized by a constant expression is a constant expression, under the
rules given in [Integer constant expressions](#untyped-constants).

Short-circuiting does not exempt an unevaluated constant operand from the
compile-time checks that apply to constant expressions. `false && 1 / 0 == 0` is
rejected.

## Statements and Execution

A function body and each nested brace-delimited block execute statements in
source order. Declarations, assignments, compound assignments, `break`,
`continue`, and `exit` statements end with `;`. A nested block, an `if`
statement, and a `for` statement need no trailing semicolon. Declarations and
references must be valid even after a statement that terminates execution.

### Conditionals

```text
if c { ... }
if c { ... } else { ... }
if c { ... } else if c2 { ... } else { ... }
```

The condition must have type `bool` or be an untyped boolean constant. It is not
parenthesized. Each body is a brace-delimited block and introduces a scope; the
braces are required, and a single statement cannot replace the block.

`else` is followed either by a block or by another `if` statement, which forms a
chain. The conditions of a chain are tested in source order, and the first
branch whose condition is `true` executes. If no condition is `true`, the
trailing `else` block executes when one is present. At most one branch of a
chain executes.

```fern
fn main() -> void {
    var count = 3;
    if count > 3 {
        exit(2);
    } else if count == 3 {
        exit(1);
    } else {
        exit(0);
    }
}
```

### Loops

```text
for { ... }
for c { ... }
for init; c; post { ... }
```

`for` is the only loop keyword. The body is a brace-delimited block, is
required, and introduces a scope entered afresh on each iteration.

`for { ... }` repeats its body indefinitely. Execution leaves it only through
`break`, `exit`, or a trap.

`for c { ... }` evaluates `c` before each iteration and executes the body while
`c` is `true`. The condition follows the same typing rules as an `if` condition.

`for init; c; post { ... }` executes `init` once, then repeats: evaluate `c`,
execute the body if `c` is `true`, then execute `post`. `init` is a declaration
or an assignment; `post` is an assignment or a compound assignment. All three
clauses are required, and `;` separates them without following `post`. To omit a
clause, write one of the other two forms instead.

A binding declared in `init` is scoped to the whole `for` statement, including
the condition, `post`, and the body, and is not visible after the loop. The body
block may shadow it. One such binding exists for the whole loop, not one per
iteration.

```fern
fn main() -> void {
    var total = 0;
    for var i = 1; i <= 10; i = i + 1 {
        total = total + i;
    }
    exit(total); // reports 55
}
```

Iteration over arrays, slices, and strings is not yet specified.

### Loop control and labels

```text
break;
continue;
break :name;
continue :name;
```

`break` ends its loop, and execution continues after that loop. `continue` ends
the current iteration; in `for init; c; post { ... }` it executes `post` and
then tests the condition again. Neither is permitted outside a loop.

A loop may carry a label, written after `for` as `:name`, ahead of the loop's
condition or clauses:

```text
for :name { ... }
for :name c { ... }
for :name init; c; post { ... }
```

Unlabeled `break` and `continue` act on the innermost enclosing loop.
`break :name` and `continue :name` act on the loop labeled `name` instead. That
label must belong to a loop enclosing the statement in the same function; any
other name is invalid. Labels occupy a namespace separate from bindings, so a
label and a binding may share a name. A label may not repeat the name of a label
on a loop that encloses it, while two loops that do not enclose each other may
use the same label name. A label that nothing refers to is permitted.

```fern
fn main() -> void {
    var found = 0;
    for :rows var r = 0; r < 3; r = r + 1 {
        for var c = 0; c < 3; c = c + 1 {
            if r * 3 + c == 4 {
                found = 1;
                break :rows;
            }
        }
    }
    exit(found); // reports 1
}
```

### Process exit

```text
exit(e);
```

`exit` is a statement and a reserved word. It uses call syntax without being a
call, as an integer conversion does. Its argument has type `int`: an untyped
constant argument must fit `int`, and a differently typed integer requires an
explicit checked or truncating conversion. A trailing comma after the argument
is permitted, as in `exit(42,);`. The statement terminates the program and does
not return. The reported exit status is the argument modulo 256, in the range
zero through 255, on every host. `exit(256);` reports zero and `exit(-1);`
reports 255.

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
`example::println`. Module and function names in this section,
`example::println` included, are illustrative; the specification does not define
a standard library.

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

Module resolution is path-based. An implementation resolves imports against an
ordered list of module search roots. It must document how callers supply those
roots and what defaults apply. A command-line implementation may accept roots
through flags or `FERNPATH`; a build tool may construct them from project
metadata. The working directory is not an implicit dependency source when an
explicit root list is supplied.

Fern source imports do not name dependency versions. Build tools may use
manifests, lockfiles, registries, vendored source, or other metadata to select
versions and construct module search roots. Those facilities are outside the
Fern source language and do not change module identity within a chosen root
list.

### Open module questions

The specification does not yet settle public-declaration syntax, import name
conflicts, dependency cycles, missing-module diagnostics, or whether setting
`FERNPATH` replaces or extends the default search list.
