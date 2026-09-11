# Fern Language Specification

This specification defines Fern's language behavior. It is authoritative
wherever it is explicit. An invalid program must be rejected during
compilation, except where this specification states that a rule is not checked;
[Pointer validity](#pointer-validity) is the only such rule.
A trap halts execution with a diagnostic on standard error identifying the
failing operation and its source location. It terminates the process abnormally
through the platform's abort mechanism. The resulting termination status is
platform-defined and is never a status that `exit` can report.

Implementations may impose documented limits on source nesting and compile-time
resource use. Source exceeding such a limit must be rejected with a compilation
diagnostic, rather than crashing the compiler or silently changing a value.

The [TODO](../eng/todo.md) records implementation status.

Fern is a low-level programming language in the spirit of C. Its goals are
clarity, defined behavior, and safe default semantics.

## Lexical Structure

### Source text

Source text is encoded in UTF-8. Source text that is not valid UTF-8 is invalid.
Unicode characters are permitted where the grammar below allows them.

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

`fn`, `void`, `var`, `const`, `type`, `struct`, `pub`, `use`, `if`, `else`,
`for`, `in`, `break`, `continue`, `return`, `exit`, `len`, `null`, `true`, and
`false` are reserved.
The ten integer type names listed under Integer types, the two floating-point
type names listed under Floating-point types, and `bool`, `rune`, and `string`
are also reserved. Reserved words cannot be identifiers.

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

### Floating-point literals

A floating-point literal is a decimal number with either a decimal point or a
decimal exponent. Its integer part or fractional part may be omitted when a
decimal point is present. An exponent begins with `e` or `E`, has an optional
`+` or `-`, and is followed by one or more decimal digits. Digit separators,
hexadecimal floating-point literals, and type suffixes are not allowed.

```fern
1.0
.5
2.
1e3
6.02e-23
```

`1` is an integer literal, while `1.` and `1e0` are floating-point literals.
Floating-point literals are untyped. Floating-point constant expressions
define how they acquire an `f32` or `f64` type.

### Escape sequences

Interpreted string literals and character literals recognize these named
escape sequences:

| Escape | Value |
|--------|-------|
| `\0` | U+0000 null |
| `\a` | U+0007 alert |
| `\b` | U+0008 backspace |
| `\f` | U+000C form feed |
| `\n` | U+000A line feed |
| `\r` | U+000D carriage return |
| `\t` | U+0009 horizontal tab |
| `\v` | U+000B vertical tab |
| `\\` | U+005C backslash |
| `\"` | U+0022 double quote |
| `\'` | U+0027 single quote |

`\u{h}` denotes the Unicode scalar value written by `h`. The braces contain
one to six hexadecimal digits, using `0`-`9`, `a`-`f`, or `A`-`F`, with no
digit separators. The value must be no greater than U+10FFFF and must not be a
surrogate in U+D800-U+DFFF. In a string literal, the scalar is encoded as
UTF-8. In a character literal, it is the literal's value.

There are no octal or raw-byte escapes. Any unrecognized or incomplete escape
sequence is invalid.

### Character literals

A character literal is one Unicode scalar value between single quotes. Its
contents are either one source character or one escape sequence. A direct
single quote or backslash must be escaped. A line feed or carriage return is
not permitted within a character literal.

```fern
'A'
'é'
'\n'
'\u{1F642}'
```

A character literal is an untyped rune constant.
[Rune constants](#rune-constants) specifies how it acquires a type. A literal
containing no scalar or more than one scalar is invalid; Unicode normalization
is not performed.

### String literals

An interpreted string literal is a sequence of Unicode scalar values and
escape sequences between double quotes. A direct double quote or backslash
must be escaped. A line feed or carriage return is not permitted within an
interpreted string literal. Each scalar is encoded as UTF-8, so every
interpreted string literal contains valid UTF-8. Literal text is not Unicode
normalized.

A raw string literal is a sequence of source characters between backticks.
Every character except a backtick is permitted, including backslashes, line
feeds, and carriage returns. Its contents are preserved exactly as their UTF-8
bytes; escape sequences have no special meaning.

```fern
"hello, 世界\n"
`C:\new\file`
`first line
second line`
```

A string literal is an untyped string constant.
[String constants](#string-constants) specifies how it acquires a type.
Adjacent string literals do not concatenate; without an operator or separator
between them, the source is invalid.

## Declarations

### Functions

Every function is a module-level declaration with an explicit result type:

```text
fn name(first: T, second: U) -> R {
    ...
}
```

Each parameter has a name and an explicit type. Parameters are fixed and
positional: a call supplies exactly one argument for each parameter in source
order. Parameter names must be unique within a function. The parameter list may
be empty, and a nonempty list may have a trailing comma. The result type is
either `void` or one value type. `void` is not a value type and is not permitted
as a parameter type.

Parameters are immutable bindings whose scope is the function body. A local
binding may shadow a parameter under the ordinary shadowing rules. Each
argument is copied into its parameter before the body begins executing.

Function names participate in the namespace described under
[Module-level declarations](#module-level-declarations). A function name is
visible throughout its module regardless of declaration order. Functions are
not overloaded. A function may call itself or participate in mutual recursion.

An executable build designates one module as its root module. The program entry
point is the root module's sole top-level function named `main`, with no
parameters and the explicit return annotation `void`:

```fern
fn main() -> void {
}
```

The body is a brace-delimited sequence of statements and may be empty. No
semicolon follows the closing brace. A program must have exactly one entry
point. A function named `main` in a dependency module is an ordinary function
and does not affect entry-point selection. Whether `main` is public does not
affect its selection. Reaching the end of `main` terminates the program with
exit status zero. Explicit termination is defined under Process exit.

### Calls

```text
name(first, second)
```

A call target is a direct function name, resolved using the ordinary lexical
name rules. An imported function may also be called through its qualified name.
The target must resolve to a function; function names are not values. A local
binding may shadow a function name, in which case that name is not callable
within the binding's scope.

A call must supply the same number of arguments as the function has parameters.
An empty argument list is permitted, and a nonempty list may have a trailing
comma. Arguments are evaluated completely from left to right, then copied into
their corresponding parameters. Each argument must be valid as the initializer
of an annotated binding whose type is the parameter type.

A call to a value-returning function is an expression of the function's result
type. A call to a `void` function produces no value and is permitted only as a
call statement. Any call may be used as a call statement ending in `;`; a value
returned by such a statement is discarded. A call is never a constant
expression, even when every argument is constant.

### Variable declarations

```text
var name = e;
const name = e;
var name: T [= e];
const name: T [= e];
```

A declaration introduces a binding. When an initializer is present, it is
initialized to the value of `e`. A type annotation determines the binding's
type. A typed initializer must have that same type, while an untyped constant
must be representable by the annotated type;
[Pointer conversion and comparison](#pointer-conversion-and-comparison) gives
the one exception. Other numeric conversions must be explicit. Without an
annotation, the binding takes the initializer's type. Untyped integer,
floating-point, boolean, rune, and string constants default to `int`, `f64`,
`bool`, `rune`, and `string`, respectively. A declaration without an
initializer must have a type annotation and receives that type's zero value.
Zero values are `0` for integer and floating-point types, `false` for `bool`,
U+0000 for `rune`, the empty string for `string`, `null` for pointer types, the
empty slice for slice types, and recursive zero values for arrays and structs.
A binding reference has the binding's type.
Initialization copies the value; later assignment to the source binding does
not change the copy.

### Module-level declarations

A source file contains `use` declarations followed by type, function, and
module-level `var` and `const` declarations, in any order. Statements appear
only in function bodies.

A module-level declaration uses the declaration syntax above. A present
initializer must be a constant expression. A declaration without an initializer
receives its type's zero value. Module-level bindings are initialized before
`main` runs.

A module-level binding is visible throughout its module, including in other
files and in declarations that appear before it. No two module-level
declarations, including types and functions, in the same module may use the
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
scopes unless those bindings are shadowed. A module's module-level bindings
enclose every function body in the module, and a local binding may shadow one
of them.

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
not require compile-time evaluation. Integer and floating-point constant
expressions define when a binding can be used in a constant expression.

```fern
fn main() -> void {
    var value = 7;
    const saved = value;
    value = 42;
    exit(saved); // reports 7
}
```

### Type declarations

```text
type Name T;
type Name struct {
    field: T,
}
```

A type declaration is module-level only. `type Name T;` creates a new nominal
type whose underlying type is `T`. The new type is distinct from `T` and every
other type, so assignments and arguments do not implicitly cross the boundary.
It retains the underlying type's representation, range, zero value, operators,
and other value semantics. Explicit conversions are required to move between
the named type and its underlying type.

The struct form creates a named struct type. A struct must contain at least one
field. Fields are declared as `name: T`, separated by commas; a trailing comma
is permitted. Field names must be unique. Struct fields have no separate
visibility modifier: fields of a public struct are accessible wherever the
struct type is accessible.

A struct declaration is invalid when following struct fields and array element
types can reach that same struct type, directly or indirectly. Every struct
therefore has a finite value size. A [pointer](#pointer-types) or
[slice](#slice-types) field breaks such a cycle, because neither one's size
depends on the type it refers to.

Type declarations whose right-hand side is a struct end at the closing `}` and
do not have a semicolon. Other type declarations end with `;`. Anonymous struct
types are not part of the language.

Structs are value types. Initialization, assignment, argument passing, and
return copy every field recursively. A field is selected with `value.field`.
Selecting a field of a `var` struct produces an assignable target; selecting a
field of a `const` struct does not.

```fern
type Point struct {
    x: int,
    y: int,
}

fn main() -> void {
    var p = Point { x = 3, y = 4 };
    p.x = 5;
    const q = p;
    exit(q.x); // reports 5
}
```

### Struct literals

```text
Name { field = e, ... }
Name { ... }
```

A struct literal names each initialized field with `=`. Field initializers are
evaluated in source order. Each field may appear at most once. Without `...`,
every field must be listed exactly once. With `...`, omitted fields receive
their recursive zero values; `...` may appear alone. A field initializer must
have the field's type, or be an untyped constant representable by it.

Struct literals whose fields are all constant expressions are constant
expressions.

The equality operators `==` and `!=` compare structs structurally, recursively
comparing corresponding fields. A struct is comparable when every field is
comparable. These operators are not defined for structs with non-comparable
fields. Ordering operators are not defined on structs.

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

### Numeric conversions

There are no implicit conversions between concrete numeric types. Every integer
type, `f32`, and `f64` is distinct. Widening, narrowing, changing signedness,
and crossing between integer and floating-point types all require an explicit
conversion. This includes conversions between `int` and `i64`, `f32` and `f64`,
or `i32` and `f32`.

#### Checked conversion

`T(x)` converts a numeric value `x` to numeric type `T`. The conversion must
preserve the value exactly. If it would discard a fractional part, precision, or
range, the program traps at runtime. A conversion between floating-point types
preserves an infinity, NaN, or signed zero when the destination has that value;
no conversion between a floating-point type and an integer type accepts an
infinity or NaN.

```fern
var a: i64 = 300;
var b = u8(a); // trap: 300 does not fit u8
var c = u16(a); // 300
var d = i8(-1); // -1
var e = u8(i8(-1)); // compile error: constant conversion would trap
var f = int(a); // 300 on every platform

var exact: f32 = 0.5;
var rounded: f32 = 0.1; // literal rounds once to the nearest f32 value
var not_exact: f64 = 0.1;
var narrow = f32(not_exact); // trap: the f64 value is not exactly an f32
var whole = i32(f64(2.0)); // 2
var fraction = i32(f64(1.5)); // trap
```

Converting an untyped constant follows the rules of Typing by context. An
untyped integer must fit the destination type. An untyped floating-point
constant rounds once to the nearest value of the destination floating-point
type, using round-to-nearest, ties-to-even, and that value must be finite.
Conversions of a constant expression that would trap are compile-time errors.

`rune` is not a numeric type. [Rune conversions](#rune-conversions) specifies
its explicit conversions to and from integer types.

#### Truncating conversion

`T.truncate(x)` is defined only when `T` is an integer type and `x` has an
integer type or is an untyped integer constant. It converts `x` to type `T` by
reinterpreting its two's complement bit pattern at the width of `T`: high bits
are discarded on narrowing, and the value is sign-extended (from a signed
source) or zero-extended (from an unsigned source) on widening. This conversion
never traps. An untyped constant is reduced modulo 2 to the power of the
destination width, then interpreted using the destination signedness; it need
not first fit `int` or the destination type.

```fern
var a: i64 = 300;
var b = u8.truncate(a); // 44
var c = u8.truncate(-1); // 255
var d = i8.truncate(200); // -56
```

### Integer arithmetic

#### Precedence, associativity, and evaluation order

Field selection, indexing, calls, conversions, and `len` bind more tightly than
any operator.
Among the operators, unary operators have the highest precedence. Binary
operators have five precedence levels, from highest to lowest:

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
of the next operand begins. `grid[r][c]` indexes the row first, and `-a[0]`
negates the element. `&&` and `||` do not always evaluate their right
operand; see [Logical operators](#logical-operators).

#### Operand types

Binary arithmetic operators require both operands to have the identical type.
There is no promotion: `u8 + u8` is `u8` and `f32 + f32` is `f32`. Untyped
constants adopt the type of the other operand. An untyped integer and an
untyped floating-point constant combine as an untyped floating-point constant.
It is a compile-time error to combine two differently typed concrete numeric
values.

```fern
var a: u8 = 10;
var b: u16 = 20;
var c = a + b; // error: u8 and u16
var d = u16(a) + b; // ok
var e = a + 5; // ok, 5 becomes u8
var n: int = 3;
var m: i64 = 4;
var p = n + m; // error: int and i64, on every platform
var q = f32(1.0) + f64(2.0); // error: f32 and f64
var r = 1 + 0.5; // untyped floating-point constant
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

The [TODO](../eng/todo.md) records implementation status. [Arrays](#arrays),
[Slices](#slices), and [Strings](#strings) specify those types, their values,
and their operations.
[Implicit dereference](#implicit-dereference) specifies indexing, slicing, and
`len` through a pointer.

The length of any array, slice, or string has type `int`. Indices and slice
bounds have type `int`. An index must have type `int` or be an untyped constant
representable by `int`; an index of any other integer type requires an explicit
conversion.

An index is valid if `0 <= i < len`. Accessing an out-of-range index traps.
Because indices are signed, an expression such as `i - 1` when `i == 0` produces
`-1` and fails the bounds check rather than wrapping to a huge value.

The maximum length of an object is `2ʷ⁻¹ − 1` elements, where `w` is the pointer
width. Allocation sizes use `uint`; converting between lengths and allocation
sizes follows Numeric conversions.

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
constants adopt the type of the other operand when their values fit. An untyped
integer and an untyped floating-point constant compare as untyped
floating-point constants. Comparison precedence is given under Precedence,
associativity, and evaluation order.

A well-typed comparison remains valid when the operand types make its result
always true or always false. For unsigned `x`, `x < 0` and `0 > x` yield `false`,
and `x >= 0` yields `true`. For `x: u8`, `x <= 255` yields `true`. A constant
that does not fit the other operand's type still makes the comparison invalid;
for example, `x < -1` is invalid when `x` is unsigned.

A comparison is a constant expression when both of its operands are constant
expressions, and its result is then an untyped boolean constant. A comparison
with an operand that is not a constant expression is evaluated at runtime, even
when its result is always true or always false.

`==` and `!=` further require comparable operand types. Comparability follows
the types whose size is part of their type; each type's own section gives its
rule.

[Array comparison](#array-comparison) specifies `==` and `!=` on arrays,
[Slice comparison](#slice-comparison) specifies that slices are not comparable,
[String comparison](#string-comparison) specifies bytewise string ordering,
[Rune comparison](#rune-comparison) specifies code-point rune ordering,
and [Pointer conversion and comparison](#pointer-conversion-and-comparison)
specifies them on pointers.

### Assignment

`x = e;` stores the value of `e` in the mutable location `x`. It requires `e` to
have the type of `x`, to be an untyped constant that fits, or to be a pointer
converted under
[Pointer conversion and comparison](#pointer-conversion-and-comparison).
Assignment is a statement and ends with a semicolon.

A *location* is a binding, a field of a location, an element of an array
location, an element of a slice, or a dereference `*p`. A location is *mutable*
when it is a `var` binding, a field or array element of a mutable location, an
element of a slice of type `[]T`, or `*p` where `p` has type `*T`. A `const`
binding, a parameter, `*p` where `p` has type `*const T`, and an element of a
slice of type `[]const T` are immutable, and so is every field and array element
of an immutable location. A slice element's mutability comes from the slice's
type alone. An expression that is not a location, such as a call result or an
arithmetic result, is neither assignable nor addressable.

An assignment target is a mutable location. `a[i] = e;` stores into the element
of `a` at index `i`. `p.x = e;` stores into field `x` of `p`. `*p = e;` stores
into the location `p` refers to. Whether the binding `p` is `var` or `const`
does not affect `*p = e;`, and whether a slice binding `s` is `var` or `const`
does not affect `s[i] = e;`: `const` prevents reassigning the binding, not
writing through it. Every index, slice-bound, field-selection, and dereference
expression in the target is evaluated left to right, then `e`: in
`grid[r][c] = e;` the order is `r`, `c`, `e`. A compound assignment to an
element, field, or dereference evaluates the target once.

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
| Constant index out of range | Compile error |
| Constant slice bound out of range for an array or constant string | Compile error |
| Integer constant expression would trap | Compile error |
| Floating-point constant expression is non-finite | Compile error |
| Constant integer-to-rune conversion receives a non-scalar value | Compile error |
| `+ - *` overflow | Trap |
| Integer `/` or `%` by zero | Trap |
| `MIN / -1`, `MIN % -1`, `-MIN` | Trap |
| Checked conversion would discard information | Trap |
| Runtime integer-to-rune conversion receives a non-scalar value | Trap |
| Negative shift count | Trap |
| Out-of-range index | Trap |
| Out-of-range slice bounds | Trap |
| Dereference of `null` | Trap |
| Runtime shift count ≥ width | Defined, no trap |
| Untyped constant shift result does not fit its target type | Compile error |
| Wrapping operators | Defined, no trap |
| Truncating conversion | Defined, no trap |
| Bitwise operators | Defined, no trap |

## Floating-point Semantics

Floating-point values use the IEEE 754 binary32 and binary64 interchange
formats. Every floating-point operation rounds its result to its operand type
with round-to-nearest, ties-to-even. Floating-point arithmetic does not trap:
it may produce infinities, NaNs, signed zeroes, and subnormal values.

### Floating-point types

| Type | Format | Width |
|------|--------|-------|
| `f32` | IEEE 754 binary32 | 32 bits |
| `f64` | IEEE 754 binary64 | 64 bits |

`f32` and `f64` are distinct value types. Their zero value is positive zero.
There is no platform-dependent floating-point type.

### Floating-point constant expressions

An untyped floating-point constant is an exact, finite mathematical value. It
is formed by a floating-point literal; by an arithmetic or unary operation on
untyped floating-point constants; by an arithmetic operation that combines an
untyped integer with an untyped floating-point constant; or by parentheses
around one. Untyped floating-point constant expressions are evaluated at
compile time with exact arithmetic.

An untyped floating-point constant acquires an `f32` or `f64` type from context.
It rounds once to the nearest value in that format, using round-to-nearest,
ties-to-even, and the rounded result must be finite. Without context, it
defaults to `f64`.

A floating-point literal, a conversion of a constant expression, parentheses
around a constant expression, and an arithmetic or unary operation whose
operands are all constant expressions are floating-point constant expressions.
All of them must have finite results. Division by a constant zero, an invalid
operation, or an overflow in a floating-point constant expression is a
compile-time error. A reference to a `const` binding initialized by a
floating-point constant expression is a constant expression.

```fern
var half: f32 = .5;
var defaulted = 1e0; // f64
const ratio = 3.0 / 2.0; // untyped floating-point constant 1.5
const bad: f64 = 1.0 / 0.0; // error
```

### Floating-point operators

`+`, `-`, `*`, `/`, and unary `-` are defined on `f32` and `f64`. Their operands
follow [Operand types](#operand-types). The result has the operand type. `%`,
wrapping operators, bitwise operators, and shifts are not defined on
floating-point values.

At runtime, floating-point operations follow IEEE 754 results. For example,
nonzero division by signed zero produces a signed infinity, zero divided by zero
produces NaN, and overflow produces a signed infinity. A floating-point
operation that produces one of these values does not trap.

### Floating-point comparison

Floating-point comparisons follow the general [Comparison](#comparison) rules
and IEEE 754 comparison semantics. `NaN == NaN` is `false`, `NaN != NaN` is
`true`, and every ordering comparison with a NaN is `false`. Positive and
negative zero compare equal.

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
comparable operands.

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
rules given in [Integer constant expressions](#untyped-constants) and
[Floating-point constant expressions](#floating-point-constant-expressions).

Short-circuiting does not exempt an unevaluated constant operand from the
compile-time checks that apply to constant expressions. `false && 1 / 0 == 0` is
rejected.

## Rune Semantics

### The rune type

`rune` is a distinct value type whose values are the Unicode scalar values
U+0000-U+D7FF and U+E000-U+10FFFF. It has the same 32-bit representation and
alignment as `u32`, but it is not an integer type. Its zero value is U+0000.

### Rune constants

A character literal is an untyped rune constant. An untyped rune constant may
acquire type `rune` or a named type whose underlying type is `rune`; it cannot
acquire an integer type. Without a contextual type, it defaults to `rune`.
Character literals, rune conversions of integer constant expressions,
parentheses around rune constant expressions, and references to `const`
bindings initialized by rune constant expressions are rune constant
expressions.

```fern
type Letter rune;

var r = 'A';        // rune
var letter: Letter = 'A';
var byte: u8 = 'A'; // invalid: a rune constant does not acquire u8
```

### Rune conversions

`R(x)`, where `R` is `rune` or a named type whose underlying type is `rune`,
converts an integer value to that rune type. The value must be in
U+0000-U+D7FF or U+E000-U+10FFFF; otherwise the conversion traps. `T(r)`
converts a rune value or untyped rune constant `r` to integer type `T` and
traps when the scalar value is not representable by `T`. Either conversion is
a compile-time error instead when its operand is constant and the conversion
would trap.

```fern
var ascii = u8('A');
var face = rune(0x1F642);
var invalid = rune(0xD800); // compile error: surrogate
```

Truncating conversions to or from `rune` are not defined. There is no built-in
conversion from a rune to a string.

### Rune comparison

`==`, `!=`, `<`, `<=`, `>`, and `>=` compare two rune values by Unicode scalar
value and yield `bool`. An untyped rune constant adopts the other operand's
rune type. A comparison is a constant expression when both operands are
constant expressions, and its result is then an untyped boolean constant.

Arithmetic, wrapping, bitwise, shift, and logical operators are not defined on
runes.

## Arrays

### Array types

```text
[N]T
```

An array holds a fixed number of elements of a single element type, stored in
order. `N` is the length and `T` is the element type, so `[3]int` holds three
`int` values. The length is part of the type: `[3]int`, `[4]int`, and `[3]i64`
are three distinct types.

The length must be a constant expression of type `int`, or an untyped constant
representable by `int`, and must be at least 1. There are no zero-length arrays.

An array type is a value type. It may be the type of a binding, a parameter, or
a function result. Its element type may be any value type, including another
array type: `[2][3]int` holds two arrays of three `int` each. `void` is not a
value type and cannot be an element type.

The zero value of an array has every element set to that element type's zero
value.

An array is a value rather than a reference. Initialization, assignment,
argument passing, and return each copy every element. Modifying one array
afterward does not modify the other. An array does not implicitly become a
slice; [Slicing expressions](#slicing-expressions) specifies how to make one.

```fern
fn main() -> void {
    var a: [3]int = [1, 2, 3];
    var b = a;
    b[0] = 99;
    exit(a[0]); // reports 1
}
```

### Array literals

```text
[e1, e2, e3]
[e1, e2, last...]
```

An array literal is a bracketed list of elements with an optional trailing
comma. It carries no type prefix and takes its type from context, as an untyped
constant does. It is valid wherever an annotated binding, a call argument, or a
`return` expression has an array type. When no context supplies a type, the
element type and the length come from the elements themselves.

Each element must have the array's element type or be an untyped constant
representable by it. In a literal with no context, every element must have one
common type, and a literal of only untyped constants uses their default type.

The number of elements must equal the length. A literal with too few or too many
elements is a compile-time error.

```fern
var primes: [3]int = [2, 3, 5];
var inferred = [2, 3, 5]; // [3]int
var bytes: [2]u8 = [0, 255];
var grid: [2][3]int = [[1, 2, 3], [4, 5, 6]];
var short: [3]int = [1, 2]; // error: two elements for [3]int
```

The type `[_]T` takes its length from the initializer. It is permitted only in a
declaration whose initializer is an array literal, never as a parameter or
result type.

```fern
var sized: [_]int = [2, 3, 5]; // [3]int
```

An array literal is a constant expression when every element is a constant
expression, which is what allows a module-level declaration to hold an array.

#### Fill

`...` after the last element repeats that element across the array's remaining
elements. The length must come from context, so `...` is a compile-time error in
a literal that would take its length from its own elements. At least one element
must precede `...`, and the elements before it must not exceed the length; when
they already fill the array, the fill repeats nothing.

```fern
var counts: [1024]i32 = [0...];
var seeded: [8]int = [1, 2, 3, 0...]; // 1, 2, 3, then five zeros
var board: [2][3]u8 = [[1...]...];
var bad = [0...]; // error: no length
var worse: [_]int = [0...]; // error: no length
```

### Indexing

`a[i]` is the element of the array `a` at index `i` and is an expression of the
element type. Indexing requires an operand of array type, or a pointer to one
under [Implicit dereference](#implicit-dereference).
[Indexing and lengths](#indexing-and-lengths) gives the type of `i` and
the range of valid indices; an out-of-range index traps. Because an array's
length is part of its type, an out-of-range constant index is rejected at
compile time instead.

`a[i]` is never a constant expression, even when `a` and `i` are both constant.

An element of a `var` array may be assigned; [Assignment](#assignment) gives the
statement form and its evaluation order. An element of a `const` array cannot be
assigned.

```fern
fn main() -> void {
    const a: [3]int = [10, 20, 30];
    var i = 2;
    exit(a[i]); // reports 30
}
```

```fern
fn main() -> void {
    const a: [3]int = [10, 20, 30];
    exit(a[3]); // error: 3 is out of range for [3]int
}
```

### Length

```text
len(a)
```

`len` is a reserved word and uses call syntax without being a call, as an
integer conversion does. Its operand is an expression of array, slice, or
string type, or a pointer to one under
[Implicit dereference](#implicit-dereference). A trailing comma after that
operand is permitted, and the result has type `int`. The rest of this section
specifies the array case; [Slice length](#slice-length) and
[String length](#string-length) specify the other cases.

`len(a)` yields the length recorded in the operand's type. The operand is still
evaluated, so a trap inside it, such as an out-of-range index, still occurs.
Because the length comes from the type rather than the value, `len(a)` is a
constant expression whenever its operand contains no call, including when the
operand refers to a `var` binding.

```fern
fn main() -> void {
    const a: [3]int = [10, 20, 30];
    exit(len(a)); // reports 3
}
```

### Array comparison

`==` and `!=` compare two arrays of identical type. `==` yields `true` when
every pair of corresponding elements is equal, and `!=` is its negation. Arrays
of different lengths or different element types are different types and cannot
be compared. `<`, `<=`, `>`, and `>=` are not defined on arrays.

An array is comparable when its element type is comparable. An array whose
element type is a slice is therefore not comparable.

An implementation may compare the elements in any order. Element comparison
cannot trap, so the order is not observable.

A comparison of two arrays is a constant expression when both operands are
constant expressions, and its result is then an untyped boolean constant.

Arithmetic, bitwise, and logical operators are not defined on arrays. There is
no elementwise arithmetic.

```fern
fn main() -> void {
    var a: [3]int = [1, 2, 3];
    const b: [3]int = [1, 2, 3];
    a[2] = 9;
    if a == b {
        exit(1);
    }
    exit(0); // reports 0
}
```

## Pointers

### Pointer types

```text
*T
*const T
```

A pointer refers to a location of its target type `T`, or is `null`. Writing
through a `*T` is permitted; writing through a `*const T` is not. The two are
distinct types, and the target type distinguishes them further: `*int`,
`*const int`, and `*i64` are three distinct types.

A pointer type is a value type. It may be the type of a binding, a parameter, a
function result, a struct field, or an array element. Its target type may be any
value type, including another pointer type. `void` is not a value type and
cannot be a target type.

The zero value of a pointer type is `null`.

A pointer is a value. Initialization, assignment, argument passing, and return
copy it, and the copy refers to the same location. Copying a pointer does not
copy the location it refers to.

Because a pointer's size does not depend on its target type, a struct may reach
itself through a pointer field:

```fern
type Node struct {
    value: int,
    next: *Node,
}
```

### The null pointer

`null` is a reserved word denoting a pointer that refers to no location. It is
an untyped constant and takes a pointer type from context under the rules of
Typing by context: an annotated binding, a parameter type, a result type, an
assignment target, or the other operand of a comparison. `null` with no such
context is invalid.

```fern
fn main() -> void {
    var p: *int = null;
    var q = null; // invalid: no context supplies a pointer type
}
```

### Address-of

`&x` yields a pointer to the location `x`. [Assignment](#assignment) defines
which expressions are locations and when a location is mutable. `&x` yields
`*T` when `x` is a mutable location of type `T`, and `*const T` when `x` is an
immutable one. Taking the address of an expression that is not a location is
invalid.

Evaluating `&x` evaluates every index, field-selection, and dereference
expression in `x`, from left to right, and traps on an out-of-range index or a
`null` dereference just as reading `x` would.

`&x` is never a constant expression. A module-level initializer therefore cannot
take an address, and `null` is the only pointer value one can hold.

```fern
fn main() -> void {
    var x = 7;
    const c = 9;
    var p = &x;       // *int
    var q = &c;       // *const int
    exit(*p + *q);    // reports 16
}
```

### Dereference

`*p` denotes the location that `p` refers to and has the pointer's target type.
Dereferencing `null` traps.

`*p` is itself a location, so it may be addressed, and it may be assigned when
`p` has type `*T`. [Assignment](#assignment) gives the statement form.

Unary `*` and unary `&` are unary operators and take the precedence given under
Precedence, associativity, and evaluation order.

```fern
fn main() -> void {
    var x = 7;
    const p = &x;
    *p = 42;
    exit(x); // reports 42
}
```

### Implicit dereference

Field selection, indexing, slicing, and `len` accept a pointer operand and
dereference it once: `p.x` means `(*p).x`, `p[i]` means `(*p)[i]`, `p[lo:hi]`
means `(*p)[lo:hi]`, and `len(p)` means `len(*p)`. Each traps when `p` is
`null`. Because a pointer operand can trap, `len(p)` is never a constant
expression.

When the corresponding non-pointer expression produces a location, its
mutability follows the pointer. For example, `p.x = e;` is valid when `p` has
type `*Point` and invalid when `p` has type `*const Point`.

One level is dereferenced, not a chain: for `pp` of type `**Point`, write
`(*pp).x`. Implicit dereference applies to these four forms and nowhere else;
iterating the value a pointer refers to is written `for v in *p`.

```fern
type Point struct {
    x: int,
    y: int,
}

fn shift(p: *Point) -> void {
    p.x = p.x + 1;
}

fn main() -> void {
    var q = Point { x = 1, y = 2 };
    shift(&q);
    exit(q.x); // reports 2
}
```

### Pointer conversion and comparison

A `*T` may be used wherever a `*const T` with the same target type is expected:
as the initializer of an annotated binding, a call argument, a `return`
expression, an assigned value, or the other operand of a comparison. The value
is unchanged; the conversion only withdraws the ability to write through the
pointer. There is no conversion in the other direction, and no conversion
between pointers with different target types.

`==` and `!=` compare two pointers whose target types are the same, whether or
not their constness matches, and compare any pointer against `null`. Two
pointers are equal when they refer to the same location, or when both are
`null`. `<`, `<=`, `>`, and `>=` are not defined on pointers, and neither are
the arithmetic, bitwise, and logical operators. There is no pointer arithmetic.
A comparison of two pointers is never a constant expression.

`uint(p)` converts a pointer to `uint`, yielding its address. It never traps,
and `null` converts to `0`. There is no conversion from an integer to a pointer.

```fern
fn read(p: *const int) -> int {
    return *p;
}

fn main() -> void {
    var x = 5;
    var p = &x;
    const q: *const int = p; // a *int is accepted here
    if p == q {
        exit(read(p)); // reports 5
    }
    exit(1);
}
```

### Pointer validity

A pointer is valid only while the location it refers to exists. A local
binding's location exists from its declaration until its enclosing block ends. A
module-level binding's location exists for the whole program run.

Using the value of a pointer to a location that no longer exists is invalid.
Fern does not detect it: neither the compiler nor the running program checks
pointer validity, and this is the only rule in the language that goes unchecked.
Keeping a pointer no longer than its referent is the programmer's obligation.

```fern
fn escape() -> *int {
    var x = 1;
    return &x; // x ceases to exist when the function returns
}
```

## Slices

### Slice types

```text
[]T
[]const T
```

A slice refers to a contiguous run of elements of a single element type `T`. It
holds the location of the first element and the number of elements, which is its
*length*. Writing through a `[]T` is permitted; writing through a `[]const T` is
not. The two are distinct types, and the element type distinguishes them
further: `[]int`, `[]const int`, and `[]i64` are three distinct types.

A slice carries no capacity and cannot grow. It is a view of storage that
something else owns, and no operation appends to a slice or reallocates the
storage it refers to.

A slice type is a value type. It may be the type of a binding, a parameter, a
function result, a struct field, an array element, or the element type of
another slice. Its element type may be any value type, including an array type
or another slice type. `void` is not a value type and cannot be an element type.

A slice is a value. Initialization, assignment, argument passing, and return
copy its location and its length, and the copy refers to the same elements.
Copying a slice does not copy the elements.

The zero value of a slice type is the empty slice: length `0`, referring to no
elements. `null` is a pointer constant and does not take a slice type, so a
slice is never `null`, and `len(s) == 0` is how a program tests for emptiness.
Every empty slice behaves alike: it yields `0` from `len`, traps on any index,
and executes a `for` body zero times.

There is no slice literal. A slice value comes from a slicing expression, from
the zero value, or from copying another slice.

Because a slice's size does not depend on its element type, a struct may reach
itself through a slice field, as it may through a pointer field.

A `[]T` may be used wherever a `[]const T` with the same element type is
expected: as the initializer of an annotated binding, a call argument, a
`return` expression, or an assigned value. The value is unchanged; the
conversion only withdraws the ability to write through the slice. There is no
conversion in the other direction, no conversion between slices with different
element types, no conversion between an array and a slice, and no conversion
between a slice and an integer.

A slice refers to storage it does not own, so
[Pointer validity](#pointer-validity) governs it unchanged: using a slice whose
elements no longer exist is invalid, and Fern does not detect it.

### Slicing expressions

```text
a[lo:hi]
a[lo:]
a[:hi]
a[:]
```

For an array or slice operand, a slicing expression yields a slice of the
elements of `a` from index `lo` up to but not including index `hi`. Its length
is `hi - lo`. An omitted `lo` means `0` and an omitted `hi` means `len(a)`, so
`a[:]` covers all of `a`. `lo` and `hi` have the type given under
[Indexing and lengths](#indexing-and-lengths). There is no third bound, because
a slice has no capacity to set. [String slicing](#string-slicing) specifies the
string case.

The operand may be an array that is a location, a slice, or a pointer to an
array under [Implicit dereference](#implicit-dereference). An array that is not
a location, such as a call result or an array literal, cannot be sliced: the
result would refer to storage that has already ceased to exist. A slice operand
carries no such restriction, because a slice already holds the location of its
elements, so the result of a call that returns a slice may be sliced.

The result's element type is the operand's element type. The result is `[]T`
when the operand is a mutable array location, a `*[N]T`, or a `[]T`, and
`[]const T` when the operand is an immutable array location, a `*const [N]T`, or
a `[]const T`. Slicing therefore grants no access the operand did not already
have.

The bounds must satisfy `0 <= lo <= hi <= len(a)`. `lo == hi` is permitted and
yields an empty slice. Bounds outside that range trap. When the operand's length
is part of its type — an array, or a pointer to one — and the offending bound
is a constant expression, it is rejected at compile time instead, as an
out-of-range constant index into an array is.

The operand is evaluated first, then `lo`, then `hi`. A slicing expression is
never a constant expression, because it takes the location of the operand's
elements. A module-level binding of slice type can therefore hold only the empty
slice, just as a module-level binding of pointer type can hold only `null`.

```fern
fn sum(xs: []const int) -> int {
    var total = 0;
    for v in xs {
        total = total + v;
    }
    return total;
}

fn main() -> void {
    const a: [5]int = [1, 2, 3, 4, 5];
    exit(sum(a[1:4])); // reports 9
}
```

```fern
fn main() -> void {
    var a: [4]int = [1, 2, 3, 4];
    var s = a[1:3]; // []int, length 2
    s[0] = 99;
    exit(a[1]); // reports 99
}
```

### Slice indexing

`s[i]` is the element of the slice `s` at index `i` and is an expression of the
element type. [Indexing and lengths](#indexing-and-lengths) gives the type of
`i` and the range of valid indices; an out-of-range index traps. A slice's
length is part of its value rather than its type, so a constant index is not
rejected at compile time, unlike an index into an array.

`s[i]` is a location. It is mutable when `s` has type `[]T` and immutable when
`s` has type `[]const T`; [Assignment](#assignment) gives the statement form and
its evaluation order. `&s[i]` yields `*T` or `*const T` accordingly.

`s[i]` is never a constant expression.

### Slice length

`len(s)` yields the length of the slice `s` and has type `int`.
[Length](#length) gives the form of `len` and its trailing-comma rule. Because a
slice's length is part of its value rather than its type, `len(s)` is never a
constant expression.

```fern
fn main() -> void {
    const a: [5]int = [1, 2, 3, 4, 5];
    exit(len(a[1:4])); // reports 3
}
```

### Slice comparison

A slice type is not comparable. `==` and `!=` are not defined on slices, and
neither are `<`, `<=`, `>`, `>=`, nor the arithmetic, bitwise, and logical
operators. A slice is never `null`, so it is not compared to `null` either;
`len(s) == 0` tests for emptiness.

A struct with a slice field and an array whose element type is a slice are also
not comparable, by the rules under [Struct literals](#struct-literals) and
[Array comparison](#array-comparison).

## Strings

### The string type

`string` is a distinct value type whose values are arbitrary, possibly empty
sequences of bytes. A string may contain NUL bytes or bytes that are not valid
UTF-8. Its length is its number of bytes. A string's bytes are immutable: no
operation can assign through a string value. NUL has no special meaning within
a string.

A string value holds its byte sequence and length. Initialization, assignment,
argument passing, and return copy the value without copying its bytes. The copy
refers to the same immutable bytes. A `var` string binding may be assigned a
different string value; this does not modify either byte sequence.

The zero value of `string` is the empty string. A string is never `null`, and
all empty strings behave alike: `len` yields `0`, every index traps, all empty
strings compare equal, and iteration executes zero times.

`string` may be the type of a binding, parameter, function result, struct
field, array element, or slice element. A named type may have `string` as its
underlying type and retains these value semantics and operations.

`[]const u8(s)` produces a slice of the bytes in string `s` without copying
them. The slice has length `len(s)` and refers to the same immutable storage.
There is no built-in conversion from a byte slice to a string or between a
string and a rune or rune slice.

### String constants

A string literal is an untyped string constant. It may acquire type `string` or
a named type whose underlying type is `string`; without a contextual type it
defaults to `string`. String literals, explicit conversions between `string`
and a named string type, parentheses around string constant expressions, and
references to `const` bindings initialized by string constant expressions are
string constant expressions.

`len(s)` is a constant expression when `s` is a string constant expression.
A string comparison is a constant expression when both operands are constant
expressions. String indexing and slicing are never constant expressions.

```fern
type Path string;

var text = "fern";       // string
var path: Path = `src/main.fern`;
const bytes = len("é"); // 2
```

### String indexing

`s[i]` is the byte at index `i` in string `s` and has type `u8`.
[Indexing and lengths](#indexing-and-lengths) gives the type and valid range of
`i`. An out-of-range index traps, except that an out-of-range constant index
into a constant string is rejected at compile time.

A string index is neither assignable nor addressable. Use `[]const u8(s)` when
an immutable byte location or slice is required. String indexing is never a
constant expression.

```fern
const s = "é";
const first = s[0]; // invalid: indexing is not a constant expression

fn main() -> void {
    exit(s[0]); // reports 195, the first UTF-8 byte
}
```

### String slicing

The slicing forms under [Slicing expressions](#slicing-expressions) also
accept a string operand. They select bytes from `lo` up to but not including
`hi`, with omitted bounds taking the same defaults, and produce a string that
refers to those bytes without copying them. The bounds must satisfy
`0 <= lo <= hi <= len(s)`. Invalid bounds trap, except that invalid constant
bounds on a constant string are rejected at compile time.

String slicing does not require UTF-8 boundaries. The result may therefore
contain invalid UTF-8 even when its operand is a valid UTF-8 string literal.
The operand is evaluated first, then `lo`, then `hi`. A string slicing
expression is never a constant expression.

```fern
const text = "é"; // C3 A9

fn main() -> void {
    const first = text[0:1];
    exit(first[0]); // reports 195; first is not valid UTF-8
}
```

### String length

`len(s)` yields the number of bytes in string `s` and has type `int`.
[Length](#length) gives the form of `len` and its trailing-comma rule. The
result is a constant expression exactly when `s` is a string constant
expression.

### String comparison

`==`, `!=`, `<`, `<=`, `>`, and `>=` compare two strings of identical type
lexicographically by their unsigned byte values. At the first unequal byte, the
string with the smaller byte compares less; if every byte of the shorter string
matches, the shorter string compares less. Two strings are equal when they
have identical lengths and bytes.

An untyped string constant adopts the other operand's string type. A comparison
is a constant expression when both operands are constant expressions, and its
result is then an untyped boolean constant.

Strings have no arithmetic, wrapping, bitwise, shift, or logical operators. In
particular, `+` does not concatenate strings. Constructing strings from runtime
data and concatenating strings are not language operations.

## Statements and Execution

A function body and each nested brace-delimited block execute statements in
source order. Declarations, assignments, compound assignments, calls, `break`,
`continue`, `return`, and `exit` statements end with `;`. A nested block, an
`if` statement, and a `for` statement need no trailing semicolon. Declarations
and references must be valid even after a statement that terminates execution.

### Conditionals

```text
if c { ... }
if c { ... } else { ... }
if c { ... } else if c2 { ... } else { ... }
```

The condition must have type `bool` or be an untyped boolean constant. It is not
parenthesized. Each body is a brace-delimited block and introduces a scope; the
braces are required, and a single statement cannot replace the block.

A named struct literal used directly in an `if` or `for` condition must be
parenthesized, either by enclosing the literal or by enclosing the whole
condition. This distinguishes the literal's opening brace from the brace that
begins the statement body.

```fern
if (Point { x = 1, y = 2 }) == expected { // valid
}
if Point { x = 1, y = 2 } == expected { // invalid
}
```

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
for v in a { ... }
for v, i in a { ... }
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

For an array or slice `a`, `for v in a { ... }` executes the body once for each
element in index order, with `v` bound to a copy of the element.
`for v, i in a { ... }` also binds `i`, of type `int`, to that element's index.
`in` is a reserved word, and the two names must differ. Iterating an empty slice
executes the body zero times. The string case is specified below.

`a` is evaluated once, before the first iteration, and the loop walks that
value, so the number of iterations is fixed before the first one. When `a` is an
array, the loop walks a copy, and assigning to an element of that array inside
the body does not change the remaining iterations. When `a` is a slice, the copy
refers to the same elements, so assigning through the slice inside the body does
change what the remaining iterations read.

`v` and `i` are immutable bindings whose scope is the loop body, bound afresh on
each iteration. The body may shadow them, and neither is visible after the loop.

```fern
fn main() -> void {
    const a: [4]int = [1, 2, 3, 4];
    var total = 0;
    for v in a {
        total = total + v;
    }
    exit(total); // reports 10
}
```

When `a` is a string, the loop decodes its bytes as UTF-8 from left to right.
Each iteration binds `v`, of type `rune`, to the next decoded scalar. The
optional `i`, of type `int`, is that scalar's starting byte index. Each byte
that cannot begin a valid encoding yields U+FFFD and advances the next
iteration by one byte. The string value is evaluated once before iteration;
its immutable byte sequence fixes every iteration value and index.

```fern
fn main() -> void {
    const text = "Aé";
    var last = 0;
    for r, i in text {
        last = i;
    }
    exit(last); // reports 1, the byte index of é
}
```

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
for :name v in a { ... }
for :name v, i in a { ... }
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

### Function return

```text
return;
return e;
```

`return` ends the current function call. A `void` function may use `return;` or
reach the end of its body, but it may not return an expression. A
value-returning function must use `return e;`. The expression is evaluated
before the call ends and must be valid as the initializer of an annotated
binding whose type is the declared result type.

The end of a value-returning function body must be unreachable under a
structural check. A `return` or `exit` makes its following path unreachable. An
`if` chain makes its following path unreachable only when it has a final `else`
and every branch does so. The path after `for { ... }` is unreachable when no
reachable `break` targets that loop. Other `for` forms may fall through.
Boolean constants and constant expressions do not remove paths for this check.
Statements on unreachable paths are still checked.

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

A module consists of all `.fern` source files directly within one directory.
Nested directories define separate modules. All source files in a module share
one module namespace and can access each other's declarations, including
private declarations. Each source file has its own imports.

A module's directory path relative to a module search root is its import path.
An import path contains one or more identifier components separated by `::`;
the components map to directories beneath the search root. The directory
defines the namespace, and source filenames do not introduce namespaces. For
example, `example/print.fern` belongs to module `example`. A function named
`println` declared in that file has the qualified name `example::println`.
Module and function names in this section, `example::println` included, are
illustrative; the specification does not define a standard library.

An executable build designates a directory as its root module. The root module
need not have a special directory name. Entry-point selection is defined under
[Functions](#functions).

### Public declarations

`pub` may precede a module-level type, function, `var`, or `const` declaration.
It makes that declaration accessible to other modules. A declaration without
`pub` is private to its module. `pub` is not permitted on a local declaration
or a `use` declaration.

```fern
pub type Point struct {
    x: int,
    y: int,
}

pub fn print() -> void {
}

pub var output_enabled = true;
const buffer_size = 4096;
```

Code in another module may read a public binding. It may assign to a public
`var`, through either a qualified or selective import, but it may not assign to
a public `const`. A private declaration cannot be accessed from another module.

### Use declarations

Imports are file-local `use` declarations terminated by `;`. Every `use` in a
file must appear before the file's first type, function, `var`, or `const`
declaration.

A whole-module import names a module by its full import path and introduces the
path's final component in that file. Its public declarations are accessed with
`::`. A selective import introduces the named public declarations for
unqualified use in that file. The brace form selects declarations only; a
nested module is imported with its full path.

```text
use fmt;             // introduce the module name fmt
use fs::{flag};      // introduce the public declaration flag
use network::http;   // introduce the module name http

fmt::println(...);
http::serve(...);
```

A name introduced by `use` must not already refer to a module-level declaration
or another imported name in that file. This rule also rejects repeating an
import that introduces the same name, even when both imports resolve to the
same declaration or module. A local binding in a function may shadow an
imported name under the ordinary shadowing rules.

Every name introduced by a `use` declaration must be referenced in that source
file. A module or declaration cannot be imported only for side effects. A `use`
declaration cannot rename a module or declaration.

A `use` declaration in one file does not introduce imported names in other
files, even when those files belong to the same module. Imported names are not
part of the importing module's public interface and are never re-exported.

### Module resolution and dependencies

Module resolution is path-based. An implementation resolves an import by
appending its path components to each module search root in order. The first
resulting directory containing at least one `.fern` source file is the imported
module. Later matching roots are ignored.

An implementation must document how callers supply the ordered roots and what
defaults apply. A command-line implementation may accept roots through flags or
`FERNPATH`; a build tool may construct them from project metadata. When
`FERNPATH` contains at least one non-empty entry, its entries replace the
command-line implementation's default roots rather than extending them. An
empty `FERNPATH` leaves the default roots in effect. The working directory is
not an implicit dependency source when an explicit root list is supplied.

If no root contains the requested module, compilation fails. The diagnostic
must point to and identify the unresolved import path and list every searched
root in search order.

Each `use` adds a dependency from the importing module to the imported module.
The dependency graph must be acyclic. A direct or indirect dependency cycle is
a compile-time error, and its diagnostic must show the cycle as a chain of
module paths. Each dependency module is included and its module-level bindings
are initialized exactly once. A dependency's bindings are initialized before
those of a module that depends on it, and all module-level bindings are
initialized before the root module's `main` runs.

Fern source imports do not name dependency versions. Build tools may use
manifests, lockfiles, registries, vendored source, or other metadata to select
versions and construct module search roots. Those facilities are outside the
Fern source language and do not change module identity within a chosen root
list.
