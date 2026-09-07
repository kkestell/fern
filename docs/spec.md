# Fern Language Specification

This specification defines the behavior every implementation of Fern must
provide. It is authoritative wherever it is explicit.

Fern is a low-level programming language in the spirit of C. Its goals are to
avoid C's legacy baggage, undefined behavior, and unsafe default semantics. A
program this specification calls invalid is ill-formed, and an implementation
must reject it. Where this specification leaves a choice to the implementation,
it says so explicitly.

## Open Questions

This section is not normative. Each item is a decision this specification has
not yet made. Items are numbered for reference and grouped by the section they
will belong to once decided. Where the current text already implies something,
the item says what.

### Lexical structure

1. **Keywords.** The Keywords section defines the words already reserved. What
   additional words belong in the complete set?

### Types

4. **Pointer lifetimes and mutability.** Which additional pointer conversions
   are allowed? What remaining lifetime and mutability rules apply to taking
   addresses and accessing pointees? After explicitly casting away pointee
   constness, may the pointer modify storage originally declared `const`?
5. **Size and index types.** What type do array indices, lengths, and the count
   argument of `alloc` have? What are the range and arithmetic rules of `size`?
6. **Integer overflow.** What happens when integer arithmetic overflows:
   wrapping, a runtime failure, or something else? Division by zero and shifts
   by at least the operand width need the same decision.
7. **Numeric conversions.** Integer conversions below settle same-signedness
   widening in initialization, assignment, function arguments, return values,
   and binary expressions. What other numeric conversions exist, and what are
   the syntax and value rules for explicit integer casts? What do conversions
   involving the null type produce? Does an unsupported `uintptr` promotion
   print a diagnostic and abort during compilation or execution? Which operators
   use the promotion rules?
8. **Strings.** What is the representation of `str`, and which standard-library
   operations provide content access, length, conversion, and validation?
9. **Slices.** A slice owns its memory, yet assigning one is shallow and creates
   an alias. Is a slice an owning buffer, a view into other storage, or both? Is
   there a slicing expression such as `xs[a..b]`? Can a slice grow?
10. **Arrays.** Does assigning an array copy its elements? Are array accesses
    bounds checked, and is a constant out-of-range index rejected at compile
    time? Is a zero-length array valid? Must the length `N` be a literal, or may
    it be any constant expression?
11. **Structs.** Must a struct literal initialize every field? Do fields have
    zero values? Can layout, padding, and alignment be controlled? Can a struct
    type be used without naming it? How are structs compared?
12. **Tuples.** Is the empty tuple `()` a type? Is a one-element tuple distinct
    from its element? Can a tuple be destructured into several bindings?
13. **Other types.** Enumerations, unions or tagged unions, optional values, and
    an error-handling mechanism are all absent. Which does the language provide?
    Beyond the compatibility rules under Function types, what are the rules for
    obtaining and calling function pointers?
14. **Type declarations.** Does `type name = ...` create a distinct type or an
    alias? Are two struct types with identical fields the same type? Can a type
    refer to itself? The example has no terminating semicolon; is one required?

### Flexible types and constants

15. **Flexible arithmetic.** How do two flexible operands combine, as in
    `1 + 2`, `1 + 2.0`, or `'a' + 1`? May a flexible integer initialize a
    floating-point binding? What precision and rounding rules apply to flexible
    floating-point values?
16. **Constant expressions.** Which `const` bindings may be used in constant
    expressions? What else may appear in a constant expression?

### Declarations

17. **Uninitialized bindings.** Is `var x: int;` valid? If so, does `x` hold a
    zero value, or must it be assigned before it is read?
18. **Top-level declarations.** Can bindings be declared outside functions? When
    are they initialized?
19. **Immutability and aliasing.** How does copying non-pointer values interact
    with const qualification? How does const qualification apply to slices and
    their elements? Can a `const`-bound allocation be freed?

### Expressions

21. **Operators.** Which arithmetic, comparison, logical, and bitwise operators
    exist, with what precedence and associativity? The pointer and null-test
    operations below settle only part of this question.
22. **Evaluation order.** In what order are operands, arguments, and the
    elements of a composite literal evaluated?
23. **Functions.** The Functions section settles parameterless declarations and
    the entry point. How are parameters declared, arguments passed, and values
    returned? What are the rules for calling user-defined functions? Does `exit`
    run deferred statements?
24. **Generics.** `[]T` and `[N]T` are parameterized by `T`. Can user code
    define generic types or functions?
25. **Compile-time evaluation.** Which expressions can be evaluated at compile
    time? How does compile-time evaluation relate to flexible types?

### Statements

26. **Control flow.** Null tests, boolean conditions, and matching nullable
    pointers are specified below. What are the full rules for conditionals,
    loops, matching other values, `yield`, `break`, `continue`, and `return`?
27. **Assignment forms.** Beyond the specified `+=` restrictions for pointers
    and `uintptr`, which compound assignments exist? Is assignment a statement
    only, or also an expression?
28. **`defer` details.** In what order do several deferred statements in one
    scope run? Are the operands of a deferred statement evaluated when it is
    deferred or when it runs? Which statements may be deferred? Does a deferred
    statement run on every way of leaving the scope?

### Memory management

29. **`alloc` forms.** Only `alloc([]T, n)` appears. Can a single struct or
    other non-slice value be allocated, and what does the result refer to? Is
    allocated memory zeroed? What alignment does it have?
30. **Allocation failure.** What happens when `alloc` cannot obtain memory?
31. **Use after free and double free.** Avoiding them is "the programmer's
    responsibility", which conflicts with the goal of no undefined behavior.
    What does the language guarantee when a dangling alias is used or an
    allocation is freed twice?
32. **`free` on other values.** What happens when `free` is applied to an array,
    a struct, a string literal, or a slice that did not come from `alloc`?
33. **Explicit copy.** Independent storage "requires an explicit copy
    operation". What is that operation, and how deep is the copy?
34. **Allocators.** Is there a single global allocator, or can allocation be
    directed to arenas or user-supplied allocators?

### Runtime checks

35. **Out-of-bounds behavior.** Out-of-bounds access "has defined behavior".
    What is it: termination with a diagnostic, a recoverable error, or something
    else? Can the check be disabled?
36. **Diagnostics.** Is every invalid program detected before it runs? What must
    an implementation report for an invalid program and for a runtime failure?
    What does a panic do, and can it be recovered from?
37. **Invalid pointer dereferences.** Does dereferencing an invalid or
    misaligned pointer panic, or can it have undefined behavior? A panic is
    preferred, but the guarantee is not yet settled.

### Program structure

38. **Module details.** How are declarations made public? How are import name
    conflicts and dependency cycles handled?
39. **C interoperability.** Can Fern call C and be called from C? What is the
    calling convention, and how are C types mapped?
40. **Standard library.** What does the language provide beyond `alloc`, `free`,
    and the explicit copy operation?
41. **Formal grammar.** Syntax is shown as forms rather than as a grammar. Once
    the lexical structure and expression syntax are settled, which grammar
    notation should this specification use?
42. **Module search.** Does setting `FERNPATH` replace the default search list
    or extend it? How are missing modules reported?

## Notation and Terminology

Syntax is shown as forms in `text` blocks. In a form, `T` stands for a type, `N`
for an array length, `name` for an identifier, `e`, `n`, and `i` for
expressions, and `s` for a statement. `...` marks repetition of the preceding
item. Every other character in a form is literal.

Examples are fragments of Fern programs. A comment on an example line states the
type or value the line produces, or marks the line as invalid.

A program this specification calls **invalid** is ill-formed. An implementation
must reject it.

A behavior this specification calls **implementation-defined** is chosen by the
implementation.

## Lexical Structure

### Source text

Source text is encoded in UTF-8. Source text that is not valid UTF-8 is invalid.
Unicode characters are permitted in comments, string literals, and rune
literals.

Outside comments and literals, ASCII space, horizontal tab, line feed, carriage
return, vertical tab, and form feed are whitespace. Whitespace and comments
separate tokens but do not otherwise affect syntax. Newlines do not insert
semicolons. Tokens use the longest matching spelling; a keyword is recognized
only when it is the entire identifier, so `exit_code` is an identifier.

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

`alloc` and `free` are reserved keywords for built-in operations. They use
function-like syntax but are not ordinary functions or library functions, and
cannot be used as identifiers.

`const` is a reserved keyword used in declarations and type qualification.

`fn` is a reserved keyword used in function declarations. `void` is reserved for
a function return annotation indicating that the function returns no value.
`exit` is a reserved keyword for the built-in process-exit operation and cannot
be used as an identifier.

Built-in type names are reserved and cannot be used as identifiers: `i8`, `i16`,
`i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `int`, `uint`, `f32`, `f64`, `bool`,
`rune`, `str`, `uintptr`, and `size`.

```text
var int = 3; // invalid: int is reserved
```

### Integer literals

An integer literal contains digits in one of four bases: decimal with no prefix,
hexadecimal with `0x`, binary with `0b`, or octal with `0o`. A prefix must be
followed by at least one digit valid in that base. Hexadecimal digits include
`a`–`f` and `A`–`F`. Digit separators are not allowed.

A literal may have one of the suffixes `i8`, `i16`, `i32`, `i64`, `u8`, `u16`,
`u32`, `u64`, `i`, `u`, or `z`. The fixed-width suffixes name their types; `i`
means `int`, `u` means `uint`, and `z` means `size`. `int`, `uint`, and `size`
are not valid suffixes. `uintptr` has no literal suffix.

```text
42
0x2A
0b101010
0o52
42u8
42i
42u
42z
```

The Flexible Types section defines literal typing and range checks.

### Floating-point literals

A floating-point literal begins with one or more decimal digits, optionally
followed by a decimal point and one or more decimal digits, then optionally an
exponent. At least the decimal point or exponent must be present. An exponent
begins with `e` or `E`, followed by an optional `+` or `-` and at least one
decimal digit. A literal may end with `f32` or `f64` to specify its type.
Hexadecimal floating-point literals are not supported.

```text
3.14
3.14f32
1e3
1.5e-2
1E+3f64
1.  // invalid
.5  // invalid
```

The Flexible Types section defines literal typing and overflow checks.

### String and rune literals

A string literal is enclosed in double quotes. Unicode characters may appear
directly, as in `"x 🌐"`. Escape sequences are decoded and Unicode characters
are encoded as UTF-8; the resulting literal always contains valid UTF-8.

A rune literal is enclosed in single quotes and denotes exactly one Unicode
scalar value, written directly or with an escape sequence. Examples include
`'x'`, `'🌐'`, and `'\n'`. The Flexible Types section defines its type.

String and rune literals support these escapes:

| Escape       | Value                                            |
| ------------ | ------------------------------------------------ |
| `\n`         | Newline                                          |
| `\r`         | Carriage return                                  |
| `\t`         | Tab                                              |
| `\\`         | Backslash                                        |
| `\"`         | Double quote                                     |
| `\0`         | NUL                                              |
| `\uXXXX`     | Unicode scalar: exactly four hexadecimal digits  |
| `\UXXXXXXXX` | Unicode scalar: exactly eight hexadecimal digits |

Rune literals additionally support `\'` for a single quote. An unsupported
escape, an incorrectly sized Unicode escape, or an escape denoting a surrogate
or a value above `U+10FFFF` is invalid.

```text
"x 🌐"
"\u0078 \U0001F310"
'\u0041'
'\U0001F310'
'\''
```

A raw string literal is enclosed in backticks. It may contain any characters
except a backtick, including newlines. Backslashes are literal characters; there
are no escape sequences. Its contents are encoded as UTF-8.

```text
`C:\tmp`
`first line
second line`
```

String literals, including raw string literals, have type `str` and use static
storage that lasts for the entire program. Evaluating a string literal does not
dynamically allocate memory. Its storage does not require `free`.

```text
const s = "hello"; // str
```

### Trailing commas

A trailing comma is permitted in every comma-separated list, on one line or
across multiple lines.

```text
[1, 2,]
f(a, b,)
```

## Types

The built-in types include integers, floating-point numbers, `bool`, `rune`,
`str`, `uintptr`, `size`, and the null type. Array, slice, tuple, struct, and
pointer types are built from other types. A type declaration (see Declarations)
gives a name to a type.

```text
i8 i16 i32 i64 u8 u16 u32 u64 int uint   integer types
f32 f64                                  floating-point types
bool                                     boolean type
rune                                     rune type
str                                      string type
uintptr                                  address integer type
size                                     type-size result type
*T                                       non-null pointer type
*const T                                 pointer to a const value
const *T                                 const pointer to a value
const *const T                           const pointer to a const value
nullable *T                              nullable pointer type
nullable *const T                        nullable pointer to a const value
[N]T                                     array type
[]T                                      slice type
(T, T, ...)                              tuple type
struct { name: T, ... }                  struct type
```

### Integer types

```text
i8  i16  i32  i64
u8  u16  u32  u64
int uint
```

`i8`, `i16`, `i32`, and `i64` are signed integer types 8, 16, 32, and 64 bits
wide. `u8`, `u16`, `u32`, and `u64` are unsigned integer types of the same
widths. These eight types are the fixed-width integer types. Fixed-width signed
integer types use two's-complement representation.

`int` is an implementation-defined signed integer type at least 32 bits wide.
`uint` is the unsigned integer type with the same width as `int`. They are
distinct types in their own right, not aliases for fixed-width integer types.
Equal storage widths do not make them identical: a 32-bit `int` is distinct from
`i32`, and a 32-bit `uint` is distinct from `u32`.

Unsigned integer types are intended for cases where the bit representation or
the unsigned range matters. Signed integer types are the normal choice for
arithmetic.

### Integer conversions

When initializing or assigning a binding of integer type, a concrete integer
source converts implicitly to the destination type if both types have the same
signedness and the source width is no greater than the destination width. The
conversion preserves the value. Implicit narrowing and implicit conversions
between signed and unsigned integers are not permitted, even when the source
value fits the destination. Such conversions require an explicit cast.

Equal-width assignment is permitted between distinct types of the same
signedness. For example, on an implementation with 32-bit `int`, `int` and `i32`
are mutually assignable. This does not make their types identical or make
function pointers with those parameter or return types interchangeable (see
Function types).

The same implicit conversions apply when passing an integer argument to an
integer parameter and when returning an integer value from a function with an
integer return type. The parameter type or declared return type is the
destination type. For example, an `i32` argument may be passed to an `i64`
parameter, and an `i32` value may be returned from a function returning `i64`.

These rules apply to the fixed-width integer types, `int`, and `uint`. The
separate rules for `uintptr`, `size`, and `rune` remain as specified in their
respective sections.

```text
var a: i64 = 42i32; // valid: signed widening
var b: u64 = 42u8;  // valid: unsigned widening
var c: u64 = 42i32; // invalid: signedness differs
var d: i64 = 42u32; // invalid: signedness differs
var e: i8 = 42i32;  // invalid: narrowing requires a cast
```

In a binary expression with two concrete integer operands of different widths,
the narrower operand is implicitly promoted to the wider operand's type only
when both have the same signedness. Signed and unsigned operands cannot be
combined without an explicit cast. Distinct types of equal width cannot be
implicitly promoted to one another in a binary expression, even when they have
the same signedness and are mutually assignable. An expression combining `int`
and `i32` is therefore invalid when `int` is 32 bits wide. The operator
determines the result type; for example, comparisons produce `bool`.

```text
1i32 + 2i64 // the i32 operand is promoted to i64
1u8 + 2u64 // the u8 operand is promoted to u64
1i32 + 2u64 // invalid: signedness differs
```

Literal typing is defined under Flexible Types. A suffixed literal retains its
specified type before any permitted conversion is applied.

### Address and size types

`uintptr` is an unsigned integer type for byte addresses. Its size equals the
size of a pointer, so `size(uintptr) == size(*int)` is valid and evaluates to
`true`. Address arithmetic and casts defines its use in byte-address
calculations.

`size` is the type of the result of `size(T)`. The Type size section defines
that operation. Its range and general arithmetic rules remain open.

### Floating-point types

```text
f32
f64
```

`f32` is the IEEE 754 binary32 format and `f64` is the IEEE 754 binary64 format.
Floating-point values use IEEE 754 representation and semantics.

### Boolean type

`bool` is a distinct type with the values `true` and `false`. Comparisons
produce `bool`. Conditions require `bool`; an integer or pointer is not a
boolean condition. For example, `if 1` is invalid.

### Rune type

`rune` is a distinct type represented as `u32`. A rune denotes a Unicode scalar
value. A concrete `rune` value is never implicitly promoted to another type. The
lowering rules for flexible rune literals are defined in Flexible Types.

### Function types

Function types distinguish their parameter types and return type. Integer
widening at a call site or return does not make different function types
interchangeable. Parameter and return types must match without applying integer
widening when assigning function pointers.

For example, `fn(i32) void` and `fn(i64) void` are different,
non-interchangeable function types. A function pointer binding of type
`*fn(i64) void` cannot accept a function declared with an `i32` parameter, even
though an `i32` argument can be passed directly to an `i64` parameter.

### Pointer types

`*T` is a non-null pointer to a value of type `T`. `nullable *T` can refer to a
value of type `T` or hold `null`, which has the null type. A non-null `*T`
converts implicitly to `nullable *T`.

`*const T` is a non-null pointer that permits reading the pointed-to value of
type `T` but not modifying it through that pointer.

`nullable *const T` is the nullable form of `*const T`.

A `*T` converts implicitly to `*const T`, including when passed as an argument
to a function that accepts `*const T`.

The reverse conversion never occurs implicitly. An explicit cast can remove
pointee constness (see Pointer casts).

A nullable pointer cannot be dereferenced directly. An `as` assertion or a
pattern match obtains a non-null pointer (see Pointer access and null handling).
Unchecked address casts are specified separately under Address arithmetic and
casts.

### Const-qualified types

`const T` qualifies the type `T` as const. A value cannot be modified through
const-qualified access. The qualifier applies to the type immediately following
it: `const *T` is a const pointer to `T`, while `*const T` is a pointer to const
`T`. `const *const T` qualifies both the pointer and its pointee.

Const qualification of a pointer does not itself qualify the pointee. The
declaration keyword and the qualifiers in its type are separate:

```text
const x: *const T = e;
```

Here the declaration prevents reassignment of `x`, and the type qualifier
prevents modifying its pointee through `x`. Writing `const x: const *const T`
adds an outer type qualifier that is redundant for preventing reassignment of
this binding.

Copying a pointer value does not carry the source binding's constness to the
destination. The copy drops any outer `const` qualifier on the pointer type,
while preserving the pointee type and all const qualifiers within it. An
explicit const qualifier on the destination still applies.

```text
const value = 1;
const other = 2;
const p: *const int = &value;
var q = p;    // *const int
q = &other;   // valid: the copied pointer can be reassigned
*q = 3;       // invalid: the pointee is still const
```

### String type

```text
str
```

`str` is the built-in string type. Dynamically allocated strings own their
memory. Strings have no built-in indexing or slicing operations. Content access
and other string-data operations belong to the standard library. The lexical
rules guarantee valid UTF-8 for literals; handling other string data belongs to
the standard library.

### Array types

```text
[N]T
```

An array holds exactly `N` values of type `T`. The length is part of the type,
so `[3]i32` and `[4]i32` are different types.

```text
const xs: [3]i32 = [1, 2, 3];
const ys = [1, 2, 3]; // [3]int
```

In a type annotation, `_` in place of `N` infers the length from the
initializer.

```text
var foo: [_]int = [1, 2, 3]; // [3]int
```

### Slice types

```text
[]T
```

A slice is a sequence of values of type `T`. A slice owns its memory. Operations
on a slice are bounds checked (see Runtime Checks). The Allocation section
defines how a slice is created.

### Tuple types

```text
(T, T, ...)
```

A tuple holds a fixed number of values, each with its own type. The type lists
the element types in order.

```text
var foo: (int, f64) = (42, 3.14);
```

### Struct types

```text
struct {
    name: T,
    ...
}
```

A struct holds a fixed set of named fields, each with its own type. Each field
is written `name: T`, and fields are separated by commas.

```text
type point = struct {
    x: int,
    y: int
}
```

## Flexible Types

Unsuffixed integer literals, unsuffixed floating-point literals, and rune
literals have distinct **flexible types**. A flexible type lowers to a concrete
type required by the context in which its value is assigned, passed, or used.
This terminology describes literal typing; it does not define named constants or
which expressions can be evaluated at compile time.

```text
var a = 42;       // int
var b: u8 = 42;   // u8
var c: i64 = 42;  // i64
var d: uint = 42; // uint
```

When no surrounding context requires another type, a flexible integer lowers to
`int`, a flexible floating-point value to `f64`, and a flexible rune to `rune`.

```text
var a = 3.14;      // f64
var b: f32 = 3.14; // f32
var r = 'a';       // rune, represented as u32
```

A suffixed literal has the concrete type specified by its suffix, regardless of
context.

```text
var x = 42u8;    // u8
var y = 3.14f32; // f32
var z = 42i;     // int
var w = 42u;     // uint
```

An integer literal outside the range of its required type is invalid and must be
rejected at compile time. A floating-point literal that overflows its required
type is also invalid; it does not produce infinity. These rules apply whether
the type comes from a suffix or from context.

```text
256u8            // invalid
var x: u8 = 256; // invalid
1e100f32         // invalid
```

A flexible rune can lower to a type that can hold its Unicode scalar value,
including integer types such as `i32`, `u32`, `u8`, and `u64`. In an expression
with a concrete operand, the flexible rune lowers to that operand's type if it
can hold the value. Lowering to a type that cannot hold the value is invalid.

```text
var a: u8 = 'a';   // 97u8
var b = 'a' + 1u8; // 98u8
var c: u8 = '🌐';  // invalid: the value does not fit
var r = 'a';       // rune
var d = r + 1u8;   // invalid: a concrete rune is not implicitly promoted
```

## Declarations

### Functions

A parameterless function declaration has this form:

```text
fn name() -> T {
    ...
}
```

`T` is the return type, or `void` when the function returns no value. The body
is a brace-delimited sequence of statements and may be empty. No semicolon
follows the closing brace.

The program entry point is a top-level function named `main`, with no parameters
and the explicit return annotation `void`:

```text
fn main() -> void {
}
```

`main` does not return an integer. Reaching the end of its body terminates the
program with exit status 0. A program can explicitly terminate with an integer
exit status using the built-in `exit` operation (see Process exit).

### Type declarations

```text
type name = T
```

A type declaration binds `name` to the type `T`. The name may then be used
wherever a type is required.

```text
type point = struct {
    x: int,
    y: int
}
```

### Variable declarations

```text
var name = e;
const name = e;
var name: T = e;
const name: T = e;
```

A variable declaration introduces a binding initialized to the value of `e`. A
binding declared with `var` permits assignment after initialization, subject to
its type's const qualification. A binding declared with `const` may not be
reassigned.

```text
var x = 10;
x = 20;

const y = 10;
y = 20; // invalid
```

When the declaration has a type annotation `T`, the binding has type `T`.
Otherwise the binding has the type of its initializer, except that a pointer
copy drops the outer const qualifier as specified under Const-qualified types.

```text
var a = 10;     // int
var b: u8 = 10; // u8
var c = 10u8;   // u8
```

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

A `const` binding cannot be reassigned. Its struct fields and array elements
cannot be modified through that binding, including fields and elements nested
within them. This restriction does not propagate through pointers: whether a
pointee may be modified is determined by its const qualification (see
Const-qualified types).

```text
const p = point { x = 10, y = 20 };
p.x = 30; // invalid

const xs = [1, 2, 3];
xs[0] = 10; // invalid

const points = [
    point { x = 1, y = 2 },
    point { x = 3, y = 4 },
];
points[0].x = 10; // invalid
```

```text
var value = 1;
var other = 2;
const p: *int = &value;
*p = 3;       // valid: the pointee is not const
p = &other;   // invalid: p is a const binding

var q: *const int = &value;
q = &other;   // valid: q is a var binding
*q = 4;       // invalid: the pointee is const
```

Taking the address of a `const` binding preserves const access as specified
under Pointer access and null handling.

## Expressions

### Identifiers

An identifier in an expression refers to the most recent declaration of that
name.

### Type size

`size(T)` returns the size of type `T` in bytes, as a value of type `size`. Its
operand is a type, not a value.

```text
size(int)
size(uintptr) == size(*int) // true
```

### Pointer access and null handling

`&x` takes the address of `x`. `*p` dereferences a non-null pointer `p`. Field
access automatically dereferences a non-null pointer: `p.field` accesses the
field of the pointed-to value.

Taking the address of a `const` binding of type `T` produces a `*const T`
pointer (see Pointer types).

```text
const x = 1;
const p: *const int = &x;
const y = *p; // 1
*p = 2;       // invalid: the pointee is const
```

For a nullable pointer `p`, `p is null` produces a `bool` indicating whether it
is null. `!(p is null)` produces the opposite result. A null test does not
narrow the pointer's type, even in a branch where it is known to be non-null.

```text
if !(p is null) {
    const value = p.value; // invalid: p still has a nullable pointer type
}
```

The assertion `p as *T` converts a `nullable *T` to `*T`. It panics if `p` is
null. The assertion produces a non-null pointer value without changing the type
of `p`.

An `as` assertion that removes nullability preserves the pointee type, including
its const qualifiers. For example, a `nullable *const T` can be asserted as
`*const T`; asserting it as `*T` is invalid because that would remove pointee
constness.

```text
const np: *node = p as *node;
const value = np.value;
```

A match expression can distinguish the null and non-null cases. `case null`
handles null. `case const nd` binds the non-null pointer, of type `*T`, in its
branch without changing the type of the matched binding. A `yield` supplies the
value of the match expression.

The non-null pattern binding also preserves the pointee type and its const
qualifiers. Matching a `nullable *const T` binds a `*const T` in the non-null
branch.

```text
const v: int = match (p) {
case null =>
    yield 0;
case const nd =>
    yield nd.value;
};
```

### Pointer casts

An explicit cast `p: *T` converts a `*const T` to `*T`, allowing the programmer
to remove pointee constness. This is an explicit escape hatch from the pointer's
write restrictions. It does not change the type of the source binding.

```text
var value = 1;
const p: *const int = &value;
var implicit: *int = p; // invalid: pointee constness cannot be removed implicitly
const q = p: *int;
*q = 2;                // valid: value was declared var
```

This cast is separate from an `as` assertion that removes nullability, which
preserves pointee constness (see Pointer access and null handling).

### Address arithmetic and casts

Pointers are non-numeric. Arithmetic operators require numeric results, so
`p + 1` and `p += n` on a pointer variable are type errors. The diagnostic for
arithmetic on a pointer is:

```text
Cannot perform arithmetic on non-numeric pointer type
```

Address arithmetic uses `uintptr` values in bytes. Pointer-to-`uintptr` and
`uintptr`-to-pointer casts use `e: T` syntax. A cast from `uintptr` to `*T` is
unchecked: it does not verify alignment or address validity. Guarantees for
dereferencing an invalid result remain unsettled (see Open Questions).

There is no implicit scaling by the size of the pointed-to type. Moving by an
element count requires multiplying by `size(T)` explicitly. In this example, `x`
holds two consecutive `int` values:

```text
var x: [2]int = [1, 2];
const y = &x: uintptr;
assert(*((y + (size(int) * 0): uintptr): *int) == 1);
assert(*((y + (size(int) * 1): uintptr): *int) == 2);
```

The following table defines promotion when combining `uintptr` with another
type. Each row lists the resulting type; arithmetic must still satisfy the
requirement that its result is numeric.

| Other type              | Result                               |
| ----------------------- | ------------------------------------ |
| `size`                  | `uintptr`                            |
| The null type           | `uintptr`                            |
| A pointer type          | The pointer type                     |
| Any other concrete type | Print a diagnostic message and abort |

The last row includes `u64`, `uint`, and every signed type. The phase at which
an unsupported promotion prints a diagnostic and aborts remains an open
question. Compound addition `+=` is permitted on a `uintptr` variable subject to
these promotion rules.

### Array literals

```text
[e, e, ...]
```

An array literal has type `[N]T`, where `N` is the number of elements and `T` is
the type of the elements.

```text
const xs = [1, 2, 3]; // [3]int
```

### Tuple literals

```text
(e, e, ...)
```

A tuple literal has a tuple type whose element types are the types of the
elements, in order.

```text
var foo: (int, f64) = (42, 3.14);
```

### Struct literals

```text
name { name = e, ... }
```

A struct literal names a struct type and gives each field a value with a
`name = e` initializer.

```text
var p = point { x = 10, y = 20 };
```

### Field and element access

`e.name` accesses the field `name` of a struct value; access through a pointer
is defined under Pointer access and null handling. `e.N` accesses element `N` of
a tuple value. Tuple elements are numbered from zero.

```text
p.x   // 10
foo.0 // 42
foo.1 // 3.14
```

### Index expressions

```text
e[i]
```

An index expression accesses the element at index `i` of an array or slice.
Indices start at zero. Index expressions on slices are bounds checked (see
Runtime Checks).

## Statements

This specification defines variable declarations (see Declarations),
assignments, calls to `free` (see Deallocation), and `defer` statements below.
These statements end with `;`. Boolean conditions and the `yield` used in
nullable-pointer match expressions are specified in the Types and Expressions
sections; the remaining control-flow rules are open.

### Assignment

```text
name = e;
```

An assignment stores the value of `e` in its target. The target is a binding, a
struct field such as `p.x`, an array or slice element such as `xs[0]`, or a
dereferenced pointer such as `*p`. Reassigning a binding requires `var`.
Assignments to fields, elements, and pointees must satisfy the Immutability and
Const-qualified types rules.

```text
var x = 10;
x = 20;
```

Assignment of a dynamically allocated string or slice is shallow (see Aliasing).

### Defer

```text
defer s
```

A `defer` statement holds the statement `s` and executes it when the current
scope exits.

```text
var xs = alloc([]u8, 100);
defer free(xs);
```

### Process exit

```text
exit(e);
```

The built-in `exit` takes one argument of type `int` and terminates the program.
The reported exit status is the argument modulo 256, in the range 0 through 255,
on every host. For example, `exit(256)` reports 0 and `exit(-1)` reports 255. It
does not return to its caller. The Flexible Types rules apply to its argument.

```text
fn main() -> void {
    exit(42);
}
```

## Memory Management

### Allocation

```text
alloc([]T, n)
```

Dynamic memory is allocated explicitly, with `alloc`. `alloc([]T, n)` allocates
memory for `n` values of type `T` and evaluates to a slice of type `[]T` that
refers to that memory.

```text
var xs = alloc([]u8, 100); // []u8
```

### Deallocation

```text
free(e);
```

Dynamic memory is released explicitly, with `free`. `free(e)` releases the
allocation that `e` refers to. There is no automatic ownership-based
deallocation.

```text
free(xs);
```

### Aliasing

Assignment of a dynamically allocated string or slice is shallow. After the
assignment, both bindings refer to the same allocation. Assignment copies
neither the contents of the allocation nor exclusive ownership of it.

```text
var a = alloc([]u8, 100);
var b = a;
free(a);
```

After `free(a)`, every alias of the allocation, such as `b`, is dangling.
Avoiding use-after-free and double-free errors is the programmer's
responsibility.

Creating independent storage requires an explicit copy operation.

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

## Runtime Checks

Operations on `[]T` are bounds checked. An out-of-bounds access has defined
behavior rather than undefined behavior.

Nullable-pointer assertions have the panic behavior specified under Pointer
access and null handling. This specification does not yet settle the behavior of
an invalid or misaligned pointer dereference.
