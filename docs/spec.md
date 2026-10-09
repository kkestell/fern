# Fern Language Specification

Fern is a small, low-level language with explicit types, value semantics, and
ordinary machine operations. Programs are responsible for memory correctness.
Invalid source is rejected with a compilation diagnostic. Undefined operations
have no specified behavior; implementations need not diagnose or check them.

## Source and syntax

Source is UTF-8. Identifiers use ASCII letters, digits, and underscores, cannot
start with a digit, and cannot be reserved words. Whitespace separates tokens;
newlines do not insert semicolons. `//` comments end at a newline. `/* */`
comments nest. Tokens use their longest spelling.

Reserved words are `fn`, `extern`, `void`, `var`, `const`, `type`, `struct`,
`pub`, `use`, `if`, `else`, `for`, `break`, `continue`, `return`, `len`,
`sizeof`, `null`, `true`, `false`, and the scalar type names.

Declarations, assignments, calls, `break`, `continue`, and `return` statements
end with `;`. Blocks, function definitions, struct declarations, `if`, and
`for` statements do not.

Integer literals are decimal, or use `0x`, `0o`, or `0b` prefixes. Leading
zeros do not select octal. Floating literals use a decimal point or `e`/`E`
exponent; `.5`, `2.`, and `1e-3` are valid. Neither kind accepts separators or
type suffixes. A literal receives its numeric type from context, otherwise an
integer defaults to `int`, a character literal to `u8`, and a float to `f64`.
An integer literal must fit its selected type. A floating literal rounds
directly to its selected format with round-to-nearest, ties-to-even; overflow
produces infinity.

A character literal is one byte between single quotes. It is an integer
literal whose value is that byte; `'é'` is invalid because it is two bytes.
A string literal is a sequence of bytes between double quotes. Its value is a
`*u8` to static storage holding those bytes followed by a zero byte, which
belongs to the sequence for pointer arithmetic. The storage lives for the
program, and writing to it is undefined. Every evaluation of one string literal
yields the same address; whether distinct literals share storage is
unspecified.

Inside either literal, `\n`, `\r`, `\t`, `\0`, `\\`, `\"`, `\'`, and `\x`
followed by exactly two hexadecimal digits each denote one byte. Every other
character denotes its UTF-8 bytes. A literal cannot contain a newline, any
other backslash sequence, or an unescaped copy of its own delimiter. There are
no raw strings and no concatenation of adjacent literals.

```fern
extern fn puts(text: *u8) -> i32;

const greeting = "hello\tworld\x21";

fn main() -> void {
    if *greeting == 'h' {
        puts(greeting);
    }
}
```

## Types

| Type | Meaning |
| --- | --- |
| `bool` | `true` or `false` |
| `i8`, `i16`, `i32`, `i64` | Signed two's-complement integers |
| `u8`, `u16`, `u32`, `u64` | Unsigned integers |
| `int`, `uint` | Signed and unsigned pointer-width integers |
| `f32`, `f64` | IEEE 754 binary32 and binary64 |
| `*T` | Raw pointer to a value of type `T`, possibly null |
| `[N]T` | Fixed array of `N` values of type `T` |
| Declared struct name | Named value type |

Every scalar type is distinct, including `int` and `i64` on a 64-bit target.
There are no implicit conversions between concrete numeric types. `void`
means no function result and is not a value type; `*void` is not a type.
Arrays have a positive, compile-time `int` length. Array type identity includes
the length and element type; pointer identity includes the target type.

```fern
type Point struct {
    x: f64,
    y: f64,
}
```

Structs are module-level declarations, have at least one field, and have
nominal identity. Fields have unique names and no independent visibility.
Following fields and array elements must not form a recursive value type;
pointers permit recursive data structures.

Scalar storage uses its actual width in bytes; `bool` occupies one byte.
Pointers occupy the platform pointer width. Arrays are contiguous. Struct
fields have declaration order, with natural host C ABI alignment and padding.
`sizeof(T)` is a constant expression of type `uint` giving the byte size,
including padding. Its operand is a type and is not evaluated.

Initialization, assignment, argument passing, and return copy values, including
all fields and elements of aggregates. Pointer copies refer to the same target.

## Declarations and scope

```fern
var count = 0;
var bytes: [3]u8 = [10, 20, 30];
const capacity: int = 16;
```

Every binding requires an initializer. `var` creates a mutable location;
`const` requires a compile-time value and creates no addressable location.
Without a type annotation, the initializer determines the type. An annotation
supplies the expected type for literals and aggregate literals.

Every block introduces a lexical scope. A local becomes visible after its
initializer. A declaration may shadow an earlier declaration in the same or
an enclosing block; its initializer still sees the earlier declaration.
Module-level declarations share one namespace and cannot repeat a name.
They are visible throughout their module regardless of order. Module-level
initializers must be constant expressions; initialization dependencies must
not contain a cycle. All globals are initialized before `main`.

There is no implicit zero initialization, aggregate fill, or inferred-length
array type annotation. Array literals infer their length when no array type
is expected. The first element determines the element type in that case;
subsequent elements are checked against it.

```fern
var a = [1, 2, 3];
var point = Point { x = 1.0, y = 2.0 };
```

Struct literals must name every field exactly once, using `=`. Initializers
execute in written order. Arrays require exactly their type's number of
elements, all of the element type.

## Functions and external calls

```fern
fn add(a: int, b: int) -> int {
    return a + b;
}

extern fn putchar(character: i32) -> i32;
extern fn malloc(bytes: uint) -> *u8;
extern fn free(memory: *u8) -> void;
extern fn exit(status: i32) -> void;
```

Functions are module-level, named, non-overloaded, and directly callable.
Parameters have explicit types and are mutable local copies of their arguments.
A result annotation is mandatory and is `void` or one value type. Recursive
and mutually recursive calls are permitted. Functions are not values.
Arguments evaluate completely from left to right before entering the callee.
A call statement may discard a result. A `void` call is only a statement.

External functions have no body and name a linker symbol exactly matching
their declared name. They use the host C calling convention. Parameters and
results are integer or pointer types; a result may also be `void`. Integer
types correspond to C integers of the same width and signedness; `int` and
`uint` correspond to `intptr_t` and `uintptr_t`. Pointer types have C pointer
representation and calling convention. The programmer must supply a declaration
matching the actual C function. Declaring one symbol with inconsistent
signatures, or calling it with an incorrect signature, is invalid. External
symbol `main` is reserved for program startup. Variadic calls, callbacks,
external globals, and aggregates or floats passed directly across the external
interface are not part of this core. Values may be exchanged through pointers.
The implementation documents which libraries it links.

An executable's root module must define `fn main() -> void`. A dependency's
`main` is ordinary. Returning from the root `main` exits successfully.
Termination, allocation, freeing, and I/O are library operations, not built-ins.
External functions returning `void` are assumed to return for control-flow
checking, including an external function named `exit`.

## Expressions

Field selection, indexing, calls, conversions, `len`, and `sizeof` bind most
tightly. Unary `-`, `^`, `!`, `&`, and `*` follow. Binary precedence, highest
first, is:

1. `* / % << >> &`
2. `+ - | ^`
3. `== != < <= > >=`
4. `&&`
5. `||`

Each level associates left to right. Parentheses override grouping. Operands
and call arguments evaluate left to right; `&&` and `||` short circuit.
A struct literal in an `if` or `for` condition must be parenthesized to
distinguish its opening brace from the statement body.

Numeric binary operands must have the same type, except shift counts.
An expected numeric type supplies the type of literal arithmetic; otherwise
concrete operand types supply it, then floating literals select `f64`, then
character literals select `u8`, then integer literals select `int`. Existing concrete values never change type
implicitly. Constants follow these same rules: intermediate operations use
their concrete type, rather than arbitrary-precision arithmetic.

### Integers

`+`, `-`, `*`, and unary `-` wrap modulo two to the type's width, with signed
results interpreted as two's complement. Bitwise `&`, `|`, `^`, and unary `^`
operate on that same representation. There are no separate wrapping operators.

`/` truncates toward zero; `%` has the dividend's sign. Division or remainder
by zero, and signed minimum divided or reduced modulo `-1`, are undefined.
Shift counts may have any integer type. Counts below zero or at least the left
operand's width are undefined. Left shift discards high bits; right shift
sign-fills signed operands and zero-fills unsigned operands.

### Floating point

Every operation on `f32` or `f64` rounds to that format with round-to-nearest,
ties-to-even. `+`, `-`, `*`, `/`, and unary `-` follow IEEE 754, including
infinities, NaNs, subnormals, and signed zero. Division by floating zero is
defined. Operations are not fused across expression boundaries.

### Numeric conversions

`T(e)` explicitly converts a numeric value. Integer-to-integer conversion
retains the destination-width low bits, sign-extending a signed source when
widening. Integer-to-float and float-to-float conversions round to the nearest
representable value, ties-to-even, and may produce infinity. Float-to-integer
conversion truncates toward zero; a non-finite value or truncated result outside
the integer range is undefined. There is no precision checking or separate
truncating-conversion syntax. `bool` does not convert to or from numbers.

### Booleans and comparisons

Conditions require `bool`. `!`, `&&`, and `||` operate only on booleans.
`==` and `!=` compare values of the same type; arrays and structs compare
corresponding elements or fields recursively. Pointer equality compares
addresses. Numeric ordering uses `<`, `<=`, `>`, and `>=`. IEEE comparison
rules apply to floats: NaNs are unequal to every value, including themselves,
and every ordering comparison involving NaN is false; signed zeros are equal.

### Arrays and pointers

`a[i]` selects an array element. Its index has type `int`; out-of-range access
is undefined. `len(a)` accepts an array value and yields its type's length as
an `int`. The operand executes normally; `len(a)` is a constant expression
only when the operand is one.

`&location` takes the address of a mutable location, including a parameter.
`*p` selects the target of a pointer and permits reading and writing. Field
access requires a struct value: write `(*p).field` to access through a pointer.
A pointer to an array must be dereferenced to select the array's elements.

`null` receives a pointer type from context and has no default type. Every
pointer type accepts null. `*T(p)` reinterprets a pointer as another pointer
type, without a check. Dereferencing null, a misaligned pointer, a dangling
pointer, or memory that does not contain the target type is undefined.

For `p: *T`, `p[i]` selects target `i`, and `p + i` and `p - i` advance by
`i * sizeof(T)` bytes. Offsets have type `int`. Arithmetic may reach elements
or one past the end of the same live array or allocated sequence of `T`; other
results are undefined. A single `T` acts as a one-element sequence. Subtracting
two pointers to the same element type yields their element distance as `int`;
the pointers must belong to that same sequence and the result must fit `int`.
A one-past pointer may not be dereferenced.

Locals live until their declaring block ends; parameter copies live until
return. Globals live for the program. Heap storage follows the external
allocator's contract. Use after lifetime end, invalid frees, and double frees
are undefined. Fern has no ownership, lifetime tracking, or read-only pointers.

## Statements and control flow

Assignment targets are mutable locations: variables, parameters, dereferenced
pointers, and fields or elements of mutable locations. Targets evaluate before
the right-hand side. `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`,
and `>>=` evaluate their target once and apply the corresponding operation.
Assignment is a statement, not an expression.

```fern
if count > 0 {
    count -= 1;
} else if count == 0 {
    count = 10;
} else {
    count = 0;
}

for count > 0 {
    count -= 1;
}

for {
    break;
}
```

Braces are required. `for` is either infinite or tests a boolean condition
before each iteration. Each iteration enters a fresh body scope. `break`
leaves the nearest loop; `continue` begins its next iteration. Neither is
permitted outside a loop. There are no labels or other loop forms.

`return;` returns from a `void` function; `return e;` returns one value of the
declared result type. A `void` function may reach its end. A value-returning
function must structurally return on every path. A final `else` whose branches
all return satisfies this check. An infinite loop without a reachable `break`
also satisfies it. Constant conditions do not remove paths. Unreachable source
is still checked.

## Constant expressions

Literals, references to `const` bindings, grouping, numeric conversions,
unary and binary operators, comparisons, logical operations, aggregate
literals, constant field selection and indexing, `len` of a constant array,
and `sizeof` may form constant expressions. Calls, addresses, dereferences,
and references to mutable bindings are never constant expressions. A string
literal is a constant expression, but its address is not a compile-time value:
comparing, converting, offsetting, indexing, or dereferencing it is not.

Constant evaluation uses the same concrete numeric rules as runtime. Integer
wrapping, IEEE rounding, infinities, and NaNs are permitted. A constant operation
that would be undefined is rejected, including an invalid shift, division by
integer zero, out-of-range constant indexing, or invalid float-to-integer
conversion. Both operands must be valid constants even in a short-circuited
constant logical expression. Implementations may impose documented compilation
resource limits and must report exceeding them as a diagnostic.

## Modules and imports

A directory's direct `.fern` files form one module and share declarations,
including private declarations. Subdirectories are separate modules. An
executable build selects its root module; source filenames do not create
namespaces. Imports are file-local and precede every other declaration.

```fern
use math;
use geometry::{Point};
use network::http;
```

A whole-module import introduces its path's final component; `math::add`
accesses a public declaration. A selective import introduces the selected
public declarations unqualified. `pub` may precede module-level functions,
external functions, structs, and bindings. Public struct fields are public.
Public variables are assignable from importing modules. Imports cannot rename
or re-export declarations. Imported names cannot conflict with module-level
or other imported names and must be referenced in their source file. Locals
may shadow imported names.

Import path components are identifiers separated by `::` and map to directories
under ordered search roots. The first matching directory with `.fern` files
wins. An implementation documents how roots are supplied; nonempty `FERNPATH`
entries replace CLI defaults, and an empty list preserves defaults. Unresolved
imports report the path and all searched roots. Module dependencies must be
acyclic; cycle diagnostics show the chain. Each module is included once.

A string type, slices, function values, generics, variadic functions, and implicit
numeric conversions are not part of this core. Stack exhaustion and allocation
failure follow the platform and library contracts; no trap runtime is required.
