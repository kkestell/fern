# Call variadic external functions

## Goal

A Fern program cannot call `printf`, `snprintf`, or any other C function that
takes extra arguments after its named parameters. Let an external declaration
end its parameter list with `...`, and pass the extra arguments at the call
with the host C ABI's variadic conventions.

## Related code

- `docs/spec.md#functions-and-external-calls` — lists variadic calls as outside
  the core; `docs/spec.md:355` lists variadic functions among omissions.
- `src/frontend/lexer.rs:73` — the `Dot` token. `...` becomes a new token.
- `src/frontend/parser.rs:359` — `function`, `parameters`, and `typed_names`,
  which parameter lists and struct field lists share.
- `src/frontend/syntax.rs:40` — the syntax `Function`.
- `src/semantic/model.rs:187` — `FunctionSignature`.
- `src/semantic/namespaces.rs:476` — `resolve_signatures`; and
  `src/semantic/namespaces.rs:190`, which rejects inconsistent declarations of
  one external symbol by comparing parameter and result types.
- `src/semantic/statements.rs:344` — `check_call` checks the argument count and
  checks each argument against its parameter type.
- `src/semantic/expressions.rs:67` — `check_expression` with no destination
  gives an untyped constant its default type and reports `null` without
  context.
- `src/types.rs` — `Type::crosses_external_interface`, the rule for fixed
  external parameters.
- `src/ir/model.rs:235` — the IR `Function`; `src/ir/lower.rs:121` builds the
  IR for an external declaration.
- `src/ir/verify.rs:381` and `src/ir/verify.rs:542` — `verify_function` checks
  external declarations; `verify_call` checks the argument count and types.
- `src/backend/emitter.rs` — `emit_call` builds the QBE argument list with
  `abi_class`.

## Decisions

- Syntax: `extern fn printf(format: *u8, ...) -> i32;`. `...` is one token and
  ends the parameter list; it may stand alone, as in `extern fn f(...)`, and
  takes no trailing comma. Only external functions may be variadic; the parser
  rejects `...` in a function with a body with
  `only external functions can be variadic`.
- An extra argument has no expected type, so an untyped literal takes its
  default type (`5` is `int`, `1.5` is `f64`, `'a'` is `u8`) and `null` is
  rejected for lack of context, as it is everywhere else.
- An extra argument must have a numeric type other than `f32`, or a pointer
  type. C never receives a `float` through `...`, and Fern does not convert
  `f32` implicitly, so `f32` reports
  `an f32 variadic argument must be converted to f64`; any other type reports
  `variadic arguments accept only numeric and pointer types`.
- Structs are excluded from extra arguments: QBE 1.3 on arm64 Apple passes both
  a two-`float` struct and a 40-byte struct after `...` incorrectly, so C reads
  the wrong values.
- An extra argument passes as C passes a variadic argument of the
  corresponding C type. The 8- and 16-bit integers go through `abi_class`'s
  `sb`, `ub`, `sh`, and `uh` classes, which QBE extends as C's integer
  promotion does, so a C callee reading them with `va_arg(ap, int)` sees the
  same value.
- The rule lives once, as `Type::passes_as_variadic_argument(&self) -> bool` in
  `src/types.rs`, called by semantic checking and IR verification.
- Being variadic is part of an external signature: declarations of one symbol
  that differ only in `...` are inconsistent.
- A variadic call needs at least the fixed parameters. Too few arguments report
  `function `f` expects at least N argument(s), found M`.
- The syntax `Function`, `FunctionSignature`, and the IR `Function` each gain a
  `variadic: bool`. `emit_call` writes `...` after the fixed arguments of every
  call to a variadic function, including a call with no extra arguments, and
  names each extra argument with `abi_class` of its operand's type.

## Test plan

- Native test `c_variadic_calls` in `tests/core.rs`. A C helper defines
  `int64_t take(int32_t count, ...)`, which reads in order an `int`, an
  `int64_t`, a `double`, a `char *`, an `int` from an `i8` of `-3`, an
  `unsigned` from a `u16` of `65535`, an `intptr_t` from untyped `5`, a `double`
  from untyped `1.5`, and an `int` from `'a'`, and returns a checksum that
  identifies a wrong value. The Fern program calls it with eleven or more
  arguments, so some pass on the stack on every target. It also calls libc
  `snprintf` into a `[32]u8` with `"%d %s %.1f"`, `i32(42)`, a string literal,
  and `2.5`, and compares the bytes with `strcmp`, and calls a variadic helper
  with no extra arguments. Each mismatch exits with a distinct status.
- `src/semantic/tests/mod.rs`: reject, against `extern fn f(x:i32,...)->void;`,
  `f();`, `f(1,f32(1.0));`, `f(1,true);`, `f(1,null);`, `f(1,[1,2]);`, and a
  struct extra argument; accept `f(1);` and `f(1,2,2.5,"s",u8(3),&x);`.
- `src/frontend/tests/mod.rs`: reject `fn f(x:int,...)->void {}`,
  `extern fn f(...,x:int)->void;`, `extern fn f(x:int,...,)->void;`, and
  `type S struct {x:int,...}`; accept `extern fn f(...)->void;`. Update the
  syntax snapshot for the new `variadic` field, and add `...` to the snapshot
  source.
- `src/frontend/tests/lexer.rs`: `...` lexes as one token, and `x.y` and `.5`
  are unchanged.
- `tests/core.rs`: a module test like
  `inconsistent_external_signatures_across_modules` where one module declares
  `extern fn printf(f:*u8,...)->i32;` and another
  `extern fn printf(f:*u8)->i32;`, expecting
  `inconsistent declarations of external symbol`. Add an `f32` extra argument
  to the diagnostic cases, expecting `converted to f64`.
- `src/ir/tests/mod.rs`: a lowered variadic call verifies; it fails to verify
  after the callee's `variadic` is cleared, and after `main`'s `variadic` is
  set.
- `src/backend/tests/mod.rs`: a call to a variadic external writes `...` after
  its fixed arguments, and a call with no extra arguments ends with `...`.

## Implementation plan

1. Update `docs/spec.md` through `/kspec`: the `...` syntax with an example
   in `#functions-and-external-calls`; only external functions may be
   variadic; a call passes at least the fixed arguments; extra arguments have
   no expected type, must be numeric other than `f32` or pointers, and pass as
   C passes a variadic argument of the corresponding C type; `...` is part of
   the signature for consistency between declarations. Remove variadic calls
   from the list of things outside the core, and say that struct extra
   arguments are outside it. At `docs/spec.md:355`, make the omission
   "variadic Fern functions".
2. Add the `...` token to `src/frontend/lexer.rs`; parse it in `parameters`,
   with `typed_names` stopping before it; add `variadic` to the syntax
   `Function`.
3. Add `Type::passes_as_variadic_argument` to `src/types.rs`.
4. Add `variadic` to `FunctionSignature`, set it in `resolve_signatures`,
   include it in the external consistency comparison, and check extra
   arguments in `check_call`.
5. Add `variadic` to the IR `Function`, set it in `lower_function`, and
   verify it in `verify_function` and `verify_call`.
6. Emit `...` and the extra arguments in `emit_call`.
7. Add the tests above.
