# Pass floating-point values across the external interface

## Goal

External functions accept only integer and pointer parameters and results, so a
Fern program cannot call `sqrt` or any C function that takes or returns a
`float` or `double`. Allow `f32` and `f64` there, corresponding to C `float` and
`double`.

## Related code

- `docs/spec.md#functions-and-external-calls` — states the types allowed across
  the external interface and excludes floats.
- `src/semantic/namespaces.rs:514` — rejects external parameter and result
  types other than integers and pointers with
  `external functions accept only integers and pointers`.
- `src/ir/verify.rs:381` — `verify_function` checks the same restriction on
  lowered external declarations.
- `src/backend/emitter.rs:410` and `src/backend/qbe.rs:271` — external calls
  type each argument and the result with `abi_class`, which falls back to
  `value_class`; `qbe_type` already maps `f32` to `s` and `f64` to `d`, so QBE
  applies the C ABI for floats with no backend change.
- `src/types.rs:107` — `Scalar::is_numeric` covers integer and floating types.
- `tests/core.rs:235` — `c_abi_small_integers_and_struct_memory` links a C
  helper object and is the pattern for the new native test.

## Decisions

- The allowed external types become numeric scalars and pointers. `bool`,
  arrays, and structs stay rejected; structs are a later todo item.
- The diagnostic becomes `external functions accept only numeric and pointer
  types`.

## Test plan

- Native test `c_abi_floating_point` in `tests/core.rs`: a C helper defines
  `float halve(float)`, `double mix(int32_t, float, double, int64_t)` that
  combines all four arguments, and `double sum10(double, ..., double)` with ten
  `double` parameters so some arguments pass on the stack. The Fern program
  calls each with values whose results are exact in binary floating point,
  passes an `f32` loaded from a struct field, and calls libm `sqrt` linked with
  `-l m`. Each mismatch exits with a distinct status.
- `src/semantic/tests/mod.rs` `external_signatures_and_returns`: replace the
  rejected `extern fn foo(x:f64)->void;` with `extern fn foo(x:bool)->void;`
  and add `extern fn foo()->bool;`. Add an accepted
  `extern fn foo(x:f32,y:f64)->f64;`.
- `tests/core.rs` diagnostic cases: replace the `f32` case with
  `extern fn x(n:bool)->void;` expecting `numeric and pointer`.

## Implementation plan

1. Update `docs/spec.md#functions-and-external-calls` through `/kspec`:
   parameters and results are numeric or pointer types; `f32` and `f64`
   correspond to C `float` and `double`; remove floats from the list of values
   that cannot pass directly across the external interface.
2. In `src/semantic/namespaces.rs`, accept a type when it is a pointer or its
   scalar `is_numeric`, and change the diagnostic text.
3. In `src/ir/verify.rs`, make `external_type` accept the same types.
4. Add and update the tests above.
