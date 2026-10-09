# Variadic external calls

## Scope and coverage

Reviewed the staged changes for
`agents/plans/2026-10-09-008-variadic-external-calls.md`: the `...` token and
parameter-list parsing, `Type::passes_as_variadic_argument`, the semantic
signature, external consistency check, and call checking, IR lowering and
verification, QBE call emission, the specification text, and the tests. Also
traced the code that consumes call arguments outside the diff: argument
lowering in `src/ir/lower.rs`, call-destination verification in
`src/ir/verify.rs`, and `abi_class` in `src/backend/qbe.rs`. Native behavior
was exercised on arm64 Apple only.

## Fixed

- **Specification restates argument evaluation order** (`docs/spec.md:183`):
  the variadic paragraph repeated that arguments evaluate left to right, a fact
  already owned by the general call rule at `docs/spec.md:151`, and said a call
  "passes every declared parameter". Removed the restatement and reworded the
  sentence to say a call passes an argument for every declared parameter.

## Findings

No open findings.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — all pass.
- Compiled and ran probe programs: `printf` with `int`, `f64`, negative,
  pointer, `i8`, `i16`, `u16`, and `u8` extra arguments printed the expected
  values; an out-of-range untyped extra argument, a `bool`, an `f32` element,
  and a `void` call as an extra argument were rejected with the expected
  diagnostics; `fn f(...)`, `f(x:int,...,)`, and `f(x:int ...)` were rejected.
- Linked a C helper and confirmed a variadic external returning a struct, and
  one with an `f32` fixed parameter followed by extra arguments, receive and
  return the correct values.

## Verdict

The change meets the plan. One documentation finding was fixed; none remain
open.
