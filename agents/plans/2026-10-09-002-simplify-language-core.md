# Simplify Fern's language core

## Goal

Implement the approved small machine-level language, retaining floating-point
numbers and directory modules. The specification owns the complete language
contract. Remove compiler machinery whose purpose is enforcing the discarded
contracts, rather than keeping inaccessible features behind parser guards.

## Related code

- `src/frontend/` — existing lexer, parser, and syntax arenas.
- `src/module.rs`, `src/source.rs`, `src/diagnostic.rs` — retain loading and
  source diagnostics.
- `src/semantic/`, `src/ir/`, `src/backend/`, `src/types.rs`, `src/layout.rs` —
  simplify the existing semantic, IR, and QBE pipeline.
- `tests/compiler.rs`, `examples/`, `README.md`, `AGENTS.md` — executable
  contracts, the language tour, and compiler development documentation.

## Decisions

- Retain `f32` and `f64`; constants and runtime use IEEE arithmetic. Numeric
  conversions round floats and truncate integers without precision checks.
- Use `extern fn name(...) -> T;` for the host C ABI. External parameters and
  results are integers or pointers; results may also be `void`. No variadics,
  callbacks, external globals, aggregate ABI, or library-selection flags.
- Retain QBE as the native backend and the existing frontend, semantic, and IR
  pipeline. Remove checks and representations made unnecessary by the approved
  language rules. Implement external functions through QBE's C ABI support.
- Use actual fixed-width scalar storage and C-compatible struct layout.
  Arrays remain copied values through the existing IR storage model.
- Constants have ordinary concrete types and fold through the same numeric
  rules as runtime expressions. Arbitrary-precision arithmetic is unnecessary.
- Preserve directory modules, visibility, file-local imports, shadowing,
  direct recursion, aggregate equality, and the command-line/output contract.
- Keep nonempty fixed arrays and structs; remove fills, inferred-length type
  annotations, automatic zero initialization, labels, three-clause and
  element-iteration loops. Explicit index loops provide iteration.
- Add `sizeof(T)` for allocating typed storage through external `malloc`.
  Pointer conversions reinterpret pointers. All pointers may be null, and
  field access through pointers requires explicit dereferencing.
- Parameters and `var` locals are mutable; `const` declarations require a
  compile-time initializer and have no addressable storage.

## Test plan

- Compile and execute integer wrapping, typed constant/runtime equivalence,
  float rounding, signed zero, infinities, NaNs, and numeric conversions.
- Check pointer arithmetic, casts, null comparisons, stack addresses, mutable
  parameters, aggregate copies, equality, nested layouts, and allocation/I/O
  through libc. Never execute programs with undefined behavior.
- Check left-to-right operands, arguments and field initializers, short
  circuiting, compound assignment evaluation once, loops, and early returns.
- Reject removed syntax, type mismatches, invalid constant operations,
  recursive value layouts, invalid external signatures, and missing returns.
- Preserve import resolution, visibility, cycles, diagnostics, and atomic
  executable publication. Compile and run the complete rewritten example tour.

## Implementation plan

1. Rewrite `docs/spec.md` around the approved core.
2. Simplify `src/types.rs`, `src/frontend/lexer.rs`,
   `src/frontend/syntax.rs`, and `src/frontend/parser.rs`.
3. Simplify `src/semantic/`, `src/ir/`, `src/layout.rs`, and `src/backend/`
   in place. Remove the trap runtime and exact untyped arithmetic; add raw
   pointer arithmetic, ordinary numeric conversions, `sizeof`, and external
   function calls to the existing models, lowering, verification, and QBE
   emission. Retain the QBE/toolchain orchestration.
4. Update phase integration in `src/lib.rs`; remove dependencies only when
   their uses have been removed.
5. Replace obsolete phase tests, snapshots, fixtures, and examples with tests
   of the core and update `README.md`, `AGENTS.md`, and `agents/bugs.md`.
6. Validate, write `agents/work/2026-10-09-002-simplify-language-core.md`,
   stage the work, review it, record findings, and commit the complete change.

## Documentation updates

Keep language rules solely in `docs/spec.md`. README documents the toolchain
and tour; AGENTS documents the resulting compiler map. Remove the obsolete
stack-check defect from `agents/bugs.md`. Preserve the user's existing removal
of the array-inference issue; its spec edits are superseded by the approved
single-pointer model.
