# Performance review: entire codebase

Scope: every production Rust source file in `src/`, with the unit and
integration tests that exercise the compiler pipeline.

Mode: performance

## Findings

### Medium: Global-data emission allocates one retained string per scalar

- `src/backend/emitter.rs:144-155` turns every scalar in every mutable global
  into a separate `String`, retains all of them in a `Vec`, then joins them.
  Lowering flattens each mutable global's constant initializer into `Global::values` at
  `src/ir/lower.rs:46-58`, so the loop runs once per scalar, not once per
  written array literal.
- A `var values: [100000]int = [1...];` has a 55-byte source file but reaches
  this path with 100,000 values. It compiled three times in 0.30 seconds in a
  release build, including QBE and the C toolchain, and produced an 826 KB
  executable. The time is end-to-end rather than emitter-isolated, but the
  100,000 temporary strings and their vector are directly established by this
  loop.
- Large static tables therefore add an avoidable allocation per scalar and
  retain every formatted item until the complete global has been joined. Peak
  compiler memory grows by the QBE data size plus the vector and its strings.
- Append the data items directly to `emitter.data`, writing separators as the
  loop advances. This keeps the required final QBE buffer while removing the
  intermediate vector and its per-value string allocations. Add a focused
  emitter test with a large filled mutable global and assert its output.

### Medium: Each function copies the whole module binding map in two phases

- Semantic checking copies `module_scope` at
  `src/semantic/namespaces.rs:604`; lowering copies `module_places` at
  `src/ir/lower.rs:81`. The former map has every module-level binding and the
  latter every mutable module-level binding.
- For a module with M bindings and F functions, these copies do O(M × F)
  hash-table entry work and allocations before checking or lowering function
  bodies. The copies occur even when a function neither reads nor writes a
  module binding.
- This is unmeasured in isolation. The call frequency and copied collection
  sizes are established by the pipeline: both sites run once for every parsed
  function, and their source maps are module-wide.
- Keep the module map immutable and shared for the phase. Give each function a
  map containing only parameters and local bindings, and resolve a name or
  place through that map before falling back to the shared module map. That
  preserves shadowing while making the per-function setup proportional to its
  own parameters and locals.

## Unresolved suspicions

None. The definite-initialization verifier uses a compact bit set and its
fixed-point work is appropriate for the current control-flow representation.
The aggregate-type deduplication uses a linear search, but array signatures are
not on the measured hot path and the extra set needed to remove it would add
complexity without evidence of a material cost.

## Checks

- Inventoried and read all 30 production Rust files (9,409 lines), then traced
  the compile path from `compile` through loading, parsing, semantic checking,
  lowering, verification, QBE emission, and native-tool execution.
- Reviewed parser, module-loader, semantic, IR, verifier, backend, source, and
  diagnostic loops for input size, call frequency, allocations, copies, I/O,
  and layout. Read and ran the unit and integration coverage for those paths.
- Built debug and release profiles successfully. `cargo clippy --all-targets
  -- -D warnings` passed. `cargo test` passed: 193 unit tests and 50 integration
  tests.
- Timed 20 release compiles of `examples/arrays.fern` in 1.06 seconds. Profiled
  it with Samply across 60 runs. Samply did not resolve Fern's local frames, so
  the profile was used only to confirm the short-lived end-to-end workload.
- Compiled and profiled the 100,000-element mutable-global workload above. The
  workload is under ignored `target/` and did not alter tracked files.
