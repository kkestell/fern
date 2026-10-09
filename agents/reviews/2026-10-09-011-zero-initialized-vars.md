# Zero-initialized vars

## Scope and coverage

Reviewed the staged changes for
`agents/plans/2026-10-09-010-zero-initialized-vars.md` against
`docs/spec.md#declarations-and-scope`: the optional initializer in the parser
and syntax tree, `check_binding`, `check_module_initializer`,
`module_dependencies`, `zero_binding_annotation`, module-level global lowering,
`lower_flow_binding` and `FlowBuilder::zero`, `Instruction::Zero` and its
verification, `emit_zero` and `zero_operand`, the `examples/raytrace.fern`
changes, and the new tests. Also traced code the change depends on outside the
diff: `binding_place` and `returned_local` for a zeroed returned local,
`emit_data` and the pointer-cell path for large zero globals, `emit_copy`'s
tiers, and `allocation`. Native behavior was exercised on arm64 Apple only; the
release memory measurement and ray tracer comparison were taken from the work
log, not re-run.

## Fixed

- **Examples tour omits the new form** (`examples/bindings_and_scopes.fern:6`):
  the tour of supported syntax showed no `var` without an initializer, so a
  reader of the examples would not learn it exists. Added a zero-initialized
  `var steps: int;` that the example then increments.

## Findings

No open findings.

## Checks run

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` — all pass.
- `cargo run -- examples/bindings_and_scopes.fern -o target/bindings` and ran
  it — exits 0; `examples_execute` and `documented_examples_compile` pass.
- Compiled and ran probe programs: two 12 MB zero globals (pointer-cell path)
  read back zero after writes to their last elements; a 21-byte align-1 struct
  (word-store path) zeroed and copied correctly; a returned zero local of a
  padded struct, used directly, as an array element, and as a binding
  initializer, kept its zero fields; a function zeroing its returned local
  while reading the global it is assigned to, and one reading through a
  pointer to its destination, both saw the old values. `var x: void;` and
  `var x: [0]int;` are rejected by the existing type checks.

## Verdict

A `var` without an initializer holds its zero value at module level and in a
function, a local is re-zeroed each time its declaration runs, and no constant
or static data holds the zero value. Ready to commit.
