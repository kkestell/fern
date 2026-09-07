# Lower and Execute Integer Programs

## Sources

- [Roadmap: Lower and execute checked programs](../roadmap.md#lower-and-execute-checked-programs)
  and [Milestone completion gates](../roadmap.md#milestone-completion-gates) —
  scope and integrated completion boundary.
- [Specification: Functions](../../docs/spec.md#functions),
  [Scope and shadowing](../../docs/spec.md#scope-and-shadowing), and
  [Process exit](../../docs/spec.md#process-exit) — execution contracts.
- `src/semantic.rs` — checked expression values, declaration identities, and
  complete source-order checking.
- `README.md` — implementation integer width and native tool requirements.

## Goal

Compile checked integer bindings and exits to native executables, completing the
current milestone. Replace the temporary nonempty-body lowering rejection with
owned IR, verification, and QBE emission.

## Implementation

- `src/ir.rs` — replace the status-only entry with an owned straight-line entry:
  ordered value definitions for integer constants and copies, followed by one
  exit terminator with a constant or value operand. Use sequential IR-local
  value IDs; retain no syntax references, interned names, or semantic binding
  IDs in the result. Represent integers with the existing `i32` choice.
- `src/ir.rs` — lower statements using checked expression facts and a map from
  semantic binding IDs to IR values. Emit a distinct definition for each
  declaration, preserving reference copies and shadowing. Stop emitting at the
  first exit; otherwise terminate with zero. Keep complete semantic checking
  ahead of lowering in the driver.
- `src/ir.rs` — add verification that rejects nonexistent and forward value
  references in copies and nonexistent references in the terminator without
  indexing unchecked IDs. Encode the single final terminator structurally.
  Provide a verified-entry wrapper whose private construction requires
  successful verification and whose accessors expose only immutable IR.
- `src/lib.rs`, `src/backend.rs` — verify after lowering and accept only
  verified entries in backend emission and building. Report verification
  failures as internal compiler errors through `CompileError`, before creating
  backend intermediates or invoking tools.
- `src/backend.rs` — emit deterministic QBE word temporaries for definitions and
  copies. Explicitly mask the exit operand with 255 before returning it from the
  generated host `main`, rather than relying on host exit-status truncation.
  Preserve the existing temporary-file and output-publication path.
- `src/semantic.rs` — remove the dead-code allowance on checked expression
  values once lowering consumes them.
- `tests/compiler.rs` — replace the valid-program lowering failure expectation
  with execution assertions. Extend existing success and output-preservation
  coverage to nonempty programs.
- `README.md` — update executable support and the existing integer and shadowing
  example descriptions now that both compile and run.

## Tests

- Snapshot Fern IR for an empty entry, literal declarations, reference chains,
  repeated shadowing, and an early exit followed by valid declarations and
  another exit. Assert deterministic IDs and absence of instructions after
  termination.
- Construct malformed IR with out-of-range and forward copy references and an
  invalid exit reference. Confirm verification rejects each; test valid copies
  and exits through the same boundary.
- Exercise the backend mask with verified IR containing negative values,
  including `-1` and `i32::MIN`; do not add source syntax for negative literals.
- Execute literal and binding exits at 0, 42, 255, 256, 257, and `i32::MAX`.
  Include both existing nonempty examples, every integer base, leading zeroes,
  optional `i`, annotations, and trailing commas across representative fixtures.
- Execute declarations without an explicit exit, copied values preserved across
  later shadowing, and `exit(42); exit(7);`. Retain rejection and output
  preservation for unknown names and overflowing literals after an exit.
