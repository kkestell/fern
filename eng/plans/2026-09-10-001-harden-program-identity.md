# Harden program identity

## Sources

- `AGENTS.md#project-documents` — architecture owns durable boundaries,
  storage lifetimes, and identity rules
- `eng/roadmap.md#compiler-architecture-hardening` — task scope and completion
  gates
- `docs/spec.md` — Fern behavior remains unchanged

## Goal

Give program files and modules typed identities, keep each file's source,
syntax, and import resolutions together, and record the compiler's durable
boundaries in `eng/architecture.md`. This completes the architecture-hardening
milestone without changing Fern behavior.

## Implementation

- `src/module.rs`, `src/source.rs`, and `src/frontend/{parser,syntax}.rs` —
  define `FileId` and `ModuleId`; replace raw file and module indices at the
  loader, parser, and semantic boundary; make one program-owned file record
  carry its source, syntax, and resolved imports.
- `src/semantic/{model,namespaces}.rs` and `src/lib.rs` — consume the typed
  program identities and file records without changing namespace resolution or
  diagnostics.
- `eng/architecture.md` — establish the phase pipeline, ownership boundaries,
  source and node lifetimes, file and module identity, diagnostic rendering,
  and the IR verification gate. Remove durable phase-boundary decisions from
  the completed decomposition plan.
- `src/module.rs` and semantic module tests — retain and extend coverage for
  dependency ordering, file-local imports, and diagnostics across modules.

## Tests

- Run module and semantic namespace tests for multi-file and imported-module
  identity.
- Run the compiler integration suite, including the arrays fixture.
