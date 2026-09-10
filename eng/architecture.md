# Fern Compiler Architecture

## Pipeline

`src/lib.rs` owns the pipeline: module loading and parsing, semantic checking,
IR lowering and verification, then native compilation. Each boundary passes one
representation forward. The frontend owns parsed syntax, semantic owns checked
meaning, IR owns executable structure, and the backend owns QBE and native-tool
execution. Phase roots declare submodules only. Cross-phase callers name the
submodule that owns the operation.

`src/module.rs` owns filesystem discovery, import resolution, and the module
dependency graph. It sequences source loading with frontend parsing because
imports are syntax. Semantic namespaces consume the resulting program; they do
not discover files or resolve paths.

## Storage and identity

A loaded `Program` owns every source file in the root module and its
dependencies. `SourceMap` assigns each one a `FileId`; that same identity names
its parsed file syntax and its program-owned import record. The loader alone
creates these records, so imports cannot become a separately indexed store.
`ModuleId` identifies a loaded module and is the value imports carry. Module
load order is dependencies before dependents, with the root module last.

Syntax owns the program-wide name interner and arenas. Syntax node IDs remain
valid for the lifetime of the syntax and are used by semantic checking. A
`CheckedProgram` borrows that syntax and owns the checked bindings, expression
facts, signatures, and declaration relations. Its temporary namespace and
import state is keyed by `ModuleId` and `FileId`.

IR lowering consumes the checked program and creates one `ir::model::Program`.
IR IDs identify positions only within that program. `Program::verify` is the
sole constructor of `VerifiedProgram`; only verified IR reaches the backend.

## Diagnostics

Source spans use the program-wide offset space in `SourceMap`. A diagnostic
renders through that map at the outer compilation boundary. The backend also
receives the map only to embed source locations in runtime trap messages. No
later phase reads source files or reconstructs source locations.

## Testing

Unit tests sit beside the phase submodule that owns the behavior. Parser and
lowering snapshots sit with those tests. `tests/compiler.rs` owns the compiler
and CLI integration boundary.
