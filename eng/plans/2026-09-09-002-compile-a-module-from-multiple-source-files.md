# Compile a module from multiple source files

## Sources

- `docs/spec.md#module-directories` — a module is every `.fern` file in one
  directory, sharing one namespace
- `docs/spec.md#functions` — entry-point selection
- `eng/roadmap.md#modules-and-imports` — second task and its boundary
- `src/source.rs`, `src/diagnostic.rs`, `src/lib.rs` — source loading,
  rendering, and the driver

## Goal

Compile the root module's whole directory as one `Syntax`, with every
diagnostic naming the file it points into. Starts from the completed parser
task, whose `use` and qualified-name guards in `semantic::check` stay in place.
Import resolution and cross-module compilation remain unfinished; this task
still compiles exactly one module.

## Implementation

- `src/source.rs` — add `SourceMap { files: Vec<Source> }`, where `Source`
  gains `base: usize`, its start in a single offset space shared by every file.
  Bases are assigned in load order as `base + text.len() + 1`; the one-byte gap
  keeps an end-of-file span from colliding with the next file's start.
  - `SourceMap::load(path)` — a directory loads every entry whose extension is
    `fern`, sorted by file name, skipping subdirectories, and fails when the
    directory holds none; a file loads that one file. Report a directory read
    failure with the same shape as the existing read error.
  - `file_at(offset) -> &Source` by `partition_point` over bases, taking the
    last file whose base is `<= offset`; `local(span) -> Range<usize>`
    subtracting that file's base.
  - `#[cfg(test)] fn from_text(text: &str) -> Self` — one file named
    `test.fern` at base 0, so existing snapshot spans do not move.

- `src/diagnostic.rs` — `DiagnosticRenderer::new(&SourceMap)` builds one
  `(path, ariadne::Source)` pair per file up front. Both render paths pick the
  file with `file_at(diagnostic.span.start)` and work in that file's local
  offsets. A span that crosses the gap between two files cannot occur; clamp the
  local end to the file's length rather than adding a check for it. Update the
  two renderer unit tests to build a `SourceMap`.

- `src/frontend.rs` — `parse(sources: &SourceMap) -> Result<Syntax, Diagnostic>`
  loops over the files in order, running the existing parser once per file with
  a new `Parser::base` field. Add `self.base` to both branches of the span
  assignment in `advance`; that is the only place spans originate.
  - Replace `Syntax::items` and `Syntax::imports` with
    `files: Vec<FileSyntax { imports: Vec<Import>, items: Vec<TopLevelItem> }>`,
    one entry per source file in load order, and add
    `Syntax::items(&self) -> impl Iterator<Item = &TopLevelItem>` flattening
    them. Imports are file-local, so grouping is what the visibility task needs.
  - In the test module, add `fn parse(text: &str) -> Result<Syntax, Diagnostic>`
    wrapping `SourceMap::from_text` and `super::parse`, so the existing test
    call sites are unchanged. Do the same in the `semantic`, `ir`, and
    `backend` test modules, and replace the hand-built `Source` values in
    `src/diagnostic.rs` and `src/backend.rs` tests.
  - The snapshot projection prints each file's path header, then its imports and
    items. Keep a single file's output byte-identical to today's so existing
    snapshots do not move.

- `src/semantic.rs` — the three `for item in &syntax.items` loops become
  `syntax.items()`. Duplicate module-level names and the sole-`main` entry point
  already cover the whole item list, so they now span files unchanged.

- `src/lib.rs` — `compile` loads a `SourceMap` and threads it through `parse`,
  `check`, and `backend::build`. `reject_input_output_alias` compares the output
  against every file in the map, not the input path.

- `src/backend.rs` — `emit` and `build` take `&SourceMap` in place of `&Source`.

- `src/main.rs` — usage becomes `fern <root> -o <output>`.

- `README.md` — document that the root module argument is a directory whose
  `.fern` files form one module, or a single `.fern` file as a one-file module.

## Tests

- Two files in one directory share a namespace: a private binding declared in
  one is read by a function in the other, and the program executes.
- A diagnostic in the second file names that file and points at the right line
  and column; a diagnostic in the first file is unchanged.
- A duplicate module-level name across two files is rejected, and the
  diagnostic points into the file holding the second declaration.
- `main` in either file is found; two `main` declarations across files and no
  `main` at all are each rejected.
- A directory holding no `.fern` file is rejected; a subdirectory's `.fern`
  files are not compiled.
- A single-file root module still compiles, keeping the existing CLI tests
  green.
- A runtime trap in the second file renders that file's path in its message.

## Decisions

- One shared offset space instead of a file identifier inside every span. Spans
  are `Range<usize>` in roughly a hundred places across four phases; a file
  identifier would touch all of them for a fact the `SourceMap` can already
  answer from an offset. Diagnostics already need the sources to render, so
  nothing that reads a span loses information.

- Load files sorted by file name. Directory order is not stable across
  platforms, and declaration order reaches diagnostics and snapshots.
