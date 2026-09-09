# Resolve imports to modules

## Sources

- `docs/spec.md#module-resolution-and-dependencies` — search-root order, the
  unresolved-import diagnostic, and the acyclic dependency graph
- `docs/spec.md#module-directories` — a module is one directory's `.fern` files
- `eng/roadmap.md#modules-and-imports` — third task and its boundary
- `src/source.rs`, `src/frontend.rs`, `src/lib.rs` — source loading, parsing,
  and the driver

## Goal

Load and parse the root module plus every module it imports, transitively,
rejecting unresolved import paths and dependency cycles. Starts from the
completed multi-file task. Cross-module name resolution, visibility, lowering,
and code generation remain unfinished: `semantic::check` still rejects any
`use` declaration, so a program that loads successfully still fails to compile.
Because a dependency is reachable only through a `use`, `check` never sees more
than one module's items and needs no change.

## Implementation

- `src/source.rs` — make the map appendable so files load as they are
  discovered.
  - Replace `from_texts` with `push(&mut self, path: PathBuf, text: String) ->
    usize`, assigning the new file's base from the previous file's base plus
    its length plus one and returning the file's index. `from_text` and
    `from_named_texts` build on `push`, so existing snapshot spans do not move.
  - Delete `SourceMap::load`. Make `read_text` `pub(crate)`, and rename
    `module_paths` to `pub(crate) fn fern_files(directory) -> Result<Vec<PathBuf>,
    CompileError>` returning an empty vector instead of an error when the
    directory holds no `.fern` file, keeping the sort and the read-failure
    message.

- `src/frontend.rs` — parse one file at a time into a shared `Syntax`.
  - Change `Parser.syntax` to `&'a mut Syntax`; the 29 `self.syntax` uses are
    unchanged.
  - Replace `parse` with `pub(crate) fn parse_file(syntax: &mut Syntax, source:
    &Source) -> Result<(), Diagnostic>`, which runs the existing parser over
    one file and pushes its `FileSyntax` onto `syntax.files`.
  - Keep a `#[cfg(test)] pub(crate) fn parse(sources: &SourceMap) ->
    Result<Syntax, Diagnostic>` looping `parse_file` over every file, so the
    test call sites in `frontend`, `semantic`, `ir`, and `backend` are
    unchanged.
  - Remove the `expect(dead_code)` attributes on `PathComponent::name` and
    `Import::path`; the loader reads both. `Import::selection` keeps its
    attribute.

- `src/module.rs` (new) — module discovery, import resolution, and the graph.
  - `Module { directory: PathBuf, path: Vec<String>, files: Range<usize>,
    dependencies: Vec<usize> }`, where `path` is the import path's components
    and is empty for the root module. Add `fn label(&self) -> String`: the
    components joined by `::`, or `directory.display()` for the root. Give
    `path`, `dependencies`, and `label` the repository's
    `#[cfg_attr(not(test), expect(dead_code, reason = ...))]` treatment where
    this task has no non-test reader.
  - `Program { sources: SourceMap, syntax: Syntax, modules: Vec<Module> }`.
  - `enum LoadError { Failed(CompileError), Source(Diagnostic, SourceMap) }`
    plus `fn into_compile_error(self) -> CompileError`, which renders the
    diagnostic against the partially loaded map. Unit tests match `Source` to
    assert the span and message.
  - `pub(crate) fn search_roots(root: &Path) -> Vec<PathBuf>`: the entries of
    `FERNPATH` in order via `std::env::split_paths`, ignoring empty entries;
    otherwise the root module directory's parent. Treat an unset or empty
    `FERNPATH` the same.
  - `pub(crate) fn load(root: &Path, roots: &[PathBuf]) -> Result<Program,
    LoadError>`: resolve the root module's directory (a directory argument, or
    a file argument's parent) and its file list (a directory argument's
    `fern_files`, rejecting an empty result with today's message; a file
    argument's single path), then walk the graph with an explicit frame stack,
    never recursing on the compiler stack:
    - A frame holds the module's directory, its file range, its imports' paths
      and spans, a cursor into those imports, and the dependencies collected so
      far. Push a frame only after reading and parsing all of that module's
      files, so each module's file range is contiguous.
    - For the frame's next import, resolve its path, advance the cursor, then:
      a directory already on the stack is a cycle; a completed module records
      its index as a dependency; otherwise load and parse its files and push a
      new frame.
    - When a frame runs out of imports, pop it, append it to
      `Program.modules`, record it in the completed map, and append its index
      to the new top frame's dependencies. Modules therefore land in dependency
      order with the root last, and every dependency index is lower than its
      dependent's.
    - Key both the completed map and the on-stack set by `fs::canonicalize` of
      the module directory, so one module loads once through different roots or
      paths.
  - Resolution appends the import's components as directories under each root
    in order and takes the first whose `fern_files` is non-empty. A root that
    does not exist is skipped rather than reported.
  - Unresolved import: a `Diagnostic` spanning the path components, from the
    first component's `name_span.start` to the last's `name_span.end`, reading
    ``unresolved import `text::format`, searched roots in order: /a/b, /c``.
  - Cycle: a `Diagnostic` on the closing import's path span reading `module
    dependency cycle: counter -> text::format -> counter`, listing the frames
    from the cycle's start through the stack top and then repeating the start's
    label.

- `src/lib.rs` — `compile` calls `module::search_roots` and `module::load`,
  mapping the failure with `into_compile_error`, then passes
  `program.sources` and `program.syntax` to the existing phases. Add `mod
  module;`.

- `README.md` — document that imports resolve against ordered search roots,
  that the default single root is the root module directory's parent, and that
  `FERNPATH` replaces that default with its platform-separated entries.

- `AGENTS.md` — add `src/module.rs` to the codebase map.

## Tests

Unit tests in `src/module.rs` build module trees under `tempfile::tempdir` and
pass explicit roots, so no test reads the process environment.

- A root module importing a module two components deep loads both, and the
  dependency's files appear in the map and in `Syntax.files`.
- A diamond (two modules importing the same third) loads the shared module
  once, and `Program.modules` lists dependencies before dependents with the
  root last.
- An import resolves through the first root holding the module when a later
  root also holds one.
- A directory that exists but holds no `.fern` file is not a match, and a
  nonexistent root is skipped.
- An unresolved import reports the path span and every searched root in order.
- A two-module cycle, an indirect three-module cycle, and a module importing
  the root module's own directory each report the chain.
- A chain of several hundred modules loads without overflowing the stack.
- A parse error in a dependency renders that dependency's path.

CLI tests in `tests/compiler.rs`, which can set the subprocess environment:

- `FERNPATH` replaces the default root, and an import that the default root
  would have found fails when `FERNPATH` points elsewhere.
- A program with a resolvable import still fails with ``use` declarations are
  not supported yet`, and existing single-file compilations stay green.

## Decisions

- The loader owns both file reading and per-file parsing because an import is
  known only after its importing file parses. `source.rs` keeps the filesystem
  primitives and `frontend.rs` keeps parsing; `module.rs` sequences them.

- Modules are identified by canonical directory rather than import path. The
  same directory reached through two roots or two paths is one module, and the
  root module participates, so a dependency that imports the root module's
  directory is a cycle instead of a second copy. For a single-file root module
  that directory holds files outside the module; the root module's own file
  list wins.
