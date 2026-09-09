//! Module discovery, import resolution, and the dependency graph.
//!
//! An import is known only after its importing file parses, so this module
//! sequences reading and parsing: `source` keeps the filesystem primitives and
//! `frontend` keeps parsing.

use crate::{
    CompileError,
    diagnostic::Diagnostic,
    frontend::{self, Syntax},
    source::{self, SourceMap},
};
use std::{
    collections::HashMap,
    env, fs, mem,
    ops::Range,
    path::{self, Path, PathBuf},
};

/// One module: the `.fern` files of a single directory. The directory itself is
/// the parent of each of those files, so `SourceMap` is its one home.
#[derive(Debug)]
pub(crate) struct Module {
    /// This module's files, as a contiguous range of `Syntax::files`.
    pub files: Range<usize>,
}

/// A loaded program: the root module and every module it imports, with
/// dependencies before dependents and the root module last.
#[derive(Debug)]
pub(crate) struct Program {
    pub sources: SourceMap,
    pub syntax: Syntax,
    pub modules: Vec<Module>,
    /// For each source file, the module each of its `use` declarations
    /// resolves to, by index into `modules`, in declaration order.
    pub imports: Vec<Vec<usize>>,
}

/// The whole of `syntax` as one root module with no imports.
#[cfg(test)]
pub(crate) fn single(syntax: &Syntax) -> (Vec<Module>, Vec<Vec<usize>>) {
    (
        vec![Module {
            files: 0..syntax.files.len(),
        }],
        vec![Vec::new(); syntax.files.len()],
    )
}

/// A loading failure. `Source` carries the partially loaded map its diagnostic
/// renders against.
#[derive(Debug)]
pub(crate) enum LoadError {
    Failed(CompileError),
    Source(Diagnostic, SourceMap),
}

impl LoadError {
    pub(crate) fn into_compile_error(self) -> CompileError {
        match self {
            Self::Failed(error) => error,
            Self::Source(diagnostic, sources) => CompileError::new(diagnostic.render(&sources)),
        }
    }
}

/// The ordered module search roots: `FERNPATH`'s entries when it holds any,
/// otherwise the root module directory's parent.
///
/// # Errors
/// Returns an error when the root module directory has no absolute form.
pub(crate) fn search_roots(root: &Path) -> Result<Vec<PathBuf>, CompileError> {
    if let Some(value) = env::var_os("FERNPATH") {
        let roots: Vec<PathBuf> = env::split_paths(&value)
            .filter(|root| !root.as_os_str().is_empty())
            .collect();
        if !roots.is_empty() {
            return Ok(roots);
        }
    }
    // A relative directory's parent is only visible in its absolute form: the
    // lexical parent of `app` is `""` and of `.` is `""` again, so both would
    // search the working directory instead of the directory above the module.
    let directory = root_directory(root);
    let absolute = path::absolute(directory)
        .map_err(|e| CompileError::new(format!("cannot resolve {}: {e}", directory.display())))?;
    Ok(absolute.parent().map(Path::to_owned).into_iter().collect())
}

/// Loads and parses the root module and, transitively, every module it imports.
pub(crate) fn load(root: &Path, roots: &[PathBuf]) -> Result<Program, LoadError> {
    let directory = root_directory(root).to_owned();
    let paths = if root.is_dir() {
        let paths = source::fern_files(&directory).map_err(LoadError::Failed)?;
        if paths.is_empty() {
            return Err(LoadError::Failed(CompileError::new(format!(
                "{}: module directory holds no `.fern` source files",
                directory.display()
            ))));
        }
        paths
    } else {
        vec![root.to_owned()]
    };

    let mut loader = Loader {
        roots,
        sources: SourceMap::default(),
        syntax: Syntax::default(),
        modules: Vec::new(),
        imports: Vec::new(),
        completed: HashMap::new(),
        stack: Vec::new(),
    };
    loader.open(directory, Vec::new(), paths, None)?;
    loader.run()?;
    Ok(Program {
        sources: loader.sources,
        syntax: loader.syntax,
        modules: loader.modules,
        imports: loader.imports,
    })
}

/// The root module's directory: a directory argument itself, or a file
/// argument's parent.
fn root_directory(root: &Path) -> &Path {
    if root.is_dir() {
        return root;
    }
    match root.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

fn label(path: &[String], directory: &Path) -> String {
    if path.is_empty() {
        directory.display().to_string()
    } else {
        path.join("::")
    }
}

/// One import path, taken from the syntax so the walk does not revisit it.
struct ImportPath {
    components: Vec<String>,
    span: Range<usize>,
    /// The file that declares this import, by index into `Syntax::files`.
    file: usize,
}

/// A module whose files are parsed and whose imports are still being loaded.
struct Frame {
    directory: PathBuf,
    /// `fs::canonicalize` of `directory`, the module's identity.
    canonical: PathBuf,
    path: Vec<String>,
    files: Range<usize>,
    imports: Vec<ImportPath>,
    cursor: usize,
    /// The file whose import opened this module, once this module completes.
    importer: Option<usize>,
}

/// The dependency walk. Modules are held on an explicit stack rather than the
/// compiler's call stack, so import depth is bounded only by memory.
struct Loader<'a> {
    roots: &'a [PathBuf],
    sources: SourceMap,
    syntax: Syntax,
    modules: Vec<Module>,
    /// One resolution list per file, filled as each import resolves.
    imports: Vec<Vec<usize>>,
    /// Loaded modules by canonical directory, so one directory reached through
    /// two roots or two paths is one module.
    completed: HashMap<PathBuf, usize>,
    stack: Vec<Frame>,
}

impl Loader<'_> {
    fn run(&mut self) -> Result<(), LoadError> {
        while !self.stack.is_empty() {
            let top = self.stack.len() - 1;
            let frame = &mut self.stack[top];
            let Some(import) = frame.imports.get(frame.cursor) else {
                self.complete_top();
                continue;
            };
            let (components, span, file) =
                (import.components.clone(), import.span.clone(), import.file);
            frame.cursor += 1;

            let Some((directory, paths)) = self.resolve(&components) else {
                return Err(self.unresolved(&components, span));
            };
            let canonical = canonicalize(&directory).map_err(LoadError::Failed)?;
            if let Some(&index) = self.completed.get(&canonical) {
                self.imports[file].push(index);
                continue;
            }
            if let Some(start) = self
                .stack
                .iter()
                .position(|frame| frame.canonical == canonical)
            {
                return Err(self.cycle(start, span));
            }
            self.open(directory, components, paths, Some(file))?;
        }
        Ok(())
    }

    /// Reads and parses every file of one module, then pushes its frame. All of
    /// a module's files load together, so its file range is contiguous.
    fn open(
        &mut self,
        directory: PathBuf,
        path: Vec<String>,
        paths: Vec<PathBuf>,
        importer: Option<usize>,
    ) -> Result<(), LoadError> {
        let canonical = canonicalize(&directory).map_err(LoadError::Failed)?;
        let start = self.syntax.files.len();
        for file in paths {
            let text = source::read_text(&file).map_err(LoadError::Failed)?;
            let index = self.sources.push(file, text);
            self.imports.push(Vec::new());
            let parsed = frontend::parse_file(&mut self.syntax, &self.sources.files()[index]);
            if let Err(diagnostic) = parsed {
                return Err(self.source_error(diagnostic));
            }
        }
        let files = start..self.syntax.files.len();
        let imports = self.import_paths(&files);
        self.stack.push(Frame {
            directory,
            canonical,
            path,
            files,
            imports,
            cursor: 0,
            importer,
        });
        Ok(())
    }

    /// The import paths of one module's files, in file and declaration order.
    fn import_paths(&self, files: &Range<usize>) -> Vec<ImportPath> {
        files
            .clone()
            .flat_map(|file| {
                self.syntax.files[file].imports.iter().map(move |import| {
                    let last = import.path.len() - 1;
                    ImportPath {
                        components: import
                            .path
                            .iter()
                            .map(|component| self.syntax.names.resolve(&component.name).to_owned())
                            .collect(),
                        span: import.path[0].name_span.start..import.path[last].name_span.end,
                        file,
                    }
                })
            })
            .collect()
    }

    /// The first root whose directory for `components` holds `.fern` files,
    /// with that module's file paths. A root that cannot be read is skipped.
    fn resolve(&self, components: &[String]) -> Option<(PathBuf, Vec<PathBuf>)> {
        self.roots.iter().find_map(|root| {
            let mut directory = root.clone();
            directory.extend(components);
            let paths = source::fern_files(&directory).ok()?;
            (!paths.is_empty()).then_some((directory, paths))
        })
    }

    /// Pops the finished module onto `modules`, resolving the import that
    /// opened it.
    fn complete_top(&mut self) {
        let frame = self.stack.pop().expect("the caller checked the stack top");
        let index = self.modules.len();
        self.completed.insert(frame.canonical, index);
        if let Some(file) = frame.importer {
            self.imports[file].push(index);
        }
        self.modules.push(Module { files: frame.files });
    }

    fn unresolved(&mut self, components: &[String], span: Range<usize>) -> LoadError {
        let searched: Vec<String> = self
            .roots
            .iter()
            .map(|root| root.display().to_string())
            .collect();
        let message = format!(
            "unresolved import `{}`, searched roots in order: {}",
            components.join("::"),
            searched.join(", ")
        );
        self.source_error(Diagnostic::new(span, message))
    }

    /// The cycle as a chain of module labels, from the module the closing
    /// import reaches back around to itself.
    fn cycle(&mut self, start: usize, span: Range<usize>) -> LoadError {
        let mut chain: Vec<String> = self.stack[start..]
            .iter()
            .map(|frame| label(&frame.path, &frame.directory))
            .collect();
        chain.push(chain[0].clone());
        let message = format!("module dependency cycle: {}", chain.join(" -> "));
        self.source_error(Diagnostic::new(span, message))
    }

    fn source_error(&mut self, diagnostic: Diagnostic) -> LoadError {
        LoadError::Source(diagnostic, mem::take(&mut self.sources))
    }
}

fn canonicalize(directory: &Path) -> Result<PathBuf, CompileError> {
    fs::canonicalize(directory)
        .map_err(|e| CompileError::new(format!("cannot resolve {}: {e}", directory.display())))
}

/// Writes `(relative path, source)` pairs under a fresh directory, creating
/// each file's parent directories.
#[cfg(test)]
pub(crate) fn tree<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for (path, source) in files {
        let path = dir.path().join(path);
        fs::create_dir_all(path.parent().expect("a file has a parent")).unwrap();
        fs::write(path, source).unwrap();
    }
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each module's directory, in load order, taken from its first file.
    fn directories(program: &Program) -> Vec<PathBuf> {
        program
            .modules
            .iter()
            .map(|module| {
                program.sources.files()[module.files.start]
                    .path
                    .parent()
                    .expect("a source file sits in its module's directory")
                    .to_owned()
            })
            .collect()
    }

    fn diagnostic(error: LoadError) -> (Diagnostic, SourceMap) {
        match error {
            LoadError::Source(diagnostic, sources) => (diagnostic, sources),
            LoadError::Failed(error) => panic!("expected a source diagnostic, got {error}"),
        }
    }

    #[test]
    fn an_import_loads_the_module_its_path_names() {
        let dir = tree([
            ("app/main.fern", "use text::format;\nfn main() -> void {}\n"),
            (
                "text/format/format.fern",
                "pub fn doubled(value: int) -> int { return value * 2; }\n",
            ),
        ]);
        let program = load(&dir.path().join("app"), &[dir.path().to_owned()]).unwrap();

        assert_eq!(
            directories(&program),
            [dir.path().join("text/format"), dir.path().join("app")]
        );
        assert_eq!(program.modules[0].files, 1..2);
        assert_eq!(program.syntax.files.len(), 2);
        let paths: Vec<_> = program
            .sources
            .files()
            .iter()
            .map(|file| file.path.clone())
            .collect();
        assert_eq!(
            paths,
            [
                dir.path().join("app/main.fern"),
                dir.path().join("text/format/format.fern"),
            ]
        );
    }

    #[test]
    fn a_single_file_root_module_resolves_its_own_imports() {
        let dir = tree([
            ("app/main.fern", "use text;\nfn main() -> void {}\n"),
            ("app/other.fern", "fn ignored() -> void {}\n"),
            ("text/text.fern", "pub const width = 1;\n"),
        ]);
        let program = load(&dir.path().join("app/main.fern"), &[dir.path().to_owned()]).unwrap();

        // The file argument's own module is that one file, not its directory.
        assert_eq!(program.syntax.files.len(), 2);
        assert_eq!(
            directories(&program),
            [dir.path().join("text"), dir.path().join("app")]
        );
    }

    #[test]
    fn a_shared_dependency_loads_once_before_its_dependents() {
        let dir = tree([
            (
                "app/main.fern",
                "use left;\nuse right;\nfn main() -> void {}\n",
            ),
            ("left/left.fern", "use shared;\npub const a = 1;\n"),
            ("right/right.fern", "use shared;\npub const b = 2;\n"),
            ("shared/shared.fern", "pub const base = 3;\n"),
        ]);
        let program = load(&dir.path().join("app"), &[dir.path().to_owned()]).unwrap();

        assert_eq!(
            directories(&program),
            [
                dir.path().join("shared"),
                dir.path().join("left"),
                dir.path().join("right"),
                dir.path().join("app"),
            ]
        );
        // The shared module's single file is loaded once.
        assert_eq!(program.syntax.files.len(), 4);
    }

    #[test]
    fn the_first_root_holding_the_module_wins() {
        let dir = tree([
            ("app/main.fern", "use text;\nfn main() -> void {}\n"),
            ("first/text/text.fern", "pub const which = 1;\n"),
            ("second/text/text.fern", "pub const which = 2;\n"),
        ]);
        let program = load(
            &dir.path().join("app"),
            &[dir.path().join("first"), dir.path().join("second")],
        )
        .unwrap();

        assert_eq!(directories(&program)[0], dir.path().join("first/text"));
    }

    #[test]
    fn a_root_without_the_module_is_skipped() {
        let dir = tree([
            ("app/main.fern", "use text;\nfn main() -> void {}\n"),
            ("without/text/notes.txt", "not a Fern source file\n"),
            ("with/text/text.fern", "pub const which = 2;\n"),
        ]);
        let program = load(
            &dir.path().join("app"),
            &[
                dir.path().join("missing"),
                dir.path().join("without"),
                dir.path().join("with"),
            ],
        )
        .unwrap();

        assert_eq!(directories(&program)[0], dir.path().join("with/text"));
    }

    #[test]
    fn an_unresolved_import_lists_every_searched_root_in_order() {
        let dir = tree([("app/main.fern", "use text::format;\nfn main() -> void {}\n")]);
        let roots = [dir.path().join("a"), dir.path().join("b")];
        let (diagnostic, sources) = diagnostic(load(&dir.path().join("app"), &roots).unwrap_err());

        assert_eq!(diagnostic.span, 4..16);
        assert_eq!(
            diagnostic.message,
            format!(
                "unresolved import `text::format`, searched roots in order: {}, {}",
                roots[0].display(),
                roots[1].display()
            )
        );
        let rendered = diagnostic.render(&sources);
        assert!(rendered.contains("main.fern:1:5"), "{rendered}");
    }

    #[test]
    fn a_dependency_cycle_reports_its_chain_of_modules() {
        for (files, chain) in [
            (
                vec![
                    ("app/main.fern", "use counter;\nfn main() -> void {}\n"),
                    ("counter/counter.fern", "use text;\npub const a = 1;\n"),
                    ("text/text.fern", "use counter;\npub const b = 2;\n"),
                ],
                "counter -> text -> counter".to_owned(),
            ),
            (
                vec![
                    ("app/main.fern", "use a;\nfn main() -> void {}\n"),
                    ("a/a.fern", "use b;\npub const a = 1;\n"),
                    ("b/b.fern", "use c;\npub const b = 2;\n"),
                    ("c/c.fern", "use a;\npub const c = 3;\n"),
                ],
                "a -> b -> c -> a".to_owned(),
            ),
        ] {
            let dir = tree(files);
            let (diagnostic, _) =
                diagnostic(load(&dir.path().join("app"), &[dir.path().to_owned()]).unwrap_err());
            assert_eq!(
                diagnostic.message,
                format!("module dependency cycle: {chain}")
            );
        }
    }

    #[test]
    fn importing_the_root_module_is_a_cycle() {
        let dir = tree([
            ("app/main.fern", "use counter;\nfn main() -> void {}\n"),
            ("counter/counter.fern", "use app;\npub const a = 1;\n"),
        ]);
        let (diagnostic, _) =
            diagnostic(load(&dir.path().join("app"), &[dir.path().to_owned()]).unwrap_err());

        let root = dir.path().join("app").display().to_string();
        assert_eq!(
            diagnostic.message,
            format!("module dependency cycle: {root} -> counter -> {root}")
        );
    }

    #[test]
    fn a_long_import_chain_loads_without_recursing() {
        const DEPTH: usize = 400;
        let mut files = vec![(
            "app/main.fern".to_owned(),
            "use m0;\nfn main() -> void {}\n".to_owned(),
        )];
        for index in 0..DEPTH {
            let source = match index + 1 {
                next if next < DEPTH => format!("use m{next};\npub const step = 1;\n"),
                _ => "pub const step = 1;\n".to_owned(),
            };
            files.push((format!("m{index}/m{index}.fern"), source));
        }
        let dir = tree(
            files
                .iter()
                .map(|(path, source)| (path.as_str(), source.as_str())),
        );
        let program = load(&dir.path().join("app"), &[dir.path().to_owned()]).unwrap();

        // The deepest module has no dependency and completes first, and the
        // root module completes last.
        let chain: Vec<PathBuf> = (0..DEPTH)
            .rev()
            .map(|index| dir.path().join(format!("m{index}")))
            .collect();
        let directories = directories(&program);
        assert_eq!(directories[..DEPTH], chain[..]);
        assert_eq!(directories[DEPTH], dir.path().join("app"));
    }

    #[test]
    fn a_parse_error_in_a_dependency_renders_that_dependency() {
        let dir = tree([
            ("app/main.fern", "use text;\nfn main() -> void {}\n"),
            ("text/broken.fern", "fn missing_body(\n"),
        ]);
        let (diagnostic, sources) =
            diagnostic(load(&dir.path().join("app"), &[dir.path().to_owned()]).unwrap_err());

        let rendered = diagnostic.render(&sources);
        assert!(rendered.contains("broken.fern:1:"), "{rendered}");
        assert!(!rendered.contains("main.fern"), "{rendered}");
    }

    #[test]
    fn a_root_module_directory_without_fern_files_is_rejected() {
        let dir = tree([("app/notes.txt", "not a Fern source file\n")]);
        let error = load(&dir.path().join("app"), &[dir.path().to_owned()]).unwrap_err();

        assert!(
            error
                .into_compile_error()
                .to_string()
                .contains("holds no `.fern` source files")
        );
    }
}
