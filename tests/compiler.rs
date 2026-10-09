use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::{TempDir, tempdir};

const INTEGER: &str = "extern fn exit(status:i32)->void; fn main() -> void { const status: int = 42; var copy = status; exit(i32(copy)); }";

fn cli(input: &Path, output: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_fern"));
    command.arg(input).arg("-o").arg(output);
    command
}

fn fixture(source: impl AsRef<[u8]>) -> (TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.fern");
    let output = dir.path().join("output");
    fs::write(&input, source).unwrap();
    (dir, input, output)
}

fn failure(result: Output, expected: &str) {
    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(
        stderr.contains(expected),
        "expected {expected:?} in {stderr}"
    );
}

#[test]
fn documented_examples_compile() {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut inputs = fs::read_dir(&examples)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "fern")
        })
        .collect::<Vec<_>>();
    inputs.sort();

    assert!(!inputs.is_empty());
    // The modules tour is a directory of modules rather than a single file.
    inputs.push(examples.join("modules_and_imports").join("app"));
    let outputs = tempdir().unwrap();
    for input in inputs {
        let output = outputs.path().join(input.file_stem().unwrap());
        fern::compile(&input, &output)
            .unwrap_or_else(|error| panic!("{}: {error}", input.display()));
    }
}

#[test]
fn cli_compiles_and_replaces_existing_output() {
    let (_dir, input, output) = fixture(INTEGER);
    fs::write(&output, "old executable").unwrap();
    let result = cli(&input, &output).output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn deep_struct_containment_reports_a_diagnostic_without_aborting() {
    let mut source = String::new();
    for index in 0..=128 {
        if index == 128 {
            source.push_str(&format!("type S{index} struct {{ value: int }}\n"));
        } else {
            source.push_str(&format!(
                "type S{index} struct {{ next: S{} }}\n",
                index + 1
            ));
        }
    }
    source.push_str("fn main() -> void {}\n");
    let (_dir, input, output) = fixture(source);
    fs::write(&output, "keep me").unwrap();
    failure(
        cli(&input, &output).output().unwrap(),
        "struct containment exceeds compiler limit of 128",
    );
    assert_eq!(fs::read_to_string(output).unwrap(), "keep me");
}

#[test]
fn oversized_integer_diagnostics_are_bounded() {
    let digits = "9".repeat(200_000);
    let (_dir, input, output) = fixture(format!("fn main() -> void {{ const value = {digits}; }}"));
    let result = cli(&input, &output).output().unwrap();
    assert!(!result.status.success());
    assert!(
        result.stderr.len() < 2_000,
        "stderr was {} bytes",
        result.stderr.len()
    );
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("integer literal exceeds compiler limit"));
    assert!(stderr.contains("bytes "));
}

#[test]
fn file_errors_identify_input() {
    let (_dir, input, output) = fixture(b"// \xff");
    let error = fern::compile(&input, &output).unwrap_err().to_string();
    assert!(
        error.contains("input.fern: invalid UTF-8 at byte 3"),
        "{error}"
    );
    assert!(!output.exists());
    fs::remove_file(&input).unwrap();
    failure(cli(&input, &output).output().unwrap(), "cannot read");
    assert!(!output.exists());
}

#[test]
fn malformed_cli_arguments_fail() {
    for args in [
        vec![],
        vec!["input"],
        vec!["input", "-o"],
        vec!["input", "--output", "out"],
        vec!["input", "-o", "out", "extra"],
        vec!["-x", "-o", "out"],
        vec!["input", "-o", ""],
        vec!["input", "-o", "-o"],
        vec!["input", "-o", "out", "--link"],
        vec!["input", "-o", "out", "-L"],
        vec!["input", "-o", "out", "-l"],
        vec!["input", "-o", "out", "-l", ""],
        vec!["input", "-o", "out", "--link", "-l", "m"],
        vec!["input", "-o", "out", "--library", "m"],
        vec!["input", "-o", "out", "-lm"],
    ] {
        failure(
            Command::new(env!("CARGO_BIN_EXE_fern"))
                .args(args)
                .output()
                .unwrap(),
            "usage:",
        );
    }
}

#[test]
fn output_cannot_alias_input() {
    let (dir, input, _) = fixture(INTEGER);
    failure(cli(&input, &input).output().unwrap(), "overwrite the input");
    let alias = dir.path().join("alias");
    fs::hard_link(&input, &alias).unwrap();
    failure(cli(&input, &alias).output().unwrap(), "overwrite the input");
    #[cfg(unix)]
    {
        let symlink = dir.path().join("symlink");
        std::os::unix::fs::symlink(&input, &symlink).unwrap();
        failure(
            cli(&input, &symlink).output().unwrap(),
            "overwrite the input",
        );
    }
    assert_eq!(fs::read_to_string(input).unwrap(), INTEGER);
}

#[test]
fn missing_tools_preserve_outputs_and_remove_intermediates() {
    for tool in ["QBE", "CC"] {
        let (dir, input, output) = fixture(INTEGER);
        for existing in [false, true] {
            if existing {
                fs::write(&output, "keep me").unwrap();
            }
            failure(
                cli(&input, &output)
                    .env(tool, dir.path().join("missing-tool"))
                    .output()
                    .unwrap(),
                "cannot run",
            );
            if existing {
                assert_eq!(fs::read_to_string(&output).unwrap(), "keep me");
            } else {
                assert!(!output.exists());
            }
            assert_eq!(
                fs::read_dir(dir.path()).unwrap().count(),
                if existing { 2 } else { 1 }
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn failing_tools_report_stderr_and_preserve_output() {
    use std::os::unix::fs::PermissionsExt;
    for tool in ["QBE", "CC"] {
        let (dir, input, output) = fixture(INTEGER);
        let script = dir.path().join("failing-tool");
        fs::write(
            &script,
            "#!/bin/sh\necho deliberate-tool-error >&2\nexit 7\n",
        )
        .unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
        for existing in [false, true] {
            if existing {
                fs::write(&output, "keep me").unwrap();
            }
            let result = cli(&input, &output).env(tool, &script).output().unwrap();
            assert!(String::from_utf8_lossy(&result.stderr).contains("7"));
            failure(result, "deliberate-tool-error");
            if existing {
                assert_eq!(fs::read_to_string(&output).unwrap(), "keep me");
            } else {
                assert!(!output.exists());
            }
            assert_eq!(
                fs::read_dir(dir.path()).unwrap().count(),
                if existing { 3 } else { 2 }
            );
        }
    }
}

#[test]
fn a_missing_named_library_preserves_output_and_removes_intermediates() {
    let (dir, input, output) = fixture(INTEGER);
    fs::write(&output, "keep me").unwrap();
    failure(
        cli(&input, &output)
            .args(["-l", "fern-missing-library"])
            .output()
            .unwrap(),
        "fern-missing-library",
    );
    assert_eq!(fs::read_to_string(&output).unwrap(), "keep me");
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
}

#[test]
fn unusable_output_destination_fails() {
    let (dir, input, output) = fixture(INTEGER);
    fs::write(&output, "not a directory").unwrap();
    failure(
        cli(&input, &output.join("child")).output().unwrap(),
        "cannot create temporary files",
    );
    assert_eq!(fs::read_to_string(&output).unwrap(), "not a directory");
    let directory = dir.path().join("directory");
    fs::create_dir(&directory).unwrap();
    failure(cli(&input, &directory).output().unwrap(), "cannot publish");
    assert!(directory.is_dir());
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 3);
}

#[cfg(unix)]
#[test]
fn unwritable_output_directory_fails() {
    use std::os::unix::fs::PermissionsExt;
    let (dir, input, output) = fixture(INTEGER);
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();
    let result = cli(&input, &output).output();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
    failure(result.unwrap(), "cannot create temporary files");
    assert!(!output.exists());
}

#[test]
fn excessive_nesting_reports_an_error_and_preserves_output() {
    for body in [
        format!(
            "{}exit(i32(42));{}",
            "{".repeat(100_000),
            "}".repeat(100_000)
        ),
        format!(
            "exit(i32({}42{}));",
            "int(".repeat(100_000),
            ")".repeat(100_000)
        ),
    ] {
        let (dir, input, output) = fixture(format!("fn main() -> void {{ {body} }}"));
        fs::write(&output, "keep me").unwrap();
        let result = cli(&input, &output).output().unwrap();
        assert_eq!(result.status.code(), Some(1));
        failure(result, "source nesting exceeds compiler limit of 128");
        assert_eq!(fs::read_to_string(&output).unwrap(), "keep me");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    }
}

/// Writes a module directory holding `(file name, source)` pairs, returning the
/// module root and an output path outside it.
fn module(files: &[(&str, &str)]) -> (TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempdir().unwrap();
    let root = dir.path().join("app");
    fs::create_dir(&root).unwrap();
    for (name, source) in files {
        fs::write(root.join(name), source).unwrap();
    }
    fs::write(
        root.join("external.fern"),
        "extern fn exit(status:i32)->void;",
    )
    .unwrap();
    let output = dir.path().join("output");
    (dir, root, output)
}

#[test]
fn module_files_share_one_namespace_and_execute() {
    let (_dir, root, output) = module(&[
        (
            "main.fern",
            "const base: int = 20;\n\nfn main() -> void {\n    exit(i32(total()));\n}\n",
        ),
        (
            "totals.fern",
            "fn total() -> int {\n    return base + 22;\n}\n",
        ),
    ]);
    fern::compile(&root, &output).unwrap();
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(42));
}

#[test]
fn the_entry_point_is_found_in_any_file_of_the_root_module() {
    let (_dir, root, output) = module(&[
        (
            "helpers.fern",
            "fn twice(value: int) -> int { return value * 2; }",
        ),
        (
            "zzz_entry.fern",
            "fn main() -> void { exit(i32(twice(21))); }",
        ),
    ]);
    fern::compile(&root, &output).unwrap();
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(42));
}

#[test]
fn module_diagnostics_identify_the_file_they_point_into() {
    for (files, message, location) in [
        (
            [
                ("first.fern", "fn main() -> void {\n    exit(i32(0));\n}\n"),
                ("second.fern", "const value = 1;\nconst value = 2;\n"),
            ],
            "duplicate module-level name `value`",
            "second.fern:2:7",
        ),
        (
            [
                ("first.fern", "fn main() -> void {\n    exit(i32(0));\n}\n"),
                (
                    "second.fern",
                    "fn helper() -> void {\n    exit(i32(missing));\n}\n",
                ),
            ],
            "unknown binding `missing`",
            "second.fern:2:14",
        ),
        (
            [
                (
                    "first.fern",
                    "fn main() -> void {\n    exit(i32(broken));\n}\n",
                ),
                ("second.fern", "const other = 1;\n"),
            ],
            "unknown binding `broken`",
            "first.fern:2:14",
        ),
    ] {
        let (_dir, root, output) = module(&files);
        let error = fern::compile(&root, &output).unwrap_err().to_string();
        assert!(error.contains(message), "{error}");
        assert!(error.contains(location), "{error}");
        assert!(!output.exists());
    }
}

#[test]
fn entry_point_failures_span_the_files_of_a_module() {
    for (files, message) in [
        (
            [
                ("first.fern", "fn main() -> void {}"),
                ("second.fern", "fn main() -> void {}"),
            ],
            "duplicate `main` function",
        ),
        (
            [
                ("first.fern", "fn helper() -> void {}"),
                ("second.fern", "fn other() -> void {}"),
            ],
            "missing `main` function",
        ),
    ] {
        let (_dir, root, output) = module(&files);
        let error = fern::compile(&root, &output).unwrap_err().to_string();
        assert!(error.contains(message), "{error}");
        assert!(!output.exists());
    }
}

#[test]
fn a_module_directory_holds_its_own_fern_files_only() {
    let (dir, root, output) = module(&[
        ("main.fern", "fn main() -> void { exit(i32(42)); }"),
        ("notes.txt", "fn main() -> void { exit(i32(1)); }"),
    ]);
    let nested = root.join("nested");
    fs::create_dir(&nested).unwrap();
    fs::write(nested.join("broken.fern"), "this is not fern").unwrap();
    fern::compile(&root, &output).unwrap();
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(42));

    let empty = dir.path().join("empty");
    fs::create_dir(&empty).unwrap();
    let error = fern::compile(&empty, &output).unwrap_err().to_string();
    assert!(error.contains("holds no `.fern` source files"), "{error}");
    let error = fern::compile(&root.join("notes.txt"), &output)
        .unwrap_err()
        .to_string();
    assert!(error.contains("is not a `.fern` file"), "{error}");
    let error = fern::compile(&nested.join("missing"), &output)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cannot read"), "{error}");
}

#[test]
fn output_cannot_alias_a_module_source_file() {
    let (_dir, root, _) = module(&[
        ("main.fern", "fn main() -> void { exit(i32(42)); }"),
        ("helper.fern", "fn helper() -> void {}"),
    ]);
    for name in ["main.fern", "helper.fern"] {
        failure(
            cli(&root, &root.join(name)).output().unwrap(),
            "overwrite the input",
        );
    }
}

/// Writes `(relative path, source)` pairs under a fresh directory, creating
/// each file's parent directories.
fn tree(files: &[(&str, &str)]) -> TempDir {
    let dir = tempdir().unwrap();
    for (path, source) in files {
        let path = dir.path().join(path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, source).unwrap();
    }
    let mut modules = std::collections::HashSet::new();
    for (path, source) in files {
        if source.contains("exit(")
            && !source.contains("extern fn exit")
            && Path::new(path).extension().is_some_and(|e| e == "fern")
        {
            modules.insert(Path::new(path).parent().unwrap());
        }
    }
    for module in modules {
        fs::write(
            dir.path().join(module).join("external.fern"),
            "extern fn exit(status:i32)->void;",
        )
        .unwrap();
    }
    dir
}

const IMPORTING_MODULE: [(&str, &str); 4] = [
    (
        "app/main.fern",
        "use text;\nextern fn exit(status:i32)->void;\nfn main() -> void { exit(i32(text::width)); }\n",
    ),
    ("text/text.fern", "pub const width = 1;\n"),
    ("vendor/text/text.fern", "pub const width = 2;\n"),
    ("empty/notes.txt", "not a Fern source file\n"),
];

#[test]
fn imports_resolve_against_the_root_modules_parent_directory() {
    let dir = tree(&IMPORTING_MODULE);
    let output = dir.path().join("output");
    assert!(
        cli(&dir.path().join("app"), &output)
            .status()
            .unwrap()
            .success()
    );
    // The exit status names the root that resolved `text`.
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(1));
}

#[test]
fn fernpath_replaces_the_default_search_roots() {
    let dir = tree(&IMPORTING_MODULE);
    let output = dir.path().join("output");

    assert!(
        cli(&dir.path().join("app"), &output)
            .env("FERNPATH", dir.path().join("vendor"))
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(2));

    // `FERNPATH` replaces the default root rather than extending it, so the
    // module beside the root module is no longer found.
    fs::remove_file(&output).unwrap();
    let roots = [dir.path().join("empty"), dir.path().join("missing")];
    failure(
        cli(&dir.path().join("app"), &output)
            .env("FERNPATH", std::env::join_paths(&roots).unwrap())
            .output()
            .unwrap(),
        &format!(
            "unresolved import `text`, searched roots in order: {}, {}",
            roots[0].display(),
            roots[1].display()
        ),
    );
    assert!(!output.exists());
}

#[test]
fn a_relative_root_module_searches_the_directory_above_it() {
    let dir = tree(&[
        (
            "proj/main.fern",
            "use text;\nextern fn exit(status:i32)->void;\nfn main() -> void { exit(i32(text::width)); }\n",
        ),
        // Beside the root module, where the default search root looks.
        ("text/text.fern", "pub const width = 2;\n"),
        // Inside the root module, where it must not look.
        ("proj/text/text.fern", "pub const width = 1;\n"),
    ]);
    let output = dir.path().join("output");
    let absolute = dir.path().join("proj");

    // The root module directory is `proj` in each form, so a relative
    // argument names the same search root as an absolute one.
    for (working_directory, root) in [
        (absolute.clone(), Path::new("main.fern")),
        (absolute.clone(), Path::new(".")),
        (dir.path().to_owned(), Path::new("proj")),
        (dir.path().to_owned(), absolute.as_path()),
    ] {
        let result = cli(root, &output)
            .current_dir(&working_directory)
            .env_remove("FERNPATH")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{root:?}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            Command::new(&output).status().unwrap().code(),
            Some(2),
            "{root:?}"
        );
        fs::remove_file(&output).unwrap();
    }
}

#[test]
fn cross_module_calls_and_public_binding_mutation_execute() {
    let dir = tree(&[
        (
            "app/main.fern",
            "use counter;\nuse counter::{bump};\n\
             fn main() -> void {\n\
             counter::value = 20;\n\
             bump(22);\n\
             exit(i32(counter::value));\n\
             }\n",
        ),
        (
            "counter/counter.fern",
            "pub var value = 0;\n\
             pub fn bump(amount: int) -> int { value = value + amount; return value; }\n",
        ),
    ]);
    let output = dir.path().join("output");
    fern::compile(&dir.path().join("app"), &output).unwrap();
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(42));
}

#[test]
fn same_named_declarations_in_two_modules_execute_independently() {
    let declarations = "pub var value = 0;\npub fn total() -> int { return value; }\n";
    let dir = tree(&[
        (
            "app/main.fern",
            "use first;\nuse second;\n\
             fn main() -> void {\n\
             first::value = 40;\n\
             second::value = 2;\n\
             exit(i32(first::total() + second::total()));\n\
             }\n",
        ),
        ("first/first.fern", declarations),
        ("second/second.fern", declarations),
    ]);
    let output = dir.path().join("output");
    fern::compile(&dir.path().join("app"), &output).unwrap();
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(42));
}

#[test]
fn a_dependency_modules_main_does_not_run() {
    let dir = tree(&[
        (
            "app/main.fern",
            "use counter;\nfn main() -> void { exit(i32(counter::value)); }\n",
        ),
        (
            "counter/counter.fern",
            "pub var value = 42;\nfn main() -> void { value = 255; }\n",
        ),
    ]);
    let output = dir.path().join("output");
    fern::compile(&dir.path().join("app"), &output).unwrap();
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(42));
}

#[test]
fn every_modules_bindings_hold_their_initial_values_when_main_begins() {
    let dir = tree(&[
        (
            "app/main.fern",
            "use first;\nuse second;\nconst local = 6;\n\
             fn main() -> void { exit(i32(first::start + second::start + local)); }\n",
        ),
        ("first/first.fern", "pub const start = 30;\n"),
        (
            "second/second.fern",
            "use first;\npub var start = first::start / 5;\n",
        ),
    ]);
    let output = dir.path().join("output");
    fern::compile(&dir.path().join("app"), &output).unwrap();
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(42));
}

#[test]
fn the_modules_and_imports_fixture_executes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/programs/modules_and_imports")
        .join("app");
    let dir = tempdir().unwrap();
    let output = dir.path().join("output");
    fern::compile(&root, &output).unwrap();
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(42));
}
