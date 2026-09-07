use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::{TempDir, tempdir};

const EMPTY: &str = "fn main() -> void {}";
const INTEGER: &str =
    "fn main() -> void { const status: int = 42; var copy = status; exit(copy,); }";

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
fn native_programs_exit_zero() {
    for source in [
        include_str!("../examples/empty.fern"),
        EMPTY,
        "fn main() -> void { const x = 42; var y = x; }",
        "\t\r\n\x0b\x0c /* 🌿 /* nested */ */ fn/*a*/main( ) -> void { // body\n } // eof",
    ] {
        let (dir, input, output) = fixture(source);
        fern::compile(&input, &output).unwrap();
        assert_eq!(Command::new(&output).status().unwrap().code(), Some(0));
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
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
fn typed_values_after_exit_do_not_change_status() {
    for body in [
        "exit(42); const value = 18446744073709551615u64;",
        "{ exit(42); const value = 1i8; } const value: u64 = 1u8;",
    ] {
        let (_dir, input, output) = fixture(format!("fn main() -> void {{ {body} }}"));
        fern::compile(&input, &output).unwrap();
        assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
    }
}

#[test]
fn typed_integer_diagnostics_precede_emission_and_preserve_output() {
    for (body, message) in [
        (
            "const x: u64 = 256u8;",
            "integer literal out of range for `u8`",
        ),
        (
            "var x: u8 = 0; { x = 256; }",
            "integer literal out of range for `u8`",
        ),
        (
            "const x = 1u64; { exit(0); exit(x); }",
            "cannot implicitly convert `u64` to `int`",
        ),
        (
            "const x = 1i64; const y: int = x;",
            "cannot implicitly convert `i64` to `int`",
        ),
        (
            "{ exit(0); } const x = 18446744073709551616u64;",
            "integer literal out of range for `u64`",
        ),
    ] {
        let (_dir, input, output) = fixture(format!("fn main() -> void {{ {body} }}"));
        fs::write(&output, "old executable").unwrap();
        let error = fern::compile(&input, &output).unwrap_err().to_string();
        assert!(error.contains(message), "{error}");
        assert!(
            error.contains(input.file_name().unwrap().to_str().unwrap()),
            "{error}"
        );
        assert_eq!(fs::read_to_string(output).unwrap(), "old executable");
    }
}

#[test]
fn source_failures_preserve_output() {
    for (source, expected) in [
        ("", "missing `main`"),
        (
            "fn main() -> void {} fn main() -> void {}",
            "duplicate `main`",
        ),
        ("fn other() -> void {}", "only the `main`"),
        (
            "fn main() -> void {} fn other() -> void {}",
            "only the `main`",
        ),
        ("fn main() -> void { const x = x; }", "unknown binding `x`"),
        (
            "fn main() -> void { exit(0); exit(missing); }",
            "unknown binding `missing`",
        ),
        (
            "fn main() -> void { exit(0); var x = 2147483648; }",
            "integer literal out of range for `int`",
        ),
        (
            "fn main() -> void { const x = 1; x = 2; }",
            "cannot assign to immutable binding `x`",
        ),
        (
            "fn main() -> void { { var x = 1; } x = 2; }",
            "unknown binding `x`",
        ),
        (
            "fn main() -> void { var x = 1; { exit(0); x = missing; } }",
            "unknown binding `missing`",
        ),
        (
            "fn main() -> void { var x = 1; { exit(0); } x = 2147483648; }",
            "integer literal out of range for `int`",
        ),
        (
            "fn main() -> void { var x = 1; { x = 1u8; } }",
            "cannot implicitly convert `u8` to `int`",
        ),
        ("fn main() -> void { x = ; }", "expected a name"),
        ("fn main() -> void { x 1; }", "expected `=`"),
        ("fn main() -> void { x = 1 }", "expected `;`"),
        ("fn main() -> void { { }", "expected `}`"),
        ("fn main(x) -> void {}", "parameters are not supported"),
        ("fn main() -> int {}", "expected `void`"),
        ("fn main() -> void {} trailing", "expected `fn`"),
        (
            "fn main() -> void {} /* unclosed",
            "unterminated block comment",
        ),
        ("fn main() -> void {} ;", "expected `fn`"),
        ("fn main() -> void {", "expected `}`"),
        ("fn main() -> void {}\u{a0}", "invalid token"),
        ("fn int() -> void {}", "reserved word"),
        ("fn main_extra() -> void {}", "only the `main`"),
    ] {
        let (dir, input, output) = fixture(source);
        failure(cli(&input, &output).output().unwrap(), expected);
        assert!(!output.exists());
        fs::write(&output, "keep me").unwrap();
        let error = fern::compile(&input, &output).unwrap_err().to_string();
        assert!(error.contains(expected), "{error}");
        assert!(
            error.contains(input.file_name().unwrap().to_str().unwrap()),
            "{error}"
        );
        assert_eq!(fs::read_to_string(&output).unwrap(), "keep me");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    }
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
fn integer_programs_execute() {
    let mut fixtures = vec![
        (
            include_str!("../examples/integer_literals.fern").to_owned(),
            42,
        ),
        (include_str!("../examples/shadowing.fern").to_owned(), 42),
        ("fn main() -> void { exit(42); exit(7); }".to_owned(), 42),
        (
            "fn main() -> void { const x = 42; exit(x); var x = 7; exit(x); }".to_owned(),
            42,
        ),
    ];
    for value in [0, 42, 255, 256, 257, i32::MAX] {
        for body in [
            format!("exit({value});"),
            format!("const x: int = {value}i; var y = x; exit(y,);"),
        ] {
            fixtures.push((format!("fn main() -> void {{ {body} }}"), value % 256));
        }
    }
    for literal in ["00042", "0x0002Ai", "0o00052", "0b000101010i"] {
        fixtures.push((
            format!("fn main() -> void {{ var x: int = {literal}; exit(x,); }}"),
            42,
        ));
    }
    for (source, expected) in fixtures {
        let (dir, input, output) = fixture(&source);
        fern::compile(&input, &output).unwrap();
        assert_eq!(
            Command::new(&output).status().unwrap().code(),
            Some(expected),
            "{source}"
        );
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    }
}

#[test]
fn assignments_and_nested_scopes_execute() {
    for (body, expected) in [
        ("var x = 1; x = 2; const y = 42; x = y; x = x; exit(x);", 42),
        ("var x = 42; const saved = x; x = 7; exit(saved);", 42),
        ("var x = 1; { x = 42; var x = 7; x = 8; } exit(x);", 42),
        ("var x = 1; { x = 42; const x = x; exit(x); }", 42),
        ("const x = 1; var x = x; x = 42; exit(x);", 42),
        ("var x = 1; const x = 42; exit(x);", 42),
        ("const x = 42; { var x = x; x = 7; } exit(x);", 42),
        ("var x = 42; { const x = 7; } x = x; exit(x);", 42),
        (
            "var x = 1; { { x = 42; exit(x); x = 7; } x = 8; } x = 9; exit(x);",
            42,
        ),
        ("{} { {} { {} } }", 0),
        ("var x = 1; { { x = 42; } } exit(x);", 42),
        ("var x = 1; { var y = x; { x = y; } } x = 42;", 0),
    ] {
        let source = format!("fn main() -> void {{ {body} }}");
        let (_dir, input, output) = fixture(&source);
        fern::compile(&input, &output).unwrap();
        assert_eq!(
            Command::new(output).status().unwrap().code(),
            Some(expected),
            "{body}"
        );
    }
    let (_dir, input, output) = fixture(include_str!("../examples/assignment_and_scopes.fern"));
    fern::compile(&input, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn typed_integer_programs_execute() {
    let mut fixtures = vec![(
        include_str!("../examples/integer_types.fern").to_owned(),
        42,
    )];
    for (body, expected) in [
        (
            "var x = 42i8; const saved = x; x = 7; var status: int = x; status = saved; exit(status);",
            42,
        ),
        (
            "var x = 1i8; { x = 42; var x: i64 = x; x = 7i16; } exit(x);",
            42,
        ),
        ("var x = 1i8; { { x = 42; exit(x); x = 7; } } exit(0);", 42),
        (
            "const x = 42; const y: i32 = x; const z: int = y; exit(z);",
            42,
        ),
        (
            "const x = 4294967295u; const y: u32 = x; var z: uint = y; z = 255u8;",
            0,
        ),
    ] {
        fixtures.push((format!("fn main() -> void {{ {body} }}"), expected));
    }
    for (suffix, max) in [
        ("i8", 127),
        ("i16", 32767),
        ("i32", i32::MAX),
        ("i", i32::MAX),
    ] {
        for value in [0, 42, max] {
            for body in [
                format!("exit({value}{suffix});"),
                format!("const x = {value}{suffix}; var y: int = 0; y = x; exit(y);"),
            ] {
                fixtures.push((format!("fn main() -> void {{ {body} }}"), value % 256));
            }
        }
    }
    for (source, expected) in fixtures {
        let (_dir, input, output) = fixture(&source);
        fern::compile(&input, &output).unwrap();
        assert_eq!(
            Command::new(output).status().unwrap().code(),
            Some(expected),
            "{source}"
        );
    }
}
