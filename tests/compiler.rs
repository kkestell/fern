use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::{TempDir, tempdir};

const EMPTY: &str = "fn main() -> void {}";

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
    let (_dir, input, output) = fixture(EMPTY);
    fs::write(&output, "old executable").unwrap();
    let result = cli(&input, &output).output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(Command::new(output).status().unwrap().code(), Some(0));
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
        (
            "fn main() -> void { const status: int = 42; var copy = status; exit(copy,); }",
            "lowering nonempty bodies is not supported",
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
            "fn main() -> void { exit(1i32); }",
            "unsupported integer suffix `i32`",
        ),
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
        assert!(error.contains("input.fern"), "{error}");
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
    let (dir, input, _) = fixture(EMPTY);
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
    assert_eq!(fs::read_to_string(input).unwrap(), EMPTY);
}

#[test]
fn missing_tools_preserve_outputs_and_remove_intermediates() {
    for tool in ["QBE", "CC"] {
        let (dir, input, output) = fixture(EMPTY);
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
        let (dir, input, output) = fixture(EMPTY);
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
    let (dir, input, output) = fixture(EMPTY);
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
    let (dir, input, output) = fixture(EMPTY);
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();
    let result = cli(&input, &output).output();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
    failure(result.unwrap(), "cannot create temporary files");
    assert!(!output.exists());
}
