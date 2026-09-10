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
        EMPTY,
        "fn main() -> void { const x = 42; var y = x; }",
        "fn helper() -> void { const value = 1; } fn main() -> void {}",
        "\t\r\n\x0b\x0c /* 🌿 /* nested */ */ fn/*a*/main( ) -> void { // body\n } // eof",
    ] {
        let (dir, input, output) = fixture(source);
        fern::compile(&input, &output).unwrap();
        assert_eq!(Command::new(&output).status().unwrap().code(), Some(0));
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    }
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
fn typed_values_after_exit_do_not_change_status() {
    for body in [
        "exit(42); const value: u64 = 18446744073709551615;",
        "{ exit(42); const value: i8 = 1; } const value: u64 = 1;",
    ] {
        let (_dir, input, output) = fixture(format!("fn main() -> void {{ {body} }}"));
        fern::compile(&input, &output).unwrap();
        assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
    }
}

#[test]
fn typed_integer_diagnostics_precede_emission_and_preserve_output() {
    for (body, message) in [
        ("const x: u64 = 256u8;", "malformed integer literal"),
        (
            "var x: u8 = 0; { x = 256; }",
            "integer literal out of range for `u8`",
        ),
        (
            "const x: u64 = 1; { exit(0); exit(x); }",
            "cannot implicitly convert `u64` to `int`",
        ),
        (
            "const x: i64 = 1; const y: int = x;",
            "cannot implicitly convert `i64` to `int`",
        ),
        (
            "{ exit(0); } const x: u64 = 18446744073709551616;",
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
        (
            "fn helper() -> void {} fn helper() -> void {} fn main() -> void {}",
            "duplicate module-level name `helper`",
        ),
        (
            "const main = 1; fn main() -> void {}",
            "duplicate module-level name `main`",
        ),
        (
            "var value = 1; const value = 2; fn main() -> void {}",
            "duplicate module-level name `value`",
        ),
        (
            "const first = second; const second = first; fn main() -> void {}",
            "module-level initializer cycle",
        ),
        (
            "var runtime = 1; const invalid = runtime; fn main() -> void {}",
            "module-level initializer must be a constant expression",
        ),
        ("fn other() -> void {}", "missing `main`"),
        ("fn main() -> void { const x = x; }", "unknown binding `x`"),
        (
            "fn main() -> void { exit(0); exit(missing); }",
            "unknown binding `missing`",
        ),
        (
            "fn main() -> void { exit(0); var x: u8 = 256; }",
            "integer literal out of range for `u8`",
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
            "fn main() -> void { var x: u8 = 1; { exit(0); } x = 256; }",
            "integer literal out of range for `u8`",
        ),
        (
            "fn main() -> void { var x: u8 = 1; { x = 256; } }",
            "integer literal out of range for `u8`",
        ),
        ("fn main() -> void { x = ; }", "expected an expression"),
        ("fn main() -> void { x 1; }", "expected `=`"),
        ("fn main() -> void { x = 1 }", "expected `;`"),
        ("fn main() -> void { { }", "expected `}`"),
        ("fn main(x) -> void {}", "expected `:` after parameter name"),
        ("fn main() -> int {}", "`main` must return `void`"),
        (
            "fn main() -> void {} trailing",
            "expected a top-level declaration",
        ),
        (
            "fn main() -> void {} /* unclosed",
            "unterminated block comment",
        ),
        ("fn main() -> void {} ;", "expected a top-level declaration"),
        ("fn main() -> void {", "expected `}`"),
        ("fn main() -> void {}\u{a0}", "invalid token"),
        ("fn int() -> void {}", "reserved word"),
        ("fn main_extra() -> void {}", "missing `main`"),
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
fn integer_operator_fixtures_execute_and_replace_output() {
    for source in [
        include_str!("fixtures/programs/integer_expressions.fern"),
        include_str!("fixtures/programs/wrapping_operators.fern"),
    ] {
        let (_dir, input, output) = fixture(source);
        fs::write(&output, "keep me").unwrap();
        fern::compile(&input, &output).unwrap();
        assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
    }
}

#[test]
fn module_level_bindings_initialize_shadow_and_mutate() {
    for source in [
        include_str!("fixtures/programs/module_level_declarations.fern"),
        "var counter = 40;
         fn main() -> void {
             { var counter: u8 = 1; counter = 2; }
             counter = counter + 2;
             exit(counter);
         }",
    ] {
        let (_dir, input, output) = fixture(source);
        fern::compile(&input, &output).unwrap();
        assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
    }
}

#[test]
fn branches_and_loops_execute() {
    for (source, expected) in [
        (include_str!("fixtures/programs/branches_and_loops.fern"), 6),
        (
            "fn main() -> void {
                var total = 0;
                for {
                    total = total + 1;
                    if total == 2 { break; }
                }
                for total < 42 { total = total + 1; }
                exit(total);
            }",
            42,
        ),
        (
            "fn main() -> void {
                var total = 24;
                for var i = 0; i < 7; i += 1 {
                    if i == 3 { continue; }
                    total += i;
                }
                exit(total);
            }",
            42,
        ),
        (
            "fn main() -> void {
                var total = 39;
                for :outer var row = 0; row < 3; row += 1 {
                    for var column = 0; column < 3; column += 1 {
                        if column == 1 { continue :outer; }
                        total += 1;
                    }
                }
                exit(total);
            }",
            42,
        ),
    ] {
        let (_dir, input, output) = fixture(source);
        fern::compile(&input, &output).unwrap();
        assert_eq!(
            Command::new(output).status().unwrap().code(),
            Some(expected)
        );
    }
}

#[test]
fn arrays_execute() {
    for (source, expected) in [
        (include_str!("fixtures/programs/arrays.fern"), 46),
        // Every whole-array copy is independent of its source.
        (
            "var shared: [3]int = [1, 2, 3];
             fn identity(row: [3]int) -> [3]int { return row; }
             fn main() -> void {
                 var local = shared;
                 local[0] = 10;
                 var returned = identity(shared);
                 returned[1] = 20;
                 exit(shared[0] + shared[1] + local[0] + returned[1]);
             }",
            33,
        ),
        // An array argument is the value the array held when it was passed.
        (
            "var shared: [3]int = [1, 2, 3];
             fn mutate() -> int { shared[0] = 50; return 0; }
             fn first(row: [3]int, ignored: int) -> int { return row[0] + ignored; }
             fn main() -> void { exit(first(shared, mutate()) * 100 + shared[0]); }",
            150,
        ),
        // A target's indices run left to right, then its value.
        (
            "var trace = 0;
             fn mark(digit: int) -> int { trace = trace * 10 + digit; return digit; }
             fn main() -> void {
                 var grid: [2][2]int = [[0, 0], [0, 0]];
                 grid[mark(0)][mark(1)] = mark(2);
                 exit(trace + grid[0][1]);
             }",
            14,
        ),
        // A compound assignment evaluates its index once.
        (
            "var trace = 0;
             fn mark(digit: int) -> int { trace = trace * 10 + digit; return digit; }
             fn main() -> void {
                 var a: [2]int = [5, 6];
                 a[mark(1)] += 4;
                 exit(a[1] + trace);
             }",
            11,
        ),
        // Equality compares elements, whatever built them.
        (
            "fn zero() -> i8 { return 0; }
             fn main() -> void {
                 var base = zero();
                 var computed: [2]i8 = [base - 1, base - 127];
                 const literal: [2]i8 = [-1, -127];
                 var flags: [2]bool = [true, false];
                 const same: [2]bool = [true, false];
                 var bytes: [3]u8 = [255, 0, 7];
                 const other: [3]u8 = [255, 0, 8];
                 var grid: [2][2]int = [[1, 2], [3, 4]];
                 const twin: [2][2]int = [[1, 2], [3, 4]];
                 var total = 0;
                 if computed == literal { total += 1; }
                 if flags == same { total += 2; }
                 if bytes != other { total += 4; }
                 if grid == twin { total += 8; }
                 exit(total);
             }",
            15,
        ),
        // A whole-array assignment copies, and `len` is a constant length.
        (
            "fn main() -> void {
                 const source: [3]int = [1, 2, 3];
                 var target: [len(source)]int = [0...];
                 target = source;
                 target[0] = 10;
                 exit(target[0] + target[2] + source[0] + len(target));
             }",
            17,
        ),
        // Iteration reads the value the operand held before the first pass.
        (
            "fn main() -> void {
                 var a: [3]int = [1, 2, 3];
                 var sum = 0;
                 for v in a {
                     a[2] = 90;
                     sum += v;
                 }
                 exit(sum + a[2]);
             }",
            96,
        ),
        // Both `for … in` forms, over a nested array and over a call result.
        (
            "fn pair() -> [3]int { var out: [3]int = [4...]; return out; }
             fn main() -> void {
                 var grid: [2][3]int = [[1, 2, 3], [4, 5, 6]];
                 var sum = 0;
                 for row, i in grid {
                     for v in row {
                         if v == 5 { continue; }
                         sum += v * (i + 1);
                     }
                 }
                 for t in pair() { sum += t; }
                 exit(sum);
             }",
            38,
        ),
        // `len` reads the operand's type, including through an index.
        (
            "fn main() -> void {
                 var a: [4]int = [1...];
                 const b: [2][3]int = [[1, 2, 3], [4, 5, 6]];
                 exit(len(a) + len(b) + len(b[0]));
             }",
            9,
        ),
        // Module-level arrays hold their initial values when `main` starts.
        (
            "var counters: [3]int = [7, 8, 9];
             const limits: [_]u8 = [1, 2];
             fn main() -> void {
                 counters[2] = 1;
                 exit(counters[0] + counters[1] + counters[2] + int(limits[1]));
             }",
            18,
        ),
        // Copies and array results in a loop reuse their storage rather than
        // growing the stack once per iteration.
        (
            "fn build(seed: int) -> [3]int { var out: [3]int = [seed...]; return out; }
             fn main() -> void {
                 var source: [3]int = [1, 2, 3];
                 var total = 0;
                 for var i = 0; i < 200000; i += 1 {
                     var copy = source;
                     total = copy[0] + build(i)[2] % 7;
                 }
                 exit(total);
             }",
            3,
        ),
    ] {
        let (_dir, input, output) = fixture(source);
        fern::compile(&input, &output).unwrap_or_else(|error| panic!("{source}: {error}"));
        assert_eq!(
            Command::new(output).status().unwrap().code(),
            Some(expected),
            "{source}"
        );
    }
}

#[test]
fn parameters_calls_and_returns_execute() {
    for source in [
        include_str!("fixtures/programs/parameters_calls_and_returns.fern"),
        // A call nested in an argument and in an expression.
        "fn twice(value: int) -> int {
             return value * 2;
         }
         fn main() -> void {
             exit(twice(twice(10)) + 2);
         }",
        // Arguments run left to right, whatever they mutate.
        "var trace = 0;
         fn mark(digit: int) -> int {
             trace = trace * 10 + digit;
             return digit;
         }
         fn pair(left: int, right: int) -> int {
             return left + right;
         }
         fn main() -> void {
             pair(mark(4), mark(2));
             exit(trace);
         }",
        // A discarded result does not change execution.
        "var trace = 0;
         fn bump() -> int {
             trace += 42;
             return 7;
         }
         fn main() -> void {
             bump();
             exit(trace);
         }",
        // A call to a function declared after `main`.
        "fn main() -> void {
             exit(answer());
         }
         fn answer() -> int {
             return 42;
         }",
        // Direct recursion.
        "fn sum_to(value: int) -> int {
             if value == 0 {
                 return 0;
             }
             return value + sum_to(value - 1);
         }
         fn main() -> void {
             exit(sum_to(6) + 21);
         }",
        // Mutual recursion, with a `bool` result used as a condition.
        "fn even(value: int) -> bool {
             if value == 0 {
                 return true;
             }
             return odd(value - 1);
         }
         fn odd(value: int) -> bool {
             if value == 0 {
                 return false;
             }
             return even(value - 1);
         }
         fn main() -> void {
             if even(10) && odd(7) {
                 exit(42);
             }
             exit(1);
         }",
        // A call whose argument short-circuits, used as an operand of a binary
        // operation, a comparison, and a compound assignment.
        "fn pick(flag: bool) -> int {
             if flag {
                 return 1;
             }
             return 2;
         }
         fn main() -> void {
             var yes = true;
             var no = false;
             var total = 40;
             total += pick(yes && no);
             if total == 40 + pick(yes && no) {
                 exit(total + pick(no || yes) - 1);
             }
             exit(1);
         }",
        // An early `return` from a function returning nothing.
        "var trace = 42;
         fn keep(flag: bool) -> void {
             if flag {
                 return;
             }
             trace = 1;
         }
         fn main() -> void {
             keep(true);
             exit(trace);
         }",
    ] {
        let (_dir, input, output) = fixture(source);
        fern::compile(&input, &output).unwrap();
        assert_eq!(
            Command::new(output).status().unwrap().code(),
            Some(42),
            "{source}"
        );
    }
}

#[test]
fn exit_inside_a_called_function_does_not_return_to_its_caller() {
    let (_dir, input, output) = fixture(
        "var trace = 42;
         fn quit(status: int) -> void {
             exit(status);
             trace = 1;
         }
         fn main() -> void {
             quit(trace);
             exit(1);
         }",
    );
    fern::compile(&input, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn runtime_failures_inside_a_called_function_report_their_source_location() {
    let source = "fn divide(left: int, right: int) -> int {\n    return left / right;\n}\n\nfn main() -> void {\n    exit(divide(1, 0));\n}\n";
    let (dir, _, output) = fixture(source);
    let input = dir.path().join("callee.fern");
    fs::write(&input, source).unwrap();
    fern::compile(&input, &output).unwrap();
    let result = Command::new(output).output().unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(
        stderr.contains("integer `/` has a zero divisor"),
        "{stderr}"
    );
    assert!(stderr.contains("callee.fern:2:"), "{stderr}");
}

#[test]
fn comparisons_and_logical_expressions_execute() {
    let (_dir, input, output) = fixture(
        "fn main() -> void {
            var negative: i8 = -1;
            var zero: i8 = 0;
            var high: u64 = 18446744073709551615;
            var low: u64 = 1;
            var no = false;
            if !(negative < zero) { exit(1); }
            if negative <= zero && zero > negative && zero >= negative
                && negative != zero && high > low && high >= low
                && low < high && low <= high && no == false
                && no < true && no <= false && true > no && true >= true
                && no != true && true == true {
                exit(42);
            } else {
                exit(2);
            }
        }",
    );
    fern::compile(&input, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn comparisons_accept_short_circuiting_boolean_operands() {
    let (_dir, input, output) = fixture(
        "fn main() -> void {
            var a = true;
            var b = true;
            var c = false;
            var d = true;
            var x = 1;
            if a == (b && c) { exit(1); }
            if (a && b) == (c && d) { exit(2); }
            if (x == 1) == (b && c) { exit(3); }
            var q = a == (b && c);
            if q { exit(4); }
            exit(42);
        }",
    );
    fern::compile(&input, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn logical_operators_skip_runtime_failures() {
    let (_dir, input, output) = fixture(
        "fn main() -> void {
            var zero = 0;
            var one = 1;
            if zero != 0 && one / zero > 0 { exit(1); }
            if one == 1 || one / zero > 0 {
                exit(42);
            }
            exit(2);
        }",
    );
    fern::compile(&input, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn grouped_contextual_integer_expressions_execute() {
    let (_dir, input, output) = fixture(
        "fn main() -> void { var count: uint = 1; const grouped: u8 = ((21)); const shifted: u64 = (10 + 11) << count; exit(int(grouped) + int(shifted / 42)); }",
    );
    fern::compile(&input, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(22));
}

#[test]
fn nested_contextual_expressions_and_typed_overshifts_execute() {
    let (_dir, input, output) = fixture(
        "fn main() -> void {
            var count: uint = 1;
            const nested: i64 = -(1 << count);
            var unsigned: u8 = 1;
            var signed: i8 = -1;
            var overshift: u64 = 18446744073709551615;
            const left = unsigned << overshift;
            const right = signed >> overshift;
            exit(int(nested) + int(left) + int(right) + 45);
        }",
    );
    fern::compile(&input, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn integer_expressions_preserve_assignments_copies_scopes_and_nested_exits() {
    for body in [
        "var value = 20; const saved = value; value = saved + 22; exit(value);",
        "var value = 40; { var value = value + 1; value = value + 1; } exit(value + 2);",
        "var value = 40; { value = value + 2; { exit(value); value = value + 1; } value = 0; } exit(0);",
        "var value: u8 = 250; const saved = value; value = value +% 48; exit(int(value));",
    ] {
        let (_dir, input, output) = fixture(format!("fn main() -> void {{ {body} }}"));
        fern::compile(&input, &output).unwrap();
        assert_eq!(
            Command::new(output).status().unwrap().code(),
            Some(42),
            "{body}"
        );
    }
}

#[test]
fn runtime_integer_failures_include_the_operation_and_source_location() {
    for (operator, expected) in [
        ("+", "integer `+` overflowed"),
        ("/", "integer `/` has a zero divisor"),
        ("<<", "integer shift count is negative"),
    ] {
        let expression = match operator {
            "+" => "value + other",
            "/" => "value / other",
            "<<" => "value << other",
            _ => unreachable!(),
        };
        let (left, right) = match operator {
            "+" => ("32767", "1"),
            "/" => ("1", "0"),
            "<<" => ("1", "-1"),
            _ => unreachable!(),
        };
        let source = format!(
            "fn main() -> void {{\n    var value: i16 = {left};\n    var other: i16 = {right};\n    const failed = {expression};\n}}"
        );
        let (dir, _, output) = fixture(&source);
        let input = dir.path().join("operation.fern");
        fs::write(&input, source).unwrap();
        fern::compile(&input, &output).unwrap();
        fs::remove_file(&input).unwrap();
        let result = Command::new(output).output().unwrap();
        assert!(!result.status.success());
        let stderr = String::from_utf8(result.stderr).unwrap();
        assert!(stderr.contains(expected), "{stderr}");
        assert!(stderr.contains("operation.fern:4:"), "{stderr}");
        assert!(stderr.contains(expression), "{stderr}");
    }
}

#[test]
fn runtime_compound_assignment_failures_name_the_compound_operator() {
    for (setup, assignment, expected) in [
        (
            "var value: u8 = 255; var other: u8 = 1;",
            "value += other;",
            "integer `+=` overflowed",
        ),
        (
            "var value: u8 = 8; var other: u8 = 0;",
            "value /= other;",
            "integer `/=` has a zero divisor",
        ),
        (
            "var value: u8 = 1; var other: int = -1;",
            "value <<= other;",
            "integer `<<=` shift count is negative",
        ),
    ] {
        let source = format!("fn main() -> void {{ {setup} {assignment} }}");
        let (_dir, input, output) = fixture(source);
        fern::compile(&input, &output).unwrap();
        let result = Command::new(output).output().unwrap();
        assert!(!result.status.success());
        assert!(
            String::from_utf8(result.stderr).unwrap().contains(expected),
            "{assignment}"
        );
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
            include_str!("fixtures/programs/integer_literals.fern").to_owned(),
            42,
        ),
        (
            include_str!("fixtures/programs/shadowing.fern").to_owned(),
            42,
        ),
        ("fn main() -> void { exit(42); exit(7); }".to_owned(), 42),
        (
            "fn main() -> void { const x = 42; exit(x); var x = 7; exit(x); }".to_owned(),
            42,
        ),
    ];
    for value in [0, 42, 255, 256, 257, i64::from(i32::MAX), isize::MAX as i64] {
        for body in [
            format!("exit({value});"),
            format!("const x: int = {value}; var y = x; exit(y,);"),
        ] {
            fixtures.push((
                format!("fn main() -> void {{ {body} }}"),
                (value % 256) as i32,
            ));
        }
    }
    for literal in ["00042", "0x0002A", "0o00052", "0b000101010"] {
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
    let (_dir, input, output) =
        fixture(include_str!("fixtures/programs/assignment_and_scopes.fern"));
    fern::compile(&input, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn integer_type_programs_execute() {
    let mut fixtures = vec![(
        include_str!("fixtures/programs/integer_types.fern").to_owned(),
        42,
    )];
    for (body, expected) in [
        (
            "var x: i8 = 42; const saved = x; x = 7; var status = int(x); status = int(saved); exit(status);",
            42,
        ),
        (
            "var x: i8 = 1; { x = 42; var x = i64(x); x = 7; } exit(int(x));",
            42,
        ),
        (
            "var x: i8 = 1; { { x = 42; exit(int(x)); x = 7; } } exit(0);",
            42,
        ),
        (
            "const x: i32 = 42; const y = i32(x); const z = int(y); exit(z);",
            42,
        ),
        (
            "const x: u32 = 4294967295; const y = u32(x); var z = uint(y); z = 255;",
            0,
        ),
    ] {
        fixtures.push((format!("fn main() -> void {{ {body} }}"), expected));
    }
    for (name, max) in [
        ("i8", 127),
        ("i16", 32767),
        ("i32", i32::MAX),
        ("int", i32::MAX),
    ] {
        for value in [0, 42, max] {
            for body in [
                format!("const x: {name} = {value}; exit(int(x));"),
                format!("const x: {name} = {value}; var y: int = int(x); exit(y);"),
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

#[test]
fn explicit_integer_conversions_execute_and_trap() {
    let (_dir, input, output) = fixture(include_str!("fixtures/programs/integer_conversions.fern"));
    fern::compile(&input, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));

    let (_dir, input, output) = fixture(
        "fn main() -> void { var value: u64 = 256; const narrowed = u8(value); exit(42); }",
    );
    fern::compile(&input, &output).unwrap();
    let result = Command::new(output).output().unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("checked integer conversion failed"));
}

#[test]
fn runtime_diagnostics_identify_the_first_failing_conversion_on_stderr() {
    for (value, source_type, destination, spelling) in [
        (65536, "u64", "u16", "u16(value)"),
        (256, "u16", "u8", "u8(u16(value))"),
    ] {
        let source = format!(
            "/* 🌿 */ fn main() -> void {{\n  var value: u64 = {value};\n  var result = u8(u16(value));\n  exit(42);\n}}"
        );
        let (dir, _, output) = fixture(&source);
        let input = dir.path().join("quoted\" path\\ 🌿.fern");
        fs::write(&input, source).unwrap();
        fern::compile(&input, &output).unwrap();
        // The executable carries its diagnostic even if the source disappears.
        fs::remove_file(&input).unwrap();
        let result = Command::new(output).output().unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        let stderr = String::from_utf8(result.stderr).unwrap();
        assert!(
            stderr.contains(&format!(
                "checked integer conversion failed: `{source_type}` to `{destination}`"
            )),
            "{stderr}"
        );
        assert!(
            stderr.contains(&format!("{}:3:", input.display())),
            "{stderr}"
        );
        assert!(stderr.contains(spelling), "{stderr}");
    }
}

#[test]
fn excessive_nesting_reports_an_error_and_preserves_output() {
    for body in [
        format!("{}exit(42);{}", "{".repeat(100_000), "}".repeat(100_000)),
        format!("exit({}42{});", "int(".repeat(100_000), ")".repeat(100_000)),
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
    let output = dir.path().join("output");
    (dir, root, output)
}

#[test]
fn module_files_share_one_namespace_and_execute() {
    let (_dir, root, output) = module(&[
        (
            "main.fern",
            "const base: int = 20;\n\nfn main() -> void {\n    exit(total());\n}\n",
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
        ("zzz_entry.fern", "fn main() -> void { exit(twice(21)); }"),
    ]);
    fern::compile(&root, &output).unwrap();
    assert_eq!(Command::new(&output).status().unwrap().code(), Some(42));
}

#[test]
fn module_diagnostics_identify_the_file_they_point_into() {
    for (files, message, location) in [
        (
            [
                ("first.fern", "fn main() -> void {\n    exit(0);\n}\n"),
                ("second.fern", "const value = 1;\nconst value = 2;\n"),
            ],
            "duplicate module-level name `value`",
            "second.fern:2:7",
        ),
        (
            [
                ("first.fern", "fn main() -> void {\n    exit(0);\n}\n"),
                (
                    "second.fern",
                    "fn helper() -> void {\n    exit(missing);\n}\n",
                ),
            ],
            "unknown binding `missing`",
            "second.fern:2:10",
        ),
        (
            [
                ("first.fern", "fn main() -> void {\n    exit(broken);\n}\n"),
                ("second.fern", "const other = 1;\n"),
            ],
            "unknown binding `broken`",
            "first.fern:2:10",
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
fn runtime_failures_report_the_file_they_occurred_in() {
    let (_dir, root, output) = module(&[
        (
            "main.fern",
            "fn main() -> void {\n    exit(divide(1, 0));\n}\n",
        ),
        (
            "math.fern",
            "fn divide(left: int, right: int) -> int {\n    return left / right;\n}\n",
        ),
    ]);
    fern::compile(&root, &output).unwrap();
    let result = Command::new(&output).output().unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(
        stderr.contains("integer `/` has a zero divisor"),
        "{stderr}"
    );
    assert!(stderr.contains("math.fern:2:"), "{stderr}");
    assert!(!stderr.contains("main.fern"), "{stderr}");
}

#[test]
fn a_module_directory_holds_its_own_fern_files_only() {
    let (dir, root, output) = module(&[
        ("main.fern", "fn main() -> void { exit(42); }"),
        ("notes.txt", "fn main() -> void { exit(1); }"),
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
    let error = fern::compile(&nested.join("missing"), &output)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cannot read"), "{error}");
}

#[test]
fn output_cannot_alias_a_module_source_file() {
    let (_dir, root, _) = module(&[
        ("main.fern", "fn main() -> void { exit(42); }"),
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
    dir
}

const IMPORTING_MODULE: [(&str, &str); 4] = [
    (
        "app/main.fern",
        "use text;\nfn main() -> void { exit(text::width); }\n",
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
            "use text;\nfn main() -> void { exit(text::width); }\n",
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
             exit(counter::value);\n\
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
             exit(first::total() + second::total());\n\
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
            "use counter;\nfn main() -> void { exit(counter::value); }\n",
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
             fn main() -> void { exit(first::start + second::start + local); }\n",
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
