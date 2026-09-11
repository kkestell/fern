use crate::{
    backend::{emitter::*, qbe::*, toolchain::build_text},
    ir::model::*,
    ir::verify::VerifiedProgram,
    source::SourceMap,
    types::Scalar,
};
use std::{collections::BTreeSet, fmt::Write, process::Command};

fn integer(value: i32) -> Operand {
    Operand::Literal(Literal::Integer {
        value: i128::from(value),
        ty: Scalar::Int,
    })
}

fn copy(operand: Operand) -> Value {
    Value {
        span: None,
        ty: Scalar::Int.into(),
        kind: ValueKind::Convert {
            operand,
            truncating: false,
        },
    }
}

fn program(values: Vec<Value>, exit: Operand) -> Program {
    Program {
        structs: vec![],
        globals: vec![],
        main: FunctionId(0),
        functions: vec![Function {
            parameters: 0,
            result: None,
            flow: ControlFlow {
                entry: BlockId(0),
                locals: vec![],
                blocks: vec![crate::ir::model::Block {
                    instructions: (0..values.len())
                        .map(|id| Instruction::Value(ValueId(id)))
                        .collect(),
                    terminator: Terminator::Exit { status: exit },
                }],
            },
            values,
        }],
    }
}

fn lowered(text: &str) -> VerifiedProgram {
    let syntax = crate::frontend::parser::parse(&SourceMap::from_text(text)).unwrap();
    crate::ir::lower::lower(crate::semantic::namespaces::check_root(&syntax).unwrap())
        .verify()
        .unwrap()
}

/// Loads, checks, lowers, and verifies a tree of `(relative path, source)`
/// files whose root module is `app`, so emission runs over a program built
/// from several modules.
fn lowered_tree<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) -> VerifiedProgram {
    let dir = crate::module::tree(files);
    let program = crate::module::load(&dir.path().join("app"), &[dir.path().to_owned()])
        .unwrap_or_else(|error| panic!("{}", error.into_compile_error()));
    crate::ir::lower::lower(
        crate::semantic::namespaces::check(&program.syntax, &program.modules, &program.files)
            .unwrap_or_else(|error| panic!("{}", error.render(&program.sources))),
    )
    .verify()
    .unwrap()
}

fn definitions(qbe: &str) -> Vec<(&str, bool)> {
    qbe.lines()
        .filter_map(|line| {
            let exported = line.starts_with("export function");
            let rest = line
                .strip_prefix("export function ")
                .or_else(|| line.strip_prefix("function "))?;
            let head = rest.split('(').next().expect("a definition head");
            // The head is either `$symbol` or `<result type> $symbol`.
            Some((head.rsplit('$').next().expect("a symbol"), exported))
        })
        .collect()
}

fn data_symbols(qbe: &str) -> Vec<&str> {
    qbe.lines()
        .filter_map(|line| line.strip_prefix("data $"))
        .map(|line| line.split(' ').next().expect("a data symbol"))
        .collect()
}

// Observe every emitted value at its full QBE width before the exit mask.
// Keeping the comparisons in the native program also prevents unused
// wide values from disappearing without their representation being tested.
fn assert_native_values(verified: &VerifiedProgram, expected: &[i128]) {
    let values = &verified.program().functions[verified.program().main.0].values;
    assert_eq!(values.len(), expected.len());
    let mut text = emit(verified, None);
    // These programs define `main` alone, which ends either in an explicit
    // `exit` or in the `ret 0` of a body that reaches its end.
    let start = text.find("export function").expect("main is exported");
    let end = start
        + text[start..]
            .find("    %block0_status =")
            .or_else(|| text[start..].rfind("    ret 0\n"))
            .expect("main ends in an exit or a void return");
    text.truncate(end);
    text.push_str("    %ok0 =w copy 1\n");
    for (id, (value, expected)) in values.iter().zip(expected).enumerate() {
        let width = qbe_type(scalar(&value.ty));
        writeln!(text, "    %check{id} =w ceq{width} %v{id}, {expected}").unwrap();
        writeln!(text, "    %ok{} =w and %ok{id}, %check{id}", id + 1).unwrap();
    }
    writeln!(text, "    ret %ok{}\n}}", expected.len()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&text, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(1));
}

/// Compiles a whole program, runs it, and reports the status it exited with.
fn native_program_status(text: &str) -> Option<i32> {
    let program = lowered(text);
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&emit(&program, None), &output).unwrap();
    Command::new(output).status().unwrap().code()
}

/// Compiles a `main` body, runs it, and reports the status it exited with.
fn native_status(body: &str) -> Option<i32> {
    native_program_status(&format!("fn main() -> void {{ {body} }}"))
}

fn assert_native_failure(body: &str, expected: &str) -> String {
    let text = format!("fn main() -> void {{ {body} }}");
    let program = lowered(&text);
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&emit(&program, None), &output).unwrap();
    let result = Command::new(output).output().unwrap();
    assert!(!result.status.success(), "{text}");
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains(expected), "{text}: {stderr}");
    stderr
}

mod emitter;
mod floating;
mod integer;
mod pointers;
mod structs;
