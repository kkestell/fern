use crate::{
    diagnostic::Diagnostic,
    frontend::{parser, syntax::*},
    semantic::{
        annotations::inferred_length,
        model::*,
        namespaces::{check, check_root},
    },
    source::SourceMap,
    types::{Float, LogicalOperator, Scalar, StructId, StructType, Type},
};
use num_bigint::BigInt;

fn parse(text: &str) -> Result<Syntax, Diagnostic> {
    parser::parse(&SourceMap::from_text(text))
}

fn literal(value: u128, base: u32) -> String {
    match base {
        2 => format!("0b{value:b}"),
        8 => format!("0o{value:o}"),
        10 => value.to_string(),
        16 => format!("0x{value:X}"),
        _ => unreachable!(),
    }
}

fn big(value: i128) -> BigInt {
    BigInt::from(value)
}

/// The recorded constant an integer-valued expression or binding folds to.
fn folded(value: i128) -> Option<Constant> {
    Some(Constant::Integer(big(value)))
}

/// The recorded constant an `f32` expression folds to. The expected value is
/// spelled as a Rust literal, which rounds to binary32 the same way Fern does.
fn binary32(value: f32) -> Option<Constant> {
    Some(Constant::Float(Float::Binary32(value.to_bits())))
}

/// The recorded constant an `f64` expression folds to.
fn binary64(value: f64) -> Option<Constant> {
    Some(Constant::Float(Float::Binary64(value.to_bits())))
}

/// The recorded constant an array of `f32` values folds to.
fn binary32_array(values: &[f32]) -> Option<Constant> {
    Some(Constant::Array(
        values
            .iter()
            .map(|&value| Constant::Float(Float::Binary32(value.to_bits())))
            .collect(),
    ))
}

/// The recorded constant an array of integers folds to.
fn folded_array(values: &[i128]) -> Option<Constant> {
    Some(Constant::Array(
        values
            .iter()
            .map(|&value| Constant::Integer(big(value)))
            .collect(),
    ))
}

/// The type and recorded constant of every binding of a checked program,
/// in the order the bindings were declared.
fn checked_bindings(text: &str) -> Vec<(Type, Option<Constant>)> {
    let syntax = parse(text).unwrap();
    let checked = check_root(&syntax).unwrap();
    checked
        .bindings
        .iter()
        .map(|(_, binding)| (binding.ty.clone(), binding.constant.clone()))
        .collect()
}

/// The value type a scalar spelling names, which is what checking records.
fn value_type(scalar: Scalar) -> Type {
    Type::Scalar(scalar)
}

/// The value type the `index`th struct declaration of a program names.
fn struct_type(index: usize, name: &str) -> Type {
    Type::Struct(StructType {
        id: StructId(index),
        name: name.to_owned(),
    })
}

/// The name and type of every field of the `index`th struct declaration.
fn struct_fields(checked: &CheckedProgram<'_>, index: usize) -> Vec<(String, Type)> {
    checked.structs[index]
        .fields
        .iter()
        .map(|field| {
            (
                checked.syntax.names.resolve(&field.name).to_owned(),
                field.ty.clone(),
            )
        })
        .collect()
}

/// An array type, innermost element type first.
fn array_type(length: u64, element: Type) -> Type {
    Type::Array {
        length,
        element: Box::new(element),
    }
}

/// A pointer type with the written target and mutability.
fn pointer_type(constant: bool, target: Type) -> Type {
    Type::Pointer {
        constant,
        target: Box::new(target),
    }
}

/// A slice type with the written element type and mutability.
fn slice_type(constant: bool, element: Type) -> Type {
    Type::Slice {
        constant,
        element: Box::new(element),
    }
}

/// A `geometry` module beside the `app` root module, with one public struct
/// type, one private struct type, and one public `const`.
const GEOMETRY: &str = "pub type Point struct {
    x: int,
    y: int,
}
type Hidden struct {
    z: int,
}
pub const origin = 0;
";

/// A `counter` module beside the `app` root module, with one public
/// function, one public `var`, one public `const`, and one private `const`.
const COUNTER: &str = "pub var value = 0;
pub const step = 2;
const origin = 10;
pub fn bump(amount: int) -> int {
    value = value + amount;
    return value;
}
";

/// Loads a tree of `(relative path, source)` files rooted at `app` and
/// checks it, so multi-module tests run through real import resolution.
fn load_tree<'a>(
    files: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> (tempfile::TempDir, crate::module::Program) {
    let dir = crate::module::tree(files);
    let program = crate::module::load(&dir.path().join("app"), &[dir.path().to_owned()])
        .unwrap_or_else(|error| panic!("{}", error.into_compile_error()));
    (dir, program)
}

fn accepts_tree<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) {
    let (_dir, program) = load_tree(files);
    check(&program.syntax, &program.modules, &program.files).unwrap();
}

fn tree_error<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) -> Diagnostic {
    let (_dir, program) = load_tree(files);
    check(&program.syntax, &program.modules, &program.files).unwrap_err()
}

fn rejects_tree<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>, message: &str) {
    assert_eq!(tree_error(files).message, message);
}

/// Rejects a tree whose root module is one `app/main.fern` file, checking
/// the diagnostic's span against `marked`, where `«»` bracket it.
fn rejects_root(marked: &str, message: &str) {
    let start = marked.find('«').unwrap();
    let end = marked.find('»').unwrap() - '«'.len_utf8();
    let main = marked.replace(['«', '»'], "");
    let error = tree_error([
        ("app/main.fern", main.as_str()),
        ("counter/counter.fern", COUNTER),
    ]);
    assert_eq!(error.message, message, "{marked}");
    assert_eq!(error.span, start..end, "{marked}");
}

fn rejects(body: &str, offending: &str, message: &str) {
    let text = format!("/* 🌿 */ fn main() -> void {{ {body} }}");
    let syntax = parse(&text).unwrap();
    let error = check_root(&syntax).unwrap_err();
    let start = text.rfind(offending).unwrap();
    assert_eq!(error.span, start..start + offending.len(), "{body}");
    assert_eq!(error.message, message, "{body}");
}

fn accepts(body: &str) {
    let text = format!("fn main() -> void {{ {body} }}");
    let syntax = parse(&text).unwrap();
    check_root(&syntax).unwrap();
}

fn rejects_source(text: &str, offending: &str, message: &str) {
    let syntax = parse(text).unwrap();
    let error = check_root(&syntax).unwrap_err();
    let start = text.rfind(offending).unwrap();
    assert_eq!(error.span, start..start + offending.len(), "{text}");
    assert_eq!(error.message, message, "{text}");
}

fn accepts_source(text: &str) {
    let syntax = parse(text).unwrap();
    check_root(&syntax).unwrap();
}

mod annotations;
mod constants;
mod expressions;
mod model;
mod namespaces;
mod statements;
