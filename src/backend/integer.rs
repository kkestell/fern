//! Integer machine operations and width normalization.
use super::qbe::*;
use crate::{
    ir::model::{Function, Operand, Value},
    types::{BinaryOperator, ComparisonOperator, Scalar, UnaryOperator},
};
use std::fmt::Write;
pub(super) fn emit_comparison(
    text: &mut String,
    id: usize,
    operator: ComparisonOperator,
    left: Operand,
    right: Operand,
    ty: Scalar,
) {
    let comparison = match operator {
        ComparisonOperator::Equal => format!("ceq{}", qbe_type(ty)),
        ComparisonOperator::NotEqual => format!("cne{}", qbe_type(ty)),
        ComparisonOperator::Less => comparison_operation("lt", ty.signed(), ty),
        ComparisonOperator::LessEqual => comparison_operation("le", ty.signed(), ty),
        ComparisonOperator::Greater => comparison_operation("gt", ty.signed(), ty),
        ComparisonOperator::GreaterEqual => comparison_operation("ge", ty.signed(), ty),
    };
    writeln!(
        text,
        "    %v{id} =w {comparison} {}, {}",
        operand(left),
        operand(right)
    )
    .unwrap();
}

pub(super) fn comparison_operation(relation: &str, signed: bool, ty: Scalar) -> String {
    format!(
        "c{}{relation}{}",
        if signed { "s" } else { "u" },
        qbe_type(ty)
    )
}

pub(super) fn emit_unary_operation(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    operator: UnaryOperator,
    source: Operand,
) {
    let ty = scalar(&value.ty);
    let class = qbe_type(ty);
    let source = operand(source);
    match operator {
        UnaryOperator::Negate => {
            writeln!(emitter.text, "    %operation{id}_raw ={class} neg {source}")
        }
        UnaryOperator::Complement => writeln!(
            emitter.text,
            "    %operation{id}_raw ={class} xor {source}, -1"
        ),
    }
    .unwrap();
    emit_truncation_operand(
        &mut emitter.text,
        id,
        &format!("%operation{id}_raw"),
        ty,
        ty,
    );
}
pub(super) fn emit_binary_operation(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    operator: BinaryOperator,
    (left, right): (Operand, Operand),
    _function: &Function,
) {
    let ty = scalar(&value.ty);
    let class = qbe_type(ty);
    let instruction = match operator {
        BinaryOperator::Add => "add",
        BinaryOperator::Subtract => "sub",
        BinaryOperator::Multiply => "mul",
        BinaryOperator::Divide => {
            if ty.signed() {
                "div"
            } else {
                "udiv"
            }
        }
        BinaryOperator::Remainder => {
            if ty.signed() {
                "rem"
            } else {
                "urem"
            }
        }
        BinaryOperator::And => "and",
        BinaryOperator::Or => "or",
        BinaryOperator::Xor => "xor",
        BinaryOperator::ShiftLeft => "shl",
        BinaryOperator::ShiftRight => {
            if ty.signed() {
                "sar"
            } else {
                "shr"
            }
        }
    };
    writeln!(
        emitter.text,
        "    %operation{id}_raw ={class} {instruction} {}, {}",
        operand(left),
        operand(right)
    )
    .unwrap();
    emit_truncation_operand(
        &mut emitter.text,
        id,
        &format!("%operation{id}_raw"),
        ty,
        ty,
    );
}
pub(super) fn emit_truncation(
    text: &mut String,
    id: usize,
    source: Operand,
    source_ty: Scalar,
    destination: Scalar,
) {
    emit_truncation_operand(text, id, &operand(source), source_ty, destination);
}

fn emit_truncation_operand(
    text: &mut String,
    id: usize,
    source: &str,
    source_ty: Scalar,
    destination: Scalar,
) {
    let raw = format!("%conversion{id}_raw");
    match destination.width() {
        8 | 16 => {
            let instruction = match (destination.width(), destination.signed()) {
                (8, true) => "extsb",
                (8, false) => "extub",
                (16, true) => "extsh",
                (16, false) => "extuh",
                _ => unreachable!(),
            };
            writeln!(text, "    %v{id} =w {instruction} {source}").unwrap();
        }
        32 => {
            writeln!(text, "    %v{id} =w copy {source}").unwrap();
        }
        64 => {
            if source_ty.width() == 64 {
                writeln!(text, "    %v{id} =l copy {source}").unwrap();
                return;
            }
            let instruction = match (source_ty.width(), source_ty.signed()) {
                (8, true) => "extsb",
                (8, false) => "extub",
                (16, true) => "extsh",
                (16, false) => "extuh",
                (32, true) => "extsw",
                (32, false) => "extuw",
                _ => unreachable!(),
            };
            if source_ty.width() == 32 {
                writeln!(text, "    %v{id} =l {instruction} {source}").unwrap();
            } else {
                writeln!(text, "    {raw} =w {instruction} {source}").unwrap();
                let extend = if source_ty.signed() { "extsw" } else { "extuw" };
                writeln!(text, "    %v{id} =l {extend} {raw}").unwrap();
            }
        }
        _ => unreachable!(),
    }
}
