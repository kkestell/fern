//! Integer operation, conversion, overflow, shift, and trap emission.

use crate::{
    ir::model::{BinaryForm, Function, Operand, Value},
    types::{BinaryOperator, ComparisonOperator, Scalar, Type, UnaryOperator},
};

use std::fmt::Write;

use super::qbe::*;

pub(super) fn emit_comparison(
    text: &mut String,
    id: usize,
    operator: ComparisonOperator,
    left: Operand,
    right: Operand,
    operand_ty: &Type,
) {
    let Some(ty) = operand_ty.scalar() else {
        return emit_array_comparison(text, id, operator, left, right, operand_ty);
    };
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

/// Two arrays are equal when every pair of corresponding elements is. Every
/// scalar is stored normalized in its whole word, so equal arrays hold
/// identical bytes and one comparison of their storage answers for all of them.
fn emit_array_comparison(
    text: &mut String,
    id: usize,
    operator: ComparisonOperator,
    left: Operand,
    right: Operand,
    ty: &Type,
) {
    writeln!(
        text,
        "    %v{id}_difference =w call $memcmp(l {}, l {}, l {})",
        operand(left),
        operand(right),
        size(ty)
    )
    .unwrap();
    let comparison = match operator {
        ComparisonOperator::Equal => "ceqw",
        ComparisonOperator::NotEqual => "cnew",
        _ => unreachable!("only equality compares arrays"),
    };
    writeln!(text, "    %v{id} =w {comparison} %v{id}_difference, 0").unwrap();
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
    let width = qbe_type(ty);
    let source = operand(source);
    match operator {
        UnaryOperator::Negate => {
            writeln!(
                emitter.text,
                "    %operation{id}_minimum =w ceq{width} {source}, {}",
                ty.min()
            )
            .unwrap();
            emit_conditional_trap(
                &mut emitter.text,
                &mut emitter.data,
                emitter.function,
                id,
                "overflow",
                &format!("%operation{id}_minimum"),
                operation_message(
                    value.span.as_ref(),
                    emitter.diagnostics.as_ref(),
                    "integer unary `-` overflowed",
                ),
            );
            writeln!(emitter.text, "    %operation{id}_raw ={width} neg {source}").unwrap();
        }
        UnaryOperator::WrappingNegate => {
            writeln!(emitter.text, "    %operation{id}_raw ={width} neg {source}").unwrap();
        }
        UnaryOperator::Complement => {
            writeln!(
                emitter.text,
                "    %operation{id}_raw ={width} xor {source}, -1"
            )
            .unwrap();
        }
    }
    emit_normalized(&mut emitter.text, id, &format!("%operation{id}_raw"), ty);
}

pub(super) fn emit_binary_operation(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    operator: BinaryOperator,
    form: BinaryForm,
    operands: (Operand, Operand),
    function: &Function,
) {
    let (left, right) = operands;
    let spelling = match form {
        BinaryForm::Infix => operator.spelling().to_owned(),
        BinaryForm::CompoundAssignment => format!("{}=", operator.spelling()),
    };
    match operator {
        BinaryOperator::Add | BinaryOperator::Subtract | BinaryOperator::Multiply => {
            emit_checked_arithmetic(emitter, id, value, operator, &spelling, left, right);
        }
        BinaryOperator::Divide | BinaryOperator::Remainder => {
            emit_division(emitter, id, value, operator, &spelling, left, right)
        }
        BinaryOperator::WrappingAdd
        | BinaryOperator::WrappingSubtract
        | BinaryOperator::WrappingMultiply => {
            let instruction = match operator {
                BinaryOperator::WrappingAdd => "add",
                BinaryOperator::WrappingSubtract => "sub",
                BinaryOperator::WrappingMultiply => "mul",
                _ => unreachable!(),
            };
            emit_binary_raw(
                &mut emitter.text,
                id,
                scalar(&value.ty),
                instruction,
                left,
                right,
            );
            emit_normalized(
                &mut emitter.text,
                id,
                &format!("%operation{id}_raw"),
                scalar(&value.ty),
            );
        }
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => emit_shift(
            emitter,
            id,
            value,
            operator,
            &spelling,
            (left, right),
            function,
        ),
        BinaryOperator::And | BinaryOperator::Xor | BinaryOperator::Or => {
            let instruction = match operator {
                BinaryOperator::And => "and",
                BinaryOperator::Xor => "xor",
                BinaryOperator::Or => "or",
                _ => unreachable!(),
            };
            emit_binary_raw(
                &mut emitter.text,
                id,
                scalar(&value.ty),
                instruction,
                left,
                right,
            );
            emit_normalized(
                &mut emitter.text,
                id,
                &format!("%operation{id}_raw"),
                scalar(&value.ty),
            );
        }
    }
}

fn emit_binary_raw(
    text: &mut String,
    id: usize,
    ty: Scalar,
    instruction: &str,
    left: Operand,
    right: Operand,
) {
    writeln!(
        text,
        "    %operation{id}_raw ={} {instruction} {}, {}",
        qbe_type(ty),
        operand(left),
        operand(right)
    )
    .unwrap();
}

fn emit_checked_arithmetic(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    operator: BinaryOperator,
    spelling: &str,
    left: Operand,
    right: Operand,
) {
    let ty = scalar(&value.ty);
    let instruction = match operator {
        BinaryOperator::Add => "add",
        BinaryOperator::Subtract => "sub",
        BinaryOperator::Multiply => "mul",
        _ => unreachable!(),
    };
    if operator == BinaryOperator::Multiply && ty.width() == 32 {
        emit_wide_word_multiply(&mut emitter.text, id, ty, left, right);
    } else {
        emit_binary_raw(&mut emitter.text, id, ty, instruction, left, right);
        if operator == BinaryOperator::Multiply && ty.width() == 64 {
            emit_long_multiply_overflow(&mut emitter.text, id, ty, left, right);
        } else {
            emit_arithmetic_overflow(&mut emitter.text, id, ty, operator, left, right);
        }
    }
    emit_conditional_trap(
        &mut emitter.text,
        &mut emitter.data,
        emitter.function,
        id,
        "overflow",
        &format!("%operation{id}_overflow"),
        operation_message(
            value.span.as_ref(),
            emitter.diagnostics.as_ref(),
            &format!("integer `{spelling}` overflowed"),
        ),
    );
    emit_normalized(&mut emitter.text, id, &format!("%operation{id}_raw"), ty);
}

fn emit_wide_word_multiply(
    text: &mut String,
    id: usize,
    ty: Scalar,
    left: Operand,
    right: Operand,
) {
    let extension = if ty.signed() { "extsw" } else { "extuw" };
    writeln!(
        text,
        "    %operation{id}_left_long =l {extension} {}",
        operand(left)
    )
    .unwrap();
    writeln!(
        text,
        "    %operation{id}_right_long =l {extension} {}",
        operand(right)
    )
    .unwrap();
    writeln!(
        text,
        "    %operation{id}_wide =l mul %operation{id}_left_long, %operation{id}_right_long"
    )
    .unwrap();
    if ty.signed() {
        writeln!(
            text,
            "    %operation{id}_too_small =w csltl %operation{id}_wide, {}",
            ty.min()
        )
        .unwrap();
        writeln!(
            text,
            "    %operation{id}_too_large =w csgtl %operation{id}_wide, {}",
            ty.max()
        )
        .unwrap();
        writeln!(
            text,
            "    %operation{id}_overflow =w or %operation{id}_too_small, %operation{id}_too_large"
        )
        .unwrap();
    } else {
        writeln!(
            text,
            "    %operation{id}_overflow =w cugtl %operation{id}_wide, {}",
            ty.max()
        )
        .unwrap();
    }
    writeln!(text, "    %operation{id}_raw =w copy %operation{id}_wide").unwrap();
}

fn emit_arithmetic_overflow(
    text: &mut String,
    id: usize,
    ty: Scalar,
    operator: BinaryOperator,
    left: Operand,
    right: Operand,
) {
    let width = qbe_type(ty);
    let left = operand(left);
    let right = operand(right);
    let raw = format!("%operation{id}_raw");
    if ty.width() < 32 {
        if ty.signed() {
            writeln!(
                text,
                "    %operation{id}_too_small =w csltw {raw}, {}",
                ty.min()
            )
            .unwrap();
            writeln!(
                text,
                "    %operation{id}_too_large =w csgtw {raw}, {}",
                ty.max()
            )
            .unwrap();
            writeln!(
                text,
                "    %operation{id}_overflow =w or %operation{id}_too_small, %operation{id}_too_large"
            )
            .unwrap();
        } else {
            writeln!(
                text,
                "    %operation{id}_overflow =w cugtw {raw}, {}",
                ty.max()
            )
            .unwrap();
        }
        return;
    }
    if !ty.signed() {
        let comparison = if operator == BinaryOperator::Subtract {
            format!("{left}, {right}")
        } else {
            format!("{raw}, {left}")
        };
        writeln!(
            text,
            "    %operation{id}_overflow =w cult{width} {comparison}"
        )
        .unwrap();
        return;
    }

    let (first_sign, first_result, second_sign, second_result) = if operator == BinaryOperator::Add
    {
        ("csgt", "cslt", "cslt", "csgt")
    } else {
        ("cslt", "cslt", "csgt", "csgt")
    };
    writeln!(
        text,
        "    %operation{id}_first_sign =w {first_sign}{width} {right}, 0"
    )
    .unwrap();
    writeln!(
        text,
        "    %operation{id}_first_result =w {first_result}{width} {raw}, {left}"
    )
    .unwrap();
    writeln!(
        text,
        "    %operation{id}_first =w and %operation{id}_first_sign, %operation{id}_first_result"
    )
    .unwrap();
    writeln!(
        text,
        "    %operation{id}_second_sign =w {second_sign}{width} {right}, 0"
    )
    .unwrap();
    writeln!(
        text,
        "    %operation{id}_second_result =w {second_result}{width} {raw}, {left}"
    )
    .unwrap();
    writeln!(
        text,
        "    %operation{id}_second =w and %operation{id}_second_sign, %operation{id}_second_result"
    )
    .unwrap();
    writeln!(
        text,
        "    %operation{id}_overflow =w or %operation{id}_first, %operation{id}_second"
    )
    .unwrap();
}

fn emit_long_multiply_overflow(
    text: &mut String,
    id: usize,
    ty: Scalar,
    left: Operand,
    right: Operand,
) {
    let left = operand(left);
    let right = operand(right);
    if ty.signed() {
        writeln!(
            text,
            "    %operation{id}_raw_minimum =w ceql %operation{id}_raw, {}",
            ty.min()
        )
        .unwrap();
        writeln!(
            text,
            "    %operation{id}_right_negative_one =w ceql {right}, -1"
        )
        .unwrap();
        writeln!(
            text,
            "    %operation{id}_special =w and %operation{id}_raw_minimum, %operation{id}_right_negative_one"
        )
        .unwrap();
        writeln!(
            text,
            "    %operation{id}_multiply_special =w copy %operation{id}_special"
        )
        .unwrap();
    } else {
        writeln!(text, "    %operation{id}_multiply_special =w copy 0").unwrap();
    }
    writeln!(text, "    %operation{id}_right_zero =w ceql {right}, 0").unwrap();
    writeln!(
        text,
        "    %operation{id}_multiply_skip =w or %operation{id}_right_zero, %operation{id}_multiply_special"
    )
    .unwrap();
    writeln!(
        text,
        "    jnz %operation{id}_multiply_skip, @operation{id}_multiply_skip_value, @operation{id}_multiply_check"
    )
    .unwrap();
    writeln!(text, "@operation{id}_multiply_skip_value").unwrap();
    writeln!(
        text,
        "    %operation{id}_multiply_skipped =w copy %operation{id}_multiply_special"
    )
    .unwrap();
    writeln!(text, "    jmp @operation{id}_multiply_done").unwrap();
    writeln!(text, "@operation{id}_multiply_check").unwrap();
    writeln!(
        text,
        "    %operation{id}_quotient =l {} %operation{id}_raw, {right}",
        if ty.signed() { "div" } else { "udiv" }
    )
    .unwrap();
    writeln!(
        text,
        "    %operation{id}_multiply_check_value =w cnel %operation{id}_quotient, {left}"
    )
    .unwrap();
    writeln!(text, "    jmp @operation{id}_multiply_done").unwrap();
    writeln!(text, "@operation{id}_multiply_done").unwrap();
    writeln!(
        text,
        "    %operation{id}_overflow =w phi @operation{id}_multiply_skip_value %operation{id}_multiply_skipped, @operation{id}_multiply_check %operation{id}_multiply_check_value"
    )
    .unwrap();
}

fn emit_division(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    operator: BinaryOperator,
    spelling: &str,
    left: Operand,
    right: Operand,
) {
    let ty = scalar(&value.ty);
    let width = qbe_type(ty);
    let left_text = operand(left);
    let right_text = operand(right);
    writeln!(
        emitter.text,
        "    %operation{id}_zero =w ceq{width} {right_text}, 0"
    )
    .unwrap();
    emit_conditional_trap(
        &mut emitter.text,
        &mut emitter.data,
        emitter.function,
        id,
        "zero",
        &format!("%operation{id}_zero"),
        operation_message(
            value.span.as_ref(),
            emitter.diagnostics.as_ref(),
            &format!("integer `{spelling}` has a zero divisor"),
        ),
    );
    if ty.signed() {
        writeln!(
            emitter.text,
            "    %operation{id}_minimum =w ceq{width} {left_text}, {}",
            ty.min()
        )
        .unwrap();
        writeln!(
            emitter.text,
            "    %operation{id}_negative_one =w ceq{width} {right_text}, -1"
        )
        .unwrap();
        writeln!(
            emitter.text,
            "    %operation{id}_overflow =w and %operation{id}_minimum, %operation{id}_negative_one"
        )
        .unwrap();
        emit_conditional_trap(
            &mut emitter.text,
            &mut emitter.data,
            emitter.function,
            id,
            "overflow",
            &format!("%operation{id}_overflow"),
            operation_message(
                value.span.as_ref(),
                emitter.diagnostics.as_ref(),
                &format!("integer `{spelling}` overflowed"),
            ),
        );
    }
    let instruction = match (operator, ty.signed()) {
        (BinaryOperator::Divide, true) => "div",
        (BinaryOperator::Divide, false) => "udiv",
        (BinaryOperator::Remainder, true) => "rem",
        (BinaryOperator::Remainder, false) => "urem",
        _ => unreachable!(),
    };
    emit_binary_raw(&mut emitter.text, id, ty, instruction, left, right);
    emit_normalized(&mut emitter.text, id, &format!("%operation{id}_raw"), ty);
}

fn emit_shift(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    operator: BinaryOperator,
    spelling: &str,
    operands: (Operand, Operand),
    function: &Function,
) {
    let (left, right) = operands;
    let ty = scalar(&value.ty);
    let count_ty = operand_scalar(function, right);
    let count_width = qbe_type(count_ty);
    let left = operand(left);
    let right = operand(right);
    if count_ty.signed() {
        writeln!(
            emitter.text,
            "    %operation{id}_negative =w cslt{count_width} {right}, 0"
        )
        .unwrap();
        let message = if spelling.ends_with('=') {
            format!("integer `{spelling}` shift count is negative")
        } else {
            "integer shift count is negative".to_owned()
        };
        emit_conditional_trap(
            &mut emitter.text,
            &mut emitter.data,
            emitter.function,
            id,
            "negative",
            &format!("%operation{id}_negative"),
            operation_message(value.span.as_ref(), emitter.diagnostics.as_ref(), &message),
        );
    }
    writeln!(
        emitter.text,
        "    %operation{id}_large =w cuge{count_width} {right}, {}",
        ty.width()
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    jnz %operation{id}_large, @operation{id}_overshift, @operation{id}_within"
    )
    .unwrap();
    writeln!(emitter.text, "@operation{id}_overshift").unwrap();
    if operator == BinaryOperator::ShiftRight && ty.signed() {
        writeln!(
            emitter.text,
            "    %operation{id}_overshift_value ={} sar {left}, {}",
            qbe_type(ty),
            ty.width() - 1
        )
        .unwrap();
    } else {
        writeln!(
            emitter.text,
            "    %operation{id}_overshift_value ={} copy 0",
            qbe_type(ty)
        )
        .unwrap();
    }
    writeln!(emitter.text, "    jmp @operation{id}_shift_done").unwrap();
    writeln!(emitter.text, "@operation{id}_within").unwrap();
    let count = if count_width == 'l' {
        writeln!(emitter.text, "    %operation{id}_count =w copy {right}").unwrap();
        format!("%operation{id}_count")
    } else {
        right
    };
    let instruction = match operator {
        BinaryOperator::ShiftLeft => "shl",
        BinaryOperator::ShiftRight if ty.signed() => "sar",
        BinaryOperator::ShiftRight => "shr",
        _ => unreachable!(),
    };
    writeln!(
        emitter.text,
        "    %operation{id}_within_value ={} {instruction} {left}, {count}",
        qbe_type(ty)
    )
    .unwrap();
    writeln!(emitter.text, "    jmp @operation{id}_shift_done").unwrap();
    writeln!(emitter.text, "@operation{id}_shift_done").unwrap();
    writeln!(
        emitter.text,
        "    %operation{id}_raw ={} phi @operation{id}_overshift %operation{id}_overshift_value, @operation{id}_within %operation{id}_within_value",
        qbe_type(ty)
    )
    .unwrap();
    emit_normalized(&mut emitter.text, id, &format!("%operation{id}_raw"), ty);
}

fn emit_normalized(text: &mut String, id: usize, source: &str, ty: Scalar) {
    emit_truncation_operand(text, id, source, ty, ty);
}

pub(super) fn emit_checked_conversion(
    emitter: &mut Emitter<'_>,
    id: usize,
    source: Operand,
    source_ty: Scalar,
    destination: Scalar,
    message: String,
) {
    let text = &mut emitter.text;
    let source = operand(source);
    let source_minimum = source_ty.min();
    let source_maximum = i128::from(source_ty.max());
    let destination_minimum = destination.min();
    let destination_maximum = i128::from(destination.max());
    if source_maximum > destination_maximum {
        writeln!(
            text,
            "    %conversion{id}_too_large =w {} {}, {}",
            comparison_operation("lt", source_ty.signed(), source_ty),
            destination_maximum,
            source
        )
        .unwrap();
    }
    if source_minimum < destination_minimum {
        writeln!(
            text,
            "    %conversion{id}_too_small =w {} {}, {}",
            comparison_operation("lt", true, source_ty),
            source,
            destination_minimum
        )
        .unwrap();
    }
    let condition = match (
        source_maximum > destination_maximum,
        source_minimum < destination_minimum,
    ) {
        (true, true) => {
            writeln!(
                text,
                "    %conversion{id}_failed =w or %conversion{id}_too_large, %conversion{id}_too_small"
            )
            .unwrap();
            format!("%conversion{id}_failed")
        }
        (true, false) => format!("%conversion{id}_too_large"),
        (false, true) => format!("%conversion{id}_too_small"),
        (false, false) => unreachable!("checked conversion requires a possible failure"),
    };
    emit_conditional_trap(
        &mut emitter.text,
        &mut emitter.data,
        emitter.function,
        id,
        "conversion",
        &condition,
        message,
    );
    emit_truncation_operand(&mut emitter.text, id, &source, source_ty, destination);
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
