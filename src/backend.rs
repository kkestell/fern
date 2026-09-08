use crate::{
    CompileError,
    diagnostic::Diagnostic,
    frontend::{BinaryOperator, UnaryOperator},
    ir::{Operand, ValueId, ValueKind, VerifiedEntry},
    semantic::Type,
    source::Source,
};
use std::{env, fmt::Write, fs, path::Path, process::Command};

fn operand(operand: Operand) -> String {
    match operand {
        Operand::Integer { value, .. } => value.to_string(),
        Operand::Value(ValueId(id)) => format!("%v{id}"),
    }
}

fn qbe_type(ty: Type) -> char {
    if ty.width() == 64 { 'l' } else { 'w' }
}

pub(crate) fn emit(verified: &VerifiedEntry, source_file: Option<&Source>) -> String {
    let entry = verified.entry();
    let mut text = String::new();
    let mut data = String::new();
    text.push_str("export function w $main() {\n@start\n");
    for (id, value) in entry.values.iter().enumerate() {
        match value.kind {
            ValueKind::Copy(source) => {
                writeln!(
                    text,
                    "    %v{id} ={} copy {}",
                    qbe_type(value.ty),
                    operand(source)
                )
                .unwrap();
            }
            ValueKind::Convert {
                operand: source,
                truncating,
            } => {
                let source_ty = operand_type(entry, source);
                if truncating || conversion_fits(source_ty, value.ty) {
                    emit_truncation(&mut text, id, source, source_ty, value.ty);
                } else {
                    let message = format!(
                        "checked integer conversion failed: `{}` to `{}`",
                        source_ty.name(),
                        value.ty.name()
                    );
                    let message = match (&value.span, source_file) {
                        (Some(span), Some(source)) => Diagnostic::new(span.clone(), message)
                            .render(source)
                            .to_string(),
                        _ => format!("{message}\n"),
                    };
                    write!(data, "data $fern_conversion{id}_message = {{ ").unwrap();
                    // Numeric bytes also handle quotes, backslashes, and UTF-8 in source paths.
                    for byte in message.bytes() {
                        write!(data, "b {byte}, ").unwrap();
                    }
                    data.push_str("b 0 }\n");
                    emit_checked_conversion(
                        &mut text,
                        id,
                        source,
                        source_ty,
                        value.ty,
                        message.len(),
                    );
                }
            }
            ValueKind::Unary { operator, operand } => {
                emit_unary_operation(
                    &mut text,
                    &mut data,
                    id,
                    value,
                    operator,
                    operand,
                    source_file,
                );
            }
            ValueKind::Binary {
                operator,
                left,
                right,
            } => {
                emit_binary_operation(
                    &mut text,
                    &mut data,
                    id,
                    value,
                    operator,
                    left,
                    right,
                    entry,
                    source_file,
                );
            }
        }
    }
    let exit = operand(entry.exit);
    let exit = if Type::Int.width() == 64 {
        writeln!(text, "    %exit_status =w copy {exit}").unwrap();
        "%exit_status"
    } else {
        exit.as_str()
    };
    writeln!(text, "    %status =w and {exit}, 255").unwrap();
    text.push_str("    ret %status\n}\n");
    data + &text
}

fn emit_unary_operation(
    text: &mut String,
    data: &mut String,
    id: usize,
    value: &crate::ir::Value,
    operator: UnaryOperator,
    source: Operand,
    source_file: Option<&Source>,
) {
    let ty = value.ty;
    let width = qbe_type(ty);
    let source = operand(source);
    match operator {
        UnaryOperator::Negate => {
            writeln!(
                text,
                "    %operation{id}_minimum =w ceq{width} {source}, {}",
                type_minimum(ty)
            )
            .unwrap();
            emit_conditional_trap(
                text,
                data,
                id,
                "overflow",
                &format!("%operation{id}_minimum"),
                operation_message(value, source_file, "integer unary `-` overflowed"),
            );
            writeln!(text, "    %operation{id}_raw ={width} neg {source}").unwrap();
        }
        UnaryOperator::WrappingNegate => {
            writeln!(text, "    %operation{id}_raw ={width} neg {source}").unwrap();
        }
        UnaryOperator::Complement => {
            writeln!(text, "    %operation{id}_raw ={width} xor {source}, -1").unwrap();
        }
    }
    emit_normalized(text, id, &format!("%operation{id}_raw"), ty);
}

#[allow(clippy::too_many_arguments)]
fn emit_binary_operation(
    text: &mut String,
    data: &mut String,
    id: usize,
    value: &crate::ir::Value,
    operator: BinaryOperator,
    left: Operand,
    right: Operand,
    entry: &crate::ir::Entry,
    source_file: Option<&Source>,
) {
    match operator {
        BinaryOperator::Add | BinaryOperator::Subtract | BinaryOperator::Multiply => {
            emit_checked_arithmetic(text, data, id, value, operator, left, right, source_file);
        }
        BinaryOperator::Divide | BinaryOperator::Remainder => {
            emit_division(text, data, id, value, operator, left, right, source_file)
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
            emit_binary_raw(text, id, value.ty, instruction, left, right);
            emit_normalized(text, id, &format!("%operation{id}_raw"), value.ty);
        }
        BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight => emit_shift(
            text,
            data,
            id,
            value,
            operator,
            left,
            right,
            entry,
            source_file,
        ),
        BinaryOperator::And | BinaryOperator::AndNot | BinaryOperator::Xor | BinaryOperator::Or => {
            let width = qbe_type(value.ty);
            let left = operand(left);
            let right = operand(right);
            if operator == BinaryOperator::AndNot {
                writeln!(text, "    %operation{id}_inverse ={width} xor {right}, -1").unwrap();
                writeln!(
                    text,
                    "    %operation{id}_raw ={width} and {left}, %operation{id}_inverse"
                )
                .unwrap();
            } else {
                let instruction = match operator {
                    BinaryOperator::And => "and",
                    BinaryOperator::Xor => "xor",
                    BinaryOperator::Or => "or",
                    _ => unreachable!(),
                };
                writeln!(
                    text,
                    "    %operation{id}_raw ={width} {instruction} {left}, {right}"
                )
                .unwrap();
            }
            emit_normalized(text, id, &format!("%operation{id}_raw"), value.ty);
        }
    }
}

fn emit_binary_raw(
    text: &mut String,
    id: usize,
    ty: Type,
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

#[allow(clippy::too_many_arguments)]
fn emit_checked_arithmetic(
    text: &mut String,
    data: &mut String,
    id: usize,
    value: &crate::ir::Value,
    operator: BinaryOperator,
    left: Operand,
    right: Operand,
    source_file: Option<&Source>,
) {
    let ty = value.ty;
    let instruction = match operator {
        BinaryOperator::Add => "add",
        BinaryOperator::Subtract => "sub",
        BinaryOperator::Multiply => "mul",
        _ => unreachable!(),
    };
    if operator == BinaryOperator::Multiply && ty.width() == 32 {
        emit_wide_word_multiply(text, id, ty, left, right);
    } else {
        emit_binary_raw(text, id, ty, instruction, left, right);
        if operator == BinaryOperator::Multiply && ty.width() == 64 {
            emit_long_multiply_overflow(text, id, ty, left, right);
        } else {
            emit_arithmetic_overflow(text, id, ty, operator, left, right);
        }
    }
    emit_conditional_trap(
        text,
        data,
        id,
        "overflow",
        &format!("%operation{id}_overflow"),
        operation_message(
            value,
            source_file,
            &format!("integer `{}` overflowed", operation_spelling(operator)),
        ),
    );
    emit_normalized(text, id, &format!("%operation{id}_raw"), ty);
}

fn emit_wide_word_multiply(text: &mut String, id: usize, ty: Type, left: Operand, right: Operand) {
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
            type_minimum(ty)
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
    ty: Type,
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
                type_minimum(ty)
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
    ty: Type,
    left: Operand,
    right: Operand,
) {
    let left = operand(left);
    let right = operand(right);
    if ty.signed() {
        writeln!(
            text,
            "    %operation{id}_raw_minimum =w ceql %operation{id}_raw, {}",
            type_minimum(ty)
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

#[allow(clippy::too_many_arguments)]
fn emit_division(
    text: &mut String,
    data: &mut String,
    id: usize,
    value: &crate::ir::Value,
    operator: BinaryOperator,
    left: Operand,
    right: Operand,
    source_file: Option<&Source>,
) {
    let ty = value.ty;
    let width = qbe_type(ty);
    let left_text = operand(left);
    let right_text = operand(right);
    writeln!(
        text,
        "    %operation{id}_zero =w ceq{width} {right_text}, 0"
    )
    .unwrap();
    emit_conditional_trap(
        text,
        data,
        id,
        "zero",
        &format!("%operation{id}_zero"),
        operation_message(
            value,
            source_file,
            &format!(
                "integer `{}` has a zero divisor",
                operation_spelling(operator)
            ),
        ),
    );
    if ty.signed() {
        writeln!(
            text,
            "    %operation{id}_minimum =w ceq{width} {left_text}, {}",
            type_minimum(ty)
        )
        .unwrap();
        writeln!(
            text,
            "    %operation{id}_negative_one =w ceq{width} {right_text}, -1"
        )
        .unwrap();
        writeln!(
            text,
            "    %operation{id}_overflow =w and %operation{id}_minimum, %operation{id}_negative_one"
        )
        .unwrap();
        emit_conditional_trap(
            text,
            data,
            id,
            "overflow",
            &format!("%operation{id}_overflow"),
            operation_message(
                value,
                source_file,
                &format!("integer `{}` overflowed", operation_spelling(operator)),
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
    emit_binary_raw(text, id, ty, instruction, left, right);
    emit_normalized(text, id, &format!("%operation{id}_raw"), ty);
}

#[allow(clippy::too_many_arguments)]
fn emit_shift(
    text: &mut String,
    data: &mut String,
    id: usize,
    value: &crate::ir::Value,
    operator: BinaryOperator,
    left: Operand,
    right: Operand,
    entry: &crate::ir::Entry,
    source_file: Option<&Source>,
) {
    let ty = value.ty;
    let count_ty = operand_type(entry, right);
    let count_width = qbe_type(count_ty);
    let left = operand(left);
    let right = operand(right);
    if count_ty.signed() {
        writeln!(
            text,
            "    %operation{id}_negative =w cslt{count_width} {right}, 0"
        )
        .unwrap();
        emit_conditional_trap(
            text,
            data,
            id,
            "negative",
            &format!("%operation{id}_negative"),
            operation_message(value, source_file, "integer shift count is negative"),
        );
    }
    writeln!(
        text,
        "    %operation{id}_large =w cuge{count_width} {right}, {}",
        ty.width()
    )
    .unwrap();
    writeln!(
        text,
        "    jnz %operation{id}_large, @operation{id}_overshift, @operation{id}_within"
    )
    .unwrap();
    writeln!(text, "@operation{id}_overshift").unwrap();
    if operator == BinaryOperator::ShiftRight && ty.signed() {
        writeln!(
            text,
            "    %operation{id}_overshift_value ={} sar {left}, {}",
            qbe_type(ty),
            ty.width() - 1
        )
        .unwrap();
    } else {
        writeln!(
            text,
            "    %operation{id}_overshift_value ={} copy 0",
            qbe_type(ty)
        )
        .unwrap();
    }
    writeln!(text, "    jmp @operation{id}_shift_done").unwrap();
    writeln!(text, "@operation{id}_within").unwrap();
    let count = if count_width == 'l' {
        writeln!(text, "    %operation{id}_count =w copy {right}").unwrap();
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
        text,
        "    %operation{id}_within_value ={} {instruction} {left}, {count}",
        qbe_type(ty)
    )
    .unwrap();
    writeln!(text, "    jmp @operation{id}_shift_done").unwrap();
    writeln!(text, "@operation{id}_shift_done").unwrap();
    writeln!(
        text,
        "    %operation{id}_raw ={} phi @operation{id}_overshift %operation{id}_overshift_value, @operation{id}_within %operation{id}_within_value",
        qbe_type(ty)
    )
    .unwrap();
    emit_normalized(text, id, &format!("%operation{id}_raw"), ty);
}

fn emit_normalized(text: &mut String, id: usize, source: &str, ty: Type) {
    emit_truncation_operand(text, id, source, ty, ty);
}

fn operation_spelling(operator: BinaryOperator) -> &'static str {
    match operator {
        BinaryOperator::Multiply => "*",
        BinaryOperator::Divide => "/",
        BinaryOperator::Remainder => "%",
        BinaryOperator::WrappingMultiply => "&*",
        BinaryOperator::Add => "+",
        BinaryOperator::Subtract => "-",
        BinaryOperator::WrappingAdd => "&+",
        BinaryOperator::WrappingSubtract => "&-",
        BinaryOperator::ShiftLeft => "<<",
        BinaryOperator::ShiftRight => ">>",
        BinaryOperator::And => "&",
        BinaryOperator::AndNot => "&^",
        BinaryOperator::Xor => "^",
        BinaryOperator::Or => "|",
    }
}

fn operation_message(
    value: &crate::ir::Value,
    source_file: Option<&Source>,
    message: &str,
) -> String {
    match (&value.span, source_file) {
        (Some(span), Some(source)) => Diagnostic::new(span.clone(), message)
            .render(source)
            .to_string(),
        _ => format!("{message}\n"),
    }
}

fn emit_conditional_trap(
    text: &mut String,
    data: &mut String,
    id: usize,
    cause: &str,
    condition: &str,
    message: String,
) {
    let failed = format!("operation{id}_{cause}_failed");
    let ready = format!("operation{id}_{cause}_ready");
    let symbol = format!("fern_operation{id}_{cause}_message");
    writeln!(text, "    jnz {condition}, @{failed}, @{ready}").unwrap();
    writeln!(text, "@{failed}").unwrap();
    write!(data, "data ${symbol} = {{ ").unwrap();
    for byte in message.bytes() {
        write!(data, "b {byte}, ").unwrap();
    }
    data.push_str("b 0 }\n");
    writeln!(
        text,
        "    call $write(w 2, l ${symbol}, {} {})",
        qbe_type(Type::Uint),
        message.len()
    )
    .unwrap();
    text.push_str("    call $abort()\n    ret 1\n");
    writeln!(text, "@{ready}").unwrap();
}

fn conversion_fits(source: Type, destination: Type) -> bool {
    type_minimum(source) >= type_minimum(destination) && source.max() <= destination.max()
}

fn operand_type(entry: &crate::ir::Entry, operand: Operand) -> Type {
    match operand {
        Operand::Integer { ty, .. } => ty,
        Operand::Value(ValueId(id)) => entry.values[id].ty,
    }
}

fn type_minimum(ty: Type) -> i128 {
    if ty.signed() {
        -(1i128 << (ty.width() - 1))
    } else {
        0
    }
}

fn comparison_name(signed: bool, ty: Type) -> String {
    format!("c{}lt{}", if signed { "s" } else { "u" }, qbe_type(ty))
}

fn emit_checked_conversion(
    text: &mut String,
    id: usize,
    source: Operand,
    source_ty: Type,
    destination: Type,
    message_len: usize,
) {
    let source = operand(source);
    let source_minimum = type_minimum(source_ty);
    let source_maximum = i128::from(source_ty.max());
    let destination_minimum = type_minimum(destination);
    let destination_maximum = i128::from(destination.max());
    let failure = format!("conversion{id}_failed");
    let ready = format!("conversion{id}_ready");
    let done = format!("conversion{id}_done");
    let upper = format!("conversion{id}_upper");

    if source_maximum > destination_maximum {
        writeln!(
            text,
            "    %conversion{id}_too_large =w {} {}, {}",
            comparison_name(source_ty.signed(), source_ty),
            destination_maximum,
            source
        )
        .unwrap();
        writeln!(
            text,
            "    jnz %conversion{id}_too_large, @{failure}, @{upper}"
        )
        .unwrap();
    }
    if source_minimum < destination_minimum {
        if source_maximum > destination_maximum {
            writeln!(text, "@{upper}").unwrap();
        }
        writeln!(
            text,
            "    %conversion{id}_too_small =w {} {}, {}",
            comparison_name(true, source_ty),
            source,
            destination_minimum
        )
        .unwrap();
        writeln!(
            text,
            "    jnz %conversion{id}_too_small, @{failure}, @{ready}"
        )
        .unwrap();
    } else if source_maximum > destination_maximum {
        writeln!(text, "@{upper}").unwrap();
    }
    writeln!(text, "@{ready}").unwrap();
    emit_truncation_operand(text, id, &source, source_ty, destination);
    writeln!(text, "    jmp @{done}").unwrap();
    writeln!(text, "@{failure}").unwrap();
    // Write directly to stderr: abort does not reliably flush C stdio buffers.
    writeln!(
        text,
        "    call $write(w 2, l $fern_conversion{id}_message, {} {message_len})",
        qbe_type(Type::Uint)
    )
    .unwrap();
    text.push_str("    call $abort()\n    ret 1\n");
    writeln!(text, "@{done}").unwrap();
}

fn emit_truncation(
    text: &mut String,
    id: usize,
    source: Operand,
    source_ty: Type,
    destination: Type,
) {
    emit_truncation_operand(text, id, &operand(source), source_ty, destination);
}

fn emit_truncation_operand(
    text: &mut String,
    id: usize,
    source: &str,
    source_ty: Type,
    destination: Type,
) {
    let raw = format!("%conversion{id}_raw");
    match destination.width() {
        8 | 16 => {
            let input = if source_ty.width() == 64 {
                writeln!(text, "    {raw} =w copy {source}").unwrap();
                raw.as_str()
            } else {
                source
            };
            let instruction = match (destination.width(), destination.signed()) {
                (8, true) => "extsb",
                (8, false) => "extub",
                (16, true) => "extsh",
                (16, false) => "extuh",
                _ => unreachable!(),
            };
            writeln!(text, "    %v{id} =w {instruction} {input}").unwrap();
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

pub(crate) fn build(
    entry: &VerifiedEntry,
    source: &Source,
    output: &Path,
) -> Result<(), CompileError> {
    build_text(&emit(entry, Some(source)), output)
}

fn build_text(text: &str, output: &Path) -> Result<(), CompileError> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temp = tempfile::Builder::new()
        .prefix(".fern-")
        .tempdir_in(parent)
        .map_err(|e| {
            CompileError::new(format!(
                "cannot create temporary files beside {}: {e}",
                output.display()
            ))
        })?;
    let qbe = temp.path().join("program.ssa");
    let assembly = temp.path().join("program.s");
    let executable = temp.path().join("program");
    fs::write(&qbe, text).map_err(|e| CompileError::new(format!("cannot write QBE input: {e}")))?;
    run(
        Command::new(env::var_os("QBE").unwrap_or_else(|| "qbe".into()))
            .arg("-o")
            .arg(&assembly)
            .arg(&qbe),
    )?;
    run(
        Command::new(env::var_os("CC").unwrap_or_else(|| "cc".into()))
            .arg(&assembly)
            .arg("-o")
            .arg(&executable),
    )?;
    fs::rename(&executable, output)
        .map_err(|e| CompileError::new(format!("cannot publish {}: {e}", output.display())))?;
    Ok(())
}

fn run(command: &mut Command) -> Result<(), CompileError> {
    let tool = command.get_program().to_string_lossy().into_owned();
    let result = command
        .output()
        .map_err(|e| CompileError::new(format!("cannot run {tool}: {e}")))?;
    if !result.status.success() {
        return Err(CompileError::new(format!(
            "{tool} failed ({}):\n{}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Entry, Value};

    fn integer(value: i32) -> Operand {
        Operand::Integer {
            value: i128::from(value),
            ty: Type::Int,
        }
    }

    fn copy(operand: Operand) -> Value {
        Value {
            span: None,
            ty: Type::Int,
            kind: ValueKind::Copy(operand),
        }
    }

    // Observe every emitted value at its full QBE width before the exit mask.
    // Keeping the comparisons in the native program also prevents unused
    // wide values from disappearing without their representation being tested.
    fn assert_native_values(verified: &VerifiedEntry, expected: &[i128]) {
        assert_eq!(verified.entry().values.len(), expected.len());
        let mut text = emit(verified, None);
        text.truncate(text.find("    %status =").unwrap());
        text.push_str("    %ok0 =w copy 1\n");
        for (id, (value, expected)) in verified.entry().values.iter().zip(expected).enumerate() {
            let width = qbe_type(value.ty);
            writeln!(text, "    %check{id} =w ceq{width} %v{id}, {expected}").unwrap();
            writeln!(text, "    %ok{} =w and %ok{id}, %check{id}", id + 1).unwrap();
        }
        writeln!(text, "    ret %ok{}\n}}", expected.len()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("program");
        build_text(&text, &output).unwrap();
        assert_eq!(Command::new(output).status().unwrap().code(), Some(1));
    }

    fn assert_native_failure(body: &str, expected: &str) -> String {
        let text = format!("fn main() -> void {{ {body} }}");
        let syntax = crate::frontend::parse(&text).unwrap();
        let entry = crate::ir::lower(crate::semantic::check(&syntax).unwrap())
            .verify()
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("program");
        build_text(&emit(&entry, None), &output).unwrap();
        let result = Command::new(output).output().unwrap();
        assert!(!result.status.success(), "{text}");
        let stderr = String::from_utf8(result.stderr).unwrap();
        assert!(stderr.contains(expected), "{text}: {stderr}");
        stderr
    }

    #[test]
    fn full_width_integer_copies_and_conversions_execute() {
        let ranges = [
            (Type::I8, -128, 127),
            (Type::I16, -32768, 32767),
            (Type::I32, -2147483648, 2147483647),
            (Type::I64, -9223372036854775808, 9223372036854775807),
            (Type::U8, 0, 255),
            (Type::U16, 0, 65535),
            (Type::U32, 0, 4294967295),
            (Type::U64, 0, 18446744073709551615),
            (
                Type::Int,
                type_minimum(Type::Int),
                i128::from(Type::Int.max()),
            ),
            (Type::Uint, 0, i128::from(Type::Uint.max())),
        ];
        let mut entry = Entry {
            values: vec![],
            exit: integer(0),
        };
        let mut expected = Vec::new();
        for (source, min, max) in ranges {
            for (destination, dest_min, dest_max) in ranges {
                let lower = min.max(dest_min);
                let upper = max.min(dest_max);
                let mut numbers = vec![lower, upper, 0, 1];
                if lower < 0 {
                    numbers.push(-1);
                }
                numbers.sort_unstable();
                numbers.dedup();
                for number in numbers {
                    let literal = Operand::Integer {
                        value: number,
                        ty: source,
                    };
                    let id = entry.values.len();
                    entry.values.push(Value {
                        span: None,
                        ty: source,
                        kind: ValueKind::Copy(literal),
                    });
                    expected.push(number);
                    for operand in [literal, Operand::Value(ValueId(id))] {
                        entry.values.push(Value {
                            span: None,
                            ty: destination,
                            kind: ValueKind::Convert {
                                operand,
                                truncating: false,
                            },
                        });
                        expected.push(number);
                        let copied = Operand::Value(ValueId(entry.values.len() - 1));
                        entry.values.push(Value {
                            span: None,
                            ty: destination,
                            kind: ValueKind::Copy(copied),
                        });
                        expected.push(number);
                    }
                }
            }
        }
        assert_native_values(&entry.verify().unwrap(), &expected);
    }

    fn truncated(value: i128, destination: Type) -> i128 {
        let modulus = 1i128 << destination.width();
        let bits = value.rem_euclid(modulus);
        if destination.signed() && bits >= (1i128 << (destination.width() - 1)) {
            bits - modulus
        } else {
            bits
        }
    }

    #[test]
    fn truncating_conversions_preserve_each_destination_bit_pattern() {
        let types = [
            Type::I8,
            Type::I16,
            Type::I32,
            Type::I64,
            Type::U8,
            Type::U16,
            Type::U32,
            Type::U64,
            Type::Int,
            Type::Uint,
        ];
        let mut entry = Entry {
            values: vec![],
            exit: integer(0),
        };
        let mut expected = Vec::new();
        for source in types {
            let minimum = type_minimum(source);
            let maximum = i128::from(source.max());
            for value in [minimum, if source.signed() { -1 } else { 1 }, maximum] {
                let source_id = entry.values.len();
                entry.values.push(Value {
                    span: None,
                    ty: source,
                    kind: ValueKind::Copy(Operand::Integer { value, ty: source }),
                });
                expected.push(value);
                for destination in types {
                    entry.values.push(Value {
                        span: None,
                        ty: destination,
                        kind: ValueKind::Convert {
                            operand: Operand::Value(ValueId(source_id)),
                            truncating: true,
                        },
                    });
                    expected.push(truncated(value, destination));
                }
            }
        }
        assert_native_values(&entry.verify().unwrap(), &expected);
    }

    #[test]
    fn native_integer_operations_preserve_values_for_every_type() {
        let types = [
            Type::I8,
            Type::I16,
            Type::I32,
            Type::I64,
            Type::U8,
            Type::U16,
            Type::U32,
            Type::U64,
            Type::Int,
            Type::Uint,
        ];
        for ty in types {
            let name = ty.name();
            let minimum = type_minimum(ty);
            let maximum = i128::from(ty.max());
            let signed_operations = if ty.signed() {
                format!(
                    "const negate = -a; var negative: {name} = -7; const negative_quotient = negative / b; const negative_remainder = negative % b;"
                )
            } else {
                String::new()
            };
            let text = format!(
                "fn main() -> void {{
                    var a: {name} = 20; var b: {name} = 3;
                    const add = a + b; const subtract = a - b;
                    const multiply = a * b; const divide = a / b; const remainder = a % b;
                    const wrapping_add = a &+ b; const wrapping_subtract = a &- b;
                    const wrapping_multiply = a &* b;
                    {signed_operations} const wrapping_negate = &-a; const complement = ^a;
                    const and = a & b; const and_not = a &^ b;
                    const xor = a ^ b; const or = a | b;
                    const shift_left = a << b; const shift_right = a >> b;
                    var maximum: {name} = {maximum}; var minimum: {name} = {minimum};
                    var one: {name} = 1;
                    const wrapped_maximum = maximum &+ one;
                    const wrapped_minimum = minimum &- one;
                    const wrapped_product = maximum &* b;
                }}",
                name = name,
            );
            let syntax = crate::frontend::parse(&text).unwrap();
            let entry = crate::ir::lower(crate::semantic::check(&syntax).unwrap())
                .verify()
                .unwrap();
            let mut expected = vec![20, 3, 23, 17, 60, 6, 2, 23, 17, 60];
            if ty.signed() {
                expected.extend([-20, -7, -2, -1]);
            }
            expected.extend([
                truncated(-20, ty),
                truncated(!20, ty),
                20 & 3,
                20 & !3,
                20 ^ 3,
                20 | 3,
                truncated(20 << 3, ty),
                20 >> 3,
                maximum,
                minimum,
                1,
                truncated(maximum + 1, ty),
                truncated(minimum - 1, ty),
                truncated(maximum * 3, ty),
            ]);
            assert_native_values(&entry, &expected);
        }
    }

    #[test]
    fn checked_arithmetic_traps_at_each_integer_width() {
        let types = [
            Type::I8,
            Type::I16,
            Type::I32,
            Type::I64,
            Type::U8,
            Type::U16,
            Type::U32,
            Type::U64,
            Type::Int,
            Type::Uint,
        ];
        for ty in types {
            let name = ty.name();
            let minimum = type_minimum(ty);
            let maximum = i128::from(ty.max());
            for (operator, left, right) in [("+", maximum, 1), ("-", minimum, 1), ("*", maximum, 2)]
            {
                assert_native_failure(
                    &format!(
                        "var left: {name} = {left}; var right: {name} = {right}; const failed = left {operator} right;"
                    ),
                    &format!("integer `{operator}` overflowed"),
                );
            }
            if ty.signed() {
                assert_native_failure(
                    &format!(
                        "var minimum: {name} = {minimum}; var negative_one: {name} = -1; const failed = minimum * negative_one;"
                    ),
                    "integer `*` overflowed",
                );
                assert_native_failure(
                    &format!("var minimum: {name} = {minimum}; const failed = -minimum;"),
                    "integer unary `-` overflowed",
                );
            }
        }
    }

    #[test]
    fn division_remainder_and_shift_failures_are_explicit() {
        let types = [
            Type::I8,
            Type::I16,
            Type::I32,
            Type::I64,
            Type::U8,
            Type::U16,
            Type::U32,
            Type::U64,
            Type::Int,
            Type::Uint,
        ];
        for ty in types {
            let name = ty.name();
            for operator in ["/", "%"] {
                assert_native_failure(
                    &format!(
                        "var value: {name} = 1; var zero: {name} = 0; const failed = value {operator} zero;"
                    ),
                    &format!("integer `{operator}` has a zero divisor"),
                );
            }
            assert_native_failure(
                &format!(
                    "var value: {name} = 1; var negative: int = -1; const failed = value << negative;"
                ),
                "integer shift count is negative",
            );
            if ty.signed() {
                let minimum = type_minimum(ty);
                for operator in ["/", "%"] {
                    assert_native_failure(
                        &format!(
                            "var minimum: {name} = {minimum}; var negative_one: {name} = -1; const failed = minimum {operator} negative_one;"
                        ),
                        &format!("integer `{operator}` overflowed"),
                    );
                }
            }
        }
    }

    #[test]
    fn runtime_shifts_define_discarded_bits_and_large_counts() {
        let syntax = crate::frontend::parse(
            "fn main() -> void {
                var high: u8 = 128; var one: uint = 1; var width: u64 = 8;
                var huge: u64 = 18446744073709551615;
                const discarded = high << one;
                const left_overshift = high << width;
                const right_overshift = high >> width;
                const huge_overshift = high << huge;
                var negative: i8 = -1;
                const sign_fill = negative >> width;
                var zero: int = 0;
                const unchanged = high << zero;
            }",
        )
        .unwrap();
        let entry = crate::ir::lower(crate::semantic::check(&syntax).unwrap())
            .verify()
            .unwrap();
        assert_native_values(
            &entry,
            &[128, 1, 8, 18446744073709551615, 0, 0, 0, 0, -1, -1, 0, 128],
        );
    }

    #[test]
    fn nested_runtime_failures_follow_left_to_right_operand_order() {
        let stderr = assert_native_failure(
            "var one: int = 1; var zero: int = 0; var negative: int = -1;
             const failed = (one / zero) + (one << negative);",
            "integer `/` has a zero divisor",
        );
        assert!(!stderr.contains("shift count"));

        let stderr = assert_native_failure(
            "var wide: u16 = 300; var one: int = 1; var zero: int = 0;
             const failed = int(u8(wide)) + (one / zero);",
            "checked integer conversion failed: `u16` to `u8`",
        );
        assert!(!stderr.contains("zero divisor"));
    }

    #[test]
    fn checked_conversions_trap_outside_each_destination_range() {
        let types = [
            Type::I8,
            Type::I16,
            Type::I32,
            Type::I64,
            Type::U8,
            Type::U16,
            Type::U32,
            Type::U64,
            Type::Int,
            Type::Uint,
        ];
        for source in types {
            let minimum = type_minimum(source);
            let maximum = i128::from(source.max());
            for destination in types {
                let lower = type_minimum(destination);
                let upper = i128::from(destination.max());
                let mut failures = vec![minimum, lower - 1, upper + 1, maximum];
                failures.retain(|value| {
                    (minimum..=maximum).contains(value) && !(lower..=upper).contains(value)
                });
                failures.sort_unstable();
                failures.dedup();
                for value in failures {
                    // Source execution also checks semantic classification and lowering.
                    // Truncation expresses negative values before unary syntax is supported.
                    let bits = value.rem_euclid(1i128 << source.width());
                    let text = format!(
                        "fn main() -> void {{ var value = {}.truncate({bits}); \
                         const result = {}(value); exit(42); }}",
                        source.name(),
                        destination.name(),
                    );
                    let syntax = crate::frontend::parse(&text).unwrap();
                    let entry = crate::ir::lower(crate::semantic::check(&syntax).unwrap())
                        .verify()
                        .unwrap();
                    let dir = tempfile::tempdir().unwrap();
                    let output = dir.path().join("program");
                    build_text(&emit(&entry, None), &output).unwrap();
                    let result = Command::new(output).output().unwrap();
                    assert!(!result.status.success(), "{text}");
                    assert!(
                        String::from_utf8_lossy(&result.stderr).contains(&format!(
                            "checked integer conversion failed: `{}` to `{}`",
                            source.name(),
                            destination.name(),
                        )),
                        "{text}: {:?}",
                        result,
                    );
                }
            }
        }
    }

    #[test]
    fn infallible_conversions_do_not_emit_trap_blocks_or_messages() {
        let syntax = crate::frontend::parse(
            "fn main() -> void { var x: u8 = 42; var y = int(x); exit(y); }",
        )
        .unwrap();
        let entry = crate::ir::lower(crate::semantic::check(&syntax).unwrap())
            .verify()
            .unwrap();
        let text = emit(&entry, None);
        assert!(!text.contains("$abort"));
        assert!(!text.contains("$write"));
        assert!(!text.contains("data $"));
        assert!(!text.contains("jnz"));
    }

    #[test]
    fn source_unsigned_high_bits_survive_assignment_copies_and_shadowing() {
        let syntax = crate::frontend::parse(
            "fn main() -> void {
            var x: u8 = 255;
            const saved = u64(x);
            { var x: u32 = 4294967295; const native = uint(x);
              x = 0; const wide = u64(native); }
            const medium = u16(x); const word = u32(medium);
            const wide = u64(word);
            x = 1;
            var result: u64 = 18446744073709551615;
            const copy = result;
            result = saved;
        }",
        )
        .unwrap();
        let checked = crate::semantic::check(&syntax).unwrap();
        assert_native_values(
            &crate::ir::lower(checked).verify().unwrap(),
            &[
                255,
                255,
                4294967295,
                4294967295,
                0,
                4294967295,
                255,
                255,
                255,
                1,
                18446744073709551615,
                18446744073709551615,
                255,
            ],
        );
    }

    #[test]
    fn negative_values_survive_chained_widening_and_exit() {
        for number in [-128, -1] {
            let entry = Entry {
                values: vec![
                    Value {
                        span: None,
                        ty: Type::I8,
                        kind: ValueKind::Copy(Operand::Integer {
                            value: number,
                            ty: Type::I8,
                        }),
                    },
                    Value {
                        span: None,
                        ty: Type::I16,
                        kind: ValueKind::Convert {
                            operand: Operand::Value(ValueId(0)),
                            truncating: false,
                        },
                    },
                    Value {
                        span: None,
                        ty: Type::I32,
                        kind: ValueKind::Convert {
                            operand: Operand::Value(ValueId(1)),
                            truncating: false,
                        },
                    },
                    Value {
                        span: None,
                        ty: Type::I64,
                        kind: ValueKind::Convert {
                            operand: Operand::Value(ValueId(2)),
                            truncating: false,
                        },
                    },
                    Value {
                        span: None,
                        ty: Type::Int,
                        kind: ValueKind::Convert {
                            operand: Operand::Value(ValueId(2)),
                            truncating: false,
                        },
                    },
                ],
                exit: Operand::Value(ValueId(4)),
            };
            let dir = tempfile::tempdir().unwrap();
            let output = dir.path().join("program");
            let verified = entry.verify().unwrap();
            build_text(&emit(&verified, None), &output).unwrap();
            assert_eq!(
                Command::new(output).status().unwrap().code(),
                Some((number & 255) as i32)
            );
            assert_native_values(&verified, &[number; 5]);
        }
    }

    #[test]
    fn negative_exit_values_are_masked_before_returning() {
        for (value, expected) in [(-1, 255), (i32::MIN, 0)] {
            for through_copy in [false, true] {
                let entry = Entry {
                    values: if through_copy {
                        vec![copy(integer(value)), copy(Operand::Value(ValueId(0)))]
                    } else {
                        vec![]
                    },
                    exit: if through_copy {
                        Operand::Value(ValueId(1))
                    } else {
                        integer(value)
                    },
                }
                .verify()
                .unwrap();
                let qbe = emit(&entry, None);
                let expected_operand = if through_copy {
                    "%v1".to_owned()
                } else {
                    value.to_string()
                };
                let expected_mask = if Type::Int.width() == 64 {
                    format!(
                        "%exit_status =w copy {expected_operand}\n    %status =w and %exit_status, 255\n    ret %status"
                    )
                } else {
                    format!("%status =w and {expected_operand}, 255\n    ret %status")
                };
                assert!(qbe.contains(&expected_mask));
                if through_copy {
                    let ty = qbe_type(Type::Int);
                    assert!(
                        qbe.contains(&format!("%v0 ={ty} copy {value}\n    %v1 ={ty} copy %v0"))
                    );
                }
                let dir = tempfile::tempdir().unwrap();
                let output = dir.path().join("program");
                build_text(&emit(&entry, None), &output).unwrap();
                assert_eq!(
                    Command::new(output).status().unwrap().code(),
                    Some(expected)
                );
            }
        }
    }
}
