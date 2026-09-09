use crate::{
    CompileError,
    diagnostic::{Diagnostic, DiagnosticRenderer},
    frontend::{BinaryOperator, ComparisonOperator, UnaryOperator},
    ir::{
        BinaryForm, BlockId, ControlFlow, Function, FunctionId, Global, Instruction, Operand,
        Place, Terminator, Value, ValueId, ValueKind, VerifiedProgram,
    },
    source::Source,
    types::Type,
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

struct Emitter<'a> {
    text: String,
    data: String,
    globals: &'a [Global],
    functions: &'a [Function],
    main: FunctionId,
    /// The index of the function being emitted, which distinguishes its
    /// module-global trap message symbols from every other function's.
    function: usize,
    /// The return type of the signature being emitted, absent for a function
    /// that returns nothing.
    qbe_result: Option<char>,
    diagnostics: Option<DiagnosticRenderer<'a>>,
}

/// Names the QBE symbol defining or calling function `id`. Fern source names
/// never reach the object file, so a function named `write` or `exit` cannot
/// collide with libc.
fn function_symbol(main: FunctionId, id: FunctionId) -> String {
    if id == main {
        "main".to_owned()
    } else {
        format!("fn{}", id.0)
    }
}

/// Names the QBE symbol holding `place`.
fn place(place: Place) -> String {
    match place {
        Place::Local(local) => format!("%local{}", local.0),
        Place::Global(global) => format!("$global{}", global.0),
    }
}

pub(crate) fn emit(verified: &VerifiedProgram, source_file: Option<&Source>) -> String {
    let program = verified.program();
    let mut emitter = Emitter {
        text: String::new(),
        data: String::new(),
        globals: &program.globals,
        functions: &program.functions,
        main: program.main,
        function: 0,
        qbe_result: None,
        diagnostics: source_file.map(DiagnosticRenderer::new),
    };
    for (id, global) in program.globals.iter().enumerate() {
        writeln!(
            emitter.data,
            "data $global{id} = {{ {} {} }}",
            qbe_type(global.ty),
            global.value
        )
        .unwrap();
    }
    for (index, function) in program.functions.iter().enumerate() {
        let id = FunctionId(index);
        let entry = id == program.main;
        emitter.function = index;
        // `main` reports the process status, so it returns a word even though
        // Fern declares it as returning nothing.
        emitter.qbe_result = if entry {
            Some('w')
        } else {
            function.result.map(qbe_type)
        };
        let parameters = (0..function.parameters)
            .map(|index| format!("{} %param{index}", qbe_type(function.flow.locals[index])))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            emitter.text,
            "{}function {}${}({parameters}) {{\n@start",
            if entry { "export " } else { "" },
            match emitter.qbe_result {
                Some(result) => format!("{result} "),
                None => String::new(),
            },
            function_symbol(program.main, id)
        )
        .unwrap();
        emit_control_flow(&mut emitter, function, &function.flow);
        emitter.text.push_str("}\n");
    }
    emitter.data + &emitter.text
}

fn emit_control_flow(emitter: &mut Emitter<'_>, function: &Function, flow: &ControlFlow) {
    for (id, ty) in flow.locals.iter().enumerate() {
        let (allocation, size) = if qbe_type(*ty) == 'l' {
            ("alloc8", 8)
        } else {
            ("alloc4", 4)
        };
        writeln!(emitter.text, "    %local{id} =l {allocation} {size}").unwrap();
    }
    // The parameters are the leading locals, so the loop above allocated them.
    for index in 0..function.parameters {
        writeln!(
            emitter.text,
            "    store{} %param{index}, %local{index}",
            qbe_type(flow.locals[index])
        )
        .unwrap();
    }
    writeln!(emitter.text, "    jmp @block{}", flow.entry.0).unwrap();

    for (block_id, block) in flow.blocks.iter().enumerate() {
        writeln!(emitter.text, "@block{block_id}").unwrap();
        for instruction in &block.instructions {
            match instruction {
                Instruction::Value(ValueId(id)) => emit_value(emitter, function, *id),
                Instruction::Store {
                    place: destination,
                    operand: source,
                } => {
                    let width = match destination {
                        Place::Local(local) => qbe_type(flow.locals[local.0]),
                        Place::Global(global) => qbe_type(emitter.globals[global.0].ty),
                    };
                    writeln!(
                        emitter.text,
                        "    store{width} {}, {}",
                        operand(*source),
                        place(*destination)
                    )
                    .unwrap();
                }
                Instruction::Call {
                    result,
                    function: callee,
                    arguments,
                    ..
                } => {
                    let target = &emitter.functions[callee.0];
                    let arguments = arguments
                        .iter()
                        .enumerate()
                        .map(|(index, argument)| {
                            format!(
                                "{} {}",
                                qbe_type(target.flow.locals[index]),
                                operand(*argument)
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    let symbol = function_symbol(emitter.main, *callee);
                    match result {
                        Some(ValueId(id)) => writeln!(
                            emitter.text,
                            "    %v{id} ={} call ${symbol}({arguments})",
                            qbe_type(
                                target
                                    .result
                                    .expect("a call result requires a callee that returns one")
                            )
                        ),
                        None => writeln!(emitter.text, "    call ${symbol}({arguments})"),
                    }
                    .unwrap();
                }
            }
        }
        emit_terminator(emitter, BlockId(block_id), &block.terminator);
    }
}

fn emit_terminator(emitter: &mut Emitter<'_>, block: BlockId, terminator: &Terminator) {
    match *terminator {
        Terminator::Jump { target } => {
            writeln!(emitter.text, "    jmp @block{}", target.0).unwrap();
        }
        Terminator::Branch {
            condition,
            then_target,
            else_target,
            ..
        } => {
            writeln!(
                emitter.text,
                "    jnz {}, @block{}, @block{}",
                operand(condition),
                then_target.0,
                else_target.0
            )
            .unwrap();
        }
        Terminator::Exit { status, .. } => {
            writeln!(
                emitter.text,
                "    %block{}_status =w and {}, 255",
                block.0,
                operand(status)
            )
            .unwrap();
            writeln!(emitter.text, "    call $exit(w %block{}_status)", block.0).unwrap();
            emitter.text.push_str("    hlt\n");
        }
        Terminator::Return { value } => match value {
            Some(value) => {
                writeln!(emitter.text, "    ret {}", operand(value)).unwrap();
            }
            // Falling off the end of `main` exits zero. Every other function
            // returning nothing has no QBE return type to give a value to.
            None => emitter.text.push_str(if emitter.qbe_result.is_some() {
                "    ret 0\n"
            } else {
                "    ret\n"
            }),
        },
        Terminator::Unreachable => emitter.text.push_str("    hlt\n"),
    }
}

fn emit_value(emitter: &mut Emitter<'_>, function: &Function, id: usize) {
    let value = &function.values[id];
    match value.kind {
        ValueKind::Load(source) => {
            let width = qbe_type(value.ty);
            writeln!(
                emitter.text,
                "    %v{id} ={width} load{width} {}",
                place(source)
            )
            .unwrap();
        }
        // A call result is defined by its call instruction.
        ValueKind::CallResult => {}
        ValueKind::Convert {
            operand: source,
            truncating,
        } => {
            let source_ty = operand_type(function, source);
            if truncating || source_ty.all_values_fit(value.ty) {
                emit_truncation(&mut emitter.text, id, source, source_ty, value.ty);
            } else {
                let message = format!(
                    "checked integer conversion failed: `{}` to `{}`",
                    source_ty.name(),
                    value.ty.name()
                );
                let message = operation_message(value, emitter.diagnostics.as_ref(), &message);
                emit_checked_conversion(emitter, id, source, source_ty, value.ty, message);
            }
        }
        ValueKind::Unary { operator, operand } => {
            emit_unary_operation(emitter, id, value, operator, operand);
        }
        ValueKind::Binary {
            operator,
            form,
            left,
            right,
        } => emit_binary_operation(emitter, id, value, operator, form, (left, right), function),
        ValueKind::Comparison {
            operator,
            left,
            right,
        } => emit_comparison(&mut emitter.text, id, operator, left, right, function),
        ValueKind::LogicalNot { operand: source } => {
            writeln!(emitter.text, "    %v{id} =w ceqw {}, 0", operand(source)).unwrap();
        }
    }
}

fn emit_comparison(
    text: &mut String,
    id: usize,
    operator: ComparisonOperator,
    left: Operand,
    right: Operand,
    function: &Function,
) {
    let ty = operand_type(function, left);
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

fn comparison_operation(relation: &str, signed: bool, ty: Type) -> String {
    format!(
        "c{}{relation}{}",
        if signed { "s" } else { "u" },
        qbe_type(ty)
    )
}

fn emit_unary_operation(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    operator: UnaryOperator,
    source: Operand,
) {
    let ty = value.ty;
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
                    value,
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

fn emit_binary_operation(
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
            emit_binary_raw(&mut emitter.text, id, value.ty, instruction, left, right);
            emit_normalized(
                &mut emitter.text,
                id,
                &format!("%operation{id}_raw"),
                value.ty,
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
            emit_binary_raw(&mut emitter.text, id, value.ty, instruction, left, right);
            emit_normalized(
                &mut emitter.text,
                id,
                &format!("%operation{id}_raw"),
                value.ty,
            );
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

fn emit_checked_arithmetic(
    emitter: &mut Emitter<'_>,
    id: usize,
    value: &Value,
    operator: BinaryOperator,
    spelling: &str,
    left: Operand,
    right: Operand,
) {
    let ty = value.ty;
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
            value,
            emitter.diagnostics.as_ref(),
            &format!("integer `{spelling}` overflowed"),
        ),
    );
    emit_normalized(&mut emitter.text, id, &format!("%operation{id}_raw"), ty);
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
    let ty = value.ty;
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
            value,
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
                value,
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
    let ty = value.ty;
    let count_ty = operand_type(function, right);
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
            operation_message(value, emitter.diagnostics.as_ref(), &message),
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

fn emit_normalized(text: &mut String, id: usize, source: &str, ty: Type) {
    emit_truncation_operand(text, id, source, ty, ty);
}

fn operation_message(
    value: &Value,
    diagnostic_renderer: Option<&DiagnosticRenderer<'_>>,
    message: &str,
) -> String {
    match (&value.span, diagnostic_renderer) {
        (Some(span), Some(renderer)) => renderer.render(&Diagnostic::new(span.clone(), message)),
        _ => format!("{message}\n"),
    }
}

fn emit_conditional_trap(
    text: &mut String,
    data: &mut String,
    function: usize,
    id: usize,
    cause: &str,
    condition: &str,
    message: String,
) {
    let failed = format!("operation{id}_{cause}_failed");
    let ready = format!("operation{id}_{cause}_ready");
    // Value IDs are function-local, so the module-global message symbol needs
    // the function to stay unique.
    let symbol = format!("fern_function{function}_operation{id}_{cause}_message");
    writeln!(text, "    jnz {condition}, @{failed}, @{ready}").unwrap();
    writeln!(text, "@{failed}").unwrap();
    emit_message_data(data, &symbol, &message);
    writeln!(
        text,
        "    call $write(w 2, l ${symbol}, l {})",
        message.len(),
    )
    .unwrap();
    text.push_str("    call $abort()\n    hlt\n");
    writeln!(text, "@{ready}").unwrap();
}

fn emit_message_data(data: &mut String, symbol: &str, message: &str) {
    write!(data, "data ${symbol} = {{ ").unwrap();
    let bytes = message.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if matches!(bytes[index], b' '..=b'~') && !matches!(bytes[index], b'"' | b'\\') {
            let start = index;
            while index < bytes.len()
                && matches!(bytes[index], b' '..=b'~')
                && !matches!(bytes[index], b'"' | b'\\')
            {
                index += 1;
            }
            write!(
                data,
                "b \"{}\", ",
                std::str::from_utf8(&bytes[start..index]).expect("printable ASCII is UTF-8")
            )
            .unwrap();
        } else {
            write!(data, "b {}, ", bytes[index]).unwrap();
            index += 1;
        }
    }
    data.push_str("b 0 }\n");
}

fn operand_type(function: &Function, operand: Operand) -> Type {
    match operand {
        Operand::Integer { ty, .. } => ty,
        Operand::Value(ValueId(id)) => function.values[id].ty,
    }
}

fn emit_checked_conversion(
    emitter: &mut Emitter<'_>,
    id: usize,
    source: Operand,
    source_ty: Type,
    destination: Type,
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

pub(crate) fn build(
    program: &VerifiedProgram,
    source: &Source,
    output: &Path,
) -> Result<(), CompileError> {
    build_text(&emit(program, Some(source)), output)
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
    use crate::ir::Program;
    use std::path::PathBuf;

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
            kind: ValueKind::Convert {
                operand,
                truncating: false,
            },
        }
    }

    fn program(values: Vec<Value>, exit: Operand) -> Program {
        Program {
            globals: vec![],
            main: FunctionId(0),
            functions: vec![Function {
                parameters: 0,
                result: None,
                flow: ControlFlow {
                    entry: BlockId(0),
                    locals: vec![],
                    blocks: vec![crate::ir::Block {
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
        let syntax = crate::frontend::parse(text).unwrap();
        crate::ir::lower(crate::semantic::check(&syntax).unwrap())
            .verify()
            .unwrap()
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

    #[test]
    fn message_data_uses_quoted_runs_and_numeric_exception_bytes() {
        let message = format!("{}\"\\é\n", "printable text ".repeat(1_000));
        let mut data = String::new();
        emit_message_data(&mut data, "message", &message);
        assert!(data.len() < message.len() + 100);
        assert!(data.contains("b \"printable text printable text"));
        for byte in [b'"', b'\\', 0xc3, 0xa9, b'\n'] {
            assert!(data.contains(&format!("b {byte},")));
        }

        let mut qbe = data;
        qbe.push_str("export function w $main() {\n@start\n    ret 0\n}\n");
        let dir = tempfile::tempdir().unwrap();
        build_text(&qbe, &dir.path().join("program")).unwrap();
    }

    #[test]
    fn many_traps_do_not_expand_qbe_byte_by_byte() {
        let mut text = String::from("fn main() -> void {\nvar left = 1; var right = 2;\n");
        for id in 0..3_000 {
            writeln!(text, "const value{id} = left + right;").unwrap();
        }
        text.push_str("}\n");
        let program = lowered(&text);
        let source = Source {
            path: PathBuf::from("many_traps.fern"),
            text,
        };
        let qbe = emit(&program, Some(&source));
        assert!(qbe.len() < 5_000_000, "QBE was {} bytes", qbe.len());
        assert!(qbe.contains("b \"Error"));
    }

    #[test]
    fn full_width_integer_copies_and_conversions_execute() {
        let ranges = Type::ALL_INTEGERS.map(|ty| (ty, ty.min(), i128::from(ty.max())));
        let mut values = vec![];
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
                    let id = values.len();
                    values.push(Value {
                        span: None,
                        ty: source,
                        kind: ValueKind::Convert {
                            operand: literal,
                            truncating: false,
                        },
                    });
                    expected.push(number);
                    for operand in [literal, Operand::Value(ValueId(id))] {
                        // Conversions between types carry the span their range trap reports.
                        values.push(Value {
                            span: Some(0..1),
                            ty: destination,
                            kind: ValueKind::Convert {
                                operand,
                                truncating: false,
                            },
                        });
                        expected.push(number);
                        let copied = Operand::Value(ValueId(values.len() - 1));
                        values.push(Value {
                            span: Some(0..1),
                            ty: destination,
                            kind: ValueKind::Convert {
                                operand: copied,
                                truncating: false,
                            },
                        });
                        expected.push(number);
                    }
                }
            }
        }
        assert_native_values(&program(values, integer(0)).verify().unwrap(), &expected);
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
        let types = Type::ALL_INTEGERS;
        let mut values = vec![];
        let mut expected = Vec::new();
        for source in types {
            let minimum = source.min();
            let maximum = i128::from(source.max());
            for value in [minimum, if source.signed() { -1 } else { 1 }, maximum] {
                let source_id = values.len();
                values.push(Value {
                    span: None,
                    ty: source,
                    kind: ValueKind::Convert {
                        operand: Operand::Integer { value, ty: source },
                        truncating: false,
                    },
                });
                expected.push(value);
                for destination in types {
                    values.push(Value {
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
        assert_native_values(&program(values, integer(0)).verify().unwrap(), &expected);
    }

    #[test]
    fn native_integer_operations_preserve_values_for_every_type() {
        let types = Type::ALL_INTEGERS;
        for ty in types {
            let name = ty.name();
            let minimum = ty.min();
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
                    const wrapping_add = a +% b; const wrapping_subtract = a -% b;
                    const wrapping_multiply = a *% b;
                    {signed_operations} const wrapping_negate = -%a; const complement = ^a;
                    const and = a & b; const and_not = a & ^b;
                    const xor = a ^ b; const or = a | b;
                    const shift_left = a << b; const shift_right = a >> b;
                    var maximum: {name} = {maximum}; var minimum: {name} = {minimum};
                    var one: {name} = 1;
                    const wrapped_maximum = maximum +% one;
                    const wrapped_minimum = minimum -% one;
                    const wrapped_product = maximum *% b;
                }}",
                name = name,
            );
            let program = lowered(&text);
            let mut expected = vec![];
            for result in [23, 17, 60, 6, 2, 23, 17, 60] {
                expected.extend([20, 3, result]);
            }
            if ty.signed() {
                expected.extend([20, -20, -7, 3, -2, -7, 3, -1]);
            }
            expected.extend([
                20,
                truncated(-20, ty),
                20,
                truncated(!20, ty),
                20,
                3,
                20 & 3,
                20,
                3,
                truncated(!3, ty),
                20 & !3,
                20,
                3,
                20 ^ 3,
                20,
                3,
                20 | 3,
                20,
                3,
                truncated(20 << 3, ty),
                20,
                3,
                20 >> 3,
                maximum,
                1,
                truncated(maximum + 1, ty),
                minimum,
                1,
                truncated(minimum - 1, ty),
                maximum,
                3,
                truncated(maximum * 3, ty),
            ]);
            assert_native_values(&program, &expected);
        }
    }

    #[test]
    fn checked_arithmetic_traps_at_each_integer_width() {
        let types = Type::ALL_INTEGERS;
        for ty in types {
            let name = ty.name();
            let minimum = ty.min();
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
        let types = Type::ALL_INTEGERS;
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
                let minimum = ty.min();
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
        let program = crate::ir::lower(crate::semantic::check(&syntax).unwrap())
            .verify()
            .unwrap();
        assert_native_values(
            &program,
            &[
                128,
                1,
                0,
                128,
                8,
                0,
                128,
                8,
                0,
                128,
                18446744073709551615,
                0,
                -1,
                8,
                -1,
                128,
                0,
                128,
            ],
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
        let types = Type::ALL_INTEGERS;
        for source in types {
            let minimum = source.min();
            let maximum = i128::from(source.max());
            for destination in types {
                let lower = destination.min();
                let upper = i128::from(destination.max());
                let mut failures = vec![minimum, lower - 1, upper + 1, maximum];
                failures.retain(|value| {
                    (minimum..=maximum).contains(value) && !(lower..=upper).contains(value)
                });
                failures.sort_unstable();
                failures.dedup();
                for value in failures {
                    // Source execution also checks semantic classification and lowering.
                    // Truncation constructs every source bit pattern uniformly.
                    let bits = value.rem_euclid(1i128 << source.width());
                    let text = format!(
                        "fn main() -> void {{ var value = {}.truncate({bits}); \
                         const result = {}(value); exit(42); }}",
                        source.name(),
                        destination.name(),
                    );
                    let program = lowered(&text);
                    let dir = tempfile::tempdir().unwrap();
                    let output = dir.path().join("program");
                    build_text(&emit(&program, None), &output).unwrap();
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
        let program = crate::ir::lower(crate::semantic::check(&syntax).unwrap())
            .verify()
            .unwrap();
        let text = emit(&program, None);
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
                4294967295,
                4294967295,
                255,
                255,
                255,
                255,
                255,
                255,
                18446744073709551615,
                255,
            ],
        );
    }

    #[test]
    fn negative_values_survive_chained_widening_and_exit() {
        for number in [-128, -1] {
            let program = program(
                vec![
                    Value {
                        span: None,
                        ty: Type::I8,
                        kind: ValueKind::Convert {
                            operand: Operand::Integer {
                                value: number,
                                ty: Type::I8,
                            },
                            truncating: false,
                        },
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
                Operand::Value(ValueId(4)),
            );
            let dir = tempfile::tempdir().unwrap();
            let output = dir.path().join("program");
            let verified = program.verify().unwrap();
            build_text(&emit(&verified, None), &output).unwrap();
            assert_eq!(
                Command::new(output).status().unwrap().code(),
                Some((number & 255) as i32)
            );
            assert_native_values(&verified, &[number; 5]);
        }
    }

    #[test]
    fn traps_at_one_value_id_in_two_functions_link_separately() {
        let program = lowered(
            "fn divide(left: int, right: int) -> int { return left / right; }
             fn modulo(left: int, right: int) -> int { return left % right; }
             fn main() -> void { exit(divide(84, 2) - modulo(84, 42)); }",
        );
        let qbe = emit(&program, None);
        let symbols: Vec<&str> = qbe
            .lines()
            .filter(|line| line.starts_with("data $fern_"))
            .map(|line| line.split(' ').nth(1).expect("a named data definition"))
            .collect();
        // Both callees trap on the same value ID, so only the function keeps
        // their message symbols apart.
        for symbol in [
            "$fern_function0_operation2_zero_message",
            "$fern_function1_operation2_zero_message",
        ] {
            assert!(symbols.contains(&symbol), "{symbols:?}");
        }
        let unique: std::collections::BTreeSet<&&str> = symbols.iter().collect();
        assert_eq!(unique.len(), symbols.len(), "{symbols:?}");

        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("program");
        build_text(&qbe, &output).unwrap();
        assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
    }

    #[test]
    fn negative_exit_values_are_masked_before_returning() {
        for (value, expected) in [(-1, 255), (i32::MIN, 0)] {
            for through_copy in [false, true] {
                let program = program(
                    if through_copy {
                        vec![copy(integer(value)), copy(Operand::Value(ValueId(0)))]
                    } else {
                        vec![]
                    },
                    if through_copy {
                        Operand::Value(ValueId(1))
                    } else {
                        integer(value)
                    },
                )
                .verify()
                .unwrap();
                let qbe = emit(&program, None);
                let expected_operand = if through_copy {
                    "%v1".to_owned()
                } else {
                    value.to_string()
                };
                let expected_mask = format!(
                    "%block0_status =w and {expected_operand}, 255\n    call $exit(w %block0_status)"
                );
                assert!(qbe.contains(&expected_mask));
                if through_copy {
                    let ty = qbe_type(Type::Int);
                    assert!(
                        qbe.contains(&format!("%v0 ={ty} copy {value}\n    %v1 ={ty} copy %v0"))
                    );
                }
                let dir = tempfile::tempdir().unwrap();
                let output = dir.path().join("program");
                build_text(&emit(&program, None), &output).unwrap();
                assert_eq!(
                    Command::new(output).status().unwrap().code(),
                    Some(expected)
                );
            }
        }
    }
}
