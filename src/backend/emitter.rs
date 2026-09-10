//! QBE program, function, block, instruction, place, aggregate, and data emission.

use crate::{
    diagnostic::DiagnosticRenderer,
    ir::model::{
        BlockId, ControlFlow, Function, FunctionId, Instruction, Literal, Operand, Place,
        Terminator, ValueId, ValueKind,
    },
    ir::verify::VerifiedProgram,
    source::SourceMap,
    types::{Scalar, Type},
};
use std::fmt::Write;

use super::{floating, integer::*, qbe::*};

fn allocation(ty: &Type) -> &'static str {
    if bytes(ty.leaf()) == 8 {
        "alloc8"
    } else {
        "alloc4"
    }
}

/// How a signature, a call argument, or a call result names this type. An
/// array is an aggregate named by its layout, so two array types made of the
/// same scalars share one QBE type definition.
fn class(ty: &Type) -> String {
    match ty {
        Type::Scalar(ty) => qbe_type(*ty).to_string(),
        Type::Array { .. } => format!(":array{}{}", word(ty), ty.element_count()),
    }
}

/// The aggregate type definitions the signatures name. QBE needs the layout of
/// every array a function takes or returns.
fn emit_aggregate_types(functions: &[Function]) -> String {
    let mut definitions: Vec<String> = Vec::new();
    for function in functions {
        let signature = function.flow.locals[..function.parameters]
            .iter()
            .chain(function.result.as_ref());
        for ty in signature.filter(|ty| ty.scalar().is_none()) {
            let definition = format!(
                "type {} = {{ {} {} }}\n",
                class(ty),
                word(ty),
                ty.element_count()
            );
            if !definitions.contains(&definition) {
                definitions.push(definition);
            }
        }
    }
    definitions.concat()
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

/// The address of a place and the type stored there. An element place emits
/// its bounds check here, so every read and write through it is checked.
fn place_address(emitter: &mut Emitter<'_>, flow: &ControlFlow, place: &Place) -> (String, Type) {
    match place {
        Place::Local(local) => (format!("%local{}", local.0), flow.locals[local.0].clone()),
        Place::Global(global) => (
            format!("$global{}", global.0),
            emitter.globals[global.0].ty.clone(),
        ),
        Place::Element { base, index, span } => {
            let (base, ty) = place_address(emitter, flow, base);
            let spelling = ty.to_string();
            let Type::Array { length, element } = ty else {
                unreachable!("a verified element place indexes an array")
            };
            let access = emitter.accesses;
            emitter.accesses += 1;
            // A constant index the checker accepted is in range, so only an
            // index computed at run time needs the check. Lowering writes an
            // array literal through constant indices, and a check for each of
            // them would carry a trap block and a message that cannot run.
            let constant = match index {
                Operand::Literal(Literal::Integer { value, .. }) => {
                    (0..i128::from(length)).contains(value)
                }
                Operand::Literal(Literal::Floating(_)) => {
                    unreachable!("a verified index has type `int`")
                }
                Operand::Value(_) => false,
            };
            let index = operand(*index);
            if !constant {
                // Indices are signed, so one unsigned comparison rejects both
                // a negative index and one at or past the length.
                writeln!(
                    emitter.text,
                    "    %access{access}_out =w {} {index}, {length}",
                    comparison_operation("ge", false, Scalar::Int)
                )
                .unwrap();
                let message = operation_message(
                    Some(span),
                    emitter.diagnostics.as_ref(),
                    &format!("array index out of range for `{spelling}`"),
                );
                emit_conditional_trap(
                    &mut emitter.text,
                    &mut emitter.data,
                    emitter.function,
                    access,
                    "bounds",
                    &format!("%access{access}_out"),
                    message,
                );
            }
            writeln!(
                emitter.text,
                "    %access{access}_offset =l mul {index}, {}",
                size(&element)
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %access{access}_address =l add {base}, %access{access}_offset"
            )
            .unwrap();
            (format!("%access{access}_address"), *element)
        }
    }
}

pub(super) fn emit(verified: &VerifiedProgram, sources: Option<&SourceMap>) -> String {
    let program = verified.program();
    let mut emitter = Emitter {
        text: String::new(),
        data: String::new(),
        globals: &program.globals,
        functions: &program.functions,
        main: program.main,
        function: 0,
        qbe_result: None,
        accesses: 0,
        diagnostics: sources.map(DiagnosticRenderer::new),
    };
    for (id, global) in program.globals.iter().enumerate() {
        // An i128, its QBE word, and a separator fit in 44 bytes. Reserve the
        // emitted data once instead of allocating one temporary string per scalar.
        emitter.data.reserve(global.values.len().saturating_mul(44));
        write!(emitter.data, "data $global{id} = {{ ").unwrap();
        for (index, &value) in global.values.iter().enumerate() {
            if index != 0 {
                emitter.data.push_str(", ");
            }
            emitter.data.push_str(&data_item(value));
        }
        emitter.data.push_str(" }\n");
    }
    for (index, function) in program.functions.iter().enumerate() {
        let id = FunctionId(index);
        let entry = id == program.main;
        emitter.function = index;
        // `main` reports the process status, so it returns a word even though
        // Fern declares it as returning nothing.
        emitter.qbe_result = if entry {
            Some("w".to_owned())
        } else {
            function.result.as_ref().map(class)
        };
        let parameters = (0..function.parameters)
            .map(|index| format!("{} %param{index}", class(&function.flow.locals[index])))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            emitter.text,
            "{}function {}${}({parameters}) {{\n@start",
            if entry { "export " } else { "" },
            match &emitter.qbe_result {
                Some(result) => format!("{result} "),
                None => String::new(),
            },
            function_symbol(program.main, id)
        )
        .unwrap();
        emit_control_flow(&mut emitter, function, &function.flow);
        emitter.text.push_str("}\n");
    }
    emit_aggregate_types(&program.functions) + &emitter.data + &emitter.text
}

fn emit_control_flow(emitter: &mut Emitter<'_>, function: &Function, flow: &ControlFlow) {
    for (id, ty) in flow.locals.iter().enumerate() {
        writeln!(
            emitter.text,
            "    %local{id} =l {} {}",
            allocation(ty),
            size(ty)
        )
        .unwrap();
    }
    // An array value holds its own copy of the array. QBE allocates once per
    // `alloc` it executes, so the storage is allocated here rather than where
    // the load runs, which may be inside a loop.
    for (id, value) in function.values.iter().enumerate() {
        if matches!(value.kind, ValueKind::Load(_)) && value.ty.scalar().is_none() {
            writeln!(
                emitter.text,
                "    %v{id}_storage =l {} {}",
                allocation(&value.ty),
                size(&value.ty)
            )
            .unwrap();
        }
    }
    // The parameters are the leading locals, so the loop above allocated them.
    for index in 0..function.parameters {
        let ty = &flow.locals[index];
        if ty.scalar().is_some() {
            writeln!(
                emitter.text,
                "    store{} %param{index}, %local{index}",
                word(ty)
            )
        } else {
            writeln!(
                emitter.text,
                "    blit %param{index}, %local{index}, {}",
                size(ty)
            )
        }
        .unwrap();
    }
    writeln!(emitter.text, "    jmp @block{}", flow.entry.0).unwrap();

    for (block_id, block) in flow.blocks.iter().enumerate() {
        writeln!(emitter.text, "@block{block_id}").unwrap();
        for instruction in &block.instructions {
            emit_instruction(emitter, function, flow, instruction);
        }
        emit_terminator(emitter, BlockId(block_id), &block.terminator);
    }
}

fn emit_instruction(
    emitter: &mut Emitter<'_>,
    function: &Function,
    flow: &ControlFlow,
    instruction: &Instruction,
) {
    match instruction {
        Instruction::Value(ValueId(id)) => emit_value(emitter, function, *id),
        Instruction::Store {
            place: destination,
            operand: source,
        } => {
            let (address, ty) = place_address(emitter, flow, destination);
            let source = operand(*source);
            if ty.scalar().is_some() {
                writeln!(emitter.text, "    store{} {source}, {address}", word(&ty))
            } else {
                writeln!(emitter.text, "    blit {source}, {address}, {}", size(&ty))
            }
            .unwrap();
        }
        Instruction::Call {
            result,
            function: callee,
            arguments,
            ..
        } => emit_call(emitter, *callee, arguments, *result),
    }
}

fn emit_call(
    emitter: &mut Emitter<'_>,
    callee: FunctionId,
    arguments: &[Operand],
    result: Option<ValueId>,
) {
    let target = &emitter.functions[callee.0];
    // The callee's leading locals hold its parameters, so they give each
    // argument its type.
    let arguments = arguments
        .iter()
        .enumerate()
        .map(|(index, argument)| {
            format!(
                "{} {}",
                class(&target.flow.locals[index]),
                operand(*argument)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let symbol = function_symbol(emitter.main, callee);
    match result {
        Some(ValueId(id)) => writeln!(
            emitter.text,
            "    %v{id} ={} call ${symbol}({arguments})",
            class(
                target
                    .result
                    .as_ref()
                    .expect("a call result requires a callee that returns one")
            )
        ),
        None => writeln!(emitter.text, "    call ${symbol}({arguments})"),
    }
    .unwrap();
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
        ValueKind::Load(ref source) => {
            let (address, _) = place_address(emitter, &function.flow, source);
            if value.ty.scalar().is_some() {
                let width = word(&value.ty);
                writeln!(emitter.text, "    %v{id} ={width} load{width} {address}")
            } else {
                // Loading an array copies it, so a later call in the same
                // expression cannot change what the value holds.
                writeln!(
                    emitter.text,
                    "    blit {address}, %v{id}_storage, {}\n    %v{id} =l copy %v{id}_storage",
                    size(&value.ty)
                )
            }
            .unwrap();
        }
        // A call result is defined by its call instruction.
        ValueKind::CallResult => {}
        ValueKind::Convert {
            operand: source,
            truncating,
        } => {
            let ty = scalar(&value.ty);
            let source_ty = operand_scalar(function, source);
            if ty.is_floating() || source_ty.is_floating() {
                floating::emit_conversion(emitter, id, value, source, source_ty, ty);
            } else if truncating || source_ty.all_values_fit(ty) {
                emit_truncation(&mut emitter.text, id, source, source_ty, ty);
            } else {
                let message = conversion_message(emitter, value, source_ty, ty);
                emit_checked_conversion(emitter, id, source, source_ty, ty, message);
            }
        }
        ValueKind::Unary { operator, operand } => {
            let ty = scalar(&value.ty);
            if ty.is_floating() {
                floating::emit_unary(&mut emitter.text, id, ty, operand);
            } else {
                emit_unary_operation(emitter, id, value, operator, operand);
            }
        }
        ValueKind::Binary {
            operator,
            form,
            left,
            right,
        } => {
            let ty = scalar(&value.ty);
            if ty.is_floating() {
                floating::emit_binary(&mut emitter.text, id, ty, operator, left, right);
            } else {
                emit_binary_operation(emitter, id, value, operator, form, (left, right), function);
            }
        }
        ValueKind::Comparison {
            operator,
            left,
            right,
        } => {
            let operand_ty = operand_type(function, left);
            if operand_ty.is_floating() {
                floating::emit_comparison(
                    &mut emitter.text,
                    id,
                    operator,
                    left,
                    right,
                    &operand_ty,
                );
            } else {
                emit_comparison(&mut emitter.text, id, operator, left, right, &operand_ty);
            }
        }
        ValueKind::LogicalNot { operand: source } => {
            writeln!(emitter.text, "    %v{id} =w ceqw {}, 0", operand(source)).unwrap();
        }
    }
}
