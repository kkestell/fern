//! QBE program, function, block, instruction, place, aggregate, and data emission.
//!
//! Generated code does not grow with an aggregate's length. Aggregates are
//! passed and returned through addresses rather than QBE aggregate types, and
//! are copied, zeroed, and filled with C library calls or loops. A callee reads
//! an aggregate parameter from the storage its caller passed, which is copied
//! only when lowering loaded the argument rather than borrowing it. Every Fern
//! call checks the available stack before entering the callee.

use crate::{
    diagnostic::DiagnosticRenderer,
    ir::model::{
        BlockId, ConstantId, ControlFlow, Function, FunctionId, Global, Instruction, Literal,
        Operand, Place, Terminator, ValueId, ValueKind,
    },
    ir::verify::VerifiedProgram,
    source::SourceMap,
    types::{Scalar, Type},
};
use std::fmt::Write;

use super::{floating, integer::*, layout::Layout, qbe::*};

/// The `alloc` an aggregate or a scalar needs. QBE has one instruction per
/// alignment, and no Fern value is aligned more strictly than eight bytes.
fn allocation(layout: &Layout<'_>, ty: &Type) -> &'static str {
    if layout.alignment(ty) == 8 {
        "alloc8"
    } else {
        "alloc4"
    }
}

/// QBE expands `blit` into one load and one store per word, so a copy is
/// emitted inline only up to this size. A longer copy calls `memcpy`, whose code
/// does not grow with the copy.
const MAX_INLINE_COPY_BYTES: u64 = 64;

/// Copies `size` bytes from `source` to `destination`, which do not overlap.
fn emit_copy(text: &mut String, source: &str, destination: &str, size: u64) {
    if size <= MAX_INLINE_COPY_BYTES {
        writeln!(text, "    blit {source}, {destination}, {size}")
    } else {
        writeln!(
            text,
            "    call $memcpy(l {destination}, l {source}, l {size})"
        )
    }
    .unwrap();
}

/// Mach-O arm64 relocations hold a signed 24-bit addend, and QBE folds a
/// constant offset into static data's relocation. Larger static data is
/// addressed through a pointer cell, whose load QBE cannot fold.
const MAX_DIRECT_STATIC_BYTES: u64 = 1 << 23;

/// Names the static data holding constant `id` of function `function`.
fn constant_symbol(function: usize, ConstantId(id): ConstantId) -> String {
    format!("function{function}_constant{id}")
}

/// Writes static data. Its scalars sit where the layout puts them, so the gaps
/// between them and the padding after the last one are zeroed, and data with
/// no values is zero throughout.
fn emit_data(emitter: &mut Emitter<'_>, symbol: &str, global: &Global) {
    let mut items = Vec::new();
    let mut written = 0;
    if let Some(values) = &global.values {
        for ((offset, ty), value) in emitter
            .layout
            .scalar_slots(&global.ty)
            .into_iter()
            .zip(values)
        {
            if offset != written {
                items.push(format!("z {}", offset - written));
            }
            items.push(data_item(value.clone()));
            written = offset + emitter.layout.size(&ty);
        }
    }
    let size = emitter.layout.size(&global.ty);
    if size != written {
        items.push(format!("z {}", size - written));
    }
    writeln!(emitter.data, "data ${symbol} = {{ {} }}", items.join(", ")).unwrap();
    if size > MAX_DIRECT_STATIC_BYTES {
        writeln!(emitter.data, "data ${symbol}_address = {{ l ${symbol} }}").unwrap();
    }
}

/// The address of static data, read from its pointer cell when the data is
/// too large to address directly.
fn static_address(emitter: &mut Emitter<'_>, symbol: String, ty: Type) -> (String, Type) {
    if emitter.layout.size(&ty) <= MAX_DIRECT_STATIC_BYTES {
        return (format!("${symbol}"), ty);
    }
    let access = emitter.accesses;
    emitter.accesses += 1;
    writeln!(
        emitter.text,
        "    %access{access}_static =l loadl ${symbol}_address"
    )
    .unwrap();
    (format!("%access{access}_static"), ty)
}

/// Names the QBE symbol defining or calling function `id`. Fern source names
/// never reach the object file, so a function named `write` or `exit` cannot
/// collide with libc.
fn function_symbol(id: FunctionId) -> String {
    format!("fn{}", id.0)
}

/// The address of a place and the type stored there. An element place emits
/// its bounds check here, so every read and write through it is checked.
fn place_address(emitter: &mut Emitter<'_>, function: &Function, place: &Place) -> (String, Type) {
    let flow = &function.flow;
    match place {
        // An aggregate parameter is never written, so it is read from the
        // storage the caller passed rather than from a copy.
        Place::Local(local)
            if local.0 < function.parameters && is_aggregate(&flow.locals[local.0]) =>
        {
            (format!("%param{}", local.0), flow.locals[local.0].clone())
        }
        Place::Local(local) => (format!("%local{}", local.0), flow.locals[local.0].clone()),
        Place::Global(global) => {
            let ty = emitter.globals[global.0].ty.clone();
            static_address(emitter, format!("global{}", global.0), ty)
        }
        Place::Constant(constant) => {
            let symbol = constant_symbol(emitter.function, *constant);
            static_address(emitter, symbol, function.constants[constant.0].ty.clone())
        }
        Place::Element { base, index, span } => {
            let (base, ty) = place_address(emitter, function, base);
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
            let index = operand(index.clone());
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
                emitter.layout.size(&element)
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %access{access}_address =l add {base}, %access{access}_offset"
            )
            .unwrap();
            (format!("%access{access}_address"), *element)
        }
        Place::Field { base, ordinal } => {
            let (base, ty) = place_address(emitter, function, base);
            let Type::Struct(declared) = ty else {
                unreachable!("a verified field place selects a field of a struct")
            };
            let (offset, field) = emitter.layout.field(declared.id, *ordinal);
            let field = field.clone();
            let access = emitter.accesses;
            emitter.accesses += 1;
            writeln!(
                emitter.text,
                "    %access{access}_address =l add {base}, {offset}"
            )
            .unwrap();
            (format!("%access{access}_address"), field)
        }
    }
}

pub(super) fn emit(verified: &VerifiedProgram, sources: Option<&SourceMap>) -> String {
    let program = verified.program();
    let layout = Layout::new(&program.structs);
    let storage_bytes = program
        .functions
        .iter()
        .map(|function| storage_bytes(&layout, function))
        .collect();
    let mut emitter = Emitter {
        text: String::new(),
        data: String::new(),
        layout,
        globals: &program.globals,
        functions: &program.functions,
        main: program.main,
        storage_bytes,
        stack_checks: 0,
        function: 0,
        qbe_result: None,
        returned_calls: Vec::new(),
        accesses: 0,
        comparisons: 0,
        fills: 0,
        diagnostics: sources.map(DiagnosticRenderer::new),
    };
    emit_entry(&mut emitter, &program.main_span);
    for (id, global) in program.globals.iter().enumerate() {
        emit_data(&mut emitter, &format!("global{id}"), global);
    }
    for (index, function) in program.functions.iter().enumerate() {
        let id = FunctionId(index);
        let entry = id == program.main;
        emitter.function = index;
        for (constant, data) in function.constants.iter().enumerate() {
            emit_data(
                &mut emitter,
                &constant_symbol(index, ConstantId(constant)),
                data,
            );
        }
        // `main` reports the process status, so it returns a word even though
        // Fern declares it as returning nothing. An aggregate result is
        // written through the address the caller passes first.
        emitter.qbe_result = if entry {
            Some('w')
        } else {
            function
                .result
                .as_ref()
                .and_then(|ty| ty.scalar())
                .map(qbe_type)
        };
        emitter.returned_calls = returned_calls(function);
        let result = function
            .result
            .as_ref()
            .filter(|ty| ty.scalar().is_none())
            .map(|_| "l %result".to_owned());
        let parameters = result
            .into_iter()
            .chain((0..function.parameters).map(|index| {
                format!(
                    "{} %param{index}",
                    emitter.layout.class(&function.flow.locals[index])
                )
            }))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            emitter.text,
            "function {}${}({parameters}) {{\n@start",
            match &emitter.qbe_result {
                Some(result) => format!("{result} "),
                None => String::new(),
            },
            function_symbol(id)
        )
        .unwrap();
        emit_control_flow(&mut emitter, function, &function.flow);
        emitter.text.push_str("}\n");
    }
    emitter.data + &emitter.text
}

/// The aggregate call results the function's `return`s return. A value is
/// used only in the block defining it, so each call shares its block with the
/// `return`, and nothing reads `%result` before the function returns.
fn returned_calls(function: &Function) -> Vec<ValueId> {
    function
        .flow
        .blocks
        .iter()
        .filter_map(|block| match &block.terminator {
            Terminator::Return {
                value: Some(Operand::Value(id)),
            } if function.values[id.0].kind == ValueKind::CallResult
                && is_aggregate(&function.values[id.0].ty) =>
            {
                Some(*id)
            }
            _ => None,
        })
        .collect()
}

fn is_aggregate(ty: &Type) -> bool {
    ty.scalar().is_none()
}

/// Leaves room for QBE spills, saved registers, alignment, libc calls, and
/// the trap's own diagnostic and abort below the checked stack limit.
const STACK_RESERVE_BYTES: u64 = 64 * 1024;

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
compile_error!("stack bounds discovery requires macOS or Linux");

/// The entry allocations shared by emission and stack accounting.
fn allocations(_layout: &Layout<'_>, function: &Function) -> Vec<(String, Type)> {
    let mut allocations = vec![("%stack_position".to_owned(), Scalar::U32.into())];
    for (id, ty) in function.flow.locals.iter().enumerate() {
        if id >= function.parameters || !is_aggregate(ty) {
            allocations.push((format!("%local{id}"), ty.clone()));
        }
    }
    let returned = returned_calls(function);
    for (id, value) in function.values.iter().enumerate() {
        if matches!(value.kind, ValueKind::Load(_) | ValueKind::CallResult)
            && is_aggregate(&value.ty)
            && !returned.contains(&ValueId(id))
        {
            allocations.push((format!("%v{id}_storage"), value.ty.clone()));
        }
    }
    allocations
}

pub(super) fn storage_bytes(layout: &Layout<'_>, function: &Function) -> u64 {
    allocations(layout, function)
        .iter()
        .fold(0u64, |bytes, (_, ty)| {
            bytes
                .saturating_add(layout.size(ty).saturating_add(7) & !7)
                .min(i64::MAX as u64)
        })
}

fn emit_stack_check(
    emitter: &mut Emitter<'_>,
    caller_bytes: u64,
    callee_bytes: u64,
    arguments: usize,
    message: String,
) {
    let id = emitter.stack_checks;
    emitter.stack_checks += 1;
    let needed = caller_bytes
        .saturating_add(callee_bytes)
        .saturating_add((arguments as u64).saturating_mul(8))
        .min(i64::MAX as u64);
    writeln!(
        emitter.text,
        "    %stack{id}_limit =l loadl $fern_stack_limit"
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %stack{id}_below =w cultl %stack_position, %stack{id}_limit"
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %stack{id}_available =l sub %stack_position, %stack{id}_limit"
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %stack{id}_short =w cultl %stack{id}_available, {needed}"
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %stack{id}_failed =w or %stack{id}_below, %stack{id}_short"
    )
    .unwrap();
    emit_conditional_trap(
        &mut emitter.text,
        &mut emitter.data,
        emitter.function,
        id,
        "stack",
        &format!("%stack{id}_failed"),
        message,
    );
}

fn emit_entry(emitter: &mut Emitter<'_>, main_span: &std::ops::Range<usize>) {
    // Keep wrapper message ownership separate from every Fern function.
    emitter.function = emitter.functions.len();
    emitter.data.push_str("data $fern_stack_limit = { l 0 }\n");
    emitter
        .text
        .push_str("export function w $main() {\n@start\n    %stack_position =l alloc4 4\n");
    if cfg!(target_os = "macos") {
        emitter.text.push_str("    %thread =l call $pthread_self()\n    %stack_top =l call $pthread_get_stackaddr_np(l %thread)\n    %stack_size =l call $pthread_get_stacksize_np(l %thread)\n    %stack_bottom =l sub %stack_top, %stack_size\n");
    } else {
        emitter.text.push_str("    %attr =l alloc8 128\n    %address =l alloc8 8\n    %size =l alloc8 8\n    %thread =l call $pthread_self()\n    %attr_status =w call $pthread_getattr_np(l %thread, l %attr)\n");
        emit_conditional_trap(
            &mut emitter.text,
            &mut emitter.data,
            emitter.function,
            0,
            "stack_bounds",
            "%attr_status",
            "cannot find the stack bounds\n".to_owned(),
        );
        emitter.text.push_str("    %bounds_status =w call $pthread_attr_getstack(l %attr, l %address, l %size)\n    call $pthread_attr_destroy(l %attr)\n");
        emit_conditional_trap(
            &mut emitter.text,
            &mut emitter.data,
            emitter.function,
            1,
            "stack_bounds",
            "%bounds_status",
            "cannot find the stack bounds\n".to_owned(),
        );
        emitter
            .text
            .push_str("    %stack_bottom =l loadl %address\n");
    }
    writeln!(emitter.text, "    %stack_limit =l add %stack_bottom, {STACK_RESERVE_BYTES}\n    storel %stack_limit, $fern_stack_limit").unwrap();
    let message = operation_message(
        Some(main_span),
        emitter.diagnostics.as_ref(),
        "stack overflow",
    );
    // The wrapper's fixed allocations occupy at most 160 bytes.
    emit_stack_check(
        emitter,
        160,
        emitter.storage_bytes[emitter.main.0],
        0,
        message,
    );
    writeln!(
        emitter.text,
        "    %status =w call ${}()\n    ret %status\n}}",
        function_symbol(emitter.main)
    )
    .unwrap();
}

fn emit_control_flow(emitter: &mut Emitter<'_>, function: &Function, flow: &ControlFlow) {
    for (name, ty) in allocations(&emitter.layout, function) {
        writeln!(
            emitter.text,
            "    {name} =l {} {}",
            allocation(&emitter.layout, &ty),
            emitter.layout.size(&ty)
        )
        .unwrap();
    }
    // The parameters are the leading locals, so the loop above allocated the
    // scalar ones. An aggregate parameter needs no entry copy.
    for index in 0..function.parameters {
        if let Type::Scalar(scalar) = &flow.locals[index] {
            writeln!(
                emitter.text,
                "    store{} %param{index}, %local{index}",
                qbe_type(*scalar)
            )
            .unwrap();
        }
    }
    writeln!(emitter.text, "    jmp @block{}", flow.entry.0).unwrap();

    for (block_id, block) in flow.blocks.iter().enumerate() {
        writeln!(emitter.text, "@block{block_id}").unwrap();
        for instruction in &block.instructions {
            emit_instruction(emitter, function, instruction);
        }
        emit_terminator(emitter, function, BlockId(block_id), &block.terminator);
    }
}

fn emit_instruction(emitter: &mut Emitter<'_>, function: &Function, instruction: &Instruction) {
    match instruction {
        Instruction::Value(ValueId(id)) => emit_value(emitter, function, *id),
        Instruction::Store {
            place: destination,
            operand: source,
        } => {
            let (address, ty) = place_address(emitter, function, destination);
            let source = operand(source.clone());
            emit_store(emitter, &source, &address, &ty);
        }
        Instruction::Zero { place } => {
            let (address, ty) = place_address(emitter, function, place);
            writeln!(
                emitter.text,
                "    call $memset(l {address}, w 0, l {})",
                emitter.layout.size(&ty)
            )
            .unwrap();
        }
        Instruction::Fill {
            place,
            start,
            operand: source,
        } => {
            let (address, ty) = place_address(emitter, function, place);
            emit_fill(emitter, &address, &ty, *start, &operand(source.clone()));
        }
        Instruction::Call {
            result,
            function: callee,
            arguments,
            span,
        } => emit_call(emitter, *callee, arguments, *result, span),
    }
}

/// Stores a value of type `ty` at `address`. An aggregate value is the
/// address of its storage, so storing one copies it.
fn emit_store(emitter: &mut Emitter<'_>, source: &str, address: &str, ty: &Type) {
    match ty {
        Type::Scalar(scalar) => writeln!(
            emitter.text,
            "    store{} {source}, {address}",
            qbe_type(*scalar)
        )
        .unwrap(),
        Type::Array { .. } | Type::Struct(_) => {
            emit_copy(&mut emitter.text, source, address, emitter.layout.size(ty));
        }
    }
}

/// Stores `source` into each element of the array at `address` from index
/// `start` on, in a loop whose code does not grow with the array.
fn emit_fill(emitter: &mut Emitter<'_>, address: &str, ty: &Type, start: u64, source: &str) {
    let Type::Array { length, element } = ty else {
        unreachable!("a verified fill stores into an array")
    };
    let stride = emitter.layout.size(element);
    let fill = emitter.fills;
    emitter.fills += 1;
    let text = &mut emitter.text;
    writeln!(text, "    jmp @fill{fill}_start").unwrap();
    writeln!(text, "@fill{fill}_start").unwrap();
    writeln!(text, "    jmp @fill{fill}_loop").unwrap();
    writeln!(text, "@fill{fill}_loop").unwrap();
    writeln!(
        text,
        "    %fill{fill}_index =l phi @fill{fill}_start {start}, @fill{fill}_body %fill{fill}_next"
    )
    .unwrap();
    writeln!(
        text,
        "    %fill{fill}_more =w csltl %fill{fill}_index, {length}"
    )
    .unwrap();
    writeln!(
        text,
        "    jnz %fill{fill}_more, @fill{fill}_body, @fill{fill}_done"
    )
    .unwrap();
    writeln!(text, "@fill{fill}_body").unwrap();
    writeln!(
        text,
        "    %fill{fill}_offset =l mul %fill{fill}_index, {stride}"
    )
    .unwrap();
    writeln!(
        text,
        "    %fill{fill}_address =l add {address}, %fill{fill}_offset"
    )
    .unwrap();
    emit_store(emitter, source, &format!("%fill{fill}_address"), element);
    let text = &mut emitter.text;
    writeln!(text, "    %fill{fill}_next =l add %fill{fill}_index, 1").unwrap();
    writeln!(text, "    jmp @fill{fill}_loop").unwrap();
    writeln!(text, "@fill{fill}_done").unwrap();
}

fn emit_call(
    emitter: &mut Emitter<'_>,
    callee: FunctionId,
    arguments: &[Operand],
    result: Option<ValueId>,
    span: &std::ops::Range<usize>,
) {
    let target = &emitter.functions[callee.0];
    // An aggregate result is written into the storage its value owns, whose
    // address the callee takes first.
    let aggregate_result = result.filter(|_| {
        target
            .result
            .as_ref()
            .is_some_and(|ty| ty.scalar().is_none())
    });
    // The callee's leading locals hold its parameters, so they give each
    // argument its type.
    let storage = |id: ValueId| {
        if emitter.returned_calls.contains(&id) {
            "%result".to_owned()
        } else {
            format!("%v{}_storage", id.0)
        }
    };
    let arguments = aggregate_result
        .map(|id| format!("l {}", storage(id)))
        .into_iter()
        .chain(arguments.iter().enumerate().map(|(index, argument)| {
            format!(
                "{} {}",
                emitter.layout.class(&target.flow.locals[index]),
                operand(argument.clone())
            )
        }))
        .collect::<Vec<_>>()
        .join(", ");
    let result_storage = aggregate_result.map(storage);
    let argument_count = target.parameters + usize::from(aggregate_result.is_some());
    let message = operation_message(Some(span), emitter.diagnostics.as_ref(), "stack overflow");
    emit_stack_check(
        emitter,
        emitter.storage_bytes[emitter.function],
        emitter.storage_bytes[callee.0],
        argument_count,
        message,
    );
    let symbol = function_symbol(callee);
    match (result, aggregate_result) {
        (Some(ValueId(id)), Some(_)) => writeln!(
            emitter.text,
            "    call ${symbol}({arguments})\n    %v{id} =l copy {}",
            result_storage.as_ref().unwrap()
        ),
        (Some(ValueId(id)), None) => writeln!(
            emitter.text,
            "    %v{id} ={} call ${symbol}({arguments})",
            emitter.layout.class(
                target
                    .result
                    .as_ref()
                    .expect("a call result requires a callee that returns one")
            )
        ),
        (None, _) => writeln!(emitter.text, "    call ${symbol}({arguments})"),
    }
    .unwrap();
}

fn emit_terminator(
    emitter: &mut Emitter<'_>,
    function: &Function,
    block: BlockId,
    terminator: &Terminator,
) {
    match terminator {
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
                operand(condition.clone()),
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
                operand(status.clone())
            )
            .unwrap();
            writeln!(emitter.text, "    call $exit(w %block{}_status)", block.0).unwrap();
            emitter.text.push_str("    hlt\n");
        }
        Terminator::Return { value } => match value {
            Some(value) => match function.result.as_ref().filter(|ty| ty.scalar().is_none()) {
                Some(ty) => {
                    // A returned call already wrote its result into `%result`.
                    let written =
                        matches!(value, Operand::Value(id) if emitter.returned_calls.contains(id));
                    if !written {
                        let size = emitter.layout.size(ty);
                        emit_copy(&mut emitter.text, &operand(value.clone()), "%result", size);
                    }
                    emitter.text.push_str("    ret\n");
                }
                None => {
                    writeln!(emitter.text, "    ret {}", operand(value.clone())).unwrap();
                }
            },
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
    match &value.kind {
        ValueKind::Load(source) => {
            let (address, _) = place_address(emitter, function, source);
            match &value.ty {
                Type::Scalar(scalar) => {
                    let width = qbe_type(*scalar);
                    writeln!(emitter.text, "    %v{id} ={width} load{width} {address}")
                }
                // Loading an aggregate copies it, so a later call in the same
                // expression cannot change what the value holds.
                Type::Array { .. } | Type::Struct(_) => {
                    let size = emitter.layout.size(&value.ty);
                    emit_copy(
                        &mut emitter.text,
                        &address,
                        &format!("%v{id}_storage"),
                        size,
                    );
                    writeln!(emitter.text, "    %v{id} =l copy %v{id}_storage")
                }
            }
            .unwrap();
        }
        // A borrowed aggregate is the address of its place's own storage.
        ValueKind::Borrow(source) => {
            let (address, _) = place_address(emitter, function, source);
            writeln!(emitter.text, "    %v{id} =l copy {address}").unwrap();
        }
        // A call result is defined by its call instruction.
        ValueKind::CallResult => {}
        ValueKind::Convert {
            operand: source,
            truncating,
        } => {
            let ty = scalar(&value.ty);
            let source_ty = operand_scalar(function, source.clone());
            if ty.is_floating() || source_ty.is_floating() {
                floating::emit_conversion(emitter, id, value, source.clone(), source_ty, ty);
            } else if *truncating || source_ty.all_values_fit(ty) {
                emit_truncation(&mut emitter.text, id, source.clone(), source_ty, ty);
            } else {
                let message = conversion_message(emitter, value, source_ty, ty);
                emit_checked_conversion(emitter, id, source.clone(), source_ty, ty, message);
            }
        }
        ValueKind::Unary { operator, operand } => {
            let ty = scalar(&value.ty);
            if ty.is_floating() {
                floating::emit_unary(&mut emitter.text, id, ty, operand.clone());
            } else {
                emit_unary_operation(emitter, id, value, *operator, operand.clone());
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
                floating::emit_binary(
                    &mut emitter.text,
                    id,
                    ty,
                    *operator,
                    left.clone(),
                    right.clone(),
                );
            } else {
                emit_binary_operation(
                    emitter,
                    id,
                    value,
                    *operator,
                    *form,
                    (left.clone(), right.clone()),
                    function,
                );
            }
        }
        ValueKind::Comparison {
            operator,
            left,
            right,
        } => {
            let operand_ty = operand_type(function, left.clone());
            match operand_ty.scalar() {
                None => emit_aggregate_comparison(
                    emitter,
                    id,
                    *operator,
                    left.clone(),
                    right.clone(),
                    &operand_ty,
                ),
                Some(scalar) if scalar.is_floating() => floating::emit_comparison(
                    &mut emitter.text,
                    id,
                    *operator,
                    left.clone(),
                    right.clone(),
                    scalar,
                ),
                Some(scalar) => emit_comparison(
                    &mut emitter.text,
                    id,
                    *operator,
                    left.clone(),
                    right.clone(),
                    scalar,
                ),
            }
        }
        ValueKind::LogicalNot { operand: source } => {
            writeln!(
                emitter.text,
                "    %v{id} =w ceqw {}, 0",
                operand(source.clone())
            )
            .unwrap();
        }
    }
}
