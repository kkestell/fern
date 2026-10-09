//! QBE program, function, block, instruction, place, aggregate, and data emission.
//!
//! Aggregates travel through addresses and are copied with blit or memcpy.
//! Callees copy aggregate arguments into their local parameter storage.

use crate::{
    ir::model::{
        BlockId, ConstantId, ControlFlow, Function, FunctionId, Global, Instruction, Operand,
        Place, Terminator, ValueId, ValueKind,
    },
    ir::verify::VerifiedProgram,
    types::{Scalar, StringId, StructId, Type},
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
    format!(".fern.function{function}_constant{id}")
}

/// Writes a string literal's bytes followed by its terminating zero.
fn emit_string(emitter: &mut Emitter<'_>, id: StringId, bytes: &[u8]) {
    let mut items = String::from("b");
    for byte in bytes.iter().chain([&0]) {
        write!(items, " {byte}").unwrap();
    }
    writeln!(emitter.data, "data ${} = {{ {items} }}", string_symbol(id)).unwrap();
}

/// Writes static data. Its scalars and pointers sit where the layout puts them,
/// so the gaps between them and the padding after the last one are zeroed, and
/// data with no values is zero throughout.
fn emit_data(emitter: &mut Emitter<'_>, symbol: &str, global: &Global) {
    let mut items = Vec::new();
    let mut written = 0;
    if let Some(values) = &global.values {
        for ((offset, ty), value) in emitter
            .layout
            .value_slots(&global.ty)
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
    format!(".fern.fn{}", id.0)
}

fn place_address(emitter: &mut Emitter<'_>, function: &Function, place: &Place) -> (String, Type) {
    let flow = &function.flow;
    match place {
        Place::Local(local) => (format!("%local{}", local.0), flow.locals[local.0].clone()),
        Place::Global(global) => {
            let ty = emitter.globals[global.0].ty.clone();
            static_address(emitter, format!(".fern.global{}", global.0), ty)
        }
        Place::Constant(constant) => {
            let symbol = constant_symbol(emitter.function, *constant);
            static_address(emitter, symbol, function.constants[constant.0].ty.clone())
        }
        Place::Element { base, index } => {
            let (base, ty) = place_address(emitter, function, base);
            let Type::Array { element, .. } = ty else {
                unreachable!("a verified element place indexes an array")
            };
            let access = emitter.accesses;
            emitter.accesses += 1;
            let index = operand(index.clone());
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
        Place::Result => (
            "%result".to_owned(),
            function
                .result
                .clone()
                .expect("verified result storage belongs to a function with a result"),
        ),
        // A pointer's value is the address of its target.
        Place::Indirect { pointer } => {
            let Type::Pointer { target, .. } = operand_type(function, pointer.clone()) else {
                unreachable!("a verified indirect place reads through a pointer")
            };
            (operand(pointer.clone()), *target)
        }
    }
}

pub(super) fn emit(verified: &VerifiedProgram) -> String {
    let program = verified.program();
    let layout = Layout::new(&program.structs);
    let mut emitter = Emitter {
        text: String::new(),
        data: String::new(),
        layout,
        globals: &program.globals,
        functions: &program.functions,
        main: program.main,
        function: 0,
        qbe_result: None,
        returned_in_place: Vec::new(),
        accesses: 0,
        comparisons: 0,
        calls: 0,
    };
    emit_external_types(&mut emitter);
    emit_entry(&mut emitter);
    for (id, global) in program.globals.iter().enumerate() {
        emit_data(&mut emitter, &format!(".fern.global{id}"), global);
    }
    for (id, bytes) in program.strings.iter().enumerate() {
        emit_string(&mut emitter, StringId(id), bytes);
    }
    for (index, function) in program.functions.iter().enumerate() {
        if function.external.is_some() {
            continue;
        }
        let id = FunctionId(index);
        emitter.function = index;
        for (constant, data) in function.constants.iter().enumerate() {
            emit_data(
                &mut emitter,
                &constant_symbol(index, ConstantId(constant)),
                data,
            );
        }
        emitter.qbe_result = function
            .result
            .as_ref()
            .filter(|ty| !ty.is_aggregate())
            .map(value_class);
        emitter.returned_in_place = returned_in_place(function);
        let result = function
            .result
            .as_ref()
            .filter(|ty| ty.is_aggregate())
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

/// Defines the QBE aggregate type of every struct an external signature
/// names, after the types of the structs it contains, since QBE resolves a
/// type name where it is written.
fn emit_external_types(emitter: &mut Emitter<'_>) {
    let mut defined = Vec::new();
    for function in emitter.functions.iter().filter(|f| f.external.is_some()) {
        for ty in function.flow.locals.iter().chain(&function.result) {
            emit_external_type(emitter, ty, &mut defined);
        }
    }
}

fn emit_external_type(emitter: &mut Emitter<'_>, ty: &Type, defined: &mut Vec<StructId>) {
    let Type::Struct(declared) = ty else {
        return;
    };
    if defined.contains(&declared.id) {
        return;
    }
    defined.push(declared.id);
    let mut items = Vec::new();
    for ordinal in 0..emitter.layout.field_count(declared.id) {
        let (_, field) = emitter.layout.field(declared.id, ordinal);
        // An array repeats its innermost element, which may be a struct.
        let mut element = field;
        let mut count = 1;
        while let Type::Array {
            length,
            element: inner,
        } = element
        {
            count *= length;
            element = inner;
        }
        emit_external_type(emitter, element, defined);
        let item = match element {
            Type::Struct(inner) => struct_type(inner.id),
            _ => memory_class(element).to_string(),
        };
        items.push(if count == 1 {
            item
        } else {
            format!("{item} {count}")
        });
    }
    writeln!(
        emitter.data,
        "type {} = {{ {} }}",
        struct_type(declared.id),
        items.join(", ")
    )
    .unwrap();
}

/// The aggregate values the function's `return`s return that already sit in
/// `%result`: call results, which each call writes straight into `%result`,
/// and loads of the result storage a returned literal or local was built in.
/// A value is used only in the block defining it, so each shares its block
/// with the `return`, and nothing writes `%result` between the value's
/// definition and the `return`.
fn returned_in_place(function: &Function) -> Vec<ValueId> {
    function
        .flow
        .blocks
        .iter()
        .filter_map(|block| match &block.terminator {
            Terminator::Return {
                value: Some(Operand::Value(id)),
            } if function.values[id.0].ty.is_aggregate()
                && matches!(
                    function.values[id.0].kind,
                    ValueKind::CallResult | ValueKind::Load(Place::Result)
                ) =>
            {
                Some(*id)
            }
            _ => None,
        })
        .collect()
}

/// The local and temporary storage allocated at function entry.
fn allocations(_layout: &Layout<'_>, function: &Function) -> Vec<(String, Type)> {
    let mut allocations = Vec::new();
    for (id, ty) in function.flow.locals.iter().enumerate() {
        allocations.push((format!("%local{id}"), ty.clone()));
    }
    let returned = returned_in_place(function);
    for (id, value) in function.values.iter().enumerate() {
        if matches!(value.kind, ValueKind::Load(_) | ValueKind::CallResult)
            && value.ty.is_aggregate()
            && !returned.contains(&ValueId(id))
        {
            allocations.push((format!("%v{id}_storage"), value.ty.clone()));
        }
    }
    allocations
}

fn emit_entry(emitter: &mut Emitter<'_>) {
    writeln!(
        emitter.text,
        "export function w $main() {{\n@start\n    call ${}()\n    ret 0\n}}",
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
    for index in 0..function.parameters {
        let ty = &flow.locals[index];
        emit_store(
            emitter,
            &format!("%param{index}"),
            &format!("%local{index}"),
            ty,
        );
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
        Instruction::Call {
            result,
            destination,
            function: callee,
            arguments,
        } => emit_call(
            emitter,
            function,
            *callee,
            arguments,
            (*result, destination.as_ref()),
        ),
    }
}

/// Stores a value of type `ty` at `address`. An aggregate value is the
/// address of its storage, so storing one copies it.
fn emit_store(emitter: &mut Emitter<'_>, source: &str, address: &str, ty: &Type) {
    if ty.is_aggregate() {
        emit_copy(&mut emitter.text, source, address, emitter.layout.size(ty));
    } else {
        writeln!(
            emitter.text,
            "    store{} {source}, {address}",
            memory_class(ty)
        )
        .unwrap();
    }
}

fn emit_call(
    emitter: &mut Emitter<'_>,
    function: &Function,
    callee: FunctionId,
    arguments: &[Operand],
    (result, destination): (Option<ValueId>, Option<&Place>),
) {
    let target = &emitter.functions[callee.0];
    // An aggregate result is written into the call's destination, or into the
    // storage its value owns, whose address the callee takes first.
    let result_storage = match destination {
        Some(destination) => Some(place_address(emitter, function, destination).0),
        None => result
            .filter(|_| target.result.as_ref().is_some_and(|ty| ty.is_aggregate()))
            .map(|id| {
                if emitter.returned_in_place.contains(&id) {
                    "%result".to_owned()
                } else {
                    format!("%v{}_storage", id.0)
                }
            }),
    };
    // The callee's leading locals hold its parameters, so they give each
    // argument its type. A Fern callee takes its result storage first.
    let arguments = result_storage
        .as_ref()
        .filter(|_| target.external.is_none())
        .map(|storage| format!("l {storage}"))
        .into_iter()
        .chain(arguments.iter().enumerate().map(|(index, argument)| {
            format!(
                "{} {}",
                if target.external.is_some() {
                    abi_class(&target.flow.locals[index])
                } else {
                    emitter.layout.class(&target.flow.locals[index]).to_string()
                },
                operand(argument.clone())
            )
        }))
        .collect::<Vec<_>>()
        .join(", ");
    let symbol = target
        .external
        .clone()
        .unwrap_or_else(|| function_symbol(callee));
    // QBE returns an external struct result in storage of its own, so the
    // call copies it into the storage a Fern callee would have written.
    if let (Some(external), Some(storage)) = (&target.external, &result_storage) {
        let ty = target.result.as_ref().unwrap();
        let call = emitter.calls;
        emitter.calls += 1;
        writeln!(
            emitter.text,
            "    %call{call} ={} call ${external}({arguments})",
            abi_class(ty)
        )
        .unwrap();
        emit_copy(
            &mut emitter.text,
            &format!("%call{call}"),
            storage,
            emitter.layout.size(ty),
        );
        if let Some(ValueId(id)) = result {
            writeln!(emitter.text, "    %v{id} =l copy {storage}").unwrap();
        }
        return;
    }
    match (result, &result_storage) {
        (Some(ValueId(id)), Some(storage)) => writeln!(
            emitter.text,
            "    call ${symbol}({arguments})\n    %v{id} =l copy {storage}"
        ),
        (Some(ValueId(id)), None) => writeln!(
            emitter.text,
            "    %v{id} ={} call ${symbol}({arguments})",
            if target.external.is_some() {
                abi_class(target.result.as_ref().unwrap())
            } else {
                emitter
                    .layout
                    .class(target.result.as_ref().unwrap())
                    .to_string()
            }
        ),
        (None, _) => writeln!(emitter.text, "    call ${symbol}({arguments})"),
    }
    .unwrap();
}

fn emit_terminator(
    emitter: &mut Emitter<'_>,
    function: &Function,
    _block: BlockId,
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
        Terminator::Return { value } => match value {
            Some(value) => match function.result.as_ref().filter(|ty| ty.is_aggregate()) {
                Some(ty) => {
                    // A returned call, literal, or local already sits in
                    // `%result`.
                    let written = matches!(value, Operand::Value(id) if emitter.returned_in_place.contains(id));
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
        ValueKind::PointerOffset {
            pointer,
            offset,
            subtract,
        } => {
            let Type::Pointer { target } = &value.ty else {
                unreachable!()
            };
            writeln!(
                emitter.text,
                "    %offset{id} =l mul {}, {}",
                operand(offset.clone()),
                emitter.layout.size(target)
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %v{id} =l {} {}, %offset{id}",
                if *subtract { "sub" } else { "add" },
                operand(pointer.clone())
            )
            .unwrap();
        }
        ValueKind::PointerDifference { left, right } => {
            let Type::Pointer { target } = operand_type(function, left.clone()) else {
                unreachable!()
            };
            writeln!(
                emitter.text,
                "    %distance{id} =l sub {}, {}",
                operand(left.clone()),
                operand(right.clone())
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %v{id} =l div %distance{id}, {}",
                emitter.layout.size(&target)
            )
            .unwrap();
        }
        ValueKind::Load(source) => {
            let (address, _) = place_address(emitter, function, source);
            // Loading an aggregate copies it, so a later call in the same
            // expression cannot change what the value holds. A returned load
            // of the result storage is already where the caller reads it.
            if emitter.returned_in_place.contains(&ValueId(id)) {
                writeln!(emitter.text, "    %v{id} =l copy {address}")
            } else if value.ty.is_aggregate() {
                let size = emitter.layout.size(&value.ty);
                emit_copy(
                    &mut emitter.text,
                    &address,
                    &format!("%v{id}_storage"),
                    size,
                );
                writeln!(emitter.text, "    %v{id} =l copy %v{id}_storage")
            } else {
                let class = value_class(&value.ty);
                writeln!(
                    emitter.text,
                    "    %v{id} ={class} {} {address}",
                    load_operation(&value.ty)
                )
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
        ValueKind::Convert { operand: source } => {
            let ty = scalar(&value.ty);
            let source_ty = operand_scalar(function, source.clone());
            if ty.is_floating() || source_ty.is_floating() {
                floating::emit_conversion(emitter, id, value, source.clone(), source_ty, ty);
            } else {
                emit_truncation(&mut emitter.text, id, source.clone(), source_ty, ty);
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
                // A pointer of either nullability is a `uint`-width word.
                None if matches!(operand_ty, Type::Pointer { .. }) => emit_comparison(
                    &mut emitter.text,
                    id,
                    *operator,
                    left.clone(),
                    right.clone(),
                    Scalar::Uint,
                ),
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
        ValueKind::Address(place) => {
            let (address, _) = place_address(emitter, function, place);
            writeln!(emitter.text, "    %v{id} =l copy {address}").unwrap();
        }
        ValueKind::PointerCast { operand: pointer } => {
            writeln!(
                emitter.text,
                "    %v{id} =l copy {}",
                operand(pointer.clone())
            )
            .unwrap();
        }
    }
}
