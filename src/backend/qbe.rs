//! The shared QBE output state, the spellings of operands and types, aggregate
//! equality, and scalar memory operations.

use crate::{
    ir::model::{Function, FunctionId, Global, Literal, Operand, ValueId},
    types::{ComparisonOperator, Scalar, StringId, StructId, Type},
};

use std::fmt::Write;

use super::{floating, layout::Layout};

pub(super) fn operand(operand: Operand) -> String {
    match operand {
        Operand::Literal(literal) => match literal {
            Literal::Integer { value, .. } => value.to_string(),
            Literal::Floating(value) => floating::literal(value),
            // `null` is the zero address.
            Literal::Null(_) => "0".to_owned(),
            Literal::String(id) => format!("${}", string_symbol(id)),
        },
        Operand::Value(ValueId(id)) => format!("%v{id}"),
    }
}

/// How static data spells one stored scalar. A QBE data item takes a decimal
/// number, so a floating-point value is written as the integer its bits spell
/// at the same width: the stored bytes are the same, and no rounding can
/// change them.
pub(super) fn data_item(literal: Literal) -> String {
    match literal {
        Literal::Integer { value, ty } => format!("{} {value}", memory_class(&ty.into())),
        Literal::Floating(value) => format!(
            "{} {}",
            if value.ty() == Scalar::F32 { 'w' } else { 'l' },
            value.bits()
        ),
        Literal::Null(ty) => format!("{} 0", value_class(&ty)),
        Literal::String(id) => format!("l ${}", string_symbol(id)),
    }
}

/// Names the static data holding string `id`'s bytes and terminator.
pub(super) fn string_symbol(StringId(id): StringId) -> String {
    format!(".fern.string{id}")
}

/// Arithmetic, conversions, and branching are defined on scalars, so the
/// emission for them states that no aggregate reaches it.
pub(super) fn scalar(ty: &Type) -> Scalar {
    ty.scalar()
        .expect("arithmetic and conversion emission runs on scalars")
}

pub(super) fn value_class(ty: &Type) -> char {
    match ty {
        Type::Scalar(scalar) => qbe_type(*scalar),
        Type::Pointer { .. } => qbe_type(Scalar::Uint),
        Type::Array { .. } | Type::Struct(_) => {
            unreachable!("an aggregate is carried by the address of its storage")
        }
    }
}

pub(super) fn qbe_type(ty: Scalar) -> char {
    match ty {
        Scalar::F32 => 's',
        Scalar::F64 => 'd',
        _ if ty.width() == 64 => 'l',
        _ => 'w',
    }
}

/// Two aggregates are equal when every pair of corresponding scalars and
/// pointers is, which is not the same question as holding identical bytes: two
/// floating-point zeroes are equal, a NaN equals nothing, and padding holds no
/// value at all. Comparing leaf by leaf answers the question one field at a
/// time, however deeply arrays and structs nest.
pub(super) fn emit_aggregate_comparison(
    emitter: &mut Emitter<'_>,
    id: usize,
    operator: ComparisonOperator,
    left: Operand,
    right: Operand,
    ty: &Type,
) {
    let (left, right) = (operand(left), operand(right));
    let equal = emit_aggregate_equal(emitter, &left, &right, ty);
    match operator {
        ComparisonOperator::Equal => writeln!(emitter.text, "    %v{id} =w copy {equal}"),
        ComparisonOperator::NotEqual => writeln!(emitter.text, "    %v{id} =w ceqw {equal}, 0"),
        _ => unreachable!("only equality compares aggregates"),
    }
    .unwrap();
}

fn emit_aggregate_equal(emitter: &mut Emitter<'_>, left: &str, right: &str, ty: &Type) -> String {
    let comparison = emitter.comparisons;
    emitter.comparisons += 1;
    match ty {
        Type::Scalar(_) | Type::Pointer { .. } => {
            let class = value_class(ty);
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_leftvalue ={class} {} {left}",
                load_operation(ty)
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_rightvalue ={class} {} {right}",
                load_operation(ty)
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_equal =w ceq{class} %aggregate{comparison}_leftvalue, %aggregate{comparison}_rightvalue"
            )
            .unwrap();
            format!("%aggregate{comparison}_equal")
        }
        Type::Array { length, element } => {
            let stride = emitter.layout.size(element);
            writeln!(emitter.text, "    jmp @aggregate{comparison}_start").unwrap();
            writeln!(emitter.text, "@aggregate{comparison}_start").unwrap();
            writeln!(emitter.text, "    jmp @aggregate{comparison}_loop").unwrap();
            writeln!(emitter.text, "@aggregate{comparison}_loop").unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_index =l phi @aggregate{comparison}_start 0, @aggregate{comparison}_next %aggregate{comparison}_next_index"
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_more =w csltl %aggregate{comparison}_index, {length}"
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    jnz %aggregate{comparison}_more, @aggregate{comparison}_body, @aggregate{comparison}_equal"
            )
            .unwrap();
            writeln!(emitter.text, "@aggregate{comparison}_body").unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_offset =l mul %aggregate{comparison}_index, {stride}"
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_left =l add {left}, %aggregate{comparison}_offset"
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_right =l add {right}, %aggregate{comparison}_offset"
            )
            .unwrap();
            let element_equal = emit_aggregate_equal(
                emitter,
                &format!("%aggregate{comparison}_left"),
                &format!("%aggregate{comparison}_right"),
                element,
            );
            writeln!(
                emitter.text,
                "    jnz {element_equal}, @aggregate{comparison}_next, @aggregate{comparison}_unequal"
            )
            .unwrap();
            writeln!(emitter.text, "@aggregate{comparison}_next").unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_next_index =l add %aggregate{comparison}_index, 1"
            )
            .unwrap();
            writeln!(emitter.text, "    jmp @aggregate{comparison}_loop").unwrap();
            emit_aggregate_result_blocks(emitter, comparison);
            format!("%aggregate{comparison}_result")
        }
        Type::Struct(declared) => {
            let fields = emitter.layout.field_count(declared.id);
            if fields == 0 {
                writeln!(emitter.text, "    %aggregate{comparison}_result =w copy 1").unwrap();
                return format!("%aggregate{comparison}_result");
            }
            writeln!(emitter.text, "    jmp @aggregate{comparison}_field0").unwrap();
            for ordinal in 0..fields {
                writeln!(emitter.text, "@aggregate{comparison}_field{ordinal}").unwrap();
                let (offset, field) = emitter.layout.field(declared.id, ordinal);
                writeln!(
                    emitter.text,
                    "    %aggregate{comparison}_left{ordinal} =l add {left}, {offset}"
                )
                .unwrap();
                writeln!(
                    emitter.text,
                    "    %aggregate{comparison}_right{ordinal} =l add {right}, {offset}"
                )
                .unwrap();
                let field_equal = emit_aggregate_equal(
                    emitter,
                    &format!("%aggregate{comparison}_left{ordinal}"),
                    &format!("%aggregate{comparison}_right{ordinal}"),
                    field,
                );
                let next = if ordinal + 1 == fields {
                    format!("aggregate{comparison}_equal")
                } else {
                    format!("aggregate{comparison}_field{}", ordinal + 1)
                };
                writeln!(
                    emitter.text,
                    "    jnz {field_equal}, @{next}, @aggregate{comparison}_unequal"
                )
                .unwrap();
            }
            emit_aggregate_result_blocks(emitter, comparison);
            format!("%aggregate{comparison}_result")
        }
    }
}

fn emit_aggregate_result_blocks(emitter: &mut Emitter<'_>, comparison: usize) {
    writeln!(emitter.text, "@aggregate{comparison}_equal").unwrap();
    writeln!(emitter.text, "    jmp @aggregate{comparison}_done").unwrap();
    writeln!(emitter.text, "@aggregate{comparison}_unequal").unwrap();
    writeln!(emitter.text, "    jmp @aggregate{comparison}_done").unwrap();
    writeln!(emitter.text, "@aggregate{comparison}_done").unwrap();
    writeln!(
        emitter.text,
        "    %aggregate{comparison}_result =w phi @aggregate{comparison}_equal 1, @aggregate{comparison}_unequal 0"
    )
    .unwrap();
}

/// The type of an operand. Only a load or a call result gives one an aggregate
/// type; literals are scalar values.
pub(super) fn operand_type(function: &Function, operand: Operand) -> Type {
    match operand {
        Operand::Literal(literal) => literal.ty(),
        Operand::Value(ValueId(id)) => function.values[id].ty.clone(),
    }
}

pub(super) fn operand_scalar(function: &Function, operand: Operand) -> Scalar {
    scalar(&operand_type(function, operand))
}

pub(super) struct Emitter<'a> {
    pub(super) text: String,
    pub(super) data: String,
    /// The QBE layout of every type the program stores, which owns aggregate
    /// classes, sizes, alignments, and field offsets.
    pub(super) layout: Layout<'a>,
    pub(super) globals: &'a [Global],
    pub(super) functions: &'a [Function],
    pub(super) main: FunctionId,
    pub(super) function: usize,
    /// The return type of the signature being emitted, absent for a function
    /// that returns nothing.
    pub(super) qbe_result: Option<char>,
    /// The aggregate values a `return` of the function being emitted returns
    /// that already sit in `%result`.
    pub(super) returned_in_place: Vec<ValueId>,
    pub(super) accesses: usize,
    /// How many aggregate comparisons have been emitted in this function.
    pub(super) comparisons: usize,
    /// How many external calls returning a struct have been emitted.
    pub(super) calls: usize,
}

/// How an external signature or call names a type. QBE's C ABI types preserve
/// the extension requirements of sub-word integers, and a struct passes as its
/// QBE aggregate type.
pub(super) fn abi_class(ty: &Type) -> String {
    match ty {
        Type::Scalar(Scalar::I8) => "sb".into(),
        Type::Scalar(Scalar::U8) => "ub".into(),
        Type::Scalar(Scalar::I16) => "sh".into(),
        Type::Scalar(Scalar::U16) => "uh".into(),
        Type::Struct(declared) => struct_type(declared.id),
        _ => value_class(ty).to_string(),
    }
}

/// Names the QBE aggregate type that passes struct `id` across the external
/// interface with the C ABI.
pub(super) fn struct_type(StructId(id): StructId) -> String {
    format!(":fern.struct{id}")
}

pub(super) fn memory_class(ty: &Type) -> char {
    match ty.scalar() {
        Some(Scalar::Bool | Scalar::I8 | Scalar::U8) => 'b',
        Some(Scalar::I16 | Scalar::U16) => 'h',
        _ => value_class(ty),
    }
}
pub(super) fn load_operation(ty: &Type) -> String {
    match ty.scalar() {
        Some(Scalar::Bool | Scalar::U8) => "loadub".into(),
        Some(Scalar::I8) => "loadsb".into(),
        Some(Scalar::U16) => "loaduh".into(),
        Some(Scalar::I16) => "loadsh".into(),
        _ => format!("load{}", value_class(ty)),
    }
}
