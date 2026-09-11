//! The shared QBE output state, the spellings of operands and types, aggregate
//! equality, and trap emission.

use crate::{
    diagnostic::{Diagnostic, DiagnosticRenderer},
    ir::model::{Function, FunctionId, Global, Literal, Operand, Value, ValueId},
    types::{ComparisonOperator, Scalar, Type},
};

use std::fmt::Write;

use super::{floating, layout::Layout};

pub(super) fn operand(operand: Operand) -> String {
    match operand {
        Operand::Literal(literal) => match literal {
            Literal::Integer { value, .. } => value.to_string(),
            Literal::Floating(value) => floating::literal(value),
            Literal::Null(_) => "0".to_owned(),
            Literal::EmptySlice(_) => "$emptyslice".to_owned(),
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
        Literal::Integer { value, ty } => format!("{} {value}", qbe_type(ty)),
        Literal::Floating(value) => format!(
            "{} {}",
            if value.ty() == Scalar::F32 { 'w' } else { 'l' },
            value.bits()
        ),
        Literal::Null(_) => "l 0".to_owned(),
        Literal::EmptySlice(_) => "l 0, l 0".to_owned(),
    }
}

/// Arithmetic, conversions, and branching are defined on scalars, so the
/// emission for them states that no aggregate reaches it.
pub(super) fn scalar(ty: &Type) -> Scalar {
    ty.scalar()
        .expect("arithmetic and conversion emission runs on scalars")
}

pub(super) fn qbe_type(ty: Scalar) -> char {
    match ty {
        Scalar::F32 => 's',
        Scalar::F64 => 'd',
        _ if ty.width() == 64 => 'l',
        _ => 'w',
    }
}

/// Two aggregates are equal when every pair of corresponding scalars is, which
/// is not the same question as holding identical bytes: two floating-point
/// zeroes are equal, a NaN equals nothing, and padding holds no value at all.
/// Comparing the layout's scalar slots answers the question one field at a
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

/// Emits an equality test without flattening an aggregate into one instruction
/// sequence per scalar. Arrays are walked at run time, while structs visit
/// their fields in declaration order. Scalar leaves retain QBE's IEEE 754
/// comparison instructions for floating-point fields.
fn emit_aggregate_equal(emitter: &mut Emitter<'_>, left: &str, right: &str, ty: &Type) -> String {
    let comparison = emitter.comparisons;
    emitter.comparisons += 1;
    match ty {
        Type::Scalar(scalar) => {
            let class = qbe_type(*scalar);
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_leftvalue ={class} load{class} {left}"
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_rightvalue ={class} load{class} {right}"
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_equal =w ceq{class} %aggregate{comparison}_leftvalue, %aggregate{comparison}_rightvalue"
            )
            .unwrap();
            format!("%aggregate{comparison}_equal")
        }
        Type::Pointer { .. } => {
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_leftvalue =l loadl {left}"
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_rightvalue =l loadl {right}"
            )
            .unwrap();
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_equal =w ceql %aggregate{comparison}_leftvalue, %aggregate{comparison}_rightvalue"
            )
            .unwrap();
            format!("%aggregate{comparison}_equal")
        }
        Type::Slice { element, .. } => {
            let equality = slice_equality(emitter, element);
            writeln!(
                emitter.text,
                "    %aggregate{comparison}_equal =w call $sliceequal{equality}(l {left}, l {right})"
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

/// Emits one equality helper per slice element type. Registering the type
/// before its body lets a comparable struct reach itself through a slice.
fn slice_equality(emitter: &mut Emitter<'_>, element: &Type) -> usize {
    if let Some(index) = emitter
        .slice_equalities
        .iter()
        .position(|known| known == element)
    {
        return index;
    }
    let index = emitter.slice_equalities.len();
    emitter.slice_equalities.push(element.clone());

    let outer = std::mem::take(&mut emitter.text);
    writeln!(
        emitter.text,
        "function w $sliceequal{index}(l %left, l %right) {{"
    )
    .unwrap();
    emitter.text.push_str("@start\n");
    writeln!(emitter.text, "    %leftbase =l loadl %left").unwrap();
    writeln!(emitter.text, "    %rightbase =l loadl %right").unwrap();
    let offset = emitter.layout.slice_length_offset();
    writeln!(
        emitter.text,
        "    %leftlengthaddress =l add %left, {offset}"
    )
    .unwrap();
    writeln!(
        emitter.text,
        "    %rightlengthaddress =l add %right, {offset}"
    )
    .unwrap();
    writeln!(emitter.text, "    %leftlength =l loadl %leftlengthaddress").unwrap();
    writeln!(
        emitter.text,
        "    %rightlength =l loadl %rightlengthaddress"
    )
    .unwrap();
    emitter
        .text
        .push_str("    %samelength =w ceql %leftlength, %rightlength\n");
    emitter
        .text
        .push_str("    jnz %samelength, @loop, @unequal\n");
    emitter.text.push_str("@loop\n");
    emitter
        .text
        .push_str("    %index =l phi @start 0, @next %nextindex\n");
    emitter
        .text
        .push_str("    %more =w csltl %index, %leftlength\n");
    emitter.text.push_str("    jnz %more, @body, @equal\n");
    emitter.text.push_str("@body\n");
    let stride = emitter.layout.size(element);
    writeln!(emitter.text, "    %offset =l mul %index, {stride}").unwrap();
    emitter
        .text
        .push_str("    %leftelement =l add %leftbase, %offset\n");
    emitter
        .text
        .push_str("    %rightelement =l add %rightbase, %offset\n");
    let equal = emit_aggregate_equal(emitter, "%leftelement", "%rightelement", element);
    writeln!(emitter.text, "    jnz {equal}, @next, @unequal").unwrap();
    emitter
        .text
        .push_str("@next\n    %nextindex =l add %index, 1\n    jmp @loop\n");
    emitter
        .text
        .push_str("@equal\n    ret 1\n@unequal\n    ret 0\n}\n");
    let helper = std::mem::replace(&mut emitter.text, outer);
    emitter.helpers.push_str(&helper);
    index
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
/// type; literals are scalar values or typed null pointers.
pub(super) fn operand_type(function: &Function, operand: Operand) -> Type {
    match operand {
        Operand::Literal(literal) => literal.ty(),
        Operand::Value(ValueId(id)) => function.values[id].ty.clone(),
    }
}

pub(super) fn operand_scalar(function: &Function, operand: Operand) -> Scalar {
    scalar(&operand_type(function, operand))
}

pub(super) fn operation_message(
    span: Option<&std::ops::Range<usize>>,
    diagnostic_renderer: Option<&DiagnosticRenderer<'_>>,
    message: &str,
) -> String {
    match (span, diagnostic_renderer) {
        (Some(span), Some(renderer)) => renderer.render(&Diagnostic::new(span.clone(), message)),
        _ => format!("{message}\n"),
    }
}

/// The message every failing checked conversion reports, whichever numeric
/// types it crosses between.
pub(super) fn conversion_message(
    emitter: &Emitter<'_>,
    value: &Value,
    source: Scalar,
    destination: Scalar,
) -> String {
    operation_message(
        value.span.as_ref(),
        emitter.diagnostics.as_ref(),
        &format!("checked conversion failed: `{source}` to `{destination}`"),
    )
}

pub(super) fn emit_conditional_trap(
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

pub(super) fn emit_message_data(data: &mut String, symbol: &str, message: &str) {
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

pub(super) struct Emitter<'a> {
    pub(super) text: String,
    pub(super) data: String,
    pub(super) helpers: String,
    pub(super) slice_equalities: Vec<Type>,
    /// The QBE layout of every type the program stores, which owns aggregate
    /// classes, sizes, alignments, and field offsets.
    pub(super) layout: Layout<'a>,
    pub(super) globals: &'a [Global],
    pub(super) functions: &'a [Function],
    pub(super) main: FunctionId,
    /// The index of the function being emitted, which distinguishes its
    /// module-global trap message symbols from every other function's.
    pub(super) function: usize,
    /// The return type of the signature being emitted, absent for a function
    /// that returns nothing.
    pub(super) qbe_result: Option<String>,
    /// How many element and field places the program has materialized, which
    /// names their address temporaries and their bounds-check symbols.
    pub(super) accesses: usize,
    /// How many aggregate comparisons have been emitted in this function.
    pub(super) comparisons: usize,
    pub(super) diagnostics: Option<DiagnosticRenderer<'a>>,
}
