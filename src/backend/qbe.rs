//! The shared QBE output state, the spellings of operands and types, and trap
//! emission.

use crate::{
    diagnostic::{Diagnostic, DiagnosticRenderer},
    ir::model::{Function, FunctionId, Global, Literal, Operand, Value, ValueId},
    types::{Scalar, Type},
};

use std::fmt::Write;

use super::floating;

pub(super) fn operand(operand: Operand) -> String {
    match operand {
        Operand::Literal(literal) => match literal {
            Literal::Integer { value, .. } => value.to_string(),
            Literal::Floating(value) => floating::literal(value),
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
    }
}

/// Arithmetic, conversions, and branching are defined on scalars, so the
/// emission for them states that no array reaches it.
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

/// Every scalar occupies its whole QBE word wherever it is stored, so an array
/// occupies that word once per element.
pub(super) fn word(ty: &Type) -> char {
    qbe_type(ty.leaf())
}

/// How many bytes one stored scalar occupies.
pub(super) fn bytes(ty: Scalar) -> u64 {
    if ty.width() == 64 { 8 } else { 4 }
}

pub(super) fn size(ty: &Type) -> u64 {
    ty.element_count() * bytes(ty.leaf())
}

/// The type of an operand. Only a load or a call result gives one an array
/// type; a literal is always a scalar.
pub(super) fn operand_type(function: &Function, operand: Operand) -> Type {
    match operand {
        Operand::Literal(literal) => literal.ty().into(),
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
    pub(super) globals: &'a [Global],
    pub(super) functions: &'a [Function],
    pub(super) main: FunctionId,
    /// The index of the function being emitted, which distinguishes its
    /// module-global trap message symbols from every other function's.
    pub(super) function: usize,
    /// The return type of the signature being emitted, absent for a function
    /// that returns nothing.
    pub(super) qbe_result: Option<String>,
    /// How many element places the program has materialized, which names their
    /// temporaries and their bounds-check symbols.
    pub(super) accesses: usize,
    pub(super) diagnostics: Option<DiagnosticRenderer<'a>>,
}
