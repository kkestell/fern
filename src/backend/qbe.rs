//! The shared QBE output state and the spellings of operands and types.

use crate::{
    diagnostic::DiagnosticRenderer,
    ir::model::{Function, FunctionId, Global, Operand, ValueId},
    types::{Scalar, Type},
};

pub(super) fn operand(operand: Operand) -> String {
    match operand {
        Operand::Integer { value, .. } => value.to_string(),
        Operand::Value(ValueId(id)) => format!("%v{id}"),
    }
}

/// Arithmetic, conversions, and branching are defined on scalars, so the
/// emission for them states that no array reaches it.
pub(super) fn scalar(ty: &Type) -> Scalar {
    ty.scalar()
        .expect("integer and boolean emission runs on scalars")
}

pub(super) fn qbe_type(ty: Scalar) -> char {
    if ty.width() == 64 { 'l' } else { 'w' }
}

/// Every scalar occupies its whole QBE word wherever it is stored, so an array
/// occupies that word once per element.
pub(super) fn word(ty: &Type) -> char {
    qbe_type(ty.leaf())
}

pub(super) fn size(ty: &Type) -> u64 {
    ty.element_count() * if word(ty) == 'l' { 8 } else { 4 }
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
