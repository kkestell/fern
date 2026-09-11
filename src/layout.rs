//! Byte size, alignment, and field offsets for Fern values.
//!
//! Semantic validation, IR verification, and the backend each hold a struct
//! table of their own, so a phase supplies its table through [`StructFields`]
//! and reads one derivation. [`Layouts`] memoizes a struct's layout the first
//! time it is derived, which keeps the total work linear in the number of
//! fields a program declares rather than exponential in containment depth.
//!
//! A cached layout is only sound once the struct's fields are final. In the
//! semantic phase that means a struct whose fields have resolved; in the IR it
//! means the program's struct table, which lowering does not revise.

use crate::types::{Scalar, StructId, Type};

use std::{cell::RefCell, collections::HashMap, rc::Rc};

/// The bytes a scalar occupies. Scalar storage is one QBE word, so a 64-bit
/// scalar takes eight bytes and every narrower one takes four.
pub(crate) fn scalar_bytes(scalar: Scalar) -> u64 {
    if scalar.width() == 64 { 8 } else { 4 }
}

/// One struct's memory: its total size including tail padding, its alignment,
/// and the byte offset of each field in declaration order.
#[derive(Debug)]
pub(crate) struct StructLayout {
    pub size: u64,
    pub alignment: u64,
    pub offsets: Vec<u64>,
}

/// A phase's struct table, read by ordinal because each phase stores fields
/// differently.
pub(crate) trait StructFields {
    fn field_count(&self, id: StructId) -> usize;
    fn field_type(&self, id: StructId, ordinal: usize) -> &Type;
}

/// The derived layout of every struct a phase has asked about. A cached `None`
/// records a layout that overflows `u64`.
#[derive(Debug, Default)]
pub(crate) struct Layouts {
    derived: RefCell<HashMap<StructId, Option<Rc<StructLayout>>>>,
}

impl Layouts {
    /// The bytes a value of this type occupies, or `None` when the layout
    /// overflows `u64`. An array is contiguous, so its size is its length
    /// times its element's.
    pub(crate) fn size(&self, table: &dyn StructFields, ty: &Type) -> Option<u64> {
        match ty {
            Type::Scalar(scalar) => Some(scalar_bytes(*scalar)),
            Type::Pointer { .. } => Some(scalar_bytes(Scalar::Uint)),
            Type::Array { length, element } => length.checked_mul(self.size(table, element)?),
            Type::Struct(declared) => Some(self.struct_layout(table, declared.id)?.size),
        }
    }

    /// The byte boundary a value of this type starts on. An array aligns like
    /// the element it repeats.
    pub(crate) fn alignment(&self, table: &dyn StructFields, ty: &Type) -> Option<u64> {
        match ty {
            Type::Scalar(scalar) => Some(scalar_bytes(*scalar)),
            Type::Pointer { .. } => Some(scalar_bytes(Scalar::Uint)),
            Type::Array { element, .. } => self.alignment(table, element),
            Type::Struct(declared) => Some(self.struct_layout(table, declared.id)?.alignment),
        }
    }

    /// One struct's layout, derived once and cached. Each field starts at the
    /// next offset its own alignment allows, the struct takes the widest
    /// field's alignment, and it is padded out to that alignment so an array
    /// of it strides evenly.
    pub(crate) fn struct_layout(
        &self,
        table: &dyn StructFields,
        id: StructId,
    ) -> Option<Rc<StructLayout>> {
        let cached = self.derived.borrow().get(&id).cloned();
        if let Some(layout) = cached {
            return layout;
        }
        let derived = self.derive(table, id);
        self.derived.borrow_mut().insert(id, derived.clone());
        derived
    }

    fn derive(&self, table: &dyn StructFields, id: StructId) -> Option<Rc<StructLayout>> {
        let count = table.field_count(id);
        let mut size = 0;
        let mut alignment = 1;
        let mut offsets = Vec::with_capacity(count);
        for ordinal in 0..count {
            let field = table.field_type(id, ordinal);
            let field_alignment = self.alignment(table, field)?;
            let field_size = self.size(table, field)?;
            alignment = alignment.max(field_alignment);
            let offset = align(size, field_alignment)?;
            offsets.push(offset);
            size = offset.checked_add(field_size)?;
        }
        Some(Rc::new(StructLayout {
            size: align(size, alignment)?,
            alignment,
            offsets,
        }))
    }
}

/// The first offset at or after `offset` that `alignment` divides.
fn align(offset: u64, alignment: u64) -> Option<u64> {
    offset
        .checked_add(alignment.checked_sub(1)?)
        .map(|value| value / alignment * alignment)
}
