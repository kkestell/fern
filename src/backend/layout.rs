//! QBE aggregate classes and the layout Fern uses for aggregate storage.

use crate::{
    ir::model::Struct,
    layout::{Layouts, StructFields},
    types::{StructId, Type},
};

use super::qbe::value_class;

/// The QBE layout of Fern values: a type's class, size, alignment, field
/// offsets, and value storage slots. Struct identity and shape both belong to
/// the verified IR program, so this is the one backend helper that reads them.
/// Size, alignment, and offsets come from the crate's shared derivation;
/// classes and value slots are the backend's own.
pub(super) struct Layout<'a> {
    structs: &'a [Struct],
    layouts: Layouts,
}

impl StructFields for Layout<'_> {
    fn field_count(&self, id: StructId) -> usize {
        self.structs[id.0].fields.len()
    }

    fn field_type(&self, id: StructId, ordinal: usize) -> &Type {
        &self.structs[id.0].fields[ordinal]
    }
}

/// Semantic checking rejects a layout the target cannot address, so the
/// backend reads a derivation that is known to be representable.
const REPRESENTABLE: &str = "verified IR has a representable layout";

impl<'a> Layout<'a> {
    pub(super) fn new(structs: &'a [Struct]) -> Self {
        Self {
            structs,
            layouts: Layouts::default(),
        }
    }

    /// How a signature or a call names this type. An aggregate travels as the
    /// address of its storage: QBE's own aggregate convention copies one word
    /// at a time, so its code would grow with the aggregate.
    pub(super) fn class(&self, ty: &Type) -> char {
        if ty.is_aggregate() {
            'l'
        } else {
            value_class(ty)
        }
    }

    pub(super) fn size(&self, ty: &Type) -> u64 {
        self.layouts.size(self, ty).expect(REPRESENTABLE)
    }

    pub(super) fn alignment(&self, ty: &Type) -> u64 {
        self.layouts.alignment(self, ty).expect(REPRESENTABLE)
    }

    /// The byte offset of a field within its struct, and the type stored
    /// there.
    pub(super) fn field(&self, id: StructId, ordinal: usize) -> (u64, &'a Type) {
        (
            self.offsets(id)[ordinal],
            &self.structs[id.0].fields[ordinal],
        )
    }

    pub(super) fn field_count(&self, id: StructId) -> usize {
        self.structs[id.0].fields.len()
    }

    fn offsets(&self, id: StructId) -> Vec<u64> {
        self.layouts
            .struct_layout(self, id)
            .expect(REPRESENTABLE)
            .offsets
            .clone()
    }

    /// Every scalar or pointer storage leaf and its byte offset, in the order
    /// IR globals flatten their literals.
    pub(super) fn value_slots(&self, ty: &Type) -> Vec<(u64, Type)> {
        let mut slots = Vec::new();
        self.collect_value_slots(ty, 0, &mut slots);
        slots
    }

    fn collect_value_slots(&self, ty: &Type, base: u64, slots: &mut Vec<(u64, Type)>) {
        match ty {
            Type::Scalar(scalar) => slots.push((base, (*scalar).into())),
            Type::Array { length, element } => {
                let stride = self.size(element);
                for index in 0..*length {
                    self.collect_value_slots(element, base + index * stride, slots);
                }
            }
            Type::Struct(declared) => {
                for (field, offset) in self.structs[declared.id.0]
                    .fields
                    .iter()
                    .zip(self.offsets(declared.id))
                {
                    self.collect_value_slots(field, base + offset, slots);
                }
            }
            Type::Pointer { .. } => slots.push((base, ty.clone())),
        }
    }
}
