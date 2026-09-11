//! QBE aggregate classes and the layout Fern uses for aggregate storage.

use crate::{
    ir::model::{Function, Struct},
    layout::{Layouts, StructFields},
    types::{StructId, Type},
};

use std::collections::BTreeSet;

use super::qbe::qbe_type;

/// The QBE layout of Fern values: a type's class, size, alignment, field
/// offsets, and scalar storage slots. Struct identity and shape both belong to
/// the verified IR program, so this is the one backend helper that reads them.
/// Size, alignment, and offsets come from the crate's shared derivation;
/// classes and scalar slots are the backend's own.
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

/// Verification rejects a layout the target cannot address, so the backend
/// reads a derivation that is known to be representable.
const REPRESENTABLE: &str = "verified IR has a representable layout";

impl<'a> Layout<'a> {
    pub(super) fn new(structs: &'a [Struct]) -> Self {
        Self {
            structs,
            layouts: Layouts::default(),
        }
    }

    /// How a signature, a call argument, or a call result names this type. Two
    /// array types made of the same elements share one QBE type definition.
    pub(super) fn class(&self, ty: &Type) -> String {
        match ty {
            Type::Scalar(scalar) => qbe_type(*scalar).to_string(),
            Type::Pointer { .. } => "l".to_owned(),
            Type::Slice { .. } => ":slice".to_owned(),
            Type::Array { .. } => self.array_class(ty),
            Type::Struct(declared) => format!(":struct{}", declared.id.0),
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

    /// The byte offset of a slice's length word from its base address.
    pub(super) fn slice_length_offset(&self) -> u64 {
        crate::layout::scalar_bytes(crate::types::Scalar::Uint)
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
    pub(super) fn scalar_slots(&self, ty: &Type) -> Vec<(u64, Type)> {
        let mut slots = Vec::new();
        self.collect_scalar_slots(ty, 0, &mut slots);
        slots
    }

    /// The aggregate type definitions the signatures name. QBE needs the
    /// layout of every aggregate a function takes or returns, and a struct's
    /// definition needs the definitions of the aggregates it contains.
    pub(super) fn type_definitions(&self, functions: &[Function]) -> String {
        let mut definitions = TypeDefinitions {
            layout: self,
            structs: vec![false; self.structs.len()],
            arrays: BTreeSet::new(),
            slice: false,
            text: String::new(),
        };
        for function in functions {
            for ty in function.flow.locals[..function.parameters]
                .iter()
                .chain(function.result.as_ref())
            {
                definitions.define(ty);
            }
        }
        definitions.text
    }

    /// An array is named by the scalar or struct it is made of and how many of
    /// them it holds, so `[2][3]int` and `[6]int` share one definition.
    fn array_class(&self, ty: &Type) -> String {
        let (element, count) = array_parts(ty);
        match element {
            Type::Scalar(scalar) => format!(":array{}{count}", qbe_type(*scalar)),
            Type::Pointer { .. } => format!(":arrayl{count}"),
            Type::Slice { .. } => format!(":arrayslice{count}"),
            Type::Struct(declared) => format!(":arraystruct{}x{count}", declared.id.0),
            Type::Array { .. } => unreachable!("array parts stop at a scalar or a struct"),
        }
    }

    fn collect_scalar_slots(&self, ty: &Type, base: u64, slots: &mut Vec<(u64, Type)>) {
        match ty {
            Type::Scalar(scalar) => slots.push((base, (*scalar).into())),
            Type::Pointer { .. } => slots.push((base, ty.clone())),
            Type::Slice { .. } => slots.push((base, ty.clone())),
            Type::Array { length, element } => {
                let stride = self.size(element);
                for index in 0..*length {
                    self.collect_scalar_slots(element, base + index * stride, slots);
                }
            }
            Type::Struct(declared) => {
                for (field, offset) in self.structs[declared.id.0]
                    .fields
                    .iter()
                    .zip(self.offsets(declared.id))
                {
                    self.collect_scalar_slots(field, base + offset, slots);
                }
            }
        }
    }
}

/// Collects the type definitions a program needs, writing each one after the
/// definitions it names and only once.
struct TypeDefinitions<'a> {
    layout: &'a Layout<'a>,
    structs: Vec<bool>,
    arrays: BTreeSet<String>,
    slice: bool,
    text: String,
}

impl TypeDefinitions<'_> {
    fn define(&mut self, ty: &Type) {
        match ty {
            Type::Scalar(_) => {}
            Type::Pointer { .. } => {}
            Type::Slice { .. } => {
                if !std::mem::replace(&mut self.slice, true) {
                    self.text.push_str("type :slice = { l, l }\n");
                }
            }
            Type::Array { .. } => self.define_array(ty),
            Type::Struct(declared) => self.define_struct(declared.id),
        }
    }

    fn define_array(&mut self, ty: &Type) {
        let (element, count) = array_parts(ty);
        self.define(element);
        let class = self.layout.array_class(ty);
        if self.arrays.insert(class.clone()) {
            let element = self.layout.class(element);
            self.text
                .push_str(&format!("type {class} = {{ {element} {count} }}\n"));
        }
    }

    fn define_struct(&mut self, id: StructId) {
        if std::mem::replace(&mut self.structs[id.0], true) {
            return;
        }
        let fields = self.layout.structs[id.0].fields.clone();
        for field in &fields {
            self.define(field);
        }
        let fields = fields
            .iter()
            .map(|field| self.layout.class(field))
            .collect::<Vec<_>>()
            .join(", ");
        self.text
            .push_str(&format!("type :struct{} = {{ {fields} }}\n", id.0));
    }
}

/// The scalar or struct an array is made of and how many of them it holds.
fn array_parts(mut ty: &Type) -> (&Type, u64) {
    let mut count = 1;
    while let Type::Array { length, element } = ty {
        count *= length;
        ty = element;
    }
    (ty, count)
}
