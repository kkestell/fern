//! Byte size, alignment, and field offsets for Fern values.
//!
//! Semantic validation and the backend each hold a struct table of their own,
//! so a phase supplies its table through [`StructFields`] and reads one
//! derivation. [`Layouts`] memoizes a struct's layout the first time it is
//! derived, which keeps the total work linear in the number of fields a
//! program declares rather than exponential in containment depth.
//!
//! A cached layout is only sound once the struct's fields are final. In the
//! semantic phase that means a struct whose fields have resolved; in the IR it
//! means the program's struct table, which lowering does not revise.

use crate::types::{Scalar, StructId, Type};

use std::{cell::RefCell, collections::HashMap, rc::Rc};

/// Scalar storage uses its byte width independently of the QBE register class.
pub(crate) fn scalar_bytes(scalar: Scalar) -> u64 {
    if scalar == Scalar::Bool {
        1
    } else {
        u64::from(scalar.width() / 8)
    }
}

pub(crate) fn pointer_bytes() -> u64 {
    scalar_bytes(Scalar::Uint)
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
    /// times its element's. A pointer's size does not depend on its target.
    pub(crate) fn size(&self, table: &dyn StructFields, ty: &Type) -> Option<u64> {
        match ty {
            Type::Scalar(scalar) => Some(scalar_bytes(*scalar)),
            Type::Array { length, element } => length.checked_mul(self.size(table, element)?),
            Type::Struct(declared) => Some(self.struct_layout(table, declared.id)?.size),
            Type::Pointer { .. } => Some(pointer_bytes()),
        }
    }

    /// The byte boundary a value of this type starts on. An array aligns like
    /// the element it repeats.
    pub(crate) fn alignment(&self, table: &dyn StructFields, ty: &Type) -> Option<u64> {
        match ty {
            Type::Scalar(scalar) => Some(scalar_bytes(*scalar)),
            Type::Array { element, .. } => self.alignment(table, element),
            Type::Struct(declared) => Some(self.struct_layout(table, declared.id)?.alignment),
            Type::Pointer { .. } => Some(pointer_bytes()),
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

#[cfg(test)]
mod tests {
    use super::{Layouts, StructFields, pointer_bytes};
    use crate::types::{Scalar, StructId, StructType, Type};

    /// One struct, `struct { a: u8, p: *int }`.
    struct Table(Vec<Type>);

    impl StructFields for Table {
        fn field_count(&self, _: StructId) -> usize {
            self.0.len()
        }

        fn field_type(&self, _: StructId, ordinal: usize) -> &Type {
            &self.0[ordinal]
        }
    }

    fn pointer_to_int() -> Type {
        Type::Pointer {
            target: Box::new(Scalar::Int.into()),
        }
    }

    #[test]
    fn a_pointer_has_the_platform_pointer_width() {
        let table = Table(Vec::new());
        let layouts = Layouts::default();
        assert_eq!(pointer_bytes(), u64::from(usize::BITS / 8));
        {
            let ty = Type::Pointer {
                target: Box::new(Type::Array {
                    length: 1 << 60,
                    element: Box::new(Scalar::Int.into()),
                }),
            };
            assert_eq!(layouts.size(&table, &ty), Some(pointer_bytes()));
            assert_eq!(layouts.alignment(&table, &ty), Some(pointer_bytes()));
        }
        let array = Type::Array {
            length: 3,
            element: Box::new(pointer_to_int()),
        };
        assert_eq!(layouts.size(&table, &array), Some(3 * pointer_bytes()));
    }

    #[test]
    fn a_pointer_field_is_aligned_to_the_pointer_width() {
        let table = Table(vec![Scalar::U8.into(), pointer_to_int()]);
        let layouts = Layouts::default();
        let layout = layouts.struct_layout(&table, StructId(0)).unwrap();
        assert_eq!(layout.offsets, [0, pointer_bytes()]);
        assert_eq!(layout.size, 2 * pointer_bytes());
        assert_eq!(layout.alignment, pointer_bytes());
        let declared = Type::Struct(StructType {
            id: StructId(0),
            name: "Holder".to_owned(),
        });
        assert_eq!(layouts.size(&table, &declared), Some(2 * pointer_bytes()));
    }
}
