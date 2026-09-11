//! Comparability of Fern value types.

use crate::{
    layout::StructFields,
    types::{StructId, Type},
};

use std::collections::HashMap;

/// Whether equality is defined on values of `ty`.
pub(crate) fn comparable(table: &dyn StructFields, ty: &Type) -> bool {
    comparable_with(table, ty, &mut HashMap::new())
}

/// Walks each distinct struct only once. Repeated fields can otherwise turn a
/// compact type graph into an exponentially large recursive walk.
fn comparable_with(
    table: &dyn StructFields,
    ty: &Type,
    structs: &mut HashMap<StructId, bool>,
) -> bool {
    match ty {
        Type::Scalar(_) | Type::Pointer { .. } => true,
        Type::Slice { .. } => false,
        Type::Array { element, .. } => comparable_with(table, element, structs),
        Type::Struct(declared) => {
            if let Some(comparable) = structs.get(&declared.id) {
                return *comparable;
            }
            let comparable = (0..table.field_count(declared.id)).all(|ordinal| {
                comparable_with(table, table.field_type(declared.id, ordinal), structs)
            });
            structs.insert(declared.id, comparable);
            comparable
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Scalar, StructType};
    use std::cell::Cell;

    struct CountingTable {
        structs: Vec<Vec<Type>>,
        field_reads: Cell<usize>,
    }

    impl StructFields for CountingTable {
        fn field_count(&self, id: StructId) -> usize {
            self.structs[id.0].len()
        }

        fn field_type(&self, id: StructId, ordinal: usize) -> &Type {
            self.field_reads.set(self.field_reads.get() + 1);
            &self.structs[id.0][ordinal]
        }
    }

    fn declared(id: usize) -> Type {
        Type::Struct(StructType {
            id: StructId(id),
            name: format!("T{id}"),
        })
    }

    #[test]
    fn repeated_structs_are_visited_once_per_query() {
        let mut structs = vec![vec![Scalar::Int.into()]];
        for id in 1..=26 {
            let previous = Type::Array {
                length: 1,
                element: Box::new(declared(id - 1)),
            };
            structs.push(vec![previous.clone(), previous]);
        }
        let table = CountingTable {
            structs,
            field_reads: Cell::new(0),
        };

        assert!(comparable(&table, &declared(26)));
        assert_eq!(table.field_reads.get(), 53);
    }
}
