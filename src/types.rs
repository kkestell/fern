use std::fmt;

/// A scalar type. Arithmetic, conversions, overflow, and shifts are defined
/// only on these, so their widths and ranges are total here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scalar {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Int,
    Uint,
}

const NAMED_TYPES: [(Scalar, &str); 11] = [
    (Scalar::Bool, "bool"),
    (Scalar::I8, "i8"),
    (Scalar::I16, "i16"),
    (Scalar::I32, "i32"),
    (Scalar::I64, "i64"),
    (Scalar::U8, "u8"),
    (Scalar::U16, "u16"),
    (Scalar::U32, "u32"),
    (Scalar::U64, "u64"),
    (Scalar::Int, "int"),
    (Scalar::Uint, "uint"),
];

impl Scalar {
    #[cfg(test)]
    pub(crate) const ALL_INTEGERS: [Self; 10] = [
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::Int,
        Self::Uint,
    ];

    pub(crate) fn named(name: &str) -> Option<Self> {
        NAMED_TYPES
            .iter()
            .find_map(|(ty, spelling)| (*spelling == name).then_some(*ty))
    }

    pub(crate) fn name(self) -> &'static str {
        NAMED_TYPES
            .iter()
            .find_map(|(ty, spelling)| (*ty == self).then_some(*spelling))
            .expect("every type has a spelling")
    }

    pub(crate) fn width(self) -> u32 {
        self.width_on(usize::BITS)
    }

    pub(crate) fn width_on(self, pointer_width: u32) -> u32 {
        match self {
            Self::Bool => 1,
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 => 16,
            Self::I32 | Self::U32 => 32,
            Self::I64 | Self::U64 => 64,
            Self::Int | Self::Uint => pointer_width,
        }
    }

    pub(crate) fn signed(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::Int
        )
    }

    pub(crate) fn is_integer(self) -> bool {
        self != Self::Bool
    }

    pub(crate) fn max(self) -> u64 {
        u64::MAX >> (64 - self.width() + u32::from(self.signed()))
    }

    pub(crate) fn min(self) -> i128 {
        if self.signed() {
            -(1i128 << (self.width() - 1))
        } else {
            0
        }
    }

    pub(crate) fn all_values_fit(self, destination: Self) -> bool {
        self.min() >= destination.min() && self.max() <= destination.max()
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The type of a value. Array types compare structurally, so two `[N]T` types
/// are the same type when their lengths and element types are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Type {
    Scalar(Scalar),
    Array { length: u64, element: Box<Type> },
}

impl Type {
    /// The scalar this type is, or `None` for an array. Callers that go on to
    /// do arithmetic use this to state that arrays do not reach them.
    pub(crate) fn scalar(&self) -> Option<Scalar> {
        match self {
            Self::Scalar(scalar) => Some(*scalar),
            Self::Array { .. } => None,
        }
    }

    /// The scalar every value of this type is made of.
    pub(crate) fn leaf(&self) -> Scalar {
        match self {
            Self::Scalar(scalar) => *scalar,
            Self::Array { element, .. } => element.leaf(),
        }
    }

    /// How many scalars a value of this type stores; a scalar stores one.
    pub(crate) fn element_count(&self) -> u64 {
        match self {
            Self::Scalar(_) => 1,
            Self::Array { length, element } => length * element.element_count(),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(scalar) => write!(f, "{scalar}"),
            Self::Array { length, element } => write!(f, "[{length}]{element}"),
        }
    }
}

impl From<Scalar> for Type {
    fn from(scalar: Scalar) -> Self {
        Self::Scalar(scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::{Scalar, Type};

    fn array(length: u64, element: Type) -> Type {
        Type::Array {
            length,
            element: Box::new(element),
        }
    }

    #[test]
    fn array_types_compare_structurally() {
        let two_by_three = array(2, array(3, Scalar::Int.into()));
        assert_eq!(two_by_three, array(2, array(3, Scalar::Int.into())));
        assert_ne!(two_by_three, array(3, array(2, Scalar::Int.into())));
        assert_ne!(two_by_three, array(2, array(3, Scalar::I64.into())));
        assert_ne!(two_by_three, Type::Scalar(Scalar::Int));
    }

    #[test]
    fn types_are_spelled_the_way_they_are_written() {
        assert_eq!(Type::Scalar(Scalar::Int).to_string(), "int");
        assert_eq!(
            array(2, array(3, Scalar::Int.into())).to_string(),
            "[2][3]int"
        );
    }

    #[test]
    fn only_a_scalar_type_has_a_scalar() {
        assert_eq!(Type::Scalar(Scalar::U8).scalar(), Some(Scalar::U8));
        assert_eq!(array(1, Scalar::U8.into()).scalar(), None);
    }

    #[test]
    fn nesting_does_not_change_the_leaf_scalar() {
        assert_eq!(Type::Scalar(Scalar::U8).leaf(), Scalar::U8);
        assert_eq!(array(2, array(3, Scalar::U8.into())).leaf(), Scalar::U8);
    }

    #[test]
    fn a_types_element_count_multiplies_its_lengths() {
        assert_eq!(Type::Scalar(Scalar::Int).element_count(), 1);
        assert_eq!(array(3, Scalar::Int.into()).element_count(), 3);
        assert_eq!(array(2, array(3, Scalar::Int.into())).element_count(), 6);
    }
}
