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
    F32,
    F64,
}

const NAMED_TYPES: [(Scalar, &str); 13] = [
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
    (Scalar::F32, "f32"),
    (Scalar::F64, "f64"),
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
            Self::F32 => 32,
            Self::F64 => 64,
        }
    }

    /// Whether this type's values are signed. This and the range helpers
    /// reached through it describe a two's complement bit pattern, which every
    /// integer type and `bool` has and a floating-point type does not.
    pub(crate) fn signed(self) -> bool {
        debug_assert!(
            !self.is_floating(),
            "`{self}` has no two's complement range"
        );
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::Int
        )
    }

    pub(crate) fn is_integer(self) -> bool {
        !matches!(self, Self::Bool | Self::F32 | Self::F64)
    }

    pub(crate) fn is_floating(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Whether this type names a number, which is what a conversion converts
    /// between.
    pub(crate) fn is_numeric(self) -> bool {
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

    /// How many bits of significand a value of this floating-point type has,
    /// counting the leading bit a normal value does not store.
    fn significand_bits(self) -> u32 {
        match self {
            Self::F32 => 24,
            Self::F64 => 53,
            _ => unreachable!("`{self}` has no significand"),
        }
    }

    /// Whether every value of this type is also a value of `destination`,
    /// which is when a checked conversion between them cannot fail.
    pub(crate) fn all_values_fit(self, destination: Self) -> bool {
        match (self.is_floating(), destination.is_floating()) {
            (false, false) => self.min() >= destination.min() && self.max() <= destination.max(),
            // A wider interchange format holds every value of a narrower one.
            (true, true) => self.width() <= destination.width(),
            // No floating-point format holds every value of an integer type,
            // and no integer type holds a NaN, an infinity, or a fraction.
            (true, false) => false,
            (false, true) => {
                self.width() - u32::from(self.signed()) <= destination.significand_bits()
            }
        }
    }
}

/// A concrete `f32` or `f64` value, stored as the bit pattern of its IEEE 754
/// interchange format. The bits keep the value's format and its signed zero,
/// which the mathematical value alone would lose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Float {
    Binary32(u32),
    Binary64(u64),
}

impl Float {
    pub(crate) fn ty(self) -> Scalar {
        match self {
            Self::Binary32(_) => Scalar::F32,
            Self::Binary64(_) => Scalar::F64,
        }
    }

    pub(crate) fn bits(self) -> u64 {
        match self {
            Self::Binary32(bits) => u64::from(bits),
            Self::Binary64(bits) => bits,
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Identifies one struct declaration in the program being checked. Semantic
/// checking allocates these, so they are unique across a whole program and
/// mean nothing outside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct StructId(pub(crate) usize);

/// A named struct type. Structs are nominal, so identity alone decides whether
/// two struct types are the same type; the name is what diagnostics display.
#[derive(Debug, Clone, Eq)]
pub(crate) struct StructType {
    pub id: StructId,
    pub name: String,
}

impl PartialEq for StructType {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

/// The type of a value. Array types compare structurally, so two `[N]T` types
/// are the same type when their lengths and element types are, while two
/// struct types are the same type only when they name one declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Type {
    Scalar(Scalar),
    Array { length: u64, element: Box<Type> },
    Struct(StructType),
}

impl Type {
    /// The scalar this type is, or `None` for an array or a struct. Callers
    /// that go on to do arithmetic use this to state that aggregates do not
    /// reach them.
    pub(crate) fn scalar(&self) -> Option<Scalar> {
        match self {
            Self::Scalar(scalar) => Some(*scalar),
            Self::Array { .. } | Self::Struct(_) => None,
        }
    }

    /// The scalar every value of this type is made of. A struct's fields have
    /// their own types, so this and `element_count` describe scalars and
    /// arrays only.
    pub(crate) fn leaf(&self) -> Scalar {
        match self {
            Self::Scalar(scalar) => *scalar,
            Self::Array { element, .. } => element.leaf(),
            Self::Struct(ty) => unreachable!("`{}` is not made of one scalar", ty.name),
        }
    }

    /// Whether values of this type are floating-point, which for an array is
    /// whether its elements are.
    pub(crate) fn is_floating(&self) -> bool {
        self.leaf().is_floating()
    }

    /// How many scalars a value of this type stores; a scalar stores one.
    pub(crate) fn element_count(&self) -> u64 {
        match self {
            Self::Scalar(_) => 1,
            Self::Array { length, element } => length * element.element_count(),
            Self::Struct(ty) => unreachable!("`{}` is not made of one scalar", ty.name),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(scalar) => write!(f, "{scalar}"),
            Self::Array { length, element } => write!(f, "[{length}]{element}"),
            Self::Struct(ty) => f.write_str(&ty.name),
        }
    }
}

impl From<Scalar> for Type {
    fn from(scalar: Scalar) -> Self {
        Self::Scalar(scalar)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOperator {
    Negate,
    WrappingNegate,
    Complement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComparisonOperator {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

impl ComparisonOperator {
    /// Equality is defined on every value type; the ordering comparisons are
    /// defined only on integers.
    pub(crate) fn is_equality(self) -> bool {
        matches!(self, Self::Equal | Self::NotEqual)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogicalOperator {
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOperator {
    Multiply,
    Divide,
    Remainder,
    WrappingMultiply,
    Add,
    Subtract,
    WrappingAdd,
    WrappingSubtract,
    ShiftLeft,
    ShiftRight,
    And,
    Xor,
    Or,
}

impl BinaryOperator {
    pub(crate) fn spelling(self) -> &'static str {
        match self {
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Remainder => "%",
            Self::WrappingMultiply => "*%",
            Self::Add => "+",
            Self::Subtract => "-",
            Self::WrappingAdd => "+%",
            Self::WrappingSubtract => "-%",
            Self::ShiftLeft => "<<",
            Self::ShiftRight => ">>",
            Self::And => "&",
            Self::Xor => "^",
            Self::Or => "|",
        }
    }

    /// A shift takes its count independently of its left operand's type, so
    /// every phase treats the two shifts apart from the other operators.
    pub(crate) fn is_shift(self) -> bool {
        matches!(self, Self::ShiftLeft | Self::ShiftRight)
    }

    /// Whether this operator is defined on floating-point operands. Remainder,
    /// the wrapping operators, the bitwise operators, and the shifts are
    /// defined only on integers.
    pub(crate) fn defined_on_floating(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Subtract | Self::Multiply | Self::Divide
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{Scalar, StructId, StructType, Type};

    fn declared(id: usize, name: &str) -> Type {
        Type::Struct(StructType {
            id: StructId(id),
            name: name.to_owned(),
        })
    }

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
    fn struct_types_compare_by_declaration_rather_than_by_name() {
        assert_eq!(declared(0, "Point"), declared(0, "Point"));
        // Two modules may each declare a `Point`, and those are two types.
        assert_ne!(declared(0, "Point"), declared(1, "Point"));
        assert_ne!(declared(0, "Point"), Type::Scalar(Scalar::Int));
        assert_ne!(declared(0, "Point"), array(1, declared(0, "Point")));
    }

    #[test]
    fn types_are_spelled_the_way_they_are_written() {
        assert_eq!(Type::Scalar(Scalar::Int).to_string(), "int");
        assert_eq!(
            array(2, array(3, Scalar::Int.into())).to_string(),
            "[2][3]int"
        );
        assert_eq!(declared(0, "Point").to_string(), "Point");
        assert_eq!(array(2, declared(0, "Point")).to_string(), "[2]Point");
    }

    #[test]
    fn a_struct_type_has_no_scalar() {
        assert_eq!(declared(0, "Point").scalar(), None);
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
    fn every_scalar_is_looked_up_and_displayed_by_its_written_name() {
        for (scalar, name) in super::NAMED_TYPES {
            assert_eq!(Scalar::named(name), Some(scalar));
            assert_eq!(scalar.name(), name);
            assert_eq!(scalar.to_string(), name);
        }
        assert_eq!(Scalar::named("f16"), None);
        assert_eq!(Scalar::named("float"), None);
    }

    #[test]
    fn floating_types_have_their_interchange_widths_on_every_target() {
        for (scalar, width) in [(Scalar::F32, 32), (Scalar::F64, 64)] {
            assert_eq!(scalar.width(), width);
            assert_eq!(scalar.width_on(32), width);
            assert_eq!(scalar.width_on(64), width);
        }
    }

    #[test]
    fn the_floating_types_are_numbers_but_not_integers() {
        for scalar in [Scalar::F32, Scalar::F64] {
            assert!(scalar.is_floating(), "{scalar}");
            assert!(scalar.is_numeric(), "{scalar}");
            assert!(!scalar.is_integer(), "{scalar}");
        }
        for scalar in Scalar::ALL_INTEGERS {
            assert!(!scalar.is_floating(), "{scalar}");
            assert!(scalar.is_numeric(), "{scalar}");
            assert!(scalar.is_integer(), "{scalar}");
        }
        assert!(!Scalar::Bool.is_floating());
        assert!(!Scalar::Bool.is_numeric());
        assert!(!Scalar::Bool.is_integer());
    }

    #[test]
    #[should_panic(expected = "`f32` has no two's complement range")]
    fn a_floating_type_has_no_two_s_complement_range() {
        Scalar::F32.signed();
    }

    #[test]
    fn an_array_is_floating_when_its_elements_are() {
        assert!(Type::Scalar(Scalar::F64).is_floating());
        assert!(array(2, array(3, Scalar::F32.into())).is_floating());
        assert!(!array(2, Scalar::Int.into()).is_floating());
        assert!(!Type::Scalar(Scalar::Bool).is_floating());
    }

    #[test]
    fn a_types_element_count_multiplies_its_lengths() {
        assert_eq!(Type::Scalar(Scalar::Int).element_count(), 1);
        assert_eq!(array(3, Scalar::Int.into()).element_count(), 3);
        assert_eq!(array(2, array(3, Scalar::Int.into())).element_count(), 6);
    }
}
