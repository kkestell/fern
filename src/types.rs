#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Type {
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

const NAMED_TYPES: [(Type, &str); 11] = [
    (Type::Bool, "bool"),
    (Type::I8, "i8"),
    (Type::I16, "i16"),
    (Type::I32, "i32"),
    (Type::I64, "i64"),
    (Type::U8, "u8"),
    (Type::U16, "u16"),
    (Type::U32, "u32"),
    (Type::U64, "u64"),
    (Type::Int, "int"),
    (Type::Uint, "uint"),
];

impl Type {
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
