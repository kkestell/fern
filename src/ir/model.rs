//! Fern IR data model.

use crate::types::{BinaryOperator, ComparisonOperator, Float, Scalar, Type, UnaryOperator};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValueId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LocalId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlobalId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FunctionId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockId(pub usize);

/// A storage location a load reads and a store writes. Locals live for one call
/// of one function; globals are the module-level `var` bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Place {
    Local(LocalId),
    Global(GlobalId),
    /// `base[index]`. The span is the one the bounds-check trap reports, so
    /// every access through this place checks its index.
    Element {
        base: Box<Place>,
        index: Operand,
        span: std::ops::Range<usize>,
    },
}

impl Place {
    /// Storing through an element place writes inside its root, which is how
    /// an array local is initialized.
    pub(super) fn root_local(&self) -> Option<LocalId> {
        match self {
            Self::Local(local) => Some(*local),
            Self::Global(_) => None,
            Self::Element { base, .. } => base.root_local(),
        }
    }
}

/// An immediate value of a scalar type. This is the only way the IR carries a
/// value that no instruction computes, so an operand and a global's static
/// data cannot disagree about a value's type or its bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Literal {
    // i128 holds both the signed minima and the full u64 range without bit reinterpretation.
    Integer {
        value: i128,
        ty: Scalar,
    },
    /// A floating-point value, which keeps the bits of its interchange format
    /// rather than the integer they spell.
    Floating(Float),
}

impl Literal {
    pub(crate) fn ty(self) -> Scalar {
        match self {
            Self::Integer { ty, .. } => ty,
            Self::Floating(value) => value.ty(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operand {
    Literal(Literal),
    Value(ValueId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ValueKind {
    Load(Place),
    /// The result of the `Instruction::Call` that defines it. The call carries
    /// the arguments and the source span.
    CallResult,
    Convert {
        operand: Operand,
        truncating: bool,
    },
    Unary {
        operator: UnaryOperator,
        operand: Operand,
    },
    Binary {
        operator: BinaryOperator,
        form: BinaryForm,
        left: Operand,
        right: Operand,
    },
    Comparison {
        operator: ComparisonOperator,
        left: Operand,
        right: Operand,
    },
    LogicalNot {
        operand: Operand,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryForm {
    Infix,
    CompoundAssignment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Value {
    pub span: Option<std::ops::Range<usize>>,
    pub ty: Type,
    pub kind: ValueKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Instruction {
    Value(ValueId),
    Store {
        place: Place,
        operand: Operand,
    },
    /// Calls `function` with `arguments` in source order. `result` names the
    /// defined value exactly when the callee returns one.
    Call {
        result: Option<ValueId>,
        function: FunctionId,
        arguments: Vec<Operand>,
        span: std::ops::Range<usize>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Terminator {
    Jump {
        target: BlockId,
    },
    Branch {
        condition: Operand,
        then_target: BlockId,
        else_target: BlockId,
    },
    Exit {
        status: Operand,
    },
    Return {
        value: Option<Operand>,
    },
    Unreachable,
}

impl Terminator {
    pub(super) fn targets(&self) -> Vec<BlockId> {
        match self {
            Self::Jump { target } => vec![*target],
            Self::Branch {
                then_target,
                else_target,
                ..
            } => vec![*then_target, *else_target],
            Self::Exit { .. } | Self::Return { .. } | Self::Unreachable => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Block {
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ControlFlow {
    pub entry: BlockId,
    pub locals: Vec<Type>,
    pub blocks: Vec<Block>,
}

/// A module-level `var` of any module in the program. Its initializer is a
/// constant expression, so it needs an initial value rather than
/// initialization code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Global {
    pub ty: Type,
    /// Its scalars in memory order: one for a scalar global, one per element
    /// for an array global.
    pub values: Vec<Literal>,
}

#[derive(Debug)]
pub(crate) struct Function {
    /// The first `parameters` entries of `flow.locals` are the parameters, in
    /// source order, and hold their arguments on entry.
    pub parameters: usize,
    pub result: Option<Type>,
    // Each position defines the corresponding function-local value ID.
    pub values: Vec<Value>,
    pub flow: ControlFlow,
}

#[derive(Debug)]
pub(crate) struct Program {
    pub globals: Vec<Global>,
    // Each position defines the corresponding function ID.
    pub functions: Vec<Function>,
    pub main: FunctionId,
}
