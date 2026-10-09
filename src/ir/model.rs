//! Fern IR data model.

use crate::types::{BinaryOperator, ComparisonOperator, Float, Scalar, Type, UnaryOperator};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValueId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LocalId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlobalId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConstantId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FunctionId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockId(pub usize);

/// A storage location a load reads and a store writes. Locals live for one call
/// of one function; globals are the module-level `var` bindings. A constant is
/// a folded aggregate the function reads from static data and never writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Place {
    Local(LocalId),
    Global(GlobalId),
    Constant(ConstantId),
    /// `base[index]`. The span is the one the bounds-check trap reports, so
    /// every access through this place checks its index.
    Element {
        base: Box<Place>,
        index: Operand,
        span: std::ops::Range<usize>,
    },
    /// `base.field`, naming the field by its position in its declaration.
    /// Checking already resolved the field and selecting one cannot fail, so
    /// this carries no span.
    Field {
        base: Box<Place>,
        ordinal: usize,
    },
    /// The target of a non-nullable pointer value.
    Indirect {
        pointer: Operand,
    },
    /// The storage a function returning an aggregate writes its result into,
    /// which its caller provides.
    Result,
}

impl Place {
    /// The local, global, constant, pointer target, or result storage beneath
    /// every element and field step.
    pub(crate) fn root(&self) -> &Place {
        match self {
            Self::Element { base, .. } | Self::Field { base, .. } => base.root(),
            Self::Local(_)
            | Self::Global(_)
            | Self::Constant(_)
            | Self::Indirect { .. }
            | Self::Result => self,
        }
    }
}

/// An immediate value no instruction computes. This is the only way the IR
/// carries a value that no instruction computes, so an operand and a global's
/// static data cannot disagree about a value's type or its bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Literal {
    // i128 holds both the signed minima and the full u64 range without bit reinterpretation.
    Integer {
        value: i128,
        ty: Scalar,
    },
    /// A floating-point value, which keeps the bits of its interchange format
    /// rather than the integer they spell.
    Floating(Float),
    /// `null` of a nullable pointer type.
    Null(Type),
}

impl Literal {
    pub(crate) fn ty(&self) -> Type {
        match self {
            Self::Integer { ty, .. } => (*ty).into(),
            Self::Floating(value) => value.ty().into(),
            Self::Null(ty) => ty.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Operand {
    Literal(Literal),
    Value(ValueId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ValueKind {
    Load(Place),
    /// An aggregate read from its place's storage rather than from a copy of
    /// it. Lowering defines one only as a call argument whose place is rooted
    /// in a constant or in a local whose address is never taken: no
    /// expression writes such a local, a called function cannot reach it,
    /// and nothing writes a constant, so the place keeps its value until the
    /// call returns.
    Borrow(Place),
    /// The address of a place, a `*T` for the place's type `T`.
    Address(Place),
    /// A pointer to new heap storage for the target of the value's type. The
    /// span is the one the `out of memory` trap reports.
    Allocate,
    /// The `*T` a `nullable *T` operand holds. The span is the one the
    /// `null pointer` trap reports.
    CheckNull {
        operand: Operand,
    },
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
    /// Sets every byte of an aggregate place to zero, which is the zero value
    /// of every Fern type.
    Zero {
        place: Place,
    },
    /// Stores `operand` into each element of an array place from index
    /// `start` through its last element.
    Fill {
        place: Place,
        start: u64,
        operand: Operand,
    },
    /// Ends the life of the heap target a pointer of either nullability
    /// refers to. Freeing `null` does nothing.
    Free {
        operand: Operand,
    },
    /// Calls `function` with `arguments` in source order. `result` names the
    /// defined value when the callee returns one, unless `destination` names
    /// the place the callee writes its aggregate result into, in which case
    /// the call defines no value.
    Call {
        result: Option<ValueId>,
        destination: Option<Place>,
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

/// Static data: a module-level `var` of any module in the program, or a
/// function's constant. Its value is known during compilation, so it needs an
/// initial value rather than initialization code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Global {
    pub ty: Type,
    /// One entry per scalar or pointer, in memory order, or `None` when every
    /// byte is zero.
    pub values: Option<Vec<Literal>>,
}

#[derive(Debug)]
pub(crate) struct Function {
    /// The first `parameters` entries of `flow.locals` are the parameters, in
    /// source order, and hold their arguments on entry.
    pub parameters: usize,
    pub result: Option<Type>,
    // Each position defines the corresponding function-local value ID.
    pub values: Vec<Value>,
    // Each position defines the corresponding function-local constant ID.
    pub constants: Vec<Global>,
    pub flow: ControlFlow,
}

/// A struct declaration's shape: the types of its fields in declaration
/// order. `Type` carries a struct's nominal identity and its spelling, and
/// this carries the field types lowering, verification, and the backend read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Struct {
    pub fields: Vec<Type>,
}

#[derive(Debug)]
pub(crate) struct Program {
    // Each position defines the corresponding StructId.
    pub structs: Vec<Struct>,
    pub globals: Vec<Global>,
    // Each position defines the corresponding function ID.
    pub functions: Vec<Function>,
    pub main: FunctionId,
    /// The source span of the entry function's name.
    pub main_span: std::ops::Range<usize>,
}
