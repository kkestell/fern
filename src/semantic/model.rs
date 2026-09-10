//! Checked values, bindings, signatures, and program storage.

use crate::{
    frontend::syntax::{Expression, Function, Statement, Syntax},
    types::{BinaryOperator, ComparisonOperator, LogicalOperator, Type, UnaryOperator},
};

use la_arena::{Arena, ArenaMap, Idx};

use lasso::Spur;

use num_bigint::BigInt;

use std::collections::HashMap;

use super::namespaces::{FileImports, Namespace};

/// The value a constant expression folds to. An array literal folds when
/// every element does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Constant {
    Integer(BigInt),
    Array(Vec<Constant>),
}

impl Constant {
    /// The integer this constant folded to, or `None` when it folded to an
    /// array.
    pub(crate) fn integer(&self) -> Option<&BigInt> {
        match self {
            Self::Integer(value) => Some(value),
            Self::Array(_) => None,
        }
    }
}

impl From<BigInt> for Constant {
    fn from(value: BigInt) -> Self {
        Self::Integer(value)
    }
}

#[derive(Debug)]
pub(crate) struct Binding {
    pub ty: Type,
    pub mutable: bool,
    pub constant: Option<Constant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ExpressionValue {
    Integer,
    Boolean,
    Reference(Idx<Binding>),
    Grouping {
        expression: Idx<Expression>,
    },
    Conversion {
        operand: Idx<Expression>,
        truncating: bool,
    },
    Unary {
        operator: UnaryOperator,
        operator_span: std::ops::Range<usize>,
        operand: Idx<Expression>,
    },
    Binary {
        operator: BinaryOperator,
        operator_span: std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    Comparison {
        operator: ComparisonOperator,
        operator_span: std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    Logical {
        operator: LogicalOperator,
        operator_span: std::ops::Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    LogicalNot {
        operator_span: std::ops::Range<usize>,
        operand: Idx<Expression>,
    },
    Call {
        function: Idx<Function>,
    },
    /// An array literal, whose fill repeats the last element across the
    /// array's remaining elements.
    Array {
        elements: Vec<Idx<Expression>>,
        fill: bool,
    },
    Index {
        operand: Idx<Expression>,
        index: Idx<Expression>,
    },
    /// `len(a)`, which reads its length from the operand's type and still
    /// evaluates the operand.
    Length {
        operand: Idx<Expression>,
    },
}

#[derive(Debug)]
pub(crate) struct CheckedExpression {
    pub ty: Type,
    pub untyped: bool,
    pub value: ExpressionValue,
    pub constant: Option<Constant>,
}

impl CheckedExpression {
    /// The integer this expression folded to, or `None` when it did not fold
    /// or folded to an array.
    pub(crate) fn integer(&self) -> Option<&BigInt> {
        self.constant.as_ref().and_then(Constant::integer)
    }
}

pub(super) struct CheckedBinaryOperand {
    pub(super) id: Option<Idx<Expression>>,
    pub(super) expression: CheckedExpression,
}

#[derive(Debug, Clone)]
pub(crate) struct FunctionSignature {
    pub parameters: Vec<Idx<Binding>>,
    pub result: Option<Type>,
}

/// Where an assignment stores: the binding its indices start from and the type
/// of the element they reach, which is the binding's own type when it has no
/// indices.
#[derive(Debug)]
pub(crate) struct CheckedTarget {
    pub binding: Idx<Binding>,
    pub ty: Type,
}

/// The bindings a `for … in` statement introduces. They belong to the
/// statement rather than to a declaration, so they are recorded on their own.
#[derive(Debug)]
pub(crate) struct IterationBindings {
    pub value: Idx<Binding>,
    pub index: Option<Idx<Binding>>,
}

#[derive(Debug)]
pub(crate) struct CheckedProgram<'a> {
    pub syntax: &'a Syntax,
    pub main: Idx<Function>,
    pub module_bindings: Vec<Idx<Statement>>,
    pub expressions: ArenaMap<Idx<Expression>, CheckedExpression>,
    pub declarations: ArenaMap<Idx<Statement>, Idx<Binding>>,
    pub bindings: Arena<Binding>,
    pub assignments: ArenaMap<Idx<Statement>, CheckedTarget>,
    pub iterations: ArenaMap<Idx<Statement>, IterationBindings>,
    pub functions: ArenaMap<Idx<Function>, FunctionSignature>,
    pub calls: ArenaMap<Idx<Statement>, Idx<Function>>,
    /// Checking state: the module-level functions of the module being checked,
    /// which an unqualified call resolves against. It is replaced per module,
    /// so it never describes the whole program.
    pub(super) function_names: HashMap<Spur, Idx<Function>>,
    /// Checking state: each checked module's namespace, in module order, which
    /// a qualified name reaches through its file's imports.
    pub(super) namespaces: Vec<Namespace>,
    /// Checking state: each source file's imported names, indexed by file.
    pub(super) imports: Vec<FileImports>,
    /// Checking state: the file whose declarations are being checked, which
    /// selects the imports name resolution sees.
    pub(super) file: usize,
}
