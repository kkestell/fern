//! Checked values, bindings, signatures, and program storage.

use crate::{
    diagnostic::Diagnostic,
    frontend::syntax::{Expression, Function, Statement, StructDeclaration, Syntax},
    source::FileId,
    types::{
        BinaryOperator, ComparisonOperator, Float, LogicalOperator, Scalar, StructId, StructType,
        Type, UnaryOperator,
    },
};

use la_arena::{Arena, ArenaMap, Idx};

use lasso::Spur;

use num_bigint::BigInt;

use num_rational::BigRational;

use std::collections::HashMap;

use super::namespaces::{FileImports, Namespace};

/// Rejects a checked program that declares a struct. Checking understands
/// structs, and the remaining struct tasks teach Fern IR and native
/// compilation to compile them; until then this stops one before lowering.
pub(crate) fn reject_uncompiled_structs(
    checked: CheckedProgram<'_>,
) -> Result<CheckedProgram<'_>, Diagnostic> {
    let Some(declared) = checked.structs.first() else {
        return Ok(checked);
    };
    let span = checked.syntax.structs[declared.declaration]
        .name_span
        .clone();
    Err(Diagnostic::new(span, "structs are not yet compiled"))
}

/// The value a constant expression folds to. An array literal folds when
/// every element does, and a struct literal when every field does.
///
/// An untyped constant keeps the exact value it was written with: an integer
/// as a `BigInt` and a floating-point value as a `Rational`. A concrete
/// floating-point value has already rounded to its format, so it is a `Float`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Constant {
    Integer(BigInt),
    Rational(BigRational),
    Float(Float),
    Array(Vec<Constant>),
    /// One value per field, in declaration order.
    Struct(Vec<Constant>),
}

impl Constant {
    /// The integer this constant folded to, or `None` when it folded to any
    /// other value.
    pub(crate) fn integer(&self) -> Option<&BigInt> {
        match self {
            Self::Integer(value) => Some(value),
            Self::Rational(_) | Self::Float(_) | Self::Array(_) | Self::Struct(_) => None,
        }
    }

    /// The exact value an untyped floating-point constant folded to.
    pub(crate) fn rational(&self) -> Option<&BigRational> {
        match self {
            Self::Rational(value) => Some(value),
            Self::Integer(_) | Self::Float(_) | Self::Array(_) | Self::Struct(_) => None,
        }
    }

    /// The concrete floating-point value this constant folded to.
    pub(crate) fn float(&self) -> Option<Float> {
        match self {
            Self::Float(value) => Some(*value),
            Self::Integer(_) | Self::Rational(_) | Self::Array(_) | Self::Struct(_) => None,
        }
    }
}

impl From<BigInt> for Constant {
    fn from(value: BigInt) -> Self {
        Self::Integer(value)
    }
}

impl From<BigRational> for Constant {
    fn from(value: BigRational) -> Self {
        Self::Rational(value)
    }
}

impl From<Float> for Constant {
    fn from(value: Float) -> Self {
        Self::Float(value)
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
    Floating,
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
    /// A struct literal. `initializers` are the written fields in source
    /// order, which is the order they are evaluated in, and `filled` holds
    /// the zero value `...` gives each field the literal omits.
    Struct {
        id: StructId,
        initializers: Vec<(usize, Idx<Expression>)>,
        filled: Vec<(usize, Constant)>,
    },
    /// `value.field`, holding the field's position in its declaration.
    Field {
        operand: Idx<Expression>,
        ordinal: usize,
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
    /// or folded to another kind of value.
    pub(crate) fn integer(&self) -> Option<&BigInt> {
        self.constant.as_ref().and_then(Constant::integer)
    }

    /// The exact value this expression folded to as an untyped
    /// floating-point constant.
    pub(crate) fn rational(&self) -> Option<&BigRational> {
        self.constant.as_ref().and_then(Constant::rational)
    }

    /// The concrete floating-point value this expression folded to.
    pub(crate) fn float(&self) -> Option<Float> {
        self.constant.as_ref().and_then(Constant::float)
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

/// One resolved step of an assignment target, in source order.
#[derive(Debug)]
pub(crate) enum CheckedStep {
    Index(Idx<Expression>),
    Field(usize),
}

/// Where an assignment stores: the binding its steps start from, those steps,
/// and the type they reach, which is the binding's own type when it has no
/// steps.
#[derive(Debug)]
pub(crate) struct CheckedTarget {
    pub binding: Idx<Binding>,
    pub steps: Vec<CheckedStep>,
    pub ty: Type,
}

/// One checked field of a struct.
#[derive(Debug)]
pub(crate) struct CheckedField {
    pub name: Spur,
    pub ty: Type,
}

/// How far a struct's fields have been resolved. Fields resolve the first time
/// a program names the type, so a struct reached again while it is resolving
/// is one that would contain itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FieldState {
    Unresolved,
    Resolving,
    Resolved,
}

/// A checked struct declaration: where it was declared, its fields in
/// declaration order, and where each field name sits in that order.
#[derive(Debug)]
pub(crate) struct CheckedStruct {
    pub declaration: Idx<StructDeclaration>,
    /// The file the declaration is in, which selects the imports its field
    /// annotations resolve against.
    pub(super) file: FileId,
    pub fields: Vec<CheckedField>,
    pub ordinals: HashMap<Spur, usize>,
    pub(super) state: FieldState,
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
    /// Every struct the program declares, indexed by its `StructId`.
    pub structs: Vec<CheckedStruct>,
    /// Checking state: the module-level structs of the module being checked,
    /// which an unqualified type name resolves against. It is replaced per
    /// module, so it never describes the whole program.
    pub(super) struct_names: HashMap<Spur, StructId>,
    /// Checking state: the module-level functions of the module being checked,
    /// which an unqualified call resolves against. It is replaced per module,
    /// so it never describes the whole program.
    pub(super) function_names: HashMap<Spur, Idx<Function>>,
    /// Checking state: each checked module's namespace, reached through a
    /// module identity rather than its position in load order.
    pub(super) namespaces: HashMap<crate::module::ModuleId, Namespace>,
    /// Checking state: each source file's imported names, reached through its
    /// file identity rather than a parallel vector index.
    pub(super) imports: HashMap<FileId, FileImports>,
    /// Checking state: the file whose declarations are being checked, which
    /// selects the imports name resolution sees.
    pub(super) file: FileId,
}

impl CheckedProgram<'_> {
    /// The value type a struct declaration names.
    pub(super) fn struct_type(&self, id: StructId) -> Type {
        let declaration = self.structs[id.0].declaration;
        Type::Struct(StructType {
            id,
            name: self
                .syntax
                .names
                .resolve(&self.syntax.structs[declaration].name)
                .to_owned(),
        })
    }

    /// The value a binding or field of this type holds before anything is
    /// stored into it: `0` for a number, `false` for a `bool`, and zero values
    /// throughout an array or struct.
    pub(super) fn zero_value(&self, ty: &Type) -> Constant {
        match ty {
            Type::Scalar(Scalar::F32) => Constant::Float(Float::Binary32(0)),
            Type::Scalar(Scalar::F64) => Constant::Float(Float::Binary64(0)),
            Type::Scalar(_) => Constant::Integer(BigInt::ZERO),
            Type::Array { length, element } => {
                let length = usize::try_from(*length).expect("an array fits in the address space");
                Constant::Array(vec![self.zero_value(element); length])
            }
            Type::Struct(ty) => Constant::Struct(
                self.structs[ty.id.0]
                    .fields
                    .iter()
                    .map(|field| self.zero_value(&field.ty))
                    .collect(),
            ),
        }
    }
}

/// The module bindings a function can read, plus its nested lexical bindings.
/// The module map is shared because a function never changes it.
pub(super) struct ScopeStack<'a> {
    module: &'a HashMap<Spur, Idx<Binding>>,
    locals: Vec<HashMap<Spur, Idx<Binding>>>,
}

impl<'a> ScopeStack<'a> {
    pub(super) fn module(module: &'a HashMap<Spur, Idx<Binding>>) -> Self {
        Self {
            module,
            locals: Vec::new(),
        }
    }

    pub(super) fn function(
        module: &'a HashMap<Spur, Idx<Binding>>,
        parameters: HashMap<Spur, Idx<Binding>>,
    ) -> Self {
        Self {
            module,
            locals: vec![parameters],
        }
    }

    pub(super) fn push(&mut self, scope: HashMap<Spur, Idx<Binding>>) {
        self.locals.push(scope);
    }

    pub(super) fn pop(&mut self) {
        self.locals.pop();
    }

    pub(super) fn insert(&mut self, name: Spur, binding: Idx<Binding>) {
        self.locals
            .last_mut()
            .expect("a statement is checked inside a local scope")
            .insert(name, binding);
    }

    pub(super) fn get(&self, name: Spur) -> Option<Idx<Binding>> {
        self.locals
            .iter()
            .rev()
            .find_map(|scope| scope.get(&name).copied())
            .or_else(|| self.module.get(&name).copied())
    }

    pub(super) fn contains(&self, name: Spur) -> bool {
        self.get(name).is_some()
    }
}
