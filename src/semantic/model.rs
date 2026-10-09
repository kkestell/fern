//! Checked values, bindings, signatures, and program storage.

use crate::{
    frontend::syntax::{Expression, Function, Statement, StructDeclaration, Syntax},
    layout::{Layouts, StructFields},
    source::FileId,
    types::{
        BinaryOperator, ComparisonOperator, Float, LogicalOperator, StringId, StructId, StructType,
        Type, UnaryOperator,
    },
};

use la_arena::{Arena, ArenaMap, Idx};

use lasso::Spur;

use num_bigint::BigInt;

use std::collections::HashMap;

use super::namespaces::{FileImports, Namespace};

/// Concrete constants. BigInt accommodates literal parsing before context
/// chooses a width; arithmetic is normalized after every operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Constant {
    Integer(BigInt),
    Float(Float),
    Array(Vec<Constant>),
    /// One value per field, in declaration order.
    Struct(Vec<Constant>),
    Null,
    /// The address of a string literal's static bytes.
    String(StringId),
}

impl Constant {
    /// The integer this constant folded to, or `None` when it folded to any
    /// other value.
    pub(crate) fn integer(&self) -> Option<&BigInt> {
        match self {
            Self::Integer(value) => Some(value),
            Self::Float(_) | Self::Array(_) | Self::Struct(_) | Self::Null | Self::String(_) => {
                None
            }
        }
    }

    /// Whether this constant is or contains a string literal's address, which
    /// is not a compile-time value.
    pub(crate) fn has_string(&self) -> bool {
        match self {
            Self::String(_) => true,
            Self::Array(values) | Self::Struct(values) => values.iter().any(Self::has_string),
            Self::Integer(_) | Self::Float(_) | Self::Null => false,
        }
    }
}

impl From<BigInt> for Constant {
    fn from(value: BigInt) -> Self {
        Self::Integer(value)
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
    PointerOffset {
        pointer: Idx<Expression>,
        offset: Idx<Expression>,
        subtract: bool,
    },
    PointerDifference {
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    Integer,
    /// A compound assignment's already-checked target, used only while
    /// selecting its binary-operator rules.
    CompoundAssignmentTarget,
    Floating,
    Boolean,
    Reference(Idx<Binding>),
    Grouping {
        expression: Idx<Expression>,
    },
    Conversion {
        operand: Idx<Expression>,
    },
    Unary {
        operator: UnaryOperator,
        operand: Idx<Expression>,
    },
    Binary {
        operator: BinaryOperator,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    Comparison {
        operator: ComparisonOperator,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    Logical {
        operator: LogicalOperator,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    LogicalNot {
        operand: Idx<Expression>,
    },
    Call {
        function: Idx<Function>,
    },
    Array {
        elements: Vec<Idx<Expression>>,
    },
    Index {
        operand: Idx<Expression>,
        index: Idx<Expression>,
    },
    Struct {
        id: StructId,
        initializers: Vec<(usize, Idx<Expression>)>,
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
    /// `&location`, whose checked location is in `CheckedProgram::addresses`.
    AddressOf,
    /// `*p`, a copy of the pointer's target.
    Dereference {
        operand: Idx<Expression>,
    },
    /// `null`, typed by its context.
    Null,
    /// A string literal, whose constant names its bytes.
    String,
    /// A raw pointer reinterpretation.
    PointerConversion {
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
}

pub(super) struct CheckedBinaryOperand {
    pub(super) id: Option<Idx<Expression>>,
    pub(super) expression: CheckedExpression,
}

#[derive(Debug, Clone)]
pub(crate) struct FunctionSignature {
    pub parameters: Vec<Idx<Binding>>,
    /// Whether calls pass extra arguments after the parameters.
    pub variadic: bool,
    pub result: Option<Type>,
}

#[derive(Debug)]
pub(crate) struct CheckedLocation {
    pub kind: CheckedLocationKind,
    pub ty: Type,
    pub span: std::ops::Range<usize>,
}

#[derive(Debug)]
pub(crate) enum CheckedLocationKind {
    Binding(Idx<Binding>),
    Index {
        operand: Box<CheckedLocation>,
        index: Idx<Expression>,
    },
    Field {
        operand: Box<CheckedLocation>,
        ordinal: usize,
    },
    /// A pointer computed by an expression that is not itself a location,
    /// such as a call. It only appears as the operand of a step or of a
    /// `Dereference`.
    Pointer(Idx<Expression>),
    /// `*p`, the target of the pointer the operand holds.
    Dereference {
        operand: Box<CheckedLocation>,
    },
}

/// Where an assignment stores: the location its steps start from, which is a
/// binding or a pointer, those steps, and the type they reach.
#[derive(Debug)]
pub(crate) struct CheckedTarget {
    pub location: CheckedLocation,
    pub ty: Type,
}

/// One checked field of a struct.
#[derive(Debug)]
pub(crate) struct CheckedField {
    pub name: Spur,
    pub ty: Type,
}

/// How far a struct's fields have been resolved. Fields resolve the first time
/// a program uses the type by value, outside a pointer target, so a struct
/// reached again while it is resolving is one that would contain itself.
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

#[derive(Debug)]
pub(crate) struct CheckedProgram<'a> {
    pub syntax: &'a Syntax,
    pub main: Idx<Function>,
    pub module_bindings: Vec<Idx<Statement>>,
    pub expressions: ArenaMap<Idx<Expression>, CheckedExpression>,
    pub declarations: ArenaMap<Idx<Statement>, Idx<Binding>>,
    pub bindings: Arena<Binding>,
    pub assignments: ArenaMap<Idx<Statement>, CheckedTarget>,
    /// The location each `&` expression takes the address of.
    pub addresses: ArenaMap<Idx<Expression>, CheckedLocation>,
    pub functions: ArenaMap<Idx<Function>, FunctionSignature>,
    pub calls: ArenaMap<Idx<Statement>, Idx<Function>>,
    /// Every struct the program declares, indexed by its `StructId`.
    pub structs: Vec<CheckedStruct>,
    /// Every distinct string literal's bytes, indexed by its `StringId`.
    pub strings: Vec<Vec<u8>>,
    /// Checking state: the `StringId` of each string literal's bytes, so equal
    /// literals share one.
    pub(super) string_ids: HashMap<Vec<u8>, StringId>,
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
    /// The inline struct declarations being resolved. It bounds the recursive
    /// field-resolution walk before it can exhaust the compiler stack.
    pub(super) struct_containment_depth: usize,
    /// The memoized layout of every struct whose size or alignment the checker
    /// has derived. A struct is only derived once its fields have resolved.
    pub(super) layouts: Layouts,
    /// Checking state: pointer target types, with their annotation spans,
    /// whose layouts wait on structs that have not resolved yet.
    pub(super) pending_targets: Vec<(Type, std::ops::Range<usize>)>,
}

impl StructFields for CheckedProgram<'_> {
    fn field_count(&self, id: StructId) -> usize {
        self.structs[id.0].fields.len()
    }

    fn field_type(&self, id: StructId, ordinal: usize) -> &Type {
        &self.structs[id.0].fields[ordinal].ty
    }
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
