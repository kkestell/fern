//! Parsed syntax nodes, type annotations, operators, arenas, source spans,
//! and expression traversal.

use crate::types::{BinaryOperator, ComparisonOperator, LogicalOperator, Scalar, UnaryOperator};

use la_arena::{Arena, Idx};

use lasso::{Rodeo, Spur};

use std::ops::Range;

/// One `::`-separated identifier in an import path, a selective import list, or
/// the qualifier of a qualified name.
#[derive(Debug)]
pub(crate) struct PathComponent {
    pub name: Spur,
    pub name_span: Range<usize>,
}

/// A `module::name` reference, call target, or assignment target. `span` covers
/// the qualifier and the name together.
#[derive(Debug)]
pub(crate) struct QualifiedName {
    pub qualifier: Option<PathComponent>,
    pub name: Spur,
    #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
    pub name_span: Range<usize>,
    pub span: Range<usize>,
}

/// A `use` declaration. `selection` is the brace form's imported names.
#[derive(Debug)]
pub(crate) struct Import {
    pub path: Vec<PathComponent>,
    pub selection: Option<Vec<PathComponent>>,
    #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
    pub span: Range<usize>,
}

#[derive(Debug)]
pub(crate) struct Function {
    pub name: Spur,
    pub name_span: Range<usize>,
    pub parameters: Vec<TypedName>,
    pub result: FunctionResult,
    pub external: bool,
    #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
    pub span: Range<usize>,
    pub body: Vec<Idx<Statement>>,
}

/// One `name: T` declaration: a function parameter or a struct field.
#[derive(Debug)]
pub(crate) struct TypedName {
    pub name: Spur,
    pub name_span: Range<usize>,
    pub annotation: Idx<TypeAnnotation>,
}

/// A `type Name struct { … }` declaration, which has at least one field.
#[derive(Debug)]
pub(crate) struct StructDeclaration {
    pub name: Spur,
    pub name_span: Range<usize>,
    pub fields: Vec<TypedName>,
    #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
    pub span: Range<usize>,
}

#[derive(Debug)]
pub(crate) enum FunctionResult {
    Void,
    Value(Idx<TypeAnnotation>),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum TopLevelItem {
    Function {
        function: Idx<Function>,
        public: bool,
    },
    Binding {
        binding: Idx<Statement>,
        public: bool,
    },
    Struct {
        declaration: Idx<StructDeclaration>,
        public: bool,
    },
}

#[derive(Debug)]
pub(crate) struct Statement {
    pub kind: StatementKind,
    pub span: Range<usize>,
}

#[derive(Debug)]
pub(crate) enum StatementKind {
    Binding {
        mutable: bool,
        name: Spur,
        name_span: Range<usize>,
        annotation: Option<Idx<TypeAnnotation>>,
        initializer: Idx<Expression>,
    },
    Assignment {
        target: Idx<Expression>,
        value: Idx<Expression>,
    },
    CompoundAssignment {
        target: Idx<Expression>,
        operator: BinaryOperator,
        operator_span: Range<usize>,
        value: Idx<Expression>,
    },
    Block {
        body: Vec<Idx<Statement>>,
    },
    Call {
        call: Call,
    },
    Return {
        value: Option<Idx<Expression>>,
    },
    If {
        condition: Idx<Expression>,
        then_body: Vec<Idx<Statement>>,
        else_branch: Option<Idx<Statement>>,
    },
    For {
        header: ForHeader,
        body: Vec<Idx<Statement>>,
    },
    Break {},
    Continue {},
}

#[derive(Debug)]
pub(crate) struct Call {
    pub target: QualifiedName,
    #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
    pub left_paren_span: Range<usize>,
    pub right_paren_span: Range<usize>,
    pub arguments: Vec<Idx<Expression>>,
}

#[derive(Debug)]
pub(crate) enum ForHeader {
    Infinite,
    Condition(Idx<Expression>),
}

#[derive(Debug)]
pub(crate) struct TypeAnnotation {
    pub kind: AnnotationKind,
    pub span: Range<usize>,
}

/// A written type with explicit array lengths.
#[derive(Debug)]
pub(crate) enum AnnotationKind {
    Scalar(Scalar),
    /// A declared type, written plainly or as `module::Name`.
    Named(QualifiedName),
    Array {
        length: Idx<Expression>,
        element: Idx<TypeAnnotation>,
    },
    Pointer {
        target: Idx<TypeAnnotation>,
    },
}

/// One `field = e` initializer of a struct literal.
#[derive(Debug)]
pub(crate) struct FieldInitializer {
    pub name: Spur,
    pub name_span: Range<usize>,
    pub value: Idx<Expression>,
}

#[derive(Debug)]
pub(crate) struct Expression {
    pub kind: ExpressionKind,
    pub span: Range<usize>,
    pub(super) depth: usize,
}

#[derive(Debug)]
pub(crate) enum ExpressionKind {
    Integer(String),
    Sizeof {
        annotation: Idx<TypeAnnotation>,
    },
    /// A floating-point literal's source spelling, kept exactly so constant
    /// evaluation can read the value the program wrote.
    Floating(String),
    Boolean(bool),
    Null,
    /// A string literal's decoded bytes, without the terminating zero.
    String(Vec<u8>),
    /// A character literal's byte.
    Character(u8),
    Reference(QualifiedName),
    Grouping {
        expression: Idx<Expression>,
    },
    Unary {
        operator: UnaryOperator,
        operator_span: Range<usize>,
        operand: Idx<Expression>,
    },
    Binary {
        operator: BinaryOperator,
        operator_span: Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    Comparison {
        operator: ComparisonOperator,
        operator_span: Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    Logical {
        operator: LogicalOperator,
        operator_span: Range<usize>,
        left: Idx<Expression>,
        right: Idx<Expression>,
    },
    LogicalNot {
        operator_span: Range<usize>,
        operand: Idx<Expression>,
    },
    /// `&x`, a pointer to the location `x`.
    AddressOf {
        #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
        operator_span: Range<usize>,
        operand: Idx<Expression>,
    },
    /// `*p`, the target of the pointer `p`.
    Dereference {
        #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
        operator_span: Range<usize>,
        operand: Idx<Expression>,
    },
    Conversion {
        destination: Scalar,
        operand: Idx<Expression>,
    },
    /// `*T(p)`, where `destination` is the whole `*T` annotation.
    PointerConversion {
        destination: Idx<TypeAnnotation>,
        operand: Idx<Expression>,
    },
    /// A directly named function call.
    Call(Call),
    ArrayLiteral {
        elements: Vec<Idx<Expression>>,
    },
    StructLiteral {
        name: QualifiedName,
        fields: Vec<FieldInitializer>,
    },
    Field {
        operand: Idx<Expression>,
        name: Spur,
        name_span: Range<usize>,
    },
    Index {
        operand: Idx<Expression>,
        index: Idx<Expression>,
    },
    Length {
        operand: Idx<Expression>,
    },
}

/// One source file's declarations. Imports are file-local, so they are kept
/// per file rather than pooled across the module.
#[derive(Debug)]
pub(crate) struct FileSyntax {
    pub imports: Vec<Import>,
    pub items: Vec<TopLevelItem>,
}

#[derive(Debug, Default)]
pub(crate) struct Syntax {
    pub names: Rodeo,
    pub files: Vec<FileSyntax>,
    pub functions: Arena<Function>,
    pub structs: Arena<StructDeclaration>,
    pub statements: Arena<Statement>,
    pub expressions: Arena<Expression>,
    pub annotations: Arena<TypeAnnotation>,
}

/// The first call anywhere in `expression`, in source order.
pub(crate) fn find_call(syntax: &Syntax, expression: Idx<Expression>) -> Option<Idx<Expression>> {
    let mut call = None;
    walk_expression(syntax, expression, &mut |id| {
        if call.is_none() && matches!(syntax.expressions[id].kind, ExpressionKind::Call(_)) {
            call = Some(id);
        }
    });
    call
}

/// Visits `expression` and every sub-expression under it, in source order.
pub(crate) fn walk_expression(
    syntax: &Syntax,
    expression: Idx<Expression>,
    visit: &mut impl FnMut(Idx<Expression>),
) {
    visit(expression);
    match &syntax.expressions[expression].kind {
        ExpressionKind::Sizeof { .. }
        | ExpressionKind::Integer(_)
        | ExpressionKind::Floating(_)
        | ExpressionKind::Boolean(_)
        | ExpressionKind::Null
        | ExpressionKind::String(_)
        | ExpressionKind::Character(_)
        | ExpressionKind::Reference(_) => {}
        ExpressionKind::Grouping { expression }
        | ExpressionKind::Unary {
            operand: expression,
            ..
        }
        | ExpressionKind::Conversion {
            operand: expression,
            ..
        }
        | ExpressionKind::LogicalNot {
            operand: expression,
            ..
        }
        | ExpressionKind::Length {
            operand: expression,
        }
        | ExpressionKind::AddressOf {
            operand: expression,
            ..
        }
        | ExpressionKind::Dereference {
            operand: expression,
            ..
        }
        | ExpressionKind::PointerConversion {
            operand: expression,
            ..
        }
        | ExpressionKind::Field {
            operand: expression,
            ..
        } => walk_expression(syntax, *expression, visit),
        ExpressionKind::Binary { left, right, .. }
        | ExpressionKind::Comparison { left, right, .. }
        | ExpressionKind::Logical { left, right, .. }
        | ExpressionKind::Index {
            operand: left,
            index: right,
            ..
        } => {
            walk_expression(syntax, *left, visit);
            walk_expression(syntax, *right, visit);
        }
        ExpressionKind::Call(call) => {
            for &argument in &call.arguments {
                walk_expression(syntax, argument, visit);
            }
        }
        ExpressionKind::ArrayLiteral { elements, .. } => {
            for &element in elements {
                walk_expression(syntax, element, visit);
            }
        }
        ExpressionKind::StructLiteral { fields, .. } => {
            for field in fields {
                walk_expression(syntax, field.value, visit);
            }
        }
    }
}
