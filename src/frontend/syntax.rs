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
    pub parameters: Vec<Parameter>,
    pub result: FunctionResult,
    #[cfg_attr(not(test), expect(dead_code, reason = "preserved in syntax snapshots"))]
    pub span: Range<usize>,
    pub body: Vec<Idx<Statement>>,
}

#[derive(Debug)]
pub(crate) struct Parameter {
    pub name: Spur,
    pub name_span: Range<usize>,
    pub annotation: Idx<TypeAnnotation>,
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
        target: AssignmentTarget,
        value: Idx<Expression>,
    },
    CompoundAssignment {
        target: AssignmentTarget,
        operator: BinaryOperator,
        operator_span: Range<usize>,
        value: Idx<Expression>,
    },
    Block {
        body: Vec<Idx<Statement>>,
    },
    Exit {
        argument: Idx<Expression>,
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
        label: Option<Label>,
        header: ForHeader,
        body: Vec<Idx<Statement>>,
    },
    Break {
        label: Option<Label>,
    },
    Continue {
        label: Option<Label>,
    },
}

#[derive(Debug)]
pub(crate) struct Label {
    pub name: Spur,
    pub name_span: Range<usize>,
}

/// The left side of an assignment: a binding, or an element reached through
/// one index per `[ … ]` group.
#[derive(Debug)]
pub(crate) struct AssignmentTarget {
    pub name: QualifiedName,
    pub indices: Vec<Idx<Expression>>,
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
    ThreeClause {
        initializer: Idx<Statement>,
        condition: Idx<Expression>,
        post: Idx<Statement>,
    },
    Iteration {
        value: Spur,
        index: Option<(Spur, Range<usize>)>,
        operand: Idx<Expression>,
    },
}

#[derive(Debug)]
pub(crate) struct TypeAnnotation {
    pub kind: AnnotationKind,
    pub span: Range<usize>,
}

/// A written type. An array's length is an unevaluated expression, and `None`
/// is the `[_]` length taken from an initializer.
#[derive(Debug)]
pub(crate) enum AnnotationKind {
    Named(Scalar),
    Array {
        length: Option<Idx<Expression>>,
        element: Idx<TypeAnnotation>,
    },
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
    /// A floating-point literal's source spelling, kept exactly so constant
    /// evaluation can read the value the program wrote.
    Floating(String),
    Boolean(bool),
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
    Conversion {
        destination: Scalar,
        truncating: bool,
        operand: Idx<Expression>,
    },
    Call(Call),
    /// `[a, b, c]`, where `fill` is the span of a trailing `...`.
    ArrayLiteral {
        elements: Vec<Idx<Expression>>,
        fill: Option<Range<usize>>,
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
        ExpressionKind::Integer(_)
        | ExpressionKind::Floating(_)
        | ExpressionKind::Boolean(_)
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
    }
}
