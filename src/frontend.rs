#[cfg(test)]
use crate::source::SourceMap;
use crate::{diagnostic::Diagnostic, source::Source, types::Scalar};
use la_arena::{Arena, Idx};
use lasso::{Rodeo, Spur};
use logos::Logos;
use std::ops::Range;

#[derive(Logos, Debug, Clone, Copy, PartialEq)]
#[logos(skip r"[ \t\n\r\x0B\x0C]+")]
enum Token {
    #[token("fn")]
    Fn,
    #[token("void")]
    Void,
    #[token("const")]
    Const,
    #[token("var")]
    Var,
    #[token("exit")]
    Exit,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("for")]
    For,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,
    #[token("in")]
    In,
    #[token("len")]
    Len,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[regex("[0-9][a-zA-Z0-9_]*")]
    Integer,
    #[token("pub")]
    Pub,
    #[token("use")]
    Use,
    #[token("::")]
    ColonColon,
    #[token(":")]
    Colon,
    #[token("=")]
    Equals,
    #[token("==")]
    EqualsEquals,
    #[token("!=")]
    NotEquals,
    #[token("<=")]
    LessEqual,
    #[token(">=")]
    GreaterEqual,
    #[token("<")]
    Less,
    #[token(">")]
    Greater,
    #[token(";")]
    Semicolon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("+%")]
    WrappingPlus,
    #[token("+%=")]
    WrappingPlusEquals,
    #[token("-%")]
    WrappingMinus,
    #[token("-%=")]
    WrappingMinusEquals,
    #[token("*%")]
    WrappingStar,
    #[token("*%=")]
    WrappingStarEquals,
    #[token("<<")]
    ShiftLeft,
    #[token("<<=")]
    ShiftLeftEquals,
    #[token(">>")]
    ShiftRight,
    #[token(">>=")]
    ShiftRightEquals,
    #[token("+")]
    Plus,
    #[token("+=")]
    PlusEquals,
    #[token("-")]
    Minus,
    #[token("-=")]
    MinusEquals,
    #[token("*")]
    Star,
    #[token("*=")]
    StarEquals,
    #[token("/")]
    Slash,
    #[token("/=")]
    SlashEquals,
    #[token("%")]
    Percent,
    #[token("%=")]
    PercentEquals,
    #[token("&")]
    Ampersand,
    #[token("&&")]
    LogicalAnd,
    #[token("&=")]
    AmpersandEquals,
    #[token("^")]
    Caret,
    #[token("^=")]
    CaretEquals,
    #[token("|")]
    Pipe,
    #[token("||")]
    LogicalOr,
    #[token("|=")]
    PipeEquals,
    #[token("!")]
    Bang,
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*")]
    Name,
    #[token("(")]
    LeftParen,
    #[token(")")]
    RightParen,
    #[token("->")]
    Arrow,
    #[token("{")]
    LeftBrace,
    #[token("}")]
    RightBrace,
    #[token("[")]
    LeftBracket,
    #[token("]")]
    RightBracket,
    #[token("...")]
    Ellipsis,
    #[regex(r"//[^\r\n]*", logos::skip, allow_greedy = true)]
    LineComment,
    #[token("/*", block_comment)]
    BlockComment,
}

fn block_comment(lexer: &mut logos::Lexer<'_, Token>) -> Result<logos::Skip, ()> {
    let bytes = lexer.remainder().as_bytes();
    let mut depth = 1;
    let mut i = 0;
    while i + 1 < bytes.len() {
        match &bytes[i..i + 2] {
            b"/*" => {
                depth += 1;
                i += 2;
            }
            b"*/" => {
                depth -= 1;
                i += 2;
                if depth == 0 {
                    lexer.bump(i);
                    return Ok(logos::Skip);
                }
            }
            _ => i += 1,
        }
    }
    lexer.bump(bytes.len());
    Err(())
}

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
    depth: usize,
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
enum InfixOperator {
    Integer(BinaryOperator),
    Comparison(ComparisonOperator),
    Logical(LogicalOperator),
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
}

#[derive(Debug)]
pub(crate) enum ExpressionKind {
    Integer(String),
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

fn reserved(name: &str) -> bool {
    matches!(
        name,
        "const"
            | "var"
            | "true"
            | "false"
            | "fn"
            | "void"
            | "exit"
            | "bool"
            | "if"
            | "else"
            | "for"
            | "break"
            | "continue"
            | "return"
            | "in"
            | "len"
            | "pub"
            | "use"
    ) || Scalar::named(name).is_some()
}

pub(crate) fn integer_parts(spelling: &str) -> (u32, &str, &str) {
    let (digits, base) = if let Some(digits) = spelling.strip_prefix("0x") {
        (digits, 16)
    } else if let Some(digits) = spelling.strip_prefix("0b") {
        (digits, 2)
    } else if let Some(digits) = spelling.strip_prefix("0o") {
        (digits, 8)
    } else {
        (spelling, 10)
    };
    let end = digits
        .bytes()
        .take_while(|b| char::from(*b).is_digit(base))
        .count();
    (base, &digits[..end], &digits[end..])
}

pub(crate) const MAX_INTEGER_LITERAL_DIGITS: usize = 4_096;

fn valid_integer(spelling: &str) -> Result<(), String> {
    let (_, digits, suffix) = integer_parts(spelling);
    if digits.is_empty() || !suffix.is_empty() {
        return Err("malformed integer literal".to_owned());
    }
    if digits.len() > MAX_INTEGER_LITERAL_DIGITS {
        return Err(format!(
            "integer literal exceeds compiler limit of {MAX_INTEGER_LITERAL_DIGITS} digits"
        ));
    }
    Ok(())
}

struct Parser<'a> {
    lexer: logos::Lexer<'a, Token>,
    current: Option<Token>,
    span: Range<usize>,
    /// The file's start in the source map's shared offset space, added to
    /// every span the lexer reports.
    base: usize,
    syntax: &'a mut Syntax,
    nesting: usize,
}

// Bound recursive parsing, checking, and lowering before entering another level.
const MAX_NESTING: usize = 128;

/// Parses one source file, appending its declarations to `syntax`, which holds
/// the whole program's name interner and arenas.
pub(crate) fn parse_file(syntax: &mut Syntax, source: &Source) -> Result<(), Diagnostic> {
    let mut parser = Parser {
        lexer: Token::lexer(&source.text),
        current: None,
        span: source.base..source.base,
        base: source.base,
        syntax,
        nesting: 0,
    };
    let file = parser.file()?;
    parser.syntax.files.push(file);
    Ok(())
}

/// Parses every file of one module into a single `Syntax`.
#[cfg(test)]
pub(crate) fn parse(sources: &SourceMap) -> Result<Syntax, Diagnostic> {
    let mut syntax = Syntax::default();
    for source in sources.files() {
        parse_file(&mut syntax, source)?;
    }
    Ok(syntax)
}

impl Parser<'_> {
    fn file(&mut self) -> Result<FileSyntax, Diagnostic> {
        self.advance()?;
        let mut imports = Vec::new();
        while self.current == Some(Token::Use) {
            imports.push(self.import()?);
        }
        let mut items = Vec::new();
        while self.current.is_some() {
            let public_span = (self.current == Some(Token::Pub)).then(|| self.span.clone());
            if public_span.is_some() {
                self.advance()?;
            }
            let public = public_span.is_some();
            items.push(match self.current {
                Some(Token::Fn) => TopLevelItem::Function {
                    function: self.function()?,
                    public,
                },
                Some(Token::Const) | Some(Token::Var) => TopLevelItem::Binding {
                    binding: self.top_level_binding()?,
                    public,
                },
                Some(Token::Use) => {
                    return Err(match public_span {
                        Some(span) => {
                            Diagnostic::new(span, "`pub` is not permitted on a `use` declaration")
                        }
                        None => self.error("`use` declarations must precede the first declaration"),
                    });
                }
                _ => return Err(self.error("expected a top-level declaration")),
            });
        }
        Ok(FileSyntax { imports, items })
    }

    fn enter_nesting(&mut self) -> Result<(), Diagnostic> {
        if self.nesting == MAX_NESTING {
            return Err(self.nesting_error());
        }
        self.nesting += 1;
        Ok(())
    }

    fn advance(&mut self) -> Result<(), Diagnostic> {
        let token = self.lexer.next();
        self.span = if token.is_some() {
            let span = self.lexer.span();
            self.base + span.start..self.base + span.end
        } else {
            let end = self.base + self.lexer.source().len();
            end..end
        };
        self.current = match token {
            Some(Ok(Token::Integer)) => {
                if let Err(message) = valid_integer(self.lexer.slice()) {
                    return Err(self.error(message));
                }
                Some(Token::Integer)
            }
            Some(Ok(token)) => Some(token),
            Some(Err(())) => {
                return Err(self.error(if self.lexer.slice().starts_with("/*") {
                    "unterminated block comment"
                } else {
                    "invalid token"
                }));
            }
            None => None,
        };
        Ok(())
    }

    fn error(&self, message: impl Into<String>) -> Diagnostic {
        Diagnostic::new(self.span.clone(), message)
    }

    fn nesting_error(&self) -> Diagnostic {
        self.nesting_error_at(self.span.clone())
    }

    fn nesting_error_at(&self, span: Range<usize>) -> Diagnostic {
        Diagnostic::new(
            span,
            format!("source nesting exceeds compiler limit of {MAX_NESTING}"),
        )
    }

    fn expect(&mut self, token: Token, message: &str) -> Result<Range<usize>, Diagnostic> {
        if self.current != Some(token) {
            return Err(self.error(message));
        }
        let span = self.span.clone();
        self.advance()?;
        Ok(span)
    }

    fn name(&mut self, expected: &str) -> Result<(Spur, Range<usize>), Diagnostic> {
        if self.current.is_some() && reserved(self.lexer.slice()) {
            return Err(self.error("reserved word cannot be used as an identifier"));
        }
        if self.current != Some(Token::Name) {
            return Err(self.error(expected));
        }
        let name = self.syntax.names.get_or_intern(self.lexer.slice());
        let span = self.span.clone();
        self.advance()?;
        Ok((name, span))
    }

    fn path_component(&mut self, expected: &str) -> Result<PathComponent, Diagnostic> {
        let (name, name_span) = self.name(expected)?;
        Ok(PathComponent { name, name_span })
    }

    fn import(&mut self) -> Result<Import, Diagnostic> {
        let start = self.expect(Token::Use, "expected `use`")?.start;
        let mut path = vec![self.path_component("expected a module name after `use`")?];
        let mut selection = None;
        while self.current == Some(Token::ColonColon) {
            self.advance()?;
            if self.current == Some(Token::LeftBrace) {
                selection = Some(self.import_selection()?);
                break;
            }
            path.push(self.path_component("expected a name after `::`")?);
        }
        let end = self
            .expect(Token::Semicolon, "expected `;` after import path")?
            .end;
        Ok(Import {
            path,
            selection,
            span: start..end,
        })
    }

    fn import_selection(&mut self) -> Result<Vec<PathComponent>, Diagnostic> {
        self.expect(Token::LeftBrace, "expected `{`")?;
        if self.current == Some(Token::RightBrace) {
            return Err(self.error("expected an imported name"));
        }
        let mut selection = Vec::new();
        loop {
            selection.push(self.path_component("expected an imported name")?);
            if self.current != Some(Token::Comma) {
                break;
            }
            self.advance()?;
            if self.current == Some(Token::RightBrace) {
                break;
            }
        }
        self.expect(Token::RightBrace, "expected `}` after imported names")?;
        Ok(selection)
    }

    /// Parses a `name` or `module::name`. Longer paths appear only in `use`.
    fn qualified_name(&mut self, expected: &str) -> Result<QualifiedName, Diagnostic> {
        let (name, name_span) = self.name(expected)?;
        if self.current != Some(Token::ColonColon) {
            let span = name_span.clone();
            return Ok(QualifiedName {
                qualifier: None,
                name,
                name_span,
                span,
            });
        }
        self.advance()?;
        let qualifier = PathComponent { name, name_span };
        let (name, name_span) = self.name("expected a name after `::`")?;
        let span = qualifier.name_span.start..name_span.end;
        Ok(QualifiedName {
            qualifier: Some(qualifier),
            name,
            name_span,
            span,
        })
    }

    fn function(&mut self) -> Result<Idx<Function>, Diagnostic> {
        let start = self.expect(Token::Fn, "expected `fn`")?.start;
        let (name, name_span) = self.name("expected a function name")?;
        let parameters = self.parameters()?;
        self.expect(Token::Arrow, "expected `->`")?;
        let result = if self.current == Some(Token::Void) {
            self.advance()?;
            FunctionResult::Void
        } else {
            FunctionResult::Value(self.type_annotation()?)
        };
        let (body, end) = self.body()?;
        Ok(self.syntax.functions.alloc(Function {
            name,
            name_span,
            parameters,
            result,
            span: start..end,
            body,
        }))
    }

    fn parameters(&mut self) -> Result<Vec<Parameter>, Diagnostic> {
        self.expect(Token::LeftParen, "expected `(`")?;
        let mut parameters = Vec::new();
        if self.current != Some(Token::RightParen) {
            loop {
                let (name, name_span) = self.name("expected a parameter name")?;
                self.expect(Token::Colon, "expected `:` after parameter name")?;
                let annotation = self.type_annotation()?;
                parameters.push(Parameter {
                    name,
                    name_span,
                    annotation,
                });
                if self.current != Some(Token::Comma) {
                    break;
                }
                self.advance()?;
                if self.current == Some(Token::RightParen) {
                    break;
                }
            }
        }
        self.expect(Token::RightParen, "expected `)` after parameter list")?;
        Ok(parameters)
    }

    fn top_level_binding(&mut self) -> Result<Idx<Statement>, Diagnostic> {
        let start = self.span.start;
        let kind = self.binding()?;
        let end = self.expect(Token::Semicolon, "expected `;`")?.end;
        Ok(self.syntax.statements.alloc(Statement {
            kind,
            span: start..end,
        }))
    }

    fn body(&mut self) -> Result<(Vec<Idx<Statement>>, usize), Diagnostic> {
        self.enter_nesting()?;
        self.expect(Token::LeftBrace, "expected `{`")?;
        let mut body = Vec::new();
        while self.current != Some(Token::RightBrace) {
            if self.current.is_none() {
                return Err(self.error("expected `}`"));
            }
            body.push(self.statement()?);
        }
        let end = self.expect(Token::RightBrace, "expected `}`")?.end;
        self.nesting -= 1;
        Ok((body, end))
    }

    fn statement(&mut self) -> Result<Idx<Statement>, Diagnostic> {
        let start = self.span.start;
        if self.current == Some(Token::Pub) {
            return Err(self.error("`pub` is not permitted on a local declaration"));
        }
        match self.current {
            Some(Token::LeftBrace) => {
                let (body, end) = self.body()?;
                return Ok(self.syntax.statements.alloc(Statement {
                    kind: StatementKind::Block { body },
                    span: start..end,
                }));
            }
            Some(Token::If) => return self.if_statement(),
            Some(Token::For) => return self.for_statement(),
            _ => {}
        }
        let kind = match self.current {
            Some(Token::Const) | Some(Token::Var) => self.binding()?,
            Some(Token::Name) if self.starts_call() => StatementKind::Call { call: self.call()? },
            Some(Token::Name) => self.assignment()?,
            Some(Token::Exit) => {
                self.advance()?;
                let (argument, _) = self.single_argument("exit")?;
                StatementKind::Exit { argument }
            }
            Some(Token::Break) => {
                self.advance()?;
                StatementKind::Break {
                    label: self.optional_label("expected a label after `:`")?,
                }
            }
            Some(Token::Continue) => {
                self.advance()?;
                StatementKind::Continue {
                    label: self.optional_label("expected a label after `:`")?,
                }
            }
            Some(Token::Return) => {
                self.advance()?;
                let value = (self.current != Some(Token::Semicolon))
                    .then(|| self.expression())
                    .transpose()?;
                StatementKind::Return { value }
            }
            _ => {
                return Err(self.error(
                    "expected a declaration, assignment, call, block, `if`, `for`, loop control, `return`, or `exit`",
                ));
            }
        };
        let end = self.expect(Token::Semicolon, "expected `;`")?.end;
        Ok(self.syntax.statements.alloc(Statement {
            kind,
            span: start..end,
        }))
    }

    /// Parses the `( expression [,] )` of `exit` or `len`, which use call
    /// syntax without being calls. Returns the operand and the `)` span.
    fn single_argument(
        &mut self,
        keyword: &str,
    ) -> Result<(Idx<Expression>, Range<usize>), Diagnostic> {
        self.expect(Token::LeftParen, "expected `(`")?;
        if self.current == Some(Token::RightParen) {
            return Err(self.error(format!("{keyword} requires one argument")));
        }
        let argument = self.expression()?;
        if self.current == Some(Token::Comma) {
            self.advance()?;
            if self.current != Some(Token::RightParen) {
                return Err(self.error(format!("{keyword} takes one argument")));
            }
        }
        let right_paren_span = self.expect(
            Token::RightParen,
            &format!("expected `)` after {keyword} argument"),
        )?;
        Ok((argument, right_paren_span))
    }

    /// Parses an assignment target: a name followed by one index per `[ … ]`.
    fn assignment_target(&mut self) -> Result<AssignmentTarget, Diagnostic> {
        let name = self.qualified_name("expected an assignment target")?;
        let mut indices = Vec::new();
        while self.current == Some(Token::LeftBracket) {
            self.enter_nesting()?;
            self.advance()?;
            indices.push(self.expression()?);
            self.expect(Token::RightBracket, "expected `]` after index")?;
            self.nesting -= 1;
        }
        Ok(AssignmentTarget { name, indices })
    }

    fn assignment(&mut self) -> Result<StatementKind, Diagnostic> {
        let target = self.assignment_target()?;
        let (operator, operator_span) = self.assignment_operator()?;
        let value = self.expression()?;
        Ok(if let Some(operator) = operator {
            StatementKind::CompoundAssignment {
                target,
                operator,
                operator_span,
                value,
            }
        } else {
            StatementKind::Assignment { target, value }
        })
    }

    fn assignment_operator(
        &mut self,
    ) -> Result<(Option<BinaryOperator>, Range<usize>), Diagnostic> {
        let operator = match self.current {
            Some(Token::Equals) => None,
            Some(Token::PlusEquals) => Some(BinaryOperator::Add),
            Some(Token::MinusEquals) => Some(BinaryOperator::Subtract),
            Some(Token::StarEquals) => Some(BinaryOperator::Multiply),
            Some(Token::SlashEquals) => Some(BinaryOperator::Divide),
            Some(Token::PercentEquals) => Some(BinaryOperator::Remainder),
            Some(Token::AmpersandEquals) => Some(BinaryOperator::And),
            Some(Token::PipeEquals) => Some(BinaryOperator::Or),
            Some(Token::CaretEquals) => Some(BinaryOperator::Xor),
            Some(Token::ShiftLeftEquals) => Some(BinaryOperator::ShiftLeft),
            Some(Token::ShiftRightEquals) => Some(BinaryOperator::ShiftRight),
            Some(Token::WrappingPlusEquals) => Some(BinaryOperator::WrappingAdd),
            Some(Token::WrappingMinusEquals) => Some(BinaryOperator::WrappingSubtract),
            Some(Token::WrappingStarEquals) => Some(BinaryOperator::WrappingMultiply),
            _ => return Err(self.error("expected `=`")),
        };
        let span = self.span.clone();
        self.advance()?;
        Ok((operator, span))
    }

    fn optional_label(&mut self, expected: &str) -> Result<Option<Label>, Diagnostic> {
        if self.current != Some(Token::Colon) {
            return Ok(None);
        }
        self.advance()?;
        let (name, name_span) = self.name(expected)?;
        Ok(Some(Label { name, name_span }))
    }

    fn if_statement(&mut self) -> Result<Idx<Statement>, Diagnostic> {
        let start = self.expect(Token::If, "expected `if`")?.start;
        let condition = self.expression()?;
        let (then_body, mut end) = self.body()?;
        let else_branch = if self.current == Some(Token::Else) {
            self.advance()?;
            let branch = if self.current == Some(Token::If) {
                self.enter_nesting()?;
                let branch = self.if_statement()?;
                self.nesting -= 1;
                branch
            } else {
                let block_start = self.span.start;
                let (body, block_end) = self.body()?;
                self.syntax.statements.alloc(Statement {
                    kind: StatementKind::Block { body },
                    span: block_start..block_end,
                })
            };
            end = self.syntax.statements[branch].span.end;
            Some(branch)
        } else {
            None
        };
        Ok(self.syntax.statements.alloc(Statement {
            kind: StatementKind::If {
                condition,
                then_body,
                else_branch,
            },
            span: start..end,
        }))
    }

    fn for_statement(&mut self) -> Result<Idx<Statement>, Diagnostic> {
        let start = self.expect(Token::For, "expected `for`")?.start;
        let label = self.optional_label("expected a loop label after `:`")?;
        let header = self.for_header()?;
        let (body, end) = self.body()?;
        Ok(self.syntax.statements.alloc(Statement {
            kind: StatementKind::For {
                label,
                header,
                body,
            },
            span: start..end,
        }))
    }

    fn for_header(&mut self) -> Result<ForHeader, Diagnostic> {
        if self.current == Some(Token::LeftBrace) {
            return Ok(ForHeader::Infinite);
        }
        if self.starts_iteration() {
            return self.iteration_header();
        }
        if matches!(self.current, Some(Token::Const | Token::Var)) || self.starts_assignment() {
            return self.three_clause_header();
        }
        Ok(ForHeader::Condition(self.expression()?))
    }

    fn three_clause_header(&mut self) -> Result<ForHeader, Diagnostic> {
        let initializer_start = self.span.start;
        let initializer_kind = if matches!(self.current, Some(Token::Const | Token::Var)) {
            self.binding()?
        } else {
            self.assignment()?
        };
        if let StatementKind::CompoundAssignment { operator_span, .. } = &initializer_kind {
            return Err(Diagnostic::new(
                operator_span.clone(),
                "for initializer does not permit compound assignment",
            ));
        }
        let initializer_end = self
            .expect(Token::Semicolon, "expected `;` after for initializer")?
            .end;
        let initializer = self.syntax.statements.alloc(Statement {
            kind: initializer_kind,
            span: initializer_start..initializer_end,
        });
        let condition = self.expression()?;
        self.expect(Token::Semicolon, "expected `;` after for condition")?;
        if !self.starts_assignment() {
            return Err(self.error("expected an assignment after second `;`"));
        }
        let post_start = self.span.start;
        let post_kind = self.assignment()?;
        let post_end = match &post_kind {
            StatementKind::Assignment { value, .. }
            | StatementKind::CompoundAssignment { value, .. } => {
                self.syntax.expressions[*value].span.end
            }
            _ => unreachable!(),
        };
        let post = self.syntax.statements.alloc(Statement {
            kind: post_kind,
            span: post_start..post_end,
        });
        Ok(ForHeader::ThreeClause {
            initializer,
            condition,
            post,
        })
    }

    fn iteration_header(&mut self) -> Result<ForHeader, Diagnostic> {
        let (value, _) = self.name("expected a loop variable")?;
        let index = (self.current == Some(Token::Comma))
            .then(|| {
                self.advance()?;
                self.name("expected an index variable after `,`")
            })
            .transpose()?;
        self.expect(Token::In, "expected `in`")?;
        let operand = self.expression()?;
        Ok(ForHeader::Iteration {
            value,
            index,
            operand,
        })
    }

    /// Reports whether a `for` header starts the `v in a` or `v, i in a` form.
    /// No condition can hold `in` or `,` at that position.
    fn starts_iteration(&self) -> bool {
        if self.current != Some(Token::Name) || reserved(self.lexer.slice()) {
            return false;
        }
        let mut lexer = self.lexer.clone();
        matches!(lexer.next(), Some(Ok(Token::In | Token::Comma)))
    }

    /// Returns the token that follows an assignment target starting at the
    /// current token, and whether the target carried any index brackets.
    /// `None` when no target starts here.
    fn token_after_target(&self) -> Option<(Token, bool)> {
        if self.current != Some(Token::Name) || reserved(self.lexer.slice()) {
            return None;
        }
        let mut lexer = self.lexer.clone();
        let mut following = lexer.next();
        if following == Some(Ok(Token::ColonColon)) {
            if lexer.next() != Some(Ok(Token::Name)) {
                return None;
            }
            following = lexer.next();
        }
        let mut indexed = false;
        while following == Some(Ok(Token::LeftBracket)) {
            indexed = true;
            let mut depth = 1usize;
            while depth > 0 {
                match lexer.next() {
                    Some(Ok(Token::LeftBracket)) => depth += 1,
                    Some(Ok(Token::RightBracket)) => depth -= 1,
                    Some(Ok(_)) => {}
                    Some(Err(())) | None => return None,
                }
            }
            following = lexer.next();
        }
        match following {
            Some(Ok(token)) => Some((token, indexed)),
            Some(Err(())) | None => None,
        }
    }

    fn starts_assignment(&self) -> bool {
        matches!(
            self.token_after_target(),
            Some((
                Token::Equals
                    | Token::PlusEquals
                    | Token::MinusEquals
                    | Token::StarEquals
                    | Token::SlashEquals
                    | Token::PercentEquals
                    | Token::AmpersandEquals
                    | Token::PipeEquals
                    | Token::CaretEquals
                    | Token::ShiftLeftEquals
                    | Token::ShiftRightEquals
                    | Token::WrappingPlusEquals
                    | Token::WrappingMinusEquals
                    | Token::WrappingStarEquals,
                _
            ))
        )
    }

    fn starts_call(&self) -> bool {
        self.token_after_target() == Some((Token::LeftParen, false))
    }

    fn call(&mut self) -> Result<Call, Diagnostic> {
        let target = self.qualified_name("expected a call target")?;
        self.call_parts(target)
    }

    fn call_parts(&mut self, target: QualifiedName) -> Result<Call, Diagnostic> {
        self.enter_nesting()?;
        let left_paren_span = self.expect(Token::LeftParen, "expected `(` after call target")?;
        let mut arguments = Vec::new();
        if self.current != Some(Token::RightParen) {
            loop {
                arguments.push(self.expression()?);
                match self.current {
                    Some(Token::Comma) => {
                        self.advance()?;
                        if self.current == Some(Token::RightParen) {
                            break;
                        }
                    }
                    Some(Token::RightParen) => break,
                    _ => return Err(self.error("expected `,` or `)` after call argument")),
                }
            }
        }
        let right_paren_span =
            self.expect(Token::RightParen, "expected `)` after call arguments")?;
        self.nesting -= 1;
        Ok(Call {
            target,
            left_paren_span,
            right_paren_span,
            arguments,
        })
    }

    fn binding(&mut self) -> Result<StatementKind, Diagnostic> {
        let mutable = self.current == Some(Token::Var);
        self.advance()?;
        let (name, name_span) = self.name(if mutable {
            "expected a binding name after `var`"
        } else {
            "expected a binding name after `const`"
        })?;
        let annotation = if self.current == Some(Token::Colon) {
            self.advance()?;
            Some(self.type_annotation()?)
        } else {
            None
        };
        self.expect(Token::Equals, "expected `=`")?;
        let initializer = self.expression()?;
        Ok(StatementKind::Binding {
            mutable,
            name,
            name_span,
            annotation,
            initializer,
        })
    }

    fn named_type(&mut self) -> Result<(Scalar, Range<usize>), Diagnostic> {
        let Some(ty) = (self.current == Some(Token::Name))
            .then(|| Scalar::named(self.lexer.slice()))
            .flatten()
        else {
            return Err(self.error("expected a type"));
        };
        let span = self.span.clone();
        self.advance()?;
        Ok((ty, span))
    }

    /// Parses a type annotation, which is a type name or one or more `[ … ]`
    /// lengths in front of an element annotation.
    fn type_annotation(&mut self) -> Result<Idx<TypeAnnotation>, Diagnostic> {
        let start = self.span.start;
        if self.current != Some(Token::LeftBracket) {
            let (ty, span) = self.named_type()?;
            return Ok(self.syntax.annotations.alloc(TypeAnnotation {
                kind: AnnotationKind::Named(ty),
                span,
            }));
        }
        self.enter_nesting()?;
        self.advance()?;
        let length = if self.current == Some(Token::Name) && self.lexer.slice() == "_" {
            self.advance()?;
            None
        } else {
            Some(self.expression()?)
        };
        self.expect(Token::RightBracket, "expected `]` after array length")?;
        let element = self.type_annotation()?;
        self.nesting -= 1;
        let span = start..self.syntax.annotations[element].span.end;
        Ok(self.syntax.annotations.alloc(TypeAnnotation {
            kind: AnnotationKind::Array { length, element },
            span,
        }))
    }

    fn expression(&mut self) -> Result<Idx<Expression>, Diagnostic> {
        self.binary_expression(1)
    }

    fn binary_expression(&mut self, minimum_precedence: u8) -> Result<Idx<Expression>, Diagnostic> {
        let mut left = self.unary_expression()?;
        while let Some((operator, precedence)) = self.binary_operator() {
            if precedence < minimum_precedence {
                break;
            }
            let operator_span = self.span.clone();
            self.advance()?;
            let right = self.binary_expression(precedence + 1)?;
            let depth = self.syntax.expressions[left]
                .depth
                .max(self.syntax.expressions[right].depth)
                + 1;
            if self.nesting + depth > MAX_NESTING {
                return Err(self.nesting_error_at(operator_span));
            }
            let span =
                self.syntax.expressions[left].span.start..self.syntax.expressions[right].span.end;
            let kind = match operator {
                InfixOperator::Integer(operator) => ExpressionKind::Binary {
                    operator,
                    operator_span,
                    left,
                    right,
                },
                InfixOperator::Comparison(operator) => ExpressionKind::Comparison {
                    operator,
                    operator_span,
                    left,
                    right,
                },
                InfixOperator::Logical(operator) => ExpressionKind::Logical {
                    operator,
                    operator_span,
                    left,
                    right,
                },
            };
            left = self
                .syntax
                .expressions
                .alloc(Expression { kind, span, depth });
        }
        Ok(left)
    }

    fn binary_operator(&self) -> Option<(InfixOperator, u8)> {
        Some(match self.current.as_ref()? {
            Token::Star => (InfixOperator::Integer(BinaryOperator::Multiply), 5),
            Token::Slash => (InfixOperator::Integer(BinaryOperator::Divide), 5),
            Token::Percent => (InfixOperator::Integer(BinaryOperator::Remainder), 5),
            Token::WrappingStar => (InfixOperator::Integer(BinaryOperator::WrappingMultiply), 5),
            Token::ShiftLeft => (InfixOperator::Integer(BinaryOperator::ShiftLeft), 5),
            Token::ShiftRight => (InfixOperator::Integer(BinaryOperator::ShiftRight), 5),
            Token::Ampersand => (InfixOperator::Integer(BinaryOperator::And), 5),
            Token::Plus => (InfixOperator::Integer(BinaryOperator::Add), 4),
            Token::Minus => (InfixOperator::Integer(BinaryOperator::Subtract), 4),
            Token::WrappingPlus => (InfixOperator::Integer(BinaryOperator::WrappingAdd), 4),
            Token::WrappingMinus => (InfixOperator::Integer(BinaryOperator::WrappingSubtract), 4),
            Token::Caret => (InfixOperator::Integer(BinaryOperator::Xor), 4),
            Token::Pipe => (InfixOperator::Integer(BinaryOperator::Or), 4),
            Token::EqualsEquals => (InfixOperator::Comparison(ComparisonOperator::Equal), 3),
            Token::NotEquals => (InfixOperator::Comparison(ComparisonOperator::NotEqual), 3),
            Token::Less => (InfixOperator::Comparison(ComparisonOperator::Less), 3),
            Token::LessEqual => (InfixOperator::Comparison(ComparisonOperator::LessEqual), 3),
            Token::Greater => (InfixOperator::Comparison(ComparisonOperator::Greater), 3),
            Token::GreaterEqual => (
                InfixOperator::Comparison(ComparisonOperator::GreaterEqual),
                3,
            ),
            Token::LogicalAnd => (InfixOperator::Logical(LogicalOperator::And), 2),
            Token::LogicalOr => (InfixOperator::Logical(LogicalOperator::Or), 1),
            _ => return None,
        })
    }

    fn unary_expression(&mut self) -> Result<Idx<Expression>, Diagnostic> {
        if self.current == Some(Token::Bang) {
            self.enter_nesting()?;
            let operator_span = self.span.clone();
            self.advance()?;
            let operand = self.unary_expression()?;
            self.nesting -= 1;
            return Ok(self.syntax.expressions.alloc(Expression {
                span: operator_span.start..self.syntax.expressions[operand].span.end,
                depth: self.syntax.expressions[operand].depth + 1,
                kind: ExpressionKind::LogicalNot {
                    operator_span,
                    operand,
                },
            }));
        }
        let operator = match self.current {
            Some(Token::Minus) => UnaryOperator::Negate,
            Some(Token::WrappingMinus) => UnaryOperator::WrappingNegate,
            Some(Token::Caret) => UnaryOperator::Complement,
            _ => return self.postfix_expression(),
        };
        self.enter_nesting()?;
        let operator_span = self.span.clone();
        self.advance()?;
        let operand = self.unary_expression()?;
        self.nesting -= 1;
        Ok(self.syntax.expressions.alloc(Expression {
            span: operator_span.start..self.syntax.expressions[operand].span.end,
            depth: self.syntax.expressions[operand].depth + 1,
            kind: ExpressionKind::Unary {
                operator,
                operator_span,
                operand,
            },
        }))
    }

    /// Parses a primary expression and the `[ … ]` indexes that follow it, so
    /// `grid[r][c]` indexes the row first and `f(x)[0]` indexes the result.
    fn postfix_expression(&mut self) -> Result<Idx<Expression>, Diagnostic> {
        let mut operand = self.primary_expression()?;
        while self.current == Some(Token::LeftBracket) {
            let bracket_span = self.span.clone();
            self.enter_nesting()?;
            self.advance()?;
            let index = self.expression()?;
            let end = self
                .expect(Token::RightBracket, "expected `]` after index")?
                .end;
            self.nesting -= 1;
            let depth = self.syntax.expressions[operand]
                .depth
                .max(self.syntax.expressions[index].depth)
                + 1;
            if self.nesting + depth > MAX_NESTING {
                return Err(self.nesting_error_at(bracket_span));
            }
            operand = self.syntax.expressions.alloc(Expression {
                span: self.syntax.expressions[operand].span.start..end,
                depth,
                kind: ExpressionKind::Index { operand, index },
            });
        }
        Ok(operand)
    }

    fn primary_expression(&mut self) -> Result<Idx<Expression>, Diagnostic> {
        let span = self.span.clone();
        let kind = if self.current == Some(Token::Integer) {
            let spelling = self.lexer.slice().to_owned();
            self.advance()?;
            ExpressionKind::Integer(spelling)
        } else if self.current == Some(Token::Name)
            && Scalar::named(self.lexer.slice()).is_some_and(Scalar::is_integer)
        {
            return self.conversion_expression(span);
        } else if matches!(self.current, Some(Token::True | Token::False)) {
            let value = self.current == Some(Token::True);
            self.advance()?;
            ExpressionKind::Boolean(value)
        } else if self.current == Some(Token::LeftParen) {
            return self.grouping_expression(span);
        } else if self.current == Some(Token::LeftBracket) {
            return self.array_literal(span);
        } else if self.current == Some(Token::Len) {
            return self.length_expression(span);
        } else {
            return self.name_expression(span);
        };
        Ok(self.syntax.expressions.alloc(Expression {
            kind,
            span,
            depth: 0,
        }))
    }

    /// Parses `[e1, e2]`, with an optional trailing comma, or `[e1, last...]`,
    /// whose `...` repeats the last element across the remaining elements.
    fn array_literal(&mut self, span: Range<usize>) -> Result<Idx<Expression>, Diagnostic> {
        self.enter_nesting()?;
        self.advance()?;
        if self.current == Some(Token::RightBracket) {
            return Err(self.error("expected an array element"));
        }
        let mut elements = Vec::new();
        let mut fill = None;
        loop {
            elements.push(self.expression()?);
            if self.current == Some(Token::Ellipsis) {
                fill = Some(self.span.clone());
                self.advance()?;
                break;
            }
            match self.current {
                Some(Token::Comma) => {
                    self.advance()?;
                    if self.current == Some(Token::RightBracket) {
                        break;
                    }
                }
                Some(Token::RightBracket) => break,
                _ => return Err(self.error("expected `,` or `]` after array element")),
            }
        }
        let end = self
            .expect(Token::RightBracket, "expected `]` after array elements")?
            .end;
        self.nesting -= 1;
        Ok(self.syntax.expressions.alloc(Expression {
            depth: elements
                .iter()
                .map(|element| self.syntax.expressions[*element].depth)
                .max()
                .expect("an array literal has at least one element")
                + 1,
            span: span.start..end,
            kind: ExpressionKind::ArrayLiteral { elements, fill },
        }))
    }

    /// Parses `len(a)`, which uses call syntax without being a call.
    fn length_expression(&mut self, span: Range<usize>) -> Result<Idx<Expression>, Diagnostic> {
        self.advance()?;
        if self.current != Some(Token::LeftParen) {
            return Err(Diagnostic::new(
                span,
                "reserved word cannot be used as an identifier",
            ));
        }
        self.enter_nesting()?;
        let (operand, right_paren_span) = self.single_argument("len")?;
        self.nesting -= 1;
        Ok(self.syntax.expressions.alloc(Expression {
            depth: self.syntax.expressions[operand].depth + 1,
            span: span.start..right_paren_span.end,
            kind: ExpressionKind::Length { operand },
        }))
    }

    /// Parses a conversion, `T(x)` or `T.truncate(x)`, whose head is the name
    /// of an integer type.
    fn conversion_expression(&mut self, span: Range<usize>) -> Result<Idx<Expression>, Diagnostic> {
        self.enter_nesting()?;
        let (destination, destination_span) = self.named_type()?;
        let truncating = self.truncate_marker()?;
        if !truncating && self.current != Some(Token::LeftParen) {
            return Err(Diagnostic::new(
                destination_span,
                "reserved word cannot be used as an identifier",
            ));
        }
        self.expect(Token::LeftParen, "expected `(`")?;
        let operand = self.expression()?;
        let end = self.expect(Token::RightParen, "expected `)`")?.end;
        self.nesting -= 1;
        Ok(self.syntax.expressions.alloc(Expression {
            kind: ExpressionKind::Conversion {
                destination,
                truncating,
                operand,
            },
            span: span.start..end,
            depth: self.syntax.expressions[operand].depth + 1,
        }))
    }

    fn truncate_marker(&mut self) -> Result<bool, Diagnostic> {
        if self.current != Some(Token::Dot) {
            return Ok(false);
        }
        self.advance()?;
        if self.current != Some(Token::Name) || self.lexer.slice() != "truncate" {
            return Err(self.error("expected `truncate`"));
        }
        self.advance()?;
        Ok(true)
    }

    fn grouping_expression(&mut self, span: Range<usize>) -> Result<Idx<Expression>, Diagnostic> {
        self.enter_nesting()?;
        self.advance()?;
        let expression = self.expression()?;
        let end = self
            .expect(Token::RightParen, "expected `)` after grouped expression")?
            .end;
        self.nesting -= 1;
        Ok(self.syntax.expressions.alloc(Expression {
            kind: ExpressionKind::Grouping { expression },
            span: span.start..end,
            depth: self.syntax.expressions[expression].depth + 1,
        }))
    }

    /// Parses a name, which is a call when an argument list follows it and a
    /// reference otherwise.
    fn name_expression(&mut self, span: Range<usize>) -> Result<Idx<Expression>, Diagnostic> {
        let target = self.qualified_name("expected an expression")?;
        if self.current != Some(Token::LeftParen) {
            return Ok(self.syntax.expressions.alloc(Expression {
                span: target.span.clone(),
                depth: 0,
                kind: ExpressionKind::Reference(target),
            }));
        }
        let call = self.call_parts(target)?;
        let end = call.right_paren_span.end;
        Ok(self.syntax.expressions.alloc(Expression {
            depth: call
                .arguments
                .iter()
                .map(|argument| self.syntax.expressions[*argument].depth)
                .max()
                .unwrap_or(0)
                + 1,
            span: span.start..end,
            kind: ExpressionKind::Call(call),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spells a qualified name as it appeared in source.
    fn spell(syntax: &Syntax, name: &QualifiedName) -> String {
        match &name.qualifier {
            Some(qualifier) => format!(
                "{}::{}",
                syntax.names.resolve(&qualifier.name),
                syntax.names.resolve(&name.name)
            ),
            None => syntax.names.resolve(&name.name).to_owned(),
        }
    }

    /// Trails a qualified name's qualifier span, so plain names project
    /// unchanged.
    fn qualifier(name: &QualifiedName) -> String {
        match &name.qualifier {
            Some(qualifier) => format!(" qualifier={:?}", qualifier.name_span),
            None => String::new(),
        }
    }

    fn spell_path(syntax: &Syntax, path: &[PathComponent], separator: &str) -> String {
        path.iter()
            .map(|component| syntax.names.resolve(&component.name))
            .collect::<Vec<_>>()
            .join(separator)
    }

    fn path_spans(path: &[PathComponent]) -> Vec<Range<usize>> {
        path.iter()
            .map(|component| component.name_span.clone())
            .collect()
    }

    fn parse(text: &str) -> Result<Syntax, Diagnostic> {
        super::parse(&SourceMap::from_text(text))
    }

    fn projected(text: &str) -> String {
        let sources = SourceMap::from_text(text);
        project(&sources, &super::parse(&sources).unwrap())
    }

    /// Projects each file's imports and items. A path header separates the
    /// files of a multi-file module.
    fn project(sources: &SourceMap, syntax: &Syntax) -> String {
        use std::fmt::Write;
        let mut output = String::new();
        for (source, file) in sources.files().iter().zip(&syntax.files) {
            if syntax.files.len() > 1 {
                writeln!(output, "// {}", source.path.display()).unwrap();
            }
            project_file(syntax, file, &mut output);
        }
        output
    }

    fn project_file(syntax: &Syntax, file: &FileSyntax, output: &mut String) {
        use std::fmt::Write;
        for import in &file.imports {
            let path = spell_path(syntax, &import.path, "::");
            match &import.selection {
                Some(selection) => writeln!(
                    output,
                    "use {path}::{{{}}} span={:?} path={:?} selection={:?}",
                    spell_path(syntax, selection, ", "),
                    import.span,
                    path_spans(&import.path),
                    path_spans(selection),
                ),
                None => writeln!(
                    output,
                    "use {path} span={:?} path={:?}",
                    import.span,
                    path_spans(&import.path),
                ),
            }
            .unwrap();
        }
        for item in &file.items {
            match item {
                TopLevelItem::Function {
                    function: id,
                    public,
                } => {
                    let function = &syntax.functions[*id];
                    writeln!(
                        output,
                        "{}fn {} name={:?} span={:?}",
                        if *public { "pub " } else { "" },
                        syntax.names.resolve(&function.name),
                        function.name_span,
                        function.span
                    )
                    .unwrap();
                    for parameter in &function.parameters {
                        writeln!(
                            output,
                            "  parameter {} name={:?}",
                            syntax.names.resolve(&parameter.name),
                            parameter.name_span,
                        )
                        .unwrap();
                        project_annotation(syntax, parameter.annotation, 2, output);
                    }
                    match function.result {
                        FunctionResult::Void => writeln!(output, "  result void").unwrap(),
                        FunctionResult::Value(annotation) => {
                            writeln!(output, "  result").unwrap();
                            project_annotation(syntax, annotation, 2, output);
                        }
                    }
                    project_body(syntax, &function.body, 1, output);
                }
                TopLevelItem::Binding {
                    binding: id,
                    public,
                } => {
                    if *public {
                        output.push_str("pub ");
                    }
                    project_body(syntax, std::slice::from_ref(id), 0, output);
                }
            }
        }
    }

    fn project_body(syntax: &Syntax, body: &[Idx<Statement>], depth: usize, output: &mut String) {
        use std::fmt::Write;
        let indent = "  ".repeat(depth);
        for &id in body {
            let statement = &syntax.statements[id];
            match &statement.kind {
                StatementKind::Binding {
                    mutable,
                    name,
                    name_span,
                    annotation,
                    initializer,
                } => {
                    writeln!(
                        output,
                        "{indent}{} {} name={name_span:?} span={:?}",
                        if *mutable { "var" } else { "const" },
                        syntax.names.resolve(name),
                        statement.span
                    )
                    .unwrap();
                    if let Some(annotation) = annotation {
                        project_annotation(syntax, *annotation, depth + 1, output);
                    }
                    project_expression(syntax, *initializer, depth + 1, output);
                }
                StatementKind::Assignment { target, value } => {
                    writeln!(
                        output,
                        "{indent}assign {} name={:?} span={:?}{}",
                        spell(syntax, &target.name),
                        target.name.name_span,
                        statement.span,
                        qualifier(&target.name)
                    )
                    .unwrap();
                    project_target_indices(syntax, target, depth + 1, output);
                    project_expression(syntax, *value, depth + 1, output);
                }
                StatementKind::CompoundAssignment {
                    target,
                    operator,
                    operator_span,
                    value,
                } => {
                    writeln!(
                        output,
                        "{indent}compound assign {} {operator:?}= name={:?} operator={operator_span:?} span={:?}{}",
                        spell(syntax, &target.name),
                        target.name.name_span,
                        statement.span,
                        qualifier(&target.name)
                    )
                    .unwrap();
                    project_target_indices(syntax, target, depth + 1, output);
                    project_expression(syntax, *value, depth + 1, output);
                }
                StatementKind::Block { body } => {
                    writeln!(output, "{indent}block span={:?}", statement.span).unwrap();
                    project_body(syntax, body, depth + 1, output);
                }
                StatementKind::Exit { argument } => {
                    writeln!(output, "{indent}exit span={:?}", statement.span).unwrap();
                    project_expression(syntax, *argument, depth + 1, output);
                }
                StatementKind::Call { call } => {
                    writeln!(
                        output,
                        "{indent}call {} target={:?} left_paren={:?} right_paren={:?} span={:?}{}",
                        spell(syntax, &call.target),
                        call.target.name_span,
                        call.left_paren_span,
                        call.right_paren_span,
                        statement.span,
                        qualifier(&call.target)
                    )
                    .unwrap();
                    for &argument in &call.arguments {
                        project_expression(syntax, argument, depth + 1, output);
                    }
                }
                StatementKind::Return { value } => {
                    writeln!(output, "{indent}return span={:?}", statement.span).unwrap();
                    if let Some(value) = value {
                        project_expression(syntax, *value, depth + 1, output);
                    }
                }
                StatementKind::If {
                    condition,
                    then_body,
                    else_branch,
                } => {
                    writeln!(output, "{indent}if span={:?}", statement.span).unwrap();
                    project_expression(syntax, *condition, depth + 1, output);
                    writeln!(output, "{indent}then").unwrap();
                    project_body(syntax, then_body, depth + 1, output);
                    if let Some(branch) = else_branch {
                        writeln!(output, "{indent}else").unwrap();
                        project_body(syntax, std::slice::from_ref(branch), depth + 1, output);
                    }
                }
                StatementKind::For {
                    label,
                    header,
                    body,
                } => {
                    let label_name = label
                        .as_ref()
                        .map(|label| syntax.names.resolve(&label.name))
                        .unwrap_or("-");
                    writeln!(
                        output,
                        "{indent}for label={label_name} label_span={:?} span={:?}",
                        label.as_ref().map(|label| &label.name_span),
                        statement.span
                    )
                    .unwrap();
                    match header {
                        ForHeader::Infinite => writeln!(output, "{indent}  infinite").unwrap(),
                        ForHeader::Condition(condition) => {
                            writeln!(output, "{indent}  condition").unwrap();
                            project_expression(syntax, *condition, depth + 2, output);
                        }
                        ForHeader::ThreeClause {
                            initializer,
                            condition,
                            post,
                        } => {
                            writeln!(output, "{indent}  initializer").unwrap();
                            project_body(
                                syntax,
                                std::slice::from_ref(initializer),
                                depth + 2,
                                output,
                            );
                            writeln!(output, "{indent}  condition").unwrap();
                            project_expression(syntax, *condition, depth + 2, output);
                            writeln!(output, "{indent}  post").unwrap();
                            project_body(syntax, std::slice::from_ref(post), depth + 2, output);
                        }
                        ForHeader::Iteration {
                            value,
                            index,
                            operand,
                        } => {
                            writeln!(
                                output,
                                "{indent}  iteration value={} index={} index_span={:?}",
                                syntax.names.resolve(value),
                                index
                                    .as_ref()
                                    .map(|(name, _)| syntax.names.resolve(name))
                                    .unwrap_or("-"),
                                index.as_ref().map(|(_, span)| span),
                            )
                            .unwrap();
                            project_expression(syntax, *operand, depth + 2, output);
                        }
                    }
                    writeln!(output, "{indent}  body").unwrap();
                    project_body(syntax, body, depth + 2, output);
                }
                StatementKind::Break { label } | StatementKind::Continue { label } => {
                    let keyword = if matches!(&statement.kind, StatementKind::Break { .. }) {
                        "break"
                    } else {
                        "continue"
                    };
                    writeln!(
                        output,
                        "{indent}{keyword} label={} label_span={:?} span={:?}",
                        label
                            .as_ref()
                            .map(|label| syntax.names.resolve(&label.name))
                            .unwrap_or("-"),
                        label.as_ref().map(|label| &label.name_span),
                        statement.span
                    )
                    .unwrap();
                }
            }
        }
    }

    /// Projects an assignment target's indices, which precede the assigned
    /// value in evaluation order.
    fn project_target_indices(
        syntax: &Syntax,
        target: &AssignmentTarget,
        depth: usize,
        output: &mut String,
    ) {
        use std::fmt::Write;
        for &index in &target.indices {
            writeln!(output, "{}index", "  ".repeat(depth)).unwrap();
            project_expression(syntax, index, depth + 1, output);
        }
    }

    fn project_annotation(
        syntax: &Syntax,
        id: Idx<TypeAnnotation>,
        depth: usize,
        output: &mut String,
    ) {
        use std::fmt::Write;
        let indent = "  ".repeat(depth);
        let annotation = &syntax.annotations[id];
        match &annotation.kind {
            AnnotationKind::Named(ty) => {
                writeln!(
                    output,
                    "{indent}named {} span={:?}",
                    ty.name(),
                    annotation.span
                )
                .unwrap();
            }
            AnnotationKind::Array { length, element } => {
                writeln!(output, "{indent}array span={:?}", annotation.span).unwrap();
                match length {
                    Some(length) => {
                        writeln!(output, "{indent}  length").unwrap();
                        project_expression(syntax, *length, depth + 2, output);
                    }
                    None => writeln!(output, "{indent}  inferred length").unwrap(),
                }
                writeln!(output, "{indent}  element").unwrap();
                project_annotation(syntax, *element, depth + 2, output);
            }
        }
    }

    fn project_expression(syntax: &Syntax, id: Idx<Expression>, depth: usize, output: &mut String) {
        use std::fmt::Write;
        let indent = "  ".repeat(depth);
        let expression = &syntax.expressions[id];
        match &expression.kind {
            ExpressionKind::Integer(spelling) => writeln!(
                output,
                "{indent}integer {spelling} span={:?}",
                expression.span
            )
            .unwrap(),
            ExpressionKind::Boolean(value) => {
                writeln!(output, "{indent}boolean {value} span={:?}", expression.span).unwrap()
            }
            ExpressionKind::Reference(name) => writeln!(
                output,
                "{indent}reference {} span={:?}{}",
                spell(syntax, name),
                expression.span,
                qualifier(name)
            )
            .unwrap(),
            ExpressionKind::Grouping { expression: inner } => {
                writeln!(output, "{indent}group span={:?}", expression.span).unwrap();
                project_expression(syntax, *inner, depth + 1, output);
            }
            ExpressionKind::Unary {
                operator,
                operator_span,
                operand,
            } => {
                writeln!(
                    output,
                    "{indent}unary {operator:?} operator={operator_span:?} span={:?}",
                    expression.span
                )
                .unwrap();
                project_expression(syntax, *operand, depth + 1, output);
            }
            ExpressionKind::Binary {
                operator,
                operator_span,
                left,
                right,
            } => {
                writeln!(
                    output,
                    "{indent}binary {operator:?} operator={operator_span:?} span={:?}",
                    expression.span
                )
                .unwrap();
                project_expression(syntax, *left, depth + 1, output);
                project_expression(syntax, *right, depth + 1, output);
            }
            ExpressionKind::Comparison {
                operator,
                operator_span,
                left,
                right,
            } => {
                writeln!(
                    output,
                    "{indent}comparison {operator:?} operator={operator_span:?} span={:?}",
                    expression.span
                )
                .unwrap();
                project_expression(syntax, *left, depth + 1, output);
                project_expression(syntax, *right, depth + 1, output);
            }
            ExpressionKind::Logical {
                operator,
                operator_span,
                left,
                right,
            } => {
                writeln!(
                    output,
                    "{indent}logical {operator:?} operator={operator_span:?} span={:?}",
                    expression.span
                )
                .unwrap();
                project_expression(syntax, *left, depth + 1, output);
                project_expression(syntax, *right, depth + 1, output);
            }
            ExpressionKind::LogicalNot {
                operator_span,
                operand,
            } => {
                writeln!(
                    output,
                    "{indent}logical Not operator={operator_span:?} span={:?}",
                    expression.span
                )
                .unwrap();
                project_expression(syntax, *operand, depth + 1, output);
            }
            ExpressionKind::Conversion {
                destination,
                truncating,
                operand,
            } => {
                writeln!(
                    output,
                    "{indent}{} conversion {} span={:?}",
                    if *truncating { "truncating" } else { "checked" },
                    destination.name(),
                    expression.span
                )
                .unwrap();
                project_expression(syntax, *operand, depth + 1, output);
            }
            ExpressionKind::Call(call) => {
                writeln!(
                    output,
                    "{indent}call {} target={:?} left_paren={:?} right_paren={:?} span={:?}{}",
                    spell(syntax, &call.target),
                    call.target.name_span,
                    call.left_paren_span,
                    call.right_paren_span,
                    expression.span,
                    qualifier(&call.target)
                )
                .unwrap();
                for &argument in &call.arguments {
                    project_expression(syntax, argument, depth + 1, output);
                }
            }
            ExpressionKind::ArrayLiteral { elements, fill } => {
                writeln!(
                    output,
                    "{indent}array literal fill={fill:?} span={:?}",
                    expression.span
                )
                .unwrap();
                for &element in elements {
                    project_expression(syntax, element, depth + 1, output);
                }
            }
            ExpressionKind::Index { operand, index } => {
                writeln!(output, "{indent}index span={:?}", expression.span).unwrap();
                project_expression(syntax, *operand, depth + 1, output);
                project_expression(syntax, *index, depth + 1, output);
            }
            ExpressionKind::Length { operand } => {
                writeln!(output, "{indent}len span={:?}", expression.span).unwrap();
                project_expression(syntax, *operand, depth + 1, output);
            }
        }
    }

    #[test]
    fn mixed_body_snapshot() {
        let source = "fn main() -> void { const exit_code = 0x2A; var copy: int = exit_code; exit(copy,); const after = missing; } fn helper() -> void { exit(000,); }";
        insta::assert_snapshot!(projected(source));
    }

    #[test]
    fn interleaved_top_level_bindings_snapshot() {
        let source = "const start: int = 40; fn main() -> void { var local = start; exit(local); } var counter = start + 2; fn helper() -> void {} const last = counter;";
        insta::assert_snapshot!(projected(source));
    }

    #[test]
    fn function_signatures_calls_and_returns_snapshot() {
        let source = "fn mark(digit: int, flag: bool,) -> int { return digit; } fn nothing() -> void { return; } fn main() -> void { mark(1, true,); nothing(); const value = mark(mark(2, false), true) + mark(3, true); if value > 0 && true { return; } }";
        insta::assert_snapshot!(projected(source));
    }

    #[test]
    fn malformed_function_signatures_calls_and_returns_report_the_offending_token() {
        for (marked, message) in [
            (
                "fn f(«,» value: int) -> void {}",
                "expected a parameter name",
            ),
            (
                "fn f(value «int») -> void {}",
                "expected `:` after parameter name",
            ),
            ("fn f(value: «)» -> void {}", "expected a type"),
            ("fn f(value: «void» ) -> void {}", "expected a type"),
            (
                "fn f(value: int «other»: bool) -> void {}",
                "expected `)` after parameter list",
            ),
            (
                "fn f(value: int «->» void {})",
                "expected `)` after parameter list",
            ),
            ("fn f(value: int) -> «{»", "expected a type"),
            (
                "fn f(value: int) -> int { return «+»; }",
                "expected an expression",
            ),
            ("fn f() -> void { target(«,»); }", "expected an expression"),
            (
                "fn f() -> void { target(1 «2»); }",
                "expected `,` or `)` after call argument",
            ),
            (
                "fn f() -> void { target(1«;» }",
                "expected `,` or `)` after call argument",
            ),
            ("fn f() -> void { target(1) «}»", "expected `;`"),
            ("fn f() -> void { return 1 «}»", "expected `;`"),
        ] {
            let prefix = "/* 🌿 */ ";
            let start = marked.find('«').unwrap();
            let end = marked.find('»').unwrap() - '«'.len_utf8();
            let source = format!("{prefix}{}", marked.replace(['«', '»'], ""));
            let error = parse(&source).unwrap_err();
            assert_eq!(error.message, message, "{marked}");
            assert_eq!(
                error.span,
                prefix.len() + start..prefix.len() + end,
                "{marked}"
            );
        }
    }

    #[test]
    fn modules_imports_and_qualified_names_snapshot() {
        let source = "\
use fmt;
use network::http;
use fs::{flag, mode,};
pub const limit: int = 4;
pub var total = 0;
const private = 1;
fn helper() -> int { return private; }
pub fn main() -> void {
    http::serve(fmt::width(limit) + helper());
    http::total = limit;
    http::total += 2;
    const value = http::total;
    exit(value);
}
";
        insta::assert_snapshot!(projected(source));
    }

    #[test]
    fn module_files_parse_into_one_syntax_with_file_local_spans_snapshot() {
        let sources = SourceMap::from_named_texts(&[
            ("first.fern", "use fmt;\nconst base = 1;\n"),
            ("second.fern", "pub fn helper() -> int { return base; }\n"),
        ]);
        let syntax = super::parse(&sources).unwrap();
        insta::assert_snapshot!(project(&sources, &syntax));

        // The second file's spans start past the first file's text and its
        // one-byte gap, so a span identifies the file it points into.
        let base = sources.files()[1].base;
        assert_eq!(base, sources.files()[0].text.len() + 1);
        let helper = syntax.functions.iter().next().unwrap().1;
        assert!(helper.name_span.start >= base);
        assert_eq!(sources.index_at(helper.name_span.start), 1);
        assert_eq!(sources.index_at(base - 1), 0);
    }

    #[test]
    fn malformed_modules_imports_and_qualified_names_report_the_offending_token() {
        for (marked, message) in [
            (
                "«pub» use fmt;",
                "`pub` is not permitted on a `use` declaration",
            ),
            (
                "fn main() -> void {} «use» fmt;",
                "`use` declarations must precede the first declaration",
            ),
            (
                "fn main() -> void { «pub» var x = 1; }",
                "`pub` is not permitted on a local declaration",
            ),
            ("use «::»fmt;", "expected a module name after `use`"),
            ("use fmt «as» f;", "expected `;` after import path"),
            ("use fmt«»", "expected `;` after import path"),
            ("use fmt::«;»", "expected a name after `::`"),
            (
                "use fmt::«const»;",
                "reserved word cannot be used as an identifier",
            ),
            ("use fs::{«}»;", "expected an imported name"),
            ("use fs::{flag,«,»};", "expected an imported name"),
            (
                "use fs::{flag«::»mode};",
                "expected `}` after imported names",
            ),
            ("fn a«::»b() -> void {}", "expected `(`"),
            (
                "fn f(a«::»b: int) -> void {}",
                "expected `:` after parameter name",
            ),
            ("fn main() -> void { var a«::»b = 1; }", "expected `=`"),
            (
                "fn main() -> void { for :a«::»b {} }",
                "expected an expression",
            ),
            ("fn main() -> void { const x = a::b«::»c; }", "expected `;`"),
            (
                "fn main() -> void { a::«=» 1; }",
                "expected a name after `::`",
            ),
            (
                "fn main() -> void { a::«(»1); }",
                "expected a name after `::`",
            ),
        ] {
            let prefix = "/* 🌿 */ ";
            let start = marked.find('«').unwrap();
            let end = marked.find('»').unwrap() - '«'.len_utf8();
            let source = format!("{prefix}{}", marked.replace(['«', '»'], ""));
            let error = parse(&source).unwrap_err();
            assert_eq!(error.message, message, "{marked}");
            assert_eq!(
                error.span,
                prefix.len() + start..prefix.len() + end,
                "{marked}"
            );
        }
    }

    #[test]
    fn public_declarations_and_qualified_names_are_recorded_on_their_nodes() {
        let syntax = parse("pub fn f() -> void {} var private = 0; pub const shared = 1;").unwrap();
        let public: Vec<_> = syntax
            .files
            .iter()
            .flat_map(|file| &file.items)
            .map(|item| match item {
                TopLevelItem::Function { public, .. } | TopLevelItem::Binding { public, .. } => {
                    *public
                }
            })
            .collect();
        assert_eq!(public, [true, false, true]);

        let syntax = parse("fn main() -> void { const x = plain; const y = a::b; }").unwrap();
        let qualifiers: Vec<_> = syntax
            .expressions
            .iter()
            .filter_map(|(_, expression)| match &expression.kind {
                ExpressionKind::Reference(name) => Some(name.qualifier.is_some()),
                _ => None,
            })
            .collect();
        assert_eq!(qualifiers, [false, true]);
    }

    #[test]
    fn statements_are_rejected_at_top_level() {
        for marked in [
            "fn main() -> void {} «exit»(0);",
            "fn main() -> void {} «value» = 1;",
            "fn main() -> void {} «{» exit(0); }",
            "fn main() -> void {} «42»;",
            "fn main() -> void {} «;»",
        ] {
            let start = marked.find('«').unwrap();
            let end = marked.find('»').unwrap() - '«'.len_utf8();
            let source = marked.replace(['«', '»'], "");
            let error = parse(&source).unwrap_err();
            assert_eq!(error.message, "expected a top-level declaration");
            assert_eq!(error.span, start..end, "{marked}");
        }
    }

    #[test]
    fn malformed_top_level_bindings_report_the_offending_token() {
        for (source, message, span) in [
            ("var = 1;", "expected a binding name after `var`", 4..5),
            ("const value int = 1;", "expected `=`", 12..15),
            ("var value: size = 1;", "expected a type", 11..15),
            ("const value = ;", "expected an expression", 14..15),
            ("var value = 1", "expected `;`", 13..13),
        ] {
            let error = parse(source).unwrap_err();
            assert_eq!(error.message, message, "{source}");
            assert_eq!(error.span, span, "{source}");
        }
    }

    #[test]
    fn assignment_and_blocks_snapshot() {
        let source = "fn main() -> void { var x = 1; x = x; {} { const copy = x; { x = 42; } exit(copy); } exit(x); }";
        insta::assert_snapshot!(projected(source));
    }

    #[test]
    fn integer_annotations_snapshot() {
        let mut source = String::from("/* 🌿 */ fn main() -> void {\n");
        for name in Scalar::ALL_INTEGERS.map(Scalar::name) {
            source.push_str(&format!("var value: {name} = 0x2A;\n"));
            source.push_str(&format!("{{ const copy: /* type */ {name} = value; }}\n"));
        }
        source.push('}');
        insta::assert_snapshot!(projected(&source));
    }

    #[test]
    fn integer_operator_precedence_and_grouping_snapshot() {
        let source = "fn main() -> void { const high = 1 * 2 / 3 % 4 *% 5 << 6 >> 7 & 8; const low = 9 + 10 - 11 +% 12 -% 13 | 14 ^ 15; const shift = 1 + 2 << 3; const add_or = 1 | 2 + 3; const and_not = 1 & ^2; const unary = -^-%u8(1); const grouping = (1 + 2) * (3 - 4); const and_negative = 7 & -2; }";
        insta::assert_snapshot!(projected(source));
    }

    #[test]
    fn boolean_expressions_and_control_flow_snapshot() {
        let source = "fn main() -> void { var ready: bool = true; const stopped = false; const result = 1 + 2 < 4 && !stopped || ready == false; if ready { exit(1); } else if stopped { exit(2); } else {} for { break; } for ready { continue; } for :rows var i: int = 0; i < 4; i += 1 { if i >= 2 { break :rows; } } for cursor = 0; cursor != 2; cursor = cursor + 1 { continue; } }";
        insta::assert_snapshot!(projected(source));
    }

    #[test]
    fn every_compound_assignment_operator_parses() {
        let operators = [
            "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "+%=", "-%=", "*%=",
        ];
        let body = operators
            .iter()
            .map(|operator| format!("value {operator} 1;"))
            .collect::<Vec<_>>()
            .join(" ");
        let syntax = parse(&format!("fn main() -> void {{ var value = 0; {body} }}")).unwrap();
        assert_eq!(syntax.statements.len(), operators.len() + 1);
        for &statement in &syntax.functions.iter().next().unwrap().1.body[1..] {
            assert!(matches!(
                syntax.statements[statement].kind,
                StatementKind::CompoundAssignment { .. }
            ));
        }
    }

    #[test]
    fn every_comparison_operator_parses() {
        for operator in ["==", "!=", "<", "<=", ">", ">="] {
            let source = format!("fn main() -> void {{ const result = left {operator} right; }}");
            let syntax = parse(&source).unwrap();
            let statement = syntax.functions.iter().next().unwrap().1.body[0];
            let StatementKind::Binding { initializer, .. } = syntax.statements[statement].kind
            else {
                panic!("expected binding")
            };
            assert!(matches!(
                syntax.expressions[initializer].kind,
                ExpressionKind::Comparison { .. }
            ));
        }
    }

    #[test]
    fn malformed_control_flow_reports_the_offending_token() {
        for (marked, message) in [
            ("if «{»}", "expected an expression"),
            ("if true «;»", "expected `{`"),
            ("if true {} else «exit»(0);", "expected `{`"),
            ("for «;»", "expected an expression"),
            (
                "for var i = 0 «i» < 2; i = i + 1 {}",
                "expected `;` after for initializer",
            ),
            ("for var i = 0; «;» i = i + 1 {}", "expected an expression"),
            (
                "for var i = 0; i < 2; «{»}",
                "expected an assignment after second `;`",
            ),
            (
                "for var i = 0; i < 2; «i» + 1 {}",
                "expected an assignment after second `;`",
            ),
            ("for :«{»}", "expected a loop label after `:`"),
            ("break :«;»", "expected a label after `:`"),
            ("continue«}»", "expected `;`"),
            (
                "for i «+=» 1; true; i = i + 1 {}",
                "for initializer does not permit compound assignment",
            ),
        ] {
            let prefix = "fn main() -> void { ";
            let start = marked.find('«').unwrap();
            let end = marked.find('»').unwrap() - '«'.len_utf8();
            let source = format!("{prefix}{}}}", marked.replace(['«', '»'], ""));
            let error = parse(&source).unwrap_err();
            assert_eq!(error.message, message, "{marked}");
            assert_eq!(
                error.span,
                prefix.len() + start..prefix.len() + end,
                "{marked}"
            );
        }
    }

    #[test]
    fn boolean_and_control_flow_nesting_obeys_the_source_limit() {
        let accepted_logical = format!(
            "fn main() -> void {{ const value = {}true; }}",
            "true && ".repeat(127)
        );
        parse(&accepted_logical).unwrap();
        let rejected_logical = format!(
            "fn main() -> void {{ const value = {}true; }}",
            "true && ".repeat(128)
        );
        assert_eq!(
            parse(&rejected_logical).unwrap_err().message,
            "source nesting exceeds compiler limit of 128"
        );

        let chain = |count: usize| {
            format!(
                "fn main() -> void {{ {}if true {{}} }}",
                "if true {} else ".repeat(count - 1)
            )
        };
        parse(&chain(127)).unwrap();
        assert_eq!(
            parse(&chain(128)).unwrap_err().message,
            "source nesting exceeds compiler limit of 128"
        );
    }

    #[test]
    fn comments_may_touch_integer_operators() {
        let source = "fn main() -> void { const x = ^/*a*/1/*b*/+%/*c*/2<<// d\n3; }";
        let syntax = parse(source).unwrap();
        assert_eq!(syntax.expressions.len(), 6);
    }

    #[test]
    fn malformed_annotations_report_the_offending_token() {
        for spelling in [
            "size",
            "uintptr",
            "void",
            "f32",
            "i128",
            "u7",
            "int_value",
            "i",
            "u",
            "42",
            "=",
            ";",
            ":",
            "",
        ] {
            let prefix = "/* 🌿 */ fn main() -> void { const x: ";
            let source = format!("{prefix}{spelling}");
            let error = parse(&source).unwrap_err();
            assert_eq!(error.message, "expected a type", "{spelling}");
            assert_eq!(error.span, prefix.len()..source.len(), "{spelling}");
        }
    }

    #[test]
    fn unsuffixed_integer_spellings_survive_parsing() {
        for digits in [
            "0",
            "00042",
            "42",
            "0xabcdefABCDEF",
            "0b001010",
            "0o00752",
            "9999999999999999999999999999999999999999999999999999999999",
        ] {
            let source = format!("fn main() -> void {{ exit({digits}); }}");
            let syntax = parse(&source).unwrap();
            let (_, expression) = syntax.expressions.iter().next().unwrap();
            let ExpressionKind::Integer(actual) = &expression.kind else {
                panic!("expected integer")
            };
            assert_eq!(actual, digits);
            assert_eq!(&source[expression.span.clone()], digits);
        }
    }

    #[test]
    fn integer_literal_digit_limit_is_checked_before_semantic_parsing() {
        let accepted = "9".repeat(MAX_INTEGER_LITERAL_DIGITS);
        parse(&format!(
            "fn main() -> void {{ const value = {accepted}; }}"
        ))
        .unwrap();

        let rejected = "9".repeat(MAX_INTEGER_LITERAL_DIGITS + 1);
        let text = format!("fn main() -> void {{ const value = {rejected}; }}");
        let error = parse(&text).unwrap_err();
        assert_eq!(error.span.len(), rejected.len());
        assert_eq!(
            error.message,
            "integer literal exceeds compiler limit of 4096 digits"
        );
    }

    #[test]
    fn nested_checked_and_truncating_conversions_preserve_their_forms() {
        let source = "fn main() -> void { exit(u8.truncate(i16(u64(42)))); }";
        let syntax = parse(source).unwrap();
        let statement = syntax.functions.iter().next().unwrap().1.body[0];
        let StatementKind::Exit { argument } = syntax.statements[statement].kind else {
            panic!("expected exit")
        };
        let ExpressionKind::Conversion {
            destination,
            truncating,
            operand,
            ..
        } = &syntax.expressions[argument].kind
        else {
            panic!("expected truncating conversion")
        };
        assert_eq!(*destination, Scalar::U8);
        assert!(*truncating);
        let ExpressionKind::Conversion {
            destination,
            truncating,
            operand,
            ..
        } = &syntax.expressions[*operand].kind
        else {
            panic!("expected checked conversion")
        };
        assert_eq!(*destination, Scalar::I16);
        assert!(!*truncating);
        let ExpressionKind::Conversion {
            destination,
            truncating,
            ..
        } = &syntax.expressions[*operand].kind
        else {
            panic!("expected nested checked conversion")
        };
        assert_eq!(*destination, Scalar::U64);
        assert!(!*truncating);

        for (body, message) in [
            ("var value = u8(1;", "expected `)`"),
            ("var value = u8.truncate(1;", "expected `)`"),
            ("var value = u8.(1);", "expected `truncate`"),
            ("var value = u8.truncate 1;", "expected `(`"),
        ] {
            let text = format!("fn main() -> void {{ {body} }}");
            assert_eq!(parse(&text).unwrap_err().message, message, "{body}");
        }
    }

    #[test]
    fn malformed_integers_cover_the_whole_candidate() {
        for spelling in [
            "0x",
            "0b",
            "0o",
            "0xi",
            "0bu8",
            "0oz",
            "0b2",
            "0b102",
            "0o8",
            "0o178",
            "0xG",
            "0x12G",
            "1_000",
            "0x_ff",
            "42foo",
            "42i",
            "42u",
            "42z",
            "42i8",
            "42i16",
            "42i32",
            "42i64",
            "42u8",
            "42u16",
            "42u32",
            "42u64",
            "42int",
            "42uint",
            "42size",
            "42uintptr",
            "42i128",
            "42u7",
            "0XFF",
            "0B10",
            "0O12",
            "0x1int",
        ] {
            let prefix = "/* é */ fn main() -> void { const x = ";
            let source = format!("{prefix}{spelling}; }}");
            let error = parse(&source).unwrap_err();
            assert_eq!(
                error.span,
                prefix.len()..prefix.len() + spelling.len(),
                "{spelling}"
            );
            assert_eq!(error.message, "malformed integer literal", "{spelling}");
        }
    }

    #[test]
    fn malformed_statements_report_the_offending_token() {
        // The marker surrounds the expected diagnostic span, including EOF.
        for marked in [
            "var «=» 1;",
            "const «;»",
            "var x = «;»",
            "var x: «=» 1;",
            "var x: «size» = 1;",
            "var x «1»;",
            "var x: int «;»",
            "const x = 1 «}»",
            "exit «0»;",
            "exit(«)»;",
            "exit(«,»);",
            "exit(0, «1»);",
            "exit(0 «1»);",
            "exit(0,«,»);",
            "exit(0«;»",
            "exit(0) «}»",
            "x «1»;",
            "x = «;»",
            "x = 1 «}»",
            "x = 1«»",
            "{ x = 1; «»",
            "{ {} } «»",
            "{}«;»",
            "«1» = 2;",
            "«=» 2;",
            "«)»",
            "x «.»field = 1;",
            "const x = 1 + «;»",
            "exit(-«)»);",
            "var x = (1 + 2«;»",
            "var x = 1«»",
            "exit(0,«»",
            "var x:«»",
        ] {
            let prefix = "/* 🌿 */ fn main() -> void { ";
            let start = marked.find('«').unwrap();
            let end = marked.find('»').unwrap() - '«'.len_utf8();
            let source = format!("{prefix}{}", marked.replace(['«', '»'], ""));
            assert_eq!(
                parse(&source).unwrap_err().span,
                prefix.len() + start..prefix.len() + end,
                "{marked}"
            );
        }
    }

    #[test]
    fn reserved_names_agree_in_all_name_positions() {
        for name in [
            "const", "var", "true", "false", "fn", "void", "exit", "i8", "i16", "i32", "i64", "u8",
            "u16", "u32", "u64", "int", "uint", "bool", "if", "else", "for", "break", "continue",
            "return", "in", "len", "pub", "use",
        ] {
            for (prefix, suffix) in [
                ("fn ", "() -> void {}"),
                ("fn main() -> void { var ", " = 0; }"),
                ("fn main() -> void { const ", " = 0; }"),
            ] {
                let error = parse(&format!("{prefix}{name}{suffix}")).unwrap_err();
                assert_eq!(error.span, prefix.len()..prefix.len() + name.len());
                assert!(error.message.contains("reserved word"));
            }
        }

        for name in [
            "const", "var", "fn", "void", "exit", "i8", "i16", "i32", "i64", "u8", "u16", "u32",
            "u64", "int", "uint", "bool", "if", "else", "for", "break", "continue", "return", "in",
            "len", "pub", "use",
        ] {
            for (prefix, suffix) in [
                ("fn main() -> void { const x = ", "; }"),
                ("fn main() -> void { exit(", "); }"),
            ] {
                let error = parse(&format!("{prefix}{name}{suffix}")).unwrap_err();
                assert_eq!(error.span, prefix.len()..prefix.len() + name.len());
                assert!(error.message.contains("reserved word"));
            }
        }

        for name in [
            "alloc", "free", "f32", "f64", "rune", "str", "uintptr", "size",
        ] {
            parse(&format!("fn {name}() -> void {{}}"))
                .unwrap_or_else(|error| panic!("{name} should not be reserved: {error:?}"));
        }
    }

    #[test]
    fn bodies_ignore_whitespace_and_comments() {
        let source = "fn main() -> void { const x: int = 1; var y = x; exit(y,); }";
        for separator in [
            " ",
            "\t",
            "\n",
            "\r",
            "\u{b}",
            "\u{c}",
            "/* 🌿 /* nested */ é */",
            "// é\n",
            "// é\r",
        ] {
            let syntax = parse(&source.replace(' ', separator)).unwrap();
            assert_eq!(syntax.statements.len(), 3);
            assert_eq!(syntax.expressions.len(), 3);
        }
        for comment in ["/*", "/* é", "/* outer /* inner */"] {
            let prefix = "fn main() -> void { exit(0); ";
            let source = format!("{prefix}{comment}");
            let error = parse(&source).unwrap_err();
            assert_eq!(error.span, prefix.len()..source.len());
            assert_eq!(error.message, "unterminated block comment");
        }
    }

    #[test]
    fn parsing_does_not_check_names_or_reachability() {
        let syntax = parse("fn exit_code() -> void { var const_value: int = unknown; exit(const_value); var after = missing; } fn exit_code() -> void {}").unwrap();
        assert_eq!(syntax.functions.len(), 2);
        assert_eq!(syntax.statements.len(), 3);
    }

    #[test]
    fn malformed_expressions_and_calls_have_specific_diagnostics() {
        for (body, message) in [
            ("var = 1;", "expected a binding name after `var`"),
            ("const = 1;", "expected a binding name after `const`"),
            ("var x = ;", "expected an expression"),
            ("exit();", "exit requires one argument"),
            ("exit(1.5);", "expected `)` after exit argument"),
            ("exit(1, 2);", "exit takes one argument"),
            ("var x = int();", "expected an expression"),
            ("var x = 1 + ;", "expected an expression"),
            ("var x = 1 + * 2;", "expected an expression"),
            ("var x = ();", "expected an expression"),
            ("var x = (1 + 2;", "expected `)` after grouped expression"),
        ] {
            let error = parse(&format!("fn main() -> void {{ {body} }}")).unwrap_err();
            assert_eq!(error.message, message, "{body}");
        }
    }

    #[test]
    fn nesting_limit_protects_parsing_checking_and_lowering() {
        for (blocks, conversions) in [(127, 0), (0, 127), (63, 64)] {
            let text = format!(
                "fn main() -> void {{ var x = 42; {}exit({}x{});{} }}",
                "{".repeat(blocks),
                "int(".repeat(conversions),
                ")".repeat(conversions),
                "}".repeat(blocks),
            );
            let syntax = parse(&text).unwrap();
            let checked = crate::semantic::check_root(&syntax).unwrap();
            crate::ir::lower(checked).verify().unwrap();
        }
        for (blocks, conversions) in [(128, 0), (0, 128), (64, 64), (100_000, 0), (0, 100_000)] {
            let text = format!(
                "fn main() -> void {{ {}exit({}42{});{} }}",
                "{".repeat(blocks),
                "int(".repeat(conversions),
                ")".repeat(conversions),
                "}".repeat(blocks),
            );
            let error = parse(&text).unwrap_err();
            assert_eq!(
                error.message,
                "source nesting exceeds compiler limit of 128"
            );
            assert!(["{", "int"].contains(&&text[error.span]));
        }

        let grouped = format!(
            "fn main() -> void {{ exit({}42{}); }}",
            "(".repeat(127),
            ")".repeat(127),
        );
        let syntax = parse(&grouped).unwrap();
        crate::ir::lower(crate::semantic::check_root(&syntax).unwrap())
            .verify()
            .unwrap();
        let binary = format!("fn main() -> void {{ exit({}1); }}", "1 + ".repeat(127));
        let syntax = parse(&binary).unwrap();
        crate::ir::lower(crate::semantic::check_root(&syntax).unwrap())
            .verify()
            .unwrap();
        for source in [
            format!(
                "fn main() -> void {{ exit({}42{}); }}",
                "(".repeat(128),
                ")".repeat(128),
            ),
            format!("fn main() -> void {{ exit({}42); }}", "-".repeat(128)),
            format!("fn main() -> void {{ exit({}1); }}", "1 + ".repeat(128)),
        ] {
            assert_eq!(
                parse(&source).unwrap_err().message,
                "source nesting exceeds compiler limit of 128"
            );
        }
    }

    #[test]
    fn array_types_literals_indexing_and_iteration_snapshot() {
        let source = "\
fn rows(grid: [2][3]int) -> [2]int {
    var out: [2]int = [0...];
    for v, i in grid {
        out[i] = v[0];
    }
    return out;
}
fn main() -> void {
    var a: [3]int = [1, 2, 3,];
    const inferred: [_]int = [0...];
    const seeded: [4]int = [1, 2, 3, 0...];
    const board: [2][3]int = [[1...]...];
    var grid: [2][3]int = [[1, 2, 3], [4, 5, 6]];
    const r = 0;
    const c = 1;
    const cell = grid[r][c];
    const first = rows(grid)[0];
    const negated = -a[0];
    const length = len(a);
    const trailing = len(a,);
    grid[r][c] = 7;
    a[r] += 1;
    for :outer t in a {
        continue :outer;
    }
    for t, i in a {
        break;
    }
}
";
        insta::assert_snapshot!(projected(source));
    }

    #[test]
    fn malformed_array_syntax_reports_the_offending_token() {
        for (marked, message) in [
            ("const x = [«]»;", "expected an array element"),
            ("const x = [1,«,»2];", "expected an expression"),
            (
                "const x = [1 «2»];",
                "expected `,` or `]` after array element",
            ),
            ("const x = [«...»];", "expected an expression"),
            ("var x: [3] «=» 1;", "expected a type"),
            ("var x: [3]«void» = 1;", "expected a type"),
            (
                "var x: [_ «+» 1]int = 1;",
                "expected `]` after array length",
            ),
            ("const x = a[«]»;", "expected an expression"),
            ("a[0 «=» 1;", "expected `]` after index"),
            (
                "const x = «len»;",
                "reserved word cannot be used as an identifier",
            ),
            ("const x = len(«)»;", "len requires one argument"),
            ("const x = len(a, «b»);", "len takes one argument"),
            (
                "for v, «in» a {}",
                "reserved word cannot be used as an identifier",
            ),
            ("for v «i» in a {}", "expected `{`"),
            ("for v, i «a» in b {}", "expected `in`"),
            (
                "for «in» a {}",
                "reserved word cannot be used as an identifier",
            ),
            ("for v in «{»}", "expected an expression"),
        ] {
            let prefix = "/* 🌿 */ fn main() -> void { ";
            let start = marked.find('«').unwrap();
            let end = marked.find('»').unwrap() - '«'.len_utf8();
            let source = format!("{prefix}{}", marked.replace(['«', '»'], ""));
            let error = parse(&source).unwrap_err();
            assert_eq!(error.message, message, "{marked}");
            assert_eq!(
                error.span,
                prefix.len() + start..prefix.len() + end,
                "{marked}"
            );
        }
    }

    #[test]
    fn array_nesting_obeys_the_source_limit() {
        // A function body already holds one level, so 127 more are accepted.
        let indexes = |count: usize| {
            format!(
                "fn main() -> void {{ const x = a{}; }}",
                "[0]".repeat(count)
            )
        };
        let literals = |count: usize| {
            format!(
                "fn main() -> void {{ const x = {}1{}; }}",
                "[".repeat(count),
                "]".repeat(count),
            )
        };
        let annotations = |count: usize| {
            format!(
                "fn main() -> void {{ var x: {}int = 1; }}",
                "[1]".repeat(count)
            )
        };
        for build in [indexes, literals, annotations] {
            parse(&build(127)).unwrap();
            assert_eq!(
                parse(&build(128)).unwrap_err().message,
                "source nesting exceeds compiler limit of 128"
            );
        }
    }

    #[test]
    fn syntax_snapshot() {
        let syntax = parse("fn main() -> void {} ").unwrap();
        let functions: Vec<_> = syntax
            .functions
            .iter()
            .map(|(_, f)| {
                (
                    syntax.names.resolve(&f.name),
                    f.name_span.clone(),
                    f.span.clone(),
                )
            })
            .collect();
        insta::assert_debug_snapshot!(functions, @r#"
        [
            (
                "main",
                3..7,
                0..20,
            ),
        ]
        "#);
    }

    #[test]
    fn errors_use_byte_spans() {
        for (text, span) in [
            ("/* é */ @", 9..10),
            ("fn main() -> void {", 19..19),
            ("fn main() -> void {} fn", 23..23),
            ("fn main() -> void {} extra", 21..26),
            ("/* é */ fn main() -> void {} 💥", 30..34),
            ("/* outer /* inner */", 0..20),
        ] {
            assert_eq!(parse(text).unwrap_err().span, span, "{text}");
        }
    }
}
