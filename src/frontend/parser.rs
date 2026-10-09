//! Precedence parsing, nesting enforcement, and syntax construction.

#[cfg(test)]
use crate::source::SourceMap;
use crate::{
    diagnostic::Diagnostic,
    source::Source,
    types::{BinaryOperator, ComparisonOperator, LogicalOperator, Scalar, UnaryOperator},
};
use la_arena::Idx;
use lasso::Spur;
use logos::Logos;
use std::ops::Range;

use super::{
    lexer::{Token, is_floating, literal_bytes, valid_number},
    syntax::*,
};

fn reserved(name: &str) -> bool {
    matches!(
        name,
        "const"
            | "var"
            | "true"
            | "false"
            | "fn"
            | "void"
            | "extern"
            | "sizeof"
            | "bool"
            | "if"
            | "else"
            | "for"
            | "break"
            | "continue"
            | "return"
            | "len"
            | "pub"
            | "use"
            | "type"
            | "struct"
            | "null"
    ) || Scalar::named(name).is_some()
}

/// The next token a lookahead lexer yields, or `None` at a lexical error or
/// the end of input.
fn lookahead(lexer: &mut logos::Lexer<'_, Token>) -> Option<Token> {
    match lexer.next() {
        Some(Ok(token)) => Some(token),
        Some(Err(())) | None => None,
    }
}

/// Skips a balanced group and returns the token following its closing delimiter.
fn token_after_group(lexer: &mut logos::Lexer<'_, Token>, opening: Token) -> Option<Token> {
    let closing = |token| match token {
        Token::LeftParen => Some(Token::RightParen),
        Token::LeftBracket => Some(Token::RightBracket),
        Token::LeftBrace => Some(Token::RightBrace),
        _ => None,
    };
    let mut endings = vec![closing(opening)?];
    while let Some(&end) = endings.last() {
        let token = lookahead(lexer)?;
        if token == end {
            endings.pop();
        } else if let Some(end) = closing(token) {
            endings.push(end);
        } else if matches!(
            token,
            Token::RightParen | Token::RightBracket | Token::RightBrace
        ) {
            return None;
        }
    }
    lookahead(lexer)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructLiterals {
    Permitted,
    Restricted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixOperator {
    Value(UnaryOperator),
    LogicalNot,
    AddressOf,
    Dereference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InfixOperator {
    Integer(BinaryOperator),
    Comparison(ComparisonOperator),
    Logical(LogicalOperator),
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
                Some(Token::Fn | Token::Extern) => TopLevelItem::Function {
                    function: self.function()?,
                    public,
                },
                Some(Token::Const) | Some(Token::Var) => TopLevelItem::Binding {
                    binding: self.top_level_binding()?,
                    public,
                },
                Some(Token::Type) => TopLevelItem::Struct {
                    declaration: self.struct_declaration()?,
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
            Some(Ok(Token::Number)) => {
                if let Err(message) = valid_number(self.lexer.slice()) {
                    return Err(self.error(message));
                }
                Some(Token::Number)
            }
            Some(Ok(Token::String)) => {
                if let Err(message) = literal_bytes(self.literal_body()) {
                    return Err(self.error(message));
                }
                Some(Token::String)
            }
            Some(Ok(Token::Character)) => {
                if let Err(message) = self.character_byte() {
                    return Err(self.error(message));
                }
                Some(Token::Character)
            }
            Some(Ok(token)) => Some(token),
            Some(Err(())) => {
                let slice = self.lexer.slice();
                return Err(self.error(if slice.starts_with("/*") {
                    "unterminated block comment"
                } else if slice.starts_with('"') {
                    "unterminated string literal"
                } else if slice.starts_with('\'') {
                    "unterminated character literal"
                } else {
                    "invalid token"
                }));
            }
            None => None,
        };
        Ok(())
    }

    /// The text between the current string or character token's delimiters.
    fn literal_body(&self) -> &str {
        let slice = self.lexer.slice();
        &slice[1..slice.len() - 1]
    }

    /// The byte the current character token denotes.
    fn character_byte(&self) -> Result<u8, String> {
        match literal_bytes(self.literal_body())?.as_slice() {
            [byte] => Ok(*byte),
            _ => Err("character literal must be exactly one byte".to_owned()),
        }
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
        let external = self.current == Some(Token::Extern);
        if external {
            self.advance()?;
        }
        let start = self.expect(Token::Fn, "expected `fn`")?.start;
        let (name, name_span) = self.name("expected a function name")?;
        let (parameters, variadic) = self.parameters(external)?;
        self.expect(Token::Arrow, "expected `->`")?;
        let result = if self.current == Some(Token::Void) {
            self.advance()?;
            FunctionResult::Void
        } else {
            FunctionResult::Value(self.type_annotation()?)
        };
        let (body, end) = if external {
            (
                Vec::new(),
                self.expect(Token::Semicolon, "expected `;` after external declaration")?
                    .end,
            )
        } else {
            self.body()?
        };
        Ok(self.syntax.functions.alloc(Function {
            name,
            name_span,
            parameters,
            variadic,
            result,
            external,
            span: start..end,
            body,
        }))
    }

    /// Parses a parenthesized parameter list and whether it ends in `...`,
    /// which only an external function may declare.
    fn parameters(&mut self, external: bool) -> Result<(Vec<TypedName>, bool), Diagnostic> {
        self.expect(Token::LeftParen, "expected `(`")?;
        let (parameters, ellipsis) = match self.current {
            Some(Token::RightParen) => (Vec::new(), None),
            Some(Token::Ellipsis) => {
                let span = self.expect(Token::Ellipsis, "expected `...`")?;
                (Vec::new(), Some(span))
            }
            _ => self.typed_names("parameter", Token::RightParen, true)?,
        };
        if let Some(span) = ellipsis.clone().filter(|_| !external) {
            return Err(Diagnostic::new(
                span,
                "only external functions can be variadic",
            ));
        }
        self.expect(Token::RightParen, "expected `)` after parameter list")?;
        Ok((parameters, ellipsis.is_some()))
    }

    /// Parses a comma-separated `name: T` list up to `close`, which a trailing
    /// comma may precede. Parameter lists and struct field lists share it.
    /// When `ellipsis` allows it, a `...` after a comma ends the list instead,
    /// and its span is returned.
    fn typed_names(
        &mut self,
        noun: &str,
        close: Token,
        ellipsis: bool,
    ) -> Result<(Vec<TypedName>, Option<Range<usize>>), Diagnostic> {
        let mut names = Vec::new();
        loop {
            let (name, name_span) = self.name(&format!("expected a {noun} name"))?;
            self.expect(Token::Colon, &format!("expected `:` after {noun} name"))?;
            let annotation = self.type_annotation()?;
            names.push(TypedName {
                name,
                name_span,
                annotation,
            });
            if self.current != Some(Token::Comma) {
                break;
            }
            self.advance()?;
            if ellipsis && self.current == Some(Token::Ellipsis) {
                let span = self.expect(Token::Ellipsis, "expected `...`")?;
                return Ok((names, Some(span)));
            }
            if self.current == Some(close) {
                break;
            }
        }
        Ok((names, None))
    }

    /// Parses `type Name struct { field: T, }`, which has at least one field
    /// and no terminating `;`.
    fn struct_declaration(&mut self) -> Result<Idx<StructDeclaration>, Diagnostic> {
        let start = self.expect(Token::Type, "expected `type`")?.start;
        let (name, name_span) = self.name("expected a type name")?;
        self.expect(
            Token::Struct,
            "named types other than structs are not yet implemented",
        )?;
        self.expect(Token::LeftBrace, "expected `{` after `struct`")?;
        if self.current == Some(Token::RightBrace) {
            return Err(self.error("expected a field declaration"));
        }
        let (fields, _) = self.typed_names("field", Token::RightBrace, false)?;
        let end = self
            .expect(Token::RightBrace, "expected `}` after struct fields")?
            .end;
        Ok(self.syntax.structs.alloc(StructDeclaration {
            name,
            name_span,
            fields,
            span: start..end,
        }))
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
            Some(Token::Name) if self.starts_assignment() => {
                self.assignment(StructLiterals::Permitted)?
            }
            Some(Token::Name) if self.starts_call() => StatementKind::Call { call: self.call()? },
            Some(Token::Name | Token::Star | Token::LeftParen) => {
                self.assignment(StructLiterals::Permitted)?
            }
            Some(Token::Break) => {
                self.advance()?;
                StatementKind::Break {}
            }
            Some(Token::Continue) => {
                self.advance()?;
                StatementKind::Continue {}
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
                    "expected a declaration, assignment, call, block, `if`, `for`, loop control, `return`, `exit`, or `free`",
                ));
            }
        };
        let end = self.expect(Token::Semicolon, "expected `;`")?.end;
        Ok(self.syntax.statements.alloc(Statement {
            kind,
            span: start..end,
        }))
    }

    /// Parses the `( expression [,] )` of `exit`, `free`, `len`, or `new`,
    /// which use call syntax without being calls. Returns the operand and the `)` span.
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

    /// Parses `target = e` or `target op= e`. A target is a unary expression,
    /// so `*p`, `(*pp).x`, and `a[i].next` all reach it through one grammar.
    /// A `{` after the target's name opens the statement's body rather than a
    /// struct literal, so the target is parsed with literals restricted.
    fn assignment(&mut self, literals: StructLiterals) -> Result<StatementKind, Diagnostic> {
        let target = self.unary_expression(StructLiterals::Restricted)?;
        let (operator, operator_span) = self.assignment_operator()?;
        let value = self.expression_with(literals)?;
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
            _ => return Err(self.error("expected `=`")),
        };
        let span = self.span.clone();
        self.advance()?;
        Ok((operator, span))
    }

    fn if_statement(&mut self) -> Result<Idx<Statement>, Diagnostic> {
        let start = self.expect(Token::If, "expected `if`")?.start;
        let condition = self.header_expression()?;
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
        let header = self.for_header()?;
        let (body, end) = self.body()?;
        Ok(self.syntax.statements.alloc(Statement {
            kind: StatementKind::For { header, body },
            span: start..end,
        }))
    }

    fn for_header(&mut self) -> Result<ForHeader, Diagnostic> {
        if self.current == Some(Token::LeftBrace) {
            return Ok(ForHeader::Infinite);
        }
        Ok(ForHeader::Condition(self.header_expression()?))
    }

    /// Returns the token that follows an assignment target starting at the
    /// current token, and whether the target carried any step: a prefix operator,
    /// a parenthesized group, an index, or a field selection. `None` when no
    /// target starts here.
    fn token_after_target(&self) -> Option<(Token, bool)> {
        let mut lexer = self.lexer.clone();
        let mut current = self.current?;
        let mut slice = self.lexer.slice();
        let mut stepped = false;
        while matches!(current, Token::Star | Token::Ampersand) {
            stepped = true;
            current = lookahead(&mut lexer)?;
            slice = lexer.slice();
        }
        let mut following = if matches!(current, Token::LeftParen | Token::LeftBracket) {
            stepped = true;
            token_after_group(&mut lexer, current)?
        } else {
            if current != Token::Name || reserved(slice) {
                return None;
            }
            let mut following = lookahead(&mut lexer)?;
            if following == Token::ColonColon {
                if lookahead(&mut lexer)? != Token::Name {
                    return None;
                }
                following = lookahead(&mut lexer)?;
            }
            if following == Token::LeftParen {
                following = token_after_group(&mut lexer, following)?;
            }
            following
        };
        loop {
            match following {
                Token::LeftBracket => {
                    following = token_after_group(&mut lexer, Token::LeftBracket)?;
                }
                Token::Dot => {
                    if lookahead(&mut lexer)? != Token::Name {
                        return None;
                    }
                    following = lookahead(&mut lexer)?;
                }
                _ => break,
            }
            stepped = true;
        }
        Some((following, stepped))
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
                    | Token::ShiftRightEquals,
                _
            ))
        )
    }

    fn starts_call(&self) -> bool {
        if self.current != Some(Token::Name) || reserved(self.lexer.slice()) {
            return false;
        }
        let mut lexer = self.lexer.clone();
        match lexer.next() {
            Some(Ok(Token::LeftParen)) => true,
            Some(Ok(Token::ColonColon)) => {
                lexer.next() == Some(Ok(Token::Name)) && lexer.next() == Some(Ok(Token::LeftParen))
            }
            Some(Ok(_)) | Some(Err(())) | None => false,
        }
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
        if self.current != Some(Token::Equals) {
            return Err(self.error("every binding requires an initializer"));
        }
        self.expect(Token::Equals, "expected `=` and an initializer")?;
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

    fn type_annotation(&mut self) -> Result<Idx<TypeAnnotation>, Diagnostic> {
        let start = self.span.start;
        if self.current == Some(Token::Star) {
            let start = self.span.start;
            self.advance()?;
            self.enter_nesting()?;
            let target = self.type_annotation()?;
            self.nesting -= 1;
            let end = self.syntax.annotations[target].span.end;
            return Ok(self.syntax.annotations.alloc(TypeAnnotation {
                kind: AnnotationKind::Pointer { target },
                span: start..end,
            }));
        }
        if self.current != Some(Token::LeftBracket) {
            // A built-in scalar name is reserved, so an ordinary identifier
            // here names a declared type.
            let (kind, span) = if self.current == Some(Token::Name) && !reserved(self.lexer.slice())
            {
                let name = self.qualified_name("expected a type")?;
                let span = name.span.clone();
                (AnnotationKind::Named(name), span)
            } else {
                let (ty, span) = self.named_type()?;
                (AnnotationKind::Scalar(ty), span)
            };
            return Ok(self.syntax.annotations.alloc(TypeAnnotation { kind, span }));
        }
        self.enter_nesting()?;
        self.advance()?;
        let length = self.expression()?;
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
        self.expression_with(StructLiterals::Permitted)
    }

    /// Parses an expression that a `{` follows, so a named struct literal in it
    /// must be parenthesized. A grouped or otherwise delimited subexpression
    /// reaches `expression` again and permits one.
    fn header_expression(&mut self) -> Result<Idx<Expression>, Diagnostic> {
        self.expression_with(StructLiterals::Restricted)
    }

    fn expression_with(&mut self, literals: StructLiterals) -> Result<Idx<Expression>, Diagnostic> {
        self.binary_expression(1, literals)
    }

    fn binary_expression(
        &mut self,
        minimum_precedence: u8,
        literals: StructLiterals,
    ) -> Result<Idx<Expression>, Diagnostic> {
        let mut left = self.unary_expression(literals)?;
        while let Some((operator, precedence)) = self.binary_operator() {
            if precedence < minimum_precedence {
                break;
            }
            let operator_span = self.span.clone();
            self.advance()?;
            let right = self.binary_expression(precedence + 1, literals)?;
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
            Token::ShiftLeft => (InfixOperator::Integer(BinaryOperator::ShiftLeft), 5),
            Token::ShiftRight => (InfixOperator::Integer(BinaryOperator::ShiftRight), 5),
            Token::Ampersand => (InfixOperator::Integer(BinaryOperator::And), 5),
            Token::Plus => (InfixOperator::Integer(BinaryOperator::Add), 4),
            Token::Minus => (InfixOperator::Integer(BinaryOperator::Subtract), 4),
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

    fn unary_expression(
        &mut self,
        literals: StructLiterals,
    ) -> Result<Idx<Expression>, Diagnostic> {
        if self.starts_pointer_conversion() {
            return self.postfix_expression(literals);
        }
        let operator = match self.current {
            Some(Token::Bang) => PrefixOperator::LogicalNot,
            Some(Token::Ampersand) => PrefixOperator::AddressOf,
            Some(Token::Star) => PrefixOperator::Dereference,
            Some(Token::Minus) => PrefixOperator::Value(UnaryOperator::Negate),
            Some(Token::Caret) => PrefixOperator::Value(UnaryOperator::Complement),
            _ => return self.postfix_expression(literals),
        };
        self.enter_nesting()?;
        let operator_span = self.span.clone();
        self.advance()?;
        let operand = self.unary_expression(literals)?;
        self.nesting -= 1;
        let span = operator_span.start..self.syntax.expressions[operand].span.end;
        let kind = match operator {
            PrefixOperator::LogicalNot => ExpressionKind::LogicalNot {
                operator_span,
                operand,
            },
            PrefixOperator::AddressOf => ExpressionKind::AddressOf {
                operator_span,
                operand,
            },
            PrefixOperator::Dereference => ExpressionKind::Dereference {
                operator_span,
                operand,
            },
            PrefixOperator::Value(operator) => ExpressionKind::Unary {
                operator,
                operator_span,
                operand,
            },
        };
        Ok(self.syntax.expressions.alloc(Expression {
            span,
            depth: self.syntax.expressions[operand].depth + 1,
            kind,
        }))
    }

    /// Parses a primary expression and the `[ … ]` indexes and `.name` field
    /// selections that follow it, in source order, so `grid[r][c]` indexes the
    /// row first and `make().rows[i].value` reads the call's result.
    fn postfix_expression(
        &mut self,
        literals: StructLiterals,
    ) -> Result<Idx<Expression>, Diagnostic> {
        let mut operand = self.primary_expression(literals)?;
        loop {
            let step_span = self.span.clone();
            // Each step's own depth is its index expression's, and a field
            // selection carries no subexpression of its own.
            let (kind, end, step_depth) = match self.current {
                Some(Token::LeftBracket) => {
                    self.enter_nesting()?;
                    self.advance()?;
                    let index = self.expression()?;
                    let end = self
                        .expect(Token::RightBracket, "expected `]` after index")?
                        .end;
                    self.nesting -= 1;
                    let depth = self.syntax.expressions[index].depth;
                    (ExpressionKind::Index { operand, index }, end, depth)
                }
                Some(Token::Dot) => {
                    self.advance()?;
                    let (name, name_span) = self.name("expected a field name after `.`")?;
                    let end = name_span.end;
                    (
                        ExpressionKind::Field {
                            operand,
                            name,
                            name_span,
                        },
                        end,
                        0,
                    )
                }
                _ => break,
            };
            let depth = self.syntax.expressions[operand].depth.max(step_depth) + 1;
            if self.nesting + depth > MAX_NESTING {
                return Err(self.nesting_error_at(step_span));
            }
            operand = self.syntax.expressions.alloc(Expression {
                span: self.syntax.expressions[operand].span.start..end,
                depth,
                kind,
            });
        }
        Ok(operand)
    }

    fn primary_expression(
        &mut self,
        literals: StructLiterals,
    ) -> Result<Idx<Expression>, Diagnostic> {
        let span = self.span.clone();
        let kind = if self.current == Some(Token::Number) {
            let spelling = self.lexer.slice().to_owned();
            self.advance()?;
            if is_floating(&spelling) {
                ExpressionKind::Floating(spelling)
            } else {
                ExpressionKind::Integer(spelling)
            }
        } else if self.current == Some(Token::String) {
            let bytes = literal_bytes(self.literal_body()).expect("validated by advance");
            self.advance()?;
            ExpressionKind::String(bytes)
        } else if self.current == Some(Token::Character) {
            let byte = self.character_byte().expect("validated by advance");
            self.advance()?;
            ExpressionKind::Character(byte)
        } else if self.current == Some(Token::Name)
            && Scalar::named(self.lexer.slice()).is_some_and(Scalar::is_numeric)
        {
            return self.conversion_expression(span);
        } else if matches!(self.current, Some(Token::True | Token::False)) {
            let value = self.current == Some(Token::True);
            self.advance()?;
            ExpressionKind::Boolean(value)
        } else if self.current == Some(Token::Null) {
            self.advance()?;
            ExpressionKind::Null
        } else if self.current == Some(Token::Star) {
            return self.pointer_conversion_expression(span);
        } else if self.current == Some(Token::LeftParen) {
            return self.grouping_expression(span);
        } else if self.current == Some(Token::LeftBracket) {
            return self.array_literal(span);
        } else if self.current == Some(Token::Len) {
            return self.length_expression(span);
        } else if self.current == Some(Token::Sizeof) {
            self.enter_nesting()?;
            self.advance()?;
            self.expect(Token::LeftParen, "expected `(` after sizeof")?;
            let annotation = self.type_annotation()?;
            let end = self
                .expect(Token::RightParen, "expected `)` after sizeof type")?
                .end;
            self.nesting -= 1;
            return Ok(self.syntax.expressions.alloc(Expression {
                kind: ExpressionKind::Sizeof { annotation },
                span: span.start..end,
                depth: 1,
            }));
        } else {
            return self.name_expression(span, literals);
        };
        Ok(self.syntax.expressions.alloc(Expression {
            kind,
            span,
            depth: 0,
        }))
    }

    /// Parses a nonempty array literal with an optional trailing comma.
    fn array_literal(&mut self, span: Range<usize>) -> Result<Idx<Expression>, Diagnostic> {
        self.enter_nesting()?;
        self.advance()?;
        if self.current == Some(Token::RightBracket) {
            return Err(self.error("expected an array element"));
        }
        let mut elements = Vec::new();
        loop {
            elements.push(self.expression()?);

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
            kind: ExpressionKind::ArrayLiteral { elements },
        }))
    }

    /// Parses a struct literal containing every field initializer.
    fn struct_literal(&mut self, name: QualifiedName) -> Result<Idx<Expression>, Diagnostic> {
        self.enter_nesting()?;
        let start = name.span.start;
        self.expect(Token::LeftBrace, "expected `{`")?;
        if self.current == Some(Token::RightBrace) {
            return Err(self.error("expected a field initializer"));
        }
        let mut fields = Vec::new();
        loop {
            let (field, name_span) = self.name("expected a field name")?;
            self.expect(Token::Equals, "expected `=` after field name")?;
            fields.push(FieldInitializer {
                name: field,
                name_span,
                value: self.expression()?,
            });
            match self.current {
                Some(Token::Comma) => {
                    self.advance()?;
                    if self.current == Some(Token::RightBrace) {
                        break;
                    }
                }
                Some(Token::RightBrace) => break,
                _ => return Err(self.error("expected `,` or `}` after field initializer")),
            }
        }
        let end = self
            .expect(Token::RightBrace, "expected `}` after field initializers")?
            .end;
        self.nesting -= 1;
        Ok(self.syntax.expressions.alloc(Expression {
            depth: fields
                .iter()
                .map(|field| self.syntax.expressions[field.value].depth)
                .max()
                .unwrap_or(0)
                + 1,
            span: start..end,
            kind: ExpressionKind::StructLiteral { name, fields },
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

    /// Distinguishes a pointer conversion from a dereference.
    fn starts_pointer_conversion(&self) -> bool {
        if self.current != Some(Token::Star) {
            return false;
        }
        let mut lexer = self.lexer.clone();
        let mut token = lookahead(&mut lexer);
        loop {
            match token {
                Some(Token::Star) => {}
                Some(Token::LeftBracket) => {
                    let mut depth = 1usize;
                    while depth > 0 {
                        match lookahead(&mut lexer) {
                            Some(Token::LeftBracket) => depth += 1,
                            Some(Token::RightBracket) => depth -= 1,
                            Some(_) => {}
                            None => return false,
                        }
                    }
                }
                _ => break,
            }
            token = lookahead(&mut lexer);
        }
        if token != Some(Token::Name) {
            return false;
        }
        let name = lexer.slice();
        if reserved(name) && Scalar::named(name).is_none() {
            return false;
        }
        token = lookahead(&mut lexer);
        if token == Some(Token::ColonColon) && Scalar::named(name).is_none() {
            if lookahead(&mut lexer) != Some(Token::Name) {
                return false;
            }
            token = lookahead(&mut lexer);
        }
        token == Some(Token::LeftParen)
    }

    fn pointer_conversion_expression(
        &mut self,
        span: Range<usize>,
    ) -> Result<Idx<Expression>, Diagnostic> {
        self.enter_nesting()?;
        let destination = self.type_annotation()?;
        self.expect(Token::LeftParen, "expected `(`")?;
        let operand = self.expression()?;
        let end = self.expect(Token::RightParen, "expected `)`")?.end;
        self.nesting -= 1;
        Ok(self.syntax.expressions.alloc(Expression {
            kind: ExpressionKind::PointerConversion {
                destination,
                operand,
            },
            span: span.start..end,
            depth: self.syntax.expressions[operand].depth + 1,
        }))
    }

    fn conversion_expression(&mut self, span: Range<usize>) -> Result<Idx<Expression>, Diagnostic> {
        self.enter_nesting()?;
        let (destination, _) = self.named_type()?;
        if self.current != Some(Token::LeftParen) {
            return Err(Diagnostic::new(
                span,
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
                operand,
            },
            span: span.start..end,
            depth: self.syntax.expressions[operand].depth + 1,
        }))
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

    /// Parses a name, which is a call when an argument list follows it, a
    /// struct literal when a permitted `{` follows it, and a reference
    /// otherwise.
    fn name_expression(
        &mut self,
        span: Range<usize>,
        literals: StructLiterals,
    ) -> Result<Idx<Expression>, Diagnostic> {
        let target = self.qualified_name("expected an expression")?;
        if self.current == Some(Token::LeftBrace) && literals == StructLiterals::Permitted {
            return self.struct_literal(target);
        }
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
