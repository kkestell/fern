use crate::{
    diagnostic::Diagnostic,
    frontend::{
        Expression, ExpressionKind, Function, Statement, StatementKind, Syntax, integer_parts,
    },
};
use la_arena::{Arena, ArenaMap, Idx};
use lasso::Spur;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Type {
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

impl Type {
    fn named(name: &str) -> Option<Self> {
        Some(match name {
            "i8" => Self::I8,
            "i16" => Self::I16,
            "i32" => Self::I32,
            "i64" => Self::I64,
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            "int" | "i" => Self::Int,
            "uint" | "u" => Self::Uint,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Int => "int",
            Self::Uint => "uint",
        }
    }

    pub(crate) fn width(self) -> u32 {
        match self {
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 => 16,
            Self::I32 | Self::U32 | Self::Int | Self::Uint => 32,
            Self::I64 | Self::U64 => 64,
        }
    }

    pub(crate) fn signed(self) -> bool {
        matches!(
            self,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::Int
        )
    }

    pub(crate) fn max(self) -> u64 {
        u64::MAX >> (64 - self.width() + u32::from(self.signed()))
    }

    pub(crate) fn converts_to(self, destination: Self) -> bool {
        self.signed() == destination.signed() && self.width() <= destination.width()
    }
}

#[derive(Debug)]
pub(crate) struct Binding {
    pub ty: Type,
    pub mutable: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ExpressionValue {
    Integer(u64),
    Reference(Idx<Binding>),
}

#[derive(Debug)]
pub(crate) struct CheckedExpression {
    // The literal or binding type before the contextual conversion.
    pub source_ty: Type,
    pub ty: Type,
    pub value: ExpressionValue,
}

#[derive(Debug)]
pub(crate) struct CheckedEntry<'a> {
    pub syntax: &'a Syntax,
    pub main: Idx<Function>,
    pub expressions: ArenaMap<Idx<Expression>, CheckedExpression>,
    pub declarations: ArenaMap<Idx<Statement>, Idx<Binding>>,
    pub bindings: Arena<Binding>,
    pub assignments: ArenaMap<Idx<Statement>, Idx<Binding>>,
}

pub(crate) fn check(syntax: &Syntax) -> Result<CheckedEntry<'_>, Diagnostic> {
    let mut main = None;
    for (id, function) in syntax.functions.iter() {
        if syntax.names.resolve(&function.name) != "main" {
            return Err(Diagnostic::new(
                function.span.clone(),
                "only the `main` function is supported",
            ));
        }
        if main.replace(id).is_some() {
            return Err(Diagnostic::new(
                function.name_span.clone(),
                "duplicate `main` function",
            ));
        }
    }
    let main = main.ok_or_else(|| Diagnostic::new(0..0, "missing `main` function"))?;
    let mut checked = CheckedEntry {
        syntax,
        main,
        expressions: ArenaMap::default(),
        declarations: ArenaMap::default(),
        bindings: Arena::default(),
        assignments: ArenaMap::default(),
    };
    checked.check_body(&syntax.functions[main].body, &mut Vec::new())?;
    Ok(checked)
}

impl CheckedEntry<'_> {
    fn check_body(
        &mut self,
        body: &[Idx<Statement>],
        scopes: &mut Vec<HashMap<Spur, Idx<Binding>>>,
    ) -> Result<(), Diagnostic> {
        scopes.push(HashMap::new());
        for &statement in body {
            match &self.syntax.statements[statement].kind {
                StatementKind::Binding {
                    name,
                    mutable,
                    annotation,
                    initializer,
                    ..
                } => {
                    let destination = annotation
                        .as_ref()
                        .map(|a| {
                            Type::named(&a.name).ok_or_else(|| {
                                Diagnostic::new(
                                    a.span.clone(),
                                    format!("unsupported integer type `{}`", a.name),
                                )
                            })
                        })
                        .transpose()?;
                    let expression = self.check_expression(*initializer, scopes, destination)?;
                    let ty = expression.ty;
                    self.expressions.insert(*initializer, expression);
                    let binding = self.bindings.alloc(Binding {
                        ty,
                        mutable: *mutable,
                    });
                    self.declarations.insert(statement, binding);
                    scopes.last_mut().unwrap().insert(*name, binding);
                }
                StatementKind::Assignment {
                    name,
                    name_span,
                    value,
                } => {
                    let binding = self.resolve(*name, name_span.clone(), scopes)?;
                    if !self.bindings[binding].mutable {
                        return Err(Diagnostic::new(
                            name_span.clone(),
                            format!(
                                "cannot assign to immutable binding `{}`",
                                self.syntax.names.resolve(name)
                            ),
                        ));
                    }
                    let expression =
                        self.check_expression(*value, scopes, Some(self.bindings[binding].ty))?;
                    self.expressions.insert(*value, expression);
                    self.assignments.insert(statement, binding);
                }
                StatementKind::Block { body } => self.check_body(body, scopes)?,
                StatementKind::Exit { argument } => {
                    let expression = self.check_expression(*argument, scopes, Some(Type::Int))?;
                    self.expressions.insert(*argument, expression);
                }
            }
        }
        scopes.pop();
        Ok(())
    }

    fn resolve(
        &self,
        name: Spur,
        span: std::ops::Range<usize>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
    ) -> Result<Idx<Binding>, Diagnostic> {
        scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(&name).copied())
            .ok_or_else(|| {
                Diagnostic::new(
                    span,
                    format!("unknown binding `{}`", self.syntax.names.resolve(&name)),
                )
            })
    }

    fn check_expression(
        &self,
        id: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        destination: Option<Type>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let expression = &self.syntax.expressions[id];
        let error = |message| Diagnostic::new(expression.span.clone(), message);
        let (source_ty, value) = match &expression.kind {
            ExpressionKind::Integer(spelling) => {
                let (base, digits, suffix) = integer_parts(spelling);
                let ty = if suffix.is_empty() {
                    destination.unwrap_or(Type::Int)
                } else {
                    Type::named(suffix)
                        .ok_or_else(|| error(format!("unsupported integer suffix `{suffix}`")))?
                };
                let mut value = 0u64;
                for digit in digits.chars() {
                    let digit = digit
                        .to_digit(base)
                        .expect("frontend validated integer digits");
                    value = value
                        .checked_mul(u64::from(base))
                        .and_then(|value| value.checked_add(u64::from(digit)))
                        .filter(|value| *value <= ty.max())
                        .ok_or_else(|| {
                            error(format!("integer literal out of range for `{}`", ty.name()))
                        })?;
                }
                (ty, ExpressionValue::Integer(value))
            }
            ExpressionKind::Reference(name) => {
                let binding = self.resolve(*name, expression.span.clone(), scopes)?;
                (
                    self.bindings[binding].ty,
                    ExpressionValue::Reference(binding),
                )
            }
        };
        let ty = destination.unwrap_or(source_ty);
        if !source_ty.converts_to(ty) {
            return Err(error(format!(
                "cannot implicitly convert `{}` to `{}`",
                source_ty.name(),
                ty.name()
            )));
        }
        Ok(CheckedExpression {
            source_ty,
            ty,
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parse;

    #[test]
    fn bindings_have_concrete_types_and_distinct_identities() {
        let text =
            "fn main() -> void { const x = 1; var x: int = x; const x = x; exit(x); exit(0i); }";
        let syntax = parse(text).unwrap();
        let checked = check(&syntax).unwrap();
        assert!(std::ptr::eq(checked.syntax, &syntax));
        let statements = &syntax.functions[checked.main].body;
        let ids: Vec<_> = statements[..3]
            .iter()
            .map(|s| checked.declarations[*s])
            .collect();
        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[1], ids[2]);
        assert_ne!(ids[0], ids[2]);
        for id in &ids {
            assert_eq!(checked.bindings[*id].ty, Type::Int);
        }
        let facts: Vec<_> = syntax
            .expressions
            .iter()
            .map(|(id, _)| &checked.expressions[id])
            .collect();
        assert_eq!(facts.len(), 5);
        assert!(facts.iter().all(|fact| fact.ty == Type::Int));
        assert_eq!(facts[0].value, ExpressionValue::Integer(1));
        assert_eq!(facts[1].value, ExpressionValue::Reference(ids[0]));
        assert_eq!(facts[2].value, ExpressionValue::Reference(ids[1]));
        assert_eq!(facts[3].value, ExpressionValue::Reference(ids[2]));
        assert_eq!(facts[4].value, ExpressionValue::Integer(0));
    }

    #[test]
    fn nested_scopes_resolve_binding_identity_and_mutability() {
        let syntax = parse("fn main() -> void { var x = 1; { x = 2; const x = x; { var x = x; x = x; } exit(x); } x = x; const x = x; exit(x); }").unwrap();
        let checked = check(&syntax).unwrap();
        let ids: Vec<_> = checked.bindings.iter().map(|(id, _)| id).collect();
        assert_eq!(ids.len(), 4);
        let mutable: Vec<_> = checked.bindings.iter().map(|(_, b)| b.mutable).collect();
        assert_eq!(mutable, [true, false, true, false]);
        let targets: Vec<_> = checked.assignments.iter().map(|(_, id)| *id).collect();
        assert_eq!(targets, [ids[0], ids[2], ids[0]]);
        let references: Vec<_> = checked
            .expressions
            .iter()
            .filter_map(|(_, expression)| match expression.value {
                ExpressionValue::Reference(id) => Some(id),
                ExpressionValue::Integer(_) => None,
            })
            .collect();
        assert_eq!(
            references,
            [ids[0], ids[1], ids[2], ids[1], ids[0], ids[0], ids[3]]
        );
    }

    #[test]
    fn assignment_errors_use_target_or_value_spans_even_after_nested_exit() {
        for (body, offending, message) in [
            (
                "const x = 1; x = 2;",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            (
                "var x = 1; const x = 2; x = 3;",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            (
                "var x = 1; { const x = 2; x = 3; }",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            (
                "const x = 1; { var x = 2; x = 3; } x = 4;",
                "x",
                "cannot assign to immutable binding `x`",
            ),
            ("x = 2;", "x", "unknown binding `x`"),
            ("{ var x = 1; } x = 2;", "x", "unknown binding `x`"),
            ("{ const x = 1; } exit(x);", "x", "unknown binding `x`"),
            ("{ var x = x; }", "x", "unknown binding `x`"),
            (
                "var x = 1; { x = missing; }",
                "missing",
                "unknown binding `missing`",
            ),
            (
                "var x = 1; { x = 2147483648; }",
                "2147483648",
                "integer literal out of range for `int`",
            ),
            (
                "var x = 1; { x = 1u8; }",
                "1u8",
                "cannot implicitly convert `u8` to `int`",
            ),
        ] {
            rejects(body, offending, message);
            rejects(&format!("{{ exit(0); }} {body}"), offending, message);
            rejects(&format!("{{ exit(0); {body} }}"), offending, message);
        }
        for body in [
            "const x = 1; var x = x; x = 2;",
            "const x = 1; { var x = x; x = 2; }",
            "var x = 1; { const x = x; } x = 2;",
        ] {
            let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
            check(&syntax).unwrap();
        }
    }

    fn rejects(body: &str, offending: &str, message: &str) {
        let text = format!("/* 🌿 */ fn main() -> void {{ {body} }}");
        let syntax = parse(&text).unwrap();
        let error = check(&syntax).unwrap_err();
        let start = text.rfind(offending).unwrap();
        assert_eq!(error.span, start..start + offending.len(), "{body}");
        assert_eq!(error.message, message, "{body}");
    }

    #[test]
    fn names_require_a_preceding_binding_even_after_exit() {
        for body in [
            "const x = x;",
            "exit(x); const x = 1;",
            "var y = x;",
            "exit(0); exit(x);",
            "const y = 1; exit(0); var x = x;",
        ] {
            // In the forward-reference case the later declaration has the same name.
            let text = format!("/* 🌿 */ fn main() -> void {{ {body} }}");
            let syntax = parse(&text).unwrap();
            let error = check(&syntax).unwrap_err();
            let start = if body.starts_with("exit(x)") {
                text.find("exit(x)").unwrap() + 5
            } else {
                text.rfind('x').unwrap()
            };
            assert_eq!(error.span, start..start + 1);
            assert_eq!(error.message, "unknown binding `x`");
        }
    }

    #[test]
    fn integer_ranges_cover_all_bases_and_supported_suffixes() {
        for (prefix, max, overflow) in [
            ("", "2147483647", "2147483648"),
            ("0x", "7FFFFFFF", "80000000"),
            ("0o", "17777777777", "20000000000"),
            (
                "0b",
                "1111111111111111111111111111111",
                "10000000000000000000000000000000",
            ),
        ] {
            for suffix in ["", "i"] {
                for (digits, expected) in [("0", 0), ("00000", 0), (max, i32::MAX as u64)] {
                    for zeros in [String::new(), "0".repeat(1000)] {
                        let literal = format!("{prefix}{zeros}{digits}{suffix}");
                        for body in [
                            format!("var x = {literal};"),
                            format!("const x: int = {literal};"),
                            format!("exit({literal});"),
                        ] {
                            let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
                            let checked = check(&syntax).unwrap();
                            let (id, _) = syntax.expressions.iter().next().unwrap();
                            assert_eq!(checked.expressions[id].ty, Type::Int);
                            assert_eq!(
                                checked.expressions[id].value,
                                ExpressionValue::Integer(expected)
                            );
                        }
                    }
                }
                for digits in [overflow.to_owned(), "1".repeat(1000)] {
                    let literal = format!("{prefix}{digits}{suffix}");
                    rejects(
                        &format!("exit(0); const x = {literal};"),
                        &literal,
                        "integer literal out of range for `int`",
                    );
                }
            }
        }
    }

    const INTEGER_TYPES: [(&str, &str, Type, u32, bool); 10] = [
        ("i8", "i8", Type::I8, 8, true),
        ("i16", "i16", Type::I16, 16, true),
        ("i32", "i32", Type::I32, 32, true),
        ("i64", "i64", Type::I64, 64, true),
        ("u8", "u8", Type::U8, 8, false),
        ("u16", "u16", Type::U16, 16, false),
        ("u32", "u32", Type::U32, 32, false),
        ("u64", "u64", Type::U64, 64, false),
        ("int", "i", Type::Int, 32, true),
        ("uint", "u", Type::Uint, 32, false),
    ];

    #[test]
    fn all_integer_ranges_in_every_base() {
        for (name, suffix, ty, width, signed) in INTEGER_TYPES {
            let max = (1u128 << (width - u32::from(signed))) - 1;
            for base in [2, 8, 10, 16] {
                let literal = |value: u128| match base {
                    2 => format!("0b{value:b}"),
                    8 => format!("0o{value:o}"),
                    10 => value.to_string(),
                    16 => format!("0x{value:X}"),
                    _ => unreachable!(),
                };
                for value in [0, max] {
                    for body in [
                        format!("const x = {}{suffix}; const copy = x;", literal(value)),
                        format!(
                            "var x: {name} = {}; x = {};",
                            literal(value),
                            literal(value)
                        ),
                    ] {
                        let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
                        let checked = check(&syntax).unwrap();
                        assert!(checked.bindings.iter().all(|(_, b)| b.ty == ty));
                        let first = checked.expressions.iter().next().unwrap().1;
                        assert_eq!(first.source_ty, ty);
                        assert_eq!(first.ty, ty);
                        assert_eq!(first.value, ExpressionValue::Integer(value as u64));
                    }
                }
                let overflow = literal(max + 1);
                for (body, offending) in [
                    (
                        format!("const x = {overflow}{suffix};"),
                        format!("{overflow}{suffix}"),
                    ),
                    (format!("const x: {name} = {overflow};"), overflow.clone()),
                    (
                        format!("var x: {name} = 0; x = {overflow};"),
                        overflow.clone(),
                    ),
                ] {
                    rejects(
                        &body,
                        &offending,
                        &format!("integer literal out of range for `{name}`"),
                    );
                }
            }
            let literal = format!("{}1{suffix}", "0".repeat(1000));
            let syntax = parse(&format!("fn main() -> void {{ const x = {literal}; }}")).unwrap();
            check(&syntax).unwrap();
            let huge = format!("{}{suffix}", "9".repeat(1000));
            rejects(
                &format!("const x = {huge};"),
                &huge,
                &format!("integer literal out of range for `{name}`"),
            );
        }
        rejects("const x = 1z;", "1z", "unsupported integer suffix `z`");
        // Context does not rescue a suffixed literal outside its own range.
        rejects(
            "const x: u64 = 256u8;",
            "256u8",
            "integer literal out of range for `u8`",
        );
    }

    #[test]
    fn conversions_in_initialization_assignment_and_exit() {
        for (source_name, suffix, source_ty, source_width, source_signed) in INTEGER_TYPES {
            for (destination_name, _, destination_ty, destination_width, destination_signed) in
                INTEGER_TYPES
            {
                let allowed =
                    source_signed == destination_signed && source_width <= destination_width;
                for (body, offending) in [
                    (
                        format!("const target: {destination_name} = 1{suffix};"),
                        format!("1{suffix}"),
                    ),
                    (
                        format!(
                            "const source = 1{suffix}; const target: {destination_name} = source;"
                        ),
                        "source".into(),
                    ),
                    (
                        format!(
                            "const source = 1{suffix}; var target: {destination_name} = 0; target = source;"
                        ),
                        "source".into(),
                    ),
                ] {
                    if allowed {
                        let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
                        let checked = check(&syntax).unwrap();
                        let last = checked.expressions.iter().last().unwrap().1;
                        assert_eq!(last.source_ty, source_ty);
                        assert_eq!(last.ty, destination_ty);
                        assert_eq!(checked.bindings.iter().last().unwrap().1.ty, destination_ty);
                    } else {
                        let message = format!(
                            "cannot implicitly convert `{source_name}` to `{destination_name}`"
                        );
                        rejects(&body, &offending, &message);
                        rejects(&format!("{{ exit(0); {body} }}"), &offending, &message);
                    }
                }
            }
            for argument in [format!("1{suffix}"), "source".into()] {
                let body = format!("const source = 1{suffix}; exit({argument});");
                if source_signed && source_width <= 32 {
                    let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
                    let checked = check(&syntax).unwrap();
                    let exit = checked.expressions.iter().last().unwrap().1;
                    assert_eq!(exit.source_ty, source_ty);
                    assert_eq!(exit.ty, Type::Int);
                } else {
                    rejects(
                        &body,
                        &argument,
                        &format!("cannot implicitly convert `{source_name}` to `int`"),
                    );
                }
            }
        }
    }

    #[test]
    fn typed_shadowing_copies_and_nested_assignments() {
        let syntax = parse(
            "fn main() -> void {
            var x = 1u8;
            const saved = x;
            { const x: u64 = x; var x = x; x = saved; }
            x = 255;
            { exit(0); const x: u16 = x; const copy = x; }
            const x: u32 = x;
            const x: uint = x;
        }",
        )
        .unwrap();
        let checked = check(&syntax).unwrap();
        let types: Vec<_> = checked.bindings.iter().map(|(_, b)| b.ty).collect();
        assert_eq!(
            types,
            [
                Type::U8,
                Type::U8,
                Type::U64,
                Type::U64,
                Type::U16,
                Type::U16,
                Type::U32,
                Type::Uint
            ]
        );
        rejects(
            "const x = 1; const y: u8 = x;",
            "x",
            "cannot implicitly convert `int` to `u8`",
        );
        rejects(
            "var x: u8 = 1; { const x: u64 = x; } x = 256;",
            "256",
            "integer literal out of range for `u8`",
        );
    }

    #[test]
    fn entry_checks_keep_their_spans_and_precede_body_checks() {
        for (text, span, message) in [
            ("/* 🌿 */", 0..0, "missing `main` function"),
            (
                "fn main() -> void {} fn main() -> void {}",
                24..28,
                "duplicate `main` function",
            ),
            (
                "fn other() -> void {}",
                0..21,
                "only the `main` function is supported",
            ),
        ] {
            let syntax = parse(text).unwrap();
            let error = check(&syntax).unwrap_err();
            assert_eq!(error.span, span);
            assert_eq!(error.message, message);
        }
        let syntax = parse("fn main() -> void { exit(x); } fn other() -> void {}").unwrap();
        assert_eq!(
            check(&syntax).unwrap_err().message,
            "only the `main` function is supported"
        );
    }
}
