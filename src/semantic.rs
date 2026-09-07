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
    Int,
}

#[derive(Debug)]
pub(crate) struct Binding {
    pub ty: Type,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ExpressionValue {
    Integer(i32),
    Reference(Idx<Binding>),
}

#[derive(Debug)]
pub(crate) struct CheckedExpression {
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
    };
    let mut scope = HashMap::new();
    for &statement in &syntax.functions[main].body {
        match &syntax.statements[statement].kind {
            StatementKind::Binding {
                name,
                int_annotation,
                initializer,
                ..
            } => {
                let expression = checked.check_expression(*initializer, &scope)?;
                let ty = match int_annotation {
                    Some(_) => Type::Int,
                    None => expression.ty,
                };
                checked.expressions.insert(*initializer, expression);
                let binding = checked.bindings.alloc(Binding { ty });
                checked.declarations.insert(statement, binding);
                scope.insert(*name, binding);
            }
            StatementKind::Exit { argument } => {
                let expression = checked.check_expression(*argument, &scope)?;
                checked.expressions.insert(*argument, expression);
            }
        }
    }
    Ok(checked)
}

impl CheckedEntry<'_> {
    fn check_expression(
        &self,
        id: Idx<Expression>,
        scope: &HashMap<Spur, Idx<Binding>>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let expression = &self.syntax.expressions[id];
        let error = |message| Diagnostic::new(expression.span.clone(), message);
        let value = match &expression.kind {
            ExpressionKind::Integer(spelling) => {
                let (base, digits, suffix) = integer_parts(spelling);
                if !matches!(suffix, "" | "i") {
                    return Err(error(format!("unsupported integer suffix `{suffix}`")));
                }
                let mut value = 0i32;
                for digit in digits.chars() {
                    let digit = digit
                        .to_digit(base)
                        .expect("frontend validated integer digits");
                    value = value
                        .checked_mul(base as i32)
                        .and_then(|value| value.checked_add(digit as i32))
                        .ok_or_else(|| error("integer literal out of range for `int`".into()))?;
                }
                ExpressionValue::Integer(value)
            }
            ExpressionKind::Reference(name) => {
                let binding = scope.get(name).ok_or_else(|| {
                    error(format!(
                        "unknown binding `{}`",
                        self.syntax.names.resolve(name)
                    ))
                })?;
                return Ok(CheckedExpression {
                    ty: self.bindings[*binding].ty,
                    value: ExpressionValue::Reference(*binding),
                });
            }
        };
        Ok(CheckedExpression {
            ty: Type::Int,
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
                for (digits, expected) in [("0", 0), ("00000", 0), (max, i32::MAX)] {
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
            for suffix in [
                "u", "z", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
            ] {
                for digits in ["1".to_owned(), "1".repeat(1000)] {
                    let literal = format!("{prefix}{digits}{suffix}");
                    rejects(
                        &format!("exit({literal});"),
                        &literal,
                        &format!("unsupported integer suffix `{suffix}`"),
                    );
                }
            }
        }
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
