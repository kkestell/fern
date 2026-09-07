use crate::{
    diagnostic::Diagnostic,
    frontend::{
        Expression, ExpressionKind, Function, Statement, StatementKind, Syntax, integer_parts,
    },
};
use la_arena::{Arena, ArenaMap, Idx};
use lasso::Spur;
use num_bigint::BigUint;
use num_traits::ToPrimitive;
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
            "int" => Self::Int,
            "uint" => Self::Uint,
            _ => return None,
        })
    }

    pub(crate) fn name(self) -> &'static str {
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
        self.width_on(usize::BITS)
    }

    pub(crate) fn width_on(self, pointer_width: u32) -> u32 {
        match self {
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 => 16,
            Self::I32 | Self::U32 => 32,
            Self::I64 | Self::U64 => 64,
            Self::Int | Self::Uint => pointer_width,
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
}

#[derive(Debug)]
pub(crate) struct Binding {
    pub ty: Type,
    pub mutable: bool,
    pub constant: Option<i128>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ExpressionValue {
    Integer(i128),
    Reference(Idx<Binding>),
    Conversion {
        operand: Idx<Expression>,
        truncating: bool,
    },
}

#[derive(Debug)]
pub(crate) struct CheckedExpression {
    pub ty: Type,
    pub value: ExpressionValue,
    pub constant: Option<i128>,
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
                        .map(|a| Type::named(&a.name).expect("frontend validates integer types"));
                    let expression = self.check_expression(*initializer, scopes, destination)?;
                    let ty = expression.ty;
                    let constant = if *mutable { None } else { expression.constant };
                    self.expressions.insert(*initializer, expression);
                    let binding = self.bindings.alloc(Binding {
                        ty,
                        mutable: *mutable,
                        constant,
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
        &mut self,
        id: Idx<Expression>,
        scopes: &[HashMap<Spur, Idx<Binding>>],
        destination: Option<Type>,
    ) -> Result<CheckedExpression, Diagnostic> {
        let expression = &self.syntax.expressions[id];
        let error = |message| Diagnostic::new(expression.span.clone(), message);
        let (source_ty, value, constant) = match &expression.kind {
            ExpressionKind::Integer(spelling) => {
                let (base, digits, suffix) = integer_parts(spelling);
                debug_assert!(suffix.is_empty(), "frontend rejects literal suffixes");
                let ty = destination.unwrap_or(Type::Int);
                let value = BigUint::parse_bytes(digits.as_bytes(), base)
                    .expect("frontend validated integer digits");
                if value > BigUint::from(ty.max()) {
                    return Err(error(format!(
                        "integer literal out of range for `{}`",
                        ty.name()
                    )));
                }
                let value = value
                    .to_i128()
                    .expect("integer value representable by every Fern type");
                (ty, ExpressionValue::Integer(value), Some(value))
            }
            ExpressionKind::Reference(name) => {
                let binding = self.resolve(*name, expression.span.clone(), scopes)?;
                (
                    self.bindings[binding].ty,
                    ExpressionValue::Reference(binding),
                    self.bindings[binding].constant,
                )
            }
            ExpressionKind::Conversion {
                destination: annotation,
                truncating,
                operand,
            } => {
                let destination =
                    Type::named(&annotation.name).expect("frontend validates integer types");
                if *truncating
                    && matches!(
                        self.syntax.expressions[*operand].kind,
                        ExpressionKind::Integer(_)
                    )
                {
                    let ExpressionKind::Integer(spelling) = &self.syntax.expressions[*operand].kind
                    else {
                        unreachable!()
                    };
                    let (base, digits, _) = integer_parts(spelling);
                    let value = BigUint::parse_bytes(digits.as_bytes(), base)
                        .expect("frontend validated integer digits");
                    let modulus = BigUint::from(1u8) << destination.width();
                    let bits = (value % modulus)
                        .to_u64()
                        .expect("truncated Fern integer fits in u64");
                    let value = integer_from_bits(bits, destination);
                    (destination, ExpressionValue::Integer(value), Some(value))
                } else {
                    let operand_destination = if !*truncating
                        && matches!(
                            self.syntax.expressions[*operand].kind,
                            ExpressionKind::Integer(_)
                        ) {
                        Some(destination)
                    } else {
                        None
                    };
                    let operand_id = *operand;
                    let checked_operand =
                        self.check_expression(operand_id, scopes, operand_destination)?;
                    let constant = checked_operand.constant;
                    self.expressions.insert(operand_id, checked_operand);
                    if let Some(value) = constant {
                        let value = if *truncating {
                            truncate_integer(value, destination)
                        } else if integer_fits(value, destination) {
                            value
                        } else {
                            return Err(error(format!(
                                "constant conversion to `{}` would trap",
                                destination.name()
                            )));
                        };
                        (destination, ExpressionValue::Integer(value), Some(value))
                    } else {
                        (
                            destination,
                            ExpressionValue::Conversion {
                                operand: operand_id,
                                truncating: *truncating,
                            },
                            None,
                        )
                    }
                }
            }
        };
        let ty = destination.unwrap_or(source_ty);
        if source_ty != ty {
            return Err(error(format!(
                "cannot implicitly convert `{}` to `{}`",
                source_ty.name(),
                ty.name()
            )));
        }
        Ok(CheckedExpression {
            ty,
            value,
            constant,
        })
    }
}

fn integer_fits(value: i128, ty: Type) -> bool {
    let minimum = if ty.signed() {
        -(1i128 << (ty.width() - 1))
    } else {
        0
    };
    value >= minimum && value <= i128::from(ty.max())
}

fn integer_from_bits(bits: u64, ty: Type) -> i128 {
    let value = i128::from(bits);
    if ty.signed() && value >= (1i128 << (ty.width() - 1)) {
        value - (1i128 << ty.width())
    } else {
        value
    }
}

fn truncate_integer(value: i128, ty: Type) -> i128 {
    let modulus = 1i128 << ty.width();
    integer_from_bits(value.rem_euclid(modulus) as u64, ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parse;

    const SPEC_INTEGER_TYPES: [(&str, Type); 10] = [
        ("i8", Type::I8),
        ("i16", Type::I16),
        ("i32", Type::I32),
        ("i64", Type::I64),
        ("u8", Type::U8),
        ("u16", Type::U16),
        ("u32", Type::U32),
        ("u64", Type::U64),
        ("int", Type::Int),
        ("uint", Type::Uint),
    ];

    fn literal(value: u128, base: u32) -> String {
        match base {
            2 => format!("0b{value:b}"),
            8 => format!("0o{value:o}"),
            10 => value.to_string(),
            16 => format!("0x{value:X}"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn bindings_have_concrete_types_and_distinct_identities() {
        let text =
            "fn main() -> void { const x = 1; var x: int = x; const x = x; exit(x); exit(0); }";
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
                ExpressionValue::Integer(_) | ExpressionValue::Conversion { .. } => None,
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
                "var x: u8 = 1; { x = 256; }",
                "256",
                "integer literal out of range for `u8`",
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
    fn integer_contract_uses_contextual_literals_and_exact_references() {
        for (name, ty) in SPEC_INTEGER_TYPES {
            let max = u128::from(ty.max());
            for base in [2, 8, 10, 16] {
                let maximum = literal(max, base);
                let syntax = parse(&format!(
                    "fn main() -> void {{ var x: {name} = {maximum}; x = {maximum}; }}"
                ))
                .unwrap();
                let checked = check(&syntax).unwrap();
                assert!(checked.bindings.iter().all(|(_, binding)| binding.ty == ty));
                assert!(
                    checked
                        .expressions
                        .iter()
                        .all(|(_, expression)| expression.ty == ty)
                );

                let overflow = literal(max + 1, base);
                rejects(
                    &format!("exit(0); var x: {name} = {overflow};"),
                    &overflow,
                    &format!("integer literal out of range for `{name}`"),
                );
            }
        }

        for (source_name, source) in SPEC_INTEGER_TYPES {
            for (destination_name, destination) in SPEC_INTEGER_TYPES {
                for target in [
                    format!("const target: {destination_name} = source;"),
                    format!("var target: {destination_name} = source;"),
                    format!("var target: {destination_name} = 0; target = source;"),
                ] {
                    for scope in [target.clone(), format!("{{ exit(0); {target} }}")] {
                        let body = format!("const source: {source_name} = 1; {scope}");
                        if source == destination {
                            let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
                            check(&syntax).unwrap();
                        } else {
                            rejects(
                                &body,
                                "source",
                                &format!(
                                    "cannot implicitly convert `{source_name}` to `{destination_name}`"
                                ),
                            );
                        }
                    }
                }
            }
            let body = format!("const source: {source_name} = 1; exit(source);");
            if source == Type::Int {
                let syntax = parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
                check(&syntax).unwrap();
            } else {
                rejects(
                    &body,
                    "source",
                    &format!("cannot implicitly convert `{source_name}` to `int`"),
                );
            }
        }
    }

    #[test]
    fn conversions_evaluate_constants_and_preserve_runtime_operands() {
        let syntax = parse(
            "fn main() -> void {
                const literal: u8 = 42;
                const reduced = u8.truncate(340282366920938463463374607431768211498);
                const signed = i8.truncate(255);
                const widened = u64(literal);
                var runtime: u64 = 42;
                const checked = u8(runtime);
                const truncated = u8.truncate(runtime);
                const later = u16(checked);
            }",
        )
        .unwrap();
        let checked = check(&syntax).unwrap();
        let constants: Vec<_> = checked
            .bindings
            .iter()
            .map(|(_, binding)| binding.constant)
            .collect();
        assert_eq!(
            constants,
            [
                Some(42),
                Some(42),
                Some(-1),
                Some(42),
                None,
                None,
                None,
                None
            ]
        );
        assert_eq!(
            checked
                .expressions
                .iter()
                .filter(|(_, expression)| {
                    matches!(expression.value, ExpressionValue::Conversion { .. })
                })
                .count(),
            3
        );

        rejects(
            "const value: u64 = 18446744073709551615; const narrowed = u8(value);",
            "u8(value)",
            "constant conversion to `u8` would trap",
        );
        rejects(
            "const narrowed = u8(256);",
            "256",
            "integer literal out of range for `u8`",
        );
        rejects(
            "exit(0); const value: u64 = 256; const narrowed = u8(value);",
            "u8(value)",
            "constant conversion to `u8` would trap",
        );
    }

    #[test]
    fn truncating_constants_keep_at_least_256_bits_before_reduction() {
        for literal in [
            format!("0x{}", "F".repeat(64)),
            format!("0b{}", "1".repeat(256)),
            format!("0x1{}FF", "0".repeat(128)),
        ] {
            let syntax = parse(&format!(
                "fn main() -> void {{ const result = int(u8.truncate({literal})); }}"
            ))
            .unwrap();
            let checked = check(&syntax).unwrap();
            assert_eq!(
                checked.bindings.iter().next().unwrap().1.constant,
                Some(255)
            );
            rejects(
                &format!("exit(0); const result = u8.truncate(u64({literal}));"),
                &literal,
                "integer literal out of range for `u64`",
            );
        }
    }

    #[test]
    fn constant_classification_follows_copies_and_shadowing() {
        let syntax = parse(
            "fn main() -> void {
                const original: u64 = 255;
                const copy = original;
                { var original = copy;
                  const saved = original;
                  const converted = u8(saved); }
                const folded = i8.truncate(copy);
                const extended = i64(folded);
                const wrapped = u64.truncate(extended);
            }",
        )
        .unwrap();
        let checked = check(&syntax).unwrap();
        assert_eq!(
            checked
                .bindings
                .iter()
                .map(|(_, binding)| binding.constant)
                .collect::<Vec<_>>(),
            [
                Some(255),
                Some(255),
                None,
                None,
                None,
                Some(-1),
                Some(-1),
                Some(i128::from(u64::MAX))
            ],
        );
        for body in [
            "const original: u64 = 256; const copy = original; const bad = u8(copy);",
            "const original: u64 = 256; { var original: u64 = 1; } const bad = u8(original);",
            "const negative = i8.truncate(255); const bad = u64(negative);",
        ] {
            let syntax = parse(&format!("fn main() -> void {{ exit(0); {body} }}")).unwrap();
            assert!(check(&syntax).unwrap_err().message.contains("would trap"));
        }
    }

    #[test]
    fn native_integer_widths_follow_the_host_and_model_both_specified_widths() {
        assert_eq!(Type::Int.width(), usize::BITS);
        assert_eq!(Type::Uint.width(), usize::BITS);
        for pointer_width in [32, 64] {
            assert_eq!(Type::Int.width_on(pointer_width), pointer_width);
            assert_eq!(Type::Uint.width_on(pointer_width), pointer_width);
            assert_eq!(Type::I32.width_on(pointer_width), 32);
            assert_eq!(Type::U64.width_on(pointer_width), 64);
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
