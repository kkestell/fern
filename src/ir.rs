use crate::{
    CompileError,
    frontend::{BinaryOperator, Expression, Statement, StatementKind, UnaryOperator},
    semantic::{Binding, CheckedEntry, ExpressionValue, Type},
};
use la_arena::Idx;
use num_traits::ToPrimitive;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValueId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operand {
    // i128 holds both the signed minima and the full u64 range without bit reinterpretation.
    Integer { value: i128, ty: Type },
    Value(ValueId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValueKind {
    Copy(Operand),
    Convert {
        operand: Operand,
        truncating: bool,
    },
    Unary {
        operator: UnaryOperator,
        operand: Operand,
    },
    Binary {
        operator: BinaryOperator,
        left: Operand,
        right: Operand,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Value {
    pub span: Option<std::ops::Range<usize>>,
    pub ty: Type,
    pub kind: ValueKind,
}

#[derive(Debug)]
pub(crate) struct Entry {
    // Each position defines the corresponding IR-local value ID.
    pub values: Vec<Value>,
    pub exit: Operand,
}

#[derive(Debug)]
pub(crate) struct VerifiedEntry(Entry);

impl VerifiedEntry {
    pub(crate) fn entry(&self) -> &Entry {
        &self.0
    }
}

impl Operand {
    fn verify(self, preceding: &[Value]) -> Result<Type, CompileError> {
        match self {
            Self::Integer { value, ty } => {
                if value < ty.min() || value > i128::from(ty.max()) {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR integer {value} out of range for {ty:?}"
                    )));
                }
                Ok(ty)
            }
            Self::Value(ValueId(id)) => preceding.get(id).map(|value| value.ty).ok_or_else(|| {
                CompileError::new(format!(
                    "internal compiler error: IR references undefined value {id}"
                ))
            }),
        }
    }
}

impl Entry {
    pub(crate) fn verify(self) -> Result<VerifiedEntry, CompileError> {
        for (index, value) in self.values.iter().enumerate() {
            let valid = match value.kind {
                ValueKind::Copy(operand) => {
                    let source = operand.verify(&self.values[..index])?;
                    source == value.ty
                }
                ValueKind::Convert { operand, .. } => {
                    operand.verify(&self.values[..index])?;
                    true
                }
                ValueKind::Unary { operator, operand } => {
                    let source = operand.verify(&self.values[..index])?;
                    value.span.is_some()
                        && source == value.ty
                        && (operator != UnaryOperator::Negate || source.signed())
                }
                ValueKind::Binary {
                    operator,
                    left,
                    right,
                } => {
                    let left = left.verify(&self.values[..index])?;
                    let right = right.verify(&self.values[..index])?;
                    value.span.is_some()
                        && left == value.ty
                        && (matches!(
                            operator,
                            BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight
                        ) || right == left)
                }
            };
            if !valid {
                return Err(CompileError::new(format!(
                    "internal compiler error: invalid IR value {index}: {:?} with result {:?}",
                    value.kind, value.ty
                )));
            }
        }
        if self.exit.verify(&self.values)? != Type::Int {
            return Err(CompileError::new(
                "internal compiler error: IR exit requires int",
            ));
        }
        Ok(VerifiedEntry(self))
    }

    fn push(&mut self, value: Value) -> ValueId {
        let id = ValueId(self.values.len());
        self.values.push(value);
        id
    }
}

pub(crate) fn lower(checked: CheckedEntry<'_>) -> Entry {
    let mut entry = Entry {
        values: Vec::new(),
        exit: Operand::Integer {
            value: 0,
            ty: Type::Int,
        },
    };
    let mut bindings = HashMap::new();
    lower_body(
        &checked,
        &checked.syntax.functions[checked.main].body,
        &mut bindings,
        &mut entry,
    );
    entry
}

fn lower_expression(
    checked: &CheckedEntry<'_>,
    id: Idx<Expression>,
    bindings: &HashMap<Idx<Binding>, ValueId>,
    entry: &mut Entry,
) -> Value {
    let expression = &checked.expressions[id];
    if matches!(
        expression.value,
        ExpressionValue::Integer
            | ExpressionValue::Grouping { .. }
            | ExpressionValue::Conversion { .. }
            | ExpressionValue::Unary { .. }
            | ExpressionValue::Binary { .. }
    ) && let Some(constant) = expression.constant.as_ref()
    {
        return constant_value(constant, expression.ty);
    }
    match &expression.value {
        ExpressionValue::Integer => unreachable!("integer expressions are constant"),
        ExpressionValue::Reference(binding) => Value {
            span: None,
            ty: expression.ty,
            kind: ValueKind::Copy(Operand::Value(bindings[binding])),
        },
        ExpressionValue::Grouping { expression } => {
            lower_expression(checked, *expression, bindings, entry)
        }
        ExpressionValue::Conversion {
            operand,
            truncating,
        } => {
            let operand = lower_operand(checked, *operand, bindings, entry);
            Value {
                span: Some(checked.syntax.expressions[id].span.clone()),
                ty: expression.ty,
                kind: ValueKind::Convert {
                    operand,
                    truncating: *truncating,
                },
            }
        }
        ExpressionValue::Unary {
            operator,
            operator_span,
            operand,
        } => {
            let operand = lower_operand(checked, *operand, bindings, entry);
            Value {
                span: Some(operator_span.clone()),
                ty: expression.ty,
                kind: ValueKind::Unary {
                    operator: *operator,
                    operand,
                },
            }
        }
        ExpressionValue::Binary {
            operator,
            operator_span,
            left,
            right,
        } => {
            let left = lower_operand(checked, *left, bindings, entry);
            let right = lower_operand(checked, *right, bindings, entry);
            Value {
                span: Some(operator_span.clone()),
                ty: expression.ty,
                kind: ValueKind::Binary {
                    operator: *operator,
                    left,
                    right,
                },
            }
        }
    }
}

fn constant_value(value: &num_bigint::BigInt, ty: Type) -> Value {
    Value {
        span: None,
        ty,
        kind: ValueKind::Copy(Operand::Integer {
            value: value
                .to_i128()
                .expect("concrete Fern integer fits in the IR representation"),
            ty,
        }),
    }
}

fn lower_operand(
    checked: &CheckedEntry<'_>,
    id: Idx<Expression>,
    bindings: &HashMap<Idx<Binding>, ValueId>,
    entry: &mut Entry,
) -> Operand {
    let value = lower_expression(checked, id, bindings, entry);
    match value.kind {
        ValueKind::Copy(operand) => operand,
        _ => Operand::Value(entry.push(value)),
    }
}

fn lower_body(
    checked: &CheckedEntry<'_>,
    body: &[Idx<Statement>],
    bindings: &mut HashMap<Idx<Binding>, ValueId>,
    entry: &mut Entry,
) -> bool {
    for &statement in body {
        match &checked.syntax.statements[statement].kind {
            StatementKind::Binding { initializer, .. } => {
                let value = lower_expression(checked, *initializer, bindings, entry);
                let id = entry.push(value);
                bindings.insert(checked.declarations[statement], id);
            }
            StatementKind::Assignment { value, .. } => {
                let value = lower_expression(checked, *value, bindings, entry);
                let id = entry.push(value);
                bindings.insert(checked.assignments[statement], id);
            }
            StatementKind::Block { body } => {
                if lower_body(checked, body, bindings, entry) {
                    return true;
                }
            }
            StatementKind::Exit { argument } => {
                entry.exit = lower_operand(checked, *argument, bindings, entry);
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{frontend, semantic};

    fn integer(value: i128, ty: Type) -> Operand {
        Operand::Integer { value, ty }
    }

    fn copy(operand: Operand, ty: Type) -> Value {
        Value {
            span: None,
            ty,
            kind: ValueKind::Copy(operand),
        }
    }

    #[test]
    fn conversions_use_existing_operands_and_keep_their_source_spans() {
        let text = "fn main() -> void { var x: u64 = 42; var y = u8(u16(x)); exit(int(y)); }";
        let syntax = frontend::parse(text).unwrap();
        let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
        let values = &entry.entry().values;
        assert_eq!(values.len(), 4);
        for (id, spelling) in [(1, "u16(x)"), (2, "u8(u16(x))"), (3, "int(y)")] {
            let start = text.find(spelling).unwrap();
            assert_eq!(values[id].span, Some(start..start + spelling.len()));
            assert_eq!(
                values[id].kind,
                ValueKind::Convert {
                    operand: Operand::Value(ValueId(id - 1)),
                    truncating: false,
                }
            );
        }
    }

    #[test]
    fn lowered_entries() {
        let fixtures = [
            ("empty", ""),
            (
                "typed_integers",
                "
                const a: i8 = 127; const b: i16 = 32767; const c: i32 = 2147483647;
                const d: i64 = 9223372036854775807; const e: u8 = 255;
                const f: u16 = 65535; const g: u32 = 4294967295;
                const h: u64 = 18446744073709551615; const i: int = 2147483647;
                const j: uint = 4294967295; const copy = h;
                const contextual: u64 = 18446744073709551615;
            ",
            ),
            (
                "typed_conversions",
                "
                var small: i8 = 42; var medium = i16(small);
                var wide = i64(medium); var unsigned: u8 = 255;
                var unsigned_wide = u64(unsigned); var native = 42; var fixed = i32(native);
                var back = int(fixed); var u: uint = 42;
                var uf = u32(u); var ub = uint(uf);
                exit(int(small));
            ",
            ),
            (
                "typed_scopes",
                "
                var x: i8 = 42; const saved = x;
                { var x = i64(x); x = i64(saved); const copy = x; }
                x = 7; var status = int(x); status = int(saved);
                { const status = saved; exit(int(status)); const ignored: u64 = 1; }
                exit(0);
            ",
            ),
            (
                "integer_expressions",
                "
                var x: int = 40; var y: int = 2;
                const add = x + y; const subtract = x - y;
                const multiply = x * y; const divide = x / y; const remainder = x % y;
                const wrapping_add = x &+ y; const wrapping_subtract = x &- y;
                const wrapping_multiply = x &* y;
                const negate = -x; const wrapping_negate = &-x; const complement = ^x;
                const and = x & y; const and_not = x &^ y;
                const xor = x ^ y; const or = x | y;
                const shift_left = x << y; const shift_right = x >> y;
                const folded: int = (250 + 10) / 2;
                exit(add);
            ",
            ),
            ("converted_literal_exit", "exit(42);"),
            ("literals", "const a = 42; var b: int = 7;"),
            ("copies", "const a = 42; var b = a; const c = b; exit(c);"),
            (
                "shadowing",
                "const x = 42; var x = x; const saved = x; const x = 7; var x = x; exit(saved);",
            ),
            (
                "assignments",
                "var x = 1; x = 42; x = x; const saved = x; x = 7; x = saved; exit(saved);",
            ),
            (
                "nested_scopes",
                "var x = 1; {} { { x = 42; } const x = x; { var x = x; x = 7; } } exit(x);",
            ),
            (
                "nested_exit",
                "var x = 1; { x = 42; { exit(x); x = 7; } x = 8; } x = 9; exit(x);",
            ),
            ("early_exit", "const x = 42; exit(x); const y = x; exit(y);"),
        ];
        for (name, body) in fixtures {
            let syntax = frontend::parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
            let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
            insta::assert_debug_snapshot!(name, entry.entry());
        }
    }

    #[test]
    fn verification_rejects_invalid_references() {
        for values in [
            vec![copy(Operand::Value(ValueId(usize::MAX)), Type::Int)],
            vec![
                copy(Operand::Value(ValueId(1)), Type::Int),
                copy(integer(42, Type::Int), Type::Int),
            ],
            vec![copy(Operand::Value(ValueId(0)), Type::Int)],
        ] {
            let error = Entry {
                values,
                exit: integer(0, Type::Int),
            }
            .verify()
            .unwrap_err();
            assert!(error.to_string().contains("references undefined value"));
        }
        for reference in [0, 1, usize::MAX] {
            assert!(
                Entry {
                    values: vec![],
                    exit: Operand::Value(ValueId(reference))
                }
                .verify()
                .is_err()
            );
        }
        assert!(
            Entry {
                values: vec![copy(integer(42, Type::Int), Type::Int)],
                exit: Operand::Value(ValueId(1))
            }
            .verify()
            .is_err()
        );
    }

    #[test]
    fn verification_accepts_constants_copies_and_exits() {
        for exit in [
            integer(-1, Type::Int),
            Operand::Value(ValueId(0)),
            Operand::Value(ValueId(1)),
        ] {
            Entry {
                values: vec![
                    copy(integer(i128::from(i32::MIN), Type::Int), Type::Int),
                    copy(Operand::Value(ValueId(0)), Type::Int),
                ],
                exit,
            }
            .verify()
            .unwrap();
        }
    }

    fn ranges() -> [(Type, i128, i128); 10] {
        Type::ALL.map(|ty| (ty, ty.min(), i128::from(ty.max())))
    }

    #[test]
    fn verification_checks_literal_ranges_and_copy_types() {
        for (ty, min, max) in ranges() {
            for value in [min, 0, max] {
                Entry {
                    values: vec![
                        copy(integer(value, ty), ty),
                        copy(Operand::Value(ValueId(0)), ty),
                    ],
                    exit: integer(0, Type::Int),
                }
                .verify()
                .unwrap();
            }
            for value in [i128::MIN, min - 1, max + 1, i128::MAX] {
                let error = Entry {
                    values: vec![copy(integer(value, ty), ty)],
                    exit: integer(0, Type::Int),
                }
                .verify()
                .unwrap_err();
                assert!(error.to_string().contains("out of range"));
            }
            for (other, _, _) in ranges() {
                if ty == other {
                    continue;
                }
                for operand in [integer(0, ty), Operand::Value(ValueId(0))] {
                    assert!(
                        Entry {
                            values: vec![copy(integer(0, ty), ty), copy(operand, other)],
                            exit: integer(0, Type::Int),
                        }
                        .verify()
                        .is_err()
                    );
                }
            }
        }
    }

    #[test]
    fn verification_checks_conversions_and_exits() {
        for (source, min, max) in ranges() {
            for (destination, _, _) in ranges() {
                for value in [min, max] {
                    for operand in [integer(value, source), Operand::Value(ValueId(0))] {
                        let result = Entry {
                            values: vec![
                                copy(integer(value, source), source),
                                Value {
                                    span: None,
                                    ty: destination,
                                    kind: ValueKind::Convert {
                                        operand,
                                        truncating: false,
                                    },
                                },
                            ],
                            exit: integer(0, Type::Int),
                        }
                        .verify();
                        assert!(result.is_ok(), "{source:?} to {destination:?}: {value}");
                    }
                }
            }
            for exit in [integer(0, source), Operand::Value(ValueId(0))] {
                assert_eq!(
                    Entry {
                        values: vec![copy(integer(0, source), source)],
                        exit,
                    }
                    .verify()
                    .is_ok(),
                    source == Type::Int
                );
            }
        }
        for operand in [
            integer(128, Type::I8),
            Operand::Value(ValueId(0)),
            Operand::Value(ValueId(usize::MAX)),
        ] {
            assert!(
                Entry {
                    values: vec![Value {
                        span: None,
                        ty: Type::Int,
                        kind: ValueKind::Convert {
                            operand,
                            truncating: false,
                        }
                    }],
                    exit: integer(0, Type::Int),
                }
                .verify()
                .is_err()
            );
        }
        for value in [
            -(1i128 << (Type::Int.width() - 1)) - 1,
            i128::from(Type::Int.max()) + 1,
        ] {
            assert!(
                Entry {
                    values: vec![],
                    exit: integer(value, Type::Int)
                }
                .verify()
                .is_err()
            );
        }
    }

    #[test]
    fn verification_accepts_checked_conversion_operands() {
        Entry {
            values: vec![
                copy(integer(42, Type::U8), Type::U8),
                Value {
                    span: None,
                    ty: Type::U64,
                    kind: ValueKind::Convert {
                        operand: Operand::Value(ValueId(0)),
                        truncating: false,
                    },
                },
            ],
            exit: integer(0, Type::Int),
        }
        .verify()
        .unwrap();
    }

    #[test]
    fn lowering_preserves_nested_operand_order_types_and_operator_spans() {
        let text = "fn main() -> void { var left: int = 8; var right: int = 2; var count: uint = 1; const result = (left + right) * (right - int(count)); exit(result); }";
        let syntax = frontend::parse(text).unwrap();
        let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
        let values = &entry.entry().values;
        assert_eq!(values.len(), 7);
        assert_eq!(
            values[3].kind,
            ValueKind::Binary {
                operator: BinaryOperator::Add,
                left: Operand::Value(ValueId(0)),
                right: Operand::Value(ValueId(1)),
            }
        );
        assert_eq!(
            values[4].kind,
            ValueKind::Convert {
                operand: Operand::Value(ValueId(2)),
                truncating: false,
            }
        );
        assert_eq!(
            values[5].kind,
            ValueKind::Binary {
                operator: BinaryOperator::Subtract,
                left: Operand::Value(ValueId(1)),
                right: Operand::Value(ValueId(4)),
            }
        );
        assert_eq!(
            values[6].kind,
            ValueKind::Binary {
                operator: BinaryOperator::Multiply,
                left: Operand::Value(ValueId(3)),
                right: Operand::Value(ValueId(5)),
            }
        );
        for (id, spelling) in [(3, "+"), (5, "-"), (6, "*")] {
            let start = text.find(&format!(" {spelling} ")).unwrap() + 1;
            assert_eq!(values[id].span, Some(start..start + spelling.len()));
            assert_eq!(values[id].ty, Type::Int);
        }
        assert_eq!(entry.entry().exit, Operand::Value(ValueId(6)));
    }

    #[test]
    fn lowering_contextualizes_an_untyped_runtime_shift_operand() {
        let syntax = frontend::parse(
            "fn main() -> void { var count: uint = 3; const shifted: u64 = (1 << count) << count; }",
        )
        .unwrap();
        let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
        assert_eq!(entry.entry().values.len(), 3);
        assert_eq!(
            entry.entry().values[1].kind,
            ValueKind::Binary {
                operator: BinaryOperator::ShiftLeft,
                left: integer(1, Type::U64),
                right: Operand::Value(ValueId(0)),
            }
        );
        assert_eq!(entry.entry().values[1].ty, Type::U64);
        assert_eq!(
            entry.entry().values[2].kind,
            ValueKind::Binary {
                operator: BinaryOperator::ShiftLeft,
                left: Operand::Value(ValueId(1)),
                right: Operand::Value(ValueId(0)),
            }
        );
        assert_eq!(entry.entry().values[2].ty, Type::U64);
    }

    #[test]
    fn lowering_preserves_contextual_types_through_grouping() {
        let syntax = frontend::parse(
            "fn main() -> void { const grouped: u8 = ((42)); var count: uint = 1; const shifted: u64 = (1 + 2) << count; }",
        )
        .unwrap();
        let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
        assert_eq!(entry.entry().values.len(), 3);
        assert_eq!(
            entry.entry().values[0],
            copy(integer(42, Type::U8), Type::U8)
        );
        assert_eq!(
            entry.entry().values[2].kind,
            ValueKind::Binary {
                operator: BinaryOperator::ShiftLeft,
                left: integer(3, Type::U64),
                right: Operand::Value(ValueId(1)),
            }
        );
        assert_eq!(entry.entry().values[2].ty, Type::U64);
    }

    #[test]
    fn verification_checks_operation_shapes_and_accepts_independent_shift_counts() {
        let operation = |ty, kind| Value {
            span: Some(0..1),
            ty,
            kind,
        };
        Entry {
            values: vec![
                copy(integer(1, Type::U8), Type::U8),
                copy(integer(1, Type::U16), Type::U16),
                operation(
                    Type::U8,
                    ValueKind::Binary {
                        operator: BinaryOperator::ShiftLeft,
                        left: Operand::Value(ValueId(0)),
                        right: Operand::Value(ValueId(1)),
                    },
                ),
            ],
            exit: integer(0, Type::Int),
        }
        .verify()
        .unwrap();

        for value in [
            operation(
                Type::U8,
                ValueKind::Unary {
                    operator: UnaryOperator::Negate,
                    operand: integer(1, Type::U8),
                },
            ),
            operation(
                Type::U16,
                ValueKind::Unary {
                    operator: UnaryOperator::Complement,
                    operand: integer(1, Type::U8),
                },
            ),
            operation(
                Type::U8,
                ValueKind::Binary {
                    operator: BinaryOperator::Add,
                    left: integer(1, Type::U8),
                    right: integer(1, Type::U16),
                },
            ),
            Value {
                span: None,
                ty: Type::U8,
                kind: ValueKind::Binary {
                    operator: BinaryOperator::Add,
                    left: integer(1, Type::U8),
                    right: integer(1, Type::U8),
                },
            },
        ] {
            assert!(
                Entry {
                    values: vec![value],
                    exit: integer(0, Type::Int),
                }
                .verify()
                .unwrap_err()
                .to_string()
                .contains("invalid IR value")
            );
        }
    }
}
