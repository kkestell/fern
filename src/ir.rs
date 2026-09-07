use crate::{
    CompileError,
    frontend::{Expression, Statement, StatementKind},
    semantic::{Binding, CheckedEntry, ExpressionValue, Type},
};
use la_arena::Idx;
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
    Convert { operand: Operand, truncating: bool },
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
                let min = if ty.signed() {
                    -(1i128 << (ty.width() - 1))
                } else {
                    0
                };
                if value < min || value > i128::from(ty.max()) {
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
            let (source, valid) = match value.kind {
                ValueKind::Copy(operand) => {
                    let source = operand.verify(&self.values[..index])?;
                    (source, source == value.ty)
                }
                ValueKind::Convert { operand, .. } => {
                    (operand.verify(&self.values[..index])?, true)
                }
            };
            if !valid {
                return Err(CompileError::new(format!(
                    "internal compiler error: invalid IR value {index}: {:?} from {source:?} to {:?}",
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
    let operand = match expression.value {
        ExpressionValue::Integer(value) => Operand::Integer {
            value,
            ty: expression.ty,
        },
        ExpressionValue::Reference(binding) => Operand::Value(bindings[&binding]),
        ExpressionValue::Conversion { operand, .. } => {
            let value = lower_expression(checked, operand, bindings, entry);
            match value.kind {
                ValueKind::Copy(operand) => operand,
                ValueKind::Convert { .. } => Operand::Value(entry.push(value)),
            }
        }
    };
    Value {
        span: matches!(expression.value, ExpressionValue::Conversion { .. })
            .then(|| checked.syntax.expressions[id].span.clone()),
        ty: expression.ty,
        kind: if let ExpressionValue::Conversion { truncating, .. } = expression.value {
            ValueKind::Convert {
                operand,
                truncating,
            }
        } else {
            ValueKind::Copy(operand)
        },
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
                let value = lower_expression(checked, *argument, bindings, entry);
                entry.exit = match value.kind {
                    ValueKind::Copy(operand) => operand,
                    ValueKind::Convert { .. } => Operand::Value(entry.push(value)),
                };
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
        [
            (Type::I8, -128, 127),
            (Type::I16, -32768, 32767),
            (Type::I32, -2147483648, 2147483647),
            (Type::I64, -9223372036854775808, 9223372036854775807),
            (Type::U8, 0, 255),
            (Type::U16, 0, 65535),
            (Type::U32, 0, 4294967295),
            (Type::U64, 0, 18446744073709551615),
            (
                Type::Int,
                -(1i128 << (Type::Int.width() - 1)),
                i128::from(Type::Int.max()),
            ),
            (Type::Uint, 0, i128::from(Type::Uint.max())),
        ]
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
}
