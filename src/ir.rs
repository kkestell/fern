use crate::{
    CompileError,
    frontend::StatementKind,
    semantic::{CheckedEntry, ExpressionValue},
};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValueId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operand {
    Integer(i32),
    Value(ValueId),
}

#[derive(Debug)]
pub(crate) struct Entry {
    // Each position defines the corresponding IR-local value ID.
    pub values: Vec<Operand>,
    pub exit: Operand,
}

#[derive(Debug)]
pub(crate) struct VerifiedEntry(Entry);

impl VerifiedEntry {
    pub(crate) fn entry(&self) -> &Entry {
        &self.0
    }
}

impl Entry {
    pub(crate) fn verify(self) -> Result<VerifiedEntry, CompileError> {
        for (index, operand) in self.values.iter().enumerate() {
            if let Operand::Value(ValueId(reference)) = operand
                && *reference >= index
            {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR value {index} references undefined value {reference}"
                )));
            }
        }
        if let Operand::Value(ValueId(reference)) = self.exit
            && reference >= self.values.len()
        {
            return Err(CompileError::new(format!(
                "internal compiler error: IR exit references undefined value {reference}"
            )));
        }
        Ok(VerifiedEntry(self))
    }
}

pub(crate) fn lower(checked: CheckedEntry<'_>) -> Entry {
    let mut entry = Entry {
        values: Vec::new(),
        exit: Operand::Integer(0),
    };
    let mut bindings = HashMap::new();
    for &statement in &checked.syntax.functions[checked.main].body {
        let operand = |expression| match checked.expressions[expression].value {
            ExpressionValue::Integer(value) => Operand::Integer(value),
            ExpressionValue::Reference(binding) => Operand::Value(bindings[&binding]),
        };
        match checked.syntax.statements[statement].kind {
            StatementKind::Binding { initializer, .. } => {
                let value = operand(initializer);
                let id = ValueId(entry.values.len());
                entry.values.push(value);
                bindings.insert(checked.declarations[statement], id);
            }
            StatementKind::Exit { argument } => {
                entry.exit = operand(argument);
                break;
            }
        }
    }
    entry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{frontend, semantic};

    #[test]
    fn lowered_entries() {
        let fixtures = [
            ("empty", ""),
            ("literals", "const a = 42; var b: int = 7i;"),
            ("copies", "const a = 42; var b = a; const c = b; exit(c);"),
            (
                "shadowing",
                "const x = 42; var x = x; const saved = x; const x = 7; var x = x; exit(saved);",
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
            vec![Operand::Value(ValueId(usize::MAX))],
            vec![Operand::Value(ValueId(1)), Operand::Integer(42)],
            vec![Operand::Value(ValueId(0))],
        ] {
            let error = Entry {
                values,
                exit: Operand::Integer(0),
            }
            .verify()
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("internal compiler error: IR value 0")
            );
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
                values: vec![Operand::Integer(42)],
                exit: Operand::Value(ValueId(1))
            }
            .verify()
            .is_err()
        );
    }

    #[test]
    fn verification_accepts_constants_copies_and_exits() {
        for exit in [
            Operand::Integer(-1),
            Operand::Value(ValueId(0)),
            Operand::Value(ValueId(1)),
        ] {
            Entry {
                values: vec![Operand::Integer(i32::MIN), Operand::Value(ValueId(0))],
                exit,
            }
            .verify()
            .unwrap();
        }
    }
}
