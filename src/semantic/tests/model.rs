use super::*;

#[test]
fn bindings_have_concrete_types_and_distinct_identities() {
    let text = "fn main() -> void { const x = 1; var x: int = x; const x = x; exit(x); exit(0); }";
    let syntax = parse(text).unwrap();
    let checked = check_root(&syntax).unwrap();
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
        assert_eq!(checked.bindings[*id].ty, value_type(Scalar::Int));
    }
    let facts: Vec<_> = syntax
        .expressions
        .iter()
        .map(|(id, _)| &checked.expressions[id])
        .collect();
    assert_eq!(facts.len(), 5);
    assert!(facts.iter().all(|fact| fact.ty == value_type(Scalar::Int)));
    assert_eq!(facts[0].value, ExpressionValue::Integer);
    assert_eq!(facts[1].value, ExpressionValue::Reference(ids[0]));
    assert_eq!(facts[2].value, ExpressionValue::Reference(ids[1]));
    assert_eq!(facts[3].value, ExpressionValue::Reference(ids[2]));
    assert_eq!(facts[4].value, ExpressionValue::Integer);
}

#[test]
fn nested_scopes_resolve_binding_identity_and_mutability() {
    let syntax = parse("fn main() -> void { var x = 1; { x = 2; const x = x; { var x = x; x = x; } exit(x); } x = x; const x = x; exit(x); }").unwrap();
    let checked = check_root(&syntax).unwrap();
    let ids: Vec<_> = checked.bindings.iter().map(|(id, _)| id).collect();
    assert_eq!(ids.len(), 4);
    let mutable: Vec<_> = checked.bindings.iter().map(|(_, b)| b.mutable).collect();
    assert_eq!(mutable, [true, false, true, false]);
    let targets: Vec<_> = checked
        .assignments
        .iter()
        .map(|(_, target)| target.binding)
        .collect();
    assert_eq!(targets, [ids[0], ids[2], ids[0]]);
    let references: Vec<_> = checked
        .expressions
        .iter()
        .filter_map(|(_, expression)| match &expression.value {
            ExpressionValue::Reference(id) => Some(*id),
            ExpressionValue::Integer
            | ExpressionValue::Boolean
            | ExpressionValue::Conversion { .. }
            | ExpressionValue::Grouping { .. }
            | ExpressionValue::Unary { .. }
            | ExpressionValue::Binary { .. }
            | ExpressionValue::Comparison { .. }
            | ExpressionValue::Logical { .. }
            | ExpressionValue::Array { .. }
            | ExpressionValue::Index { .. }
            | ExpressionValue::Length { .. }
            | ExpressionValue::LogicalNot { .. }
            | ExpressionValue::Call { .. } => None,
        })
        .collect();
    assert_eq!(
        references,
        [ids[0], ids[1], ids[2], ids[1], ids[0], ids[0], ids[3]]
    );
}
