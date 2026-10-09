use crate::{
    frontend::parser,
    ir::{lower::lower, model::*},
    semantic::namespaces,
    source::SourceMap,
    types::{Scalar, StringId},
};
fn program(source: &str) -> Program {
    let syntax = parser::parse(&SourceMap::from_text(source)).unwrap();
    lower(namespaces::check_root(&syntax).unwrap())
}
#[test]
fn raw_pointers_and_external_calls_verify() {
    let p = program(
        "extern fn putchar(c:i32)->i32;fn main()->void {var a=[1,2];var p=&a[0];p[1]=3;var n=(p+2)-p;putchar(i32(n));}",
    );
    assert!(
        p.functions
            .iter()
            .flat_map(|f| &f.values)
            .any(|v| matches!(v.kind, ValueKind::PointerDifference { .. }))
    );
    p.verify().unwrap();
}
#[test]
fn rejects_invalid_control_flow_and_types() {
    let mut p = program("fn main()->void {var x=1;}");
    p.functions[0].flow.blocks[0].terminator = Terminator::Jump {
        target: BlockId(999),
    };
    assert!(p.verify().is_err());
    let mut p = program("fn main()->void {var x=1;}");
    p.functions[0].flow.blocks[0]
        .instructions
        .push(Instruction::Store {
            place: Place::Local(LocalId(0)),
            operand: Operand::Literal(Literal::Integer {
                value: 1,
                ty: Scalar::U8,
            }),
        });
    assert!(p.verify().is_err());
    let mut p = program("extern fn f()->i32;fn main()->void {}");
    p.functions[0].result = Some(Scalar::F64.into());
    p.verify().unwrap();
    let mut p = program("extern fn f()->i32;fn main()->void {}");
    p.functions[0].result = Some(Scalar::Bool.into());
    assert!(p.verify().is_err());
}
#[test]
fn rejects_values_used_before_definition() {
    let mut p = program("fn main()->void {var x=1;var y=x+2;}");
    let f = &mut p.functions[0];
    let id = ValueId(f.values.len() - 1);
    f.flow.blocks[0].instructions.insert(
        0,
        Instruction::Store {
            place: Place::Local(LocalId(0)),
            operand: Operand::Value(id),
        },
    );
    assert!(p.verify().is_err());
}
#[test]
fn len_does_not_load_its_operand() {
    let p = program(
        "fn count(a:[4]i64)->void {var p=&a;var n=len(*p);var m=len(a);}fn main()->void {}",
    );
    assert!(!p.functions.iter().flat_map(|f| &f.values).any(|v| {
        matches!(v.kind, ValueKind::Load(_)) && matches!(v.ty, crate::types::Type::Array { .. })
    }));
    p.verify().unwrap();
}
#[test]
fn equal_string_literals_share_one_entry() {
    let p = program(
        "extern fn puts(s:*u8)->i32;var m:*u8=\"hi\";fn f()->void {puts(\"hi\");} fn main()->void {f();puts(\"hi\");}",
    );
    assert_eq!(p.strings, [b"hi".to_vec()]);
    assert_eq!(
        p.globals[0].values,
        Some(vec![Literal::String(StringId(0))])
    );
    p.verify().unwrap();
}
