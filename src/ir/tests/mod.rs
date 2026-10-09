use crate::{
    frontend::parser,
    ir::{lower::lower, model::*},
    semantic::namespaces,
    source::SourceMap,
    types::Scalar,
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
    let mut p = program("extern fn f()->i32;fn main()->void {f();}");
    p.functions[0].result = Some(Scalar::F64.into());
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
