use crate::{frontend::parser, ir::lower::lower, semantic::namespaces, source::SourceMap};
fn emitted(source: &str) -> String {
    let syntax = parser::parse(&SourceMap::from_text(source)).unwrap();
    let p = lower(namespaces::check_root(&syntax).unwrap())
        .verify()
        .unwrap();
    super::emitter::emit(&p)
}
#[test]
fn emits_qbe_machine_operations() {
    let qbe = emitted(
        "extern fn putchar(c:i32)->i32;fn main()->void {var x:u8=255;x+=1;putchar(i32(x));}",
    );
    assert!(qbe.contains("export function w $main"));
    assert!(qbe.contains("call $putchar(w"));
    assert!(qbe.contains("storeb"));
    assert!(qbe.contains("extub"));
    assert!(!qbe.contains("abort"));
    assert!(!qbe.contains("pthread"));
}
#[test]
fn external_symbols_cannot_collide_with_generated_names() {
    let qbe = emitted("extern fn global0()->i32;var n=1;fn main()->void {global0();}");
    assert!(qbe.contains("data $.fern.global0"));
    assert!(qbe.contains("call $global0("));
}
#[test]
fn external_structs_use_qbe_aggregate_types() {
    let qbe = emitted(
        "type Outer struct {flag:u8,inner:[2][3]Inner,p:*Outer} type Inner struct {x:f32,n:[3]i16}
         extern fn f(o:Outer)->Inner;fn main()->void {var i=Inner {x=1.0,n=[1,2,3]};
         var o=Outer {flag=1,inner=[[i,i,i],[i,i,i]],p=null};var r=f(o);}",
    );
    let inner = qbe.find("type :fern.struct1 = { s, h 3 }").unwrap();
    let outer = qbe
        .find("type :fern.struct0 = { b, :fern.struct1 6, l }")
        .unwrap();
    assert!(inner < outer);
    assert!(qbe.contains("=:fern.struct1 call $f(:fern.struct0 "));
}

#[test]
fn core_qbe_snapshot() {
    insta::assert_snapshot!(
        "core_qbe",
        emitted(
            "extern fn putchar(c:i32)->i32;fn main()->void {var x:u8=255;x+=1;putchar(i32(x));}"
        )
    );
}
