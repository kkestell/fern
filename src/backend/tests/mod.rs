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
fn variadic_calls_mark_their_extra_arguments() {
    let qbe = emitted("extern fn f(x:i32,...)->void;fn main()->void {var b:i8=1;f(1,b,2.5);f(2);}");
    assert!(qbe.contains("call $f(w 1, ..., sb %"));
    assert!(qbe.contains("call $f(w 2, ...)"));
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

#[test]
fn small_aggregates_copy_field_by_field() {
    let qbe = emitted(
        "type V struct {x:f64,y:f64,z:f64} type B struct {b:[16]u8} type L struct {a:[9]f64}
         fn v(p:V)->f64 {return p.y;} fn b(p:B)->u8 {return p.b[0];} fn l(p:L)->f64 {return p.a[0];}
         fn call(pv:*V,pb:*B,pl:*L)->void {var x=v(*pv);var y=b(*pb);var z=l(*pl);}
         fn main()->void {}",
    );
    let function = |name: &str| {
        let start = qbe.find(&format!("${name}(")).unwrap();
        &qbe[start..start + qbe[start..].find("\n}").unwrap()]
    };
    let small = function(".fern.fn0");
    assert!(small.contains("=d loadd %access0_from\n    stored %access0_value, %access0_to"));
    assert!(small.contains("%access2_from =l add %param0, 16"));
    assert!(!small.contains("blit"));
    assert!(function(".fern.fn1").contains("blit %param0, %local0, 16"));
    assert!(function(".fern.fn2").contains("call $memcpy(l %local0, l %param0, l 72)"));
}
