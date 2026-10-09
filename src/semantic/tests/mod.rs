use super::model::Constant;
use crate::{
    frontend::parser,
    semantic::namespaces::check_root,
    source::SourceMap,
    types::{Float, Scalar},
};
fn constants(body: &str) -> Vec<Constant> {
    let source = SourceMap::from_text(&format!("fn main()->void {{{body}}}"));
    let syntax = parser::parse(&source).unwrap();
    check_root(&syntax)
        .unwrap()
        .bindings
        .iter()
        .filter_map(|(_, b)| b.constant.clone())
        .collect()
}
#[test]
fn concrete_integer_folding() {
    assert_eq!(
        constants(
            "const a:u8=255+1; const b:i8=127+1; const c:u8=-1;const d:i8=-128;const e=i8(511);"
        ),
        [0, -128, 255, -128, -1].map(|n| Constant::Integer(n.into()))
    );
}
#[test]
fn ieee_folding() {
    let values = constants(
        "const a:f32=16777216.0+1.0;const b:f32=1.0/0.0;const c:f64=0.0/0.0;const d:f64=-0.0;",
    );
    assert_eq!(
        values[0],
        Constant::Float(Float::Binary32(16777216f32.to_bits()))
    );
    assert_eq!(
        values[1],
        Constant::Float(Float::Binary32(f32::INFINITY.to_bits()))
    );
    assert!(matches!(values[2],Constant::Float(Float::Binary64(n)) if f64::from_bits(n).is_nan()));
    assert_eq!(
        values[3],
        Constant::Float(Float::Binary64((-0f64).to_bits()))
    );
}
#[test]
fn invalid_core_programs() {
    for body in [
        "const x=1/0;",
        "const x=1<<64;",
        "const x=i8(128)/i8(-1);",
        "const x=i8(1e30);",
        "var x:u8=256;",
        "var x:i8=1;var y:u8=1;var z=x+y;",
        "const p=&x;var x=1;",
        "var x=1;const y=x;",
        "var a=[1,u8(2)];",
        "var x=true<false;",
        "break;",
        "var a=[1,2];const n=len(a);",
        "var a=[1];var x=a[2];",
    ] {
        let source = SourceMap::from_text(&format!("fn main()->void {{{body}}}"));
        let syntax = parser::parse(&source).unwrap();
        assert!(check_root(&syntax).is_err(), "{body}");
    }
}
#[test]
fn external_signatures_and_returns() {
    for source in [
        "extern fn foo(x:bool)->void;fn main()->void {}",
        "extern fn foo()->bool;fn main()->void {}",
        "extern fn foo()->[2]int;fn main()->void {}",
        "type S struct {b:bool} extern fn foo(s:S)->void;fn main()->void {}",
        "type S struct {a:[2]bool} extern fn foo()->S;fn main()->void {}",
        "type I struct {b:bool} type S struct {i:I} extern fn foo(s:S)->void;fn main()->void {}",
        "extern fn main()->void;",
        "fn f()->int {for true {}} fn main()->void {}",
        "fn f()->int {for {break;}} fn main()->void {}",
    ] {
        let syntax = parser::parse(&SourceMap::from_text(source)).unwrap();
        assert!(check_root(&syntax).is_err(), "{source}");
    }
    let syntax = parser::parse(&SourceMap::from_text(
        "fn f()->int {for {continue;break;}} fn main()->void {}",
    ))
    .unwrap();
    check_root(&syntax).unwrap();
    let syntax = parser::parse(&SourceMap::from_text(
        "extern fn foo(x:f32,y:f64)->f64;fn main()->void {}",
    ))
    .unwrap();
    check_root(&syntax).unwrap();
    for call in [
        "f();",
        "f(1,f32(1.0));",
        "f(1,true);",
        "f(1,null);",
        "f(1,[1,2]);",
        "f(1,S {x=1});",
    ] {
        let source = format!(
            "type S struct {{x:int}} extern fn f(x:i32,...)->void;fn main()->void {{{call}}}"
        );
        let syntax = parser::parse(&SourceMap::from_text(&source)).unwrap();
        assert!(check_root(&syntax).is_err(), "{call}");
    }
    let syntax = parser::parse(&SourceMap::from_text(
        "extern fn f(x:i32,...)->void;fn main()->void {var x=1;f(1);f(1,2,2.5,\"s\",u8(3),&x);}",
    ))
    .unwrap();
    check_root(&syntax).unwrap();
    let syntax = parser::parse(&SourceMap::from_text(
        "type N struct {x:f32} type S struct {x:f32,a:[3]u8,p:*S,n:[2]N}
         extern fn foo(s:S)->S;fn main()->void {}",
    ))
    .unwrap();
    check_root(&syntax).unwrap();
    assert_eq!(Scalar::Int.width(), usize::BITS);
}

#[test]
fn sizeof_resolves_forward_array_lengths() {
    let source =
        SourceMap::from_text("const bytes=sizeof([count]u8);const count=3;fn main()->void {}");
    let syntax = parser::parse(&source).unwrap();
    let checked = check_root(&syntax).unwrap();
    assert!(checked.bindings.iter().any(
        |(_, b)| b.constant == Some(Constant::Integer(3.into())) && b.ty == Scalar::Uint.into()
    ));
}

#[test]
fn unreachable_break_after_continuing_branches_does_not_require_a_return() {
    let source = SourceMap::from_text(
        "fn f(x:bool)->int {for {if x {continue;} else {continue;} break;}}fn main()->void {}",
    );
    let syntax = parser::parse(&source).unwrap();
    check_root(&syntax).unwrap();
}
#[test]
fn character_literals_are_u8_integers() {
    assert_eq!(
        constants("const c='a';const d='a'+1;const e:i32='\\xFF';"),
        [97, 98, 255].map(|n| Constant::Integer(n.into()))
    );
    let source = SourceMap::from_text("fn main()->void {var c='a';var d='a'+1;var e='a'+1.0;}");
    let syntax = parser::parse(&source).unwrap();
    let checked = check_root(&syntax).unwrap();
    let types: Vec<_> = checked
        .bindings
        .iter()
        .map(|(_, b)| b.ty.to_string())
        .collect();
    assert_eq!(types, ["u8", "u8", "f64"]);
    assert_eq!(
        constants("const a=97=='a';const b='a'==97;const c='a'==97.0;"),
        [1, 1, 1].map(|n| Constant::Integer(n.into()))
    );
    let source = SourceMap::from_text("fn main()->void {var bad=300=='a';}");
    let syntax = parser::parse(&source).unwrap();
    assert!(check_root(&syntax).is_err());
}
#[test]
fn string_literals_are_constant_u8_pointers() {
    let source = SourceMap::from_text(
        "var m:*u8=\"hi\"; fn main()->void {const s=\"hi\";var t=s;var u:*u8=\"other\";}",
    );
    let syntax = parser::parse(&source).unwrap();
    let checked = check_root(&syntax).unwrap();
    assert_eq!(checked.strings, [b"hi".to_vec(), b"other".to_vec()]);
    for (_, binding) in checked.bindings.iter() {
        assert_eq!(binding.ty.to_string(), "*u8");
    }
}
#[test]
fn string_literal_addresses_are_not_compile_time_values() {
    for body in [
        "var x:i8='\\xFF';",
        "var p:*i8=\"x\";",
        "var p=&\"x\";",
        "const b=\"a\"==\"a\";",
        "const p=*i8(\"a\");",
        "const c=*\"a\";",
        "const q=\"ab\"+1;",
        "var n=len(\"a\");",
    ] {
        let source = SourceMap::from_text(&format!("fn main()->void {{{body}}}"));
        let syntax = parser::parse(&source).unwrap();
        assert!(check_root(&syntax).is_err(), "{body}");
    }
}
