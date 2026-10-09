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
        "extern fn foo(x:f64)->void;fn main()->void {}",
        "extern fn foo()->[2]int;fn main()->void {}",
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
