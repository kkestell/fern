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
fn core_qbe_snapshot() {
    insta::assert_snapshot!(
        "core_qbe",
        emitted(
            "extern fn putchar(c:i32)->i32;fn main()->void {var x:u8=255;x+=1;putchar(i32(x));}"
        )
    );
}
