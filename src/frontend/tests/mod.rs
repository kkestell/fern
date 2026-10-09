use super::syntax::*;
use crate::{diagnostic::Diagnostic, source::SourceMap};
fn parse(text: &str) -> Result<Syntax, Diagnostic> {
    super::parser::parse(&SourceMap::from_text(text))
}
mod lexer;
#[test]
fn core_syntax() {
    let syntax=parse("extern fn malloc(n:uint)->*u8; type S struct { x:f32 } fn main()->void {var a:[2]u8=[1,2];for a[0]>0 {a[0]-=1;continue;} var n=sizeof(S);}").unwrap();
    assert!(syntax.functions.iter().next().unwrap().1.external);
    assert_eq!(syntax.structs.len(), 1);
    assert!(
        syntax
            .expressions
            .iter()
            .any(|(_, e)| matches!(e.kind, ExpressionKind::Sizeof { .. }))
    );
}
#[test]
fn removed_syntax_is_rejected() {
    for body in [
        "var x:int;",
        "var x:[3]int=[1...];",
        "var x=1+%2;",
        "var x=i8.truncate(1);",
        "for var i=0;i<2;i+=1 {}",
        "for x in [1] {}",
        "for:outer {break:outer;}",
        "var p:nullable *int=null;",
    ] {
        assert!(
            parse(&format!("fn main()->void {{{body}}}")).is_err(),
            "{body}"
        );
    }
}

#[test]
fn syntax_spans_and_snapshot() {
    let source = "use lib; type S struct { x:int } extern fn call(p:*int)->int; fn main()->void {var x=1;var p=&x;*p=call(p);var s=\"a\\tb\";var c='c';}";
    let syntax = parse(source).unwrap();
    let mut spans = Vec::new();
    for file in &syntax.files {
        for import in &file.imports {
            spans.push(import.span.clone());
        }
    }
    for (_, function) in syntax.functions.iter() {
        spans.push(function.span.clone());
    }
    for (_, declared) in syntax.structs.iter() {
        spans.push(declared.span.clone());
    }
    for (_, expression) in syntax.expressions.iter() {
        match &expression.kind {
            ExpressionKind::Reference(name) => spans.push(name.name_span.clone()),
            ExpressionKind::Call(call) => spans.push(call.left_paren_span.clone()),
            ExpressionKind::AddressOf { operator_span, .. }
            | ExpressionKind::Dereference { operator_span, .. } => {
                spans.push(operator_span.clone())
            }
            _ => {}
        }
    }
    for span in spans {
        assert!(!source[span].is_empty());
    }
    insta::assert_debug_snapshot!(
        "core_syntax",
        (syntax.statements, syntax.expressions, syntax.annotations)
    );
}

#[test]
fn sizeof_spans_cover_the_type_and_parentheses() {
    let source = "fn main()->void {var n=sizeof([2]u8);}";
    let syntax = parse(source).unwrap();
    let expression = syntax
        .expressions
        .iter()
        .find(|(_, e)| matches!(e.kind, ExpressionKind::Sizeof { .. }))
        .unwrap()
        .1;
    assert_eq!(&source[expression.span.clone()], "sizeof([2]u8)");
}
