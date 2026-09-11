use super::*;

#[test]
fn annotations_resolve_to_array_types() {
    // An array annotation only resolves while its value is still rejected,
    // so each case reads the type back from a rejected initializer.
    for (source, expected) in [
        (
            "var a: [3]int = 1; fn main() -> void {}",
            array_type(3, value_type(Scalar::Int)),
        ),
        (
            "var a: [2][3]int = 1; fn main() -> void {}",
            array_type(2, array_type(3, value_type(Scalar::Int))),
        ),
        (
            "const n = 4; var a: [n]u8 = 1; fn main() -> void {}",
            array_type(4, value_type(Scalar::U8)),
        ),
        (
            "var a: [n]u8 = 1; const n = 4; fn main() -> void {}",
            array_type(4, value_type(Scalar::U8)),
        ),
        (
            "var a: [1 + 1]bool = 1; fn main() -> void {}",
            array_type(2, value_type(Scalar::Bool)),
        ),
    ] {
        let error = check_root(&parse(source).unwrap()).unwrap_err();
        assert_eq!(
            error.message,
            format!("cannot implicitly convert `int` to `{expected}`"),
            "{source}"
        );
    }
}

#[test]
fn slice_annotations_are_accepted_in_every_position() {
    for source in [
        "var values: []int; fn main() -> void {}",
        "fn main() -> void { var values: []int; }",
        "fn take(values: []int) -> void {} fn main() -> void {}",
        "fn make() -> []int { for {} } fn main() -> void {}",
        "type Node struct { children: []Node } fn main() -> void {}",
        "var rows: [3][]int; fn main() -> void {}",
        "var values: [][]int; fn main() -> void {}",
    ] {
        accepts_source(source);
    }
}

#[test]
fn slice_annotations_resolve_to_their_written_types() {
    assert_eq!(
        checked_bindings(
            "var mutable: []int;
             const immutable: []const int;
             var nested: [][]int;
             var pointers: []*const int;
             var slice_pointer: *[]int;
             var rows: [3][]int;
             type Point struct { x: int }
             var points: []Point;
             fn main() -> void {}"
        ),
        [
            (slice_type(false, value_type(Scalar::Int)), None),
            (
                slice_type(true, value_type(Scalar::Int)),
                Some(Constant::EmptySlice)
            ),
            (
                slice_type(false, slice_type(false, value_type(Scalar::Int))),
                None
            ),
            (
                slice_type(false, pointer_type(true, value_type(Scalar::Int))),
                None
            ),
            (
                pointer_type(false, slice_type(false, value_type(Scalar::Int))),
                None
            ),
            (
                array_type(3, slice_type(false, value_type(Scalar::Int))),
                None
            ),
            (slice_type(false, struct_type(0, "Point")), None),
        ]
    );
}

#[test]
fn slices_break_struct_containment_and_have_a_fixed_layout() {
    let syntax = parse(
        "type Node struct { children: []Node }
         fn main() -> void {}",
    )
    .unwrap();
    let checked = check_root(&syntax).unwrap();
    let node = struct_type(0, "Node");
    assert_eq!(
        struct_fields(&checked, 0),
        [("children".to_owned(), slice_type(false, node.clone()))]
    );
    assert_eq!(
        checked.layouts.size(&checked, &node),
        Some(2 * crate::layout::scalar_bytes(Scalar::Uint))
    );

    for source in [
        "type Node struct { child: Node } fn main() -> void {}",
        "type Node struct { children: [2]Node } fn main() -> void {}",
    ] {
        rejects_source(source, "Node", "recursive struct type `Node`");
    }
}

#[test]
fn slice_elements_retain_annotation_element_rules() {
    rejects_root(
        "var values: []«[_]int»; fn main() -> void {}",
        "`[_]` requires an array-literal initializer",
    );
}

#[test]
fn slice_layouts_count_as_two_pointer_width_words() {
    accepts_source("fn take(value: [70368744177664][]int) -> void {} fn main() -> void {}");
    rejects_root(
        "fn take(value: «[70368744177665][]int») -> void {} fn main() -> void {}",
        "aggregate layout exceeds compiler limit of 1 PiB",
    );
}

#[test]
fn an_underscore_length_comes_from_an_array_literal_initializer() {
    assert_eq!(
        checked_bindings("var a: [_]int = [1, 2, 3]; fn main() -> void {}")[0].0,
        array_type(3, value_type(Scalar::Int))
    );

    for marked in [
        "const a: «[_]int» = 1; fn main() -> void {}",
        "fn main() -> void { var a: «[_]int» = 1; }",
    ] {
        rejects_root(marked, "`[_]` requires an array-literal initializer");
    }
    rejects_root(
        "const a: «[_]int» = [0...]; fn main() -> void {}",
        "`[_]` cannot take a length from a literal with a fill",
    );
    rejects_root(
        "const a: [_]«[_]int» = [[1, 2]]; fn main() -> void {}",
        "`[_]` requires an array-literal initializer",
    );
    for marked in [
        "fn f(a: «[_]int») -> void {} fn main() -> void {}",
        "fn f() -> «[_]int» { return 1; } fn main() -> void {}",
    ] {
        rejects_root(marked, "`[_]` requires an array-literal initializer");
    }
}

#[test]
fn an_array_length_is_a_constant_int_of_at_least_one() {
    for (marked, message) in [
        (
            "var a: [«0»]int = 1; fn main() -> void {}",
            "array length must be at least 1",
        ),
        (
            "var a: [«-1»]int = 1; fn main() -> void {}",
            "array length must be at least 1",
        ),
        (
            "var n = 3; var a: [«n»]int = 1; fn main() -> void {}",
            "array length must be a constant expression",
        ),
        (
            "var a: [«true»]int = 1; fn main() -> void {}",
            "cannot implicitly convert `bool` to `int`",
        ),
        (
            "const n: [«n»]int = 1; fn main() -> void {}",
            "module-level initializer cycle involving `n`",
        ),
    ] {
        rejects_root(marked, message);
    }
    rejects_root(
        "fn main() -> void { var i = 0; var a: [«i»]int = 1; }",
        "array length must be a constant expression",
    );
    // A call is rejected before signatures are resolved, at module level
    // and in a function body alike.
    for marked in [
        "var a: [«size()»]int = 1; fn size() -> int { return 3; } fn main() -> void {}",
        "fn size() -> int { return 3; } fn main() -> void { var a: [«size()»]int = 1; }",
    ] {
        rejects_root(marked, "array length must be a constant expression");
    }
}

#[test]
fn aggregate_layouts_must_fit_the_target_address_space() {
    rejects_root(
        "fn consume(value: «[9223372036854775807]int») -> void {} fn main() -> void {}",
        "aggregate layout exceeds compiler limit of 1 PiB",
    );
    rejects_source(
        "type Huge struct { values: [9223372036854775807]int } fn main() -> void {}",
        "[9223372036854775807]int",
        "aggregate layout exceeds compiler limit of 1 PiB",
    );
}

#[test]
fn an_underscore_takes_its_length_from_the_literal_element_count() {
    for (source, expected) in [
        ("var a: [_]int = [1, 2, 3]; fn main() -> void {}", 3),
        (
            "var a: [_][2]int = [[1, 2], [3, 4]]; fn main() -> void {}",
            2,
        ),
        ("var a: [_]int = [7]; fn main() -> void {}", 1),
    ] {
        let syntax = parse(source).unwrap();
        let StatementKind::Binding { initializer, .. } =
            &syntax.statements[match syntax.files[0].items[0] {
                TopLevelItem::Binding { binding, .. } => binding,
                TopLevelItem::Function { .. } | TopLevelItem::Struct { .. } => {
                    unreachable!("the first item is a binding")
                }
            }]
            .kind
        else {
            unreachable!("the first item is a binding")
        };
        let length = inferred_length(&syntax, &(0..0), *initializer).unwrap();
        assert_eq!(length, expected, "{source}");
    }
}

#[test]
fn struct_fields_resolve_to_their_written_types() {
    let source = "type Inner struct { flag: bool }
type Outer struct {
    value: int,
    row: [2]u8,
    inner: Inner,
    grid: [2][3]Inner,
}
fn main() -> void {}";
    let syntax = parse(source).unwrap();
    let checked = check_root(&syntax).unwrap();
    let inner = struct_type(0, "Inner");
    assert_eq!(
        struct_fields(&checked, 1),
        [
            ("value".to_owned(), value_type(Scalar::Int)),
            ("row".to_owned(), array_type(2, value_type(Scalar::U8))),
            ("inner".to_owned(), inner.clone()),
            ("grid".to_owned(), array_type(2, array_type(3, inner))),
        ]
    );
}

#[test]
fn a_field_reaches_a_struct_declared_after_it() {
    let syntax = parse(
        "type Outer struct { inner: Inner }
         type Inner struct { flag: bool }
         fn main() -> void {}",
    )
    .unwrap();
    let checked = check_root(&syntax).unwrap();
    assert_eq!(
        struct_fields(&checked, 0),
        [("inner".to_owned(), struct_type(1, "Inner"))]
    );
}

#[test]
fn field_names_are_unique_within_their_struct() {
    rejects_source(
        "type Point struct { x: int, x: u8 } fn main() -> void {}",
        "x",
        "duplicate field name `x`",
    );
    // Two structs may each declare the same field name.
    accepts_source(
        "type Point struct { x: int }
         type Offset struct { x: u8 }
         fn main() -> void {}",
    );
}

#[test]
fn a_field_array_length_reads_a_module_constant_declared_later() {
    // The binding needs `Row`, whose field length needs `count`, so `count`
    // is checked first however the declarations are ordered.
    assert_eq!(
        checked_bindings(
            "const head = Row { cells = [0...] };
             type Row struct { cells: [count]int }
             const count = 3;
             fn main() -> void {}"
        )[0],
        (
            struct_type(0, "Row"),
            Some(Constant::Struct(vec![Constant::Array(vec![
                Constant::Integer(big(0)),
                Constant::Integer(big(0)),
                Constant::Integer(big(0)),
            ])]))
        )
    );
    // The same ordering holds when an annotation is what names the struct,
    // which the resolved type in this mismatch reports.
    rejects_source(
        "const head: Row = 0;
         type Row struct { cells: [count]int }
         const count = 3;
         fn main() -> void {}",
        "0",
        "cannot implicitly convert `int` to `Row`",
    );
}

#[test]
fn a_struct_must_not_contain_itself() {
    for source in [
        "type Node struct { next: Node } fn main() -> void {}",
        "type Node struct { children: [2]Node } fn main() -> void {}",
        "type Node struct { rows: [2][3]Node } fn main() -> void {}",
    ] {
        rejects_source(source, "Node", "recursive struct type `Node`");
    }
    rejects_source(
        "type A struct { b: B } type B struct { a: A } fn main() -> void {}",
        "A",
        "recursive struct type `A`",
    );
    // A cycle is rejected even when nothing names the struct.
    rejects_source(
        "type Point struct { x: int } type Node struct { next: Node } fn main() -> void {}",
        "Node",
        "recursive struct type `Node`",
    );
    // Two fields of one struct type are not a cycle.
    accepts_source(
        "type Point struct { x: int }
         type Segment struct { start: Point, end: Point }
         fn main() -> void {}",
    );
}

#[test]
fn deep_struct_containment_reports_a_compiler_limit() {
    let mut source = String::new();
    for index in 0..=128 {
        if index == 128 {
            source.push_str(&format!("type S{index} struct {{ value: int }}\n"));
        } else {
            source.push_str(&format!(
                "type S{index} struct {{ next: S{} }}\n",
                index + 1
            ));
        }
    }
    source.push_str("fn main() -> void {}\n");
    let error = check_root(&parse(&source).unwrap()).unwrap_err();
    assert_eq!(
        error.message,
        "struct containment exceeds compiler limit of 128"
    );
}

#[test]
fn a_binary_struct_tree_derives_each_layout_once() {
    // Every level names the level below it twice, so a layout derived per
    // reference would cost 2^40 derivations. Memoizing costs one per level.
    let mut source = String::new();
    for index in 0..40 {
        source.push_str(&format!(
            "type S{index} struct {{ left: S{0}, right: S{0} }}\n",
            index + 1
        ));
    }
    source.push_str("type S40 struct { value: u8 }\n");
    source.push_str("fn walk(node: S0) -> void {}\nfn main() -> void {}\n");
    let syntax = parse(&source).unwrap();
    let checked = check_root(&syntax).unwrap();
    assert_eq!(checked.structs.len(), 41);
}

#[test]
fn pointer_annotations_resolve_and_break_struct_containment_cycles() {
    assert_eq!(
        checked_bindings(
            "var mutable: *int;
             const immutable: *const [2]u8;
             var nested: [3]**int;
             fn main() -> void {}"
        ),
        [
            (pointer_type(false, value_type(Scalar::Int)), None),
            (
                pointer_type(true, array_type(2, value_type(Scalar::U8))),
                Some(Constant::Null),
            ),
            (
                array_type(
                    3,
                    pointer_type(false, pointer_type(false, value_type(Scalar::Int))),
                ),
                None,
            ),
        ]
    );

    let syntax = parse(
        "type Node struct { next: *Node }
         type Pair struct { left: *const Node, right: *[2]*Node }
         fn take(node: *Node) -> void {}
         fn main() -> void { var node: Node; }",
    )
    .unwrap();
    let checked = check_root(&syntax).unwrap();
    let node = struct_type(0, "Node");
    assert_eq!(
        struct_fields(&checked, 0),
        [("next".to_owned(), pointer_type(false, node.clone()))]
    );
    assert_eq!(
        struct_fields(&checked, 1),
        [
            ("left".to_owned(), pointer_type(true, node.clone())),
            (
                "right".to_owned(),
                pointer_type(false, array_type(2, pointer_type(false, node.clone()))),
            ),
        ]
    );
    assert_eq!(
        checked.layouts.size(&checked, &node),
        Some(crate::layout::scalar_bytes(Scalar::Uint))
    );
    let (_, take) = checked.functions.iter().next().unwrap();
    assert_eq!(
        checked.bindings[take.parameters[0]].ty,
        pointer_type(false, struct_type(0, "Node"))
    );
    rejects_source(
        "fn result() -> *int { return 0; } fn main() -> void {}",
        "0",
        "cannot implicitly convert `int` to `*int`",
    );
}
