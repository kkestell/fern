use super::*;

/// A struct whose fields need padding, an array of it, a struct containing
/// both, and an array of scalars, so one program exercises every aggregate
/// class the layout spells.
const SHAPES: &str = "type Inner struct { flag: u8, count: i64 }
     type Outer struct { head: Inner, tail: [2]Inner, weight: f64 }
     fn pick(rows: [2]Outer, index: int) -> Inner { return rows[index].tail[1]; }
     fn total(values: [3]int) -> int { return values[0] + values[2]; }
     fn build() -> Outer { return Outer { ... }; }
";

#[test]
fn struct_signatures_pass_addresses_rather_than_aggregate_types() {
    let qbe = emit(
        &lowered(&format!(
            "{SHAPES}
             fn main() -> void {{
                 var rows: [2]Outer = [build()...];
                 var values: [3]int = [1, 2, 3];
                 exit(total(values) + int(pick(rows, 0).flag));
             }}"
        )),
        None,
    );
    assert!(!qbe.contains("type "), "{qbe}");
    for signature in [
        "function $fn0(l %result, l %param0, l %param1) {",
        "function l $fn1(l %param0) {",
        "function $fn2(l %result) {",
    ] {
        assert!(qbe.contains(signature), "{signature} missing from:\n{qbe}");
    }

    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(4));
}

#[test]
fn a_field_place_adds_its_layout_offset_to_its_base() {
    let qbe = emit(
        &lowered(
            "type Inner struct { flag: u8, count: i64 }
             type Outer struct { head: Inner, tail: [2]Inner, weight: f64 }
             var rows: [2]Outer = [Outer { ... }...];
             fn main() -> void {
                 var index = 1;
                 rows[index].tail[1].count = 7;
                 exit(int(rows[0].head.flag));
             }",
        ),
        None,
    );
    for expected in [
        // `count` follows a word-sized field, so it starts eight bytes in, and
        // `tail` follows the sixteen-byte `head`.
        "    %access1_address =l add %access0_address, 16",
        "    %access2_offset =l mul 1, 16",
        "    %access3_address =l add %access2_address, 8",
        // Reading through a constant index reaches `head.flag` at offset zero.
        "    %access6_address =l add %access5_address, 0",
    ] {
        assert!(qbe.contains(expected), "{expected} missing from:\n{qbe}");
    }
}

#[test]
fn a_struct_global_holds_its_scalars_at_their_layout_offsets() {
    let qbe = emit(
        &lowered(
            "type Gapped struct { flag: u8, count: i64 }
             type Trailing struct { count: i64, flag: u8 }
             var gapped: Gapped = Gapped { flag = 1, count = 2 };
             var trailing: [2]Trailing = [Trailing { count = 3, flag = 4 }...];
             var zeroed: Trailing = Trailing { ... };
             fn main() -> void { exit(int(gapped.flag) + int(trailing[1].flag)); }",
        ),
        None,
    );
    for expected in [
        "data $global0 = { w 1, z 4, l 2 }",
        "data $global1 = { l 3, w 4, z 4, l 3, w 4, z 4 }",
        "data $global2 = { z 16 }",
    ] {
        assert!(qbe.contains(expected), "{expected} missing from:\n{qbe}");
    }

    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(5));
}

#[test]
fn native_struct_values_copy_independently() {
    assert_eq!(
        native_program_status(
            "type Point struct { x: int, y: int }
             type Pair struct { left: Point, right: [2]Point }
             var origin: Point = Point { x = 1, y = 2 };
             fn shift(point: Point) -> Point {
                 var moved = point;
                 moved.x = moved.x + 10;
                 return moved;
             }
             fn main() -> void {
                 var local = origin;
                 local.x = 5;
                 // Copying a global left the global alone.
                 if origin.x != 1 { exit(1); }
                 var shifted = shift(local);
                 // The callee mutated its own copy of the argument.
                 if local.x != 5 { exit(2); }
                 if shifted.x != 15 { exit(3); }
                 var alias = shifted;
                 shifted.y = 99;
                 if alias.y != 2 { exit(4); }
                 // Assigning a whole struct through a global copies it.
                 origin = alias;
                 if origin.x != 15 || origin.y != 2 { exit(5); }
                 var pair = Pair { left = origin, right = [local...] };
                 pair.left.x = 0;
                 pair.right[1].y = 0;
                 if origin.x != 15 { exit(6); }
                 if pair.right[0].y != 2 { exit(7); }
                 exit(42);
             }"
        ),
        Some(42)
    );
}

#[test]
fn native_struct_equality_compares_every_nested_scalar() {
    assert_eq!(
        native_program_status(
            "type Inner struct { flag: u8, count: i64 }
             type Outer struct { head: Inner, tail: [2]Inner }
             fn main() -> void {
                 var left = Outer {
                     head = Inner { flag = 1, count = 2 },
                     tail = [Inner { flag = 3, count = 4 }, Inner { flag = 5, count = 6 }],
                 };
                 var right = left;
                 if left != right { exit(1); }
                 if !(left == right) { exit(2); }
                 // Each nested scalar decides equality on its own.
                 right.head.flag = 9;
                 if left == right { exit(3); }
                 right.head.flag = 1;
                 right.tail[1].count = 9;
                 if left == right { exit(4); }
                 if !(left != right) { exit(5); }
                 right.tail[1].count = 6;
                 if left != right { exit(6); }
                 // Padding holds no value, so it cannot decide equality.
                 var zeroed = Outer { ... };
                 var also = Outer { head = Inner { ... }, tail = [Inner { ... }...] };
                 if zeroed != also { exit(7); }
                 exit(42);
             }"
        ),
        Some(42)
    );
}

#[test]
fn native_floating_struct_equality_follows_ieee_754() {
    assert_eq!(
        native_program_status(
            "type Weighted struct { count: i64, weight: f64, narrow: [2]f32 }
             fn main() -> void {
                 var zero: f64 = 0.0;
                 var left = Weighted { count = 1, weight = zero, narrow = [0.5, 1.5] };
                 var right = left;
                 right.weight = -zero;
                 // The two zeroes are equal even though their bytes differ.
                 if left != right { exit(1); }
                 var nan = zero / zero;
                 right.weight = nan;
                 if left == right { exit(2); }
                 if !(left != right) { exit(3); }
                 // A NaN field equals nothing, not even itself.
                 left.weight = nan;
                 if left == right { exit(4); }
                 left.weight = zero;
                 right.weight = zero;
                 right.narrow[1] = 2.5;
                 if left == right { exit(5); }
                 exit(42);
             }"
        ),
        Some(42)
    );
    // Equal structs holding floating-point fields need not hold identical
    // bytes, so the comparison loads each field rather than calling `memcmp`.
    let qbe = emit(
        &lowered(
            "type Weighted struct { count: i64, weight: f64 }
             fn main() -> void {
                 var left = Weighted { count = 1, weight = 0.5 };
                 var right = left;
                 if left == right { exit(0); }
                 exit(1);
             }",
        ),
        None,
    );
    assert!(!qbe.contains("memcmp"), "{qbe}");
    assert!(qbe.contains(" =l loadl "), "{qbe}");
    assert!(qbe.contains(" =d loadd "), "{qbe}");
    assert!(qbe.contains(" =w ceqd "), "{qbe}");
}

#[test]
fn array_equality_emits_a_runtime_loop() {
    let qbe = emit(
        &lowered(
            "fn same(left: [100000]int, right: [100000]int) -> bool {
                 return left == right;
             }
             fn main() -> void { exit(0); }",
        ),
        None,
    );
    assert!(qbe.contains("@aggregate0_loop"), "{qbe}");
    assert!(qbe.contains("csltl %aggregate0_index, 100000"), "{qbe}");
    assert!(qbe.lines().count() < 100, "{qbe}");
}

#[test]
fn native_arguments_keep_their_values_while_the_call_runs() {
    assert_eq!(
        native_program_status(
            "type Point struct { x: int, y: int }
             var shared: Point = Point { x = 1, y = 2 };
             fn overwrite() -> int { shared = Point { x = 50, y = 60 }; return 0; }
             fn reread(point: Point) -> int {
                 shared.x = 9;
                 return point.x;
             }
             fn first(point: Point, unused: int) -> int { return point.x; }
             fn depth(point: Point, count: int) -> int {
                 if count == 0 { return point.y; }
                 return depth(point, count - 1);
             }
             fn main() -> void {
                 // The callee writing the global left its argument alone.
                 if reread(shared) != 1 { exit(1); }
                 if shared.x != 9 { exit(2); }
                 // A later argument writing the global left an earlier one
                 // alone.
                 if first(shared, overwrite()) != 9 { exit(3); }
                 if shared.x != 50 { exit(4); }
                 // The caller writing a local after the call cannot reach
                 // back into what the callee returned.
                 var local = Point { x = 3, y = 4 };
                 const seen = first(local, 0);
                 local.x = 30;
                 if seen != 3 { exit(5); }
                 // A parameter passed on through recursion keeps its value.
                 if depth(local, 5) != 4 { exit(6); }
                 exit(42);
             }"
        ),
        Some(42)
    );
}

#[test]
fn native_returned_calls_chain_their_results() {
    assert_eq!(
        native_program_status(
            "type Point struct { x: int, y: int }
             fn make(x: int, y: int) -> Point { return Point { x = x, y = y }; }
             fn swap(point: Point) -> Point { return make(point.y, point.x); }
             fn twice(point: Point) -> Point { return swap(swap(point)); }
             fn main() -> void {
                 const p = twice(make(40, 2));
                 if p.x != 40 || p.y != 2 { exit(1); }
                 exit(p.x + p.y);
             }"
        ),
        Some(42)
    );
}
