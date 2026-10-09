use super::*;

#[test]
fn native_pointers_read_and_write_their_targets() {
    let status = native_program_status(
        "type S struct { x: int, a: [3]int }
type Holder struct { p: *int }
var counter = 10;
fn set(p: *S, value: int) -> void { p.x = value; }
fn pick(a: *S, b: *S, first: bool) -> *S { if first { return a; } return b; }
fn main() -> void {
    var x = 1;
    const p = &x;
    *p = 2;
    *p += 3;
    var s = S { x = 0, a = [0, 0, 0] };
    const ps = &s;
    ps.x = 4;
    ps.a[1] = 6;
    var vps = ps;
    const pps = &vps;
    pps.x += 1;
    var t = S { x = 0, a = [0, 0, 0] };
    set(&t, 7);
    pick(&s, &t, false).a[2] = 8;
    var row = [1, 2, 3];
    const pe = &row[2];
    *pe = 9;
    const pc = &counter;
    *pc += 20;
    var h = Holder { p = &x };
    *h.p += 1;
    var pointers = [&x, &row[0]];
    *pointers[1] = 11;
    exit(x + s.x + s.a[1] + t.x + t.a[2] + row[2] + counter + row[0]);
}",
    );
    // x = 6, s.x = 5, s.a[1] = 6, t.x = 7, t.a[2] = 8, row[2] = 9,
    // counter = 30, and row[0] = 11.
    assert_eq!(status, Some(82));
}

#[test]
fn native_indexing_through_a_pointer_checks_its_bounds() {
    assert_native_failure(
        "var a = [1, 2, 3]; const q = &a; var i = 3; exit(q[i]);",
        "array index out of range for `[3]int`",
    );
}

#[test]
fn native_pointers_compare_by_target() {
    let status = native_program_status(
        "type Holder struct { p: nullable *int }
fn main() -> void {
    var x = 1;
    var y = 1;
    const p = &x;
    var q: nullable *int = p;
    var z: nullable *int;
    var status = 0;
    if &x == &x { status += 1; }
    if &x != &y { status += 2; }
    if p == q { status += 4; }
    if q != null { status += 8; }
    if z == null { status += 16; }
    const a = Holder { p = p };
    const b = Holder { p = q };
    const c = Holder { p = null };
    if a == b { status += 32; }
    if a != c { status += 64; }
    exit(status);
}",
    );
    assert_eq!(status, Some(127));
}

#[test]
fn native_static_data_holds_null_as_a_zero_word() {
    let source = "type Node struct { value: int, next: nullable *Node }
var n = Node { value = 1, next = null };
fn main() -> void {
    if n.next == null { exit(n.value + 1); }
    exit(0);
}";
    let qbe = emit(&lowered(source), None);
    assert!(qbe.contains("data $global0 = { l 1, l 0 }"), "{qbe}");
    assert_eq!(native_program_status(source), Some(2));
}

#[test]
fn native_arguments_a_pointer_can_reach_are_copied() {
    let status = native_program_status(
        "fn f(a: [2]int, p: *[2]int) -> int { p[0] = 9; return a[0]; }
fn main() -> void { var arr = [1, 2]; exit(f(arr, &arr)); }",
    );
    assert_eq!(status, Some(1));
}

#[test]
fn pointer_signatures_pass_words() {
    let source = "type S struct { x: int }
fn pass(p: *S) -> *S { return p; }
fn main() -> void { var s = S { x = 3 }; exit(pass(&s).x); }";
    let qbe = emit(&lowered(source), None);
    assert!(qbe.contains("function l $fn0(l %param0) {"), "{qbe}");
    assert_eq!(native_program_status(source), Some(3));
}
