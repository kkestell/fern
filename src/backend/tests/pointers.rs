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

#[test]
fn native_specification_pointer_examples_exit_as_specified() {
    for (source, status) in [
        (
            "type Point struct { x: int, y: int }
fn main() -> void {
    var p = new(Point { x = 1, y = 2 });
    var n = new(0);
    p.x = 5;
    exit(p.x);
}",
            5,
        ),
        (
            "fn main() -> void {
    var p = new([10, 20, 30]);
    p[1] = 25;
    var copy = *p;
    copy[0] = 0;
    exit(p[0] + p[1]);
}",
            35,
        ),
        (
            "type Node struct {
    value: int,
    next: nullable *Node,
}

fn sum(list: nullable *Node) -> int {
    var total = 0;
    var cursor = list;
    for cursor != null {
        const node = *Node(cursor);
        total += node.value;
        cursor = node.next;
    }
    return total;
}

fn main() -> void {
    const tail = new(Node { value = 2, next = null });
    const head = new(Node { value = 1, next = tail });
    exit(sum(head));
}",
            3,
        ),
        (
            "fn main() -> void {
    var a = new(1);
    var b = new(1);
    const c = a;
    if a == b { exit(1); }
    if a == c { exit(0); }
    exit(2);
}",
            0,
        ),
    ] {
        assert_eq!(native_program_status(source), Some(status), "{source}");
    }
}

#[test]
fn native_free_releases_targets_and_ignores_null() {
    assert_eq!(
        native_status("const p = new(1); free(p); var q: nullable *int; free(q); free(q,);"),
        Some(0)
    );
}

#[test]
fn native_pointer_conversion_traps_on_null() {
    assert_native_failure(
        "var q: nullable *int; const p = *int(q); exit(*p);",
        "null pointer",
    );
    assert_eq!(
        native_status("var q: nullable *int = new(7); const p = *int(q); exit(*p);"),
        Some(7)
    );
}

#[test]
fn native_new_stores_its_operand_before_it_is_read() {
    let status = native_program_status(
        "type Pair struct { first: int, second: int }
fn seven() -> int { return 7; }
fn main() -> void {
    const p = new(Pair { first = seven(), second = 2 });
    exit(p.first * 10 + p.second);
}",
    );
    assert_eq!(status, Some(72));
}

#[test]
fn heap_operations_call_the_c_library() {
    let qbe = emit(
        &lowered("fn main() -> void { const p = new(1); free(p); }"),
        None,
    );
    for expected in ["call $malloc(l 8)", "b \"out of memory\"", "call $free(l "] {
        assert!(qbe.contains(expected), "{expected} missing from:\n{qbe}");
    }
}

#[test]
fn native_allocation_failure_traps_with_out_of_memory() {
    // No portable program exhausts memory, so the allocation calls a
    // stand-in that always fails, which runs the real trap path.
    let qbe = emit(
        &lowered("fn main() -> void { const p = new(1); exit(*p); }"),
        None,
    )
    .replace("call $malloc(", "call $fern_test_failing_malloc(")
        + "function l $fern_test_failing_malloc(l %size) {\n@start\n    ret 0\n}\n";
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    let result = Command::new(output).output().unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("out of memory"), "{stderr}");
}
