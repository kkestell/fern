use super::*;

#[test]
fn native_pointer_copies_addresses_and_conversions_preserve_aliasing() {
    assert_eq!(
        native_program_status(
            "type Node struct { value: int, next: *Node }
             var zero: *Node;
             fn echo(pointer: *Node) -> *Node { return pointer; }
             fn values(pointer: *[2]int) -> *[2]int { return pointer; }
             fn read(pointer: *const Node) -> int { return pointer.value; }
             fn main() -> void {
                 var first: Node;
                 first.value = 7;
                 first.next = &first;
                 var copied = first;
                 if copied != first { exit(1); }
                 var pointers: [2]*Node;
                 pointers[0] = &first;
                 pointers[1] = pointers[0];
                 const immutable: *const Node = pointers[0];
                 if immutable != pointers[1] || zero != null { exit(2); }
                 if uint(pointers[0]) == 0 || uint(zero) != 0 { exit(3); }
                 var returned = echo(pointers[1]);
                 returned.value = 9;
                 echo(returned).value += 1;
                 var array: [2]int = [1, 2];
                 values(&array)[0] += 2;
                 if first.value != 10 || read(immutable) != 10 || array[0] != 3 { exit(4); }
                 if copied == first { exit(5); }
                 exit(42);
             }"
        ),
        Some(42)
    );
}

#[test]
fn pointer_globals_and_recursive_aggregate_types_emit_as_word_storage() {
    let qbe = emit(
        &lowered(
            "type Node struct { value: int, next: *Node }
             var root: Node;
             var empty: *Node;
             fn main() -> void {
                 root.next = &root;
                 if root.next == empty { exit(1); }
                 exit(42);
             }",
        ),
        None,
    );
    assert!(!qbe.contains("type :struct"), "{qbe}");
    assert!(qbe.contains("data $global1 = { l 0 }"), "{qbe}");
    assert!(qbe.contains("storel"), "{qbe}");
    assert!(qbe.contains("loadl"), "{qbe}");
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn null_indirect_accesses_abort_with_the_reached_source_location() {
    for source in [
        "fn main() -> void { var pointer: *int = null; exit(*pointer); }",
        "fn main() -> void { var pointer: *int = null; const address = &*pointer; }",
        "fn main() -> void { var pointer: *[2]int = null; exit(pointer[0]); }",
        "fn main() -> void { var pointer: *[2]int = null; const address = &pointer[0]; }",
        "fn main() -> void { var pointer: *[2]int = null; exit(len(pointer)); }",
        "type Point struct { value: int }
         fn main() -> void { var pointer: *Point = null; exit(pointer.value); }",
        "type Point struct { value: int }
         fn main() -> void { var pointer: *Point = null; const address = &pointer.value; }",
    ] {
        let program = lowered(source);
        let sources = SourceMap::from_text(source);
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("program");
        build_text(&emit(&program, Some(&sources)), &output).unwrap();
        let result = Command::new(output).output().unwrap();
        assert!(!result.status.success(), "{source}");
        let stderr = String::from_utf8(result.stderr).unwrap();
        assert!(
            stderr.contains("null pointer dereference"),
            "{source}: {stderr}"
        );
        assert!(stderr.contains("test.fern:"), "{source}: {stderr}");
    }
}
