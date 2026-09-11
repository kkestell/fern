use super::*;

#[test]
fn native_slices_cover_array_ranges_access_and_lengths() {
    assert_eq!(
        native_status(
            "var values: [4]int = [1, 2, 3, 4];
             var first = values[1:3];
             var rest = values[2:];
             var prefix = values[:2];
             var whole = values[:];
             first[0] = 9;
             if values[1] != 9 || rest[1] != 4 || prefix[1] != 9 { exit(1); }
             if len(first) != 2 || len(rest) != 2 || len(prefix) != 2 || len(whole) != 4 { exit(2); }
             var nested = whole[1:3][0:1];
             if len(nested) != 1 || nested[0] != 9 { exit(3); }
             exit(42);"
        ),
        Some(42)
    );
}

#[test]
fn native_slice_copies_calls_results_and_aggregates_preserve_the_value() {
    assert_eq!(
        native_program_status(
            "type Holder struct { values: []int }
             fn echo(values: []int) -> []int { return values; }
             fn main() -> void {
                 var array: [3]int = [1, 2, 3];
                 var values = array[:];
                 var copied = values;
                 var returned = echo(copied);
                 var holder = Holder { values = returned };
                 var many: [2][]int = [holder.values...];
                 copied[1] = 9;
                 if values[1] != 9 || returned[1] != 9 || holder.values[1] != 9 || many[1][1] != 9 { exit(1); }
                 const immutable: []const int = returned;
                 if len(immutable) != 3 { exit(2); }
                 exit(42);
             }"
        ),
        Some(42)
    );
}

#[test]
fn native_slices_support_pointer_and_iteration_operands() {
    assert_eq!(
        native_status(
            "var array: [3]int = [2, 3, 4];
             var pointer = &array;
             var values = pointer[1:];
             var slice_pointer = &values;
             var total = 0;
             for value, index in *slice_pointer { total += value + index; }
             if total != 8 || len(slice_pointer) != 2 { exit(1); }
             slice_pointer[0] = 9;
             if array[1] != 9 { exit(2); }
             exit(42);"
        ),
        Some(42)
    );
}

#[test]
fn slice_layout_globals_and_traps_follow_the_shared_two_word_representation() {
    let source = "type Holder struct { before: int, values: []int }
                  var global: []int;
                  var holder: Holder;
                  var many: [2][]int;
                  fn take(values: []int) -> []int { return values; }
                  fn take_many(values: [2][]int) -> [2][]int { return values; }
                  fn main() -> void {
                      var values: [2]int = [1, 2];
                      var slice = take(values[:]);
                      if len(global) != 0 || len(holder.values) != 0 || len(many[1]) != 0 { exit(1); }
                      exit(slice[1]);
                  }";
    let qbe = emit(&lowered(source), None);
    assert!(qbe.contains("type :slice = { l, l }"), "{qbe}");
    assert!(qbe.contains("type :arrayslice2 = { :slice 2 }"), "{qbe}");
    assert!(qbe.contains("data $emptyslice = { l 0, l 0 }"), "{qbe}");
    assert!(qbe.contains("data $global0 = { l 0, l 0 }"), "{qbe}");
    assert_eq!(native_program_status(source), Some(2));

    for (body, expected) in [
        (
            "var values: [1]int = [1]; var slice = values[:]; exit(slice[1]);",
            "slice index out of range for `[]int`",
        ),
        (
            "var values: [1]int = [1]; var slice = values[:]; var index = 2; exit(slice[index]);",
            "slice index out of range for `[]int`",
        ),
        (
            "var values: [1]int = [1]; var slice = values[:]; var index = -1; exit(slice[index]);",
            "slice index out of range for `[]int`",
        ),
        (
            "var empty: []int; exit(empty[0]);",
            "slice index out of range for `[]int`",
        ),
        (
            "var values: [1]int = [1]; var low = 1; var high = 0; var slice = values[low:high]; exit(len(slice));",
            "slice bounds out of range for `[]int`",
        ),
        (
            "var values: [1]int = [1]; var low = -1; var high = 0; var slice = values[low:high]; exit(len(slice));",
            "slice bounds out of range for `[]int`",
        ),
        (
            "var values: [1]int = [1]; var low = 0; var high = 2; var slice = values[low:high]; exit(len(slice));",
            "slice bounds out of range for `[]int`",
        ),
        (
            "var pointer: *[1]int = null; var slice = pointer[:]; exit(len(slice));",
            "null pointer dereference",
        ),
    ] {
        assert_native_failure(body, expected);
    }

    let source = "fn main() -> void {
                      var values: [1]int = [1];
                      var slice = values[:];
                      exit(slice[1]);
                  }";
    let program = lowered(source);
    let sources = SourceMap::from_text(source);
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&emit(&program, Some(&sources)), &output).unwrap();
    let result = Command::new(output).output().unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(
        stderr.contains("slice index out of range for `[]int`"),
        "{stderr}"
    );
    assert!(stderr.contains("test.fern:"), "{stderr}");
}
