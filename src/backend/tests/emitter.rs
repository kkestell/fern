use super::*;

#[test]
fn aggregates_pass_and_return_through_addresses() {
    let program = lowered(
        "fn row(seed: int) -> [3]int { var out: [3]int = [seed...]; return out; }
             fn total(values: [3]int) -> int { return values[0] + values[2]; }
             fn grid() -> [2][3]int { var out: [2][3]int = [[1, 2, 3]...]; return out; }
             fn main() -> void {
                 if grid()[1][2] == 3 { exit(total(row(21))); }
                 exit(255);
             }",
    );
    let qbe = emit(&program, None);
    // QBE's aggregate types would copy an argument one word at a time, so no
    // signature names one. An aggregate result is written through the
    // address the caller passes first.
    assert!(!qbe.contains("type "), "{qbe}");
    assert!(
        qbe.contains("function $fn0(l %result, l %param0) {"),
        "{qbe}"
    );
    assert!(qbe.contains("function l $fn1(l %param0) {"), "{qbe}");
    assert!(qbe.contains("function $fn2(l %result) {"), "{qbe}");
    // A callee reads an aggregate parameter from the caller's storage.
    let total = function_text(&qbe, "$fn1");
    assert!(!total.contains("blit"), "{total}");
    assert!(!total.contains("%local0 ="), "{total}");

    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

/// The text of the function whose symbol is `symbol`, through its closing
/// brace.
fn function_text<'a>(qbe: &'a str, symbol: &str) -> &'a str {
    // A call to the function names its symbol too, so find its header.
    let header = qbe
        .lines()
        .find(|line| line.contains("function ") && line.contains(&format!(" {symbol}(")))
        .unwrap_or_else(|| panic!("{qbe}"));
    let start = qbe.find(header).unwrap();
    let end = start + qbe[start..].find("\n}\n").expect("a closing brace");
    &qbe[start..end]
}

#[test]
fn a_returned_call_writes_its_result_straight_into_the_callers_storage() {
    let program = lowered(
        "type Big struct { values: [10]int, last: int }
             fn build(seed: int) -> Big { return Big { values = [seed...], last = seed }; }
             fn relay(seed: int) -> Big { return build(seed); }
             fn chain(seed: int) -> Big { return (relay(seed + 1)); }
             fn main() -> void { exit(chain(41).last); }",
    );
    let qbe = emit(&program, None);
    for (symbol, callee) in [("$fn1", "$fn0"), ("$fn2", "$fn1")] {
        let text = function_text(&qbe, symbol);
        assert!(
            text.contains(&format!("call {callee}(l %result,")),
            "{text}"
        );
        assert!(!text.contains("blit"), "{text}");
        assert!(!text.contains("memcpy"), "{text}");
        assert!(!text.contains("_storage"), "{text}");
    }

    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));
}

#[test]
fn a_borrowed_arguments_bounds_check_runs_before_later_arguments() {
    // An out-of-range index traps, which aborts rather than exiting with a
    // status, so `quit` never runs.
    assert_eq!(
        native_program_status(
            "fn quit() -> int { exit(7); }
             fn first(row: [2]int, unused: int) -> int { return row[0]; }
             fn main() -> void {
                 var rows: [2][2]int;
                 var i = 2;
                 exit(first(rows[i], quit()));
             }"
        ),
        None
    );
}

#[test]
fn large_aggregates_copy_zero_and_fill_without_unrolling() {
    let program = lowered(
        "fn main() -> void {
                 var zeroed: [100]int;
                 var n = 3;
                 var filled: [100]int = [n...];
                 zeroed = filled;
                 exit(zeroed[99] + filled[0]);
             }",
    );
    let qbe = emit(&program, None);
    assert!(qbe.contains("call $memset(l %local0, w 0, l 800)"), "{qbe}");
    assert!(qbe.contains("call $memcpy("), "{qbe}");
    assert!(qbe.contains("@fill0_loop"), "{qbe}");
    assert!(!qbe.contains("blit"), "{qbe}");

    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(6));
}

#[test]
fn an_array_global_holds_one_data_item_for_each_element() {
    let program = lowered(
        "var flags: [2][2]u8 = [[1, 2], [3, 4]];
             fn main() -> void { exit(int(flags[1][1])); }",
    );
    let qbe = emit(&program, None);
    assert!(
        qbe.contains("data $global0 = { w 1, w 2, w 3, w 4 }"),
        "{qbe}"
    );

    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(4));
}

#[test]
fn every_allocation_is_emitted_in_the_entry_block() {
    // QBE allocates once per `alloc` it executes, so an `alloc` reached by
    // a loop would grow the stack on every iteration.
    let program = lowered(
        "fn build(seed: int) -> [3]int { var out: [3]int = [seed...]; return out; }
             fn main() -> void {
                 var source: [3]int = [1, 2, 3];
                 var total = 0;
                 for var i = 0; i < 3; i += 1 {
                     var copy = source;
                     total += copy[0] + build(i)[2];
                 }
                 exit(total);
             }",
    );
    let qbe = emit(&program, None);
    let mut block = "";
    let mut allocations = 0;
    for line in qbe.lines() {
        if let Some(label) = line.strip_prefix('@') {
            block = label;
        }
        if line.contains(" alloc") {
            allocations += 1;
            assert_eq!(block, "start", "{line}");
        }
    }
    assert!(allocations > 0, "{qbe}");
}

#[test]
fn out_of_range_indices_trap_and_name_the_array_type() {
    for (body, expected) in [
        (
            "var a: [3]int = [1, 2, 3]; var i = 3; exit(a[i]);",
            "[3]int",
        ),
        (
            "var a: [3]int = [1, 2, 3]; var i = 0; exit(a[i - 1]);",
            "[3]int",
        ),
        (
            "var g: [2][3]int = [[1, 2, 3], [4, 5, 6]]; var i = 2; exit(g[i][0]);",
            "[2][3]int",
        ),
        // `len` reads the length from the type, and still evaluates its
        // operand, including when an operator around it folds the result.
        (
            "var g: [2][3]int = [[1, 2, 3], [4, 5, 6]]; var i = 2; exit(len(g[i]));",
            "[2][3]int",
        ),
        (
            "var g: [2][3]int = [[1, 2, 3], [4, 5, 6]]; var i = 2; exit(len(g[i]) + 1);",
            "[2][3]int",
        ),
    ] {
        assert_native_failure(body, &format!("array index out of range for `{expected}`"));
    }
}

#[test]
fn two_bounds_checks_in_one_function_link_separately() {
    let program = lowered(
        "fn main() -> void {
                 var a: [3]int = [1, 2, 3];
                 var i = 1;
                 exit(a[i] + a[i + 1]);
             }",
    );
    let qbe = emit(&program, None);
    let symbols: Vec<&str> = data_symbols(&qbe)
        .into_iter()
        .filter(|symbol| symbol.contains("_bounds_"))
        .collect();
    assert_eq!(symbols.len(), 2, "{symbols:?}");
    assert_eq!(symbols.iter().collect::<BTreeSet<_>>().len(), 2);

    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(5));
}

#[test]
fn only_the_root_modules_main_is_exported() {
    let program = lowered_tree([
        (
            "app/main.fern",
            "use counter;\nfn main() -> void { exit(counter::bump()); }\n",
        ),
        (
            "counter/counter.fern",
            "pub fn bump() -> int { return 42; }\nfn main() -> void {}\n",
        ),
    ]);
    let qbe = emit(&program, None);
    let definitions = definitions(&qbe);
    assert_eq!(definitions.len(), program.program().functions.len());
    let exported: Vec<&str> = definitions
        .iter()
        .filter(|(_, exported)| *exported)
        .map(|(symbol, _)| *symbol)
        .collect();
    assert_eq!(exported, ["main"]);
    // A dependency's `main` is an ordinary function, so it takes an `fn`
    // symbol and collides with neither the entry point nor libc.
    for (symbol, _) in definitions.iter().filter(|(symbol, _)| *symbol != "main") {
        assert!(symbol.starts_with("fn"), "{symbol}");
    }
}

#[test]
fn same_named_declarations_in_two_modules_take_distinct_symbols() {
    let declarations = "pub var value = 0;\n\
             pub fn bump(amount: int) -> int { return value + amount; }\n";
    let program = lowered_tree([
        (
            "app/main.fern",
            "use first;\nuse second;\n\
                 fn main() -> void {\n\
                 first::value = 1;\n\
                 second::value = 2;\n\
                 exit(first::bump(0) + second::bump(0));\n\
                 }\n",
        ),
        ("first/first.fern", declarations),
        ("second/second.fern", declarations),
    ]);
    let qbe = emit(&program, None);

    // `main` and the two same-named `bump` functions.
    let definitions = definitions(&qbe);
    assert_eq!(definitions.len(), 3);
    let functions: BTreeSet<&str> = definitions.iter().map(|(symbol, _)| *symbol).collect();
    assert_eq!(functions.len(), definitions.len(), "{definitions:?}");

    let data = data_symbols(&qbe);
    assert_eq!(
        data.iter().collect::<BTreeSet<_>>().len(),
        data.len(),
        "{data:?}"
    );
    let globals: Vec<&str> = data
        .iter()
        .copied()
        .filter(|symbol| symbol.starts_with("global"))
        .collect();
    assert_eq!(globals.len(), 2, "{data:?}");

    // Each same-named `bump`, and `main`, owns its own trap messages.
    let owners: BTreeSet<&str> = data
        .iter()
        .filter(|symbol| symbol.ends_with("_message"))
        .map(|symbol| {
            symbol
                .split_once("_operation")
                .expect("a trap message symbol")
                .0
        })
        .collect();
    assert_eq!(owners.len(), 3, "{data:?}");

    // The two same-named `pub var`s are separate storage.
    let stores: BTreeSet<&str> = qbe
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("store"))
        .filter_map(|line| line.rsplit(", ").next())
        .filter(|target| target.starts_with('$'))
        .collect();
    assert_eq!(stores.len(), 2, "{qbe}");
}
