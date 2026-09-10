use super::*;

#[test]
fn array_signatures_declare_one_aggregate_type_for_each_layout() {
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
    let types: Vec<&str> = qbe
        .lines()
        .filter(|line| line.starts_with("type "))
        .collect();
    // `[3]int` is a parameter and a result, and both name one definition.
    assert_eq!(
        types,
        ["type :arrayl3 = { l 3 }", "type :arrayl6 = { l 6 }"]
    );
    let first_use = qbe
        .find(":arrayl3 %param")
        .expect("an aggregate parameter names its type");
    assert!(qbe.find("type :arrayl3").unwrap() < first_use, "{qbe}");
    assert!(qbe.contains("blit %param0, %local0, 24"), "{qbe}");

    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("program");
    build_text(&qbe, &output).unwrap();
    assert_eq!(Command::new(output).status().unwrap().code(), Some(42));

    let scalars = emit(&lowered("fn main() -> void { exit(42); }"), None);
    assert!(!scalars.contains("type "), "{scalars}");
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
