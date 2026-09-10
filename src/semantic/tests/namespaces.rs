use super::*;

#[test]
fn public_declarations_are_reachable_through_both_import_forms() {
    accepts_tree([
        (
            "app/main.fern",
            "use counter;
                 fn main() -> void {
                     counter::value = counter::bump(counter::step);
                     exit(counter::value + doubled());
                 }",
        ),
        (
            "app/totals.fern",
            "use counter::{step, bump};
                 fn doubled() -> int { return bump(step) * step; }",
        ),
        ("counter/counter.fern", COUNTER),
    ]);
}

#[test]
fn a_private_declaration_is_module_wide_but_not_visible_to_importers() {
    // Another file of the same module reads the private `origin`.
    accepts_tree([
        (
            "app/main.fern",
            "use counter::{doubled_origin};
                 fn main() -> void { exit(doubled_origin); }",
        ),
        ("counter/counter.fern", COUNTER),
        (
            "counter/extra.fern",
            "pub const doubled_origin = origin * 2;",
        ),
    ]);
    rejects_root(
        "use counter; fn main() -> void { exit(«counter::origin»); }",
        "declaration `origin` is private to module `counter`",
    );
    rejects_root(
        "use counter::{«origin»}; fn main() -> void { exit(origin); }",
        "declaration `origin` is private to module `counter`",
    );
    rejects_root(
        "use counter; fn main() -> void { exit(«counter::missing»); }",
        "module `counter` has no declaration named `missing`",
    );
    rejects_root(
        "use counter::{«missing»}; fn main() -> void { exit(missing); }",
        "module `counter` has no declaration named `missing`",
    );
}

#[test]
fn imported_bindings_keep_their_mutability_across_modules() {
    accepts_tree([
        (
            "app/main.fern",
            "use counter::{value};
                 fn main() -> void { value = 7; exit(value); }",
        ),
        ("counter/counter.fern", COUNTER),
    ]);
    rejects_root(
        "use counter; fn main() -> void { «counter::step» = 1; exit(0); }",
        "cannot assign to immutable binding `step`",
    );
    rejects_root(
        "use counter::{step}; fn main() -> void { «step» = 1; exit(0); }",
        "cannot assign to immutable binding `step`",
    );
}

#[test]
fn a_module_level_initializer_reads_an_imported_constant() {
    accepts_tree([
        (
            "app/main.fern",
            "use counter::{step};
                 const total = step * 3;
                 fn main() -> void { exit(total); }",
        ),
        ("counter/counter.fern", COUNTER),
    ]);
    accepts_tree([
        (
            "app/main.fern",
            "use counter;
                 const total = counter::step * 3;
                 fn main() -> void { exit(total); }",
        ),
        ("counter/counter.fern", COUNTER),
    ]);
}

#[test]
fn an_introduced_name_must_not_collide_in_its_file() {
    for (marked, name) in [
        (
            "use counter::{«step»}; const step = 1; fn main() -> void { exit(step); }",
            "step",
        ),
        (
            "use counter::{«bump»}; fn bump() -> void {} fn main() -> void { bump(); }",
            "bump",
        ),
    ] {
        rejects_root(
            marked,
            &format!("imported name `{name}` conflicts with a module-level declaration"),
        );
    }
    rejects_root(
        "use counter::{step}; use counter::{«step»}; fn main() -> void { exit(step); }",
        "duplicate imported name `step`",
    );
    rejects_root(
        "use counter; use «counter»; fn main() -> void { exit(counter::step); }",
        "duplicate imported name `counter`",
    );
}

#[test]
fn an_introduced_name_must_be_referenced_in_its_own_file() {
    rejects_root(
        "use «counter»; fn main() -> void { exit(0); }",
        "imported name `counter` is never referenced",
    );
    rejects_root(
        "use counter::{«step»}; fn main() -> void { exit(0); }",
        "imported name `step` is never referenced",
    );
    // An import is file-local, so the other file neither sees it nor keeps
    // it referenced.
    rejects_tree(
        [
            (
                "app/main.fern",
                "use counter::{step}; fn main() -> void { exit(0); }",
            ),
            (
                "app/totals.fern",
                "fn tripled() -> int { return step * 3; }",
            ),
            ("counter/counter.fern", COUNTER),
        ],
        "unknown binding `step`",
    );
    rejects_tree(
        [
            (
                "app/main.fern",
                "use counter::{step}; fn main() -> void { exit(0); }",
            ),
            (
                "app/totals.fern",
                "use counter::{step}; fn tripled() -> int { return step * 3; }",
            ),
            ("counter/counter.fern", COUNTER),
        ],
        "imported name `step` is never referenced",
    );
}

#[test]
fn a_local_binding_shadows_an_imported_name() {
    accepts_tree([
        (
            "app/main.fern",
            "use counter::{step};
                 fn main() -> void {
                     const outer = step;
                     { var step = 5; step = 7; }
                     exit(outer);
                 }",
        ),
        ("counter/counter.fern", COUNTER),
    ]);
    rejects_root(
        "use counter; fn main() -> void { const counter = 1; exit(«counter»::step); }",
        "cannot use binding `counter` as a module",
    );
}

#[test]
fn a_qualified_name_resolves_only_through_an_imported_module() {
    rejects_root(
        "use counter; fn main() -> void { exit(counter::step + «text»::width); }",
        "unknown module `text`",
    );
    rejects_root(
        "use counter::{step}; fn main() -> void { exit(«step»::inner); }",
        "`step` is not a module",
    );
    rejects_root(
        "use counter; fn main() -> void { exit(«counter»); }",
        "module `counter` is not a value",
    );
    rejects_root(
        "use counter; fn main() -> void { «counter»(); }",
        "module `counter` is not a function",
    );
    rejects_root(
        "use counter; fn main() -> void { const f = «counter::bump»; exit(f); }",
        "cannot use function `bump` as a value",
    );
    rejects_root(
        "use counter; fn main() -> void { «counter::value»(); }",
        "cannot call non-function declaration `value`",
    );
    rejects_root(
        "use counter::{value}; fn main() -> void { «value»(); }",
        "cannot call non-function declaration `value`",
    );
}

#[test]
fn only_the_root_modules_main_is_the_entry_point() {
    let (_dir, program) = load_tree([
        (
            "app/main.fern",
            "use counter; fn main() -> void { exit(counter::bump(1)); }",
        ),
        (
            "counter/counter.fern",
            "pub var value = 0;
                 pub fn bump(amount: int) -> int { value = value + amount; return value; }
                 fn main(flag: int) -> int { return flag; }",
        ),
    ]);
    let checked = check(&program.syntax, &program.modules, &program.imports).unwrap();
    let root = program.modules.last().unwrap();
    let entry = program.syntax.files[root.files.start].items[0];
    assert!(matches!(
        entry,
        TopLevelItem::Function { function, .. } if function == checked.main
    ));

    rejects_tree(
        [
            (
                "app/main.fern",
                "use counter; fn helper() -> int { return counter::value; }",
            ),
            (
                "counter/counter.fern",
                "pub var value = 1; fn main() -> void {}",
            ),
        ],
        "missing `main` function",
    );
}

#[test]
fn module_bindings_resolve_forward_references_and_enclose_every_function() {
    let syntax = parse(
        "var counter = base;
             fn helper() -> void { const saved = counter; }
             const base: int = 40;
             fn main() -> void {
                 counter = counter + 2;
                 { const counter: u8 = 1; }
             }",
    )
    .unwrap();
    let checked = check_root(&syntax).unwrap();
    let module_statements: Vec<_> = syntax
        .files
        .iter()
        .flat_map(|file| &file.items)
        .filter_map(|item| match item {
            TopLevelItem::Binding { binding, .. } => Some(*binding),
            TopLevelItem::Function { .. } => None,
        })
        .collect();
    let counter = checked.declarations[module_statements[0]];
    let base = checked.declarations[module_statements[1]];
    assert!(checked.bindings[counter].mutable);
    assert_eq!(checked.bindings[counter].ty, value_type(Scalar::Int));
    assert_eq!(checked.bindings[counter].constant, None);
    assert_eq!(checked.bindings[base].constant, folded(40));
    assert!(
        checked
            .assignments
            .iter()
            .any(|(_, target)| target.binding == counter)
    );
    assert_eq!(checked.bindings.len(), 4);
}

#[test]
fn module_binding_failures_identify_the_declaration_contract() {
    for (text, offending, message) in [
        (
            "var value = 1; const value = 2; fn main() -> void {}",
            "value = 2",
            "duplicate module-level name `value`",
        ),
        (
            "const value = value; fn main() -> void {}",
            "value;",
            "module-level initializer cycle involving `value`",
        ),
        (
            "const first = second; const second = third; const third = first; fn main() -> void {}",
            "first;",
            "module-level initializer cycle involving `first`",
        ),
        (
            "var runtime = 1; const invalid = runtime; fn main() -> void {}",
            "runtime;",
            "module-level initializer must be a constant expression",
        ),
        (
            "const value = missing; fn main() -> void {}",
            "missing",
            "unknown binding `missing`",
        ),
    ] {
        let syntax = parse(text).unwrap();
        let error = check_root(&syntax).unwrap_err();
        let start = text.rfind(offending).unwrap();
        let expected_len = offending
            .trim_end_matches(';')
            .split(' ')
            .next()
            .unwrap()
            .len();
        assert_eq!(error.span, start..start + expected_len, "{text}");
        assert_eq!(error.message, message, "{text}");
    }
}

#[test]
fn module_dependency_ordering_does_not_recurse_on_the_compiler_stack() {
    let mut source = String::new();
    for index in 0..10_000 {
        source.push_str(&format!("const value{index} = value{};\n", index + 1));
    }
    source.push_str("const value10000 = 1; fn main() -> void {}");
    check_root(&parse(&source).unwrap()).unwrap();
}

#[test]
fn entry_checks_keep_their_spans_and_function_bodies_are_all_checked() {
    for (text, span, message) in [
        ("/* 🌿 */", 0..0, "missing `main` function"),
        (
            "fn main() -> void {} fn main() -> void {}",
            24..28,
            "duplicate `main` function",
        ),
    ] {
        let syntax = parse(text).unwrap();
        let error = check_root(&syntax).unwrap_err();
        assert_eq!(error.span, span);
        assert_eq!(error.message, message);
    }
    for (text, name) in [
        (
            "fn helper() -> void {} fn helper() -> void {} fn main() -> void {}",
            "helper",
        ),
        ("const main = 1; fn main() -> void {}", "main"),
    ] {
        let syntax = parse(text).unwrap();
        let error = check_root(&syntax).unwrap_err();
        let start = text.rfind(name).unwrap();
        assert_eq!(error.span, start..start + name.len());
        assert_eq!(
            error.message,
            format!("duplicate module-level name `{name}`")
        );
    }
    let text = "fn helper() -> void { exit(missing); } fn main() -> void {}";
    let syntax = parse(text).unwrap();
    let error = check_root(&syntax).unwrap_err();
    let start = text.find("missing").unwrap();
    assert_eq!(error.span, start..start + "missing".len());
    assert_eq!(error.message, "unknown binding `missing`");
}
