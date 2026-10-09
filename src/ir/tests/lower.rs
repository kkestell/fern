use super::*;

#[test]
fn array_globals_hold_their_elements_in_memory_order() {
    let program = lowered(
        "var row: [3]int = [1, 2, 3];
             var grid: [2][2]u8 = [[1, 2], [3, 4]];
             var count = 7;
             fn main() -> void { exit(count); }",
    );
    assert_eq!(
        program.program().globals,
        vec![
            Global {
                ty: array(3, Scalar::Int.into()),
                values: Some(integers([1, 2, 3], Scalar::Int)),
            },
            Global {
                ty: array(2, array(2, Scalar::U8.into())),
                values: Some(integers([1, 2, 3, 4], Scalar::U8)),
            },
            Global {
                ty: Scalar::Int.into(),
                values: Some(integers([7], Scalar::Int)),
            },
        ]
    );
}

#[test]
fn an_array_literal_stores_each_element_and_a_fill_repeats_the_last_one() {
    let program = lowered(
        "fn side() -> int { return 4; }
             fn main() -> void { var a: [4]int = [1, side()...]; exit(a[3]); }",
    );
    let main = main_of(&program);
    assert_eq!(
        stored_places(main),
        ["local0[0]", "local0[1]", "local0[2..]", "local1"]
    );
    // The fill evaluates its element once and repeats that value.
    assert_eq!(call_targets(main), [FunctionId(0)]);
    let filled = instructions(main)
        .find_map(|instruction| match instruction {
            Instruction::Fill { operand, .. } => Some(operand.clone()),
            _ => None,
        })
        .expect("the literal fills its remaining elements");
    assert_eq!(filled, stores(main)[1].1);
}

#[test]
fn a_nested_literal_stores_each_row_through_the_outer_element_place() {
    // A folded literal is static data, copied whole into the binding.
    let folded =
        lowered("fn main() -> void { var grid: [2][2]int = [[1, 2], [3, 4]]; exit(grid[0][0]); }");
    let main = main_of(&folded);
    assert_eq!(stored_places(main), ["local0"]);
    assert_eq!(loads(main)[0], Place::Constant(ConstantId(0)));

    // A constant row is copied in from static data, while a row that is not
    // constant is built on its own and copied in.
    let runtime = lowered(
        "fn side() -> int { return 4; }
             fn main() -> void {
                 var grid: [2][2]int = [[1, 2], [3, side()]];
                 exit(grid[0][0]);
             }",
    );
    assert_eq!(
        stored_places(main_of(&runtime)),
        ["local0[0]", "local1[0]", "local1[1]", "local0[1]", "local2"]
    );
}

#[test]
fn a_constant_array_is_read_from_static_data_at_each_use() {
    let program = lowered(
        "const a: [2]int = [10, 20];
             fn main() -> void { var i = 1; exit(a[i] + a[0]); }",
    );
    let main = main_of(&program);
    // Each use of the folded array reads its own static data, which nothing
    // stores into.
    assert_eq!(stored_places(main), ["local0"]);
    let loaded: Vec<String> = loads(main).iter().map(describe).collect();
    assert_eq!(loaded, ["local0", "constant0[v0]", "constant1[0]"]);
    assert_eq!(main.constants.len(), 2);
}

#[test]
fn assigning_a_whole_array_copies_it() {
    let program = lowered(
        "fn main() -> void {
                 var a: [3]int = [1, 2, 3];
                 var b = a;
                 b[0] = 99;
                 exit(a[0]);
             }",
    );
    let main = main_of(&program);
    assert_eq!(stored_places(main), ["local0", "local1", "local1[0]"]);
    // `b` is its own array, so writing an element of it leaves `a` alone.
    let copy = stores(main)[1].1.clone();
    assert_eq!(
        main.values[match copy {
            Operand::Value(ValueId(id)) => id,
            _ => panic!("a whole-array copy reads a place"),
        }]
        .kind,
        ValueKind::Load(Place::Local(LocalId(0)))
    );
}

#[test]
fn indexing_carries_each_index_span_for_its_bounds_check() {
    let text = "fn main() -> void {
             var grid: [2][2]int = [[1, 2], [3, 4]];
             var row = 1;
             exit(grid[row][0]);
         }";
    let program = lowered(text);
    let load = loads(main_of(&program))
        .into_iter()
        .find(|place| matches!(place, Place::Element { base, .. } if matches!(**base, Place::Element { .. })))
        .expect("`grid[row][0]` loads through two element places");
    let Place::Element {
        base, span: inner, ..
    } = &load
    else {
        unreachable!("the load reaches an element")
    };
    let Place::Element { span: outer, .. } = &**base else {
        unreachable!("its base reaches an element")
    };
    assert_eq!(&text[outer.clone()], "row");
    assert_eq!(&text[inner.clone()], "0");
}

#[test]
fn an_element_assignment_lowers_its_index_before_the_value() {
    let program = lowered(
        "fn side() -> int { return 1; }
             fn main() -> void {
                 var a: [2]int = [0, 0];
                 var i = 0;
                 a[i] = side();
                 exit(a[0]);
             }",
    );
    let main = main_of(&program);
    let instructions = &main.flow.blocks[0].instructions;
    let call = instructions
        .iter()
        .position(|instruction| matches!(instruction, Instruction::Call { .. }))
        .expect("the value is a call");
    let stored = instructions
        .iter()
        .position(|instruction| {
            matches!(
                instruction,
                Instruction::Store {
                    place: Place::Element { .. },
                    operand: Operand::Value(_),
                }
            )
        })
        .expect("the assignment stores through an element place");
    let Instruction::Store {
        place: Place::Element { index, .. },
        ..
    } = &instructions[stored]
    else {
        unreachable!("the store reaches an element")
    };
    let Operand::Value(ValueId(index)) = index else {
        unreachable!("the index is a runtime value")
    };
    let index = instructions
        .iter()
        .position(|instruction| *instruction == Instruction::Value(ValueId(*index)))
        .expect("the index is defined in the block");
    assert!(index < call, "the index is evaluated before the value");
}

#[test]
fn a_compound_element_assignment_evaluates_its_index_once() {
    let program = lowered(
        "fn main() -> void {
                 var a: [2]int = [1, 2];
                 var i = 1;
                 a[i] += 3;
                 exit(a[i]);
             }",
    );
    let main = main_of(&program);
    // The one index value is reused by the read and by the write, so the
    // target's indices are evaluated once.
    let runtime = |place: &Place| {
        matches!(
            place,
            Place::Element {
                index: Operand::Value(_),
                ..
            }
        )
    };
    let read = loads(main)
        .into_iter()
        .find(runtime)
        .expect("the compound assignment reads its element");
    let written = stores(main)
        .into_iter()
        .map(|(place, _)| place)
        .find(runtime)
        .expect("the compound assignment writes its element");
    assert_eq!(read, written);
}

#[test]
fn len_folds_to_the_length_while_its_operand_still_runs() {
    let folded = lowered("fn main() -> void { var a: [3]int = [1, 2, 3]; exit(len(a)); }");
    let main = main_of(&folded);
    assert_eq!(
        main.flow.blocks[0].terminator,
        Terminator::Exit {
            status: integer(3, Scalar::Int),
        }
    );

    let called = lowered(
        "fn make() -> [2]int { return [1, 2]; }
             fn main() -> void { exit(len(make())); }",
    );
    let main = main_of(&called);
    assert_eq!(call_targets(main), [FunctionId(0)]);
    assert_eq!(
        main.flow.blocks[0].terminator,
        Terminator::Exit {
            status: integer(2, Scalar::Int),
        }
    );

    let indexed = lowered(
        "fn main() -> void {
                 var grid: [2][2]int = [[1, 2], [3, 4]];
                 var row = 1;
                 exit(len(grid[row]));
             }",
    );
    let main = main_of(&indexed);
    assert!(
        loads(main)
            .iter()
            .any(|place| matches!(place, Place::Element { .. })),
        "the operand's bounds-checked access still happens"
    );

    let inside_a_fold = lowered(
        "fn main() -> void {
                 var grid: [2][2]int = [[1, 2], [3, 4]];
                 var row = 1;
                 exit(len(grid[row]) + 1);
             }",
    );
    let main = main_of(&inside_a_fold);
    assert_eq!(
        main.flow.blocks[0].terminator,
        Terminator::Exit {
            status: integer(3, Scalar::Int),
        }
    );
    assert!(
        loads(main)
            .iter()
            .any(|place| matches!(place, Place::Element { .. })),
        "an operator that folds the length does not swallow the operand"
    );

    for statement in [
        "const n = len(grid[row]);",
        "var n = u8(len(grid[row]));",
        "var n = f64(len(grid[row]));",
        "var n = u8.truncate(len(grid[row]));",
    ] {
        let program = lowered(&format!(
            "fn main() -> void {{
                 var grid: [2][2]int = [[1, 2], [3, 4]];
                 var row = 1;
                 {statement}
             }}"
        ));
        assert!(
            loads(main_of(&program))
                .iter()
                .any(|place| matches!(place, Place::Element { .. })),
            "{statement} still evaluates the operand"
        );
    }
}

#[test]
fn for_in_walks_a_copy_of_the_array_taken_once() {
    let program = lowered(
        "fn main() -> void {
                 var a: [3]int = [1, 2, 3];
                 var total = 0;
                 for v in a {
                     a[0] = 0;
                     total = total + v;
                 }
                 exit(total);
             }",
    );
    let main = main_of(&program);
    // Only two whole-array values exist: the literal into `a`, and `a` into
    // the copy the loop walks.
    let copies = main
        .values
        .iter()
        .filter(|value| matches!(value.ty, Type::Array { .. }))
        .count();
    assert_eq!(copies, 2);
}

#[test]
fn the_index_binding_reads_the_loop_counter() {
    let program = lowered(
        "fn main() -> void {
                 var a: [2]int = [5, 6];
                 var last = 0;
                 for v, i in a { last = i; }
                 exit(last);
             }",
    );
    // `a`, `last`, the copy the loop walks, the counter, and `v`. The index
    // binding is the counter rather than a sixth local.
    assert_eq!(
        main_of(&program).flow.locals,
        [
            array(2, Scalar::Int.into()),
            Scalar::Int.into(),
            array(2, Scalar::Int.into()),
            Scalar::Int.into(),
            Scalar::Int.into(),
        ]
    );
}

#[test]
fn continue_inside_for_in_advances_to_the_next_element() {
    let program = lowered(
        "fn main() -> void {
                 var a: [3]int = [1, 2, 3];
                 var total = 0;
                 for v in a {
                     if v == 2 { continue; }
                     total = total + v;
                 }
                 exit(total);
             }",
    );
    let main = main_of(&program);
    let post = main
        .flow
        .blocks
        .iter()
        .position(|block| {
            block.instructions.iter().any(|instruction| {
                matches!(
                    instruction,
                    Instruction::Value(ValueId(id))
                        if matches!(
                            main.values[*id].kind,
                            ValueKind::Binary {
                                right: Operand::Literal(Literal::Integer { value: 1, .. }),
                                ..
                            }
                        )
                )
            })
        })
        .expect("the loop increments its counter");
    let reaching = main
        .flow
        .blocks
        .iter()
        .filter(|block| block.terminator.targets().contains(&BlockId(post)))
        .count();
    assert_eq!(reaching, 2, "the body and its `continue` both increment");
}

#[test]
fn arrays_pass_through_parameters_arguments_and_results() {
    let program = lowered(
        "fn first(row: [2]int) -> int { return row[0]; }
             fn swapped(row: [2]int) -> [2]int { return [row[1], row[0]]; }
             fn main() -> void { var a: [2]int = [1, 2]; exit(first(swapped(a))); }",
    );
    let functions = &program.program().functions;
    let row = array(2, Scalar::Int.into());
    assert_eq!(functions[0].flow.locals[0], row);
    assert_eq!(functions[1].result, Some(row.clone()));
    let main = main_of(&program);
    assert_eq!(call_targets(main), [FunctionId(1), FunctionId(0)]);
    // The inner call's array result is passed on from its own storage.
    assert_eq!(main.flow.locals, [row]);
}

/// The kind of each value `function` passes as a call argument, in
/// instruction order.
fn argument_kinds(function: &Function) -> Vec<&ValueKind> {
    instructions(function)
        .filter_map(|instruction| match instruction {
            Instruction::Call { arguments, .. } => Some(arguments),
            _ => None,
        })
        .flatten()
        .map(|argument| match argument {
            Operand::Value(ValueId(id)) => &function.values[*id].kind,
            Operand::Literal(literal) => {
                panic!("an aggregate argument is a value, not {literal:?}")
            }
        })
        .collect()
}

#[test]
fn an_aggregate_argument_is_borrowed_unless_a_called_function_can_write_it() {
    let program = lowered(
        "var shared: [2]int = [1, 2];
             var grid: [2][2]int;
             fn take(row: [2]int) -> int { return row[0]; }
             fn pass(row: [2]int) -> int { return take(row); }
             fn main() -> void {
                 var local: [2]int = [3, 4];
                 var rows: [2][2]int;
                 var i = 1;
                 exit(take(local) + take(rows[i]) + take([5, 6]) + take([i, 6])
                     + take(shared) + take(grid[i]) + pass(local));
             }",
    );
    let main = main_of(&program);
    let kinds = argument_kinds(main);
    let borrowed = |kind: &ValueKind| match kind {
        ValueKind::Borrow(place) => Some(describe(place.root())),
        _ => None,
    };
    let loaded = |kind: &ValueKind| match kind {
        ValueKind::Load(place) => Some(describe(place.root())),
        _ => None,
    };
    let local = describe(&Place::Local(LocalId(0)));
    let rows = describe(&Place::Local(LocalId(1)));
    let shared = describe(&Place::Global(GlobalId(0)));
    let grid = describe(&Place::Global(GlobalId(1)));
    assert_eq!(borrowed(kinds[0]), Some(local.clone()));
    assert_eq!(borrowed(kinds[1]), Some(rows));
    assert!(matches!(kinds[2], ValueKind::Borrow(Place::Constant(_))));
    // A literal that does not fold is built in a fresh local and borrowed
    // from there.
    assert!(matches!(
        kinds[3],
        ValueKind::Borrow(Place::Local(LocalId(id))) if *id > 2
    ));
    assert_eq!(loaded(kinds[4]), Some(shared));
    assert_eq!(loaded(kinds[5]), Some(grid));
    assert_eq!(borrowed(kinds[6]), Some(local));

    // A parameter passed on is borrowed too.
    let pass = &program.program().functions[1];
    let kinds = argument_kinds(pass);
    assert_eq!(
        kinds.iter().map(|kind| borrowed(kind)).collect::<Vec<_>>(),
        [Some(describe(&Place::Local(LocalId(0))))]
    );
}

#[test]
fn returning_an_aggregate_call_returns_its_result() {
    let program = lowered(
        "fn make() -> [2]int { return [1, 2]; }
             fn relay() -> [2]int { return make(); }
             fn grouped() -> [2]int { return (make()); }
             fn main() -> void { exit(relay()[0] + grouped()[1]); }",
    );
    for function in &program.program().functions[1..3] {
        assert!(function.flow.locals.is_empty(), "{function:?}");
        let [block] = function.flow.blocks.as_slice() else {
            panic!("{function:?}");
        };
        let Terminator::Return {
            value: Some(Operand::Value(ValueId(id))),
        } = block.terminator
        else {
            panic!("{function:?}");
        };
        assert_eq!(function.values[id].kind, ValueKind::CallResult);
    }
}

#[test]
fn conversions_use_existing_operands_and_keep_their_source_spans() {
    let text = "fn main() -> void { var x: u64 = 42; var y = u8(u16(x)); exit(int(y)); }";
    let program = lowered(text);
    let values = &main_of(&program).values;
    assert_eq!(values.len(), 5);
    for (id, operand, spelling) in [
        (1, ValueId(0), "u16(x)"),
        (2, ValueId(1), "u8(u16(x))"),
        (4, ValueId(3), "int(y)"),
    ] {
        let start = text.find(spelling).unwrap();
        assert_eq!(values[id].span, Some(start..start + spelling.len()));
        assert_eq!(
            values[id].kind,
            ValueKind::Convert {
                operand: Operand::Value(operand),
                truncating: false,
            }
        );
    }
}

#[test]
fn lowered_programs() {
    let fixtures = [
        ("empty", ""),
        (
            "typed_integers",
            "
                const a: i8 = 127; const b: i16 = 32767; const c: i32 = 2147483647;
                const d: i64 = 9223372036854775807; const e: u8 = 255;
                const f: u16 = 65535; const g: u32 = 4294967295;
                const h: u64 = 18446744073709551615; const i: int = 2147483647;
                const j: uint = 4294967295; const copy = h;
                const contextual: u64 = 18446744073709551615;
            ",
        ),
        (
            "typed_conversions",
            "
                var small: i8 = 42; var medium = i16(small);
                var wide = i64(medium); var unsigned: u8 = 255;
                var unsigned_wide = u64(unsigned); var native = 42; var fixed = i32(native);
                var back = int(fixed); var u: uint = 42;
                var uf = u32(u); var ub = uint(uf);
                exit(int(small));
            ",
        ),
        (
            "typed_scopes",
            "
                var x: i8 = 42; const saved = x;
                { var x = i64(x); x = i64(saved); const copy = x; }
                x = 7; var status = int(x); status = int(saved);
                { const status = saved; exit(int(status)); const ignored: u64 = 1; }
                exit(0);
            ",
        ),
        (
            "integer_expressions",
            "
                var x: int = 40; var y: int = 2;
                const add = x + y; const subtract = x - y;
                const multiply = x * y; const divide = x / y; const remainder = x % y;
                const wrapping_add = x +% y; const wrapping_subtract = x -% y;
                const wrapping_multiply = x *% y;
                const negate = -x; const wrapping_negate = -%x; const complement = ^x;
                const and = x & y; const and_not = x & ^y;
                const xor = x ^ y; const or = x | y;
                const shift_left = x << y; const shift_right = x >> y;
                const folded: int = (250 + 10) / 2;
                exit(add);
            ",
        ),
        ("converted_literal_exit", "exit(42);"),
        ("literals", "const a = 42; var b: int = 7;"),
        ("copies", "const a = 42; var b = a; const c = b; exit(c);"),
        (
            "shadowing",
            "const x = 42; var x = x; const saved = x; const x = 7; var x = x; exit(saved);",
        ),
        (
            "assignments",
            "var x = 1; x = 42; x = x; const saved = x; x = 7; x = saved; exit(saved);",
        ),
        (
            "nested_scopes",
            "var x = 1; {} { { x = 42; } const x = x; { var x = x; x = 7; } } exit(x);",
        ),
        (
            "nested_exit",
            "var x = 1; { x = 42; { exit(x); x = 7; } x = 8; } x = 9; exit(x);",
        ),
        ("early_exit", "const x = 42; exit(x); const y = x; exit(y);"),
        (
            "floating_expressions",
            "
                var x: f64 = 40.5; var y: f64 = 2.0; var narrow: f32 = .5;
                const add = x + y; const subtract = x - y;
                const multiply = x * y; const divide = x / y; const negate = -x;
                const equal = x == y; const less = x < y;
                const rounded = f32(x); const widened = f64(narrow);
                const whole = int(y); const from_integer = f64(whole);
                const folded: f32 = 1.0 / 4.0;
                exit(whole);
            ",
        ),
    ];
    for (name, body) in fixtures {
        let program = lowered(&format!("fn main() -> void {{ {body} }}"));
        insta::assert_debug_snapshot!(name, program.program());
    }
}

#[test]
fn module_bindings_become_globals_with_constant_initial_values() {
    let program = lowered(
        "var counter = start;
             const start: int = 40;
             const step = 2;
             fn main() -> void {
                 { var counter: u8 = 1; counter = 2; }
                 counter = counter + step;
                 exit(counter);
             }",
    );
    // Only the `var` needs storage; the `const` bindings fold into their uses.
    assert_eq!(
        program.program().globals,
        vec![Global {
            ty: Scalar::Int.into(),
            values: Some(integers([40], Scalar::Int)),
        }]
    );
    insta::assert_debug_snapshot!("module_bindings", program.program());
}

#[test]
fn declarations_without_an_initializer_lower_to_their_zero_value() {
    let program = lowered(
        "type Pair struct { a: int, b: u8 }
             var counter: int;
             fn main() -> void {
                 var row: [2]int;
                 var pair: Pair;
                 counter = counter + row[1] + pair.a;
                 exit(counter);
             }",
    );
    assert_eq!(
        program.program().globals,
        vec![Global {
            ty: Scalar::Int.into(),
            values: None,
        }]
    );
    // A local aggregate is zeroed whole rather than one scalar at a time.
    assert_eq!(
        stored_places(main_of(&program))[..2],
        ["zero local0", "zero local1"]
    );
    insta::assert_debug_snapshot!("zero_declarations", program.program());
}

#[test]
fn every_function_is_lowered_with_its_signature_and_body() {
    let program = lowered(
        "fn helper() -> void { if true {} }
             fn main() -> void { exit(0); }",
    );
    let functions = &program.program().functions;
    assert_eq!(functions.len(), 2);
    assert_eq!(program.program().main, FunctionId(1));

    // The unreferenced function is lowered in place, not pruned or inlined.
    assert_eq!(functions[0].parameters, 0);
    assert_eq!(functions[0].result, None);
    assert!(functions[0].flow.blocks.len() > 1);
    assert!(matches!(
        functions[0].flow.blocks.last().unwrap().terminator,
        Terminator::Return { value: None }
    ));

    // and leaves `main` exactly as it would be on its own.
    assert_eq!(functions[1].flow.blocks.len(), 1);
    assert!(functions[1].values.is_empty());
}

#[test]
fn parameters_are_the_first_locals_in_source_order() {
    let program = lowered(
        "fn pick(first: u8, second: i64) -> i64 { return second; }
             fn main() -> void { exit(0); }",
    );
    let pick = &program.program().functions[0];
    assert_eq!(pick.parameters, 2);
    assert_eq!(pick.result, Some(Scalar::I64.into()));
    assert_eq!(
        pick.flow.locals[..2],
        [Type::from(Scalar::U8), Scalar::I64.into()]
    );
    assert!(matches!(
        pick.flow.blocks[0].terminator,
        Terminator::Return {
            value: Some(Operand::Value(_))
        }
    ));
}

#[test]
fn lowers_the_milestone_program() {
    let program = lowered(
        "var trace = 0;

             fn mark(digit: int) -> int {
                 trace = trace * 10 + digit;
                 return digit;
             }

             fn difference(left: int, right: int) -> int {
                 return left - right;
             }

             fn sum_to(value: int) -> int {
                 if value == 0 {
                     return 0;
                 }
                 return value + sum_to(value - 1);
             }

             fn positive(value: int) -> bool {
                 return value > 0;
             }

             fn remember_zero(value: int) -> void {
                 if value == 0 {
                     return;
                 }
                 trace = 255;
             }

             fn main() -> void {
                 difference(mark(4), mark(2));
                 const total = sum_to(3);
                 remember_zero(0);

                 if positive(total) {
                     exit(trace);
                 }
                 exit(255);
             }",
    );
    insta::assert_debug_snapshot!("functions", program.program());
}

#[test]
fn lowers_the_arrays_milestone_program() {
    let program = lowered(
        "const weights: [_]int = [1, 2, 3];

             fn weighted(row: [3]int) -> int {
                 var total = 0;
                 for v, i in row {
                     total = total + v * weights[i];
                 }
                 return total;
             }

             fn totals(grid: [2][3]int) -> [2]int {
                 var out: [2]int = [0...];
                 for var r = 0; r < len(grid); r = r + 1 {
                     out[r] = weighted(grid[r]);
                 }
                 return out;
             }

             fn main() -> void {
                 var grid: [2][3]int = [[1, 2, 3], [4, 5, 6]];

                 var copy = grid;
                 copy[0][0] = 99;
                 if copy == grid {
                     exit(255);
                 }

                 var sum = 0;
                 for t in totals(grid) {
                     sum = sum + t;
                 }
                 exit(sum);
             }",
    );
    insta::assert_debug_snapshot!("arrays", program.program());
}

/// The milestone example's module tree, with `app` as the root module.
const MODULE_TREE: [(&str, &str); 5] = [
    (
        "app/main.fern",
        "use counter;

const base: int = 20;

fn main() -> void {
    counter::value = base;
    counter::bump(step_total());
    exit(counter::value + base - 10);
}
",
    ),
    (
        "app/totals.fern",
        "use counter::{step};

fn step_total() -> int {
    return step * 3;
}
",
    ),
    (
        "counter/counter.fern",
        "use text::format;

pub var value = 0;
const origin = 10;

pub fn bump(amount: int) -> int {
    value = value + format::doubled(amount);
    return value;
}

fn main() -> void {
    value = 255;
}
",
    ),
    ("counter/step.fern", "pub const step = origin - 8;\n"),
    (
        "text/format/format.fern",
        "pub fn doubled(value: int) -> int {
    return value * 2;
}
",
    ),
];

#[test]
fn cross_module_calls_reads_and_assignments_lower_like_local_ones() {
    let program = lowered_tree([
        (
            "app/main.fern",
            "use dep;
                 use dep::{doubled};
                 fn main() -> void {
                     dep::value = 1;
                     dep::bump(doubled(2));
                     exit(dep::value);
                 }",
        ),
        (
            "dep/dep.fern",
            "pub var value = 0;
                 pub fn doubled(amount: int) -> int { return amount * 2; }
                 pub fn bump(amount: int) -> void { value = value + amount; }",
        ),
    ]);
    let program = program.program();
    // One global for `dep::value`, addressed from both modules.
    let value = Place::Global(GlobalId(0));
    assert_eq!(
        program.globals,
        vec![Global {
            ty: Scalar::Int.into(),
            values: None,
        }]
    );

    // The root module's files parse first, so `main` is the first function.
    assert_eq!(program.main, FunctionId(0));
    let main = main_of_program(program);
    assert_eq!(call_targets(main), [FunctionId(1), FunctionId(2)]);
    assert_eq!(stores(main), [(value.clone(), integer(1, Scalar::Int))]);
    assert_eq!(loads(main), std::slice::from_ref(&value));

    // `value = value + amount` in the callee reads the same global and its
    // own parameter local.
    let bump = &program.functions[2];
    assert_eq!(loads(bump), [value.clone(), Place::Local(LocalId(0))]);
    assert_eq!(stores(bump).len(), 1);
    assert_eq!(stores(bump)[0].0, value);
}

#[test]
fn every_module_function_is_lowered_once_under_its_own_id() {
    let program = lowered_tree([
        (
            "app/main.fern",
            "use dep;
                 fn main() -> void { exit(dep::used()); }",
        ),
        (
            "dep/dep.fern",
            "fn unused() -> int { return 7; }
                 pub fn used() -> int { return 1; }",
        ),
    ]);
    let program = program.program();
    // A dependency's private, uncalled function is still lowered in place.
    assert_eq!(program.functions.len(), 3);
    assert_eq!(program.main, FunctionId(0));
    assert_eq!(call_targets(main_of_program(program)), [FunctionId(2)]);
    assert_eq!(
        program.functions[1].flow.blocks[0].terminator,
        Terminator::Return {
            value: Some(integer(7, Scalar::Int))
        }
    );
}

#[test]
fn a_dependencys_main_is_lowered_as_an_ordinary_function() {
    let program = lowered_tree([
        (
            "app/main.fern",
            "use dep;
                 fn main() -> void { exit(dep::probe()); }",
        ),
        (
            "dep/dep.fern",
            "pub var value = 1;
                 pub fn probe() -> int { return value; }
                 fn main() -> void { value = 255; }",
        ),
    ]);
    let program = program.program();
    assert_eq!(program.main, FunctionId(0));

    // The dependency's `main` is neither the entry point nor pruned.
    let dependency_main = &program.functions[2];
    assert_eq!(dependency_main.result, None);
    assert_eq!(
        stores(dependency_main),
        [(Place::Global(GlobalId(0)), integer(255, Scalar::Int))]
    );
}

#[test]
fn a_binding_two_modules_use_is_one_global() {
    let program = lowered_tree([
        (
            "app/main.fern",
            "use left;
                 use right;
                 fn main() -> void { left::add(); exit(right::read()); }",
        ),
        (
            "left/left.fern",
            "use shared;
                 pub fn add() -> void { shared::total = shared::total + 1; }",
        ),
        (
            "right/right.fern",
            "use shared::{total};
                 pub fn read() -> int { return total; }",
        ),
        ("shared/shared.fern", "pub var total = 0;"),
    ]);
    let program = program.program();
    let total = Place::Global(GlobalId(0));
    assert_eq!(
        program.globals,
        vec![Global {
            ty: Scalar::Int.into(),
            values: None,
        }]
    );

    // `shared` loads once, so both dependents address the same storage.
    let add = &program.functions[1];
    assert_eq!(loads(add), std::slice::from_ref(&total));
    assert_eq!(stores(add).len(), 1);
    assert_eq!(stores(add)[0].0, total);
    assert_eq!(loads(&program.functions[2]), [total]);
}

#[test]
fn globals_follow_dependency_order_and_fold_imported_constants() {
    let program = lowered_tree([
        (
            "app/main.fern",
            "use dep;
                 var here = dep::seed + 1;
                 fn main() -> void { here = here + dep::there; exit(here); }",
        ),
        (
            "dep/dep.fern",
            "pub const seed = 5;
                 pub var there = 2;",
        ),
    ]);
    let program = program.program();
    // The dependency's global comes first; the imported `const` folds into
    // the root module's initializer instead of taking storage.
    assert_eq!(
        program.globals,
        vec![
            Global {
                ty: Scalar::Int.into(),
                values: Some(integers([2], Scalar::Int)),
            },
            Global {
                ty: Scalar::Int.into(),
                values: Some(integers([6], Scalar::Int)),
            },
        ]
    );
}

#[test]
fn lowers_the_module_tree() {
    insta::assert_debug_snapshot!("modules", lowered_tree(MODULE_TREE).program());
}

#[test]
fn operands_lowered_before_a_short_circuit_are_read_back_in_its_join_block() {
    let program = lowered(
        "fn take(count: int, flag: bool) -> void {}
             fn main() -> void {
                 var a = true;
                 var b = false;
                 var c = 1;
                 take(c, a && b);
                 exit(0);
             }",
    );
    let main = main_of(&program);
    let (block_index, arguments) = main
        .flow
        .blocks
        .iter()
        .enumerate()
        .find_map(|(index, block)| {
            block
                .instructions
                .iter()
                .find_map(|instruction| match instruction {
                    Instruction::Call { arguments, .. } => Some((index, arguments)),
                    _ => None,
                })
        })
        .expect("the call is lowered");
    assert_eq!(arguments.len(), 2);

    // The short-circuiting second argument splits blocks, so the first one
    // is held in a local and read back where the call runs.
    let block = &main.flow.blocks[block_index];
    let mut locals = Vec::new();
    for (argument, ty) in arguments.iter().zip([Scalar::Int, Scalar::Bool]) {
        let Operand::Value(id) = argument else {
            panic!("an argument read in the call's block is a value")
        };
        assert!(
            block.instructions.contains(&Instruction::Value(*id)),
            "the argument is defined in the call's block"
        );
        assert_eq!(
            main.values[id.0].ty,
            Type::from(ty),
            "arguments keep source order"
        );
        let ValueKind::Load(Place::Local(local)) = main.values[id.0].kind else {
            panic!("an argument held across a split is loaded from its local")
        };
        locals.push(local);
    }

    let holding_block = main
        .flow
        .blocks
        .iter()
        .position(|block| {
            block.instructions.iter().any(|instruction| {
                matches!(
                    instruction,
                    Instruction::Store {
                        place: Place::Local(local),
                        ..
                    } if *local == locals[0]
                )
            })
        })
        .expect("the first argument is held in a local");
    assert!(
        holding_block < block_index,
        "the hold happens in the block that defined the argument"
    );
}

#[test]
fn a_discarded_call_result_is_still_defined() {
    let program = lowered(
        "fn value() -> int { return 1; }
             fn main() -> void { value(); exit(0); }",
    );
    let main = main_of(&program);
    let result = main.flow.blocks[0]
        .instructions
        .iter()
        .find_map(|instruction| match instruction {
            Instruction::Call { result, .. } => Some(*result),
            _ => None,
        })
        .expect("the call is lowered");
    let ValueId(id) = result.expect("a value call defines its result");
    assert_eq!(main.values[id].kind, ValueKind::CallResult);
    assert_eq!(main.values[id].ty, Type::from(Scalar::Int));
}

#[test]
fn lowering_preserves_nested_operand_order_types_and_operator_spans() {
    let text = "fn main() -> void { var left: int = 8; var right: int = 2; var count: uint = 1; const result = (left + right) * (right - int(count)); exit(result); }";
    let lowered = lowered(text);
    let main = main_of(&lowered);
    let values = &main.values;
    assert_eq!(values.len(), 9);
    assert_eq!(
        values[2].kind,
        ValueKind::Binary {
            operator: BinaryOperator::Add,
            form: BinaryForm::Infix,
            left: Operand::Value(ValueId(0)),
            right: Operand::Value(ValueId(1)),
        }
    );
    assert_eq!(
        values[5].kind,
        ValueKind::Convert {
            operand: Operand::Value(ValueId(4)),
            truncating: false,
        }
    );
    assert_eq!(
        values[6].kind,
        ValueKind::Binary {
            operator: BinaryOperator::Subtract,
            form: BinaryForm::Infix,
            left: Operand::Value(ValueId(3)),
            right: Operand::Value(ValueId(5)),
        }
    );
    assert_eq!(
        values[7].kind,
        ValueKind::Binary {
            operator: BinaryOperator::Multiply,
            form: BinaryForm::Infix,
            left: Operand::Value(ValueId(2)),
            right: Operand::Value(ValueId(6)),
        }
    );
    for (id, spelling) in [(2, "+"), (6, "-"), (7, "*")] {
        let start = text.find(&format!(" {spelling} ")).unwrap() + 1;
        assert_eq!(values[id].span, Some(start..start + spelling.len()));
        assert_eq!(values[id].ty, Type::from(Scalar::Int));
    }
    assert!(matches!(
        main.flow.blocks.last().unwrap().terminator,
        Terminator::Exit { .. }
    ));
}

#[test]
fn lowering_contextualizes_an_untyped_runtime_shift_operand() {
    let program = lowered(
        "fn main() -> void { var count: uint = 3; const shifted: u64 = (1 << count) << count; }",
    );
    let values = &main_of(&program).values;
    assert_eq!(values.len(), 4);
    assert_eq!(
        values[1].kind,
        ValueKind::Binary {
            operator: BinaryOperator::ShiftLeft,
            form: BinaryForm::Infix,
            left: integer(1, Scalar::U64),
            right: Operand::Value(ValueId(0)),
        }
    );
    assert_eq!(values[1].ty, Type::from(Scalar::U64));
    assert_eq!(
        values[3].kind,
        ValueKind::Binary {
            operator: BinaryOperator::ShiftLeft,
            form: BinaryForm::Infix,
            left: Operand::Value(ValueId(1)),
            right: Operand::Value(ValueId(2)),
        }
    );
    assert_eq!(values[3].ty, Type::from(Scalar::U64));
}

#[test]
fn lowering_preserves_contextual_types_through_grouping() {
    let program = lowered(
        "fn main() -> void { const grouped: u8 = ((42)); var count: uint = 1; const shifted: u64 = (1 + 2) << count; }",
    );
    let values = &main_of(&program).values;
    assert_eq!(values.len(), 2);
    assert_eq!(
        values[1].kind,
        ValueKind::Binary {
            operator: BinaryOperator::ShiftLeft,
            form: BinaryForm::Infix,
            left: integer(3, Scalar::U64),
            right: Operand::Value(ValueId(0)),
        }
    );
    assert_eq!(values[1].ty, Type::from(Scalar::U64));
}

#[test]
fn lowers_nested_control_flow_and_mutations() {
    let program = lowered(
        "fn main() -> void {
                var total = 0;
                for :outer var r = 0; r < 3; r += 1 {
                    for var c = 0; c < 3; c += 1 {
                        if c == 0 { continue; }
                        if r == c { total += 1; break :outer; }
                    }
                }
                exit(total);
            }",
    );
    insta::assert_debug_snapshot!("nested_control_flow", program.program());
}

#[test]
fn lowers_short_circuit_paths_before_their_uses() {
    let program = lowered(
        "fn main() -> void {
                var divisor = 0;
                var enabled = true;
                if enabled && divisor != 0 && 10 / divisor > 1 { exit(1); }
                exit(0);
            }",
    );
    insta::assert_debug_snapshot!("short_circuit_control_flow", program.program());
}

#[test]
fn comparisons_preserve_left_operands_across_short_circuiting_right_operands() {
    for source in [
        "var a = true; var b = true; var c = false; if a == (b && c) {}",
        "var a = true; var b = true; var c = false; var d = true; if (a && b) == (c && d) {}",
        "var x = 1; var b = true; var c = false; if (x == 1) == (b && c) {}",
        "var a = true; var b = true; var c = false; var q = a == (b && c);",
    ] {
        lowered(&format!("fn main() -> void {{ {source} }}"));
    }
}

#[test]
fn floating_constants_lower_to_the_bits_of_their_format() {
    let program = lowered(
        "var narrow: f32 = 0.1;
             var wide: f64 = 0.1;
             const zero: f32 = 0.0;
             var signed_zero: f32 = -zero;
             var table: [2][1]f64 = [[0.5], [-1.5]];
             fn main() -> void { exit(len(table)); }",
    );
    assert_eq!(
        program.program().globals,
        vec![
            Global {
                ty: Scalar::F32.into(),
                values: Some(vec![Literal::Floating(Float::Binary32(0.1f32.to_bits()))]),
            },
            Global {
                ty: Scalar::F64.into(),
                values: Some(vec![Literal::Floating(Float::Binary64(0.1f64.to_bits()))]),
            },
            // A literal rounds once to its format, so `f32` and `f64` hold
            // different values for the same source text. Negating a typed
            // zero keeps the sign bit an untyped constant cannot carry.
            Global {
                ty: Scalar::F32.into(),
                values: Some(vec![Literal::Floating(Float::Binary32(
                    (-0.0f32).to_bits()
                ))]),
            },
            Global {
                ty: array(2, array(1, Scalar::F64.into())),
                values: Some(doubles([0.5, -1.5])),
            },
        ]
    );
}

#[test]
fn a_folded_floating_expression_lowers_to_one_literal_operand() {
    let program = lowered("fn main() -> void { var value: f32 = 0.25 + 0.25; exit(0); }");
    let main = main_of(&program);
    assert_eq!(
        stores(main),
        [(
            Place::Local(LocalId(0)),
            floating(Float::Binary32(0.5f32.to_bits())),
        )]
    );
    // The whole expression folded, so nothing computes the value at run time.
    assert!(main.values.is_empty());
}

#[test]
fn floating_signatures_arrays_and_globals_reach_the_common_ir_forms() {
    let program = lowered(
        "var scale: f64 = 2.0;
             fn weigh(weights: [2]f64, index: int) -> f64 { return weights[index] * scale; }
             fn main() -> void {
                 var weights: [2]f64 = [0.5, 1.5];
                 var total = weigh(weights, 1) + weigh(weights, 0);
                 if total == 4.0 { exit(42); }
                 exit(0);
             }",
    );
    let weigh = &program.program().functions[0];
    assert_eq!(weigh.result, Some(Scalar::F64.into()));
    assert_eq!(
        weigh.flow.locals,
        [array(2, Scalar::F64.into()), Scalar::Int.into()]
    );
    assert_eq!(loads(weigh).len(), 3);
    insta::assert_debug_snapshot!("floating_functions", program.program());
}

/// The struct declarations the lowering tests build their programs on.
const STRUCTS: &str = "type Point struct { x: int, y: u8 }
     type Cell struct { point: Point, weights: [2]f64 }
     type Row struct { count: int }
     type Table struct { rows: [2]Row }
     ";

fn lowered_with_structs(body: &str) -> VerifiedProgram {
    lowered(&format!("{STRUCTS}{body}"))
}

#[test]
fn the_program_owns_every_struct_definition_in_field_order() {
    let program = lowered_with_structs("fn main() -> void { exit(0); }");
    assert_eq!(
        program.program().structs,
        vec![
            Struct {
                fields: vec![Scalar::Int.into(), Scalar::U8.into()],
            },
            Struct {
                fields: vec![declared(0, "Point"), array(2, Scalar::F64.into())],
            },
            Struct {
                fields: vec![Scalar::Int.into()],
            },
            Struct {
                fields: vec![array(2, declared(2, "Row"))],
            },
        ]
    );
}

#[test]
fn heterogeneous_struct_globals_flatten_in_field_order() {
    let program = lowered_with_structs(
        "var cell = Cell { point = Point { x = 1, y = 2 }, weights = [0.5, 1.5] };
         var table = Table { ... };
         fn main() -> void { exit(0); }",
    );
    assert_eq!(
        program.program().globals,
        vec![
            // A struct's fields have their own types, so one global holds
            // `int`, `u8`, and `f64` literals in declaration order.
            Global {
                ty: declared(1, "Cell"),
                values: Some(vec![
                    Literal::Integer {
                        value: 1,
                        ty: Scalar::Int,
                    },
                    Literal::Integer {
                        value: 2,
                        ty: Scalar::U8,
                    },
                    Literal::Floating(Float::Binary64(0.5f64.to_bits())),
                    Literal::Floating(Float::Binary64(1.5f64.to_bits())),
                ]),
            },
            Global {
                ty: declared(3, "Table"),
                values: None,
            },
        ]
    );
}

#[test]
fn a_struct_literal_stores_its_written_fields_and_then_its_filled_ones() {
    let complete = lowered_with_structs(
        "var y: u8 = 2;
         fn main() -> void { var p = Point { y = y, x = 1 }; exit(0); }",
    );
    // The written fields store through their own ordinals, in source order,
    // rather than in declaration order.
    assert_eq!(
        stored_places(main_of(&complete)),
        ["local0.1", "local0.0", "local1"]
    );

    let filled = lowered_with_structs(
        "var x = 7;
         fn main() -> void { var c = Cell { point = Point { x = x, ... }, ... }; exit(0); }",
    );
    assert_eq!(
        stored_places(main_of(&filled)),
        [
            // `Point { x = x, ... }` writes `x` and zeroes `y` into its own
            // local, which then stores into the enclosing literal's `point`
            // field before that literal zeroes its whole array field.
            "local1.0",
            "local1.1",
            "local0.0",
            "zero local0.1",
            "local2",
        ]
    );
}

#[test]
fn a_folded_struct_is_static_data_holding_each_scalar() {
    let program = lowered_with_structs(
        "fn main() -> void {
             const cell = Cell { point = Point { x = 1, y = 2 }, weights = [0.5, 1.5] };
             var copy = cell;
             exit(0);
         }",
    );
    let main = main_of(&program);
    // A folded struct reaches storage the same way a folded array does: its
    // static data holds one literal per scalar, and the binding copies it.
    assert_eq!(stored_places(main), ["local0"]);
    assert_eq!(loads(main), [Place::Constant(ConstantId(0))]);
    assert_eq!(
        main.constants[0].values,
        Some(vec![
            Literal::Integer {
                value: 1,
                ty: Scalar::Int
            },
            Literal::Integer {
                value: 2,
                ty: Scalar::U8
            },
            Literal::Floating(Float::Binary64(0.5f64.to_bits())),
            Literal::Floating(Float::Binary64(1.5f64.to_bits())),
        ])
    );
}

#[test]
fn field_selection_reads_scalar_and_aggregate_fields_through_one_place() {
    let program = lowered_with_structs(
        "fn main() -> void {
             var cell = Cell { point = Point { x = 4, y = 5 }, weights = [0.5, 1.5] };
             var point = cell.point;
             var weight = cell.weights[1];
             exit(point.x);
         }",
    );
    let main = main_of(&program);
    // `constant0` holds the folded literal and `local0` is `cell`. A scalar
    // field and an aggregate field both load through a field place, and no
    // separate struct copy instruction exists.
    let cell = Place::Local(LocalId(0));
    let loaded: Vec<String> = loads(main).iter().map(describe).collect();
    assert_eq!(loaded, ["constant0", "local0.0", "local0.1[1]", "local1.0"]);
    assert_eq!(
        loads(main)[1],
        field(cell.clone(), 0),
        "the aggregate field loads through one place"
    );
    let Place::Element { base, .. } = &loads(main)[2] else {
        unreachable!("indexing an array field ends in an element step")
    };
    assert_eq!(**base, field(cell, 1));
}

#[test]
fn copying_a_struct_copies_its_whole_value() {
    let program = lowered_with_structs(
        "fn main() -> void {
             var p = Point { x = 1, y = 2 };
             var copy = p;
             copy.x = 9;
             exit(p.x);
         }",
    );
    let main = main_of(&program);
    // The copy loads the whole struct once and stores it into its own local,
    // so assigning through the copy cannot reach the source.
    let whole = main
        .values
        .iter()
        .filter(|value| {
            value.kind == ValueKind::Load(Place::Local(LocalId(0)))
                && value.ty == declared(0, "Point")
        })
        .count();
    assert_eq!(whole, 1);
    assert_eq!(stored_places(main), ["local0", "local1", "local1.0"]);
}

#[test]
fn structs_pass_through_parameters_arguments_and_results() {
    let program = lowered_with_structs(
        "fn shifted(p: Point) -> Point { return Point { x = p.x + 1, y = p.y }; }
         fn main() -> void {
             var p = Point { x = 1, y = 2 };
             exit(shifted(shifted(p)).x);
         }",
    );
    let functions = &program.program().functions;
    let point = declared(0, "Point");
    assert_eq!(functions[0].flow.locals[0], point);
    assert_eq!(functions[0].result, Some(point.clone()));
    let main = main_of(&program);
    assert_eq!(call_targets(main), [FunctionId(0), FunctionId(0)]);
    // The inner call's struct result lands in a local before it is passed on.
    assert!(main.flow.locals.contains(&point));
}

#[test]
fn struct_equality_compares_whole_values() {
    for (source, operator) in [
        ("a == b", ComparisonOperator::Equal),
        ("a != b", ComparisonOperator::NotEqual),
    ] {
        let program = lowered_with_structs(&format!(
            "fn main() -> void {{
                 var a = Point {{ x = 1, y = 2 }};
                 var b = Point {{ x = 1, y = 2 }};
                 if {source} {{ exit(1); }}
                 exit(0);
             }}"
        ));
        let main = main_of(&program);
        let comparison = main
            .values
            .iter()
            .find(|value| matches!(value.kind, ValueKind::Comparison { .. }))
            .expect("the condition compares the two structs");
        assert_eq!(comparison.ty, Scalar::Bool.into());
        let ValueKind::Comparison {
            operator: found,
            left,
            right,
        } = &comparison.kind
        else {
            unreachable!("the value is a comparison")
        };
        assert_eq!(*found, operator);
        // Both operands are whole struct values, not field-by-field reads.
        for operand in [left, right] {
            let Operand::Value(ValueId(id)) = operand else {
                unreachable!("a struct operand is a loaded value")
            };
            assert_eq!(main.values[*id].ty, declared(0, "Point"));
        }
    }
}

#[test]
fn written_field_initializers_lower_in_source_order() {
    let program = lowered_with_structs(
        "fn one() -> int { return 1; }
         fn two() -> u8 { return 2; }
         fn main() -> void { var p = Point { y = two(), x = one() }; exit(0); }",
    );
    // `y` is written first, so its call runs first even though `x` is
    // declared first.
    assert_eq!(
        call_targets(main_of(&program)),
        [FunctionId(1), FunctionId(0)]
    );
    assert_eq!(
        stored_places(main_of(&program)),
        ["local0.1", "local0.0", "local1"]
    );
}

#[test]
fn mixed_field_and_element_targets_build_their_steps_in_order() {
    let program = lowered_with_structs(
        "fn main() -> void {
             var grid: [2]Cell = [Cell { ... }, Cell { ... }];
             var table = Table { ... };
             var i = 1;
             grid[i].point.x = 3;
             table.rows[i].count = 4;
             exit(0);
         }",
    );
    let places = stored_places(main_of(&program));
    let mixed: Vec<&String> = places.iter().filter(|place| place.contains("[v")).collect();
    assert_eq!(mixed, ["local0[v2].0.0", "local1.0[v3].0"]);
}

#[test]
fn an_assignment_through_a_mixed_target_rebuilds_it_after_a_split_value() {
    let program = lowered_with_structs(
        "type Flags struct { seen: [2]bool }
         fn main() -> void {
             var flags = Flags { ... };
             var i = 0;
             var a = true;
             var b = false;
             flags.seen[i] = a && b;
             exit(0);
         }",
    );
    let main = main_of(&program);
    let store = stores(main)
        .pop()
        .expect("the assignment stores through the mixed target");
    // The index was lowered before the short-circuiting value, so the store's
    // block reads it back through its spill rather than through the original
    // value.
    let Place::Element { base, index, .. } = &store.0 else {
        unreachable!("the target ends in an element step")
    };
    assert_eq!(**base, field(Place::Local(LocalId(0)), 0));
    let Operand::Value(ValueId(id)) = index else {
        unreachable!("a runtime index is a value")
    };
    assert!(matches!(
        main.values[*id].kind,
        ValueKind::Load(Place::Local(_))
    ));
}

#[test]
fn an_element_read_rebuilds_its_base_after_a_split_index() {
    // `grid[i]` is lowered before the short-circuiting index, so the load
    // reads `i` back in the index's final block, and verification accepts a
    // value read only in the block that defines it.
    lowered(
        "fn pick(c: bool) -> int { if c { return 1; } return 0; }
         fn main() -> void {
             var grid = [[1, 2], [3, 4]];
             var i = 1;
             var b = true;
             exit(grid[i][pick(b && b)]);
         }",
    );
}

#[test]
fn a_compound_assignment_evaluates_each_runtime_index_once() {
    let program = lowered_with_structs(
        "fn next() -> int { return 0; }
         fn main() -> void {
             var grid: [2]Cell = [Cell { ... }, Cell { ... }];
             grid[next()].point.x = grid[0].point.x;
             grid[next()].point.x += 5;
             exit(0);
         }",
    );
    let main = main_of(&program);
    // Two statements name `grid[next()]`, and each evaluates its index once
    // even though the compound assignment reads and writes through it.
    assert_eq!(call_targets(main), [FunctionId(0), FunctionId(0)]);
    let compound = main
        .values
        .iter()
        .filter(|value| {
            matches!(
                value.kind,
                ValueKind::Binary {
                    form: BinaryForm::CompoundAssignment,
                    ..
                }
            )
        })
        .count();
    assert_eq!(compound, 1);
}

#[test]
fn lowers_the_structs_milestone_program() {
    let program = lowered_with_structs(
        "var start = Point { x = 19, y = 23 };

         fn shifted(cell: Cell) -> Cell {
             return Cell { point = Point { x = cell.point.x + 1, y = cell.point.y }, ... };
         }

         fn main() -> void {
             var cell = Cell { point = start, weights = [0.5, 1.5] };
             cell.weights[1] = 2.5;
             var moved = shifted(cell);
             const expected = Point { x = 20, y = 23 };
             if moved.point == expected {
                 exit(moved.point.x + int(moved.point.y));
             }
             exit(255);
         }",
    );
    insta::assert_debug_snapshot!("structs", program.program());
}

/// The kind of every value `main` defines, spelled with `{:?}`.
fn value_kinds(function: &Function) -> Vec<String> {
    function
        .values
        .iter()
        .map(|value| format!("{:?}", value.kind))
        .collect()
}

#[test]
fn assignments_store_through_pointers_loaded_before_the_value() {
    let program = lowered(
        "type S struct { x: int, a: [3]int }
fn get(p: *S) -> *S { return p; }
fn main() -> void {
    var x = 1;
    const p = &x;
    *p = 2;
    var s: S;
    const ps = &s;
    ps.x = 3;
    ps.a[2] = 4;
    var vps = ps;
    const pps = &vps;
    pps.x = 5;
    get(ps).x = 6;
    *p = get(ps).x;
}",
    );
    let main = main_of(&program);
    assert_eq!(
        stored_places(main),
        [
            "local0",
            "local1",
            "*(v1)",
            "zero local2",
            "local3",
            "*(v3).0",
            "*(v4).1[2]",
            "local4",
            "local5",
            "*(v8).0",
            "*(v10).0",
            "*(v11)",
        ]
    );
    let kinds = value_kinds(main);
    assert_eq!(kinds[0], "Address(Local(LocalId(0)))");
    assert_eq!(kinds[6], "Address(Local(LocalId(4)))");
    // `pps.x` loads `pps`, then the `*S` it points at.
    assert_eq!(kinds[7], "Load(Local(LocalId(5)))");
    assert_eq!(kinds[8], "Load(Indirect { pointer: Value(ValueId(7)) })");
    // `*p = get(ps).x;` loads `p` before the call.
    assert_eq!(kinds[11], "Load(Local(LocalId(1)))");
    assert_eq!(kinds[13], "CallResult");
}

#[test]
fn addresses_name_element_and_field_places() {
    let program = lowered(
        "type S struct { x: int }
fn main() -> void {
    var a = [1, 2];
    var s = S { x = 1 };
    const pa = &a[1];
    const ps = &s.x;
    const p = &s;
    const px = &p.x;
}",
    );
    let main = main_of(&program);
    let addresses = main
        .values
        .iter()
        .filter_map(|value| match &value.kind {
            ValueKind::Address(place) => Some((describe(place), value.ty.to_string())),
            _ => None,
        })
        .collect::<Vec<_>>();
    let expected = [
        ("local0[1]", "*int"),
        ("local1.0", "*int"),
        ("local1", "*S"),
        ("*(v5).0", "*int"),
    ];
    assert_eq!(
        addresses,
        expected.map(|(place, ty)| (place.to_owned(), ty.to_owned()))
    );
}

#[test]
fn values_are_read_through_pointers() {
    let program = lowered(
        "type S struct { x: int, a: [3]int }
fn main() -> void {
    var x = 1;
    const p = &x;
    var s: S;
    const ps = &s;
    var a = [1, 2, 3];
    const q = &a;
    const c = *ps;
    exit(*p + ps.x + q[1] + len(q) + c.x);
}",
    );
    let main = main_of(&program);
    assert_eq!(
        loads(main).iter().map(describe).collect::<Vec<_>>(),
        [
            "constant0",
            "local3",
            "*(v4)",
            "local1",
            "*(v6)",
            "local3",
            "*(v8).0",
            "local5",
            "*(v11)[1]",
            "local5",
            "local6.0",
        ]
    );
    // `*ps` copies the whole struct target.
    assert_eq!(main.values[5].ty.to_string(), "S");
    // `len(q)` folds to the length and still reads `q`.
    assert_eq!(value_kinds(main)[14], "Load(Local(LocalId(5)))");
    assert!(matches!(
        &main.values[15].kind,
        ValueKind::Binary {
            right: Operand::Literal(Literal::Integer { value: 3, .. }),
            ..
        }
    ));
}

#[test]
fn heap_and_nullable_pointer_operations_lower_to_their_own_instructions() {
    let source = "type S struct { x: int }
fn f() -> int { return 1; }
fn main() -> void {
    const n = new(1);
    const m = new(S { x = f() });
    const o = new(f());
    var q: nullable *int = n;
    const r = *int(q);
    const t = *int(n);
    free(n);
    q = null;
    var z: nullable *int;
    if n == q { exit(1); }
}";
    let program = lowered(source);
    let main = main_of(&program);
    assert_eq!(
        stored_places(main),
        [
            "*(v0)", "local0", "local1.0", "*(v3)", "local2", "*(v5)", "local3", "local4",
            "local5", "local6", "free v10", "local4", "local7",
        ]
    );
    let span = |text: &str| {
        let start = source.find(text).unwrap();
        Some(start..start + text.len())
    };
    // Each allocation follows its operand and reports its own span.
    for (id, text) in [(0, "new(1)"), (3, "new(S { x = f() })"), (5, "new(f())")] {
        assert_eq!(main.values[id].kind, ValueKind::Allocate);
        assert_eq!(main.values[id].span, span(text));
    }
    assert_eq!(main.values[1].kind, ValueKind::CallResult);
    assert_eq!(main.values[4].kind, ValueKind::CallResult);
    // Only the nullable operand is checked.
    assert_eq!(
        main.values[8].kind,
        ValueKind::CheckNull {
            operand: Operand::Value(ValueId(7))
        }
    );
    assert_eq!(main.values[8].span, span("*int(q)"));
    assert_eq!(value_kinds(main)[9], "Load(Local(LocalId(0)))");
    // `null` and a nullable zero value are stored as `null` literals, and a
    // `*int` widens where a `nullable *int` is expected.
    let nulls = stores(main)
        .into_iter()
        .filter(|(_, operand)| matches!(operand, Operand::Literal(Literal::Null(_))))
        .map(|(place, _)| describe(&place))
        .collect::<Vec<_>>();
    assert_eq!(nulls, ["local4", "local7"]);
    assert_eq!(main.values[6].ty.to_string(), "nullable *int");
    assert!(matches!(
        &main.values[13].kind,
        ValueKind::Comparison {
            operator: ComparisonOperator::Equal,
            ..
        }
    ));
}

#[test]
fn widened_pointers_pass_as_arguments_and_results() {
    lowered(
        "fn take(q: nullable *int) -> nullable *int { return q; }
fn widen(p: *int) -> nullable *int { return p; }
fn main() -> void {
    const p = new(1);
    var q = take(p);
    q = widen(p);
    var a: [2]nullable *int = [p...];
    if p != q { exit(1); }
}",
    );
}

#[test]
fn globals_hold_null_for_their_pointer_slots() {
    let program = lowered(
        "type Node struct { value: int, next: nullable *Node }
var n = Node { value = 1, next = null };
var empty = Node { value = 0, next = null };
fn main() -> void {}",
    );
    let node = declared(0, "Node");
    let nullable_node = Type::Pointer {
        nullable: true,
        target: Box::new(node.clone()),
    };
    assert_eq!(
        program.program().globals,
        [
            Global {
                ty: node.clone(),
                values: Some(vec![
                    Literal::Integer {
                        value: 1,
                        ty: Scalar::Int
                    },
                    Literal::Null(nullable_node),
                ]),
            },
            Global {
                ty: node,
                values: None,
            },
        ]
    );
}

#[test]
fn an_aggregate_a_pointer_can_reach_is_copied_into_a_call() {
    let program = lowered(
        "fn sum(values: [2]int) -> int { return values[0] + values[1]; }
fn main() -> void {
    var kept = [1, 2];
    var shared = [3, 4];
    const p = &shared;
    var total = sum(kept) + sum(shared) + sum(*p);
}",
    );
    let arguments = main_of(&program)
        .values
        .iter()
        .filter_map(|value| match &value.kind {
            ValueKind::Borrow(place) => Some(format!("borrow {}", describe(place))),
            ValueKind::Load(place) if value.ty.is_aggregate() => {
                Some(format!("load {}", describe(place)))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        arguments,
        [
            "load constant0",
            "load constant1",
            "borrow local0",
            "load local1",
            "load *(v8)",
        ]
    );
}

#[test]
fn the_specification_pointer_examples_lower() {
    for source in [
        "type Point struct { x: int, y: int }
fn main() -> void {
    var p = new(Point { x = 1, y = 2 });
    var n = new(0);
    p.x = 5;
    exit(p.x);
}",
        "fn main() -> void {
    var p = new([10, 20, 30]);
    p[1] = 25;
    var copy = *p;
    copy[0] = 0;
    exit(p[0] + p[1]);
}",
        "type Point struct { x: int, y: int }
fn main() -> void {
    var p = Point { x = 1, y = 2 };
    const a = &p;
    a.x = 5;
    exit(p.x);
}",
        "fn main() -> void {
    var a = new(1);
    var b = new(1);
    const c = a;
    if a == b { exit(1); }
    if a == c { exit(0); }
    exit(2);
}",
    ] {
        lowered(source);
    }
}

#[test]
fn pointers_snapshot() {
    let program = lowered(
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
    );
    insta::assert_debug_snapshot!("pointers", program.program());
}

#[test]
fn a_returned_literal_is_built_in_the_result_storage() {
    let program = lowered(
        "type S struct { a: int, b: [2]int }
fn pair(x: int) -> [2]int { return [x, x + 1]; }
fn grouped(x: int) -> S { return (S { a = x, ... }); }
fn filled(x: int) -> [3]int { return [x...]; }
fn folded() -> [2]int { return [1, 2]; }
fn copied(x: int) -> [2]int { var local = [x, x]; return local; }
fn main() -> void {}",
    );
    let functions = &program.program().functions;
    let returned = |function: &Function| {
        function
            .flow
            .blocks
            .iter()
            .find_map(|block| match &block.terminator {
                Terminator::Return {
                    value: Some(Operand::Value(ValueId(id))),
                } => Some(format!("{:?}", function.values[*id].kind)),
                _ => None,
            })
            .unwrap()
    };
    assert_eq!(stored_places(&functions[0]), ["result[0]", "result[1]"]);
    assert_eq!(returned(&functions[0]), "Load(Result)");
    // Only the parameter is a local.
    assert_eq!(functions[0].flow.locals.len(), 1);
    assert_eq!(stored_places(&functions[1]), ["result.0", "zero result.1"]);
    assert_eq!(returned(&functions[1]), "Load(Result)");
    assert_eq!(stored_places(&functions[2]), ["result[0]", "result[1..]"]);
    assert_eq!(returned(&functions[2]), "Load(Result)");
    assert_eq!(returned(&functions[3]), "Load(Constant(ConstantId(0)))");
    assert_eq!(returned(&functions[4]), "Load(Local(LocalId(2)))");
}
