use super::*;
use crate::types::{ComparisonOperator, UnaryOperator};

#[test]
fn verification_rejects_invalid_array_places() {
    let row = array(2, Scalar::Int.into());
    let load = |ty: Type, place| Value {
        span: None,
        ty,
        kind: ValueKind::Load(place),
    };
    for (values, locals, initialize, expected) in [
        (
            vec![load(
                Scalar::Int.into(),
                element(Place::Local(LocalId(0)), integer(0, Scalar::Int)),
            )],
            vec![Type::from(Scalar::Int)],
            Place::Local(LocalId(0)),
            "IR indexes `int`",
        ),
        (
            vec![load(
                Scalar::Int.into(),
                element(Place::Local(LocalId(0)), integer(0, Scalar::U8)),
            )],
            vec![row.clone()],
            element(Place::Local(LocalId(0)), integer(0, Scalar::Int)),
            "IR index has type `u8`",
        ),
        (
            vec![load(
                Scalar::U8.into(),
                element(Place::Local(LocalId(0)), integer(0, Scalar::Int)),
            )],
            vec![row.clone()],
            element(Place::Local(LocalId(0)), integer(0, Scalar::Int)),
            "invalid IR value",
        ),
    ] {
        let blocks = vec![Block {
            instructions: vec![
                Instruction::Store {
                    place: initialize,
                    operand: integer(0, Scalar::Int),
                },
                Instruction::Value(ValueId(0)),
            ],
            terminator: Terminator::Exit {
                status: integer(0, Scalar::Int),
            },
        }];
        let error = one_function(main_function(values, locals, blocks))
            .verify()
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn verification_rejects_array_values_outside_copies_and_equality() {
    let row = array(2, Scalar::Int.into());
    let operation = |kind| Value {
        span: Some(0..1),
        ty: Scalar::Bool.into(),
        kind,
    };
    let load = Value {
        span: None,
        ty: row.clone(),
        kind: ValueKind::Load(Place::Local(LocalId(0))),
    };
    let rows = Operand::Value(ValueId(0));
    for (values, expected) in [
        (
            vec![
                load.clone(),
                operation(ValueKind::Comparison {
                    operator: ComparisonOperator::Less,
                    left: rows,
                    right: rows,
                }),
            ],
            "invalid IR value",
        ),
        (
            vec![
                load.clone(),
                Value {
                    span: Some(0..1),
                    ty: row.clone(),
                    kind: ValueKind::Binary {
                        operator: BinaryOperator::Add,
                        form: BinaryForm::Infix,
                        left: rows,
                        right: rows,
                    },
                },
            ],
            "where a scalar is required",
        ),
    ] {
        let blocks = vec![Block {
            instructions: vec![
                Instruction::Store {
                    place: element(Place::Local(LocalId(0)), integer(0, Scalar::Int)),
                    operand: integer(0, Scalar::Int),
                },
                Instruction::Value(ValueId(0)),
                Instruction::Value(ValueId(1)),
            ],
            terminator: Terminator::Exit {
                status: integer(0, Scalar::Int),
            },
        }];
        let error = one_function(main_function(values, vec![row.clone()], blocks))
            .verify()
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }

    // Equality on identical array types is the one array operation.
    let equality = vec![
        load,
        operation(ValueKind::Comparison {
            operator: ComparisonOperator::Equal,
            left: rows,
            right: rows,
        }),
    ];
    let blocks = vec![Block {
        instructions: vec![
            Instruction::Store {
                place: element(Place::Local(LocalId(0)), integer(0, Scalar::Int)),
                operand: integer(0, Scalar::Int),
            },
            Instruction::Value(ValueId(0)),
            Instruction::Value(ValueId(1)),
        ],
        terminator: Terminator::Exit {
            status: integer(0, Scalar::Int),
        },
    }];
    one_function(main_function(equality, vec![row], blocks))
        .verify()
        .unwrap();
}

#[test]
fn verification_rejects_a_copy_between_different_array_types() {
    let function = main_function(
        vec![Value {
            span: None,
            ty: array(3, Scalar::Int.into()),
            kind: ValueKind::Load(Place::Local(LocalId(1))),
        }],
        vec![array(2, Scalar::Int.into()), array(3, Scalar::Int.into())],
        vec![Block {
            instructions: vec![
                Instruction::Store {
                    place: element(Place::Local(LocalId(1)), integer(0, Scalar::Int)),
                    operand: integer(0, Scalar::Int),
                },
                Instruction::Value(ValueId(0)),
                Instruction::Store {
                    place: Place::Local(LocalId(0)),
                    operand: Operand::Value(ValueId(0)),
                },
            ],
            terminator: Terminator::Exit {
                status: integer(0, Scalar::Int),
            },
        }],
    );
    let error = one_function(function).verify().unwrap_err();
    assert!(
        error.to_string().contains("IR store has type `[3]int`"),
        "{error}"
    );
}

#[test]
fn verification_checks_a_globals_values_against_its_type() {
    for (ty, values, expected) in [
        (
            array(2, Scalar::Int.into()),
            vec![1],
            "holds 1 values, expected 2",
        ),
        (
            array(2, array(2, Scalar::Int.into())),
            vec![1, 2, 3, 4, 5],
            "holds 5 values, expected 4",
        ),
        (
            array(2, Scalar::U8.into()),
            vec![0, 256],
            "IR integer 256 out of range",
        ),
    ] {
        let mut program = one_function(main_function(
            vec![],
            vec![],
            vec![Block {
                instructions: vec![],
                terminator: Terminator::Exit {
                    status: integer(0, Scalar::Int),
                },
            }],
        ));
        program.globals = vec![Global { ty, values }];
        let error = program.verify().unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn verification_rejects_invalid_references() {
    for values in [
        vec![convert(Operand::Value(ValueId(usize::MAX)), Scalar::Int)],
        vec![
            convert(Operand::Value(ValueId(1)), Scalar::Int),
            convert(integer(42, Scalar::Int), Scalar::Int),
        ],
        vec![convert(Operand::Value(ValueId(0)), Scalar::Int)],
    ] {
        let error = program(values, integer(0, Scalar::Int))
            .verify()
            .unwrap_err();
        assert!(
            error.to_string().contains("undefined value")
                || error.to_string().contains("not defined earlier")
        );
    }
    for reference in [0, 1, usize::MAX] {
        assert!(
            program(vec![], Operand::Value(ValueId(reference)))
                .verify()
                .is_err()
        );
    }
    assert!(
        program(
            vec![convert(integer(42, Scalar::Int), Scalar::Int)],
            Operand::Value(ValueId(1)),
        )
        .verify()
        .is_err()
    );
}

#[test]
fn verification_accepts_constants_conversions_and_exits() {
    for exit in [
        integer(-1, Scalar::Int),
        Operand::Value(ValueId(0)),
        Operand::Value(ValueId(1)),
    ] {
        program(
            vec![
                convert(integer(i128::from(i32::MIN), Scalar::Int), Scalar::Int),
                convert(Operand::Value(ValueId(0)), Scalar::Int),
            ],
            exit,
        )
        .verify()
        .unwrap();
    }
}

fn ranges() -> [(Scalar, i128, i128); 10] {
    Scalar::ALL_INTEGERS.map(|ty| (ty, ty.min(), i128::from(ty.max())))
}

#[test]
fn verification_checks_literal_ranges_and_conversion_types() {
    for (ty, min, max) in ranges() {
        for value in [min, 0, max] {
            program(
                vec![
                    convert(integer(value, ty), ty),
                    convert(Operand::Value(ValueId(0)), ty),
                ],
                integer(0, Scalar::Int),
            )
            .verify()
            .unwrap();
        }
        for value in [i128::MIN, min - 1, max + 1, i128::MAX] {
            let error = program(
                vec![convert(integer(value, ty), ty)],
                integer(0, Scalar::Int),
            )
            .verify()
            .unwrap_err();
            assert!(error.to_string().contains("out of range"));
        }
        assert!(
            program(
                vec![convert(integer(0, ty), Scalar::Bool)],
                integer(0, Scalar::Int),
            )
            .verify()
            .is_err()
        );
    }
}

#[test]
fn verification_checks_conversions_and_exits() {
    for (source, min, max) in ranges() {
        for (destination, _, _) in ranges() {
            for value in [min, max] {
                for operand in [integer(value, source), Operand::Value(ValueId(0))] {
                    // A conversion that can trap must carry the span its trap reports.
                    for span in [None, Some(0..1)] {
                        let reportable = span.is_some();
                        let result = program(
                            vec![
                                convert(integer(value, source), source),
                                Value {
                                    span,
                                    ty: destination.into(),
                                    kind: ValueKind::Convert {
                                        operand,
                                        truncating: false,
                                    },
                                },
                            ],
                            integer(0, Scalar::Int),
                        )
                        .verify();
                        assert_eq!(
                            result.is_ok(),
                            reportable || source.all_values_fit(destination),
                            "{source:?} to {destination:?}: {value}"
                        );
                    }
                }
            }
        }
        for exit in [integer(0, source), Operand::Value(ValueId(0))] {
            assert_eq!(
                program(vec![convert(integer(0, source), source)], exit)
                    .verify()
                    .is_ok(),
                source == Scalar::Int
            );
        }
    }
    for operand in [
        integer(128, Scalar::I8),
        Operand::Value(ValueId(0)),
        Operand::Value(ValueId(usize::MAX)),
    ] {
        assert!(
            program(
                vec![Value {
                    span: None,
                    ty: Scalar::Int.into(),
                    kind: ValueKind::Convert {
                        operand,
                        truncating: false,
                    }
                }],
                integer(0, Scalar::Int),
            )
            .verify()
            .is_err()
        );
    }
    for value in [
        -(1i128 << (Scalar::Int.width() - 1)) - 1,
        i128::from(Scalar::Int.max()) + 1,
    ] {
        assert!(
            program(vec![], integer(value, Scalar::Int))
                .verify()
                .is_err()
        );
    }
}

#[test]
fn verification_accepts_checked_conversion_operands() {
    program(
        vec![
            convert(integer(42, Scalar::U8), Scalar::U8),
            Value {
                span: None,
                ty: Scalar::U64.into(),
                kind: ValueKind::Convert {
                    operand: Operand::Value(ValueId(0)),
                    truncating: false,
                },
            },
        ],
        integer(0, Scalar::Int),
    )
    .verify()
    .unwrap();
}

#[test]
fn verification_checks_operation_shapes_and_accepts_independent_shift_counts() {
    let operation = |ty: Scalar, kind| Value {
        span: Some(0..1),
        ty: ty.into(),
        kind,
    };
    program(
        vec![
            convert(integer(1, Scalar::U8), Scalar::U8),
            convert(integer(1, Scalar::U16), Scalar::U16),
            operation(
                Scalar::U8,
                ValueKind::Binary {
                    operator: BinaryOperator::ShiftLeft,
                    form: BinaryForm::Infix,
                    left: Operand::Value(ValueId(0)),
                    right: Operand::Value(ValueId(1)),
                },
            ),
        ],
        integer(0, Scalar::Int),
    )
    .verify()
    .unwrap();

    for value in [
        operation(
            Scalar::U8,
            ValueKind::Unary {
                operator: UnaryOperator::Negate,
                operand: integer(1, Scalar::U8),
            },
        ),
        operation(
            Scalar::U16,
            ValueKind::Unary {
                operator: UnaryOperator::Complement,
                operand: integer(1, Scalar::U8),
            },
        ),
        operation(
            Scalar::U8,
            ValueKind::Binary {
                operator: BinaryOperator::Add,
                form: BinaryForm::Infix,
                left: integer(1, Scalar::U8),
                right: integer(1, Scalar::U16),
            },
        ),
        Value {
            span: None,
            ty: Scalar::U8.into(),
            kind: ValueKind::Binary {
                operator: BinaryOperator::Add,
                form: BinaryForm::Infix,
                left: integer(1, Scalar::U8),
                right: integer(1, Scalar::U8),
            },
        },
    ] {
        assert!(
            program(vec![value], integer(0, Scalar::Int))
                .verify()
                .unwrap_err()
                .to_string()
                .contains("invalid IR value")
        );
    }
}

#[test]
fn verification_rejects_invalid_control_flow() {
    let error = one_function(main_function(
        vec![],
        vec![],
        vec![Block {
            instructions: vec![],
            terminator: Terminator::Jump { target: BlockId(1) },
        }],
    ))
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("unknown block"));

    let error = one_function(main_function(
        vec![],
        vec![],
        vec![Block {
            instructions: vec![],
            terminator: Terminator::Branch {
                condition: integer(1, Scalar::Int),
                then_target: BlockId(0),
                else_target: BlockId(0),
            },
        }],
    ))
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("branch requires bool"));

    let error = one_function(main_function(
        vec![],
        vec![],
        vec![Block {
            instructions: vec![],
            terminator: Terminator::Unreachable,
        }],
    ))
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("not terminated"));
}

#[test]
fn verification_rejects_loads_before_definite_initialization() {
    let load = || Value {
        span: None,
        ty: Scalar::Int.into(),
        kind: ValueKind::Load(Place::Local(LocalId(0))),
    };
    let same_block = main_function(
        vec![load()],
        vec![Scalar::Int.into()],
        vec![Block {
            instructions: vec![
                Instruction::Value(ValueId(0)),
                Instruction::Store {
                    place: Place::Local(LocalId(0)),
                    operand: integer(1, Scalar::Int),
                },
            ],
            terminator: Terminator::Exit {
                status: integer(0, Scalar::Int),
            },
        }],
    );
    let missing_path = main_function(
        vec![load()],
        vec![Scalar::Int.into()],
        vec![
            Block {
                instructions: vec![],
                terminator: Terminator::Branch {
                    condition: integer(1, Scalar::Bool),
                    then_target: BlockId(1),
                    else_target: BlockId(2),
                },
            },
            Block {
                instructions: vec![Instruction::Store {
                    place: Place::Local(LocalId(0)),
                    operand: integer(1, Scalar::Int),
                }],
                terminator: Terminator::Jump { target: BlockId(3) },
            },
            Block {
                instructions: vec![],
                terminator: Terminator::Jump { target: BlockId(3) },
            },
            Block {
                instructions: vec![Instruction::Value(ValueId(0))],
                terminator: Terminator::Exit {
                    status: integer(0, Scalar::Int),
                },
            },
        ],
    );
    for function in [same_block, missing_path] {
        let error = one_function(function).verify().unwrap_err();
        assert!(error.to_string().contains("uninitialized local 0"));
    }
}

#[test]
fn arguments_initialize_the_parameters_before_the_entry_block() {
    let reads_parameter = Function {
        parameters: 1,
        result: Some(Scalar::Int.into()),
        values: vec![Value {
            span: None,
            ty: Scalar::Int.into(),
            kind: ValueKind::Load(Place::Local(LocalId(0))),
        }],
        flow: ControlFlow {
            entry: BlockId(0),
            locals: vec![Scalar::Int.into()],
            blocks: vec![Block {
                instructions: vec![Instruction::Value(ValueId(0))],
                terminator: Terminator::Return {
                    value: Some(Operand::Value(ValueId(0))),
                },
            }],
        },
    };
    two_functions(reads_parameter, vec![], vec![])
        .verify()
        .unwrap();
}

/// `main` runs `instructions` and exits; `callee` is `FunctionId(1)`.
fn two_functions(callee: Function, values: Vec<Value>, instructions: Vec<Instruction>) -> Program {
    Program {
        globals: vec![],
        functions: vec![
            main_function(
                values,
                vec![],
                vec![Block {
                    instructions,
                    terminator: Terminator::Exit {
                        status: integer(0, Scalar::Int),
                    },
                }],
            ),
            callee,
        ],
        main: FunctionId(0),
    }
}

fn callee(parameters: Vec<Scalar>, result: Option<Scalar>) -> Function {
    Function {
        parameters: parameters.len(),
        result: result.map(Type::from),
        values: vec![],
        flow: ControlFlow {
            entry: BlockId(0),
            locals: parameters.into_iter().map(Type::from).collect(),
            blocks: vec![Block {
                instructions: vec![],
                terminator: Terminator::Return {
                    value: result.map(|ty| integer(0, ty)),
                },
            }],
        },
    }
}

fn call(result: Option<ValueId>, function: usize, arguments: Vec<Operand>) -> Instruction {
    Instruction::Call {
        result,
        function: FunctionId(function),
        arguments,
        span: 0..1,
    }
}

fn call_result(ty: Scalar) -> Value {
    Value {
        span: None,
        ty: ty.into(),
        kind: ValueKind::CallResult,
    }
}

#[test]
fn verification_checks_call_targets_arity_and_argument_types() {
    // A well-formed call to a two-parameter function.
    two_functions(
        callee(vec![Scalar::U8, Scalar::Int], None),
        vec![],
        vec![call(
            None,
            1,
            vec![integer(1, Scalar::U8), integer(2, Scalar::Int)],
        )],
    )
    .verify()
    .unwrap();

    for (instruction, expected) in [
        (call(None, 7, vec![]), "unknown function 7"),
        (
            call(None, 1, vec![integer(1, Scalar::U8)]),
            "passes 1 argument",
        ),
        (
            call(
                None,
                1,
                vec![
                    integer(1, Scalar::U8),
                    integer(2, Scalar::Int),
                    integer(3, Scalar::Int),
                ],
            ),
            "passes 3 arguments",
        ),
        (
            call(
                None,
                1,
                vec![integer(1, Scalar::Int), integer(2, Scalar::Int)],
            ),
            "argument 0 has type",
        ),
    ] {
        let error = two_functions(
            callee(vec![Scalar::U8, Scalar::Int], None),
            vec![],
            vec![instruction],
        )
        .verify()
        .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn verification_checks_every_signature_before_any_call() {
    // The callee promises two parameters but names only one local, so the
    // caller's argument check has no parameter type to read.
    let mut short_of_locals = callee(vec![Scalar::Int], None);
    short_of_locals.parameters = 2;
    let error = two_functions(
        short_of_locals,
        vec![],
        vec![call(
            None,
            1,
            vec![integer(1, Scalar::Int), integer(2, Scalar::Int)],
        )],
    )
    .verify()
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("declares 2 parameters but only 1 locals"),
        "{error}"
    );
}

#[test]
fn verification_checks_call_results_against_the_callee_signature() {
    // A void call defines nothing; a value call defines its result.
    two_functions(callee(vec![], None), vec![], vec![call(None, 1, vec![])])
        .verify()
        .unwrap();
    two_functions(
        callee(vec![], Some(Scalar::Int)),
        vec![call_result(Scalar::Int)],
        vec![call(Some(ValueId(0)), 1, vec![])],
    )
    .verify()
    .unwrap();

    let error = two_functions(
        callee(vec![], None),
        vec![call_result(Scalar::Int)],
        vec![call(Some(ValueId(0)), 1, vec![])],
    )
    .verify()
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("defines a result for void function")
    );

    let error = two_functions(
        callee(vec![], Some(Scalar::Int)),
        vec![],
        vec![call(None, 1, vec![])],
    )
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("discards the result"));

    let error = two_functions(
        callee(vec![], Some(Scalar::Int)),
        vec![call_result(Scalar::U8)],
        vec![call(Some(ValueId(0)), 1, vec![])],
    )
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("call result has type"));
}

#[test]
fn verification_ties_call_results_to_their_defining_call() {
    // A `CallResult` value that no call defines.
    let error = one_function(main_function(
        vec![call_result(Scalar::Int)],
        vec![],
        vec![Block {
            instructions: vec![Instruction::Value(ValueId(0))],
            terminator: Terminator::Exit {
                status: integer(0, Scalar::Int),
            },
        }],
    ))
    .verify()
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("disagree on being a call result")
    );

    // A call defining a value that is not a `CallResult`.
    let error = two_functions(
        callee(vec![], Some(Scalar::Int)),
        vec![convert(integer(1, Scalar::Int), Scalar::Int)],
        vec![call(Some(ValueId(0)), 1, vec![])],
    )
    .verify()
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("disagree on being a call result")
    );
}

#[test]
fn verification_checks_return_forms_against_the_declared_result() {
    let returning = |result, value| {
        two_functions(
            Function {
                parameters: 0,
                result,
                values: vec![],
                flow: ControlFlow {
                    entry: BlockId(0),
                    locals: vec![],
                    blocks: vec![Block {
                        instructions: vec![],
                        terminator: Terminator::Return { value },
                    }],
                },
            },
            vec![],
            vec![],
        )
        .verify()
    };
    returning(None, None).unwrap();
    returning(Some(Scalar::Int.into()), Some(integer(0, Scalar::Int))).unwrap();

    for (result, value, expected) in [
        (
            None,
            Some(integer(0, Scalar::Int)),
            "returns a value from a void function",
        ),
        (Some(Scalar::Int), None, "return is missing its value"),
        (
            Some(Scalar::Int),
            Some(integer(0, Scalar::U8)),
            "internal compiler error: IR returns",
        ),
    ] {
        let error = returning(result.map(Type::from), value).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn verification_rejects_a_value_function_that_falls_off_its_end() {
    let error = two_functions(
        Function {
            parameters: 0,
            result: Some(Scalar::Int.into()),
            values: vec![],
            flow: ControlFlow {
                entry: BlockId(0),
                locals: vec![],
                blocks: vec![Block {
                    instructions: vec![],
                    terminator: Terminator::Unreachable,
                }],
            },
        },
        vec![],
        vec![],
    )
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("not terminated"));
}

#[test]
fn verification_checks_main_globals_and_global_places() {
    let error = one_function(Function {
        parameters: 1,
        result: None,
        values: vec![],
        flow: ControlFlow {
            entry: BlockId(0),
            locals: vec![Scalar::Int.into()],
            blocks: vec![Block {
                instructions: vec![],
                terminator: Terminator::Return { value: None },
            }],
        },
    })
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("IR main takes parameters"));

    let stores_to_global = |globals: Vec<Global>| Program {
        globals,
        functions: vec![main_function(
            vec![],
            vec![],
            vec![Block {
                instructions: vec![Instruction::Store {
                    place: Place::Global(GlobalId(0)),
                    operand: integer(1, Scalar::Int),
                }],
                terminator: Terminator::Exit {
                    status: integer(0, Scalar::Int),
                },
            }],
        )],
        main: FunctionId(0),
    };
    stores_to_global(vec![Global {
        ty: Scalar::Int.into(),
        values: vec![0],
    }])
    .verify()
    .unwrap();

    let error = stores_to_global(vec![]).verify().unwrap_err();
    assert!(error.to_string().contains("unknown global 0"));

    let error = stores_to_global(vec![Global {
        ty: Scalar::U8.into(),
        values: vec![0],
    }])
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("IR store has type"));

    let error = Program {
        globals: vec![Global {
            ty: Scalar::U8.into(),
            values: vec![256],
        }],
        functions: vec![main_function(
            vec![],
            vec![],
            vec![Block {
                instructions: vec![],
                terminator: Terminator::Exit {
                    status: integer(0, Scalar::Int),
                },
            }],
        )],
        main: FunctionId(0),
    }
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("out of range"), "{error}");
}
