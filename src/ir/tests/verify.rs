use super::*;
use crate::types::{ComparisonOperator, MAX_STRUCT_CONTAINMENT_DEPTH, UnaryOperator};

#[test]
fn verification_checks_slice_literals_places_and_values() {
    let ints = slice(false, Scalar::Int.into());
    let values = vec![
        Value {
            span: None,
            ty: ints.clone(),
            kind: ValueKind::Load(Place::Local(LocalId(0))),
        },
        Value {
            span: Some(0..1),
            ty: ints.clone(),
            kind: ValueKind::SliceRange {
                slice: Operand::Value(ValueId(0)),
                low: integer(0, Scalar::Int),
                high: integer(1, Scalar::Int),
            },
        },
        Value {
            span: None,
            ty: Scalar::Int.into(),
            kind: ValueKind::Load(Place::SliceElement {
                slice: Operand::Value(ValueId(1)),
                index: integer(0, Scalar::Int),
                span: 0..1,
            }),
        },
        Value {
            span: None,
            ty: slice(true, Scalar::Int.into()),
            kind: ValueKind::WholeSlice(Place::Local(LocalId(1))),
        },
        Value {
            span: None,
            ty: Scalar::Int.into(),
            kind: ValueKind::SliceLength {
                slice: Operand::Value(ValueId(3)),
            },
        },
    ];
    let function = main_function(
        values,
        vec![ints.clone(), array(2, Scalar::Int.into())],
        vec![Block {
            instructions: vec![
                Instruction::Store {
                    place: Place::Local(LocalId(0)),
                    operand: empty_slice(ints.clone()),
                },
                Instruction::Value(ValueId(0)),
                Instruction::Value(ValueId(1)),
                Instruction::Value(ValueId(2)),
                Instruction::Value(ValueId(3)),
                Instruction::Value(ValueId(4)),
            ],
            terminator: Terminator::Exit {
                status: integer(0, Scalar::Int),
            },
        }],
    );
    one_function(function).verify().unwrap();
}

#[test]
fn verification_rejects_malformed_slice_values_and_places() {
    let ints = slice(false, Scalar::Int.into());
    let bytes = slice(false, Scalar::U8.into());

    let error = program(vec![], empty_slice(Scalar::Int.into()))
        .verify()
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("IR empty slice has non-slice type `int`"),
        "{error}"
    );

    for (value, expected) in [
        (
            Value {
                span: None,
                ty: Scalar::Int.into(),
                kind: ValueKind::Load(Place::SliceElement {
                    slice: integer(0, Scalar::Int),
                    index: integer(0, Scalar::Int),
                    span: 0..1,
                }),
            },
            "IR indexes slice value `int`",
        ),
        (
            Value {
                span: None,
                ty: Scalar::Int.into(),
                kind: ValueKind::Load(Place::SliceElement {
                    slice: empty_slice(ints.clone()),
                    index: integer(0, Scalar::U8),
                    span: 0..1,
                }),
            },
            "IR index has type `u8`, expected `int`",
        ),
        (
            Value {
                span: None,
                ty: Scalar::Int.into(),
                kind: ValueKind::SliceLength {
                    slice: integer(0, Scalar::Int),
                },
            },
            "invalid IR value 0",
        ),
        (
            Value {
                span: Some(0..1),
                ty: ints.clone(),
                kind: ValueKind::SliceRange {
                    slice: empty_slice(ints.clone()),
                    low: integer(0, Scalar::U8),
                    high: integer(1, Scalar::Int),
                },
            },
            "invalid IR value 0",
        ),
        (
            Value {
                span: Some(0..1),
                ty: bytes.clone(),
                kind: ValueKind::SliceRange {
                    slice: empty_slice(ints.clone()),
                    low: integer(0, Scalar::Int),
                    high: integer(1, Scalar::Int),
                },
            },
            "invalid IR value 0",
        ),
        (
            Value {
                span: Some(0..1),
                ty: Scalar::Bool.into(),
                kind: ValueKind::Comparison {
                    operator: ComparisonOperator::Equal,
                    left: empty_slice(ints.clone()),
                    right: empty_slice(bytes.clone()),
                },
            },
            "invalid IR value 0",
        ),
        (
            Value {
                span: Some(0..1),
                ty: Scalar::Bool.into(),
                kind: ValueKind::Comparison {
                    operator: ComparisonOperator::Equal,
                    left: empty_slice(ints.clone()),
                    right: empty_slice(ints.clone()),
                },
            },
            "invalid IR value 0",
        ),
    ] {
        let error = program(vec![value], integer(0, Scalar::Int))
            .verify()
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }

    let mismatched_whole = Value {
        span: None,
        ty: bytes.clone(),
        kind: ValueKind::WholeSlice(Place::Local(LocalId(0))),
    };
    let function = main_function(
        vec![mismatched_whole],
        vec![array(2, Scalar::Int.into())],
        vec![Block {
            instructions: vec![Instruction::Value(ValueId(0))],
            terminator: Terminator::Exit {
                status: integer(0, Scalar::Int),
            },
        }],
    );
    let error = one_function(function).verify().unwrap_err();
    assert!(error.to_string().contains("invalid IR value 0"), "{error}");
}

#[test]
fn verification_rejects_equality_on_aggregates_that_contain_slices() {
    let ints = slice(false, Scalar::Int.into());
    let array_of_slices = array(2, ints.clone());
    let values = vec![
        Value {
            span: None,
            ty: array_of_slices.clone(),
            kind: ValueKind::Load(Place::Local(LocalId(0))),
        },
        Value {
            span: Some(0..1),
            ty: Scalar::Bool.into(),
            kind: ValueKind::Comparison {
                operator: ComparisonOperator::Equal,
                left: Operand::Value(ValueId(0)),
                right: Operand::Value(ValueId(0)),
            },
        },
    ];
    let function = main_function(
        values,
        vec![array_of_slices],
        vec![Block {
            instructions: vec![
                Instruction::Store {
                    place: element(Place::Local(LocalId(0)), integer(0, Scalar::Int)),
                    operand: empty_slice(ints.clone()),
                },
                Instruction::Value(ValueId(0)),
                Instruction::Value(ValueId(1)),
            ],
            terminator: Terminator::Exit {
                status: integer(0, Scalar::Int),
            },
        }],
    );
    let error = one_function(function).verify().unwrap_err();
    assert!(error.to_string().contains("invalid IR value 1"), "{error}");

    let holder = declared(0, "Holder");
    let values = vec![
        Value {
            span: None,
            ty: holder.clone(),
            kind: ValueKind::Load(Place::Local(LocalId(0))),
        },
        Value {
            span: Some(0..1),
            ty: Scalar::Bool.into(),
            kind: ValueKind::Comparison {
                operator: ComparisonOperator::Equal,
                left: Operand::Value(ValueId(0)),
                right: Operand::Value(ValueId(0)),
            },
        },
    ];
    let function = main_function(
        values,
        vec![holder],
        vec![Block {
            instructions: vec![
                Instruction::Store {
                    place: field(Place::Local(LocalId(0)), 0),
                    operand: empty_slice(ints.clone()),
                },
                Instruction::Value(ValueId(0)),
                Instruction::Value(ValueId(1)),
            ],
            terminator: Terminator::Exit {
                status: integer(0, Scalar::Int),
            },
        }],
    );
    let error = with_structs(vec![Struct { fields: vec![ints] }], function)
        .verify()
        .unwrap_err();
    assert!(error.to_string().contains("invalid IR value 1"), "{error}");
}

#[test]
fn verification_accepts_pointers_to_non_comparable_types() {
    let pointer = Type::Pointer {
        constant: false,
        target: Box::new(slice(false, Scalar::Int.into())),
    };
    let holder = declared(0, "Holder");
    let values = vec![
        Value {
            span: Some(0..1),
            ty: Scalar::Bool.into(),
            kind: ValueKind::Comparison {
                operator: ComparisonOperator::Equal,
                left: Operand::Literal(Literal::Null(pointer.clone())),
                right: Operand::Literal(Literal::Null(pointer.clone())),
            },
        },
        Value {
            span: None,
            ty: holder.clone(),
            kind: ValueKind::Load(Place::Local(LocalId(0))),
        },
        Value {
            span: Some(0..1),
            ty: Scalar::Bool.into(),
            kind: ValueKind::Comparison {
                operator: ComparisonOperator::Equal,
                left: Operand::Value(ValueId(1)),
                right: Operand::Value(ValueId(1)),
            },
        },
    ];
    let function = main_function(
        values,
        vec![holder],
        vec![Block {
            instructions: vec![
                Instruction::Store {
                    place: field(Place::Local(LocalId(0)), 0),
                    operand: Operand::Literal(Literal::Null(pointer.clone())),
                },
                Instruction::Value(ValueId(0)),
                Instruction::Value(ValueId(1)),
                Instruction::Value(ValueId(2)),
            ],
            terminator: Terminator::Exit {
                status: integer(0, Scalar::Int),
            },
        }],
    );

    with_structs(
        vec![Struct {
            fields: vec![pointer],
        }],
        function,
    )
    .verify()
    .unwrap();
}

#[test]
fn verification_rejects_non_pointer_nulls_and_indirect_places() {
    let invalid_null = Global {
        ty: Scalar::Int.into(),
        values: vec![Literal::Null(Scalar::Int.into())],
    };
    let mut malformed = one_function(main_function(
        vec![],
        vec![],
        vec![Block {
            instructions: vec![],
            terminator: Terminator::Exit {
                status: integer(0, Scalar::Int),
            },
        }],
    ));
    malformed.globals = vec![invalid_null];
    assert!(
        malformed
            .verify()
            .unwrap_err()
            .to_string()
            .contains("IR null has non-pointer type `int`")
    );

    let invalid_indirect = Value {
        span: None,
        ty: Scalar::Int.into(),
        kind: ValueKind::Load(Place::Indirect {
            pointer: integer(0, Scalar::Int),
            span: 0..1,
        }),
    };
    let error = program(vec![invalid_indirect], integer(0, Scalar::Int))
        .verify()
        .unwrap_err();
    assert!(
        error.to_string().contains("IR dereferences `int`"),
        "{error}"
    );
}

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
                    left: rows.clone(),
                    right: rows.clone(),
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
                        left: rows.clone(),
                        right: rows.clone(),
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
            left: rows.clone(),
            right: rows.clone(),
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
            integers([1], Scalar::Int),
            "holds 1 values, expected 2",
        ),
        (
            array(2, array(2, Scalar::Int.into())),
            integers([1, 2, 3, 4, 5], Scalar::Int),
            "holds 5 values, expected 4",
        ),
        (
            array(2, Scalar::U8.into()),
            integers([0, 256], Scalar::U8),
            "IR integer 256 out of range",
        ),
        (
            array(2, Scalar::Int.into()),
            integers([1, 2], Scalar::I64),
            "holds a `i64` value",
        ),
        (
            array(2, Scalar::F64.into()),
            integers([1, 2], Scalar::Int),
            "holds a `int` value",
        ),
        (Scalar::F32.into(), doubles([1.0]), "holds a `f64` value"),
        (
            Scalar::F32.into(),
            integers([1], Scalar::F32),
            "IR integer 1 has floating-point type F32",
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
fn verification_rejects_an_aggregate_too_large_for_backend_layout() {
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
    program.globals = vec![Global {
        ty: array(1 << 60, Scalar::Int.into()),
        values: vec![],
    }];
    let error = program.verify().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("IR aggregate layout exceeds compiler limit of 1 PiB"),
        "{error}"
    );
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
                                        operand: operand.clone(),
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
fn verification_admits_only_nontruncating_pointer_to_uint_conversion() {
    let pointer = Type::Pointer {
        constant: false,
        target: Box::new(Scalar::Int.into()),
    };
    let conversion = |destination, truncating| Value {
        span: None,
        ty: destination,
        kind: ValueKind::Convert {
            operand: Operand::Literal(Literal::Null(pointer.clone())),
            truncating,
        },
    };
    program(
        vec![conversion(Scalar::Uint.into(), false)],
        integer(0, Scalar::Int),
    )
    .verify()
    .unwrap();
    for (destination, truncating) in [
        (Scalar::Uint.into(), true),
        (Scalar::Int.into(), false),
        (pointer.clone(), false),
    ] {
        assert!(
            program(
                vec![conversion(destination, truncating)],
                integer(0, Scalar::Int),
            )
            .verify()
            .is_err()
        );
    }
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
        structs: vec![],
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
        structs: vec![],
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
        values: integers([0], Scalar::Int),
    }])
    .verify()
    .unwrap();

    let error = stores_to_global(vec![]).verify().unwrap_err();
    assert!(error.to_string().contains("unknown global 0"));

    let error = stores_to_global(vec![Global {
        ty: Scalar::U8.into(),
        values: integers([0], Scalar::U8),
    }])
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("IR store has type"));

    let error = Program {
        structs: vec![],
        globals: vec![Global {
            ty: Scalar::U8.into(),
            values: integers([256], Scalar::U8),
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

/// One operation value with the span a trapping operation carries.
fn operation(ty: Scalar, kind: ValueKind) -> Value {
    Value {
        span: Some(0..1),
        ty: ty.into(),
        kind,
    }
}

fn binary(ty: Scalar, operator: BinaryOperator, left: Operand, right: Operand) -> Value {
    operation(
        ty,
        ValueKind::Binary {
            operator,
            form: BinaryForm::Infix,
            left,
            right,
        },
    )
}

fn unary(ty: Scalar, operator: UnaryOperator, operand: Operand) -> Value {
    operation(ty, ValueKind::Unary { operator, operand })
}

fn double(value: f64) -> Operand {
    floating(Float::Binary64(value.to_bits()))
}

fn single(value: f32) -> Operand {
    floating(Float::Binary32(value.to_bits()))
}

/// The zero of a scalar type, spelled as the literal that holds it.
fn zero(ty: Scalar) -> Operand {
    match ty {
        Scalar::F32 => single(0.0),
        Scalar::F64 => double(0.0),
        _ => integer(0, ty),
    }
}

#[test]
fn verification_admits_only_the_floating_point_operations() {
    let accepted = [
        binary(Scalar::F64, BinaryOperator::Add, double(1.0), double(2.0)),
        binary(
            Scalar::F64,
            BinaryOperator::Subtract,
            double(1.0),
            double(2.0),
        ),
        binary(
            Scalar::F32,
            BinaryOperator::Multiply,
            single(1.0),
            single(2.0),
        ),
        binary(
            Scalar::F32,
            BinaryOperator::Divide,
            single(1.0),
            single(2.0),
        ),
        unary(Scalar::F64, UnaryOperator::Negate, double(1.0)),
        operation(
            Scalar::Bool,
            ValueKind::Comparison {
                operator: ComparisonOperator::Less,
                left: double(1.0),
                right: double(2.0),
            },
        ),
    ];
    for value in accepted {
        program(vec![value.clone()], integer(0, Scalar::Int))
            .verify()
            .unwrap_or_else(|error| panic!("{error}: {value:?}"));
    }
    let rejected = [
        // `%`, the wrapping operators, the bitwise operators, and the shifts
        // read integers.
        binary(
            Scalar::F64,
            BinaryOperator::Remainder,
            double(1.0),
            double(2.0),
        ),
        binary(
            Scalar::F64,
            BinaryOperator::WrappingAdd,
            double(1.0),
            double(2.0),
        ),
        binary(Scalar::F64, BinaryOperator::And, double(1.0), double(2.0)),
        binary(
            Scalar::F64,
            BinaryOperator::ShiftLeft,
            double(1.0),
            integer(1, Scalar::Int),
        ),
        unary(Scalar::F64, UnaryOperator::WrappingNegate, double(1.0)),
        unary(Scalar::F64, UnaryOperator::Complement, double(1.0)),
        // An operation takes one format, and no operation mixes a
        // floating-point operand with an integer one.
        binary(Scalar::F32, BinaryOperator::Add, single(1.0), double(2.0)),
        binary(Scalar::F64, BinaryOperator::Add, double(1.0), single(2.0)),
        binary(
            Scalar::F64,
            BinaryOperator::Add,
            double(1.0),
            integer(1, Scalar::Int),
        ),
        binary(
            Scalar::Int,
            BinaryOperator::Add,
            integer(1, Scalar::Int),
            double(1.0),
        ),
        unary(Scalar::F32, UnaryOperator::Negate, double(1.0)),
    ];
    for value in rejected {
        let error = program(vec![value.clone()], integer(0, Scalar::Int))
            .verify()
            .unwrap_err();
        assert!(
            error.to_string().contains("invalid IR value 0"),
            "{error}: {value:?}"
        );
    }
}

#[test]
fn verification_checks_conversions_between_the_numeric_categories() {
    // A conversion that cannot lose its value needs no span; every other one
    // carries the span its trap reports.
    for (source, destination, span) in [
        (Scalar::F32, Scalar::F64, None),
        (Scalar::U8, Scalar::F32, None),
        (Scalar::U32, Scalar::F64, None),
        (Scalar::F64, Scalar::F32, Some(0..1)),
        (Scalar::F64, Scalar::I32, Some(0..1)),
        (Scalar::I32, Scalar::F32, Some(0..1)),
        (Scalar::U64, Scalar::F64, Some(0..1)),
        (Scalar::F32, Scalar::F32, None),
    ] {
        let value = Value {
            span,
            ty: destination.into(),
            kind: ValueKind::Convert {
                operand: zero(source),
                truncating: false,
            },
        };
        program(vec![value], integer(0, Scalar::Int))
            .verify()
            .unwrap_or_else(|error| panic!("{error}: {source} to {destination}"));
    }
    for (source, destination, truncating, span) in [
        // A conversion that can lose its value must be reportable.
        (Scalar::F64, Scalar::F32, false, None),
        (Scalar::F64, Scalar::I32, false, None),
        (Scalar::I32, Scalar::F32, false, None),
        // Truncation reinterprets a bit pattern, which no floating-point
        // value has.
        (Scalar::F64, Scalar::I32, true, Some(0..1)),
        (Scalar::I32, Scalar::F64, true, Some(0..1)),
        (Scalar::F64, Scalar::F32, true, Some(0..1)),
        // `bool` is not a number.
        (Scalar::F64, Scalar::Bool, false, Some(0..1)),
        (Scalar::Bool, Scalar::F64, false, Some(0..1)),
    ] {
        let value = Value {
            span,
            ty: destination.into(),
            kind: ValueKind::Convert {
                operand: zero(source),
                truncating,
            },
        };
        let error = program(vec![value], integer(0, Scalar::Int))
            .verify()
            .unwrap_err();
        assert!(
            error.to_string().contains("invalid IR value 0"),
            "{error}: {source} to {destination}"
        );
    }
}

#[test]
fn verification_rejects_an_integer_literal_of_a_floating_point_type() {
    let error = program(
        vec![convert(integer(1, Scalar::F32), Scalar::F32)],
        integer(0, Scalar::Int),
    )
    .verify()
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("IR integer 1 has floating-point type F32"),
        "{error}"
    );
}

/// `Point { x: int, y: u8 }` and `Cell { point: Point, weights: [2]f64 }`.
fn point_and_cell() -> Vec<Struct> {
    vec![
        Struct {
            fields: vec![Scalar::Int.into(), Scalar::U8.into()],
        },
        Struct {
            fields: vec![declared(0, "Point"), array(2, Scalar::F64.into())],
        },
    ]
}

/// A `main` that stores into `place` and then runs `values`, over the struct
/// table `point_and_cell` defines and one `Cell` local.
fn cell_program(place: Place, values: Vec<Value>) -> Program {
    let blocks = vec![Block {
        instructions: std::iter::once(Instruction::Store {
            place,
            operand: integer(0, Scalar::Int),
        })
        .chain((0..values.len()).map(|id| Instruction::Value(ValueId(id))))
        .collect(),
        terminator: Terminator::Exit {
            status: integer(0, Scalar::Int),
        },
    }];
    with_structs(
        point_and_cell(),
        main_function(values, vec![declared(1, "Cell")], blocks),
    )
}

#[test]
fn verification_accepts_a_field_place_nested_through_a_struct_and_an_array() {
    let cell = Place::Local(LocalId(0));
    let x = field(field(cell.clone(), 0), 0);
    let weight = element(field(cell, 1), integer(1, Scalar::Int));
    cell_program(
        x.clone(),
        vec![
            Value {
                span: None,
                ty: Scalar::Int.into(),
                kind: ValueKind::Load(x),
            },
            Value {
                span: None,
                ty: Scalar::F64.into(),
                kind: ValueKind::Load(weight),
            },
        ],
    )
    .verify()
    .unwrap();
}

#[test]
fn verification_rejects_invalid_field_places() {
    let cell = Place::Local(LocalId(0));
    let load = |ty: Type, place| Value {
        span: None,
        ty,
        kind: ValueKind::Load(place),
    };
    for (place, expected) in [
        // A field of the `[2]f64` field, and a field of one of its elements.
        (field(field(cell.clone(), 1), 0), "IR selects a field of"),
        (
            field(element(field(cell.clone(), 1), integer(0, Scalar::Int)), 0),
            "IR selects a field of `f64`",
        ),
        (field(cell.clone(), 2), "IR selects field 2 of `Cell`"),
    ] {
        let error = cell_program(
            field(field(cell.clone(), 0), 0),
            vec![load(Scalar::Int.into(), place)],
        )
        .verify()
        .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }

    // A field place types the store through it, so storing the wrong type is
    // rejected the same way an element store is.
    let error = with_structs(
        point_and_cell(),
        main_function(
            vec![],
            vec![declared(0, "Point")],
            vec![Block {
                instructions: vec![Instruction::Store {
                    place: field(Place::Local(LocalId(0)), 1),
                    operand: integer(0, Scalar::Int),
                }],
                terminator: Terminator::Exit {
                    status: integer(0, Scalar::Int),
                },
            }],
        ),
    )
    .verify()
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("IR store has type `int`, expected `u8`"),
        "{error}"
    );
}

#[test]
fn verification_rejects_a_type_naming_a_struct_the_program_does_not_define() {
    let exit = vec![Block {
        instructions: vec![],
        terminator: Terminator::Exit {
            status: integer(0, Scalar::Int),
        },
    }];
    // A local, a global, a value, and a field of a defined struct each reach
    // type validation, so an unknown struct cannot arrive through any of them.
    let unknown = array(2, declared(7, "Missing"));
    let error = one_function(main_function(vec![], vec![unknown.clone()], exit.clone()))
        .verify()
        .unwrap_err();
    assert!(error.to_string().contains("unknown struct 7"), "{error}");

    let error = Program {
        structs: vec![],
        globals: vec![Global {
            ty: declared(0, "Missing"),
            values: vec![],
        }],
        functions: vec![main_function(vec![], vec![], exit.clone())],
        main: FunctionId(0),
    }
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("unknown struct 0"), "{error}");

    let error = with_structs(
        vec![Struct {
            fields: vec![unknown],
        }],
        main_function(vec![], vec![], exit),
    )
    .verify()
    .unwrap_err();
    assert!(error.to_string().contains("unknown struct 7"), "{error}");
}

#[test]
fn verification_rejects_a_struct_table_whose_fields_form_a_cycle() {
    let exit = vec![Block {
        instructions: vec![],
        terminator: Terminator::Exit {
            status: integer(0, Scalar::Int),
        },
    }];
    for structs in [
        // A struct holding itself, and two structs holding each other through
        // an array, which stores its elements inline just as a field does.
        vec![Struct {
            fields: vec![declared(0, "Loop")],
        }],
        vec![
            Struct {
                fields: vec![array(2, declared(1, "Second"))],
            },
            Struct {
                fields: vec![declared(0, "First")],
            },
        ],
    ] {
        let error = with_structs(structs, main_function(vec![], vec![], exit.clone()))
            .verify()
            .unwrap_err();
        assert!(error.to_string().contains("contains itself"), "{error}");
    }
}

#[test]
fn verification_limits_deep_acyclic_struct_tables() {
    let structs = (0..=MAX_STRUCT_CONTAINMENT_DEPTH)
        .map(|index| Struct {
            fields: if index == MAX_STRUCT_CONTAINMENT_DEPTH {
                vec![Scalar::Int.into()]
            } else {
                vec![declared(index + 1, &format!("S{}", index + 1))]
            },
        })
        .collect();
    let error = with_structs(
        structs,
        main_function(
            vec![],
            vec![],
            vec![Block {
                instructions: vec![],
                terminator: Terminator::Exit {
                    status: integer(0, Scalar::Int),
                },
            }],
        ),
    )
    .verify()
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("IR struct containment exceeds compiler limit of 128"),
        "{error}"
    );
}

#[test]
fn verification_checks_a_struct_global_against_its_scalar_types_in_field_order() {
    let cell = |values: Vec<Literal>| Program {
        structs: point_and_cell(),
        globals: vec![Global {
            ty: declared(1, "Cell"),
            values,
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
    };
    let x = Literal::Integer {
        value: 1,
        ty: Scalar::Int,
    };
    let y = Literal::Integer {
        value: 2,
        ty: Scalar::U8,
    };
    let weights = doubles([0.5, 1.5]);
    let complete = || vec![x.clone(), y.clone(), weights[0].clone(), weights[1].clone()];

    cell(complete()).verify().unwrap();

    for (values, expected) in [
        (
            vec![x.clone(), y.clone(), weights[0].clone()],
            "holds 3 values, expected 4",
        ),
        (
            [complete(), vec![weights[1].clone()]].concat(),
            "holds 5 values, expected 4",
        ),
        // The scalars are heterogeneous, so their order is part of the check.
        (
            vec![y.clone(), x.clone(), weights[0].clone(), weights[1].clone()],
            "holds a `u8` value where a `int` value belongs",
        ),
        (
            vec![
                x,
                Literal::Integer {
                    value: 256,
                    ty: Scalar::U8,
                },
                weights[0].clone(),
                weights[1].clone(),
            ],
            "IR integer 256 out of range",
        ),
    ] {
        let error = cell(values).verify().unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}
