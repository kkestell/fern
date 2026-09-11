use crate::{
    frontend::parser,
    ir::{lower::lower, model::*, verify::VerifiedProgram},
    semantic::namespaces,
    types::{BinaryOperator, ComparisonOperator, Float, Scalar, StructId, StructType, Type},
};

pub(super) fn integer(value: i128, ty: Scalar) -> Operand {
    Operand::Literal(Literal::Integer { value, ty })
}

pub(super) fn floating(value: Float) -> Operand {
    Operand::Literal(Literal::Floating(value))
}

/// The integer literals a global of leaf type `ty` holds, in memory order.
pub(super) fn integers(values: impl IntoIterator<Item = i128>, ty: Scalar) -> Vec<Literal> {
    values
        .into_iter()
        .map(|value| Literal::Integer { value, ty })
        .collect()
}

/// The floating-point literals a global of leaf type `f64` holds.
pub(super) fn doubles(values: impl IntoIterator<Item = f64>) -> Vec<Literal> {
    values
        .into_iter()
        .map(|value| Literal::Floating(Float::Binary64(value.to_bits())))
        .collect()
}

pub(super) fn convert(operand: Operand, ty: Scalar) -> Value {
    Value {
        span: None,
        ty: ty.into(),
        kind: ValueKind::Convert {
            operand,
            truncating: false,
        },
    }
}

/// A whole-program wrapper around a single `main` whose block runs `values`
/// in order and then exits.
pub(super) fn program(values: Vec<Value>, exit: Operand) -> Program {
    let blocks = vec![Block {
        instructions: (0..values.len())
            .map(|id| Instruction::Value(ValueId(id)))
            .collect(),
        terminator: Terminator::Exit { status: exit },
    }];
    one_function(main_function(values, vec![], blocks))
}

pub(super) fn main_function(values: Vec<Value>, locals: Vec<Type>, blocks: Vec<Block>) -> Function {
    Function {
        parameters: 0,
        result: None,
        values,
        flow: ControlFlow {
            entry: BlockId(0),
            locals,
            blocks,
        },
    }
}

pub(super) fn one_function(main: Function) -> Program {
    with_structs(vec![], main)
}

/// A whole program of one `main` over a struct table.
pub(super) fn with_structs(structs: Vec<Struct>, main: Function) -> Program {
    Program {
        structs,
        globals: vec![],
        functions: vec![main],
        main: FunctionId(0),
    }
}

pub(super) fn lowered(source: &str) -> VerifiedProgram {
    let syntax = parser::parse(&crate::source::SourceMap::from_text(source)).unwrap();
    lower(namespaces::check_root(&syntax).unwrap())
        .verify()
        .unwrap()
}

/// Loads, checks, lowers, and verifies a tree of `(relative path, source)`
/// files whose root module is `app`, so multi-module lowering runs through
/// real import resolution.
pub(super) fn lowered_tree<'a>(
    files: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> VerifiedProgram {
    let dir = crate::module::tree(files);
    let program = crate::module::load(&dir.path().join("app"), &[dir.path().to_owned()])
        .unwrap_or_else(|error| panic!("{}", error.into_compile_error()));
    lower(
        namespaces::check(&program.syntax, &program.modules, &program.files)
            .unwrap_or_else(|error| panic!("{}", error.render(&program.sources))),
    )
    .verify()
    .unwrap()
}

pub(super) fn main_of(program: &VerifiedProgram) -> &Function {
    main_of_program(program.program())
}

pub(super) fn main_of_program(program: &Program) -> &Function {
    &program.functions[program.main.0]
}

pub(super) fn instructions(function: &Function) -> impl Iterator<Item = &Instruction> {
    function
        .flow
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
}

/// Every function `function` calls, in instruction order.
pub(super) fn call_targets(function: &Function) -> Vec<FunctionId> {
    instructions(function)
        .filter_map(|instruction| match instruction {
            Instruction::Call { function, .. } => Some(*function),
            _ => None,
        })
        .collect()
}

pub(super) fn array(length: u64, element: Type) -> Type {
    Type::Array {
        length,
        element: Box::new(element),
    }
}

/// The type of struct `id`, spelled `name` the way diagnostics show it.
pub(super) fn declared(id: usize, name: &str) -> Type {
    Type::Struct(StructType {
        id: StructId(id),
        name: name.to_owned(),
    })
}

pub(super) fn element(base: Place, index: Operand) -> Place {
    Place::Element {
        base: Box::new(base),
        index,
        span: 0..1,
    }
}

pub(super) fn field(base: Place, ordinal: usize) -> Place {
    Place::Field {
        base: Box::new(base),
        ordinal,
    }
}

/// Spells a place the way a test reads it: `local0[1].2`, `global2`.
pub(super) fn describe(place: &Place) -> String {
    match place {
        Place::Local(LocalId(id)) => format!("local{id}"),
        Place::Global(GlobalId(id)) => format!("global{id}"),
        Place::Element { base, index, .. } => {
            let index = match index {
                Operand::Literal(Literal::Integer { value, .. }) => value.to_string(),
                Operand::Literal(Literal::Floating(_)) => {
                    unreachable!("an index has type `int`")
                }
                Operand::Literal(Literal::Null(_)) => unreachable!("an index has type `int`"),
                Operand::Value(ValueId(id)) => format!("v{id}"),
            };
            format!("{}[{index}]", describe(base))
        }
        Place::Field { base, ordinal } => format!("{}.{ordinal}", describe(base)),
        Place::Indirect { .. } => "*pointer".to_owned(),
    }
}

pub(super) fn stored_places(function: &Function) -> Vec<String> {
    stores(function)
        .iter()
        .map(|(place, _)| describe(place))
        .collect()
}

pub(super) fn stores(function: &Function) -> Vec<(Place, Operand)> {
    instructions(function)
        .filter_map(|instruction| match instruction {
            Instruction::Store { place, operand } => Some((place.clone(), operand.clone())),
            _ => None,
        })
        .collect()
}

pub(super) fn loads(function: &Function) -> Vec<Place> {
    function
        .values
        .iter()
        .filter_map(|value| match &value.kind {
            ValueKind::Load(place) => Some(place.clone()),
            _ => None,
        })
        .collect()
}

mod lower;
mod verify;
