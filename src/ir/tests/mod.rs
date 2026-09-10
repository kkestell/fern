use crate::{
    frontend::parser,
    ir::{lower::lower, model::*, verify::VerifiedProgram},
    semantic::namespaces,
    types::{BinaryOperator, Scalar, Type},
};

pub(super) fn integer(value: i128, ty: Scalar) -> Operand {
    Operand::Integer { value, ty }
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
    Program {
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

pub(super) fn element(base: Place, index: Operand) -> Place {
    Place::Element {
        base: Box::new(base),
        index,
        span: 0..1,
    }
}

/// Spells a place the way a test reads it: `local0[1]`, `global2`.
fn describe(place: &Place) -> String {
    match place {
        Place::Local(LocalId(id)) => format!("local{id}"),
        Place::Global(GlobalId(id)) => format!("global{id}"),
        Place::Element { base, index, .. } => {
            let index = match index {
                Operand::Integer { value, .. } => value.to_string(),
                Operand::Value(ValueId(id)) => format!("v{id}"),
            };
            format!("{}[{index}]", describe(base))
        }
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
            Instruction::Store { place, operand } => Some((place.clone(), *operand)),
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
