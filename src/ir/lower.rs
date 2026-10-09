//! Lowering from the checked program to Fern IR.

use crate::{
    frontend::syntax::{
        Expression, ExpressionKind, ForHeader, Function as SyntaxFunction, Statement, StatementKind,
    },
    semantic::model::{
        Binding, CheckedLocation, CheckedLocationKind, CheckedProgram, Constant, ExpressionValue,
    },
    types::{BinaryOperator, ComparisonOperator, Float, LogicalOperator, Scalar, Type},
};
use la_arena::Idx;
use lasso::Spur;
use num_traits::ToPrimitive;
use std::{
    collections::{HashMap, HashSet},
    ops::Index,
};

use super::model::*;
/// The IR function ID of a syntax function. `syntax.functions` holds every
/// function of every module in the program, and lowering visits it in arena
/// order, so a function's raw arena index is its program-wide ID and two
/// modules cannot claim one ID.
fn function_id(id: Idx<SyntaxFunction>) -> FunctionId {
    FunctionId(id.into_raw().into_u32() as usize)
}

/// Lowers a checked program, whose syntax and bindings span every module it
/// loaded, into one IR program.
///
/// `checked.module_bindings` covers every module's bindings with dependencies
/// before dependents, so each module-level `var` becomes exactly one `Global`
/// and a dependency's globals precede its dependents'. Module-level
/// initializers are constant expressions, so globals are static data: nothing
/// initializes them while the program runs, and no ordering code is emitted.
pub(crate) fn lower(checked: CheckedProgram<'_>) -> Program {
    let structs = checked
        .structs
        .iter()
        .map(|declared| Struct {
            fields: declared
                .fields
                .iter()
                .map(|field| field.ty.clone())
                .collect(),
        })
        .collect();
    let mut globals = Vec::new();
    let mut module_places = HashMap::new();
    for &statement in &checked.module_bindings {
        let binding = checked.declarations[statement];
        // A module-level `const` folds into its use sites and needs no storage.
        if checked.bindings[binding].constant.is_some() {
            continue;
        }
        let StatementKind::Binding { initializer, .. } = &checked.syntax.statements[statement].kind
        else {
            unreachable!("module bindings are binding statements")
        };
        let ty = checked.bindings[binding].ty.clone();
        let values = initializer.and_then(|initializer| {
            let constant = checked.expressions[initializer]
                .constant
                .as_ref()
                .expect("module-level initializers are constant expressions");
            static_values(&checked, constant, &ty)
        });
        module_places.insert(binding, Place::Global(GlobalId(globals.len())));
        globals.push(Global { ty, values });
    }

    let addressed = checked
        .addresses
        .iter()
        .filter_map(|(_, location)| addressed_binding(location))
        .collect();
    let functions = checked
        .syntax
        .functions
        .iter()
        .map(|(id, _)| lower_function(&checked, id, &module_places, &addressed))
        .collect();
    Program {
        structs,
        globals,
        functions,
        main: function_id(checked.main),
        main_span: checked.syntax.functions[checked.main].name_span.clone(),
    }
}

/// The binding whose storage `&location` addresses, or `None` when the
/// location is reached through a pointer and so is no binding's storage.
fn addressed_binding(location: &CheckedLocation) -> Option<Idx<Binding>> {
    match &location.kind {
        CheckedLocationKind::Binding(binding) => Some(*binding),
        CheckedLocationKind::Index { operand, .. } | CheckedLocationKind::Field { operand, .. } => {
            if matches!(operand.ty, Type::Pointer { .. }) {
                None
            } else {
                addressed_binding(operand)
            }
        }
        CheckedLocationKind::Dereference { .. } | CheckedLocationKind::Pointer(_) => None,
    }
}

fn lower_function(
    checked: &CheckedProgram<'_>,
    id: Idx<SyntaxFunction>,
    module_places: &HashMap<Idx<Binding>, Place>,
    addressed: &HashSet<Idx<Binding>>,
) -> Function {
    let signature = &checked.functions[id];
    let mut builder = FlowBuilder::new();
    let mut bindings = Places::new(module_places, addressed);
    for &parameter in &signature.parameters {
        let local = builder.local(checked.bindings[parameter].ty.clone());
        bindings.insert(parameter, Place::Local(local));
    }
    bindings.returned = returned_local(checked, id, addressed);
    let parameters = signature.parameters.len();
    let result = signature.result.clone();
    let terminated = lower_flow_body(
        checked,
        &checked.syntax.functions[id].body,
        &mut bindings,
        &[],
        &mut builder,
    );
    // A value function that reaches its end is rejected during checking, so an
    // unterminated block here is a lowering bug and verification reports it.
    if !terminated && result.is_none() {
        builder.terminate(Terminator::Return { value: None });
    }
    builder.finish(parameters, result)
}

/// The aggregate local every `return` of `function` returns, which can live in
/// the result storage so that no `return` copies it. Nothing else reads or
/// writes the result storage: the local is never addressed, an argument read
/// from it is copied, and no call returns into the storage because every
/// `return` returns the local. A parameter, a `for … in` binding, and a folded
/// `const` do not qualify, since each has storage of its own or none.
fn returned_local(
    checked: &CheckedProgram<'_>,
    function: Idx<SyntaxFunction>,
    addressed: &HashSet<Idx<Binding>>,
) -> Option<Idx<Binding>> {
    if !checked.functions[function]
        .result
        .as_ref()
        .is_some_and(Type::is_aggregate)
    {
        return None;
    }
    let mut returns = Vec::new();
    let mut declared = HashSet::new();
    collect_returns(
        checked,
        &checked.syntax.functions[function].body,
        &mut returns,
        &mut declared,
    );
    let mut returned = None;
    for value in returns {
        let binding = value.and_then(|value| returned_binding(checked, value))?;
        if returned
            .replace(binding)
            .is_some_and(|other| other != binding)
        {
            return None;
        }
    }
    returned.filter(|binding| {
        declared.contains(binding)
            && checked.bindings[*binding].constant.is_none()
            && !addressed.contains(binding)
    })
}

/// Every `return` value in `body`, and every binding a declaration in it
/// introduces, however deeply nested.
fn collect_returns(
    checked: &CheckedProgram<'_>,
    body: &[Idx<Statement>],
    returns: &mut Vec<Option<Idx<Expression>>>,
    declared: &mut HashSet<Idx<Binding>>,
) {
    for &statement in body {
        collect_statement_returns(checked, statement, returns, declared);
    }
}

fn collect_statement_returns(
    checked: &CheckedProgram<'_>,
    statement: Idx<Statement>,
    returns: &mut Vec<Option<Idx<Expression>>>,
    declared: &mut HashSet<Idx<Binding>>,
) {
    match &checked.syntax.statements[statement].kind {
        StatementKind::Binding { .. } => {
            declared.insert(checked.declarations[statement]);
        }
        StatementKind::Return { value } => returns.push(*value),
        StatementKind::Block { body } => collect_returns(checked, body, returns, declared),
        StatementKind::If {
            then_body,
            else_branch,
            ..
        } => {
            collect_returns(checked, then_body, returns, declared);
            if let Some(branch) = else_branch {
                collect_statement_returns(checked, *branch, returns, declared);
            }
        }
        StatementKind::For { header, body, .. } => {
            if let ForHeader::ThreeClause {
                initializer, post, ..
            } = header
            {
                collect_statement_returns(checked, *initializer, returns, declared);
                collect_statement_returns(checked, *post, returns, declared);
            }
            collect_returns(checked, body, returns, declared);
        }
        StatementKind::Assignment { .. }
        | StatementKind::CompoundAssignment { .. }
        | StatementKind::Free { .. }
        | StatementKind::Exit { .. }
        | StatementKind::Call { .. }
        | StatementKind::Break { .. }
        | StatementKind::Continue { .. } => {}
    }
}

/// The binding a `return` value reads, looking through any parentheses.
fn returned_binding(checked: &CheckedProgram<'_>, id: Idx<Expression>) -> Option<Idx<Binding>> {
    match &checked.expressions[id].value {
        ExpressionValue::Reference(binding) => Some(*binding),
        ExpressionValue::Grouping { expression } => returned_binding(checked, *expression),
        _ => None,
    }
}

/// The immutable module storage and the locals one function owns while it is
/// lowered. Local bindings shadow module bindings without copying the latter.
/// `addressed` holds every binding whose address an `&` takes, whose storage
/// a pointer can therefore write.
struct Places<'a> {
    module: &'a HashMap<Idx<Binding>, Place>,
    addressed: &'a HashSet<Idx<Binding>>,
    locals: HashMap<Idx<Binding>, Place>,
    /// The locals of addressed bindings.
    addressed_locals: HashSet<LocalId>,
    /// The local every `return` of the function returns, which lives in the
    /// result storage; see `returned_local`.
    returned: Option<Idx<Binding>>,
}

impl<'a> Places<'a> {
    fn new(module: &'a HashMap<Idx<Binding>, Place>, addressed: &'a HashSet<Idx<Binding>>) -> Self {
        Self {
            module,
            addressed,
            locals: HashMap::new(),
            addressed_locals: HashSet::new(),
            returned: None,
        }
    }

    fn insert(&mut self, binding: Idx<Binding>, place: Place) {
        if let Place::Local(local) = place
            && self.addressed.contains(&binding)
        {
            self.addressed_locals.insert(local);
        }
        self.locals.insert(binding, place);
    }

    /// Whether an aggregate argument may be read from `place` without a copy:
    /// its root is a constant, which nothing writes, or a local no pointer
    /// can reach. A called function can write a global or a pointer target.
    fn borrowable(&self, place: &Place) -> bool {
        match place.root() {
            Place::Constant(_) => true,
            Place::Local(local) => !self.addressed_locals.contains(local),
            Place::Global(_) | Place::Indirect { .. } | Place::Result => false,
            Place::Element { .. } | Place::Field { .. } => {
                unreachable!("a place's root is not a step")
            }
        }
    }
}

impl Index<&Idx<Binding>> for Places<'_> {
    type Output = Place;

    fn index(&self, binding: &Idx<Binding>) -> &Self::Output {
        self.locals
            .get(binding)
            .or_else(|| self.module.get(binding))
            .expect("checked bindings have a lowered place")
    }
}

fn integer(value: i128, ty: Scalar) -> Operand {
    Operand::Literal(Literal::Integer { value, ty })
}

/// The IR literal a concrete scalar or `null` constant of type `ty` holds.
/// Semantic checking has already given every untyped constant its context
/// type, so an untyped rational never reaches lowering.
fn literal(constant: &Constant, ty: &Type) -> Literal {
    match constant {
        Constant::Integer(value) => Literal::Integer {
            value: value
                .to_i128()
                .expect("concrete Fern value fits in the IR representation"),
            ty: ty.scalar().expect("an integer constant has a scalar type"),
        },
        Constant::Float(value) => Literal::Floating(*value),
        Constant::Null => Literal::Null(ty.clone()),
        Constant::Rational(_) => {
            unreachable!("an untyped constant is contextualized before lowering")
        }
        Constant::Array(_) | Constant::Struct(_) => {
            unreachable!("an aggregate constant is not a scalar")
        }
    }
}

/// The scalars a constant holds, in memory order, or `None` when every one of
/// them is stored as zero bytes and static data needs no values at all.
fn static_values(
    checked: &CheckedProgram<'_>,
    constant: &Constant,
    ty: &Type,
) -> Option<Vec<Literal>> {
    let mut values = Vec::new();
    flatten(checked, constant, ty, &mut values);
    values
        .iter()
        .any(|value| match value {
            Literal::Integer { value, .. } => *value != 0,
            Literal::Floating(value) => value.bits() != 0,
            Literal::Null(_) => false,
        })
        .then_some(values)
}

/// The zero value of a scalar or nullable pointer type.
fn zero_literal(ty: &Type) -> Literal {
    match ty {
        Type::Scalar(Scalar::F32) => Literal::Floating(Float::Binary32(0)),
        Type::Scalar(Scalar::F64) => Literal::Floating(Float::Binary64(0)),
        Type::Scalar(scalar) => Literal::Integer {
            value: 0,
            ty: *scalar,
        },
        Type::Pointer { .. } => Literal::Null(ty.clone()),
        Type::Array { .. } | Type::Struct(_) => unreachable!("an aggregate zero is a `Zero`"),
    }
}

/// The scalars a constant holds, in memory order. A struct's fields have
/// their own types, so the constant is walked together with its type rather
/// than under one leaf type.
fn flatten(
    checked: &CheckedProgram<'_>,
    constant: &Constant,
    ty: &Type,
    values: &mut Vec<Literal>,
) {
    match (ty, constant) {
        (Type::Scalar(_) | Type::Pointer { .. }, scalar_constant) => {
            values.push(literal(scalar_constant, ty));
        }
        (Type::Array { element, .. }, Constant::Array(elements)) => {
            for value in elements {
                flatten(checked, value, element, values);
            }
        }
        (Type::Struct(_), Constant::Struct(fields)) => {
            for (ordinal, value) in fields.iter().enumerate() {
                flatten(checked, value, &field_type(checked, ty, ordinal), values);
            }
        }
        _ => unreachable!("a folded constant has the shape of its type"),
    }
}

/// The declared type of one field of a struct type. Checking resolved the
/// field, so the ordinal names a field the declaration has.
fn field_type(checked: &CheckedProgram<'_>, ty: &Type, ordinal: usize) -> Type {
    let Type::Struct(declared) = ty else {
        unreachable!("a field belongs to a struct type")
    };
    checked.structs[declared.id.0].fields[ordinal].ty.clone()
}

struct BuildingBlock {
    instructions: Vec<Instruction>,
    terminator: Option<Terminator>,
}

struct FlowBuilder {
    values: Vec<Value>,
    locals: Vec<Type>,
    constants: Vec<Global>,
    blocks: Vec<BuildingBlock>,
    current: BlockId,
}

impl FlowBuilder {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            locals: Vec::new(),
            constants: Vec::new(),
            blocks: vec![BuildingBlock {
                instructions: Vec::new(),
                terminator: None,
            }],
            current: BlockId(0),
        }
    }

    fn block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len());
        self.blocks.push(BuildingBlock {
            instructions: Vec::new(),
            terminator: None,
        });
        id
    }

    fn select(&mut self, block: BlockId) {
        self.current = block;
    }

    fn local(&mut self, ty: Type) -> LocalId {
        let id = LocalId(self.locals.len());
        self.locals.push(ty);
        id
    }

    fn value(&mut self, value: Value) -> Operand {
        let id = ValueId(self.values.len());
        self.values.push(value);
        self.blocks[self.current.0]
            .instructions
            .push(Instruction::Value(id));
        Operand::Value(id)
    }

    fn store(&mut self, place: Place, operand: Operand) {
        self.store_in(self.current, place, operand);
    }

    /// Stores the zero value of `ty` into `place`.
    fn store_zero(&mut self, place: Place, ty: &Type) {
        let instruction = if ty.is_aggregate() {
            Instruction::Zero { place }
        } else {
            Instruction::Store {
                place,
                operand: Operand::Literal(zero_literal(ty)),
            }
        };
        self.blocks[self.current.0].instructions.push(instruction);
    }

    fn free(&mut self, operand: Operand) {
        self.blocks[self.current.0]
            .instructions
            .push(Instruction::Free { operand });
    }

    /// The IR type of an operand this builder defined or a literal.
    fn operand_type(&self, operand: &Operand) -> Type {
        match operand {
            Operand::Literal(literal) => literal.ty(),
            Operand::Value(ValueId(id)) => self.values[*id].ty.clone(),
        }
    }

    fn fill(&mut self, place: Place, start: u64, operand: Operand) {
        self.blocks[self.current.0]
            .instructions
            .push(Instruction::Fill {
                place,
                start,
                operand,
            });
    }

    /// The static data holding a folded aggregate.
    fn constant(&mut self, global: Global) -> Place {
        let id = ConstantId(self.constants.len());
        self.constants.push(global);
        Place::Constant(id)
    }

    /// Stores into `block` rather than the current one. The store runs after
    /// everything already in that block and before its terminator, so it can
    /// hold a value the block defined for a later block to read.
    fn store_in(&mut self, block: BlockId, place: Place, operand: Operand) {
        self.blocks[block.0]
            .instructions
            .push(Instruction::Store { place, operand });
    }

    fn call(
        &mut self,
        function: FunctionId,
        arguments: Vec<Operand>,
        result: Option<Type>,
        destination: Option<Place>,
        span: std::ops::Range<usize>,
    ) -> Option<Operand> {
        // A call with a destination writes its result there and defines no
        // value.
        let result = result.filter(|_| destination.is_none()).map(|ty| {
            let id = ValueId(self.values.len());
            self.values.push(Value {
                span: None,
                ty,
                kind: ValueKind::CallResult,
            });
            id
        });
        self.blocks[self.current.0]
            .instructions
            .push(Instruction::Call {
                result,
                destination,
                function,
                arguments,
                span,
            });
        result.map(Operand::Value)
    }

    fn terminate(&mut self, terminator: Terminator) {
        let slot = &mut self.blocks[self.current.0].terminator;
        assert!(slot.is_none(), "IR builder terminated a block twice");
        *slot = Some(terminator);
    }

    fn finish(self, parameters: usize, result: Option<Type>) -> Function {
        Function {
            parameters,
            result,
            values: self.values,
            constants: self.constants,
            flow: ControlFlow {
                entry: BlockId(0),
                locals: self.locals,
                blocks: self
                    .blocks
                    .into_iter()
                    .map(|block| Block {
                        instructions: block.instructions,
                        terminator: block.terminator.unwrap_or(Terminator::Unreachable),
                    })
                    .collect(),
            },
        }
    }
}

#[derive(Clone)]
struct LoopTarget {
    label: Option<Spur>,
    break_target: BlockId,
    continue_target: BlockId,
}

fn lower_flow_binding(
    checked: &CheckedProgram<'_>,
    statement: Idx<Statement>,
    bindings: &mut Places<'_>,
    builder: &mut FlowBuilder,
) {
    let StatementKind::Binding { initializer, .. } = &checked.syntax.statements[statement].kind
    else {
        unreachable!("binding lowering requires a binding statement")
    };
    let binding = checked.declarations[statement];
    if checked.bindings[binding].constant.is_some() {
        if let Some(initializer) = initializer {
            lower_folded_effects(checked, *initializer, bindings, builder);
        }
        return;
    }
    let ty = checked.bindings[binding].ty.clone();
    let Some(initializer) = initializer else {
        let place = binding_place(binding, &ty, bindings, builder);
        builder.store_zero(place.clone(), &ty);
        bindings.insert(binding, place);
        return;
    };
    // A binding is not in scope in its own initializer and no pointer reaches
    // an unaddressed one, so a call can write its result straight into the
    // binding's storage without any argument seeing the write.
    if ty.is_aggregate()
        && !bindings.addressed.contains(&binding)
        && let Some(call) = unwrapped_call(checked, *initializer)
    {
        let place = binding_place(binding, &ty, bindings, builder);
        lower_call_into(checked, call, place.clone(), bindings, builder);
        bindings.insert(binding, place);
        return;
    }
    let operand = lower_flow_operand(checked, *initializer, bindings, builder);
    let place = binding_place(binding, &ty, bindings, builder);
    builder.store(place.clone(), operand);
    bindings.insert(binding, place);
}

/// The storage a declared binding lives in: the result storage for the
/// function's returned local, and a fresh local otherwise.
fn binding_place(
    binding: Idx<Binding>,
    ty: &Type,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Place {
    if bindings.returned == Some(binding) {
        Place::Result
    } else {
        Place::Local(builder.local(ty.clone()))
    }
}

fn lower_flow_body(
    checked: &CheckedProgram<'_>,
    body: &[Idx<Statement>],
    bindings: &mut Places<'_>,
    loops: &[LoopTarget],
    builder: &mut FlowBuilder,
) -> bool {
    for &statement in body {
        if lower_flow_statement(checked, statement, bindings, loops, builder) {
            return true;
        }
    }
    false
}

fn lower_flow_statement(
    checked: &CheckedProgram<'_>,
    statement: Idx<Statement>,
    bindings: &mut Places<'_>,
    loops: &[LoopTarget],
    builder: &mut FlowBuilder,
) -> bool {
    match &checked.syntax.statements[statement].kind {
        StatementKind::Binding { .. } => {
            lower_flow_binding(checked, statement, bindings, builder);
            false
        }
        StatementKind::Assignment { value, .. } => {
            let mut place = lower_target(checked, statement, bindings, builder);
            let operand = lower_flow_operand(checked, *value, bindings, builder);
            let place = place.read(builder);
            builder.store(place, operand);
            false
        }
        StatementKind::CompoundAssignment {
            operator,
            operator_span,
            value,
            ..
        } => {
            let mut held = lower_target(checked, statement, bindings, builder);
            let ty = checked.assignments[statement].ty.clone();
            let place = held.read(builder);
            let left_operand = load_place(builder, place, ty.clone());
            let (left, right) =
                lower_second_operand(checked, left_operand, ty.clone(), *value, bindings, builder);
            let result = builder.value(Value {
                span: Some(operator_span.clone()),
                ty,
                kind: ValueKind::Binary {
                    operator: *operator,
                    form: BinaryForm::CompoundAssignment,
                    left,
                    right,
                },
            });
            let place = held.read(builder);
            builder.store(place, result);
            false
        }
        StatementKind::Block { body } => lower_flow_body(checked, body, bindings, loops, builder),
        StatementKind::Free { operand } => {
            let operand = lower_flow_operand(checked, *operand, bindings, builder);
            builder.free(operand);
            false
        }
        StatementKind::Exit { argument } => {
            let status = lower_flow_operand(checked, *argument, bindings, builder);
            builder.terminate(Terminator::Exit { status });
            true
        }
        StatementKind::Call { call } => {
            lower_call(
                checked,
                checked.calls[statement],
                &call.arguments,
                call.target.span.clone(),
                bindings,
                builder,
            );
            false
        }
        StatementKind::Return { value } => {
            let value = value.map(|value| match returned_literal(checked, value) {
                // The result storage aliases nothing the literal reads, so the
                // literal is built there rather than in a local it would be
                // copied out of.
                Some(literal) => {
                    store_literal(checked, &Place::Result, literal, bindings, builder);
                    load_place(
                        builder,
                        Place::Result,
                        checked.expressions[value].ty.clone(),
                    )
                }
                None => lower_flow_operand(checked, value, bindings, builder),
            });
            builder.terminate(Terminator::Return { value });
            true
        }
        StatementKind::If {
            condition,
            then_body,
            else_branch,
        } => lower_if(
            checked,
            *condition,
            then_body,
            *else_branch,
            bindings,
            loops,
            builder,
        ),
        StatementKind::For { .. } => {
            lower_for(checked, statement, bindings, loops, builder);
            false
        }
        StatementKind::Break { label } | StatementKind::Continue { label } => {
            let loop_target = loop_target(loops, label.as_ref().map(|label| label.name));
            let target = if matches!(
                checked.syntax.statements[statement].kind,
                StatementKind::Break { .. }
            ) {
                loop_target.break_target
            } else {
                loop_target.continue_target
            };
            builder.terminate(Terminator::Jump { target });
            true
        }
    }
}

/// The place an assignment target names, with its steps lowered left to right
/// before the value.
fn lower_target(
    checked: &CheckedProgram<'_>,
    statement: Idx<Statement>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> HeldPlace {
    let target = &checked.assignments[statement];
    lower_location(checked, &target.location, bindings, builder)
}

fn lower_location(
    checked: &CheckedProgram<'_>,
    location: &CheckedLocation,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> HeldPlace {
    match &location.kind {
        CheckedLocationKind::Binding(binding) => HeldPlace {
            root: HeldRoot::Place(bindings[binding].clone()),
            steps: Vec::new(),
        },
        CheckedLocationKind::Index { operand, index } => {
            let mut held = lower_step_base_location(checked, operand, bindings, builder);
            let syntax_index = *index;
            let index = lower_flow_operand(checked, syntax_index, bindings, builder);
            held.steps.push(HeldStep::Index(
                hold_operand(builder, index, Scalar::Int.into()),
                checked.syntax.expressions[syntax_index].span.clone(),
            ));
            held
        }
        CheckedLocationKind::Field { operand, ordinal } => {
            let mut held = lower_step_base_location(checked, operand, bindings, builder);
            held.steps.push(HeldStep::Field(*ordinal));
            held
        }
        CheckedLocationKind::Dereference { operand } => {
            let pointer = lower_location_pointer(checked, operand, bindings, builder);
            hold_place(builder, Place::Indirect { pointer })
        }
        CheckedLocationKind::Pointer(_) => {
            unreachable!("a pointer location is only the operand of a step or a dereference")
        }
    }
}

/// The place a field or index step on `location` acts on: the location
/// itself, or the target its pointers reach. The pointers are loaded now, so
/// the target is fixed before anything the step is followed by.
fn lower_step_base_location(
    checked: &CheckedProgram<'_>,
    location: &CheckedLocation,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> HeldPlace {
    if !matches!(location.ty, Type::Pointer { .. }) {
        return lower_location(checked, location, bindings, builder);
    }
    let pointer = lower_location_pointer(checked, location, bindings, builder);
    let place = follow_pointers(builder, pointer, &location.ty);
    hold_place(builder, place)
}

/// The pointer a pointer-typed location holds: the value of the expression a
/// `Pointer` location computes, or a load from any other location.
fn lower_location_pointer(
    checked: &CheckedProgram<'_>,
    location: &CheckedLocation,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Operand {
    if let CheckedLocationKind::Pointer(id) = &location.kind {
        return lower_flow_operand(checked, *id, bindings, builder);
    }
    let place = lower_location(checked, location, bindings, builder).read(builder);
    load_place(builder, place, location.ty.clone())
}

/// The target of the last of `ty`'s non-nullable pointer layers, starting
/// from `pointer` of type `ty`. Each intermediate pointer is loaded from the
/// target of the one before it.
fn follow_pointers(builder: &mut FlowBuilder, pointer: Operand, ty: &Type) -> Place {
    let Type::Pointer {
        nullable: false,
        target,
    } = ty
    else {
        unreachable!("a step follows a non-nullable pointer")
    };
    let place = Place::Indirect { pointer };
    if matches!(
        **target,
        Type::Pointer {
            nullable: false,
            ..
        }
    ) {
        let next = load_place(builder, place, (**target).clone());
        return follow_pointers(builder, next, target);
    }
    place
}

/// Lowers arguments left to right, holding each so a later argument that splits
/// blocks cannot leave an earlier one unreadable in the call's block. An
/// aggregate argument is passed from its own storage unless a called function
/// could write it; see `lower_argument`.
fn lower_call(
    checked: &CheckedProgram<'_>,
    function: Idx<SyntaxFunction>,
    arguments: &[Idx<Expression>],
    span: std::ops::Range<usize>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Option<Operand> {
    let arguments = lower_arguments(checked, arguments, bindings, builder);
    builder.call(
        function_id(function),
        arguments,
        checked.functions[function].result.clone(),
        None,
        span,
    )
}

/// Lowers the call `id`, whose callee writes its aggregate result straight
/// into `destination` rather than into storage the call owns.
fn lower_call_into(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    destination: Place,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) {
    let ExpressionValue::Call { function } = &checked.expressions[id].value else {
        unreachable!("a call expression is lowered from its own value")
    };
    let ExpressionKind::Call(call) = &checked.syntax.expressions[id].kind else {
        unreachable!("a checked call expression is a call")
    };
    let arguments = lower_arguments(checked, &call.arguments, bindings, builder);
    builder.call(
        function_id(*function),
        arguments,
        checked.functions[*function].result.clone(),
        Some(destination),
        checked.syntax.expressions[id].span.clone(),
    );
}

/// Lowers arguments left to right, holding each so a later argument that
/// splits blocks cannot leave an earlier one unreadable in the call's block.
fn lower_arguments(
    checked: &CheckedProgram<'_>,
    arguments: &[Idx<Expression>],
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Vec<Operand> {
    let mut held = Vec::with_capacity(arguments.len());
    for &argument in arguments {
        let ty = checked.expressions[argument].ty.clone();
        let operand = lower_argument(checked, argument, bindings, builder);
        held.push(hold_operand(builder, operand, ty));
    }
    held.iter_mut()
        .map(|operand| operand.read(builder))
        .collect()
}

/// An argument's operand. An aggregate whose place `Places::borrowable`
/// accepts is borrowed rather than copied, since nothing can write that place
/// before the call returns. A module-level `var` or a place a pointer reaches
/// can be written by the callee or by a later argument, so an argument read
/// from one is loaded.
fn lower_argument(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Operand {
    let ty = checked.expressions[id].ty.clone();
    if !ty.is_aggregate() || unwrapped_call(checked, id).is_some() {
        return lower_flow_operand(checked, id, bindings, builder);
    }
    let place = lower_aggregate_place(checked, id, bindings, builder);
    let kind = if bindings.borrowable(&place) {
        ValueKind::Borrow(place)
    } else {
        ValueKind::Load(place)
    };
    builder.value(Value {
        span: None,
        ty,
        kind,
    })
}

/// The unfolded array or struct literal a `return` of `id` returns, looking
/// through any parentheses around it.
fn returned_literal(checked: &CheckedProgram<'_>, id: Idx<Expression>) -> Option<Idx<Expression>> {
    let expression = &checked.expressions[id];
    if expression.constant.is_some() {
        return None;
    }
    match &expression.value {
        ExpressionValue::Array { .. } | ExpressionValue::Struct { .. } => Some(id),
        ExpressionValue::Grouping { expression } => returned_literal(checked, *expression),
        _ => None,
    }
}

/// Stores the array or struct literal `id` into `place`.
fn store_literal(
    checked: &CheckedProgram<'_>,
    place: &Place,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) {
    match &checked.expressions[id].value {
        ExpressionValue::Array { elements, fill } => {
            store_elements(checked, place, id, elements, *fill, bindings, builder);
        }
        ExpressionValue::Struct {
            initializers,
            filled,
            ..
        } => store_fields(checked, place, id, initializers, filled, bindings, builder),
        _ => unreachable!("only an array or struct literal is stored as a literal"),
    }
}

/// The unfolded call `id` evaluates, looking through any parentheses around
/// it.
fn unwrapped_call(checked: &CheckedProgram<'_>, id: Idx<Expression>) -> Option<Idx<Expression>> {
    let expression = &checked.expressions[id];
    if expression.constant.is_some() {
        return None;
    }
    match &expression.value {
        ExpressionValue::Call { .. } => Some(id),
        ExpressionValue::Grouping { expression } => unwrapped_call(checked, *expression),
        _ => None,
    }
}

fn lower_call_expression(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Operand {
    let ExpressionValue::Call { function } = &checked.expressions[id].value else {
        unreachable!("a call expression is lowered from its own value")
    };
    let ExpressionKind::Call(call) = &checked.syntax.expressions[id].kind else {
        unreachable!("a checked call expression is a call")
    };
    lower_call(
        checked,
        *function,
        &call.arguments,
        checked.syntax.expressions[id].span.clone(),
        bindings,
        builder,
    )
    .expect("a call used as a value returns one")
}

fn lower_if(
    checked: &CheckedProgram<'_>,
    condition: Idx<Expression>,
    then_body: &[Idx<Statement>],
    else_branch: Option<Idx<Statement>>,
    bindings: &mut Places<'_>,
    loops: &[LoopTarget],
    builder: &mut FlowBuilder,
) -> bool {
    let condition_operand = lower_flow_operand(checked, condition, bindings, builder);
    let then_block = builder.block();
    let else_block = builder.block();
    let join_block = builder.block();
    builder.terminate(Terminator::Branch {
        condition: condition_operand,
        then_target: then_block,
        else_target: else_block,
    });

    builder.select(then_block);
    let then_terminated = lower_flow_body(checked, then_body, bindings, loops, builder);
    if !then_terminated {
        builder.terminate(Terminator::Jump { target: join_block });
    }

    builder.select(else_block);
    let else_terminated = else_branch
        .is_some_and(|branch| lower_flow_statement(checked, branch, bindings, loops, builder));
    if !else_terminated {
        builder.terminate(Terminator::Jump { target: join_block });
    }

    builder.select(join_block);
    then_terminated && else_terminated
}

/// A `for … in` loop's array, captured once before the first iteration, its
/// length, and the counter driving it.
struct Iteration {
    place: Place,
    length: u64,
    element: Type,
    counter: LocalId,
    value: LocalId,
    span: std::ops::Range<usize>,
}

impl Iteration {
    fn counter(&self, builder: &mut FlowBuilder) -> Operand {
        load_place(builder, Place::Local(self.counter), Scalar::Int.into())
    }

    fn condition(&self, builder: &mut FlowBuilder) -> Operand {
        let counter = self.counter(builder);
        builder.value(Value {
            span: None,
            ty: Scalar::Bool.into(),
            kind: ValueKind::Comparison {
                operator: ComparisonOperator::Less,
                left: counter,
                right: integer(i128::from(self.length), Scalar::Int),
            },
        })
    }

    /// Binds the value to a copy of the element the counter reaches.
    fn bind(&self, builder: &mut FlowBuilder) {
        let index = self.counter(builder);
        let element = Place::Element {
            base: Box::new(self.place.clone()),
            index,
            span: self.span.clone(),
        };
        let operand = load_place(builder, element, self.element.clone());
        builder.store(Place::Local(self.value), operand);
    }

    fn advance(&self, builder: &mut FlowBuilder) {
        let counter = self.counter(builder);
        let next = builder.value(Value {
            span: Some(self.span.clone()),
            ty: Scalar::Int.into(),
            kind: ValueKind::Binary {
                operator: BinaryOperator::Add,
                form: BinaryForm::Infix,
                left: counter,
                right: integer(1, Scalar::Int),
            },
        });
        builder.store(Place::Local(self.counter), next);
    }
}

/// What a loop's condition block tests and its post block does.
enum LoopKind {
    Clauses {
        condition: Option<Idx<Expression>>,
        post: Option<Idx<Statement>>,
    },
    Iteration(Iteration),
}

impl LoopKind {
    fn has_post(&self) -> bool {
        match self {
            Self::Clauses { post, .. } => post.is_some(),
            Self::Iteration(_) => true,
        }
    }
}

/// Lowers whatever a loop header does before its first iteration.
fn lower_for_header(
    checked: &CheckedProgram<'_>,
    statement: Idx<Statement>,
    header: &ForHeader,
    bindings: &mut Places<'_>,
    loops: &[LoopTarget],
    builder: &mut FlowBuilder,
) -> LoopKind {
    match header {
        ForHeader::Infinite => LoopKind::Clauses {
            condition: None,
            post: None,
        },
        ForHeader::Condition(condition) => LoopKind::Clauses {
            condition: Some(*condition),
            post: None,
        },
        ForHeader::ThreeClause {
            initializer,
            condition,
            post,
        } => {
            assert!(!lower_flow_statement(
                checked,
                *initializer,
                bindings,
                loops,
                builder
            ));
            LoopKind::Clauses {
                condition: Some(*condition),
                post: Some(*post),
            }
        }
        ForHeader::Iteration { operand, .. } => {
            LoopKind::Iteration(capture(checked, statement, *operand, bindings, builder))
        }
    }
}

/// Captures the array a `for … in` walks, so assigning to its original binding
/// inside the body cannot change the remaining iterations.
fn capture(
    checked: &CheckedProgram<'_>,
    statement: Idx<Statement>,
    operand: Idx<Expression>,
    bindings: &mut Places<'_>,
    builder: &mut FlowBuilder,
) -> Iteration {
    let ty = checked.expressions[operand].ty.clone();
    let span = checked.syntax.expressions[operand].span.clone();
    let Type::Array { length, element } = &ty else {
        unreachable!("checking requires an array operand for `for … in`")
    };
    let (length, element) = (*length, (**element).clone());
    let source = lower_aggregate_place(checked, operand, bindings, builder);
    let loaded = load_place(builder, source, ty.clone());
    let place = Place::Local(builder.local(ty.clone()));
    builder.store(place.clone(), loaded);

    let counter = builder.local(Scalar::Int.into());
    builder.store(Place::Local(counter), integer(0, Scalar::Int));
    let value = builder.local(element.clone());

    let iteration = &checked.iterations[statement];
    bindings.insert(iteration.value, Place::Local(value));
    // The index binding is immutable, so reading the counter directly is the
    // same as reading a copy taken at the top of the body.
    if let Some(index) = iteration.index {
        bindings.insert(index, Place::Local(counter));
    }
    Iteration {
        place,
        length,
        element,
        counter,
        value,
        span,
    }
}

fn lower_for(
    checked: &CheckedProgram<'_>,
    statement: Idx<Statement>,
    bindings: &mut Places<'_>,
    loops: &[LoopTarget],
    builder: &mut FlowBuilder,
) {
    let StatementKind::For {
        label,
        header,
        body,
    } = &checked.syntax.statements[statement].kind
    else {
        unreachable!("a `for` statement is lowered from its own syntax")
    };
    let label = label.as_ref().map(|label| label.name);
    let kind = lower_for_header(checked, statement, header, bindings, loops, builder);

    let condition_block = builder.block();
    let body_block = builder.block();
    let post_block = kind.has_post().then(|| builder.block());
    let after_block = builder.block();
    builder.terminate(Terminator::Jump {
        target: condition_block,
    });

    builder.select(condition_block);
    let condition = match &kind {
        LoopKind::Clauses { condition, .. } => {
            condition.map(|condition| lower_flow_operand(checked, condition, bindings, builder))
        }
        LoopKind::Iteration(iteration) => Some(iteration.condition(builder)),
    };
    if let Some(condition) = condition {
        builder.terminate(Terminator::Branch {
            condition,
            then_target: body_block,
            else_target: after_block,
        });
    } else {
        builder.terminate(Terminator::Jump { target: body_block });
    }

    let mut nested_loops = loops.to_vec();
    nested_loops.push(LoopTarget {
        label,
        break_target: after_block,
        continue_target: post_block.unwrap_or(condition_block),
    });
    builder.select(body_block);
    if let LoopKind::Iteration(iteration) = &kind {
        iteration.bind(builder);
    }
    if !lower_flow_body(checked, body, bindings, &nested_loops, builder) {
        builder.terminate(Terminator::Jump {
            target: post_block.unwrap_or(condition_block),
        });
    }

    if let Some(post_block) = post_block {
        builder.select(post_block);
        lower_for_post(checked, &kind, bindings, loops, builder);
        builder.terminate(Terminator::Jump {
            target: condition_block,
        });
    }
    builder.select(after_block);
}

fn lower_for_post(
    checked: &CheckedProgram<'_>,
    kind: &LoopKind,
    bindings: &mut Places<'_>,
    loops: &[LoopTarget],
    builder: &mut FlowBuilder,
) {
    match kind {
        LoopKind::Clauses { post, .. } => {
            let post = post.expect("a post block exists only for a post clause");
            assert!(!lower_flow_statement(
                checked, post, bindings, loops, builder
            ));
        }
        LoopKind::Iteration(iteration) => iteration.advance(builder),
    }
}

fn loop_target(loops: &[LoopTarget], label: Option<Spur>) -> &LoopTarget {
    loops
        .iter()
        .rev()
        .find(|target| label.is_none() || target.label == label)
        .expect("semantic checking resolves loop targets")
}

fn element_at(base: &Place, index: u64, span: &std::ops::Range<usize>) -> Place {
    Place::Element {
        base: Box::new(base.clone()),
        index: integer(i128::from(index), Scalar::Int),
        span: span.clone(),
    }
}

fn field_at(base: &Place, ordinal: usize) -> Place {
    Place::Field {
        base: Box::new(base.clone()),
        ordinal,
    }
}

/// Stores a struct literal's fields into `place`: the written initializers in
/// source order, which is the order they are evaluated in, and then the zero
/// value each omitted field is filled with.
fn store_fields(
    checked: &CheckedProgram<'_>,
    place: &Place,
    id: Idx<Expression>,
    initializers: &[(usize, Idx<Expression>)],
    filled: &[(usize, Constant)],
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) {
    let ty = checked.expressions[id].ty.clone();
    for &(ordinal, value) in initializers {
        let operand = lower_flow_operand(checked, value, bindings, builder);
        builder.store(field_at(place, ordinal), operand);
    }
    for (ordinal, _) in filled {
        builder.store_zero(
            field_at(place, *ordinal),
            &field_type(checked, &ty, *ordinal),
        );
    }
}

/// Stores an array literal's elements into `place`. A fill evaluates its last
/// element once and copies that value into every remaining element.
fn store_elements(
    checked: &CheckedProgram<'_>,
    place: &Place,
    id: Idx<Expression>,
    elements: &[Idx<Expression>],
    fill: bool,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) {
    let Type::Array { length, .. } = &checked.expressions[id].ty else {
        unreachable!("an array literal has an array type")
    };
    let span = &checked.syntax.expressions[id].span;
    let mut last = None;
    for (index, &element) in elements.iter().enumerate() {
        let index = u64::try_from(index).expect("an array fits in the address space");
        let operand = lower_flow_operand(checked, element, bindings, builder);
        builder.store(element_at(place, index, span), operand.clone());
        last = Some(operand);
    }
    if !fill {
        return;
    }
    let last = last.expect("a fill follows at least one element");
    let filled = u64::try_from(elements.len()).expect("an array fits in the address space");
    if filled < *length {
        builder.fill(place.clone(), filled, last);
    }
}

/// The place holding an aggregate-typed expression. An expression with no
/// storage of its own materializes into a fresh local.
fn lower_aggregate_place(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Place {
    let expression = &checked.expressions[id];
    let ty = &expression.ty;
    if let Some(constant) = expression.constant.as_ref() {
        lower_folded_effects(checked, id, bindings, builder);
        return builder.constant(Global {
            ty: ty.clone(),
            values: static_values(checked, constant, ty),
        });
    }
    match &expression.value {
        ExpressionValue::Reference(binding) => bindings[binding].clone(),
        ExpressionValue::Grouping { expression } => {
            lower_aggregate_place(checked, *expression, bindings, builder)
        }
        ExpressionValue::Index { .. } => element_place(checked, id, bindings, builder),
        ExpressionValue::Field { .. } => field_place(checked, id, bindings, builder),
        ExpressionValue::Array { elements, fill } => {
            let place = Place::Local(builder.local(ty.clone()));
            store_elements(checked, &place, id, elements, *fill, bindings, builder);
            place
        }
        ExpressionValue::Struct {
            initializers,
            filled,
            ..
        } => {
            let place = Place::Local(builder.local(ty.clone()));
            store_fields(checked, &place, id, initializers, filled, bindings, builder);
            place
        }
        ExpressionValue::Call { .. } => {
            let operand = lower_call_expression(checked, id, bindings, builder);
            let place = Place::Local(builder.local(ty.clone()));
            builder.store(place.clone(), operand);
            place
        }
        ExpressionValue::Dereference { operand } => Place::Indirect {
            pointer: lower_flow_operand(checked, *operand, bindings, builder),
        },
        _ => unreachable!(
            "an aggregate value is a reference, a literal, an element, a field, a call, or a dereference"
        ),
    }
}

fn element_place(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Place {
    let ExpressionValue::Index { operand, index } = &checked.expressions[id].value else {
        unreachable!("an element place is lowered from an index expression")
    };
    let span = checked.syntax.expressions[*index].span.clone();
    let base = lower_step_base(checked, *operand, bindings, builder);
    let mut base = hold_place(builder, base);
    let index = lower_flow_operand(checked, *index, bindings, builder);
    Place::Element {
        base: Box::new(base.read(builder)),
        index,
        span,
    }
}

fn field_place(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Place {
    let ExpressionValue::Field { operand, ordinal } = &checked.expressions[id].value else {
        unreachable!("a field place is lowered from a field expression")
    };
    let base = lower_step_base(checked, *operand, bindings, builder);
    field_at(&base, *ordinal)
}

/// The place a field or index step on `operand` acts on: the operand's own
/// place when it is an aggregate, or the target its pointers reach.
fn lower_step_base(
    checked: &CheckedProgram<'_>,
    operand: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Place {
    let ty = &checked.expressions[operand].ty;
    if ty.is_aggregate() {
        return lower_aggregate_place(checked, operand, bindings, builder);
    }
    let pointer = lower_flow_operand(checked, operand, bindings, builder);
    follow_pointers(builder, pointer, ty)
}

/// The length `len(operand)` reads from its operand's type. The operand is
/// still evaluated, so a trap inside it still happens.
fn lower_length(
    checked: &CheckedProgram<'_>,
    operand: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Operand {
    let Type::Array { length, .. } = checked.expressions[operand].ty.through_pointers() else {
        unreachable!("checking requires an array operand for `len`")
    };
    lower_flow_operand(checked, operand, bindings, builder);
    integer(i128::from(*length), Scalar::Int)
}

/// Lowers the evaluation a folded expression still owes. `len` reads its
/// length from its operand's type, so it folds while the operand still runs;
/// every other sub-expression of a folded constant folded too and has nothing
/// left to evaluate.
fn lower_folded_effects(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) {
    let mut operands = Vec::new();
    match &checked.expressions[id].value {
        ExpressionValue::Length { operand } => {
            lower_flow_operand(checked, *operand, bindings, builder);
            return;
        }
        ExpressionValue::Grouping { expression } => operands.push(*expression),
        ExpressionValue::Conversion { operand, .. }
        | ExpressionValue::Unary { operand, .. }
        | ExpressionValue::LogicalNot { operand, .. } => operands.push(*operand),
        ExpressionValue::Binary { left, right, .. }
        | ExpressionValue::Comparison { left, right, .. }
        | ExpressionValue::Logical { left, right, .. } => operands.extend([*left, *right]),
        ExpressionValue::Array { elements, .. } => operands.extend(elements),
        ExpressionValue::Struct { initializers, .. } => {
            operands.extend(initializers.iter().map(|(_, value)| value));
        }
        // A call, an index, a field, and the pointer operations never fold,
        // so a folded expression only reaches them under a `len`, and the
        // rest hold no sub-expression at all.
        ExpressionValue::Integer
        | ExpressionValue::CompoundAssignmentTarget
        | ExpressionValue::Floating
        | ExpressionValue::Boolean
        | ExpressionValue::Null
        | ExpressionValue::Reference(_)
        | ExpressionValue::Call { .. }
        | ExpressionValue::Index { .. }
        | ExpressionValue::Field { .. }
        | ExpressionValue::AddressOf
        | ExpressionValue::Dereference { .. }
        | ExpressionValue::New { .. }
        | ExpressionValue::PointerConversion { .. } => {}
    }
    for operand in operands {
        lower_folded_effects(checked, operand, bindings, builder);
    }
}

fn lower_flow_operand(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Operand {
    let expression = &checked.expressions[id];
    if expression.ty.is_aggregate() {
        return lower_aggregate_operand(checked, id, bindings, builder);
    }
    let ty = expression.ty.clone();
    if let Some(constant) = expression.constant.as_ref() {
        lower_folded_effects(checked, id, bindings, builder);
        return Operand::Literal(literal(constant, &ty));
    }
    match &expression.value {
        ExpressionValue::Integer
        | ExpressionValue::CompoundAssignmentTarget
        | ExpressionValue::Floating
        | ExpressionValue::Boolean
        | ExpressionValue::Null => {
            unreachable!("literal expressions are constant")
        }
        ExpressionValue::Array { .. } => unreachable!("an array literal has an array type"),
        ExpressionValue::Struct { .. } => unreachable!("a struct literal has a struct type"),
        // A `len` reaches here only when its operand holds a call, which is
        // what stops it from folding.
        ExpressionValue::Length { operand } => lower_length(checked, *operand, bindings, builder),
        ExpressionValue::Index { .. } => {
            let place = element_place(checked, id, bindings, builder);
            load_place(builder, place, ty)
        }
        ExpressionValue::Field { .. } => {
            let place = field_place(checked, id, bindings, builder);
            load_place(builder, place, ty)
        }
        ExpressionValue::Dereference { operand } => {
            let pointer = lower_flow_operand(checked, *operand, bindings, builder);
            load_place(builder, Place::Indirect { pointer }, ty)
        }
        ExpressionValue::AddressOf => {
            let location = &checked.addresses[id];
            let place = lower_location(checked, location, bindings, builder).read(builder);
            builder.value(Value {
                span: None,
                ty: Type::Pointer {
                    nullable: false,
                    target: Box::new(location.ty.clone()),
                },
                kind: ValueKind::Address(place),
            })
        }
        ExpressionValue::New { operand } => {
            let target = checked.expressions[*operand].ty.clone();
            let value = lower_flow_operand(checked, *operand, bindings, builder);
            let pointer = builder.value(Value {
                span: Some(checked.syntax.expressions[id].span.clone()),
                ty: Type::Pointer {
                    nullable: false,
                    target: Box::new(target),
                },
                kind: ValueKind::Allocate,
            });
            builder.store(
                Place::Indirect {
                    pointer: pointer.clone(),
                },
                value,
            );
            pointer
        }
        ExpressionValue::PointerConversion { operand } => {
            let value = lower_flow_operand(checked, *operand, bindings, builder);
            match &checked.expressions[*operand].ty {
                // A `*T` operand already holds a `*T`, so only a nullable one
                // is checked.
                Type::Pointer {
                    nullable: true,
                    target,
                } => builder.value(Value {
                    span: Some(checked.syntax.expressions[id].span.clone()),
                    ty: Type::Pointer {
                        nullable: false,
                        target: target.clone(),
                    },
                    kind: ValueKind::CheckNull { operand: value },
                }),
                _ => value,
            }
        }
        ExpressionValue::Reference(binding) => load_place(builder, bindings[binding].clone(), ty),
        ExpressionValue::Grouping { expression } => {
            lower_flow_operand(checked, *expression, bindings, builder)
        }
        ExpressionValue::Conversion {
            operand,
            truncating,
        } => {
            let operand = lower_flow_operand(checked, *operand, bindings, builder);
            builder.value(Value {
                span: Some(checked.syntax.expressions[id].span.clone()),
                ty,
                kind: ValueKind::Convert {
                    operand,
                    truncating: *truncating,
                },
            })
        }
        ExpressionValue::Unary {
            operator,
            operator_span,
            operand,
        } => {
            let operand = lower_flow_operand(checked, *operand, bindings, builder);
            builder.value(Value {
                span: Some(operator_span.clone()),
                ty,
                kind: ValueKind::Unary {
                    operator: *operator,
                    operand,
                },
            })
        }
        ExpressionValue::Binary {
            operator,
            operator_span,
            left,
            right,
        } => {
            let left_operand = lower_flow_operand(checked, *left, bindings, builder);
            let (left, right) = lower_second_operand(
                checked,
                left_operand,
                checked.expressions[*left].ty.clone(),
                *right,
                bindings,
                builder,
            );
            builder.value(Value {
                span: Some(operator_span.clone()),
                ty,
                kind: ValueKind::Binary {
                    operator: *operator,
                    form: BinaryForm::Infix,
                    left,
                    right,
                },
            })
        }
        ExpressionValue::Comparison {
            operator,
            left,
            right,
            ..
        } => {
            let left_operand = lower_flow_operand(checked, *left, bindings, builder);
            let (left, right) = lower_second_operand(
                checked,
                left_operand,
                checked.expressions[*left].ty.clone(),
                *right,
                bindings,
                builder,
            );
            builder.value(Value {
                span: None,
                ty: Scalar::Bool.into(),
                kind: ValueKind::Comparison {
                    operator: *operator,
                    left,
                    right,
                },
            })
        }
        ExpressionValue::LogicalNot { operand, .. } => {
            let operand = lower_flow_operand(checked, *operand, bindings, builder);
            builder.value(Value {
                span: None,
                ty: Scalar::Bool.into(),
                kind: ValueKind::LogicalNot { operand },
            })
        }
        ExpressionValue::Logical {
            operator,
            left,
            right,
            ..
        } => lower_logical(checked, *operator, *left, *right, bindings, builder),
        ExpressionValue::Call { .. } => lower_call_expression(checked, id, bindings, builder),
    }
}

/// An aggregate operand. A call's result already owns its storage, so it is
/// the operand itself; anything else is loaded from its place.
fn lower_aggregate_operand(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Operand {
    if let Some(call) = unwrapped_call(checked, id) {
        return lower_call_expression(checked, call, bindings, builder);
    }
    let place = lower_aggregate_place(checked, id, bindings, builder);
    load_place(builder, place, checked.expressions[id].ty.clone())
}

/// An already-lowered operand and the block that produced it. Lowering what
/// follows it can split blocks, and a value is readable only in the block
/// defining it, so reading it elsewhere goes through a local. However often it
/// is read, it spills at most once.
struct HeldOperand {
    operand: Operand,
    block: BlockId,
    ty: Type,
    spill: Option<LocalId>,
}

fn hold_operand(builder: &FlowBuilder, operand: Operand, ty: Type) -> HeldOperand {
    HeldOperand {
        operand,
        block: builder.current,
        ty,
        spill: None,
    }
}

impl HeldOperand {
    fn read(&mut self, builder: &mut FlowBuilder) -> Operand {
        if !matches!(self.operand, Operand::Value(_)) || self.block == builder.current {
            return self.operand.clone();
        }
        let local = match self.spill {
            Some(local) => local,
            None => {
                let local = builder.local(self.ty.clone());
                builder.store_in(self.block, Place::Local(local), self.operand.clone());
                self.spill = Some(local);
                local
            }
        };
        load_place(builder, Place::Local(local), self.ty.clone())
    }
}

/// One step of a held place. An index holds the operand it lowered before
/// whatever follows; a field is just its ordinal.
enum HeldStep {
    Index(HeldOperand, std::ops::Range<usize>),
    Field(usize),
}

/// Where a held place starts: storage of its own, or the target of a pointer
/// lowered before whatever follows it.
enum HeldRoot {
    Place(Place),
    Indirect(HeldOperand),
}

/// A place whose steps were lowered before something that can split blocks,
/// such as the value an assignment stores or an index into the place, so the
/// place is rebuilt in whichever block uses it.
struct HeldPlace {
    root: HeldRoot,
    steps: Vec<HeldStep>,
}

fn hold_place(builder: &FlowBuilder, place: Place) -> HeldPlace {
    match place {
        Place::Local(_) | Place::Global(_) | Place::Constant(_) | Place::Result => HeldPlace {
            root: HeldRoot::Place(place),
            steps: Vec::new(),
        },
        Place::Indirect { pointer } => {
            let ty = builder.operand_type(&pointer);
            HeldPlace {
                root: HeldRoot::Indirect(hold_operand(builder, pointer, ty)),
                steps: Vec::new(),
            }
        }
        Place::Element { base, index, span } => {
            let mut held = hold_place(builder, *base);
            held.steps.push(HeldStep::Index(
                hold_operand(builder, index, Scalar::Int.into()),
                span,
            ));
            held
        }
        Place::Field { base, ordinal } => {
            let mut held = hold_place(builder, *base);
            held.steps.push(HeldStep::Field(ordinal));
            held
        }
    }
}

impl HeldPlace {
    fn read(&mut self, builder: &mut FlowBuilder) -> Place {
        let mut place = match &mut self.root {
            HeldRoot::Place(place) => place.clone(),
            HeldRoot::Indirect(pointer) => Place::Indirect {
                pointer: pointer.read(builder),
            },
        };
        for step in &mut self.steps {
            place = match step {
                HeldStep::Index(index, span) => Place::Element {
                    base: Box::new(place),
                    index: index.read(builder),
                    span: span.clone(),
                },
                HeldStep::Field(ordinal) => field_at(&place, *ordinal),
            };
        }
        place
    }
}

/// Lowers `right` after an already-lowered `left`, holding `left` so it is
/// readable in whichever block `right` lowering ends in.
fn lower_second_operand(
    checked: &CheckedProgram<'_>,
    left: Operand,
    left_type: Type,
    right: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> (Operand, Operand) {
    let mut held_left = hold_operand(builder, left, left_type);
    let right = lower_flow_operand(checked, right, bindings, builder);
    (held_left.read(builder), right)
}

fn load_place(builder: &mut FlowBuilder, place: Place, ty: Type) -> Operand {
    builder.value(Value {
        span: None,
        ty,
        kind: ValueKind::Load(place),
    })
}

fn lower_logical(
    checked: &CheckedProgram<'_>,
    operator: LogicalOperator,
    left: Idx<Expression>,
    right: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Operand {
    let left = lower_flow_operand(checked, left, bindings, builder);
    let right_block = builder.block();
    let short_block = builder.block();
    let join_block = builder.block();
    let result = builder.local(Scalar::Bool.into());
    let (then_target, else_target, short_value) = match operator {
        LogicalOperator::And => (right_block, short_block, 0),
        LogicalOperator::Or => (short_block, right_block, 1),
    };
    builder.terminate(Terminator::Branch {
        condition: left,
        then_target,
        else_target,
    });

    builder.select(short_block);
    builder.store(Place::Local(result), integer(short_value, Scalar::Bool));
    builder.terminate(Terminator::Jump { target: join_block });

    builder.select(right_block);
    let right = lower_flow_operand(checked, right, bindings, builder);
    builder.store(Place::Local(result), right);
    builder.terminate(Terminator::Jump { target: join_block });

    builder.select(join_block);
    load_place(builder, Place::Local(result), Scalar::Bool.into())
}
