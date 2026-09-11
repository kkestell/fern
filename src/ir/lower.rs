//! Lowering from the checked program to Fern IR.

use crate::{
    frontend::syntax::{
        Expression, ExpressionKind, ForHeader, Function as SyntaxFunction, Statement, StatementKind,
    },
    semantic::model::{
        Binding, CheckedLocation, CheckedLocationKind, CheckedProgram, Constant, ExpressionValue,
    },
    types::{BinaryOperator, ComparisonOperator, LogicalOperator, Scalar, Type},
};
use la_arena::Idx;
use lasso::Spur;
use num_traits::ToPrimitive;
use std::{collections::HashMap, ops::Index};

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
        let constant = match initializer {
            Some(initializer) => checked.expressions[*initializer]
                .constant
                .as_ref()
                .expect("module-level initializers are constant expressions"),
            None => &checked.zero_declarations[&statement],
        };
        let mut values = Vec::new();
        flatten(&checked, constant, &ty, &mut values);
        module_places.insert(binding, Place::Global(GlobalId(globals.len())));
        globals.push(Global { ty, values });
    }

    let functions = checked
        .syntax
        .functions
        .iter()
        .map(|(id, _)| lower_function(&checked, id, &module_places))
        .collect();
    Program {
        structs,
        globals,
        functions,
        main: function_id(checked.main),
    }
}

fn lower_function(
    checked: &CheckedProgram<'_>,
    id: Idx<SyntaxFunction>,
    module_places: &HashMap<Idx<Binding>, Place>,
) -> Function {
    let signature = &checked.functions[id];
    let mut builder = FlowBuilder::new();
    let mut bindings = Places::new(module_places);
    for &parameter in &signature.parameters {
        let local = builder.local(checked.bindings[parameter].ty.clone());
        bindings.insert(parameter, Place::Local(local));
    }
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

/// The immutable module storage and the locals one function owns while it is
/// lowered. Local bindings shadow module bindings without copying the latter.
struct Places<'a> {
    module: &'a HashMap<Idx<Binding>, Place>,
    locals: HashMap<Idx<Binding>, Place>,
}

impl<'a> Places<'a> {
    fn new(module: &'a HashMap<Idx<Binding>, Place>) -> Self {
        Self {
            module,
            locals: HashMap::new(),
        }
    }

    fn insert(&mut self, binding: Idx<Binding>, place: Place) {
        self.locals.insert(binding, place);
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

/// The IR literal a concrete scalar constant of type `ty` holds. Semantic
/// checking has already given every untyped constant its context type, so an
/// untyped rational never reaches lowering.
fn scalar_literal(constant: &Constant, ty: Scalar) -> Literal {
    match constant {
        Constant::Integer(value) => Literal::Integer {
            value: value
                .to_i128()
                .expect("concrete Fern value fits in the IR representation"),
            ty,
        },
        Constant::Float(value) => Literal::Floating(*value),
        Constant::Null => unreachable!("a null constant has a pointer type"),
        Constant::EmptySlice => unreachable!("a slice constant does not reach lowering"),
        Constant::Rational(_) => {
            unreachable!("an untyped constant is contextualized before lowering")
        }
        Constant::Array(_) | Constant::Struct(_) => {
            unreachable!("an aggregate constant is not a scalar")
        }
    }
}

fn pointer_literal(constant: &Constant, ty: &Type) -> Literal {
    match constant {
        Constant::Null => Literal::Null(ty.clone()),
        _ => unreachable!("a pointer constant is null"),
    }
}

fn slice_literal(constant: &Constant, ty: &Type) -> Literal {
    match constant {
        Constant::EmptySlice => Literal::EmptySlice(ty.clone()),
        _ => unreachable!("a slice constant is empty"),
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
        (Type::Scalar(scalar), scalar_constant) => {
            values.push(scalar_literal(scalar_constant, *scalar));
        }
        (Type::Pointer { .. }, constant) => values.push(pointer_literal(constant, ty)),
        (Type::Slice { .. }, constant) => values.push(slice_literal(constant, ty)),
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
    blocks: Vec<BuildingBlock>,
    current: BlockId,
}

impl FlowBuilder {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            locals: Vec::new(),
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

    fn check(&mut self, place: Place) {
        self.blocks[self.current.0]
            .instructions
            .push(Instruction::Check { place });
    }

    fn store(&mut self, place: Place, operand: Operand) {
        self.store_in(self.current, place, operand);
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
        span: std::ops::Range<usize>,
    ) -> Option<Operand> {
        let result = result.map(|ty| {
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
        return;
    }
    let ty = checked.bindings[binding].ty.clone();
    let Some(initializer) = initializer else {
        let local = builder.local(ty.clone());
        let place = Place::Local(local);
        store_constant(
            checked,
            builder,
            &place,
            &ty,
            &checked.zero_declarations[&statement],
            &checked.syntax.statements[statement].span,
        );
        bindings.insert(binding, place);
        return;
    };
    let operand = lower_flow_operand(checked, *initializer, bindings, builder);
    let local = builder.local(ty);
    builder.store(Place::Local(local), operand);
    bindings.insert(binding, Place::Local(local));
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
            let value = value.map(|value| lower_flow_operand(checked, value, bindings, builder));
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
            root: Some(bindings[binding].clone()),
            steps: Vec::new(),
        },
        CheckedLocationKind::SliceValue { .. } => {
            unreachable!("a slice value is only the operand of an index step")
        }
        CheckedLocationKind::Index {
            operand,
            index,
            implicit_dereference,
        } => {
            let reached = if implicit_dereference.is_some() {
                let Type::Pointer { target, .. } = &operand.ty else {
                    unreachable!("implicit index dereferences a pointer")
                };
                &**target
            } else {
                &operand.ty
            };
            if matches!(reached, Type::Slice { .. }) {
                let syntax_index = *index;
                let slice = if let Some(span) = implicit_dereference {
                    let mut held = lower_location(checked, operand, bindings, builder);
                    let place = held.read(builder);
                    let pointer = load_place(builder, place, operand.ty.clone());
                    load_place(
                        builder,
                        Place::Indirect {
                            pointer,
                            span: span.clone(),
                        },
                        reached.clone(),
                    )
                } else {
                    lower_location_slice(checked, operand, bindings, builder)
                };
                let index = lower_flow_operand(checked, syntax_index, bindings, builder);
                return HeldPlace {
                    root: None,
                    steps: vec![HeldStep::SliceIndex {
                        slice: hold_operand(builder, slice, reached.clone()),
                        index: hold_operand(builder, index, Scalar::Int.into()),
                        span: checked.syntax.expressions[syntax_index].span.clone(),
                    }],
                };
            }
            let mut held = lower_location(checked, operand, bindings, builder);
            if let Some(span) = implicit_dereference {
                let place = held.read(builder);
                let pointer = load_place(builder, place, operand.ty.clone());
                held.steps.push(HeldStep::Indirect(
                    hold_operand(builder, pointer, operand.ty.clone()),
                    span.clone(),
                ));
            }
            let syntax_index = *index;
            let index = lower_flow_operand(checked, syntax_index, bindings, builder);
            held.steps.push(HeldStep::Index(
                hold_operand(builder, index, Scalar::Int.into()),
                checked.syntax.expressions[syntax_index].span.clone(),
            ));
            held
        }
        CheckedLocationKind::Field {
            operand,
            ordinal,
            implicit_dereference,
        } => {
            let mut held = lower_location(checked, operand, bindings, builder);
            if let Some(span) = implicit_dereference {
                let place = held.read(builder);
                let pointer = load_place(builder, place, operand.ty.clone());
                held.steps.push(HeldStep::Indirect(
                    hold_operand(builder, pointer, operand.ty.clone()),
                    span.clone(),
                ));
            }
            held.steps.push(HeldStep::Field(*ordinal));
            held
        }
        CheckedLocationKind::Dereference { operand } => {
            let pointer = lower_flow_operand(checked, *operand, bindings, builder);
            HeldPlace {
                root: None,
                steps: vec![HeldStep::Indirect(
                    hold_operand(builder, pointer, checked.expressions[*operand].ty.clone()),
                    location.span.clone(),
                )],
            }
        }
    }
}

/// Reads the slice value an index location consumes. A slicing expression has
/// no storage, while a binding, field, or element does; both become the same
/// slice operand before `SliceElement` is built.
fn lower_location_slice(
    checked: &CheckedProgram<'_>,
    location: &CheckedLocation,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Operand {
    match &location.kind {
        CheckedLocationKind::SliceValue { operand } => {
            lower_flow_operand(checked, *operand, bindings, builder)
        }
        _ => {
            let mut held = lower_location(checked, location, bindings, builder);
            let place = held.read(builder);
            load_place(builder, place, location.ty.clone())
        }
    }
}

/// Lowers arguments left to right, holding each so a later argument that splits
/// blocks cannot leave an earlier one unreadable in the call's block.
fn lower_call(
    checked: &CheckedProgram<'_>,
    function: Idx<SyntaxFunction>,
    arguments: &[Idx<Expression>],
    span: std::ops::Range<usize>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Option<Operand> {
    let mut held = Vec::with_capacity(arguments.len());
    for &argument in arguments {
        let ty = checked.expressions[argument].ty.clone();
        let operand = lower_flow_operand(checked, argument, bindings, builder);
        held.push(hold_operand(builder, operand, ty));
    }
    let arguments = held
        .iter_mut()
        .map(|operand| operand.read(builder))
        .collect();
    builder.call(
        function_id(function),
        arguments,
        checked.functions[function].result.clone(),
        span,
    )
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

/// The captured value a `for … in` walks.
enum IterationSource {
    Array { place: Place, length: u64 },
    Slice { local: LocalId, ty: Type },
}

/// A `for … in` loop's storage, captured once before the first iteration, and
/// the counter driving it.
struct Iteration {
    source: IterationSource,
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
        let right = match &self.source {
            IterationSource::Array { length, .. } => integer(i128::from(*length), Scalar::Int),
            IterationSource::Slice { local, ty } => {
                let slice = load_place(builder, Place::Local(*local), ty.clone());
                builder.value(Value {
                    span: None,
                    ty: Scalar::Int.into(),
                    kind: ValueKind::SliceLength { slice },
                })
            }
        };
        builder.value(Value {
            span: Some(self.span.clone()),
            ty: Scalar::Bool.into(),
            kind: ValueKind::Comparison {
                operator: ComparisonOperator::Less,
                left: counter,
                right,
            },
        })
    }

    /// Binds the value to a copy of the element the counter reaches.
    fn bind(&self, builder: &mut FlowBuilder) {
        let index = self.counter(builder);
        let element = match &self.source {
            IterationSource::Array { place, .. } => Place::Element {
                base: Box::new(place.clone()),
                index,
                span: self.span.clone(),
            },
            IterationSource::Slice { local, ty } => Place::SliceElement {
                slice: load_place(builder, Place::Local(*local), ty.clone()),
                index,
                span: self.span.clone(),
            },
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

/// Captures the value a `for … in` walks, so assigning to its original binding
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
    let (source, element) = match &ty {
        Type::Array { length, element } => {
            let source = lower_aggregate_place(checked, operand, bindings, builder);
            let loaded = load_place(builder, source, ty.clone());
            let place = Place::Local(builder.local(ty.clone()));
            builder.store(place.clone(), loaded);
            (
                IterationSource::Array {
                    place,
                    length: *length,
                },
                (**element).clone(),
            )
        }
        Type::Slice { element, .. } => {
            let loaded = lower_flow_operand(checked, operand, bindings, builder);
            let local = builder.local(ty.clone());
            builder.store(Place::Local(local), loaded);
            (
                IterationSource::Slice {
                    local,
                    ty: ty.clone(),
                },
                (**element).clone(),
            )
        }
        _ => unreachable!("checking requires an array or slice operand for `for … in`"),
    };

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
        source,
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

/// Stores a folded aggregate into `place`, one scalar per element or field.
fn store_constant(
    checked: &CheckedProgram<'_>,
    builder: &mut FlowBuilder,
    place: &Place,
    ty: &Type,
    constant: &Constant,
    span: &std::ops::Range<usize>,
) {
    match (ty, constant) {
        (Type::Scalar(scalar), constant) => {
            builder.store(
                place.clone(),
                Operand::Literal(scalar_literal(constant, *scalar)),
            );
        }
        (Type::Pointer { .. }, constant) => {
            builder.store(
                place.clone(),
                Operand::Literal(pointer_literal(constant, ty)),
            );
        }
        (Type::Slice { .. }, constant) => {
            builder.store(place.clone(), Operand::Literal(slice_literal(constant, ty)));
        }
        (Type::Array { element, .. }, Constant::Array(elements)) => {
            for (index, value) in elements.iter().enumerate() {
                let index = u64::try_from(index).expect("an array fits in the address space");
                store_constant(
                    checked,
                    builder,
                    &element_at(place, index, span),
                    element,
                    value,
                    span,
                );
            }
        }
        (Type::Struct(_), Constant::Struct(fields)) => {
            for (ordinal, value) in fields.iter().enumerate() {
                store_constant(
                    checked,
                    builder,
                    &field_at(place, ordinal),
                    &field_type(checked, ty, ordinal),
                    value,
                    span,
                );
            }
        }
        _ => unreachable!("a folded constant has the shape of its type"),
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
    let span = &checked.syntax.expressions[id].span;
    for &(ordinal, value) in initializers {
        let operand = lower_flow_operand(checked, value, bindings, builder);
        builder.store(field_at(place, ordinal), operand);
    }
    for (ordinal, constant) in filled {
        store_constant(
            checked,
            builder,
            &field_at(place, *ordinal),
            &field_type(checked, &ty, *ordinal),
            constant,
            span,
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
    for index in filled..*length {
        builder.store(element_at(place, index, span), last.clone());
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
        let place = Place::Local(builder.local(ty.clone()));
        let span = &checked.syntax.expressions[id].span;
        store_constant(checked, builder, &place, ty, constant, span);
        return place;
    }
    match &expression.value {
        ExpressionValue::Reference(binding) => bindings[binding].clone(),
        ExpressionValue::Grouping { expression } => {
            lower_aggregate_place(checked, *expression, bindings, builder)
        }
        ExpressionValue::Index { .. } => element_place(checked, id, bindings, builder),
        ExpressionValue::Field { .. } => field_place(checked, id, bindings, builder),
        ExpressionValue::Dereference { operand } => Place::Indirect {
            pointer: lower_flow_operand(checked, *operand, bindings, builder),
            span: checked.syntax.expressions[id].span.clone(),
        },
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
        ExpressionValue::Slice { .. } => {
            let operand = lower_slicing(checked, id, bindings, builder);
            let place = Place::Local(builder.local(ty.clone()));
            builder.store(place.clone(), operand);
            place
        }
        _ => unreachable!(
            "an aggregate value is a reference, a literal, an element, a field, or a call"
        ),
    }
}

/// Lowers a slicing expression into a range of a two-word slice value.
fn lower_slicing(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Operand {
    let ExpressionValue::Slice {
        operand,
        low,
        high,
        implicit_dereference,
    } = &checked.expressions[id].value
    else {
        unreachable!("a slice value is lowered from a slicing expression")
    };
    let operand_ty = &checked.expressions[*operand].ty;
    let reached = if *implicit_dereference {
        let Type::Pointer { target, .. } = operand_ty else {
            unreachable!("implicit slicing dereferences a pointer")
        };
        &**target
    } else {
        operand_ty
    };
    let (slice, array_length) = match reached {
        Type::Array { length, .. } => {
            let place = if *implicit_dereference {
                Place::Indirect {
                    pointer: lower_flow_operand(checked, *operand, bindings, builder),
                    span: checked.syntax.expressions[*operand].span.clone(),
                }
            } else {
                lower_aggregate_place(checked, *operand, bindings, builder)
            };
            let ty = checked.expressions[id].ty.clone();
            (
                builder.value(Value {
                    span: None,
                    ty,
                    kind: ValueKind::WholeSlice(place),
                }),
                Some(*length),
            )
        }
        Type::Slice { .. } => {
            let slice = if *implicit_dereference {
                let pointer = lower_flow_operand(checked, *operand, bindings, builder);
                load_place(
                    builder,
                    Place::Indirect {
                        pointer,
                        span: checked.syntax.expressions[*operand].span.clone(),
                    },
                    reached.clone(),
                )
            } else {
                lower_flow_operand(checked, *operand, bindings, builder)
            };
            (slice, None)
        }
        _ => unreachable!("checking requires an array or slice operand for slicing"),
    };
    let mut slice = hold_operand(builder, slice, checked.expressions[id].ty.clone());
    let low = low
        .map(|low| lower_flow_operand(checked, low, bindings, builder))
        .unwrap_or_else(|| integer(0, Scalar::Int));
    let mut low = hold_operand(builder, low, Scalar::Int.into());
    let high = match high {
        Some(high) => lower_flow_operand(checked, *high, bindings, builder),
        None => match array_length {
            Some(length) => integer(i128::from(length), Scalar::Int),
            None => {
                let slice = slice.read(builder);
                builder.value(Value {
                    span: None,
                    ty: Scalar::Int.into(),
                    kind: ValueKind::SliceLength { slice },
                })
            }
        },
    };
    let slice = slice.read(builder);
    let low = low.read(builder);
    builder.value(Value {
        span: Some(checked.syntax.expressions[id].span.clone()),
        ty: checked.expressions[id].ty.clone(),
        kind: ValueKind::SliceRange { slice, low, high },
    })
}

fn element_place(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Place {
    let ExpressionValue::Index {
        operand,
        index,
        implicit_dereference,
    } = &checked.expressions[id].value
    else {
        unreachable!("an element place is lowered from an index expression")
    };
    let span = checked.syntax.expressions[*index].span.clone();
    let operand_ty = &checked.expressions[*operand].ty;
    let reached = if *implicit_dereference {
        let Type::Pointer { target, .. } = operand_ty else {
            unreachable!("implicit index dereferences a pointer")
        };
        &**target
    } else {
        operand_ty
    };
    if matches!(reached, Type::Slice { .. }) {
        let slice = if *implicit_dereference {
            let pointer = lower_flow_operand(checked, *operand, bindings, builder);
            load_place(
                builder,
                Place::Indirect {
                    pointer,
                    span: checked.syntax.expressions[*operand].span.clone(),
                },
                reached.clone(),
            )
        } else {
            lower_flow_operand(checked, *operand, bindings, builder)
        };
        let index = lower_flow_operand(checked, *index, bindings, builder);
        return Place::SliceElement { slice, index, span };
    }
    let base = if *implicit_dereference {
        Place::Indirect {
            pointer: lower_flow_operand(checked, *operand, bindings, builder),
            span: checked.syntax.expressions[*operand].span.clone(),
        }
    } else {
        lower_aggregate_place(checked, *operand, bindings, builder)
    };
    let index = lower_flow_operand(checked, *index, bindings, builder);
    Place::Element {
        base: Box::new(base),
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
    let ExpressionValue::Field {
        operand,
        ordinal,
        implicit_dereference,
    } = &checked.expressions[id].value
    else {
        unreachable!("a field place is lowered from a field expression")
    };
    let base = if *implicit_dereference {
        Place::Indirect {
            pointer: lower_flow_operand(checked, *operand, bindings, builder),
            span: checked.syntax.expressions[*operand].span.clone(),
        }
    } else {
        lower_aggregate_place(checked, *operand, bindings, builder)
    };
    field_at(&base, *ordinal)
}

fn lower_expression_location(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Place {
    match &checked.expressions[id].value {
        ExpressionValue::Reference(binding) => bindings[binding].clone(),
        ExpressionValue::Grouping { expression } => {
            lower_expression_location(checked, *expression, bindings, builder)
        }
        ExpressionValue::Index { .. } => element_place(checked, id, bindings, builder),
        ExpressionValue::Field { .. } => field_place(checked, id, bindings, builder),
        ExpressionValue::Dereference { operand } => Place::Indirect {
            pointer: lower_flow_operand(checked, *operand, bindings, builder),
            span: checked.syntax.expressions[id].span.clone(),
        },
        _ => unreachable!("semantic checking requires an addressable expression"),
    }
}

/// The length `len(operand)` reads from its operand's type. The operand is
/// still evaluated, so a trap inside it still happens.
fn lower_length(
    checked: &CheckedProgram<'_>,
    operand: Idx<Expression>,
    bindings: &Places<'_>,
    builder: &mut FlowBuilder,
) -> Operand {
    let operand_ty = &checked.expressions[operand].ty;
    let (array, implicit) = match operand_ty {
        Type::Pointer { target, .. } => (&**target, true),
        _ => (operand_ty, false),
    };
    match array {
        Type::Array { length, .. } => {
            let value = lower_flow_operand(checked, operand, bindings, builder);
            if implicit {
                builder.check(Place::Indirect {
                    pointer: value,
                    span: checked.syntax.expressions[operand].span.clone(),
                });
            }
            integer(i128::from(*length), Scalar::Int)
        }
        Type::Slice { .. } => {
            let slice = if implicit {
                let pointer = lower_flow_operand(checked, operand, bindings, builder);
                load_place(
                    builder,
                    Place::Indirect {
                        pointer,
                        span: checked.syntax.expressions[operand].span.clone(),
                    },
                    array.clone(),
                )
            } else {
                lower_flow_operand(checked, operand, bindings, builder)
            };
            builder.value(Value {
                span: None,
                ty: Scalar::Int.into(),
                kind: ValueKind::SliceLength { slice },
            })
        }
        _ => unreachable!("checking requires an array or slice operand for `len`"),
    }
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
        ExpressionValue::Length { operand, .. } => {
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
        // A call, an index, and a field never fold, so a folded expression
        // only reaches them under a `len`, and the rest hold no
        // sub-expression at all.
        ExpressionValue::Integer
        | ExpressionValue::CompoundAssignmentTarget
        | ExpressionValue::Floating
        | ExpressionValue::Boolean
        | ExpressionValue::Null
        | ExpressionValue::Reference(_)
        | ExpressionValue::Call { .. }
        | ExpressionValue::Index { .. }
        | ExpressionValue::Field { .. } => {}
        ExpressionValue::Slice { .. } => unreachable!("a slicing expression is never constant"),
        ExpressionValue::AddressOf { .. } | ExpressionValue::Dereference { .. } => {
            unreachable!("pointers stop before IR lowering")
        }
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
    if let Type::Pointer { .. } = &expression.ty {
        if let Some(constant) = expression.constant.as_ref() {
            lower_folded_effects(checked, id, bindings, builder);
            return Operand::Literal(pointer_literal(constant, &expression.ty));
        }
        return match &expression.value {
            ExpressionValue::Reference(binding) => {
                load_place(builder, bindings[binding].clone(), expression.ty.clone())
            }
            ExpressionValue::Grouping { expression } => {
                lower_flow_operand(checked, *expression, bindings, builder)
            }
            // Converting `*T` to `*const T` changes only the type-system
            // permission to write through the pointer, never its bits.
            ExpressionValue::Conversion { operand, .. } => {
                lower_flow_operand(checked, *operand, bindings, builder)
            }
            ExpressionValue::Index { .. } => {
                let place = element_place(checked, id, bindings, builder);
                load_place(builder, place, expression.ty.clone())
            }
            ExpressionValue::Field { .. } => {
                let place = field_place(checked, id, bindings, builder);
                load_place(builder, place, expression.ty.clone())
            }
            ExpressionValue::Dereference { operand } => {
                let pointer = lower_flow_operand(checked, *operand, bindings, builder);
                load_place(
                    builder,
                    Place::Indirect {
                        pointer,
                        span: checked.syntax.expressions[id].span.clone(),
                    },
                    expression.ty.clone(),
                )
            }
            ExpressionValue::AddressOf { operand } => {
                let place = lower_expression_location(checked, *operand, bindings, builder);
                builder.value(Value {
                    span: Some(checked.syntax.expressions[id].span.clone()),
                    ty: expression.ty.clone(),
                    kind: ValueKind::AddressOf(place),
                })
            }
            ExpressionValue::Call { .. } => lower_call_expression(checked, id, bindings, builder),
            _ => unreachable!("a pointer value is a reference, null, address, grouping, or call"),
        };
    }
    if matches!(expression.ty, Type::Slice { .. }) {
        if let Some(constant) = expression.constant.as_ref() {
            lower_folded_effects(checked, id, bindings, builder);
            return Operand::Literal(slice_literal(constant, &expression.ty));
        }
        return match &expression.value {
            ExpressionValue::Grouping { expression } => {
                lower_flow_operand(checked, *expression, bindings, builder)
            }
            ExpressionValue::Conversion { operand, .. } => {
                lower_flow_operand(checked, *operand, bindings, builder)
            }
            ExpressionValue::Slice { .. } => lower_slicing(checked, id, bindings, builder),
            _ => {
                let place = lower_aggregate_place(checked, id, bindings, builder);
                load_place(builder, place, expression.ty.clone())
            }
        };
    }
    let Some(scalar) = expression.ty.scalar() else {
        let place = lower_aggregate_place(checked, id, bindings, builder);
        return load_place(builder, place, expression.ty.clone());
    };
    let ty: Type = scalar.into();
    if let Some(constant) = expression.constant.as_ref() {
        lower_folded_effects(checked, id, bindings, builder);
        return Operand::Literal(scalar_literal(constant, scalar));
    }
    match &expression.value {
        ExpressionValue::Integer
        | ExpressionValue::CompoundAssignmentTarget
        | ExpressionValue::Floating
        | ExpressionValue::Boolean => {
            unreachable!("literal expressions are constant")
        }
        ExpressionValue::Array { .. } => unreachable!("an array literal has an array type"),
        ExpressionValue::Struct { .. } => unreachable!("a struct literal has a struct type"),
        // A `len` reaches here only when its operand holds a call, which is
        // what stops it from folding.
        ExpressionValue::Length {
            operand,
            implicit_dereference: _,
        } => lower_length(checked, *operand, bindings, builder),
        ExpressionValue::Index { .. } => {
            let place = element_place(checked, id, bindings, builder);
            load_place(builder, place, ty)
        }
        ExpressionValue::Field { .. } => {
            let place = field_place(checked, id, bindings, builder);
            load_place(builder, place, ty)
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
                ty: Scalar::Bool.into(),
                kind: ValueKind::Comparison {
                    operator: *operator,
                    left,
                    right,
                },
            })
        }
        ExpressionValue::LogicalNot {
            operator_span,
            operand,
        } => {
            let operand = lower_flow_operand(checked, *operand, bindings, builder);
            builder.value(Value {
                span: Some(operator_span.clone()),
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
        ExpressionValue::Dereference { operand } => {
            let pointer = lower_flow_operand(checked, *operand, bindings, builder);
            load_place(
                builder,
                Place::Indirect {
                    pointer,
                    span: checked.syntax.expressions[id].span.clone(),
                },
                ty,
            )
        }
        ExpressionValue::Null | ExpressionValue::AddressOf { .. } => {
            unreachable!("pointer expressions are handled above")
        }
        ExpressionValue::Slice { .. } => unreachable!("slice values are handled above"),
    }
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

/// One step of an assignment target. An index holds the operand it lowered
/// before the assigned value; a field is just its ordinal.
enum HeldStep {
    Index(HeldOperand, std::ops::Range<usize>),
    SliceIndex {
        slice: HeldOperand,
        index: HeldOperand,
        span: std::ops::Range<usize>,
    },
    Field(usize),
    Indirect(HeldOperand, std::ops::Range<usize>),
}

/// An assignment target whose steps were lowered before the value stored
/// through it, so the place is rebuilt in whichever block the store lands in.
struct HeldPlace {
    root: Option<Place>,
    steps: Vec<HeldStep>,
}

impl HeldPlace {
    fn read(&mut self, builder: &mut FlowBuilder) -> Place {
        let mut place = self.root.clone();
        for step in &mut self.steps {
            place = Some(match step {
                HeldStep::Index(index, span) => Place::Element {
                    base: Box::new(place.expect("an index follows a base place")),
                    index: index.read(builder),
                    span: span.clone(),
                },
                HeldStep::SliceIndex { slice, index, span } => Place::SliceElement {
                    slice: slice.read(builder),
                    index: index.read(builder),
                    span: span.clone(),
                },
                HeldStep::Field(ordinal) => {
                    field_at(&place.expect("a field follows a base place"), *ordinal)
                }
                HeldStep::Indirect(pointer, span) => Place::Indirect {
                    pointer: pointer.read(builder),
                    span: span.clone(),
                },
            });
        }
        place.expect("a checked location has a place")
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
