use crate::{
    CompileError,
    frontend::{
        BinaryOperator, ComparisonOperator, Expression, ExpressionKind, ForHeader,
        Function as SyntaxFunction, LogicalOperator, Statement, StatementKind, UnaryOperator,
    },
    semantic::{Binding, CheckedProgram, ExpressionValue},
    types::Type,
};
use la_arena::Idx;
use lasso::Spur;
use num_traits::ToPrimitive;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ValueId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LocalId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlobalId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FunctionId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockId(pub usize);

/// A storage location a load reads and a store writes. Locals live for one call
/// of one function; globals are the module-level `var` bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Place {
    Local(LocalId),
    Global(GlobalId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operand {
    // i128 holds both the signed minima and the full u64 range without bit reinterpretation.
    Integer { value: i128, ty: Type },
    Value(ValueId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValueKind {
    Load(Place),
    /// The result of the `Instruction::Call` that defines it. The call carries
    /// the arguments and the source span.
    CallResult,
    Convert {
        operand: Operand,
        truncating: bool,
    },
    Unary {
        operator: UnaryOperator,
        operand: Operand,
    },
    Binary {
        operator: BinaryOperator,
        form: BinaryForm,
        left: Operand,
        right: Operand,
    },
    Comparison {
        operator: ComparisonOperator,
        left: Operand,
        right: Operand,
    },
    LogicalNot {
        operand: Operand,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryForm {
    Infix,
    CompoundAssignment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Value {
    pub span: Option<std::ops::Range<usize>>,
    pub ty: Type,
    pub kind: ValueKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Instruction {
    Value(ValueId),
    Store {
        place: Place,
        operand: Operand,
    },
    /// Calls `function` with `arguments` in source order. `result` names the
    /// defined value exactly when the callee returns one.
    Call {
        result: Option<ValueId>,
        function: FunctionId,
        arguments: Vec<Operand>,
        span: std::ops::Range<usize>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Terminator {
    Jump {
        target: BlockId,
    },
    Branch {
        condition: Operand,
        then_target: BlockId,
        else_target: BlockId,
    },
    Exit {
        status: Operand,
    },
    Return {
        value: Option<Operand>,
    },
    Unreachable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Block {
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ControlFlow {
    pub entry: BlockId,
    pub locals: Vec<Type>,
    pub blocks: Vec<Block>,
}

/// A module-level `var`. Its initializer is a constant expression, so it needs
/// an initial value rather than initialization code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Global {
    pub ty: Type,
    pub value: i128,
}

#[derive(Debug)]
pub(crate) struct Function {
    /// The first `parameters` entries of `flow.locals` are the parameters, in
    /// source order, and hold their arguments on entry.
    pub parameters: usize,
    pub result: Option<Type>,
    // Each position defines the corresponding function-local value ID.
    pub values: Vec<Value>,
    pub flow: ControlFlow,
}

#[derive(Debug)]
pub(crate) struct Program {
    pub globals: Vec<Global>,
    // Each position defines the corresponding function ID.
    pub functions: Vec<Function>,
    pub main: FunctionId,
}

#[derive(Debug)]
pub(crate) struct VerifiedProgram(Program);

impl VerifiedProgram {
    pub(crate) fn program(&self) -> &Program {
        &self.0
    }
}

fn verify_integer(value: i128, ty: Type) -> Result<Type, CompileError> {
    if value < ty.min() || value > i128::from(ty.max()) {
        return Err(CompileError::new(format!(
            "internal compiler error: IR integer {value} out of range for {ty:?}"
        )));
    }
    Ok(ty)
}

fn valid_value(
    value: &Value,
    mut operand_type: impl FnMut(Operand) -> Result<Type, CompileError>,
    mut place_type: impl FnMut(Place) -> Result<Type, CompileError>,
) -> Result<bool, CompileError> {
    Ok(match value.kind {
        ValueKind::Load(place) => place_type(place)? == value.ty && value.span.is_none(),
        // The defining call checks the result type against the callee's
        // signature, and owns the span the call reports.
        ValueKind::CallResult => value.span.is_none(),
        ValueKind::Convert {
            operand,
            truncating,
        } => {
            let source = operand_type(operand)?;
            source != Type::Bool
                && value.ty != Type::Bool
                && (truncating || source.all_values_fit(value.ty) || value.span.is_some())
        }
        ValueKind::Unary { operator, operand } => {
            let source = operand_type(operand)?;
            value.span.is_some()
                && source == value.ty
                && source != Type::Bool
                && (operator != UnaryOperator::Negate || source.signed())
        }
        ValueKind::Binary {
            operator,
            left,
            right,
            ..
        } => {
            let left = operand_type(left)?;
            let right = operand_type(right)?;
            value.span.is_some()
                && left == value.ty
                && left != Type::Bool
                && right != Type::Bool
                && (matches!(
                    operator,
                    BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight
                ) || right == left)
        }
        ValueKind::Comparison { left, right, .. } => {
            let left = operand_type(left)?;
            let right = operand_type(right)?;
            value.span.is_some() && value.ty == Type::Bool && left == right
        }
        ValueKind::LogicalNot { operand } => {
            value.span.is_some() && value.ty == Type::Bool && operand_type(operand)? == Type::Bool
        }
    })
}

/// Where a value is defined, and whether a call defined it.
#[derive(Clone, Copy)]
struct Definition {
    block: usize,
    position: usize,
    from_call: bool,
}

impl Program {
    pub(crate) fn verify(self) -> Result<VerifiedProgram, CompileError> {
        for (id, global) in self.globals.iter().enumerate() {
            verify_integer(global.value, global.ty).map_err(|error| {
                CompileError::new(format!("{error} (IR global {id} initial value)"))
            })?;
        }
        let main = self.function(self.main)?;
        if main.parameters != 0 || main.result.is_some() {
            return Err(CompileError::new(
                "internal compiler error: IR main takes parameters or returns a value",
            ));
        }
        for function in &self.functions {
            verify_parameters(function)?;
        }
        for function in &self.functions {
            self.verify_function(function)?;
        }
        Ok(VerifiedProgram(self))
    }

    fn function(&self, FunctionId(id): FunctionId) -> Result<&Function, CompileError> {
        self.functions.get(id).ok_or_else(|| {
            CompileError::new(format!(
                "internal compiler error: IR calls unknown function {id}"
            ))
        })
    }

    fn place_type(&self, flow: &ControlFlow, place: Place) -> Result<Type, CompileError> {
        match place {
            Place::Local(LocalId(id)) => flow.locals.get(id).copied().ok_or_else(|| {
                CompileError::new(format!(
                    "internal compiler error: IR uses unknown local {id}"
                ))
            }),
            Place::Global(GlobalId(id)) => {
                self.globals.get(id).map(|global| global.ty).ok_or_else(|| {
                    CompileError::new(format!(
                        "internal compiler error: IR uses unknown global {id}"
                    ))
                })
            }
        }
    }

    fn verify_function(&self, function: &Function) -> Result<(), CompileError> {
        let flow = &function.flow;
        verify_target(flow.entry, flow)?;

        let mut definitions: Vec<Option<Definition>> = vec![None; function.values.len()];
        for (block_index, block) in flow.blocks.iter().enumerate() {
            for (position, instruction) in block.instructions.iter().enumerate() {
                let defined = match instruction {
                    Instruction::Value(id) => Some((*id, false)),
                    Instruction::Call { result, .. } => result.map(|id| (id, true)),
                    Instruction::Store { .. } => None,
                };
                let Some((ValueId(id), from_call)) = defined else {
                    continue;
                };
                let Some(definition) = definitions.get_mut(id) else {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR defines unknown value {id}"
                    )));
                };
                if definition
                    .replace(Definition {
                        block: block_index,
                        position,
                        from_call,
                    })
                    .is_some()
                {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR defines value {id} more than once"
                    )));
                }
            }
        }
        if let Some(id) = definitions.iter().position(Option::is_none) {
            return Err(CompileError::new(format!(
                "internal compiler error: IR does not define value {id}"
            )));
        }
        // A call result exists only where a call produces it, and a call
        // produces nothing else.
        for (id, (value, definition)) in function.values.iter().zip(&definitions).enumerate() {
            let from_call = definition.expect("every value is defined").from_call;
            if (value.kind == ValueKind::CallResult) != from_call {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR value {id} and its definition disagree on being a call result"
                )));
            }
        }

        let reachable = reachable_blocks(flow)?;
        let initialized_at_entry = initialized_locals(function, &reachable);
        for (block_index, block) in flow.blocks.iter().enumerate() {
            if reachable[block_index] && block.terminator == Terminator::Unreachable {
                return Err(CompileError::new(format!(
                    "internal compiler error: reachable IR block {block_index} is not terminated"
                )));
            }
            let mut initialized = initialized_at_entry[block_index].clone();
            for (position, instruction) in block.instructions.iter().enumerate() {
                let operand_type = |operand| {
                    flow_operand_type(
                        operand,
                        &function.values,
                        &definitions,
                        block_index,
                        position,
                    )
                };
                match instruction {
                    Instruction::Value(ValueId(id)) => {
                        let value = &function.values[*id];
                        if let ValueKind::Load(Place::Local(LocalId(local))) = value.kind
                            && local < flow.locals.len()
                            && !initialized.contains(local)
                        {
                            return Err(CompileError::new(format!(
                                "internal compiler error: IR loads uninitialized local {local}"
                            )));
                        }
                        if !valid_value(value, operand_type, |place| self.place_type(flow, place))?
                        {
                            return Err(invalid_value(*id, value));
                        }
                    }
                    Instruction::Store { place, operand } => {
                        let destination = self.place_type(flow, *place)?;
                        let source = operand_type(*operand)?;
                        if source != destination {
                            return Err(CompileError::new(format!(
                                "internal compiler error: IR store has type {source:?}, expected {destination:?}"
                            )));
                        }
                        if let Place::Local(local) = place {
                            initialized.insert(local.0);
                        }
                    }
                    Instruction::Call {
                        result,
                        function: callee,
                        arguments,
                        ..
                    } => self.verify_call(function, *callee, arguments, *result, operand_type)?,
                }
            }
            let end = block.instructions.len();
            let operand_type = |operand| {
                flow_operand_type(operand, &function.values, &definitions, block_index, end)
            };
            match &block.terminator {
                Terminator::Jump { target, .. } => verify_target(*target, flow)?,
                Terminator::Branch {
                    condition,
                    then_target,
                    else_target,
                    ..
                } => {
                    verify_target(*then_target, flow)?;
                    verify_target(*else_target, flow)?;
                    if operand_type(*condition)? != Type::Bool {
                        return Err(CompileError::new(
                            "internal compiler error: IR branch requires bool",
                        ));
                    }
                }
                Terminator::Exit { status, .. } => {
                    if operand_type(*status)? != Type::Int {
                        return Err(CompileError::new(
                            "internal compiler error: IR exit requires int",
                        ));
                    }
                }
                Terminator::Return { value } => match (function.result, value) {
                    (None, None) => {}
                    (None, Some(_)) => {
                        return Err(CompileError::new(
                            "internal compiler error: IR returns a value from a void function",
                        ));
                    }
                    (Some(_), None) => {
                        return Err(CompileError::new(
                            "internal compiler error: IR return is missing its value",
                        ));
                    }
                    (Some(result), Some(value)) => {
                        let source = operand_type(*value)?;
                        if source != result {
                            return Err(CompileError::new(format!(
                                "internal compiler error: IR returns {source:?}, expected {result:?}"
                            )));
                        }
                    }
                },
                Terminator::Unreachable => {}
            }
        }
        Ok(())
    }

    fn verify_call(
        &self,
        caller: &Function,
        callee: FunctionId,
        arguments: &[Operand],
        result: Option<ValueId>,
        mut operand_type: impl FnMut(Operand) -> Result<Type, CompileError>,
    ) -> Result<(), CompileError> {
        let target = self.function(callee)?;
        if arguments.len() != target.parameters {
            return Err(CompileError::new(format!(
                "internal compiler error: IR call passes {} arguments to function {} taking {}",
                arguments.len(),
                callee.0,
                target.parameters
            )));
        }
        for (index, &argument) in arguments.iter().enumerate() {
            let source = operand_type(argument)?;
            let parameter = target.flow.locals[index];
            if source != parameter {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR call argument {index} has type {source:?}, expected {parameter:?}"
                )));
            }
        }
        match (target.result, result) {
            (None, None) => {}
            (None, Some(_)) => Err(CompileError::new(format!(
                "internal compiler error: IR call defines a result for void function {}",
                callee.0
            )))?,
            (Some(_), None) => Err(CompileError::new(format!(
                "internal compiler error: IR call discards the result of function {} in the IR",
                callee.0
            )))?,
            (Some(expected), Some(ValueId(id))) => {
                let defined = caller.values[id].ty;
                if defined != expected {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR call result has type {defined:?}, expected {expected:?}"
                    )));
                }
            }
        }
        Ok(())
    }
}

const LOCAL_BITS_PER_WORD: usize = u64::BITS as usize;

#[derive(Clone, PartialEq, Eq)]
struct LocalSet(Vec<u64>);

impl LocalSet {
    fn empty(local_count: usize) -> Self {
        Self(vec![0; local_count.div_ceil(LOCAL_BITS_PER_WORD)])
    }

    fn full(local_count: usize) -> Self {
        Self(vec![u64::MAX; local_count.div_ceil(LOCAL_BITS_PER_WORD)])
    }

    fn contains(&self, local: usize) -> bool {
        self.0
            .get(local / LOCAL_BITS_PER_WORD)
            .is_some_and(|word| word & (1 << (local % LOCAL_BITS_PER_WORD)) != 0)
    }

    fn insert(&mut self, local: usize) {
        if let Some(word) = self.0.get_mut(local / LOCAL_BITS_PER_WORD) {
            *word |= 1 << (local % LOCAL_BITS_PER_WORD);
        }
    }

    fn intersect(&mut self, other: &Self) {
        for (word, other) in self.0.iter_mut().zip(&other.0) {
            *word &= other;
        }
    }
}

fn initialized_locals(function: &Function, reachable: &[bool]) -> Vec<LocalSet> {
    let flow = &function.flow;
    let mut predecessors = vec![Vec::new(); flow.blocks.len()];
    let mut stores = vec![Vec::new(); flow.blocks.len()];
    for (block_index, block) in flow.blocks.iter().enumerate() {
        for instruction in &block.instructions {
            if let Instruction::Store {
                place: Place::Local(local),
                ..
            } = instruction
            {
                stores[block_index].push(local.0);
            }
        }
        match block.terminator {
            Terminator::Jump { target } => {
                if let Some(target_predecessors) = predecessors.get_mut(target.0) {
                    target_predecessors.push(block_index);
                }
            }
            Terminator::Branch {
                then_target,
                else_target,
                ..
            } => {
                if let Some(target_predecessors) = predecessors.get_mut(then_target.0) {
                    target_predecessors.push(block_index);
                }
                if let Some(target_predecessors) = predecessors.get_mut(else_target.0) {
                    target_predecessors.push(block_index);
                }
            }
            Terminator::Exit { .. } | Terminator::Return { .. } | Terminator::Unreachable => {}
        }
    }

    let mut initialized = vec![LocalSet::full(flow.locals.len()); flow.blocks.len()];
    // Arguments initialize the parameters before the entry block runs.
    let mut at_entry = LocalSet::empty(flow.locals.len());
    for parameter in 0..function.parameters {
        at_entry.insert(parameter);
    }
    initialized[flow.entry.0] = at_entry;
    for (block, is_reachable) in reachable.iter().enumerate() {
        if !is_reachable {
            initialized[block] = LocalSet::empty(flow.locals.len());
        }
    }
    loop {
        let mut changed = false;
        for block in 0..flow.blocks.len() {
            if !reachable[block] || block == flow.entry.0 {
                continue;
            }
            let mut incoming = LocalSet::full(flow.locals.len());
            for &predecessor in predecessors[block]
                .iter()
                .filter(|predecessor| reachable[**predecessor])
            {
                let mut predecessor_initialized = initialized[predecessor].clone();
                for &local in &stores[predecessor] {
                    predecessor_initialized.insert(local);
                }
                incoming.intersect(&predecessor_initialized);
            }
            if incoming != initialized[block] {
                initialized[block] = incoming;
                changed = true;
            }
        }
        if !changed {
            return initialized;
        }
    }
}

fn invalid_value(index: usize, value: &Value) -> CompileError {
    CompileError::new(format!(
        "internal compiler error: invalid IR value {index}: {:?} with result {:?}",
        value.kind, value.ty
    ))
}

/// Checks that a function's parameters name locals. Every function passes this
/// before any body is verified, so a call can read its callee's parameter
/// types.
fn verify_parameters(function: &Function) -> Result<(), CompileError> {
    if function.parameters > function.flow.locals.len() {
        return Err(CompileError::new(format!(
            "internal compiler error: IR function declares {} parameters but only {} locals",
            function.parameters,
            function.flow.locals.len()
        )));
    }
    Ok(())
}

fn verify_target(target: BlockId, flow: &ControlFlow) -> Result<(), CompileError> {
    if flow.blocks.get(target.0).is_none() {
        return Err(CompileError::new(format!(
            "internal compiler error: IR targets unknown block {}",
            target.0
        )));
    }
    Ok(())
}

fn reachable_blocks(flow: &ControlFlow) -> Result<Vec<bool>, CompileError> {
    let mut reachable = vec![false; flow.blocks.len()];
    let mut pending = VecDeque::from([flow.entry]);
    while let Some(block) = pending.pop_front() {
        verify_target(block, flow)?;
        if std::mem::replace(&mut reachable[block.0], true) {
            continue;
        }
        match flow.blocks[block.0].terminator {
            Terminator::Jump { target } => pending.push_back(target),
            Terminator::Branch {
                then_target,
                else_target,
                ..
            } => {
                pending.push_back(then_target);
                pending.push_back(else_target);
            }
            Terminator::Exit { .. } | Terminator::Return { .. } | Terminator::Unreachable => {}
        }
    }
    Ok(reachable)
}

fn flow_operand_type(
    operand: Operand,
    values: &[Value],
    definitions: &[Option<Definition>],
    block: usize,
    position: usize,
) -> Result<Type, CompileError> {
    match operand {
        Operand::Integer { value, ty } => verify_integer(value, ty),
        Operand::Value(ValueId(id)) => {
            let Some(Some(definition)) = definitions.get(id) else {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR references undefined value {id}"
                )));
            };
            if definition.block != block || definition.position >= position {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR value {id} is not defined earlier in block {block}"
                )));
            }
            Ok(values[id].ty)
        }
    }
}

/// The IR function ID of a syntax function. Lowering visits `syntax.functions`
/// in arena order, so a function's position is its raw arena index.
fn function_id(id: Idx<SyntaxFunction>) -> FunctionId {
    FunctionId(id.into_raw().into_u32() as usize)
}

pub(crate) fn lower(checked: CheckedProgram<'_>) -> Program {
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
        let value = checked.expressions[*initializer]
            .constant
            .as_ref()
            .expect("module-level initializers are constant expressions")
            .to_i128()
            .expect("concrete Fern value fits in the IR representation");
        module_places.insert(binding, Place::Global(GlobalId(globals.len())));
        globals.push(Global {
            ty: checked.bindings[binding].ty,
            value,
        });
    }

    let functions = checked
        .syntax
        .functions
        .iter()
        .map(|(id, _)| lower_function(&checked, id, &module_places))
        .collect();
    Program {
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
    let mut bindings = module_places.clone();
    for &parameter in &signature.parameters {
        let local = builder.local(checked.bindings[parameter].ty);
        bindings.insert(parameter, Place::Local(local));
    }
    let parameters = signature.parameters.len();
    let result = signature.result;
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

fn integer(value: i128, ty: Type) -> Operand {
    Operand::Integer { value, ty }
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
    bindings: &mut HashMap<Idx<Binding>, Place>,
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
    let operand = lower_flow_operand(checked, *initializer, bindings, builder);
    let local = builder.local(checked.bindings[binding].ty);
    builder.store(Place::Local(local), operand);
    bindings.insert(binding, Place::Local(local));
}

fn lower_flow_body(
    checked: &CheckedProgram<'_>,
    body: &[Idx<Statement>],
    bindings: &mut HashMap<Idx<Binding>, Place>,
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
    bindings: &mut HashMap<Idx<Binding>, Place>,
    loops: &[LoopTarget],
    builder: &mut FlowBuilder,
) -> bool {
    match &checked.syntax.statements[statement].kind {
        StatementKind::Binding { .. } => {
            lower_flow_binding(checked, statement, bindings, builder);
            false
        }
        StatementKind::Assignment { value, .. } => {
            let operand = lower_flow_operand(checked, *value, bindings, builder);
            builder.store(bindings[&checked.assignments[statement]], operand);
            false
        }
        StatementKind::CompoundAssignment {
            operator,
            operator_span,
            value,
            ..
        } => {
            let binding = checked.assignments[statement];
            let place = bindings[&binding];
            let ty = checked.bindings[binding].ty;
            let left_operand = load_place(builder, place, ty);
            let (left, right) =
                lower_second_operand(checked, left_operand, ty, *value, bindings, builder);
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
                call.target_span.clone(),
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
        StatementKind::For {
            label,
            header,
            body,
        } => {
            lower_for(
                checked,
                label.as_ref().map(|label| label.name),
                header,
                body,
                bindings,
                loops,
                builder,
            );
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

/// Lowers arguments left to right, holding each so a later argument that splits
/// blocks cannot leave an earlier one unreadable in the call's block.
fn lower_call(
    checked: &CheckedProgram<'_>,
    function: Idx<SyntaxFunction>,
    arguments: &[Idx<Expression>],
    span: std::ops::Range<usize>,
    bindings: &HashMap<Idx<Binding>, Place>,
    builder: &mut FlowBuilder,
) -> Option<Operand> {
    let mut held = Vec::with_capacity(arguments.len());
    for &argument in arguments {
        let ty = checked.expressions[argument].ty;
        let operand = lower_flow_operand(checked, argument, bindings, builder);
        held.push(hold_operand(builder, operand, ty));
    }
    let arguments = held
        .into_iter()
        .map(|operand| operand.read(builder))
        .collect();
    builder.call(
        function_id(function),
        arguments,
        checked.functions[function].result,
        span,
    )
}

fn lower_if(
    checked: &CheckedProgram<'_>,
    condition: Idx<Expression>,
    then_body: &[Idx<Statement>],
    else_branch: Option<Idx<Statement>>,
    bindings: &mut HashMap<Idx<Binding>, Place>,
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

fn lower_for(
    checked: &CheckedProgram<'_>,
    label: Option<Spur>,
    header: &ForHeader,
    body: &[Idx<Statement>],
    bindings: &mut HashMap<Idx<Binding>, Place>,
    loops: &[LoopTarget],
    builder: &mut FlowBuilder,
) {
    let (condition, post) = match header {
        ForHeader::Infinite => (None, None),
        ForHeader::Condition(condition) => (Some(*condition), None),
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
            (Some(*condition), Some(*post))
        }
    };

    let condition_block = builder.block();
    let body_block = builder.block();
    let post_block = post.map(|_| builder.block());
    let after_block = builder.block();
    builder.terminate(Terminator::Jump {
        target: condition_block,
    });

    builder.select(condition_block);
    if let Some(condition) = condition {
        let operand = lower_flow_operand(checked, condition, bindings, builder);
        builder.terminate(Terminator::Branch {
            condition: operand,
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
    if !lower_flow_body(checked, body, bindings, &nested_loops, builder) {
        builder.terminate(Terminator::Jump {
            target: post_block.unwrap_or(condition_block),
        });
    }

    if let (Some(post), Some(post_block)) = (post, post_block) {
        builder.select(post_block);
        assert!(!lower_flow_statement(
            checked, post, bindings, loops, builder
        ));
        builder.terminate(Terminator::Jump {
            target: condition_block,
        });
    }
    builder.select(after_block);
}

fn loop_target(loops: &[LoopTarget], label: Option<Spur>) -> &LoopTarget {
    loops
        .iter()
        .rev()
        .find(|target| label.is_none() || target.label == label)
        .expect("semantic checking resolves loop targets")
}

fn lower_flow_operand(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &HashMap<Idx<Binding>, Place>,
    builder: &mut FlowBuilder,
) -> Operand {
    let expression = &checked.expressions[id];
    if let Some(constant) = expression.constant.as_ref() {
        return integer(
            constant
                .to_i128()
                .expect("concrete Fern value fits in the IR representation"),
            expression.ty,
        );
    }
    match &expression.value {
        ExpressionValue::Integer | ExpressionValue::Boolean => {
            unreachable!("literal expressions are constant")
        }
        ExpressionValue::Reference(binding) => {
            load_place(builder, bindings[binding], expression.ty)
        }
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
                ty: expression.ty,
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
                ty: expression.ty,
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
                checked.expressions[*left].ty,
                *right,
                bindings,
                builder,
            );
            builder.value(Value {
                span: Some(operator_span.clone()),
                ty: expression.ty,
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
                checked.expressions[*left].ty,
                *right,
                bindings,
                builder,
            );
            builder.value(Value {
                span: Some(operator_span.clone()),
                ty: Type::Bool,
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
                ty: Type::Bool,
                kind: ValueKind::LogicalNot { operand },
            })
        }
        ExpressionValue::Logical {
            operator,
            left,
            right,
            ..
        } => lower_logical(checked, *operator, *left, *right, bindings, builder),
        ExpressionValue::Call { function } => {
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
    }
}

/// An already-lowered operand and the block that produced it. Lowering the
/// operands that follow it can split blocks, and a value is readable only in
/// the block defining it, so reading it later goes through a local.
struct HeldOperand {
    operand: Operand,
    block: BlockId,
    ty: Type,
}

fn hold_operand(builder: &FlowBuilder, operand: Operand, ty: Type) -> HeldOperand {
    HeldOperand {
        operand,
        block: builder.current,
        ty,
    }
}

impl HeldOperand {
    /// Reads the operand in the builder's current block, routing a value
    /// through a local when lowering has moved on to another block.
    fn read(self, builder: &mut FlowBuilder) -> Operand {
        match self.operand {
            Operand::Value(_) if self.block != builder.current => {
                let local = builder.local(self.ty);
                builder.store_in(self.block, Place::Local(local), self.operand);
                load_place(builder, Place::Local(local), self.ty)
            }
            operand => operand,
        }
    }
}

/// Lowers `right` after an already-lowered `left`, holding `left` so it is
/// readable in whichever block `right` lowering ends in.
fn lower_second_operand(
    checked: &CheckedProgram<'_>,
    left: Operand,
    left_type: Type,
    right: Idx<Expression>,
    bindings: &HashMap<Idx<Binding>, Place>,
    builder: &mut FlowBuilder,
) -> (Operand, Operand) {
    let held_left = hold_operand(builder, left, left_type);
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
    bindings: &HashMap<Idx<Binding>, Place>,
    builder: &mut FlowBuilder,
) -> Operand {
    let left = lower_flow_operand(checked, left, bindings, builder);
    let right_block = builder.block();
    let short_block = builder.block();
    let join_block = builder.block();
    let result = builder.local(Type::Bool);
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
    builder.store(Place::Local(result), integer(short_value, Type::Bool));
    builder.terminate(Terminator::Jump { target: join_block });

    builder.select(right_block);
    let right = lower_flow_operand(checked, right, bindings, builder);
    builder.store(Place::Local(result), right);
    builder.terminate(Terminator::Jump { target: join_block });

    builder.select(join_block);
    load_place(builder, Place::Local(result), Type::Bool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{frontend, semantic};

    fn integer(value: i128, ty: Type) -> Operand {
        Operand::Integer { value, ty }
    }

    fn convert(operand: Operand, ty: Type) -> Value {
        Value {
            span: None,
            ty,
            kind: ValueKind::Convert {
                operand,
                truncating: false,
            },
        }
    }

    /// A whole-program wrapper around a single `main` whose block runs `values`
    /// in order and then exits.
    fn program(values: Vec<Value>, exit: Operand) -> Program {
        let blocks = vec![Block {
            instructions: (0..values.len())
                .map(|id| Instruction::Value(ValueId(id)))
                .collect(),
            terminator: Terminator::Exit { status: exit },
        }];
        one_function(main_function(values, vec![], blocks))
    }

    fn main_function(values: Vec<Value>, locals: Vec<Type>, blocks: Vec<Block>) -> Function {
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

    fn one_function(main: Function) -> Program {
        Program {
            globals: vec![],
            functions: vec![main],
            main: FunctionId(0),
        }
    }

    fn lowered(source: &str) -> VerifiedProgram {
        let syntax = frontend::parse(source).unwrap();
        lower(semantic::check(&syntax).unwrap()).verify().unwrap()
    }

    fn main_of(program: &VerifiedProgram) -> &Function {
        &program.program().functions[program.program().main.0]
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
                ty: Type::Int,
                value: 40,
            }]
        );
        insta::assert_debug_snapshot!("module_bindings", program.program());
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
        assert_eq!(pick.result, Some(Type::I64));
        assert_eq!(pick.flow.locals[..2], [Type::U8, Type::I64]);
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
        for (argument, ty) in arguments.iter().zip([Type::Int, Type::Bool]) {
            let Operand::Value(id) = argument else {
                panic!("an argument read in the call's block is a value")
            };
            assert!(
                block.instructions.contains(&Instruction::Value(*id)),
                "the argument is defined in the call's block"
            );
            assert_eq!(main.values[id.0].ty, ty, "arguments keep source order");
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
        assert_eq!(main.values[id].ty, Type::Int);
    }

    #[test]
    fn verification_rejects_invalid_references() {
        for values in [
            vec![convert(Operand::Value(ValueId(usize::MAX)), Type::Int)],
            vec![
                convert(Operand::Value(ValueId(1)), Type::Int),
                convert(integer(42, Type::Int), Type::Int),
            ],
            vec![convert(Operand::Value(ValueId(0)), Type::Int)],
        ] {
            let error = program(values, integer(0, Type::Int)).verify().unwrap_err();
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
                vec![convert(integer(42, Type::Int), Type::Int)],
                Operand::Value(ValueId(1)),
            )
            .verify()
            .is_err()
        );
    }

    #[test]
    fn verification_accepts_constants_conversions_and_exits() {
        for exit in [
            integer(-1, Type::Int),
            Operand::Value(ValueId(0)),
            Operand::Value(ValueId(1)),
        ] {
            program(
                vec![
                    convert(integer(i128::from(i32::MIN), Type::Int), Type::Int),
                    convert(Operand::Value(ValueId(0)), Type::Int),
                ],
                exit,
            )
            .verify()
            .unwrap();
        }
    }

    fn ranges() -> [(Type, i128, i128); 10] {
        Type::ALL_INTEGERS.map(|ty| (ty, ty.min(), i128::from(ty.max())))
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
                    integer(0, Type::Int),
                )
                .verify()
                .unwrap();
            }
            for value in [i128::MIN, min - 1, max + 1, i128::MAX] {
                let error = program(vec![convert(integer(value, ty), ty)], integer(0, Type::Int))
                    .verify()
                    .unwrap_err();
                assert!(error.to_string().contains("out of range"));
            }
            assert!(
                program(
                    vec![convert(integer(0, ty), Type::Bool)],
                    integer(0, Type::Int),
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
                                        ty: destination,
                                        kind: ValueKind::Convert {
                                            operand,
                                            truncating: false,
                                        },
                                    },
                                ],
                                integer(0, Type::Int),
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
                    source == Type::Int
                );
            }
        }
        for operand in [
            integer(128, Type::I8),
            Operand::Value(ValueId(0)),
            Operand::Value(ValueId(usize::MAX)),
        ] {
            assert!(
                program(
                    vec![Value {
                        span: None,
                        ty: Type::Int,
                        kind: ValueKind::Convert {
                            operand,
                            truncating: false,
                        }
                    }],
                    integer(0, Type::Int),
                )
                .verify()
                .is_err()
            );
        }
        for value in [
            -(1i128 << (Type::Int.width() - 1)) - 1,
            i128::from(Type::Int.max()) + 1,
        ] {
            assert!(program(vec![], integer(value, Type::Int)).verify().is_err());
        }
    }

    #[test]
    fn verification_accepts_checked_conversion_operands() {
        program(
            vec![
                convert(integer(42, Type::U8), Type::U8),
                Value {
                    span: None,
                    ty: Type::U64,
                    kind: ValueKind::Convert {
                        operand: Operand::Value(ValueId(0)),
                        truncating: false,
                    },
                },
            ],
            integer(0, Type::Int),
        )
        .verify()
        .unwrap();
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
            assert_eq!(values[id].ty, Type::Int);
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
                left: integer(1, Type::U64),
                right: Operand::Value(ValueId(0)),
            }
        );
        assert_eq!(values[1].ty, Type::U64);
        assert_eq!(
            values[3].kind,
            ValueKind::Binary {
                operator: BinaryOperator::ShiftLeft,
                form: BinaryForm::Infix,
                left: Operand::Value(ValueId(1)),
                right: Operand::Value(ValueId(2)),
            }
        );
        assert_eq!(values[3].ty, Type::U64);
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
                left: integer(3, Type::U64),
                right: Operand::Value(ValueId(0)),
            }
        );
        assert_eq!(values[1].ty, Type::U64);
    }

    #[test]
    fn verification_checks_operation_shapes_and_accepts_independent_shift_counts() {
        let operation = |ty, kind| Value {
            span: Some(0..1),
            ty,
            kind,
        };
        program(
            vec![
                convert(integer(1, Type::U8), Type::U8),
                convert(integer(1, Type::U16), Type::U16),
                operation(
                    Type::U8,
                    ValueKind::Binary {
                        operator: BinaryOperator::ShiftLeft,
                        form: BinaryForm::Infix,
                        left: Operand::Value(ValueId(0)),
                        right: Operand::Value(ValueId(1)),
                    },
                ),
            ],
            integer(0, Type::Int),
        )
        .verify()
        .unwrap();

        for value in [
            operation(
                Type::U8,
                ValueKind::Unary {
                    operator: UnaryOperator::Negate,
                    operand: integer(1, Type::U8),
                },
            ),
            operation(
                Type::U16,
                ValueKind::Unary {
                    operator: UnaryOperator::Complement,
                    operand: integer(1, Type::U8),
                },
            ),
            operation(
                Type::U8,
                ValueKind::Binary {
                    operator: BinaryOperator::Add,
                    form: BinaryForm::Infix,
                    left: integer(1, Type::U8),
                    right: integer(1, Type::U16),
                },
            ),
            Value {
                span: None,
                ty: Type::U8,
                kind: ValueKind::Binary {
                    operator: BinaryOperator::Add,
                    form: BinaryForm::Infix,
                    left: integer(1, Type::U8),
                    right: integer(1, Type::U8),
                },
            },
        ] {
            assert!(
                program(vec![value], integer(0, Type::Int))
                    .verify()
                    .unwrap_err()
                    .to_string()
                    .contains("invalid IR value")
            );
        }
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
                    condition: integer(1, Type::Int),
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
            ty: Type::Int,
            kind: ValueKind::Load(Place::Local(LocalId(0))),
        };
        let same_block = main_function(
            vec![load()],
            vec![Type::Int],
            vec![Block {
                instructions: vec![
                    Instruction::Value(ValueId(0)),
                    Instruction::Store {
                        place: Place::Local(LocalId(0)),
                        operand: integer(1, Type::Int),
                    },
                ],
                terminator: Terminator::Exit {
                    status: integer(0, Type::Int),
                },
            }],
        );
        let missing_path = main_function(
            vec![load()],
            vec![Type::Int],
            vec![
                Block {
                    instructions: vec![],
                    terminator: Terminator::Branch {
                        condition: integer(1, Type::Bool),
                        then_target: BlockId(1),
                        else_target: BlockId(2),
                    },
                },
                Block {
                    instructions: vec![Instruction::Store {
                        place: Place::Local(LocalId(0)),
                        operand: integer(1, Type::Int),
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
                        status: integer(0, Type::Int),
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
            result: Some(Type::Int),
            values: vec![Value {
                span: None,
                ty: Type::Int,
                kind: ValueKind::Load(Place::Local(LocalId(0))),
            }],
            flow: ControlFlow {
                entry: BlockId(0),
                locals: vec![Type::Int],
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
    fn two_functions(
        callee: Function,
        values: Vec<Value>,
        instructions: Vec<Instruction>,
    ) -> Program {
        Program {
            globals: vec![],
            functions: vec![
                main_function(
                    values,
                    vec![],
                    vec![Block {
                        instructions,
                        terminator: Terminator::Exit {
                            status: integer(0, Type::Int),
                        },
                    }],
                ),
                callee,
            ],
            main: FunctionId(0),
        }
    }

    fn callee(parameters: Vec<Type>, result: Option<Type>) -> Function {
        Function {
            parameters: parameters.len(),
            result,
            values: vec![],
            flow: ControlFlow {
                entry: BlockId(0),
                locals: parameters,
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

    fn call_result(ty: Type) -> Value {
        Value {
            span: None,
            ty,
            kind: ValueKind::CallResult,
        }
    }

    #[test]
    fn verification_checks_call_targets_arity_and_argument_types() {
        // A well-formed call to a two-parameter function.
        two_functions(
            callee(vec![Type::U8, Type::Int], None),
            vec![],
            vec![call(
                None,
                1,
                vec![integer(1, Type::U8), integer(2, Type::Int)],
            )],
        )
        .verify()
        .unwrap();

        for (instruction, expected) in [
            (call(None, 7, vec![]), "unknown function 7"),
            (
                call(None, 1, vec![integer(1, Type::U8)]),
                "passes 1 argument",
            ),
            (
                call(
                    None,
                    1,
                    vec![
                        integer(1, Type::U8),
                        integer(2, Type::Int),
                        integer(3, Type::Int),
                    ],
                ),
                "passes 3 arguments",
            ),
            (
                call(None, 1, vec![integer(1, Type::Int), integer(2, Type::Int)]),
                "argument 0 has type",
            ),
        ] {
            let error = two_functions(
                callee(vec![Type::U8, Type::Int], None),
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
        let mut short_of_locals = callee(vec![Type::Int], None);
        short_of_locals.parameters = 2;
        let error = two_functions(
            short_of_locals,
            vec![],
            vec![call(
                None,
                1,
                vec![integer(1, Type::Int), integer(2, Type::Int)],
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
            callee(vec![], Some(Type::Int)),
            vec![call_result(Type::Int)],
            vec![call(Some(ValueId(0)), 1, vec![])],
        )
        .verify()
        .unwrap();

        let error = two_functions(
            callee(vec![], None),
            vec![call_result(Type::Int)],
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
            callee(vec![], Some(Type::Int)),
            vec![],
            vec![call(None, 1, vec![])],
        )
        .verify()
        .unwrap_err();
        assert!(error.to_string().contains("discards the result"));

        let error = two_functions(
            callee(vec![], Some(Type::Int)),
            vec![call_result(Type::U8)],
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
            vec![call_result(Type::Int)],
            vec![],
            vec![Block {
                instructions: vec![Instruction::Value(ValueId(0))],
                terminator: Terminator::Exit {
                    status: integer(0, Type::Int),
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
            callee(vec![], Some(Type::Int)),
            vec![convert(integer(1, Type::Int), Type::Int)],
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
        returning(Some(Type::Int), Some(integer(0, Type::Int))).unwrap();

        for (result, value, expected) in [
            (
                None,
                Some(integer(0, Type::Int)),
                "returns a value from a void function",
            ),
            (Some(Type::Int), None, "return is missing its value"),
            (
                Some(Type::Int),
                Some(integer(0, Type::U8)),
                "internal compiler error: IR returns",
            ),
        ] {
            let error = returning(result, value).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn verification_rejects_a_value_function_that_falls_off_its_end() {
        let error = two_functions(
            Function {
                parameters: 0,
                result: Some(Type::Int),
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
                locals: vec![Type::Int],
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
                        operand: integer(1, Type::Int),
                    }],
                    terminator: Terminator::Exit {
                        status: integer(0, Type::Int),
                    },
                }],
            )],
            main: FunctionId(0),
        };
        stores_to_global(vec![Global {
            ty: Type::Int,
            value: 0,
        }])
        .verify()
        .unwrap();

        let error = stores_to_global(vec![]).verify().unwrap_err();
        assert!(error.to_string().contains("unknown global 0"));

        let error = stores_to_global(vec![Global {
            ty: Type::U8,
            value: 0,
        }])
        .verify()
        .unwrap_err();
        assert!(error.to_string().contains("IR store has type"));

        let error = Program {
            globals: vec![Global {
                ty: Type::U8,
                value: 256,
            }],
            functions: vec![main_function(
                vec![],
                vec![],
                vec![Block {
                    instructions: vec![],
                    terminator: Terminator::Exit {
                        status: integer(0, Type::Int),
                    },
                }],
            )],
            main: FunctionId(0),
        }
        .verify()
        .unwrap_err();
        assert!(error.to_string().contains("out of range"), "{error}");
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
}
