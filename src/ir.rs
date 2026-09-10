use crate::{
    CompileError,
    frontend::{
        AssignmentTarget, BinaryOperator, ComparisonOperator, Expression, ExpressionKind,
        ForHeader, Function as SyntaxFunction, LogicalOperator, Statement, StatementKind,
        UnaryOperator,
    },
    semantic::{Binding, CheckedProgram, Constant, ExpressionValue},
    types::{Scalar, Type},
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Place {
    Local(LocalId),
    Global(GlobalId),
    /// `base[index]`. The span is the one the bounds-check trap reports, so
    /// every access through this place checks its index.
    Element {
        base: Box<Place>,
        index: Operand,
        span: std::ops::Range<usize>,
    },
}

impl Place {
    /// Storing through an element place writes inside its root, which is how
    /// an array local is initialized.
    fn root_local(&self) -> Option<LocalId> {
        match self {
            Self::Local(local) => Some(*local),
            Self::Global(_) => None,
            Self::Element { base, .. } => base.root_local(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operand {
    // i128 holds both the signed minima and the full u64 range without bit reinterpretation.
    Integer { value: i128, ty: Scalar },
    Value(ValueId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

impl Terminator {
    fn targets(&self) -> Vec<BlockId> {
        match self {
            Self::Jump { target } => vec![*target],
            Self::Branch {
                then_target,
                else_target,
                ..
            } => vec![*then_target, *else_target],
            Self::Exit { .. } | Self::Return { .. } | Self::Unreachable => Vec::new(),
        }
    }
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

/// A module-level `var` of any module in the program. Its initializer is a
/// constant expression, so it needs an initial value rather than
/// initialization code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Global {
    pub ty: Type,
    /// Its scalars in memory order: one for a scalar global, one per element
    /// for an array global.
    pub values: Vec<i128>,
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

fn verify_integer(value: i128, ty: Scalar) -> Result<Scalar, CompileError> {
    if value < ty.min() || value > i128::from(ty.max()) {
        return Err(CompileError::new(format!(
            "internal compiler error: IR integer {value} out of range for {ty:?}"
        )));
    }
    Ok(ty)
}

/// The IR defines arithmetic, conversions, and branching on scalars only.
fn scalar(ty: &Type) -> Result<Scalar, CompileError> {
    ty.scalar().ok_or_else(|| {
        CompileError::new(format!(
            "internal compiler error: IR uses `{ty}` where a scalar is required"
        ))
    })
}

fn valid_value(
    value: &Value,
    operand_type: &impl Fn(Operand) -> Result<Type, CompileError>,
    place_type: &impl Fn(&Place) -> Result<Type, CompileError>,
) -> Result<bool, CompileError> {
    Ok(match &value.kind {
        ValueKind::Load(place) => place_type(place)? == value.ty && value.span.is_none(),
        // The defining call checks the result type against the callee's
        // signature, and owns the span the call reports.
        ValueKind::CallResult => value.span.is_none(),
        ValueKind::Comparison {
            operator,
            left,
            right,
        } => {
            let left = operand_type(*left)?;
            value.span.is_some()
                && value.ty == Scalar::Bool.into()
                && left == operand_type(*right)?
                && (operator.is_equality() || left.scalar().is_some())
        }
        ValueKind::LogicalNot { operand } => {
            value.span.is_some()
                && value.ty == Scalar::Bool.into()
                && operand_type(*operand)? == Scalar::Bool.into()
        }
        ValueKind::Convert { .. } | ValueKind::Unary { .. } | ValueKind::Binary { .. } => {
            valid_integer_operation(value, operand_type)?
        }
    })
}

/// The value kinds defined only on integers, whose result and operands are all
/// non-`bool` scalars.
fn valid_integer_operation(
    value: &Value,
    operand_type: &impl Fn(Operand) -> Result<Type, CompileError>,
) -> Result<bool, CompileError> {
    let operand = |operand| scalar(&operand_type(operand)?);
    let ty = scalar(&value.ty)?;
    Ok(match &value.kind {
        ValueKind::Convert {
            operand: source,
            truncating,
        } => {
            let source = operand(*source)?;
            source.is_integer()
                && ty.is_integer()
                && (*truncating || source.all_values_fit(ty) || value.span.is_some())
        }
        ValueKind::Unary {
            operator,
            operand: source,
        } => {
            let source = operand(*source)?;
            value.span.is_some()
                && source == ty
                && source.is_integer()
                && (*operator != UnaryOperator::Negate || source.signed())
        }
        ValueKind::Binary {
            operator,
            left,
            right,
            ..
        } => {
            let (left, right) = (operand(*left)?, operand(*right)?);
            value.span.is_some()
                && left == ty
                && left.is_integer()
                && right.is_integer()
                && (operator.is_shift() || right == left)
        }
        _ => unreachable!("an integer operation is a conversion, a unary, or a binary value"),
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
            verify_global(global)
                .map_err(|error| CompileError::new(format!("{error} (IR global {id})")))?;
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

    fn place_type(
        &self,
        flow: &ControlFlow,
        place: &Place,
        operand_type: &impl Fn(Operand) -> Result<Type, CompileError>,
    ) -> Result<Type, CompileError> {
        match place {
            Place::Local(LocalId(id)) => flow.locals.get(*id).cloned().ok_or_else(|| {
                CompileError::new(format!(
                    "internal compiler error: IR uses unknown local {id}"
                ))
            }),
            Place::Global(GlobalId(id)) => self
                .globals
                .get(*id)
                .map(|global| global.ty.clone())
                .ok_or_else(|| {
                    CompileError::new(format!(
                        "internal compiler error: IR uses unknown global {id}"
                    ))
                }),
            Place::Element { base, index, .. } => {
                let base = self.place_type(flow, base, operand_type)?;
                let Type::Array { element, .. } = base else {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR indexes `{base}`"
                    )));
                };
                let index = operand_type(*index)?;
                if index != Scalar::Int.into() {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR index has type `{index}`, expected `int`"
                    )));
                }
                Ok(*element)
            }
        }
    }

    fn verify_function(&self, function: &Function) -> Result<(), CompileError> {
        let flow = &function.flow;
        verify_target(flow.entry, flow)?;
        let definitions = value_definitions(function)?;
        let reachable = reachable_blocks(flow)?;
        let initialized_at_entry = initialized_locals(function, &reachable);
        for (index, block) in flow.blocks.iter().enumerate() {
            if reachable[index] && block.terminator == Terminator::Unreachable {
                return Err(CompileError::new(format!(
                    "internal compiler error: reachable IR block {index} is not terminated"
                )));
            }
            self.verify_block(
                function,
                index,
                &definitions,
                initialized_at_entry[index].clone(),
            )?;
        }
        Ok(())
    }

    /// Checks one block's instructions and its terminator. `initialized` holds
    /// the locals initialized on entry to the block and grows as stores in the
    /// block initialize more.
    fn verify_block(
        &self,
        function: &Function,
        block_index: usize,
        definitions: &[Definition],
        mut initialized: LocalSet,
    ) -> Result<(), CompileError> {
        let block = &function.flow.blocks[block_index];
        for (position, instruction) in block.instructions.iter().enumerate() {
            self.verify_instruction(
                function,
                definitions,
                (block_index, position),
                instruction,
                &mut initialized,
            )?;
        }
        self.verify_terminator(function, block_index, definitions)
    }

    fn verify_instruction(
        &self,
        function: &Function,
        definitions: &[Definition],
        (block_index, position): (usize, usize),
        instruction: &Instruction,
        initialized: &mut LocalSet,
    ) -> Result<(), CompileError> {
        let flow = &function.flow;
        let operand_type = |operand| {
            flow_operand_type(
                operand,
                &function.values,
                definitions,
                block_index,
                position,
            )
        };
        match instruction {
            Instruction::Value(ValueId(id)) => {
                let value = &function.values[*id];
                if let ValueKind::Load(place) = &value.kind
                    && let Some(LocalId(local)) = place.root_local()
                    && local < flow.locals.len()
                    && !initialized.contains(local)
                {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR loads uninitialized local {local}"
                    )));
                }
                let place_type = |place: &Place| self.place_type(flow, place, &operand_type);
                if !valid_value(value, &operand_type, &place_type)? {
                    return Err(invalid_value(*id, value));
                }
                Ok(())
            }
            Instruction::Store { place, operand } => {
                let destination = self.place_type(flow, place, &operand_type)?;
                let source = operand_type(*operand)?;
                if source != destination {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR store has type `{source}`, expected `{destination}`"
                    )));
                }
                if let Some(local) = place.root_local() {
                    initialized.insert(local.0);
                }
                Ok(())
            }
            Instruction::Call {
                result,
                function: callee,
                arguments,
                ..
            } => self.verify_call(function, *callee, arguments, *result, &operand_type),
        }
    }

    fn verify_terminator(
        &self,
        function: &Function,
        block_index: usize,
        definitions: &[Definition],
    ) -> Result<(), CompileError> {
        let flow = &function.flow;
        let block = &flow.blocks[block_index];
        // A terminator reads values defined anywhere in its own block.
        let end = block.instructions.len();
        let operand_type =
            |operand| flow_operand_type(operand, &function.values, definitions, block_index, end);
        for target in block.terminator.targets() {
            verify_target(target, flow)?;
        }
        match &block.terminator {
            Terminator::Branch { condition, .. } => {
                if operand_type(*condition)? != Scalar::Bool.into() {
                    return Err(CompileError::new(
                        "internal compiler error: IR branch requires bool",
                    ));
                }
                Ok(())
            }
            Terminator::Exit { status, .. } => {
                if operand_type(*status)? != Scalar::Int.into() {
                    return Err(CompileError::new(
                        "internal compiler error: IR exit requires int",
                    ));
                }
                Ok(())
            }
            Terminator::Return { value } => {
                verify_return(function.result.as_ref(), *value, &operand_type)
            }
            Terminator::Jump { .. } | Terminator::Unreachable => Ok(()),
        }
    }

    fn verify_call(
        &self,
        caller: &Function,
        callee: FunctionId,
        arguments: &[Operand],
        result: Option<ValueId>,
        operand_type: &impl Fn(Operand) -> Result<Type, CompileError>,
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
            let parameter = &target.flow.locals[index];
            if source != *parameter {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR call argument {index} has type `{source}`, expected `{parameter}`"
                )));
            }
        }
        match (target.result.as_ref(), result) {
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
                let defined = &caller.values[id].ty;
                if *defined != *expected {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR call result has type `{defined}`, expected `{expected}`"
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

    fn union(&mut self, other: &Self) {
        for (word, other) in self.0.iter_mut().zip(&other.0) {
            *word |= other;
        }
    }

    fn intersect(&mut self, other: &Self) {
        for (word, other) in self.0.iter_mut().zip(&other.0) {
            *word &= other;
        }
    }
}

/// The locals initialized on entry to each block. A local counts as
/// initialized only when every reachable path to the block initializes it, so
/// the analysis starts optimistic and shrinks to a fixed point.
fn initialized_locals(function: &Function, reachable: &[bool]) -> Vec<LocalSet> {
    let flow = &function.flow;
    let predecessors = block_predecessors(flow);
    let stores = local_stores(flow, flow.locals.len());
    let locals = flow.locals.len();
    let mut initialized = vec![LocalSet::full(locals); flow.blocks.len()];
    // Arguments initialize the parameters before the entry block runs.
    let mut at_entry = LocalSet::empty(locals);
    for parameter in 0..function.parameters {
        at_entry.insert(parameter);
    }
    initialized[flow.entry.0] = at_entry;
    for (block, is_reachable) in reachable.iter().enumerate() {
        if !is_reachable {
            initialized[block] = LocalSet::empty(locals);
        }
    }
    loop {
        let mut changed = false;
        for block in 0..flow.blocks.len() {
            // Both of these blocks keep the set they started with.
            if !reachable[block] || block == flow.entry.0 {
                continue;
            }
            let mut incoming = LocalSet::full(locals);
            for &predecessor in predecessors[block]
                .iter()
                .filter(|predecessor| reachable[**predecessor])
            {
                let mut leaving = initialized[predecessor].clone();
                leaving.union(&stores[predecessor]);
                incoming.intersect(&leaving);
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

fn block_predecessors(flow: &ControlFlow) -> Vec<Vec<usize>> {
    let mut predecessors = vec![Vec::new(); flow.blocks.len()];
    for (block_index, block) in flow.blocks.iter().enumerate() {
        for target in block.terminator.targets() {
            if let Some(target_predecessors) = predecessors.get_mut(target.0) {
                target_predecessors.push(block_index);
            }
        }
    }
    predecessors
}

fn local_stores(flow: &ControlFlow, locals: usize) -> Vec<LocalSet> {
    flow.blocks
        .iter()
        .map(|block| {
            let mut stored = LocalSet::empty(locals);
            for instruction in &block.instructions {
                if let Instruction::Store { place, .. } = instruction
                    && let Some(local) = place.root_local()
                {
                    stored.insert(local.0);
                }
            }
            stored
        })
        .collect()
}

fn verify_global(global: &Global) -> Result<(), CompileError> {
    let expected = global.ty.element_count();
    let found = global.values.len() as u64;
    if found != expected {
        return Err(CompileError::new(format!(
            "internal compiler error: IR global of type `{}` holds {found} values, expected {expected}",
            global.ty
        )));
    }
    for &value in &global.values {
        verify_integer(value, global.ty.leaf())?;
    }
    Ok(())
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
        pending.extend(flow.blocks[block.0].terminator.targets());
    }
    Ok(reachable)
}

/// Where each value of `function` is defined. Every value is defined exactly
/// once, and only a call defines a call result.
fn value_definitions(function: &Function) -> Result<Vec<Definition>, CompileError> {
    let mut found: Vec<Option<Definition>> = vec![None; function.values.len()];
    for (block, block_data) in function.flow.blocks.iter().enumerate() {
        for (position, instruction) in block_data.instructions.iter().enumerate() {
            record_definition(&mut found, instruction, block, position)?;
        }
    }
    let definitions = found
        .into_iter()
        .enumerate()
        .map(|(id, definition)| {
            definition.ok_or_else(|| {
                CompileError::new(format!(
                    "internal compiler error: IR does not define value {id}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (id, (value, definition)) in function.values.iter().zip(&definitions).enumerate() {
        if (value.kind == ValueKind::CallResult) != definition.from_call {
            return Err(CompileError::new(format!(
                "internal compiler error: IR value {id} and its definition disagree on being a call result"
            )));
        }
    }
    Ok(definitions)
}

/// Records the value one instruction defines, rejecting a second definition of
/// it.
fn record_definition(
    found: &mut [Option<Definition>],
    instruction: &Instruction,
    block: usize,
    position: usize,
) -> Result<(), CompileError> {
    let defined = match instruction {
        Instruction::Value(id) => Some((*id, false)),
        Instruction::Call { result, .. } => result.map(|id| (id, true)),
        Instruction::Store { .. } => None,
    };
    let Some((ValueId(id), from_call)) = defined else {
        return Ok(());
    };
    let Some(definition) = found.get_mut(id) else {
        return Err(CompileError::new(format!(
            "internal compiler error: IR defines unknown value {id}"
        )));
    };
    if definition
        .replace(Definition {
            block,
            position,
            from_call,
        })
        .is_some()
    {
        return Err(CompileError::new(format!(
            "internal compiler error: IR defines value {id} more than once"
        )));
    }
    Ok(())
}

fn verify_return(
    result: Option<&Type>,
    value: Option<Operand>,
    operand_type: &impl Fn(Operand) -> Result<Type, CompileError>,
) -> Result<(), CompileError> {
    match (result, value) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(CompileError::new(
            "internal compiler error: IR returns a value from a void function",
        )),
        (Some(_), None) => Err(CompileError::new(
            "internal compiler error: IR return is missing its value",
        )),
        (Some(result), Some(value)) => {
            let source = operand_type(value)?;
            if source != *result {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR returns `{source}`, expected `{result}`"
                )));
            }
            Ok(())
        }
    }
}

fn flow_operand_type(
    operand: Operand,
    values: &[Value],
    definitions: &[Definition],
    block: usize,
    position: usize,
) -> Result<Type, CompileError> {
    match operand {
        Operand::Integer { value, ty } => verify_integer(value, ty).map(Type::from),
        Operand::Value(ValueId(id)) => {
            let Some(definition) = definitions.get(id) else {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR references undefined value {id}"
                )));
            };
            if definition.block != block || definition.position >= position {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR value {id} is not defined earlier in block {block}"
                )));
            }
            Ok(values[id].ty.clone())
        }
    }
}

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
        let mut values = Vec::new();
        flatten(
            checked.expressions[*initializer]
                .constant
                .as_ref()
                .expect("module-level initializers are constant expressions"),
            &mut values,
        );
        module_places.insert(binding, Place::Global(GlobalId(globals.len())));
        globals.push(Global {
            ty: checked.bindings[binding].ty.clone(),
            values,
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

fn integer(value: i128, ty: Scalar) -> Operand {
    Operand::Integer { value, ty }
}

/// The scalars a constant holds, in memory order.
fn flatten(constant: &Constant, values: &mut Vec<i128>) {
    match constant {
        Constant::Integer(value) => values.push(
            value
                .to_i128()
                .expect("concrete Fern value fits in the IR representation"),
        ),
        Constant::Array(elements) => {
            for element in elements {
                flatten(element, values);
            }
        }
    }
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
    let local = builder.local(checked.bindings[binding].ty.clone());
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
        StatementKind::Assignment { target, value } => {
            let mut place = lower_target(checked, statement, target, bindings, builder);
            let operand = lower_flow_operand(checked, *value, bindings, builder);
            let place = place.read(builder);
            builder.store(place, operand);
            false
        }
        StatementKind::CompoundAssignment {
            target,
            operator,
            operator_span,
            value,
        } => {
            let mut held = lower_target(checked, statement, target, bindings, builder);
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

/// The place an assignment target names, with its indices lowered left to
/// right before the value.
fn lower_target(
    checked: &CheckedProgram<'_>,
    statement: Idx<Statement>,
    target: &AssignmentTarget,
    bindings: &HashMap<Idx<Binding>, Place>,
    builder: &mut FlowBuilder,
) -> HeldPlace {
    let root = bindings[&checked.assignments[statement].binding].clone();
    let mut indices = Vec::with_capacity(target.indices.len());
    for &index in &target.indices {
        let span = checked.syntax.expressions[index].span.clone();
        let operand = lower_flow_operand(checked, index, bindings, builder);
        indices.push((hold_operand(builder, operand, Scalar::Int.into()), span));
    }
    HeldPlace { root, indices }
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
    bindings: &HashMap<Idx<Binding>, Place>,
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

/// A `for … in` loop's storage: the array it walks, captured once before the
/// first iteration, and the counter driving it.
struct Iteration {
    array: Place,
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
            span: Some(self.span.clone()),
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
            base: Box::new(self.array.clone()),
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
    bindings: &mut HashMap<Idx<Binding>, Place>,
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

/// Copies the array a `for … in` walks into a local, so assigning to the
/// original inside the body cannot change the remaining iterations.
fn capture(
    checked: &CheckedProgram<'_>,
    statement: Idx<Statement>,
    operand: Idx<Expression>,
    bindings: &mut HashMap<Idx<Binding>, Place>,
    builder: &mut FlowBuilder,
) -> Iteration {
    let ty = checked.expressions[operand].ty.clone();
    let Type::Array { length, element } = &ty else {
        unreachable!("checking requires an array operand for `for … in`")
    };
    let span = checked.syntax.expressions[operand].span.clone();
    let source = lower_array_place(checked, operand, bindings, builder);
    let loaded = load_place(builder, source, ty.clone());
    let array = Place::Local(builder.local(ty.clone()));
    builder.store(array.clone(), loaded);

    let counter = builder.local(Scalar::Int.into());
    builder.store(Place::Local(counter), integer(0, Scalar::Int));
    let element = (**element).clone();
    let value = builder.local(element.clone());

    let iteration = &checked.iterations[statement];
    bindings.insert(iteration.value, Place::Local(value));
    // The index binding is immutable, so reading the counter directly is the
    // same as reading a copy taken at the top of the body.
    if let Some(index) = iteration.index {
        bindings.insert(index, Place::Local(counter));
    }
    Iteration {
        array,
        length: *length,
        element,
        counter,
        value,
        span,
    }
}

fn lower_for(
    checked: &CheckedProgram<'_>,
    statement: Idx<Statement>,
    bindings: &mut HashMap<Idx<Binding>, Place>,
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
    bindings: &mut HashMap<Idx<Binding>, Place>,
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

/// Stores a folded array into `place`, one scalar per element.
fn store_constant(
    builder: &mut FlowBuilder,
    place: &Place,
    ty: &Type,
    constant: &Constant,
    span: &std::ops::Range<usize>,
) {
    match (ty, constant) {
        (Type::Scalar(scalar), Constant::Integer(value)) => {
            let value = value
                .to_i128()
                .expect("concrete Fern value fits in the IR representation");
            builder.store(place.clone(), integer(value, *scalar));
        }
        (Type::Array { element, .. }, Constant::Array(elements)) => {
            for (index, value) in elements.iter().enumerate() {
                let index = u64::try_from(index).expect("an array fits in the address space");
                store_constant(
                    builder,
                    &element_at(place, index, span),
                    element,
                    value,
                    span,
                );
            }
        }
        _ => unreachable!("a folded constant has the shape of its type"),
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
    bindings: &HashMap<Idx<Binding>, Place>,
    builder: &mut FlowBuilder,
) {
    let Type::Array { length, .. } = &checked.expressions[id].ty else {
        unreachable!("an array literal has an array type")
    };
    let span = &checked.syntax.expressions[id].span.clone();
    let mut last = None;
    for (index, &element) in elements.iter().enumerate() {
        let index = u64::try_from(index).expect("an array fits in the address space");
        let operand = lower_flow_operand(checked, element, bindings, builder);
        builder.store(element_at(place, index, span), operand);
        last = Some(operand);
    }
    if !fill {
        return;
    }
    let last = last.expect("a fill follows at least one element");
    let filled = u64::try_from(elements.len()).expect("an array fits in the address space");
    for index in filled..*length {
        builder.store(element_at(place, index, span), last);
    }
}

/// The place holding an array-typed expression. An expression with no storage
/// of its own materializes into a fresh local.
fn lower_array_place(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &HashMap<Idx<Binding>, Place>,
    builder: &mut FlowBuilder,
) -> Place {
    let expression = &checked.expressions[id];
    let ty = &expression.ty;
    if let Some(constant) = expression.constant.as_ref() {
        lower_folded_effects(checked, id, bindings, builder);
        let place = Place::Local(builder.local(ty.clone()));
        let span = &checked.syntax.expressions[id].span.clone();
        store_constant(builder, &place, ty, constant, span);
        return place;
    }
    match &expression.value {
        ExpressionValue::Reference(binding) => bindings[binding].clone(),
        ExpressionValue::Grouping { expression } => {
            lower_array_place(checked, *expression, bindings, builder)
        }
        ExpressionValue::Index { .. } => element_place(checked, id, bindings, builder),
        ExpressionValue::Array { elements, fill } => {
            let place = Place::Local(builder.local(ty.clone()));
            store_elements(checked, &place, id, elements, *fill, bindings, builder);
            place
        }
        ExpressionValue::Call { .. } => {
            let operand = lower_call_expression(checked, id, bindings, builder);
            let place = Place::Local(builder.local(ty.clone()));
            builder.store(place.clone(), operand);
            place
        }
        _ => unreachable!("an array value is a reference, a literal, an element, or a call"),
    }
}

fn element_place(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &HashMap<Idx<Binding>, Place>,
    builder: &mut FlowBuilder,
) -> Place {
    let ExpressionValue::Index { operand, index } = &checked.expressions[id].value else {
        unreachable!("an element place is lowered from an index expression")
    };
    let base = lower_array_place(checked, *operand, bindings, builder);
    let span = checked.syntax.expressions[*index].span.clone();
    let index = lower_flow_operand(checked, *index, bindings, builder);
    Place::Element {
        base: Box::new(base),
        index,
        span,
    }
}

/// The length `len(operand)` reads from its operand's type. The operand is
/// still evaluated, so a trap inside it still happens.
fn lower_length(
    checked: &CheckedProgram<'_>,
    operand: Idx<Expression>,
    bindings: &HashMap<Idx<Binding>, Place>,
    builder: &mut FlowBuilder,
) -> Operand {
    let Type::Array { length, .. } = checked.expressions[operand].ty else {
        unreachable!("checking requires an array operand for `len`")
    };
    lower_flow_operand(checked, operand, bindings, builder);
    integer(i128::from(length), Scalar::Int)
}

/// Lowers the evaluation a folded expression still owes. `len` reads its
/// length from its operand's type, so it folds while the operand still runs;
/// every other sub-expression of a folded constant folded too and has nothing
/// left to evaluate.
fn lower_folded_effects(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &HashMap<Idx<Binding>, Place>,
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
        // A call and an index never fold, so a folded expression only reaches
        // them under a `len`, and the rest hold no sub-expression at all.
        ExpressionValue::Integer
        | ExpressionValue::Boolean
        | ExpressionValue::Reference(_)
        | ExpressionValue::Call { .. }
        | ExpressionValue::Index { .. } => {}
    }
    for operand in operands {
        lower_folded_effects(checked, operand, bindings, builder);
    }
}

fn lower_flow_operand(
    checked: &CheckedProgram<'_>,
    id: Idx<Expression>,
    bindings: &HashMap<Idx<Binding>, Place>,
    builder: &mut FlowBuilder,
) -> Operand {
    let expression = &checked.expressions[id];
    let Some(scalar) = expression.ty.scalar() else {
        let place = lower_array_place(checked, id, bindings, builder);
        return load_place(builder, place, expression.ty.clone());
    };
    let ty: Type = scalar.into();
    if let Some(constant) = expression.constant.as_ref() {
        lower_folded_effects(checked, id, bindings, builder);
        return integer(
            constant
                .integer()
                .expect("a scalar expression folds to an integer")
                .to_i128()
                .expect("concrete Fern value fits in the IR representation"),
            scalar,
        );
    }
    match &expression.value {
        ExpressionValue::Integer | ExpressionValue::Boolean => {
            unreachable!("literal expressions are constant")
        }
        ExpressionValue::Array { .. } => unreachable!("an array literal has an array type"),
        // A `len` reaches here only when its operand holds a call, which is
        // what stops it from folding.
        ExpressionValue::Length { operand } => lower_length(checked, *operand, bindings, builder),
        ExpressionValue::Index { .. } => {
            let place = element_place(checked, id, bindings, builder);
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
            return self.operand;
        }
        let local = match self.spill {
            Some(local) => local,
            None => {
                let local = builder.local(self.ty.clone());
                builder.store_in(self.block, Place::Local(local), self.operand);
                self.spill = Some(local);
                local
            }
        };
        load_place(builder, Place::Local(local), self.ty.clone())
    }
}

/// An assignment target whose indices were lowered before the value stored
/// through it, so the place is rebuilt in whichever block the store lands in.
struct HeldPlace {
    root: Place,
    indices: Vec<(HeldOperand, std::ops::Range<usize>)>,
}

impl HeldPlace {
    fn read(&mut self, builder: &mut FlowBuilder) -> Place {
        let mut place = self.root.clone();
        for (index, span) in &mut self.indices {
            place = Place::Element {
                base: Box::new(place),
                index: index.read(builder),
                span: span.clone(),
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
    bindings: &HashMap<Idx<Binding>, Place>,
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
    bindings: &HashMap<Idx<Binding>, Place>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{frontend, semantic};

    fn integer(value: i128, ty: Scalar) -> Operand {
        Operand::Integer { value, ty }
    }

    fn convert(operand: Operand, ty: Scalar) -> Value {
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
        let syntax = frontend::parse(&crate::source::SourceMap::from_text(source)).unwrap();
        lower(semantic::check_root(&syntax).unwrap())
            .verify()
            .unwrap()
    }

    /// Loads, checks, lowers, and verifies a tree of `(relative path, source)`
    /// files whose root module is `app`, so multi-module lowering runs through
    /// real import resolution.
    fn lowered_tree<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) -> VerifiedProgram {
        let dir = crate::module::tree(files);
        let program = crate::module::load(&dir.path().join("app"), &[dir.path().to_owned()])
            .unwrap_or_else(|error| panic!("{}", error.into_compile_error()));
        lower(
            semantic::check(&program.syntax, &program.modules, &program.imports)
                .unwrap_or_else(|error| panic!("{}", error.render(&program.sources))),
        )
        .verify()
        .unwrap()
    }

    fn main_of(program: &VerifiedProgram) -> &Function {
        main_of_program(program.program())
    }

    fn main_of_program(program: &Program) -> &Function {
        &program.functions[program.main.0]
    }

    fn instructions(function: &Function) -> impl Iterator<Item = &Instruction> {
        function
            .flow
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
    }

    /// Every function `function` calls, in instruction order.
    fn call_targets(function: &Function) -> Vec<FunctionId> {
        instructions(function)
            .filter_map(|instruction| match instruction {
                Instruction::Call { function, .. } => Some(*function),
                _ => None,
            })
            .collect()
    }

    fn array(length: u64, element: Type) -> Type {
        Type::Array {
            length,
            element: Box::new(element),
        }
    }

    fn element(base: Place, index: Operand) -> Place {
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

    fn stored_places(function: &Function) -> Vec<String> {
        stores(function)
            .iter()
            .map(|(place, _)| describe(place))
            .collect()
    }

    fn stores(function: &Function) -> Vec<(Place, Operand)> {
        instructions(function)
            .filter_map(|instruction| match instruction {
                Instruction::Store { place, operand } => Some((place.clone(), *operand)),
                _ => None,
            })
            .collect()
    }

    fn loads(function: &Function) -> Vec<Place> {
        function
            .values
            .iter()
            .filter_map(|value| match &value.kind {
                ValueKind::Load(place) => Some(place.clone()),
                _ => None,
            })
            .collect()
    }

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
                    values: vec![1, 2, 3],
                },
                Global {
                    ty: array(2, array(2, Scalar::U8.into())),
                    values: vec![1, 2, 3, 4],
                },
                Global {
                    ty: Scalar::Int.into(),
                    values: vec![7],
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
            ["local0[0]", "local0[1]", "local0[2]", "local0[3]", "local1"]
        );
        // The fill evaluates its element once and copies that value.
        assert_eq!(call_targets(main), [FunctionId(0)]);
        let filled: Vec<_> = stores(main)[1..4].iter().map(|(_, value)| *value).collect();
        assert_eq!(filled, [filled[0]; 3]);
    }

    #[test]
    fn a_nested_literal_stores_each_row_through_the_outer_element_place() {
        // A folded literal writes its leaves straight through nested places.
        let folded = lowered(
            "fn main() -> void { var grid: [2][2]int = [[1, 2], [3, 4]]; exit(grid[0][0]); }",
        );
        assert_eq!(
            stored_places(main_of(&folded)),
            [
                "local0[0][0]",
                "local0[0][1]",
                "local0[1][0]",
                "local0[1][1]",
                "local1",
            ]
        );

        // A row that is not constant is built on its own and copied in.
        let runtime = lowered(
            "fn side() -> int { return 4; }
             fn main() -> void {
                 var grid: [2][2]int = [[1, 2], [3, side()]];
                 exit(grid[0][0]);
             }",
        );
        assert_eq!(
            stored_places(main_of(&runtime)),
            [
                "local1[0]",
                "local1[1]",
                "local0[0]",
                "local2[0]",
                "local2[1]",
                "local0[1]",
                "local3",
            ]
        );
    }

    #[test]
    fn a_constant_array_materializes_into_a_local_at_each_use() {
        let program = lowered(
            "const a: [2]int = [10, 20];
             fn main() -> void { var i = 1; exit(a[i] + a[0]); }",
        );
        let main = main_of(&program);
        // Two uses of the folded array, each filled elementwise before its read.
        assert_eq!(
            stored_places(main),
            ["local0", "local1[0]", "local1[1]", "local2[0]", "local2[1]"]
        );
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
        assert_eq!(
            stored_places(main),
            [
                "local0[0]",
                "local0[1]",
                "local0[2]",
                "local1",
                "local2",
                "local2[0]"
            ]
        );
        // `b` is its own array, so writing an element of it leaves `a` alone.
        let copy = stores(main)[4].1;
        assert_eq!(
            main.values[match copy {
                Operand::Value(ValueId(id)) => id,
                _ => panic!("a whole-array copy reads a place"),
            }]
            .kind,
            ValueKind::Load(Place::Local(LocalId(1)))
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
        // The literal, `a`, `last`, the copy the loop walks, the counter, and
        // `v`. The index binding is the counter rather than a seventh local.
        assert_eq!(
            main_of(&program).flow.locals,
            [
                array(2, Scalar::Int.into()),
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
                                    right: Operand::Integer { value: 1, .. },
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
        // The inner call's array result lands in a local before it is passed on.
        assert_eq!(main.flow.locals[2], row);
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
                ty: Scalar::Int.into(),
                values: vec![40],
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
                values: vec![0],
            }]
        );

        // The root module's files parse first, so `main` is the first function.
        assert_eq!(program.main, FunctionId(0));
        let main = main_of_program(program);
        assert_eq!(call_targets(main), [FunctionId(1), FunctionId(2)]);
        assert_eq!(
            stores(main),
            [(
                value.clone(),
                Operand::Integer {
                    value: 1,
                    ty: Scalar::Int,
                }
            )]
        );
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
                value: Some(Operand::Integer {
                    value: 7,
                    ty: Scalar::Int,
                })
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
            [(
                Place::Global(GlobalId(0)),
                Operand::Integer {
                    value: 255,
                    ty: Scalar::Int,
                }
            )]
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
                values: vec![0],
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
                    values: vec![2],
                },
                Global {
                    ty: Scalar::Int.into(),
                    values: vec![6],
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
