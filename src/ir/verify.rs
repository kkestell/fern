//! Structural, typing, reachability, and definite-initialization verification.

use crate::{
    CompileError,
    types::{Scalar, Type, UnaryOperator},
};

use std::collections::VecDeque;

use super::model::*;

/// A program that passed verification. Only `Program::verify` constructs one.
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
