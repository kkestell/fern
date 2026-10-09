//! Structural, typing, and reachability verification.

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

fn fits(source: &Type, expected: &Type) -> bool {
    source == expected
}

fn verify_literal(literal: &Literal) -> Result<Type, CompileError> {
    if let Literal::Null(ty) = literal {
        if !matches!(ty, Type::Pointer { .. }) {
            return Err(CompileError::new(format!(
                "internal compiler error: IR null has type `{ty}`"
            )));
        }
        return Ok(ty.clone());
    }
    let Literal::Integer { value, ty } = literal else {
        return Ok(literal.ty());
    };
    if ty.is_floating() {
        return Err(CompileError::new(format!(
            "internal compiler error: IR integer {value} has floating-point type {ty:?}"
        )));
    }
    if *value < ty.min() || *value > i128::from(ty.max()) {
        return Err(CompileError::new(format!(
            "internal compiler error: IR integer {value} out of range for {ty:?}"
        )));
    }
    Ok((*ty).into())
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
        ValueKind::PointerOffset {
            pointer, offset, ..
        } => {
            matches!(value.ty, Type::Pointer { .. })
                && operand_type(pointer.clone())? == value.ty
                && operand_type(offset.clone())? == Scalar::Int.into()
        }
        ValueKind::PointerDifference { left, right } => {
            let left = operand_type(left.clone())?;
            matches!(left, Type::Pointer { .. })
                && operand_type(right.clone())? == left
                && value.ty == Scalar::Int.into()
        }
        ValueKind::Load(place) => fits(&place_type(place)?, &value.ty),
        // A called function can write a global or a pointer target, and a
        // returned call writes the result storage, so an argument read from
        // one is copied.
        ValueKind::Borrow(place) => {
            value.ty.is_aggregate()
                && !matches!(
                    place.root(),
                    Place::Global(_) | Place::Indirect { .. } | Place::Result
                )
                && place_type(place)? == value.ty
        }
        ValueKind::Address(_) => {
            unreachable!("an address is checked against its place as a destination")
        }
        ValueKind::PointerCast { operand } => {
            matches!(value.ty, Type::Pointer { .. })
                && matches!(operand_type(operand.clone())?, Type::Pointer { .. })
        }
        // The defining call checks the result type against the callee's
        // signature.
        ValueKind::CallResult => true,
        ValueKind::Comparison {
            operator,
            left,
            right,
        } => {
            // Equality compares every type; ordering requires numbers.
            let left = operand_type(left.clone())?;
            let right = operand_type(right.clone())?;
            value.ty == Scalar::Bool.into()
                && (fits(&left, &right) || fits(&right, &left))
                && (operator.is_equality() || left.scalar().is_some_and(Scalar::is_numeric))
        }
        ValueKind::LogicalNot { operand } => {
            value.ty == Scalar::Bool.into() && operand_type(operand.clone())? == Scalar::Bool.into()
        }
        ValueKind::Convert { .. } | ValueKind::Unary { .. } | ValueKind::Binary { .. } => {
            valid_numeric_operation(value, operand_type)?
        }
    })
}

/// The value kinds defined only on numbers, whose result and operands are all
/// numeric scalars. A conversion crosses between the integer and
/// floating-point types; every operation keeps its operands' type, and the
/// operators defined on each of the two categories differ.
fn valid_numeric_operation(
    value: &Value,
    operand_type: &impl Fn(Operand) -> Result<Type, CompileError>,
) -> Result<bool, CompileError> {
    let operand = |operand| scalar(&operand_type(operand)?);
    let ty = scalar(&value.ty)?;
    Ok(match &value.kind {
        ValueKind::Convert { operand: source } => {
            operand(source.clone())?.is_numeric() && ty.is_numeric()
        }
        ValueKind::Unary {
            operator,
            operand: source,
        } => {
            let source = operand(source.clone())?;
            source == ty
                && match operator {
                    // Wrapping negation and complement read a bit pattern.
                    UnaryOperator::Complement => ty.is_integer(),
                    UnaryOperator::Negate => ty.is_numeric(),
                }
        }
        ValueKind::Binary {
            operator,
            left,
            right,
            ..
        } => {
            let (left, right) = (operand(left.clone())?, operand(right.clone())?);
            left == ty
                && if ty.is_floating() {
                    // Only the four arithmetic operators are defined on
                    // floating-point operands, which share one format.
                    right == ty && operator.defined_on_floating()
                } else {
                    // A shift takes its count independently of its left
                    // operand's type; every other operator takes one type.
                    ty.is_integer() && right.is_integer() && (operator.is_shift() || right == ty)
                }
        }
        _ => unreachable!("a numeric operation is a conversion, a unary, or a binary value"),
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
            self.verify_global(global)
                .map_err(|error| CompileError::new(format!("{error} (IR global {id})")))?;
        }
        let main = self.function(self.main)?;
        if main.parameters != 0 || main.result.is_some() || main.external.is_some() {
            return Err(CompileError::new(
                "internal compiler error: IR main takes parameters or returns a value",
            ));
        }
        for function in &self.functions {
            verify_parameters(function)?;
            for (id, constant) in function.constants.iter().enumerate() {
                self.verify_global(constant)
                    .map_err(|error| CompileError::new(format!("{error} (IR constant {id})")))?;
            }
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
        function: &Function,
        place: &Place,
        operand_type: &impl Fn(Operand) -> Result<Type, CompileError>,
    ) -> Result<Type, CompileError> {
        match place {
            Place::Local(LocalId(id)) => function.flow.locals.get(*id).cloned().ok_or_else(|| {
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
            Place::Constant(ConstantId(id)) => function
                .constants
                .get(*id)
                .map(|constant| constant.ty.clone())
                .ok_or_else(|| {
                    CompileError::new(format!(
                        "internal compiler error: IR uses unknown constant {id}"
                    ))
                }),
            Place::Element { base, index, .. } => {
                let base = self.place_type(function, base, operand_type)?;
                let Type::Array { element, .. } = base else {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR indexes `{base}`"
                    )));
                };
                let index = operand_type(index.clone())?;
                if index != Scalar::Int.into() {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR index has type `{index}`, expected `int`"
                    )));
                }
                Ok(*element)
            }
            Place::Field { base, ordinal } => {
                let base = self.place_type(function, base, operand_type)?;
                let Type::Struct(declared) = &base else {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR selects a field of `{base}`"
                    )));
                };
                self.structs[declared.id.0]
                    .fields
                    .get(*ordinal)
                    .cloned()
                    .ok_or_else(|| {
                        CompileError::new(format!(
                            "internal compiler error: IR selects field {ordinal} of `{base}`"
                        ))
                    })
            }
            Place::Result => function
                .result
                .clone()
                .filter(Type::is_aggregate)
                .ok_or_else(|| {
                    CompileError::new(
                        "internal compiler error: IR uses result storage in a function without an aggregate result",
                    )
                }),
            Place::Indirect { pointer } => match operand_type(pointer.clone())? {
                Type::Pointer {
                    target,
                } => Ok(*target),
                other => Err(CompileError::new(format!(
                    "internal compiler error: IR reads through `{other}`"
                ))),
            },
        }
    }

    /// The type of a place an instruction writes. Constants are static data
    /// that every use reads, so nothing may write one.
    fn destination_type(
        &self,
        function: &Function,
        place: &Place,
        operand_type: &impl Fn(Operand) -> Result<Type, CompileError>,
    ) -> Result<Type, CompileError> {
        if let Place::Constant(ConstantId(id)) = place.root() {
            return Err(CompileError::new(format!(
                "internal compiler error: IR writes constant {id}"
            )));
        }
        self.place_type(function, place, operand_type)
    }

    /// The scalar types a value of `ty` stores, in memory order. A struct's
    /// fields have their own types, so the sequence is derived from the
    /// program's struct table rather than from one leaf type.
    fn scalar_types(&self, ty: &Type, types: &mut Vec<Type>) {
        match ty {
            Type::Scalar(scalar) => types.push((*scalar).into()),
            Type::Array { length, element } => {
                for _ in 0..*length {
                    self.scalar_types(element, types);
                }
            }
            Type::Struct(declared) => {
                for field in &self.structs[declared.id.0].fields {
                    self.scalar_types(field, types);
                }
            }
            Type::Pointer { .. } => types.push(ty.clone()),
        }
    }

    fn verify_global(&self, global: &Global) -> Result<(), CompileError> {
        let Some(values) = &global.values else {
            return Ok(());
        };
        let expected_count = self.scalar_count(&global.ty).ok_or_else(|| {
            CompileError::new("internal compiler error: IR global scalar count overflows")
        })?;
        if u64::try_from(values.len()).expect("a Vec length fits u64") != expected_count {
            return Err(CompileError::new(format!(
                "internal compiler error: IR global of type `{}` holds {} values, expected {}",
                global.ty,
                values.len(),
                expected_count
            )));
        }
        let mut expected = Vec::with_capacity(values.len());
        self.scalar_types(&global.ty, &mut expected);
        for (value, expected) in values.iter().zip(&expected) {
            let ty = verify_literal(value)?;
            if ty != *expected {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR global of type `{}` holds a `{ty}` value where a `{expected}` value belongs",
                    global.ty
                )));
            }
        }
        Ok(())
    }

    fn scalar_count(&self, ty: &Type) -> Option<u64> {
        match ty {
            Type::Scalar(_) => Some(1),
            Type::Array { length, element } => length.checked_mul(self.scalar_count(element)?),
            Type::Struct(declared) => self.structs[declared.id.0]
                .fields
                .iter()
                .try_fold(0u64, |count, field| {
                    count.checked_add(self.scalar_count(field)?)
                }),
            Type::Pointer { .. } => Some(1),
        }
    }

    fn verify_function(&self, function: &Function) -> Result<(), CompileError> {
        if let Some(symbol) = &function.external {
            let external_type = |ty: &Type| {
                matches!(ty, Type::Pointer { .. }) || ty.scalar().is_some_and(Scalar::is_integer)
            };
            if symbol == "main"
                || symbol.is_empty()
                || !symbol.bytes().enumerate().all(|(i, b)| {
                    b == b'_' || b.is_ascii_alphabetic() || (i > 0 && b.is_ascii_digit())
                })
                || function.flow.locals.len() != function.parameters
                || !function.flow.locals.iter().all(external_type)
                || function
                    .result
                    .as_ref()
                    .is_some_and(|ty| !external_type(ty))
                || !function.values.is_empty()
                || !function.constants.is_empty()
                || function.flow.blocks.len() != 1
                || !function.flow.blocks[0].instructions.is_empty()
                || function.flow.blocks[0].terminator != Terminator::Unreachable
            {
                return Err(CompileError::new(
                    "internal compiler error: invalid external declaration",
                ));
            }
            return Ok(());
        }
        let flow = &function.flow;
        verify_target(flow.entry, flow)?;
        let definitions = value_definitions(function)?;
        let reachable = reachable_blocks(flow)?;
        for (index, block) in flow.blocks.iter().enumerate() {
            if reachable[index] && block.terminator == Terminator::Unreachable {
                return Err(CompileError::new(format!(
                    "internal compiler error: reachable IR block {index} is not terminated"
                )));
            }
            self.verify_block(function, index, &definitions)?;
        }
        Ok(())
    }

    /// Checks one block's instructions and its terminator.
    fn verify_block(
        &self,
        function: &Function,
        block_index: usize,
        definitions: &[Definition],
    ) -> Result<(), CompileError> {
        let block = &function.flow.blocks[block_index];
        for (position, instruction) in block.instructions.iter().enumerate() {
            self.verify_instruction(function, definitions, (block_index, position), instruction)?;
        }
        self.verify_terminator(function, block_index, definitions)
    }

    fn verify_instruction(
        &self,
        function: &Function,
        definitions: &[Definition],
        (block_index, position): (usize, usize),
        instruction: &Instruction,
    ) -> Result<(), CompileError> {
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
                // A pointer may be written through, so an addressed place is
                // checked as a destination: no constant. No
                // pointer reaches the result storage, so only the returned
                // call, literal, or local the function builds there reads or
                // writes it.
                let valid = if let ValueKind::Address(place) = &value.kind {
                    let target = self.destination_type(function, place, &operand_type)?;
                    !matches!(place.root(), Place::Result)
                        && value.ty
                            == Type::Pointer {
                                target: Box::new(target),
                            }
                } else {
                    let place_type =
                        |place: &Place| self.place_type(function, place, &operand_type);
                    valid_value(value, &operand_type, &place_type)?
                };
                if !valid {
                    return Err(invalid_value(*id, value));
                }
                Ok(())
            }
            Instruction::Store { place, operand } => {
                let destination = self.destination_type(function, place, &operand_type)?;
                let source = operand_type(operand.clone())?;
                if !fits(&source, &destination) {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR store has type `{source}`, expected `{destination}`"
                    )));
                }
                Ok(())
            }
            Instruction::Call {
                result,
                destination,
                function: callee,
                arguments,
                ..
            } => self.verify_call(
                function,
                *callee,
                arguments,
                (*result, destination.as_ref()),
                &operand_type,
            ),
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
                if operand_type(condition.clone())? != Scalar::Bool.into() {
                    return Err(CompileError::new(
                        "internal compiler error: IR branch requires bool",
                    ));
                }
                Ok(())
            }
            Terminator::Return { value } => {
                verify_return(function.result.as_ref(), value.clone(), &operand_type)
            }
            Terminator::Jump { .. } | Terminator::Unreachable => Ok(()),
        }
    }

    fn verify_call(
        &self,
        caller: &Function,
        callee: FunctionId,
        arguments: &[Operand],
        (result, destination): (Option<ValueId>, Option<&Place>),
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
        for (index, argument) in arguments.iter().enumerate() {
            let source = operand_type(argument.clone())?;
            let parameter = &target.flow.locals[index];
            if !fits(&source, parameter) {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR call argument {index} has type `{source}`, expected `{parameter}`"
                )));
            }
        }
        if let Some(destination) = destination {
            return self.verify_call_destination(caller, target, result, destination, operand_type);
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

impl Program {
    /// Checks a call that writes its aggregate result straight into
    /// `destination`. The callee may build its result there while it reads
    /// its arguments and any global or pointer target, so the destination is
    /// rooted in a local or the caller's own result storage, which no
    /// argument shares.
    fn verify_call_destination(
        &self,
        caller: &Function,
        target: &Function,
        result: Option<ValueId>,
        destination: &Place,
        operand_type: &impl Fn(Operand) -> Result<Type, CompileError>,
    ) -> Result<(), CompileError> {
        if result.is_some() {
            return Err(CompileError::new(
                "internal compiler error: IR call both defines a result and writes a destination",
            ));
        }
        if !matches!(destination.root(), Place::Local(_) | Place::Result) {
            return Err(CompileError::new(format!(
                "internal compiler error: IR call writes its result into `{destination:?}`"
            )));
        }
        let expected = self.destination_type(caller, destination, operand_type)?;
        match target.result.as_ref() {
            Some(result) if result.is_aggregate() && *result == expected => Ok(()),
            result => Err(CompileError::new(format!(
                "internal compiler error: IR call writes a `{}` result into a `{expected}` destination",
                result.map_or_else(|| "void".to_owned(), ToString::to_string)
            ))),
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
            if !fits(&source, result) {
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
        Operand::Literal(literal) => verify_literal(&literal),
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
