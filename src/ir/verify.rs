//! Structural, typing, reachability, and definite-initialization verification.

use crate::{
    CompileError,
    layout::{Layouts, StructFields},
    types::{
        MAX_AGGREGATE_LAYOUT_BYTES, MAX_STRUCT_CONTAINMENT_DEPTH, Scalar, StructId, StructType,
        Type, UnaryOperator,
    },
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

/// The type a literal has, rejecting an integer whose value its scalar type
/// cannot hold.
fn verify_literal(literal: &Literal) -> Result<Type, CompileError> {
    if let Literal::Null(ty) = literal {
        if !matches!(ty, Type::Pointer { .. }) {
            return Err(CompileError::new(format!(
                "internal compiler error: IR null has non-pointer type `{ty}`"
            )));
        }
        return Ok(ty.clone());
    }
    if let Literal::EmptySlice(ty) = literal {
        if !matches!(ty, Type::Slice { .. }) {
            return Err(CompileError::new(format!(
                "internal compiler error: IR empty slice has non-slice type `{ty}`"
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
        ValueKind::Load(place) => place_type(place)? == value.ty && value.span.is_none(),
        ValueKind::AddressOf(place) => {
            let place = place_type(place)?;
            value.span.is_some()
                && matches!(&value.ty, Type::Pointer { target, .. } if **target == place)
        }
        // The defining call checks the result type against the callee's
        // signature, and owns the span the call reports.
        ValueKind::CallResult => value.span.is_none(),
        ValueKind::Comparison {
            operator,
            left,
            right,
        } => {
            let left = operand_type(left.clone())?;
            value.span.is_some()
                && value.ty == Scalar::Bool.into()
                && (if operator.is_equality() {
                    let right = operand_type(right.clone())?;
                    left == right
                        || matches!(
                            (&left, &right),
                            (Type::Pointer { target: left, .. }, Type::Pointer { target: right, .. }) if left == right
                        )
                        || matches!(
                            (&left, &right),
                            (Type::Slice { element: left, .. }, Type::Slice { element: right, .. }) if left == right
                        )
                } else {
                    left == operand_type(right.clone())? && left.scalar().is_some()
                })
        }
        ValueKind::LogicalNot { operand } => {
            value.span.is_some()
                && value.ty == Scalar::Bool.into()
                && operand_type(operand.clone())? == Scalar::Bool.into()
        }
        ValueKind::WholeSlice(place) => {
            let place = place_type(place)?;
            value.span.is_none()
                && matches!(
                    (&place, &value.ty),
                    (Type::Array { element: source, .. }, Type::Slice { element: destination, .. })
                        if source == destination
                )
        }
        ValueKind::SliceRange { slice, low, high } => {
            let source = operand_type(slice.clone())?;
            value.span.is_some()
                && operand_type(low.clone())? == Scalar::Int.into()
                && operand_type(high.clone())? == Scalar::Int.into()
                && matches!(&source, Type::Slice { .. })
                && source.value_compatible(&value.ty)
        }
        ValueKind::SliceLength { slice } => {
            value.span.is_none()
                && value.ty == Scalar::Int.into()
                && matches!(operand_type(slice.clone())?, Type::Slice { .. })
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
    if let ValueKind::Convert {
        operand: source,
        truncating: false,
    } = &value.kind
    {
        let source = operand_type(source.clone())?;
        if matches!(source, Type::Pointer { .. }) {
            return Ok(value.ty == Scalar::Uint.into());
        }
    }
    let operand = |operand| scalar(&operand_type(operand)?);
    let ty = scalar(&value.ty)?;
    Ok(match &value.kind {
        ValueKind::Convert {
            operand: source,
            truncating,
        } => {
            let source = operand(source.clone())?;
            if *truncating {
                // Truncation reinterprets a two's complement bit pattern.
                source.is_integer() && ty.is_integer()
            } else {
                source.is_numeric()
                    && ty.is_numeric()
                    && (source.all_values_fit(ty) || value.span.is_some())
            }
        }
        ValueKind::Unary {
            operator,
            operand: source,
        } => {
            let source = operand(source.clone())?;
            value.span.is_some()
                && source == ty
                && match operator {
                    // Wrapping negation and complement read a bit pattern.
                    UnaryOperator::WrappingNegate | UnaryOperator::Complement => ty.is_integer(),
                    UnaryOperator::Negate => ty.is_floating() || (ty.is_integer() && ty.signed()),
                }
        }
        ValueKind::Binary {
            operator,
            left,
            right,
            ..
        } => {
            let (left, right) = (operand(left.clone())?, operand(right.clone())?);
            value.span.is_some()
                && left == ty
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

/// How far one struct's fields have been walked while looking for a cycle.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Resolution {
    Unvisited,
    Visiting,
    Acyclic,
}

/// The struct a field of this type stores inline. An array stores its elements
/// inline too, so `[N]Point` contains `Point`.
fn contained_struct(ty: &Type) -> Option<usize> {
    match ty {
        Type::Scalar(_) => None,
        Type::Pointer { .. } => None,
        Type::Slice { .. } => None,
        Type::Array { element, .. } => contained_struct(element),
        Type::Struct(declared) => Some(declared.id.0),
    }
}

/// Where a value is defined, and whether a call defined it.
#[derive(Clone, Copy)]
struct Definition {
    block: usize,
    position: usize,
    from_call: bool,
}

impl StructFields for Program {
    fn field_count(&self, id: StructId) -> usize {
        self.structs[id.0].fields.len()
    }

    fn field_type(&self, id: StructId, ordinal: usize) -> &Type {
        &self.structs[id.0].fields[ordinal]
    }
}

impl Program {
    pub(crate) fn verify(self) -> Result<VerifiedProgram, CompileError> {
        let layouts = Layouts::default();
        self.verify_structs(&layouts)?;
        for (id, global) in self.globals.iter().enumerate() {
            self.verify_global(&layouts, global)
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
            self.verify_function(&layouts, function)?;
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
                let index = operand_type(index.clone())?;
                if index != Scalar::Int.into() {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR index has type `{index}`, expected `int`"
                    )));
                }
                Ok(*element)
            }
            Place::Field { base, ordinal } => {
                let base = self.place_type(flow, base, operand_type)?;
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
            Place::Indirect { pointer, .. } => {
                let pointer = operand_type(pointer.clone())?;
                let Type::Pointer { target, .. } = pointer else {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR dereferences `{pointer}`"
                    )));
                };
                Ok(*target)
            }
            Place::SliceElement { slice, index, .. } => {
                let slice = operand_type(slice.clone())?;
                let Type::Slice { element, .. } = slice else {
                    return Err(CompileError::new(format!(
                        "internal compiler error: IR indexes slice value `{slice}`"
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
        }
    }

    /// Checks the program's struct table before anything reads it: every
    /// struct a field names is defined, and no struct contains itself. A field
    /// stores its value inline, directly or through an array, so a cycle among
    /// them has no finite layout.
    fn verify_structs(&self, layouts: &Layouts) -> Result<(), CompileError> {
        for definition in &self.structs {
            for field in &definition.fields {
                self.verify_type(field)?;
            }
        }
        let mut state = vec![Resolution::Unvisited; self.structs.len()];
        for id in 0..self.structs.len() {
            self.verify_acyclic(id, &mut state, 0)?;
        }
        for id in 0..self.structs.len() {
            self.verify_aggregate_layout(
                layouts,
                &Type::Struct(StructType {
                    id: StructId(id),
                    name: String::new(),
                }),
            )?;
        }
        Ok(())
    }

    fn verify_acyclic(
        &self,
        id: usize,
        state: &mut [Resolution],
        depth: usize,
    ) -> Result<(), CompileError> {
        match state[id] {
            Resolution::Acyclic => return Ok(()),
            Resolution::Visiting => {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR struct {id} contains itself"
                )));
            }
            Resolution::Unvisited => {}
        }
        if depth == MAX_STRUCT_CONTAINMENT_DEPTH {
            return Err(CompileError::new(format!(
                "internal compiler error: IR struct containment exceeds compiler limit of {MAX_STRUCT_CONTAINMENT_DEPTH}"
            )));
        }
        state[id] = Resolution::Visiting;
        for field in &self.structs[id].fields {
            if let Some(contained) = contained_struct(field) {
                self.verify_acyclic(contained, state, depth + 1)?;
            }
        }
        state[id] = Resolution::Acyclic;
        Ok(())
    }

    /// Rejects a type naming a struct the program does not define, so an
    /// unknown struct cannot reach the backend through a global, a signature,
    /// a local, or a value that nothing else looks at.
    fn verify_type(&self, ty: &Type) -> Result<(), CompileError> {
        match ty {
            Type::Scalar(_) => Ok(()),
            Type::Pointer { target, .. } => self.verify_type(target),
            Type::Slice { element, .. } => self.verify_type(element),
            Type::Array { element, .. } => self.verify_type(element),
            Type::Struct(declared) if declared.id.0 < self.structs.len() => Ok(()),
            Type::Struct(declared) => Err(CompileError::new(format!(
                "internal compiler error: IR uses unknown struct {}",
                declared.id.0
            ))),
        }
    }

    /// Repeats the backend's layout arithmetic with checked operations at the
    /// verified-IR boundary. The backend may consequently keep its compact
    /// layout API without accepting an overflowing source-controlled type.
    fn verify_aggregate_layout(&self, layouts: &Layouts, ty: &Type) -> Result<(), CompileError> {
        let Some(size) = layouts.size(self, ty) else {
            return Err(CompileError::new(
                "internal compiler error: IR aggregate layout overflows the target address space",
            ));
        };
        if size > MAX_AGGREGATE_LAYOUT_BYTES {
            return Err(CompileError::new(format!(
                "internal compiler error: IR aggregate layout exceeds compiler limit of {} PiB",
                MAX_AGGREGATE_LAYOUT_BYTES >> 50
            )));
        }
        Ok(())
    }

    /// The scalar types a value of `ty` stores, in memory order. A struct's
    /// fields have their own types, so the sequence is derived from the
    /// program's struct table rather than from one leaf type.
    fn scalar_types(&self, ty: &Type, types: &mut Vec<Type>) {
        match ty {
            Type::Scalar(scalar) => types.push((*scalar).into()),
            Type::Pointer { .. } => types.push(ty.clone()),
            Type::Slice { .. } => types.push(ty.clone()),
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
        }
    }

    fn verify_global(&self, layouts: &Layouts, global: &Global) -> Result<(), CompileError> {
        self.verify_type(&global.ty)?;
        self.verify_aggregate_layout(layouts, &global.ty)?;
        let expected_count = self.scalar_count(&global.ty).ok_or_else(|| {
            CompileError::new("internal compiler error: IR global scalar count overflows")
        })?;
        if u64::try_from(global.values.len()).expect("a Vec length fits u64") != expected_count {
            return Err(CompileError::new(format!(
                "internal compiler error: IR global of type `{}` holds {} values, expected {}",
                global.ty,
                global.values.len(),
                expected_count
            )));
        }
        let mut expected = Vec::with_capacity(global.values.len());
        self.scalar_types(&global.ty, &mut expected);
        for (value, expected) in global.values.iter().zip(&expected) {
            let ty = verify_literal(value)?;
            if !ty.value_compatible(expected) {
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
            Type::Pointer { .. } => Some(1),
            Type::Slice { .. } => Some(1),
            Type::Array { length, element } => length.checked_mul(self.scalar_count(element)?),
            Type::Struct(declared) => self.structs[declared.id.0]
                .fields
                .iter()
                .try_fold(0u64, |count, field| {
                    count.checked_add(self.scalar_count(field)?)
                }),
        }
    }

    fn verify_function(&self, layouts: &Layouts, function: &Function) -> Result<(), CompileError> {
        let flow = &function.flow;
        for ty in flow
            .locals
            .iter()
            .chain(function.result.as_ref())
            .chain(function.values.iter().map(|value| &value.ty))
        {
            self.verify_type(ty)?;
            self.verify_aggregate_layout(layouts, ty)?;
        }
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
            Instruction::Check { place } => {
                self.place_type(flow, place, &operand_type)?;
                Ok(())
            }
            Instruction::Store { place, operand } => {
                let destination = self.place_type(flow, place, &operand_type)?;
                let source = operand_type(operand.clone())?;
                if !source.value_compatible(&destination) {
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
                if operand_type(condition.clone())? != Scalar::Bool.into() {
                    return Err(CompileError::new(
                        "internal compiler error: IR branch requires bool",
                    ));
                }
                Ok(())
            }
            Terminator::Exit { status, .. } => {
                if operand_type(status.clone())? != Scalar::Int.into() {
                    return Err(CompileError::new(
                        "internal compiler error: IR exit requires int",
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
        for (index, argument) in arguments.iter().enumerate() {
            let source = operand_type(argument.clone())?;
            let parameter = &target.flow.locals[index];
            if !source.value_compatible(parameter) {
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
        Instruction::Check { .. } | Instruction::Store { .. } => None,
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
            if !source.value_compatible(result) {
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
