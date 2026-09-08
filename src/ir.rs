use crate::{
    CompileError,
    frontend::{
        BinaryOperator, ComparisonOperator, Expression, ForHeader, LogicalOperator, Statement,
        StatementKind, UnaryOperator,
    },
    semantic::{Binding, CheckedEntry, ExpressionValue},
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
pub(crate) struct BlockId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operand {
    // i128 holds both the signed minima and the full u64 range without bit reinterpretation.
    Integer { value: i128, ty: Type },
    Value(ValueId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValueKind {
    Load(LocalId),
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
    Store { local: LocalId, operand: Operand },
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

#[derive(Debug)]
pub(crate) struct Entry {
    // Each position defines the corresponding IR-local value ID.
    pub values: Vec<Value>,
    pub flow: ControlFlow,
}

#[derive(Debug)]
pub(crate) struct VerifiedEntry(Entry);

impl VerifiedEntry {
    pub(crate) fn entry(&self) -> &Entry {
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
    mut local_type: impl FnMut(LocalId) -> Result<Type, CompileError>,
) -> Result<bool, CompileError> {
    Ok(match value.kind {
        ValueKind::Load(local) => local_type(local)? == value.ty && value.span.is_none(),
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

impl Entry {
    pub(crate) fn verify(self) -> Result<VerifiedEntry, CompileError> {
        self.verify_flow(&self.flow)?;
        Ok(VerifiedEntry(self))
    }

    fn verify_flow(&self, flow: &ControlFlow) -> Result<(), CompileError> {
        verify_target(flow.entry, flow)?;
        let mut definitions = vec![None; self.values.len()];
        for (block_index, block) in flow.blocks.iter().enumerate() {
            for (position, instruction) in block.instructions.iter().enumerate() {
                if let Instruction::Value(ValueId(id)) = *instruction {
                    let Some(definition) = definitions.get_mut(id) else {
                        return Err(CompileError::new(format!(
                            "internal compiler error: IR defines unknown value {id}"
                        )));
                    };
                    if definition.replace((block_index, position)).is_some() {
                        return Err(CompileError::new(format!(
                            "internal compiler error: IR defines value {id} more than once"
                        )));
                    }
                }
            }
        }
        if let Some(id) = definitions.iter().position(Option::is_none) {
            return Err(CompileError::new(format!(
                "internal compiler error: IR does not define value {id}"
            )));
        }

        let reachable = reachable_blocks(flow)?;
        let initialized_at_entry = initialized_locals(flow, &reachable);
        for (block_index, block) in flow.blocks.iter().enumerate() {
            if reachable[block_index] && block.terminator == Terminator::Unreachable {
                return Err(CompileError::new(format!(
                    "internal compiler error: reachable IR block {block_index} is not terminated"
                )));
            }
            let mut initialized = initialized_at_entry[block_index].clone();
            for (position, instruction) in block.instructions.iter().enumerate() {
                match instruction {
                    Instruction::Value(ValueId(id)) => {
                        let value = &self.values[*id];
                        if let ValueKind::Load(LocalId(local)) = value.kind
                            && local < flow.locals.len()
                            && !initialized.contains(local)
                        {
                            return Err(CompileError::new(format!(
                                "internal compiler error: IR loads uninitialized local {local}"
                            )));
                        }
                        let valid = valid_value(
                            value,
                            |operand| {
                                flow_operand_type(
                                    operand,
                                    &self.values,
                                    &definitions,
                                    block_index,
                                    position,
                                )
                            },
                            |LocalId(id)| {
                                flow.locals.get(id).copied().ok_or_else(|| {
                                    CompileError::new(format!(
                                        "internal compiler error: IR loads unknown local {id}"
                                    ))
                                })
                            },
                        )?;
                        if !valid {
                            return Err(invalid_value(*id, value));
                        }
                    }
                    Instruction::Store { local, operand } => {
                        let destination = flow.locals.get(local.0).copied().ok_or_else(|| {
                            CompileError::new(format!(
                                "internal compiler error: IR stores to unknown local {}",
                                local.0
                            ))
                        })?;
                        let source = flow_operand_type(
                            *operand,
                            &self.values,
                            &definitions,
                            block_index,
                            position,
                        )?;
                        if source != destination {
                            return Err(CompileError::new(format!(
                                "internal compiler error: IR store has type {source:?}, expected {destination:?}"
                            )));
                        }
                        initialized.insert(local.0);
                    }
                }
            }
            let end = block.instructions.len();
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
                    if flow_operand_type(*condition, &self.values, &definitions, block_index, end)?
                        != Type::Bool
                    {
                        return Err(CompileError::new(
                            "internal compiler error: IR branch requires bool",
                        ));
                    }
                }
                Terminator::Exit { status, .. } => {
                    if flow_operand_type(*status, &self.values, &definitions, block_index, end)?
                        != Type::Int
                    {
                        return Err(CompileError::new(
                            "internal compiler error: IR exit requires int",
                        ));
                    }
                }
                Terminator::Unreachable => {}
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

fn initialized_locals(flow: &ControlFlow, reachable: &[bool]) -> Vec<LocalSet> {
    let mut predecessors = vec![Vec::new(); flow.blocks.len()];
    let mut stores = vec![Vec::new(); flow.blocks.len()];
    for (block_index, block) in flow.blocks.iter().enumerate() {
        for instruction in &block.instructions {
            if let Instruction::Store { local, .. } = instruction {
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
            Terminator::Exit { .. } | Terminator::Unreachable => {}
        }
    }

    let mut initialized = vec![LocalSet::full(flow.locals.len()); flow.blocks.len()];
    initialized[flow.entry.0] = LocalSet::empty(flow.locals.len());
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
            Terminator::Exit { .. } | Terminator::Unreachable => {}
        }
    }
    Ok(reachable)
}

fn flow_operand_type(
    operand: Operand,
    values: &[Value],
    definitions: &[Option<(usize, usize)>],
    block: usize,
    position: usize,
) -> Result<Type, CompileError> {
    match operand {
        Operand::Integer { value, ty } => verify_integer(value, ty),
        Operand::Value(ValueId(id)) => {
            let Some(Some((definition_block, definition_position))) = definitions.get(id) else {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR references undefined value {id}"
                )));
            };
            if *definition_block != block || *definition_position >= position {
                return Err(CompileError::new(format!(
                    "internal compiler error: IR value {id} is not defined earlier in block {block}"
                )));
            }
            Ok(values[id].ty)
        }
    }
}

pub(crate) fn lower(checked: CheckedEntry<'_>) -> Entry {
    lower_control_flow(checked)
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

    fn store(&mut self, local: LocalId, operand: Operand) {
        self.blocks[self.current.0]
            .instructions
            .push(Instruction::Store { local, operand });
    }

    fn terminate(&mut self, terminator: Terminator) {
        let slot = &mut self.blocks[self.current.0].terminator;
        assert!(slot.is_none(), "IR builder terminated a block twice");
        *slot = Some(terminator);
    }

    fn finish(self) -> Entry {
        Entry {
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

fn lower_control_flow(checked: CheckedEntry<'_>) -> Entry {
    let mut builder = FlowBuilder::new();
    let mut bindings = HashMap::new();
    for &statement in &checked.module_bindings {
        lower_flow_binding(&checked, statement, &mut bindings, &mut builder);
    }
    let terminated = lower_flow_body(
        &checked,
        &checked.syntax.functions[checked.main].body,
        &mut bindings,
        &[],
        &mut builder,
    );
    if !terminated {
        builder.terminate(Terminator::Exit {
            status: integer(0, Type::Int),
        });
    }
    builder.finish()
}

fn lower_flow_binding(
    checked: &CheckedEntry<'_>,
    statement: Idx<Statement>,
    bindings: &mut HashMap<Idx<Binding>, LocalId>,
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
    builder.store(local, operand);
    bindings.insert(binding, local);
}

fn lower_flow_body(
    checked: &CheckedEntry<'_>,
    body: &[Idx<Statement>],
    bindings: &mut HashMap<Idx<Binding>, LocalId>,
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
    checked: &CheckedEntry<'_>,
    statement: Idx<Statement>,
    bindings: &mut HashMap<Idx<Binding>, LocalId>,
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
            let local = bindings[&binding];
            let left = load_local(builder, local, checked.bindings[binding].ty);
            let right = lower_flow_operand(checked, *value, bindings, builder);
            let result = builder.value(Value {
                span: Some(operator_span.clone()),
                ty: checked.bindings[binding].ty,
                kind: ValueKind::Binary {
                    operator: *operator,
                    form: BinaryForm::CompoundAssignment,
                    left,
                    right,
                },
            });
            builder.store(local, result);
            false
        }
        StatementKind::Block { body } => lower_flow_body(checked, body, bindings, loops, builder),
        StatementKind::Exit { argument } => {
            let status = lower_flow_operand(checked, *argument, bindings, builder);
            builder.terminate(Terminator::Exit { status });
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

fn lower_if(
    checked: &CheckedEntry<'_>,
    condition: Idx<Expression>,
    then_body: &[Idx<Statement>],
    else_branch: Option<Idx<Statement>>,
    bindings: &mut HashMap<Idx<Binding>, LocalId>,
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
    checked: &CheckedEntry<'_>,
    label: Option<Spur>,
    header: &ForHeader,
    body: &[Idx<Statement>],
    bindings: &mut HashMap<Idx<Binding>, LocalId>,
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
    checked: &CheckedEntry<'_>,
    id: Idx<Expression>,
    bindings: &HashMap<Idx<Binding>, LocalId>,
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
            load_local(builder, bindings[binding], expression.ty)
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
            let left = lower_flow_operand(checked, *left, bindings, builder);
            let right = lower_flow_operand(checked, *right, bindings, builder);
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
            let left_id = *left;
            let left = lower_flow_operand(checked, left_id, bindings, builder);
            let left_type = checked.expressions[left_id].ty;
            let saved_left = builder.local(left_type);
            builder.store(saved_left, left);
            let right = lower_flow_operand(checked, *right, bindings, builder);
            let left = load_local(builder, saved_left, left_type);
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
    }
}

fn load_local(builder: &mut FlowBuilder, local: LocalId, ty: Type) -> Operand {
    builder.value(Value {
        span: None,
        ty,
        kind: ValueKind::Load(local),
    })
}

fn lower_logical(
    checked: &CheckedEntry<'_>,
    operator: LogicalOperator,
    left: Idx<Expression>,
    right: Idx<Expression>,
    bindings: &HashMap<Idx<Binding>, LocalId>,
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
    builder.store(result, integer(short_value, Type::Bool));
    builder.terminate(Terminator::Jump { target: join_block });

    builder.select(right_block);
    let right = lower_flow_operand(checked, right, bindings, builder);
    builder.store(result, right);
    builder.terminate(Terminator::Jump { target: join_block });

    builder.select(join_block);
    load_local(builder, result, Type::Bool)
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

    fn entry(values: Vec<Value>, exit: Operand) -> Entry {
        Entry {
            flow: ControlFlow {
                entry: BlockId(0),
                locals: vec![],
                blocks: vec![Block {
                    instructions: (0..values.len())
                        .map(|id| Instruction::Value(ValueId(id)))
                        .collect(),
                    terminator: Terminator::Exit { status: exit },
                }],
            },
            values,
        }
    }

    #[test]
    fn conversions_use_existing_operands_and_keep_their_source_spans() {
        let text = "fn main() -> void { var x: u64 = 42; var y = u8(u16(x)); exit(int(y)); }";
        let syntax = frontend::parse(text).unwrap();
        let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
        let values = &entry.entry().values;
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
    fn lowered_entries() {
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
            let syntax = frontend::parse(&format!("fn main() -> void {{ {body} }}")).unwrap();
            let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
            insta::assert_debug_snapshot!(name, entry.entry());
        }
    }

    #[test]
    fn module_bindings_are_initialized_before_the_entry_body() {
        let syntax = frontend::parse(
            "var counter = start;
             const start: int = 40;
             const step = 2;
             fn main() -> void {
                 { var counter: u8 = 1; counter = 2; }
                 counter = counter + step;
                 exit(counter);
             }",
        )
        .unwrap();
        let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
        insta::assert_debug_snapshot!("module_bindings", entry.entry());
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
            let error = entry(values, integer(0, Type::Int)).verify().unwrap_err();
            assert!(
                error.to_string().contains("undefined value")
                    || error.to_string().contains("not defined earlier")
            );
        }
        for reference in [0, 1, usize::MAX] {
            assert!(
                entry(vec![], Operand::Value(ValueId(reference)))
                    .verify()
                    .is_err()
            );
        }
        assert!(
            entry(
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
            entry(
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
                entry(
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
                let error = entry(vec![convert(integer(value, ty), ty)], integer(0, Type::Int))
                    .verify()
                    .unwrap_err();
                assert!(error.to_string().contains("out of range"));
            }
            assert!(
                entry(
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
                            let result = entry(
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
                    entry(vec![convert(integer(0, source), source)], exit)
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
                entry(
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
            assert!(entry(vec![], integer(value, Type::Int)).verify().is_err());
        }
    }

    #[test]
    fn verification_accepts_checked_conversion_operands() {
        entry(
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
        let syntax = frontend::parse(text).unwrap();
        let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
        let values = &entry.entry().values;
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
            entry.entry().flow.blocks.last().unwrap().terminator,
            Terminator::Exit { .. }
        ));
    }

    #[test]
    fn lowering_contextualizes_an_untyped_runtime_shift_operand() {
        let syntax = frontend::parse(
            "fn main() -> void { var count: uint = 3; const shifted: u64 = (1 << count) << count; }",
        )
        .unwrap();
        let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
        assert_eq!(entry.entry().values.len(), 4);
        assert_eq!(
            entry.entry().values[1].kind,
            ValueKind::Binary {
                operator: BinaryOperator::ShiftLeft,
                form: BinaryForm::Infix,
                left: integer(1, Type::U64),
                right: Operand::Value(ValueId(0)),
            }
        );
        assert_eq!(entry.entry().values[1].ty, Type::U64);
        assert_eq!(
            entry.entry().values[3].kind,
            ValueKind::Binary {
                operator: BinaryOperator::ShiftLeft,
                form: BinaryForm::Infix,
                left: Operand::Value(ValueId(1)),
                right: Operand::Value(ValueId(2)),
            }
        );
        assert_eq!(entry.entry().values[3].ty, Type::U64);
    }

    #[test]
    fn lowering_preserves_contextual_types_through_grouping() {
        let syntax = frontend::parse(
            "fn main() -> void { const grouped: u8 = ((42)); var count: uint = 1; const shifted: u64 = (1 + 2) << count; }",
        )
        .unwrap();
        let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
        assert_eq!(entry.entry().values.len(), 2);
        assert_eq!(
            entry.entry().values[1].kind,
            ValueKind::Binary {
                operator: BinaryOperator::ShiftLeft,
                form: BinaryForm::Infix,
                left: integer(3, Type::U64),
                right: Operand::Value(ValueId(0)),
            }
        );
        assert_eq!(entry.entry().values[1].ty, Type::U64);
    }

    #[test]
    fn verification_checks_operation_shapes_and_accepts_independent_shift_counts() {
        let operation = |ty, kind| Value {
            span: Some(0..1),
            ty,
            kind,
        };
        entry(
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
                entry(vec![value], integer(0, Type::Int))
                    .verify()
                    .unwrap_err()
                    .to_string()
                    .contains("invalid IR value")
            );
        }
    }

    fn lowered_flow(source: &str) -> VerifiedEntry {
        let syntax = frontend::parse(source).unwrap();
        lower(semantic::check(&syntax).unwrap()).verify().unwrap()
    }

    #[test]
    fn lowers_nested_control_flow_and_mutations() {
        let entry = lowered_flow(
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
        insta::assert_debug_snapshot!("nested_control_flow", entry.entry());
    }

    #[test]
    fn lowers_short_circuit_paths_before_their_uses() {
        let entry = lowered_flow(
            "fn main() -> void {
                var divisor = 0;
                var enabled = true;
                if enabled && divisor != 0 && 10 / divisor > 1 { exit(1); }
                exit(0);
            }",
        );
        insta::assert_debug_snapshot!("short_circuit_control_flow", entry.entry());
    }

    #[test]
    fn control_flow_in_an_unreferenced_function_does_not_change_main_ir() {
        let syntax = frontend::parse(
            "fn helper() -> void { if true {} }
             fn main() -> void { exit(0); }",
        )
        .unwrap();
        let entry = lower(semantic::check(&syntax).unwrap()).verify().unwrap();
        assert_eq!(entry.entry().flow.blocks.len(), 1);
        assert!(entry.entry().values.is_empty());
    }

    fn flow_entry(values: Vec<Value>, blocks: Vec<Block>) -> Entry {
        flow_entry_with_locals(values, vec![], blocks)
    }

    fn flow_entry_with_locals(values: Vec<Value>, locals: Vec<Type>, blocks: Vec<Block>) -> Entry {
        Entry {
            flow: ControlFlow {
                entry: BlockId(0),
                locals,
                blocks,
            },
            values,
        }
    }

    #[test]
    fn verification_rejects_invalid_control_flow() {
        let error = flow_entry(
            vec![],
            vec![Block {
                instructions: vec![],
                terminator: Terminator::Jump { target: BlockId(1) },
            }],
        )
        .verify()
        .unwrap_err();
        assert!(error.to_string().contains("unknown block"));

        let error = flow_entry(
            vec![],
            vec![Block {
                instructions: vec![],
                terminator: Terminator::Branch {
                    condition: integer(1, Type::Int),
                    then_target: BlockId(0),
                    else_target: BlockId(0),
                },
            }],
        )
        .verify()
        .unwrap_err();
        assert!(error.to_string().contains("branch requires bool"));

        let error = flow_entry(
            vec![],
            vec![Block {
                instructions: vec![],
                terminator: Terminator::Unreachable,
            }],
        )
        .verify()
        .unwrap_err();
        assert!(error.to_string().contains("not terminated"));
    }

    #[test]
    fn verification_rejects_loads_before_definite_initialization() {
        let load = || Value {
            span: None,
            ty: Type::Int,
            kind: ValueKind::Load(LocalId(0)),
        };
        let same_block = flow_entry_with_locals(
            vec![load()],
            vec![Type::Int],
            vec![Block {
                instructions: vec![
                    Instruction::Value(ValueId(0)),
                    Instruction::Store {
                        local: LocalId(0),
                        operand: integer(1, Type::Int),
                    },
                ],
                terminator: Terminator::Exit {
                    status: integer(0, Type::Int),
                },
            }],
        );
        let missing_path = flow_entry_with_locals(
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
                        local: LocalId(0),
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
        for entry in [same_block, missing_path] {
            let error = entry.verify().unwrap_err();
            assert!(error.to_string().contains("uninitialized local 0"));
        }
    }

    #[test]
    fn comparisons_preserve_left_operands_across_short_circuiting_right_operands() {
        for source in [
            "var a = true; var b = true; var c = false; if a == (b && c) {}",
            "var a = true; var b = true; var c = false; var d = true; if (a && b) == (c && d) {}",
            "var x = 1; var b = true; var c = false; if (x == 1) == (b && c) {}",
            "var a = true; var b = true; var c = false; var q = a == (b && c);",
        ] {
            lowered_flow(&format!("fn main() -> void {{ {source} }}"));
        }
    }
}
