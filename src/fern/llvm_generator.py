from __future__ import annotations

from llvmlite import ir

from src.fern.ssa_nodes import SSAValue
from ssa_nodes import SSAPhi, SSAFunction, SSAInstr, SSABlock, SSAName, SSAProgram, SSAOp, SSAConst


class LLVMGenerator:
    def __init__(self) -> None:
        self.module = ir.Module(name="module")
        self.ssa_values: dict[str, ir.Value] = {}
        self.blocks: dict[str, ir.Block] = {}
        self.current_function: ir.Function | None = None
        self.phi_nodes: list[tuple[ir.PhiInstr, SSAPhi]] = []

    def _get_llvm_type(self, type_name: str) -> ir.Type:
        """Convert AST types to LLVM types."""
        type_map = {
            'int': ir.IntType(32),
            'float': ir.FloatType(),
            'bool': ir.IntType(1),
            'void': ir.VoidType(),
        }
        return type_map.get(type_name.lower(), ir.VoidType())

    def generate(self, ssa_program: SSAProgram) -> str:
        """Generate LLVM IR for all functions."""
        # First declare all functions (needed for calls between them)
        for func in ssa_program.functions:
            ret_type = self._get_llvm_type(func.return_type.name)
            param_types = [self._get_llvm_type(t.name) for t in func.param_types]
            func_type = ir.FunctionType(ret_type, param_types)
            ir.Function(self.module, func_type, name=func.name)

        # Now generate the bodies
        for func in ssa_program.functions:
            self._generate_function(func)

        return str(self.module).strip()

    def _generate_function(self, func: SSAFunction) -> None:
        """Generate a single function."""
        # Reset state for new function
        self.ssa_values.clear()
        self.blocks.clear()
        self.phi_nodes.clear()

        # Get the pre-declared function
        llvm_func = self.module.get_global(func.name)
        assert isinstance(llvm_func, ir.Function)
        self.current_function = llvm_func

        # Set parameter names and add to SSA values
        for i, (param, param_name) in enumerate(zip(llvm_func.args, func.params)):
            param.name = param_name
            self.ssa_values[param_name] = param

        # Create all blocks first
        for block in func.blocks:
            self.blocks[block.name] = llvm_func.append_basic_block(block.name)

        # Generate code for each block
        for block in func.blocks:
            self._generate_block(block)

        # Add phi incoming values after all blocks are complete
        self._add_phi_incoming()

    def _generate_block(self, block: SSABlock) -> None:
        """Generate code for a basic block."""
        builder = ir.IRBuilder(self.blocks[block.name])

        # Generate phi nodes first
        for phi in block.phis:
            phi_node = builder.phi(ir.IntType(32), name=str(phi.result))
            self.ssa_values[str(phi.result)] = phi_node
            self.phi_nodes.append((phi_node, phi))

        # Generate instructions
        for instr in block.instructions:
            self._generate_instruction(instr, builder)

    def _add_phi_incoming(self) -> None:
        """Add incoming values to phi nodes after all blocks are generated."""
        for phi_node, phi in self.phi_nodes:
            incoming = []
            for pred_name, value in phi.args.items():
                pred_block = self.blocks[pred_name]
                if isinstance(value, SSAConst):
                    val = ir.Constant(ir.IntType(32), value.value)
                else:
                    val = self.ssa_values[str(value)]
                incoming.append((val, pred_block))
            phi_node.incomings.extend(incoming)

    def _generate_instruction(self, instr: SSAInstr, builder: ir.IRBuilder) -> None:
        """Generate code for a single instruction with dynamic type handling."""

        def get_type(value: SSAValue) -> ir.Type:
            """Determine the type based on operand."""
            if isinstance(value, SSAConst):
                if isinstance(value.value, bool):
                    return ir.IntType(1)
                elif isinstance(value.value, int):
                    return ir.IntType(32)
                elif isinstance(value.value, float):
                    return ir.FloatType()
            return ir.IntType(32)  # Default to int32 if unknown

        if instr.op == SSAOp.ASSIGN:
            value = self._get_operand(instr.args[0], builder)
            value_type = get_type(instr.args[0])

            # Ensure the constant zero matches the type of `value`
            zero_value = ir.Constant(value_type, 0)

            # If `value` is a boolean (i1), we need to extend it to match the type expected by the operation
            if value.type != zero_value.type:
                if value.type == ir.IntType(1) and zero_value.type == ir.IntType(32):
                    value = builder.zext(value, ir.IntType(32))  # Extend boolean to int32 if needed
                else:
                    raise ValueError("Incompatible operand types for ASSIGN")

            result = builder.add(value, zero_value, name=str(instr.result))
            self.ssa_values[str(instr.result)] = result

        elif instr.op == SSAOp.AND:
            left = self._get_operand(instr.args[0], builder)
            right = self._get_operand(instr.args[1], builder)

            # Ensure both operands are of boolean type (i1)
            if left.type != ir.IntType(1):
                left = builder.trunc(left, ir.IntType(1))
            if right.type != ir.IntType(1):
                right = builder.trunc(right, ir.IntType(1))

            result = builder.and_(left, right, name=str(instr.result))
            self.ssa_values[str(instr.result)] = result

        elif instr.op == SSAOp.OR:
            left = self._get_operand(instr.args[0], builder)
            right = self._get_operand(instr.args[1], builder)

            # Ensure both operands are of boolean type (i1)
            if left.type != ir.IntType(1):
                left = builder.trunc(left, ir.IntType(1))
            if right.type != ir.IntType(1):
                right = builder.trunc(right, ir.IntType(1))

            result = builder.or_(left, right, name=str(instr.result))
            self.ssa_values[str(instr.result)] = result

        elif instr.op == SSAOp.CALL:
            func_name = str(instr.args[0])
            callee = self.module.get_global(func_name)
            assert isinstance(callee, ir.Function)
            args = [self._get_operand(arg, builder) for arg in instr.args[1:]]
            result = builder.call(callee, args, name=str(instr.result) if instr.result else "")
            if instr.result:
                self.ssa_values[str(instr.result)] = result

        elif instr.op in (SSAOp.NOT, SSAOp.NEG):
            value = self._get_operand(instr.args[0], builder)
            if instr.op == SSAOp.NOT:
                result = builder.xor(value, ir.Constant(value.type, -1), name=str(instr.result))
            elif instr.op == SSAOp.NEG:
                result = builder.neg(value, name=str(instr.result))
            else :
                raise NotImplementedError(f"Operation not implemented: {instr.op}")
            self.ssa_values[str(instr.result)] = result


        elif instr.op in (SSAOp.ADD, SSAOp.SUB, SSAOp.MUL, SSAOp.DIV):
            left = self._get_operand(instr.args[0], builder)
            right = self._get_operand(instr.args[1], builder)
            op_type = get_type(instr.args[0])

            if op_type == ir.FloatType():
                if instr.op == SSAOp.ADD:
                    result = builder.fadd(left, right, name=str(instr.result))
                elif instr.op == SSAOp.SUB:
                    result = builder.fsub(left, right, name=str(instr.result))
                elif instr.op == SSAOp.MUL:
                    result = builder.fmul(left, right, name=str(instr.result))
                elif instr.op == SSAOp.DIV:
                    result = builder.fdiv(left, right, name=str(instr.result))
            else:
                if instr.op == SSAOp.ADD:
                    result = builder.add(left, right, name=str(instr.result))
                elif instr.op == SSAOp.SUB:
                    result = builder.sub(left, right, name=str(instr.result))
                elif instr.op == SSAOp.MUL:
                    result = builder.mul(left, right, name=str(instr.result))
                elif instr.op == SSAOp.DIV:
                    result = builder.sdiv(left, right, name=str(instr.result))
            if instr.result:
                self.ssa_values[str(instr.result)] = result

        elif instr.op in (SSAOp.LT, SSAOp.LTE, SSAOp.GT, SSAOp.GTE, SSAOp.EQ, SSAOp.NEQ):
            left = self._get_operand(instr.args[0], builder)
            right = self._get_operand(instr.args[1], builder)
            op_type = get_type(instr.args[0])

            if instr.op == SSAOp.LT:
                result = builder.fcmp_ordered('<', left, right, name=str(instr.result)) if op_type == ir.FloatType() else builder.icmp_signed('<', left, right, name=str(instr.result))
            if instr.op == SSAOp.LTE:
                result = builder.fcmp_ordered('<=', left, right, name=str(instr.result)) if op_type == ir.FloatType() else builder.icmp_signed('<=', left, right, name=str(instr.result))
            elif instr.op == SSAOp.GT:
                result = builder.fcmp_ordered('>', left, right, name=str(instr.result)) if op_type == ir.FloatType() else builder.icmp_signed('>', left, right, name=str(instr.result))
            elif instr.op == SSAOp.GTE:
                result = builder.fcmp_ordered('>=', left, right, name=str(instr.result)) if op_type == ir.FloatType() else builder.icmp_signed('>=', left, right, name=str(instr.result))
            elif instr.op == SSAOp.EQ:
                result = builder.fcmp_ordered('==', left, right, name=str(instr.result)) if op_type == ir.FloatType() else builder.icmp_signed('==', left, right, name=str(instr.result))
            elif instr.op == SSAOp.NEQ:
                result = builder.fcmp_ordered('!=', left, right, name=str(instr.result)) if op_type == ir.FloatType() else builder.icmp_signed('!=', left, right, name=str(instr.result))

            if instr.result:
                self.ssa_values[str(instr.result)] = result

        elif instr.op == SSAOp.RET:
            if instr.args:
                value = self._get_operand(instr.args[0], builder)
                builder.ret(value)
            else:
                builder.ret_void()

        elif instr.op == SSAOp.BR:
            # Get the condition and ensure it is of type i1
            cond = self._get_operand(instr.args[0], builder)
            if cond.type != ir.IntType(1):
                cond = builder.icmp_signed('!=', cond,
                                           ir.Constant(cond.type, 0))  # Convert non-i1 condition to boolean (i1)

            # Define true and false branches
            true_block = self.blocks[str(instr.args[1])]
            false_block = self.blocks[str(instr.args[2])]
            builder.cbranch(cond, true_block, false_block)

        elif instr.op == SSAOp.JUMP:
            target_block = self.blocks[str(instr.args[0])]
            builder.branch(target_block)

        else:
            raise NotImplementedError(f"Operation not implemented: {instr.op}")

    def _get_operand(self, arg: SSAName | str | SSAConst, builder: ir.IRBuilder) -> ir.Value:
        """Get LLVM value for an SSA operand."""
        if isinstance(arg, SSAConst):
            if isinstance(arg.value, bool):
                return ir.Constant(ir.IntType(1), int(arg.value))
            elif isinstance(arg.value, int):
                return ir.Constant(ir.IntType(32), arg.value)
            elif isinstance(arg.value, float):
                return ir.Constant(ir.FloatType(), arg.value)
            else:
                raise ValueError(f"Unsupported constant type: {type(arg.value)}")
        elif isinstance(arg, (SSAName, str)):
            name = str(arg)
            if name in self.ssa_values:
                return self.ssa_values[name]
            raise ValueError(f"Unknown SSA value: {name}")
        else:
            raise ValueError(f"Unknown operand type: {type(arg)}")
