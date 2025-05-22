from __future__ import annotations

from llvmlite import ir

from src.fern.ssa_nodes import SSAPhi, SSAFunction, SSAInstr, SSABlock, SSAName, SSAProgram, SSAOp, SSAConst, SSAValue


class LLVMGenerator:
    def __init__(self) -> None:
        self.module = ir.Module(name="module")
        self.ssa_values: dict[str, ir.Value] = {} # Maps SSA name string to LLVM value
        self.blocks: dict[str, ir.Block] = {} # Maps block name to LLVM block
        self.current_function: ir.Function | None = None
        # Store tuples of (LLVM Phi Instr, Original SSAPhi node)
        self.phi_nodes: list[tuple[ir.PhiInstr, SSAPhi]] = []

    def _get_llvm_type(self, type_name: str) -> ir.Type:
        """Convert simple type names to LLVM types."""
        type_map = {
            'int': ir.IntType(32),
            'float': ir.FloatType(), # Assuming standard float
            'bool': ir.IntType(1),
            'void': ir.VoidType(),
        }
        # Handle potential ASTType object if passed
        actual_type_name = getattr(type_name, 'name', str(type_name)).lower()
        return type_map.get(actual_type_name, ir.VoidType())

    def generate(self, ssa_program: SSAProgram) -> str:
        """Generate LLVM IR for the entire SSA program."""
        # Declare all functions first
        for func in ssa_program.functions:
            ret_type = self._get_llvm_type(func.return_type) # Pass ASTType directly
            param_types = [self._get_llvm_type(t) for t in func.param_types] # Pass ASTType directly
            func_type = ir.FunctionType(ret_type, param_types)
            ir.Function(self.module, func_type, name=func.name)

        # Generate bodies
        for func in ssa_program.functions:
            self._generate_function(func)

        return str(self.module).strip()

    def _generate_function(self, func: SSAFunction) -> None:
        """Generate LLVM IR for a single SSA function."""
        self.ssa_values.clear()
        self.blocks.clear()
        self.phi_nodes.clear()

        llvm_func = self.module.get_global(func.name)
        assert isinstance(llvm_func, ir.Function)
        self.current_function = llvm_func

        # Map parameters (version 0) to LLVM arguments
        for i, (param, param_name) in enumerate(zip(llvm_func.args, func.params)):
            param.name = param_name
            ssa_param_name_v0 = SSAName(base_name=param_name, version=0)
            self.ssa_values[str(ssa_param_name_v0)] = param

        # Create LLVM blocks
        for block in func.blocks:
            self.blocks[block.name] = llvm_func.append_basic_block(block.name)

        # Generate code for each block
        for block in func.blocks:
            self._generate_block(block)

        # Add phi incoming values now that all values are defined
        self._add_phi_incoming()

    def _generate_block(self, block: SSABlock) -> None:
        """Generate LLVM IR for a single SSA basic block."""
        builder = ir.IRBuilder(self.blocks[block.name])

        # Create LLVM phi nodes (without incoming values yet)
        for phi in block.phis:
            if not isinstance(phi.result, SSAName):
                 print(f"Warning: Phi node in {block.name} has non-SSAName result {phi.result}. Skipping.")
                 continue # Cannot determine type or name

            # TODO: Implement proper type analysis for phi nodes
            # Placeholder: Assume i32 for now. This needs to be derived from
            # the types of incoming values or variable declarations.
            phi_type = ir.IntType(32)
            phi_name = str(phi.result)

            # Check if phi result name already exists (shouldn't for phi definition)
            if phi_name in self.ssa_values:
                 print(f"Warning: SSA value '{phi_name}' already exists before phi node creation in {block.name}.")
                 # Potentially skip or handle error

            phi_node = builder.phi(phi_type, name=phi_name)
            self.ssa_values[phi_name] = phi_node
            # Store the LLVM phi and the original SSA phi together
            self.phi_nodes.append((phi_node, phi))

        # Generate instructions
        for instr in block.instructions:
            self._generate_instruction(instr, builder)

    def _add_phi_incoming(self) -> None:
        """Add incoming values to LLVM phi nodes. Handles undef values."""
        for phi_node, ssa_phi in self.phi_nodes:
            # Ensure ssa_phi.args is a dictionary
            if not isinstance(ssa_phi.args, dict):
                print(f"Warning: SSAPhi arguments for {phi_node.name} is not a dict: {type(ssa_phi.args)}. Skipping.")
                continue

            for pred_name, ssa_value in ssa_phi.args.items():
                pred_block = self.blocks.get(pred_name)
                if pred_block is None:
                    print(f"Warning: Predecessor block '{pred_name}' not found for phi {phi_node.name}. Skipping entry.")
                    continue

                llvm_value: ir.Value | None = None
                if ssa_value is None:
                    # *** CORRECTED: Use ir.Constant(type, None) for undef ***
                    llvm_value = ir.Constant(phi_node.type, None)
                elif isinstance(ssa_value, SSAConst):
                    # Handle constants (ensure type matches phi)
                    const_val = ssa_value.value
                    # Determine LLVM type from Python type
                    if isinstance(const_val, bool): const_type = ir.IntType(1)
                    elif isinstance(const_val, int): const_type = ir.IntType(32)
                    elif isinstance(const_val, float): const_type = ir.FloatType()
                    else:
                        print(f"Warning: Unsupported constant type {type(const_val)} in phi {phi_node.name}. Using undef.")
                        # *** CORRECTED: Use ir.Constant(type, None) for undef ***
                        llvm_value = ir.Constant(phi_node.type, None)
                        phi_node.add_incoming(llvm_value, pred_block)
                        continue # Skip rest of processing for this entry

                    llvm_const = ir.Constant(const_type, const_val)

                    # Handle type mismatch between constant and phi node
                    if llvm_const.type != phi_node.type:
                         print(f"Warning: Type mismatch for constant in phi {phi_node.name}. Got {llvm_const.type}, expected {phi_node.type}. Attempting cast/default.")
                         # Example: Cast i1 constant to i32 phi
                         if phi_node.type == ir.IntType(32) and llvm_const.type == ir.IntType(1):
                             llvm_value = ir.Constant(phi_node.type, int(const_val)) # Use integer value of bool
                         else:
                             # *** CORRECTED: Use ir.Constant(type, None) for undef ***
                             llvm_value = ir.Constant(phi_node.type, None) # Fallback to undef
                    else:
                         llvm_value = llvm_const

                elif isinstance(ssa_value, (SSAName, str)):
                    # Handle SSA variables
                    val_name = str(ssa_value)
                    llvm_value_lookup = self.ssa_values.get(val_name)
                    if llvm_value_lookup is None:
                        print(f"Warning: SSA value '{val_name}' not found for phi {phi_node.name} from pred {pred_name}. Using undef.")
                        # *** CORRECTED: Use ir.Constant(type, None) for undef ***
                        llvm_value = ir.Constant(phi_node.type, None)
                    elif llvm_value_lookup.type != phi_node.type:
                        # Handle type mismatches if necessary
                        print(f"Warning: Type mismatch for variable '{val_name}' in phi {phi_node.name}. Got {llvm_value_lookup.type}, expected {phi_node.type}. Using undef.")
                        # *** CORRECTED: Use ir.Constant(type, None) for undef ***
                        llvm_value = ir.Constant(phi_node.type, None) # Fallback
                    else:
                         llvm_value = llvm_value_lookup # Use the found value
                else:
                    print(f"Warning: Unexpected value type {type(ssa_value)} in phi args for {phi_node.name}. Using undef.")
                    # *** CORRECTED: Use ir.Constant(type, None) for undef ***
                    llvm_value = ir.Constant(phi_node.type, None)

                # Add the determined incoming value and block to the LLVM phi node
                phi_node.add_incoming(llvm_value, pred_block)


    def _generate_instruction(self, instr: SSAInstr, builder: ir.IRBuilder) -> None:
        """Generate LLVM IR for a single SSA instruction."""
        # Helper to get operand, assuming _get_operand handles constants and SSA names
        def get_op(arg_index: int) -> ir.Value:
            # Check bounds
            if arg_index >= len(instr.args):
                 raise IndexError(f"Argument index {arg_index} out of bounds for instruction {instr}")
            return self._get_operand(instr.args[arg_index], builder)

        # Helper to store result
        def store_result(llvm_val: ir.Value):
            if instr.result:
                 res_name = str(instr.result)
                 # Optional: Check if name already exists, might indicate SSA error
                 # if res_name in self.ssa_values:
                 #     print(f"Warning: Overwriting SSA value '{res_name}' in block {builder.block.name}")
                 self.ssa_values[res_name] = llvm_val

        op = instr.op

        # --- Assignment (Handled by mapping in SSA pass, usually no LLVM op needed) ---
        if op == SSAOp.ASSIGN:
             value_to_assign = get_op(0)
             store_result(value_to_assign) # Just map the name

        # --- Binary Arithmetic ---
        elif op in (SSAOp.ADD, SSAOp.SUB, SSAOp.MUL, SSAOp.DIV):
            left, right = get_op(0), get_op(1)
            # Basic type check (assuming operands have same type)
            is_float = isinstance(left.type, ir.FloatType)
            if op == SSAOp.ADD: store_result(builder.fadd(left, right) if is_float else builder.add(left, right, name=str(instr.result) if instr.result else ""))
            elif op == SSAOp.SUB: store_result(builder.fsub(left, right) if is_float else builder.sub(left, right, name=str(instr.result) if instr.result else ""))
            elif op == SSAOp.MUL: store_result(builder.fmul(left, right) if is_float else builder.mul(left, right, name=str(instr.result) if instr.result else ""))
            elif op == SSAOp.DIV: store_result(builder.fdiv(left, right) if is_float else builder.sdiv(left, right, name=str(instr.result) if instr.result else "")) # Signed integer div

        # --- Binary Comparison ---
        elif op in (SSAOp.LT, SSAOp.LTE, SSAOp.GT, SSAOp.GTE, SSAOp.EQ, SSAOp.NEQ):
            left, right = get_op(0), get_op(1)
            is_float = isinstance(left.type, ir.FloatType)
            op_map_float = {SSAOp.LT: '<', SSAOp.LTE: '<=', SSAOp.GT: '>', SSAOp.GTE: '>=', SSAOp.EQ: '==', SSAOp.NEQ: '!='}
            op_map_int = {SSAOp.LT: '<', SSAOp.LTE: '<=', SSAOp.GT: '>', SSAOp.GTE: '>=', SSAOp.EQ: '==', SSAOp.NEQ: '!='} # Use signed comparison for int
            cmp_op = op_map_float[op] if is_float else op_map_int[op]
            cmp_func = builder.fcmp_ordered if is_float else builder.icmp_signed
            store_result(cmp_func(cmp_op, left, right, name=str(instr.result) if instr.result else ""))

        # --- Unary Operations ---
        elif op == SSAOp.NEG:
            operand = get_op(0)
            is_float = isinstance(operand.type, ir.FloatType)
            store_result(builder.fneg(operand, name=str(instr.result) if instr.result else "") if is_float else builder.neg(operand, name=str(instr.result) if instr.result else ""))
        elif op == SSAOp.NOT: # Logical not (usually on i1)
             operand = get_op(0)
             if operand.type == ir.IntType(1):
                 # XOR with true (1) flips the boolean
                 store_result(builder.xor(operand, ir.Constant(ir.IntType(1), 1), name=str(instr.result) if instr.result else ""))
             else:
                 # Handle NOT on non-boolean if necessary (e.g., bitwise not)
                 # For now, assume it's logical not on i1
                 print(f"Warning: Applying logical NOT to non-boolean type {operand.type}. Result may be unexpected.")
                 # Example: treat non-zero as true, zero as false -> result = icmp eq operand, 0
                 zero = ir.Constant(operand.type, 0)
                 store_result(builder.icmp_eq(operand, zero, name=str(instr.result) if instr.result else ""))


        # --- Control Flow ---
        elif op == SSAOp.RET:
            if instr.args: builder.ret(get_op(0))
            else: builder.ret_void()
        elif op == SSAOp.BR: # Conditional Branch
            cond = get_op(0)
            # Ensure condition is i1
            if cond.type != ir.IntType(1):
                print(f"Warning: Conditional branch on non-boolean type {cond.type}. Comparing with zero.")
                zero = ir.Constant(cond.type, 0)
                cond = builder.icmp_ne(cond, zero) # Treat non-zero as true
            true_dest = self.blocks[str(instr.args[1])]
            false_dest = self.blocks[str(instr.args[2])]
            builder.cbranch(cond, true_dest, false_dest)
        elif op == SSAOp.JUMP: # Unconditional Branch
            target = self.blocks[str(instr.args[0])]
            builder.branch(target)

        # --- Function Call ---
        elif op == SSAOp.CALL:
            # First arg is function name (string or SSAName referencing function)
            func_ref = instr.args[0]
            if isinstance(func_ref, SSAName): func_name = func_ref.base_name
            elif isinstance(func_ref, str): func_name = func_ref
            else: raise TypeError(f"Unexpected function reference type: {type(func_ref)}")

            callee = self.module.get_global(func_name)
            if not isinstance(callee, ir.Function):
                 raise LookupError(f"Function '{func_name}' not found in module.")

            call_args = [self._get_operand(arg, builder) for arg in instr.args[1:]]
            result = builder.call(callee, call_args, name=str(instr.result) if instr.result else "")
            store_result(result) # Store if call is not void

        else:
            raise NotImplementedError(f"SSA Operation not implemented in LLVM Generator: {op}")


    def _get_operand(self, arg: SSAValue, builder: ir.IRBuilder) -> ir.Value:
        """Get LLVM value for an SSA operand (Constant or Name)."""
        if isinstance(arg, SSAConst):
            val = arg.value
            if isinstance(val, bool): return ir.Constant(ir.IntType(1), int(val))
            elif isinstance(val, int): return ir.Constant(ir.IntType(32), val)
            elif isinstance(val, float): return ir.Constant(ir.FloatType(), val)
            else: raise ValueError(f"Unsupported constant type: {type(val)}")
        elif isinstance(arg, (SSAName, str)): # Allow plain strings for globals/funcs
            name = str(arg)
            if name in self.ssa_values:
                return self.ssa_values[name]
            # Check if it's a global function name
            elif glob := self.module.globals.get(name):
                 if isinstance(glob, ir.Function): return glob
                 else: raise ValueError(f"Global '{name}' is not a function.")
            else:
                 raise ValueError(f"Unknown SSA value or global function: {name}")
        elif arg is None: # Handle None placeholder from SSA phi generation
             # This case should ideally be handled in _add_phi_incoming
             # If reached here, it's likely an error elsewhere. Return undef i32.
             print("Warning: _get_operand received None, returning undef i32.")
             # *** CORRECTED: Use ir.Constant(type, None) for undef ***
             # Need context for type, assume IntType(32) as placeholder
             return ir.Constant(ir.IntType(32), None)
        else:
            raise TypeError(f"Unknown operand type: {type(arg)}")

