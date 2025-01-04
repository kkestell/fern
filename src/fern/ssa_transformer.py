from __future__ import annotations

from ssa_nodes import (
    SSAFunction, SSABlock, SSAPhi, SSAInstr, SSAName, SSAProgram, SSAOp, SSAConst
)
from cfg_nodes import CFG
from tac_nodes import TACConst, TACProgram, TACBlock, TACInstr, TACTemp, TACLabel


class SSATransformer:
    def __init__(self) -> None:
        """Initialize transformation state, including variable versions and definition maps."""
        self._var_versions: dict[str, int] = {}  # Track latest version number for each variable
        self._var_defs: dict[tuple[str, str], SSAName | str] = {}  # Map (block name, var) to SSA name
        self._current_block: SSABlock | None = None  # Tracks the SSA block currently processed
        self._program: TACProgram | None = None  # TAC program being transformed
        self._current_proc_params: set[str] = set()  # Set of parameter names for the current function
        self._cfg: CFG | None = None  # CFG of the function currently being transformed

    def transform(self, program: TACProgram, cfgs: list[CFG]) -> SSAProgram:
        """
        Convert a TAC program and its control flow graphs (CFGs) into SSA form.

        :param program: The TAC program containing multiple procedures.
        :param cfgs: Control flow graphs for each procedure.
        :return: The resulting SSA program.
        """
        self._program = program
        ssa_funcs = {cfg.proc_name: self._transform_function(cfg) for cfg in cfgs}
        return SSAProgram(functions=list(ssa_funcs.values()))

    def _new_name(self, base: str, block_name: str) -> SSAName | str:
        """
        Create a new SSA version of a variable name within the given block.

        Avoids renaming if the name is a function, temporary, or parameter.

        :param base: Base variable name.
        :param block_name: The name of the block where the variable is defined.
        :return: A new SSAName or the original name if renaming is unnecessary.
        """
        if self._is_function_name(base) or base == "print" or base in self._current_proc_params or self._is_temporary(
                base):
            return base

        # Increment and assign a new version for this variable
        version = self._var_versions.get(base, -1) + 1
        self._var_versions[base] = version
        name = SSAName(base_name=base, version=version)
        self._var_defs[(block_name, base)] = name
        return name

    def _transform_function(self, cfg: CFG) -> SSAFunction:
        """
        Convert a function's TAC CFG into SSA form.

        :param cfg: Control flow graph for the function to transform.
        :return: SSA representation of the function.
        """
        if self._program is None:
            raise Exception("No program available")

        # Locate the TAC procedure matching the CFG function name
        proc = next((p for p in self._program.procedures if p.name == cfg.proc_name), None)
        if proc is None:
            raise Exception(f"No procedure found for {cfg.proc_name}")

        # Reset state for new function
        self._cfg = cfg
        self._var_versions.clear()
        self._var_defs.clear()
        self._current_proc_params = {p.name for p in proc.params}

        # Create SSA blocks and link their predecessors and successors
        ssa_blocks = {node.block.name: SSABlock(name=node.block.name) for node in cfg.nodes}
        for node in cfg.nodes:
            ssa_block = ssa_blocks[node.block.name]
            ssa_block.predecessors = [ssa_blocks[pred.block.name] for pred in node.predecessors]
            ssa_block.successors = [ssa_blocks[succ.block.name] for succ in node.successors]

        # First pass: transform blocks
        for node in cfg.nodes:
            self._current_block = ssa_blocks[node.block.name]
            self._transform_block(node.block, self._current_block)

        # Add phi nodes where multiple definitions meet
        self._add_phi_nodes(ssa_blocks)

        # Update variable uses to use phi results
        self._update_variable_uses(ssa_blocks)

        return SSAFunction(
            name=cfg.proc_name,
            params=[p.name for p in proc.params],
            param_types=[p.param_type for p in proc.params],
            return_type=proc.return_type,
            entry_block=ssa_blocks[cfg.entry.block.name],
            blocks=list(ssa_blocks.values())
        )

    def _update_variable_uses(self, blocks: dict[str, SSABlock]) -> None:
        """
        Update variable uses to refer to the correct SSA versions.

        :param blocks: Dictionary of SSA blocks
        """
        latest_defs: dict[str, SSAName | str] = {}  # Track latest definition of each variable

        # Process blocks in dominator tree order
        if self._cfg is None:
            raise Exception("No CFG available")

        for node in self._cfg.nodes:
            block = blocks[node.block.name]

            # Update from phi nodes
            for phi in block.phis:
                if isinstance(phi.result, SSAName):
                    latest_defs[phi.result.base_name] = phi.result

            # Update instructions to use latest definitions
            for instr in block.instructions:
                # Update arguments to use latest definitions
                new_args = []
                for arg in instr.args:
                    if isinstance(arg, SSAName):
                        base_name = arg.base_name
                        if base_name in latest_defs:
                            new_args.append(latest_defs[base_name])
                        else:
                            new_args.append(arg)
                    else:
                        new_args.append(arg)
                instr.args = new_args

                # Update latest definition if this instruction defines a variable
                if instr.result and isinstance(instr.result, SSAName):
                    latest_defs[instr.result.base_name] = instr.result

    def _propagate_versions(self, blocks: dict[str, SSABlock]) -> None:
        """
        Propagate the latest version of each variable through the CFG.

        :param blocks: Dictionary of SSA blocks
        """
        # Track the latest version of each variable
        latest_versions: dict[str, SSAName | str] = {}

        # Process blocks in dominator tree order
        if self._cfg is None:
            raise Exception("No CFG available")

        for node in self._cfg.nodes:
            block = blocks[node.block.name]

            # Update with versions from predecessors
            for pred in block.predecessors:
                for (pred_name, var), defn in self._var_defs.items():
                    if pred_name == pred.name:
                        if var not in latest_versions:
                            latest_versions[var] = defn
                        else:
                            current = latest_versions[var]
                            if (isinstance(defn, SSAName) and
                                    isinstance(current, SSAName) and
                                    defn.version > current.version):
                                latest_versions[var] = defn
                            elif isinstance(defn, str):
                                latest_versions[var] = defn

            # Process phis and instructions to update latest versions
            for phi in block.phis:
                if isinstance(phi.result, SSAName):
                    latest_versions[phi.result.base_name] = phi.result

            for instr in block.instructions:
                if instr.result and isinstance(instr.result, SSAName):
                    latest_versions[instr.result.base_name] = instr.result

            # Update var_defs with latest versions
            for var, version in latest_versions.items():
                self._var_defs[(block.name, var)] = version

    def _get_reaching_def(self, var: str, block_name: str) -> SSAName | str:
        """
        Determine the reaching definition of a variable within a block.

        :param var: Variable name to resolve.
        :param block_name: Block name for the definition scope.
        :return: The SSAName or original name of the reaching definition.
        """
        if self._current_block is None:
            raise Exception("No current block")

        if self._is_function_name(var) or var == "print" or var in self._current_proc_params or self._is_temporary(var):
            return var

        # First check current block
        if (block_name, var) in self._var_defs:
            return self._var_defs[(block_name, var)]

        # Find the most recent version from any predecessor
        latest_version: SSAName | str | None = None
        for pred in self._current_block.predecessors:
            if (pred.name, var) in self._var_defs:
                pred_def = self._var_defs[(pred.name, var)]
                if isinstance(pred_def, SSAName):
                    if latest_version is None or (
                            isinstance(latest_version, SSAName) and
                            pred_def.version > latest_version.version
                    ):
                        latest_version = pred_def

        if latest_version is not None:
            self._var_defs[(block_name, var)] = latest_version
            return latest_version

        # If no definition exists, create new version
        version = 0
        name = SSAName(base_name=var, version=version)
        self._var_versions[var] = version
        self._var_defs[(block_name, var)] = name
        return name

    def _transform_block(self, tac_block: TACBlock, ssa_block: SSABlock) -> None:
        """
        Convert a TAC block's instructions to SSA form, renaming variables as necessary.

        :param tac_block: Original TAC block to transform.
        :param ssa_block: SSA block to populate with transformed instructions.
        """
        for instr in tac_block.instructions:
            ssa_instr = self._transform_instruction(instr)
            if ssa_instr:
                ssa_block.instructions.append(ssa_instr)

    def _transform_instruction(self, instr: TACInstr) -> SSAInstr | None:
        """Transform a TAC instruction to SSA form by renaming arguments and result."""
        if self._current_block is None:
            raise Exception("No current block")

        # Rename arguments to current SSA definitions
        new_args = []
        for arg in instr.args:
            if isinstance(arg, TACTemp):
                new_args.append(self._get_reaching_def(arg.name, self._current_block.name))
            elif isinstance(arg, TACConst):
                # Convert TACConst to SSAConst
                new_args.append(SSAConst(value=arg.value))
            elif isinstance(arg, TACLabel):
                new_args.append(arg.name)
            else:
                raise Exception(f"Unexpected argument type: {type(arg)}")

        # Generate a new SSA name for the result, if applicable
        result = self._new_name(instr.result.name, self._current_block.name) if instr.result else None

        # Convert TACOp to SSAOp
        ssa_op = SSAOp[instr.op.name]

        return SSAInstr(op=ssa_op, result=result, args=new_args)

    def _add_phi_nodes(self, blocks: dict[str, SSABlock]) -> None:
        """
        Insert phi nodes in SSA blocks where a variable has multiple incoming definitions.

        :param blocks: SSA blocks dictionary.
        """
        for block in blocks.values():
            if len(block.predecessors) > 1:
                # Identify variables needing phi nodes due to multiple incoming definitions
                vars_needing_phi = {
                    var for pred in block.predecessors
                    for (pred_name, var), _ in self._var_defs.items()
                    if pred_name == pred.name and not self._is_temporary(var)
                }

                # Create phi nodes for each variable needing unified values
                for var in vars_needing_phi:
                    incoming = {
                        pred.name: self._var_defs[(pred.name, var)]
                        for pred in block.predecessors if (pred.name, var) in self._var_defs
                    }

                    # Insert phi node only if incoming values differ
                    if len(set(str(v) for v in incoming.values())) > 1:
                        result = self._new_name(var, block.name)
                        phi = SSAPhi(result=result, args=incoming)
                        block.phis.append(phi)
                        self._var_defs[(block.name, var)] = result

    @staticmethod
    def _is_temporary(name: str) -> bool:
        """Check if a variable is a temporary."""
        return name.startswith('t')

    def _is_function_name(self, name: str) -> bool:
        """Check if a name corresponds to a function in the TAC program."""
        if self._program is None:
            raise Exception("No program available")
        return any(proc.name == name for proc in self._program.procedures)