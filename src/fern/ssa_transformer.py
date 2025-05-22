from __future__ import annotations

import collections

from src.fern.cfg_nodes import CFG
from src.fern.ssa_nodes import SSAFunction, SSABlock, SSAPhi, SSAInstr, SSAName, SSAProgram, SSAOp, SSAConst
from src.fern.tac_nodes import TACConst, TACProgram, TACTemp, TACLabel, TACProc


class SSATransformer:
    """
    Transforms a TAC program into Static Single Assignment (SSA) form
    using the algorithm by Cytron et al. (dominance frontiers).
    (Includes fix for incomplete Phi node arguments)
    """

    def __init__(self) -> None:
        # State per function transformation
        self._program: TACProgram | None = None
        self._cfg: CFG | None = None
        self._ssa_blocks: dict[str, SSABlock] = {}
        self._var_defs_blocks: dict[str, set[str]] = collections.defaultdict(set) # var -> set of block names where it's defined
        self._globals: set[str] = set() # Names of global variables/functions

        # Renaming state
        self._counters: dict[str, int] = collections.defaultdict(int) # base_name -> current_version
        self._stacks: dict[str, list[int]] = collections.defaultdict(list) # base_name -> stack of versions

        # Dominance information (to be computed)
        self._dom: dict[str, set[str]] | None = None # block_name -> set of dominator block names
        self._idom: dict[str, str | None] | None = None # block_name -> immediate dominator block name
        self._dom_tree_children: dict[str, list[str]] | None = None # block_name -> list of children in dominator tree
        self._df: dict[str, set[str]] | None = None # block_name -> dominance frontier set

    def transform(self, program: TACProgram, cfgs: list[CFG]) -> SSAProgram:
        """
        Convert a TAC program and its CFGs into SSA form.

        :param program: The TAC program.
        :param cfgs: List of Control Flow Graphs, one per procedure.
        :return: The SSA program.
        """
        self._program = program
        # Identify global names (functions)
        self._globals = {proc.name for proc in program.procedures}

        ssa_funcs = []
        for cfg in cfgs:
            # Find the corresponding TACProc
            proc = next((p for p in program.procedures if p.name == cfg.proc_name), None)
            if proc is None:
                raise ValueError(f"Procedure {cfg.proc_name} not found in TAC program")
            ssa_funcs.append(self._transform_function(proc, cfg))

        return SSAProgram(functions=ssa_funcs)

    # --------------------------------------------------------------------------
    # Dominance Calculation Implementation (Assumed Correct)
    # --------------------------------------------------------------------------
    def _compute_dominators(self, cfg: CFG) -> dict[str, set[str]]:
        """Computes the dominator sets for each block in the CFG (using iterative dataflow)."""
        if not cfg.entry: return {}
        all_nodes = {node.block.name for node in cfg.nodes}
        entry_name = cfg.entry.block.name
        dom: dict[str, set[str]] = {entry_name: {entry_name}}
        for node_name in all_nodes:
            if node_name != entry_name: dom[node_name] = all_nodes.copy()

        changed = True
        while changed:
            changed = False
            for node in cfg.nodes: # Consider reverse post-order for efficiency
                node_name = node.block.name
                if node_name == entry_name: continue
                pred_dom_intersection = all_nodes.copy()
                if node.predecessors:
                    first_pred = True
                    for pred_node in node.predecessors:
                        pred_name = pred_node.block.name
                        current_pred_dom = dom.get(pred_name, all_nodes) # Handle potential processing order issues
                        if first_pred:
                            pred_dom_intersection = current_pred_dom.copy()
                            first_pred = False
                        else:
                            pred_dom_intersection.intersection_update(current_pred_dom)
                else:
                    pred_dom_intersection = set() # Unreachable?

                new_dom = {node_name}.union(pred_dom_intersection)
                if new_dom != dom[node_name]:
                    dom[node_name] = new_dom
                    changed = True
        return dom

    def _compute_dominator_tree(self, cfg: CFG, dom: dict[str, set[str]]) -> tuple[dict[str, str | None], dict[str, list[str]]]:
        """Computes the immediate dominator (idom) and dominator tree children."""
        if not cfg.entry: return {}, {}
        entry_name = cfg.entry.block.name
        all_block_names = {node.block.name for node in cfg.nodes}
        idom: dict[str, str | None] = {name: None for name in all_block_names}
        children: dict[str, list[str]] = collections.defaultdict(list)
        for node_name in all_block_names:
            if node_name == entry_name: continue
            proper_dominators = dom.get(node_name, {node_name}) - {node_name}
            if not proper_dominators: continue
            # Find idom: the proper dominator 'd' dominated by all other proper dominators 's'.
            candidate_idom = None
            for d in proper_dominators:
                is_idom = True
                for s in proper_dominators:
                    if d == s: continue
                    if d not in dom.get(s, set()): is_idom = False; break
                if is_idom: candidate_idom = d; break
            idom[node_name] = candidate_idom
            if candidate_idom: children[candidate_idom].append(node_name)
        return idom, dict(children)

    def _compute_dominance_frontiers(self, cfg: CFG, dom: dict[str, set[str]], idom: dict[str, str | None]) -> dict[str, set[str]]:
        """Computes the dominance frontier using the direct definition."""
        if not cfg.entry or idom is None: return {}
        df_direct: dict[str, set[str]] = collections.defaultdict(set)
        node_map = {node.block.name: node for node in cfg.nodes}
        all_block_names = set(node_map.keys())

        for n_name in all_block_names:
            n_node = node_map.get(n_name)
            if n_node is None: continue
            for y_name in all_block_names:
                 y_node = node_map.get(y_name)
                 if y_node is None: continue
                 for p_node in y_node.predecessors:
                     p_name = p_node.block.name
                     if n_name in dom.get(p_name, set()): # n dom p
                         n_dominates_y = n_name in dom.get(y_name, set())
                         if not (n_dominates_y and n_name != y_name): # n does not strictly dominate y
                             df_direct[n_name].add(y_name)
                             break # y is in DF(n)
        return dict(df_direct)

    # --------------------------------------------------------------------------
    # SSA Transformation Steps
    # --------------------------------------------------------------------------

    def _transform_function(self, proc: TACProc, cfg: CFG) -> SSAFunction:
        """Transforms a single function/procedure into SSA form."""
        self._cfg = cfg
        self._ssa_blocks = {}
        self._var_defs_blocks.clear()
        self._counters.clear()
        self._stacks.clear()

        # 1. Compute Dominance Information
        self._dom = self._compute_dominators(cfg)
        self._idom, self._dom_tree_children = self._compute_dominator_tree(cfg, self._dom)
        self._df = self._compute_dominance_frontiers(cfg, self._dom, self._idom)

        # Create basic SSA blocks structure
        for node in cfg.nodes:
            ssa_block = SSABlock(name=node.block.name)
            self._ssa_blocks[node.block.name] = ssa_block
            for tac_instr in node.block.instructions:
                 try: ssa_op = SSAOp[tac_instr.op.name]
                 except KeyError: raise ValueError(f"Unsupported TAC op: {tac_instr.op.name}")
                 ssa_instr = SSAInstr(op=ssa_op, result=None, args=[])
                 ssa_instr.original_tac = tac_instr # type: ignore[attr-defined]
                 ssa_block.instructions.append(ssa_instr)

        # Link predecessors/successors
        for node in cfg.nodes:
            ssa_block = self._ssa_blocks[node.block.name]
            ssa_block.predecessors = [self._ssa_blocks[pred.block.name] for pred in node.predecessors]
            ssa_block.successors = [self._ssa_blocks[succ.block.name] for succ in node.successors]

        # 2. Identify Original Variable Definitions
        self._identify_variable_defs(proc)

        # 3. Insert Phi Functions
        self._insert_phi_functions()

        # 4. Rename Variables
        param_names = {p.name for p in proc.params}
        for param_name in param_names:
             if param_name in self._var_defs_blocks:
                 self._generate_new_name(param_name) # Create version 0

        if cfg.entry:
            self._rename_variables(cfg.entry.block.name, param_names)
        else: print(f"Warning: CFG for {proc.name} has no entry, skipping rename.")

        # Clean up temporary attributes
        for block in self._ssa_blocks.values():
            for instr in block.instructions:
                if hasattr(instr, 'original_tac'): del instr.original_tac # type: ignore[attr-defined]
            for phi in block.phis:
                 if hasattr(phi, 'variable'): del phi.variable # type: ignore[attr-defined]

        return SSAFunction(
            name=proc.name, params=[p.name for p in proc.params],
            param_types=[p.param_type for p in proc.params],
            return_type=proc.return_type,
            entry_block=self._ssa_blocks[cfg.entry.block.name] if cfg.entry else None,
            blocks=list(self._ssa_blocks.values())
        )

    def _identify_variable_defs(self, proc: TACProc):
        """Find all blocks where each variable (non-temporary) is defined."""
        entry_block_name = self._cfg.entry.block.name if self._cfg and self._cfg.entry else None
        if entry_block_name:
            for param in proc.params:
                if not self._is_temporary(param.name) and param.name not in self._globals:
                    self._var_defs_blocks[param.name].add(entry_block_name)
        else: print(f"Warning: No entry block for {proc.name}, param defs missed.")

        for block_name, ssa_block in self._ssa_blocks.items():
            for instr in ssa_block.instructions:
                if not hasattr(instr, 'original_tac'): continue
                tac_instr = instr.original_tac # type: ignore[attr-defined]
                if tac_instr.result and isinstance(tac_instr.result, TACTemp):
                     var_name = tac_instr.result.name
                     if not self._is_temporary(var_name) and var_name not in self._globals:
                         self._var_defs_blocks[var_name].add(block_name)

    def _insert_phi_functions(self):
        """Insert necessary phi functions based on dominance frontiers."""
        if self._df is None: raise RuntimeError("Dominance frontiers not computed.")
        variables = list(self._var_defs_blocks.keys())
        phi_locations: dict[tuple[str, str], bool] = collections.defaultdict(bool)
        var_worklist: dict[str, collections.deque[str]] = collections.defaultdict(collections.deque)
        for var, blocks in self._var_defs_blocks.items():
            for block_name in blocks: var_worklist[var].append(block_name)

        for var in variables:
            queued_blocks: set[str] = set(var_worklist[var])
            while var_worklist[var]:
                block_name = var_worklist[var].popleft()
                queued_blocks.remove(block_name)
                for frontier_block_name in self._df.get(block_name, set()):
                    if not phi_locations[(frontier_block_name, var)]:
                        target_block = self._ssa_blocks[frontier_block_name]
                        phi = SSAPhi(result=None, args={}) # type: ignore[arg-type]
                        phi.variable = var # type: ignore[attr-defined]
                        target_block.phis.insert(0, phi)
                        phi_locations[(frontier_block_name, var)] = True
                        if frontier_block_name not in queued_blocks:
                             var_worklist[var].append(frontier_block_name)
                             queued_blocks.add(frontier_block_name)

    def _rename_variables(self, block_name: str, params: set[str]):
        """Recursively rename variables traversing the dominator tree."""
        if self._idom is None or self._dom_tree_children is None:
             raise RuntimeError("Dominator tree not computed.")

        ssa_block = self._ssa_blocks[block_name]
        pushed_stack_counts = collections.defaultdict(int)

        # 1. Rename Phi results (Definitions)
        for phi in ssa_block.phis:
             if not hasattr(phi, 'variable'): continue
             var = phi.variable # type: ignore[attr-defined]
             new_name = self._generate_new_name(var)
             phi.result = new_name
             pushed_stack_counts[var] += 1

        # 2. Rename Instruction results (Definitions) and arguments (Uses)
        for instr in ssa_block.instructions:
             if not hasattr(instr, 'original_tac'): continue
             tac_instr = instr.original_tac # type: ignore[attr-defined]
             # Rename arguments (Uses)
             new_args = []
             for arg in tac_instr.args:
                  if isinstance(arg, TACTemp):
                      base_name = arg.name
                      if self._is_variable(base_name, params):
                          if not self._stacks[base_name]:
                              # This indicates use before def unless it's a parameter
                              # Parameters should have version 0 pushed initially
                              raise NameError(f"Variable '{base_name}' used before definition in block {block_name}")
                          current_version = self._stacks[base_name][-1]
                          new_args.append(SSAName(base_name=base_name, version=current_version))
                      else: new_args.append(base_name) # Global/Func/Temp
                  elif isinstance(arg, TACConst): new_args.append(SSAConst(value=arg.value))
                  elif isinstance(arg, TACLabel): new_args.append(arg.name)
                  else: raise TypeError(f"Unexpected TAC arg type: {type(arg)}")
             instr.args = new_args
             # Rename result (Definition)
             if tac_instr.result and isinstance(tac_instr.result, TACTemp):
                  base_name = tac_instr.result.name
                  if self._is_variable(base_name, params):
                      new_name = self._generate_new_name(base_name)
                      instr.result = new_name
                      pushed_stack_counts[base_name] += 1
                  else: instr.result = base_name # Temp

        # 3. Fill Phi arguments in successor blocks (Uses) - *** CORRECTED LOGIC ***
        for succ_node in ssa_block.successors: # Iterate through actual successor SSABlocks
            succ_block_name = succ_node.name
            for phi in succ_node.phis: # For each phi in the successor
                 if not hasattr(phi, 'variable'): continue
                 var = phi.variable # type: ignore[attr-defined] # Original variable name

                 # Check the stack for the current block's version of 'var'
                 if self._stacks[var]:
                     # If defined, use the current version from this predecessor path
                     current_version = self._stacks[var][-1]
                     phi.args[block_name] = SSAName(base_name=var, version=current_version)
                 else:
                     # If 'var' has no version on the stack for this path,
                     # it means it's undefined coming from this predecessor.
                     # Add a placeholder (None) to signify this.
                     # The LLVM generator will handle this (e.g., create undef).
                     phi.args[block_name] = None # Use None as placeholder for undef

        # 4. Recurse down the dominator tree
        for child_block_name in self._dom_tree_children.get(block_name, []):
            self._rename_variables(child_block_name, params)

        # 5. Pop definitions created in this block from stacks
        for var, count in pushed_stack_counts.items():
            for _ in range(count):
                if self._stacks[var]: self._stacks[var].pop()
                else: print(f"Warning: Popped empty stack for '{var}' after {block_name}")

    def _generate_new_name(self, base_name: str) -> SSAName:
        """Generates a new SSA version for a variable and updates the stack."""
        version = self._counters[base_name]
        self._counters[base_name] += 1
        self._stacks[base_name].append(version)
        return SSAName(base_name=base_name, version=version)

    def _is_temporary(self, name: str) -> bool:
        """Checks if a name represents a temporary variable."""
        return name.startswith('t')

    def _is_variable(self, name: str, params: set[str]) -> bool:
        """Checks if a name is a user variable or parameter needing SSA renaming."""
        return not self._is_temporary(name) and name not in self._globals
