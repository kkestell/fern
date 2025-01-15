from __future__ import annotations

from typing import cast

from tac_nodes import TACProgram, TACProc, TACOp, TACLabel
from cfg_nodes import CFG, CFGNode


class CFGBuilder:
    """
    Construct Control Flow Graphs (CFGs) for a TACProgram.

    Control Flow Graphs are used in compiler design and program analysis to represent the flow 
    of control within procedures. Each CFG represents one procedure and consists of nodes 
    representing blocks of code and edges that represent control flow paths between them.
    """

    def build(self, program: TACProgram) -> list[CFG]:
        """
        Build CFGs for all procedures in a TACProgram.

        :param program: The entire TAC program containing multiple procedures.
        :type program: TACProgram

        :return: A list of CFGs, each corresponding to a single procedure.
        :rtype: list[CFG]
        """
        cfgs: list[CFG] = []

        # Iterate over each procedure in the program and build a CFG for it.
        for proc in program.procedures:
            cfg = self._build_cfg(proc)
            cfgs.append(cfg)

        return cfgs

    @staticmethod
    def _build_cfg(proc: TACProc) -> CFG:
        """
        Construct a CFG for a single procedure.

        :param proc: A procedure in the TAC program, containing basic blocks of code.
        :type proc: TACProc

        :return: The constructed CFG for the procedure.
        :rtype: CFG
        """
        nodes: list[CFGNode] = []  # Nodes for the CFG representing blocks of code.
        node_map: dict[str, CFGNode] = {}  # Map to find nodes by block name.

        # First, create a CFGNode for each block in the procedure.
        for block in proc.blocks:
            node = CFGNode(block=block)  # Initialize a node with the block.
            nodes.append(node)
            node_map[block.name] = node  # Map block name to CFGNode for easy access.

        # Connect nodes based on control flow, linking blocks based on the type of TAC operation.
        exit_nodes: set[CFGNode] = set()  # Set to track potential exit nodes.

        for node in nodes:
            # Get the last instruction in the block to determine control flow.
            last_instr = node.block.instructions[-1] if node.block.instructions else None

            if last_instr is None:
                # If there are no instructions, continue without adding successors.
                continue

            # RET (return) operation indicates an exit from the procedure.
            if last_instr.op == TACOp.RET:
                exit_nodes.add(node)
                continue

            # BR (branch) operation has two targets, leading to two possible successor nodes.
            elif last_instr.op == TACOp.BR:
                # Extract true and false labels for branching.
                true_label = cast(TACLabel, last_instr.args[1])
                false_label = cast(TACLabel, last_instr.args[2])
                true_target = node_map[true_label.name]
                false_target = node_map[false_label.name]
                # Link current node to both possible outcomes.
                node.add_successor(true_target)
                node.add_successor(false_target)

            # JUMP operation has a single target, redirecting control to one node.
            elif last_instr.op == TACOp.JUMP:
                target_label = cast(TACLabel, last_instr.args[0])
                target_node = node_map[target_label.name]
                node.add_successor(target_node)

            else:
                # For other operations (without explicit branching), assume fall-through to the next sequential block.
                curr_idx = nodes.index(node)
                if curr_idx + 1 < len(nodes):
                    next_node = nodes[curr_idx + 1]
                    node.add_successor(next_node)

        # Define the entry point (first node) for this CFG.
        entry_node = nodes[0]

        # Define the exit node. If there are multiple return nodes, arbitrarily select one. No additional exit node is
        # required if there's no post-return code.
        exit_node = next(iter(exit_nodes)) if exit_nodes else nodes[-1]

        # Return a CFG object encapsulating the entry, exit, and all nodes in the CFG.
        return CFG(
            proc_name=proc.name,
            entry=entry_node,
            exit=exit_node,
            nodes=nodes
        )
