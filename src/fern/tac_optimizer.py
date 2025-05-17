from __future__ import annotations

from typing import cast

from .tac_transformer import TACProgram, TACProc, TACBlock, TACOp, TACLabel


class TACDeadCodeEliminator:
    def eliminate(self, program: TACProgram) -> TACProgram:
        """Eliminates dead code from all procedures in the program."""
        for proc in program.procedures:
            self._eliminate_in_procedure(proc)
        return program

    def _eliminate_in_procedure(self, proc: TACProc) -> None:
        """Eliminates dead code within a single procedure."""
        # First eliminate dead code within blocks
        for block in proc.blocks:
            self._eliminate_in_block(block)

        # Then identify and remove unreachable blocks
        reachable_names = self._find_reachable_block_names(proc)
        proc.blocks = [block for block in proc.blocks if block.name in reachable_names]

    @staticmethod
    def _eliminate_in_block(block: TACBlock) -> None:
        """Eliminates dead code within a basic block."""
        return_idx = -1
        for idx, instr in enumerate(block.instructions):
            if instr.op == TACOp.RET:
                return_idx = idx
                break

        if return_idx != -1:
            block.instructions = block.instructions[:return_idx + 1]

    @staticmethod
    def _find_reachable_block_names(proc: TACProc) -> set[str]:
        """Identifies names of all blocks reachable from the entry block."""
        reachable = set()
        worklist = [proc.entry_block.name]
        name_to_block = {block.name: block for block in proc.blocks}

        while worklist:
            block_name = worklist.pop()
            if block_name in reachable:
                continue

            reachable.add(block_name)
            block = name_to_block[block_name]

            # Follow control flow
            for instr in block.instructions:
                if instr.op == TACOp.RET:
                    break  # Stop analyzing this block after return
                elif instr.op == TACOp.BR:
                    true_label = cast(TACLabel, instr.args[1])
                    false_label = cast(TACLabel, instr.args[2])
                    worklist.extend([true_label.name, false_label.name])
                elif instr.op == TACOp.JUMP:
                    target_label = cast(TACLabel, instr.args[0])
                    worklist.append(target_label.name)

        return reachable
