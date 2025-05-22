from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.fern.tac_nodes import TACBlock


@dataclass(kw_only=True, frozen=True)
class CFGNode:
    """A node in the control flow graph representing a basic block."""
    block: TACBlock

    _successors: list['CFGNode'] = field(default_factory=list, hash=False, compare=False)
    _predecessors: list['CFGNode'] = field(default_factory=list, hash=False, compare=False)

    def __hash__(self) -> int:
        return hash(self.block.name)

    def __eq__(self, other: object) -> bool | Any:
        if not isinstance(other, CFGNode):
            return NotImplemented
        return self.block.name == other.block.name

    @property
    def successors(self) -> list[CFGNode]:
        return self._successors

    @property
    def predecessors(self) -> list[CFGNode]:
        return self._predecessors

    def add_successor(self, node: CFGNode) -> None:
        if node not in self._successors:
            self._successors.append(node)
        if self not in node._predecessors:
            node._predecessors.append(self)


@dataclass(kw_only=True)
class CFG:
    """Control flow graph for a procedure."""
    proc_name: str
    entry: CFGNode
    exit: CFGNode
    nodes: list[CFGNode]
