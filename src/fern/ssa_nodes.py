from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from src.fern.ast_nodes import ASTType


class SSAOp(Enum):
    # Arithmetic Operations
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()

    # Logical Operations
    AND = auto()
    OR = auto()

    # Assignment
    ASSIGN = auto()

    # Comparison Operations
    LT = auto()  # Less than
    GT = auto()  # Greater than
    EQ = auto()  # Equal
    NEQ = auto()  # Not equal
    GTE = auto()  # Greater than or equal
    LTE = auto()  # Less than or equal

    # Unary Operations
    NEG = auto()  # Negation
    NOT = auto()  # Logical negation

    # Control Flow
    BR = auto()  # Conditional branch
    JUMP = auto()  # Unconditional jump
    RET = auto()  # Return

    # Function Calls
    CALL = auto()


@dataclass
class SSAValue:
    def __str__(self) -> str:
        return str(self.value) if hasattr(self, 'value') else super().__str__()


@dataclass
class SSAConst(SSAValue):
    value: int | bool


@dataclass(frozen=True)
class SSAName:
    """A variable in SSA form with a version number."""
    base_name: str
    version: int

    def __hash__(self) -> int:
        return hash((self.base_name, self.version))

    def __str__(self) -> str:
        return f"{self.base_name}.{self.version}"


@dataclass(kw_only=True)
class SSAInstr:
    """An instruction in SSA form."""
    op: SSAOp
    result: SSAName | str | None = None
    args: list[SSAName | str | SSAConst] = field(default_factory=list)

    def __str__(self) -> str:
        op_str = {
            SSAOp.ADD: '+',
            SSAOp.SUB: '-',
            SSAOp.MUL: '*',
            SSAOp.DIV: '/',
            SSAOp.LT: '<',
            SSAOp.GT: '>',
            SSAOp.EQ: '==',
            SSAOp.NEQ: '!=',
            SSAOp.GTE: '>=',
            SSAOp.LTE: '<='
        }

        if self.op == SSAOp.ASSIGN:
            return f"{self.result} = {self.args[0]}"
        elif self.op in {SSAOp.NEG, SSAOp.NOT}:
            return f"{self.result} = {self.op.name.lower()} {self.args[0]}"
        elif self.op in op_str:
            return f"{self.result} = {self.args[0]} {op_str[self.op]} {self.args[1]}"
        elif self.op == SSAOp.OR:
            return f"{self.result} = {self.args[0]} or {self.args[1]}"
        elif self.op == SSAOp.CALL:
            func_args = ', '.join(str(arg) for arg in self.args[1:])
            return f"{self.result} = {self.args[0]}({func_args})"
        elif self.op == SSAOp.RET:
            return f"return {self.args[0]}" if self.args else "return"
        elif self.op == SSAOp.BR:
            return f"if {self.args[0]} goto {self.args[1]} else goto {self.args[2]}"
        elif self.op == SSAOp.JUMP:
            return f"goto {self.args[0]}"
        else:
            args_str = ' '.join(str(arg) for arg in self.args)
            return f"{self.result} = {self.op.name} {args_str}" if self.result else f"{self.op.name} {args_str}"


@dataclass(kw_only=True)
class SSAPhi:
    """A phi node in SSA form."""
    result: SSAName
    args: dict[str, SSAName | str | SSAConst]  # block name -> value

    def __str__(self) -> str:
        args_str = ", ".join(f"{pred}: {arg}" for pred, arg in self.args.items())
        return f"{self.result} = φ({args_str})"


@dataclass(kw_only=True)
class SSABlock:
    """Basic block in SSA form."""
    name: str
    phis: list[SSAPhi] = field(default_factory=list)
    instructions: list[SSAInstr] = field(default_factory=list)
    predecessors: list['SSABlock'] = field(default_factory=list)
    successors: list['SSABlock'] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"block {self.name}:"]
        lines.extend(f"  {phi}" for phi in self.phis)
        lines.extend(f"  {instr}" for instr in self.instructions)
        return "\n".join(lines)


@dataclass(kw_only=True)
class SSAFunction:
    """A function in SSA form."""
    name: str
    params: list[str]
    param_types: list[ASTType]
    return_type: ASTType
    entry_block: SSABlock
    blocks: list[SSABlock]

    def __str__(self) -> str:
        param_strs = [f"{param}: {typ.name.lower()}"
                      for param, typ in zip(self.params, self.param_types)]
        lines = [f"proc {self.name}({', '.join(param_strs)}) -> {self.return_type.name.lower()}:"]
        lines.extend(str(block) for block in self.blocks)
        return "\n".join(lines)


@dataclass(kw_only=True)
class SSAProgram:
    functions: list[SSAFunction] = field(default_factory=list)

    def __str__(self) -> str:
        return "\n\n".join(str(func) for func in self.functions)