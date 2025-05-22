from __future__ import annotations

from dataclasses import field, dataclass
from enum import Enum, auto

from src.fern.ast_nodes import ASTType


class TACOp(Enum):
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
    LT = auto()   # Less than
    GT = auto()   # Greater than
    EQ = auto()   # Equal
    NEQ = auto()  # Not equal
    GTE = auto()  # Greater than or equal
    LTE = auto()  # Less than or equal

    # Unary Operations
    NEG = auto()  # Negation
    NOT = auto()  # Logical negation

    # Control Flow
    BR = auto()   # Conditional branch
    JUMP = auto() # Unconditional jump
    RET = auto()  # Return

    # Function Calls
    CALL = auto()


@dataclass
class TACValue:
    def __str__(self) -> str:
        return str(self.value) if hasattr(self, 'value') else super().__str__()


@dataclass
class TACTemp(TACValue):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class TACConst(TACValue):
    value: int | bool

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class TACLabel(TACValue):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class TACInstr:
    op: TACOp
    result: TACTemp | None = None
    args: list[TACValue] = field(default_factory=list)

    def __str__(self) -> str:
        binary_op_str_map = {
            TACOp.ADD: '+', TACOp.SUB: '-', TACOp.MUL: '*', TACOp.DIV: '/',
            TACOp.LT: '<', TACOp.GT: '>', TACOp.EQ: '==', TACOp.NEQ: '!=',
            TACOp.GTE: '>=', TACOp.LTE: '<=', TACOp.AND: '&&', TACOp.OR: '||',
        }
        unary_op_str_map = {
            TACOp.NOT: '!', TACOp.NEG: '-'
        }

        if self.op == TACOp.ASSIGN:
            return f"{self.result} = {self.args[0]}"
        elif self.op in unary_op_str_map:
            return f"{self.result} = {unary_op_str_map[self.op]}{self.args[0]}"
        elif self.op in binary_op_str_map:
            return f"{self.result} = {self.args[0]} {binary_op_str_map[self.op]} {self.args[1]}"
        elif self.op == TACOp.CALL:
            func_args = ', '.join(str(arg) for arg in self.args[1:])
            return f"{self.result} = {self.args[0]}({func_args})"
        elif self.op == TACOp.RET:
            return f"return {self.args[0]}" if self.args else "return"
        elif self.op == TACOp.BR:
            return f"if {self.args[0]} goto {self.args[1]} else goto {self.args[2]}"
        elif self.op == TACOp.JUMP:
            return f"goto {self.args[0]}"
        else:
            args_str = ', '.join(str(arg) for arg in self.args)
            return f"{self.op.name} {args_str}"


@dataclass
class TACBlock:
    name: str
    instructions: list[TACInstr] = field(default_factory=list)
    predecessors: list[TACBlock] = field(default_factory=list)
    successors: list[TACBlock] = field(default_factory=list)

    def append(self, instr: TACInstr) -> None:
        self.instructions.append(instr)

    def __str__(self) -> str:
        lines = [f"{self.name}:"]
        lines.extend(f"  {instr}" for instr in self.instructions)
        return "\n".join(lines)


@dataclass
class TACParameter:
    name: str
    param_type: ASTType

    def __str__(self) -> str:
        return f"{self.name}: {self.param_type.name.lower()}"


@dataclass
class TACProc:
    name: str
    params: list[TACParameter]
    return_type: ASTType
    entry_block: TACBlock
    blocks: list[TACBlock] = field(default_factory=list)

    def __str__(self) -> str:
        param_strs = [str(p) for p in self.params]
        lines = [f"procedure {self.name}({', '.join(param_strs)}) -> {self.return_type.name.lower()}:"]
        lines.extend(str(block) for block in self.blocks)
        return "\n".join(lines)


class TACProgram:
    def __init__(self) -> None:
        self.procedures: list[TACProc] = []
        self._temp_counter: int = 0
        self._block_counter: int = 0

    def new_temp(self) -> TACTemp:
        self._temp_counter += 1
        return TACTemp(name=f"t{self._temp_counter}")

    def new_block(self) -> TACBlock:
        self._block_counter += 1
        return TACBlock(name=f"bb{self._block_counter}")

    def __str__(self) -> str:
        return "\n\n".join(str(proc) for proc in self.procedures)