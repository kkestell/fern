from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class ASTBinaryOperator(Enum):
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    AND = auto()
    OR = auto()


class ASTUnaryOperator(Enum):
    NOT = auto()
    NEGATE = auto()


@dataclass
class ASTExpression:
    pass


@dataclass
class ASTVariableReferenceExpression(ASTExpression):
    name: str


@dataclass
class ASTType:
    name: str


@dataclass
class ASTBooleanLiteralExpression(ASTExpression):
    value: bool


@dataclass
class ASTIntegerLiteralExpression(ASTExpression):
    value: int


@dataclass
class ASTUnaryExpression(ASTExpression):
    operator: ASTUnaryOperator
    operand: ASTExpression


@dataclass
class ASTBinaryExpression(ASTExpression):
    operator: ASTBinaryOperator
    left: ASTExpression
    right: ASTExpression


@dataclass
class ASTFunctionCallExpression(ASTExpression):
    name: str
    arguments: list[ASTExpression]


@dataclass
class ASTStatement:
    pass


@dataclass
class ASTAssignmentStatement(ASTStatement):
    name: str
    expression: ASTExpression


@dataclass
class ASTReturnStatement(ASTStatement):
    expression: ASTExpression | None


@dataclass
class ASTParameter:
    name: str
    type: ASTType


@dataclass
class ASTBlock:
    statements: list[ASTStatement]


@dataclass
class ASTVariableDeclarationStatement(ASTStatement):
    name: str
    type: ASTType | None
    initial_value: ASTExpression | None


@dataclass
class ASTIfStatement(ASTStatement):
    condition: ASTExpression
    then_block: ASTBlock
    else_block: ASTBlock | None


@dataclass
class ASTFunction:
    name: str
    parameters: list[ASTParameter]
    return_type: ASTType
    body: ASTBlock


@dataclass
class ASTProgram:
    functions: list[ASTFunction]
