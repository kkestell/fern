from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from lark import Lark
from lark.tree import Tree
from lark.lexer import Token

from ast_nodes import (
    ASTProgram, ASTFunction, ASTParameter, ASTType, ASTBlock, ASTStatement, ASTAssignmentStatement, ASTReturnStatement,
    ASTVariableDeclarationStatement, ASTIfStatement, ASTBinaryExpression, ASTVariableReferenceExpression,
    ASTIntegerLiteralExpression, ASTBooleanLiteralExpression, ASTFunctionCallExpression, ASTBinaryOperator,
    ASTExpression, ASTUnaryExpression, ASTUnaryOperator
)


class Parser:
    def __init__(self) -> None:
        grammar_path = Path(__file__).parent / "fern.lark"
        with open(grammar_path, 'r') as f:
            grammar = f.read()
        self.parser = Lark(grammar, parser='lalr', start='start')

    def parse(self, code: str) -> ASTProgram:
        tree = self.parser.parse(code)
        return self._convert_program(tree)

    def _convert_program(self, tree: Tree[Any]) -> ASTProgram:
        functions = []
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'function':
                func = self._convert_function(child)
                functions.append(func)
        return ASTProgram(functions=functions)

    def _convert_function(self, tree: Tree[Any]) -> ASTFunction:
        idx = 0
        name_token = tree.children[idx]
        if isinstance(name_token, Token):
            name = name_token.value
        else:
            raise TypeError("Expected Token for function name")
        idx += 1

        parameters = []
        if isinstance(tree.children[idx], Tree) and tree.children[idx].data == 'parameters':
            parameters = self._convert_parameters(tree.children[idx])
            idx += 1

        return_type_tree = tree.children[idx]
        return_type = self._convert_type(return_type_tree)
        idx += 1

        block_tree = tree.children[idx]
        if isinstance(block_tree, Tree):
            body = self._convert_block(block_tree)
        else:
            raise TypeError("Expected Tree for function body block")

        return ASTFunction(name=name, parameters=parameters, return_type=return_type, body=body)

    def _convert_parameters(self, tree: Tree[Any]) -> list[ASTParameter]:
        parameters = []
        for param_tree in tree.children:
            param = self._convert_parameter(param_tree)
            parameters.append(param)
        return parameters

    def _convert_parameter(self, tree: Tree[Any]) -> ASTParameter:
        name_token = tree.children[0]
        type_tree = tree.children[1]

        if isinstance(name_token, Token):
            name = name_token.value
        else:
            raise TypeError("Expected Token for parameter name")

        param_type = self._convert_type(type_tree)
        return ASTParameter(name=name, type=param_type)

    def _convert_type(self, tree: Tree[Any]) -> ASTType:
        token = tree.children[0]
        if isinstance(token, Token):
            if token.type == 'VOID':
                return ASTType(name='void')
            elif token.type == 'INT':
                return ASTType(name='int')
            elif token.type == 'BOOL':
                return ASTType(name='bool')
            else:
                raise ValueError(f"Unknown type token: {token.type}")
        else:
            raise TypeError("Expected Token for type")

    def _convert_block(self, tree: Tree[Any]) -> ASTBlock:
        statements = []
        for child in tree.children:
            stmt = self._convert_statement(child)
            statements.append(stmt)
        return ASTBlock(statements=statements)

    def _convert_statement(self, tree: Tree[Any]) -> ASTStatement:
        if isinstance(tree, Tree):
            if tree.data == 'var_decl_stmt':
                return self._convert_var_decl_stmt(tree)
            elif tree.data == 'if_stmt':
                return self._convert_if_stmt(tree)
            elif tree.data == 'return_stmt':
                return self._convert_return_stmt(tree)
            elif tree.data == 'expr_stmt':
                return self._convert_expr_stmt(tree)
            else:
                raise ValueError(f"Unknown statement type: {tree.data}")
        else:
            raise ValueError(f"Expected Tree node for statement, got {type(tree)}")

    def _convert_var_decl_stmt(self, tree: Tree[Any]) -> ASTVariableDeclarationStatement:
        idx = 0
        name_token = tree.children[idx]
        if isinstance(name_token, Token):
            name = name_token.value
        else:
            raise TypeError("Expected Token for variable name")
        idx += 1

        type_tree = tree.children[idx]
        var_type = self._convert_type(type_tree)
        idx += 1

        initial_value = None
        if idx < len(tree.children):
            expr_tree = tree.children[idx]
            initial_value = self._convert_expression(expr_tree)

        return ASTVariableDeclarationStatement(name=name, type=var_type, initial_value=initial_value)

    def _convert_if_stmt(self, tree: Tree[Any]) -> ASTIfStatement:
        idx = 0
        condition_tree = tree.children[idx]
        condition = self._convert_expression(condition_tree)
        idx += 1

        then_block_tree = tree.children[idx]
        then_block = self._convert_block(then_block_tree)
        idx += 1

        else_block = None
        if idx < len(tree.children):
            else_block_tree = tree.children[idx]
            else_block = self._convert_block(else_block_tree)

        return ASTIfStatement(condition=condition, then_block=then_block, else_block=else_block)

    def _convert_return_stmt(self, tree: Tree[Any]) -> ASTReturnStatement:
        if len(tree.children) > 0:
            expr_tree = tree.children[0]
            expression = self._convert_expression(expr_tree)
        else:
            expression = None
        return ASTReturnStatement(expression=expression)

    def _convert_expr_stmt(self, tree: Tree[Any]) -> ASTAssignmentStatement:
        name_token = tree.children[0]
        expr_tree = tree.children[1]

        if isinstance(name_token, Token):
            name = name_token.value
        else:
            raise TypeError("Expected Token for assignment name")

        expression = self._convert_expression(expr_tree)
        return ASTAssignmentStatement(name=name, expression=expression)

    def _convert_expression(self, tree: Tree[Any]) -> ASTExpression:
        if isinstance(tree, Tree):
            if tree.data == 'number':
                token = cast(Token, tree.children[0])
                return ASTIntegerLiteralExpression(value=int(token.value))
            elif tree.data == 'var':
                token = cast(Token, tree.children[0])
                return ASTVariableReferenceExpression(name=token.value)
            elif tree.data == 'unary_op':
                token = cast(Token, tree.children[0])
                operator = {
                    'EXCLAMATION': ASTUnaryOperator.NOT,
                    'MINUS': ASTUnaryOperator.NEGATE
                }[token.type]
                operand = self._convert_expression(tree.children[1])
                return ASTUnaryExpression(operator=operator, operand=operand)
            elif tree.data == 'binary_op':
                left = self._convert_expression(tree.children[0])
                token = cast(Token, tree.children[1])
                operator = {
                    'PLUS': ASTBinaryOperator.ADD,
                    'MINUS': ASTBinaryOperator.SUBTRACT,
                    'ASTERISK': ASTBinaryOperator.MULTIPLY,
                    'SLASH': ASTBinaryOperator.DIVIDE,
                    'GREATER': ASTBinaryOperator.GREATER,
                    'LESS': ASTBinaryOperator.LESS,
                    'EQUAL': ASTBinaryOperator.EQUAL,
                    'NOT_EQUAL': ASTBinaryOperator.NOT_EQUAL,
                    'GREATER_EQUAL': ASTBinaryOperator.GREATER_EQUAL,
                    'LESS_EQUAL': ASTBinaryOperator.LESS_EQUAL,
                    'AMPERSAND_AMPERSAND': ASTBinaryOperator.AND,
                    'PIPE_PIPE': ASTBinaryOperator.OR
                }[token.type]
                right = self._convert_expression(tree.children[2])
                return ASTBinaryExpression(operator=operator, left=left, right=right)
            elif tree.data == 'funccall':
                return self._convert_funccall(tree)
            elif tree.data == 'true':
                return ASTBooleanLiteralExpression(value=True)
            elif tree.data == 'false':
                return ASTBooleanLiteralExpression(value=False)
            else:
                raise ValueError(f"Unknown expression type: {tree.data}")
        else:
            raise ValueError(f"Expected Tree node for expression, got {type(tree)}")

    def _convert_funccall(self, tree: Tree[Any]) -> ASTFunctionCallExpression:
        name_token = tree.children[0]
        args_tree = tree.children[1]

        if isinstance(name_token, Token):
            name = name_token.value
        else:
            raise TypeError("Expected Token for function call name")

        arguments = [self._convert_expression(arg) for arg in args_tree.children]
        return ASTFunctionCallExpression(name=name, arguments=arguments)

    @staticmethod
    def _convert_unary_operator(token: Token) -> ASTUnaryOperator:
        operator_map = {
            '!': ASTUnaryOperator.NOT,
            '-': ASTUnaryOperator.NEGATE
        }
        if token.value not in operator_map:
            raise ValueError(f"Unknown unary operator: {token.value}")
        return operator_map[token.value]

    @staticmethod
    def _convert_binary_operator(token: Token) -> ASTBinaryOperator:
        operator_map = {
            '+': ASTBinaryOperator.ADD,
            '-': ASTBinaryOperator.SUBTRACT,
            '*': ASTBinaryOperator.MULTIPLY,
            '/': ASTBinaryOperator.DIVIDE,
            '>': ASTBinaryOperator.GREATER,
            '<': ASTBinaryOperator.LESS,
            '==': ASTBinaryOperator.EQUAL,
            '!=': ASTBinaryOperator.NOT_EQUAL,
            '>=': ASTBinaryOperator.GREATER_EQUAL,
            '<=': ASTBinaryOperator.LESS_EQUAL,
            '&&': ASTBinaryOperator.AND,
            '||': ASTBinaryOperator.OR
        }
        if token.value not in operator_map:
            raise ValueError(f"Unknown binary operator: {token.value}")
        return operator_map[token.value]
