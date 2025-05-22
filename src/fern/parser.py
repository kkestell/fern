from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from lark import Lark
from lark.tree import Tree
from lark.lexer import Token

from src.fern.ast_nodes import (
    ASTProgram, ASTFunction, ASTParameter, ASTType, ASTBlock, ASTStatement, ASTAssignmentStatement, ASTReturnStatement,
    ASTVariableDeclarationStatement, ASTIfStatement, ASTBinaryExpression, ASTVariableReferenceExpression,
    ASTIntegerLiteralExpression, ASTBooleanLiteralExpression, ASTFunctionCallExpression, ASTBinaryOperator,
    ASTExpression, ASTUnaryExpression, ASTUnaryOperator
)


class Parser:
    def __init__(self) -> None:
        grammar_path = Path(__file__).parent / "fern.lark"
        if not grammar_path.exists():
             grammar_path = Path("fern.lark") # Fallback to current directory

        if not grammar_path.exists():
            raise FileNotFoundError(f"Grammar file 'fern.lark' not found at {grammar_path.resolve()} or current directory.")

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
        if isinstance(name_token, Token) and name_token.type == 'NAME':
            name = name_token.value
        else:
            raise TypeError(f"Expected NAME Token for function name, got {type(name_token)}")
        idx += 1

        parameters = []
        if idx < len(tree.children) and isinstance(tree.children[idx], Tree) and tree.children[idx].data == 'parameters':
            parameters = self._convert_parameters(tree.children[idx])
            idx += 1

        return_type_tree = tree.children[idx]
        if isinstance(return_type_tree, Tree) and return_type_tree.data in ('int_type', 'bool_type', 'void_type'):
             return_type = self._convert_type(return_type_tree)
        else:
             expected_data = "'int_type', 'bool_type', or 'void_type'"
             actual_data = f"'{return_type_tree.data}'" if isinstance(return_type_tree, Tree) else "None"
             raise TypeError(f"Expected Tree with data {expected_data} for return type, got {type(return_type_tree)} with data {actual_data}")
        idx += 1

        block_tree = tree.children[idx]
        if isinstance(block_tree, Tree) and block_tree.data == 'block':
            body = self._convert_block(block_tree)
        else:
            raise TypeError(f"Expected Tree 'block' for function body, got {type(block_tree)}")

        return ASTFunction(name=name, parameters=parameters, return_type=return_type, body=body)

    def _convert_parameters(self, tree: Tree[Any]) -> list[ASTParameter]:
        parameters = []
        for param_tree in tree.children:
             if isinstance(param_tree, Tree) and param_tree.data == 'parameter':
                 param = self._convert_parameter(param_tree)
                 parameters.append(param)
             else:
                 raise TypeError(f"Expected Tree 'parameter', got {type(param_tree)}")
        return parameters

    def _convert_parameter(self, tree: Tree[Any]) -> ASTParameter:
        name_token = tree.children[0]
        if isinstance(name_token, Token) and name_token.type == 'NAME':
            name = name_token.value
        else:
            raise TypeError(f"Expected NAME Token for parameter name, got {type(name_token)}")

        type_tree = tree.children[1]
        if isinstance(type_tree, Tree) and type_tree.data in ('int_type', 'bool_type', 'void_type'):
            param_type = self._convert_type(type_tree)
        else:
             expected_data = "'int_type', 'bool_type', or 'void_type'"
             actual_data = f"'{type_tree.data}'" if isinstance(type_tree, Tree) else "None"
             raise TypeError(f"Expected Tree with data {expected_data} for parameter type, got {type(type_tree)} with data {actual_data}")

        return ASTParameter(name=name, type=param_type)

    def _convert_type(self, tree: Tree[Any]) -> ASTType:
        token = tree.children[0]
        if isinstance(token, Token):
            type_name_map = {
                'INT': 'int',
                'BOOL': 'bool',
                'VOID': 'void'
            }
            if token.type in type_name_map:
                return ASTType(name=type_name_map[token.type])
            else:
                raise ValueError(f"Unknown type token inside type tree: {token.type}")
        else:
            raise TypeError(f"Expected Token child for type tree, got {type(token)}")

    def _convert_block(self, tree: Tree[Any]) -> ASTBlock:
        statements = []
        for child in tree.children:
            stmt = self._convert_statement(child)
            statements.append(stmt)
        return ASTBlock(statements=statements)

    def _convert_statement(self, node: Tree[Any] | Token[Any]) -> ASTStatement:
        if isinstance(node, Tree):
            if node.data == 'var_decl_stmt':
                return self._convert_var_decl_stmt(node)
            elif node.data == 'if_stmt':
                return self._convert_if_stmt(node)
            elif node.data == 'return_stmt':
                return self._convert_return_stmt(node)
            elif node.data == 'expr_stmt':
                return self._convert_expr_stmt(node)
            elif node.data in ['expr', 'or_expr', 'and_expr', 'comparison', 'sum', 'product', 'unary', 'atom', 'funccall']:
                 raise ValueError(f"Unexpected expression node '{node.data}' found where statement was expected.")
            else:
                raise ValueError(f"Unknown statement type: {node.data}")
        else:
            raise ValueError(f"Expected Tree node for statement, got {type(node)}")

    def _convert_var_decl_stmt(self, tree: Tree[Any]) -> ASTVariableDeclarationStatement:
        idx = 0
        name_token = tree.children[idx]
        if isinstance(name_token, Token) and name_token.type == 'NAME':
            name = name_token.value
        else:
            raise TypeError(f"Expected NAME Token for variable name, got {type(name_token)}")
        idx += 1

        var_type: ASTType | None = None
        if idx < len(tree.children) and isinstance(tree.children[idx], Tree) and tree.children[idx].data in ('int_type', 'bool_type', 'void_type'):
            var_type = self._convert_type(tree.children[idx])
            idx += 1

        initial_value: ASTExpression | None = None
        if idx < len(tree.children):
             expr_node = tree.children[idx]
             if isinstance(expr_node, Tree) and expr_node.data not in ('int_type', 'bool_type', 'void_type'):
                 initial_value = self._convert_expression(expr_node)
                 idx += 1

        return ASTVariableDeclarationStatement(name=name, type=var_type, initial_value=initial_value)

    # --- Corrected _convert_if_stmt Method ---
    def _convert_if_stmt(self, tree: Tree[Any]) -> ASTIfStatement:
        """Converts an if/else-if/else statement tree."""
        children = tree.children
        num_children = len(children)
        idx = 0

        # First child is the main 'if' condition
        if idx >= num_children or not isinstance(children[idx], Tree):
             raise ValueError("If statement missing condition.")
        main_condition = self._convert_expression(children[idx])
        idx += 1

        # Second child is the main 'then' block
        if idx >= num_children or not isinstance(children[idx], Tree) or children[idx].data != 'block':
             raise ValueError("If statement missing 'then' block.")
        main_then_block = self._convert_block(children[idx])
        idx += 1

        # Process remaining children for 'else if' and 'else'
        else_if_conditions = []
        else_if_blocks = []
        final_else_block = None

        while idx < num_children:
            # The grammar structure puts condition and block pairs directly as children.
            # Check for an 'else if' condition (which must be an expression Tree)
            # followed by a block Tree.
            if idx + 1 < num_children and isinstance(children[idx], Tree) and isinstance(children[idx+1], Tree) and children[idx+1].data == 'block':
                # This pair is an 'else if' condition and block
                else_if_conditions.append(self._convert_expression(children[idx]))
                else_if_blocks.append(self._convert_block(children[idx+1]))
                idx += 2
            # Check for a final 'else' block (must be the last child and a block Tree)
            elif idx == num_children - 1 and isinstance(children[idx], Tree) and children[idx].data == 'block':
                final_else_block = self._convert_block(children[idx])
                idx += 1
                break # Final else block found, exit loop
            else:
                # Structure doesn't match expected pattern from the grammar
                # (e.g., condition without block, or non-block as final else)
                raise ValueError(f"Unexpected structure in 'if' statement's else/else-if part near child index {idx}. Child type: {type(children[idx])}, Data: {getattr(children[idx], 'data', 'N/A')}")

        # Build the nested ASTIfStatement structure from the end (right-associative)
        current_else_node: ASTStatement | None = final_else_block # Start with the final else block, if any

        # Iterate backwards through the collected 'else if' parts
        for i in range(len(else_if_conditions) - 1, -1, -1):
            condition = else_if_conditions[i]
            then_block = else_if_blocks[i]
            # Create a new ASTIfStatement for this 'else if', nesting the previous one
            current_else_node = ASTIfStatement(
                condition=condition,
                then_block=then_block,
                else_block=current_else_node # Nest the previously built else part
            )

        # Return the top-level ASTIfStatement, connecting the main if/then to the nested else structure
        return ASTIfStatement(
            condition=main_condition,
            then_block=main_then_block,
            else_block=current_else_node # Attach the fully nested else structure
        )
    # --- End Corrected Method ---

    def _convert_return_stmt(self, tree: Tree[Any]) -> ASTReturnStatement:
        expression: ASTExpression | None = None
        if len(tree.children) > 0:
            expr_tree = tree.children[0]
            if isinstance(expr_tree, Tree):
                 expression = self._convert_expression(expr_tree)

        return ASTReturnStatement(expression=expression)

    def _convert_expr_stmt(self, tree: Tree[Any]) -> ASTAssignmentStatement:
        name_token = tree.children[0]
        if isinstance(name_token, Token) and name_token.type == 'NAME':
            name = name_token.value
        else:
            raise TypeError(f"Expected NAME Token for assignment target, got {type(name_token)}")

        expr_tree = tree.children[1]
        if isinstance(expr_tree, Tree):
             expression = self._convert_expression(expr_tree)
        else:
             raise TypeError(f"Expected Tree node for assignment expression, got {type(expr_tree)}")

        return ASTAssignmentStatement(name=name, expression=expression)

    def _convert_expression(self, tree: Tree[Any] | Token[Any]) -> ASTExpression:
        if isinstance(tree, Token):
             raise ValueError(f"Unexpected Token '{tree.value}' ({tree.type}) found directly in expression conversion.")

        elif isinstance(tree, Tree):
            rule_name = tree.data
            children = tree.children

            if rule_name in ('int_type', 'bool_type', 'void_type'):
                 raise ValueError(f"Type node '{rule_name}' found unexpectedly during general expression conversion.")

            if rule_name == 'number':
                token = cast(Token, children[0])
                return ASTIntegerLiteralExpression(value=int(token.value))
            elif rule_name == 'var':
                token = cast(Token, children[0])
                return ASTVariableReferenceExpression(name=token.value)
            elif rule_name == 'true':
                return ASTBooleanLiteralExpression(value=True)
            elif rule_name == 'false':
                return ASTBooleanLiteralExpression(value=False)
            elif rule_name == 'funccall':
                return self._convert_funccall(tree)
            elif rule_name == 'unary_op':
                op_token = cast(Token, children[0])
                operator = self._convert_unary_operator(op_token)
                operand = self._convert_expression(children[1])
                return ASTUnaryExpression(operator=operator, operand=operand)
            elif rule_name == 'binary_op':
                left = self._convert_expression(children[0])
                op_token = cast(Token, children[1])
                operator = self._convert_binary_operator(op_token)
                right = self._convert_expression(children[2])
                return ASTBinaryExpression(operator=operator, left=left, right=right)
            elif rule_name in ['expr', 'or_expr', 'and_expr', 'comparison', 'sum', 'product', 'unary', 'atom']:
                 if len(children) == 1:
                     return self._convert_expression(children[0])
                 else:
                     raise ValueError(f"Unexpected structure for expression rule '{rule_name}' with {len(children)} children.")
            else:
                raise ValueError(f"Unknown or unhandled expression type (Tree data): {rule_name}")
        else:
            raise ValueError(f"Expected Tree or Token node for expression, got {type(tree)}")


    def _convert_funccall(self, tree: Tree[Any]) -> ASTFunctionCallExpression:
        idx = 0
        name_token = tree.children[idx]
        if isinstance(name_token, Token) and name_token.type == 'NAME':
            name = name_token.value
        else:
            raise TypeError(f"Expected NAME Token for function call name, got {type(name_token)}")
        idx += 1

        arguments = []
        if idx < len(tree.children) and isinstance(tree.children[idx], Tree) and tree.children[idx].data == 'arguments':
            args_tree = tree.children[idx]
            arguments = [self._convert_expression(arg) for arg in args_tree.children]

        return ASTFunctionCallExpression(name=name, arguments=arguments)

    @staticmethod
    def _convert_unary_operator(token: Token) -> ASTUnaryOperator:
        operator_map = {
            'EXCLAMATION': ASTUnaryOperator.NOT,
            'MINUS': ASTUnaryOperator.NEGATE
        }
        if token.type in operator_map:
            return operator_map[token.type]
        else:
            value_map = {'!': ASTUnaryOperator.NOT, '-': ASTUnaryOperator.NEGATE}
            if token.value in value_map:
                 return value_map[token.value]
            else:
                 raise ValueError(f"Unknown unary operator token: type={token.type}, value={token.value}")


    @staticmethod
    def _convert_binary_operator(token: Token) -> ASTBinaryOperator:
        operator_map = {
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
        }
        if token.type in operator_map:
            return operator_map[token.type]
        else:
             value_map = {
                 '+': ASTBinaryOperator.ADD, '-': ASTBinaryOperator.SUBTRACT, '*': ASTBinaryOperator.MULTIPLY, '/': ASTBinaryOperator.DIVIDE,
                 '>': ASTBinaryOperator.GREATER, '<': ASTBinaryOperator.LESS, '==': ASTBinaryOperator.EQUAL, '!=': ASTBinaryOperator.NOT_EQUAL,
                 '>=': ASTBinaryOperator.GREATER_EQUAL, '<=': ASTBinaryOperator.LESS_EQUAL, '&&': ASTBinaryOperator.AND, '||': ASTBinaryOperator.OR
             }
             if token.value in value_map:
                 return value_map[token.value]
             else:
                 raise ValueError(f"Unknown binary operator token: type={token.type}, value={token.value}")
