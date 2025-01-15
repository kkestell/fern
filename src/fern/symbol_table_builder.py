from __future__ import annotations

from ast_nodes import (
    ASTProgram, ASTFunction, ASTBlock, ASTStatement, ASTVariableDeclarationStatement, ASTIfStatement
)
from symbol_table import SymbolTable, FunctionSymbol, VariableSymbol


class SymbolTableBuilder:
    def __init__(self) -> None:
        self.symbol_table = SymbolTable()

    def build(self, program: ASTProgram) -> SymbolTable:
        for function in program.functions:
            self.visit_function(function)
        return self.symbol_table

    def visit_function(self, function: ASTFunction) -> None:
        # Define the function in the global scope
        func_symbol = FunctionSymbol(
            name=function.name,
            return_type=function.return_type,
            parameters=function.parameters
        )
        self.symbol_table.define(func_symbol)

        # Enter the function's scope
        self.symbol_table.enter_scope()

        # Define parameters in the function's scope
        for param in function.parameters:
            var_symbol = VariableSymbol(name=param.name, var_type=param.type)
            self.symbol_table.define(var_symbol)

        # Visit the function body
        self.visit_block(function.body)

        # Exit the function's scope
        self.symbol_table.exit_scope()

    def visit_block(self, block: ASTBlock) -> None:
        # Enter a new scope for the block
        self.symbol_table.enter_scope()

        for statement in block.statements:
            self.visit_statement(statement)

        # Exit the block's scope
        self.symbol_table.exit_scope()

    def visit_statement(self, statement: ASTStatement) -> None:
        if isinstance(statement, ASTVariableDeclarationStatement):
            statement_type = statement.type
            if statement_type is None:
                raise ValueError("Variable declaration statement must have a type")
            var_symbol = VariableSymbol(name=statement.name, var_type=statement_type)
            self.symbol_table.define(var_symbol)
        elif isinstance(statement, ASTIfStatement):
            self.visit_if_statement(statement)
        # Handle other statement types as needed

    def visit_if_statement(self, if_statement: ASTIfStatement) -> None:
        # Visit the condition expression (not shown here)
        # self.visit_expression(if_statement.condition)

        # Visit the 'then' block
        self.visit_block(if_statement.then_block)

        # Check the type of the else_block
        if if_statement.else_block:
            if isinstance(if_statement.else_block, ASTIfStatement):
                # Recursively visit the nested if statement
                self.visit_if_statement(if_statement.else_block)
            else:
                # Otherwise, visit the block as usual
                self.visit_block(if_statement.else_block)
