from __future__ import annotations

from src.fern.ast_nodes import (
    ASTProgram, ASTFunction, ASTBlock, ASTStatement, ASTVariableDeclarationStatement, ASTIfStatement, ASTExpression,
    ASTBinaryExpression, ASTVariableReferenceExpression, ASTIntegerLiteralExpression, ASTBooleanLiteralExpression,
    ASTFunctionCallExpression, ASTBinaryOperator, ASTReturnStatement, ASTAssignmentStatement, ASTUnaryOperator,
    ASTUnaryExpression
)
from src.fern.symbol_table import SymbolTable
from src.fern.tac_nodes import (
    TACProgram, TACProc, TACBlock, TACInstr, TACOp, TACConst, TACTemp, TACValue, TACLabel, TACParameter
)


class TACTransformer:
    """
    Transforms an abstract syntax tree (AST) into a three-address code (TAC) representation.

    This class is responsible for traversing the AST, handling various types of nodes (statements and expressions),
    and generating corresponding TAC instructions, facilitating intermediate representation for further compiler phases.
    """

    def __init__(self) -> None:
        """
        Initialize a TACTransformer object with program storage and context variables.
        """
        self._program = TACProgram()
        self._current_block: TACBlock | None = None
        self._current_proc: TACProc | None = None
        self._symbol_table: SymbolTable | None = None

    def generate(self, ast: ASTProgram, symbol_table: SymbolTable) -> TACProgram:
        """
        Generate a TACProgram from an AST program and symbol table.

        :param ast: The abstract syntax tree for the entire program.
        :type ast: ASTProgram
        :param symbol_table: The symbol table for resolving identifiers.
        :type symbol_table: SymbolTable

        :return: The generated TAC program.
        :rtype: TACProgram
        """
        self._symbol_table = symbol_table
        for func in ast.functions:
            self._generate_procedure(func)
        return self._program

    def _add_instruction(self, instr: TACInstr) -> None:
        """
        Append an instruction to the current TAC block.

        :param instr: The TAC instruction to be added.
        :type instr: TACInstr

        :raises Exception: If no current block is set.
        """
        if self._current_block is None:
            raise Exception("No current block")
        self._current_block.append(instr)

    def _generate_procedure(self, func: ASTFunction) -> None:
        """
        Generate TAC for a function procedure.

        :param func: The AST function to convert.
        :type func: ASTFunction
        """
        entry_block = self._program.new_block()
        self._current_block = entry_block

        # Initialize TACProc for the function, storing its entry block and parameters.
        proc = TACProc(
            name=func.name,
            params=[TACParameter(name=p.name, param_type=p.type) for p in func.parameters],
            return_type=func.return_type,
            entry_block=entry_block,
            blocks=[entry_block]
        )
        self._program.procedures.append(proc)
        self._current_proc = proc

        # Generate TAC for the function body.
        self._generate_block(func.body)

        # Add default return if function type is non-void.
        if func.return_type.name != "void":
            temp = self._program.new_temp()
            self._add_instruction(TACInstr(
                op=TACOp.ASSIGN,
                result=temp,
                args=[TACConst(value=0)]
            ))
            self._add_instruction(TACInstr(op=TACOp.RET, args=[temp]))
        else:
            self._add_instruction(TACInstr(op=TACOp.RET))

        # Reset the current procedure and block context.
        self._current_proc = None
        self._current_block = None

    def _generate_block(self, block: ASTBlock) -> None:
        """
        Generate TAC for each statement in a block.

        :param block: The AST block to convert.
        :type block: ASTBlock
        """
        for stmt in block.statements:
            self._generate_statement(stmt)

    def _generate_statement(self, stmt: ASTStatement) -> None:
        """
        Generate TAC for various statement types.

        :param stmt: The AST statement to convert.
        :type stmt: ASTStatement
        """
        if isinstance(stmt, ASTVariableDeclarationStatement):
            if stmt.initial_value:
                value = self._generate_expression(stmt.initial_value)
                self._add_instruction(TACInstr(
                    op=TACOp.ASSIGN,
                    result=TACTemp(name=stmt.name),
                    args=[value]
                ))
            else:
                # Initialize variable with default value 0.
                self._add_instruction(TACInstr(
                    op=TACOp.ASSIGN,
                    result=TACTemp(name=stmt.name),
                    args=[TACConst(value=0)]
                ))
        elif isinstance(stmt, ASTAssignmentStatement):
            value = self._generate_expression(stmt.expression)
            self._add_instruction(TACInstr(
                op=TACOp.ASSIGN,
                result=TACTemp(name=stmt.name),
                args=[value]
            ))
        elif isinstance(stmt, ASTReturnStatement):
            if stmt.expression:
                result = self._generate_expression(stmt.expression)
                self._add_instruction(TACInstr(
                    op=TACOp.RET,
                    args=[result]
                ))
            else:
                self._add_instruction(TACInstr(op=TACOp.RET))
        elif isinstance(stmt, ASTIfStatement):
            self._generate_if_statement(stmt)
        else:
            raise Exception(f"Unexpected statement type: {type(stmt)}")

    def _generate_if_statement(self, stmt: ASTIfStatement) -> None:
        """
        Generate TAC for an if-statement, handling both branches and merging.

        This method creates distinct blocks for the "then" branch, "else" branch (if it exists), and the "after" block
        (common post-if continuation). It performs branching based on the condition's truth value and merges control
        flow after each branch.

        :param stmt: The AST if-statement to convert.
        :type stmt: ASTIfStatement
        """
        # Evaluate the condition and create TAC for it.
        condition = self._generate_expression(stmt.condition)

        # Pre-create blocks for the then, else, and after sections.
        then_block = self._program.new_block()
        else_block = self._program.new_block() if stmt.else_block else None
        after_block = self._program.new_block()

        # Add these blocks to the procedure.
        if self._current_proc is None:
            raise Exception("No current procedure")
        self._current_proc.blocks.extend([then_block])
        if else_block:
            self._current_proc.blocks.append(else_block)
        self._current_proc.blocks.append(after_block)

        # Add conditional branching to then and else blocks based on the condition.
        if else_block:
            self._add_instruction(TACInstr(
                op=TACOp.BR,
                args=[condition, TACLabel(name=then_block.name), TACLabel(name=else_block.name)]
            ))
        else:
            self._add_instruction(TACInstr(
                op=TACOp.BR,
                args=[condition, TACLabel(name=then_block.name), TACLabel(name=after_block.name)]
            ))

        # Generate TAC for the 'then' branch and ensure it jumps to the 'after' block.
        self._current_block = then_block
        self._generate_block(stmt.then_block)
        self._add_instruction(TACInstr(
            op=TACOp.JUMP,
            args=[TACLabel(name=after_block.name)]
        ))

        # Generate TAC for the 'else' branch, if it exists, and ensure it also jumps to 'after' block.
        if else_block and stmt.else_block:
            self._current_block = else_block
            if isinstance(stmt.else_block, ASTIfStatement):
                # Handle nested if-statements recursively within the else branch.
                self._generate_if_statement(stmt.else_block)
            else:
                # Generate TAC for a regular else block.
                self._generate_block(stmt.else_block)
            self._add_instruction(TACInstr(
                op=TACOp.JUMP,
                args=[TACLabel(name=after_block.name)]
            ))

        # Continue TAC generation in the 'after' block, where control flow converges.
        self._current_block = after_block

    def _generate_expression(self, expr: ASTExpression) -> TACValue:
        """
        Generate TAC for expressions, handling various expression types.

        :param expr: The AST expression to convert.
        :type expr: ASTExpression

        :return: The generated TAC value for the expression.
        :rtype: TACValue
        """
        if isinstance(expr, ASTUnaryExpression):
            value = self._generate_expression(expr.operand)
            result = self._program.new_temp()

            op_map = {
                ASTUnaryOperator.NEGATE: TACOp.NEG,
                ASTUnaryOperator.NOT: TACOp.NOT
            }

            op = op_map[expr.operator]

            self._add_instruction(TACInstr(
                op=op,
                result=result,
                args=[value]
            ))
            return result
        elif isinstance(expr, ASTBinaryExpression):
            left = self._generate_expression(expr.left)
            right = self._generate_expression(expr.right)
            result = self._program.new_temp()

            op_map = {
                ASTBinaryOperator.ADD: TACOp.ADD,
                ASTBinaryOperator.SUBTRACT: TACOp.SUB,
                ASTBinaryOperator.MULTIPLY: TACOp.MUL,
                ASTBinaryOperator.DIVIDE: TACOp.DIV,
                ASTBinaryOperator.EQUAL: TACOp.EQ,
                ASTBinaryOperator.NOT_EQUAL: TACOp.NEQ,
                ASTBinaryOperator.GREATER: TACOp.GT,
                ASTBinaryOperator.GREATER_EQUAL: TACOp.GTE,
                ASTBinaryOperator.LESS: TACOp.LT,
                ASTBinaryOperator.LESS_EQUAL: TACOp.LTE,
                ASTBinaryOperator.AND: TACOp.AND,
                ASTBinaryOperator.OR: TACOp.OR,
            }

            op = op_map[expr.operator]

            self._add_instruction(TACInstr(
                op=op,
                result=result,
                args=[left, right]
            ))
            return result
        elif isinstance(expr, ASTIntegerLiteralExpression):
            return TACConst(value=expr.value)
        elif isinstance(expr, ASTBooleanLiteralExpression):
            return TACConst(value=expr.value)
        elif isinstance(expr, ASTVariableReferenceExpression):
            return TACTemp(name=expr.name)
        elif isinstance(expr, ASTFunctionCallExpression):
            arg_values = [self._generate_expression(arg) for arg in expr.arguments]
            result = self._program.new_temp()
            self._add_instruction(TACInstr(
                op=TACOp.CALL,
                result=result,
                args=[TACTemp(name=expr.name), *arg_values]
            ))
            return result
        else:
            raise Exception(f"Unexpected expression type: {type(expr)}")