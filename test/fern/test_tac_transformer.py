import pytest

from src.fern.ast_nodes import (
    ASTProgram, ASTFunction, ASTBlock, ASTReturnStatement,
    ASTVariableDeclarationStatement, ASTAssignmentStatement, ASTIfStatement,
    ASTBinaryExpression, ASTUnaryExpression, ASTIntegerLiteralExpression,
    ASTBooleanLiteralExpression, ASTVariableReferenceExpression,
    ASTFunctionCallExpression, ASTType, ASTParameter, ASTExpression,
    ASTBinaryOperator, ASTUnaryOperator
)
from src.fern.symbol_table import (
    SymbolTable, VariableSymbol, FunctionSymbol
)
from src.fern.tac_nodes import (
    TACProgram, TACProc, TACBlock, TACInstr, TACOp, TACConst, TACTemp, TACValue, TACLabel, TACParameter
)
from src.fern.tac_transformer import TACTransformer


@pytest.fixture
def int_type() -> ASTType:
    return ASTType(name="int")


@pytest.fixture
def bool_type() -> ASTType:
    return ASTType(name="bool")


@pytest.fixture
def void_type() -> ASTType:
    return ASTType(name="void")


@pytest.fixture
def transformer() -> TACTransformer:
    return TACTransformer()


@pytest.fixture
def symbol_table() -> SymbolTable:
    st = SymbolTable()
    return st


# Helper to get instructions from a TACProgram/TACProc
def get_all_instructions(program: TACProgram, proc_name: str) -> list[TACInstr]:
    proc = next((p for p in program.procedures if p.name == proc_name), None)
    if not proc:
        return []
    instructions = []
    # Ensure blocks are processed in a somewhat predictable order if possible
    # (Entry first, then others - though order isn't strictly guaranteed)
    ordered_blocks = [proc.entry_block] + [b for b in proc.blocks if b is not proc.entry_block]
    for block in ordered_blocks:
        instructions.extend(block.instructions)
    return instructions


def test_generate_empty_function(transformer, symbol_table, void_type):
    func = ASTFunction(name="main", parameters=[], return_type=void_type, body=ASTBlock(statements=[]))
    program_ast = ASTProgram(functions=[func])
    symbol_table.define(FunctionSymbol(name="main", return_type=void_type, parameters=[]))

    program_tac = transformer.generate(program_ast, symbol_table)

    assert len(program_tac.procedures) == 1
    proc = program_tac.procedures[0]
    assert proc.name == "main"
    assert proc.return_type == void_type
    assert len(proc.params) == 0
    assert len(proc.blocks) == 1

    entry_block = proc.entry_block
    assert len(entry_block.instructions) == 1
    assert entry_block.instructions[0].op == TACOp.RET
    assert len(entry_block.instructions[0].args) == 0


def test_generate_function_with_return_value(transformer, symbol_table, int_type):
    ret_stmt = ASTReturnStatement(expression=ASTIntegerLiteralExpression(value=42))
    func = ASTFunction(name="get_answer", parameters=[], return_type=int_type, body=ASTBlock(statements=[ret_stmt]))
    program_ast = ASTProgram(functions=[func])
    symbol_table.define(FunctionSymbol(name="get_answer", return_type=int_type, parameters=[]))

    program_tac = transformer.generate(program_ast, symbol_table)

    assert len(program_tac.procedures) == 1
    proc = program_tac.procedures[0]
    assert proc.name == "get_answer"
    assert proc.return_type == int_type

    instructions = get_all_instructions(program_tac, "get_answer")
    # Expect: RET 42; t = 0; RET t
    assert len(instructions) == 3

    # Check explicit return value generation
    assert instructions[0].op == TACOp.RET
    assert len(instructions[0].args) == 1
    assert isinstance(instructions[0].args[0], TACConst)
    assert instructions[0].args[0].value == 42

    # Check default return sequence generation
    assert instructions[1].op == TACOp.ASSIGN
    assert isinstance(instructions[1].result, TACTemp)
    assert len(instructions[1].args) == 1
    assert isinstance(instructions[1].args[0], TACConst)
    assert instructions[1].args[0].value == 0

    assert instructions[2].op == TACOp.RET
    assert len(instructions[2].args) == 1
    assert instructions[2].args[0] == instructions[1].result # Should return the temp assigned 0


def test_generate_var_decl_no_init(transformer, symbol_table, int_type, void_type):
    var_decl = ASTVariableDeclarationStatement(name="x", type=int_type, initial_value=None)
    func = ASTFunction(name="test", parameters=[], return_type=void_type, body=ASTBlock(statements=[var_decl]))
    program_ast = ASTProgram(functions=[func])
    symbol_table.define(FunctionSymbol(name="test", return_type=void_type, parameters=[]))

    program_tac = transformer.generate(program_ast, symbol_table)
    instructions = get_all_instructions(program_tac, "test")

    assert len(instructions) == 2
    assert instructions[0].op == TACOp.ASSIGN
    assert isinstance(instructions[0].result, TACTemp)
    assert instructions[0].result.name == "x"
    assert len(instructions[0].args) == 1
    assert isinstance(instructions[0].args[0], TACConst)
    assert instructions[0].args[0].value == 0
    assert instructions[1].op == TACOp.RET


def test_generate_var_decl_with_init(transformer, symbol_table, int_type, void_type):
    init_expr = ASTIntegerLiteralExpression(value=123)
    var_decl = ASTVariableDeclarationStatement(name="y", type=int_type, initial_value=init_expr)
    func = ASTFunction(name="test", parameters=[], return_type=void_type, body=ASTBlock(statements=[var_decl]))
    program_ast = ASTProgram(functions=[func])
    symbol_table.define(FunctionSymbol(name="test", return_type=void_type, parameters=[]))

    program_tac = transformer.generate(program_ast, symbol_table)
    instructions = get_all_instructions(program_tac, "test")

    assert len(instructions) == 2
    assert instructions[0].op == TACOp.ASSIGN
    assert isinstance(instructions[0].result, TACTemp)
    assert instructions[0].result.name == "y"
    assert len(instructions[0].args) == 1
    assert isinstance(instructions[0].args[0], TACConst)
    assert instructions[0].args[0].value == 123
    assert instructions[1].op == TACOp.RET


def test_generate_assignment(transformer, symbol_table, int_type, void_type):
    assign_expr = ASTVariableReferenceExpression(name="z")
    assign_stmt = ASTAssignmentStatement(name="x", expression=assign_expr)
    func = ASTFunction(name="test", parameters=[ASTParameter(name="z", type=int_type)], return_type=void_type, body=ASTBlock(statements=[assign_stmt]))
    program_ast = ASTProgram(functions=[func])
    symbol_table.define(FunctionSymbol(name="test", return_type=void_type, parameters=[ASTParameter(name="z", type=int_type)]))

    program_tac = transformer.generate(program_ast, symbol_table)
    instructions = get_all_instructions(program_tac, "test")

    assert len(instructions) == 2
    assert instructions[0].op == TACOp.ASSIGN
    assert isinstance(instructions[0].result, TACTemp)
    assert instructions[0].result.name == "x"
    assert len(instructions[0].args) == 1
    assert isinstance(instructions[0].args[0], TACTemp)
    assert instructions[0].args[0].name == "z"
    assert instructions[1].op == TACOp.RET


def test_generate_binary_expression(transformer, symbol_table, int_type, void_type):
    expr = ASTBinaryExpression(
        operator=ASTBinaryOperator.ADD,
        left=ASTIntegerLiteralExpression(value=5),
        right=ASTVariableReferenceExpression(name="y")
    )
    assign_stmt = ASTAssignmentStatement(name="x", expression=expr)
    func = ASTFunction(name="test", parameters=[ASTParameter(name="y", type=int_type)], return_type=void_type, body=ASTBlock(statements=[assign_stmt]))
    program_ast = ASTProgram(functions=[func])
    symbol_table.define(FunctionSymbol(name="test", return_type=void_type, parameters=[ASTParameter(name="y", type=int_type)]))

    program_tac = transformer.generate(program_ast, symbol_table)
    instructions = get_all_instructions(program_tac, "test")

    assert len(instructions) == 3
    assert instructions[0].op == TACOp.ADD
    assert isinstance(instructions[0].result, TACTemp)
    assert len(instructions[0].args) == 2
    assert isinstance(instructions[0].args[0], TACConst)
    assert instructions[0].args[0].value == 5
    assert isinstance(instructions[0].args[1], TACTemp)
    assert instructions[0].args[1].name == "y"

    assert instructions[1].op == TACOp.ASSIGN
    assert isinstance(instructions[1].result, TACTemp)
    assert instructions[1].result.name == "x"
    assert len(instructions[1].args) == 1
    assert instructions[1].args[0] == instructions[0].result

    assert instructions[2].op == TACOp.RET


def test_generate_unary_expression(transformer, symbol_table, bool_type, void_type):
    expr = ASTUnaryExpression(operator=ASTUnaryOperator.NOT, operand=ASTVariableReferenceExpression(name="y"))
    assign_stmt = ASTAssignmentStatement(name="x", expression=expr)
    func = ASTFunction(name="test", parameters=[ASTParameter(name="y", type=bool_type)], return_type=void_type, body=ASTBlock(statements=[assign_stmt]))
    program_ast = ASTProgram(functions=[func])
    symbol_table.define(FunctionSymbol(name="test", return_type=void_type, parameters=[ASTParameter(name="y", type=bool_type)]))

    program_tac = transformer.generate(program_ast, symbol_table)
    instructions = get_all_instructions(program_tac, "test")

    assert len(instructions) == 3
    assert instructions[0].op == TACOp.NOT
    assert isinstance(instructions[0].result, TACTemp)
    assert len(instructions[0].args) == 1
    assert isinstance(instructions[0].args[0], TACTemp)
    assert instructions[0].args[0].name == "y"

    assert instructions[1].op == TACOp.ASSIGN
    assert isinstance(instructions[1].result, TACTemp)
    assert instructions[1].result.name == "x"
    assert len(instructions[1].args) == 1
    assert instructions[1].args[0] == instructions[0].result

    assert instructions[2].op == TACOp.RET


def test_generate_function_call(transformer, symbol_table, int_type, void_type):
    call_expr = ASTFunctionCallExpression(name="foo", arguments=[ASTIntegerLiteralExpression(value=5)])
    assign_stmt = ASTAssignmentStatement(name="x", expression=call_expr)
    func = ASTFunction(name="main", parameters=[], return_type=void_type, body=ASTBlock(statements=[assign_stmt]))
    program_ast = ASTProgram(functions=[func])
    symbol_table.define(FunctionSymbol(name="main", return_type=void_type, parameters=[]))
    symbol_table.define(FunctionSymbol(name="foo", return_type=int_type, parameters=[ASTParameter(name="p1", type=int_type)]))

    program_tac = transformer.generate(program_ast, symbol_table)
    instructions = get_all_instructions(program_tac, "main")

    assert len(instructions) == 3
    assert instructions[0].op == TACOp.CALL
    assert isinstance(instructions[0].result, TACTemp)
    assert len(instructions[0].args) == 2
    assert isinstance(instructions[0].args[0], TACTemp)
    assert instructions[0].args[0].name == "foo"
    assert isinstance(instructions[0].args[1], TACConst)
    assert instructions[0].args[1].value == 5

    assert instructions[1].op == TACOp.ASSIGN
    assert isinstance(instructions[1].result, TACTemp)
    assert instructions[1].result.name == "x"
    assert len(instructions[1].args) == 1
    assert instructions[1].args[0] == instructions[0].result

    assert instructions[2].op == TACOp.RET


def test_generate_if_statement_no_else(transformer, symbol_table, bool_type, int_type, void_type):
    condition = ASTVariableReferenceExpression(name="a")
    then_stmt = ASTAssignmentStatement(name="x", expression=ASTIntegerLiteralExpression(value=1))
    then_block = ASTBlock(statements=[then_stmt])
    if_stmt = ASTIfStatement(condition=condition, then_block=then_block, else_block=None)

    func = ASTFunction(name="test_if", parameters=[ASTParameter(name="a", type=bool_type)], return_type=void_type, body=ASTBlock(statements=[if_stmt]))
    program_ast = ASTProgram(functions=[func])
    symbol_table.define(FunctionSymbol(name="test_if", return_type=void_type, parameters=[ASTParameter(name="a", type=bool_type)]))

    program_tac = transformer.generate(program_ast, symbol_table)
    proc = program_tac.procedures[0]

    assert len(proc.blocks) == 3
    entry_b, then_b, after_b = proc.blocks

    assert len(entry_b.instructions) == 1
    br_instr = entry_b.instructions[0]
    assert br_instr.op == TACOp.BR
    assert len(br_instr.args) == 3
    assert isinstance(br_instr.args[0], TACTemp) and br_instr.args[0].name == "a"
    assert isinstance(br_instr.args[1], TACLabel) and br_instr.args[1].name == then_b.name
    assert isinstance(br_instr.args[2], TACLabel) and br_instr.args[2].name == after_b.name

    assert len(then_b.instructions) == 2
    assign_instr = then_b.instructions[0]
    jump_instr = then_b.instructions[1]
    assert assign_instr.op == TACOp.ASSIGN
    assert isinstance(assign_instr.result, TACTemp) and assign_instr.result.name == "x"
    assert isinstance(assign_instr.args[0], TACConst) and assign_instr.args[0].value == 1
    assert jump_instr.op == TACOp.JUMP
    assert len(jump_instr.args) == 1
    assert isinstance(jump_instr.args[0], TACLabel) and jump_instr.args[0].name == after_b.name

    assert len(after_b.instructions) == 1
    assert after_b.instructions[0].op == TACOp.RET


def test_generate_if_else_statement(transformer, symbol_table, bool_type, int_type, void_type):
    condition = ASTVariableReferenceExpression(name="a")
    then_stmt = ASTAssignmentStatement(name="x", expression=ASTIntegerLiteralExpression(value=1))
    else_stmt = ASTAssignmentStatement(name="x", expression=ASTIntegerLiteralExpression(value=0))
    then_block = ASTBlock(statements=[then_stmt])
    else_block = ASTBlock(statements=[else_stmt])
    if_stmt = ASTIfStatement(condition=condition, then_block=then_block, else_block=else_block)

    func = ASTFunction(name="test_if_else", parameters=[ASTParameter(name="a", type=bool_type)], return_type=void_type, body=ASTBlock(statements=[if_stmt]))
    program_ast = ASTProgram(functions=[func])
    symbol_table.define(FunctionSymbol(name="test_if_else", return_type=void_type, parameters=[ASTParameter(name="a", type=bool_type)]))

    program_tac = transformer.generate(program_ast, symbol_table)
    proc = program_tac.procedures[0]

    assert len(proc.blocks) == 4
    entry_b, then_b, else_b, after_b = proc.blocks

    assert len(entry_b.instructions) == 1
    br_instr = entry_b.instructions[0]
    assert br_instr.op == TACOp.BR
    assert len(br_instr.args) == 3
    assert isinstance(br_instr.args[0], TACTemp) and br_instr.args[0].name == "a"
    assert isinstance(br_instr.args[1], TACLabel) and br_instr.args[1].name == then_b.name
    assert isinstance(br_instr.args[2], TACLabel) and br_instr.args[2].name == else_b.name

    assert len(then_b.instructions) == 2
    assign_then = then_b.instructions[0]
    jump_then = then_b.instructions[1]
    assert assign_then.op == TACOp.ASSIGN
    assert isinstance(assign_then.result, TACTemp) and assign_then.result.name == "x"
    assert isinstance(assign_then.args[0], TACConst) and assign_then.args[0].value == 1
    assert jump_then.op == TACOp.JUMP
    assert isinstance(jump_then.args[0], TACLabel) and jump_then.args[0].name == after_b.name

    assert len(else_b.instructions) == 2
    assign_else = else_b.instructions[0]
    jump_else = else_b.instructions[1]
    assert assign_else.op == TACOp.ASSIGN
    assert isinstance(assign_else.result, TACTemp) and assign_else.result.name == "x"
    assert isinstance(assign_else.args[0], TACConst) and assign_else.args[0].value == 0
    assert jump_else.op == TACOp.JUMP
    assert isinstance(jump_else.args[0], TACLabel) and jump_else.args[0].name == after_b.name

    assert len(after_b.instructions) == 1
    assert after_b.instructions[0].op == TACOp.RET

