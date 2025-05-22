import pytest
from lark import LarkError, Tree

from src.fern.ast_nodes import (
    ASTProgram, ASTFunction, ASTBlock, ASTReturnStatement,
    ASTVariableDeclarationStatement, ASTAssignmentStatement, ASTIfStatement,
    ASTBinaryExpression, ASTUnaryExpression, ASTIntegerLiteralExpression,
    ASTBooleanLiteralExpression, ASTVariableReferenceExpression,
    ASTFunctionCallExpression, ASTType, ASTParameter,
    ASTBinaryOperator, ASTUnaryOperator
)

from src.fern.parser import Parser


@pytest.fixture
def fern_parser():
    return Parser()

def test_empty_program(fern_parser):
    ast = fern_parser.parse("")
    assert isinstance(ast, ASTProgram)
    assert hasattr(ast, 'functions')
    assert len(ast.functions) == 0

def test_empty_function(fern_parser):
    source = """
    fn main() -> void {
    }
    """
    ast = fern_parser.parse(source)
    assert isinstance(ast, ASTProgram)
    assert len(ast.functions) == 1
    function = ast.functions[0]
    assert isinstance(function, ASTFunction)
    assert function.name == "main"
    assert isinstance(function.return_type, ASTType)
    assert function.return_type.name == "void"
    assert hasattr(function, 'parameters')
    assert len(function.parameters) == 0
    assert isinstance(function.body, ASTBlock)
    assert hasattr(function.body, 'statements')
    assert len(function.body.statements) == 0

def test_multiple_functions(fern_parser):
    source = """
    fn foo() -> int {
        return 0;
    }
    fn bar() -> bool {
        return true;
    }
    """
    ast = fern_parser.parse(source)
    assert isinstance(ast, ASTProgram)
    assert len(ast.functions) == 2
    assert isinstance(ast.functions[0], ASTFunction)
    assert ast.functions[0].name == "foo"
    assert isinstance(ast.functions[1], ASTFunction)
    assert ast.functions[1].name == "bar"

def test_function_with_parameters(fern_parser):
    source = """
    fn add(a: int, b: int) -> int {
        return a + b;
    }
    """
    ast = fern_parser.parse(source)
    function = ast.functions[0]
    assert len(function.parameters) == 2
    param_a = function.parameters[0]
    param_b = function.parameters[1]

    assert isinstance(param_a, ASTParameter)
    assert param_a.name == "a"
    assert isinstance(param_a.type, ASTType)
    assert param_a.type.name == "int"

    assert isinstance(param_b, ASTParameter)
    assert param_b.name == "b"
    assert isinstance(param_b.type, ASTType)
    assert param_b.type.name == "int"

def test_return_statement(fern_parser):
    source = """
    fn test() -> void {
        return;
    }
    fn test2() -> int {
        return 5;
    }
    """
    ast = fern_parser.parse(source)
    block1 = ast.functions[0].body
    assert len(block1.statements) == 1
    ret1 = block1.statements[0]
    assert isinstance(ret1, ASTReturnStatement)
    assert ret1.expression is None

    block2 = ast.functions[1].body
    assert len(block2.statements) == 1
    ret2 = block2.statements[0]
    assert isinstance(ret2, ASTReturnStatement)
    assert isinstance(ret2.expression, ASTIntegerLiteralExpression)
    assert ret2.expression.value == 5

def test_var_declaration(fern_parser):
    source = """
    fn main() -> void {
        var x;
        var y : int;
        var z = 10;
        var a : bool = true;
    }
    """
    ast = fern_parser.parse(source)
    block = ast.functions[0].body
    assert len(block.statements) == 4

    stmt1 = block.statements[0]
    assert isinstance(stmt1, ASTVariableDeclarationStatement)
    assert stmt1.name == "x"
    assert stmt1.type is None
    assert stmt1.initial_value is None

    stmt2 = block.statements[1]
    assert isinstance(stmt2, ASTVariableDeclarationStatement)
    assert stmt2.name == "y"
    assert isinstance(stmt2.type, ASTType)
    assert stmt2.type.name == "int"
    assert stmt2.initial_value is None

    stmt3 = block.statements[2]
    assert isinstance(stmt3, ASTVariableDeclarationStatement)
    assert stmt3.name == "z"
    assert stmt3.type is None
    assert isinstance(stmt3.initial_value, ASTIntegerLiteralExpression)
    assert stmt3.initial_value.value == 10

    stmt4 = block.statements[3]
    assert isinstance(stmt4, ASTVariableDeclarationStatement)
    assert stmt4.name == "a"
    assert isinstance(stmt4.type, ASTType)
    assert stmt4.type.name == "bool"
    assert isinstance(stmt4.initial_value, ASTBooleanLiteralExpression)
    assert stmt4.initial_value.value is True

def test_assignment_statement(fern_parser):
    source = """
    fn main() -> void {
        x = 5 + y;
    }
    """
    ast = fern_parser.parse(source)
    block = ast.functions[0].body
    assert len(block.statements) == 1
    stmt = block.statements[0]
    assert isinstance(stmt, ASTAssignmentStatement)
    assert stmt.name == "x"
    assert isinstance(stmt.expression, ASTBinaryExpression)
    assert stmt.expression.operator == ASTBinaryOperator.ADD

def test_basic_expressions(fern_parser):
    source = """
    fn main() -> void {
        a = 123;
        b = true;
        c = false;
        d = myVar;
    }
    """
    ast = fern_parser.parse(source)
    block = ast.functions[0].body
    assert isinstance(block.statements[0].expression, ASTIntegerLiteralExpression)
    assert block.statements[0].expression.value == 123
    assert isinstance(block.statements[1].expression, ASTBooleanLiteralExpression)
    assert block.statements[1].expression.value is True
    assert isinstance(block.statements[2].expression, ASTBooleanLiteralExpression)
    assert block.statements[2].expression.value is False
    assert isinstance(block.statements[3].expression, ASTVariableReferenceExpression)
    assert block.statements[3].expression.name == "myVar"

def test_binary_operations(fern_parser):
    source = """
    fn main() -> void {
        res = 1 + 2 * 3;
    }
    """
    ast = fern_parser.parse(source)
    block = ast.functions[0].body
    expr = block.statements[0].expression
    assert isinstance(expr, ASTBinaryExpression)
    assert expr.operator == ASTBinaryOperator.ADD
    assert isinstance(expr.left, ASTIntegerLiteralExpression)
    assert expr.left.value == 1
    assert isinstance(expr.right, ASTBinaryExpression)
    assert expr.right.operator == ASTBinaryOperator.MULTIPLY
    assert isinstance(expr.right.left, ASTIntegerLiteralExpression)
    assert expr.right.left.value == 2
    assert isinstance(expr.right.right, ASTIntegerLiteralExpression)
    assert expr.right.right.value == 3

def test_unary_operations(fern_parser):
    source = """
    fn main() -> void {
        a = !true;
        b = -5;
    }
    """
    ast = fern_parser.parse(source)
    block = ast.functions[0].body

    expr_a = block.statements[0].expression
    assert isinstance(expr_a, ASTUnaryExpression)
    assert expr_a.operator == ASTUnaryOperator.NOT
    assert isinstance(expr_a.operand, ASTBooleanLiteralExpression)
    assert expr_a.operand.value is True

    expr_b = block.statements[1].expression
    assert isinstance(expr_b, ASTUnaryExpression)
    assert expr_b.operator == ASTUnaryOperator.NEGATE
    assert isinstance(expr_b.operand, ASTIntegerLiteralExpression)
    assert expr_b.operand.value == 5

def test_function_call(fern_parser):
    source = """
    fn main() -> void {
        dummy = do_nothing();
        result = calculate(x, 5 + y, true);
    }
    """
    ast = fern_parser.parse(source)
    block = ast.functions[0].body

    call1_expr = block.statements[0].expression
    assert isinstance(call1_expr, ASTFunctionCallExpression)
    assert call1_expr.name == "do_nothing"
    assert hasattr(call1_expr, 'arguments')
    assert len(call1_expr.arguments) == 0

    call2_expr = block.statements[1].expression
    assert isinstance(call2_expr, ASTFunctionCallExpression)
    assert call2_expr.name == "calculate"
    assert len(call2_expr.arguments) == 3
    assert isinstance(call2_expr.arguments[0], ASTVariableReferenceExpression)
    assert call2_expr.arguments[0].name == "x"
    assert isinstance(call2_expr.arguments[1], ASTBinaryExpression)
    assert call2_expr.arguments[1].operator == ASTBinaryOperator.ADD
    assert isinstance(call2_expr.arguments[2], ASTBooleanLiteralExpression)
    assert call2_expr.arguments[2].value is True

def test_if_statement(fern_parser):
    source = """
    fn main() -> void {
        if (x > 0) {
            y = 1;
        }
    }
    """
    ast = fern_parser.parse(source)
    block = ast.functions[0].body
    assert len(block.statements) == 1
    if_stmt = block.statements[0]

    assert isinstance(if_stmt, ASTIfStatement)
    assert isinstance(if_stmt.condition, ASTBinaryExpression)
    assert if_stmt.condition.operator == ASTBinaryOperator.GREATER
    assert isinstance(if_stmt.condition.left, ASTVariableReferenceExpression)
    assert if_stmt.condition.left.name == "x"
    assert isinstance(if_stmt.condition.right, ASTIntegerLiteralExpression)
    assert if_stmt.condition.right.value == 0
    assert isinstance(if_stmt.then_block, ASTBlock)
    assert len(if_stmt.then_block.statements) == 1
    then_stmt = if_stmt.then_block.statements[0]
    assert isinstance(then_stmt, ASTAssignmentStatement)
    assert then_stmt.name == "y"
    assert isinstance(then_stmt.expression, ASTIntegerLiteralExpression)
    assert then_stmt.expression.value == 1
    assert if_stmt.else_block is None

def test_if_else_statement_raw_tree(fern_parser):
    source = """
    fn main() -> void {
        if (x) {
            y = 1;
        } else {
            y = 0;
        }
    }
    """
    lark_parser = fern_parser.parser
    tree = lark_parser.parse(source)

    if_stmt_tree = tree.children[0].children[-1].children[0]
    assert isinstance(if_stmt_tree, Tree)
    assert if_stmt_tree.data == "if_stmt"
    assert len(if_stmt_tree.children) == 3
    assert isinstance(if_stmt_tree.children[0], Tree)
    assert if_stmt_tree.children[0].data == "var"
    assert isinstance(if_stmt_tree.children[1], Tree)
    assert if_stmt_tree.children[1].data == "block"
    assert isinstance(if_stmt_tree.children[2], Tree)
    assert if_stmt_tree.children[2].data == "block"

def test_if_else_if_statement_raw_tree(fern_parser):
    source = """
    fn main() -> void {
        if (a) {
            x = 1;
        } else if (b) {
            x = 2;
        } else if (c) {
            x = 3;
        } else {
            x = 4;
        }
    }
    """
    lark_parser = fern_parser.parser
    tree = lark_parser.parse(source)
    if_stmt_tree = tree.children[0].children[-1].children[0]
    assert isinstance(if_stmt_tree, Tree)
    assert if_stmt_tree.data == "if_stmt"
    assert len(if_stmt_tree.children) == 7
    assert if_stmt_tree.children[0].data == "var"
    assert if_stmt_tree.children[1].data == "block"
    assert if_stmt_tree.children[2].data == "var"
    assert if_stmt_tree.children[3].data == "block"
    assert if_stmt_tree.children[4].data == "var"
    assert if_stmt_tree.children[5].data == "block"
    assert if_stmt_tree.children[6].data == "block"

def test_syntax_error(fern_parser):
    source = """
    fn main() -> int {
        return 1 +;
    }
    """
    with pytest.raises(LarkError):
        fern_parser.parse(source)

def test_missing_semicolon(fern_parser):
    source = """
    fn main() -> void {
        x = 5
    }
    """
    with pytest.raises(LarkError):
        fern_parser.parse(source)

def test_unbalanced_parentheses(fern_parser):
    source = """
    fn main() -> void {
        x = (5 + 2;
    }
    """
    with pytest.raises(LarkError):
        fern_parser.parse(source)

def test_invalid_standalone_call(fern_parser):
    source = """
    fn main() -> void {
        my_func();
    }
    """
    with pytest.raises(LarkError):
        fern_parser.parse(source)
