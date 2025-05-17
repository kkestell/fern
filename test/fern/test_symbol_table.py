import pytest

from src.fern.ast_nodes import (
    ASTProgram, ASTFunction, ASTBlock, ASTReturnStatement,
    ASTVariableDeclarationStatement, ASTAssignmentStatement, ASTIfStatement,
    ASTBinaryExpression, ASTUnaryExpression, ASTIntegerLiteralExpression,
    ASTBooleanLiteralExpression, ASTVariableReferenceExpression,
    ASTFunctionCallExpression, ASTType, ASTParameter, ASTExpression
)
from src.fern.symbol_table import (
    SymbolTable, Scope, Symbol, SymbolKind, VariableSymbol, FunctionSymbol
)
from src.fern.symbol_table_builder import SymbolTableBuilder


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
def dummy_expr() -> ASTExpression:
    return ASTIntegerLiteralExpression(value=0)


def test_scope_define_resolve_current(int_type):
    scope = Scope()
    symbol = VariableSymbol(name="x", var_type=int_type)
    scope.define(symbol)
    resolved = scope.resolve("x")
    assert resolved is symbol
    assert resolved.name == "x"
    assert resolved.kind == SymbolKind.VARIABLE
    assert resolved.symbol_type == int_type


def test_scope_define_redefinition():
    scope = Scope()
    scope.define(VariableSymbol(name="x", var_type=ASTType(name="int")))
    with pytest.raises(Exception, match="Symbol 'x' already defined"):
        scope.define(VariableSymbol(name="x", var_type=ASTType(name="float")))


def test_scope_resolve_parent(int_type, bool_type):
    parent_scope = Scope()
    child_scope = Scope(parent=parent_scope)
    parent_scope.add_child(child_scope)

    parent_symbol = VariableSymbol(name="x", var_type=int_type)
    child_symbol = VariableSymbol(name="y", var_type=bool_type)

    parent_scope.define(parent_symbol)
    child_scope.define(child_symbol)

    assert child_scope.resolve("y") is child_symbol
    assert child_scope.resolve("x") is parent_symbol
    assert parent_scope.resolve("y") is None
    assert parent_scope.resolve("x") is parent_symbol


def test_scope_resolve_shadowing(int_type, bool_type):
    parent_scope = Scope()
    child_scope = Scope(parent=parent_scope)
    parent_scope.add_child(child_scope)

    parent_symbol = VariableSymbol(name="x", var_type=int_type)
    child_symbol = VariableSymbol(name="x", var_type=bool_type)

    parent_scope.define(parent_symbol)
    child_scope.define(child_symbol)

    resolved_in_child = child_scope.resolve("x")
    assert resolved_in_child is child_symbol
    assert resolved_in_child.symbol_type == bool_type

    resolved_in_parent = parent_scope.resolve("x")
    assert resolved_in_parent is parent_symbol
    assert resolved_in_parent.symbol_type == int_type


def test_scope_resolve_not_found():
    scope = Scope()
    assert scope.resolve("non_existent") is None


def test_symbol_table_initialization():
    st = SymbolTable()
    assert st.global_scope is not None
    assert st._current_scope is st.global_scope
    assert st.global_scope.parent is None


def test_symbol_table_enter_exit_scope():
    st = SymbolTable()
    initial_scope = st._current_scope
    st.enter_scope()
    function_scope = st._current_scope
    assert function_scope is not None
    assert function_scope.parent is initial_scope
    assert function_scope in initial_scope.children

    st.enter_scope()
    block_scope = st._current_scope
    assert block_scope is not None
    assert block_scope.parent is function_scope
    assert block_scope in function_scope.children

    st.exit_scope()
    assert st._current_scope is function_scope

    st.exit_scope()
    assert st._current_scope is initial_scope


def test_symbol_table_exit_global_scope():
    st = SymbolTable()
    with pytest.raises(Exception, match="Cannot exit from the global scope"):
        st.exit_scope()


def test_symbol_table_define_resolve(int_type, bool_type):
    st = SymbolTable()
    global_var = VariableSymbol(name="g", var_type=int_type)
    st.define(global_var)

    st.enter_scope()
    func_var = VariableSymbol(name="f", var_type=bool_type)
    st.define(func_var)

    st.enter_scope()
    block_var = VariableSymbol(name="b", var_type=int_type)
    st.define(block_var)

    assert st.resolve("b") is block_var
    assert st.resolve("f") is func_var
    assert st.resolve("g") is global_var

    st.exit_scope()
    assert st.resolve("b") is None
    assert st.resolve("f") is func_var
    assert st.resolve("g") is global_var

    st.exit_scope()
    assert st.resolve("b") is None
    assert st.resolve("f") is None
    assert st.resolve("g") is global_var


@pytest.fixture
def builder() -> SymbolTableBuilder:
    return SymbolTableBuilder()


def test_builder_empty_program(builder):
    program = ASTProgram(functions=[])
    st = builder.build(program)
    assert st.global_scope.symbols == {}
    assert st.global_scope.children == []


def test_builder_simple_function(builder, void_type):
    func = ASTFunction(
        name="main",
        parameters=[],
        return_type=void_type,
        body=ASTBlock(statements=[])
    )
    program = ASTProgram(functions=[func])
    st = builder.build(program)

    assert len(st.global_scope.symbols) == 1
    func_symbol = st.global_scope.resolve("main")
    assert isinstance(func_symbol, FunctionSymbol)
    assert func_symbol.name == "main"
    assert func_symbol.symbol_type == void_type
    assert func_symbol.parameters == []

    assert len(st.global_scope.children) == 1
    func_scope = st.global_scope.children[0]
    assert func_scope.parent is st.global_scope
    assert func_scope.symbols == {}

    assert len(func_scope.children) == 1
    block_scope = func_scope.children[0]
    assert block_scope.parent is func_scope
    assert block_scope.symbols == {}


def test_builder_function_with_parameters(builder, int_type, bool_type):
    param_a = ASTParameter(name="a", type=int_type)
    param_b = ASTParameter(name="b", type=bool_type)
    func = ASTFunction(
        name="process",
        parameters=[param_a, param_b],
        return_type=int_type,
        body=ASTBlock(statements=[])
    )
    program = ASTProgram(functions=[func])
    st = builder.build(program)

    func_symbol = st.global_scope.resolve("process")
    assert isinstance(func_symbol, FunctionSymbol)
    assert func_symbol.parameters == [param_a, param_b]

    func_scope = st.global_scope.children[0]
    assert len(func_scope.symbols) == 2
    param_a_sym = func_scope.resolve("a")
    param_b_sym = func_scope.resolve("b")

    assert isinstance(param_a_sym, VariableSymbol)
    assert param_a_sym.name == "a"
    assert param_a_sym.symbol_type == int_type

    assert isinstance(param_b_sym, VariableSymbol)
    assert param_b_sym.name == "b"
    assert param_b_sym.symbol_type == bool_type


def test_builder_variable_declaration(builder, int_type, void_type):
    var_decl = ASTVariableDeclarationStatement(name="x", type=int_type, initial_value=None)
    func = ASTFunction(
        name="test",
        parameters=[],
        return_type=void_type,
        body=ASTBlock(statements=[var_decl])
    )
    program = ASTProgram(functions=[func])
    st = builder.build(program)

    func_scope = st.global_scope.children[0]
    block_scope = func_scope.children[0]
    assert len(block_scope.symbols) == 1
    var_x_sym = block_scope.resolve("x")

    assert isinstance(var_x_sym, VariableSymbol)
    assert var_x_sym.name == "x"
    assert var_x_sym.symbol_type == int_type

    assert st.global_scope.resolve("x") is None
    assert func_scope.resolve("x") is None


def test_builder_variable_decl_no_type(builder, void_type):
    var_decl = ASTVariableDeclarationStatement(name="x", type=None, initial_value=None)
    func = ASTFunction(
        name="test",
        parameters=[],
        return_type=void_type,
        body=ASTBlock(statements=[var_decl])
    )
    program = ASTProgram(functions=[func])

    with pytest.raises(ValueError, match="Variable declaration statement must have a type"):
        builder.build(program)


def test_builder_nested_blocks_if(builder, bool_type, int_type, void_type, dummy_expr):
    inner_var_decl = ASTVariableDeclarationStatement(name="y", type=int_type, initial_value=None)
    then_block = ASTBlock(statements=[inner_var_decl])
    outer_var_decl = ASTVariableDeclarationStatement(name="x", type=bool_type, initial_value=None)
    if_stmt = ASTIfStatement(condition=dummy_expr, then_block=then_block, else_block=None)

    func = ASTFunction(
        name="nested",
        parameters=[],
        return_type=void_type,
        body=ASTBlock(statements=[outer_var_decl, if_stmt])
    )
    program = ASTProgram(functions=[func])
    st = builder.build(program)

    func_scope = st.global_scope.children[0]
    outer_block_scope = func_scope.children[0]

    assert len(outer_block_scope.symbols) == 1
    var_x_sym = outer_block_scope.resolve("x")
    assert isinstance(var_x_sym, VariableSymbol)
    assert var_x_sym.symbol_type == bool_type

    assert len(outer_block_scope.children) == 1
    then_scope = outer_block_scope.children[0]
    assert then_scope.parent is outer_block_scope

    assert len(then_scope.symbols) == 1
    var_y_sym = then_scope.resolve("y")
    assert isinstance(var_y_sym, VariableSymbol)
    assert var_y_sym.symbol_type == int_type

    assert then_scope.resolve("x") is var_x_sym
    assert then_scope.resolve("y") is var_y_sym

    assert outer_block_scope.resolve("y") is None


def test_builder_if_else(builder, bool_type, int_type, void_type, dummy_expr):
    then_var_decl = ASTVariableDeclarationStatement(name="t", type=int_type, initial_value=None)
    else_var_decl = ASTVariableDeclarationStatement(name="e", type=bool_type, initial_value=None)
    then_block = ASTBlock(statements=[then_var_decl])
    else_block = ASTBlock(statements=[else_var_decl])
    if_stmt = ASTIfStatement(condition=dummy_expr, then_block=then_block, else_block=else_block)

    func = ASTFunction(
        name="if_else_test",
        parameters=[],
        return_type=void_type,
        body=ASTBlock(statements=[if_stmt])
    )
    program = ASTProgram(functions=[func])
    st = builder.build(program)

    func_scope = st.global_scope.children[0]
    outer_block_scope = func_scope.children[0]

    assert len(outer_block_scope.children) == 2
    then_scope = outer_block_scope.children[0]
    else_scope = outer_block_scope.children[1]

    assert then_scope.parent is outer_block_scope
    assert else_scope.parent is outer_block_scope

    assert len(then_scope.symbols) == 1
    var_t_sym = then_scope.resolve("t")
    assert isinstance(var_t_sym, VariableSymbol)
    assert var_t_sym.symbol_type == int_type
    assert then_scope.resolve("e") is None

    assert len(else_scope.symbols) == 1
    var_e_sym = else_scope.resolve("e")
    assert isinstance(var_e_sym, VariableSymbol)
    assert var_e_sym.symbol_type == bool_type
    assert else_scope.resolve("t") is None

