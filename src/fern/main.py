import textwrap

from ast_debug import debug_ast
from cfg_builder import CFGBuilder
from cfg_debug import debug_cfg
from llvm_compiler import LLVMCompiler
from llvm_generator import LLVMGenerator
from llvm_runner import LLVMRunner
from parser import Parser
from ssa_transformer import SSATransformer
from symbol_table_builder import SymbolTableBuilder
from symbol_table_debug import debug_symbol_table
from tac_optimizer import TACDeadCodeEliminator
from tac_transformer import TACTransformer
from utils_debug import format_header


def main() -> None:
    source = """
    def fib(n: int): int
        if n <= 1 then
            return n
        end
        return fib(n - 1) + fib(n - 2)
    end
    
    def main(): int
        return fib(10)
    end
    """

    print(format_header("Source Code"))
    print(textwrap.dedent(source).strip())

    parser = Parser()
    ast_program = parser.parse(source)
    print(format_header("AST"))
    print(debug_ast(ast_program))

    symbol_table_builder = SymbolTableBuilder()
    symbol_table = symbol_table_builder.build(ast_program)
    print(format_header("Symbol Table"))
    print(debug_symbol_table(symbol_table))

    tac_generator = TACTransformer()
    tac_program = tac_generator.generate(ast_program, symbol_table)
    print(format_header("Three-Address Code"))
    print(tac_program)

    tac_eliminator = TACDeadCodeEliminator()
    tac_program = tac_eliminator.eliminate(tac_program)
    print(format_header("Three-Address Code (After Dead Code Elimination)"))
    print(tac_program)

    cfg_builder = CFGBuilder()
    cfgs = cfg_builder.build(tac_program)
    print(format_header("Control Flow Graphs"))
    for cfg in cfgs:
        print(debug_cfg(cfg))

    ssa_transform = SSATransformer()
    ssa_program = ssa_transform.transform(tac_program, cfgs)
    print(format_header("Static Single Assignment"))
    print(ssa_program)

    llvm_generator = LLVMGenerator()
    llvm_ir = llvm_generator.generate(ssa_program)
    print(format_header("LLVM IR"))
    print(llvm_ir)

    llvm_runner = LLVMRunner()
    result = llvm_runner.run_ir(llvm_generator.module)
    print(format_header("Result"))
    print(result)

    llvm_compiler = LLVMCompiler()
    llvm_compiler.create_executable(llvm_generator.module, "a")
    print(format_header("Executable Generated"))
    print("a.exe")


if __name__ == "__main__":
    main()
