import argparse
import platform
import sys
from pathlib import Path

from src.fern.ast_debug import debug_ast
from src.fern.cfg_builder import CFGBuilder
from src.fern.cfg_debug import debug_cfg
from src.fern.llvm_compiler import LLVMCompiler
from src.fern.llvm_generator import LLVMGenerator
from src.fern.llvm_runner import LLVMRunner
from src.fern.parser import Parser
from src.fern.ssa_transformer import SSATransformer
from src.fern.symbol_table_builder import SymbolTableBuilder
from src.fern.symbol_table_debug import debug_symbol_table
from src.fern.tac_optimizer import TACDeadCodeEliminator
from src.fern.tac_transformer import TACTransformer
from src.fern.utils_debug import format_header


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fern",
        description="Fern programming language compiler"
    )
    parser.add_argument(
        "source_file",
        type=Path,
        help="Source file to compile (e.g. test.fern)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output executable name (default: source file name without extension)"
    )
    args = parser.parse_args()

    try:
        source = args.source_file.read_text()
    except Exception as e:
        print(f"Error reading source file: {e}", file=sys.stderr)
        sys.exit(1)

    print(format_header("Source Code"))
    print(source.strip())

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

    output_name = args.output
    if output_name is None:
        output_name = args.source_file.with_suffix('')
        if platform.system() == 'Windows':
            output_name = output_name.with_suffix('.exe')

    llvm_compiler = LLVMCompiler()
    llvm_compiler.create_executable(llvm_generator.module, str(output_name))
    print(format_header("Executable Generated"))
    print(output_name)


if __name__ == "__main__":
    main()
