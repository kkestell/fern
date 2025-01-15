from __future__ import annotations

from ctypes import CFUNCTYPE, c_int

import llvmlite.binding as llvm
from llvmlite import ir
from llvmlite.binding import ExecutionEngine


class LLVMRunner:
    def __init__(self) -> None:
        """Initialize the LLVMRunner with necessary LLVM settings."""
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        self.engine = self._create_execution_engine()

    @staticmethod
    def _create_execution_engine() -> ExecutionEngine:
        """Create an ExecutionEngine for JIT compilation."""
        # Create a target machine representing the host
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine()
        # Create execution engine with an empty backing module
        backing_mod = llvm.parse_assembly("")
        engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
        return engine

    def compile_ir(self, ir_module: ir.Module) -> llvm.ModuleRef:
        """Compile the LLVM IR module."""
        # Create LLVM module from IR
        mod = llvm.parse_assembly(str(ir_module))
        mod.verify()
        # Add module and prepare for execution
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()
        return mod

    def run_ir(self, ir_module: ir.Module) -> int:
        """Run LLVM IR and return the integer result."""
        # Compile IR module and retrieve function pointer
        self.compile_ir(ir_module)
        func_ptr = self.engine.get_function_address("main")
        cfunc = CFUNCTYPE(c_int)(func_ptr)
        result: int = cfunc()
        return result
