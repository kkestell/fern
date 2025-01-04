from __future__ import annotations

import os
import platform
from ctypes import CFUNCTYPE, c_int
from pathlib import Path

import llvmlite.binding as llvm
from llvmlite import ir
from llvmlite.binding import ExecutionEngine


class LLVMCompiler:
    def __init__(self) -> None:
        """Initialize the LLVMCompiler with necessary LLVM settings."""
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        self.engine = self._create_execution_engine()

        # Get target triple for current platform
        self.target_triple = llvm.get_default_triple()

        # Create target machine
        self.target = llvm.Target.from_triple(self.target_triple)
        self.target_machine = self.target.create_target_machine(
            opt=2,  # Optimization level
            reloc='pic',  # Position independent code
            codemodel='default'
        )

    @staticmethod
    def _create_execution_engine() -> ExecutionEngine:
        """Create an ExecutionEngine for JIT compilation."""
        target = llvm.Target.from_default_triple()
        target_machine = target.create_target_machine()
        backing_mod = llvm.parse_assembly("")
        engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
        return engine

    def compile_ir(self, ir_module: ir.Module) -> llvm.ModuleRef:
        """Compile the LLVM IR module."""
        # Create LLVM module from IR
        mod = llvm.parse_assembly(str(ir_module))
        mod.verify()
        return mod

    def run_ir(self, ir_module: ir.Module) -> int:
        """Run LLVM IR and return the integer result."""
        # Compile IR module and retrieve function pointer
        mod = self.compile_ir(ir_module)
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()

        func_ptr = self.engine.get_function_address("main")
        cfunc = CFUNCTYPE(c_int)(func_ptr)
        result: int = cfunc()
        return result

    def emit_object_file(self, ir_module: ir.Module, output_path: str | Path) -> None:
        """Generate an object file from the IR module."""
        output_path = Path(output_path)
        mod = self.compile_ir(ir_module)

        # Generate object code
        obj_bytes = self.target_machine.emit_object(mod)

        # Write to file
        with open(output_path, 'wb') as f:
            f.write(obj_bytes)

    def create_executable(self, ir_module: ir.Module, output_path: str | Path) -> None:
        """Create a native executable from the IR module."""
        output_path = Path(output_path)
        obj_path = output_path.with_suffix('.o')

        # Generate object file
        self.emit_object_file(ir_module, obj_path)

        # Determine system-specific linker command
        if platform.system() == "Windows":
            link_cmd = (
                f"\"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\Tools\\MSVC\\14.41.34120\\bin\\Hostx64\\x64\\link.exe\" /ENTRY:main /SUBSYSTEM:CONSOLE /NOLOGO "
                f"/OUT:{output_path} {obj_path}"
            )
        else:
            link_cmd = f"cc -o {output_path} {obj_path}"

        # Link the object file
        result = os.system(link_cmd)
        if result != 0:
            raise RuntimeError(f"Linking failed with exit code {result}")

        # Clean up object file
        obj_path.unlink()
