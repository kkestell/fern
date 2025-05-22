from src.fern.symbol_table import SymbolTable, Scope, FunctionSymbol


def debug_symbol_table(symbol_table: SymbolTable) -> str:
    def format_scope(scope: Scope, indent: int = 0) -> list[str]:
        indent_str = '  ' * indent
        lines = []

        # Add symbols in the current scope
        for name, symbol in scope.symbols.items():
            if isinstance(symbol, FunctionSymbol):
                params = ', '.join(f"{param.name}: {param.type.name}" for param in symbol.parameters)
                lines.append(f"{indent_str}{symbol.name}({params}) -> {symbol.symbol_type.name}")
            else:
                lines.append(f"{indent_str}{symbol.name}: {symbol.symbol_type.name}")

        # Process child scopes with arrows to indicate nested structure
        for child in scope.children:
            lines.append(f"{indent_str}→")
            lines.extend(format_scope(child, indent + 1))
            lines.append(f"{indent_str}←")

        return lines

    # Start formatting from the global scope
    lines = format_scope(symbol_table.global_scope)
    return "\n".join(lines)
