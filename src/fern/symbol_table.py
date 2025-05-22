from __future__ import annotations

from dataclasses import field, dataclass
from enum import Enum, auto

from src.fern.ast_nodes import ASTParameter, ASTType


class SymbolKind(Enum):
    VARIABLE = auto()
    FUNCTION = auto()


@dataclass(kw_only=True)
class Symbol:
    name: str
    kind: SymbolKind
    symbol_type: ASTType


class VariableSymbol(Symbol):
    def __init__(self, name: str, var_type: ASTType) -> None:
        super().__init__(name=name, kind=SymbolKind.VARIABLE, symbol_type=var_type)


class FunctionSymbol(Symbol):
    def __init__(self, name: str, return_type: ASTType, parameters: list[ASTParameter]) -> None:
        super().__init__(name=name, kind=SymbolKind.FUNCTION, symbol_type=return_type)
        self.parameters = parameters


@dataclass
class Scope:
    parent: Scope | None = None
    symbols: dict[str, Symbol] = field(default_factory=dict)
    children: list[Scope] = field(default_factory=list)

    def define(self, symbol: Symbol) -> None:
        if symbol.name in self.symbols:
            raise Exception(f"Symbol '{symbol.name}' already defined in this scope")
        self.symbols[symbol.name] = symbol

    def resolve(self, name: str) -> Symbol | None:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.resolve(name)
        return None

    def add_child(self, child: 'Scope') -> None:
        if child not in self.children:
            self.children.append(child)
            child.parent = self


class SymbolTable:
    def __init__(self) -> None:
        self.global_scope = Scope()
        self._current_scope = self.global_scope

    def enter_scope(self) -> None:
        new_scope = Scope(parent=self._current_scope)
        self._current_scope.add_child(new_scope)
        self._current_scope = new_scope

    def exit_scope(self) -> None:
        if self._current_scope.parent is not None:
            self._current_scope = self._current_scope.parent
        else:
            raise Exception("Cannot exit from the global scope")

    def define(self, symbol: Symbol) -> None:
        self._current_scope.define(symbol)

    def resolve(self, name: str) -> Symbol | None:
        return self._current_scope.resolve(name)
