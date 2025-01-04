import json
from enum import Enum
from typing import Any


def debug_ast(node: Any, indent: int = 2) -> str:
    def ast_to_dict(n: Any) -> Any:
        if isinstance(n, list):
            return [ast_to_dict(child) for child in n]
        elif hasattr(n, '__dataclass_fields__'):
            return {field: ast_to_dict(getattr(n, field)) for field in n.__dataclass_fields__}
        elif isinstance(n, Enum):
            return n.name
        else:
            return n

    return json.dumps(ast_to_dict(node), indent=indent)
