# Fern

A multi-stage optimizing compiler for a statically-typed imperative programming language.

This project exists primarily as a learning exercise in compiler design and implementation. While it implements fundamental language features, it's intended for educational purposes and isn't suitable for production use.

## Language Features

The compiler supports core imperative programming constructs including static typing with `int`, `bool`, and `void` types, function definitions with typed parameters, arithmetic and logical operations, control flow statements, and recursion.

## Language Syntax

Fern uses a Lua-inspired syntax with static typing. Here's a quick overview:

Function definitions use the `def` keyword, followed by the function name and typed parameters. The return type comes after the parameter list:

```
def add(x: int, y: int): int
    return x + y
end
```

Variables must be declared with the `var` keyword and require explicit type annotations. They can be initialized at declaration or will be zero-initialized:

```
var x: int = 42    // initialized to 42
var y: int         // zero-initialized
var z: bool
```

Control flow uses `if-then-end` blocks with optional `else` clauses:

```
if x > y then
    return x
else 
    return y
end
```

### Example: Fibonacci Sequence

```
def fib(n: int): int
    if n <= 1 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end

def main(): int
    return fib(10)
end
```

## Compilation Pipeline

The compiler follows a modern compilation pipeline with several optimization stages:

1. Source code is parsed into an Abstract Syntax Tree (AST)
2. A symbol table is generated to track variable and function scopes
3. The AST is converted to Three-Address Code (TAC) for optimization
4. Dead code elimination removes unreachable code paths
5. A Control Flow Graph is built for analysis
6. The program is converted to Static Single Assignment (SSA) form
7. LLVM Intermediate Representation is generated
8. Finally, LLVM compiles the IR to a native executable

## Future Enhancements

- [ ] Type inference
- [ ] Arrays and composite types
- [ ] Loop constructs (while, for)
- [ ] Standard library functions
- [ ] More aggressive optimizations
- [ ] Better error messages

## License

Permission to use, copy, modify, and/or distribute this software for
any purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY
DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
