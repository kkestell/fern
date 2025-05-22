# Fern

A multi-stage optimizing compiler for a statically-typed imperative programming language.

## Example

```
fn fib(n: int) -> int
    if n <= 1 then
        return n
    end
    return fib(n - 1) + fib(n - 2)
end

fn main() -> int
    return fib(10)
end
```

```
$ fern example.fern
$ ./example
$ echo $?
55
```

## Compilation Pipeline

The compiler implements a modern compilation pipeline with several optimization stages:

1. Source code is parsed into an Abstract Syntax Tree (AST)
2. A symbol table is generated to track variable and function scopes
3. The AST is converted to Three-Address Code (TAC)
4. Dead code elimination removes unreachable code paths
5. A Control Flow Graph (CFG) is built for analysis
6. Conversion to Static Single Assignment (SSA) form
7. SSA is translated to LLVM Intermediate Representation (IR)
8. Finally, LLVM compiles the IR to a native executable

## TODO

* [ ] Type inference
* [ ] Arrays and composite types
* [ ] Loop constructs (while, for)
* [ ] Standard library functions
* [ ] More aggressive optimizations
* [ ] Better error messages
