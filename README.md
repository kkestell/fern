# Fern

Fern is a low-level programming language. The
[language specification](docs/spec.md) defines its behavior.

The [examples](examples/) form a topic-based tour:

- [Literals](examples/literals.fern)
- [Bindings and scopes](examples/bindings_and_scopes.fern)
- [Integer types and conversions](examples/integer_types_and_conversions.fern)
- [Integer operations](examples/integer_operations.fern)
- [Control flow](examples/control_flow.fern)
- [Functions](examples/functions.fern)
- [Arrays](examples/arrays.fern)
- [Structs](examples/structs.fern)
- [Pointers](examples/pointers.fern)
- [Strings](examples/strings.fern)
- [Floating-point numbers](examples/floating_point.fern)
- [Modules and imports](examples/modules_and_imports/), whose root module is
  `examples/modules_and_imports/app`.

## Build and run

Install Rust with Cargo, [QBE](https://c9x.me/compile/), and a host C toolchain
providing `cc`. On macOS, install the Xcode command-line tools with
`xcode-select --install` and QBE with `brew install qbe`.

```sh
cargo build
cargo run -- examples/literals.fern -o target/literals
target/literals
echo $?
```

The CLI is `fern <root> -o <output>`. The root module argument is either a
directory, whose `.fern` files together form one module, or a single `.fern`
file forming a one-file module. Files in nested directories are not part of the
module, and a directory holding no `.fern` file is rejected. The output
directory must already exist. Prefix filenames beginning with `-` with `./`.

Imports resolve against an ordered list of module search roots. The default
list is one root: the root module directory's parent. `FERNPATH` replaces that
default with its platform-separated entries, in order. Empty entries are
ignored, and a `FERNPATH` left with no entry keeps the default root rather than
searching nowhere. An import path's components name directories beneath a root,
and the first root whose directory holds a `.fern` file is the imported module.
A root that does not exist is skipped. An unresolved import lists every searched
root in search order, and a module dependency cycle reports its chain of
modules.

`QBE` and `CC` can each specify a tool executable name or path. They do not
accept command-line flags. QBE must default to the development host's target;
cross-compilation is not supported.

Source diagnostics include the source file's path and the location within it.
Invalid UTF-8 reports a zero-based byte offset. File and tool failures return a
nonzero compiler status; tool failures include stderr. The compiler uses
temporary intermediates beside the requested output and publishes the executable
only after QBE and the C toolchain succeed. A failed compilation preserves an
existing output, and an output aliasing a source file is rejected.

The compiler uses QBE to emit host assembly, then links it with `cc` and the
host's default C libraries. There is no language runtime or additional library
selection interface.

Compilation limits are 128 levels of recursive source nesting, 128 levels of
inline struct containment, 4,096 digits in an integer literal, and 1 PiB per
aggregate layout. Diagnostics excerpt source lines longer than 240 bytes.

## Development

See [AGENTS.md](AGENTS.md#development-commands) for development commands. Native
integration tests require real QBE and C tools and fail if they are unavailable.
Run tests as an ordinary user so filesystem permission tests are meaningful.
