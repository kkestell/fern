# Fern

Fern is a low-level programming language. The
[language specification](docs/spec.md) defines its behavior; the
[TODO](eng/todo.md) tracks implementation work. The specification includes
language features ahead of the compiler.

The examples form a tour of the currently supported language, grouped by topic:

- [Literals](examples/literals.fern) covers the supported integer literal forms.
- [Bindings and scopes](examples/bindings_and_scopes.fern) covers module and
  local bindings, assignment, nested scopes, and shadowing.
- [Integer types and conversions](examples/integer_types_and_conversions.fern)
  covers fixed-width and native integer types, checked conversions, and
  truncation.
- [Integer operations](examples/integer_operations.fern) covers arithmetic,
  wrapping arithmetic, bitwise operations, and shifts.
- [Control flow](examples/control_flow.fern) covers booleans, conditionals, the
  three `for` forms, and loop control.
- [Functions](examples/functions.fern) covers parameters, nested and recursive
  calls, and value and `void` returns.
- [Arrays](examples/arrays.fern) covers array types, literals and fills,
  indexing, `len`, array comparison, whole-array copies, and both `for … in`
  forms.
- [Structs](examples/structs.fern) covers declarations, literals and fill,
  field selection and assignment, copying, and equality.
- [Pointers](examples/pointers.fern) covers mutable and const pointer types,
  `null`, address-of, explicit and implicit dereference, indirect assignment,
  pointer copies, comparison, and conversion to `uint`.
- [Modules and imports](examples/modules_and_imports/) covers module
  directories, `pub` declarations, whole-module, nested-path, and selective
  imports, and qualified references. Its root module is
  `examples/modules_and_imports/app`.
- [Floating-point numbers](examples/floating_point.fern) covers floating-point
  literal forms, `f32` and `f64` bindings, arithmetic, comparison, and
  conversions between floating-point and integer types.

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

The compiler accepts recursive source nesting up to 128 levels, counting the
function body, blocks, conversions, parenthesized groups, unary operators, and
binary expression-tree depth. Deeper nesting produces a source diagnostic
before recursive compiler phases can exhaust the stack. Inline struct
containment uses the same 128-level limit. Untyped constant left-shift counts
above 1,000,000 produce a compiler resource-limit diagnostic. Integer literals
are limited to 4,096 digits, and untyped constant-folding results are limited
to 2,000,000 significant bits. Diagnostics excerpt source lines longer than
240 bytes. Aggregate layouts are limited to 1 PiB. One initializer's repeated
array fills and struct-literal zero fills expand to at most 1,000,000 scalar
values in total.

## Development

See [AGENTS.md](AGENTS.md#development-commands) for development commands. Native
integration tests require real QBE and C tools and fail if they are unavailable.
Run tests as an ordinary user so filesystem permission tests are meaningful.
