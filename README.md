# Fern

Fern is a low-level programming language. The
[language specification](docs/spec.md) defines its behavior; the
[roadmap](eng/roadmap.md) tracks implementation scope and completion gates.
The specification includes language features ahead of the compiler. See the
roadmap's completed milestones for the currently supported subset.

See [the empty entry](examples/empty.fern),
[integer literals](examples/integer_literals.fern) for bases, annotations,
comments, and a trailing call comma, and
[shadowing](examples/shadowing.fern) for binding copies and initializer
visibility. See [assignment and scopes](examples/assignment_and_scopes.fern) for
reassignment, saved copies, and nested shadowing, and
[integer types](examples/integer_types.fern) for typed literals and explicit
conversions. See [integer conversions](examples/integer_conversions.fern) for
checked and truncating forms, and
[integer expressions](examples/integer_expressions.fern) for arithmetic,
bitwise, and shift operators. See
[hardened integers](examples/hardened_integers.fern) for a runtime shift with
an inferred count,
[module-level declarations](examples/module_level_declarations.fern) for
file-scope initialization and mutation, and
[branches and loops](examples/branches_and_loops.fern) for boolean expressions,
conditionals, and loop control. See
[parameters, calls, and returns](examples/parameters_calls_and_returns.fern) for
typed parameters, nested and recursive calls, and value and `void` returns.

## Build and run

Install Rust with Cargo, [QBE](https://c9x.me/compile/), and a host C toolchain
providing `cc`. On macOS, install the Xcode command-line tools with
`xcode-select --install` and QBE with `brew install qbe`.

```sh
cargo build
cargo run -- examples/empty.fern -o target/empty
target/empty
echo $?
```

The CLI is `fern <input.fern> -o <output>`. The output directory must already
exist. Prefix filenames beginning with `-` with `./`.

`QBE` and `CC` can each specify a tool executable name or path. They do not
accept command-line flags. QBE must default to the development host's target;
cross-compilation is not supported.

Source diagnostics include the input path and source location. Invalid UTF-8
reports a zero-based byte offset. File and tool failures return a nonzero
compiler status; tool failures include stderr. The compiler uses temporary
intermediates beside the requested output and publishes the executable only
after QBE and the C toolchain succeed. A failed compilation preserves an
existing output, and an output aliasing the input is rejected.

The compiler accepts recursive source nesting up to 128 levels, counting the
function body, blocks, conversions, parenthesized groups, unary operators, and
binary expression-tree depth. Deeper nesting produces a source diagnostic
before recursive compiler phases can exhaust the stack. Untyped constant
left-shift counts above 1,000,000 produce a compiler resource-limit diagnostic.
Integer literals are limited to 4,096 digits, and untyped constant-folding
results are limited to 2,000,000 significant bits. Diagnostics excerpt source
lines longer than 240 bytes.

## Development

See [AGENTS.md](AGENTS.md#development-commands) for development commands. Native
integration tests require real QBE and C tools and fail if they are unavailable.
Run tests as an ordinary user so filesystem permission tests are meaningful.
