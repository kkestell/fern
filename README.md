# Fern

Fern is a low-level programming language. The
[language specification](docs/spec.md) defines its behavior; the
[roadmap](eng/roadmap.md) tracks implementation scope and completion gates.
The specification includes language features ahead of the compiler. See the
roadmap's completed milestones for the currently supported subset.

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
