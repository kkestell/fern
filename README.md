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
checked and truncating forms. The nonempty examples exit with status 42.

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

The example exits with status zero. The CLI is `fern <input.fern> -o <output>`.
The output directory must already exist. Prefix filenames beginning with `-`
with `./`.

`QBE` and `CC` can each specify a tool executable name or path. They do not
accept command-line flags. QBE must default to the development host's target;
cross-compilation is not supported.

Source diagnostics include the input path and source location. Invalid UTF-8
reports a zero-based byte offset. File and tool failures return a nonzero
compiler status; tool failures include stderr. The compiler uses temporary
intermediates beside the requested output and publishes the executable only
after QBE and the C toolchain succeed. A failed compilation preserves an
existing output, and an output aliasing the input is rejected.

The compiler accepts up to 128 enclosing blocks and conversion expressions
combined, counting the function body. Deeper nesting produces a source
diagnostic before recursive compiler phases can exhaust the stack.

## Development

See [AGENTS.md](AGENTS.md#development-commands) for development commands. Native
integration tests require real QBE and C tools and fail if they are unavailable.
Run tests as an ordinary user so filesystem permission tests are meaningful.
