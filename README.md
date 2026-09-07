# Fern

Fern is a low-level programming language. The
[language specification](docs/spec.md) defines its behavior; the
[roadmap](eng/roadmap.md) tracks implementation scope and completion gates.

The compiler checks initialized `var` and `const` bindings, integer literals,
binding references, and `exit` arguments in one parameterless `main` returning
`void`. Its `int` is signed 32-bit, independent of host pointer width. Literals
may be unsuffixed or use `i`; other integer suffixes are not supported yet.

Executable compilation currently supports only an empty body. Nonempty bodies
receive semantic diagnostics first, then an unsupported-lowering diagnostic if
checking succeeds. See [examples/empty.fern](examples/empty.fern).

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
cross-compilation is not supported. Native execution was tested on macOS arm64
with Rust 1.95.0, QBE's `arm64_apple` default target, and Apple clang 21.0.0.

Source diagnostics include the input path and source location. Invalid UTF-8
reports a zero-based byte offset. File and tool failures return a nonzero
compiler status; tool failures include stderr. The compiler uses temporary
intermediates beside the requested output and publishes the executable only
after QBE and the C toolchain succeed. A failed compilation preserves an
existing output, and an output aliasing the input is rejected.

## Development

See [AGENTS.md](AGENTS.md#development-commands) for development commands. Native
integration tests require real QBE and C tools and fail if they are unavailable.
Run tests as an ordinary user so filesystem permission tests are meaningful.
