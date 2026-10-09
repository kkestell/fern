# String and character literals

## Plan

`agents/plans/2026-10-09-004-string-and-character-literals.md`

## Summary

String and character literals lex, decode, check, lower, and emit as the plan
describes. `extern fn puts(text: *u8) -> i32; puts("hi");` prints `hi`, and
equal literals share one static entry. The plan's goal is met.

## Departures from the plan

- An `\x` escape without two hexadecimal digits reports the digits it found,
  such as ``invalid escape sequence `\x4` ``, under the plan's
  `invalid escape sequence` message.
- `verify_literal` and `flow_operand_type` take `Program.strings` so the
  verifier can check a string ID; `verify_literal` had no access to the
  program.
- `run` in `tests/core.rs` returns the program's `Output` so
  `string_literals_reach_c` can assert standard output.

## Decisions

- The lexer callback `quoted` is generic over its delimiter, so one function
  scans both tokens. A backslash before a line break does not skip it, so
  `"a\` followed by a newline is unterminated rather than an invalid escape.
- A character literal in a comparison lets `unify_comparison` choose the
  numeric type before the operands are concretized, so `97 == 'a'` uses `u8`.

## Checks run

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `cargo test` — passed.

## Manual verification

1. The example prints through `puts` and compares bytes with character
   literals.

   ```sh
   cargo run -- examples/strings.fern -o target/strings && target/strings
   ```

   Printed `hello	world!` and `it's "quoted"`, and exited 0.

2. String literals inside constant arrays, module-level arrays, and constant
   structs lower to static data and compare at run time.

   ```fern
   extern fn puts(text: *u8) -> i32;
   type S struct { p: *u8 }
   const names = ["one", "two"];
   var table: [2]*u8 = ["three", "four"];
   fn main() -> void {
       puts(names[1]);
       puts(table[0]);
       const s = S { p = "five" };
       if s == s && names == names {
           puts(s.p);
       }
   }
   ```

   Printed `two`, `three`, and `five`.
