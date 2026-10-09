# String and character literals

## Goal

Implement the string and character literals in `docs/spec.md#source-and-syntax`.
A character literal is an integer literal holding one byte. It defaults to
`u8`, and in literal arithmetic with no other type it selects `u8` ahead of
`int`. A string literal is a constant `*u8` that points to static, zero-terminated
bytes. It is a constant expression, but comparing, converting, offsetting,
indexing, or dereferencing it is not. When the work is done,
`extern fn puts(text: *u8) -> i32; puts("hi");` prints `hi`.

## Related code

- `src/frontend/lexer.rs` — the `Token` enum, the `block_comment` callback
  pattern, and `valid_number`, which the parser calls on each number token.
- `src/frontend/parser.rs` `advance` — validates number tokens and reports
  lexical errors; `primary_expression` builds literal expressions.
- `src/frontend/syntax.rs` — `ExpressionKind` and `walk_expression`.
- `src/semantic/model.rs` — `Constant`, `ExpressionValue`, `CheckedProgram`.
- `src/semantic/expressions.rs` — `infer_expression`, `type_hint`,
  `has_float_literal` and its callers in `infer_unary_context` and
  `infer_binary_context`, `unify_comparison`, `infer_comparison`, and
  `infer_pointer_conversion`, which passes its operand's constant through.
- `src/semantic/constants.rs` — `compare_constants`, `integer_literal`.
- `src/types.rs` — `StructId`, the pattern for an ID shared by the semantic
  model and the IR.
- `src/ir/model.rs` — `Literal`, `Program`.
- `src/ir/lower.rs` `lower`, `literal`, `static_values` — turn constants into
  operands and static data; `lower_folded_effects` and `lower_flow_operand`
  list the literal `ExpressionValue`s.
- `src/ir/verify.rs` `verify_literal`.
- `src/backend/qbe.rs` `operand`, `data_item`; `src/backend/emitter.rs`
  `emit_data` and the program loop that emits globals.

## Decisions

- **Lexing.** `Token::String` and `Token::Character` start at `"` and `'`. A
  callback scans to the matching unescaped delimiter, skipping the byte after
  each backslash. A newline, a carriage return, or the end of input before the
  delimiter is an error, which `advance` reports as
  ``unterminated string literal`` or ``unterminated character literal``.
- **Decoding.** `lexer::literal_bytes(body: &str) -> Result<Vec<u8>, String>`
  decodes the text between the delimiters. It copies every source byte except
  a backslash, which must begin `\n`, `\r`, `\t`, `\0`, `\\`, `\"`, `\'`, or
  `\x` followed by exactly two hexadecimal digits; anything else is
  `` invalid escape sequence `\q` ``. `advance` validates each string and
  character token with it, and a character literal that decodes to anything
  other than exactly one byte is ``character literal must be exactly one byte``.
  `primary_expression` decodes again, as it does with numbers after
  `valid_number`, and builds `ExpressionKind::String(Vec<u8>)` or
  `ExpressionKind::Character(u8)`.
- **Character typing.** A character literal checks as an untyped integer with
  type `u8`, `ExpressionValue::Integer`, and its byte as the constant. So
  `check_expression` defaults it to `u8` and `concretize` rejects a value that
  does not fit the type from context. `type_hint` returns `None` for it, as it
  does for an integer literal. A new `has_character_literal`, shaped like
  `has_float_literal`, makes the fallbacks in `infer_unary_context` and
  `infer_binary_context` select `u8` after `f64` and before `int`.
  `unify_comparison` selects `u8` for two untyped operands when either is
  `u8` and neither is floating.
- **String constants.** `StringId` in `src/types.rs` names one distinct byte
  sequence, without its terminator. `CheckedProgram` gains
  `strings: Vec<Vec<u8>>`, indexed by `StringId`, and a checking-state map from
  bytes to `StringId`. Checking a string literal interns its bytes and yields
  type `*u8`, `ExpressionValue::String`, and constant `Constant::String(id)`.
  Interning by content keeps one address for one literal across every
  `const` use the constant is folded into. The spec leaves sharing between
  distinct literals unspecified, so equal literals share one `StringId`.
- **Address-dependent operations do not fold.** `Constant::has_string`
  reports whether a constant is or contains a `Constant::String`.
  `infer_comparison` folds only when neither operand's constant has a string.
  `infer_pointer_conversion` keeps its operand's constant only when it is
  `Constant::Null`. Dereference, pointer indexing, and pointer offsets already
  yield no constant.
- **IR.** `Literal::String(StringId)` has type `*u8`. `lower` copies
  `checked.strings` into `Program.strings`, and `literal` maps
  `Constant::String(id)` to `Literal::String(id)`. `static_values` treats a
  string literal as nonzero. Verification checks that the ID names an entry of
  `Program.strings`.
- **Backend.** String `id` is the symbol `.fern.string{id}`, emitted once per
  entry as `data $.fern.string{id} = { b … 0 }` with its bytes in decimal and
  the terminating zero. `operand` spells `Literal::String` as the symbol and
  `data_item` as `l $.fern.string{id}`.

## Test plan

- `src/frontend/tests/lexer.rs`: string and character tokens decode every
  escape, including `\xNN`, and copy non-ASCII text as UTF-8 bytes. A string
  may hold an unescaped `'` and a character literal an unescaped `"`. Errors:
  a newline inside a literal, end of input, `\q`, `\x4`, `''`, `'ab'`, and `'é'`.
- `src/frontend/tests/mod.rs`: add a string and a character literal to the
  `core_syntax` snapshot source.
- `src/semantic/tests/mod.rs`:
  - `constants` folds `const c='a';`, `const d='a'+1;`, and
    `const e:i32='\xFF';` to 97, 98, and 255.
  - `var c='a';` and `var d='a'+1;` check as `u8`; `var e='a'+1.0;` is `f64`.
  - `const s="hi";`, `var t=s;`, and a module-level `var m:*u8="hi";`
    check.
  - Rejections: `var x:i8='\xFF';`, `var p:*i8="x";`, `var p=&"x";`,
    `const b="a"=="a";`, `const p=*i8("a");`, `const c=*"a";`,
    `const q="ab"+1;`, and `var n=len("a");`.
- `src/ir/tests/mod.rs`: two equal literals in different functions and one
  in a module-level `var` lower to one `Program.strings` entry, and the
  program verifies.
- `tests/core.rs`:
  - `string_literals_are_static_terminated_bytes` walks `"a\tb\x21\0c"` and
    `"é"` through pointers and indexing, checks each byte and the
    terminator, and checks that a literal named by a `const` and used in two
    functions has one address.
  - `string_literals_reach_c` calls `puts` with a module-level `*u8` and a
    literal argument, and asserts the program's standard output.

## Implementation plan

1. Lexer: add `Token::String`, `Token::Character`, the scanning callback, and
   `literal_bytes` in `src/frontend/lexer.rs`.
2. Parser and syntax: validate the tokens in `advance`, build the new
   `ExpressionKind` variants in `primary_expression`, and add them to
   `walk_expression`'s leaves.
3. Semantic: add `StringId` to `src/types.rs`, `Constant::String`,
   `Constant::has_string`, `ExpressionValue::String`, and the string table on
   `CheckedProgram`. Check both literals in `infer_expression`, add
   `has_character_literal`, and update `type_hint`, the unary, binary, and
   comparison fallbacks, comparison folding, and `infer_pointer_conversion`.
4. IR: add `Literal::String` and `Program.strings`, and update `lower`,
   `literal`, `static_values`, the literal lists in `lower_folded_effects`
   and `lower_flow_operand`, and `verify_literal`.
5. Backend: emit string data and spell string operands and data items.
6. Add the tests from the test plan.
7. Add `examples/strings.fern`, which prints through `puts` and compares
   bytes with character literals, and list it in `README.md` after Pointers.

## Documentation updates

- `README.md` — the Strings entry in the example tour.
