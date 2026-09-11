# String Design Options

This note preserves two competing proposals for Fern strings. It is not a
language contract. [The language specification](../docs/spec.md) remains the
authority for Fern behavior.

The proposals differ on whether `string` is principally an immutable byte view
with text conveniences or a managed Unicode text value. Both keep binary data
and C interoperability available, but they put the boundary in different
places.

## Proposal One: Immutable Byte Views

### Purpose

`string` is an immutable view of an arbitrary sequence of bytes. It is useful
for text, operating-system data, protocol fields, and other byte sequences for
which comparison and slicing are convenient. UTF-8 is a supported
interpretation, not a validity invariant.

This follows the broad model used by Go and Odin: length and indexing are in
bytes, while iteration can decode UTF-8 runes. Malformed UTF-8 remains a valid
string value and has defined decoding behavior.

### Values and storage

A string contains a reference to bytes and a byte length. Copying a string
copies this descriptor and does not copy the bytes.

Strings do not own their storage. The storage must remain valid for every use
of the string and every slice derived from it. This is the same unchecked
pointer-validity obligation that governs pointers and slices.

Immutability is an access property: bytes cannot be changed through a string.
If the same storage is also reachable through a mutable byte slice, later
writes through that slice are visible through the string. String iteration
captures the descriptor once but reads bytes as it proceeds.

### Literals

Interpreted and raw literals produce string constants. Interpreted literals
encode source scalars and Unicode escapes as UTF-8. They also support `\xHH`,
which contributes exactly one byte and permits literals to express the entire
string value domain.

String constant expressions support `+`. Concatenation is performed during
compilation and produces another string constant. Runtime `+` is not defined,
so the operator never introduces an implicit runtime allocation.

### Conversion and construction

`[]const u8(s)` produces a byte slice that aliases a string's storage.
`string(bytes)`, where `bytes` has type `[]const u8`, produces a string that
aliases the slice's storage. Neither operation copies or allocates.

Runtime code constructs owned string data by allocating and filling byte
storage, then creating a string view over it. The library API that allocates
the storage also defines how the caller releases it. Concatenation, formatting,
and rune encoding are library operations with explicit allocation behavior.

### Operations

- `len(s)` returns the number of bytes in `s`.
- `s[i]` returns the `u8` at byte index `i`.
- `s[lo:hi]` returns a non-copying byte substring and does not require UTF-8
  boundaries.
- Comparisons are lexicographic by unsigned byte value.
- `for r, i in s` decodes UTF-8. `r` is a rune and `i` is its starting byte
  index. Malformed input yields U+FFFD and advances one byte.
- `for b in []const u8(s)` iterates the exact bytes.

Indexing and slicing never allocate. String operations remain constant-time
where the corresponding byte-slice operations are constant-time.

### C interoperability

No NUL terminator or accessible byte after `len(s)` is guaranteed. A foreign C
signature does not accept a Fern string directly. Code passes an explicit
pointer and length or uses a library operation to construct a NUL-terminated
buffer. Constructing a C string rejects an embedded NUL that would truncate the
value.

### Consequences

This design is transparent and cheap. It represents arbitrary host and
protocol data without another scalar type, and it does not require managed
values in the compiler.

Its cost is that ordinary string code must understand byte offsets. Slicing can
create malformed UTF-8, UTF-8 iteration is lossy on malformed data, and storage
lifetime remains a manual obligation. The type prevents mutation through a
string but does not establish stable or independently owned contents.

## Proposal Two: Managed UTF-8 Text

### Purpose

`string` is a high-level immutable text value. Every string contains valid
UTF-8. Ordinary programs operate in runes and do not need to understand the
encoding. Arbitrary bytes remain byte arrays and slices.

This follows Hare's validity boundary while giving more string operations
language-level support. It differs from Go and Odin, whose ordinary indexing
and slicing expose byte offsets.

### Values and storage

String contents live in runtime-managed immutable storage. Assignment,
argument passing, return, and aggregate copying preserve the value without
requiring the programmer to manage its lifetime.

A suitable initial implementation is reference-counted string storage.
Strings cannot create reference cycles through their byte buffers, literals
can use immortal storage, and releasing the last value can deterministically
release a runtime allocation. Because arrays and structs may contain strings,
the compiler and runtime must apply string retain and release behavior
recursively when those aggregate values are copied or cease to exist.

The language specifies the value and lifetime behavior rather than a public
descriptor layout. Runtime representation can change without changing the
language.

### Literals

Interpreted and raw literals produce valid UTF-8 string constants. Unicode
escapes contribute the UTF-8 encoding of a scalar. There is no raw-byte escape:
bytes that do not encode valid text belong in a byte array or slice.

`+` concatenates strings. Constant operands are folded during compilation.
Otherwise, concatenation may allocate runtime-managed storage. Allocation is
part of the operator's documented cost.

### Conversion and construction

Converting bytes to a string validates the complete UTF-8 encoding and copies
the result into managed immutable storage. Invalid UTF-8 cannot produce a
string. The eventual error-result facilities determine how validation failure
is represented; an initial checked conversion may trap.

Encoding a rune or a rune sequence as a string produces its UTF-8 text and may
allocate. Formatting, joining, replacement, and builder APIs live in the
standard library and return managed strings.

There is no unchecked conversion that violates the UTF-8 invariant. Code that
needs to preserve or manipulate malformed encodings continues using bytes.

### Operations

- `len(s)` returns the number of runes in `s`.
- `s[i]` returns the rune at rune index `i`.
- `s[lo:hi]` returns the substring selected by rune indices. The result is
  valid UTF-8 and may allocate.
- `for r, i in s` iterates runes, with `i` as the rune index.
- Equality compares the encoded text exactly, without Unicode normalization.
- Ordering, if retained as an operator, compares Unicode scalar sequences
  lexicographically rather than locale-sensitive text.

Length, indexing, and slicing may require a linear scan. Repeated indexing may
therefore be quadratic. Iteration is the preferred traversal operation. User-
perceived characters can contain multiple runes, so grapheme segmentation and
locale-aware comparison remain standard-library concerns.

### Byte and C interoperability

An explicit byte-view operation exposes the UTF-8 encoding for low-level APIs.
The view cannot outlive the string storage that backs it; the runtime or the
foreign-call boundary must keep that storage alive for the operation.

C interoperability uses an explicit conversion that allocates a NUL-terminated
copy when necessary. A separate C-string library type may describe that buffer,
but ordinary Fern strings neither contain a terminator nor admit embedded
invalid UTF-8. Constructing a C string rejects an embedded NUL that would
truncate the value.

### Consequences

This design makes the common text operations direct and preserves the UTF-8
invariant across every language operation. Concatenation, slicing, and
conversion can allocate, and ordinary programs do not manage string storage.

Its cost is a managed-value boundary in an otherwise explicit low-level
language. String fields make aggregate copying and destruction non-trivial,
rune indexing hides linear work, binary and operating-system data require byte
APIs, and FFI needs explicit encoding and lifetime bridges.

## Decisions That Separate the Proposals

| Decision | Immutable byte views | Managed UTF-8 text |
| --- | --- | --- |
| Valid values | Any bytes | Valid UTF-8 only |
| Storage | Borrowed | Runtime-managed |
| Copy | Descriptor copy | Managed shared value |
| `len` and indices | Bytes | Runes |
| Slicing | Aliases bytes | Rune-based, may allocate |
| Runtime `+` | Not defined | Allocating concatenation |
| Byte-to-string | Aliases | Validates and copies |
| Malformed UTF-8 | Valid data | Cannot inhabit `string` |
| Lifetime | Pointer-validity obligation | Automatic |
| FFI | Direct byte view, manual lifetime | Explicit encoding bridge |

The first proposal keeps strings within Fern's existing pointer and slice
model. The second makes strings the first runtime-managed Fern value. Choosing
between them therefore decides more than Unicode behavior: it determines
whether ordinary string use participates in manual storage lifetime or creates
a new automatic lifetime mechanism for values and aggregates.

## Reference Models

- [The Go Programming Language Specification](https://go.dev/ref/spec)
- [Odin language overview](https://odin-lang.org/docs/overview/)
- [Hare language specification](https://harelang.org/specification.pdf)
- [Hare strings library](https://docs.harelang.org/strings/)
