# Represent pointer types across the compiler

## Goal

Pointer annotations resolve to a pointer type instead of reporting the
temporary guard. `Type` gains a pointer variant with a target type and
nullability, spelled and compared as `docs/spec.md#pointer-types` defines. A
pointer has the platform pointer width in the shared layout and counts as one
value in the semantic model. Recursive uses such as `next: nullable *Node`
resolve, and declarations of types without a zero value are rejected.

Pointer expressions keep their temporary guard, and the compiler driver
rejects any program that writes a pointer annotation, so no pointer type
reaches IR lowering until the IR task lands.

## Related code

- `src/types.rs` — `Type`, its `Display`, `Type::scalar`, and the unit tests
  for spelling and sameness.
- `src/layout.rs` — `scalar_bytes`, `Layouts::size`, and `Layouts::alignment`.
- `src/semantic/annotations.rs`:
  - `resolve_annotation` — the `Pointer` arm currently returns
    `pointers_not_implemented`.
  - `resolve_struct_fields` and `resolve_fields` — field resolution, which
    reports a struct reached again while `Resolving` as recursive.
  - `validate_aggregate_layout` — fills the `Layouts` cache, so it must only
    see a struct whose fields are final.
- `src/semantic/model.rs` — `Constant`, `CheckedStruct`, `zero_value`,
  `zero_value_unchecked`, and `aggregate_value_count`.
- `src/semantic/namespaces.rs` `check_module` — module initializers are
  checked before `resolve_module_structs`, and signatures and bodies after it.
  `check_module_initializer` handles a module-level declaration without an
  initializer.
- `src/semantic/statements.rs` `check_binding` — a local declaration without
  an initializer.
- `src/semantic/expressions.rs` — the struct-literal fill that gives omitted
  fields `zero_value_unchecked`.
- `src/semantic/constants.rs` — `contextualize_constant`, `converted`, and
  `constants_equal`, which match on `Constant`.
- `src/lib.rs` `compile` — runs semantic checking, then lowering.
- `src/ir/lower.rs` (`scalar_literal`, `flatten`), `src/ir/verify.rs`
  (`scalar_types`, `scalar_count`), `src/backend/qbe.rs`,
  `src/backend/layout.rs` (`class`, `collect_scalar_slots`), and
  `src/backend/emitter.rs` (`emit_aggregate_equal`, `emit_store`,
  `emit_value`) — exhaustive matches on `Type` or `Constant`.
- `src/semantic/tests/expressions.rs`
  `pointer_syntax_reports_that_pointers_are_not_yet_implemented` — asserts the
  annotation guard this change removes.

## Decisions

**The pointer type.** `Type::Pointer { nullable: bool, target: Box<Type> }`.
The derived `PartialEq` gives the spec's sameness rule: same target type and
same nullability. `Display` writes `*T` and `nullable *T`, so nested forms
read as written: `*nullable *int`, `[2]*int`, `*[2]int`. `Type::scalar`
returns `None` for a pointer, so arithmetic paths keep rejecting it.

**Layout.** `src/layout.rs` gains `pointer_bytes()`, which returns
`scalar_bytes(Scalar::Uint)`: `uint` has the platform pointer width, so a
pointer has its size and alignment. `size` and `alignment` return it for
`Type::Pointer` without looking at the target.

**Pointer targets resolve without their fields.** A struct may hold a pointer
to itself, so a target never resolves struct fields. `resolve_annotation`
delegates to a private `resolve_written` that takes a `Placement`:

- `Placement::Value` — today's behavior: a named struct resolves its fields,
  and the resolved type's layout is validated.
- `Placement::PointerTarget` — a named struct resolves its name only, through
  `resolve_type_name` and `struct_type`. An array length is still evaluated,
  and an array element stays in `PointerTarget`. No layout is validated.

The `Pointer` arm resolves its target in `PointerTarget` and builds the
pointer. Because field resolution never follows a pointer, `A { b: *B }` with
`B { a: A }` is valid, and `Node { next: Node }` is still recursive. A `[_]`
in a target has no initializer and reports the existing
`` `[_]` requires an array-literal initializer ``.

**Target layouts are validated once final.** A target can still exceed the
layout limit, such as `*[4]Half` where `Half` is half a pebibyte, and later
phases lay out a dereferenced target. The `Pointer` arm validates the target
with `validate_aggregate_layout` at the target annotation's span when every
struct the target holds by value is `Resolved` (`layout_final`: scalars and
pointers are final, an array is final when its element is, a struct when its
state is `Resolved`). Otherwise it pushes the type and span onto a new
checking-state field, `CheckedProgram::pending_targets`. Only module
initializers and struct field resolution run before a module's structs
resolve, so `resolve_module_structs` ends by draining `pending_targets` through
`validate_aggregate_layout`. Validating a non-final target early would cache a
partial struct layout.

**Zero values.** `docs/spec.md#variable-declarations` gives `nullable *T` the
zero value `null` and gives no zero value to `*T` or to an array or struct that
holds one by value.

- `Constant::Null` is the zero value of a nullable pointer.
  `zero_value_unchecked` returns it, and `aggregate_value_count` counts a
  pointer as one value.
- `CheckedStruct` gains `has_zero_value: bool`, set by `resolve_fields` once
  every field resolves. Computing it there keeps the check linear in the
  declared fields.
- `CheckedProgram::has_zero_value(&Type)`: true for scalars, `nullable` for
  pointers, the element's answer for arrays, and the stored flag for structs.
- `check_binding` and `check_module_initializer` report
  `` `{ty}` has no zero value `` at the statement span when a declaration has
  no initializer and its type has no zero value, for `var` and `const` alike.
- A struct literal with `...` reports
  `` omitted field `{name}` has no zero value `` at the literal span for the
  first omitted field whose type has none.
- `zero_value_unchecked` treats a non-nullable pointer as `unreachable!`.

`constants_equal` already compares `Null` to `Null` through its `==` fallback.
`contextualize_constant` and `converted` add `Null` to their `unreachable!`
arms.

**Temporary driver guard.** Every pointer type originates in an annotation,
because pointer expressions still report `pointers are not yet implemented`.
`reject_pointer_annotations(syntax: &Syntax) -> Result<(), Diagnostic>` in
`src/semantic/annotations.rs`, re-exported from `src/semantic/mod.rs`, returns
`pointers_not_implemented` at the span of the first `AnnotationKind::Pointer`
in `syntax.annotations`. `compile` calls it after semantic checking succeeds
and before lowering, so a pointer program reports its semantic errors first.
Semantic unit tests call the checker directly and see resolved pointer types.

Behind that guard, every `Type::Pointer` and `Constant::Null` arm in
`src/ir/` and `src/backend/` is
`unreachable!("pointer types are rejected before lowering")`.

**`*void`.** The parser already rejects it with `expected a type`, and the
parser tests already cover it.

## Implementation plan

1. `src/types.rs` — add `Type::Pointer`, its `Display` arm, and its
   `Type::scalar` arm. Update the `Type` doc comment: arrays and pointers
   compare structurally.
2. `src/layout.rs` — add `pointer_bytes` and the `Pointer` arms of `size` and
   `alignment`.
3. `src/semantic/model.rs` — add `Constant::Null` and its arms in `integer`,
   `rational`, and `float`; add `CheckedStruct::has_zero_value`,
   `CheckedProgram::has_zero_value`, `CheckedProgram::pending_targets`, and
   the `Pointer` arms of `zero_value_unchecked` and `aggregate_value_count`.
   Initialize the new fields where `CheckedProgram` and `CheckedStruct` are
   built in `src/semantic/namespaces.rs`.
4. `src/semantic/annotations.rs` — add `Placement`, `resolve_written`, and
   `layout_final`; replace the `Pointer` guard arm with resolution; set
   `has_zero_value` in `resolve_fields`; add `reject_pointer_annotations`.
5. `src/semantic/namespaces.rs` — drain `pending_targets` at the end of
   `resolve_module_structs`; add the zero-value check to
   `check_module_initializer`.
6. `src/semantic/statements.rs` `check_binding` — add the zero-value check.
7. `src/semantic/expressions.rs` — add the omitted-field zero-value check
   before the fill builds zero values.
8. `src/semantic/constants.rs` — add the `Null` arms.
9. `src/semantic/mod.rs` and `src/lib.rs` — re-export and call
   `reject_pointer_annotations`.
10. `src/ir/` and `src/backend/` — add the `unreachable!` arms the compiler
    asks for.

## Test plan

- `src/types.rs`:
  - spellings: `*int`, `nullable *Point`, `*nullable *int`, `**int`,
    `[2]*int`, and `*[2]int`;
  - sameness: pointers with the same target and nullability are equal; a
    different nullability, a different target, or a struct target from another
    declaration makes them different; a pointer differs from its target;
  - a pointer has no scalar.
- `src/layout.rs` unit tests with a small `StructFields` table: a pointer's
  size and alignment are `pointer_bytes()`, `[3]*int` is three times that, and
  `struct { a: u8, p: *int }` puts `p` at offset `pointer_bytes()`.
- `src/semantic/tests/annotations.rs`:
  - binding, parameter, result, field, and array-element annotations record
    `*int`, `nullable *int`, `**int`, `*nullable *int`, `*[3]int`, `[2]*int`,
    and `nullable *geometry::Point` across a module tree;
  - `type Node struct { value: int, next: nullable *Node }`,
    `type Tree struct { children: nullable *[2]Tree }`, and the pair
    `A { b: *B }` and `B { a: A }` resolve, with the expected field types;
  - `type Node struct { next: Node }` still reports
    `` recursive struct type `Node` ``;
  - `fn f(p: *[1 << 48]int) -> void {}`, a struct field of that type, and a
    module-level `var p: nullable *[4]Half;` declared before
    `type Half struct { cells: [1 << 46]int }` each report
    `aggregate layout exceeds compiler limit of 1 PiB` at the target
    annotation;
  - `var p: *[_]int = [1];` reports the `[_]` error at `[_]int`;
  - an unknown target name reports the existing unknown-type diagnostic.
- `src/semantic/tests/statements.rs`:
  - `var p: nullable *int;` records no constant, and `const p: nullable *int;`
    records `Constant::Null`; a struct with a nullable pointer field zeroes it
    to `Null`;
  - `var p: *int;`, `const p: *int;`, `var a: [2]*int;`, `var h: Holder;`
    where `Holder` holds a `*int`, and the module-level `var p: *int;` each
    report `` `…` has no zero value `` at the statement.
- `src/semantic/tests/expressions.rs`:
  - `Holder { ... }` reports `` omitted field `p` has no zero value ``;
    `Holder { p = q, ... }` with `q` a parameter is accepted, and a nullable
    pointer field omitted under `...` fills with `Null`;
  - `pointer_syntax_reports_that_pointers_are_not_yet_implemented` keeps only
    the expression and statement cases, and `const p: *int = new(0);` reports
    the guard at `new(0)`.
- `tests/compiler.rs` `source_failures_preserve_output` — a program with
  `fn f(p: *int) -> void {}` reports `pointers are not yet implemented`.
