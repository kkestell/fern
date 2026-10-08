# Remove pointers, slices, and unsettled features

## Plan

`agents/plans/2026-10-08-001-remove-pointers-slices-and-unsettled-features.md`

## Summary

Pointers, slices, `null`, comparability, and the rune, string,
character-literal, and non-struct named-type rules are gone from the compiler,
tests, examples, and documents. The IR verifier no longer re-checks the struct
table, layouts, or definite initialization, and comparison and logical-not
values carry no span. The plan's goal is met: the specification describes what
the compiler accepts.

## Departures from the plan

- `docs/spec.md` was edited directly rather than through `kspec`, because the
  plan fixes every specification change.
- `CheckedLocation.mutable` is removed. Mutability comes from the root binding
  through `location_binding`, which would otherwise have no caller.
- `Place::root_local` is removed. Only the definite-initialization analysis
  used it.
- The Checked conversion section of the specification still pointed to the
  deleted Rune conversions section. That paragraph is removed.
- The work log lives in `agents/work/`, alongside `agents/plans/` and
  `agents/reviews/`. `AGENTS.md` does not list a work-log directory.

## Decisions

- The verifier now indexes the struct table without checking it. IR that names
  an unknown struct panics instead of returning an internal compiler error.
  Semantic checking already rules that IR out.
- Doc comments that named IR verification as a layout reader or a
  containment-depth enforcer now name semantic checking: `src/layout.rs`,
  `MAX_STRUCT_CONTAINMENT_DEPTH`, and the backend's `REPRESENTABLE`.
- The lexer test that lists non-reserved names now includes `null` and
  `string`.

## Checks run

- `cargo fmt --check`: passed.
- `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo test`: 283 unit tests and 57 integration tests passed. The seven
  changed IR snapshots differ only by `span: Some(..)` becoming `span: None` on
  `Comparison` values.

## Manual verification

1. Every remaining example compiles and runs.

   ```sh
   cargo build
   for f in examples/*.fern examples/modules_and_imports/app; do
     n=$(basename "$f" .fern)
     target/debug/fern "$f" -o "target/examples/$n" && "target/examples/$n"
     echo "$n: $?"
   done
   ```

   `control_flow` exits 6. Every other example exits 42.

2. Address-of is no longer syntax.

   ```sh
   printf 'fn main() -> void { var x = 1; var p = &x; }\n' > /tmp/a.fern
   target/debug/fern /tmp/a.fern -o /tmp/a
   ```

   Reports `expected an expression` at `&`.

3. Assigning an element of a `const` array names the binding.

   ```sh
   printf 'fn main() -> void { const a = [1, 2]; a[0] = 3; }\n' > /tmp/c.fern
   target/debug/fern /tmp/c.fern -o /tmp/c
   ```

   Reports ``cannot assign to immutable binding `a` `` at `a`.

## Follow-up work

- `agents/todo.md` is deleted. `kplan` needs a new one before it can plan the
  next task.
- Decide whether `AGENTS.md` should list `agents/work/`.
