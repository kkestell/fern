# Parse `pub`, `use`, and qualified names

## Sources

- `docs/spec.md#module-directories` — import paths and their `::` components
- `docs/spec.md#public-declarations` — where `pub` is and is not permitted
- `docs/spec.md#use-declarations` — the whole-module, nested-path, and
  selective forms, their placement, and the ban on renaming
- `eng/roadmap.md#modules-and-imports` — first task and its parser boundary
- `src/frontend.rs` — lexer, syntax arenas, parser, and syntax snapshots

## Goal

Parse the full module syntax into `Syntax`: `pub` on module-level declarations,
`use` declarations, and qualified `module::name` references, call targets, and
assignment targets. This is a parser-only slice. Multi-file compilation, import
resolution, and visibility checking arrive in the following tasks, so semantic
checking rejects the new forms for now.

## Implementation

- `src/frontend.rs` — add `pub`, `use`, and `::` tokens. `pub` and `use` are
  already in `reserved`, so name positions keep their current diagnostic.

- `src/frontend.rs` — add `PathComponent { name, name_span }` and
  `QualifiedName { qualifier: Option<PathComponent>, name, name_span, span }`,
  where `span` covers the whole `module::name`. Replace `Call`'s `target` and
  `target_span` with one `QualifiedName`, replace `ExpressionKind::Reference`'s
  `Spur` with one, and replace the `name`/`name_span` pair in
  `StatementKind::Assignment` and `StatementKind::CompoundAssignment` with one.
  A qualified name has at most two components; longer paths appear only in
  `use`.

- `src/frontend.rs` — add `Import { path: Vec<PathComponent>, selection:
  Option<Vec<PathComponent>>, span }` and `Syntax::imports: Vec<Import>`.
  Parse `use` only before the first item in `Syntax::items`; a later `use` is an
  error on its own span. Reject `pub use`, a renaming form (a name where `;` is
  expected after the path), and an empty `{}` selection.

- `src/frontend.rs` — make `TopLevelItem` struct variants carrying
  `public: bool` alongside the existing index, set from an optional leading
  `pub` in the top-level loop. Reject `pub` in `statement` with a message naming
  local declarations.

- `src/frontend.rs` — extend the one-token lookahead in `starts_assignment` and
  `starts_call` to skip an optional `:: name` first, and parse a qualified name
  in `call`, `assignment`, and the name-led branch of `primary_expression`.
  Keep one qualified-name parser shared by all three.

- `src/frontend.rs` — project imports, a `pub ` prefix on public items, and a
  qualified name's qualifier in the test projection. Print the prefix only when
  the item is public so existing snapshots do not move.

- `src/semantic.rs` — update the `TopLevelItem` matches and the sites that read
  the replaced `Spur`/span pairs (`collect_references`, `assignment_target`,
  `resolve`, `check_call`) to go through `QualifiedName`. Same for
  `call.target_span` in `src/ir.rs`.

- `src/semantic.rs` — at the top of `check`, reject the first `Import` in
  `syntax.imports` and the first qualified name found by scanning the statement
  and expression arenas. These two guards exist only to prevent unresolvable
  names from reaching lowering; the import-resolution tasks delete them.

## Tests

- A frontend snapshot covering `pub` on a `fn`, `var`, and `const`; the
  single-component, nested-path, and selective `use` forms with several selected
  names and a trailing comma; and qualified references, call targets, and
  assignment targets, including a qualified call nested in an expression.

- Parser diagnostics with exact offending spans for: `pub` on a local binding,
  `pub use`, a `use` after the first declaration, `use fmt as f;`, `use fs::{};`
  a missing `;`, a missing or reserved component after `::`, and a qualified
  name in a position that takes a plain one (function name, parameter name,
  binding name, loop label).

- A semantic test that a program with a `use` declaration and one with a
  qualified name each fail with the temporary guard's diagnostic.

## Decisions

- Allow a trailing comma in a selective `use` list and reject an empty one. The
  specification leaves the list shape open; trailing commas match the parameter
  and argument lists, and an empty selection introduces no name.

- Store `public` on `TopLevelItem` rather than on `Function` and
  `StatementKind::Binding`, so both module-level declaration kinds record
  visibility in one place and local bindings gain no unusable field.
