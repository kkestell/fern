---
name: kspec
description: "Create or update docs/spec.md through an interactive product-specification process. Use when defining, changing, or recording product behavior; do not invoke merely to read the specification."
argument-hint: "[behavior to specify, or blank to create/review the specification]"
---

## Workflow

This skill owns changes to `docs/spec.md`. Reading the specification does not
require it.

### Establish the specification work

1. Resolve `<specification_work> $ARGUMENTS </specification_work>` and the
   repository root. Confirm that the path is a Git repository.
2. Read `AGENTS.md` and `docs/spec.md` when they exist. For a new specification,
   also read the project README and the smallest useful set of user-facing docs,
   examples, and tests.
3. Separate facts the user has decided from behavior that remains unsettled.
   Existing implementation is evidence of current behavior, not permission to
   turn every accidental detail into a requirement.

### Resolve behavior with the user

4. When the request already settles the behavior, accept it as decided. Do not
   ask for confirmation the user has already supplied.
5. When a material behavior is unsettled, resolve one decision at a time.
   - Ask one plain question.
   - Present numbered viable options, recommend one, and mention relevant
     precedent from comparable software.
   - Include a small concrete example when the choice affects an interface.
   - Note the decision and move to the next question. Hold all writing to
     `docs/spec.md` until the session's questions are resolved.
6. For an empty repository or a new product, begin with what the software is for
   and its observable interface. Let the user's answers determine which behavior
   needs specification. Do not infer a product or roadmap from the template.

Use general knowledge for prior art. Consult an official reference only when a
technical detail is both uncertain and material to the decision.

### Write docs/spec.md

7. Write `docs/spec.md` once, after the last decision of the session, carrying
   every noted decision into that single write. Create `docs/` when needed and
   use `assets/spec-template.md` for a new file. Use the project name in the
   title when it is established; otherwise keep the neutral `Specification`
   title.
8. Write rules as behavior a reader can check against the running product. Cover
   inputs, outputs, state changes, errors, ordering, and intentional omissions
   when they apply. Use small examples where prose alone is ambiguous.
9. Keep implementation structure, algorithms, dependencies, work ordering, and
   task status out of the specification. Those belong in architecture, code, or
   the roadmap.
10. Preserve unrelated existing behavior. Do not leave template prompts in the
    written file or state an undecided behavior as settled. If the user ends the
    session before every question is resolved, write only the decisions that
    have actually been made.
11. Report what changed and name any specification question that remains
    unresolved.

## Principles

- `docs/spec.md` is authoritative for behavior where it is explicit.
- Specify observable contracts, not the current implementation.
- Give each fact one home; refer to facts owned elsewhere instead of copying
  them.
- Record intentional absence only when it is a product decision, not merely
  because work has not been implemented yet.
