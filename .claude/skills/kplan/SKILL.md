---
name: kplan
description: Explore how a code change fits the repository, resolve material decisions, and write a concise implementation plan in `eng/plans/`. Use when the user wants to plan code work or requests a significant behavior change. Do not use it to write or update documentation such as a TODO, specification, design note, or README.
argument-hint: "[feature idea, bug report, or improvement to explore]"
---

## Workflow

This skill produces an implementation plan. It never implements the plan.

### Establish the work

1. Require `docs/spec.md` before doing any planning, and `eng/todo.md` as well
   when the work is a feature slice.
   - If a required document is missing, list the missing path and stop.
   - Tell the user to run `kspec` for a missing specification or add a TODO for
     a missing `eng/todo.md`. Do not create placeholders or infer either document
     from the requested implementation work.
2. Resolve `<feature_description> $ARGUMENTS </feature_description>` and decide
   which kind of work it is.
   - Defect repair is planned directly against `docs/spec.md` and the code: a
     bug report, a review finding, a regression, or an implementation that
     contradicts the specification. The TODO tracks feature scope, so a defect
     repair needs no TODO entry and reaching one is not a prerequisite. This
     holds even when the defect sits inside a task already marked complete.
   - Feature work is planned against `eng/todo.md`. When the description names
     it, identify the corresponding unchecked task. When the description is
     empty, take the first unchecked task in file order; if that task has
     unchecked subtasks, select its first unchecked subtask. Inspect plans,
     history, and code only as needed to establish which listed task is next.
   - When feature work is requested and no unchecked task exists, stop and tell
     the user that `eng/todo.md` is complete and needs another task before
     planning can continue.
   - Feature scope and ordering belong to `eng/todo.md`. When a request adds new
     feature scope the TODO does not list, stop and tell the user to add it
     there.
   - Continue without confirmation when the user has already authorized planning
     the next repository-defined slice. Ask one focused question only when the
     next work remains ambiguous.
3. Read `docs/spec.md` and `eng/todo.md` before exploring implementation.
   - The specification owns behavior. The TODO owns task scope, order, and
     completion state.
   - Read `eng/architecture.md` when it exists and the work touches structural
     decisions. Its absence is not a blocker.
   - Reference those sources from the plan. Do not restate their contents.
4. Confirm that the source documents settle the proposed work.
   - If required product behavior is missing or ambiguous, invoke `kspec` to
     resolve it, then resume this workflow from the updated specification. Do
     not modify the specification directly from this skill.
   - When a request adds feature scope that the TODO does not list, or conflicts
     with its order or boundaries, stop and identify the decision the user must
     resolve in `eng/todo.md`. Do not modify the TODO from this skill.
   - Confirm a reported defect against the specification and the code before
     planning its repair, and plan only the repairs that hold up.
   - Resolve only implementation decisions that the specification, TODO, and
     established codebase patterns leave open.
   - Compare alternatives only when the repository leaves a real choice. Follow
     an established local pattern directly when it already settles the design.

### Explore and size

5. Inspect the relevant parts of the repository.
   - Find the attachment points, adjacent patterns, affected public boundaries,
     and tests.
   - Use targeted searches and file ranges. Do not dump whole directories or
     reread guidance already present in the session.
   - Check architectural fit directly.
6. Write one plan for the selected work.
   - Scope a feature plan from its TODO task, and a defect plan from the
     confirmed defects and the rules they violate.
   - State the starting state, dependencies, and any integration the plan leaves
     unfinished. One plan may cover one implementation phase or several related
     phases.
   - A parent task is complete only after all its subtasks are complete. Identify
     when this plan completes a parent task and therefore reaches an integrated
     feature boundary.
   - Preserve existing supported behavior. Add temporary safeguards only where
     needed for correctness.
   - The plan should be detailed enough that a lesser model can complete it
     without requiring advanced reasoning or problem solving.

### Write

7. Name each plan `eng/plans/YYYY-MM-DD-NNN-slug.md`, using the next sequence
   for the day.
8. Write from `assets/plan-template.md`.
   - Keep only information the implementer needs to execute this slice:
     source-of-truth references, concrete file-oriented tasks, decisions not
     obvious from those sources, and tests unique to the change.
   - Do not copy language rules, TODO scope, architecture guidance,
     repository instructions, conversation history, rejected alternatives,
     generic risks, standard validation commands, or follow-up work owned by the
     TODO.
   - Use concrete bullets. Add detail only where an implementer could otherwise
     make a materially wrong choice.
9. Print the final plan path and stop.
   - Do not review the plan locally or with a subagent.
   - Suggest a fresh session only when the remaining work needs one.
