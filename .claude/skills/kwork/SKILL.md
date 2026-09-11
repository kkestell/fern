---
name: kwork
description: "Execute a repository plan with focused checks and validate the integrated feature at its completion boundary."
argument-hint: "[plan, specification, or todo file path]"
---

## Workflow

### Pre-flight

1. Resolve `<input_document> $ARGUMENTS </input_document>`.
   - Prefer `eng/plans/` for plans and `eng/todo.md` for tracked follow-up work.
   - If none is supplied, use repository ordering and current state to identify
     the next clearly unimplemented plan. Ask only when more than one candidate
     remains plausible.
2. Check `git status`.
   - Preserve unrelated changes. Continue from earlier implementation steps.
   - Ask only when overlapping changes leave ownership or intended behavior
     unclear.

### Implement

3. Read the work document completely, then inspect only the related code and
   nearby patterns needed to execute it.
4. Turn its implementation and test bullets into a short working checklist. Skip
   a separate todo tool when the plan is already small enough to track directly.
5. Implement the selected plan's task and stop when its scoped changes and
   focused checks are done.
   - Intermediate tasks may leave the feature partially implemented. Preserve
     existing supported behavior and record unfinished integration for later
     TODO tasks.
   - Keep architecture coherent as the feature develops. Add temporary guards
     only when needed to prevent incorrect behavior.
   - Follow the plan and repository guidance.
   - Write focused tests through public boundaries, emphasizing edge cases,
     failures, and semantic boundaries.
   - Ask the user only when the plan and its authoritative sources leave a
     material decision unresolved.
   - Keep tool output targeted. Batch independent reads and checks when useful.

### Validate

6. Run focused checks while implementing.
7. Choose checks in proportion to risk and the completion boundary.
   - Use focused tests and checks for intermediate plans.
   - Run the repository's broad gates when an integrated feature is
     integrated. Run them earlier when a concrete regression risk warrants it or
     repository instructions explicitly require it.
   - For documentation-only, filename-only, and test-only changes, run focused
     checks and inspect the diff unless repository guidance explicitly requires
     broader validation for that change category.
   - Fix regressions. Report expected failures from unfinished integration
     explicitly; do not claim the feature is complete while they remain.
   - Do not repeat passing broad gates unless later changes could affect them.
8. Do not run a plan review or post-implementation review and do not spawn a
   review agent. Review is performed once across the completed integrated
   feature with `kreview`.

### Record completion and hand off

9. When the plan implements a task listed in `eng/todo.md` and its required
   checks pass, change that task's checkbox from `- [ ]` to `- [x]`. If this
   completes all direct subtasks of an unchecked parent task, mark that parent
   task complete too. A defect-repair plan has no TODO task, so it leaves
   `eng/todo.md` untouched. Checking a box is the only TODO edit made by
   `kwork`; leave task text and ordering intact. Adding or revising tasks
   belongs to the user.
10. Commit only when authorized by the user and allowed by repository guidance.
    A plan boundary does not require a commit.
11. Keep the handoff focused on what changed, checks run, and any unfinished
    integration. Distinguish completion of an intermediate plan from completion
    of the feature. Discuss review timing only when the user asks about it;
    deferred review is not unfinished implementation or a routine next step.

## Principles

- **Use the plan as an index** — follow its source links and tasks without
  recreating the planning exploration.
- **Validate proportionately** — use focused checks during implementation and
  broad gates at the integrated feature boundary.
- **Defer review to the integrated feature** — individual slices stay cheap
  while the cumulative result still receives an independent review.
- **Separate subtask and parent completion** — finish the selected task; a
  parent task completes only when all its subtasks are complete.
