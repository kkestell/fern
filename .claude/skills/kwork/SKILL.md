---
name: kwork
description: "Take a repository task from lightweight planning through implementation and focused validation in one session. Use for roadmap work; keep the plan in context without writing a plan file."
argument-hint: "[task description or existing work document; blank selects the next roadmap task]"
---

## Workflow

### Establish the work

1. Resolve `<task> $ARGUMENTS </task>`.
   - Require `docs/spec.md` and `eng/roadmap.md`. If either is missing, name
     the missing document and direct the user to `kspec` or `kroadmap`.
   - Read both documents and identify the requested roadmap task. If no task is
     supplied, select the first unchecked task in milestone and task order,
     using implementation state to continue any work already started.
   - Read any supplied work document as context. An implementation plan file is
     not required.
   - Task creation and scope belong to `kroadmap`. If the task is absent from
     the roadmap, direct the user there. Ask only when task selection remains
     ambiguous.
2. Check `git status`.
   - Preserve unrelated changes. Continue from earlier implementation steps.
   - Ask only when overlapping changes leave ownership or intended behavior
     unclear.

### Plan lightly, then implement

3. Read the owning documents before making behavioral or structural decisions.
   Read `eng/architecture.md` when it exists and the task touches structure.
   Inspect only the related code and nearby patterns needed for this task.
   Resolve missing behavior with `kspec` before implementation depends on it.
4. Form a short checklist in context: affected files, implementation steps, and
   focused checks. Scale the detail to the task; do not write a plan file.
   Continue directly into implementation in the same session. Ask only when a
   material decision remains unresolved by the task and its authoritative
   sources.
5. Implement the selected task and stop when its scoped changes and
   focused checks are done.
   - Intermediate tasks may leave the feature partially implemented. Preserve
     existing supported behavior and report unfinished integration in the
     handoff.
   - Keep architecture coherent as the feature develops. Add temporary guards
     only when needed to prevent incorrect behavior.
   - Follow the task scope and repository guidance.
   - Write focused tests through public boundaries, emphasizing edge cases,
     failures, and semantic boundaries.
   - Keep tool output targeted. Batch independent reads and checks when useful.

### Validate

6. Run focused checks while implementing.
7. Choose checks in proportion to risk and the completion boundary.
   - Use focused tests and checks for intermediate tasks.
   - Run the repository's broad gates when the feature or milestone is
     integrated. Run them earlier when a concrete regression risk warrants it or
     repository instructions explicitly require it.
   - For documentation-only, filename-only, and test-only changes, run focused
     checks and inspect the diff unless repository guidance explicitly requires
     broader validation for that change category.
   - Fix regressions. Report expected failures from unfinished integration
     explicitly; do not claim the feature is complete while they remain.
   - Do not repeat passing broad gates unless later changes could affect them.
8. Do not run a plan review or post-implementation review and do not spawn a
   review agent. Review is performed once across the completed milestone with
   `kreview`.

### Record completion and hand off

9. After the selected task and its required checks pass, change its roadmap
   checkbox from `- [ ]` to `- [x]`. This is the only roadmap edit made by
   `kwork`; leave milestone headings, task text, examples, gates, and ordering
   intact. Adding or revising milestones and tasks belongs to `kroadmap`.
10. Commit only when authorized by the user and allowed by repository guidance.
    A task boundary does not require a commit.
11. Keep the handoff focused on what changed, checks run, and any unfinished
    integration. Distinguish completion of an intermediate task from completion
    of the feature. Discuss review timing only when the user asks about it;
    deferred review is not unfinished implementation or a routine next step.

## Principles

- **Plan in context** — use the roadmap and owning documents to form a short
  checklist, then do the work without a separate planning handoff.
- **Validate proportionately** — use focused checks during implementation and
  broad gates at the feature or milestone boundary.
- **Defer review to the milestone** — individual slices stay cheap while the
  cumulative result still receives an independent review.
- **Separate task and milestone completion** — finish the selected task;
  integration gates belong to the completed milestone.
