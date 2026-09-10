---
name: kplan
description: Explore how a code change fits the repository, resolve material decisions, and write a concise implementation plan in `eng/plans/`. Use when the user wants to plan code work or requests a significant behavior change. Do not use it to write or update documentation such as a roadmap, specification, design note, or README.
argument-hint: "[feature idea, bug report, or improvement to explore]"
---

## Workflow

This skill produces an implementation plan. It never implements the plan.

### Establish the work

1. Require both `docs/spec.md` and `eng/roadmap.md` before doing any planning.
   - If either is missing, list the missing path and stop.
   - Tell the user to run `kspec` for a missing specification, then `kroadmap`
     for a missing roadmap. Do not create placeholders or infer either document
     from the requested implementation work.
2. Resolve `<feature_description> $ARGUMENTS </feature_description>`.
   - When it is present, identify the corresponding task in `eng/roadmap.md`.
   - When it is empty, take the next unimplemented task from `eng/roadmap.md`.
     Inspect plans, history, and code only as needed to establish which listed
     task is next.
   - Task creation and scope belong to `kroadmap`. If the roadmap does not
     define the task to plan, stop and direct the user to `kroadmap`.
   - Continue without confirmation when the user has already authorized planning
     the next repository-defined slice. Ask one focused question only when the
     next work remains ambiguous.
3. Read `docs/spec.md` and `eng/roadmap.md` before exploring implementation.
   - The specification owns behavior. The roadmap owns scope, order, and
     completion gates.
   - Read `eng/architecture.md` when it exists and the work touches structural
     decisions. Its absence is not a blocker.
   - Reference those sources from the plan. Do not restate their contents.
4. Confirm that the source documents settle the proposed work.
   - If required product behavior is missing or ambiguous, invoke `kspec` to
     resolve it, then resume this workflow from the updated specification. Do
     not modify the specification directly from this skill.
   - If the work is absent from or conflicts with the roadmap, stop and identify
     the scope or ordering decision the user must resolve with `kroadmap`. Do
     not modify the roadmap from this skill.
   - Resolve only implementation decisions that the specification, roadmap, and
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
6. Write one plan for the selected roadmap task.
   - Use the task's roadmap-defined scope to identify concrete changes and
     focused checks.
   - State the task's starting state, dependencies, and any integration it
     leaves unfinished. A task may cover one implementation phase or several
     related phases.
   - The milestone's integrated outcome determines broad validation and review
     timing, not the scope of each plan. Identify when this task reaches that
     boundary.
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
   - Do not copy language rules, roadmap scope, architecture guidance,
     repository instructions, conversation history, rejected alternatives,
     generic risks, standard validation commands, or follow-up work owned by the
     roadmap.
   - Use concrete bullets. Add detail only where an implementer could otherwise
     make a materially wrong choice.
9. Print the final plan path and stop.
   - Do not review the plan locally or with a subagent.
   - Suggest a fresh session only when the remaining work needs one.
