---
name: kreview
description: "Review a completed milestone for omissions, correctness problems, and unnecessary complexity. Use once after all milestone slices are implemented, not after each plan. Do not use it to review prose alone."
argument-hint: "[milestone description] [--base <commit-or-ref>]"
---

## Workflow

Run this skill once from a fresh session after every planned slice in a
milestone has been implemented and validated. Perform the review directly; do
not spawn additional review agents.

### Establish scope

1. Resolve the milestone and comparison range from
   `<input_document> $ARGUMENTS </input_document>`.
   - Prefer an explicit base ref when supplied.
   - Otherwise use the roadmap, milestone plans, and repository history to find
     the commit immediately before the milestone began.
   - Ask one focused question only when the range remains ambiguous.
2. Read the milestone's authoritative specification and roadmap sections, then
   the plans for its slices.
3. Inspect the cumulative diff and changed-file list before opening files.
   - Read changed code and nearby context needed to judge the diff.
   - Do not dump every changed file end to end when the diff and focused context
     are sufficient.

### Completeness and correctness

4. Check the complete milestone against its specification, roadmap gates, and
   plans.
   - Find omitted behavior, partial slices, disabled checks, workarounds, and
     contract drift.
   - Check interactions between slices that a per-plan review would miss.
   - Check likely correctness errors at public and semantic boundaries.

### Simplification

5. Review the cumulative implementation for unnecessary complexity.
   - Look for abstractions without current clients, duplicated machinery,
     speculative flexibility, hidden state, and code that can be made direct.
   - Preserve required behavior and established repository boundaries.

### Report

6. Return one concise report:
   - comparison range and milestone reviewed;
   - `PASS` or `ISSUES FOUND`;
   - actionable issues only, each with path, line, consequence, and suggested
     fix;
   - separate completeness/correctness and simplification headings when both
     have findings.
7. Do not edit files or rerun the project's validation suite. The caller fixes
   findings and reruns affected gates.

## Principles

- **Milestone-wide** — review the integrated result once, after all slices.
- **Independent** — start fresh and judge the final diff rather than the
  implementer's transcript.
- **Diff-first** — load only the context needed to assess changed behavior.
- **Actionable** — omit reassurance, style nits, and cosmetic suggestions.
