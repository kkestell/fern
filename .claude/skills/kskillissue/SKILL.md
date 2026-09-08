---
name: kskillissue
description: Diagnose and fix a `k*` skill that made an agent behave badly, then commit, push, and reinstall it. Use when the user reports that an agent did something wrong and suspects the skill it was following — "the skill told it to", "kwork skipped the review", "kplan wrote the plan to the wrong place", "that's a skill issue". Works on the source checkout of the skills repo, not the installed copy.
argument-hint: "[what the agent did wrong — and which skill, if you know]"
---

## Workflow

### Phase 1 — Capture the failure

1. Read `<issue_report> $ARGUMENTS </issue_report>`. If it is empty, ask what
   the agent did and which skill was in play, then stop.
2. Write down the evidence before touching any file:
   - which skill was active,
   - what its text told the agent to do,
   - what the agent actually did,
   - what it should have done instead.
3. If the failure happened in this session, take the evidence from the
   transcript. If it happened elsewhere, ask the user for what they typed and
   what the agent did in response.
4. The failure report is the specification for the fix. If it is too vague to
   point at specific text, ask one focused question rather than guessing.

### Phase 2 — Find the source

5. Locate the checkout. Try `~/src/skills` first and confirm it is the right
   one: `git -C ~/src/skills remote -v` names `kkestell/skills`, and the tree
   has `profiles.py` and `skills/`. If it is not there, search the usual
   development directories before asking the user.
6. Edit the repo copy only. Installed skills live in `~/.agents/skills/<name>`,
   symlinked from `~/.claude/skills/<name>`; every sync overwrites them.
7. Confirm the suspect skill is in this repo under
   `skills/<group>/<name>`. If the behavior came from a built-in agent skill, a
   plugin skill, `AGENTS.md`, or the user's `CLAUDE.md`, say where it actually
   comes from and stop.

### Phase 3 — Diagnose

8. Compare the installed copy against the source:

   ```bash
   diff -ru ~/.agents/skills/<name> skills/<group>/<name>
   ```

   A difference means the install is stale, and the fix may be nothing but a
   sync — confirm first that the repo copy would have produced the right
   behavior.
9. Read the whole `SKILL.md` and every asset, reference, and script it loads.
   Find the text responsible. Common root causes:
   - the instruction is missing entirely,
   - it is present but ambiguous, and the agent took the other reading,
   - two instructions conflict and the agent followed the wrong one,
   - the ordering is wrong — the right instruction arrives after the damage,
   - the `description` never triggers the skill, or triggers it too eagerly,
   - the flawed step lives in an asset template or subagent prompt,
   - the problem is repo tooling (`profiles.py`, validation), not skill prose.
10. State the root cause in a sentence or two with the exact `file:line`.
11. If the text was clear and correct and the agent simply ignored it, say so
    and stop. Do not paper over an agent's mistake with more prose.

### Phase 4 — Fix

12. Make the smallest edit that closes the gap, in the layer that owns the
    behavior: prose problem in the prose, template problem in the template,
    trigger problem in the `description`.
13. Write the fix as clean, declarative instruction, as if the right way had
    been known from the start. No warnings built around what the agent did, no
    rationale that exists only to justify the incident, no dated notes.
14. Keep the repo's conventions: allowed frontmatter fields only and `SKILL.md`
    body under 500 lines.
15. Do not grow the scope. A prose bug does not need a new skill, a new asset
    file, or a restructure. If the diagnosis genuinely calls for more than a
    contained edit, describe it and get approval before making it.

### Phase 5 — Verify

16. Run `npm run check` from the repo root, or validate the one skill with
    `npm run validate -- skills/<group>/<name>`. Run `npm install` first if
    `node_modules/` is missing. Fix what fails.
17. Re-read the edited skill from the top the way a fresh agent would, and
    confirm the reported scenario now lands correctly.
18. For a fix whose effect is not obvious, hand the edited `SKILL.md` and the
    failure scenario to a fresh subagent and ask what it would do. It has no
    memory of the incident, so its answer is real evidence.

### Phase 6 — Commit and push

19. Show the user the diff, then commit and push without waiting for approval —
    invoking this skill authorizes both.
20. Stage the files this fix touched, by path. The checkout often carries
    unrelated work in progress, so never stage the whole tree.
21. Write the commit subject around the behavior the skill now produces, not the
    incident that prompted it. No AI attribution or co-author trailer.
22. Push to `origin`. If the branch has diverged from its remote, report that
    and stop.

### Phase 7 — Reinstall

23. Run `./profiles.py sync-skills core,ext --claude --codex` from the repo
    root. Pass a different set of harness flags only when the user names one;
    never infer harnesses from which configuration directories happen to exist
    on the machine.
24. Confirm the install landed — re-run the `diff -ru` from step 8 and see it
    come back clean.

### Phase 8 — Report

25. Report the root cause with its `file:line`, what changed, check results, the
    commit hash, the push, and the sync result.
26. Note that instructions already loaded in this session stay in context; the
    new text applies to the next invocation and to later sessions.

## Principles

- **The failure report is the spec** — fix the behavior the user described, not
  everything else you notice in the file.
- **Fix the source, not the install** — `~/.agents/skills/` is generated.
- **Smallest edit, right layer** — a cosmetic wording bug gets a wording fix.
- **Write it as if it were always right** — the skill reads as instruction, not
  as a record of what went wrong.
- **Not every failure is a skill issue** — sometimes the answer is that the
  skill was fine.
- **Not finished until it is synced** — commit, push, and reinstall are part of
  the job.
