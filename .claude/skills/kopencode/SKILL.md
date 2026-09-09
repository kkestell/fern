---
name: kopencode
description: Delegate a task to opencode headlessly with `opencode run`, let opencode work autonomously, then rewrite only opencode's final response for the user. Use when the user asks to hand work to opencode, delegate to opencode, or use the opencode CLI without the current agent duplicating the work. Do not use when the user asks the current agent itself to perform or independently verify the task.
argument-hint: "[task to delegate to opencode]"
---

# opencode Delegate

Act as a thin dispatcher. Minimize the current agent's token use.

## Rules

1. Do not investigate, plan, inspect files, read the repository, or solve the
   task yourself before delegating.
2. Pass the user's actual task to opencode with only the minimal wrapper below.
3. Run opencode once and let it work autonomously.
4. Do not supervise opencode while it works.
5. Do not independently inspect opencode's changes, run tests, read diffs, check
   git status, or verify opencode's claims afterward unless the user explicitly
   asked for verification.
6. After opencode exits, read only its streamed assistant messages. The last
   message is opencode's final report and is the source material.
7. Rewrite that final response into clear, concise language for the user.
8. Preserve substantive facts, filenames, commands, results, caveats, and
   unresolved problems.
9. Remove verbosity, repetition, canned sections, excessive explanation, and
   awkward model prose.
10. Do not add analysis or claims that were not present in opencode's result.
11. Do not mention this delegation workflow unless it is relevant to an error or
    limitation.
12. If opencode fails or produces no usable final response, report the failure
    directly. Do not take over the task unless the user asks.

## Invocation

Run from the directory in which the task should be performed:

```bash
opencode run \
  -m 'openrouter/~z-ai/glm-flash-latest' \
  --variant high \
  --format json \
  - <<'KOPENCODE_TASK' \
  | jq -r --unbuffered 'select(.type == "text") | .part.text'
Complete the following task autonomously. Work until it is finished. Use your tools as needed. Do not ask the user questions. Do not create or use git worktrees; work directly in the current checkout. While you work, output a concise one-line progress update at least every 5 minutes saying what you are doing. Make any necessary edits and run whatever checks you judge appropriate. Finish with a concise final report containing what you did, relevant validation results, and any unresolved issues.

TASK:
<USER_TASK>
KOPENCODE_TASK
```

The `--format json` + `jq` pipeline is required. Plain output does not stream
usefully; the JSON event stream emits one line per event, and the filter prints
each completed assistant text part as opencode produces it, ending with the
final report.

Replace `<USER_TASK>` with the user's task as faithfully and compactly as
possible.

Do not add repository context that opencode can discover itself.

Do not tell opencode how to solve the task unless the user supplied those
instructions.

Use `openrouter/~z-ai/glm-flash-latest` unless the user names a different
model, in which case pass that identifier exactly as given.

Single-quote the model identifier. It begins with `~`, and an unquoted `~z-ai/`
is a tilde expansion that fails with `no such user or named directory: z-ai`
before opencode ever runs.

`opencode run` applies the permission rules from the user's opencode config. If
the run stalls or reports a denied action, say so rather than escalating. Add
`--auto`, which auto-approves every permission that is not explicitly denied,
only when the user asks for it or the surrounding environment authorizes it.

The run is recorded as a normal opencode session; there is no ephemeral mode.
Pass `--title` when a recognizable session name is useful.

## Getting the most out of this model

`~z-ai/glm-flash-latest` takes a reasoning effort through `--variant`, with the
values `low`, `high`, and `max`. Use `high`.

Always pass one. With no `--variant`, opencode omits `reasoning_effort` from the
request and the provider picks the effort, so the run is at the mercy of a
default nobody here controls.

Avoid `max` on this model. It stalls: two of three trial runs on a trivial
prompt produced no output at all, wrote nothing to stderr, and had to be killed
after 200 and 560 seconds. A stalled delegated run looks identical to a slow one.

An unrecognized variant is accepted silently rather than rejected, so a typo
degrades the run without any error—copy the value, do not retype it.

The model holds roughly 1.3M tokens of context and cached input is read back at
a fraction of fresh input. One long autonomous run is therefore both cheaper and
better than several short ones. Delegate the whole task, and let opencode
explore the repository itself instead of pasting file contents into the prompt.

Give the task a finish line. State what done looks like—the test that passes,
the command that exits clean, the output that changes—so the model has something
to check itself against rather than stopping at the first plausible edit. The
wrapper prompt already asks for validation; a concrete criterion is what makes
that ask mean something.

Keep the task description faithful and specific, but do not turn it into a
method. Say what must be true at the end, not which functions to touch.

## Verification

If the user explicitly requests verification, first delegate normally. After
opencode finishes, perform only the verification the user requested. Do not redo
opencode's entire investigation or implementation.

## Final response

Treat opencode's final report—the last assistant text part in the filtered
stream—as the source material.

Rewrite it for clarity and brevity. Do not re-investigate the task merely to
improve the wording.

If opencode reports successful work, state the result directly. If opencode
reports uncertainty or failure, preserve that qualification.
