---
name: guard
description: "Radient Guard is on for this member's Team org: when you spot an exposed credential or a guardrail blocking legitimate work, draft a report for the user to send."
requires: radient-guard
---

# Radient Guard: self-reporting

The user's Radient Team organization turned on Radient Guard. Guard gives the
org's admins guardrail stats from members who opt in, and lets members send
one-shot reports (report sending isn't live yet: a confirmed report stays on
this computer, and the user is asked once more before it is sent). You help by **drafting** reports; the user decides
whether to send them.

## When to draft a report

Draft one as soon as you notice either of these in your own work. Don't wait
to be asked:

- `credential_exposure`: a credential is exposed, for example an API key, token,
  password or private key printed in command output, committed to a file,
  written to a log or pasted into a message.
- `excessive_guardrail`: a guardrail blocked or asked about work that was
  legitimate and in scope, and the block cost real progress. Don't draft one
  when the guardrail stopped something genuinely risky, even if it slowed you
  down.

Use `other` only when the user asks you to flag something that fits neither.

```
guard(action="report_draft", kind="credential_exposure")
```

The report carries only the kind. The tool has no field for detail, and none
is needed:

- **Never** repeat the credential's value anywhere: not in the report, your
  reply, a memory, a commit or a message.
- **Do** tell the user, in your reply, where it leaked and which credential to
  rotate, so they can fix it. Keep that location out of everything else: the
  report has no field for it, and it doesn't belong in a memory, a commit, or a
  peer or Team Bridge message.
- Still fix the exposure where you can (remove it from the file, tell the user
  to rotate the key). The report doesn't replace the fix.

## What happens next

The draft waits in `/guard` on this computer. Tell the user plainly:
"I drafted a Guard report (suspected exposed credential). Send or discard it in
/guard." You can't send it, and you shouldn't ask the user to let you. Drafting
the same kind again before they answer returns the open draft.

If the user asks who sees it: the org's admins see only how many reports of
each kind were sent, and only for a kind at least 5 people sent that day. No
one at the org can open a report.

## Status

`guard(action="status")` shows, per org, whether this computer shares Guard
stats, is waiting for the user's choice, or is paused. It never shows numbers.

Whether to share is the user's choice, made in `/guard`. You can't answer that
prompt: no tool can, and `/guard` opened from an agent's shell or console pane
is read-only. If it's pending, mention `/guard` once and carry on with the
task. Never pause work for it.
