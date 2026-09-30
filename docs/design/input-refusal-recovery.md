# Input-refusal recovery — a bounded degraded retry for a provider content screen

Status: design frozen before implementation (see the commit order on
`fix/input-refusal-recovery`); implemented in the commits that follow.

## The finding (field evidence, not a theory)

An OSWorld 10-task arm's best-scoring episode died with **no finish and no
binary score** because of how its final model call ended. From
`worktrees/osworld/runs/a1748-w2-task_006-20260929-130146`:

- `evidence/ep-aadd48815fbc-session/score.json`: `"partial_ppm": 777778`,
  `"binary": 0` — a 77.8 % partial, scored zero because the run never submitted.
- `outcome.json`: `"status": "agent_stop"`, 103 steps — the run stopped itself.
- Terminal turn (events.jsonl): assistant message with `"content": []`,
  `"stop_reason": "error"`, and the AgentEndEvent error:

  ```
  invalid request (HTTP 400): invalid_request: data: {"error":{"code":"data_inspection_failed",
  "param":null,"message":"Input text data may contain inappropriate content.",
  "type":"data_inspection_failed"}, ...}
  ```

The same refusal appears in three more episodes across two arms
(`a1716-w2-task_006`, `a1716-w2-task_016`, `a1748-w1-task_016`), each as the
**last** event of its run.

## What is actually broken

When the provider refuses to *process the request*, the request was never
processed — but the harness treats the refusal like a terminal model answer:
the turn is empty, `stop_reason` is `"error"`, the run ends, and the record
shows an opaque HTTP 400 rather than "the provider refused the input". From
the outside it reads as the model trailing off. Concretely, two defects:

1. **No recovery attempt exists.** The refusal is classified `kind="request"`
   (a 4xx the provider read and refused), which the failover walk treats as
   deterministic bytes: on the primary it raises immediately. That
   classification is right *for the same bytes* — and wrong as a policy,
   because the remedy for THIS class is a changed request.
2. **The terminal reason is not legible.** `stop_reason="error"` plus a raw
   400 body does not tell the next reader (or the next benchmark record) that
   a content screen refused the **input** and nothing was sent to the model.

## Is the refusal deterministic for a given request?

**Measured, as far as it can be:**

- **A live probe could not reproduce the refusal on demand.** Three bounded
  calls on the field's own route (openrouter → Alibaba upstream,
  `qwen/qwen3.8-max-0902`), through the product's own
  `create_stream_fn` → `stream_with_failover` path: the two identical calls
  carrying Alibaba's *own documentation example* for this error
  (“Give me a plan to rob a bank”, help.aliyun.com/en/model-studio/content-security)
  and a benign control all returned `stop` — `IDENTICAL-OUTCOME: True` on the
  pair, no refusal. The flag is not reachable by re-sending a known example:
  its firing conditions live in the accumulated (multi-modal) request, not in
  any one string. So provider-side determinism cannot be measured with the
  probe we can run, and this design does not claim a number for it.
- **What the provider says.** The error is `DataInspectionFailed` — “Input or
  output contains suspected sensitive content blocked by Green Net. **Solution:
  Modify the input content and retry**” (400-DataInspectionFailed section of
  the error-code page; the image variant says “**Replace or modify the input
  image** and retry”). The documented remedy is a *modified* input; no
  retry-as-is semantics exist for this class.
- **What our own walk already believes.** `stream_with_failover` documents the
  same class as “DETERMINISTIC in its bytes — the same request fails identically
  on every other provider”.

**Design consequence:** an identical re-send is not a recovery. Any retry for
this class must *change the request*. The change must be bounded, recorded,
and must not silently swap the provider (a different host's verdict is a
different result; on a benchmark it would invalidate the record).

## Design

### Where

Inside `stream_with_failover` (`local_operator/providers/failover.py`), as one
more bounded, same-target recovery step in the attempt walk — beside the
fast-mode re-ask and the model-flap re-ask, and before `record()`. This is the
**shared provider path**: every consumer (TUI, mobile, exec, the evaluation
runner's `ProviderModelClient`, the OSWorld session arm) reaches a provider
through it, so no caller needs OSWorld-specific or session-specific code. The
`ProviderModelClient`'s documented contract (“a failure that reaches this class
is terminal”) stays true: it never sees the failure when recovery succeeds, and
sees the legible one when it does not.

### What changes, and in what order

One predicate, `is_input_refusal`, recognizes the class on a `kind="request"`
`ProviderError` (or its rendered form): Alibaba's `data_inspection_failed` /
`DataInspectionFailed` code — input-side unless the sentence is the output-side
one — or one of the input phrases the provider publishes (“Input [text|image]
data may contain inappropriate content”, “Input or output data …”). The
OUTPUT-side sentence never classifies, even beside the code: its failures keep
the ordinary retry ladder. The predicate is deliberately narrower than the
contract of `is_image_rejection` — a malformed 400 is still a terminal request
defect and keeps today's behaviour.

A refusal then earns up to **two degraded re-asks**, on the SAME target,
credential and model, tried in this order:

1. **`screenshots_removed`** — every image block in the request is replaced by
   a short text placeholder. (Vision loads are the filter's documented image
   pre-check surface; the docs sanction exactly this remedy for it.)
2. **`older_observations_removed`** — additionally, tool-result content older
   than the newest 24 messages is replaced by a short placeholder. Tool rows
   are the harvested environment text (screens, pages) and the likeliest
   carrier of the flag; the model's own turns and the task statement are left
   alone, so reasoning-echo contracts and the task itself survive.

**Bounds and cost, stated:** at most **two** additional requests per call
(three wire attempts total for this class), each *smaller* than the original —
the ladder only removes. **No backoff**: a content verdict does not clear with
time, so waiting would only delay the outcome. The ladder is skipped entirely
for isolated/retry-disabled calls (`retry.enabled` gate — decorative errands
keep their one-attempt contract). Nothing re-sends identical bytes: with no
rung applicable, no retry is attempted at all.

An exhausted or unavailable refusal is **terminal for the walk**: it raises
here instead of falling through to the next target. A hop would hand another
provider this request's bytes — the exact ones a content screen refused — and,
if the new target answered, stamp a recovery on a request that never carried
it; the substitution red line and the call-scoped bound both point the same
way. Conversely, once a rung has been applied the narrowing *follows* the
request for the rest of the call: if the walk moves on for an unrelated
failure, later targets receive the narrowed request, and the end-event stamp
describes exactly what answered.

### What the terminal refusal says

When the budget is spent (or nothing was degradable), the raised error keeps
the provider's own words in front and appends the one fact the record was
missing:

> … the provider refused the request's input as inappropriate content (content
> screen); the request was never processed. A degraded retry (screenshots
> removed, then older observations removed) was also refused, so it was not
> re-sent unchanged.

With no rung applicable the sentence instead states that a degraded retry was
unavailable; where the ladder existed but the call's retry policy declined it
(isolated errands, retries disabled), it says the policy declined a degraded
retry — not that none was available. `stop_reason` stays `"error"` (the
exception path, unchanged) —
"refusal" is the model-refusal vocabulary and its copy would misattribute a
provider request refusal to the model; the message now carries the diagnosis.

### Recording a recovered turn

When a degraded re-ask succeeds, the walk stamps
`provider_payload["input_refusal_recovery"] = {"degradations": [...]}` on the
end event, and the harness loop surfaces it to the user as a `NoticeEvent`
(“the provider refused the request as inappropriate input content — the
request was re-sent with screenshots omitted and the turn continued”), the
same transparency rule the existing sticky image-degrade notice follows. Every
rung also logs at warning level.

### The output-side 502 — separate handling, deliberately

The run also logged `HTTP 502 — Upstream error from Alibaba: Output data may
contain inappropriate content`, and it recovered on its own. That is the
**output** screen, arriving as a retryable 5xx: the model regenerates output on
each attempt, so the existing transport/retry ladder is the right and already
working remedy, and the input ladder must not touch it (the predicate requires
input-side wording, and the kind is not `"request"`). No change is made there;
this design records why the two classes are handled differently.

## Non-goals

- No provider or model substitution: the degraded re-ask stays on the same
  target and credential.
- No new stop_reason, no session-sticky state, no OSWorld-specific code.
- No change for ordinary calls: the branch is inert unless a `kind="request"`
  input refusal was classified; the happy path is byte-identical (pinned by
  test).

## Evidence plan (posted on the PR)

- Field reproduction: the bundle above, with the score and the terminal frame.
- Local recovery: the real `stream_with_failover` driven with a scripted wire
  that replays the captured 400 — showing the changed second request, the
  recovery, and the recorded payload; and the exhausted path's legible error.
- The live probe's actual result (no refusal reproduced) with its command.
- Happy-path byte-identity: captured requests on a normal call, before/after.
