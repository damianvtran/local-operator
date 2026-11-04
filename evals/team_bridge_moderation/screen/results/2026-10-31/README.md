# Screen-bench results, 2026-10-31

The per-run directories (`trials.jsonl`, `summary.json`, `summary.md`) and the write-ups
(`baseline.md`, `patch-c.md`, `variant-d.md`) for the study registered in
[`../../PROTOCOL.md`](../../PROTOCOL.md). Both candidates were rejected; the template is unchanged.

## Read this before re-summarising

These runs were produced by bench commit `52a98363305`, before the round-1 and round-2
remediation. They use the **legacy trial schema**:

- The temperature field is `temperature_sent` (later `temperature_requested` plus
  `temperature_reached_provider`). Loaded at a later head, the new columns read `None`.
- A runner stream refusal was recorded as `provider:_ScreeningStreamRefused`, **with no
  stop reason**. Later heads record `model_refused:<reason>` only when production's own
  hold code is `model_refused`, and send other stops to `answered:*` or
  `provider:other:stream_<reason>`.

So the "model refusals" in `baseline.md` (55 on Opus across all splits) are runner
**stream-refused stops whose reason was not recorded**. They were most likely model
refusals, but this data cannot prove it.

Re-summarising these files at the current head counts them as provider failures (invalid,
excluded), not as `model_refused`, so the refused-as-held sensitivity figure will not
include them. The adoption decisions are unchanged under both readings: excluding them
as invalid is the protocol's primary figure, and counting them as held was checked when
each decision was made (PROTOCOL.md A3).

Every legit message in all 12 runs has at least 3 valid trials, so the later
≥3-valid-trial floors on both arms of the newly-blocked rule change no result.

Read the false-block column split by disposition. These are pre-cache study results;
the historical retry/re-draw interpretation is preserved in
[What a false block means over time](../../README.md#what-a-false-block-means-over-time),
alongside the dated **2026-10-31 retrospective clarification**: #3220 shipped in
v0.94.2, and cached model-answer holds do not clear merely with elapsed time or retries.
That later policy clarification does not rewrite these runs or adoption decisions.
