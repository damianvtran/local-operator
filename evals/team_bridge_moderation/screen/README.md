# Screen bench: protocol, corpora, templates, results

The screen-only bench itself is `scripts/eval_bridge_screen.py`; its usage and the
rates it reports are described in
[`../README.md#screen-only-bench-the-inbound-screener-as-a-classifier`](../README.md#screen-only-bench-the-inbound-screener-as-a-classifier).

- [`PROTOCOL.md`](PROTOCOL.md): the pre-registered evaluation protocol and its
  amendments (A1 corpora sealed, A2 candidates frozen, A3 rule correction).
- [`corpus/`](corpus/): the sealed `adv`, `legit` and `replay` corpora.
- [`templates/`](templates/): the embedded `bridge-screen-v2` bytes and, under
  `candidates/`, the two REJECTED candidates kept as a record.
- [`results/2026-10-31/`](results/2026-10-31/): the runs, summaries and write-ups.

## What a false block means over time

A legitimate message that was not delivered is a false block; the two ways differ in
duration.

- **A `bounce` is terminal.** The verdict commits a write-once receipt and the hold row
  is retired rather than re-driven, so a bounced message is a permanent block.
- **A `hold` is not terminal today.** `screening.py` maps the model's own uncertainty
  verdict to `hold_code="model_hold"`, the row is persisted so it comes due (no
  `retry_at` counts as due), and the connector's `retry_due_holds` re-drives it through
  the same spine: a fresh model call and a fresh verdict draw, backed off over time. So
  a hold-block may later deliver (good for availability), and a hold on an adversarial
  message may later admit - the re-draw exposure.
- **Read the false-block column split by disposition.** The summary already reports
  `false_hold_rate` and `false_bounce_rate` separately: a candidate adding `hold`-blocks
  is riskier than one with the same number of `bounce`-blocks, because a hold is a live
  draw.
- **A retry cache is planned** (hotfix lane, not open yet): a model-answer `hold` becomes
  reusable under a receipt key (body_sha256 + template_sha256 + model + policy
  revision), route failures stay uncached, and only a republish re-draws. Once it lands,
  a `hold`-block is a permanent block.
