# Multi-generation release-gate evidence — 2026-10-18

**Release remains HELD for operator review.** This evidence does not merge a PR, request a human reviewer, create an issue, bump a version or publish a release.

Capture agent: coder. Actual served model **openai/gpt-6.1-sol**, after host fallback from **anthropic/claude-sonnet-5-5** on quota exhaustion. No role/default model/tool settings changed.

## Source and truthful boundaries

- UI pinned to merged **`5ffa8dfcb19790e2e3ecb2879ed1e4f7b95c7592`**; harness pinned to merged **`30fbee16141dd7f237fd5893202cd572012ddc7d`**. Separate detached worktrees; unrelated moves of shared main are not used.
- Fresh real provider calls made by the merged `execute_generate_sound_effect` tool against production `https://api.radienthq.com/v1`. The persisted signed-in Radient identity was seeded into an isolated runtime auth database; `radient_available` and the actual tool builder were asserted true before spending. No exported raw-key bypass, provider mocks, invented costs or synthetic audio.
- The screenshots are **recorded-artifact replay of those fresh ToolResults** in an isolated conversation, not a claim that the generations took place inside the photographed chat. The production journal reducer, batch widget, approved player, attachment relay, decoder and audio graph run in the real compiled Electron app. The widget/product source is unchanged from the specified UI head.
- `supporting/rig10.patch` records capture-only journal/scene/driver changes. The replay keeps `provider_payload.details`, names, costs and attachment identities from the real tool result. Source URLs are removed from retained provenance and local attachments are used by digest; no signed URL is published.
- Real native headless window: invisible and unfocused, no notification/telemetry output, isolated HOME/config, inherited CMUX/LOP/XPC flags stripped, mock keychain switch, exact owned process groups reaped. The shipped **explicit opaque window preference** (`window-glass.json`, enabled false) is seeded in the isolated profile, rather than compositing a transparent capture after the fact. Final full frames are 2760×3600 (1380×1800 CSS at DPR 2), opaque RGB.
- A zero-gain graph is armed **before any audio element exists**. Decoder time advances on each row, pre-gain RMS is positive, post-gain RMS is zero. No host audio output. Re-arming the graph is a tested refusal. The local expired-URL control returns 403 once, and the app never calls it.

## 10-item all-success batch

One real call:

```python
await execute_generate_sound_effect(
    "multi10-call", {"sounds": exact_reference_items},
    context=isolated_context, on_update=record_actual_update,
)
```

The name/prompt/duration list in `provenance/multi10-args.json` is copied verbatim from the server lane's supplied `multi10-reference.json` (session `6c9c8f09af70/scratchpad/audio-live`). All requested durations are 0.5 s. One batch approval is recorded in `provenance/multi10-approval.txt`; the tool owns its two-worker bounded queue.

**Actual result:** `is_error:false`, `completed:10`, `failed:0`, 10 attached audio blocks, per-item `cost_usd:0.002`, `basis:billed`; total `cost.amount_usd:0.020`, `basis:billed`, `cost_complete:true`. Elapsed live generation: 11.78 s. Complete request IDs, cost objects, units and digests are retained in `provenance/multi10-result.json`; audio file SHA256s and image hashes are in `manifest.json`.

| # | Name | Per-item billed cost |
|---|---|---|
| 1 | thunder rumble | $0.0020 |
| 2 | door creak | $0.0020 |
| 3 | glass shatter | $0.0020 |
| 4 | ui blip | $0.0020 |
| 5 | rain window | $0.0020 |
| 6 | coin drop | $0.0020 |
| 7 | sword unsheathe | $0.0020 |
| 8 | camera shutter | $0.0020 |
| 9 | water drop | $0.0020 |
| 10 | crowd gasp | $0.0020 |

**Total billed: $0.0200.** The unchanged UI formats those amounts as `$0.002` and `$0.02`; the numeric values and billed basis match. All 10 rows, names, transports, individual costs and total are visible together.

- [Completed ten-row frame](multi10-completed.png): real batch result, 10 completed / 0 failed, all names/costs and preview controls visible. Model fallback: openai/gpt-6.1-sol.
- [Actual playing frame](multi10-playing.png): thunder rumble's real pause control and played waveform state; the other nine independently previewable rows remain visible. Model fallback: openai/gpt-6.1-sol.
- [Per-row playback measurements](provenance/playback10.json): each of the 10 rows was pressed independently and decoded from a local blob with distinct owner identity, positive input RMS, output RMS 0 and gain 0. The browser decoder observes 0.48 s for these MP3s; the provider/container attachment duration is 0.52244898 s. Neither is replaced with the requested 0.5 s.
- [Generated MP3 files](audio/): these are the actual fresh attachments, named for review, never replacement/synthetic samples.

## Partial-failure status (pending corrected-head capture)

The first live 4-item call on harness `30fbee16` produced **3 attached successes + 1 failed item**, `is_error:false`, known billed total `$0.006`, `cost_complete:false`. The failed glass-shatter request has `http_status:400`, `error_type:audio_rejected`, null request ID/digest/cost. It did not discard siblings or invent a charge.

A discriminating assertion found that this pinned harness drops the server's actionable `details.reason`, retaining only the generic error `This audio request could not be accepted.` This is a real limitation, not relabelled as a pass. The manager is landing a narrowly scoped fix; the partial render is intentionally **held** until a fresh live 4-item call on the corrected merged harness head. No detailed reason is synthesized into the original ToolResult.

[Fresh raw HTTP 400 response](provenance/fresh-raw400.json), independently repeated against production with the same persisted identity and invalid duration (no audio/no charge):

```json
{"status":400,"body":{"code":"audio_rejected","details":{"reason":"duration_seconds must be between 0.5 and 30 seconds, got 31"},"error":"This audio request could not be accepted."}}
```

## Reproduction commands and preflight

Pinned source worktrees: `~/local-operator/.worktrees/multi-capture-626a` and `~/local-operator-ui/.worktrees/multi-capture-626a`. Existing shared Python/pnpm/Electron dependencies only; no dependency installation. The source-only checks are preflight, **not** the native evidence.

```sh
# Live invocation script, preflight and exact fresh args/results retained in
# capture session 626adcfcc558/scratchpad/multi-evidence; credentials never printed.
cd ~/local-operator/.worktrees/multi-capture-626a
env -u XPC_FLAGS PYTHONPATH="$PWD" ~/local-operator/.venv/bin/python \
  "$LOCAL_OPERATOR_SCRATCHPAD/multi-evidence/live_batches.py"

# Apply supporting/rig10.patch in the pinned UI worktree. INPUT contains the
# fresh scrubbed results + their local content-addressed attachment store.
cd ~/local-operator-ui/.worktrees/multi-capture-626a
node --check scripts/sounds-fixture.mjs
node --check scripts/sounds-scene.mjs
node --check scripts/run-sounds-scene.mjs
export NODE_PATH="$HOME/local-operator-ui/node_modules/.pnpm/@babel+plugin-transform-arrow-functions@7.25.9_@babel+core@7.26.10/node_modules"
export MULTI_EVIDENCE_INPUT="$LOCAL_OPERATOR_SCRATCHPAD/multi-evidence"
export LOP_SOUNDS_HARNESS="$HOME/local-operator/.worktrees/multi-capture-626a"
env -u XPC_FLAGS node scripts/run-sounds-scene.mjs
```

Actual preflight: history HTTP 200; production reducer preserves all 10 attachments and all 10 artifact details; full source build including main bytecode; renderer's compiled-source guard; isolated backend connection and no connection to operator backend. Native scene: all ten playback checks PASS; graph rearm refused; expired URL unused; exact teardown PASS; `ALL CHECKS PASSED`. Actual build/fixture/scene logs retained in `supporting/`.

One failed rig attempt omitted the production `provider_payload.details` carrier; it was corrected and the reducer preflight strengthened before the successful dispatch. A later image preflight caught the Mac glass default's nonopaque alpha; those frames are not published, and the final capture uses the app's ordinary explicit opaque preference. No paid regeneration was needed for either rig correction.

No fabricated loading/pending frame is supplied: the visible captures replay actual completed fresh results. Actual generation progress updates are retained with elapsed timestamps; they are not portrayed as a continuously live UI stream. No new code review, full suite, merge or release is claimed by this evidence task. Partial-failure final capture and operator review remain outstanding gates.
