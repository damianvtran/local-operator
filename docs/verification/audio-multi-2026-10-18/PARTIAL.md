# Four-item partial — corrected merged harness, fresh live generation

**Release remains HELD for the operator.** This is evidence only; no merge, reviewer request, issue, release, tag or version bump.

Actual capture agent model **openai/gpt-6.1-sol**, host fallback from Anthropic after quota exhaustion. The served provider/model were independently checked from this capture session's transcript metadata (`provider:openai`, `model:gpt-6.1-sol`, `model_id:gpt-6.1-sol`); no role/model/tool configuration changed.

## Exact source and real calls

- UI remains pinned to merged **`5ffa8dfcb19790e2e3ecb2879ed1e4f7b95c7592`**. Production UI source is byte-unchanged; capture-only journal/scene/driver instruments are recorded in `supporting/rig-partial.patch`.
- Harness pinned to corrected merged **`4dbbc476941f0d20fa843adf25e4f7b088df8723`** (PR #2603). New isolated worktree `~/local-operator/.worktrees/multi-partial-626a`; no harness source changes.
- This is a **fresh real four-item call** to merged `execute_generate_sound_effect`, through the advertised persisted signed-in Radient route to production `https://api.radienthq.com/v1`. `radient_available` and the actual tool builder were asserted true before the call. Isolated HOME/config with a seeded copy of the existing persisted login; no raw-key tool bypass, mocked provider, invented cost or replacement audio.
- The photographed native UI **replays the exact fresh completed ToolResult and real content-addressed attachments** in an isolated conversation. It is not a claim those generations happened inside the photographed chat. Production reducer, batch widget, approved player, decoder, relay and graph execute in the compiled Electron app.
- [Original 10-item evidence comment](https://github.com/damianvtran/local-operator-ui/pull/1158#issuecomment-6251900438) retains its original harness `30fbee16` pin and immutable commit `633028bd8426b4cd5e57bcce21661dd4b03e07ca`. That successful call was not needlessly regenerated. This partial explicitly uses the newer fixed harness pin.

## One canonical batch, one real rejection

```python
await execute_generate_sound_effect(
    "partial4-call",
    {"sounds": [
        {"name":"thunder rumble", "prompt":"low distant thunder rolling, close perspective", "duration_s":0.5},
        {"name":"door creak", "prompt":"old wooden door creaking open slowly", "duration_s":0.5},
        {"name":"glass shatter", "prompt":"small glass pane shattering on a hard floor", "duration_s":31},
        {"name":"ui blip", "prompt":"short soft digital confirmation blip", "duration_s":0.5},
    ]},
    context=isolated_context, on_update=record_actual_update,
)
```

Elapsed live generation **4.48 seconds**. Actual output: `is_error:false`, `completed:3`, `failed:1`, three audio attachments, `cost_usd:0.006`, `cost.basis:billed`, **`cost_complete:false`**. Successful items cost **$0.0020 each**; the rejected item has null cost, digest and request ID. The unchanged UI prints `$0.002` per success and **`Known total cost $0.006 (billed). Unknown or late charges are not included.`** No failed-item charge is invented.

| Item | Actual provider/harness outcome | Cost | UI |
|---|---|---|---|
| thunder rumble | completed, `aud_6ad4f640e0050304150a7663` | $0.002 billed | Named playable row |
| door creak | completed, `aud_6ad4f640e73ab027182a5ce2` | $0.002 billed | Named playable row |
| glass shatter | **HTTP 400**, `error_type:audio_rejected` | **unknown/null; no invented charge** | Named failed row with actual duration reason and no player |
| ui blip | completed, `aud_6ad4f642e73ab027182a5ce8` | $0.002 billed | Named playable row |

Actual rejected row error retained by the corrected harness, verbatim:

```text
Audio generation failed on every available provider:
- Radient: This audio request could not be accepted. (duration_seconds must be between 0.5 and 30 seconds, got 31)
```

The earlier [raw HTTP 400 body](provenance/fresh-raw400.json) independently establishes the server's exact `details.reason`. This fresh batch's actual result now propagates that reason; it was not edited or synthesized into the replay.

## Native frames and preview verification

- [partial4-completed.png](partial4-completed.png): real native partial result, all four names, three playable rows, rejected row/reason, individual costs and honest known total visible together. Actual model openai/gpt-6.1-sol.
- [partial4-playing.png](partial4-playing.png): genuine first-row playback, pause control and moving playhead, not a painted state. Actual model openai/gpt-6.1-sol.
- [playback-partial.json](provenance/playback-partial.json): each of the three successful rows individually pressed and decoded, time advancing, nonzero pre-gain RMS, **gain 0/output RMS 0**, local blob source and distinct owner IDs. No host audio output.
- [partial4-result.json](provenance/partial4-result.json): exact fresh tool result, removing only source URL fields. Full args, approval, spend callbacks and actual timestamped generation updates are adjacent.
- [manifest.json](manifest.json): SHA256, dimensions and opacity for both images and hashes for the three actual MP3 attachments in `audio/partial-*.mp3`.

Final frames are opaque **RGB, 2760×2200** (1380×1100 CSS at DPR 2). Native profile uses the app's ordinary explicit opaque preference (`window-glass.json`, enabled false); captured alpha was verified 255 throughout before removing its redundant alpha plane. No compositing, drawn controls, fabricated pending frames, crops or pixel replacement.

Headless/invisible/unfocused; zero-gain safety route installed before any audio element. This partial uses the native Web Audio **`AudioContext.setSinkId({type:'none'})`** no-device processing sink in addition to gain zero: real graph processing/decoder/analyser still execute, without a physical output device. The [documented API semantics](https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/setSinkId#parameters) explicitly process audio without playing it through any output device. Constructor/play wrappers only retain references for observation and forward the originals. Re-arm refusal negative control PASS; expired local URL returns 403 exactly once and app never requests it. HOME/config isolated, CMUX/LOP/XPC stripped, telemetry/notifications off, mock keychain; exact owned process teardown PASS. No signed URL or credential is published. Full compile/fixture/native logs retained in `supporting/partial-*.log`.

The half-second clips honestly render `0:00` in the player's whole-second clock; actual provider metadata and advancing decoded subsecond times remain in the provenance. No duration is fabricated to make the clock larger. Live progress updates are retained but no loading UI is fabricated from a completed replay.

## Commands and rig correction

```sh
cd ~/local-operator/.worktrees/multi-partial-626a
env -u XPC_FLAGS PYTHONPATH="$PWD" ~/local-operator/.venv/bin/python \
  "$LOCAL_OPERATOR_SCRATCHPAD/multi-evidence/live_partial_fixed.py"

cd ~/local-operator-ui/.worktrees/multi-capture-626a
# Pinned production source plus supporting/rig-partial.patch. Existing shared
# dependency tree; no installations. Syntax/reducer preflight before native run.
node --check scripts/sounds-fixture.mjs
node --check scripts/sounds-scene.mjs
node --check scripts/run-sounds-scene.mjs
export NODE_PATH="$HOME/local-operator-ui/node_modules/.pnpm/@babel+plugin-transform-arrow-functions@7.25.9_@babel+core@7.26.10/node_modules"
export MULTI_EVIDENCE_INPUT="$LOCAL_OPERATOR_SCRATCHPAD/multi-evidence"
export LOP_SOUNDS_HARNESS="$HOME/local-operator/.worktrees/multi-partial-626a"
env -u XPC_FLAGS node scripts/run-sounds-scene.mjs
```

A first native attempt correctly found all rows and the real reason, but the rig's total-cost assertion scoped to the item-list element instead of its adjacent aggregate caption. Actual body text already held the correct `$0.006` billed known total; the assertion was corrected. Subsequent playback attempts stayed at t=0 even with bounded observation rather than a fixed 110ms delay. A discriminating graph probe measured **context state running but clock frozen at 0.005333 s**, media readyState 4/networkState 1/paused false/no media error. The hardware sink was stalled, not absent provider bytes. The capture-only repair selects the documented native no-device processing sink before playback, preserving the real decoder and zero-gain analyser. The final run directly observes all three successful decoders advancing with positive input and zero output; it does not synthesize playback state. No product edit or paid regeneration was required for any rig correction. Failed probe logs remain in capture-session scratch; final PASS logs and the full instrument patch are published here.

The initial missing-reason product gap is now resolved by merged PR #2603 and directly verified here. This document makes no new code-review/CI claim. **All requested multi-generation captures are supplied; operator review and release authorization remain outstanding.**
