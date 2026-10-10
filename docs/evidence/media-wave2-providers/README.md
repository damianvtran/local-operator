# Media wave-2 providers — tool-level capture

The PR's tool-level evidence for the artifact-generation interface + provider
breadth (`feat/media-provider-breadth`): one run that drives the REAL
`generate_image` path end to end over a **scripted socket** — spend $0.00, no
real endpoint contacted, no live credential involved.

What it exercises (nothing stubbed in the data path except the socket):
resolver (`resolve_image_route`) → cascade walk → `_run_route` → call-time
credential resolution against a synthetic store → `run_openrouter` →
asset decode → `RungResult` → the tool's caption / attachment / details
assembly, plus the no-provider refusal path.

## Run it

```sh
ISO=$(mktemp -d)
env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" \
    PATH="$PATH" TERM=xterm-256color \
    <worktree>/.venv/bin/python docs/evidence/media-wave2-providers/capture_tool_level.py
```

(The script refuses to run with `OPENROUTER_API_KEY` in its environment — an
exported key would bypass the stored row the capture is about — and redirects
the attachment store to a scratch dir; the operator's live config is never
touched.)

## What the output reads (captured 2026-10-09 from this script at the branch tip)

- `route: openrouter | reason: An OpenRouter key is stored.` and the rung
  report shows the six earlier rungs `False` and `openrouter` `True` — the
  append-only order resolving to the synthetic row.
- the tool's caption: `Generated 1 image with OpenRouter
  (bytedance-seed/seedream-4.5), square_hd — attached to the session (digest
  f8b9599fb418b563e2882fe3d64a8a63). Cost $0.04.` — the reported
  `usage.cost` surfacing in the caption.
- `store round-trip: … -> 73 bytes, byte-equal: True` — the attachment
  landed in (and reads back from) the store surfaces fetch.
- the request the REAL executor sent: `POST
  https://openrouter.ai/api/v1/images`, bearer masked by the script, body
  `{"model": "bytedance-seed/seedream-4.5", "prompt": …, "aspect_ratio":
  "1:1"}` — no `n`, no `seed` (their documented "only when asked for"
  semantics).
- the refusal path: `route: none` with the full seven-rung remedy sentence
  (`lop login radient` … `lop login openrouter`).

## Scope and honesty

This is the wave-2 rung evidence, NOT a live validation: no provider was
called for real in this wave (operator spend policy). The four new rungs'
live acceptance probes are listed per rung in
`docs/design/image-providers.md` § Live validation status — openai-sub /
Google / xAI live validation is pending credentials.
