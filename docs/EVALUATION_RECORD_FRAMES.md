# Session-arm event frame references

Session-arm `events.jsonl` records now replace inline `ImageContent` blocks
with `image_reference` blocks. This is a **record-only projection**: the agent,
provider, adapter protocol and normal CLI JSON stream still receive inline
images. Text, IDs, errors, timings and tool-call `arguments` / execution-event
`args` are unchanged.

```json
{
  "type": "image_reference",
  "mime_type": "image/png",
  "artifact": {
    "sha256": "<64 hexadecimal characters>",
    "media_type": "image/png",
    "byte_count": 123456,
    "path": "../artifacts/<digest>"
  }
}
```

Other image metadata (for example capture provenance) remains on the block.
`artifact.path` is relative to the directory containing `events.jsonl`.
Preserve the complete run directory when copying evidence: adapter artifacts
can be shared by multiple episode directories. The writer reuses the adapter's
`artifact_root/<sha256>` publication when present. For images not yet persisted,
it writes `frames/<sha256>` beside the record, once. It verifies bytes, size,
digest and media type before emitting a reference. Corrupt publications and
storage failures go through `RecordSink.failure` and the existing session-arm
`record_incomplete` diagnostic rather than publishing a dangling reference or
turning a completed paid episode into an execution failure.

## Reading old and new records

Generic JSON readers can keep consuming event fields and text. An inline-only
`ImageContent` validator rejects `type: image_reference` explicitly. To restore
the old shape, including exact original image bytes:

```python
import json
from pathlib import Path
from local_operator.evaluation.record_frames import resolve_record_images

record = Path("run/evidence/episode-session/events.jsonl")
for line in record.read_text().splitlines():
    event = resolve_record_images(json.loads(line), record)
    # event now contains ordinary type=image, data=<base64> blocks.
```

Resolution also accepts legacy inline rows unchanged. Missing artifacts, digest
or byte-count mismatches and wrong media types fail loudly through the existing
artifact verifier; never ignore these exceptions when claiming complete
trajectory evidence. Only resolve trusted local records: reference paths can
point to an adapter store outside the episode directory.

Reader census for this change:

- `local_operator/evaluation/triage.py` discovers `events.jsonl` roots but does
  not interpret session-arm image blocks; no change needed.
- `scripts/repro-enospc-record.py` counts JSONL lines and reports line-size
  summaries; the new shape remains ordinary JSON and its readers still work.
- `local_operator/evaluation/evidence/{store,verify}.py` reads a **different**
  typed runner journal named `events.jsonl`; session-arm `RecordSink` does not
  write that schema and it is unchanged.
- Existing session-arm tests read event text and metadata, not inline images.
- `scripts/analyze_offline_subset.py` does not exist in this repository at the
  change's base (`c7f065f3ce`). External inline-image analysis scripts must call
  the resolver above, or reject `image_reference`; treating it as an image with
  missing `data` silently is unsupported.

## Measured writer comparison

A synthetic 100-turn episode, each turn carrying one distinct 1280×720 PNG
(229,342–235,047 bytes), was written through the actual baseline
`c7f065f3ce` `RecordSink` and the new writer using identical typed `turn_end`
events projected by `printable_event`:

| Metric | Inline baseline | Referenced frames |
| --- | ---: | ---: |
| `events.jsonl` bytes | 30,994,490 | 57,682 |
| Largest new event line | — | 578 |
| Existing adapter artifact bytes | 23,219,594 | 23,219,594 |
| Additional frame copies | — | 0 |

This is a 99.8139% reduction in **event journal** bytes for that fixture, not a
claim that unique screenshot storage disappears. Every new row was resolved
and compared against the baseline row, including exact frame bytes (100/100).
The screenshots were synthetic desktop/window/table scenes with textured
backgrounds, not captures of a running benchmark or a paid evaluation.

## Size guard

Normal projected lines must fit within `MAX_RECORD_LINE_BYTES` (256 KiB).
Unusually large text or arguments are **not truncated**. Such lines carry
`record_oversized` with `reason: non_image_payload_exceeds_limit`, `limit_bytes`
and the unflagged byte count. A frame-heavy regression test exercises the real
writer with twelve 1280×720 images and asserts a bounded, unflagged line, a
single artifact and byte-equal resolution. A separate test requires the explicit
marker on oversized text. This is a diagnostic bound, not permission to discard
reviewable evidence.
