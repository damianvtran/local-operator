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
it writes `frames/<sha256>` beside the record, once. This **does create one
additional stored copy** when an artifact root is not supplied or does not
contain those bytes; the zero-copy result in the measurement below applies only
to its prepopulated adapter-store fixture. It verifies bytes, size, digest and
media type before emitting a reference.

Projection is an optimization, never a prerequisite for writing a trajectory.
A block that cannot be decoded, published or verified stays **exactly inline**;
other blocks on the same event can still become references. The event receives
`record_image_fallbacks`, a list of `path` / `reason` notes outside its content.
For example, an SVG block is retained with a note like:

```json
{"path":"$.event.content[1]","reason":"ValueError: image projection does not support mime_type 'image/svg+xml'"}
```

An image fallback alone does not set `RecordSink.failure` or mark the record
incomplete, because its original bytes and text are still written. A failure of
the JSONL append itself still uses the existing incomplete-record diagnostic.
A corrupt, directory or foreign file at an adapter digest name is not rewritten:
that block stays inline on this and later events, with no dangling reference.

### Predictable image projection

Only exact MIME strings `image/png`, `image/jpeg`, `image/gif` and `image/webp`
are candidates for projection (the existing media module's supported image
set). They require string `mime_type` and `data`, strict base64 decoding and
successful strict media validation, including the existing trailing-byte rules.
SVG, BMP, aliases such as `image/jpg`, case variants such as `image/PNG`, empty
images, malformed base64, URL-shaped `data`, missing/non-string MIME values,
and otherwise unpublishable images remain inline with fallback notes. No MIME
normalization or transcoding changes the original evidence. Even a supported
MIME can fall back if its bytes or its storage cannot be verified.

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
| Additional frame copies (prepopulated adapter root) | — | 0 |

This is a 99.8139% reduction in **event journal** bytes for that fixture, not a
claim that unique screenshot storage disappears. Every new row was resolved
and compared against the baseline row, including exact frame bytes (100/100).
The screenshots were synthetic desktop/window/table scenes with textured
backgrounds, not captures of a running benchmark or a paid evaluation.

## Size annotation (not an enforced limit)

`MAX_RECORD_LINE_BYTES` (256 KiB) is an **annotation threshold, not a size cap**.
Lines can exceed it: inline image fallbacks, `image_url` payloads, other media,
text and arguments are **never refused or truncated** solely to fit a line.
Oversized lines carry `record_oversized` with
`reason: projected_record_exceeds_annotation_threshold`, `limit_bytes` (the
historical field name for the annotation threshold), and `unflagged_bytes`.
Consumers must not interpret the constant or field as an enforced upper bound.
A frame-heavy regression test exercises the real writer with twelve supported
1280×720 images and asserts a small unflagged line, one artifact and byte-equal
resolution. Separate tests require the annotation on oversized text and inline
fallbacks; they do not assert that all possible event lines are bounded.
