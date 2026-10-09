# Output attachments: agent-produced media, first-class

Status: implemented in `feat/output-attachments` (harness) + `feat/output-attachments-ui`
(desktop UI). This is the contract the image-generation feature builds on; the
surfaces lane consumes it and adds no second binary transport.

## The problem

A tool that produces a binary — a generated image, a downloaded video, a
captured frame — had no first-class way to hand it to the conversation. The
generated file typically arrived as a TEXT path: the model could cite it, the
user could not see it, and nothing downstream (desktop UI, TUI, phone) had a
reference to render. The earlier condensed-view image strip only taught
*existing* markdown/legacy images how to fit the folded view; it did not give
producers a contract.

The harness already had the other half: user-pasted images ride
`ImageContent` blocks, durable rows externalise them into a content-addressed
store (`session/attachments.py`), the desktop UI fetches them by digest
through `/v1/desktop/sessions/{id}/attachments/{digest}`, the TUI mounts them
as `ImageBlock`s, and the phone fetches them by `(entry, index)` from the
mobile daemon. Output media reuses EVERY one of those mechanisms; what it adds
is a generic, pointer-shaped block and one registration call.

## The contract

### The block — `AttachmentContent` (`harness/types.py`)

```jsonc
{
  "type": "attachment",        // discriminant on LIVE frames only
  "kind": "image",             // "image" | "video" | "audio" — REQUIRED, no default
  "content_type": "image/png", // MIME of the cached bytes
  "attachment": "<sha256[:32] digest>", // local copy in the attachment store
  "source_url": "https://…",   // provider URL: provenance + re-fetch fallback
  "size_bytes": 123456,
  "width": 1024, "height": 1024,
  "duration_s": 3.2,
  "name": "flux-dev-01.png"    // short display handle, never a filesystem path
}
```

All fields nullable except `kind`; absent where unknown. Generic by
construction — video and audio ride the same fields; nothing assumes pixels or
a screen.

Four decisions are load-bearing and recorded here because each one is a place
a future change can silently break the contract:

1. **`kind` is required with no default.** The transcript encoder dumps with
   `exclude_defaults=True`: a defaulted `kind` is ABSENT from every durable
   row, which would make an image artifact's row (`{content_type, attachment}`)
   indistinguishable from a legacy image reference. A required field is always
   on the row, so the durable identity is an invariant of the format rather
   than a producer habit.
2. **No `data` field.** Producers cache bytes at registration time; the block
   is a pointer. Nothing inlines artifact bytes into a frame, a row or LLM
   history — video especially cannot afford it. Consumers read the store on
   demand; a missing store degrades visibly (unavailable receipt / 404), never
   silently (an empty `data` is how a missing image disappears without trace).
3. **A sibling mechanism to `spill://`, not the spill store.** Spill
   (`tools/spill.py`) is LRU-evicted under a byte ceiling and is ALLOWED to
   forget content a transcript still references — correct for oversized TEXT
   tool output, fatal for media a conversation's rows point at. The
   content-addressed attachment store never evicts; its bytes are binary;
   its facts travel as sidecars. The store module docstring carries the whole
   argument, and this is one mechanism for BOTH directions (user pastes and
   output artifacts) so surfaces resolve one digest format, not two.
4. **The block is never sent to a model.** Provider dispatch skips unknown
   block types, and that is deliberate: the artifact is for the USER's
   surfaces, and the model's reading of it is the caption text that rides
   FIRST in the result (`_image()`'s existing convention). A tool that wants
   the MODEL to see the pixels also attaches an `ImageContent` — both blocks
   can coexist in one result.

### Registration — the one call (`session/attachments.py`)

```python
from local_operator.session.attachments import cache_media

block = cache_media(
    raw,                      # decoded bytes (not base64)
    "image/png",              # content_type
    kind=None,                # derived from content_type when omitted; refused outside the vocabulary
    name="flux-dev-01.png",   # optional display handle
    source_url="https://…",   # optional provider URL
    width=None, height=None,  # images: sniffed from the header for free when omitted
    duration_s=None,          # audio/video
)  # -> AttachmentContent | None; None is a normal outcome, never an exception
```

Content-addressed (`sha256[:32]` of the bytes), idempotent, deduplicated
across sessions, written under `<config>/attachments/<digest>.bin` with a
`<digest>.json` sidecar. `None` on any failure (empty payload, non-media
MIME, read-only home, full disk) — the caller degrades to a text error and
must NOT build a block around a failed registration.

### Where it rides

`ToolResult.content`, after a caption `TextContent`:

```python
ToolResult(
    tool_call_id=...,
    tool_name="generate_image",
    content=[TextContent(text="Generated flux-dev (1024x1024, 412 KB)."), *blocks],
    details={...provider, model, prompt, seed, generation_id, cancel handle...},
)
```

Convention: the caption is self-sufficient (it is the ENTIRE result for
text-only consumers and for the model); provenance and lifecycle data go in
`details`, not the block. N artifacts per result are legal.

### ref → pixels

- **Local cache first**: `attachment` digest →
  `GET /v1/desktop/sessions/{session_id}/attachments/{digest}` (bearer auth,
  `nosniff`, `no-store`; the digest is a 32-hex path parameter and is
  traversal-gated by construction; the served mime allowlist covers image,
  audio and video types and falls back to `application/octet-stream`).
  Client caches keyed by digest are safe — the digest IS the content.
- **Phone**: the projection emits `{"index", "mime_type"}` references on the
  tool row (and keeps emitting them for user rows); the daemon serves bytes
  by `(entry id, index)` through `/api/sessions/{id}/image`.
- **TUI**: reads the store directly (`tool_result_image_blocks`); kind
  `image` mounts as an `ImageBlock`, others contribute nothing yet.
- **Fallback**: digest absent (session moved without its store, hand-pruned
  store) → surfaces may use `source_url` where fetchable, else their existing
  unavailable state. Never break the row.

### Durable shape, detection, and the coercion layer

Durable rows drop `type` (it is the pydantic default, and the encoder excludes
defaults). Readers detect an artifact by `kind` + a media fact
(`content_type` / `attachment` / `source_url`) — NEVER by `type`.

`harness.types.coerce_content_blocks` routes raw block dicts to their models
BEFORE the `Content` union parses them, for a measured reason: a pydantic
smart union landed a durable artifact dict on `TextContent` (leftmost member,
accepts any dict by ignoring unknown keys) where the artifact read as EMPTY
TEXT and vanished with no error anywhere. Routing keys on the facts above;
every legacy shape is handed to the union untouched. The same rule protects
the two transcript passes (`_externalize_attachments` skips artifact blocks —
their reference is payload, not an encoding convention —
and `_resolve_attachments` skips them too, so the digest SURVIVES replay and
`compact_file` folds).

### Back-compat

Additive only. Old rows (inline images, digest references, audio, markdown
images and plain paths in text) parse and render exactly as before; the
existing regression tests for those shapes are unchanged and still run
(`tests/unit/session/test_attachments.py`). An older BUILD reading a NEW row
is out of scope: the artifact block parses as empty text in a build without
the coercion, which is invisible rather than broken, and the desktop UI /
harness update together.

## Surfaces map (who owns what)

- **Harness** (this repo): block + coercion, `cache_media`, externalize/
  resolve guards, desktop route allowlist, TUI mount adapter
  (`session_presentation.tool_result_image_blocks` + its call sites), mobile
  projection references + daemon byte resolution.
- **Surfaces lane** (`feat/imagegen-surfaces`): desktop UI presentation
  (inline + folded/canvas treatments, progress UX), TUI progress card,
  mobile web + native app rendering. The UI data path (reducer extraction
  into `images[]` with the additive metadata fields) is the thin piece in
  this contract's UI branch.
- **Harness tool lane** (`feat/image-generation-tool`): the producer —
  `generate_image` returning captions + `cache_media` blocks + details.

## Evidence

The PRs carry end-to-end proof, not unit counts: a real artifact generated
through the real tool path, rendered in the real desktop UI and the real TUI
(screenshots), surviving a reload, and appearing on the phone web relay —
with the commands that produced each frame, and the tolerant-reader proof
against a real pre-change transcript.
