# Owner audit before session-file content

## Boundary

This change hardens the shipped **S1 session-file read API**, not all mesh
traffic. It grants no new access. The existing `view` capability, owner-session
scope, attachment-transcript membership, scratchpad parser, no-follow open,
credential denylist, configuration-directory refusal and source fingerprint
checks remain prerequisites. It does not add outside-scratchpad file serving,
requesting-session identity, file writes, or UI behavior.

`net_sync` whole-session replication is a separate existing transport and is
**not covered by this S1 admission event**. Its transcript/attachment export is
not a preview/download alias; do not infer that every mesh byte is audited here.

## Admission and durability

Every owner `net_file_get` **fetch** phase calls the same admission callback
before reading file content for its response. The owner emits
`session_file_read_admitted` through the existing rotating `AuditLog` writer.
The callback requires an enabled, non-degraded writer and an exact-row receipt
while holding the writer's existing lock. The receipt binds the requested
sequence to the inode/generation that actually contains it, not the latest
bookkeeping sequence or an increased sync counter. Live-file bytes and directory
publication (including newly created ancestors) must be fsynced. Rotation closes
the streamed gzip, fsyncs its complete bytes/trailer and published directory name
**before** unlinking the prior live copy; archive/publication failure preserves
that copy and refuses content. The active receipt's generation is not pruned
before acknowledgement. Later `audit_rotated` bookkeeping does not invalidate a
healthy admission or certify an unrelated replacement inode. A failed append,
file/archive/directory fsync, buffered-only acceptance, disabled writer or already
degraded writer refuses that fetch with `audit_unavailable`. Refusal/control frames are
not FILE content. Direct fetch calls without an admission callback also refuse.
Other existing events retain their existing batching/durability policy.

The row records the authenticated **requester device** as `actor` (device kind),
the owner session as `session_id`, and an allow decision, offset, and
`admitted_bytes` in `detail`. No trusted requesting-session identity exists in
this protocol; it is intentionally absent, not copied from an arbitrary peer
field or confused with the owner session. Network and epoch come from the link.

`admitted_bytes = min(bounded chunk limit, max(0, owner stat size - offset))`
is a **planned maximum for this fetch**, never actual transfer completion.
The actual read is bounded to that admitted span, including at/beyond EOF.
A source read error, truncation, interrupted response or lost link can leave
an allow row with fewer (even zero) delivered bytes. A resumed fetch is a new
admission: authorization and source checks happen again for each request.

References contain only a canonical pad-relative `scratchpad://` reference,
an attachment digest, or a typed `reference:sha256:<hex>` fallback. The fallback
hashes the UTF-8 **canonical relative reference string**, never file contents;
it cannot trigger a fetch-content read before admission. The owner checks the
existing sanitizer's actual serialized detail budget as well as string/control
scrubbing: legal short Unicode paths can expand under JSON ASCII escaping.
Required acknowledgement refuses missing/changed detail fields, including the
actual rendered row, rather than silently accepting `truncated=true`. Neither
caps nor authority are widened. Existing stat/verify content hashing is separate
metadata/integrity behavior, not a fetch admission or an all-reads guarantee.
Compression/content hashing remain streamed and fetch buffering remains bounded
by the existing chunk size.

## Source inventory

| Surface | Source path to owner admission |
| --- | --- |
| CLI `network files pull` | `network/cli.py` → local `session_file_pull` → `files.pull_file` / `_Asker.ask` → peer `net_file_get` → `files.make_handler` / `serve_file(fetch)` |
| Transcript preview/media, canvas attachments and waveform source | `server/utils/desktop_sessions.py` `peer_blob` → local cache lookup first → `_fetch_peer` → `remote_files/fetch.py` `_pull` → `session_file_pull` → same owner fetch |
| Files-panel content/range and download | `server/utils/remote_session_files.py` `_resolve` / `_opened` → `peer_blob` or `_fetch_peer` → same pull and owner fetch |
| Library content | `remote_session_files.library_content` → local digest lookup first; digest/remote-file miss follows `peer_blob` / `_resolve` → same pull and owner fetch |
| Precache media and pad assets | `remote_files/precache.py` → `remote_files.fetch.fetch` → same pull and owner fetch |
| Library thumbnail/list/detail | Local metadata/cache/replica only; `library_content` explicitly excludes these callers, so no peer content admission is triggered |
| `net_session_files` | `network/file_index.py` `make_handler`: only `list` and `row`, metadata/source references, **no content phase**; returned source must subsequently pass `net_file_get` |

Local viewer cache/replica hits do not become new peer content reads. No audit
or mesh dependency is added to them or to local listing. Remote listing may
still query owner **metadata**, as before; it is not claimed to be mesh-free.
The owner index handler itself performs no nested mesh request. `stat` returns
metadata/digest and `verify` returns a prefix-match boolean; neither returns
file content. `tools/builtin.py` was searched without a file-size exclusion:
there is no additional `session_file_pull`, `net_file_get` or
`net_session_files` content alias there. General local `read`/web-fetch/tool
attachment operations are not peer session-file serving.

## Validation contract

`tests/unit/network/test_peer_file_read_audit.py` uses the existing real paired
relay fixture and CLI control-socket path. Positive controls inspect committed
owner rows and observed successful fsync before the source fetch read and first
received content, for both pad and attachment sources. Negative controls inject
the **new admission's own** append/file/directory-fsync failures after fresh-link
lifecycle writes, checking healthy entry and exact target sequence, requiring zero
source fetch reads/received FILE bytes/landed payload for pad and digest sources.
Archive fault controls target the generation holding that admission and preserve
the prior copy. Descriptor-aware controls cover first creation, complete archive
publication-before-unlink, healthy pre/post-append rotation and wrong-inode sync.
Long/control/Unicode fallback controls instrument all source `pread` calls only
within fetch, retaining separate stat/verify behavior, and require exact usable
fields before content. EOF plans, post-admission read failure, sanitizer refusal
and buffer-only acceptance remain covered. Existing S1 authorization/path/
sensitivity tests remain neighbors.

The positive and injected-failure relay controls deliberately use existing-base
APIs, so base-red must be unaudited content behavior, not missing new helpers.
Execution evidence is separate from this source inventory; source inspection
alone does not establish that these checks passed.
