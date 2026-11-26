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

### Separate policy for replica refreshes

The source relay admits an authorized `net_sync` replica plan in
`session.replica.admitted`, with the authenticated peer as actor, session as
subject, and only the admission phase (`plan`, `fetch`, or `verify`) in detail.
This records admission, not delivery. An owner-built replica plan on the same
link and session covers its planned and contiguous chunks while the writer is
healthy. Prefix verification, noncontiguous resumes, and continuation after an
observed outage can take a fresh admission. In-flight ownership handoffs and
whole-transcript move plans are outside this policy.

A confirmed audit outage does not block an otherwise-authorized replica
refresh. The replica receipt (`lop sessions sync --json`) and cursor expose
`audit_status` separately from transfer success: `audited` for confirmed
coverage throughout the attempt, `degraded` for known failed durability,
`unconfirmed` for incomplete evidence, or `legacy` when all source replies
omit the field. A later audited span does not erase a prior gap. These statuses
neither grant peer access nor participate in local replica promotion. The S1
peer-file policy below still fails closed, and moves keep their existing policy.

## Admission and durability

Every owner `net_file_get` **fetch**, nonempty **stat**, and valid-prefix
**verify** phase calls the same admission callback immediately before its first
source content read. Stat admits the owner-observed file size at offset zero;
verify admits exactly its validated prefix at offset zero; fetch retains its
existing clipped chunk plan. Empty stat and invalid-prefix verify perform no
content read and need no admission. The owner emits
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
degraded writer refuses that read with `audit_unavailable`, without target reads
or a digest/match response. Refusal/control frames are not FILE content. Direct
calls to a content-reading phase without an admission callback also refuse.
Other existing events retain their existing batching/durability policy.

The row records the authenticated **requester device** as `actor` (device kind),
the owner session as `session_id`, and an allow decision, offset, and
`admitted_bytes` in `detail`. No trusted requesting-session identity exists in
this protocol; it is intentionally absent, not copied from an arbitrary peer
field or confused with the owner session. Network and epoch come from the link.

Historical **and future fetch** rows have exactly the original four detail
fields (`reference`, `admitted_bytes`, `offset`, `decision`), with `phase` **absent**.
For fetch, `admitted_bytes = min(bounded chunk limit, max(0, owner stat size - offset))`
is a **planned maximum for this fetch**, never actual transfer completion.
The actual read is bounded to that admitted span, including at/beyond EOF.
A source read error, truncation, interrupted response or lost link can leave
an allow row with fewer (even zero) delivered bytes. A resumed fetch is a new
admission: authorization and source checks happen again for each request.

Stat/verify hashing extends the producer with a closed five-field detail variant:
the same four fields plus literal `phase: "stat"` or `phase: "verify"`. These are
planned **owner hash reads**, not fetch bytes, delivered bytes or completion.
Fetch consumers select key **absence** (`"phase" not in detail`), never null,
truthiness or an unknown-phase fallback. There is no `phase: "fetch"` variant.
Counts/offsets are strict nonnegative integers, not booleans or floats; extra,
missing, null, unknown or scrubbed fields fail closed. The trusted owner branch
passes `expected_file_phase` outside mutable event detail, so a missing hash
phase cannot be accepted as a historical fetch row. Raw, sanitized, rendered and
durable-receipt detail must match an independent snapshot with the same strict
shape/types/values. Rendered candidates are checked under the existing writer
lock **before append**; rejected candidates consume no sequence or row/counter.
The durable acknowledgement remains bound to the exact published inode/row.

References contain only a canonical pad-relative `scratchpad://` reference,
an attachment digest, or a typed `reference:sha256:<hex>` fallback. The fallback
hashes the UTF-8 **canonical relative reference string**, never file contents;
it cannot trigger a fetch-content read before admission. The owner checks the
existing sanitizer's actual serialized detail budget as well as string/control
scrubbing: legal short Unicode paths can expand under JSON ASCII escaping.
Required acknowledgement refuses missing/changed detail fields, including the
actual rendered row, rather than silently accepting `truncated=true`. Neither
caps nor authority are widened. Reference normalization budgets the **final**
four- or five-field variant, including the hash phase, before target content is
read. Stat/verify admission covers these S1 hash reads only, not unrelated local
reads, net_sync replication, P3 provenance or final socket-release fencing.
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
The original #4228 evidence instrumented fetch only and did not claim audited
stat/verify hashing. This A extension intentionally adds hash-phase admission;
new direct-handler controls observe durable exact-phase rows before target
`pread`, and reject failed hash admission without digest/match disclosure.
Long/control/Unicode fallback controls now observe stat/verify/fetch source
`pread` and require each operation's own exact usable receipt. Fetch totals
remain fetch-only by phase-key absence, not sums of all owner-read work. EOF plans, post-admission read failure, sanitizer refusal
and buffer-only acceptance remain covered. Existing S1 authorization/path/
sensitivity tests remain neighbors.

The positive and injected-failure relay controls deliberately use existing-base
APIs, so base-red must be unaudited content behavior, not missing new helpers.
Execution evidence is separate from this source inventory; source inspection
alone does not establish that these checks passed.
