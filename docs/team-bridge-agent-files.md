# Dark agent generic-file sends (partial, send-only)

This core slice reuses the existing hub file store and actor-bound stage/bind
routes. It is **not** end-to-end agent file reading and does not activate files.
The server's production agent-file gate remains unwired/default-OFF. No download
router, generic-original export, projection admission or model grant is added.

## Discovery and capability

`bridge` explicitly advertises `files: list[str]` for `dm` and `channel_post`.
Unlike the older hidden image argument, this is a deliberate published-schema
addition: its model-facing description states the runtime gate and count-only
recipient limitation. Review this small static context addition explicitly; no
session-specific schema or network call during schema construction is assumed.
The `attachments` image argument keeps its existing policy.

A file send requires a fresh, account-pinned status answer with a strict integer
`files.version == 1`, `enabled is True`, `download_only is True`, a usable bounded
class list and **`agent_upload is True`**. Missing, false, malformed, future-version
or failed status refuses before staging or sending. Human `files.enabled` and
image `attachments.agent_upload` never substitute for the agent-file grant.
The OFF answer is normal; there is no silent text-only or human-actor fallback.

```json
{"op":"channel_post","channel_id":"bcv_example","body":"","files":["/absolute/report.pdf"]}
```

The same argument works on `dm`, and may accompany existing image `attachments`
and `thread_root_id`. Pair-inbox `send`/`reply`, schedules and read operations do
not accept files. Combined image/file count is bounded by the hub and at most10;
images retain their own unchanged cap. The filesystem paths are sender-side input,
not message content or recipient metadata.

## Authority and bytes

The endpoint/account/tenant/hub origin are captured from runtime ownership, never
model overrides. The resolved conversation is pinned before staging. An owner,
credential generation, tenant, endpoint, origin or conversation switch invalidates
the call; expected account and actor remain on upload, bind and staged cleanup.
The HTTP guard rechecks after credential/refresh waits.

Generic bytes use the image lane's existing credential/config-root path policy,
regular-file FD checks and bounded reads, **not** image decoding/resizing or archive
expansion. Parent directory descriptors are opened without following symlinks so
a parent swapped after the path check cannot redirect a generic upload into a
protected root. Platforms without that anchored-open primitive refuse generic
files; they do not fall back to a weaker read. Empty files remain unsupported.
The hub remains authoritative for sniffed class, class-specific limits and quota.

Only ordered staged IDs enter the existing `files` message wire; mixed images
remain in `attachment_ids`. No paths, original names, byte buffers or digests are
added to sender receipts. Receipts expose the sender's own file IDs only. No
receiving-agent history, screen/context, audit or model-admission path is widened:
agent recipients continue to get file counts, not descriptors/names/originals.

## Failure and uncertainty

A partial stage or definitive refused send attempts deletion of known staged IDs
using the **same actor tuple**. Cleanup is shielded and holds only in-flight task
references; failures are observable as counts. Bound rows remain undeletable.
Existing removed-member tenant-resolution limitations still apply: cleanup may
be refused until expiry/sweep. Stage authorization expiry is not a physical-erasure
SLA, and the existing quota/sweeper semantics are unchanged.

After a physical send starts, a missing/unconfirmed answer may hide a commit.
The call replays at most once with the **same client reference, body, ordered
image IDs and ordered file IDs**, without staging again. If still unknown, it reports delivery uncertain. An interrupted call also retains
staged/possibly-bound rows, and callers must treat that interruption as uncertain;
never retry by restaging with a new reference. No body/byte/token retry journal,
spool, cache or payload telemetry is added. After process loss there is no automatic
cross-process recovery: an interrupted file send must be treated as uncertain,
not declared failed/successful or automatically sent again.

## Activation and evidence boundaries

The existing `other` classification policy does not fence bind/replay revocation
as a new rule in this slice. That unresolved policy remains an explicit activation
blocker. Capability snapshots are not distributed revocation barriers. Activation,
all-replica rollout and recipient read admission need their own accepted controls.

Unit tests drive the real tool and client wire with recorders/MockTransport; they
are not hub/Mongo/GridFS/Redis proof. The accepted C1–C9 matrix must additionally
exercise real tool→client→disposable hub with pinned core/server identities,
capability ON/OFF, actor switches, exact stored bytes, all-or-nothing bind/replay,
revocation, cleanup and count-only receiving-agent controls. Full automated suites
belong to CI, independent security/QA review remains required, and no production
messages or flags are needed to validate this dark slice.
