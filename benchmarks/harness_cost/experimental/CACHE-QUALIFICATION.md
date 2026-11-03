# EXPERIMENTAL cache namespace qualification — limited development evidence

**Draft pending; not an authorization for cold-session or headline collection.**
This is a sanitized methodological port of the private development report, not a
live extracted reproduction. No model request was made while preparing this port.
The private exploratory source artifacts, prompts and profiles are not shipped.
The candidate development qualification's private spend accounting is not
reproduced by the minimal evidence here and is not an allowance for a new run.

## Proposed intervention

`cache_namespace.py` implements `cache-namespace-decimal-v2`. Derive a stable,
public, schedule-owned scope per (task, harness, repetition, workload), shared by
main, summary and side calls. Do not derive it from prompt contents or outcomes.
The helper converts a SHA256-derived value to fixed-format decimal digits and
prepends one administrative tag to the first tool description and first system
text block. This is an additive intervention, not native harness behavior.

Apply **after** native discovery, tool-search expansion and Responses-to-Messages
translation, once the actual tool order is known, **before** the meter observes
the final body. Preserve every native cache marker/TTL, original tool/schema,
message and thinking block. Never normalize one-hour TTLs to five minutes or
recursively remove schema properties named `cache_control`. A changed first tool
must be tagged from its unmodified descriptor; do not accumulate tags. Missing
first descriptions, deferred first tools, unsupported server-native tools,
non-text first system blocks and already-tagged input must fail before billing.
Absent system requires an explicitly disclosed administrative-only first block.

`cache_forwarder.py` is an optional localhost-only development adapter for
`/r/cachedev__<trial>/v1/messages`, configured with `CACHE_METER_BASE` and
`CACHE_FORWARD_LOG`. It buffers SSE and is not a latency benchmark or a concurrent
budget admission service. The upstream meter must enforce all budget gates. Own
and reap only the exact PIDs you start; never restart someone else's services.
The adapter logs metadata/hashes, not private request bodies.

## Minimal supplied evidence and reproducible arithmetic

`evidence/cache-controls.json` contains eight allowlisted historical metadata
records: four decimal-tag first-tool controls and four deliberately failing
system-only controls. It contains token counters and request hashes but no
prompt bodies, credentials, response text or private profile content.
`python3 verify_cache_evidence.py` reproduces these statements from those rows:

- Decimal A/A/B/A reads: **0 / 18081 / 0 / 18081 tokens**.
- Negative system-only A/A/B/A reads: **0 / 17991 / 8287 / 17991 tokens**.

The B read in the negative control demonstrates that a system-only tag can leave
an earlier native tool prefix warm. The supplied rows permit counter and hash
consistency checks, **not** independent re-creation of provider responses or
request hashes without the withheld bodies. They are historical evidence, not
new public-loadout results or proof of equal task behavior.

`evidence/private-layout-hashes.json` retains eight archived layout audits as
hashes, counts and marker-position summaries only. Original private profiles and
request filenames are omitted. Canonical JSON byte lengths are not model token
positions. These hashes are provenance references, not downloadable fixtures.
No measured figure/table beyond the minimal supplied controls is reproduced here.
The earlier hexadecimal policy was rejected for variable token overhead; its
private measurements are not part of this public minimal-data package.

## Zero-spend checks versus unverified gates

`test_cache_namespace.py` exercises synthetic inputs, mutation preservation,
namespace separation and rejected layouts. `test_cache_bridge.py` exercises the
real installed LiteLLM transformation classes when that optional dependency is
available; it does not launch Codex or prove model choice/search behavior. Neither
unit test measures provider caching. Record installed dependency versions when
running these checks; do not install dependencies implicitly.

Before any paid qualification or freeze, supply all of:

1. A frozen schedule and empirical token-equality/overhead tolerance for its tag
   set. Equal ASCII length is not token-equality evidence.
2. Route-specific A/A/B/A controls for all actual public-loadout runtime layouts,
   native/deferred/server tools, streaming, thinking, prefix growth, summary and
   side-call paths. Native changes may legitimately invalidate a warm prefix.
3. Meter evidence that archived final bodies are post-transformation; streaming
   usage/cost reconciliation; failure, retry, crash and missing-usage controls.
4. Independent behavioral negative controls. Existing private mechanical controls
   do not establish quality parity or full agent compaction behavior.
5. Separate LOP classifier attribution, bounded reservation and defensible billed
   USD accounting. Do not claim classifier caches are cold from main-model tests.
6. Atomic per-trial and total maximum-exposure admission before every dispatch,
   explicitly configured provider/rates/output limits and no new implicit tier.
7. Immutable preregistration/runtime/tool-wire attestations and independent review.

If namespaces cannot be qualified, preregister measured TTL isolation with an
actually quiet account/model/cache scope. Wait beyond the longest applicable TTL
since the last response completion plus a documented margin; unrelated traffic
resets the clock. Do not silently substitute prompt normalization or an account
change. See `PREREGISTRATION.md`; unresolved gates mean **not ready**.

Public cache semantics reference (documentation must be rechecked before a run):
https://platform.claude.com/docs/en/build-with-claude/prompt-caching
