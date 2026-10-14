# The native session caller (OSWorld v2.1): what is declared, observed and refused

Companion to [`README.md`](README.md) (the apparatus/methods for the v2.0
release). This file covers ONLY what changed for the native session caller
against the v2.1 release, and it is **not a readiness claim**: the fixed
cohort below has not been run, no paid episode backs any statement here, and
"declared" means declared-and-testable, not measured-at-scale.

## The official pin

| What | Value |
| --- | --- |
| Official release | `osworld-v2.1` |
| Official code commit | `3d778a3c9a34a079316f70df023b166700445792` |
| Official runner entry | `lib_run_single.run_single_example` (`run_multienv_claude.py:638` calls it) |
| Official predictor | `mm_agents/anthropic/main.py` (`AnthropicAgent.predict`) |
| Official guide | `docs/PUBLIC_EVALUATION_GUIDELINE_v2.1.md` |
| Offline population | 82 unique IDs (`evaluation_examples/test_v2*.json` at the commit above); list sha256 `a14975fe0972530993d634d380ba574ecbd19b730d14716a392efe02faa62257` |

The v2.1 tag is a **new pin**: v2.0-era bundles remain readable, but no
number from one release may be compared with the other, and the runner
scripts (`scripts/run_episode.py`, `scripts/launch_episode_21.sh` wrappers)
must name the release they ran.

## Four clocks — never equated

| Clock | What it charges | Honest claim |
| --- | --- | --- |
| Official external epoch | ONE `agent.predict` output group, or one ASK_USER exchange (all returned actions execute under one epoch; `lib_run_single.py:1529–1533`, increment at `:1616`) | The selected official horizon. Internal predict retries do NOT add epochs (`mm_agents/anthropic/main.py:844–845, :894–896`). |
| `ActionBridge._steps` (native output epochs) | ONE per ADMITTED output decision: executed batch, finish proposal (incl. the challenged one), ask exchange, committed read-back loss | The native mapping of the official epoch. Consumed at admission, BEFORE the guest await; pre-admission refusals charge compute/wall only. |
| `ActionBridge.attempted_groups` (attempted output groups) | ONE per COMPLETED root output group, keyed by the generated assistant message: world batches, refused frame/schema attempts, local-tool-only turns, ask/finish, terminal prose | The horizon the arm ENFORCES (`max_steps`), beside the admitted epoch the public `steps` reports. A refused group consumes the attempt but not the admitted epoch and never rebinds the screen token. A turn whose typed signal says the stream faulted never completed and is not charged. |
| Root model-request cap | Unique root `purpose=turn` request-final records (incl. native format/recovery re-asks; excludes children/compaction/naming) | A conservative native-generation safety cap — NOT the official 500, because upstream can retry inside one predict. |
| Whole-system all-actor compute | Root + child/grandchild/helper requests, observed and reported | Reported, not capped: final-row callbacks alone cannot bound concurrent child calls. Any optional bound is declared separately. |

The episode's Nth-epoch stop is enforced by the arm itself: when the tool
result that consumed the Nth admitted epoch is paired — a waiting batch, an
accepted or failed finish, or a challenged proposal — the caller requests
the loop's existing graceful boundary: the current group's calls finish and
pair, and no further root request is admitted. Terminal finishes are
INCLUDED (`end_requested` is deliberately not read): a finish ends guest
admission but NOT the SDK turn, so without the stop the model spends one
more root request on a decision the terminal bridge can only refuse
(measured in review round 1: a finish at epoch N recorded N+1 root requests
with no stop marker). The bridge's own budget refusal remains the
fail-closed fallback, so guest admission closes AT the budget, never one
decision later.

The ATTEMPTED clock (audit 2026-10-13, `native-osworld-v21-epoch-audit.md`)
extends enforcement to every completed root output group. `max_steps` is
enforced on it: the Nth group is reserved at the bridge's entry BEFORE batch
validation, its legal actions/results pair, and group N+1 is refused before
any of its work happens (the admitted-count check remains the fail-closed
fallback). The stop seam also covers the local-tool-only and prose-only Nth
group — a turn boundary with no action-tool result at all — via the loop's
own turn end while the episode is live. A pre-admission refusal after
explicit action output consumes an ATTEMPTED group without touching the
admitted `_steps` or the screen-token binding: a run whose whole budget went
to refused frames would otherwise report a spend the world never saw. The
same ledger carries both: one `output_epoch` row per admitted epoch, one
`attempted_group` row per completed group (`group_kind` carries the per-kind
reason; a same-group follow-up call never renames it). **Official-equivalence
of this root-all mapping (local tools, ask/finish, terminal prose included)
remains UNVERIFIED** — it is a declared conservative choice pending
maintainer confirmation of the multi-agent/Full-tools mapping, not a claim
that every counted group is an official epoch. Internal bounded
format-recovery requests, helper, compaction and naming compute stay
separately accounted and disclosed (see the request-cap and all-actor rows).
The fault filter is the assistant message's own typed `stop_reason == "error"`
-- the typed fact at the bridge layer; the wire-level `retryable_interruption`
marker (core #2337, merged `1684c66b`; this slice cut from `7c88ef5a`) lives
UPSTREAM (the loop consumes it to continue the turn). The direct
clock×recovery integration surface is exercised only once both changes share
one base. Any selector cut before
this slice merges carries a stale `session_arm` hash and is refused by
design at preflight — re-cut the selector from the merged main.

Wall bounds, stated once: the native timer is the caller's `--session-wall-s`
(passed straight to `run_session_episode`; a KILL switch on the arm), while
`--max-wall-s` bounds the owner lease and frontend close-out on either side
of it. Launch records must name BOTH — a record carrying only `--max-wall-s`
cannot say which bound actually fired. `18000 s` is this harness's OWN
runaway guard, NOT a published OSWorld time allowance: the pinned
guideline/runner carries no whole-episode wall ceiling, and `500` is the
Claude sample's SELECTED comparison horizon, not a universal maximum (the
GPT sample uses 200).

## Model policy (preflight requirement, not a proved admission mechanism)

- Hosting/route: `deepseek` / `deepseek-flash` **direct** (the registry
  declares 1M context, vision and tools for this route). The prior OpenRouter
  route is not used.
- Subagent tiers `lo`/`med`/`hi`: the EXISTING default inherit sentinel (the
  root model), `model_choice=operator`; fresh shipped roles only — no custom
  role/team pins are copied into the episode root, and no team is provisioned
  (the per-turn `team` tool remains available; a runtime team scaffold is NOT
  invented).
- Fallback: `retry.modelFallback=False` set in the BENCHMARK config only —
  never in the operator's live configuration.
- **Stated gap**: explicit role/team model pins can still override tiers, so
  the initial configuration is not an all-actor admission guarantee. Actual
  requested/served identities are reported per request (the observer's
  `requested_*`/`served_*` fields); mixed or unknown serving is disclosed,
  never silently averaged. The same-substrate claim requires verifying every
  actor's actual served route — not made here.

## What the caller observes, and how

`run_session_episode` subscribes the request observer immediately after the
session opens and BEFORE the first prompt (records are not replayed). The
handler deduplicates by `request_id` and queues; the ONE flush at close-out
serializes scalars plus the detached usage dump into the record's `requests`
row. Money is kept as separate figures through the canonical
`costs.cost_summary` (no second ledger):

- `recorded_usd`: receipts actually stamped — known charged money;
- `estimated_usd`: the same canonical table projection where only an
  estimate exists (direct DeepSeek rows do not carry `estimated_usd_cost`, so
  `recorded_only` alone would lose them);
- `missing_usage_request_ids`: rows with no usage frame at all — unknown,
  NEVER a priced zero;
- `recorded_is_lower_bound`: TRUE whenever any rows were observed, and FALSE
  only under a future invoice-wide reconciliation. The observer reports
  LOGICAL-REQUEST finals, and a final's coverage cannot certify the complete
  bill: a retried request bills every attempt while the deduplicated final
  records only the last receipt (measured: .03 + .12 billed, .12 recorded,
  coverage still reading `reported_final`). The recorded total is therefore
  conservatively a lower bound, never a certified total;
- `invoice_coverage`: names the coverage actually held
  (`logical_final_only`); `usage_coverage_counts` reports the per-row labels
  as observed.

Cache read/write and reasoning ride inside the usage buckets the shared
pricer already separates; nothing re-adds them to input/output.

## Ask delivery

The adapter-owned ask exchange returns the canonical
`AskUserExchangeResult` to the caller, and the receipt — `ask_id`, request
digest, accepted/declined, public answer, never simulator state — is
rendered INTO the native tool result's CONTENT beside the answer sentence.
That placement is load-bearing: the MCP forwarding layer delivers only the
reply's content blocks to the next root request, so a receipt kept in the
bridge-local `details` never reaches the model (measured in review round 1).
One exchange RPC per ask, one epoch consumed at admission, and the frame
refresh rides the same epoch — no extra call, no double charge.

## Shutdown and the score gate

The close-out calls `shutdown_owned_work(reason, timeout_s=…)` explicitly:
delegation admission closes, the turn aborts, and the canonical disposal
joins every child/grandchild job. The subscription outlives the event sink
and detaches only AFTER the SDK context exits, so cancellation-final usage is
captured. Scoring is REFUSED unless the shutdown receipt is `complete=True`;
an incomplete receipt keeps its pending job ids and errors attached
(`complete=False` is unverified, never a pretend kill), and the outcome says
why.

**The release gate.** With an incomplete (or absent) receipt the worker is
NOT released: no cleanup RPC (`close_session`/`release_instance`/
`revoke_lease`), no adapter close, no supervisor terminate — an unverified
cleanup may still have live owned calls into that worker. The rescue
descriptor stays retained without release or a terminal seal. The managed
OWNER retains its loop/context/bridge/observer/worker FD and paced public joins
for ALL unverified outcomes, including a settled failed canonical. There is no
elapsed-time exit or inference that a missing receipt is clean. Only a fresh
complete receipt re-enables context close, scoring and normal release. The
FRONTEND can return boundedly without closing any of those owning scopes.
Ordinary
abort semantics are unchanged: first-Esc child continuation, TUI
cancellation and the wall kill switch behave exactly as before; the explicit
shutdown is the OWNING CALLER's terminal, not a keybinding.

### Admission, reference syntax and uncertain supervisor readings

The production owner requires `OwnershipGuarantee.MANAGED_RETENTION` at the
shared `verify_owned_admission` seam. A genuine detached lease is still valid
for the generic utility's default `IDENTIFIED_PROCESS`, but process identity
alone cannot admit this caller. Strong admission derives the real guarded
supervisor domain/registration/PID, exact lease birth/source and the matching
owned no-restart plist contract. Unknown proof refuses before config/auth/MCP
or preparation. This is retention, **not** proof of forced descendant death.

All three name-only channels (`--secret`, `--secret-env`, infrastructure
`STORE_REF`) use the existing `SecretRef` syntax before relaunch/persistence
or lookup; default and programmatic Namespace inputs take the same pure path.
Malformed values are never echoed. Syntax cannot detect a secret that happens
to look like a valid uppercase NAME: valid syntax is solely an opaque ref, not
proof the ref exists. Env-only/literal infrastructure transport remains
unsupported by both production frontend and direct owning entry.

Loaded policy is independently required by that SAME supervisor snapshot.
Restoring a KeepAlive=False plist on disk after bootstrapping KeepAlive=True
cannot certify retention: launchd keeps its loaded policy and can restart the
job. The supported proof parses a complete target/PID-specific top-level
`launchctl print` shape, explicit properties and absence of loaded timer/event
triggers. Nested argv/environment text is not policy. Missing, duplicate,
truncated, unfamiliar fields/flags are unknown and strong admission refuses;
this deliberately fails closed on unsupported supervisor output, without OS
version guesses. Disk/source/birth checks remain additional constraints.

Cross-platform fixture-only detached composition binds both frontend and host
through an explicit Python-injected entry/admission callable; it verifies the
generic identity lease, then exercises the real SDK/MCP/verified worker. No
production CLI/env/lease/provider/path datum selects that injection. Production
always requires managed retention and refuses unsupported standalone lifetime.
These fixture semantics are not a production detached-lifetime claim.

Supervisor diagnostics project ONE typed `launchd.observe_job` reading plus
exact owner process checks: live, positively absent, or unknown. Timeout or
incomplete output is unknown, not death **or** positive life; the frontend
reports that uncertainty without tearing down or claiming cleanup. Normal
SDK/UI operation requires neither a managed owner nor this observation path.

## The selector's core-source pin

The arm's correctness lives in the core tree, so the selector now carries
`core_source` (`CoreSourcePin`): the checkout commit plus sha256 per pinned
module (`sdk`, `session`, `session_arm`, `loop`, `configure`, `subagent`,
`action_tool`). `scripts/run_episode.py --engagement session` verifies the pin
against the RUNNING interpreter before launch/allocation and REFUSES a
selector without one — a selector built before this composition fails
clearly instead of launching against a tree nobody identified. The module
set is exact: a partial or over-broad pin is refused, not read as a pass.
The declared `commit` must be a hex SHA (7–40 characters) and is accepted
only as a PREFIX of the running checkout's commit — never the reverse —
and an absent pin is OMITTED from the wire so an old selector's canonical
bytes stay byte-identical for old consumers (a PRESENT pin sent to an old
worker fails closed at the hello handshake, before allocation).
Build a pin for a tree with:

```sh
.venv/bin/python -m local_operator.evaluation.source_pin
```

and copy `commit` + `module_hashes` into the selector JSON. A wheel install
without `.git` records `commit_source="unavailable"` — the hashes are then
the artifact identity, and the run says so rather than substituting a
version string.

## The managed owner (R2 lifetime)

`--engagement session` no longer runs inside the standalone entry's own
`asyncio.run`: that loop's exit cancels the episode's canonical cleanup at
teardown (measured: a delayed dispose hook stayed pending while the process
exited). The entry now has TWO roles:

* **frontend** (default): refuses unless a managed supervisor exists (Darwin
  launchd; unsupported platforms refuse before preparation); starts the SAME script/interpreter under a
  per-episode managed lease; verifies the owner's identity from the lease
  record (pid + birth token, and on launchd the OS's own `job_pid` answer for
  the job label); waits boundedly for a terminal status; and then reports the
  owner's SEALED outcome. On its own timeout it exits `failed_provisional`
  and NEVER stops the owner -- detaching is not a stop; missing admission
  status is uncertainty, not permission to reap. An owner that vanishes
  without a terminal status is reported `owner_lost_unverified`; nothing is
  inferred, deleted or re-run from a dead pid.
* **owner** (`--owned-host`, internal): verifies its lease BEFORE anything is
  prepared (a forged flag with no record/label refuses with zero allocation),
  then runs the ordinary session arm. An incomplete owned-work join no longer
  falls through to a descriptor-only failure: the owner STAYS IN the join --
  the same public `shutdown_owned_work`, bounded attempts, nothing torn down
  in the meantime (no context exit, no adapter close, no bridge stop, no
  rescue, no descriptor discard) -- until a fresh COMPLETE receipt admits the
  normal close-out. Settled unverifiable canonicals stay retained (never
  restarted/relabelled) pending a separately authorized proved disposition.
  Internal rig composition is injected through Python, not public unmanaged
  flags; the fixture explicitly owns that process's teardown.

The owner writes a small non-secret status file under the run root
(`owner/<owner_id>/status.json`): phases `starting -> admitted -> running ->
provisional|closing -> complete|ended`, plus the public shutdown receipt and
the outcome path. A retained owner writes `provisional`; if its canonical
settles unverifiable it REMAINS provisional in the live owner; a descriptor
and TTL are not a substitute for that lifetime or a cleanup proof -- never
inferred as clean.

The retention contract is deliberately bounded: it proves a SURVIVING OWNER
and a bounded frontend report. It is NOT proof that infinitely resistant
descendants can be force-killed (POSIX per-group signalling cannot prove a
whole-tree stop -- `procstate.terminate_process_tree` returns signal
DELIVERY): a run that requires finite whole-tree death must be refused
before it starts unless an independently proven supervisor supplies it. No
privileged tracker, no silent Linux migration (systemd cgroup semantics
remain a separate, unproven route), and the Darwin job runs with
`KeepAlive=false` so a crashed owner is never re-run -- a restart can never
re-allocate an episode.

Owner jobs are session-unique (`com.damian.lop-episode.<owner_id>` under the
real passwd home's `~/Library/LaunchAgents`) and are cleaned up by exact
label once terminal. Secrets never ride argv, the plist, the environment or
the status file: values travel once through a 0600 handoff file under the
run root that the owner reads and unlinks before anything else runs.

## AWS / VM references

The adapter README
([`benchmarks/osworld_v2_adapter/README.md`](../../../benchmarks/osworld_v2_adapter/README.md))
remains the authority for the workspace build, the release manifest and the
provider (`AWS_REGION`, `AWS_SUBNET_ID`, AMI and TTL infra values). The
episode's VM allocation happens at adapter `prepare`/`reset` — strictly
AFTER this document's preflights (secret resolution and the source pin), and
teardown/rescue semantics are unchanged (see README §5).

## Unknowns that must ride every report

- Whether the maintainers accept the native system's internal epoch
  allocation for ranking; JSONL report format itself is permitted
  (guide 357–359), but that is a format allowance, not a protocol sign-off.
- Exact upstream-equivalence of the native epoch mapping is UNVERIFIED until
  the neutral fixture runs and a maintainer reads it.
- Multi-agent rank acceptance is unspecified in the public sources.
- Per-run route/capability provenance is what the observer records; the
  configuration is not its own proof.

## The frozen cohort (candidate, not launch authorization)

[`offline20-v21-candidate.json`](offline20-v21-candidate.json) records the
PROSPECTIVE 20-task cohort: tasks
`002,021,026,029,031,042,043,048,054,056,058,063,065,072,077,080,082,092,104,106`
in the archived seeded run order, `selection_manifest_digest`
`ee5f560d416961df1259caf5eecb25b456724fea97292ba96cc7a7c9eabc369e`
(verify by removing that field and sha256-ing sorted-key compact JSON).
Status: `candidate_not_launch_authorization` — ONE generation, no reroll, no
post-outcome substitution; historical readiness flags (task_026 user
simulator/GitLab dependencies, 029 setup timing, 043 parser) are leads for a
separate preflight, not results. Blocked cases are held and reported, never
silently dropped or replaced.


### Round-4 managed-owner lifetime and transport contract

Managed execution retains the **whole existing SDK host**, not merely a rescue
file, for every unverified public join: pending, None, raised, cancelled/failed
canonical, record-write failure and host-task cancellation. Bounded public
joins are paced; an immediate false answer is never interpreted through elapsed
time. A settled failed canonical is **not recreated**. The same context,
observer, MCP socket and adapter owner FD remain owned until a fresh public
complete receipt permits close-out. The frontend may exit provisionally; it does
not reap a host because a best-effort status file is absent or unreadable.
The descriptor and cloud TTL do not certify cleanup; a permanently failed
canonical remains retained pending a separately authorized, proved disposition.

The production script cannot select the old `--in-process-host` or detached
rig mode. Tests compose those through an internal injected parser entry; the
ordinary CLI requires its existing managed Darwin supervisor. This is retention,
**not** a guarantee of force-killing descendants that leave the process group.
No normal CLI/TUI/SDK session policy or model loop is changed.

There is **no plaintext `secrets.json` handoff**. Managed session invocations
refuse env-only overrides and all inline `--infra NAME=VALUE` before launch.
They cannot safely classify arbitrary inline values as non-secret, and argv is
persisted in the LaunchAgent plist. Use `--secret NAME` and
`--infra-ref [purpose:]NAME=STORE_REF`: only sealed-store names cross the boundary;
the already-existing CredentialStoreResolver resolves them inside the isolated
owner after core-source validation. Infrastructure ref values participate in the
same secret redaction set. No new encryption, broker, pricing ledger, model driver
or credential staging is introduced. Reply-arm inline/env inputs are unchanged.
Missing refs fail before worker preparation. Real credentials are never used in
these scripted proofs.

Start ownership is exclusive: an existing lease/plist/registration is not this
attempt's to overwrite, stop or delete. Failed-start teardown requires this
attempt's exact written plist bytes and a positively pending (unadmitted) lease.
The owner retires its own exact LaunchAgent only after canonical join, context
exit, close-out, seal and output flush. That retirement checks owner nonce,
source/lease, current PID/birth and launchd's own running PID, and does not depend
on the frontend surviving. Missing admission status means **unknown**, not zero
allocation. Detached fixture liveness rejects a zombie, not just a reused PID.

Zero-cost real SDK/native-child/verified-worker hosted evidence is recorded per
fault in the round-4 remediation notes. The old in-process matrix is not a claim
that all hosted cases passed. None of this is paid-pilot approval or a live AWS
cleanup/composite proof; integrated merged source, loader wheel and direct-route
infrastructure still require separate preflight before the frozen twenty tasks.
