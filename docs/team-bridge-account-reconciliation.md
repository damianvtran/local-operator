# Account-wide Team Bridge reconciliation — decision contract only

**Status: pure contract with scoped validation; not product readiness.** This milestone adds only
`local_operator/team_bridge/reconcile_plan.py`,
`tests/unit/team_bridge/test_reconcile_plan.py`, and this document. No caller is
wired to the planner; existing enrollment, transport, service, and UI behavior
are unchanged. No new feature flags are claimed or introduced. Tests were
written first under the LIGHT-only allocation, then exercised in a separately
granted serial planner-only validation slot; see the scoped evidence below.

The stacked branch is `feat/bridge-account-reconcile-contract`, based on
`feat/team-bridge-default-on` at `424561faa9`. Its stacked PR targets that base,
not `main`. Only the contract branch is published by this slice; #2790's remote
is unchanged.

## Immutable inputs and proposed decisions

`plan_reconciliation(AccountSnapshot) -> tuple[Intent, ...]` is pure. Frozen
`AccountSnapshot`, `TenantSnapshot`, `Offer`, `Binding`, `ReceiverBinding`, and
`OwnerKey` carry captured facts; frozen `Intent` carries `CREATE`,
`ATTACH_CURRENT`, or `HOLD` and a coded `Reason`. CREATE is a proposal, never a
claim of a successful enrollment. Deterministic tenant ordering makes repeated
snapshots equal without mutation. Duplicate events repeat the same proposals,
including CREATE: there is no executor ledger or completed/no-op inference here.
The later executor must revalidate captures and enforce idempotency separately.
Duplicate tenant rows and malformed identities
reject the entire plan with `ValueError`; namespace mismatches return explicit
holds. Nothing silently picks a duplicate or normalizes an untrusted hub URL.

The authoritative ingress adapter must capture one canonical hub/account/device
snapshot, its revision, every tenant's membership revision/currentness and
eligibility, accepted receipt policy, server state, and any current binding.
It must map unreadable configuration to `Preference.UNREADABLE`, a persisted
user opt-out/provenance to `EXPLICIT_FALSE`, a persisted opt-in to `EXPLICIT_TRUE`,
and absence of a user choice to `UNSET`. A default/schema False is **not** a
persisted opt-out: empty configuration remains eligible under a valid new v2
offer. Bare booleans are rejected by the planner's preference API. Explicit
false and unreadable configuration propose no enabling actions.

A new-enrollment `Offer` is separate from each binding's receipt. Offer version,
text identity/reference, byte-validation result, and policy acceptance are facts
validated elsewhere; a version string alone is never evidence of acceptance.
Only a served, validated, accepted `bridge_consent_v2` offer permits automatic
new enrollment. Missing/v1/unaccepted/unvalidated offers hold new enrollments.
An existing valid, currently accepted v1 receipt stays v1 and may attach while a
different tenant enrolls using a separate v2 offer. The planner does not render
consent, fetch bytes, or broaden the accepted-version policy.

Disabled, paused, stale-consent, ineligible, and noncurrent-membership tenants
hold without server writes. Noncurrent/inactive or mismatched binding captures
hold. Ambiguous `active_owner` and ambiguous legacy account-wide done markers
produce visible coded holds; they are not silently ignored. Known successful
per-tenant enrollment bookkeeping does not veto another tenant. Without a
current binding it holds **that tenant only**, never inferring creation from
provenance or using a success marker as reenrollment authority.

The exact owner namespace remains compact JSON
`[canonical_hub, account_id, tenant_id, endpoint_id]`, matching
`local_operator/team_bridge/store.py:_owner_key` (around line 839). Generation is
specific to that namespace, not global. Binding proposals capture the namespace,
generation, receipt version, initial replay fence, membership revision, receiver
binding, and account/device/snapshot identity. Executors must recheck all of
these at consumption time, including account switching and membership removal;
this planner provides no lock, CAS, authorization proof, or owner-row writer.
Other tenants, accounts, hubs, and generations must never be clobbered. The
existing global `Store.active_owner` ambiguity guard remains intact.

### Existing-current attachment and replay are separate contracts

#183's existing default-on response has no explicit created-versus-no-op flag.
An endpoint with `consented_via=default_on` does not prove this request created it.
The future adapter needs an explicit **current-safe attachment** contract, with
validated tenant/account/device/endpoint identity, active/current state, accepted
receipt, namespace generation, and rechecked membership. It must not manufacture
creation evidence or re-enable a disabled, paused, or stale-consent endpoint.

`pre_enable_head` remains the initial replay fence
(`TeamBridgeClient._pre_enable_head` and the store's enrollment path). A captured
fence is retained, including zero; an absent fence holds attachment. Late-attach
backlog handling needs its own contract and negative tests: never silently
reset the fence, import old body history, or treat ATTACH_CURRENT as permission
to deliver pre-fence messages. This slice tests missing-fence hold and exact
preservation, **not** transport/backlog behavior. Received-cache data must remain
references/hashes only, not bodies; neither snapshot nor intent carries tokens,
secrets, or message bodies.

## Scheduling contract, not a new loop

One coalesced event worker will reconcile the **whole account/hub/device** from
each fresh authoritative snapshot, including every eligible organization and
later joins. Do not select the first tenant or create per-org pollers/timers.
When push/refresh events cannot cover later joins with UI windows closed, allow
one account-wide bounded maintenance tick; it exists to discover changed
membership, not to allocate an independent cadence to every organization.

The optional pure helper
`select_rotation(tuple[OwnerKey, ...], budget=..., cursor=...) -> Rotation`
returns a deterministic bounded owner selection and last-served namespace
cursor. It rejects duplicate namespaces and nonpositive/noninteger budgets.
For a stable set of N owners, every owner is visited within `ceil(N / budget)`
cycles. Changing membership requires a new snapshot; no stronger fairness bound
is claimed under unlimited churn. Cursor removal resumes at the next namespace.
No timer, async loop, or transport is implemented here.

The later global shared transport uses `wait=0` and a **total request budget**,
not a full budget per org. Rotation counts selected owners, so the executor must
also count every actual request (including pagination/retries); selection alone
is not a network limiter. With budgeted bounded handling, stable-set latency is
approximately `ceil(N / budget) * cadence + bounded handling`. Consume the
separately owned #1073 loop fix later; do not duplicate its loop implementation.

## Canonical receiver and service safety are unresolved seams

CoS account isolation is **not solved** by this module. A receiver must have
proven canonical-hub/account/device binding; a missing or mismatched receiver
holds proposals. No blind reuse of the config-global Aida session is safe.
`local_operator/aida/bootstrap.py:ensure_session` reads the global session via
`state.session_id_of(root)` around line 254, without account/device arguments.
`local_operator/aida/state.py:session_id_of` (around line 174) reads the
config-global `aida/state.json` identity. Reusing that identity can expose
account A's history to account B. Further, `ensure_session` calls
`proactive.ensure_armed(root, hers)` around lines 324–325: unchanged bootstrap is
**not** an identity-only, no-model/no-new-work factory.

A later receiver factory must separate identity creation from proactive arming,
reuse an existing session only with a proven matching binding, and otherwise
create a fresh account-specific canonical receiver with empty history. No
automatic body/history migration is permitted. The tests here cover mismatched
hub/account/device receiver captures; bootstrap and state are untouched.

Keep `local_operator/services.py` / `local_operator/launchd.py` service rendering
pure. Actual ensure/start/update-health actions belong in a later guarded module,
not a renderer or this planner. LaunchAgent `RunAtLoad`/`KeepAlive` resume **at
login**, not while the user is logged out; do not promise logged-out operation
or add a privileged daemon.

## Current blockers and milestone boundaries

1. **Planner (this slice):** immutable decisions and authored negative tests only.
   #183 supplies a per-org enrollment primitive, not an account reconciler; its
   separately prepared candidate is unvalidated. #2790 remains single-owner /
   account-done: `local_operator/team_bridge/client.py:default_on_in_force`,
   `StatusDiscipline`, and tenant selection still select one org, while
   `local_operator/team_bridge/store.py:active_owner` retains the singleton guard.
2. **Owner set + CoS:** design namespace-safe owner enumeration/generation fencing,
   explicit current-safe attachment, legacy-state resolution, and the isolated
   identity-only receiver factory. Do not bypass `active_owner` during migration.
3. **Guarded service ensure + update health:** retain pure rendering, add guarded
   account/device-safe service lifecycle and health/update handling. Security
   #2827/#2828 own the client/desktop-route ingress seams; consume their outcome
   instead of overlapping their files or relaxing consent/authorization checks.
4. **Reconciler + shared transport:** one coalesced account worker, all-org fresh
   snapshots, later-join discovery, bounded global requests/fair rotation,
   membership/identity revalidation, and separate late-attach/backlog semantics.
   `local_operator/team_bridge/connector/loop.py` belongs to #1073; consume its
   fixed loop only after that stream is ready.

## Scoped validation and remaining evidence

The authored tests cover unset/default-False versus explicit-false/unreadable;
two eligible orgs and a later join; v1 receipt with separate v2 offer; missing,
v1, unaccepted and unvalidated offers; all server-state holds; stale membership,
noncurrent/inactive bindings and receipt rejection; hub/account/tenant/device
mismatches; duplicate/invalid rows; visible legacy done and owner ambiguity;
per-org bookkeeping independence; receiver isolation; exact JSON namespace;
missing replay fence; truthy malformed facts; bounded fair rotation, removed
cursors and invalid budgets; repeated deterministic snapshots and frozen values.
An AST import/call negative control guards the decision-only architectural seam;
live file-I/O tripwires exercise both CREATE and ATTACH_CURRENT planning. These
prove neither account-history isolation nor native/runtime integration security.

The serial validation command is:

```sh
python -m pytest -n 0 -p no:cacheprovider tests/unit/team_bridge/test_reconcile_plan.py -q
```

The primary repository interpreter is reused with explicit worktree PYTHONPATH,
an asserted exact module source path, isolated scratch HOME/config/cache/temp,
no bytecode output, and no dependency installation. Each scoped command has a
90-second / 1536-MiB process-group watchdog and a >=5-GiB disk pre/post check.
The 21 test functions collect 58 parameterized cases. Flake8 and pyright target
only the two new Python files; formatting uses existing cached Black 26.1.0 and
isort 5.13.2. Actual frozen-head results and resource measurements belong in the
PR evidence, not a promise that broader gates passed.

No reconciliation worker, enrollment I/O, account receiver factory, service
ensure/autoconfiguration, native compiler path, or end-to-end integration is
implemented or validated. No full suite, builds, installs, service/DB writes,
browser runs, production flags, or deploys are authorized by this milestone.
Later milestones require their own real-path QA and independent review, with
full automated coverage once at the terminal frozen head (CI where covered).
