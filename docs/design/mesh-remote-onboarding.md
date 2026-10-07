# Design: agent-runnable remote onboarding, one-approval authority, and state carry-over

**Status:** design note, RATIFIED — the workstream builds from it (core slices (a)–(e);
the UI surface rides slice (c)). Written for the Aida chief-of-staff session and the lop-dev
build team. **Committed as:** `docs/design/mesh-remote-onboarding.md` (sits beside the
`mesh-*.md` family it extends).

**Revision 1 (2026-09-30, Aida review):** preflight narrowed to a **credential-free transport
handshake** — reachability + host-key fingerprint + banner only, no `-i`, no auth (§3.1,
§3.3 step 1, §7 step 1; card example §2.1 and slice (b) contract §6 updated to match); everything credentialed runs AFTER the signed approval as the
record's "step zero" (§3.3 step 4), which HALTS on a contradiction with the card and asks
with a fresh card rather than proceeding. All OQ1–OQ14 defaults accepted as written; a
reviewer pass on this note is running (findings to the manager).

**Revision 2 (2026-09-30, Aida review 2 — operator requirement):** the self-install principle
extends to the LOCAL side. The agent bootstraps operator authority on the user's own machine
end-to-end (`lop operator init` + install; new sibling `kind: local_authority`, §2.2/§3.7);
consent is a GESTURE (native admin dialog in the desktop app; Touch ID/OS sheet where offered;
interim ask-sudo via the credential prompt); failure copy is repaved so no surface tells a
user to run a terminal command (§2.9 lists the sites); readiness carries a "Finish setup"
affordance, never a dead end; the drill gains Step 0 (fresh-machine acceptance — THIS Mac is
measured no-anchor: `spawn-capability-only`, §1.1). OQ15–OQ16 added; all other defaults stand.

**Revision 3 (2026-09-30, design review R1 — remediation):** F1 FIXED (idempotency frozen:
digest-bound ids, conflict refusal, terminal immutability, tombstones, single-use defined —
§2.2); F2 FIXED (full transition matrix, writers, expiry owner, deny semantics; diagram
corrected — §2.4); F3 FIXED (write-once decision + per-record cross-process flock +
device-local rule — §2.3; queue record and mesh.json covered in §5.3/§5.4); F4 FIXED
(canonical `lop-approval-v1` payload; verify at decision-write and before every step; anchor
provenance trio derived from the local store — §2.4/§2.5/§3.3 step 7); F5 FIXED
(prune-after-commit, [retire→prune] guard, no-consume on a refused engage, promote-gap named
— §5.3). M1–M4 FIXED (§2.1/§3.3 step 1; §2.3/§2.6; §7 step 3; §2.6). N1–N4 FIXED (§5.4 doc
name; §2.6 sibling-doc task; §2.8 direction+ack; §2.1/§2.2 single-use wording). Operator
addendum (local self-install): COVERED BY REVISION 2 — kind `local_authority` (§2.2),
mechanism (§3.7), copy rule (§2.9), slice map (§6), drill Step 0 (level now ASSERTED, §7);
no rework, per the review's own note.

**Base refs.** All core citations are `origin/main` @ `fff390360` (v0.64.11), read
read-only via `git show`. UI citations are `local-operator-ui` `origin/main` @ `4ea1635904`.
Scout reports folded in @ the workstream scratchpad (`recon/scout-core-gate.md`,
`recon/scout-state-ui.md`); items I could not re-verify myself are marked
**[scout-verified]** / **[unverified]**.

**Shorthand:** `approval.py` = `local_operator/harness/approval.py`; `readiness.py` =
`local_operator/network/readiness.py`; `serving.py` = `local_operator/session/runtime/serving.py`;
`server.py` = `local_operator/session/runtime/server.py`; `relay.py` = `local_operator/network/relay.py`;
`types.py` = `local_operator/network/types.py`; `sync.py` = `local_operator/network/sync.py`;
`mobility.py` = `local_operator/network/mobility.py`. The design docs (`docs/design/mesh-*.md`,
`docs/design/approval-authority.md`) are cited by section.

---

## 0. TL;DR for Aida

Today, the agent can drive everything about the mesh **except bringing a device to the point
where it can complete work**: offloaded writes/execs on `cloud-node-1` park with
"only the operator can allow it … run [the now-retired privileged install step] on it
(one privileged step)",
and nothing short of a human on that box (or a lot of manual SSH) changes that. This note
makes that whole path agent-runnable behind **one remote approval** — a card in the Mesh tab
that shows *what / where / who*, is answered with the operator's Touch ID, and authorizes:
connect → install the current build → install the operator's public anchor (root-owned) →
join as a member → optionally trust the device for unattended work. It also carries the rest
of the operator's asks: routine approvals become answerable from his own device (end-to-end
signature, verified on the node), full-auto survives remote execution, a session's wakes and
monitors travel with it, and a move to a device can be **queued** instead of dead-ending when
another window is attached. Five PR slices (a)–e plus two small defects; a four-step E2E drill
on this Mac + `cloud-node-1` closes it. **Revision 2** adds the local half of the same
principle: the operator's own machine bootstraps its operator authority the same
agent-runnable way — one gesture plus one admin prompt, no terminal (§3.7).

---

## 1. Problem & current state

### 1.1 The hard gate, measured

`cloud-node-1` (ec2-user@99.79.190.164, Amazon Linux, lop v0.63.2, `damian-mesh` member with
role capabilities `list/prompt/slash/steer/stop/view`) parks every write/exec because it has
no operator authority. Measured 2026-09-30 (the card refusal the operator pasted; its remedy
clause is bracketed — §2.9 retires that copy):

> "Your answer was not sent; this approval is still waiting: only the operator can allow it,
> but operator authority is not installed on the machine running the session yet, so nothing
> there can check a signature — [the now-retired privileged install step on it].
> Denying it works from here."

(Revision 2 makes this family of sentences the copy this workstream REPAVES: a refusal's
remedy must name a product action, never a terminal instruction — enumerated sites in §2.9.)

That sentence is `CARD_APPROVAL_REFUSED_UNCONFIGURED_NOTICE` (`approval.py:744` @ ff39; the
configured variant is at `:735`; the command variants live beside them). It is emitted
because the runtime's authority seam can admit an allow only through (a) a proven spawn
capability or (b) a signature that verifies against the pinned anchor — and on this host the
anchor does not exist. `readiness.py:747-785` states the same fact per peer ("operator
authority is not installed on {peer} …: an approval that needs the operator — a write or
command offloaded there — parks until someone installs it") with the remedy at `:806-810`.

**The same gate exists, in its basic form, on the operator's own machine.** Measured
2026-09-30 on this Mac (`lop operator status`, read-only): level `spawn-capability-only`;
`anchor installed: False`; private-half backend `(none)`; reason "no anchor is installed,
so the runtime trusts no key yet". Revision 2's fresh-machine acceptance (§1.3) is
therefore exercised on this host, not hypothetically.

The refusal machinery has three layers worth naming because the new design must sit inside
them, not beside them:

- the **class predicate** `transition_authority` and the sink table `AUTHORITY_OPS`
  (`approval.py`, w/ tests `tests/unit/harness/test_approval_authority.py`;
  `tests/unit/session/runtime/test_approval_authority_seam.py` re-derives the op table from
  `server.py`'s dispatch source);
- the **runtime gate** (`serving.py` `_install_gates.approval_gate`, **scout-verified**
  L939-965: auto short-circuit, else mint `request_id`, publish a card, await a future in
  `_pending_futures` — *in-process memory only*);
- the **park notice at the origin**, which learns of a remote park by diffing peer rows it
  already polls (no wire message): `session/peer_rows.py` `park_edges` (20 s TTL),
  `RemotePark` — "The origin cannot answer a remote ALLOW (`operator_challenge` is
  deliberately absent from the mesh transport's capability set)" **[scout-verified]**, and
  the copy constants `REMOTE_PARK_APPROVAL_*` at `tui/notify.py:299-431` (read at ff39).

Why the origin cannot answer: `types.py:719` records the deliberate absence —
`operator_challenge` is *refused* when it arrives wrapped in `net_forward`, and there is no
`NET_OPS` row for it; the relay forwards inner session ops through `INNER_OP_CAPABILITY`
(`types.py:508+`), where `approval_answer` maps to `prompt` — but an `approval_answer(approved=True)`
is authority-increasing and the *runtime* rejects it without a signature (`approval.py:309-349`,
`admit_increasing`). So a remote viewer can deny (ordinary, safe direction) and cannot allow.

### 1.2 The other measured blockers (2026-09-30)

| # | Blocker | Evidence |
|---|---|---|
| B1 | Build behind (0.63.2 vs current tag; readiness `build` row is an **exact-version** comparison) | `readiness.py` `compare_builds`; beat-2 matrix P5 |
| B2 | No anchor: `/etc/local-operator/operators/<uid>.json` absent | measured; `operator/trust.py:26-31` (anchor paths) |
| B3 | Relay supervision is launchd-only; on Linux the relay "degrades to a foreground process" | `network/cli.py:3290-3330` ("no launchd here: run `lop network serve` in the foreground") |
| B4 | Trying to begin or receive a peer-created unattended session is refused **structurally** — *superseded by slice (a): the boundary is now the `unattended` GRANT, checked receiver-side (§2 OQ4, §6 defect 2)* | `relay.py:6038-6044` (`_op_session_create`) and `:9103-9109` (`_ctl_peer_create`) — the claim at this doc's revision read: "a session created on another device cannot start unattended (yolo) … no capability that unlocks it" |
| B5 | No challenge/signature path crosses the mesh (the ALLOW half) | `types.py:719`, `peer_rows.py:170-189` |
| B6 | A joined device has no way to gain `move/delete/broker` without a manual grant | via `lop network member grant` (`network/cli.py:259-266`); measured node caps lack them |
| B7 | A move dead-ends on attached clients | `server.py:5699-5702`, `:6248-6251`: "kept: This session is open in another terminal or attached client. Disconnect that client, then move again." |
| B8 | No remote-exec machinery exists in core at all — SSH hits are detection only **[scout-verified]**; the install runner is new code | grep evidence in scout-core-gate §5 |

### 1.3 Acceptance scenarios (what "done" means for this workstream)

Operator requirements (verbatim intent preserved):

1. User provides credential(s) via `/credential` or keys on disk.
2. **The agent asks permission to connect+install on the remote; approval granted remotely;
   the agent proceeds** — the operator never performs the installs.
3. Mesh tab approval UX: sidebar badge/tray like browser-approval requests; click shows
   pending approvals (**what / where / who**); approve/deny.
4. Feedback loop approved → connecting → connected; the device appears; the chat header
   dropdown can send/move sessions there.
5. Full state carry-over (wakes / monitors / schedules / automations; LAN or public internet).

Aida additions: **(A)** queued/deferred move — today's attached-client dead-end must become
"complete current step → pause → copy → resume", with UI states queued → finishing → paused →
copying → resumed, and the attached client follows or detaches/re-attaches; **(B)** new-session-to-remote
acceptance — device selection must persist across send (today it reverts to local), full-auto
must apply to remote execution (one-time install/connect approval OK; no routine re-prompts
once trusted), and the probe "what OS are you on" must answer from the node with no dead-end.

**Aida amendment 2 (local self-install):** a fresh machine (no anchor) goes zero→onboarded
with only (a) the approval gesture and (b) the OS admin prompt — no terminal; the drill
exercises that path on THIS Mac (§1.1's measurement).

Board items this retires (manager, folded): (A) build parity on the node — the drill's "bring
it current" covers it, **with the exact-version caveat: update the node to the then-current
tag AT DRILL TIME and restart its relay, or readiness will say `behind` again**; (B) node-side
operator authority — the approval record + Mesh approval UI REPLACES the manual step, with the
`lop network ready` `operator_authority` row kept as the **acceptance surface**; (C) git
identity/auth on the node — carried by the state-carry-over slice; do not duplicate the
broker-vs-documented decision (their Slice 4 shipped with #1751 + broker).

---

## 2. The approval model

### 2.1 One approval, four scopes, one gesture

**Decision.** The ONE user approval is a single signed record answering a single card, with an
explicit scope list on its face. All four scopes ride it, each individually revocable later:

```
approval ap_4f2k… — onboard cloud-node-1            [requested by: session 1a2b3c4d5e6f — "onboard cloud-node-1"]
  where : ec2-user@99.79.190.164 via ssh (credential-free handshake ok; host key fingerprint
          SHA256:…), key ref: secret-store name (never the value)
  probed: transport handshake at 2026-09-30T20:12Z — reachability, host key, banner;
          no credential, no writes (M1; Rev 1 narrowed this to the credential-free shape)
  who   : d_428cb39f…  (mesh device id)  · device fingerprint ABCD-EF12-3456
          operator key to install: kid lop-op-7c1f… (spki 9A3C-… — matches your operator key ✓)
  what  : (1) connect         (2) install lop build v0.64.xx   (3) install operator anchor (root-owned)
          (4) join damian-mesh as `drive`   (5) trust: unattended sessions ✔  (6) grant: approve ✔
  expires: 2026-09-30 21:40Z (60 min) — one use
```

Recommended defaults, each argued:

- **One card, not four.** The user's words are "approval to install on the remote device and
  connect"; splitting install from join gives the operator more prompts for one logical act.
  The card lists every scope so the gesture is informed. Scopes 5–6 are checkboxes with
  defaults ON (requirement B: "no routine re-prompts once trusted") but VISIBLE — an operator
  who does not want unattended can untick before signing.
- **The record is single-use and expiring** (default 60 min; the pairing's own window is the
  invite TTL, default 10 min, extendable via `--expires`; `mesh-transport-identity.md` §5.1).
  "Single-use" is defined ONCE in §2.2: ONE record = ONE onboarding target, steps resumable
  INSIDE the window, never re-runnable after a terminal state; a retry or a new target mints
  a NEW `request_id`. Expiry frees the badge; a new request is one command.
- **Deny is ordinary; approve is the operator gesture.** Mirrors `approval-authority.md` §1:
  a deny settles in the safe direction and keeps working from every surface; an approve
  requires a verified operator signature (Touch ID via `lop operator sign`; the TUI/desktop
  backend may sign in-process — same prompt, `approval-authority.md` §2.4).

A second KIND rides the same machinery: `local_authority` (§2.2, §3.7) bootstraps operator
authority on the reader's OWN machine — same card grammar and lifecycle, its own scope
list, and the OS admin prompt as its privileged gesture.

### 2.2 Record schema (v1, frozen in slice (a))

```jsonc
// <config>/network/approvals/<approval_id>.json   (0600, atomic write, one file per record)
{
  "schema": 1,
  "approval_id": "ap_4f2k…",            // crockford(8)
  "kind": "device_onboard",             // open enum; v1 mints device_onboard + local_authority (§2.2)
  "request_id": "req_…",                // minted by the requester; rules in "Idempotency" below
  "request_digest": "sha256:…",         // jcs(immutable request): binds id to payload (F1)
  "requested_by": {"session_id": "…", "device_id": "d_fe100bae…", "surface": "cli|desktop|tui"},
  "device": {"device_id": "d_428cb39f…", "name": "cloud-node-1", "fingerprint": "ABCD-EF12-3456",
             "host": "99.79.190.164", "user": "ec2-user", "transport": "ssh",
             "host_key_fp": "SHA256:…"},
  "what": {"install": true, "connect": true, "network_id": "n_98af…", "role": "drive",
           "anchor": {"key_id": "lop-op-7c1f…", "spki_fp": "9A3C-…",
                      "statement_digest": "sha256:…"},
           "unattended": true, "grant": ["approve"]},
  "credential_ref": {"kind": "ssh", "ref": "secret-store name"},   // NEVER material
  "state": "requested",                 // requested|approved|connecting|connected|denied|expired|failed
  "created_at": 1790734000.0, "decided_at": 0.0, "expires_at": 1790737600.0,
  "signature": {"by": "device:d_fe100bae…|operator:lop-op-7c1f…", "alg": "ES256",
                "sig": "…", "key_id": "…", "cert": null},   // null = operator key directly
  "receipts": [ {"run_id": "run_…", "step": "connect", "at": …, "ok": true,
                 "detail": "…", "digest": "sha256:…"} ],
  "audit": ["onboard_requested", "onboard_approved", …]
}
```

Rationale for each choice is written into the slice; the load-bearing ones: `credential_ref`
carries a **reference** and never a value (the secret store's discipline, `guide://credentials`;
`redaction_shapes.py` is the shape guard); `what.anchor` binds the exact public key being
planted (the card can show its fingerprint — see §2.5); `receipts` are append-only so the
"connecting/connected" feedback loop (§1.3.4) is a fold, not a second state machine.

**Idempotency (frozen — F1).** The `request_id` (`req_<crockford(8)>`, minted by the
requesting surface; scope = this device's store) is bound to a payload digest:
`request_digest = "sha256:" + sha256(jcs(immutable_request))`, where `immutable_request` =
`{kind, requested_by, device, what, credential_ref, created_at, expires_at}` and `jcs` is the
mesh transcript's canonical JSON (`sort_keys=True, separators=(",", ":"),
ensure_ascii=False` — `mesh-transport-identity.md` §6.2). All under the store lock (§2.3):

- **same id + same digest** ⇒ the existing record is returned verbatim — no merge, no state
  change;
- **same id + different digest** ⇒ typed refusal `approval_request_conflict`; nothing is
  written;
- **re-request after a terminal state** returns that terminal record unchanged — a deny or an
  expiry is never reset by re-request; a NEW target, or any request after the terminal state,
  is a NEW `request_id` (a `failed`-record retry inside the window is NOT a new request —
  §2.4: same record, new `run_id`);
- **who may mint:** any local surface (CLI, desktop route, agent tool) — minting is NOT
  authority-increasing (it creates a request; approve still needs the operator signature and
  deny stays ordinary, §2.4);
- **prune vs id:** the 30-day prune removes the file and leaves a TOMBSTONE in
  `<config>/network/approvals/index.jsonl` (append-only: `{request_id, request_digest,
  terminal_state, pruned_at}`); re-minting a tombstoned id is refused
  (`approval_request_conflict`) for 180 days, after which tombstones prune too (the id is 8
  random bytes, so collision resistance is the real guarantee; the tombstone is the guard);
- **"single-use" means:** ONE record = ONE onboarding target, steps resumable inside the
  60-min window (§2.4), never re-runnable after `connected` or any terminal state. (This is
  the ONE wording §2.1 / §2.4 / §3.6 refer to.)

**Sibling kind, frozen (revision 2): `local_authority`.** The local bootstrap (§3.7) rides the
SAME record type, store and card grammar — `kind` is what differs. Its `device` block is
replaced by `machine` (hostname, platform, uid, backend, level), `credential_ref` is `null`,
and its receipt vocabulary is `proposed → consent → generated → installed → verified` (same
terminal states). Rationale: badge, gesture, receipts fold and retention are identical; only
the steps and the consent channel differ — which is exactly what a kind is for. (Rejected:
folding these steps into `device_onboard`'s receipts — the two flows are independent and one
may run with no device in sight.)

### 2.3 Where it lives; how surfaces read it

- **Store:** `<config>/network/approvals/<approval_id>.json` beside the mesh store's other
  durable objects (`network/audit.jsonl`, `network/outbox/<invite_id>.invite`; same atomic
  write + 0600 file conventions, `network/store.py`). A flat directory means the badge is a
  cold scan (the `wakes/`-index discipline, `wakes/store.py` docstring) and the desktop/CLI
  surfaces never open the network store to answer "is anything pending?".
- **Store discipline (frozen — F3): atomic write is not concurrency control.** The record
  is read-modify-written by ≥3 processes (CLI approve/deny, the desktop daemon's routes,
  the runner's step transitions), and `network/store.py`'s own lock is IN-PROCESS only
  (its docstring names the cross-process gap) — so this store adds: a **per-record
  cross-process lock** (`flock(LOCK_EX)` on `<approval_id>.lock`) around every
  read-modify-write, plus the temp-file + `os.replace` write; a **write-once decision**
  (a second decision write refuses `approval_decision_conflict` — approve-after-deny is
  refused; a receipt append never touches `state`/`signature` and is refused on a decided
  record except the runner's own in-flight path); a **monotone state** (only §2.4's matrix
  transitions; anything else fails closed).
- **Device-local (frozen — M2):** `<config>/network/approvals/` is never synced, never
  copied by a move, never carried by `net_sync` — the record carries host/user/device ids
  and a secret-store reference, and exists on the origin device only. The step-3
  pre-answered pairing confirm is one-shot: bound to the invite id, consumed by the
  pairing, never reusable (the invite's own single-use discipline,
  `mesh-transport-identity.md` §5.1/§5.4) — an answer file, not a standing credential.
- **Same discipline elsewhere:** the §5.4 queue record (per-file flock; relay transitions
  vs `--cancel-queued`, first-terminal-wins) and the `mesh.json` re-stamp (one writer by
  construction — the adopter at promote, atomic replace; the source's copy is deleted at
  commit).
- **CLI:** `lop network approvals [list|show|approve|deny] [--json]` — the handler verbs.
- **Desktop:** thin routes over the same functions, in the `desktop_mesh.py` pattern (bearer
  gated; "thin HTTP skin over one function"; refusals as `{code, message}`):
  `GET /v1/desktop/approvals`, `POST /v1/desktop/approvals/{id}/approve|deny`. New feature key
  `features.approvals = 1` so an old renderer never calls them; old daemon ⇒ the UI hides the
  surface (the additive rule the mesh routes already follow).
- **TUI:** the same list renders as a card/dock (§6 slice e); approve requires the presence
  gesture, so TUI approve = `lop operator sign` forwarded (mechanism exists, slice (e)).

### 2.4 Lifecycle (frozen matrix), verification, audit

```
              ┌───────────────────────── denied ◄── any surface, write-once, before the
              │                                      runner's next step check
requested ────┤
              ├─ approve ─► approved ─► connecting ─► connected        (terminal)
              │                │            │    ▲
              │                │            │    └── retry: SAME record, new run_id
              │                │            ▼        (window open, not denied)
              └─ time ──────► expired ◄─── failed   (retry-eligible until expiry)
                                              ▲
                               any writer observing now ≥ expires_at, or the sweep
```

**The transition matrix (frozen — F2). Every allowed transition, its writer, its rule:**

| from → to | writer | rule |
|---|---|---|
| (none) → `requested` | any local surface | create-if-absent under the store lock; digest-bound (§2.2) |
| `requested` → `approved` | approve op | signature verified BEFORE the write (contract below); first terminal decision wins |
| `requested` → `denied` | deny op | ordinary; write-once |
| `requested` → `expired` | any writer / sweep | `now ≥ expires_at` |
| `approved` → `denied` | deny op | ordinary; write-once; allowed any time before the runner's next step check |
| `connecting` → `denied` | deny op (decision), runner (observation) | the operator aborts an in-flight run: the runner observes the decision at its next step check and stops, recording `denied` + receipts showing where it stopped |
| `failed` → `denied` | deny op | an operator may abandon a failed-but-retryable request; write-once |
| `approved` → `connecting` | runner | first credentialed step begins; opens a `run_id` |
| `approved`/`connecting`/`failed` → `expired` | any writer / sweep | mid-run: the runner refuses the NEXT step on sight of expiry and records `expired` + receipts |
| `connecting` → `connected` | runner | all steps verified; terminal — never re-openable, never re-runnable |
| `connecting` → `failed` | runner | a step failed (incl. halt-on-contradiction); receipts name the step |
| `failed` → `connecting` | runner | RETRY re-enters execution on the SAME record with a new `run_id` (window open, not denied) |
| `failed` → `expired` | any writer / sweep | window passed with no retry |

- **Terminal states are `denied`, `expired`, `connected`.** `failed` is retry-eligible only
  until expiry; a deny is never reset and an expiry is never extended (the signature covers
  `expires_at`).
- **Deny semantics, exactly:** deny lands any time before the runner's next step check; the
  runner observes state + expiry before EVERY step and stops, recording `denied` with receipts
  showing where it stopped. Deny does NOT roll back steps already executed (an anchor once
  planted stays — remediation is `lop network uninstall [--purge]` / `lop network member rm`
  on the node, named in the receipt). After `connected`, deny is refused
  (`approval_already_connected`).
- **Expiry owner:** lazy-on-read + sweep — the first WRITER to observe `now ≥ expires_at`
  materializes `expired` under the lock (reads fold the same way, the asks-store discipline);
  the retention sweep materializes it for untouched records. No timer process. The sweep
  rotates from the create path and, since 2026-10-07, once at relay start
  (`relay.RelayServer._sweep_approvals_once`) — a device that stops onboarding stops touching
  create, and the boot seat is the one non-timer event left that still reaps.
- **Who may deny:** any surface that can reach the record (the safe-direction rule); the
  denial is written with `decided_at` + audit event.
- **Who may approve:** a verified operator signature only (Touch ID on macOS; device cert from
  the paired phone later — §8 OQ9). Verified locally at the surfaces; the node cannot verify
  it before its anchor exists (§2.5 for the honest bound).
- **Verification contract (frozen — F4).** The signature covers a canonical,
  domain-separated, length-prefixed message in the operator module's exact shape
  (`operator/verify.py:42-91`; `_lp` = 4-byte big-endian length + UTF-8):

  ```
  approval_message =
      b"lop-approval-v1\x00"    # a NEW, versioned domain beside lop-operator-v1 and
    || _lp(kind)                 # lop-operator-device-v1 (verify.py:42-47), so a signature
    || _lp(request_id)           # from any other protocol can never be replayed here
    || _lp(request_digest)       # §2.2 — binds every immutable field, incl. expires_at
    || _lp(decision)             # "approve" | "deny"
    || _lp(decided_at)           # UTC epoch seconds, "%.6f"
  ```

  Composed by ONE builder (`approvals.signed_payload(...)` beside the store; the domain
  constant added to `operator/verify.py` so every tag lives in one module) and verified with
  `operator.verify.verify_signature` against the LOCAL operator key at TWO points:
  1. **at decision write** — refusal `approval_signature_invalid` before the file is touched;
  2. **in `approvals run`, before EVERY step** — re-derive `request_digest` from the file's
     current immutable fields (mismatch ⇒ `approval_record_tampered`, receipt + refuse) and
     re-verify the signature; any mismatch is a typed refusal + a `failed` receipt naming the
     check.
  This is what makes §2.5's "enforcement token" claim true against a same-uid file:
  laundering an approval through the product path would require forging the operator
  signature. A same-uid writer can still delete the file or DoS the run — a denial, not an
  escalation (§4).
- **Audit:** every transition is an `AuditEvent` in the mesh audit log (`network/audit.py`
  closed `EVENT_KINDS`, detail whitelist, no key material): `onboard_requested`,
  `onboard_approved`, `onboard_denied`, `onboard_expired`, `onboard_connected`,
  `onboard_failed`, plus the step events the receipts carry.
- **Retention:** terminal records pruned after 30 days; the TOMBSTONE row survives 180 days
  (§2.2's id rules); audit events persist under the log's own retention.

### 2.5 The trust root, and the invariant

**Scout-core Q2, answered: install operator authority on the peer (scout's option (i)), and
add the wire path so the signature crosses.**

The three options and why this one:

- **(i) Node-side anchor install, at onboarding.** The node's anchor pins the *operator's
  public key* — the same statement the operator's own Mac installed (`operator/handlers.py`
  `stage_anchor` :79-94 writes it; `install_anchor` :96-139 lands it root-owned via sudo;
  paths per `operator/trust.py:26-31`). Consequences: the node can VERIFY operator/device
  signatures (Mac Touch ID signatures and the phone's certificate chain — "a device public key
  taken from a certificate that verifies under the anchored operator key",
  `approval-authority.md` §2.3), and nothing on the node can sign. **This is a new kind of
  host for the product to name: a verify-only holder of the operator anchor.** Its readiness
  row must say that truthfully (§8 OQ6).
- **(ii) Origin-signed challenge verified by the peer without an anchor.** Cannot be done
  without trusting the peer relay's word, which the mesh's zero-trust rule
  (`mesh-network.md` §5, R4) rejects — and a hash-at-first-use pin stored in a user-writable
  file is exactly the substitution the anchor's root-ownership exists to prevent
  (`approval-authority.md` §2.1 measurements).
- **(iii) Mesh device certificate as the trust root for allows.** Rejected as an *authority*:
  the mesh identity key is a 0600 file readable by any same-uid process
  (`network/identity.py`), not presence-gated; accepting it as an allow-root would let a
  model's own tool child on the Mac escalate on the node. The mesh link still carries the
  frames and still authenticates the *transport* peer — it is the carrier, not the authority.

- **Anchor provenance (frozen — F4b).** `what.anchor` is never caller-supplied:
  `{key_id, spki_fp, statement_digest}` are derived AT MINT from the local operator store —
  `key_id` = `verify.key_id_for(spki)` (sha256(SPKI)[:32], `verify.py`), `statement_digest` =
  sha256 of the exact bytes `install_anchor` writes (`operator/trust.py:342` `anchor_bytes`;
  staged by `handlers.py:79-94`, landed by `:96-139`). The approve op re-derives from the
  store and refuses `approval_anchor_mismatch` on any difference (the card renders "matches
  your operator key ✓" from the same comparison). The runner re-checks a third time before
  planting and installs EXACTLY the statement whose digest matches the record; its receipt
  carries all three values. A swapped anchor would be a persistent signing root on the node —
  the invariant's own class — so the three checks are the design, not decoration. (For
  `local_authority`, the same trio derives from the just-staged statement and is consumed by
  the install step.)

**Invariant reconciliation** (`approval-authority.md` §0: "a constrained subject must not be
able to mint the authority that removes its own approval requirement"). Preserved, with the
new elements named: the ONBOARDING approval is minted by the operator's presence-gated key and
is single-use/expiring; the ANCHOR it authorizes contains only public data and pins the
operator's key, so no node-side subject gains signing power; routine ALLOWS (once the anchor
is installed) are single-use challenges popped before verification, verified offline against
the anchor (`approval-authority.md` §2.3) — unchanged from the local case, only transported.
What is **not** claimed: the node cannot verify the onboarding record at install time (no
anchor yet), so the record is (a) the human gesture, (b) the enforcement token of the
product's own execution path (§3.3), (c) an audit artifact. A same-uid subject on the Mac
with raw SSH+sudo on the node can bypass the product path — the same class as
`file-only`-is-not-a-boundary and root-defeats-userland; stated here, not defended (§4).

### 2.6 Identity proof in the card; A8.1 mapping; the join ceremony

The card carries both identities: the **device** (mesh device id + its key fingerprint, from a
key the node mints locally) and the **authority being planted** (the anchor key id + SPKI
fingerprint). This is the A8.1 pattern from `mesh-compute-pool.md` §3.2 (a human confirms once,
at the inviter, and the joining side is admitted on a grant) adapted from pools to devices:
A8.1 says "the SAS screen is shown to the human at mint time, describing the member about to be
admitted". So the join itself uses the **existing pairing ceremony** with its human step
confirmed once, at the operator:

- The operator's relay parks the pairing question when it has no terminal (`lop network
  confirm` exists for exactly this; `network/cli.py:530-544`), and the Mesh card renders the
  same two codes the prompt carries (`invite_prompt`/`inviter_prompt_for`,
  `network/invite.py:751-812`: "transcribed X … YOUR screen shows Y … Do they match?").
- The node side runs the **two-phase pair** (`join @<token> --park` prints its own derived
  code and waits; `join --confirm <code>` records the transcription; `_park_join`,
  `network/cli.py:2841+`). For a device with no human at its end we add the transport doc's
  already-specified flag — **`--automated`** (`mesh-transport-identity.md` §12.4: "requires a
  human on the inviter only") — under which the node supplies its own derived code as the
  transcription and the operator's confirm is the only human act. **Doc/code divergence to
  resolve: `--automated` appears in the design docs (§12.4) and in NO code at this head
  [scout-verified]; this workstream implements it.** Naming reconciliation: `--automated`
  means "no human at the joining end"; the admitted row stays `kind: "device"` (the
  compute-pool doc's coupling of `--automated` to `kind: "pool"` is pool-flavored; pools will
  pass `--kind pool` when they arrive — `MemberKind`/`MemberLifecycle` already exist,
  `types.py:988-989`).
- **Compare-then-admit holds under `--automated` (frozen — M4):** the confirm performs the
  compare — the node's supplied code against the inviter's derived value — and a MISMATCH
  REFUSES, spends an attempt (the ceremony's existing repeated-failure policy applies — up to
  three, then the invite is consumed; pre-existing policy Q-XH-6) and audits `sas_mismatch`.
  An unconditional bless would drop the MITM check silently; the implementer must not add
  one. *(Wording reconciled with the shipped policy — QA round 1, Q-1.)*
- **Sibling-doc amendments land in the same change (N2):** `mesh-transport-identity.md`
  §12.4 ("`--automated` … marks `kind: \"pool\"`") and §12.5 ("can never complete a
  pairing on its own") are updated by this workstream's PRs so the divergence does not flip
  direction.
- What is lost vs. a two-human SAS: the second human's transcription check; what replaces it:
  the approval record (bound device fingerprint + the operator's presence gesture), and the
  wire still catches a mismatch (A refuses when the transcribed value ≠ A's derived value —
  the check is by construction MITM-detecting in both flows). Weaker than two humans; stated,
  as A8.1 states its own weaker guarantee.

### 2.7 Durability (scout-core Q1)

**Decision: the approval RECORD is durable; the PARK/CARD stays a live gate (v1).** The
record must survive daemon restarts (the agent may execute minutes after approval; the
desktop may reload) — hence the JSON store. The parked *card* on a node session remains
in-process memory (`serving.py` `_pending_futures`), which dies with its runtime; making parks
durable is precisely the scope of the separate, proposed `docs/design/ask-nonblocking.md`
(its §0 says "Approvals are untouched and keep blocking"), and pulling it in here would
double the workstream. Consequences we accept and state: if the node's runtime dies while
parked, the card is gone and the turn is re-driven by the ordinary resume path; the queued
move NEVER crosses a park (it waits for it — §5.4); phone-answered cards stay live-gate-bound.
The approval record's mechanics reuse the ask queue's *shape* (append-only + fold + derived
index, stdlib-only) but NOT its records — one status vocabulary cannot carry two authority
classes (ask answers are ordinary `prompt` acts; allows are authority-increasing)
[scout-verified derivation, agreement recorded].

### 2.8 The wire for a remote ALLOW (scout-core Q3)

**Decision: extend the existing forward path; do not invent a parallel op family.**

- The challenge and the signed answer are ordinary session-plane ops the runtime already
  speaks (`operator_challenge`; the signed frame carrying `operator_sig`/`operator_key_id`/
  `operator_cert`, `approval-authority.md` §2.3). They cross the mesh inside `net_forward`,
  whose inner-op table lives in `types.py` `INNER_OP_CAPABILITY` (`types.py:508+`) — today's
  deliberate absence is recorded at `types.py:719`.
- New/modified rows (additive, no `MESH_PROTOCOL_VERSION`/`PROTOCOL_VERSION` bump —
  `mesh-transport-identity.md` §12.3):
  - `INNER_OP_CAPABILITY["operator_challenge"] = "approve"`;
  - `INNER_OP_CAPABILITY["approval_answer"]` stays `"prompt"` (deny must keep working from
    everywhere; the runtime remains the layer that refuses a signature-less allow);
  - a new capability name `approve` in the grantable vocabulary (`types.py:415` region),
    grantable per-member via the existing `lop network member grant/revoke`
    (`network/cli.py:259-266`) and included in the onboarding scopes (§2.1). It gates who may
    ATTEMPT; the SIGNATURE gates admission.
- **Direction is pinned (N3):** the capability is checked receiver-side against the SENDER's
  member row — one node-side grant (`member grant … approve`, §3.3 step 9) is sufficient;
  nothing reciprocal is needed. The challenge reply rides the existing `ack` frame shape
  (`approval-authority.md` §2.3's protocol requirement) — never a novel reply op.
- The runtime side is unchanged: challenges are single-use (popped before verification), the
  counts bounded per connection and in aggregate (`server.py` `operator_challenges`;
  `approval-authority.md` §2.3-2.4), verification offline against the anchor.
- The refusal path for an old/anchorless peer is unchanged and truthful: the unconfigured
  notice (§1.1), and `lop network ready` names the remedy; a peer that predates readiness
  answers `peer_too_old` — the card copy already names that answer (F-D fix, commit
  `72330e62c`).

### 2.9 Copy rules (the repave, revision 2)

**Rule (frozen): no failure, refusal, remedy, or readiness sentence may instruct its reader to
run a terminal command.** Setup remedies name PRODUCT actions: on the reader's own machine,
"set up operator authority" (one approval + one admin prompt, §3.7); for a peer, "approve
setup for <device> in the Mesh tab" (the onboarding path, §2.1). Help text and guides may
still document the CLI verbs — that is documentation, not a remedy — and agent-facing guidance
teaches the self-install verbs so the agent's default move is "set it up for me" (§6 slice (e)).

Sites to repave in slice (a) (verified on `fff390360`): the notice constants and their builder
(`harness/approval.py:716`, `:735`, `:744`, `:795-807`); `session/errors.py:383-430` (category
docstrings + sentence selection); `network/readiness.py:766-810` (`operator_row` detail +
remedy); `operator/handlers.py:286`, `:347-350`, `:388-400` (init/status receipts);
`operator/pair_handlers.py:104`, `:224-234`, `:300` (pair-flow prompts); `tui/app.py:25896-25913`
and its `serving.py` mirror (the report's missing-anchor clause); `tui/notify.py:337-341` (the
remote park card's clause); `operator/cli.py` help strings for `init`/`install`. The sweep
cells pinning these strings (`tests/unit/harness/test_approval_authority.py`,
`tests/unit/tui/test_approvals_ux.py`) move their pins in the same commit, and the sweep's
phrase list gains the retired terminal-command shape deliberately — it is not widened to
silence anything.

---

## 3. Install mechanism (no CLI presence on the remote, no human either)

### 3.1 Transports

**SSH is first-class.** Reasons: it is what the operator already has (measured: key
`~/.ssh/lop-mesh-nprod.pem` works, passwordless sudo, from this host); it needs nothing
installed at the far end; and "no remote-exec machinery exists in core" means we build exactly
one path, deliberately. The runner is a new core module — proposed
`local_operator/network/onboard.py` — with an explicit transport interface in TWO phases
(`probe/connect/run/copy/close`): (a) a **credential-free handshake** — reachability, the
host-key fingerprint (`ssh-keyscan`-equivalent shape; no `-i`, no auth, nothing that could
require a secret) and the SSH banner, which is what makes the approval card factual with
zero credentialed probing; and (b) the **credentialed phase**, whose every action runs only
after the signed approval, starting with the receipted pre-read (§3.3 step 4, "step zero").
The credentialed implementation is `ssh(1)` with `BatchMode=yes`,
`StrictHostKeyChecking=accept-new` pinned by the fingerprint the handshake observed, and
per-step timeouts. **Other transports, and when:**
a *Radient tunnel* or the *mesh relay itself* could carry a bootstrap later (a node that is
already a member but not yet installed is a chicken-and-egg only the relay could solve); do
NOT build them now — the runner interface leaves room, and the E2E fallback (§7) proves the
core flow without SSH.

### 3.2 Credential intake & handling

- Intake: `/credential` or `lop secret` (the secret store is the mechanism; `guide://credentials`),
  or a key path on disk. The record stores a REFERENCE (a secret-store name or
  `path:~/.ssh/lop-mesh-nprod.pem`).
- **The credential is not touched before the signed approval:** steps 1–3 of §3.3 are
  credential-free, and the credential is first resolved inside the receipted pre-read (step 4).
- At execution the runner resolves the reference **in place**: `$(lop secret get NAME)` writes
  to a 0600 temp file (or `SSH_AUTH_SOCK` passthrough via `lop secret run`), the ssh child gets
  it via `-i`, and the temp is unlinked in a `finally`. Never printed: not to stdout, not into
  receipts, not into the audit (`network/audit.py` `FORBIDDEN_DETAIL_KEYS` already forbids the
  obvious spellings; the runner's receipt detail is a fixed vocabulary plus digests).
- The existing guard `redaction_shapes.py` (16 dump shapes, `credential_dump_notice` without
  values) stays the backstop on anything the runner echoes.

### 3.3 Bootstrap sequence (fresh or stale Linux box)

Steps 1–2 run BEFORE the approval and are **credential-free by contract** (a transport
handshake only — reachability, host-key fingerprint, banner); every credentialed action runs
after it, beginning with step 4. Each step is a named receipt; the runner re-checks the
approval record (state + expiry) before EVERY credentialed step and refuses to continue on a
denial.

**The LOCAL anchor bootstrap (§3.7) is a separate flow** — it is the prerequisite for the
operator's own signing surfaces (this Mac's own anchor), not a step of a remote onboarding.

1. **transport handshake (credential-free, pre-approval):** reachability, host-key
   fingerprint, SSH banner — no `-i`, no auth, no secret. This is the whole preflight; it is
   what the card's "where" is built from (host, port, user, host-key fp, banner). The
   card discloses this pre-approval contact verbatim (`probed: transport handshake at …;
   no credential, no writes` — M1), so "ask permission to connect+install" stays
   literally true.
2. **request approval** (`lop network approvals request … --json`): mints the record; the
   Mesh badge appears; the agent blocks (bounded, `--wait`) or polls the record. The card's
   "who" binds what is known up front: for a KNOWN host (`cloud-node-1`: `d_428cb39f…`) the
   device id + fingerprint; for a FRESH host the host-key fingerprint + `user@host` (from the
   credential ref / operator input — never from a credentialed probe), with the device
   fingerprint landing in the join receipts. **No writes and no credentialed reads happen
   before the approval** — identity minting moved after the install (step 6); the fallback
   (mint identity pre-approval so a fresh host's card can show it) writes one 0600 file
   pre-consent and is noted for review if card-time fingerprints are preferred there too.
3. **on approve:** mint the invite (`lop network invite --role drive [--device <id>]`;
   `--device` binding exists, transport §5.1 — bind when the id is known) and have the
   record pre-answer the relay's parked pairing confirm for THIS invite (one-shot, consumed
   by the pairing, visible on the badge until then). THIS is what makes one gesture enough:
   the operator's approve is also the admit; both SAS codes are recorded in receipts +
   audit rather than typed by a person (§2.6's stated weakening).
4. **credentialed pre-read — the record's "step zero" (FIRST credentialed action):** with
   the credential resolved in place (§3.2), record OS/arch, whether `lop` exists and its
   version, whether the anchor path exists, `systemctl --user` availability. Discipline from
   `readiness.py`'s "READ-ONLY, AND PROVEN SO" contract (its own suite pins "reads create
   nothing"); writes nothing. **HALT-ON-CONTRADICTION:** if a fact contradicts what the card
   was approved against (OS/arch mismatch, the tool's presence or version outside the
   approved expectation, sudo absent), the runner HALTS **before any state-changing step**,
   marks the record `failed` with the receipt, and files a FRESH card describing the finding
   (a new `requested` record superseding the old id) rather than proceeding — the operator
   re-approves on the corrected facts.
5. **install build:** `uv tool install local-operator==<tag>` (or the node's `lop-update`
   when a build already exists — the generations layout is local-only; `lop install` is NOT
   a bootstrap, `cli.py:1184-1190` noted [scout-verified]); the tag is pinned in the record
   because readiness compares versions exactly (B1).
6. **provision identity + join:** `lop network identity show` on the node (mints the device
   key if absent: 0600 under 0700); its fingerprint is recorded to the receipts; then
   `lop network join @<token-file> --automated` — one-phase; the operator's approve (step 3)
   is what admits the parked pairing. Bounded by the invite's TTL; on expiry the invite
   returns to `minted` (transport §5.4) and a re-run is one command.
7. **install operator anchor:** export the operator's public anchor statement from the Mac
   (`lop operator anchor export --file`), copy it to the node, run `lop operator install
   --from <file>` there — privileged, root-owned landing via `install_commands`
   (`operator/handlers.py`); the verb verifies after (`load_anchor().usable`) and the runner
   records `lop operator trust` output. **Required** under the approval model: it is the
   verification half that makes routine allows answerable and is the acceptance surface's
   subject (§2.5, OQ6); and the runner re-derives `what.anchor` from the local store
   (refusing `approval_anchor_mismatch` on any difference — F4b) and copies exactly the
   bytes whose `statement_digest` matches the record; the receipt carries
   `{key_id, spki_fp, statement_digest}`.
8. **relay supervision:** install + start the relay as a service using the EXISTING
   three-platform abstraction (`local_operator/supervisors.py`; systemd `--user` arm already
   used by `mobile/install.py:10-12`, `:1637+` and `wakes/install.py` whose docstring names
   "a systemd --user service on Linux"). The network group currently drives launchd only
   (`network/cli.py:3290-3330`); slice (b) wires it to the same abstraction. It also rolls a
   relay that is ALREADY RUNNING onto the build the install step landed — and it runs BEFORE
   the grants step (F7b): with a relay answering on the node, the grant write is executed by
   the node's own relay, and a relay still on the previous build refuses it without falling
   back, so the move must precede the write. Note for
   headless: `systemd --user` needs `loginctl enable-linger` — the check lands in the
   credentialed pre-read (step 4) and the step is recorded there (OQ11).
9. **grants:** on the NODE, `lop network member grant damian-mesh <mac-device-id> approve
   unattended` (per scopes 5-6). Receipt records the resulting member row. Runs after the
   relay step, so the write meets the run's relay rather than a pre-run one (F7b).
10. **verify + receipts:** `lop network doctor` (link), `lop network ready --peer cloud-node-1`
    (the acceptance surface — build row `equal`, authority rows updated per §8 OQ6),
    `lop network peers`; the record folds to `connected`; the Mesh UI flips the card.

### 3.4 Upgrade path for a stale node

`lop-update` on the node (rebuilds the uv tool install from a clean clone — same rules as
this machine's; the release-owner section of AGENTS.md governs what ref it may point at) then
`lop network restart` there. The drill must pin the node to the then-current tag at drill
time (B1). The runner's upgrade step is `lop-update` + `restart` + re-verify, and it is the
same receipts shape; the update-rollover work (#1838, `fff390360`) gives the node's own
daemon the concurrent-refresh discipline so this step is safe while the relay is live.

### 3.5 CLI surface (proposed; --json shapes frozen in slice (a)/(b))

| Verb | Purpose | `--json` shape (keys) |
|---|---|---|
| `lop network approvals request --host … --json` | File the record; prints `approval_id` + card payload | `{approval_id, state, device, what, expires_at}` |
| `lop network approvals run <approval_id> --json` | Execute an approved record (the agent's path); streams receipts | `{approval_id, state, steps:[{step, ok, detail, at}], next}` |
| `lop network approvals list / show <id>` | The badge/CLI reads | `{approvals:[{approval_id, state, device, what, requested_by, expires_at}]}` |
| `lop network approvals approve/deny <id>` | The operator's answer (approve requires a signing surface; refuses headless without one) | `{approval_id, state, signature:{key_id}}` |
| `lop operator anchor export [--file P] [--json]` | Emit the PUBLIC anchor statement for transfer | `{path, key_id, spki_fp}` |
| `lop operator install --from <file> [--print-only]` | Node-side privileged landing of an imported statement | exit code + the existing receipt lines |
| `lop network join @f --automated [--json]` | Two-phase-pair automation (human on inviter only) | `{state, sas, fingerprint, invite_id}` / exit 3 = awaiting confirm |
| `lop network member grant <net> <dev> approve unattended [--json]` | The grants step | existing member-row shape |

`lop network ready` keeps its shape; the `operator_authority` row stays the acceptance
surface and its sentences gain the verify-only case (§8 OQ6). Nothing here replaces the
existing verbs; everything is additive.

### 3.6 Failure / rollback semantics

- **Each step is resumable**: receipts + idempotent steps (install is idempotent; join is one
  invite; grants are set-union). Re-running `approvals run` on a `failed` record re-enters
  execution (SAME record, new `run_id`; §2.4's matrix) while the record is inside its
  window, and refuses with the record's own sentence otherwise.
- **Anchoring is the one privileged, root-owned write**; its failure leaves the node exactly
  as before (staged file removed; `install_anchor` verifies after and returns 1 when
  unusable).
- **Join failures** burn or release the invite per the existing state machine (transport
  §5.4); nothing half-joined exists (admission is a single member-row write).
- **Relay supervision failure** leaves the tree installed but the relay unfenced; the record
  is FAILED with receipts, the badge offers "retry step 8".
- **Nothing on the Mac rolls back the node**; the reverse path is `lop network uninstall
  [--purge]` on the node (exists; §6 CLI table of the spine) plus the member `rm` rotation.

### 3.7 Local anchor bootstrap — operator authority on THIS machine (revision 2)

**Self-install principle, local half.** When operator authority is needed on the operator's OWN
machine and absent (measured so, §1.1), the AGENT bootstraps it end-to-end: `lop operator init`
(key generation; idempotent) → **consent** (the record's approve gesture, then the OS admin
sheet at the privileged step) → `lop operator install` (the one privileged landing) → verify (`load_anchor().usable`, `lop operator trust`). Proposed single verb for the
agent path: `lop operator setup [--json]`, receipts `proposed → consent → generated →
installed → verified` (the record vocabulary, §2.2's sibling kind; the OS sheet's outcome is
recorded on the `installed` transition, so a declined or failed sheet names its reason there).

**Consent split (frozen).** The product path is a GESTURE raised by the surface hosting the
install. Desktop: the native macOS admin/authorization sheet for the single privileged command
— candidate mechanism to MEASURE in slice (c)'s design round: `do shell script … with
administrator privileges`, which raises the system sheet; a signed privileged helper is the
heavier alternative, not preferred unless the measure fails. Where the OS offers Touch ID for
admin authorization it is the same gesture class as the key's presence check — claimed only
where the OS actually offers it. CLI/TUI keep the existing sudo prompt — as a PROMPT, not as
instructions (§2.9). Interim agent-runnable fallback: the established ask-sudo pattern — the
agent asks, the user supplies the secret ONCE through the credential prompt (`/credential`),
it is passed to `sudo -S` for that one step and never logged (§4).

**Visibility (frozen).** The bootstrap is a first-class request of kind `local_authority`: a
card in the Mesh tab or a first-run card, showing the machine, the key (label, backend) and
what will be written (`/Library/Application Support/local-operator/operators/<uid>.json`,
root-owned, PUBLIC data); state `proposed → consent → generated → installed → verified`; a
DECLINE leaves the same "not installed" state with a "Finish setup" affordance — never a dead
end. The readiness surface stays truthfully not-ok until verified (no premature ok), and its
remedy names the setup action, not a command.

**Failure shape.** `init` failures (e.g. the macOS keyagent absent) are receipts carrying the
state `lop operator status` already diagnoses; the setup card names the fault, not a command.

**Level handling.** On a host whose presence store is unavailable the local key is `file-only`
— accepted and REPORTED as such ("not a boundary", the existing ladder), with TPM+PIN still
stage F; the bootstrap must not claim a presence guarantee the host cannot deliver
(`operator/__init__.py` levels; `lop operator status` prints the meaning).

---

## 4. Security boundaries (what each scope authorizes; what we do NOT defend)

- **Scope-by-scope**: (1) connect — read-only probes + identity provisioning; (2) install —
  write a tool install and a service unit as the login user; (3) anchor — one root-owned
  0644 file containing PUBLIC data; (4) join — one member row; (5) unattended — the right to
  start sessions on the node without a per-card gate; (6) approve — the right to ATTEMPT an
  allow (the signature still decides).
- **The credential**: the agent may READ the referenced secret at execution time and use it
  for this host's SSH only; the runner may not copy it to the node, echo it, or write it into
  receipts/audit; failure to resolve ⇒ step fails, record keeps `credential_ref` only.
- **Replay bounds**: one approval = one use; challenge single-use by pop-before-verify;
  bounded challenge counts; invite single-use w/ device binding; TTLs at every layer (invite
  10 min default, approval 60 min default, pairing remains what it is).
- **Revocation & deny**: deny is ordinary everywhere; `member rm` rotates; `member revoke
  approve|unattended` narrows live; anchor-held revocation lists reach the node at the next
  anchor refresh (bounded staleness — §8 OQ7); deny is its own terminal state and expiry is
  time-driven only (§2.4's matrix) — a deny stops the runner at its next step check, and
  after `connected` it is refused (`approval_already_connected`).
- **Least privilege for the onboarded device**: role `drive` (no move/delete/broker by
  default — measured today's node caps; grant only `approve`/`unattended` when the operator
  ticks them). Offload needs: `prompt`, `list`, `view`, `steer`, `stop`, `slash` + `approve`
  to answer its own cards remotely; `move`/`delete` only if the operator grants them.
- **Deliberately NOT defended** (honest list): a same-uid subject on the Mac using raw SSH
  with the referenced key and the node's sudo can bypass the product path — this is
  authorization-by-the-user's-own-machine, same class as root/sudo and `file-only`; the
  verifier tree on the node stays user-writable (`approval-authority.md` §4 residual); the
  node's anchor copy is a snapshot (revocation lag, OQ7); availability attacks
  (delete/deny/flood) are denials not escalations; multi-user networks are out of scope
  (spine §5).
- **Record tamper detection (frozen — F4):** the operator signature is verified at decision
  write AND before every runner step; a mutated record is a typed refusal, not a silent run
  (§2.4). Deleting the file or flooding requests remains the availability class the
  not-defended list names above.
- **Local-bootstrap consent (revision 2):** the privileged write is still ONE root-owned file
  of public data; consent is the hosting surface's gesture (§3.7); the ask-sudo secret, when
  used, lives only in the child's stdin for that step and never reaches logs/audit/receipts;
  a declined consent changes nothing — state remains not-installed + "Finish setup".
- **Copy as a boundary:** no failure/refusal/readiness copy may name a terminal command as
  the reader's remedy (§2.9); the existing string sweeps pin this, with their phrase lists
  moved deliberately.

---

## 5. State carry-over model

### 5.1 What travels, what does not, and why

Session-adjacent state is **transcript-first**: wakes and monitors live as custom transcript
entries (`wake_schedules`, `monitor_schedules` — `wakes/store.py` and `monitors/store.py`
docstrings both say "the transcript's entry is the source of truth"), and the copy set already
carries the transcript plus its sidecars (`sync.py:133-192`: `transcript.jsonl`, `title.json`,
`attachment.json`, `origin.json`, `turn-journal.json`, `runtime-stop.json`, `inbox.jsonl`,
`asks.jsonl`, `fork-boundary.json`, `desktop.json`, `created_at.json`, `goal.json`,
`boot-prompt.json`, `eval-kernel.json`, plus the `scratchpad/` tree and attachment blobs).

**Transfers (by this design's delta):** the two *derived indexes* — `<config>/wakes/<sid>.json`
and `<config>/monitors/<sid>.json` (+ `monitors/state/<sid>/…`) — are OUTSIDE the session dir
and are NOT in the copy set today; a moved session therefore arrives with no index, and the
index only rebuilds "on every open" (store docstrings). Slice (d) adds a **cold rebuild at
promote** (stdlib-only, reading the copied transcript) and a **prune at commit** on the
source. Without this, a moved daily wake is invisible to the destination's supervisor scan
until someone opens the session — the silent-loss class both stores' docstrings warn about.

**Does NOT transfer, with reasons:** leases and locks (`.session.pid`, `.execution-lease*`,
`.wake-write.lock`, `.monitor-write.lock` — "per-device, meaningful only where it was taken",
`sync.py:265-271`); `mesh.json` (re-stamped by the adopter — and it gains one field per §5.3);
`subagent-roster.v1.json` (the source process's roster); scan sentinels; replicas' cursors;
archived index; `origin-verdicts.json`. **Agent-level schedules** (`scheduler_service.py`,
`agents/<id>/schedules.jsonl`, APScheduler, "frozen-but-live"): they stay on the device that
owns the agent — they are per-agent, and silently following a session would double-fire across
devices. The product's user-facing "automations" (the Schedules page) mint WAKE rows
[scout-verified, UI schedules-page], which ARE carried. Stated as a scope boundary, not an
oversight.

### 5.2 Move vs recreate; idempotency

The move keeps the session id and retires the source (`move`); `--keep` mints a new id; a
replica recovery promotes as a fork with a NEW id (single-writer invariant,
`sync.py:promote_replica`). Carry-over contract for both shapes: **dedupe on
`(session_id, kind, row id)`** — wake rows carry stable `w1…` ids + per-arm `request_id`; monitor
rows `m1…` + `next_seq` high-water [scout-verified]; the transfer request key
`transfer:{session_id}:{request_id}` already exists on the desktop route. Promote is
idempotent: re-running it re-stamps `mesh.json`, re-derives indexes (a second rebuild is a
no-op), and never duplicates a row.

### 5.3 No-double-fire ordering (exact)

```
prepare (journal + retire source runtime)   ← the runtime is stopped FIRST; its last
                                              append is final; INV-1 guard makes a source-side
                                              wake engage during the window impossible
                                              (launch.py handoff guard, verified)
copy → commit (owner: staging os.replace, source dir deleted, tombstone written)
prune: source wakes/monitors index entries for <sid>          ← new step (d), AFTER a
                                                              SUCCESSFUL commit only (F5)
promote: write mesh.json (home_device=<dest>; + approval carried, below) ; REBUILD dest
         wakes/monitors indexes from the copied transcript    ← new step (d)
ensure: wake supervisor installed/running on the destination  ← idempotent installer exists
[optional] engage-on-arrival (existing flag) — the first open then rewrites the indexes again,
                                              which is a no-op by construction
```

- **Wake rows** carry `next_due_at` (absolute epoch ms), `every_ms`, `until_at`, `limit`,
  `fired_count` — carried as stored; a past-due row fires on the destination's first eligible
  tick after promote. The `[commit → promote-rebuild]` gap is named rather than hidden: for
  the move's own seconds neither supervisor can see the row, and the due wake fires on the
  destination's first tick — intended.
- **No-double-fire, exactly (F5-fixed):** the prune happens AFTER a successful commit
  (prune-first would strand the source on a failed copy) and the source runtime is retired
  BEFORE the copy; the `[retire → prune]` guard is `placement.handoff_guard_refusal` (wired
  at `launch.py:1068-1081`, `session_factory.py:3960-3972`) — an engage against a session
  with a move in flight is refused. **A refused engage must not consume or advance the
  row** (`fired_count`/`next_due_at` untouched), so the destination still fires it exactly
  once; the OQ14 e2e cell asserts this no-consume property, not just the happy path.
- **The `mesh.json` re-stamp has exactly one writer by construction** — the adopting device
  at promote (atomic replace); the source's copy is deleted at commit, so no two devices
  ever write one session's stamp.
- **Monitors**: specs carried (transcript); counters/snapshots are rebuildable caches; the
  destination **re-baselines** — the `.snap` is a device-local observation, so the first check
  after arrival establishes a baseline **without firing** (no alert produced by the move
  itself). Stated as behavior; OQ8 asks the reviewer to keep or change it.
- **The first move after this ships** should verify the supervisor scan sees the rebuilt
  index within one tick (E2E step 4).

### 5.4 The queued / deferred move (Aida's A)

**Design.** A move request gains `--queue` (CLI) / "Move at the next safe point" (both UI
surfaces); when granted:

- **The queue record** is owned by the SOURCE device, durable, keyed by the existing transfer
  key: `<config>/network/queue/move-<session_id>.json` (0600, atomic) — it must survive the
  requesting window closing, so the WRITER is the source relay, and the requesting surface
  reads the same record. The record obeys §2.3's store discipline: per-file flock; the
  relay's transitions and `--cancel-queued` serialize; the first terminal state wins.
- **Queue phases** (wire vocabulary extends the existing phase list
  `prepared→handing_off→committed→done` additively): `queued → finishing → paused →
  copying → resumed`, mapped: `queued` (accepted, no state change), `finishing` (waiting for
  the single in-flight turn to reach a turn boundary — **the design refuses to drain turns**
  (mesh-session-mobility.md §6.4), so "complete the current step" means "until the turn
  ends"; a session mid-turn shows `finishing`), `paused` (quiesced: no new turns
  admitted; clients announced), `copying`
  (staging/commit), `resumed` (destination engaged / origin follows as remote).
- **Who watches the boundary:** the RUNTIME, not the requester — it is the only component
  that knows when the turn ends. The runtime holds a `queued_move` record (a flag, NOT the
  exclusive fence — holding the fence blocks attach admission, `server.py:3897-3900`); at the
  turn boundary it triggers the ordinary retire path (`announce_retiring` /
  `_retire_for("moved")`). The source relay drives the copy after that, as today.
- **The attached-client dead-end is replaced by announce-then-proceed:** at the safe point
  the runtime ANNOUNCES the pending move to every attached client (`move_pending` with the
  target device), holds a bounded window for them to convert to a remote viewer (the
  attached-remote follow path exists — beat-2 (b1) shows a TUI auto-following a moved session,
  and F-A was fixed on main in `72330e62c` by making the follow view's suppression
  card-presence-based, `_remote_park_yields_to_card`) or detach cleanly; after the window the
  move commits and any client that could not follow is disconnected with a sentence naming
  the device where the conversation now lives. This is a deliberate reversal of today's
  "refuse while any observer exists" (`server.py:5699-5702`) — justified because the queue is
  an EXPLICIT, informed request (the UI dialog states that other windows will follow or be
  disconnected), which is what today's blanket refusal protected against.
- **Park interaction (scout-core Q5):** a parked gate is mid-turn, so it is by definition
  before the safe point — the queue WAITS for it (`finishing… (waiting for an approval)`;
  deny resolves it; the park itself is not carried — it cannot coexist with the boundary).
  If the runtime dies while queued, the queue record survives and re-arms on the next engage
  (the record is a relay-owned file, not runtime memory).
- **Refusal taxonomy split (scout-state-ui Q3):** emit the already-reserved
  `viewed_elsewhere` code at the producer (`mobility.py:288` holds the code; nothing produces
  it today) so notices can offer Queue vs Wait for the two different blockers; `busy` (turn)
  keeps "Wait for the turn to finish", `viewed` gains "Move at the next step".
- **Cancel:** `lop sessions move --cancel-queued <id>` / the UI's notice button; the queue
  record is deleted; nothing changed.
- **Failure modes:** destination unreachable at copy time → the existing recovery table
  (`mesh-session-mobility.md` §6.5) and the queue record folds to failed with the source
  untouched; refused at commit → the queue record reports the refusal verbatim (the
  `_source_refused` path already carries refusals verbatim, `mobility.py:1906+`).
- **F-A (manager addition):** the follow view must RECEIVE the (approval) card — for this
  workstream's new approval surface too, the suppression predicate must not count an
  attachment unless the card is actually present (main's fix is the precedent: suppression
  is "yields to card", `require_card=True`); acceptance test in slice (c).

---

## 6. Slice plan: (a)–(e) + the two defects

Every slice is its own PR, conventional commits, with the standing gates (flake8, black,
isort, pyright via `make type-check`, full unit suite; reviewer + QA rounds per team policy;
UI slices add design rounds). No version bumps in PRs.

| # | Slice | Repo(s) | Interface freeze | Depends on | Evidence the PR carries | Parallelism |
|---|---|---|---|---|---|---|
| a | **Approval-request API + gate refactor**: `network/approvals.py` (schema lifecycle), CLI verbs, desktop routes (`desktop_approvals.py`), audit events, `features.approvals`; `INNER_OP_CAPABILITY` rows + `approve` capability + role table; readiness `operator_fact` verify-only case + the "Finish setup" state; **`kind: local_authority` (§2.2) + the copy repave (§2.9)**; **full-auto retention defect** (`unattended` capability + create acceptance replacing the `relay.py:6038/9103` refusals + `mesh.json` carry of auto authority) | core | record schema v1 (+ `local_authority`; F1–F4 frozen shapes: digest idempotency, transition matrix, write-once concurrency, signed payload + anchor provenance); op names; capability names; route paths | — | unit: store lifecycle, seam totality, create-with-grant/refuse-without; e2e cell: a parked node session answered by a signature over a forwarded challenge (two local config roots) | can start immediately |
| b | **Install/connect execution**: `network/onboard.py` runner (SSH transport; credential-free handshake → approved credential phase with halt-on-contradiction), anchor export/import verbs, `join --automated`, relay supervision via `supervisors.py` (systemd arm), receipts; `lop operator setup` (init→consent→install→verify; consent hook: native-sheet runner / ask-sudo fallback); drill runbook; sibling-doc amendments (transport §12.4/§12.5) in the same PR | core | runner step names; receipts JSON; `--automated` semantics; halt-on-contradiction contract; consent-hook shape | (a) schema | unit with a fake transport (no real SSH in CI); a real node drill as PR evidence (commands + outputs); `lop network ready` flips | after (a) freeze; parallel with (c)/(d) |
| c | **Mesh UI approval surface**: badge/tray (browser-approvals pattern: rail badge + tray + card, shared clock), approval card (what/where/who + scopes + phases approved→connecting→connected), routes consumption, **new-chat device-selection persistence defect**, **follow-view card acceptance (F-A)**; **"Finish setup" affordance + native admin sheet (first privileged-step UI here — none exists today)** | UI (+ core routes from (a)) | consumes (a)'s routes | (a) routes frozen | Storybook states + live-app frames (both themes), the revert repro before/after, click-through driving the real routes | parallel with (b)/(d) |
| d | **State carry-over + queued move**: index prune/rebuild at commit/promote, ordering, `mesh.json` additions; queue record + phases + announce-to-clients + `viewed_elsewhere`; UI queue affordances | core + UI | queue record schema; phase names | (a) not required; mobility internals | unit: ordering/no-double-fire/rebuild idempotency; e2e: queued move with a second attached client; frames for the phases | parallel with (b); UI part after (c) visual language lands (or same PR, per review economics) |
| e | **CLI/TUI parity**: `/network` vocab + TUI cards for approvals, `lop network approvals` rendering, GUIDE updates, README notes + a TUI "Finish setup" surface; guides/prompts teach self-install as the default move | core | slash vocab | (a)/(b) | parity cells (slash + CLI + desktop same verbs), TUI frames | last |

**Defect 1 — new-chat device selection reverts to local** (UI): root cause traced by scout
(panePlacement falls through to `{kind:"local"}` once the draft stops existing; the canonical
row carries `locality/owner_device/owner_device_name/reachable` that the chat device family
never reads — UI `4ea16359` citations [scout-verified]). Fix: feed the canonical row into the
slot's placement inputs (or synthesize a remote placement on create-success when `draft.peer`
was set). Acceptance: pick device → send → chip reads `On cloud-node-1` and the next send does
not revert; frames + a live-app drive. Rides (c) or ships independently first (recommended:
independently, it is small and unblocks requirement 4's acceptance).

**Defect 2 — full-auto on remote execution**: rides (a) — it IS the `unattended` grant +
create/move acceptance (§5.3; slice (a)). Acceptance: a full-auto send to the node runs with no card;
a moved auto session stays auto on the node; a NON-granted member still gets the structural
refusal (sentence updated to name the grant).

---

## 7. E2E plan — the four-step drill (this Mac + `cloud-node-1`, fallback below)

Synthetic sessions only; the operator's live sessions are untouched (the drill commander
holds their ids for the drill window — this doc and the evidence never carry them). Isolated
config dirs for tests; TUI boots unset every inherited `CMUX_*`; kills scoped to own pids.
Evidence goes to a scratchpad matrix file (the beat-2 pattern), never the repo.

**Step 0 — Local anchor bootstrap (fresh-machine acceptance; run FIRST).** This Mac is measured
no-anchor (§1.1), so the drill begins with the local half: the agent files/uses the
`local_authority` request ("Finish setup"); operator approves (gesture); the OS admin prompt
runs the privileged landing (gesture); receipts fold to verified and the level is ASSERTED,
not merely recorded (`lop operator status` → installed/trusted; `operator-presence` if the
SE/keyagent path works here, else `operator-file-only`) — **the drill cannot proceed to
Step 1's Touch ID cells without this flip.** Artifacts: card frames, `lop operator setup --json`
receipts, `lop operator status` before/after. **Done:** zero→onboarded with only the two
gestures, no terminal — and Step 1's Touch ID signing now has a key to use.

**Step 1 — Onboard (build + anchor + join + grants).** Agent first runs the credential-free
transport handshake (reachability + host-key fp + banner) and files the request: `lop network
approvals request` → card appears (frame: badge + card, both themes); operator approves (Touch
ID); `approvals run` executes, starting with the credentialed pre-read ("step zero" — its
receipts must not contradict the card, or the run halts and files a fresh card): node updated
to the then-current tag, anchor installed, join completed, grants written, relay supervised
(systemd). Artifacts: card frames before/after; `approvals run --json` receipts (including
step zero); node-side `lop --version`, `ls -l /etc/local-operator/operators/`, `systemctl
--user status` for the relay; `lop network ready --peer cloud-node-1 --json` (build row
`equal`; authority rows per OQ6); `lop network show` member row with `approve`/`unattended`.
**Done:** readiness `ok:true` on every row except any explicitly-scoped residual; device
visible in Mesh tab.

**Step 2 — Remote create + probe (no dead-end).** Desktop: pick `cloud-node-1` in the chat
header, send "what OS are you on"; the created session lives on the node (transcript there),
answers from the node's own shell; full-auto variant: send again with full-auto on — no card.
Artifacts: chip/placement frames (persistence), node-side transcript + reply, `origin.json`,
`mesh.json` `home_device`; the denied case: an ungranted member still refused.

**Step 3 — Park answered remotely (the new ALLOW path).** In an ask-mode node session,
prompt a write; it parks; the origin notice fires (timing budget ≤30 s p95 — beat-2 measured
≤15 s); the origin surface offers "Authorise on this machine"; approve with Touch ID (Step 0's key); the
tool runs on the node; repeat with deny-from-origin (works today, must not regress); record
the audit lines. Artifacts: card frames, the signed frame's audit, the tool's output,
deny receipt. Also: anchorless case reverts to the unconfigured copy (regression cell).
**Negative cells (M3):** (i) an unsigned/forged allow over the forwarded wire → refused at
the node (typed refusal + audit line); (ii) expiry mid-run → the next step refused and the
record `expired` with receipts; (iii) replay of a consumed challenge → refused end-to-end
(pop-before-verify). Artifacts: the refusal frames' audit lines beside the positive cells.

**Step 4 — Carry-over + queued move.** In a session on the Mac: arm a wake (and a monitor);
attach a second client (TUI #2 and/or desktop pane); `/move … --to cloud-node-1 --queue`;
watch phases queued→finishing→paused→copying→resumed (frames per phase); the attached client
follows as remote or gets the clean detach sentence; on the node: wake index present
(cold rebuild), wake fires at its due time, no double fire (source index pruned; source
silent); monitor re-baselines without alerting; then recall home; indexes rebuild again.
Artifacts: phase frames, node `wakes/<sid>.json`, supervisor logs, both-device audit.

**Fallback:** two local config roots + a second relay on a LAN address (the mesh suite's local
topology) — loses SSH-transport coverage, keeps approval/carry-over/queue coverage; the SSH
path then gets covered only by unit cells + a manual run. State it in the QA report rather
than implying coverage.

**Done =** all four steps green with artifacts (commands + actual outputs + frames), no false
success anywhere, matrix verdict posted, findings filed as F-questions with repros. After the
drill: the readiness flip is the go signal for the mesh lane's full offload E2E (manager
sequencing: drill → readiness flip → ping mesh lane).

**Existing findings folded in:** F-A (follow-view card) — fixed on main `72330e62c` for the
TUI; slice (c) extends the invariant to the new card and the Mesh surfaces. F-B (refusal
repeat) — fixed on main `72330e62c` (speaks once per refusal state). F-C — assessed
not-reproducible on main (rc is 1 in the lost-turn arm). F-D — copy fixed on main. F-E
(archive receipt) — deferred; needs the live two-device rig; carried as an open item to the
mesh lane's E2E.

---

## 8. Open questions, each with a recommendation (defaults so build proceeds)

- **OQ1 (scout-core 1) — park durability.** Default: records durable; parks stay live gates;
  queued move waits for parks; revisit only when `ask-nonblocking.md` lands. *(§2.7)*
- **OQ2 (scout-core 2) — trust root for a remote ALLOW.** Default: anchor install on the
  node + forwarded challenge/answer; mesh identity is transport only; no TOFU. *(§2.5, §2.8)*
- **OQ3 (scout-core 3) — wire shape.** Default: extend `net_forward` inner ops + `approve`
  capability; no new `net_*` pair; no version bump. *(§2.8)*
- **OQ4 (scout-core 4) — full-auto semantics.** Default: `unattended` member capability,
  granted at onboarding (visible scope) or via `member grant`; accepted on create (replace the
  structural refusal) and re-applied on move via the carried `mesh.json` authority field;
  revocation takes effect at the next create/engage. *(§2.1; slice (a))*
- **OQ5 (scout-core 5) — queued move + park.** Default: waits; parks do not copy; queue
  survives runtime death; the boundary is a turn boundary. *(§5.4)*
- **OQ6 — the verify-only node's readiness/report truthfulness.** Default: add a
  `verify_only` fact to `operator_fact` and adjust the `operator_authority` row's sentences
  ("operator authority is installed on X: approvals for offloaded work can be signed from
  your devices"), keeping level values unchanged (additive). The row remains the acceptance
  surface (manager). *(§2.5)*
- **OQ7 — revocation lag to the node's anchor copy.** Default: refresh on every onboard run +
  a `lop network ready` remedy line; bounded staleness stated in copy; push-on-revoke is a
  follow-up. Until refreshed, the node may accept a device the Mac has revoked — named risk.
- **OQ8 — monitor re-baseline.** Default: re-baseline silently (no alert from the move);
  keep if review prefers carrying `.snap` per-monitor; neither is silent data loss.
- **OQ9 — who may approve.** Default: desktop + CLI with the presence gesture; the paired
  phone follows (device cert path exists) as a fast follow, not v1-blocking. *(§2.4)*
- **OQ10 — `--automated` naming.** Default: implement as the transport doc names it; keep
  `kind: device`; pools take `--kind pool`. *(§2.6)*
- **OQ11 — Linux relinquish/linger.** Default: supervision via `supervisors.py` systemd arm +
  a documented `loginctl enable-linger` step; if absent, the runner records "relay will stop
  at logout" as a receipted caveat rather than failing the onboard. (The availability check itself rides the credentialed pre-read, §3.3 step 4.)
- **OQ12 — approval record retention.** Default: 30-day prune after terminal state; audit
  events persist under the audit log's retention.
- **OQ13 — attach window length for the queued move.** Default: 30 s (matches existing
  handoff announcements; configurable `network.move.attach_window_s`).
- **OQ14 — engagement of a parked node session by a source-side wake during the copy
  window.** Default: none needed — the INV-1 handoff guard already refuses new runtimes for a
  session with a move in flight (`launch.py`); keep an e2e cell for it and assert the
  no-consume property there (F5): a refused engage leaves `fired_count`/`next_due_at`
  untouched so the destination fires the row exactly once.
- **OQ15 — local-bootstrap consent hook (rev. 2).** Default: the hosting surface raises consent
  — desktop: the native OS sheet (mechanism measured in slice (c)); CLI/TUI: the existing sudo
  prompt; agent-interim: ask-sudo via the credential prompt — the hook shape is frozen in slice
  (b). The copy rule (§2.9) is independent of the hook.
- **OQ16 — privileged steps that need a password (node sudo without NOPASSWD; native sheets).**
  Default: ask-once via the credential prompt (same discipline as §3.2); the drill host's
  passwordless sudo is recorded as the DRILL's assumption, not the product's.
- **Unknowns I could not settle (flagged, not hedged):** the exact gate-construction of a
  relay-engaged runtime on a peer when a config exists there local-and-different (slice (a)
  must pin it with a test before the full-auto acceptance is claimed) **[partially verified:
  the daemon/phone spawn path reads the machine's own config; the relay engage path was not
  exercised]**; `lop network start`'s Linux behavior at this head (`serve` is promised; the
  service branch was unread — slice (b) settles it); UI revert line numbers re-anchor by
  symbol if UI head moves.

---

## Sources read for this note

- Recon set @ the workstream scratchpad (`recon/`): `approval-authority.md`,
  `mesh-network.md`, `mesh-transport-identity.md` (§5.1-5.4, §12), `mesh-compute-pool.md`
  (§3.2 A8.1), `mesh-session-mobility.md` (via scout), `mesh-ui.md`, `AGENTS.md` (origin rev).
- Core @ `fff390360`: `harness/approval.py`, `session/runtime/{serving,server,launch}.py`,
  `session/peer_rows.py` (scout), `network/{cli,readiness,relay,types,sync,mobility,
  invite,handshake,tool,audit,store}.py`, `session/{cleanup,retention,placement}.py`,
  `operator/{__init__,handlers,pair_handlers,trust}.py`, `wakes/{store,install}.py`,
  `monitors/store.py`, `supervisors.py`, `mobile/install.py`, `cli.py`, `slash_commands.py`,
  `tui/notify.py`, `docs/design/*.md` (all read via `git show`, read-only).
- UI (`local-operator-ui`) @ `4ea1635904`: `docs/design/browser-approval-ux.md`,
  `docs/evidence/{mesh-tab,chat-device}/`, `src/renderer/src/features/{mesh,chat/device}/*`
  (scout-verified where noted).
- Evidence: beat-2 matrix @ the workstream scratchpad (`e2e-beat2/e2e-beat2-matrix.md`;
  P1-P6, beats a-f, F-A..F-E); offload runbook (`e2e-beat2-runbook.md`).
