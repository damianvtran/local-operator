# Design: operator-signed remote ask answers (option D)

Status: proposal (architect, 2026-11-17). Base: `origin/main` 14ea6de98. No code.
Supersedes the guide's "needs a node-bound signed answer record, not built yet"
(`local_operator/guides/guardrails/GUIDE.md:48`).

## 1. Problem as found

Damian answers, from his Mac, an `ask` raised by a session on a mesh node. On
the node that answer is stamped `remote` and never counts as consent, and that
is correct today:

- The node relay forwards a viewer's frame **verbatim** into the owner runtime
  (`network/relay.py:12294-12317`; the dial carries the link's authenticated
  `peer` block, `relay.py:12235-12241`). The runtime stamps the surface from
  connection facts only. Any `locality != "local"` is `REMOTE`
  (`asks/surfaces.py:107-138`, called from `session/runtime/server.py:162-197`
  and `:9380-9385`).
- The guard trusts only `terminal|tui|desktop|mirror`
  (`guardrails/provenance.py:123`, read by `_ask_words` at `:4145-4151` and
  `verified_ask` at `:4092`).
- Measured on node2 (`~/.local-operator/logs/runtime.log`):
  `ask answer stamped: op=ask_respond ask='a-bf67' surface=remote conn.kind=attach conn.surface='desktop' locality=remote proved=False`
  (13:40:05Z), and the same line for `a-c293` at 13:40:06Z. Both answered rows
  in `sessions/d5188ee174d4/asks.jsonl` carry `"by": {"surface": "remote"}`.

So every approval he gives a node session's guarded action is void. The fix is
a record the node can verify **without trusting the head machine's user
account**, because agents on the head run as that same account.

## 2. The inconsistency: why a-bf67 "worked" and a-c293 did not

**a-c293 (17-rig delete) was refused deterministically.** The flagged targets
are concrete paths, so `unanswered_asks` (`provenance.py:4727`) lists a-c293.
Its answer is untrusted, so `answer=None` (`:4807`) and `_ask_block` returns
`(a, True)` (`check.py:1184-1200`). `build_delete_case` then marks the path
`lapsed_only` (`check.py:1560-1561`), and `_dangerous_delete` refuses before any
model call (`check.py:1840`). Log: `risk=high verdict=block latency_s=0.000
targets=~/qa-1366 … basis=lapsed ask` (13:40:40Z).

**The a-bf67 import was allowed, and the remote answer was NOT the carrier.**
I traced it from the child transcript `sessions/feac6b2db7a4/transcript.jsonl`
and reproduced the classification with the base code
(`$LOCAL_OPERATOR_SCRATCHPAD/probe_case.py`):

1. The command passes another session's scratch dir to `python`. `classify_bash`
   ranks it HIGH with `op=touch`, so `build_delete_case` files it as
   **unverifiable** (`check.py:1436-1450`). Probe output:
   `paths [] unverifiable ['…/1164c8450503/scratchpad/i18n-wire.settings-m4135'] lapsed_only {}`.
2. With `paths == []`, no lapsed asks are read at all
   (`check.py:1499`: `… if ctx.session_id and paths else []`). The **context-free**
   judge decides (`check.py:1878`), and `verify()` has no path to demand a
   citation for (`check.py:1063-1109`). A plain ALLOW therefore passes, and the
   log shows `basis=` empty (13:40:38Z).
3. What did that judge read as the operator? In a subagent, `operator_brief`
   correctly refuses to quote the delegating agent's message
   (`provenance.py:5138-5141`), and `verified_ask` refuses the `remote` a-bf67,
   so `operator_words == ()`. But `CallContext.user_request` is built by
   `last_user_request` (`harness/loop.py:7758` → `check.py:2097-2119`), which
   returns the newest non-harness user row **with no lineage check**. When
   `operator_words` is empty, the judge brief renders that text as
   `<operator source="operator, direct">` (`guardrails/judge.py:204-209`).
4. That row was the parent's resume brief (child row 151, `role=user`, no
   `harness_injected` stamp): *"The OPERATOR answered ask a-bf67 … on their own
   surface: 'Yes, import and append them' … Re-issue your ORIGINAL blocked call
   VERBATIM."* Step 82→83 had been refused while the newest user row was the
   earlier, narrower brief. Step 92→93, an identical call, was allowed once the
   newest user row quoted the approval.

**Verdict: a real hole, and independent of signing.** The guard presented a
delegating agent's text to the judge as the operator's own words. The same
fallback feeds user-rule judges (`check.py:2062-2069`, `user_request=ctx.user_request`).
Any agent text (a hub resume, a steer, a peer relayed by a parent) can lift any
judge-decided refusal in a delegated session, including one that cites a remote
answer. This design closes it in Phase 1, before any signing ships. Otherwise
"remote answers can't lift any refusal until signed" is false by a side door.

**A second side door, not closed here (S-3, owned by the Phase 1 lane #4164,
pending Damian).** A *remote prompt* counts as consent today. The mesh `drive`
member's or a relayed phone's prompt reaches the runtime as a plain user turn;
the only remote-specific handling is `owner_typed=False`
(`session/runtime/server.py:9081-9088`, `:9128`). `operator_brief` then quotes it
as `operator, direct` in an operator-opened session (`provenance.py:5130-5144`),
and `verify()` accepts an `operator` citation of it (`check.py:1089`). So a
remote-typed "delete ~/x" lifts a delete that a remote ask answer cannot. The
#4164 lane excludes `locality != "local"` prompts from operator words. Its UX
cost: an operator typing on the Mac into a node session must answer a signed ask
(§6) for guarded actions, instead of typing "yes, delete it".

## 3. Rejected alternatives

- **Sign with the mesh device key.** `network/identity.py:9-29` keeps the Ed25519
  seed in a 0600 user file and says plainly: "A process running as this account
  can read it … it does not protect the device from itself." Any head-agent
  `bash` could forge answers.
- **Trust the roster `admin` role.** Roles live in node-local `networks/<id>.json`,
  "unsigned and writable by anything running as the node's user"
  (`network/operator_binding.py:9-14`). They are not a trust root; the binding
  code never reads them (`:36`).
- **A per-process answer key, certified once per launch.** This removes the
  per-answer prompt, but certifying still needs the anchor key's gesture. It also
  adds a second key type with its own revocation. It loses to D on simplicity,
  and the Touch ID cost is acceptable because signing is offered only on
  path-naming asks (§6).
- **The operator capability as signer** (`harness/approval.py:376-384`). The
  capability is per runtime, and any `lop` launch mints one
  (`session/runtime/launch.py:674`, `:854`), including a launch from agent `bash`.
  It proves "this process started that runtime", not "a person decided".
- **Doing nothing.** Rejected by the operator's request: node work keeps stalling
  on a hand-run delete.

## 4. Option D: the operator approval key, verified against the root-owned anchor

The signer is the **operator ES256 key** that the anchor pins: the Mac Secure
Enclave key with a per-signature Touch ID (`docs/design/approval-authority.md`
§2.1-2.2), or a phone key whose device cert verifies under the operator key
(`operator/verify.py:285`) and that the anchor admits (`operator/trust.py:1501`).
This is the same machinery as the F4 approval decisions
(`network/approvals.py:1077-1110` builds the message, `:1125-1160` signs it), and
the verifier follows `operator_binding._check_signature` (`operator_binding.py:918-965`).

### 4.1 Rollout blocker found on node2 (needs the operator's privileged step)

`/etc/local-operator/operators/1000.json` on node2 is root-owned and usable, but
it reads `backend: file-only, presence: False, devices: []`
(key_id `94a6a6c1…`, created 2026-10-02). node2 does not hold the private half
(`anchor_key_held_locally` → False). A `file-only` key, though, is a 0600 PEM
wherever it was created (`approval-authority.md:183-189`: "not a boundary"), so
head-agent `bash` can read it. Constraint (a) therefore **requires** the verifier
to refuse anchors with `presence: False`, unless the signature arrives through an
admitted phone key. Against today's anchor, D accepts **nothing**. Before D can
help, the operator must re-anchor to a Secure Enclave key (`lop operator init`
on the Mac, then one `sudo` install per node). This is his decision, it has a
cost, and it is listed in §9. Phone signing is deferred (§7a, S-7).

**The re-anchor ceremony must be authenticated by a human comparison (S-4).**
Neither transfer path proves the new key is the Mac's Secure Enclave key:
`lop operator anchor export` prefers the user-writable **staged** statement over
the installed one (`operator/handlers.py:1040-1057`), and the F4b onboarding pin
(`operator/cli.py:66`, `network/onboard.py:1932`) is minted by the current
file-only key, which a head agent can read. A head agent could therefore stage
its own key and get it pinned. Required:

1. The node-side `install` prints `key_id` and `spki_fp` (`trust.py:859-880` `anchor_trio`, the
   one fingerprint builder) and waits for confirmation.
2. The operator compares them against a value the agent cannot plant: the Mac's
   **installed** root-owned anchor (`lop operator trust`/`status`, which read
   `trust.load_anchor`), or the key agent's `public` verb
   (`packaging/macos/lop-keyagent/se-keyagent.c:507`, `:958`), which reads the
   Enclave key itself. Never `anchor export` while a staged statement exists.
3. The runbook states that **re-anchoring voids**: every operator-binding proof
   (`old_signer`, `operator_binding.py:938-939`), every device cert issued under
   the old key (`verify.py:285` re-verifies under the current operator key), and
   every signed answer (§7d). Phones must re-pair and bindings re-present.

### 4.2 Where the signature attaches

1. **Head surface** (desktop facade or TUI dock; the same opt-in seam as
   `operator_ask=True`, `mobile/attach_client.py:1876-1920`). The operator answers
   a card for a remote session whose card carries `signable` (§6). The surface
   builds the payload (§5) from the question **it rendered**, then calls the key
   agent's new `sign-answer` verb (§4.3). The Touch ID sheet text is built
   **inside the helper** from the bytes it signs, so a caller cannot make the
   text differ from what is signed.
2. The surface adds the `operator_answer_sig` field to the `ask_respond` or
   `ask_revise` frame. The frame then travels the existing stream path unchanged
   (`relay.py:12640` → peer → `:12317`). `INNER_OP_CAPABILITY` is unchanged
   (`network/types.py:828`).
3. **Node runtime ingest** (`server.py:9385`): `signed_answers.verify_ingest()`
   runs (§7). If it passes, the raw record is stored in the answered or revised
   row as `by.operator_sig` and the nonce is journalled. If it fails, the answer
   is **still recorded** as plain `remote`, so the agent gets its answer, and the
   failure fact is logged. The runtime keeps stamping `surface=remote`; it never
   writes a trusted label.
4. **Guard** (`provenance._ask_words`): it re-verifies the stored record at
   consent time and derives `operator-signed` itself (§8).

### 4.3 A domain-aware signing verb (S-1)

Today the sheet is generic. `cmd_sign` signs whatever is on stdin
(`se-keyagent.c:535-595`), and `_KeyagentSigner` records that the sheet cannot
carry our text (`operator/keychain.py:846-856`). A head agent could feed it a
forged answer and get it signed with one blind touch. The fix lives in the only
process that can use the key: the entitled helper (`keyagent.entitlements`; a
modified binary loses its application identifier, so it cannot reach the key).

- **New verb `sign-answer`.** It reads the framed bytes of §5 on stdin and
  parses them itself (fixed field order, length-prefixed, no JSON in C). It
  refuses anything that is not exactly `lop-ask-answer-v1` framing: wrong field
  count, `exp_min > 240`, a glob character in `scope`, or over 8 KiB. It then
  builds `localizedReason` from the parsed fields: *"approve '<label>' on
  <target_device_id> for session <session_id>, ask <ask_id>: <first 3 scope
  paths> (+N more), valid <exp_min> min"*. It evaluates an `LAContext` with
  `evaluateAccessControl:operation:kLASecAccessControlOperationUseKeySign
  localizedReason:` (with `touchIDAuthenticationAllowableReuseDuration = 0`),
  finds the key with `kSecUseAuthenticationContext` set to that context, and
  signs **exactly** those bytes. LocalAuthentication is Objective-C, so the
  helper gains one `.m` unit, or `objc_msgSend` calls in the C file.
- **Raw `sign` refuses reserved prefixes:** `lop-ask-answer-v1\0`, and
  (recommended, same line) `lop-operator-device-v1\0`, so blind device-cert
  minting dies too.
- **Settling evidence:** one operator-run `sign-answer` on the Mac whose sheet
  shows the payload's node, session and scope. CI cannot raise a sheet.
- *Tests* (source pins in the style of `tests/unit/operator/test_operator_keyagent.py:44-62`,
  plus the `keyagent-macos` publish job that compiles the helper):
  `test_raw_sign_refuses_answer_domain_prefix`,
  `test_raw_sign_refuses_device_cert_prefix`,
  `test_sign_answer_reason_built_from_parsed_fields` (no reason text arrives from
  argv or env), `test_sign_answer_rejects_glob_and_long_exp`,
  `test_sign_answer_uses_authentication_context` (the context is passed to the key
  lookup, so there is one gesture with our text, not a second generic sheet).
- *Residual, for the operator to accept:* a head agent can still raise a sheet,
  but it now reads as the forged action it is. Approving it requires a human to
  approve text that names the wrong thing.

**Relay-surfaced asks.** A relay is a pointer: the origin holds the question
and the answer, and "one answer settles both" (`asks/queue.py:648-657`). The
signature always binds the **origin** `(session_id, ask_id)`. For a-bf67 that is
`d5188ee174d4/a-bf67`, not ce618's `a-c381`. The surface resolves the pointer
before building the payload, and a payload that names a `relayed` row is refused
(`relay_pointer`).

## 5. Record schema and canonical serialisation

```
operator_answer_sig = {
  "v": 1,
  "payload": {
    "domain": "lop-ask-answer-v1",
    "network_id":  "n_…",           # the network the stream link is in
    "target_device_id": "d_…",      # the NODE that owns the session
    "session_id": "d5188ee174d4",   # ORIGIN session (never a relay pointer)
    "ask_id": "a-bf67",
    "qid": "i18n_receipts_4135",
    "question_digest": "sha256:…",  # §5.1
    "answer": ["Yes, import and append them"],
    "scope": ["~/x/a", "/abs/b"],   # §5.2, sorted, literal as rendered
    "ts": 1794922805,               # signer clock, seconds
    "exp_min": 60,                  # 1..240 (S-6)
    "nonce": "<32 hex>",            # 128-bit, signer-minted
    "key_id": "<anchor or device key id>",
    "surface": "desktop"            # (f) informational only
  },
  "sig": "<DER hex>",
  "cert": "<device cert>"           # phone signer only
}
```

Signed bytes (revised for §4.3, so the helper can parse what it signs):
`b"lop-ask-answer-v1\x00"` followed by `_lp` fields in the fixed order
network_id, target_device_id, session_id, ask_id, qid, question_digest, answer
(count, then each), scope (count, then each), ts, exp_min, nonce, key_id,
surface. This is the length-prefix framing of `operator/verify.py:84-88`, used by
F4 (`approvals.py:1077-1110`). Integers are decimal text. `question_digest` itself
is `sha256(network.wire.canonical_json(...))` (`network/wire.py:378`, as in
`approvals.py:430-441`). Declare the domain tag beside the other four in
`operator/verify.py:42-67`. `sign_challenge` refuses any purpose outside
`ACTIONS` (`operator/sign.py:163-164`), so `lop operator sign` cannot produce it.

**5.1 `question_digest`** = `sha256(canonical_json({question, options:[{label,
description}], command, multi}))` over the ONE question `qid` exactly as it
appears in the folded origin record. Head and node compute it with the same
function, so a relay that rewrote the text, an option or the suggested command
breaks verification (`question_mismatch`).

**5.2 `scope`** contains the path words the guard's own consent reader would
extract from the question text plus the **picked** labels. That is the
`_approved_text` / `ask_reach` vocabulary (`provenance.py:4160-4185`,
`:5281`), brace-expanded, never descriptions (`:4211-4240`). Words stay literal
(`~/…`); the node expands `~` with its own HOME. The node recomputes the set
from the stored question plus `answer`, and it must equal `scope`
(`scope_mismatch`). After brace expansion, a scope word containing `*`, `?`, `[`
or `]` is refused (`scope_glob`, S-9). A glob approves an open set the operator
never saw, so a pattern ask must list its paths. That makes the "exact paths/args" binding concrete: the
operator signs the path words he saw, and consent can never cover more than them.

## 6. UX

- One Touch ID (or one phone approval) per signed answer. It is offered **only**
  when the forwarded card carries `signable`, which the node sets at enqueue when
  (i) the question or labels name a path (non-empty `scope` for any option), or
  (ii) the asking task tree has a guard refusal journalled in the last 24 h
  (Phase 3). Everything else stays tap-free.
- A node agent could mark every ask `signable`. That costs prompts, not security.
- The card says what the gesture does: "Sign this answer so node2 can act on it
  (Touch ID)". It also says when no signer exists ("this Mac has no presence key;
  the answer is recorded but cannot authorise a guarded action on node2").
- **No retro-signing.** A signature is accepted only inside the answering
  `ask_respond` or `ask_revise` frame. There is no op that attaches a signature to
  an existing row. a-c293 and a-bf67 stay unverified forever, and the agent must
  raise one fresh ask.

## 7. Security constraints (a)–(h), remapped onto D

Each check fails with a named fact (§7g). `sa` below is the new module
`local_operator/asks/signed_answers.py`, which is stdlib plus lazy `cryptography`.

**(a) The private key never leaves the Secure Enclave or the phone.**
*Mechanism:* in Phases 2-3 the verifier accepts **only** the anchor key, when
`anchor.presence` is True and the backend is a presence backend. A file-only
anchor fails as `signer_not_presence_bound`, because that key is a readable
file. `lop operator sign` cannot emit the domain (§5), and the helper's raw
`sign` refuses it (§4.3). A record carrying a `cert` is refused as
`device_signing_deferred`.
**Phone keys are deferred (S-7) until the phone signer renders the exact parsed
payload, as §4.3 does.** When they are enabled (Phase 4) they must not use
`device_is_admitted` (`trust.py:1501-1557`). That predicate returns True for any
unrevoked, id-consistent device when `anchor.presence` is True (`:1551-1552`), or
when the anchor has no `admission` marker (`:1553-1554`, the migration guard).
node2's 2026-10-02 anchor predates the marker. On such a host, a head agent
holding the file-only operator PEM could mint a cert for its own key and sign
unlimited answers (S-2). Phase 4 adds `trust.device_listed_strict(anchor,
device_id, spki)`: not revoked (`device_is_revoked`), `device_id ==
key_id_for(spki)` (`verify.py:114`), `device_id in open_device_ids(anchor)`
(`trust.py:1405`), and **no** presence or migration exemption. Listing is a
root-owned step, so the agent cannot reach it.
*Residual:* a forged-payload sheet (§4.3).
*Tests:* `test_file_only_anchor_signature_refused` (a valid ES256 signature from
a file-only key: refused); `test_device_identity_key_signature_refused` (record
signed with `identity.py`'s Ed25519 seed); `test_lop_operator_sign_refuses_answer_domain`;
`test_cert_record_refused_until_phone_phase`;
`test_strict_listing_refuses_unmarked_anchor_unlisted_cert` and
`test_strict_listing_refuses_presence_anchor_unlisted_cert` (S-2, Phase 4; both
pass under `device_is_admitted` today, which is the point);
`test_head_bash_cannot_read_presence_key` (the file-backend contract only:
`FileKeyBackend` is the sole on-disk key type and is refused; the Secure Enclave
non-exportability is operator-run evidence, as `approval-authority.md:160-166`
records, and is never claimed from CI).

**(b) Bind to the exact rendered question.** `question_digest`, `qid`, `answer`,
`scope`, `session_id`, `ask_id`, `target_device_id`, `network_id`. The node
re-derives each value from its own store, its own id (`_node_self_id`,
`operator_binding.py:1044`) and the stream link's network.
*Tests:* `test_digest_mismatch_after_relay_rewrite`, `test_scope_superset_refused`,
`test_other_session_or_ask_refused`, `test_other_node_refused`,
`test_relay_pointer_refused_origin_binds`.

**(c) Single-use nonce, recorded at the verifier.** At ingest, `sa` appends
`{nonce, ask_id, qid, sig_sha256, at}` to `sessions/<sid>/signed-answers.jsonl`
(O_APPEND, 0600) and refuses a seen nonce (`replay`). Rows older than 24 h plus
skew are pruned, since freshness refuses those anyway. At consent time the guard
also refuses a nonce that appears on more than one answer row. A replayed old
"Yes" after a "No" revision is therefore refused at both ends.
*Tests:* `test_replay_same_frame_refused`, `test_replay_after_revision_refused`,
`test_nonce_on_two_rows_not_consent`.

**(d) The anchor-listed key, not roster admin.** `key_id` must equal the
**currently installed** root-owned anchor's `key_id`; a re-anchor drops every
older record (`old_signer`), the same rule as `operator_binding.py:938-939`.
A phone key (Phase 4) must be listed **now** (`device_listed_strict`, never
`device_is_admitted`): revoking it in the anchor voids its pending consents at the
next guard check. The anchor is read by
`trust.load_anchor()` only, never `load_local_anchor` (`trust.py:937`;
`operator_binding.py:853-870`, R4). Roles are never read.
*Tests:* `test_staged_anchor_ignored`, `test_reanchor_voids_record`,
`test_revoked_phone_voids_consent`, `test_roster_admin_alone_is_not_signer`.

**(e) Freshness.** `exp_min ≤ 240` (S-6: the nonce journal is a same-uid,
truncatable file, so a short window bounds what a truncation could replay;
default 60), and `ts + exp_min·60 ≤ ask.expires_at`
(the node's stored deadline). Skew is bounded by `SKEW_S = 120`: refuse
`ts > now + SKEW_S`, and refuse when `now > ts + exp_min·60 + SKEW_S`, at ingest
and again at consent. The default is `exp_min = 60`, and the card states it.
*Tests:* `test_future_ts_beyond_skew_refused`, `test_expired_at_consent_time`,
`test_exp_beyond_ask_deadline_refused`, `test_exp_over_240_min_refused`.
A `stale` or `future` refusal's facts carry the signer `ts`, node `now`, the
delta and `SKEW_S` (S-10), so a clock problem reads as one.

**(f) The surface kind is in the payload; trust comes from the key.**
`payload.surface` is logged and shown, and never changes the verdict.
`answered_by.surface` is never read as trust: the guard derives `operator-signed`
from a verifying stored record.
*Tests:* `test_forged_surface_label_rejected` (a row stamped `desktop` with no
record, written by a direct same-uid append to `asks.jsonl`: not consent through
this path; a row with `operator_sig` whose payload says `surface:"desktop"` and
whose signature fails: not consent), `test_surface_field_does_not_affect_verdict`.

**(g) Fail closed with named facts.** `sa.verify_*` returns
`(ok, reason, facts)`. Possible reasons: `malformed`, `domain`,
`no_installed_anchor`, `no_approver`, `old_signer`, `signer_not_presence_bound`,
`cert_invalid`, `device_not_admitted`, `signature_invalid`, `network_mismatch`,
`node_mismatch`, `session_or_ask_mismatch`, `relay_pointer`, `question_mismatch`,
`scope_mismatch`, `scope_glob`, `answer_mismatch`, `stale`, `future`, `replay`,
`device_signing_deferred`, `device_not_listed`. Any exception
counts as `malformed`. The facts name the check, the ask and the key_id prefix,
never the signature, the spki or the payload bytes. A refusal for an ask that
carries a failed record says *"signed answer to a-xxxx failed: question_mismatch"*.
(S-5: fail-closed holds only with S-2's strict predicate in place; with
`device_is_admitted`, an agent-minted cert would *pass* every check.)
*Tests:* `test_every_reason_reachable_and_named`, `test_facts_carry_no_key_material`.

**(h) CRITICAL stays refused; relay, phone-portal and agent relays stay
excluded.** A signed answer becomes an `OperatorWords` row and goes through the
existing consent readers. CRITICAL is refused before any consent is read
(`check.py:1807-1815`), and `operator_only` targets still need the operator's own
words (`check.py:1430-1435`). An unsigned portal answer (`remote`), an agent's
`answered_in_chat`, a hub, steer or Team Bridge relay, and any parent brief all
stay non-consent. The phone counts only through its strictly listed key, and only from Phase 4 (a).
`no_approver`: a host holding the anchor private key accepts no record
(`site_approval.anchor_key_held_locally`, `browser_bridge/site_approval.py:451-471`).
*Tests:* `test_critical_refused_despite_signed_answer`,
`test_no_approver_host_accepts_nothing`, `test_portal_answer_unsigned_excluded`,
plus the Phase 1 headline test below.

## 8. How the guard consumes it

- `store.fold` keeps each contributing write's `by` (it already walks every
  `revised` row, `asks/store.py:878-891`). It exposes `answered_by.operator_sig`
  from the write that supplied the **effective** answers, plus a list of the
  per-write signed flags.
- One new helper, `provenance.consent_source(record, session_id)`, returns
  `("trusted", label)`, `("signed", OperatorWords)` or `("none", reason)`. A
  signed answer counts only when **every** write that shaped the answer was
  trusted or signed. That is the `weakest_surface` rule
  (`asks/surfaces.py:83-104`) applied to signatures, so an unsigned revision
  launders nothing.
- `_ask_words` (`provenance.py:4145`), `_not_consent_label` (`:4711`) and
  `_untrusted_yes` (`:4817`) all call `consent_source`, so consent and refusal
  can never disagree. For a signed source the `OperatorWords` carries
  `label="operator, signed answer to ask"`, and `approved` and `firm` hold the
  signed `scope` paths only. Free label prose never widens it. From there the
  existing machinery applies unchanged: `ask_consents`, `_approving_ask`,
  `evidence_basis` (`check.py:943-976`, including `_ask_settles_git_root`) and
  `verify()`'s `ask=` citation (`check.py:1075-1096`).
- **Exception 2 then holds in code:** a verified record counts as an
  own-surface answer in this session or its task tree, naming the exact path.

## 9. Threat model

| Threat | Outcome under D |
| --- | --- |
| Compromised head agent (same uid as the surface) | Cannot sign: no readable presence key (a). Raw `sign` refuses the domain (§4.3). Can raise a `sign-answer` sheet, but its text names the forged node, session and scope. Cannot mint an admitted phone cert (strict listing, S-2). Cannot plant a re-anchor without failing the human fingerprint check (S-4). |
| Compromised node agent | Cannot forge a record. Can replay a record onto the same ask and qid only, which yields the same answer and is refused by nonce at ingest and consent (c). Can append a fake `desktop` row: the store-integrity residual (`surfaces.py:52-53`), unchanged by this design. |
| Compromised node (root) | Out of scope: it owns its own guard. Passwordless sudo makes the anchor advisory (`operator_binding.py:57-60`). |
| Replay or reorder over the mesh | Refused by nonce, freshness, and the per-write rule. |
| Roster change mid-flight | The roster is never trusted. Anchor re-key or phone revocation is read live at consent time (d). |
| Relay rewrites the question | `question_mismatch` (b). |
| Delegating agent quotes an approval | Closed in Phase 1 (§2, §11). |
| Agent re-issues a refused call hoping the judge flips | Refused deterministically unless newer consent exists (S-8, Phase 2). |
| Remote typed prompt | Consent today; the #4164 lane, not this design (S-3). |

## 10. Migration

- An unsigned remote answer stays excluded forever (no retro-signing, §6).
- An **older node** ignores the unknown frame field: `server.py:9386-9392` reads
  `answers` and `images` selectively, so the answer is stored unsigned and stays
  excluded. The node advertises capability `signed-ask-answer-v1`, and the head
  offers signing only when a node advertises it, so no gesture is wasted.
- An **older head** sends no signature, so nothing changes.
- An anchor with `presence: False` (node2 today) accepts nothing. The status
  line and the card say so and name the setup step.

## 11. Phased plan (one coder, ~5-6 h for Phases 1-3; "very significant" lane)

**Phase 1: close the delegated-words hole (fix lane, ~1.5 h; ships alone, first).**
- `harness/loop.py:7753-7762`: pass `user_request` only when
  `session_lineage(...).operator_opened` (`provenance.py:4892`). Otherwise
  pass `""`, and the excluded count already reports "delegating-agent message".
  This covers the dangerous-delete judge and the user-rule judges, because both
  read `ctx.user_request`.
- `guardrails/judge.py:204-209`: drop the `user_request → operator, direct`
  fallback when an `OperatorBrief` was supplied (old callers keep it).
- Tests in `tests/unit/guardrails/test_check_delegated_words.py`:
  `test_parent_resume_quoting_approval_is_not_operator_consent`, the headline
  test. It replays #4135: a subagent context whose newest user row is a resume
  brief quoting "The OPERATOR answered ask a-bf67 … Yes". The command passes a
  sibling pad to `python`, and a stub judge that ALLOWs iff the brief contains an
  operator block → BLOCK, and the brief contains no `<operator` block.
  `test_steer_and_hub_message_never_operator`;
  `test_user_rule_judge_gets_no_delegated_words`;
  `test_operator_opened_session_keeps_direct_words` (no regression).

**Phase 2: verify at the node (feature, ~2 h).** New
`local_operator/asks/signed_answers.py` (payload builder, digest, scope,
`verify_ingest`, `verify_stored`, nonce journal). It hooks `server.py:9380-9420`
(ingest) and `asks/store.py:831-891` (fold) to `provenance.consent_source`, with
the `_ask_words`, `_not_consent_label` and `_untrusted_yes` call sites. The domain
tag goes in `operator/verify.py`. Tests: every (a)–(h) test above, in
`tests/unit/asks/test_signed_answers.py` and
`tests/unit/guardrails/test_signed_consent.py`. Signatures come from the
file-only test backend (`approval-authority.md:160-166`) with the anchor seam
patched to report presence. A real-path QA rig runs two isolated configs (head
and node) over a loopback mesh.
- **Re-issue rule (S-8).** On every guard refusal, record `(task_tree_root,
  sha256(tool || canonical_json(args)), refused_at, guard_version)` in
  `<config>/guardrails/refusals.sqlite`, beside the existing `created.sqlite`.
  An identical call in the same tree is refused deterministically, before any
  judge, unless the tree holds a trusted or verified-signed consent row newer
  than `refused_at`, or an operator-opened session has a direct operator turn
  newer than it. Exception 1 (re-evaluation) is preserved: a different installed
  guard version falls through to normal evaluation. Tests:
  `test_identical_reissue_refused_without_newer_consent` (#4135 step 92 replayed:
  refused), `test_reissue_allowed_after_newer_signed_consent`,
  `test_reissue_reevaluated_on_new_guard_version`.

**Phase 3: sign at the head (feature, ~2.5 h, including the §4.3 helper verb and
its operator-run sheet evidence).** Signing goes in
`mobile/attach_client.py` (beside `_present_operator_signature`, `:1922`); the
desktop facade and the TUI dock call it. The node sets `signable`
(`asks/queue.py`), and the wire capability is declared. The sheet copy lives in
`operator/sign.py`. Tests: `test_signable_only_on_path_asks`,
`test_payload_binds_origin_for_relayed_card`,
`test_no_signature_offered_to_old_node`. Visual before/after frames of the card
go to the designer.

**Phase 4 (deferred; separate PR): phone signing**, gated on the phone rendering
the parsed payload (S-7), with `device_listed_strict` and its two tests (S-2).

**Docs:** `guides/guardrails/GUIDE.md:48-52` and `check.REFUSAL_SCOPE`
(`check.py:303-345`) replace "not built yet" with "an answer signed with the
operator key (Touch ID or a paired phone) counts; an unsigned remote answer
never does".

## 12. Risks to watch during rollout

1. **Phase 1 raises subagent refusals** of opaque HIGH calls that a parent's
   brief used to "approve". That is intended. Watch the `guardrail.verdict … block`
   ratio for delegated sessions for a week, and expect operators to answer asks
   instead.
2. **Nothing verifies until re-anchoring.** Every current anchor that resembles
   node2's is file-only. Report the anchor level in `lop network status`, so
   "signed answers unavailable" is visible rather than silent.
3. **Prompt fatigue.** Count signable asks per day. If it exceeds ~10, revisit
   the per-launch delegation (§3), not the trigger.
4. **Scope extraction drift** between head and node versions would show up as
   `scope_mismatch` spikes. Version the extractor inside `domain` (`-v1`).
5. The unverifiable-target path still ends in a context-free plain ALLOW
   (`check.py:1878`, `:1015-1018`) for a call never refused before. Phase 1
   removes delegated words from that judge, and S-8 makes a refused call's
   identical re-issue deterministic. Watch `refusals.sqlite` growth, and prune
   rows older than 7 days.
6. The re-anchor ceremony (S-4) depends on a human comparing fingerprints. The
   install step must make the comparison impossible to skip.

## 13. Security review round 1 — disposition

| Finding | Severity | Disposition | Where |
| --- | --- | --- | --- |
| S-1 generic Touch ID sheet | MAJOR | Fixed: `sign-answer` verb with an LAContext reason built from the parsed bytes; raw `sign` refuses reserved prefixes; 5 tests; residual stated | §4.2(1), §4.3, §5, §9 |
| S-2 `device_is_admitted` exemptions | BLOCKER | Fixed: phone signing deferred (`device_signing_deferred`); Phase 4 requires `device_listed_strict` with no presence or migration exemption; 2 named tests | §7a, §7d, §11 Phase 4 |
| S-3 remote prompts are consent | MAJOR (not this lane) | Noted, with UX cost; owned by the #4164 Phase 1 lane, pending Damian | §2, §9 |
| S-4 unauthenticated re-anchor | MAJOR | Fixed: install prints `key_id`/`spki_fp`, compared against the Mac's installed anchor or the keyagent `public` verb, never staged export; re-anchor voids proofs, certs and signed answers | §4.1 |
| S-5 fail-closed depends on S-2 | minor | Stated | §7g |
| S-6 cap `exp_min` | minor | Cap 240, default 60 | §5, §7e |
| S-7 phone renders exact payload | minor | Phone signing deferred until it does | §7a, §11 Phase 4 |
| S-8 deterministic re-issue rule | minor | Added to Phase 2 with 3 tests | §11 Phase 2, §9, §12(5) |
| S-9 globs in scope | nit | `scope_glob` refusal in the verifier and the helper | §5.2, §4.3, §7g |
| S-10 skew facts | nit | `ts`, `now`, delta and `SKEW_S` in the facts | §7e |
