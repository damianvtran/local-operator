# Design: operator authority for gate-loosening control requests

Status: revision 2 implemented through stage C (the authority is the OPERATOR,
verified against a root-owned anchor); stages D/E/F outstanding. Issue:
`damianvtran/local-operator#1310`. Predecessor: #1282 / PR #1291, which closed
the *settings-write* half of the same invariant.

## 0. The invariant

```text
a constrained subject must not be able to mint the authority
that removes its own approval requirement
```

#1282 enforced it for `config.yml` writes: `tool_approval_mode` is live, a model
tool can write the file, so a loosening is authorised on the write's
**provenance** (`ConfigChange.source == "local"` in the process holding the
gate) rather than on its value.

This document covers the other boundary, which #1282 does not touch: the
**control plane**. A running session publishes a discovery record at
`<config>/run/mobile/<pid>.json`, mode 0600 under a 0700 directory, carrying
`control_key` and `control_port`. The runtime used to accept
`{"op": "slash_result", "command": "approvals", "args": "auto"}` — and
`{"op": "approval_answer", "approved": true}` — on that key alone.
`slash_result` rather than a bare `slash` is the op that reaches the sink: the
runtime answers a bare `slash` with "terminal-only here", so the route an
attacker actually uses is the `_result` one. A model-authored `bash` tool call
runs as the **same uid**, so it could read the record (it is 0600 *under its own
account*), dial loopback and remove its own gate. Same invariant, same subject,
one route over.

The two halves are not equally easy for that attacker, and the difference is
worth stating exactly: `/approvals auto` is a single frame, while the card half
needs the `request_id`, which is 8 random bytes and never guesses. It is not out
of reach, though — the welcome projection hands the pending card, id included, to
any client that authenticates with the record key, and a same-uid process can be
that client (that is what the negative control does). So the card half is
reachable *after one dial*, not by guessing (QA round 1, Q2).

The distinction #1310 draws, in one sentence: **same-account process identity is
not operator consent when the operation expands that process's own authority.**

## 1. The class

Exactly one predicate, `harness/approval.transition_authority`, beside
`loosening_is_authorised`:

```python
def transition_authority(command, args, *, approved=None) -> Authority
```

- **Authority-increasing** — the request removes or answers the gate that
  constrains the caller:
  - `/approvals auto|off|yolo` → `ServingSessionHandle._auto_approve = True`
    (or the app's `_set_approve_all` for the gate a TUI owns);
  - `approval_answer(approved=True)` → resolves the parked card.
- **Ordinary** — everything else: `ask|on|prompt`, any unknown word, a bare
  `/approvals`, `default …`, `read`/`status`/`stop`/`prompt`/`model`/`rename`,
  `peer_message`, `ask_answer`. An `approval_answer(approved=False)` is
  ordinary **on purpose**: a deny settles the card in the safe direction, so it
  must keep working from every surface that can reach the session.

Aliases resolve through `primary_slash_name` before matching — the same
resolution both dispatch hosts already perform — so an alias cannot slip past
the seam and reach a sink that would have honoured it.

The class is machine-testable rather than inferred from UI wording, which was
an explicit acceptance requirement: the predicate is a pure function, and the
seam's op table is re-derived from `session/runtime/server.py`'s own dispatch
source by `tests/unit/session/runtime/test_approval_authority_seam.py`.

## 2. The mechanism (revision 2): the operator, verified offline

**The mechanism changed; nothing else did.** The class (§1), the seam, the op set,
the typed refusal and the proof machinery are all as they were. What changed is
what "authorised" MEANS: revision 1 asked "did the process that spawned this
runtime say so?", and revision 2 asks "did the OPERATOR, or the operator's
device, say so — offline, with a human gesture?"

Why the change was forced is recorded in §2.5 rather than here, because it is a
capability argument and the mechanism only makes sense on top of it.

### 2.1 The trust anchor — decided by measurement, not assumption

Authority is verified against an **operator ES256 (P-256) key** whose public half
is pinned in a file the gated subject cannot write:

| platform | anchor path | ownership |
| --- | --- | --- |
| macOS | `/Library/Application Support/local-operator/operators/<uid>.json` | `root:wheel`, dir 0755, file 0644 |
| Linux | `/etc/local-operator/operators/<uid>.json` | `root:root`, same modes |
| Windows | `%PROGRAMDATA%\local-operator\operators\<sid>.json` | Administrators |

The file holds PUBLIC data only (`{key_id, alg, spki, backend, presence, label,
created_at, devices}`), costs **one `sudo`/admin step at onboarding** (and again on
rotation), and is written by an explicit `lop operator install`. The code works
without it: a host with no anchor reports a lower level and refuses loosening
rather than failing to start.

The probe measured, in a throwaway keychain so nothing touched the operator's
(measured 2026-09-19, macOS 26.6.2, uid 501, SIP on):

- **a keychain item is silently substitutable**: `delete-generic-password` then
  `add-generic-password` with a different value both return 0, the readback shows
  the substituted value, and **no prompt and no privilege** are involved;
- two items with the same service+account cannot coexist inside ONE keychain
  (`errSecDuplicateItem`, rc=45) — the attacker must *replace*; but **shadowing
  across keychains works**, and **search order is rewritable with no
  authentication** (`security list-keychains -s`, rc=0);
- **`/Library/*` is not writable by this admin user at all**.

Two consequences the implementation respects, stated because they bound the
claim:

- **the enforcement point matters as much as the anchor.** A verifier inside a
  user-writable tree can be replaced or bypassed by the same subject. The harness
  install (`~/.local/share/uv/tools/local-operator`) is user-writable today; that
  is the pre-existing Stage-2 confining epic (§4), not something this change
  fixes;
- **the anchor path must not be redirectable** by anything the uid can write. It
  is built from constants only — no environment variable, no config value, no
  `PATH` lookup — every component is `lstat`-checked for symlinks, and the file is
  opened `O_NOFOLLOW` with owner and mode validated **from the descriptor**, not
  from a second `stat` of the path. The test seam for the anchor is a
  module-level global a test patches, never an environment lookup, and
  `test_the_anchor_path_cannot_be_redirected` pins that.

### 2.2 The private half: what a signature costs, per platform

| platform | private half | presence per signature | guarantee |
| --- | --- | --- | --- |
| macOS | Secure Enclave P-256, `SecAccessControlCreateWithFlags(… kSecAccessControlPrivateKeyUsage \| kSecAccessControlUserPresence)` via `ctypes` on Security.framework | every `SecKeyCreateSignature` raises Touch ID / login-password | strong |
| Windows | CNG key in the Microsoft Software KSP with `NCRYPT_UI_POLICY` / `NCRYPT_UI_PROTECT_KEY_FLAG` — **not implemented on this build** | none on this build | **`file-only`, which is NOT a boundary** (below) |
| Linux | no presence store implemented yet — see below | none | **`file-only`, which is NOT a boundary** |

macOS reachability was verified separately from the key itself: `ctypes` loads
Security.framework with all 13 needed symbols resolved
(`SecKeyCreateRandomKey`, `SecAccessControlCreateWithFlags`, `SecItemAdd`,
`SecItemCopyMatching`, `SecKeyCreateSignature`, …), `CFDictionaryCreate` accepts
the attribute shapes below, and the framework's own errors are readable through
`CFErrorCopyDescription`.

**MEASURED, and it shapes the whole backend: a Secure Enclave key cannot be
created in a legacy file keychain at all.** Every attribute shape tried against a
throwaway keychain — with and without `kSecUseKeychain`, with each of
`kSecAttrAccessibleWhenUnlocked`, `WhenUnlockedThisDeviceOnly` and
`WhenPasscodeSetThisDeviceOnly`, and with each of the `privateKeyUsage` and
`userPresence` flags — returns `errSecParam` ("inconsistent private key
parameters for key generation"). Secure Enclave keys live in the
**data-protection keychain**, i.e. the user's login keychain, and that placement
is the OS's choice rather than ours. Two things follow, both recorded rather than
worked around:

- the private half's own substitutability is **irrelevant here**, and it is worth
  saying why: replacing the key does not help an attacker, because the ANCHOR
  pins the public half. A substituted private key produces signatures that fail
  verification. The measured substitutability above matters for the ANCHOR, which
  is why the anchor is a root-owned file;
- a test can never exercise this backend, because doing so writes an item to the
  operator's login keychain — which the design forbids and which would be a
  prompt on the operator's screen. The backend's contract is therefore exercised
  through the `file-only` implementation (same interface, real ES256 through
  `cryptography`), and the Secure Enclave path is exercised by the operator's own
  `lop operator init` / `lop operator sign` runs. The level report keeps the
  difference visible instead of implying the strong path was measured.

**Windows is not a presence tier on this build, and the ladder now says so
instead of dying on it.** This row used to read "per-use consent dialog · strong",
which the code contradicted: `CngBackend.create()` raises *"not implemented on this
build"*, and `choose_backend("auto")` returned that backend for `os.name == "nt"`
without asking whether the host could USE it — so `lop operator init` exited 1
naming an internal backend, on a host whose honest level is `file-only`. The
ladder now falls to `file-only` (reported as the level it is), an explicit
`--backend cng-presence` fails with an operator-readable sentence naming
`--backend file-only` and saying that a file-backed key raises no prompt, and
`CngBackend.supported()` answers for the BUILD rather than for the host — which is
the question its only caller asks. **Read, not run**: there is no Windows host in
this CI, so this is a claim about the selection logic and the failure text, and the
`NCRYPT_UI_POLICY` call itself remains unexercised here exactly as the macOS
presence path is.

**Linux has no presence backend in this change.** `file-only` — a 0600 PKCS#8
P-256 key — is what this build creates there, and it is **not a boundary**: the
same uid can read it and sign for the operator with no human act. TPM+PIN sealing
is named as stage F work rather than shipped untested, because an unverified
sealing implementation would make the level report claim a guarantee nothing
demonstrates. `operator_authority_level()` returns `operator-file-only` there and
`lop operator status` prints what it means.

### 2.3 The wire (additive — no `PROTOCOL_VERSION` bump)

- new **ordinary** op `{"op":"operator_challenge","action":"loosen"|"approve","request_id":…}`
  answered by `{"op":"ack","req":…,"challenge":"<64 hex>","expires_s":30}`.
  Ordinary **by construction** — it grants nothing on its own; the signature it
  is used to produce is what carries authority — so it rides the record key like
  every other control op. The reply rides the existing **`ack`** shape rather
  than a frame with its own op name, and that is a protocol requirement rather
  than style: `AttachClient`'s reader routes replies only for `ack`/`error`/
  `result`, so a reply carrying a novel op name would fall through every branch
  and tear down the whole connection — taking the caller's in-flight request
  with it, which is exactly the "an old front end must keep working" guarantee
  this additive design exists to keep (found by measurement; see
  `test_an_operator_signature_loosens_a_runtime_this_process_never_spawned`);
- the runtime binds the challenge to `(connection, session_id, action,
  request_id, expiry)` and **single-uses** it: the entry is `pop`ped by the first
  frame that presents a signature for that `(action, request_id)`, before
  verification, so a captured signature has exactly one use. A missing or expired
  challenge is a REFUSAL rather than "not offered";
- an increasing frame may carry `operator_sig`, `operator_key_id` and (for a
  device) `operator_cert`. The signed message is domain-separated and
  length-prefixed:

  ```text
  b"lop-operator-v1\x00" + lp(action) + lp(session_id) + lp(request_id or "") + lp(challenge)
  ```

  The length prefixes are not decoration: without them an attacker who controls
  one field can shift bytes between adjacent fields and produce a valid-looking
  message for a different action. The domain tag stops a signature from any other
  protocol this key signs from being replayed here;
- verification is offline ES256 against the pinned anchor, or against a device
  public key taken from a certificate that verifies under the anchored operator
  key and has not expired;
- `operator-signature-v1` is advertised in the record's `capabilities`, and
  **unconditionally** — the runtime can always verify (the public half is all
  verification needs), and gating the advertisement on an installed anchor would
  make a host mid-onboarding answer "unsupported" where the truthful answer is
  "not yet installed";
- an OLD runtime answers the unknown op with its generic error frame, which the
  client reads as "predates the feature" and handles by not being able to loosen
  — exactly what that runtime could do before. The engine has no
  `PROTOCOL_VERSION` bump because nothing *must* be understood for ordinary
  control to keep working, which is the test the bump exists for.

**Two sources, one predicate, in one stdlib-only module.**
`harness/approval.admit_increasing(*, capability, signature)` is the whole policy:
an increasing frame is admitted by a proven spawn capability, or by a signature
that VERIFIES, and a signature that was offered and failed admits nothing on its
own (presenting a bad signature must not be a way in — nor a way to lock out a
capability holder whose client also attached a stale one). The crypto that
produces the verdict lives in `local_operator.operator.verify` behind a lazily
imported `cryptography`, so `harness/approval.py` keeps its stdlib-only import
graph and gains one function that takes a **verdict**.

### 2.4 The sources at the seam

`_authority_admitted` admits any of:

1. the **spawn capability** — unchanged, no prompt (the common interactive case);
2. an **operator signature** (presence-gated), obtained through the one signing
   entry point — `lop operator sign --challenge … --purpose … [--session …]` →
   `{sig, key_id}`. In-process use is allowed for the TUI and the desktop backend
   because the presence enforcement is in the OS call, so a caller in this process
   gets the same prompt a subprocess would;
3. a **device signature** under an operator-signed certificate (the phone —
   stage D);
4. the **run-scoped supervisor credential** (stage E).

**One signing entry point, and the prompt names the session and the effect.**
Anything on the machine may invoke it, including a model's tool child — and it
always prompts. That is the boundary: not the caller's identity but "only a human
can answer this". Prompts are rate-limited by construction: one signature per
action, because the memo is **popped** by the frame that uses it, and the number
of live challenges is bounded per connection and in aggregate
(`server._MAX_CHALLENGES_PER_CONN`, `server._MAX_LIVE_CHALLENGES`).

**WHERE THAT COPY ACTUALLY REACHES A HUMAN — corrected, because this paragraph
used to claim more than the code did.** The copy is built once
(`operator/sign.py::effect_copy`, e.g. *"Authorise the operator key to LOOSEN the
approval gate of `<session>`"*), and there are exactly three routes it can take,
none of them the OS dialog:

- the CLI prints it to **stderr** before signing (`lop operator sign`; stdout
  stays the value and nothing else), so the person who typed the command reads it
  before the gesture;
- an attached viewer's pane paints it as a transcript notice. The wiring is
  `AttachClient(on_operator_prompt=…)` at the pane's construction site, which is
  what this paragraph previously asserted and what no production site passed —
  the callback took its fallback branch and the sentence became a log line;
- the relay's own clients log it at `warning`, because that surface has no human
  at a terminal and a machine-side prompt with nobody looking is the event an
  operator most needs to find afterwards.

**The OS sheet itself cannot carry it, and that is a property of the API rather
than an omission.** `SecKeyCreateSignature` takes no parameters dictionary, so
there is nowhere to pass a reason; `kSecUseOperationPrompt` was the key that would
have carried one and Apple deprecated it in macOS 11 (availability 10.10–11.0),
before the key type this backend creates was in use here. On Windows the same
holds for the CNG backend's unimplemented consent dialog. So the mitigation for
the misread-prompt residual is the product's own copy on the surfaces above, and
this document says that rather than claiming the dialog speaks for us.

### 2.5 Why the spawn capability was not enough

The measured defect (#1310) was a same-uid tool child reading the session record,
dialling the loopback control socket and setting its own gate to `auto`. Revision
1 closed it by making the spawner the authority — and cost capability the
operator will not accept:

- **the phone can never loosen in any session.** Its spawn path
  (`mobile/daemon.py`) never passes `--operator-fd`, and `remember_operator_cap`
  has exactly one production call site (`session/runtime/launch.py`), so
  `entry.operator_cap` is always `None`. The unit test that appeared to cover the
  relay case fabricated the spawn;
- a pane attached to a runtime **another process started** (wake supervisor, peer
  send, `lop refresh`) cannot loosen;
- the desktop app can loosen only for sessions its own backend spawned;
- a supervised `lop exec --control` run can be **denied** but not **approved** by
  its supervisor;
- background-engaged runs (wake/peer) park a card nobody can answer, for up to
  `runtime.unattended_gate_timeout` — 24 h by default.

Revision 2 restores every one of those while keeping the invariant, because the
authority is no longer the spawner: it is a fact about the operator that a
constrained subject cannot mint, verified against a root-owned anchor.
## 3. Surface by surface

| surface | loosens? | approves a card? | how it proves it is the operator |
| --- | --- | --- | --- |
| TUI pane that OWNS the session (gate in-process) | yes, unchanged | yes | the human's keystroke; nothing crosses the wire |
| pane / CLI / desktop that SPAWNED the runtime | yes, unchanged, **no prompt** | yes | the spawn capability |
| pane ATTACHED to a runtime another process started | **yes — one presence prompt** | yes | `lop operator sign`, forwarded by `AttachClient` |
| desktop app, **any** session | **yes — one presence prompt** | yes | operator signature via the backend |
| phone / relay, **any** session (spawn-independent) | **yes** | **yes** | device signature under an operator-signed certificate (`lop pair`; the portal signs with a non-extractable WebCrypto key). The relay mints nothing: `POST /api/sessions/<id>/operator/challenge` forwards one ordinary frame and carries `operator_sig`/`operator_key_id`/`operator_cert` back |
| `lop exec --control` supervisor | **yes** with `--supervisor-fd N` | **yes** | run-scoped supervisor credential: the run mints its own capability, writes it UP the inherited descriptor, and closes it |
| `lop exec --background --control` (launcher gone) | from a human surface only | yes | operator/device signature |
| CLI one-shot run by a script | yes, with a prompt or a paired device | yes | operator/device signature |
| tightening / deny / report / read / status / stop / prompt / model / rename / peer_message / ask_answer | n/a — ordinary | n/a | the record key, unchanged, on every surface including the phone |
| `--yolo` / headless non-TTY | n/a (born `auto` / born denying) | n/a | a launch-time human act |
| wake/peer-engaged background run | yes | **yes — the card is answerable** from phone/desktop/pane for a runtime none of them spawned | operator/device signature, plus the policy levers `--yolo`, `tool_approval_mode: auto` |
| old front end (before `operator-signature-v1`) | refused until it updates — bounded to one release, and **loosening only** | same | n/a |

The rows that changed are the four the operator named as unacceptable: the phone
row no longer depends on the spawn path, an attached pane loosens, the desktop
app works for sessions its backend did not start, and a background-engaged run's
card is answerable. **No row regressed**: the owning pane and the spawning
console are untouched (the latter deliberately prompt-free), and every ordinary,
tightening and deny route is byte-identical to what it was.

**The supervisor row is driven, not reasoned about.** `--supervisor-fd` was a flag
that parsed, validated and then did nothing: `run_session` reads the descriptor off
the `ExecArgs` object rather than off the argparse namespace, and the construction
in `cli.py` omitted the field, so the run minted no capability and a supervisor
waited out its whole timeout. Found by the e2e cell that drives a real supervised
run — socketpair first, descriptor number in argv, capability read upward,
`remember_operator_cap(pid, cap)`, then a TUI supervisor whose `y` is ACCEPTED and
whose gated tool actually runs (`tests/e2e/test_exec_startup_e2e.py::
test_a_supervised_run_is_approved_through_the_handoff`). Its sibling keeps the
`--background` shape, where nobody may approve, so the pair states the rule and its
exception.

Two rows were narrower than this document could honestly make them while stages D
and E were unlanded, and both have since landed:

* the **phone** row no longer depends on the spawn path at all. Its authority is a
device signature under an operator-signed certificate, and the relay is a
forwarder — it holds no operator key, so it cannot make anything a signer even
when it is driven by a same-uid process holding the portal password. The body
scrub was NARROWED to make this real: `operator_cap` stays dropped (machine-held
proof material), and `operator_sig`/`operator_key_id`/`operator_cert` are now
admitted, because a signature is unforgeable and its challenge is single-used.
* the **exec supervisor** row is no longer a deny-only surface. The run mints its
own capability and hands it UP a descriptor the supervisor opened
(`--supervisor-fd N`), which is the mirror of the downward handoff
(`OperatorCapHandoff.deliver`); the descriptor is closed before the run's first
tool call, so no tool child of the run can find it. `--background --supervisor-fd`
is refused outright: the launcher exits and takes the far end of the socketpair
with it, so the combination could only produce a run that LOOKS supervised while
nobody can approve its cards.

One path deviation from §2.3, recorded because the design text says otherwise:
this repository has no `/v1/mobile` namespace — every route it serves is under
`/api` (`mobile/daemon.py`'s route table and the portal's `api.ts`), so the
challenge endpoint is `POST /api/sessions/<session_id>/operator/challenge`. A
second namespace invented to match the document would be the duplication this
project's conventions forbid, and the served paths are pinned by a test.

### 4.1 The remedies, with their conditions

| remedy | what it does | when it works | in the refusal copy? |
| --- | --- | --- | --- |
| authorise from this machine | one presence gesture (Touch ID / CNG consent) signs a per-action challenge; the gate loosens or the card resolves | a local surface (pane, CLI, desktop backend) and an anchor installed. On a host whose private half is `file-only` the same command works **without** a gesture, and `lop operator status` is where that is said | yes, as the primary remedy |
| authorise from your paired phone | the phone signs the challenge with its own key | after pairing (stage D) | yes |
| `/approvals ask` | tightens | everywhere, including a follower, the phone and the desktop | yes |
| `--yolo`, or `tool_approval_mode: auto` in `config.yml` | the NEXT session starts loosened | at launch; the config write is a file edit (or the desktop app's settings), not a session command — no control connection may write it | yes |
| `lop refresh` | asks a live runtime to move to the install on disk and leave at its next boundary | the install on disk has MOVED (an update). Without a move a runtime answers "already current" and stays | **no** — conditional on a move, and the copy has no room to state the condition without pushing the reason off a narrow screen |
| `lop stop <session>`, then reopen | ends this session so a new runtime starts under a console that owns it | always, and it ENDS the running turn: named here for completeness, never as a remedy | **no** |

**The remedy that is gone, and why that is the point of this revision.** Revision
1's copy ended with *"let its runtime retire and reopen it here — the window that
opens a runtime owns its gate"*. That remedy was true and useless at the same
time: it made the next runtime the reader's, which is a workaround for an
authority model that should never have required one, and it was unavailable
exactly where it was most needed — a background-started runtime has no window to
reopen it from. It is **deleted**, and a test asserts the phrase is absent rather
than merely that the new phrases are present, so it cannot creep back as a
"harmless" sentence.

**...on every surface a reader meets, which took a round to be true.** The refusal
notices were rewritten first; the REPORT — the sentence an operator reads *before*
acting — still said *"typed in the terminal or app window that started this
session"* and then appended the deleted remedy, and the suite pinned BOTH ends
(`test_serving_approvals_live.py` required the clause PRESENT while
`test_approval_authority_seam.py` required it absent from the refusal). Under this
model that was wrong twice over: it withheld capability that now exists (a pane
attached to another process's runtime loosens with one gesture, §3), and it named
a window a background-started runtime does not have. Both handles
(`session/runtime/serving.py::_adopt_remedy` and `tui/app.py::_adopt_remedy`, kept
in the same words) now name the real levers, and the contradiction is closed by
inverting the report cell rather than by narrowing this paragraph.

**The missing-anchor state is named wherever it is the reason.** Between
`lop operator init` (which stages the anchor) and `lop operator install` (the
privileged step that lands it) neither named surface can sign, so the refusal, the
`/approvals default` receipt and the report all name the install step instead of
offering remedies that cannot run — and the refusals do it under a distinct typed
code (`operator_authority_unconfigured`, a subclass of
`operator_authority_required`, so every route that keys on the base keeps working)
which the phone can act on without pattern-matching English. `lop pair` and the
portal's pairing screen report the same host state, from the same predicate, rather
than promising authority the machine cannot honour.
### 4.2 What the refusal looks like on a narrow screen

The copy is 288 characters (the card's is 228) and both are under the runtime's
400-character error-frame cap, which is asserted as a number
(`test_the_refusal_copy_names_the_remedies_and_not_a_rule`) so a longer copy
fails a test rather than a phone.

The revision-2 copy is SHORTER for the command (288 against 345) and longer for
the card (228 against 162): the retired remedy sentence was the longest clause in
the command's, and naming the operator's levers honestly costs the card a line.
Both are re-measured rather than assumed, and the row pins moved with them.

Rows are a property of the RENDERER, not of the characters: the notice block
wraps at its own content width — measured **40 cells** at a 44-column terminal,
not 44 — where the command's copy renders as **9 rows** (was 12) and the card's
as **8** (was 6). Both measured on the real widget at 44×20 with the production
CSS, with an SVG frame either side of the change; the pins live in
`tests/unit/tui/test_approvals_ux.py::test_the_refused_card_notice_reaches_the_screen`.

**The block does not fit at every height, and the copy's order is the reason it
matters.** Measured at 44x20 on this head, the transcript's content area is
**13 rows** in the pin's staging, **11 rows** in a conversation (`refusal-44x20`:
`region [1,1,42,13] size [41,11] virtual [40,15]`, `scroll_y=4` — the maximum,
15 − 11, i.e. scrolled to the bottom; the notice is `region [2,1,40,12] size
[40,12]`, so **exactly ONE row of it is above the fold** — its first. The other
three rows off the top are the prompt's own two (`UserBlock region [2,-2,40,2]`)
and the container's adaptive gap row, not more of the notice. What the operator
reads first is therefore the remainder of the reason sentence, not its opening
words), and **2 rows** with the re-armed card docked (`region [1,1,42,4] size [41,2] virtual [40,8] scroll_y=6`, the notice
itself at `region [2,-2,40,6]`), where the rows that paint are the block's **last
two** — `blocked until someone does.` / `Denying it works from here.` So at 44
columns the operator gets the reason and the remedies — minus the block's first
row — in a conversation, and gets **the notice's tail with the card up**, which is the
frame where the card is the thing that has to be answered anyway and the notice's
detail is recoverable once it is.
Both block heights are pinned as exact numbers in
`tests/unit/tui/test_approvals_ux.py::test_the_refused_card_notice_reaches_the_screen`
(design round 3 D14 and agent R3-5 corrected a wrap-based pin that said "9 rows"
and measured a wrapping the frame does not do; agent round 4 R4-2 corrected the
area figure this text used to claim).

## 4. The residual

The capability is a real boundary only where the operating system makes it one.
It does **not** defend against a process that can read the console's memory or
file descriptors: Linux with `ptrace_scope=0`, Windows, or an attached debugger.
That is the honest statement of what this closes and what it does not — it
removes "read a world-readable-under-this-uid record and dial a port" as an
attack, which is what a model-authored tool call can actually do today.

**A revocation reaches a RUNNING runtime, within a bound.** The anchor is read
once and pinned in memory, which is what stops a same-uid subject forcing a read
of the root-owned file per frame — and it used to mean the revocation list was
frozen for the lifetime of a session, so a device the operator had revoked went on
loosening and approving on every runtime started before that edit (measured:
`load_anchor` calls: 1). The cache now re-reads the anchor on a bounded window
(`trust.ANCHOR_REFRESH_S`, 30 s, never per frame), adopting only a load that names
the same operator key — a rotation is a privileged step and does not take effect
mid-session — and the certificate cache re-checks revocation on every use, so the
window is the ONLY staleness a revoked device gets. `lop operator devices
--revoke`'s receipt states that window, and the aggregate challenge count is bounded
across connections rather than per connection, because the per-connection maximum
alone left a subject that can dial as many sockets as it likes unbounded.

Also deliberately not fixed here, recorded so it is not mistaken for covered:

- **background-spawned runtimes have no console**, so nothing may loosen them
  until someone attaches one that does (a wake supervisor's runtime is exactly
  case 3 in the table). The open question this change used to carry — could a
  supervisor get a credential without reopening the hole for the most unattended
  surface — is now answered for the SUPERVISED case by `--supervisor-fd` (stage
  E): the credential is the run's own, handed up a descriptor the supervisor
  holds, and closed before any tool child exists. It is deliberately NOT answered
  for a DETACHED run (`--background --control`), where the launcher that would
  own the far end has exited: unattended approval there stays `--yolo` /
  `tool_approval_mode: auto`, because any credential that survives on disk is
  readable by the attack this issue exists for. `tests/e2e/
  test_exec_startup_e2e.py::test_exec_supervisor_approval_ui` drives a parked
  `--control` gate and pins the behaviour;
- **Windows is not a presence tier on this build, and no Windows host was run in
  this change.** The ladder's decision and the failure text are read from the code
  and pinned by a test; the `NCRYPT_UI_POLICY` call, like the macOS presence
  prompt, remains unexercised here. A host in that state gets `file-only`, which
  this document already grades as not-a-boundary;
- **the device certificate store is writable by the subject it is meant to
  constrain**, and that is by design rather than an oversight: it sits under the
  operator's config root (`<config>/operator/devices/<device_id>.json`, 0644 under
  0700) because the relay and every runtime have to read it. What defends it is
  the SIGNATURE — a substituted or forged certificate fails verification against
  the anchored operator key, pinned by a test — and the id is DERIVED from the
  public point, so a revoked device cannot relabel itself. What is NOT defended is
  AVAILABILITY: a same-uid subject can delete certificates and flood pairing
  requests, which are denials rather than escalations. The revocation list is the
  opposite case and lives in the ROOT-OWNED anchor (`lop operator devices
  --revoke`, through the same privileged step the anchor install uses), so a
  revoked device cannot be un-revoked by the device. It is written in TWO places,
  and the second one cost two rounds to notice (round 9's Q9-1/R9-2): the relay
  keeps its own copy under the config root so the common case needs no privileged
  read, that copy is SCOPED TO THE ANCHOR'S KEY ID it was stamped with (a record
  naming a superseded key describes devices of an anchor that is gone, which is
  what made a genuinely new anchor fail to lift a revocation), and
  `lop operator devices --authorise <device id>` is the host-side inverse verb
  that clears both halves — the route the phone's refusal copy names;
- **a process that proxies the whole session** can relay a connection's proof
  and that connection's requests — which is what a proxy is — but it never
  learns the capability and cannot originate a request of its own: nothing that
  crosses the wire is reusable on another connection (agent review round 1,
  R1-1; `test_an_impostor_endpoint_learns_nothing_it_can_replay`);
- **the phone relay's own password** remains the authority for reaching a
  runtime the relay started, over a remote transport — Stage 3 replaces it with
  a device-bound credential;
- **the desktop `claim` handshake and the serve record's `claim_key`** are a
  different plane (governing a *daemon*, not a session gate) and are unchanged;
- **nothing identifies WHICH window owns a session when several are live.** Under
  revision 1 the copy said "the window that started this session" and that
  referent was resolvable only by elimination: no `lop info` field, no band
  marker named the owner. Revision 2 removes the question from the refusal (no
  remedy depends on owning a window any more) and answers the part that remains
  in the PROMPT: `effect_copy` names the session and the effect, so the person
  answering a Touch ID dialog is told which gate their gesture is about to
  loosen (raised round 2 UX U9). **That sentence took three sweeps to become
  true.** The refusal and the card refusal were rewritten first, the `/approvals`
  REPORT in round 6, and `LOOSENING_REFUSED_NOTICE` — which reaches a reader on
  both hosts — only in round 7 (QA Q7-1), each round having fixed the sentences
  someone had happened to look at. It is now a claim a test can fail on rather
  than a claim in prose: three cells in
  `tests/unit/harness/test_approval_authority.py` sweep an explicit subject list —
  every notice constant, every non-docstring string literal under `local_operator/`,
  every string in the portal sources (`local_operator/mobile/web/src`, comments
  stripped) and every `docs/**/*.md` — judged twice. Once against the phrasings this
  class has produced, sentence by sentence in documents so a wrapped line cannot hide
  one. Then against a phrase-independent SHAPE. The shape is a surface noun — a
  window, a console, a terminal — near one of the two remedy names the shipped
  phrasings carried. The two names are `approvals` and `retire and reopen`. In a
  document the shape additionally requires a spawner relation, which is what made the
  deleted rule false: a start, launch or open word taking a session or runtime as its
  object. The one exemption is THIS file, and it is checked rather than asserted: a
  mention here must sit beside a word that makes it history. **The exclusions are
  deliberate and they are part of the bound** (agent review round 10, MINOR-1): a bare
  verb meaning "to loosen" is not one of the two names, so copy pairing a surface noun
  with it is not caught, and ownership of the gate is excluded as a relation because
  this PR's own repaired sentence uses that word. The concrete counterexample is
  recorded in the cell's own docstring rather than quoted here. What the cells do
  catch is every wording this class actually shipped — each is in a mutation battery
  — and, in a document, any re-statement of the spawner rule. A claim outside those
  shapes is the phrase list's job; an un-qualified claim in this file still fails;
- **the phone's card component renders the refusal body raw.** The sentence it
  now receives is the CARD's — the question survived, and a deny works from there
  — but `pending-card.tsx`'s `humanizeError` has no arm for it, so the copy is
  shown as sent rather than as a card-shaped message. That file is in THIS
  repository — the portal is `local_operator/mobile/web` (`src/pending-card.tsx`)
  — so the remainder is in scope for stage D rather than a cross-repo errand
  (UX review round 3, U11's remainder);
- **a relay dial that has ENDED leaves ``SessionEntry.authority_bearing`` set**
  until the next dial replaces the connection. Pre-existing and outside this
  delta, and no user-visible consequence has been produced from it — the next
  request is written on the next dial, which resets both fields — but it cost a
  reviewer one false reading, so it is recorded (QA review round 3, Q7);
- **D5's per-row "why" is deferred.** `auto` is tinted because this connection
  may not loosen THIS session and `default auto` because it is machine-locality
  and is refused from every surface, including a console — two different reasons
  behind one tint, distinguishable only by spending the keystroke. The refusal
  explains each; the list has no room at 60 or 44 columns (UX review round 3,
  U15).
- **REVISION 2, and the two that matter most:**
  - **`file-only` hosts have no presence gate.** Linux (and any macOS host whose
    Secure Enclave refuses) stores the private half as a 0600 file, so ANY process
    running as the operator — including the model's own `bash` child — can sign
    for them without a gesture, and the invariant is not enforced there. The
    level is reported (`operator-file-only`) and `lop operator status` prints what
    it means; it is not papered over with a claimed boundary. TPM+PIN sealing is
    the named follow-up;
  - **the prompt itself is an attack surface.** A model can TRIGGER a presence
    prompt; only a human can answer it, but prompts can be spammed and misread.
    Mitigations that ARE in place: one signature per action (the memo is popped by
    the frame that uses it), copy that names the session and the effect, and a
    per-connection challenge cap. Prompt-rate limiting beyond that is not
    implemented;
  - **`root`/`sudo` defeats any userland scheme**, exactly as it did before;
  - **the harness install tree stays user-writable**
    (`~/.local/share/uv/tools/local-operator`). The anchor is root-owned, but the
    VERIFIER lives in a tree the same uid can write, so a subject that can rewrite
    the installed package can bypass verification entirely. Pre-existing, and the
    Stage-2 confining epic is where it is fixed — recorded here because the anchor
    work makes the claim look stronger than the install does;
  - **availability is not protected**: an attacker can delete the operator's key,
    delete device certificates, or flood prompts. That is denial rather than
    escalation, and it is stated rather than implied;
  - **a tool child CAN shell out to `lop operator sign`.** It gets a real prompt
    naming the session and the effect, and a signature only if a human answers —
    which is the boundary working as designed, recorded because it looks alarming
    and is not;
  - **the phone's ordinary login is still a password this machine stores**
    (`mobile/auth.py`). Moving it to a device challenge-response is the natural
    next ticket, and stage D's device certificate is the groundwork for it rather
    than a replacement for it.

## 5. Staged plan

- **Stage 1** — the class, the seam, the in-memory capability, the handoff.
- **Stage 1b** — the guarantee-level report, the daemon's
  HTTP-boundary drop, and the not-at-rest probes. **Superseded by revision 2 for
  the AUTHORITY model** — the class, the seam, the op set, the typed refusal and
  the proof machinery all survive it; what revision 2 replaces is what
  "authorised" means (the spawner becomes the operator).
- **Revision 2, stage A (this change)** — `local_operator/operator/`
  (`__init__`, `trust`, `keychain`, `verify`, `sign`), `lop operator
  init|trust|install|sign|status`, `operator_authority_level()` absorbing
  `operator_cap_guarantee()`, and §2 of this document.
- **Revision 2, stage B (this change)** — the wire and the seam: the
  `operator_challenge` op, the three optional fields, the domain-separated
  length-prefixed message, `admit_increasing` consulting every source,
  `operator-signature-v1` in the record's `capabilities`, and the refusal copy
  that stops offering the retire-and-reopen remedy.
- **Revision 2, stage C (this change)** — the local surfaces: the one signing
  entry point, and `AttachClient` presenting a signature for an
  authority-increasing frame when it holds no spawn capability — which is the
  path the TUI's attached pane and the desktop backend both take.
- **Revision 2, stage D (this change, landed)** — the phone: `local_operator/
  operator/devices.py` (the certificate store plus the pairing handshake), `lop
  pair`, the relay's `POST /api/pair` + `GET /api/pair/<device_id>` +
  `POST /api/sessions/<id>/operator/challenge`, the narrowed body scrub, the
  relay declaring its paired certificate on its auth frame (which is what widens
  `_connection_may_loosen` for a phone), and the portal's signing UI in
  `local_operator/mobile/web` (`lib/operator-device.ts`, `screens/pair.tsx`,
  `components/gate-sheet.tsx`, and the pending card's signed retry). The
  fabricated relay-spawn cell was DELETED and replaced by a real one: the old
  test called `remember_operator_cap` itself, so it proved a hand-built state
  rather than the phone, and no production path ever registers a capability for a
  relay-spawned runtime (now pinned as a source fact).
- **Revision 2, stage E (this change, landed)** — the exec supervisor:
  `--supervisor-fd N`, the argv serialization with its refusal for
  `--background`, `deliver_operator_cap_to` + `SupervisorCapChannel` in
  `harness/approval.py`, and the supervisor-side `remember_operator_cap` keyed by
  the pid the endpoint line prints. The takeover route
  (`--resume <own live session> --control --supervisor-fd`) was PROBED rather
  than reasoned about: the run exits 1 on the session LEASE's refusal
  (`session <id> is already open in another process`) and the supervisor receives
  no capability, because the run never wrote.
- **Round 6 remediation (this change, landed)** — the four review streams' MAJORs,
  all of them about a claim that outran the code rather than about the mechanism:
  a revocation that never reached a running runtime; the deleted remedy still
  shipping on the `/approvals` REPORT; the pre-install state refused with remedies
  that cannot run and no mention of `lop operator install`; the prompt copy wired
  at no production site; the phone's loosen receipt set inside the sheet that then
  unmounted; and Windows graded `strong` while `create()` raises. Each one is
  fixed at the source and pinned by the cell named beside it in this document.
- **Revision 2, stage F (partly this change)** — the docs, the surface table, the
  residual and the refusal copy, all updated here: the conservative branch of
  `approvals_default_notice` stopped naming "the window that started it" (a
  window a background-started runtime does not have — the user-visible regression
  this redesign deletes) and names the three levers that work from anywhere.
  STILL OUTSTANDING as its own round: TPM+PIN sealing on Linux, prompt rate
  limiting, and the operator's design/UX review rounds for the portal screens.
- **Stage 2 (unchanged)** — an OS boundary confining the model-code spawn sites,
  so the residuals above stop being reachable by construction: macOS
  `sandbox-exec`, Linux Landlock/`bwrap`, Windows restricted token plus a deny
  ACE. It is also where the user-writable install tree stops being a bypass.
- **Stage 3 (unchanged)** — a device-bound credential for the phone's ordinary
  login, retiring the stored portal password.
