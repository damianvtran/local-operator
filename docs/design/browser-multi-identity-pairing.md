# Browser bridge: multiple extension identities paired at once

Status: **DESIGN PROPOSAL — no implementation.** Written against `origin/main`
at `99e81a1c53502effaf350309f1f964d30b09019a` (the merge of PR #996, which
rewrote this exact handshake/authorisation path). Every line number below is
that tree. Change classification: **C2** — standard feature iteration inside an
existing subsystem, pre-authorised, no RFC. It is not C3 (no new product,
platform or page) and not C5 (it narrows nothing foundational; §4 argues it is
a net security *improvement* over the alternative the operator would otherwise
reach for).

Companion documents: the contract is `docs/design/browser-extension.md` §6; the
operational playbook is `guide://browser`; the repo rules are `AGENTS.md`.

---

## 0. The answer, up front

> *Can we be paired to both at once, can config be shared, and at what cost?*

| Question | Answer |
|---|---|
| Can two extension identities be **authorised** at once? | **Yes.** Cheap and low-risk — one file-format change plus one predicate. §3. |
| Can both be **connected** at once? | **Yes, but only one may drive.** A standby role is not a nicety here: allowing two identities *without* one would produce a 1 Hz reconnect war between the two installs. §2 proves the mechanism. |
| Can both **drive** at once? | **No, and it must stay no.** Two installs cannot share a surface handle (§5.3), and letting both drive re-opens the exact hazard `daemon.py:1105` was written to prevent. |
| Can **config** be shared? | **Partly, and the useful half is already shared.** Pairing authority, port and discovery state are daemon-side and shared for free. Per-origin grants are extension-side *by design* and must NOT be shared — §6 explains why sharing them would break the property §6.3 of the contract exists to provide. |
| Does the user's actual pain go away? | **Only if the dev build gets a manifest `key`.** Allow-listing an ID does nothing for a path-derived ID that changes when the directory moves (§4.1 measures it). The allow-list is necessary; the `key` is what makes it sufficient. |
| Cost | ~1 file format revision, ~1 new daemon concept (`links` instead of `link`), 3 new user-visible surfaces, one extension version bump, **no `PROTO_VERSION` bump** (§7 — a bump would hard-refuse the released store build). |

**The smallest change that solves the stated problem** is §3 (allow-list) +
§4.2 (dev manifest `key`) + §2 (standby, which the allow-list *forces* on us) —
in that order of importance. §6 (sharing origin grants) is the part I recommend
**not** building.

---

## 1. What the code does today, verified

### 1.1 The hard rejection

`local_operator/browser_bridge/daemon.py:1100-1103`:

```python
saved = _read_json(_pairing_path(self.root))
if saved and saved.get("extension_id") != extension_id:
    await websocket.close(code=4004)
    return
```

One saved record, one pinned ID, everything else refused before the link is
even installed. Downstream of it, two more places make the single-ID assumption
load-bearing:

- `_valid_saved_token()` (`daemon.py:1018-1023`) returns `False` unless
  `saved["extension_id"] == extension_id`, so the stored `token_sha256` is
  bound to one identity.
- `_live_pairing_matches()` (`daemon.py:866-876`) compares the on-disk ID to
  `self.link.extension_id` and severs a live socket when they diverge, polled
  every `REVOKE_WATCH_S = 3.0` s (`daemon.py:123`, `daemon.py:918-937`).

The record itself is written at `daemon.py:1063-1070`:
`{extension_id, token_sha256, paired_at}` into `browser/pairing.json`
(`PAIRING_FILENAME`, `daemon.py:126`), 0600 under a 0700 dir
(`_private_write`, `daemon.py:142-148`).

### 1.2 Proof, on a real daemon

Run against an **isolated** daemon — random port, `LOCAL_OPERATOR_CONFIG_DIR`
redirected to `/tmp/mid-probe/cfg`. Port 4099, `~/.local-operator`, the
operator's Chrome and the live pairing file were never touched.

```
# isolated daemon on 127.0.0.1:54576
1. store id, no pairing file      -> ack {'event': 'hello_ack', 'proto': 1, 'paired': False}
2. pair with code                 -> ok=True token_len=43
3. pairing.json keys              -> ['extension_id', 'paired_at', 'token_sha256']
                                     id= omibaecbjdhgbbcedbnnnmjpmopfheof
4. unpacked id, store paired      -> closed 4004      <-- the reported defect
5. store id again w/ token        -> ack {..., 'paired': True}
   incumbent socket               -> closed, code= 4000
6. hello with an EXTRA field      -> closed 4001      <-- see §7
7. store id, wrong token          -> ack {..., 'paired': False}
```

Rows 4 and 6 are the two facts the rest of this document is built on. Row 5 is
"a later extension wins" (`daemon.py:1105-1155`) working as designed.

### 1.3 A second, unreported defect the multi-ID world would expose

The pending-pairing record is a **single slot keyed by one ID**
(`PENDING_FILENAME`, `daemon.py:127`; written by `_rotate_pending`,
`daemon.py:981-998`). `_ensure_pending` (`daemon.py:1000-1016`) reuses a
pending code only when `pending["extension_id"] == extension_id`, and otherwise
rotates — destroying the other identity's live code. Measured on the isolated
daemon:

```
A. store code 765294 -> after unpacked dial, pending is for UNPACKED code 362304
   store's code still valid?   False
B. store pairs with its old code -> ok=False
   "That code didn't match. Codes expire after two minutes — check the app."
```

Today this is unreachable (§1.1 refuses the second identity before
`_ensure_pending` runs). The moment the allow-list lands, two unpaired installs
racing to pair steal each other's codes and the user meets a mismatch error for
a code they read correctly. **Any implementation of §3 must also do §3.4.**

### 1.4 Why "allow both and let later-wins sort it out" is actively wrong

`worker.ts:408-441` (`teardown`) treats close code 4000 as *not* a loss of
connectivity — it suppresses the `connState` write and comments that
"the worker has to re-dial with the new token" — but it still calls
`scheduleReconnect()` at `worker.ts:440`. And `worker.ts:389` resets
`attempt = 0` on every successful `onopen`. With `backoffDelayMs(0) === 1_000`
(`reconnect.ts:63-65`), two installs that are both authorised and both dial
produce:

> A attaches → B attaches, evicting A with 4000 → A's `attempt` was reset to 0,
> so A re-dials after 1 s, evicting B → B re-dials after 1 s → …

A stable ~1 Hz mutual-eviction war, indefinitely, with `forget_link_state()`
(`daemon.py:502-520`) failing every in-flight future on each flip. That is
strictly worse than today's honest 4004. **Standby is therefore a requirement
of the allow-list, not an enhancement of it** — this is the single most
important structural finding in this note.

### 1.5 Blast radius of touching `ExtensionLink`

`self.link.<attr>` appears **106 times** across `daemon.py`, spanning 25
distinct attributes (19 × `.websocket`, 11 × `.paired`, 8 × `.extension_id`,
7 × `.awaiting_origin`, …). PR #996 threaded generation fencing through most of
them (`is_authoritative`, `daemon.py:473-482`; the "one uninterrupted block"
install at `daemon.py:1124-1148`; the send-side `wire=` discipline at
`daemon.py:484-496`). §5 keeps every one of those invariants by construction.

---

## 2. Decision 1 — Simultaneous connections: STANDBY, with a deterministic rule

### 2.1 The rule

> **Later-wins WITHIN an identity. Incumbency ACROSS identities.**

- A new socket whose `extension_id` equals the current **driver's** replaces the
  driver, exactly as today (`daemon.py:1105-1155`, unchanged). This is what
  reconnect-after-worker-death depends on, what the popup's own pairing socket
  depends on (`reconnect.ts:49-64` documents the eviction as load-bearing), and
  what #996 hardened.
- A new socket from a **different** authorised identity is accepted and told
  `role: "standby"`. It receives no `Request` frames. It does not evict.
- **Paired outranks unpaired.** (Added in remediation round 1 — review finding M1
  — and recorded here because the lines above did not carry it.) A dial that
  presents a valid token takes the wheel from an incumbent that is connected but
  *unpaired*: the cold-start rule below lets such an incumbent hold it, and every
  session would otherwise answer `not_paired` while an authorised, paired install
  stood waiting. The reverse is impossible — an unpaired dial never evicts a
  paired driver, which is decision 4. It is the same refinement
  `_promote_standby` and `POST /driver` already applied, that the wheel only goes
  to a link that can serve a command; the handshake was the third place it holds.
- Cold-start tie-break: **first to complete `hello` drives.** Deliberately not a
  configured priority — a priority list makes the preferred install evict the
  incumbent on *every* one of its reconnects, re-creating §1.4's war at the
  alarm period instead of at 1 Hz. Self-stable beats "correct but flapping".
  Interaction with the rule above: a token-less dial can still win a cold start,
  but its win is revocable by the first paired dial.
- Escape hatch: `lop browser drive <id-or-label>` pins a driver explicitly
  (§8.2). One command, for when the operator cares which one is driving.

**Rejected alternative — refuse the second identity with a typed error**
(`already_driving`) instead of accepting it as standby. It is smaller and it
does prevent the war. I reject it because it does not deliver what was asked
for: the operator explicitly wants to be "paired to both at once", and a refused
socket means the standby install's popup shows a disconnect state and its worker
retries forever on the alarm floor. Standby costs one extra card in the popup
and gives the operator the thing they asked for, with a real failover story.

### 2.2 Failover

When the driver's link ends — peer close (`daemon.py:1280-1291`), a
drop-for-silence (`_drop_unproven_link`, `daemon.py:787-858`), or a per-ID
revoke — promote the **longest-attached** surviving standby, publish, and send
it `{event: "role", role: "driver"}`. Promotion runs inside the same "no await
between decision and publish" discipline as `daemon.py:1124-1148`.

The promotion is *not* free for sessions holding a handle. See §5.3.

### 2.3 The failure mode where two installs collide

They cannot hold the same handle: a surface handle is `bridge:<tabId>:<nonce>`
(`state.ts:335-340`) and the map lives in each install's own
`chrome.storage.session` (`state.ts:70-77`), so install B's exact-token lookup
(`resolveSurfaceToken`, `state.ts:350-356`) can never resolve install A's
handle. A session presenting a stale handle to a newly-promoted driver gets the
already-handled typed refusal `tab_closed` / "the browser tab handle is stale"
(`cdp.ts:33-35`).

The real collision is **orphaned tabs**. A demoted install keeps its surfaces in
session storage and its `chrome.debugger` attachments live (`attached` set,
`cdp.ts:96,114`), so the user keeps seeing "Local Operator is debugging this
browser" banners on tabs that nothing can ever close — `close` only reaches the
driver. Both installs attaching to the *same numeric tab* would surface as
`debugger_conflict` via the foreign-attachment probe (`cdp.ts:121-133`,
`ownAttachment` at `cdp.ts:157-176`), which is the honest error but a confusing
one here.

**Mitigation, and the reason the extension must change:** on receiving
`role: "standby"`, the worker detaches its debugger sessions and clears its
surface map. A standby install holds nothing. This is the whole of the
extension-side work and it is what forces the version bump in §9.

---

## 3. Decision 2 — Authorization: an allow-list with ONE TOKEN PER ID

### 3.1 The format

Keep the legacy top-level keys verbatim and add a list:

```json
{
  "extension_id":  "omibaecbjdhgbbcedbnnnmjpmopfheof",
  "token_sha256":  "<sha256 of the store build's token>",
  "paired_at":     1757000000.0,

  "schema": 2,
  "identities": [
    {"extension_id": "omibaec…", "token_sha256": "<…>", "paired_at": 1757000000.0,
     "label": "Chrome Web Store build", "last_seen_at": 1757600000.0},
    {"extension_id": "jbadjea…", "token_sha256": "<…>", "paired_at": 1757500000.0,
     "label": "unpacked dev build",     "last_seen_at": 1757600100.0}
  ]
}
```

The top-level trio is the **downgrade contract**, not redundancy: it names
whichever identity is currently driving, so an older daemon (or an older `lop`)
reading this file still works. Verified by running today's code against a
hand-written schema-2 file:

```
today's pairing_status(): {'paired': True, 'extension_id': 'omibaec…'}
today's _valid_saved_token(store, tok)    : True
today's _valid_saved_token(unpacked, tok) : False     # correctly ignores what it cannot parse
today's _live_pairing_matches(store)      : True
today's reset_pairing removed file        : True
```

Zero migration in the other direction too: a legacy `{extension_id,
token_sha256, paired_at}` file is read as a one-entry allow-list. **An
already-paired operator is never asked to re-pair**, which is a hard
requirement of this work.

### 3.2 Per-ID tokens, not one shared token

**Rejected alternative — one shared token accepted from any allow-listed ID.**
It is simpler and it is what the request literally suggests ("share the
config"). I reject it on one argument that survives scrutiny and one that does
not; both are stated honestly because the difference matters for review.

*The argument that does not survive:* "a shared token widens the blast radius
of a token leak." Against a **local** attacker it barely does. A local process
that can read `browser/pairing.json` can equally read `run/browser/bridge.json`
and take `session_key`, which is the credential for `POST /rpc`
(`daemon.py:1294-1296`) — the leg that actually drives the browser. It does not
need a pairing token at all. Both files are 0600 under 0700
(`_private_write`, `daemon.py:142-148`; `state.py:80-96`), and
`docs/design/browser-extension.md` §6 threat 3 already concedes this case. The
Origin header is likewise no defence against a local process: it is whatever
the client sends, and `_origin_extension_id` (`daemon.py:969-979`) only
validates its *shape* (32 chars from `a`–`p`). I could forge it from a shell in
one line, and the probe in §1.2 did exactly that.

*The argument that does survive:* **revocation must be real per identity.** With
a shared token, "revoke the dev build" removes an ID from the allow-list while
the revoked install still holds a working credential on disk. The only thing
stopping its re-entry is the Origin check — which the paragraph above just
established is not a security boundary against anything running locally. And
the dev build is loaded unpacked from a directory any local process can write,
which is precisely threat 2 (rogue extension) of the contract's §6. With
per-ID tokens, revoking an ID deletes the only hash that could have accepted
its secret, and revocation is a fact about the file rather than a hope about a
header. Per-ID tokens cost one nested dict lookup. Take them.

### 3.3 The two predicates that change

```python
def _identities(root) -> list[dict]:
    """Every authorised identity, legacy record included. One reader, so the
    schema-1 upgrade path exists in exactly one place."""
    saved = _read_json(_pairing_path(root))
    if not saved:
        return []
    listed = saved.get("identities")
    if isinstance(listed, list) and listed:
        return [entry for entry in listed if isinstance(entry, dict)]
    return [saved]          # schema 1: the record IS the sole identity
```

- `daemon.py:1100-1103` becomes "close 4004 unless `extension_id` appears in
  `_identities()`" — still refusing before `attach()`, so the unbounded-close
  rule stated at `daemon.py:1078-1085` is preserved unchanged.
  **IMPLEMENTATION NOTE (coder, 2026-09-12): read literally, this sentence
  reproduces the transcript §1.2 records as the reported defect — its row 4 is
  "unpacked id, store paired → closed 4004" — and it contradicts §0 ("can two
  identities be authorised at once? Yes"), §3.4 ("two unpaired installs racing to
  pair"), §8.1 (a code per waiting install) and §9.3's happy cell ("both paired
  through the real popup form"). Once the store build is paired, every later
  install IS unlisted, so a literal gate leaves it with no code to enter and it
  can never be added. Measured on the isolated rig before the change below: 97
  `/extension` accepts while the second install dialled and was refused, no
  pending code for it, `/health` authorised list unchanged.**
  **ROUND 3 SUPERSESSION (coder, 2026-09-13): the token branch is gone, because
  it was a DEAD END rather than a defence.** A revoked install keeps the token it
  was issued (`worker.ts` does not clear it on the 4003 close), so it re-dialled
  with it, was closed before `attach()`, and therefore never reached
  `_ensure_pending`: no code was ever minted for it, `lop browser pair` answered
  "already paired … use --reset", and the pairing form its own popup showed could
  not be completed by any code — the only escapes were `--reset` (which revokes
  the working install too) or Settings → unpair. Measured (design review D1 / UX
  U3): unlisted id **with** a token → closed 4004 with `pairing_status().pending`
  empty and the popup painting *"Could not reach Local Operator on this machine."*
  about a daemon it was fetching `/health` from; the same id **without** a token →
  admitted, `hello_ack{paired:false}`, code minted. Since an unlisted id cannot be
  authorised either way (authority comes from the FILE's entry for THIS id, and
  `_valid_saved_token` selects by id), refusing a token that could never match
  bought nothing that admitting it does not, and it cost the user the only route
  back. **As implemented: every unlisted dial is admitted as an ASKER** —
  `hello_ack{paired:false}`, its own code minted, refused every RPC until that
  code is entered — which is the rule the rest of the daemon already followed and
  the one §6.2's "every identity goes through the code dance" assumes. The
  unbounded-close rule is untouched for the refusals that remain (no usable
  ORIGIN, a malformed `hello`, a proto mismatch, a superseded socket).
  Pinned by `test_u4_*`: admission + code + no authority for the token case,
  re-pairing with the dead token for a revoked install (`test_u4d`), and a row
  that the presented token still buys nothing (`_valid_saved_token` False for the
  third id while the store build's own token stays valid for it).
- `_valid_saved_token()` (`daemon.py:1018-1023`) selects the entry by ID and
  `compare_digest`s that entry's hash. Keep `secrets.compare_digest`; keep
  returning `False` on an empty token.
- `_live_pairing_matches()` (`daemon.py:866-876`) becomes "is *this link's* ID
  still listed", evaluated per link.

### 3.4 The pending-code fix (required, see §1.3)

`PENDING_FILENAME` becomes a map keyed by extension ID, each entry retaining
`{code, expires_at, attempts}`. `_rotate_pending` / `_ensure_pending` /
`_try_pair` (`daemon.py:981-1075`) operate on one entry. `PAIR_TTL_S = 120`,
`PAIR_MAX_ATTEMPTS = 5` and the rotate-on-exhaustion lockout (finding A1, whose
reasoning is at `daemon.py:1034-1041`) are per-entry and otherwise untouched.
`lop browser pair` then prints one line per waiting identity, naming which
install each code belongs to.

### 3.5 Two invariants that are easy to get wrong

1. **`pair` and `unpair` must work on a STANDBY link.** The popup opens its own
   socket to submit the code (`popup.ts:650-690`), and after §2's rule that
   socket is a standby link whenever another identity is driving. If the `pair`
   handler (`daemon.py:1203-1214`) requires driver status, pairing the second
   install becomes impossible — the flow would deadlock on itself.
2. **`unpair` must be per-identity.** The options page sends `{event:"unpair"}`
   on its own socket (`options.ts:154-180`, handled at `daemon.py:1235-1240`)
   and today calls `revoke()`, which does `reset_pairing()` — deleting the whole
   file. In the multi-ID world it must remove **only the sending link's own
   identity** and sever only that link. `revoke()` splits into
   `revoke_identity(extension_id)` and the all-identities `revoke_all()` behind
   `pair --reset`. The audit-A1 re-check at `daemon.py:902-914` (a handshake
   installing itself while the close is in flight) applies unchanged to the
   per-identity form.

---

## 4. Decision 3 — Dev-mode widenings and what they actually cost

### 4.1 The user's pain is the path, not the pin

A Chromium unpacked ID is the first 16 bytes of the SHA-256 of the absolute
directory path, nibble-mapped to `a`–`p`. Three plausible locations for the same
build:

```
jbadjeaodkoboanppmpjiifpconegdcj  /Users/damian/local-operator/extension/dist
odohekclkjcllncpbifladmkaakoigdp  …/local-operator-worktrees/bridge-multi-id/extension/dist
gdmnboeijgcnngijgdnilkhmdaanmneg  /tmp/lo-ext/dist
```

Three identities for one artifact. Since this repo is worked through many
concurrent worktrees (`AGENTS.md`, "Environment"), **an allow-list alone leaves
the operator re-pairing on every worktree** — a smaller pain than today's, but
the same pain. This is why §3 is necessary and not sufficient.

### 4.2 A dev manifest `key` — recommended

`manifest.json`'s `key` field is the base64 DER of an RSA **public** key; the ID
is derived from it and **no path enters the hash** (Chrome's manifest-key
reference; derivation reproduced locally against a throwaway keypair, which
produced a stable ID for a key that was never written into the repo).

Recommendation: generate one dev keypair, commit **only the public `key`**, and
ship it in a dev-only manifest overlay that `build.mjs` applies when *not*
`--zip` (`build.mjs:14` already branches on `isStore`, and `build.mjs:29`
already copies the manifest — this is a few lines at an existing seam). The
store build must keep shipping no `key`, since the store assigns the published
identity.

**Is a committed public key acceptable in a public repo?** Yes, and the threat
model is worth stating precisely rather than waving at:

- A `key` is a **public identity, not a secret**. The private `.pem` is never
  committed and is not needed to load unpacked.
- Anyone can copy that key and build an extension that claims the dev ID. That
  buys them nothing: they still have to get past §6.2 of the contract, where the
  6-digit code flows **terminal → browser**. The attacker's extension can dial
  and be told `paired: false`; it cannot learn a code only the operator's
  terminal printed. This is the property that already defends the store ID,
  which is equally public.
- Therefore: commit the key, and **never auto-trust it.** A well-known ID that
  the daemon allows without pairing would be a genuine hole — that is what
  turns a public identity into a credential.

### 4.3 "Trust locally-loaded unpacked extensions" — rejected as not implementable

The daemon's only evidence about the peer is the Origin header
(`_origin_extension_id`, `daemon.py:969-979`). There is **no** field in the
handshake, and no property of the WebSocket, that distinguishes an unpacked
install from a packed one — `Hello` carries only `proto`, `token`,
`extension_version`, `browser` (`protocol.py:224-229`). So "trust unpacked IDs"
can only be implemented as "trust any ID", i.e. delete the Origin pin and defeat
threat 2 of the contract outright. Reject on feasibility before security.

*(A future daemon could read Chrome's own `Preferences`/`Secure Preferences` to
learn which IDs are unpacked. That is a profile-location guessing game across
Chrome/Edge/Arc/Brave, it reads a file the browser owns, and it would still be a
local-file assertion. Not worth it; named so nobody rediscovers it as clever.)*

### 4.4 Uncertainty I could not settle, and the experiment that would

**Can Chrome load an unpacked build carrying a dev `key` while the Web Store
build of the same extension is installed?** They are different IDs, so I expect
yes; but Chrome has historically refused some duplicate-identity loads, and I
have no browser in this session. **Experiment:** in a throwaway Chrome profile
(`--user-data-dir` under `/tmp`, `--remote-debugging-port=0`, per `AGENTS.md`
"Capturing a browser surface"), install the store build, then Load-unpacked a
dist carrying the dev `key`, and read both IDs off `chrome://extensions`.
Fifteen minutes with the `bridge-rig` harness
(`docs/design/browser-bridge-realchrome-rig.md` on `feat/bridge-realchrome-rig`,
which already builds scratch extension copies from distinct directories for
exactly this reason — see its `attach-refusal` scenario). **If it refuses, §3
still lands and the dev build simply keeps its path-derived ID, with the
allow-list making the re-pair a per-worktree one-off instead of a per-switch
one.** The design does not depend on this answer.

---

## 5. Decision 4 — Implementation shape inside the daemon

### 5.1 `link` → `links`, with the driver named

`ExtensionLink` (`daemon.py:244-520`) is per-connection state and already
carries everything a standby needs (socket, generation, `extension_id`,
`last_frame_at`, the unproven-drop latch). The change is one level up:

```python
self.links: dict[int, ExtensionLink] = {}   # keyed by generation
self.driver_generation: int = 0

@property
def link(self) -> ExtensionLink:
    """The DRIVING link, or a null link when nothing drives.

    Retained as a property so the ~106 existing `self.link.*` call sites keep
    meaning what they meant — every one of them is about the link that serves
    commands — and the diff stays reviewable instead of touching all of them.
    """
```

This is deliberate minimalism: `rpc()` (`daemon.py:1293-1385`), `_admit`,
`_complete`, `/health`, `repair`, `publish` and the whole lock topology keep
reading `self.link` and keep being correct, because the driver *is* the link
they were always about. Standby links are served only by the handshake and the
receive loop.

Generation stays **global and monotonic** (`attach`, `daemon.py:467-471`) so
`is_authoritative` (`daemon.py:473-482`) remains a total order over every socket
the daemon has ever installed. `is_authoritative` gains the "and this link is
the driver" meaning by construction: `self.link` resolves to the driver, so a
standby's `(socket, generation)` never matches it. The A1 fences added by #996
therefore keep holding without being rewritten.

### 5.2 What `publish()` and `/health` say

`publish()` (`daemon.py:557-575`) keeps writing the **driver's** ID into
`BridgeState.extension_id` (`state.py:37`). That preserves the one consumer that
matters: `tools/builtin.py:9433` reads `current.extension_id` as "this bridge
was paired once", which drives the demotion diagnostic at
`builtin.py:8194-8210`. No change there, deliberately.

`/health` (`daemon.py:1918-1977`) gains **additive optional** fields only — the
file already establishes that pattern and the reason (`daemon.py:1936-1942`:
HTTP, not the WS protocol, so an old client ignores what it does not know):

```json
"driver_extension_id": "omibaec…",
"standby_extension_ids": ["jbadjea…"],
"authorized_extension_ids": ["omibaec…", "jbadjea…"]
```

`extension_connected`, `paired`, `extension_unresponsive`, `link_attached` and
`link_silent_s` keep describing the **driver**, because every existing reader —
the popup's wedge card (`popup.ts:518-546`), `lop browser status`
(`cli.py:1891-1902`), `backend._health_ok` — is asking about the link that
serves commands.

### 5.3 The cost of a failover, stated plainly

A session's `BrowserResource` persists `surface_id` — a handle minted by the
*previous driver* (`resources.py:284`, `resources.py:304-311`). After a failover
the new driver cannot resolve it (§2.3), so the next command returns
`tab_closed` / "the browser tab handle is stale", and `owner_recover`
(`ownership.ts:112-141`) answers `state: "unresolved"` because the new install's
`ownerScopes` has no such proof. The session then re-opens a fresh tab — which
is the *already-designed* behaviour for "a full browser restart destroys
session-storage authority" (`resources.py:280-284`).

So failover behaves exactly like a browser restart from the session's point of
view. That is an acceptable cost and it needs **no new code** — but it must be
said out loud in the standby copy (§8.3), because the user will see a new tab
appear and deserves to know why.

---

## 6. Decision 5 — What "share the config" can and cannot mean

| Config | Where it lives | Shared across identities? |
|---|---|---|
| Pairing authority (allow-list) | `browser/pairing.json`, daemon-side | **Yes**, by §3 — this is the ask |
| Daemon port, session key, discovery | `run/browser/bridge.json` (`state.py:28-52`) | **Yes**, already |
| Pairing token | `chrome.storage.local` per install (`state.ts:58-60`) | **No** — one per ID by §3.2 |
| Daemon port *setting* | `chrome.storage.local.port`, per install (`options.ts:125-130`) | **No** — retyped once per install, trivial |
| **Per-origin grants** | `origins` / `hostGrants` / `siteGrants` / `allowAllSites`, per install (`state.ts:58-68`) | **No — and deliberately not** |

The last row is the one that will disappoint, so here is the argument rather
than the verdict. `docs/design/browser-extension.md` §6 threat 3 says the
per-origin allowlist lives **in the extension** because "the prompt renders in
browser UI that no local process can click", so "the agent opened the user's
bank" always passed through a human click on that machine's screen. A shared
grant store is necessarily a *daemon-side* store — a file a local process can
write. Syncing grants between installs through the daemon therefore hands a
local process the ability to pre-grant origins, which is the exact capability
§6.3 exists to deny. It would trade a real security property for the
convenience of not re-approving a handful of sites on a dev build.

**Recommendation: do not build it.** Re-approving a few origins on the dev
install is cheap, it is the security property working, and the user only meets
it once per install rather than once per switch. If it later proves genuinely
painful, the right shape is an **options-page export/import** — a human
clicking "export" in one browser and "import" in the other keeps the human in
the loop and never gives a local process a writable grant store. Note it as
future work, not as scope here.

---

## 7. Decision 6 — Protocol compatibility: NO `PROTO_VERSION` bump

### 7.1 The asymmetry, verified

`WireModel` sets `extra="forbid"` (`protocol.py:199-202`), so the daemon
rejects unknown fields on models it validates. The extension does **not**
validate: `worker.ts:395` is `JSON.parse(String(message.data)) as DaemonMessage`
— an unchecked TypeScript cast with no runtime schema. Measured:

```
Hello + extra field -> REJECTED  ['extra_forbidden']     # and on the wire: close 4001
HelloAck dump        -> {'event': 'hello_ack', 'proto': 1, 'paired': True}
```

That gives three rules, and they are not symmetric:

1. **Daemon → extension: additive fields are SAFE.** New `HelloAck` fields (and
   a new `role` frame) are ignored by the store 0.1.10 build.
2. **Extension → daemon: additive fields on an EXISTING model are FATAL.** A new
   extension that added a field to `Hello` would be closed 4001 by every already
   released daemon. **Do not add fields to `Hello`.**
3. **Extension → daemon: an entirely NEW event is SAFE.** The receive loop
   dispatches on `frame.get("event")` string equality (`daemon.py:1203-1259`)
   and falls through to `Response.model_validate`, which raises
   `ValidationError` and `continue`s (`daemon.py:1261-1264`). An old daemon
   silently ignores an unknown event rather than closing.

### 7.2 The compat matrix

| daemon \ extension | store 0.1.10 (old) | new build |
|---|---|---|
| **released daemon (old)** | today's behaviour | works; extension sees no `role` in `HelloAck` and **must default to driver**, i.e. behave exactly as today |
| **new daemon** | works; single identity, always the driver, ignores the extra ack fields | full multi-identity + standby |

### 7.3 The bump would be the breakage

`daemon.py:1097-1099` closes 4001 on any `hello.proto != PROTO_VERSION`, and
`worker.ts:424` renders 4001 as `connState: "incompatible"` — the popup's
"update needed" card that **pairing cannot fix**. Bumping `PROTO_VERSION` from 1
(`protocol.py:16`) would hard-refuse the published store build the operator is
running right now. **No bump.** Signal the daemon's capability with an additive
`HelloAck` field instead:

```python
class HelloAck(WireModel):
    event: Literal["hello_ack"] = "hello_ack"
    proto: int = PROTO_VERSION
    paired: bool
    role: Literal["driver", "standby"] = "driver"   # absent ⇒ driver, for old daemons
    authorized_count: int = 1
```

plus a new daemon→extension `Role` frame for a live promotion/demotion, added to
`DaemonMessage` in `gen_ts.py:78`. `protocol.gen.ts` is generated
(`gen_ts.py:26-79`, checked by `--check` at `gen_ts.py:87-98` and in the
extension CI workflow), so the TS side follows automatically — **regenerate it
in the same commit or CI fails.**

---

## 8. Decision 7 — CLI, status and popup (USER-VISIBLE — schedule the rounds)

> **Everything in this section is user-visible.** Per the team's review gate
> that means a **designer** round (`### Design review`, D-findings) and a
> **copy-reviewer** round on the rendered strings; §8.3 additionally changes an
> interaction flow, so it needs a **ux-reviewer** round (U-findings). I am
> deliberately specifying *what must be knowable*, not pixels or final wording.

### 8.1 `lop browser pair`

- `pair` (no flags): unchanged in spirit — shows the live code(s). With more
  than one identity waiting it must name **which install** each code belongs to
  (§3.4), because a user staring at two popups cannot otherwise tell.
- `pair --list`: the authorised identities, their labels, when each was paired
  and last seen, and which is driving.
- **Round 3 (UX U2 / design D3): the label is NOT sufficient on its own.**
  `_browser_label` is `<browser> extension <version>`, so two installs of the
  SAME build — two profiles loading one unpacked build, or any two builds at one
  version — are labelled byte-identically, and every surface that addresses an
  identity by label then has no token that resolves it. The id prefix is the one
  thing that always does, so it is printed beside the label everywhere an
  identity is named: `status`/`pair --list` rows (`- label (cmadnonj…) driving`),
  `--list` adds `paired <when>, last seen <when>` from the file (§8.1's promise,
  and the only remaining field that differs when labels collide), the popup's
  driver line (`Chrome extension 0.1.13 (cmadnonj…) is driving right now.`), and
  the `waiting:` rows of `pair` (`009319  (Chrome extension 0.1.13 · omibaecb…)`).
  The `note:` about a pre-0.1.13 standby is also gated on the **standby's** own
  build (from `/health`'s `standby_labels`) rather than printed for any standby,
  and it names the symptom the user sees (copy review C4 / UX U7).
- `pair --revoke <id-or-label>`: removes one identity and severs only its link.
- `pair --reset`: unchanged meaning — revoke **everything**. Must keep its
  UX-N1 property of exiting 0 when nothing is waiting (`cli.py:2005-2019`).
- **No `pair --allow <id>`.** Adding an identity keeps going through the code
  dance, so §6.2's "the secret flows terminal → browser" property holds for
  every identity rather than just the first. A bare `--allow` is one social-
  engineering step away from authorising a rogue extension, and it saves the
  operator about four seconds.

### 8.2 `lop browser drive <id-or-label>`

Pins the driver explicitly and demotes the incumbent. The escape hatch from
§2.1. Small, honest, and the thing an operator reaches for when both installs
are up and the wrong one has the wheel.

**Round 3 (copy review C1 / UX U2): the failure has to say WHICH failure.**
Asked for a label that two installs share, the command answered "no connected
 extension matches '…'" — i.e. *nothing* matched, about a target two installs
matched — and listed bare id prefixes with no labels, so it did not teach the
token that would have worked. It now says **"no single connected extension
matches"** (the wording `--revoke` already used for the same refusal) and lists
`id  label` per candidate. Separately, an install that is **authorised but not
connected** — the state a handover leaves behind, where `status` says "paired,
not connected" — is answered `409 not_connected` with "that install is authorised
but not connected right now — open its browser, then retry" instead of a
not-found that reads as a typo'd id (UX U4).

### 8.3 `lop browser status`

Today's output (`cli.py:1891-1937`) never prints an extension ID at all. It must
now answer, without the user running anything else: **which identities are
authorised, which one is driving, and which is on standby.** Keep the existing
`extension connected` / `paired` / driving-tabs lines meaning what they mean
(the driver, §5.2) and add an identities block. The `extension_unresponsive`
discriminator and its two tenses (`cli.py:1917-1932`) are hard-won #996 copy —
do not disturb them.

### 8.4 The extension popup

One new state alongside `connected` / `paired` / `pairing` / `disconnected` /
`incompatible` / `unresponsive` (`popup.ts:22-42`): **standby**. It must convey
that this browser *is* paired and *is* connected but is not the one taking
commands right now, name the other install if the daemon reported it, and say
what happens when the driver goes away (§5.3: the session gets a fresh tab —
this is the part users will otherwise file as a bug). The popup card is pinned
per state with measured heights (`popup.ts:215-241`), so a new state needs its
own pin measured the same way, not a guess.

---

## 9. Test plan (coder-executable)

### 9.1 Unit — `tests/unit/browser_bridge/`

Follow the existing file conventions (`test_daemon.py`, and the
round-scoped `test_remediation_round*.py` pattern). Every daemon test uses an
isolated `root=tmp_path`; none may touch 4099 or `~/.local-operator`.

| # | Assertion |
|---|---|
| U1 | A legacy `{extension_id, token_sha256, paired_at}` file authorises that ID and **is not rewritten on read** — zero-migration. |
| U2 | A schema-2 file authorises both IDs; each token validates only against its own ID. |
| U3 | Today's `pairing_status()` / `_valid_saved_token` / `_live_pairing_matches` still work against a schema-2 file (the downgrade contract — reproduce the §3.1 transcript as a test). |
| U4 | An unlisted ID is closed **4004 before `attach()`** (assert no link state was created). |
| U5 | Revoking one identity severs only its link; the other keeps driving and its token still validates. |
| U6 | `pair --reset` / `revoke_all` still removes everything. |
| U7 | Second identity connects → `HelloAck.role == "standby"`; an `rpc()` while it is standby is served by the **driver** (assert the frame went to the driver's socket, by identity). |
| U8 | Driver disconnects → longest-attached standby is promoted, gets a `role` frame, and `publish()` now names its ID. |
| U9 | Same-ID reconnect **replaces** the driver (4000 on the incumbent) and does **not** demote it to standby — §2.1's within-identity rule. |
| U10 | Two unpaired identities each get their own live code; neither rotates the other's (§1.3 — assert §1.3's transcript is no longer reproducible). |
| U11 | `pair` and `unpair` are served on a **standby** link (§3.5). |
| U12 | `unpair` from install B removes only B from the file. |
| U13 | `Hello` with an extra field is still 4001 (pins rule 2 of §7.1 so nobody "helpfully" relaxes `extra="forbid"`). |
| U14 | An unknown extension→daemon event is ignored, not fatal (pins rule 3). |
| U15 | `PROTO_VERSION == 1` and a store-shaped `hello` with `proto: 1` is accepted (pins §7.3). |
| U16 | The #996 fences still hold with several links: a superseded socket's frames stamp no liveness, resolve no future, and publish no driven record. Mutate `is_authoritative` to prove each can fail. |

**Prove each guard can fail** (`AGENTS.md`, "Prove the test can still fail"):
mutate the fix in place, watch the row go red, revert. Record the table in the
PR as #996 did.

### 9.2 Extension — `extension/tests/*.mjs`

Mirror `worker-pairing-eviction.integration.test.mjs` and
`worker-wire-generation.integration.test.mjs`:

| # | Assertion |
|---|---|
| E1 | `HelloAck` **without** `role` ⇒ the worker behaves as driver (old-daemon compat). |
| E2 | `role: "standby"` ⇒ the worker detaches its debugger sessions and clears its surface map (§2.3). |
| E3 | A live `role: "driver"` promotion re-arms the worker without a reconnect. |
| E4 | The popup renders the standby card for a standby `/health`, and the pinned height matches the measured value. |

### 9.3 The real two-identity end-to-end proof — **this is the evidence that counts**

A green unit suite proves nothing here. The rig already exists for this shape of
problem: `scripts/bridge_rig.py` +
`docs/design/browser-bridge-realchrome-rig.md` on `feat/bridge-realchrome-rig`
(not yet on `main` — land it or cherry-pick it into the test worktree). Its
`attach-refusal` scenario **already builds a second extension copy in a distinct
directory to get a distinct ID**, which is exactly the second identity needed
here. Its isolation guarantees are non-negotiable and already asserted: daemon
on 4599 with its own config root, headless Chrome on a `mktemp` profile with
`--remote-debugging-port=0`, `_assert_no_production_port` refusing any build
that still contains `4099`, and `teardown: ALL CHECKS PASSED`.

Run against **one** isolated daemon with **two** real installs — a
store-*shaped* build (a committed `key` giving a fixed ID; the real store build
cannot be side-loaded) and a path-derived unpacked build:

| Cell | What must be shown |
|---|---|
| **Happy** | Both paired through the **real popup form**; `status` lists two identities, one driving; a real `open`/`read`/`screenshot` succeeds on the driver. |
| **No war** | Both installs connected for ≥120 s with `/health` sampled throughout: exactly one driver the whole time, **zero** 4000 evictions after settle. This is the §1.4 regression and it must be sampled, not assumed. |
| **Standby cannot drive** | A session command while B is standby is served by A; B's debugger attachments are gone; B's popup shows the standby card (screenshot). |
| **Failover** | Kill A's worker; B is promoted within one observed interval; a session command on a **stale handle** returns typed `tab_closed`, and the following `open` succeeds on a fresh tab (§5.3). Capture the timing. |
| **Unknown ID** | A third build in a third directory is **admitted as an asker** — with or without a token, since neither can be authorised (`§3.3`, round 3): `hello_ack{paired:false}`, its own code minted, no authority until that code is entered, nothing in the daemon's link state changed beyond a bounded unlisted link. **The pre-round-3 `4004` for the token-bearing half is gone** — measured as the revoked-install dead end that made re-pairing impossible without a storage wipe. |
| **Revoked ID** | `pair --revoke B` severs B's link within `REVOKE_WATCH_S`; A keeps driving and keeps answering; B's popup shows the pairing form. |
| **Token mismatch** | B with a corrupted stored token gets `paired: false`, is offered a code, and cannot issue RPCs (`not_paired`). |
| **Old extension, new daemon** | The **released 0.1.10** build against the new daemon: pairs, drives, ignores the additive ack fields. Load the published store build in the rig profile for this one. |
| **New extension, old daemon** | The new build against a daemon built from the pre-change commit: pairs and drives as a single identity, no 4001. |

Capture commands and their **actual** responses, the `/health` samples, and
popup screenshots for every user-visible state (standby, two-identity pairing,
post-revoke). Reproduce the §1.4 war on a deliberately broken build (allow-list
without standby) and show it gone on the real head — that is the "reproduce
before fixing" evidence for the most important structural claim in this note.

### 9.4 Gates (`AGENTS.md`)

```sh
.venv/bin/python -m flake8 .
uvx --from black==26.1.0 black --check .
uvx isort==5.13.2 --check .
.venv/bin/python -m pyright --pythonpath .venv/bin/python .
.venv/bin/python -m pytest tests/unit -q
cd extension && npx tsc --noEmit && node --test tests/*.test.mjs && node build.mjs
.venv/bin/python -m local_operator.browser_bridge.gen_ts --check
.venv/bin/python -m pytest tests/e2e/test_browser_ownership.py -m e2e -n0 -q
```

Never `.venv/bin/black` / `flake8` / `isort` / `pyright` directly
(`AGENTS.md:172-183`).

---

## 10. Rollout and rollback

- **Runtime + extension, together.** The daemon change alone is inert for the
  user's pain: the standby card, the demotion-detach and the `role` handling are
  extension-side. Both in **one PR**.
- **Extension version bump is mandatory and belongs in this PR.** `AGENTS.md`
  ("Releasing the browser extension"): the version in `extension/manifest.json`
  **and** `extension/package.json` tracks extension code and must be bumped in
  the same PR that changes extension behaviour, so every submitted version
  identifies exactly one tree. Both files currently read **0.1.11** on
  `origin/main` (bumped by #996, not yet submitted — the store is on 0.1.10). If
  0.1.11 has **not** been submitted when this lands, it may ride 0.1.11; if it
  has, bump to 0.1.12. **Check the store's current submitted version before
  choosing** — this is a decision that must be made against reality at merge
  time, not inherited from this document.
- **Runtime release is the combined release, and this PR does not bump
  `pyproject.toml`** (`AGENTS.md:449+`; `version-bump-guard` fails a feature PR
  that does).
- **Store submission is a separate, later step** on its own two-phase workflow
  (`chrome-web-store.yml` → `publish`), gated on Google's review queue. Plan for
  the new daemon to run against the **old** store build for days or weeks: §7.2's
  matrix is not a formality, it is the steady state during rollout.
- **Rollback.** The daemon is trivially revertible — an older daemon reading a
  schema-2 file authorises the driving identity from the top-level keys (§3.1,
  verified), so a downgrade degrades to single-identity rather than breaking
  pairing. No forced re-pair in either direction. The extension cannot be rolled
  back quickly (store review), which is one more reason the extension-side
  change must be small: `role` handling, demotion detach, one popup card.

### Risks to watch during rollout

1. **The reconnect war (§1.4)** is the one that can escape unit tests, because
   it needs two real workers with real alarms. Watch for repeated close-4000
   pairs in `lop browser logs`; that signature *is* the regression.
2. **Orphaned tabs from a demoted install (§2.3)** if the detach-on-standby path
   fails or the install is running an old build that has no `role` handling at
   all. The user-visible tell is a debugger banner on a tab nothing can close.
   `lop browser status --repair` (`daemon.py:1979-2048`) reconciles the
   *daemon's* records but cannot reach a standby install's tabs — a gap worth
   naming in the PR rather than discovering in an incident.
3. **Pending-code confusion (§1.3)** if §3.4 is skipped or half-done.
4. **The `link` property (§5.1)** silently changing meaning for a call site that
   was about "any connection" rather than "the driving connection". I believe
   all 106 are about the driver; the reviewer should spot-check the ones inside
   `_drop_unproven_link` (`daemon.py:787-858`) and `repair`, where the
   distinction is subtlest.

---

## 11. Decisions, numbered

1. **Allow-list several identities in `pairing.json`, one entry per ID, keeping
   the legacy top-level keys as the driver's record.**
   *Rejected:* a separate `identities.json` beside the existing file — two files
   to keep consistent, two revocation paths, and no downgrade story.
   *Rejected:* rewriting the file into schema 2 on first read — a write on a
   read path, which `state.py:63-78` documents as the defect class that broke
   boot on a full disk.

2. **One token per identity, not one shared token.**
   *Rejected:* shared token. Simpler, and the blast-radius argument against it is
   weaker than it looks (a local attacker already has `session_key`) — but
   per-ID revocation must be a fact about the file rather than a hope about a
   spoofable Origin header, and per-ID tokens cost one dict lookup. §3.2.

3. **Adding an identity keeps going through the 6-digit code dance; no
   `pair --allow <id>`.**
   *Rejected:* CLI allow. Saves four seconds, discards the terminal→browser
   secret-flow property for every identity after the first.

4. **Accept later identities as STANDBY links that receive no commands, rather
   than refusing them with a typed error.**
   *Rejected:* typed refusal. Smaller, and it does prevent the war — but it does
   not deliver "paired to both at once", which is the request. §2.1.

5. **Driver rule: later-wins WITHIN an identity, incumbency ACROSS identities;
   cold-start tie-break is first-to-complete-`hello`; `lop browser drive` pins;
   and a PAIRED dial outranks an unpaired incumbent (§2.1, remediation round 1),
   so a token-less dial's cold-start win is revocable.**
   *Rejected:* configured source priority (store preferred over unpacked) — the
   preferred install would evict the incumbent on every reconnect, re-creating
   the war at the alarm period. *Rejected:* first-paired — a stable rule that
   makes the common dev case ("I just loaded the new build") do the wrong thing
   silently. §2.1.

6. **Failover promotes the longest-attached standby; a session's stale handle
   gets the existing typed `tab_closed` and re-opens.**
   *Rejected:* handing surfaces over between installs — the handle's nonce is the
   anti-guessing capability (`state.ts:350-356`) and transferring it daemon-side
   would hand one install another's capability. §5.3.

7. **Commit a dev-only manifest `key` so the unpacked build has a stable,
   path-independent ID; never auto-trust it.**
   *Rejected:* "trust locally-loaded unpacked IDs" — not implementable, since the
   handshake carries no packed/unpacked signal; it degrades to "trust any ID".
   *Rejected:* documenting the store key for dev use — same ID as the store
   build, so the two cannot coexist, which is the case the operator is in. §4.

8. **No `PROTO_VERSION` bump. Capability goes in additive `HelloAck` fields plus
   a new daemon→extension `role` frame; nothing is added to `Hello`.**
   *Rejected:* bumping to 2 — `daemon.py:1097` would close the released store
   build 4001, rendering the popup's unfixable "update needed" card. §7.

9. **Make the pending-pairing record per-ID.**
   *Rejected:* leaving it. It is invisible today only because the second identity
   is refused earlier; the allow-list makes it a live footgun. §1.3/§3.4.

10. **Do NOT share per-origin grants across installs.**
    *Rejected:* daemon-mediated grant sync — it moves allowlist authority to a
    file a local process can write, defeating the property
    `browser-extension.md` §6.3 exists to provide. Future work, if it ever
    matters, is a human-driven options-page export/import. §6.

11. **Keep `self.link` as a property resolving to the driver, rather than
    rewriting 106 call sites.**
    *Rejected:* threading an explicit link parameter through `rpc` / `_admit` /
    `_complete` / `/health` / `repair` — a far larger diff across the code #996
    has just fenced, for no behavioural gain, since every one of those sites is
    about the driving link. §5.1.

---

## 12. Implementation sketch for the coder

Order matters — each step is independently testable and the tree is shippable
after 1, 2 and 3 even if 4-6 slip.

1. **`daemon.py` — the allow-list.** Add `_identities(root)`; rewrite
   `_valid_saved_token` (`:1018`), the gate at `:1100-1103`, and
   `_live_pairing_matches` (`:866`) on top of it. Add
   `add_identity` / `revoke_identity` / `revoke_all` and teach
   `pairing_status()` (`:167`) to return `identities` **while keeping
   `extension_id`** for `install.py:908` and `state.py:37`. Tests U1-U6.
2. **`daemon.py` — per-ID pending.** Map form; rework `_rotate_pending`,
   `_ensure_pending`, `_try_pair` (`:981-1075`). Keep `PAIR_TTL_S`,
   `PAIR_MAX_ATTEMPTS` and the finding-A1 lockout semantics exactly. Test U10.
3. **`daemon.py` — `links` + driver.** `self.links`, `driver_generation`, the
   `link` property (§5.1), standby acceptance in `extension()` (`:1077`),
   promotion on driver loss, per-link `pair`/`unpair` (§3.5), additive `/health`
   fields (§5.2). **Do not move any statement inside the uninterrupted install
   block at `:1124-1148`** — re-read that comment before editing, it encodes
   audit A1. Tests U7-U9, U11-U12, U16.
4. **`protocol.py` + `gen_ts.py`.** `HelloAck.role` / `authorized_count`, a
   `Role` frame added to `DaemonMessage` (`gen_ts.py:78`). Run
   `python -m local_operator.browser_bridge.gen_ts` and **commit the regenerated
   `protocol.gen.ts`**. Tests U13-U15.
5. **`extension/`.** `worker.ts:398-403` handles `role`; add detach-and-clear on
   demotion (reuse `cdp.ts`'s `detach` and `state.ts`'s surface removal, do not
   write a second teardown path); the popup standby card with a measured pin;
   bump **both** `manifest.json` and `package.json`. Tests E1-E4.
6. **`cli.py` + `install.py`.** `pair --list/--revoke`, `drive`, the identities
   block in `status` (`cli.py:1879+`). Leave the `extension_unresponsive` copy
   alone.
7. **Evidence.** §9.3 on the rig, screenshots for every new user-visible state,
   the §1.4 war reproduced-then-fixed, and the proof-of-failure table.

Worktree hygiene: one worktree, reclaimed after merge; the rig's `down --purge`
after evidence capture (`AGENTS.md`, shared-machine hygiene).

---

## 13. Open questions that genuinely need the operator

1. **Dev `key` in the public repo — approve or not?** §4.2 argues it is a public
   identity and not a secret, and that it is the only thing that actually fixes
   the reported pain. It is still a deliberate act of committing a key to a
   public repository and it is the operator's call.
2. **Which extension version does this ride?** 0.1.11 is on `main` and not yet
   submitted. If it is submitted before this merges, bump to 0.1.12. Needs the
   store state at merge time (§10).
3. **`lop browser drive` — worth it, or is the automatic rule enough?** I
   recommend shipping it; it is ~20 lines and it is the difference between "the
   wrong install has the wheel and I must quit a browser" and one command. Say
   if you would rather keep the CLI surface smaller.
4. **Is failover-costs-you-a-tab acceptable?** §5.3 makes a driver change look
   like a browser restart to a session: the current tab is abandoned and a new
   one opens. Preserving the tab across installs is not possible without
   transferring a capability between them (decision 6), so the alternative is
   *refusing* to fail over. I recommend accepting the tab cost.

**Not blocking, already decided by me:** the standby-vs-refuse choice (§2.1),
per-ID tokens (§3.2), no `--allow` (§8.1), no `PROTO_VERSION` bump (§7), and not
sharing origin grants (§6). Each has its rejected alternative recorded in §11 so
a reviewer can overturn it on evidence rather than re-litigate it from scratch.
