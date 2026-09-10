# Long-term encrypted secret store (`lop secret`)

Status: proposal. Author: architect subagent. Date: 2026-09-08.

This designs the encrypted, long-lived credential store the operator asked for,
the inline `/credential` composer redaction that feeds it, and the four access
surfaces around it. It also states, without softening, what the design does
**not** protect against, because the honest answer to part of the ask is "that
cannot be fully achieved on a same-uid macOS box, and here is the strongest
approximation".

Every empirical claim below was measured on this machine (macOS 25.6.0, arm64,
CPython 3.14.3) with the spikes recorded in [Appendix A](#appendix-a--spikes).
Where a mechanism is named, it was run, not recalled.

---

## 1. The problem as I found it

### 1.1 What exists today

Three separate things are called "credentials" in this tree, and the new store
is a fourth. Keeping them distinct is load-bearing:

| Store | Lives in | Lifetime | Encrypted | Purpose |
|---|---|---|---|---|
| `CredentialManager` (`credentials.py:43`) | `~/.local-operator/credentials.env`, 0600, plaintext `KEY=VALUE` | forever | **no** | PROVIDER API keys (Anthropic, OpenAI…) read at boot by the model layer |
| `VariableStore` session credentials (`variables.py:305-361`) | process memory only | one session | n/a | a secret the agent must USE and must never READ |
| `VariableStore` variables (`variables.py:273-302`) | config / `.local-operator.env` / `LOCAL_OPERATOR_*` env | n/a | no | non-secret config, denylist-filtered |
| **new: secret store** | `~/.local-operator/secrets/` | forever | **yes** | operator secrets, agent-retrievable from bash and eval |

The session-credential path is well built and I am not replacing it. The
operator hands over a secret; it is trimmed and stored in
`VariableStore._credentials` (`variables.py:323`); the name only is advertised
through `credential_names()` (`builtin.py:5360`) and the system-prompt tail; the
value is injected into every `bash` child's environment (`builtin.py:1483-1492`);
and it is scrubbed on the way back out by two cooperating filters — the loop's
`redact_tool_result` choke point (`session.py:6420` → `variables.py:360`) for
model-visible text, and `_PipeRedactor` (`builtin.py:1298-1328`) for live stream
bytes, which deliberately holds back a possible credential suffix so a secret
split across two reads cannot leak.

**What is missing is only persistence and agent-initiated retrieval.** Nothing
survives the session, and the agent can never read a value back — by design.

### 1.2 The actual baseline, measured

The comparison that matters is not against a perfect store, it is against what
holds the operator's secrets *right now*:

```
~/.minerva/credentials/.env exists=True size=6664 mode=0o600
-> 58 KEY=VALUE lines readable by ANY same-uid process, right now.
*.json under ~/.minerva: 31   (service-account and secret-backup files, 0600)
```

Every one of those is plaintext, at a predictable path, matching a predictable
filename pattern. `grep -rl "API_KEY\|SECRET" ~` finds all of it in seconds.
That is the bar to clear.

### 1.3 The central question, answered honestly

> The operator wants secrets a "random script on the device" cannot sniff out,
> while lop agents retrieve them freely from bash and eval with no prompt.

A script running as the operator's UID can read any file that UID can read.
So an encrypted store plus a key file beside it is **pure obfuscation** — the
attacker reads both and decrypts. That much is true and I am not going to
dress it up.

But the naive framing hides a real asymmetry that the spikes confirm, and it
is the asymmetry the whole design stands on:

- **A same-uid process can read another process's ENVIRONMENT.** Verified:
  `ps eww -p <pid>` and `sysctl KERN_PROCARGS2` both returned the marker from a
  sibling process (spike 2). A capability token in the environment **is
  stealable**. This kills option (b) as stated.
- **A same-uid process CANNOT read another process's MEMORY without an
  authorization prompt.** Verified: `task_for_pid()` on a sibling returned
  `rc=5` (`KERN_FAILURE`), and `lldb -p` **hung** rather than attaching —
  `authd` logged `Validating session owner damian (501) for
  system.privilege.taskport.debug` and a `SecurityAgent` process appeared
  (spike 3, 5). Developer mode is disabled and the user is not in `_developer`.
  **A memory scrape puts a password dialog on the operator's screen.**

That asymmetry is the entire security budget available. A master key **in the
memory of a running daemon** is protected by a real kernel boundary; the same
key **in a file** is protected by nothing beyond the file mode already on
`.env`. Everything below follows from spending that budget well.

---

## 2. Evaluating the operator's three candidate mechanisms

### 2.1 (a) Broker daemon with peer authentication — **recommended, with a caveat**

What macOS actually gives a unix-socket server about its peer (spike 1, all
succeeded):

| Option | Result on this machine |
|---|---|
| `LOCAL_PEERCRED` (`SOL_LOCAL`, `xucred`) | uid=501, ngroups=16 — **uid only, no pid** |
| `LOCAL_PEERPID` | peer pid (21276) |
| `LOCAL_PEEREPID` | effective peer pid |
| `LOCAL_PEERTOKEN` | audit token: `(auid, euid, egid, ruid, rgid, pid, asid, **pidversion**)` |
| `proc_pidpath(pid)` | full executable path — `…/cpython-3.14.3-…/bin/python3.14` |
| `sysctl KERN_PROC_PID` | ppid (offset 560) and start time (offset **0**, not 8) |
| `csops(CS_OPS_CDHASH)` | **rc=-1, denied** — no code-signature check available |

So: uid, pid, executable path and full ancestry are readable without privilege;
**code signature is not**. Any design element depending on verifying the peer's
signature is off the table. I checked, rather than assuming it would work.

**Pid reuse is solved, not merely mitigated.** The audit token carries a
`pidversion` (34652033 in spike 1), a monotonic per-process generation, and
`kinfo_proc` carries a start time. The broker pins `(pid, pidversion,
start_time)` at *connect* time and never re-resolves a bare pid later. A
recycled pid presents a different generation and fails the match. Verified in
spike 7 and spike 9.

The full decision procedure works end to end (spike 9):

```
registered 'lop session' pid=28578 start=1788847125
[a] agent bash grandchild  -> 28580 SECRET_VALUE
[b] detached attacker script -> 28582 ppid=1 DENIED

broker decisions:
  pid=28580   ALLOW  descendant of session 28578 (start 1788847125 matches)
  pid=28582   DENY   no registered session ancestor within 1 hops
```

The attacker there is exactly the threat-model script: same uid, double-forked
and `setsid`'d so it reparents to launchd — the shape a script dropped by a bad
link actually has. It was denied while the agent's real `bash` grandchild was
allowed.

**The caveat, stated plainly:** a script that does *not* detach, and instead
runs as a child of some process that is itself a descendant of a lop session,
inherits that ancestry and is allowed. Ancestry authenticates *lineage*, not
*intent*. See §9.

> **Amended during PR 2 (review round 1): becoming a session must itself be
> authenticated, and `register` was not.** The decision procedure above assumes
> the set of registered sessions is trustworthy. As first implemented it was
> not: `register` was dispatched *before* the authorization gate, and because
> the ancestry walk yields the peer as the first element of its own chain, any
> process that registered itself became its own authorizing ancestor.
> Reproduced from a double-forked `setsid` process reparented to launchd —
> the shape this section marks DENIED — against an *unlocked passphrase-tier*
> broker: one 60-byte frame yielded both the plaintext secret and the 32-byte
> master key. Two fixes, each sufficient alone and both kept:
>
> 1. **`register` presents a ticket** — 32 bytes from the CSPRNG in the 0700
>    secrets directory, compared with `compare_digest`. Note honestly what this
>    is worth per tier: in `keyfile` mode the ticket sits beside `master.key`,
>    so it stops nobody who could not already decrypt the store, and §8 says so.
>    In `passphrase` mode the ticket is the only thing on disk, and holding it
>    grants standing to *ask* — not to decrypt.
> 2. **A peer is never its own authorizing ancestor.** A registered session is
>    authorized *as itself*, with its connect-time pin verified; an
>    unregistered peer gains nothing from heading its own chain.
>
> **No process-shape check was available as an alternative, and this was
> measured rather than assumed.** The obvious repairs — require the registrant
> to be a session leader, to have a controlling tty, to have a particular ppid
> — all fail against a same-uid attacker, which is the entire threat model
> here: a detached script reparents to launchd (`ppid=1`, exactly like a real
> detached session), calls `setsid()` to lead its own session, and can allocate
> its own pty with `pty.fork()`. Code-signature verification is unavailable
> (`csops` denied, above). A secret the attacker must *read from a 0700
> directory* is the only discriminator this design has.
>
> **In `passphrase` mode the ticket alone is deliberately not enough.** An
> attacker that can reach the socket can read the ticket file too, so
> registering there additionally requires lineage from a live session or from
> the terminal that proved knowledge of the passphrase by unlocking the broker
> (see §13's amendment). Caught by an adversarial test written for this fix,
> not by review — the first version of the fix was ticket-only and the detached
> attacker still walked in.

> **Amended during PR 2 (implementation).** Three corrections, each from
> executed measurement rather than review:
>
> 1. **The walk must include pid 1, not stop above it.** Spike 9's loop is
>    `while pid > 1`, which never examines pid 1 itself. That is invisible on
>    this machine, where no session is pid 1, and wrong inside a container
>    where a session frequently *is* pid 1 — measured in Docker, where a
>    legitimate descendant was DENIED until the bound was corrected.
> 2. **Ancestors are pinned by `p_uniqueid`, not by `pidversion`.** The audit
>    token's `pidversion` is only readable for the *connecting* peer; the
>    kernel exposes no way to read it for an arbitrary ancestor. Measured
>    instead: `proc_pidinfo(PROC_PIDUNIQIDENTIFIERINFO)` yields a monotonic,
>    never-recycled 64-bit `p_uniqueid` for any pid, plus `p_puniqueid` — the
>    parent's — which lets each hop of the walk VERIFY it links to the process
>    it is about to examine rather than trusting a bare ppid. `pidversion` is
>    still pinned for the peer itself; start time remains the Linux fallback.
> 3. **Linux gets full ancestry authentication, not a fail-closed stub.** §14
>    anticipated `SO_PEERCRED` yielding a pid and left the rest open. Measured
>    in Docker: `SO_PEERCRED` plus `/proc/<pid>/stat` (ppid and field-22 start
>    ticks) reproduces both spike-9 outcomes exactly — agent grandchild
>    ALLOWED, detached `setsid` script DENIED. So Linux is a peer platform
>    here. Only genuinely unsupported platforms fail closed, and they do so by
>    refusing everyone rather than degrading open.
>
> A fourth correction is about the socket rather than the walk: macOS
> `sockaddr_un.sun_path` is 104 bytes and `bind()` fails past it (measured: 103
> binds, 104 fails). The default config dir yields a 49-byte path, but
> `LOCAL_OPERATOR_CONFIG_DIR` is operator-controlled, so the socket relocates
> to a short private `TMPDIR` directory when the natural path does not fit.
> Only the rendezvous point moves; the key, database and audit log stay put.

### 2.2 (b) Capability tokens in the environment — **rejected on evidence**

Spike 2, the measurement that decides it:

```
[1] ps eww -p 21277: rc=0 MARKER_VISIBLE=True
[1b] ps -E -p 21277: MARKER_VISIBLE=True
[2] sysctl KERN_PROCARGS2: rc=0 bytes=3444 MARKER_VISIBLE=True
```

A same-uid process reads a sibling's environment three different ways. A
capability token injected into agent children's environment is readable by the
very attacker it is meant to exclude, and worse, it would be readable by any
`bash` command the agent runs on behalf of a *prompt-injected* instruction.

**The alternative the operator suggested — passing the capability over an
inherited FD — also fails, for a subtler reason** (spike 6, 7). It works
mechanically: a child handed a pre-connected socket on fd 3 talks to the broker
with no token and no path. But `LOCAL_PEERPID` then reports the **connector's**
pid, not the user's:

```
connector pid 27033 | hello pid=27033
connector exiting, grandchild 27034
    connector pid 27033 alive now? False
=> the pid the broker sees is the CONNECTING process, pinned at connect();
   the grandchild using the fd is invisible to LOCAL_PEERPID.
```

An inherited connection is therefore *anti*-authenticating: it launders the
identity of whoever ends up holding it, and it survives the death of the
process that was authenticated. **Each client must make its own connection**
so that each request is peer-authenticated on its own merits.

Conclusion: no ambient capability token, in env or on an fd. Authentication is
the connection itself.

### 2.3 (c) Where the master key lives at rest — **two-tier, default no prompt**

The socket is a real barrier: macOS **enforces** mode bits on a unix socket at
`connect()`. Verified (spike 7) — after `chmod 000`, even the owner got
`[Errno 13] Permission denied`. So 0600 excludes other uids outright, and
peer authentication handles the same-uid case.

But when nothing is running the key must live somewhere. Options:

| Option | Stops the targeted attacker? | Cost to operator |
|---|---|---|
| 0600 key file in a 0700 dir | **No** — a script reads it | none |
| Key split (file + machine-bound value) | No — both are same-uid readable | none |
| macOS keychain without prompt | No — same-uid ACL grants read | none |
| **Passphrase, entered once per boot, cached in broker memory** | **Yes, for at-rest theft** | one prompt per reboot |

This is where I disagree slightly with a strict reading of the operator's
"no admin gate" constraint, and I want to be explicit about it rather than
quietly picking one. The operator rejected *admin-gated and prompting* stores —
keychain dialogs on every access, `op` CLI round-trips. A **once-per-boot**
unlock is a different thing: it is not per-access, not admin, not a system
dialog, and the broker then serves every retrieval for the rest of the uptime
with no prompt at all. Agents never see it.

What it buys is large and specific: **without it, the entire design collapses
to obfuscation against an attacker who reads the key file.** With it, the
master key exists only in the memory of a process that `task_for_pid` cannot
open without raising a `SecurityAgent` prompt. That is the difference between
"a script decrypts the store" and "a script cannot decrypt the store without
the operator watching a password dialog appear for no reason".

**Recommendation: ship both, default to the no-prompt mode, make the upgrade
one command.**

- **Default (`keyfile` mode).** Master key in `~/.local-operator/secrets/
  master.key`, 0600, in a 0700 directory. Honest claim: *equivalent to today's
  `.env` against a targeted attacker, materially better against opportunistic
  malware* (§8). No prompt, no behaviour change, works headless, survives
  reboots and `launchd` restarts.
- **Opt-in (`passphrase` mode), `lop secret harden`.** Master key wrapped with
  scrypt over an operator passphrase. The broker prompts once per boot, caches
  the unwrapped key in memory, and serves everything after that silently. A
  session that starts while the broker is locked gets a clear "locked, run
  `lop secret unlock`" error rather than a hang.

scrypt cost, measured (spike 8), for picking the parameter:

```
scrypt n=2^14 r=8 p=1:    92.3 ms, ~16 MiB
scrypt n=2^15 r=8 p=1:   182.1 ms, ~32 MiB
scrypt n=2^17 r=8 p=1:   840.7 ms, ~128 MiB
```

Pick **n=2^15, r=8, p=1** (182 ms, 32 MiB): unnoticeable once per boot, and a
meaningful brute-force cost against a stolen wrapped key.

---

## 3. Crypto

**No new install.** `cryptography` 50.0.x is already present in every
environment, so this adds nothing to install time or Windows wheel risk — which
the dependency comments in `pyproject.toml` treat as a first-class concern. Do
not add `argon2-cffi` (not present) or `pynacl`.

> **Amended during PR 1.** It arrives only as a *transitive* of
> `pyjwt[crypto]>=2.10`, not as a direct dependency. A direct import belongs in
> a direct declaration — if that extra were ever narrowed or replaced, the
> transitive copy would vanish and `lop secret` would fail at import with
> nothing pointing at the cause. PR 1 therefore declares `cryptography>=42`
> explicitly in `pyproject.toml`. The install footprint is unchanged.

**Primitives:**

- **AEAD: AES-256-GCM** (`AESGCM` from `cryptography.hazmat.primitives.ciphers.
  aead`). Hardware-accelerated on arm64; the alternative, ChaCha20-Poly1305,
  buys nothing here.
- **Nonce: 12 random bytes per record, per write.** Never a counter, never
  reused across a re-seal. Every update generates a fresh nonce. At 96 bits
  random with a store of thousands of records, collision risk is negligible.
- **KDF for the passphrase mode: scrypt**, n=2^15, r=8, p=1, 32-byte output,
  16-byte random salt stored beside the wrapped key.
- **Subkey derivation: HKDF-SHA256** from the master key, `info` carrying the
  key generation. Verified distinct per generation (spike 8, `[5]`).

**Metadata is authenticated, and this is not optional.** The name and
description are bound into the AAD, so an attacker with write access to the DB
cannot rename a record or swap two records' ciphertexts to make a consumer
fetch the wrong secret under a trusted name. Both attacks were tested and both
fail closed:

```
[2] renaming the record fails decryption: InvalidTag (metadata is bound)
[3] swapping record ciphertexts fails:    InvalidTag (record id is bound)
```

**The AAD**, canonically encoded so it cannot be ambiguous:

```
AAD = b"lopsec\x00" || format_version(u8) || key_generation(u32be) || record_id || \x00
      || name_index || \x00 || kind
```

`record_id` is an immutable UUIDv4 assigned at creation. `name_index` is the
HMAC blind index of §4 — the *column*, not the cleartext label. `kind` is
`string` or `file` (§7).

> **Amended during PR 1 (implementation).** This section originally bound the
> cleartext `name` and `sha256(description)`. That is not implementable against
> §4's schema, where both fields are stored *encrypted inside the ciphertext*:
> a GCM tag covers the AAD, so the AAD must be known **before** decrypting, and
> `list`/`rotate` enumerate rows with no candidate name to reconstruct it from.
> The two sections contradicted each other. Binding the `name_index` column
> instead is non-circular and preserves every property this section claims:
> the blind index is a deterministic function of the name, so **renaming still
> fails closed**; `record_id` is still bound, so **ciphertext swaps still fail
> closed**; and the name and description remain authenticated by virtue of
> being sealed *inside* the AEAD rather than beside it. The implementation
> additionally re-derives the blind index from the decrypted name and rejects a
> mismatch, so the inner and outer copies of the name cannot drift apart. Both
> tamper cases are proven against a real on-disk database in
> `tests/unit/secrets/test_tamper.py`.

**Rotation.** `lop secret rotate` generates a new master key, re-seals every
record under an incremented `key_generation`, and writes the new DB by the
atomic path in §4. Records carry their generation, so a partially-rotated store
is still fully readable — the reader selects the subkey by the record's own
generation. The old key is held until the last record has moved, then wiped.

---

## 4. Storage

**SQLite in WAL mode**, at `~/.local-operator/secrets/store.db`, directory 0700,
file 0600.

**Concurrency is the deciding factor, not a hypothetical.** The operator runs
~10 lop sessions at once. Individual files per secret would need a lock protocol
invented here; SQLite already has one that works. Measured with 10 concurrent
readers doing 60 decrypts each against a live writer (spike 8):

```
[6] WAL: 10 concurrent readers x60 decrypts + 1 writer x60 commits in 21 ms
    reads completed: [60, 60, 60, 60, 60, 60, 60, 60, 60, 60] | errors: NONE
```

Zero errors, 21 ms. WAL gives readers that never block on the writer, which is
exactly the ~10-session shape.

**Metadata leakage is handled by encrypting metadata too.** The operator
correctly notes that names and descriptions are themselves sensitive — a store
listing `MINERVA_PROD_DB_PASSWORD` tells an attacker where to go next. So:

- `name` and `description` are stored **encrypted**, in the same record.
- The `name` column additionally holds a **blind index**: `HMAC-SHA256(
  hkdf(master, "name-index"), normalized_name)`, truncated to 16 bytes. That
  gives exact-match lookup by name without storing the name in the clear, and
  a `UNIQUE` constraint for free.
- `list` therefore requires the key — which is correct: listing what secrets
  exist is itself a privileged operation, and it goes through the same broker
  authentication as a retrieval.

Schema:

```sql
PRAGMA journal_mode=WAL;
CREATE TABLE meta(k TEXT PRIMARY KEY, v BLOB);       -- schema_version, key_generation,
                                                     -- key_fingerprint, kdf params
CREATE TABLE secrets(
  id            TEXT PRIMARY KEY,                     -- UUIDv4, immutable
  name_index    BLOB NOT NULL UNIQUE,                 -- HMAC blind index
  key_generation INTEGER NOT NULL,
  format_version INTEGER NOT NULL,                    -- record format, for §13 skew
  nonce         BLOB NOT NULL,
  ciphertext    BLOB NOT NULL,                        -- {value, name, description, kind}
  kind          TEXT NOT NULL,                        -- 'string' | 'file'
  created_at    REAL NOT NULL,
  updated_at    REAL NOT NULL,
  last_used_at  REAL
);
CREATE TABLE audit(
  ts REAL NOT NULL, event TEXT NOT NULL, secret_id TEXT,
  session_id TEXT, pid INTEGER, exe TEXT, outcome TEXT, prev_hash BLOB, hash BLOB
);
```

**Atomicity and durability.** Single-statement writes inside a transaction;
SQLite's WAL handles torn-write recovery, which is strictly better than the
write-temp-then-`os.replace` dance `credentials.py:79` has to do by hand for a
flat file. `PRAGMA synchronous=FULL` on the secrets table's connection: this
store is small and written rarely, so durability beats throughput.

---

## 5. The four access surfaces

### 5.1 The CLI — `lop secret` (primary surface)

`AGENTS.md`'s tool-surface footprint ladder is explicit that a skill + `bash` is
rung 2 and a new core tool is rung 5, "last resort". The CLI is therefore the
**primary** surface and carries the full verb set:

```
lop secret get NAME              # value to stdout, no trailing newline
lop secret set NAME [--description TEXT] [--kind string|file]   # value from stdin
lop secret list [--json]         # names + descriptions, never values
lop secret describe NAME
lop secret rm NAME
lop secret rotate | harden | unlock | status
lop secret file NAME -- CMD...   # materialise a file secret for CMD (§7)
lop secret run -- CMD...         # run CMD with named secrets in its env
```

**Usable inside `$( )` without the value reaching the transcript.** The whole
point:

```bash
curl -H "Authorization: Bearer $(lop secret get GITHUB_TOKEN)" https://api.github.com/user
```

The value crosses a pipe into `curl`'s argv inside the child; the model sees
only the command text it wrote. Interoperating with the existing redaction is
the part that needs care and is specified in §6.

**`lop secret set` reads the value from stdin, never argv.** argv is world-
readable via the same `KERN_PROCARGS2` path spike 2 used for the environment.
The CLI must refuse a `--value` flag; `eval_worker.py:225-266` already redacts
argv in error rendering precisely because argv leaks.

### 5.2 The agent tool — one `createIf`-gated tool

Rung 3 on the ladder. One tool, `secret`, with an `op` parameter
(`store|retrieve|list|describe|update|delete`), returning `None` from its
builder when the broker is unavailable — mirroring `build_wake_tool` and
`build_browser_tool`. Six separate tools would be six schemas in every cache
prefix; one gated tool with a verb parameter is a fraction of that.

**`retrieve` deliberately does not return the value to the model.** It returns
a receipt naming the secret and stating that the value is available as
`$NAME` in the next `bash` call and as `secrets["NAME"]` in `eval` — the same
inversion the session-credential path already implements
(`variables.py:32-40`). An agent that genuinely needs the bytes in-process uses
`eval` (§5.3), where they never enter the transcript.

`store` is the verb that makes persistence **a decision the agent makes**, as
the operator asked. Guidance (§10) tells agents to prefer this store over
`~/.minerva/credentials/.env`.

### 5.3 The eval runtime — a library, not injected variables

`eval` runs in a **separate worker process** (`tools/eval.py`, `eval_worker.py`).
Injecting every secret into that worker's globals at spawn would:

- put every secret in memory whether or not the cell wants one,
- make them visible to `list_variables`-style introspection of the namespace,
  and to any `print(globals())`,
- and require deciding *which* secrets to inject before knowing what the cell
  does.

Instead, a lazy accessor is pre-imported into the worker namespace:

```python
from local_operator.secrets import secrets
token = secrets["GITHUB_TOKEN"]      # connects to broker, authenticates, returns str
```

`secrets` is a lazy `Mapping`. Each `__getitem__` is one broker round trip and
one audit entry, so retrievals are logged at the granularity they actually
happen. The returned object is a `str` subclass whose `__repr__` and `__str__`
in a *display* context render `[redacted]`, so an accidental bare `token` at
the end of a cell does not paint the secret — while `+` concatenation and
`.encode()` still yield real bytes for actual use. It also registers the value
with the worker's redaction ledger (§6).

The eval worker is a descendant of the session process, so it authenticates by
ancestry exactly as `bash` does. No token, no injection.

### 5.4 The existing session path keeps working, and gains "persist this"

`/credential` and `ask secret=true` are unchanged in behaviour: still
memory-only by default, still injected into bash, still unreadable. What is
added is a **route**, not a replacement:

- `/credential --persist <KEY>` promotes an existing session credential into
  the long-term store.
- The inline paste flow (§7 of the operator's brief, specified below in §11)
  writes to the session store **and** offers persistence, with the agent
  deciding.
- `ask secret=true` gains an optional `persist: bool` on the question, and
  `_report_secret_answers` (`builtin.py:9854`) routes to the long-term store
  when set — it already has the shape for this, storing and reporting only the
  key name.

---

## 6. Redaction: how a retrieved value stays out of the transcript

This is the part most likely to be got wrong, because **the subprocess case is
genuinely not solvable by the existing filters alone**, and I would rather name
that than paper over it.

Three cases, three mechanisms:

**(1) Value retrieved by the agent tool or eval.** The session process knows the
value. It registers it with the `VariableStore`, which already backs both
`redact_tool_result` (`session.py:6420`) and `_redact_tool_text`
(`builtin.py:1354`). Nothing new is needed: a new `VariableStore.
register_redaction(value)` adds to the same `_credentials`-backed set that
`redact()` iterates (`variables.py:360`), without making the value readable —
it must go into a *separate* `_redactions` set that `redact()` reads but
`credential_env()` and `credential_names()` do not, so registering a value for
scrubbing never accidentally injects or advertises it.

**(2) Value used by a `bash` child via `$(lop secret get NAME)`.** The session
process never sees the bytes — that is the whole point of the `$( )` form — so
the filter has nothing to match on. **This is a real hole and it is closed at
the broker, not at the filter.** When the broker serves a retrieval to a peer it
has authenticated to session `S`, it notifies `S` over the same socket
(`retrieved: {id, value}` on the session's own registration connection). The
session registers the value with its redactor **before** the child's output can
be read, because the broker's reply to the child and the notification to the
session are both written before the child can print anything. The existing
`_PipeRedactor` (`builtin.py:1298`) then scrubs the stream bytes, including a
secret split across two reads, which it already handles.

There is a narrow race — a child that retrieves and prints within the same
microseconds as the notification crossing the socket. Mitigate by having the
broker **write the session notification first and wait for its ack before
replying to the retrieving child**. That makes the ordering an invariant rather
than a hope, at the cost of one extra round trip on retrieval (sub-millisecond
on a unix socket). Take that cost.

> **Amended during PR 2 (implementation): the invariant only became one when it
> started failing CLOSED.** As first written the broker waited 2 s for the ack
> and then **served the value anyway**, which made the word "invariant" false —
> a session that never acked (a wedged UI, or an attacker who simply chose not
> to ack) got the value into the transcript with nothing downstream able to
> scrub it, since the whole premise of this section is that a `$( )` value
> passes through no filter. Measured: `CHILD GOT VALUE … after 2.00s WITHOUT
> the session ever acking`.
>
> A descendant's retrieval is now **denied** when the owning session does not
> acknowledge within the timeout, with an error naming the wedged session. The
> availability objection that motivated serving-anyway is real but much
> narrower than it looks, and is answered by scope rather than by weakening the
> rule:
>
> - A session retrieving **its own** value never waits for an ack — that is
>   case (1) above, where the value lands in the session's own memory and it
>   registers the redaction directly. Requiring an ack from the process blocked
>   on the reply would deadlock the operator's own terminal, so it is excluded
>   by construction, not by timeout.
> - `lop secret get` typed at a prompt has no owning session, so there is no
>   notice to wait for and nothing to deny.
>
> What remains deniable is exactly the case the ordering exists for: an agent's
> child fetching a secret through a session that has stopped answering. A
> failed command there is recoverable and visible; a leaked credential in a
> transcript is neither.

**(3) Value in a subprocess the filter never sees at all** — e.g. the agent
pipes it to a file, or a background job started before registration. Not
solvable in general, and the guidance (§10) must say so: `lop secret get` is
for interpolating into a command, not for `echo`-ing. The `secret` tool's
description should carry that sentence, because tool descriptions are the only
guidance a model reliably reads.

---

## 7. File-shaped secrets, and `GOOGLE_APPLICATION_CREDENTIALS`

The 31 JSON files under `~/.minerva` (9 of them service-account keys per the
operator's brief) are not strings, and a consumer like google-auth needs a
**real path it can open, possibly more than once**.

I tested the three candidate mechanisms (spike 11):

| Mechanism | Result |
|---|---|
| FIFO | `fifo read 73 bytes; seek FAILS: UnsupportedOperation` — single-shot, not seekable. **Unsafe.** |
| Unlinked file via `/dev/fd/N` | Child reads it (`CHILD read 73 bytes`), sibling cannot (`Bad file descriptor`) — but **`open#1 49, open#2 0, open#3 0`: `/dev/fd/N` on macOS is a DUP with a shared offset, not a re-open.** Single-shot. **Unsafe for repeat readers.** |
| 0600 file in a 0700 private dir, unlinked after the command | `repeat opens: 49 49` — works, seekable, re-openable |

macOS `/dev/fd` semantics differ from Linux here, and this is exactly the kind
of thing that would have been wrong if assumed. **Use the third option:**

`lop secret file NAME -- CMD...` creates a per-invocation directory under
`$TMPDIR` with mode 0700, writes the decrypted content 0600, exports
`GOOGLE_APPLICATION_CREDENTIALS` (or a `--env-var` name) pointing at it, runs
`CMD`, and removes the file and directory in a `finally` — including on signal,
which needs an explicit handler, not just `atexit`.

Honest note for the guidance: the plaintext **is** on disk for the lifetime of
that command. It is 0700-dir-scoped, randomly named, and gone afterwards, which
is far better than a permanent well-known path — but it is not zero exposure,
and an attacker sampling the filesystem during that window finds it.

---

## 8. What each design actually stops — the honest comparison

Measured against today's plaintext `~/.minerva/credentials/.env` (58 keys).

> **Corrected during PR 2 (implementation), and this correction matters more
> than the rest of the table.** Two cells previously read "**Stopped**" for the
> default tier. Both were false, and a reviewer and QA independently *measured*
> them false: a detached `setsid` script is denied at the socket and then
> **served anyway** from the key file, `rc=0`, value on stdout. The rule that
> residual risk is never overclaimed applies to this table above all, because
> it is the comparison an operator reads to choose a tier.
>
> Stated plainly, replacing the two claims: **the default `keyfile` tier
> enforces no ancestry at all.** A denial there is a fallback, not a refusal —
> `access.py` reads the key file instead, deliberately (see §13's amendment),
> because a caller the broker just refused could read that file directly and
> refusing would only break the operator's own terminal. The broker's
> contribution in that tier is the audit trail and the §6 redaction notice,
> **not access control**. `lop secret harden` is the only tier where the
> ancestry boundary is load-bearing.
>
> **Corrected again in PR-2 review round 4, and this is the more general
> lesson.** Because the hardened tier is the only one where the gate is real,
> *deciding which tier is in force is itself a security decision* — and it was
> being made by the same predicate `status` uses to print the mode line. That
> predicate answers on what is ABSENT from disk, which is right for honesty and
> wrong for authorization: absence, and presence, are attacker-controlled. Any
> same-uid process could write 32 random bytes to `master.key`, and the broker
> would conclude it was in the keyfile tier and stop requiring lineage —
> measured 3/3 from a detached `setsid` process at ppid=1 against a genuinely
> hardened, unlocked store, which then registered and was served the unwrapped
> master key out of broker memory and decrypted every record. The planted key
> was junk, so this was never key theft; it was a **lie told to a predicate that
> only observed**.
>
> The rule that replaces it: **a display answer may observe, an authorization
> answer must validate.** The two are now separate functions — `key_mode()`
> reports the tier honestly (a store carrying both files says `keyfile`, so
> `status` warns and `harden` repairs it), while `key_of_record_is_plaintext()`
> decides the gate by checking the installed key against the fingerprint the
> database records for itself. That fingerprint is public by design and cannot
> be forged without the key it names, so a plant fails it and the store is
> correctly still treated as hardened. Both behaviours are pinned by tests, and
> the authorization one is mutation-tested: restoring the existence check turns
> it red.

| Attacker behaviour | Today | Encrypted DB + key file (default) | Broker + passphrase (opt-in) |
|---|---|---|---|
| `grep -r` / "find .env files" opportunistic malware | **Loses everything** | **Stopped** — ciphertext, names blind-indexed | **Stopped** |
| Script reading a known key file path | n/a | **Not stopped** — reads key + DB, decrypts | **Stopped** — key is not on disk unwrapped |
| Script camping on the socket | n/a | **Not stopped** — 0600 excludes other *uids* (spike 7), but a same-uid denial falls back to the key file | **Stopped** — denial is enforced; no key on disk to fall back to |
| Detached script (`setsid`, reparented to launchd) | Loses everything | **Not stopped** — denied at the socket, then served from the key file (measured: `rc=0`, value on stdout) | **Stopped** — denied at the socket and there is no fallback (measured) |
| Script dumping the broker's memory | n/a | Needs `task_for_pid` → **denied, rc=5**; `lldb` → **SecurityAgent prompt** (spikes 3, 5) | Same |
| Script that self-registers as a session over the socket | n/a | **Not stopped** — but irrelevant, it can read the key file anyway | **Stopped** — `register` needs the 0700 ticket *and*, in this tier, lineage from an unlocked terminal or a live session (PR-2 review R1), and *which tier applies* is decided by validating the installed key against the store's fingerprint rather than by observing that a file exists (R4-1) |
| Script that runs `lop secret get` itself | n/a | **Not stopped** (§9) | **Not stopped** while unlocked (§9.1, §9.4) |
| Script that rewrites the lop code it can write (`~/.local/bin/lop` is 0755 **writable**, spike 10) | n/a | **Not stopped** | **Not stopped** |
| Attacker with a backup copy of the store only | n/a | Stopped if the key file was not in the backup | **Stopped** |

The honest summary: **the default mode's real win is against opportunistic and
automated theft, which is the overwhelming majority of what "clicked a bad link"
produces** — it turns 58 plaintext keys at a predictable path into ciphertext
with blind-indexed names. What it does **not** do is stop a script that looks
for the key file, and it does not stop one at the socket either.

The passphrase mode is what converts "targeted attacker wins" into "targeted
attacker must run code that impersonates a lop session *while the broker is
unlocked*, or trip a visible authorization prompt". Between reboot and the
first `lop secret unlock`, an attacker with the whole disk gets nothing *from
this store* — whatever plaintext `.env` files have not been migrated yet are
of course still readable (review R11).

---

## 9. Residual risk

Plain sentences, for the operator to act on. **This design does not protect
against the following, and no design confined to one macOS user account can.**

1. **A script that simply runs `lop secret get` itself.** `lop`, `local-operator`
   and `lo` are all on `PATH` and executable by any process running as you
   (spike 10). If that script spawns its own lop session, it becomes a
   legitimate descendant and the broker authorizes it. Ancestry proves lineage,
   not intent. **This is the fundamental limit of the whole design**, and it is
   why the store is a large improvement over `.env` for automated malware and a
   modest one against an attacker who has specifically studied your setup.

2. **A script that modifies the lop runtime.** `~/.local/bin/lop` and the entire
   `~/.local/share/uv/tools/local-operator` tree are mode 0755 and **writable by
   your user** (spike 10). An attacker who patches the CLI, or drops a
   `DYLD_INSERT_LIBRARIES` shim into a lop process they launch (the interpreter
   is adhoc-signed, not hardened — spike 10), reads secrets as they are
   decrypted. Code integrity is a prerequisite this design assumes and cannot
   itself provide.

3. **Anything already in a session's memory.** Session credentials, and any
   secret an agent has retrieved this turn, are in the session process's RAM.
   Protected by the same `task_for_pid` boundary as the broker — meaningful, but
   it is the same single control.

4. **The terminal you unlock in is authorized for as long as it lives.**
   `lop secret unlock` records the shell it was typed in as an authorizing
   ancestor, because without it the operator's own `lop secret get` — which has
   no lop session above it — would be denied and the tier would be unusable.
   The consequence is wider than case 1 and is a *different* mechanism than
   case 1 describes: a process in that terminal does not need to run `lop` at
   all, it can speak to the broker socket directly and be authorized on lineage
   alone. Measured, not inferred:

   - **Served**, for the shell's whole life: direct children, subshells, `( )`,
     grandchildren, `xargs` and `nohup` children, a `make` invocation running an
     unrelated Makefile, and a *later, unrelated* command run in that terminal
     long after the unlock. Anything you run there can read every secret.
   - **Denied**: any other terminal, a fresh shell started after the granting
     shell exits, and anything that detaches (a `setsid` child orphans to
     launchd and leaves the lineage).

   The grant is in memory only, is pinned to the shell's process identity so a
   recycled pid inherits nothing, and dies with the shell or with the broker —
   `lop secret broker stop`/`restart` revokes it, and there is no separate
   relock verb. A session that registered itself *inside* the granted terminal
   is revoked with that terminal too, so the grant cannot be promoted into
   something that outlives the shell (QA Q8).

   So: once you unlock after a reboot the broker serves silently until it
   exits, and the practical boundary during that window is **the terminal**,
   not the process. Treat an unlocked terminal as holding the whole store. The
   passphrase protects the store **at rest**, not while you are using it.

5. **A file secret during its command.** §7 writes plaintext to a 0700 dir for
   the duration of one command. An attacker sampling the filesystem in that
   window gets it.

6. **The audit log is tamper-EVIDENT, not tamper-proof.** `chflags uappnd`
   works without sudo and does block truncation and unlink (verified: "truncate
   /overwrite BLOCKED", "unlink BLOCKED"), **but the owner can clear the flag**
   with `chflags nouappnd` (verified, rc=0). Combined with the hash chain (§12)
   an attacker cannot silently *edit* history, but they can remove the file and
   you will notice a gap rather than being fooled by a forgery. Do not describe
   this as an immutable log.

7. **Prompt injection.** An agent that reads a malicious web page and is
   convinced to `lop secret get` and exfiltrate is fully authorized. The store
   defends against processes, not against the agent's own judgement. This is
   arguably the *most likely* real path to loss, and it is unchanged by this
   work.

**One-sentence version for the operator:** *this makes your secrets invisible to
the automated "scan the disk for credentials" malware that a bad link actually
drops, and makes a targeted attacker either impersonate lop or trip a macOS
password prompt — but anything running as you that is willing to run `lop`
itself can still read them, and in the hardened tier anything at all running in
a terminal you have unlocked can read them, so it is a large and worthwhile
increase in cost, not a guarantee.*

That is a real improvement over 58 plaintext keys at a predictable path. It is
not a vault, and it should not be described as one.

---

## 10. Documentation: "credentials guidance"

A new section in `AGENTS.md` (harness-agnostic, `~/` paths only per the
operator's standing rules) plus the same content as a `skill://` for Minerva
sessions. It must state:

- **Prefer the lop store over plaintext `.env`** when storing anything new.
  Storing is the agent's decision.
- **From bash:** `$(lop secret get NAME)` interpolated directly into the command
  that needs it. Never `echo`, never assign-then-print, never into a file.
- **From eval:** `from local_operator.secrets import secrets; secrets["NAME"]`.
- **File secrets:** `lop secret file NAME -- CMD`.
- **What is logged** — every retrieval, with pid and executable.
- **The residual risk in one paragraph**, so an agent does not overstate the
  guarantee to the operator.

---

## 11. Inline `/credential` and the random-name scheme

The operator's UX ask: type `/credential` anywhere in the composer line, paste
the secret right after it, see it replaced by a bracketed marker in the style of
image and large-text pastes, then keep typing to describe it.

**Reuse the existing marker machinery, do not build a second one.**
`editor.py` already has exactly this: `_collapse_paste` (`editor.py:5107`)
issues `[Paste #N, {_paste_label(payload)}]`, stores the payload in
`self._attachments[index]`, and returns `marker + " "`. `Marked = Attachment |
PastedText` (`editor.py:619`) is a two-variant union whose docstring
explicitly warns against adding a parallel map — so add a **third variant,
`PastedCredential`**, to that same union rather than a `_credentials` dict
beside it. Everything keyed on the marker number (the counter, the chip, the
atomic-token gate, the aside stash, `EditorSubmitted`, `/reload`) is
payload-agnostic and keeps working.

Marker: `[Credential #1, 64 chars]` — `_paste_label` already produces
`"64 chars"` for a single-line payload, which is exactly right for a secret.

Two behaviours that must differ from `PastedText`:

- **The payload must never be spliced back into the submitted text.** Where
  `expand_pastes` re-inserts a `PastedText` payload at submit, a
  `PastedCredential` is *removed* from the outgoing text and its value routed
  to the store. The marker text itself stays, so the model sees
  `[Credential #1, 64 chars]` followed by the operator's description — which is
  precisely the message the agent needs.
- **It must never enter prompt history.** `_record_history` already strips
  paste citations (`editor.py:6546`) for a subtler reason; credentials must be
  stripped there unconditionally, and never restored by `_navigate_history`.

**Detection.** A `/credential` token anywhere in the line, followed by a paste
event, arms the next paste for capture. Because `consumes_prompt` and
`ArgumentMode` (`slash_commands.py:494-500`) only handle a *leading* slash
command, the inline form is an editor-level concern, not a slash-command one —
the existing `/credential` command keeps working unchanged for the leading case.

### 11.1 The TYPED capture

The paste path above shipped in v0.53.0 and was, for two releases, the **only**
path. Typing the secret — `/credential 12345`, the obvious human gesture —
produced no chip at all: the line fell through to the legacy `/credential <KEY>`
command with the secret as its **key argument**, so the app answered "Paste the
value for 12345" and the secret sat in the transcript in plaintext. No chip, no
error, no warning. That is a security defect rather than a missing convenience,
and it is what this section closes.

**The state machine.**

```
token typed or completed  ──▶ ARMED
ARMED + SPACE             ──▶ TYPING   (characters masked as typed)
TYPING + printable        ──▶ held out of the document, one mask cell painted
TYPING + Enter            ──▶ chip minted; composer keeps the draft
TYPING + Esc              ──▶ unredacted: the characters come back as text
ARMED + paste             ──▶ chip in place, instantly (unchanged)
```

**The space is the boundary, and it arms from both routes.** A hand-typed space
and the trailing space `_apply_command` inserts when a picker row is accepted
are the same character at the same offset, so the capture opens on the shared
edit funnel rather than at either call site — the two routes are identical *by
construction* instead of by two code paths agreeing. The space is a delimiter
and is **not** part of the secret, so `K` in `[Credential #N, K chars]` counts
the value alone.

**Two different Enters, separated by state, not by timing.** While the picker is
open Enter accepts the completion; once a capture is open Enter mints the chip.
Exactly one of those states holds at any instant, which is what makes them
impossible to confuse. Accepting the row therefore must not *run* the command —
`Editor.arms_a_capture` is the predicate that suppresses it, beside
`opens_a_list` and for the same reason.

**Enter ends the secret; it does not submit.** The gesture is specified as "hand
over a secret and then describe it", so an Enter that also sent the message
would make the description impossible to write. The operator presses Enter a
second time to send.

**The secret is held out of the document.** Characters typed while a capture is
open never enter `self.text`; they accumulate in `Editor._credential_typed`
while the buffer receives `CREDENTIAL_MASK_CHAR` cells. This is the whole safety
property, and it is why the mask is *not* a render-time effect over real
characters: every disclosure seam (history, the draft store, undo, `ctrl+o`,
the submit path) reads the buffer, so a document that never held the secret
closes all of them at once rather than one at a time.

**Esc unredacts rather than discards, and says so.** Retyping a secret from
memory is exactly what an operator cannot do, so the recoverable reading is the
only safe one. Esc also ends the arm: the operator has visibly backed out, so
the next paste must not still be swallowed. But the unredact is the one exit
that *ends with the secret in the buffer*, and the frame after it looks entirely
ordinary — the amber marker reverts, the masking notice goes — while the next
Enter re-commits the original leak; measured, the post-Esc-then-Enter outcome
was byte-identical to the pre-fix base. So it posts `CredentialUnredacted` and
the app warns that N characters are now plain text (design round 1, D1).

**Esc works on an EMPTY span too**, which took a mechanism rather than a
comment. The cancel's own `replace()` re-enters the edit funnel, which
re-derived the arm from a buffer still ending in `/credential ` and re-opened
the capture the cancel had just closed — so Esc was inert before any character
was typed, on the exact key the on-screen notice advertises, and the prose typed
next was captured as a secret. `_suspend_credential_sync` scopes the
re-derivation out across the widget's own edit, the same idiom
`_suspend_picker_sync` uses (review round 1, R1; UX round 1, U2).

**The masked span is positional — and so is the held value.** The span is open
exactly while the caret is inside it, asked on `watch_selection` — `move_cursor`,
a mouse click and an app-set selection all move the caret without a caret key
being pressed. Because the capture deliberately stays open *anywhere* in the
span ("the operator may be mid-word"), the interaction invites an edit inside it,
so `_credential_typed` is a **positional mirror** of the mask cells rather than
an append-only buffer: `Editor.edit` maps every insertion, deletion and
selection-replacement onto the held value at the same index. An append-only
value beside a caret-positioned cell is the one arrangement that fails
*silently* — the count stays right while the order goes wrong, so the chip's
length (documented as an integrity check) passes on a value that is wrong and
can never be displayed again to catch it. Measured before the fix: `ABCDEFGH`,
`←←`, `xy` stored `ABCDEFGHxy` for an intended `ABCDEFxyGH` (UX round 1, U1).

**The typed `/credential <KEY>` form is retired.** The space always opens a
masked capture, so a hand-typed key name is minted as a short secret rather than
reaching the `<KEY>` prompt. The command is still *parsed* — a pasted whole line
reaches it, and it stays the route a viewer session uses to hand a secret to its
runtime — but nothing advertises it as typable any more: `CREDENTIAL_USAGE` and
the `--persist` advice both name the inline gesture and the generated
`LOP_SECRET_` name instead. Usage text and behaviour have to agree (QA round 1,
Q1).

**The arm stays on its own token while that token is being typed out.** `/cred`
is a token and `/credential` is a token, but the five spellings between them are
not — so an operator re-arming on the same line (the state the now-working Esc
leaves them in) walked the latched arm through a window where it matched nothing
at its own anchor, and the nearest-match tie-break migrated it back onto the
FIRST token earlier in the line. The migration is one-way, because the anchor
moves with it: the arm never came home, the caret-at-span-end gate never fired,
and the secret typed next was painted in the clear and then parsed as a
credential NAME — the half the model does learn on a later turn (UX round 2,
U6). `_token_being_typed_at` answers the anchor's own word first, and the test
is PREFIX-OF the token rather than starts-with, so `/credentials` cannot inherit
an arm and a word shortened past `/cred` still reads as withdrawing the gesture.

**A leading `-` escapes the mask**, so `--forget-all` stays typable. The
credential verbs are flag-shaped precisely so they cannot collide with a key
(keys normalize to `[A-Z0-9_]`), which is the same partition `CREDENTIAL_ARGUMENT`
already draws on the pasted path. Only the *first* character is tested: `-` is
common inside real keys.

**A trap worth recording.** Textual spells punctuation keys as words (`minus`,
`full_stop`), so a handler gated on `len(event.key) == 1` masks letters and
digits while every punctuation character falls through into the document.
Measured, the canary `zQ7-TYPED-LEAK-CANARY-4417` rendered as
`•••-TYPED-LEAK-CANARY-4417` — the first hyphen ended the masking and the rest
was typed in plaintext and submitted. Real credentials are mostly punctuation,
so that gate leaks nearly every actual secret while passing against an
alphanumeric test value. Gate on `event.is_printable`, never on the key name.

**Cost accepted.** Prose typed after the token — `fix the /credential command` —
is masked, because no rule keyed on the buffer can separate it from
`deploy with /credential the prod key`, which must stay armed (design round 1,
D2). The mask is loud and immediate and Esc restores the text in one keystroke,
which is the false-positive direction; the alternative fails toward a plaintext
secret in scrollback that nothing can recall.

**Random naming.** The operator does not invent a name; the store does:

```
LOP_SECRET_<8 chars of base32(random)>      e.g. LOP_SECRET_K3RQ7WZM
```

Env-var-shaped so it drops straight into `credential_env()` injection and
`normalize_credential_key` (`variables.py:92`) round-trips it unchanged.
Collision-free without consulting the store.

**How a random name coexists with a meaningful one.** The `record_id` (UUIDv4)
is the immutable identity; the **name is mutable metadata**. The random name is
just the first value of that field. When the agent later understands what the
secret is — from the operator's description in the same message — it calls
`secret update --id <id> --name GITHUB_TOKEN --description "..."`, which
re-seals the record under the new AAD (§3) and updates the blind index. The old
random name stops resolving; the id never changes, so the audit chain stays
continuous across the rename.

---

## 12. Audit

Every retrieval, store, update, delete, rename and failed authorization is
appended to the `audit` table **and** mirrored to
`~/.local-operator/secrets/audit.log` (0600, `chflags uappnd`).

> **Shipped in PR 1 vs. still outstanding.** PR 1 writes an audit row for every
> *successful* retrieval, store, update, delete and rotation, with the hash
> chain and `audit --verify` below. It does **not** yet record FAILED
> operations — a `get` for a name that does not exist, or a record that fails
> authentication, currently leaves no row — and it does not yet mirror to
> `audit.log` or set `chflags uappnd`. Both gaps belong to the broker PR, which
> owns the peer identity (pid, executable path) that makes a failure row worth
> recording and is the process that can hold an append-only file open. Until
> then the trail shows what succeeded, not an attacker probing for secret
> names; this paragraph describes the destination, and the note describes what
> is actually on disk today.

Recorded: timestamp, event, `secret_id` (never the value), session id, peer pid,
peer executable path (`proc_pidpath`, spike 1), and outcome.

**Tamper evidence.** Each row carries `hash = SHA256(prev_hash || canonical
(row))`, so an attacker who deletes or edits a middle entry breaks the chain and
`lop secret audit --verify` reports where. Combined with `uappnd` this means an
attacker must remove the whole file (detectable as a gap, since the DB table and
the log must agree) rather than forge a plausible history. As §9.6 says
explicitly: evident, not proof.

---

## 13. Compatibility and failure modes

**Forward/backward skew.** The operator runs ~10 sessions and updates the
runtime with `lop-update` while sessions are live, so version skew is routine,
not exceptional.

- `meta.schema_version` is checked on every open. A runtime seeing a **newer**
  schema than it knows **refuses to write** and serves reads only if the record
  format is one it understands; otherwise it fails with "store written by a
  newer local-operator; upgrade this runtime". It must never migrate downward.
- Records carry their own `key_generation` and format version, so a partially
  rotated or partially migrated store is always coherent.
- New optional columns only; never repurpose a column meaning.

**When the broker dies with sessions running.** The broker is not the session's
lifeline — it is consulted per retrieval. So:

- A retrieval attempted while the broker is down returns a clear error
  ("secret broker unavailable"), never a hang. Hard timeout on connect, ~2 s.
- Already-injected session credentials and already-retrieved values are
  unaffected: they are in the session's own memory.
- **Auto-restart, carefully.** A session that finds no broker starts one, under
  a `flock`-guarded startup so ten sessions racing produce one broker. In
  `keyfile` mode the restart is silent; in `passphrase` mode the new broker
  starts locked and says so. Note `AGENTS.md`'s #401 lesson — a blocking `flock`
  deadlocked the event loop — so this lock must be non-blocking-with-retry off
  the event loop, and the e2e stage should cover broker-down boot.

> **Amended during PR 2 (implementation).** What a broker outage costs depends
> on the tier, and the two must not be conflated:
>
> - In `keyfile` mode the master key is on disk beside the store, so a caller
>   the broker would refuse can read it directly — §8's own table says as much
>   ("Script reading a known key file path: **Not stopped**"). Failing a
>   retrieval on a broker outage or an ancestry DENIAL would therefore add no
>   security whatsoever while breaking the store's primary surface: the
>   operator's own `lop secret get`, typed in their own terminal, has no lop
>   session among its ancestors and is denied by construction. This mode
>   therefore falls back to the key file, and the broker's contribution is the
>   audit trail and the §6 redaction notice rather than access control.
> - In `passphrase` mode there is no unwrapped key on disk to fall back to, so
>   a denial is enforced and an unreachable broker is a hard, clearly-worded
>   failure. This is the tier in which the ancestry boundary is load-bearing.
>
> Verified by execution in both tiers: a detached `setsid` script is denied
> against an unlocked hardened store, and a SIGKILLed broker leaves an
> in-flight retrieval with `BrokerUnavailable` in ~5 ms rather than a hang, a
> stale value, or a wrong one.

> **Amended again during PR 2 (QA round 1): the hardened tier could not be
> entered, and the fix is the unlock grant.** Two defects compounded. `unlock`
> was dispatched *behind* the ancestry gate, so unlocking required already
> descending from a registered session — while **nothing in shipping code ever
> registered one** (only tests did). After `harden` the correct passphrase was
> refused and even `status` failed, with the plaintext key already deleted. The
> tier had therefore never worked end to end in either direction: unreachable
> for its runtime, and bypassable by anyone else (§2.1's amendment).
>
> Three changes, which have to land together:
>
> - **`unlock` is authenticated by the passphrase, so it is dispatched before
>   the ancestry gate.** The passphrase is a stronger credential than ancestry
>   and the only one never written to disk in any form. Wrong guesses are
>   audited and cost a geometric backoff (0.25 s → 8 s), since this verb is now
>   reachable without lineage and scrypt's ~180 ms alone is a thin defence
>   against an online oracle.
> - **Interactive sessions register themselves** (`local_operator/secrets/
>   session.py`, wired into `run_tui`). Without this, authenticating `register`
>   would have converted the bypass into a permanent lockout. Closing the
>   channel deregisters, which is what revokes descendants promptly.
> - **Unlocking grants the operator's terminal standing for this boot.** `lop
>   secret get` typed at a prompt has no lop session among its ancestors and is
>   denied by construction — in `keyfile` mode the key-file fallback hides
>   that, but in `passphrase` mode there is no fallback, so the operator's own
>   store stayed unreachable even after a successful unlock. The parent of the
>   process that proved knowledge of the passphrase — the shell it was typed
>   into — is recorded as an authorizing ancestor, pinned by identity like any
>   session. The detached attacker does not descend from that shell and remains
>   denied, verified as real processes against an unlocked broker.
>
> **The residual risk IS widened, and §9.4 now says so.** An earlier draft of
> this note claimed it was not — that the grant was covered by §9.1's "a script
> that runs `lop` itself". That is wrong and the correction matters: a process
> in the granted terminal does not have to run `lop`, it can speak to the
> socket directly and be authorized on lineage alone. The honest statement is
> that anything running in an unlocked terminal — including build tools invoked
> there — can read every secret for as long as that shell lives. It stays
> strictly narrower than the ancestry-free access it replaces, and the detached
> attacker remains denied.
>
> The grant is bounded by the terminal's LIFETIME, which required a fix rather
> than only a doc change (QA Q8): a process inside the granted terminal may
> register itself as a session, and a registered session is an independent
> authorizing entry, so killing the shell used to leave the self-registered
> squatter serving secrets for the broker's whole life. A session admitted
> *because* it descended from a granted terminal now records that terminal, and
> is revoked when the terminal dies; standing is inherited down a chain of such
> registrations, so one sweep removes the whole chain rather than the first hop.
>
> Verified end to end through the real CLI with the passphrase typed at a pty:
> `set` → `get` → `harden` → broker restart (a reboot) → `get` refused, 0 bytes
> on stdout → `unlock` → `get` serves the value → `status` reports
> `passphrase`.

> **Amended again during PR 2 (QA round 3): the key of record is TIER-SPECIFIC,
> and `rotate` did not know it.** §3's rotation paragraph says the new key is
> generated, every record re-sealed, and the new key installed — without ever
> saying WHICH FILE "installed" means, and the implementation resolved that
> ambiguity the same way in both tiers: it wrote the plaintext `master.key`.
> On a hardened store the key of record is `master.key.wrapped`, so a single
> `lop secret rotate` re-sealed the database under a new key while `unlock`
> kept unwrapping the old one. Every secret became undecryptable, exit code 0,
> `status` reporting `secrets 0 / damaged 1`.
>
> The second consequence is the worse one and is the reason this is recorded
> here rather than only in a commit message. `key_mode()` answered
> `passphrase` on the mere presence of the wrapped file, so the store kept
> reporting the hardened tier while the live master key sat UNWRAPPED on disk
> beside the database — the exact property §2.3 sells this tier on, silently
> not provided. A tier is defined by what is ABSENT from disk, so it is now
> decided by looking for the plaintext key: a store carrying both files
> reports `keyfile`, which is the truth, and `status` names the inconsistency
> outright.
>
> That honest answer is also the recovery path. `harden` refuses a store
> already in `passphrase` mode, so under the old answer an operator whose
> rotation had produced this state had no CLI way out at all; with it, `harden`
> sees a tier it can act on and re-wraps the live key. No `--force` flag was
> added — the condition it would guard is precisely "this store is not hardened
> right now", which is what the verb already means.
>
> Three structural consequences, so the class is closed rather than the
> instance:
>
> - **One choke point.** `install_key_of_record_if_current` holds the
>   compare-and-swap both tiers need and takes the install as a callback, so
>   the hardened path cannot acquire the concurrency exposure the keyfile path
>   was fixed for in round 1. `assert_key_of_record_invariant` is a
>   post-condition on every key install: no plaintext `master.key` in the
>   hardened tier, ever.
> - **Hardened rotation stages WRAPPED.** Staging the raw key would have put a
>   plaintext master key on disk for the duration of every rotation — the same
>   defeat of §2.3, in a narrower window. Recovery from the commit-to-install
>   crash window therefore moves to `unlock`, the only moment the passphrase
>   exists; the keyfile tier keeps its unattended repair.
> - **The broker validates its cached key on every use.** Found by the same
>   sweep: the broker holds the key for a whole boot while `rotate` runs in
>   another process, and nothing invalidated it. Because the blind index is
>   derived from the master key, a stale key does not fail loudly — every
>   lookup returned "No secret named X", so an intact store read as empty.
>
> Verified through the real CLI at a pty: `set` → `harden` → broker stop (a
> reboot) → `unlock` → `rotate` → no plaintext `master.key` on disk → broker
> restart → `unlock` → `get` returns the original value.
- A `launchd` agent (`com.damian.lop-secretd.plist`) is the tidier long-term
  answer, but it is **out of scope for these PRs**: it changes machine state
  outside the repo, and the flock-guarded lazy start is sufficient and testable.

---

## 14. PR split

Four PRs, in dependency order. Each is independently reviewable and independently
useful; the first three are backend-only.

### PR 1 — store core: crypto, schema, CLI (no broker)

`local_operator/secrets/` — record sealing/opening, SQLite WAL schema, blind
index, key generation, rotation, and the `lop secret` CLI reading the key file
directly. No daemon, no socket. Ships the default `keyfile` mode end to end.

- **Risk:** crypto misuse (nonce reuse on update, AAD not covering a mutable
  field). Mitigate with round-trip tests including the two tamper cases proven
  in spike 8, and a test asserting a fresh nonce on every update.
- **Rounds:** reviewer + qa-tester. No designer/ux — no user-visible surface
  beyond CLI text.

### PR 2 — broker daemon and peer authentication

The unix-socket daemon, `LOCAL_PEERPID` + `LOCAL_PEERTOKEN` + ancestry walk with
the `(pid, pidversion, start_time)` pin, session registration, flock-guarded
lazy start, audit log with hash chain and `uappnd`. CLI switches to talking to
the broker; the direct-key path stays as the fallback when the broker cannot
start.

- **Risk:** the highest-risk PR. A hang here freezes sessions (`AGENTS.md` #401
  is the precedent — a blocking lock deadlocked the event loop). Requires: hard
  connect timeouts, no blocking lock on the event loop, an e2e case for
  broker-down and broker-killed-mid-session, and a **Linux fallback** using
  `SO_PEERCRED` (which yields pid directly) since CI runs an
  `[ubuntu-latest, macos-latest]` matrix.
- **Rounds:** reviewer + qa-tester. QA must include the spike-9 attacker case
  (detached `setsid` script must be DENIED) as an actual test.

### PR 3 — agent surfaces: `secret` tool, eval library, redaction wiring

The `createIf`-gated `secret` tool, the `secrets` lazy mapping in the eval
worker, `VariableStore.register_redaction`, the broker→session notification with
the ack-before-reply ordering from §6, and the `--persist` route from the
session store.

- **Risk:** a retrieved value escaping into the transcript — the failure that
  matters most. QA must specifically test `$(lop secret get X)` output scrubbing,
  including a value split across two pipe reads (the case `_PipeRedactor` exists
  for), and the notification race.
- **Rounds:** reviewer + qa-tester.

### PR 4 — inline `/credential` composer capture + guidance docs

The `PastedCredential` variant in `editor.py`'s `Marked` union, inline detection,
the `[Credential #1, 64 chars]` marker, history stripping, random naming, the
`ask secret=true` persist option, and the credentials-guidance documentation.

- **Risk:** the marker machinery has a documented history of regressions
  (rounds 17-22 in `editor.py`'s docstrings: markers resolving to the wrong
  payload, orphaned markers, numbering collisions). A credential resolving to
  the wrong payload is worse than an image doing so.
- **Rounds:** reviewer + qa-tester + **designer** (new visible marker style —
  needs rendered before/after frames per `AGENTS.md` §"Visual validation", using
  `scripts.visual_capture.save_capture`) + **ux-reviewer** (a new interaction
  flow: type, paste, watch it redact, keep typing).

**Ordering:** 1 → 2 → 3 → 4. PR 1 is independently shippable. PR 4 depends on 3
for the persist route but its composer half could be split out if 3 stalls.

**Version bumps:** none, per the standing rule — `pyproject.toml` stays at the
last released version on every branch.

---

## Appendix A — spikes

All spikes are in `/tmp/spike-secrets/`, run against
`~/local-operator/.venv/bin/python` (CPython 3.14.3, macOS 25.6.0, arm64).

| # | File | Question | Verdict |
|---|---|---|---|
| 1 | `spike1_peercred.py` | what does macOS tell a socket server about its peer? | uid/pid/audit-token/exe path/ancestry **yes**; code signature **no** (`csops` rc=-1) |
| 2 | `spike2_env.py` | is a sibling's env readable? | **yes**, 3 ways — env tokens rejected |
| 3 | `spike3_memory.py` | can a sibling read another process's memory? | `task_for_pid` **rc=5 denied** |
| 4 | `spike4_ancestry.py` | ppid/start-time offsets in `kinfo_proc` | ppid @560, pid @40; start time **@0**, not @8 |
| 5 | `spike5_debugger.py` | why did lldb hang? | blocked on `system.privilege.taskport.debug` in `authd`; `SecurityAgent` spawned |
| 6 | `spike6_fdcap.py` | is an inherited fd a good capability? | works, but launders identity — rejected |
| 7 | `spike7_perms.py` | are socket mode bits enforced? peer pid after connector exits? | **enforced** (0000 → EACCES); pid pinned at connect |
| 8 | `spike8_crypto.py` | AEAD, AAD binding, scrypt cost, WAL under 10 sessions | all pass; tamper cases fail closed; 21 ms |
| 9 | `spike9_broker.py` | does ancestry auth work end to end? append-only log? | agent ALLOW / detached attacker DENY; `uappnd` blocks truncate+unlink, owner can clear |
| 10 | `spike10_bypass.py` | the bypasses | `lop` on PATH, runtime tree writable, interpreter adhoc-signed |
| 11 | `spike11_files.py` | file-shaped secrets | FIFO not seekable; `/dev/fd/N` is a **dup** (single-shot) on macOS; private-dir file works |
