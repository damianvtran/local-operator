---
name: qwencloud
description: Show QwenCloud Token Plan usage on /usage — store the login_qwencloud_ticket console cookie with lop qwencloud-ticket; fix a blank or expired window.
---

# QwenCloud Token Plan usage on /usage

Use `lop qwencloud-ticket` when the user asks how to see their QwenCloud (Qwen)
usage, why `/usage` shows no QwenCloud window, or wants their personal Token
Plan quota on that panel. This file is the operational playbook an agent
follows; `local_operator/providers/qwencloud_console.py` and the QwenCloud
fetchers in `local_operator/providers/usage.py` are authoritative, and this
guide is a reading of them that can fall behind.

## Two routes, and which one answers

`/usage` can report QwenCloud quota two ways. The console route runs **first**;
BSS is the fall-through.

- **BSS route** — `fetch_qwencloud_token_plan`, pre-existing. Uses the
  management OAuth token from `/login alibaba-token-plan-oauth` against
  `cli.qwencloud.com`. It serves **teams and seat accounts**, which need no
  ticket. For a **personal** Token Plan account it returns nothing at all:
  `QuerySubscriptionGray` reports `IsGray: true`, but
  `GetSeatSubscriptionSummary` comes back `Data: {}` and `DescribeFrInstances`
  reports `TotalCount: 0` on all three commodities. An empty BSS answer is
  therefore not a broken login — it is the signal that the account is personal
  and needs the console route instead.
- **Console route** — `fetch_qwencloud_console_usage`. POSTs form-encoded to
  `https://cs-data.qwencloud.com/data/api.json`, authenticated by **one**
  cookie, `login_qwencloud_ticket`. The response carries `per1WeekPercentage`,
  a **fraction used** (`0.2405` means 24.05% consumed), and `per1WeekResetTime`
  in epoch milliseconds; the panel renders that as the `7 Day Credits` window.
  `sec_token` is sent but the server does **not** validate it. An expired
  ticket returns HTTP **200** carrying
  `data.errorCode == "BailianGateway.Login.NotLogined"` — a 200 is not success
  here.

**Why a second credential exists:** the management Bearer token cannot reach the
console endpoint — it returns `BailianGateway.Login.NotLogined`. No login flow
can mint a browser session cookie, and one cannot be refreshed headlessly. That
is the whole justification for asking the user to capture it by hand.

## Setup

1. Run `lop login-status` and confirm an `alibaba-token-plan` credential row
   exists. The ticket **augments** that row; it does not replace one, and with
   no such row `/usage` cannot report at all, because the ticket deliberately
   cannot authenticate the provider. If the row is missing, have the user run
   `/login alibaba-token-plan-oauth` first. Do not read auth.db or print
   credentials.
2. Have the user sign in to QwenCloud in a browser, open devtools, and copy the
   **value** of the `login_qwencloud_ticket` cookie. Do not ask them to paste it
   into chat, an issue, or a file: the value belongs in the store and in
   nothing that keeps a transcript.
3. Have the user store it from their own shell, through STDIN:

   ```bash
   printf %s '<TICKET>' | lop qwencloud-ticket set
   ```

   **STDIN only, never argv.** A command line lands in shell history and is
   readable in `ps` by every process running as the user, so `set` refuses a tty
   with exit 2 instead of prompting into one. It also refuses a store directory
   wider than `0700` or a file wider than `0600`, rejects embedded newlines,
   control characters and non-latin-1 bytes, and rejects anything over
   `QWENCLOUD_TICKET_MAX_LENGTH = 4096` characters.
4. Run `lop qwencloud-ticket status`. It reports presence, character count and
   age — **never the value** — and warns on the three states that silently
   produce an empty panel: the cookie is older than ~7 days
   (`QWENCLOUD_TICKET_STALE_MS`); no `alibaba-token-plan` credential row exists
   for it to augment; this build has no console fetcher, meaning a `/update`
   reverted it.
5. Open `/usage` in lop. `r` force-refreshes past the 5-minute cache.

The ticket is split across two stores. Its **value** goes into the encrypted
`lop secret` store as `QWENCLOUD_CONSOLE_TICKET`
(`QWENCLOUD_TICKET_SECRET_NAME`); only its **metadata** — capture time, length,
the secret's name — stays in `~/.local-operator/auth.db` under the provider
namespace `qwencloud-console` (`QWENCLOUD_CONSOLE_PROVIDER`). That namespace is
not a login method, so it appears in no `/login` list; `status` is how you
inspect it, and it reads the metadata only. What that encryption is and is not
worth is "What this actually protects against" below — read it before telling a
user their cookie is safe.

## The panel is not the diagnostic

On expiry `/usage` shows `no provider reports no usage — no quota endpoint, or
no credential for one` (`alibaba-token-plan reports no usage — …` when the panel
is scoped to the provider). That is worse than a neutral "nothing to show": it
names a **missing credential** when the credential is stored and merely expired,
and it carries no hint to re-run `set`. It can also keep serving the last good
percentage for minutes after the ticket dies: a failed fetch writes the previous
value back (`USAGE_REPORT_TTL_MS` is 5 minutes, and an account whose probes keep
failing re-probes on a jittered ~10-minute cadence). So a dead ticket can look
healthy for a while, and an empty panel can mean nothing more specific than
those two guesses — neither of which is "your ticket expired". Do not tell the
user to trust the panel in either direction. `lop qwencloud-ticket status` is
the **only** diagnostic: run it before believing anything the panel implies
about QwenCloud.

## What this actually protects against — do not overstate it

Be accurate if the user asks, because the honest answer is the useful one.

The ticket's value is held in the encrypted `lop secret` store (AES-256-GCM,
blind-indexed names) under `QWENCLOUD_CONSOLE_TICKET`; only its metadata —
length, capture time — remains in `~/.local-operator/auth.db`. That is a real
improvement over the plaintext row it replaces: disk-scanning malware finds no
readable `API_KEY=`, and not even the name.

It is **not a vault**. In the default `keyfile` mode the master key is a file
beside the store (`<config>/secrets/master.key`), so an attacker who knows to
look reads both. More fundamentally, **anything running as the user that is
willing to run `lop` can retrieve this value**, exactly as local-operator does.
The opt-in `lop secret harden` passphrase mode holds the key only in a running
broker's memory.

This is still the **broadest credential in the store** — a full-account console
session, not a scoped read-only key. Encrypting it raises the cost of stealing
it; it does not make it safe. Never tell the user it is "safe", "protected" or
"secure" without that qualification, and never describe either store as a
vault.

## Removal is not revocation

`lop qwencloud-ticket rm` removes the row from the API's view and **confirms the
delete by re-reading**. If it cannot confirm, it exits **non-zero** saying the
ticket **MAY STILL BE STORED** — read that as "still there", not as a warning to
wave off. It now has two stores to clear, so that message names which: *"in the
credential store, the encrypted secret store, or both"*. A locked hardened store
gets its own line naming `lop secret unlock`.

Even a confirmed delete is not a revoke. The session **stays valid server-side**
whatever the local store says. **`rm` is not a substitute for signing out or
revoking the session in the QwenCloud console**; if the cookie may have been
exposed, tell the user to do that as well.

## Migrating a ticket stored before the move to the secret store

A ticket captured by an older build sits in `auth.db` as plaintext JSON.
`lop qwencloud-ticket migrate` moves that value into the encrypted store and
rewrites the row as metadata only, keeping `captured_at` so the staleness clock
is not reset. It is idempotent: with nothing to migrate it reports that and
exits 0.

**Plaintext can survive in SQLite freelist pages until a `VACUUM`.** That is
still true, and it is now true of exactly two things: a **pre-migration**
install, and the bytes `migrate` itself leaves behind. It is no longer true of a
fresh `set`, which never writes the value to `auth.db` at all. It is also the
reason `migrate` vacuums.

**`migrate` clears those bytes only when the vacuum and its checkpoint succeed,
and it tells the user when they did not.** `auth.db` runs in WAL mode, so the
vacuum's rebuild goes through the write-ahead log and a
`PRAGMA wal_checkpoint(TRUNCATE)` is what actually removes the old bytes. A
reader on the database makes that checkpoint return **busy rather than raise**,
and a running lop session is such a reader — so on a machine with sessions open
this is the **likely** outcome, not an edge case. When it happens `migrate`
prints a WARNING that the old plaintext **could NOT be cleared** and remains
readable on disk until a later `VACUUM`, and the success line claiming the row
is metadata only is **withheld**. The remedy it names is real: quit the running
lop sessions and re-run `migrate`. Read that warning the same way as `rm`'s
**MAY STILL BE STORED** — as "still there".

## When the secret store is locked

The value now lives behind the secret store, so that store's state is the
user's problem too. In the opt-in `lop secret harden` passphrase mode a store
that has not been unlocked since the last reboot cannot be read:

- `/usage` keeps the QwenCloud block on screen with the note
  `locked — lop secret unlock` rather than silently dropping the window.
- `lop qwencloud-ticket status` exits **non-zero** saying whether a ticket is
  stored is **UNKNOWN** — *"this is not the same as none being stored"* — and
  the store's message names `lop secret unlock`.

A third state is new and is reported rather than silent: metadata in `auth.db`
with **no encrypted value** behind it. `status` prints a WARNING that the value
is missing and tells the user to re-run `set`; `/usage` shows
`no ticket — lop qwencloud-ticket set`.

## Known limits

Accepted behaviour, not bugs to fix or promise away:

- A **hybrid** account — a personal plan that also has BSS Credit Packs — loses
  its `credits-packs` row, because the console route answers first and BSS never
  runs. Narrow: it takes a dead OAuth grant **and** BSS packs at once.
- Any `lop /update` reinstalls from PyPI and **silently reverts the feature**.
  The ticket row survives but nothing reads it. `status` self-diagnoses exactly
  this, so run it after an update before concluding the cookie broke.

## Prohibitions

Never accept the ticket in chat, an issue, a file, or a commit. Never pass it as
argv. Never print a `raw` or `data` dict read back from the store — that is how
a live credential ends up at rest in a transcript. Never describe the store as a
vault, and never report a QwenCloud window as healthy on the panel's word alone.
