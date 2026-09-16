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

The ticket is stored in `~/.local-operator/auth.db` under the provider
namespace `qwencloud-console` (`QWENCLOUD_CONSOLE_PROVIDER`). That namespace is
not a login method, so it appears in no `/login` list; `status` is how you
inspect it.

## The panel is not the diagnostic

On expiry `/usage` shows `no windows reported` — **the same string** it uses for
"this provider has no quota endpoint" — with no hint to re-run `set`. It can
also keep serving the last good percentage for minutes after the ticket dies:
a failed fetch writes the previous value back (`USAGE_REPORT_TTL_MS` is 5
minutes, and an account whose probes keep failing re-probes on a jittered
~10-minute cadence). So a dead ticket can look healthy for a while, and an
empty panel can mean nothing more specific than "no endpoint". Do not tell the
user to trust the panel in either direction. `lop qwencloud-ticket status` is
the **only** diagnostic: run it before believing anything the panel implies
about QwenCloud.

## What this actually protects against — do not overstate it

Be accurate if the user asks, because the honest answer is the useful one.

The ticket sits in `~/.local-operator/auth.db` as **plaintext JSON**, protected
only by file mode `0600`. There is **no OS keychain** behind it, unlike the
encrypted `lop secret` store. It is also the **broadest credential in that
store**: a full-account console session rather than a scoped read-only key. This
feature only ever reads usage with it, but anyone who reads the file holds the
user's QwenCloud console.

So: a file permission, not a vault. Never call it safe, protected, or encrypted,
and never describe `auth.db` as a secure store.

## Removal is not revocation

`lop qwencloud-ticket rm` removes the row from the API's view and **confirms the
delete by re-reading**. If it cannot confirm, it exits **non-zero** saying the
ticket **MAY STILL BE STORED** — read that as "still there", not as a warning to
wave off.

Even a confirmed delete is not a revoke. Plaintext can survive in SQLite
freelist pages until a `VACUUM`, and the session **stays valid server-side**
whatever the local store says. **`rm` is not a substitute for signing out or
revoking the session in the QwenCloud console**; if the cookie may have been
exposed, tell the user to do that as well.

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
