---
name: network
description: Pair this device into a lop mesh network, see which peers are reachable, drive the network's lifecycle, and respond to a mesh incident such as disconnect or panic.
---

# Network: pair your devices, and see what is on the other end

A mesh network is a group of devices that trust each other by a shared secret,
each addressed by a device keypair of its own. The one rule that matters:
`lop network` is about **this** device, and a session's owner is the device it
runs on.

Read this guide before pairing two devices, before answering "what is on my
other machine", and before touching an incident control. The design contract
lives in `docs/design/mesh-network.md` (§6 is the CLI surface),
`docs/design/mesh-transport-identity.md` (op and capability vocabulary) and
`docs/design/mesh-ui.md` (§3, the agent half). This file is the playbook you
execute.

Drive every command below with `--json`: the agent path parses it, and the human
rendering is not a contract.

## When the user asks to set it up

1. `lop network status --json` first — it answers "is there a relay, is there an
   identity, which networks" without changing anything. Expect `installed`,
   `relay_running`, `identity_present`, `networks`.
2. On the device that should own the network:
   `lop network init <name> --json`. Creates the network, this device's identity
   if it has none, this device's member row (role `admin`), and starts the relay
   under a LaunchAgent (`~/Library/LaunchAgents/com.local-operator.network.plist`).
   Pass `--no-start` when the user does not want a supervisor yet, and
   `--listen-address 127.0.0.1` when this device should only dial out and never
   accept (the default bind is `0.0.0.0:4097`; `--port` and `--advertise-host`
   override the rest).
3. Mint the invite on that device: `lop network invite --role drive --json`.
   Add `--network <name>` when the device is in more than one network, `--role
   read` for a viewer, `--expires 30m` to change the ten-minute default. The
   token is written to a **file** and the JSON gives you `path`, never the token;
   hand that file to the other machine out of band — AirDrop, a shared directory,
   or the user's own copy at a terminal they control. Never into a chat
   transcript, and never by printing it from here.
   `--print` exists for a human at a keyboard and is refused with `--json`.
   To bind the token to one device, pass
   `--device <device id>` with the id the joining device reports in
   `lop network identity show --json`; any other device redeeming it is refused
   and the invite is burned. That is the form to use after a compromise.
4. On the other device, **the human** runs it in a terminal:
   `lop network join @<token-file>` (or with the token itself).
   The joining device prints the inviter, the network, the offered role, its
   derived `code` (`481 926`) and a `fingerprint`, then asks for the code. The
   inviter admits the device only if the typed value matches its **own**
   derivation, and a mismatch burns the invite — so the human must check that
   the network and role are the ones they asked for and that the code agrees on
   both screens. `--verify` makes the 160-bit fingerprint the compared value
   instead of the six digits: use it when the two machines do not share a
   private path. `--name` sets the name this device will be known by, `--host
   host:port` overrides the endpoint to dial.
   **You cannot do this step.** There is deliberately no flag that completes a
   pairing, and the code is read at the joining device's own prompt — that is
   what makes the interlock real rather than a round trip. Hand over the file and
   the command, and let the user run it. The one non-interactive spelling,
   `--sas-stdin`, is refused unless `LOP_NETWORK_TEST_MODE=1` is set: it is the
   e2e harness's seam, and setting that variable to finish a real pairing would
   turn the human check into a formality. Never set it.
5. Verify from both sides: `lop network peers --json` must show the other device
   with `reachable: true`, and `lop network ls --json` must agree on the epoch
   and member count. `peers` exits 1 and returns
   `{"ok": false, "error": "the relay is not running"}` when this device's relay
   is down — fix that (`lop network start --json`) before reading anything into
   an empty list.
6. Tell the user what they now have: a relay this device supervises, an identity
   keypair other networks will address it by, and a member list they can inspect.
   The next section says which of the session features are not in this build yet.

## Which device should run this session

Not yet available in this build. The design puts session placement here
(`lop sessions --all-peers`, `lop exec --peer`, `lop sessions move <id> --to
<peer>|local`, `/new remote <peer>`, `/move`), but the relay answers every
session-plane op by name:

```
net_forward_session -> unknown local op    (mesh-session-mobility.md owns it)
```

Until that slice lands, do not promise a user that a session can be created on,
driven from, or moved to a peer, and do not retry those verbs hoping for a
different answer. What works today is pairing, membership, the peer table, the
audit log and the incident controls — the transport half of the network.

The rule the design fixes, which will hold when it lands: a session with a
strong local dependency (a repository that exists only on this machine, an
attached browser, a terminal the user is watching) should stay where it is.
Mobility is for work that follows the person, not for work that follows the
machine.

## Credentials on a peer

Credentials are brokered, never mirrored (decision A5). Operationally: never
copy a token or a key from one device to another, never run a login on a peer,
and never "fix" an expiry by re-authenticating for someone else. A refresh is
requested from the device that owns the credential. Nothing in this build
brokers credentials yet — the `broker_credential` capability is in the member
vocabulary, and the op that would use it is not.

## When something looks wrong

Diagnose in this order, and stop at the first answer that explains it:

1. `lop network doctor --json` — identity present, records healthy, endpoints
   present, and (with the relay up) each endpoint actually dialled: latency,
   `epoch_skew`, and the failure name (`connect_timeout`, `connection_refused`,
   or a refusal code). Without a relay it says `not probed (the relay is not
   running)` rather than guessing.
2. `lop network log --since 1h --json` — what actually happened: `member_admitted`,
   `member_removed`, `invite_minted`, `panic_raised`, `trust_changed`,
   `pairing_refused`, each with `ts_iso`, `event`, `outcome`, `network_id` and a
   `detail` object.
3. `lop network peers --json` — reachability right now, per peer: `device_id`,
   `name`, `network_id`, `reachable`, `reason`, `endpoints`.
4. `lop network status --json` — install and health: `relay_running`, `relay`,
   `identity_present`, the per-network rows and the log path.

Two things that look like failures and are not: a peer that is unreachable is
**not** an error and its sessions are simply not reachable from here; a network
this device has left shows as `trust: disconnected` and keeps its audit trail.
An unreachable peer is a `doctor` question — never a reason to re-run the same
list.

## Incident: stop the network

Two controls, and what each does:

- `lop network disconnect [<network>] --json` — leave: stop trusting the
  network, close links, delete the local copy of the epoch secret (the audit
  trail is kept). Measured scope: `secret_deleted: true`, and the record stays
  with `trust: disconnected`. Because the secret is gone, a later `panic` on
  that network fails on the missing secret — re-join to use it again.
- `lop network panic [<network>] --json` — the incident control: broadcast a
  revoke, rotate the secret and bump the epoch for every member, and mark the
  network untrusted locally. Every other device must then be re-admitted with
  `lop network trust <network> --active --json`.

**Run neither without an explicit instruction that names the network and the
action, and never as a retry after a failed command.** Neither takes a
confirmation flag; both mutate the trust state of every device in the network,
and panic invalidates an invite or a session the user may be in the middle of.

```bash
lop network trust <network> --active --json      # re-admit after a panic
lop network trust <network> --untrusted --json   # refuse it again
```

## What never to do

- Do not edit anything under the network directory by hand (identity, network
  records, secrets, outbox) — one writer owns each file, and a hand edit is a
  state no code path expects.
- Do not run the join step for the user, and never type a code you did not see
  on the other device.
- Do not print an invite token, or read the token file into a result: it is a
  single-use bearer credential, and a tool result is the most-copied text in the
  system.
- Do not add a member by editing a member list. Membership changes through
  `invite`/`join`, `member rm`, and the epoch rotation they carry.
- Do not force a full re-sync or invent a `--force`: nothing here takes one.

## Reference

**Exit codes.** `0` success; `1` a refusal or a failed operation (the sentence is
on stderr, and it is the sentence to show the user); `2` a usage error, including
`--print` together with `--json`. The design reserves `3` for "a human decision
is required"; this build never emits it — the one human step (`join`) is a
prompt on the joining device rather than a two-phase command, so there is no
`--confirm` flag to reach for.

**Commands.**

```bash
# reading
lop network status --json
lop network ls --json
lop network show <network> --json
lop network peers --json
lop network doctor --json
lop network log --json              # --follow, --limit 50, --since 15m, --export <file>

# lifecycle
lop network init <name> --json
lop network rename <network> <name> --json
lop network rm <network> --json
lop network invite --role drive --json
lop network join @<token-file>
lop network member rm <network> <device> --json

# incident
lop network disconnect [<network>] --json
lop network panic [<network>] --json
lop network trust <network> --active --json
lop network trust <network> --untrusted --json

# the relay and this device's key
lop network serve                   # foreground
lop network start --json
lop network stop --json
lop network restart --json
lop network identity show --json
lop network identity rotate --json
lop network uninstall --json        # --purge, whose real scope is below
```

`serve` runs the relay in the foreground (supervision belongs to launchd or to
that terminal); `start`/`stop`/`restart` drive the LaunchAgent. `log --export`
copies the audit log somewhere retention will not prune it — take that copy
before an incident review, not after. `identity rotate` replaces the device key
and announces the new device id to peers (the old id is stale from then on): it
is for a suspected key compromise, and the previous id is gone for good.

**What is not implemented in this build**, so nothing above should be promised as
existing:

- Session placement, forwarding and mobility (`mesh-session-mobility.md`).
- Credential brokering (`mesh-credentials.md`).
- The console/relay UI surfaces (`mesh-ui.md` §1–2).
- `uninstall --purge` is narrower than its step message claims: it removes files
  directly under the network directory (the audit log) and reports "deleted this
  device's mesh identity and network records", but the identity and the network
  records live in subdirectories and survive. The design splits this properly —
  `--purge` for a network's records, invites and outbox, and a separate
  `--purge-identity` with a TTY confirmation naming every network the identity
  knows — and neither that split nor `--purge-identity` exists yet. Tell the user
  what actually happens; to leave a single network, `lop network rm <network>`.

**File locations** (under the config directory — `LOCAL_OPERATOR_CONFIG_DIR`, by
default `~/.local-operator`): the network directory holds `identity/device.json`
(0600 keypair), `networks/<network-id>.json` and `.secrets.json`, `outbox/`
(invite token files and queued frames), and `audit.jsonl` (append-only, rotated).
The relay's own log is `logs/network.log`, and its LaunchAgent is
`com.local-operator.network`.

**Capability vocabulary** — one list, the authorizer's:
`admin`, `broker_credential`, `list`, `view`, `prompt`, `steer`, `stop`,
`slash`, `delete`, `move`. Roles map onto them (`read` is viewer-shaped, `drive`
adds `prompt`/`steer`/`stop`/`slash`, `admin` adds `admin`);
`lop network show <network> --json` prints each member's granted list.
