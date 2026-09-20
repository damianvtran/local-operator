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
   `{"ok": false, "code": "relay_unavailable", "message": ...}` when this device's
   relay cannot answer — the message names which relay it was (a stopped one and a
   wedged one are different incidents), so fix that (`lop network start --json`)
   before reading anything into an empty list.
   MEMBERSHIP CONVERGES ON A SCHEDULE, NOT ON A LINK BEING ESTABLISHED. A member
   admitted after this device joined becomes visible without anything being
   restarted and without anyone running a command: a live link is asked for its
   member table again every fifteen seconds or so, and because a table transfers
   transitively, a device two hops from the newcomer learns it too — including on a
   dial-only machine whose only neighbour never holds a link to the newcomer. A
   peer that accepts the connection and then does not answer its table read is
   skipped after four seconds and asked again on the next pass, so one unresponsive
   member delays nobody else's convergence. A
   device that has been switched off still shows the table it had when it went down
   until its next contact, and a member that IS reachable is never hidden by that:
   run any of `ls`, `show`, `peers` rather than restarting the relay.
   **Say what the count rests on.** Every member count travels beside a
   `membership` block, and that block is what to read before believing the number:
   `membership.table.answered` is the peers whose table answered, `not_answered`
   names the ones that could not be asked and why, and `complete` is true only when
   EVERY other active member answered. A device with no route to a peer reads
   `complete: false` permanently — that is the honest answer, not a fault — and
   the human `ls` line carries the short form of it (`[members verified with 7 of
   9 peer(s)]`, or `[members NOT verified: no peer answered]`). `--all-peers`
   refreshes the table before it merges, so an incomplete peer set is named rather
   than silently merged.
6. Tell the user what they now have: a relay this device supervises, an identity
   keypair other networks will address it by, and a member list they can inspect.
   The next section says what the session plane can and cannot do across the mesh.

## Which device should run this session

A session can be listed, created, warmed and stopped ON a peer over a paired
mesh. From a shell:

| Command | Effect |
|---|---|
| `lop network sessions --all-peers --json` | every peer's sessions, merged; each row names the device holding it |
| `lop network sessions --peer <id\|name> --json` | one device's own catalogue |
| `lop sessions --peer <id\|name>` / `--all-peers` | the same rows through the ordinary session list |
| `lop network sessions --peer <id> --create --name <n> [--prompt <p>]` | create the session ON the peer, which mints its id |
| `lop network sessions --peer <id> --engage <session>` | warm a stored session on the peer |
| `lop network sessions --peer <id> --stop <session>` | stop it where it lives |

WHAT IS **NOT** IN THIS BUILD, although the design names it: `lop exec --peer`,
`lop send --peer`, `lop sessions move <id> --to <peer>|local`, and the TUI's
`/new remote` and `/move`. Do not promise a user that a session can be MOVED
between devices, and do not retry those verbs hoping for a different answer —
`--peer` is not a flag on `exec` or `send`, and `move` is not a `sessions`
subcommand. Credentials are the other gap (next section): a session created on a
peer needs a model THAT PEER can reach, and this build does not broker one.

DIAL-ONLY DEVICES. A machine with no inbound path (behind NAT, or
`network.listen_address: 127.0.0.1`) can reach a peer but cannot be reached by
one. That is a supported configuration, not a fault, and the rule is symmetric:
whichever side can dial forms the link, and the link is then bidirectional. So a
peer's listing works from the dial-only side, and a session on the dial-only
device is best created from there. `lop network peers --json` says which kind each
member is, with the reason in `reason`: `no_endpoint` means that member never
declared an address (admitted by an older build), a single `connect_failed:*`
means nothing answered at any address it declared, `unreachable: <address>
<why>; …` names every address it declared and what each one did (a member often
has one address that is a black hole from where you are and one that answers),
and `not_attempted:` means nothing was dialled at all — the listing gave up
before this member's turn rather than claiming anything about it.
`handshake_refused:*` means the peer ACCEPTED the connection and then closed it
during the handshake: that is what a REFUSAL looks like on this wire, because the
protocol never explains one (an open port that answers "that token is not mine"
would let a stranger enumerate what is real). The suffix keeps what was observed —
`ConnectionError` is a close, `TimeoutError` is a peer that never answered — and
`lop network ls`/`doctor` say the interpretation: the network is marked
`[refused_by_peers]` and `doctor` carries a `membership` check naming the states
that look like this (this device was removed, the network was marked untrusted
after a panic, or this device's epoch is behind). A device with
several addresses is probed at ALL of them at once, so the order its row happens
to list them in cannot decide whether it looks reachable. `lop network doctor
--json` reports the same per link, one row per address plus the handshake, and
its top-level `ok` is FALSE whenever any check in its own array is false.

The rule the design fixes: a session with a strong local dependency (a repository
that exists only on this machine, an attached browser, a terminal the user is
watching) should stay where it is. Mobility is for work that follows the person,
not for work that follows the machine.

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
   `epoch_skew`, and the failure name (`connect_failed:*`, `no_answer`, a refusal
   code). A row that was not dialled says so and is `ok: false` rather than
   passing unchecked. A `{"check": "membership"}` row appears ONLY when something
   is wrong with this device's own standing, and it carries the sentence and the
   remedies — read it before the endpoint rows, because a removed or refused
   device's endpoints all fail for one reason. The `relay` field is read from the
   relay itself — replied to, `running (pid N)` and, if its control socket did not
   answer, that too (a relay that is up but silent is a WEDGED relay: its pid is
   alive and it is not answering, which is a different incident from a stopped
   one).
2. `lop network log --since 1h --json` — what actually happened: `member_admitted`,
   `member_removed`, `membership_learned`, `invite_minted`, `panic_raised`,
   `trust_changed`, `pairing_refused`, `handshake_refused`, each with `ts_iso`,
   `event`, `outcome`, `network_id` and a `detail` object. `handshake_refused` with
   `detail.cause: "peer_closed_silently"` is THIS device being refused by a peer —
   the only place that fact is written down, since the peer never says it.
3. `lop network peers --json` — reachability right now, per peer: `device_id`,
   `name`, `network_id`, `reachable`, `reason`, `endpoints`.
4. `lop network status --json` — install and health: `relay_running`,
   `relay_answering`, `relay_state` (`live` / `wedged` / `stopped`), `relay`,
   `identity_present`, the per-network rows and the log path. `relay_running` is
   true whenever a relay process of ours exists, so it is never false beside a
   `record.pid`; `relay_answering` is what this probe actually got back.

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

**A REMOVED DEVICE LEARNS IT FROM ITS OWN SURFACE, AND RE-ADMISSION HAS ONE
ROUTE.** If this device was removed by an admin, its own `ls`/`show`/`status` say
`this device is no longer a member of <network> (removed by <who>)` — from the
tombstone the rotation delivered, or, if it was offline when that happened, from
the refusal it observes on its next dial. Its epoch and member list are the last
delivered state, and its `doctor` carries a `membership` check with a code
(`removed` / `refused`), so an operator is not left reading a transport error
alongside `trust: active`.

Re-admitting it is NOT `lop network trust`: that verb re-admits a NETWORK that was
marked untrusted after a panic, and it does not restore a member that was removed —
the removed device id is burned on every device that saw the rotation (`removed_ids`,
kept forever on purpose), and a fresh invite does not revive it. Do not tell a user
to try it, and do not retry a join that already answered `device_id_conflict`.
The route that works is a NEW IDENTITY on the removed device:

```bash
lop network identity rotate --json    # on the removed device: mints a new id
# then mint a fresh invite on a member, and join with it as usual
```

The refusal now says so on the joiner's own screen: `device_id_conflict` comes back
with the admitting device's sentence plus that remedy, instead of the bare code.

```bash
lop network identity rotate --json    # for a suspected key compromise
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

- Session MOBILITY — `lop sessions move <id> --to <peer>|local`, move-with-`--keep`
  (fork), and the TUI's `/move` and `/new remote`. Creating, listing, warming and
  stopping a session ON a peer IS implemented (see "Which device should run this
  session"); moving one is not (`mesh-session-mobility.md`).
- Credential brokering (`mesh-credentials.md`).
- The console/relay UI surfaces (`mesh-ui.md` §1–2).

`lop network uninstall --purge` removes this device's NETWORK records, invite
token files, outbound queues, parked pairings and the audit log, and deliberately
KEEPS the identity keypair — every network addresses this device by it, and losing
it silently would break networks this purge does not cover. `--purge-identity` is
the separate verb that deletes the keypair; it needs a terminal to confirm and
names every network the identity knows. On a host with no launchd (Linux) neither
verb needs launchd: `lop network serve` is how the relay runs there.

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
