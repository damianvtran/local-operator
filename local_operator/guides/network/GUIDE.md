---
name: network
description: Pair this device into a lop mesh network, see which peers answer, move a session between devices, drive the network's lifecycle, and answer a mesh incident.
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
   read` for a viewer, `--expires 30m` to change the ten-minute default. `drive`
   is the right default for a device that will PROMPT from here and never hold
   anything: it may list, view, prompt, steer, stop and slash, and it may not
   take a session (`move`), delete one on a peer (`delete`) or borrow a login
   (`broker_credential`). Grant those to a device you trust to carry work — see
   "Moving a session between devices". The
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
   derived `code` (`481 926`) and a `fingerprint`, then asks for the code, and
   says how long the person has (up to 180 s, and it reports the time **left** on
   the invite rather than the duration it was minted with, so a token that has
   been carried around for a while shows a smaller window). The inviter admits the device
   only if the typed value matches its **own** derivation. A mismatch refuses the
   join, and the human must check that the network and role are the ones they
   asked for and that the code agrees on both screens: **a delay and a mistyped
   digit do not spend the invite** — run `lop network join` again with the same
   token — and it takes repeated failures (three) before one is burned, so read
   the two screens carefully rather than assuming a retry will fail. `--verify` makes the 160-bit fingerprint the compared value
   instead of the six digits: use it when the two machines do not share a
   private path. `--name` sets the name this device will be known by, `--host
   host:port` overrides the endpoint to dial.

   WHEN BOTH BUILDS ARE NEW ENOUGH, BOTH SCREENS ALSO SHOW WHAT WILL BE SHARED
   before anything is admitted: the joining side under the code (`Credentials
   <inviter> will serve to this device:` with one row per credential — `will be
   served` / `not offered`), and the same rows on the inviter's confirm screen.
   The owner may only REMOVE rows (`[t]` at its prompt; nothing can be added in
   a pairing), the confirm screen re-renders the list after each edit so the
   frame you answer `y` at matches the grant, and the final set is what the
   joined receipt reports (`available here: …`; a promised key the result does not
   carry is named either as deliberately not shared or, if the grant failed,
   with the remedy).
   An older build on EITHER end shows no list and the ceremony is exactly what
   it was before — the joiner's screen says the same, so an absence is never
   read as "nothing to share".

   THE INVITER'S HALF, when its relay runs under a daemon and has no terminal to
   ask at, is `lop network confirm`: `--list` shows the parked pairing with both
   codes, and answering it records the person's decision for the waiting pairing
   loop. It refuses without a TTY — a foreground `lop network serve` is the
   alternative on that side.

   THE TWO-PHASE PAIR, which is how you do this step WITHOUT a terminal. When the
   person is not sitting at the joining device's keyboard, start the ceremony and
   park it:

   ```bash
   lop network join @<token-file> --park --json
   # → {"status":"awaiting_confirmation","sas":"481926","shown":"481 926",
   #    "fingerprint":…,"expires_at":…,"seconds_left":…,
   #    "offers":[{"key":"openai","kind":"oauth-rotating","label":"d***@example.com",
   #               "share":true},
   #              {"key":"anthropic","kind":"api-key-static","label":"","share":false}],
   #    "sentence":…}   exit code 0
   ```

   (`sas` is the compact spelling, `shown` and the `sentence` carry the spaced one —
   the same six digits either way. `offers` is the share list: PRESENT as a (possibly
   empty) list only when the other device sent one, absent for an older build — and
   the `sentence` carries the same fact in words (`… will serve: openai.` /
   `No credentials were offered.` / `The other device did not offer credentials …`),
   so an agent that only reads the sentence can still say what will be shared. A
   ceremony nobody answers exits `3`
   (`pairing_unanswered`), having sent nothing.)

   The first call prints this device's code and then WAITS (up to the same window the
   prompt does). Show the user the code and the `sentence` it carries, and ask them to
   read the code off the OTHER device's screen. The ceremony's socket belongs to the
   parked process, so leave it running: do not kill it, and do not start a second park
   for the same invite.

   **You cannot do this step.** The second phase is the user's, run at their own
   terminal:

   ```bash
   lop network join --confirm <code> --json      # the code THEY read, never one you chose
   # → {"ok":true,"status":"joined","network_id":…,"epoch":…,"shares":["openai"]}
   ```

   The joined receipt also names the final set in words — `available here: openai`,
   plus `not available here:` lines beside it: a key the owner's person removed says so
   ("the other device chose not to share it"), while a grant that failed names
   the remedy.

   Hand them that command and the code, and let them run it. There is deliberately no
   flag on the `network` tool that completes a pairing: the digits both devices derive
   are the SAME ones, so an agent able to answer with the code it just printed would
   satisfy the very comparison that exists to catch a substitution — the interlock is
   real only because the second phase is not the agent's to run. If the user cannot
   read the code, that is the interlock working, not a problem to route around. A wrong
   code is refused with `sas_mismatch` and leaves the ceremony open (the invite is not
   spent).
   The one non-interactive spelling of the PROMPT, `--sas-stdin`, is refused unless
   `LOP_NETWORK_TEST_MODE=1` is set: it is the e2e harness's seam, and setting that
   variable to finish a real pairing would turn the human check into a formality.
   Never set it.

   The `network` tool PARKS a pairing (`action="join"` with `token`) and reports the
   code, the sentence and the window for the user; it has no `confirm` field, so no
   argument combination lets an agent answer a park. The inviter's half
   (`lop network confirm`) is deliberately NOT in that tool either: a full pairing
   always needs one person, and that is the side where their comparison decides.
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
   `complete: false` for as long as that is true — that is the honest answer, not a
   fault — and the human `ls` line carries the short form of it, WITH THE AGE OF
   THE EVIDENCE: `[members verified with 7 of 9 peer(s) (12s ago); NOT verified
   with d_9f8e7d6c5b (nothing is connected to it)]` — the oldest answer's age, and
   every member that did not answer named with its reason — or, when no table came
   back, `[members NOT verified: no table came back in the last read (12s ago) —
   retrying: d_1a2b3c4d5e (nothing is connected to it)]`. The age is when that
   read ran, and "retrying" is the cadence in the reader's words: the next pass,
   not a four-second timer (four seconds is one pull's timeout) — a snapshot, not
   a verdict, since a link is asked again once about fifteen seconds have passed
   since its last answer. `lop network status` asks the relay for a fresh pass
   before it reports (bounded: a pass that does not land within a moment is
   reported with the age it actually has, never dressed up as fresh), so its member
   block answers from a read it asked for rather than from the last cadence tick.
   And a count that NO read has fed yet says `no table read has completed yet —
   retrying` — never the failure's words, which would claim an ask nobody made.
   `--all-peers` refreshes the table before it merges, so an incomplete peer set is
   named rather than silently merged.
6. Tell the user what they now have: a relay this device supervises, an identity
   keypair other networks will address it by, and a member list they can inspect.
   The next section says what the session plane can and cannot do across the mesh.

## Who the session on the peer runs as

A create on a peer can name the agent PROFILE it runs as and the TEAM it manages, and
it may also name a legacy agent (`--agent NAME` / `--agent-id ID`) whose model and
hosting that device will use. The definitions travel with the create:
`lop network sessions --peer <id> --create --profile reviewer --team release`
reconciles those two definitions onto that device FIRST (idempotent, by name, only
what the frame mentions), so this works against a device that has never seen them —
including a bare install that was paired a minute ago.

Use `lop network definitions push [--peer <id>|--all-peers]` to bring a peer up to
date deliberately, and `lop network definitions state` to see what THIS device holds
and what it mirrored from elsewhere (agents and teams are configuration and are never
secret; a row whose text looks like a credential is withheld and named rather than
sent, and a definition is refused on arrival for the same reason).

The same deliberate half exists for MCP servers, and an offload usually needs it: a
workload that assumed a GitLab or Linear server finds none on a freshly paired peer.
`lop network mcp push [--peer <id>|--all-peers]` reconciles this device's user-scope
MCP servers onto a peer (the mesh-definitions cadence also carries them after pairing,
so a device converges on its own), and `lop network mcp state` shows what THIS device
holds and which keys a mirror still needs. **Values never travel**: `env` and
`headers` move as `${NAME}` references and per-key state (a literal value stays on
the device that wrote it), an OAuth server re-registers on the peer instead of
copying its client secret, and a mirrored server whose key the peer's store lacks is
NOT hidden — it refuses at connect, by name, and the fix is `lop secret set <KEY>`
on the device that runs it. A row whose text looks like a credential is withheld and
named rather than sent, on both ends.

WHAT A PEER MAY AND MAY NOT BE ASKED FOR, and each is refused in words rather than
quietly dropped:

- **A name that does not resolve there is refused BY NAME**, never run as the default
  agent — a session that runs the wrong thing under the right name is invisible to the
  user. The sentence names the missing name and the remedy (`definitions push`), and
  it is the PEER's own sentence, with this side's push failure appended if there was
  one.
- **`--yolo` is refused**, on both ends, with no capability that unlocks it: it would
  let one device make another run unattended with nobody there to see the approval.
  Start such a session on the machine it runs on.
- **A profile's instructions are applied when it is a role, a specialist or a package
  seed.** A legacy chat agent's own prompt is deliberately not attachable (that is the
  product's rule, not the mesh's), so such a session runs its OWN instructions on that
  agent's model — and the receipt says exactly that instead of implying the whole row
  arrived.
- **A profile outranks `--model`**, which is this product's precedence on a local
  create too (agent > flag > config). The receipt says the requested model was not
  applied and names the profile that overrode it.
- **An edited copy is never overwritten.** Each device remembers what it mirrored; if
  the local copy has been edited since, a later push REFUSES that row by name (`the
  copy of that name here has local edits`) and leaves the edit alone. A row this device
  authored itself is refused the same way.

## Which device should run this session

A session can be listed, created, warmed, stopped, archived or deleted ON a peer
over a paired mesh. From a shell:

| Command | Effect |
|---|---|
| `lop network sessions --all-peers --json` | every peer's sessions, merged; each row names the device holding it |
| `lop network sessions --peer <id\|name> --json` | one device's own catalogue |
| `lop sessions --peer <id\|name>` / `--all-peers` | the same rows through the ordinary session list |
| `lop network sessions --peer <id> --create --name <n> [--prompt <p>] [--profile <role>] [--agent <name>] [--team <name>] [--effort <level>]` | create the session ON the peer, which mints its id |
| `lop network sessions --peer <id> --engage <session>` | warm a stored session on the peer |
| `lop network sessions --peer <id> --stop <session>` | stop it where it lives |
| `lop network sessions --peer <id> --stop <session> --force` | the same stop on a target whose turn is in flight, or that will not answer its socket — it WAITS for the owner's ladder to resolve, which can be minutes (see below) |
| `lop network sessions --peer <id> --archive <session>` | hide it on the device that holds it |
| `lop network sessions --peer <id> --unarchive <session>` | restore it there |
| `lop network sessions --peer <id> --delete <session> [--yes]` | delete it where it lives — a dry run until `--yes` |
| `lop network sessions --peer <id> --send <session> <text>` | deliver a TURN to a conversation that is already there, and wait for its outcome |
| `lop network sessions --peer <id> --steer <session> <text>` | inject into the turn that session is running there |
| `lop network sessions --peer <id> --slash <session> /<command> [args]` | run a slash command in that session, ON its device |

THE THREE PILOT VERBS DRIVE A CONVERSATION THAT ALREADY EXISTS (`--create` is the
one that starts a new one). They open the same viewer the TUI's sidebar pick and `lop
--resume <a peer's id>` open, so what they act on is the OWNER's runtime: a routed
slash changes the peer's own record (a rename is visible in that device's listing),
and `--send` waits for the owner's terminal turn outcome rather than returning on
admission. The text is the positional — or stdin, for a body with newlines in it —
and, like every verb here, the tail may also be typed as `/network sessions …` from
inside a session. `--send` exits 0 only for a turn that REACHED its end; a turn that
failed there, is still running, or was queued for a retiring runtime exits 1 with
`outcome` naming which (`failed`, `running`, `queued`) so a script cannot read a
delivery as a completion.

THE TEXT IS WHAT FOLLOWS THE LAST FLAG THIS COMMAND READS, taken as-is to the end
of the command line. A prompt that talks about flags is delivered whole —
`--send <session> check the --name field` sends those five words — because nothing
after the first plain word is re-read as a flag. The boundary is the parser's own
and it is worth knowing exactly: **a token that IS one of this command's flags is
still that flag**, so `--send <session> --json is the field` delivers `is the field`
with JSON output on, and this command's own flags therefore go BEFORE the session
id. `--` is the separator for text that starts with a dash, in either direction
(`--send <session> -- --json is the field I mean`), and a flag that only describes a
NEW session — `--model`, `--hosting`, `--run-in`, `--name`, `--cwd`, `--prompt`,
`--profile`, `--team`, `--effort`, `--agent`, `--agent-name`, `--agent-id` — is
**refused** when an act is present rather than accepted and dropped, which was the
other way words went missing here (review round 2, MINOR-1).

EVERY SUCCESS RECEIPT NAMES THE DEVICE THE ACT RAN ON, and `--peer` is not that
name — the session id is what routes the act, so the receipt answers from the
session's own row and costs no extra read. `peer` is therefore the device
(`cloud-node-1`), not whatever `--peer` was typed; when the two disagree, the
caller's own word is carried beside it as `peer_named` and the human run says so in
a line of its own. A receipt you can trust to name the machine that did the work is
the point: before this, `--peer no-such-device --send <id> hello` answered
`peer: no-such-device` with exit 0 while the turn really ran elsewhere.

EVERY REFUSAL NAMES THE COMPONENT THAT CAUSED IT, so read the `code` before
acting on one (QA round 5, Q-R5-1):

- `relay_unavailable` — **this** device's relay could not be asked: it is not
  running, or it is running and wedged (the message names which, from its own
  record). The sibling `lop network peers` has always refused this way, and
  `sessions --peer` / `--all-peers` refuse in the SAME words rather than answering
  an empty list. Remedy: `lop network start`, or `lop network restart` for a wedged
  relay. An empty list from either listing means "asked, and nothing is held" —
  never "could not ask".
- `relay_refused` — the relay answered and declined; its sentence is the relay's
  own.
- `peer_unreachable` — the relay ANSWERED, and the DEVICE NAMED is the reason: it
  is not in the network, or it is in it and did not answer. An unreachable peer is a
  `doctor` question and not an error to retry (`lop network doctor --json`).
- `stop_unreported` / `engage_unreported` / `create_unreported` / `slash_unreported`
  — a receipt arrived without the field it is a receipt for, so whether the peer
  acted is UNKNOWN: restart the relay and ask again, and do not read the answer as a
  failure of the act. `slash_unreported` is the routed-slash case: the owner's
  `SlashResult` is `kind`/`text`/`style`/`data`, and one whose `style` this build
  does not know (including a receipt with no `style` at all, which used to read as
  success) cannot be read as an outcome.
- `turn_not_running` — a `--steer` reached a session that is not running a turn
  there, so there was nothing for it to correct. The sentence names the two ways
  forward (`--send` to start a turn, `--slash` to change the session); nothing was
  written to the peer.
- `session_unknown` — the named device ANSWERED and holds no such conversation (its
  own listing is the sentence's second half). Distinct from `peer_unreachable` on
  purpose: one means "ask again when it is back", the other means "that id is not
  there".
- `session_unreachable` — the peer accepted the act and then its runtime stopped
  answering WITHIN THE TIME THIS VERB ALLOWS: a bind that ran out of its own budget
  (that runtime may be starting, or wedged), or the act as a whole running out of
  `PILOT_ACT_TIMEOUT_S`. The peer's own sentence is carried verbatim, and nothing on
  this side changed. A dial that produced NO stream at all is `peer_unreachable` (or
  `relay_unavailable` with no local relay) even when the rung that noticed was the
  bind: the two codes answer "which side of the open died", not "which sentence did
  I get", because a stopped device and a stopped runtime look identical from here.

`--all-peers` merges the rows it could read and NAMES the peers it could not
(`<device>: unreachable (<reason>)` on stderr), so a partial listing is never
mistaken for a complete one.

A STOP THAT DID NOT ACT EXITS NON-ZERO. `--stop` answers `{"ok": true}` with rc 0
only for an outcome that ENDED the target: `stopped`, `killed`, `already-gone`, or
`not_running` (which is itself the answer to "did it stop"). A target whose turn is
in flight is `{"ok": false, "outcome": "skipped", "rung": "busy"}` with rc 1, a
sentence naming both ways forward, and the target LEFT UNTOUCHED — stop it again
once the turn ends, or add `--force`. The same rule covers `refused`, where the
owner could not prove the process it would signal was the one it recorded. The
`outcome` word is what says which happened; rc alone does not.

A FORCED STOP CAN TAKE MINUTES, and the wait is the owner's own ladder, not a
hang. `--force` against a target that will not answer its socket signals it and
then waits out the owner's SIGTERM grace: the rung cannot tell "draining
politely" from "wedged" while the socket is silent, so it is deliberately as
long as the `lop stop --force` you would run on the owner's own machine (about
two to three minutes). THIS SIDE WAITS IT OUT, because the alternative is what a
previous round shipped and QA measured: the signal landing while the caller was
told `peer_unreachable` with no outcome, no rung and no pid. So read the answer
that arrives — it names the rung that actually acted (`sigterm` or `sigkill`) and
`ok: true` with rc 0 — rather than interrupting the command and assuming it
failed. `--force` on a target whose socket does answer is unaffected and returns
in about a second.

MOVING A SESSION **IS** IN THIS BUILD. `lop sessions move <id> --to <peer>|local`
moves a conversation between devices, `--keep` copies it instead of moving it, and
the TUI's `/move <id> --to <peer|local> [--keep]` is the same act from inside a
session — see "Moving a session between devices". They are named here because this
paragraph used to list them as unbuilt while they worked, and a guide that tells an
agent a working verb does not exist is the same defect in the other direction.

WHAT IS **NOT** IN THIS BUILD, although the design names it: `lop exec --peer` and
`lop send --peer`. Do not retry those two hoping for a different answer: `--peer`
is a flag on `lop sessions` and on `lop network sessions`, never on `exec` or
`send`. **`/new remote <peer> [prompt]` IS in this build** (see "Which device
should run this session"): it creates, lists, warms and stops a session on a peer.
Credentials are brokered too (next section): `lop network credential share` lends
a short-lived token from the device that owns the login, so a session on a peer no
longer needs its own login for that provider — `kimi` is the one provider that can
never be lent.

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

## Moving a session between devices

A conversation moves with its own id and its transcript:

```bash
lop sessions move <id> --to <peer>          # hand it to that device; the copy HERE is retired
lop sessions move <id> --to local           # bring a remote conversation home to this device
lop sessions move <id> --to <peer> --keep   # copy it and leave the original running
```

THE DIRECTION IS THE PROTOCOL: the device that will HOLD the conversation issues
the move, so `--to <peer>` is this device asking the peer to pull and `--to local`
is this device pulling. There is no push verb.

RECEIVING A CONVERSATION IS ITS OWN CAPABILITY, AND `drive` DOES NOT HAVE IT. The
peer must hold `move` ON THE DEVICE THAT HOLDS THE WORK, or the move is refused
before anything is copied — `code: not_authorised`, naming the peer and the remedy.
A `--role drive` device (what the setup above mints, and the right default for a
laptop you only want to prompt from) may list, view, prompt, steer, stop and slash;
taking ownership of a conversation, deleting one on a peer and borrowing a login
are the three it may not. Grant one of them, on the device that owns the work:

```bash
lop network member grant <network> <device-id> move     # may take sessions from here
lop network member grant <network> <device-id> delete   # may delete sessions here
```

or pair that device with `--role admin`. The device id is the one the refusal
prints (a name will not resolve for this verb). Read the `code` before retrying:
the identical move succeeds unchanged once the grant is in place, and re-pairing to
"fix" it burns the device id instead.

THE DEFAULT IS A MOVE, NOT A COPY. Without `--keep` the conversation keeps its id
and the copy on the device it left is deleted once the handoff commits — which is
what a recall (`--to local`) means: the conversation comes home and is removed on
the remote rather than left behind as a second divergent transcript. `--keep`
forks instead: the destination mints a NEW id, copies the transcript, and leaves
the source running untouched, with the copy's origin recorded as a fork; the two
transcripts diverge from there.

WHAT TRAVELS OF A SESSION'S SCHEDULED STATE. A wake travels with the conversation,
and the destination installs and starts its wake supervisor for it as part of the
move — the receipt confirms it ("supervisor running on <device>") — so the schedule
fires there with nothing open; if that device cannot run a supervisor, the receipt
says so in one line and names the fix to run ON THAT DEVICE (`lop wake install`),
so the right machine gets repaired. **Monitor state does not travel with a move**: a
monitor's counters and snapshots are device-local observations, so their checks
start fresh on the destination and the move itself never fires an alert. Watch the
receipt's `carry` block on `--json` for both halves.

OTHER FLAGS. `--wait [SECONDS]` re-checks a conversation whose turn is in flight
every five seconds, up to the design's thirty minutes (a bare `--wait`). A session
with a turn in flight is otherwise REFUSED rather than interrupted, and the refusal
changes nothing. `--queue` is the alternative to waiting, for exactly the sessions
a plain move refuses: an attached-or-busy conversation records a durable intent on
the source (`queued`), which runs the move itself at the conversation's next safe
point and tells attached windows first (`move_pending`) so they can follow or be
disconnected. `--json` reports the record in `queue` with its phase (`queued`,
`finishing`, `paused`, `copying`, `resumed`); cancel before it starts with
`lop sessions move --cancel-queued <id>` (from `paused` on it is too late — the
runtime owns the outcome). `--from-replica` recovers THIS device's last synced copy
of a conversation as a NEW session — the path when the device that held it is gone;
`lop sessions sync <id>` is what keeps that copy fresh.

IN THE TUI the same act is `/move [<id>] --to <peer|local> [--keep] [--queue]`: it
runs the CLI and renders its phase transcript or the queue's receipt. **TWO
DIFFERENT THINGS ANSWER TO `/move`**
and they must not be confused: bare `/move` opens the working-directory picker and
`/move <path>` moves the SESSION'S DIRECTORY, both frontend-local. The mobility
form is discriminated by `--to` and by nothing else (`mesh-ui.md` §1.7 — an id can
look like a directory name and a directory name like an id, so the first token
never decides). That surface takes `--keep` and `--queue` and no other flag: there
is no `--wait` there (the refusal names the shell route for it), and the queue is
how an attached-or-busy conversation moves from the composer. Moving the session
you are IN leaves it first, because a move cannot retire a runtime this terminal is
attached to.

WHAT MOBILITY IS NOT. The desktop app has no mesh surface yet (`mesh-ui.md` §2 is
`local-operator-ui`'s). And the design's rule still holds: a session with a strong
local dependency (a repository that exists only on this machine, an attached
browser, a terminal the user is watching) should stay where it is — mobility is
for work that follows the person, not for work that follows the machine.

## Credentials on a peer

Credentials are brokered, never mirrored (decision A5). Operationally: never
copy a token or a key from one device to another, never run a login on a peer,
and never "fix" an expiry by re-authenticating for someone else. A refresh is
requested from the device that owns the credential, which lends a short-lived
access token and never its refresh token (`lop network credential share|revoke`,
`lop network credentials`).

Provider logins are shareable by name (except device-bound ones such as kimi),
and the ledger lists them beside the MCP servers: `lop network credentials`
shows a row per provider login this device holds — e.g.
`radient  oauth-rotating  login held`, with the share command right beside it —
and a peer that borrows one runs the call as that account — including
publishing and deleting agents and teams — without any login of its own. The
Radient organization login is person-scoped: share it only to your own paired
devices. If the borrowing device points at a local or staging hub, set
`RADIENT_ORG_ALLOW_NONCANONICAL_BASE=1` on that device: the check runs where
the request is sent, so setting it on the owner does nothing.

A granted share is learned by PULLING, never pushed. The owner's document changes
when `lop network credential share` runs there; this device reads it on its next
`lop network credentials`, and a key it has not read yet is a key it has never
heard of — so the provider reports `No API key configured for provider '…'`
rather than the refusal that names the owner. When a peer's operator has just
shared something and it still looks absent, run `lop network credentials` before
asking them to share it again.

A SHARE IS ALSO WIRED IN WHEN A SESSION STARTS, and that is the second reason the
same sentence can appear while the share is right there in the listing: a session
that was already running when the device's FIRST borrowable share arrived was
built while this device had nothing to borrow, so it runs without the brokering
rung for its whole life — no pull in it will ever make that key usable, and the
remedy is to start a new session (or restart that one). A session started after
the share brokers normally, and a session that was already brokering picks up a
newly shared key on the next pull without a restart. `lop network credentials`
says `note: '…' is now available to borrow on this device…` when a pull makes a
key newly borrowable, because that is the moment the distinction matters.

Brokering needs a link the BORROWER opens: the session's device dials the device
that owns the credential and runs `net_broker` over it. An owner behind NAT with
no reverse path to it — the laptop that owns the key, reached from a cloud peer —
is therefore the topology this does not survive on its own, and it does not
report itself as a link failure: with the key present in this device's document
the borrow says `No credential for '…' is reachable: <owner> owns it and was last
seen …`, and without it, the unconfigured-provider sentence above. Neither is a
broken login, and re-authenticating does not change either.
`lop network doctor --json` on the borrower is what names the endpoint and why it
did not answer.

An MCP login that has died on the owner is refused `interactive_required`, and
the fix is an interactive sign-in THERE — the one repair a borrower cannot run
for the owner. The owner's own surfaces carry the notice: `lop network doctor
--json` (and the `/network` panel) shows a `credential_repair` row naming the
login to run; it clears by itself once the next borrow succeeds.

Revocation is not instant, and an incident response must not assume it is.
`credential revoke` refuses new borrows at once; a grant already lent is dropped
by the borrower within `network.credentials.grant_ttl_s` (900 s by default); and
an OAuth bearer copied out of the borrower stays valid at the provider until the
token itself expires — to end it now, sign the account out at the provider. A
shared static API key never expires, so a copy of one lives until the key is
rotated at the provider; rotating it is the only way to end it.

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
5. `lop network ready [--peer <device>]` — readiness for offload: observed
   reachability first, then the peer's install checks (operator authority, build
   parity, git identity, MCP servers, the model and per-server MCP credentials),
   every failing row carrying its exact remedy and the side to run it on. The
   git-identity remedy suggests this device's own `user.name`/`user.email` — edit
   them if the peer should commit as someone else. It reads and creates nothing on
   either device.
6. A session on a peer is waiting for approval and it cannot be allowed from
   here — `lop network ready --peer <device>`: its `operator_authority` row says
   whether anything there can allow, and SETTING THAT UP IS SELF-INSTALLABLE:
   offer to do it rather than handing the user a terminal step.
   - **On THIS machine**, run `lop operator setup --json` — receipts `proposed →
     consent → generated → installed → verified`, all in product words — then
     `lop operator status` for the level achieved (`operator-presence` where the
     host offers a presence store, `operator-file-only` where it does not). The
     one admin prompt can be answered in the terminal it runs in; when there is
     none, the user supplies their admin password ONCE through the credential
     prompt (a masked capture, `/credential`, which reports the generated
     `LOP_SECRET_…` name it is stored under) and PROMOTES that name into the
     encrypted store the flag reads — `/credential --persist LOP_SECRET_…`.
     Both steps are required: the capture writes session memory only, and
     `--sudo-secret <name>` resolves through `lop secret get`. The value is
     never printed or logged.
   - **On a PEER**, file the request — `lop network approvals request --host
     <host> --user <user> --network <name> --json` — show the user the card
     (`lop network approvals list --json`, then `show <id> --json`), let THEM
     answer it (`lop network approvals approve <id>` signs with the operator
     key, Touch ID where the host offers it; never approve on their behalf, and
     `deny <id>` is ordinary), then run `lop network approvals run <id> --json`:
     installs/updates the build on it, lands the operator's public anchor
     root-owned, joins the member, writes the grants and supervises the relay.
     Receipts, in order: `invite`, `pre_read`, `install`, `join`, `anchor`,
     `grants`, `relay`, `verify`. A pre-read that CONTRADICTS the card halts the
     run and files a fresh request carrying the corrected facts — take that one
     to approval; never proceed on the wrong facts.
     Filing needs an operator key on THIS machine to name on the card, so on a
     fresh machine the local bullet above comes first.
   Denying a parked session works from any attached viewer. An ALLOW does not
   ride the slash: `/approvals` is terminal-only from a remote surface — the
   answer that reaches the owner is an `approval_answer` from the session's
   approval card, the same card any attached viewer shows, and a paired phone
   answers that card from its approvals panel (`Approvals in this session`).
   The `lop network approvals` cards are a different family — a device's
   onboarding request, answered with those verbs — not this card.

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
- Do not run the join step for the user, and never type a code you did not see on
  the other device. `--confirm` takes the value the USER read back from the other
  screen: never the code this device printed, never a value you derived.
- Do not print an invite token, or read the token file into a result: it is a
  single-use bearer credential, and a tool result is the most-copied text in the
  system.
- Do not add a member by editing a member list. Membership changes through
  `invite`/`join`, `member rm`, and the epoch rotation they carry.
- Do not force a full re-sync, and do not invent a `--force` on a verb that does
  not take one: the only verb in this family with a `--force` is `network sessions
  --stop`, and it means the owner's own `lop stop --force` — signal a target whose
  turn is in flight or whose socket will not answer, accepting that the turn goes
  with it.

## Reference

**Exit codes.** `0` success; `1` a refusal or a failed operation — INCLUDING a verb
that answered and did not act, such as a stop whose `outcome` is `skipped` — with
the sentence on stderr, which is the sentence to show the user (read `outcome`, not
the code alone: a receipt is not a success); `2` a usage error, including `--print`
together with `--json` and `--force` without `--stop`; `3` a human decision is
required and did not arrive — a parked pair (`join --park`) whose window closed with
nobody answering, reported as `pairing_unanswered`. Nothing is joined in that case,
the invite is untouched, and the ceremony is not resumed: park a new one.

**Commands.**

```bash
# reading
lop network status --json
lop network ls --json
lop network show <network> --json
lop network peers --json
lop network doctor --json
lop network ready --json            # --peer <device>: what a peer still needs for offload
lop network definitions state --json
lop network mcp state --json        # user-scope MCP servers, provenance, keys still needed
lop network log --json              # --follow, --limit 50, --since 15m, --export <file>

# lifecycle
lop network init <name> --json
lop network rename <network> <name> --json
lop network rm <network> --json
lop network invite --role drive --json
lop network join @<token-file>      # a terminal: prompts for the code, then joins
lop network join @<token-file> --park --json   # no terminal: park and print the code
lop network join --confirm <code> --json       # answer the parked ceremony
lop network confirm --list          # a pairing parked on THIS device, with both codes
lop network confirm <invite-id>     # answer it (--decline refuses; needs a TTY)
lop network member rm <network> <device> --json

# approvals (remote onboarding: the card, the decision, the run)
lop network approvals list --json
lop network approvals show <id> --json
lop network approvals request --host <host> --user <user> --json
lop network approvals approve <id> --json   # signs with the operator key (their gesture)
lop network approvals deny <id> --json
lop network approvals run <id> --json

# operator authority (this machine)
lop operator setup --json    # the agent-runnable self-install: key, one admin prompt,
                             # anchor installed root-owned, verified
lop operator status          # the level achieved (`operator-presence`/`operator-file-only`)

lop sessions move <id> --to <peer> --queue --json   # queued move: runs at the next safe point
lop sessions move --cancel-queued <id> --json       # cancel before the safe point passes

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

- The DESKTOP half of the mesh surfaces (`mesh-ui.md` §2) — the networks-and-
  devices view belongs to `local-operator-ui`, not to this build. The TUI half
  (§1) is built: `/network` (including `/network approvals` — the approval cards
  are readable and answerable from the composer), `/network peers`, `/new remote
  <peer>` and `/move --to` (see "Moving a session between devices").
- `--peer` on `lop exec` and on `lop send` (see "Which device should run this
  session").
- `--wait` on the TUI's `/move --to`: the slash takes `--keep` and `--queue`, so a
  busy or attached conversation is queued at the next safe point (`--queue`) or
  waited for through the CLI (`--wait`).

Session mobility (`mesh-session-mobility.md`) and credential brokering
(`mesh-credentials.md`) are BOTH in this build, as described above — they are
listed here only to say where their missing halves are, not to deny them.

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
