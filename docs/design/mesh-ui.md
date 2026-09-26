# Mesh surfaces — the TUI, the desktop, and the agent-facing half

Status: **design, pre-implementation**. This is the brief a designer and a UI
implementer work from. It **owns R6** (one session list with a local/remote
annotation, inside the desktop's existing space budget) and **R19** (the
agent-facing guide and tool — `mesh-transport-identity.md` §12.5 supplies R19's
transport half but does not claim the requirement). It is subordinate to
`docs/design/mesh-network.md` — where this document and the spine disagree about
a *mechanism*, the spine's requirements win and the divergence is called out
inline with its reason.

Provenance of every anchor:

| Tree | Revision | Used for |
|---|---|---|
| `~/local-operator-worktrees/mesh-network` (`feat/mesh-network`) | `a7e6b9bd` | TUI, CLI, guides, tools, scripts |
| `~/local-operator-ui-worktrees/mesh-network` (`feat/mesh-network`) | `305efc363` | desktop components, contracts, evidence tooling |

Both are clean checkouts; **line numbers move, symbol names do not** — re-anchor
by symbol. Where a citation is to the head of a function, the name is the
citation.

### Anchors and sibling decisions this brief defers to

The five sibling mesh designs landed while this one was written, and three of
them own facts this brief would otherwise have invented. **Where they and this
document differ, they win, and the difference is named inline:**

| Fact | Owner | Consequence here |
|---|---|---|
| `locality`, `owner_device`, `owner_device_name`, `placement`, `reachable`, `unreachable_reason`, `last_synced_at`, `origin` — the row fields | `mesh-session-mobility.md` §9.2 | §2.6 uses those names, not my first draft's `peer_id`/`peer_label`/`peer_reachable` |
| `features.peers: 1`, `GET /v1/desktop/peers`, `include_peers` on the session list, `POST /v1/desktop/sessions/{id}/transfer` | `mesh-session-mobility.md` §9.3; `mesh-transport-identity.md` §9.4 | §2.6 states the shapes and adds only the keys the sibling does not name |
| "the annotation joins the leading status-glyph slot or the tooltip; a peer is a group heading; 32px rows; the trailing slot is already spent" | the spine §8, quoted by all three | §2.1-§2.3 are the *brief* for that constraint: I keep the group heading and the tooltip, and I refine "joins the leading slot" into "a sibling mark in the leading cluster, rendered **only** on remote rows" with the measurement (and the repo's own precedent) that says why a reserved slot is the wrong reading |
| The mobility command's spelling | the spine §6, implemented by `mesh-session-mobility.md` §4.2 | §1.7 follows it (`/move <id> --to <peer\|local>`) and specifies the disambiguation the overload requires — my own recommendation (`/handoff`) is now listed as a *rejected* alternative, with the cost it avoids and the cost the overload pays |
| The agent tool is **unconditional** and shells the CLI with `--json`; the invite token never enters a tool result | `mesh-transport-identity.md` §12.5 | §3.2 adopts both and keeps this document's contribution: which operations the tool must *not* be able to complete alone, enforced in the CLI where the guards already live |

---

## 1. TUI

### 1.1 `/network` — the command family

**Decision: one grouped command, `/network`, whose argument is a declared
subcommand vocabulary** — the `/mcp` shape (`slash_commands.py` L693 region),
because a mesh has several verbs that are variations of one noun and the
argument picker enumerates them with a line of help each, which one truncating
`/help` row cannot.

Registry entry, added to `SLASH_COMMANDS` (`slash_commands.py:67`) beside
`/mobile` (`:770`), since both are "this machine's connectivity":

```python
SlashCommand(
    "network",
    "Networks, peers, and this device's mesh state",   # 46 cells, inside the ~55 /help budget
    arguments=ArgumentMode.OPTIONAL,
    argument_shape=ArgumentShape.SUBCOMMAND,
    subcommands=NETWORK_SUBCOMMANDS,        # NEW field — §1.1.2
    # NOT offered on the desktop in this pass: same call as `/mobile`
    # (desktop_destination deliberately unset) — see §2.7.
)
```

`echo=False` (the default): the listing or the receipt is the answer, and every
handler already reports what it did — the rule `/approvals`/`/rename` follow.

#### 1.1.1 Subcommands, help text, and what each shows

| Subcommand | Picker row (name — description) | What the receipt shows |
|---|---|---|
| `ls` (default: bare `/network`) | `ls` — `Networks this device is in` | a `NetworkScreen` (§1.1.3) |
| `status` | `status` — `Relay health and this device's links` | the same screen, scrolled to the relay section |
| `peers` (alias `/peers`) | `peers` — `Reachable peers right now` | the peer table (§1.2) |
| `new <name>` | `new` — `Create a network on this device` | `network <id> created` + the invite hint (the id in full, no truncation) |
| `invite` | `invite` — `Mint a single-use invite` | the token, plus `expires in 10m`, plus the SAS hint |
| `join <token>` | `join` — `Join from an invite (shows a code to confirm)` | **two-phase** (§1.4.3): phase one prints the SAS and refuses to continue without a human |
| `show <network>` | `show` — `Members, roles and endpoints` | the member table |
| `member rm <device>` | `member rm` — `Revoke a member (rotates the secret)` | a confirm, then `epoch → 8` and the audit line |
| `disconnect` | `disconnect` — `Leave; stop trusting; close links` | a confirm, then the receipt (§1.5) |
| `panic` | `panic` — `Broadcast revoke and rotate the secret` | a typed confirmation (§1.5) |
| `log` | `log` — `Recent mesh events` | the audit tail (last N, no `--follow` in the TUI) |
| `trust <network>` | `trust` — `Trust an untrusted network again` | a confirm, then the re-admission receipt (IR's `trust` verb) |
| `pool request\|ls\|cancel` | `pool` — `Ask for extra compute, list, or cancel` | the `PoolRequest` receipt (compute-pool §4.1; nothing else acts on it) |
| `doctor` | `doctor` — `Diagnose a link` | handshake/epoch/clock rows |

**One list, with the installation verbs deliberately outside it.** The CLI's
`lop network` group carries five verbs the TUI must not: `serve`, `start`, `stop`,
`restart` and `uninstall [--purge] [--purge-identity]`. They install, supervise and remove a
LaunchAgent, and a picker row that boots out the operator's relay — or deletes
this device's identity keypair and its networks — from the composer is the same
class of one-keystroke mistake §1.5 refuses for `panic`. They are therefore not in
`NETWORK_SUBCOMMANDS`: the two lists are one list for everything that acts *on a
network*, and installation is the CLI's alone (a TUI cannot even ask the
supervisor to reinstall itself safely). `lop network doctor`, when it finds no
relay, names the CLI command rather than offering to run it.

`NETWORK_SUBCOMMANDS` is a frozen tuple in `slash_commands.py` beside the
registry, because **the CLI's verb list and the TUI's must be one list** — the
same rule `MCP_SUBCOMMANDS` already follows (`session/frontend_state.py`, read
by the route, the picker and the desktop catalogue through
`command_argument_words`). Drift between the typed word a handler accepts and
the word the picker offers is a defect class this repo has already paid for.

#### 1.1.2 The one registry change this needs (and why it is not a new shape)

`ArgumentShape.SUBCOMMAND`'s vocabulary is currently hardcoded to MCP's
(`command_argument_words`, `slash_commands.py:942-960`: `if spec.argument_shape is
ArgumentShape.SUBCOMMAND: return tuple(sorted(MCP_SUBCOMMANDS))`). `/network`
needs a per-entry vocabulary, and `/new remote <peer>` needs a second shape.

**Change A — one new keyword-only field on `SlashCommand`** (`tui/autocomplete.py`,
beside `argument_shape`):

```python
#: The first-token vocabulary for ``ArgumentShape.SUBCOMMAND``. Empty keeps the
#: MCP set, which is what every existing entry declares by silence — so `/mcp`
#: does not change by one byte.
subcommands: tuple[str, ...] = field(default=(), kw_only=True)
```

and one line in `command_argument_words`:

```python
if spec.argument_shape is ArgumentShape.SUBCOMMAND:
    if spec.subcommands:
        return spec.subcommands
    return tuple(sorted(MCP_SUBCOMMANDS))
```

`_is_mcp_invocation` (`:973`) already accepts "at most two tokens, first in the
vocabulary, second a name" and returns **False** for a sentence — which is what
keeps `/network disconnect please` prose. (`/network` takes no second token
today; `member rm` is the one two-token form and it reuses the same rule.)

**Change B — one new shape** (`ArgumentShape.REMOTE_PEER = "remote_peer"`), the
exact shape of `PROVIDER` (a *live* vocabulary resolved lazily in
`command_argument_words`), for `/new remote <peer>`:

* vocabulary: `peers.known_peer_names()` — the same cached peer catalogue the
  autocomplete fills from (§1.4), so the word the desktop accepts and the word
  the picker offers cannot drift;
* validator: `_is_remote_peer(args)` — exactly two tokens, first is `remote`
  case-folded, second a peer name in the catalogue;
* refusal sentence (§1.4.2): `Choose a peer for the remote session`.

**Why a shape rather than reusing `WORD`.** `/new` is `WORD` today because one
word is the desktop picker's selection (`selected=args`), and two tokens are
prose. Without a new shape, `/new remote devon` on the desktop would be planned
as **prose** — a paid model turn for a control, which is the failure the
admission rule exists to prevent (`slash_commands.py:1037-1100`). Widening `WORD`
instead would wrongly accept `/new /tmp/x y`.

Both changes are pinned entry-by-entry by the existing tests
(`tests/unit/tui/test_slash_prefixes_text.py`, `test_slash_echo.py`), so the
implementing PR must state the new rows there rather than inherit a default.

#### 1.1.3 Where it executes: this device, not the session's owner

**Decision: `/network` is `FRONTEND_LOCAL`, and `"network"` joins
`_FRONTEND_LOCAL_SLASHES`** (`session/frontend_state.py:828`).

The reason is the set's own stated criterion, and `/info`'s entry is the exact
precedent: *every fact this command reports is about the machine the user is
sitting at* — this device's networks, this device's relay, this device's peer
links, this device's audit log. Routed to a remote session's owner it would
describe the runtime's machine while the user reads it as theirs — the
wrong-machine answer on the one screen whose job is "what is my mesh state".

Consequences, and they are the design:

* `disconnect` / `panic` are **this device's** incident controls by
  construction (R17 is a local act), not the remote runtime's;
* the commands therefore keep working when the session is cold, unfollowed, or
  remote — which is when an incident is most likely;
* `_FRONTEND_LOCAL_SLASHES` is a *complement*: the classification cannot be
  forgotten, because a new entry without one fails the test
  (`session/frontend_state.py:824-827`);
* the exception table the spine names (`mesh-session-mobility.md`) gains one
  row: `/network*` never leaves the viewer, beside `/quit`, `/resume` and the
  pickers.

#### 1.1.4 The screen

**New file: `local_operator/tui/widgets/network_panel.py`**, holding
`NetworkScreen(ModalScreen)` — modelled on `InfoScreen`
(`tui/widgets/info_panel.py`, opened by `OperatorApp._cmd_info`,
`app.py:34604`) because it is the same shape: a multi-record state with slow
facts, a two-phase capture, and a worker for the probes.

The two-phase split is copied deliberately from `/info` and for its stated
reason: the **synchronous** phase reads the relay's own state (it is in-process
memory and a couple of small files) and paints immediately; the **worker** runs
the reachability probes (TCP handshakes to peers, `rtt_ms`, epoch/clock skew)
because they are the only slow, blocking facts, and a `/new` arriving mid-probe
must not paint a new session's state under a header describing the old one.

Sections, in paint order:

```
Mesh                                     ← the screen's title (muted, like "Sessions")
  relay          running · pid 41233 · 127.0.0.1:0 · udp/tcp 7431 · since 2h14m
  device         9f2c…  "damian-mbp"      ← this device's id + label; the id is copyable
Networks
  devmesh        0e15…  role owner  epoch 7  members 3  active
  lab            77a1…  role drive  epoch 2  members 1  unreachable · since 4m
Peers
  devon-laptop   9c02…  drive  rtt 24ms  sessions 2  live
  studio-mini    41ab…  read   rtt 210ms sessions 0  live
  radient-m-4h   9f2c…  drive  pool · size m · expires 1h52m  sessions 1  live
Recent activity
  14:02:11  session_handoff_completed  b71e…  local → d_9c02…  ok
  14:01:40  member_admitted    41ab…               ok
```

Keys (only these; the screen is a read surface plus two deliberate actions):

| Key | Action |
|---|---|
| `↑`/`↓`, `pgup`/`pgdn` | move between **actionable rows only** (peers and networks); chrome rows are skipped — the same "rows, not lines" rule the sidebar's hit-testing follows |
| `enter` | peer → `/new remote <peer>` (the picker, §1.4); network → the member table for it |
| `l` | open the audit tail (the `/network log` receipt) |
| `d` | disconnect the selected network — confirmation (§1.5) |
| `P` | panic the selected network — typed confirmation (§1.5) |
| `r` | re-probe (the worker) |
| `esc` | close |

`d`/`P` are **selection-scoped**, which is what makes them safe on a screen that
can hold several networks: a bare global "panic" key would have to guess which
network, and guessing is the accident this design is avoiding. `shift+P` rather
than a lowercase key, and a typed confirmation, are the two guards — see §1.5.

Clamping, not wrapping: this is a full-page mode whose list can exceed the
viewport (the `/settings` exception in `~/local-operator/AGENTS.md`'s TUI
conventions), so its movement clamps.

### 1.2 The peer list

`/network peers` (and the screen's `Peers` section) is a **table with fixed
columns**, not a sidebar row. Columns, in order, with the width budget of a
100-column terminal:

| Column | Width | Content | Missing/degraded value |
|---|---|---|---|
| `PEER` | 16 | the member's `name` (transport's field; `label` in my earlier draft), ellipsised | `—` |
| `DEVICE` | 8 | the first 8 chars of `device_id` | never missing |
| `ROLE` | 6 | highest grant: `owner`/`admin`/`drive`/`read` | `?` (unknown, older peer) |
| `RTT` | 6 | `rtt_ms` as `24ms` / `1.2s`, dimmed past 500 ms | `—` when unreachable or `null` (then `STATE` leads with `unreachable`) |
| `SESS` | 4 | `session_count`, from the last projection | `—` when unreachable and no cached projection |
| `KIND` | 8 | `device` or `pool · m` (size class), pool rows in `accent` | `device` |
| `STATE` | rest | `live` / `unreachable · 4m` / `joining` / `draining · expires 12m` / `expired` | `unknown` |

**Unreachable is a state of the row, not an absence of the row.** Spine §7: a
session on an unreachable peer is shown as unreachable, never deleted. The same
rule governs the peer itself: a peer whose link is down keeps its row, with
`STATE` saying so and the reason in `--json`'s `unreachable_reason`. The one
thing that removes a row is revocation or `expired` past the retention window —
and even then `lop network log` can explain it.

An empty list is a sentence, not a blank: `No peers yet — /network invite mints
a token, /network join accepts one.`

### 1.3 The session sidebar: peer grouping and the locality mark

The sidebar is the surface R6 has to fit into. Its geometry, read from
`tui/widgets/session_sidebar.py`:

* width `SIDEBAR_WIDTH = 30`, cap `SIDEBAR_MAX_WIDTH = 44` (`:34`, `:51`), plus
  `SIDEBAR_GUTTER = 3` cells that are **added**, never taken from the content
  (`:117`);
* every row is composed of exactly: **columns 0-1** the cursor prefix
  (`› `, `» `, `★ ` or two spaces, `:113-118`), **column 2** the state mark
  (`row_state_mark`, `session_picker.py:920`; `_special_mark`'s `⌥` for subagent
  rows, `:410`), then a space, then the title, then the right-aligned age
  (`:119-169`);
* sections are **contiguous render rows that are never entries**
  (`_display_rows`, `:466`), named by `_SECTION_NAMES` (`:197`): `pinned`,
  `active`, `previous`, `subagent`, each with a blank above and below, and an
  empty section contributes no header;
* the title's width is `width - 4 - (len(age) + 1)` (`:152`).

**Decision 1 — peer grouping is a new section axis; the existing tiers stay.**
`_SECTION_NAMES`'s four keys become a *(rank, peer key)* pair, so a peer's rows
form one contiguous section each, placed **after `previous` and before
`subagent`**:

| rank | section key | heading label |
|---|---|---|
| 0 | `pinned` | `★ Pinned` (unchanged) |
| 1 | `active` | `Active Sessions` (unchanged) |
| 2 | `previous` | `Previous Sessions` (unchanged) |
| 3 | `peer:<device_id>` | `⇄ <label>` — `⇄ <label> (unreachable)` when the link is down |
| 4 | `subagent` | `⌥ Subagent Runs` (unchanged) |

Why this shape and not the alternatives:

* *Nested (active/previous inside each peer)* — up to 3 headers per peer; the
  page-size arithmetic (`page_size`, `_display_rows`, `_entry_at`) is shared
  with keyboard traversal and every nested header narrows the window for rows.
* *Peer as a per-row mark only, no grouping* — a fifty-row list of mixed
  devices with a glyph per row is exactly the "which of these is mine" question
  the list exists to answer, answered badly.
* *Local rows merged into peer sections too (every row under a device)* — the
  **zero-peer and single-device cases would change**, and R9/R16's regression
  requirement is that an install with no network behaves exactly as today. Under
  this design a lone device paints byte-identically to the base revision, which
  is a testable invariant and the *before* frame of the visual pair (§4.1).

**Decision 2 — the locality mark goes in the CURSOR-PREFIX slot (columns 0-1),
not the mark column.** The slot holds **two facts**: cell 0 is the caret or the
pin, cell 1 is ALWAYS the locality mark. So:

```
›⇄ ⬤ Fix the sidebar reflow…        2m      ← remote, cursor here
 ⇄ ● Review the mesh brief          1h      ← remote, idle
★⇄ ● Draft the RFC                  3d      ← remote, pinned
    ● Article-search-svc sweep       5m      ← local
```

**Why both cells, and not the precedence this section first drew** (design
round 1, D4 — settled here, with the frame `sidebar_shot-peers-focus` in §4.1).
The first cut gave cell 0 *and* cell 1 to whichever mark won:
`»`/`›` → `★` → `⇄` → two blanks. That is right about which fact leads — the
caret says "here"/"opening", the pin says "kept", and both outrank a durable
property — and wrong about the cost: the `⇄` then vanished on the **one row the
user is deciding about**, and the peer heading that carries the same fact
independently sat two lines above it on a list being scrolled. The precedence is
kept for CELL 0; cell 1 is the mark's unconditionally. Nothing grows: the slot
was reserved and its second cell was blank on these rows anyway, so the title
starts at column 4 in every one of the four states above. This also replaces the
`› ⇄ ⬤ …` example this section carried — four lead cells that a 2-cell slot
cannot hold, and which the review correctly called unrenderable.

**This diverges from one *mechanism* in the spine** (§8: "a per-row locality
mark, reusing the existing status-glyph slot") **and keeps its requirement** (a
per-row mark that does not grow the row). Column 2 belongs to the urgency ladder,
and the file says so in the same words the correction that moved `★` out of it
used: *"`row_state_mark` owns column 2 and its urgency ladder must not be
displaced by a durable property"* (`session_sidebar.py:406-436`, and the render
comment at `:106-112`). Locality is the same kind of durable property. Columns
0-1 are already reserved and are otherwise blank, so `⇄` there costs **zero new
cells and no change to the title arithmetic** (`width - 4 - …` is untouched),
and it is painted with the `muted` ink the existing prefixes use, so it cannot
be confused with a state: it never spins, never turns `danger`, and never
replaces a caret. The `_special_mark`/`_advance_spinner` pair is untouched —
column 2's contract does not move.

**Tooltip on the row** (the existing `_describe`, `:950`) gains one clause, in
this order after the state sentence: `on <peer label>` (or
`on <peer label> — unreachable: <reason>`). This is where a peer's *name* is
readable when the user has scrolled past the heading, and it costs no cells.

**Fields the sidebar needs, and where they come from.** `resume.SessionRow`
(`resume.py:1826`) gains, defaulted exactly like its existing live-state fields
so every other construction site renders as before:

```python
locality: str = ""            # "", "local", "remote"   ("" = unknown ⇒ render as local, no mark)
owner_device: str = ""        # the member's device id (mobility §9.2's name)
owner_device_name: str = ""   # the member's human name, for the heading and the tooltip
reachable: bool = True        # False ⇒ the group heading carries `(unreachable)`
placement_stale: bool = False # the projection is cache-only (owner unreachable)
```

and `session/catalog.decorate_rows` (`session/catalog.py:569`) is **the one join
point**: it already reads the live registry and the wake index for the whole
list, and it is where the peer projection is merged in (a scan of the local
relay's cached projection, not one socket per row — the projection is already a
local cache; the sidebar never blocks on a network read).

`CatalogEntry` (`session/catalog.py:37`) needs nothing new: it wraps `SessionRow`
and the `_section_of` change reads the row's `owner_device`/`owner_device_name`
(mobility §9.2's names, adopted in §2.6).

**Empty and degraded states.**

* **Zero peers / feature off:** no peer section exists, `_SECTION_NAMES`'s four
  keys and every label are what they are today. **Byte-identical frame** (§4.1).
* **Peer unreachable, rows cached:** the section renders from the cache with
  `(unreachable)` in the heading and the rows' marks in `dim`; pressing Enter on
  one opens the existing unreachable/unavailable path, whose sentence names the
  `/resume` and the last-known state rather than "session not found".
* **Peer unreachable, nothing cached:** the section is absent (an empty section
  contributes no header) and the *peer* is still visible on `/network peers` and
  on `/network`'s screen — which is the reason those exist separately from the
  sidebar.
* **Degraded projection read:** the existing `SessionRow.degraded` tuple
  (`resume.py:1930`) gains `"peers"` when the projection could not be read, and
  the sidebar's existing "could not read" treatment covers it — the same
  mechanism that exists because a swallowed registry failure once rendered as
  "Nothing running right now" (`session/catalog.py:569-606`).

### 1.4 `/new remote <peer>` with autofill

#### 1.4.1 Registry and completion

`/new` keeps `desktop_destination="sessions.new"`, and its `argument_shape`
becomes `ArgumentShape.REMOTE_PEER`. The TUI handler signature changes from
`_cmd_new(self, notice)` (`app.py:14667`) to `_cmd_new(self, arg, notice)`, with
the dispatch line at `app.py:29386` passing `arg` — today the argument is
**dropped**, which is the bug this fixes (a user typing `/new remote x` gets a
plain new session and no explanation).

Grammar:

```
/new                        → today's behaviour, byte for byte
/new <word>                 → today's behaviour (the desktop picker's selection;
                              the TUI ignores it as it does now)
/new remote <peer> [prompt] → create ON <peer>; a trailing text becomes the
                              session's first prompt (the R8 verbatim form)
/new remote                 → refusal: names the two forms
/new remote <unknown>       → refusal: "No peer named <x>. /network peers lists them."
/new remote <peer> extra words that are not a prompt → the prompt path (any tail
                              after the peer name is a prompt, because that is
                              what `/new remote devon summarise the RFC` means)
```

Autofill comes from `on_argument_query_opened` (`app.py:35658`), the one handler
that fills every command's argument list, dispatching on the word. A new arm:

```python
if message.command == "new":
    choices = [ArgumentChoice(name="remote", description="Create it on a peer")]
    # then one row per known peer, `name` the peer's label, `detail` role · rtt · sessions
    picker.set_choices(choices, prefix="remote ")   # the buffer completes to `/new remote `
```

so typing `/new ` offers `remote`, and typing `/new remote ` offers the peers.
`ArgumentChoice`'s existing `detail` column carries the live facts, exactly as
the `/stop` arm does with pids.

#### 1.4.2 The offline and stale cases, and the refusals

**A stale list is usable; an unreachable peer is refused with its reason** (spine
§8). Concretely:

| Case | The picker shows | Submitting it does |
|---|---|---|
| Peer live | `devon-laptop   drive · 24ms · 2 sessions` | creates remotely |
| Peer unreachable, cached projection | `devon-laptop   drive · unreachable · 2 sessions (cached)` — row still selectable, ink `warning` | refuses **before** opening a session: `devon-laptop is unreachable (<reason>). /network doctor devon-laptop diagnoses the link.` |
| Peer `draining`/`expired` | `radient-m-4h   pool · draining — expires 12m` | refuses: `radient-m-4h is draining; it will not accept new sessions.` |
| No peers at all | no rows, with the list's notice: `No peers yet. /network invite mints a token.` — the word `remote` is still offered, and submitting it says the same sentence | refusal, not a half-created session |
| Peer refuses after the request (grants) | — | `devon-laptop refused: <reason from the peer>` with the peer's own words, never paraphrased |

Every refusal is a **notice with `error` kind** and never a silently dropped
keystroke: the message names the peer, the reason, and the one command that
would diagnose or fix it.

#### 1.4.3 `/network join` is two-phase, because R3 needs a human

The agent-facing path (§3) must drive pairing without a TTY, and R3 needs a
human to compare a code. Resolution, and it is the interface both the CLI and
the guide use:

```sh
lop network join lop-inv-… --json
#  → {"status":"awaiting_confirmation","sas":"K7QF-2M4D","peer_name":"damian-mbp",
#     "expires_at":1789400600.0}      # exit code 3: "needs a human"
lop network join --confirm K7QF-2M4D --json
#  → {"status":"joined","device_id":"9f2c…","network_id":"n_4a1c","epoch":7}
```

Phase one writes nothing durable except the pending join (in the join state file,
0600). Phase two refuses a mismatched or expired code by name. A TTY session
gets the same two steps with the code printed and a `y/n` prompt in between;
`--yes` exists for scripts but is **not** accepted with `--confirm` omitted, so
nothing can join a network without either a human or a code the human read.

### 1.5 Incident controls: one step, never by accident

`/network disconnect` and `/network panic` are the two shapes (spine A6). Both
are typed commands — a word, not a chord — which is the "one step" half.

| Control | Reachable in one step by | Guard against accident | What it says |
|---|---|---|---|
| disconnect | `/network disconnect` (bare resolves to the only network; with several, it refuses and lists them) or `d` on the selected row of the screen | a single `y/n` confirm naming the network and the count of links that will close; `--yes` for scripts | `Left devmesh. 3 links closed, sessions on peers are now unreachable. The audit trail is kept. /network join <token> rejoins.` |
| panic | `/network panic <network>` or `shift+P` on the selected row | **typed confirmation of the network's name** (the same shape as `sessions cleanup --force` and GitHub's repo deletion), plus the screen's row selection | `Revoked the network. Secret rotated (epoch 7 → 8). Broadcast to 4 peers; 1 unreachable and will be refused at its next connect. Re-admit devices with /network invite.` |

**Why panic needs a typed name.** Panic is not reversible by the same command:
it rotates the network secret, so every *other* device must be re-invited. That
is the definition of an irreversible act, and the repo's precedent for those is
a typed confirmation, not a yes/no. The reachability requirement (R17: "must be
reachable from the TUI in one command") is satisfied by the typed command; the
typing is what stops a stray keypress.

**Placement.** The controls live (a) in the command family, always, and (b) on
the `NetworkScreen`, selection-scoped. They are **not** bound to a global chord,
and they are **not** in the sidebar's context: a binding that fires from any
screen would have to guess which network, and one that fires from the sidebar
would put a destructive affordance one keystroke from the row a user was
scanning. The screen's `d`/`P` are only live while the screen has focus, and the
footer states them (`esc close · enter open · d disconnect · P panic`).

**The audit lines are `mesh-incident-response.md` §4.3's names**, not new ones —
this surface produces the *commands*, not the vocabulary: `disconnect_initiated`
(`{epoch, reachable_peers, sessions_became_unreachable}`), `net_disconnect`,
`panic_broadcast_result` (`{sent, acked, unacked, failed, duration_ms}`),
`panic_delivered` / `panic_undelivered` (per peer) and `epoch_rotated`. A surface
that invented its own event names would put two vocabularies in one log, which is
the thing the taxonomy's single owner exists to prevent.

### 1.6 Every user-facing string this document introduces

Collected so a designer reviews copy in one place rather than hunting the doc.
(Each is sized to its tightest site; the `/help` description column wraps past
~55 cells, `slash_commands.py:96-110`.)

| Site | String |
|---|---|
| `/help` row | `/network` — `Networks, peers, and this device's mesh state` |
| picker rows | §1.1.1 table, `Picker row` column |
| sidebar headings | `⇄ <label>` · `⇄ <label> (unreachable)` |
| device with no name (both surfaces, one string) | `unnamed device` (`resume.UNNAMED_DEVICE`) |
| `/network` panel, section subtitles | `checking with the relay…` · `found by the relay` · `from this device's records` |
| `/network` panel, a peer that did not answer | `did not answer` · `no address of it answered` · `no address published for it` · `it is not a member of this network` · `it was removed from this network` (`network_panel.peer_reason_words` — the relay's protocol token, in words) |
| `/network` panel, no relay | `relay: not running on this device` |
| `/network sessions` help row | `Sessions on other devices: list, engage, stop` |
| a remote session picked from the list | `<id> is running on <device> — /network sessions --peer <device> lists it, --engage warms it, --stop ends it` |
| sidebar tooltip clause | `on <label>` · `on <label> — unreachable: <reason>` |
| `/new` refusals | `Use /new remote <peer> <prompt> to create it on a peer.` · `No peer named <x>. /network peers lists them.` · `<label> is unreachable (<reason>). /network doctor <label> diagnoses the link.` · `<label> is draining; it will not accept new sessions.` |
| move refusals | §1.7 |
| join | `Code on this device: K7QF-2M4D — compare it with the other device, then /network join --confirm K7QF-2M4D.` |
| peers empty | `No peers yet — /network invite mints a token, /network join accepts one.` |
| move done | `Moved “<title>” to devon-laptop. <n> turns, resumed there.` / `… home from devon-laptop (the copy there was deleted; --keep would have left it).` |
| move failed | `Could not move “<title>”: <refusal>. Nothing changed.` |
| move ambiguous | `Use /move <path> for a working directory, or /move <session> --to <peer> to move a session.` |

### 1.7 Mobility in the TUI — `/move <id> --to <peer|local>`

**The spelling is the spine's** (its §6 table) and `mesh-session-mobility.md`
§4.2 has landed it: `/move <path>` keeps its cwd meaning, and
`/move <id|session> --to <peer|local> [--keep]` is the mobility form. Both are
`ArgumentShape.ANY`, `ArgumentMode.OPTIONAL`, and `FRONTEND_LOCAL`
(`session/frontend_state.py` , the `"move"` entry, with its own stated reason:
the picker and the paths belong to the machine the user is sitting at).

**The disambiguation rule, stated exactly**, because one word carrying two
grammars is the defect this section exists to contain:

| Buffer | Meaning | Route |
|---|---|---|
| `/move` (bare) | the cwd picker | unchanged, `session.move` |
| `/move <path>` | change working directory | unchanged, `session.move` |
| `/move <id> --to <peer>` | move that session to a peer | mobility, `session.transfer` |
| `/move --to <peer>` (no id) | move **this** session to that peer | mobility, `session.transfer` |
| `/move <path> --to …` | ambiguous ⇒ **refused**, naming both forms | no route |
| `/move <id> --to <unknown>` | refused: `No peer named <x>. /network peers lists them.` | no route |

Three implementation consequences, each of which a coder must not miss:

1. **The discriminant is the presence of `--to`**, not the shape of the first
   token. A path can look like an id and vice versa, so nothing may guess.
2. **The desktop needs two destinations on one registry entry.** `session.move`
   executes the path today (`argsBehavior: "execute"`); the `--to` form must
   reach `POST /v1/desktop/sessions/{id}/transfer` (`mesh-session-mobility.md`
   §9.3; §2.6 here). Recommendation: keep the entry's `desktop_destination`
   as `session.move` and let *that* destination resolve `--to` by forwarding —
   one entry, one destination, the branch inside the adapter where it can see the
   args. (The alternative — a second entry, or a destination that is a function
   of args — either splits the command in the registry or moves a host-specific
   fact out of the field documented as holding exactly one.)
3. **The face of the command is its grammar.** `/move --to` with no id is the
   form the operator reaches for when shedding load ("move this heavy session")
   and it must work: the spine's own story is "move my heavy session over", not
   "go and find its id".

`--keep` means fork-and-copy (the source stays); without it the source is
retired (R11). Nothing is reversible-by-typing-the-other-way alone, so the
receipt states what happened on **both** ends, and `/move --to local --keep` is
the sentence to expect when the operator is unsure.

**Rejected alternative, and why it lost.** I recommended a separate `/handoff`
in an earlier draft: one word, one grammar, no ambiguity to reason about. It
loses on the authority of the spine (§6 names `/move`) and on what the siblings
have already specified and tested against. What it would have bought is the
table above; what the overload pays is that table — a parse rule, the ambiguity
refusal, and one destination that branches. That is a price worth naming and
paying, and it *would* become the wrong call the moment a third meaning for
`/move` is proposed: two grammars under one word is a documented decision, three
is a design smell, and the entry's own comment should say so.

The CLI keeps the spine's name and its own namespace —
`lop sessions move <session> --to <peer|local> [--keep] [--json]` — because the
only `lop sessions` subcommand today is `cleanup` (`cli.py:697`), so nothing
collides there.

## 2. Desktop (`local-operator-ui`)

Anchors below are from `~/local-operator-ui-worktrees/mesh-network` @
`305efc363` (`feat/mesh-network`, clean, cut from `origin/main`).

### 2.1 What the sidebar may spend, measured

From the tree and from the design history recorded in it:

* the sidebar is a dragged pixel width, **default 280, clamped 240-360**
  (`src/renderer/src/shared/components/common/chat-layout.tsx`, the
  `ChatLayout` width clamp);
* a session row is `flex h-8 … gap-1 px-1`, i.e. **32px tall with a 4px gap**
  between its children (`chat-sidebar.tsx`, the `rowStyle` constant);
* a row is: the leading status cluster (`<ChatSessionStatus row={row} />`, a
  `size-4 shrink-0` wrapper = **16px**), the truncating title (`flex-1
  truncate`), and **exactly one trailing statement** decided by
  `rowTrailingStatement` (`src/renderer/src/features/chat/chat-search.ts`,
  `rowTrailingStatement`), whose slot is already spoken for by the search mark,
  `· Not sent yet`, or the binding;
* that trailing slot is **capped at one claim by design**, and the docstring
  records what fighting it cost: a single truncating span rendered an orphan
  `·`; unbounded slots starved the title to 38px of 179px; a floor then
  overflowed the row at 240px. Measured title budget in that analysis: **179px
  at the default width**;
* section headings are the `heading(key, label, initial, count?, action?,
  toggleRef?)` primitive — `h-7`, collapsible, `label + count` already rendered,
  and **zero pins renders nothing** (no heading, no empty section).

**Therefore** (and this is the budget argument the design round must be able to
check):

1. **No new control in the trailing slot.** It is capped at one claim and the
   three existing claims already lose ties.
2. **No reserved leading slot.** A permanently reserved second icon costs
   **20px of a 179px title budget ≈ 11%** on *every* row to say something true
   of *no* row in the common case. The repo's own rule (from the search mark)
   is the alternative: *render the mark on the rows that carry it, never reserve
   it list-wide*. So the remote mark is rendered **only on remote rows**, which
   cost 20px each, and a local row's geometry is **unchanged by this change**.
3. **A peer is a group, not a column.** Peer sections reuse the existing
   `heading()` primitive with `count`, so the label+count shape costs no new
   widget and no new width.

### 2.2 Files to touch

| File | Change |
|---|---|
| `src/renderer/src/features/chat/components/chat-sidebar.tsx` | one section per peer after `Previous chats`; the leading remote mark inside `sessionRow`; the heading tooltip; the `Peers` group (§2.4) |
| `src/renderer/src/features/chat/components/chat-remote-mark.tsx` **(new)** | `ChatRemoteMark({row})` — the 16px locality glyph, its `title`, and its `sr-only` sentence, modelled on `chat-session-status.tsx` (which is the precedent for "icon `aria-hidden`, meaning in an `sr-only` span, `title` for the hover case") |
| `src/renderer/src/shared/store/canonical-sessions-store.ts` | `CanonicalSessionRow` gains `locality`, `owner_device`, `owner_device_name`, `reachable`, `unreachable_reason`, `last_synced_at`, `placement`, `origin` (§2.6) |
| `src/renderer/src/features/chat/peers-store.ts` **(new)** | a small zustand store over `GET /v1/desktop/peers` — deliberately **not** folded into `canonical-sessions-store`: sessions and peers have different fetch cadences, different failure modes, and folding them would make one backend error blank both surfaces |
| `src/renderer/src/features/chat/sidebar-catalogue-gate.ts` | nothing; the peers surface is gated by its own key at its own call site (§2.6) rather than by widening this gate's four inputs |
| `src/shared/desktop-session-contract.ts` | `SessionCatalogueRow` gains mobility §9.2's fields; new `PeerRow`/`PeerList` shapes as §2.6 states them |
| `src/shared/desktop-contract.ts` | the control/documentation half (`docs/desktop-controls.md` in that repo is the spec to extend) |
| `src/renderer/src/features/chat/components/chat-sidebar-*.stories.tsx` | the state matrix (§2.5) as Storybook cases — the evidence tool photographs stories, not the app |

**Not touched, and that is a design statement:** the transport. All of a peer's
sessions arrive through **the one backend the app is already talking to**
(`src/main/desktop-transport.ts` is one `backendUrl` per app). The renderer
never dials a peer, never learns a peer's address, and holds no mesh credential.
A UI that could reach a peer directly would need the mesh's authorisation model
re-implemented in JavaScript — the exact thing the relay exists to avoid.

### 2.3 The annotation: a leading mark on remote rows, plus the group

**Decision.** A remote session row renders `ChatRemoteMark` as the **first**
child of the row, before `ChatSessionStatus`; a local row renders nothing there.
The mark is a single `size-4 shrink-0` icon, `aria-hidden`, with the meaning in
an `sr-only` span and the peer's name in the row's `title`. In the sectioned
list the governing heading is the peer's; the mark is what survives scrolling
past the heading and is the **only** annotation the flat/search list has — which
is why it is not dropped as "redundant with the heading". One mechanism, both
lists.

Rejected alternatives, with the reason:

| Rejected | Why |
|---|---|
| A per-row trailing statement (`· devon-laptop`) | The slot is capped at ONE claim and is already the search mark / `· Not sent yet` / the binding. A fourth claimant re-opens the layout the docstring says took five rounds to settle. |
| A second icon on **every** row, local included | Costs 20px ≈ 11% of the 179px title budget on rows that carry no such fact, and the repo's search-mark history is precisely "do not reserve list-wide". |
| An ink or weight change on remote titles | Free in cells, but weight is already the unread signal and ink is already the state signal; a third meaning for the same channel is a colour-blindness and a consistency defect at once. |
| A separate "Remote chats" list | Splits the listing the user is scanning (R6's "one session list") and loses the interleaving that makes "which of my chats is where" answerable. |
| Inferring locality from an id shape or a title prefix | Spine §8 explicitly refuses it: the row gains an explicit `locality` field. |

### 2.4 The peer group heading, and the `Peers` group

**Sections, in paint order** (when the mesh feature is unavailable or no peers
are configured, the first three are the entire list — today's rendering):

```
Pinned chats                     ← unchanged (only when pinned.length > 0)
Active chats                     ← local rows only
Previous chats                   ← local rows only
  ⇄ devon-laptop            2    ← one section per peer with rows, count = its rows
  ⇄ studio-mini             0    ← rendered even at 0, because the count says which
                                    peer is quiet, and there is no other place to say it
Peers                            ← collapsed by default (§below)
```

`heading("peer:" + peer.device_id, `⇄ ${peer.name}`, false, n)` — the existing
primitive, `initial=false` (collapsed, like `Previous chats`), with the count
from the rows the store holds for that peer. The **key** is the device id and the
**label** is the name: two devices may carry the same human name, and a section
keyed on the label would merge them. The label carries the peer's name
and its state when it is not live:

* live: `⇄ devon-laptop`
* unreachable: `⇄ studio-mini · unreachable` (label ink stays `ink-muted`; the
  suffix is what changes)
* `draining` / `expiring`: `⇄ radient-m-4h · draining — 12m`

**The `Peers` group** is the one addition to the entity list (where `Agents` and
`Teams` already live), rendered **only when at least one peer exists**, collapsed
by default, one `h-8` row per peer:

| Row | Content |
|---|---|
| `devon-laptop` | a locality/state glyph, the label, then the trailing statement — this row's trailing slot is **its own** (peers are a different list from chats, so its slot has no competing claim): `24ms · 2 chats` |
| unreachable peer | `unreachable · last seen 4m ago` in `text-warning` |
| pool member | `pool · size m · expires 1h52m` |
| a peer row's action | `Move a chat here…` — opens a picker of **local** sessions and calls the transfer route (§2.6). Mounted only when `features.session_transfer` is advertised |

Why a peers list at all, when the section headings already name peers: a peer
with **no cached sessions** would otherwise be invisible, and an unreachable peer
is exactly the thing the user wants to see. This also gives session *movement* an
entry point that does not add a hover control to every chat row (rejected: a
per-row "move" affordance — a third control in a row that has room for one, on a
gesture that is rare).

### 2.5 The states to design

Each is a Storybook case for `chat-sidebar` (or `peers-*`) so
`scripts/capture-evidence.mjs` can photograph it, and each must be *looked at*
in every theme per that repo's review process.

| # | State | Renders | The sentence/annotation |
|---|---|---|---|
| S1 | **Zero peers** (mesh unavailable or one device) | exactly today's sidebar | nothing — the *before* frame |
| S2 | **One peer, live, with 2 sessions** | one `⇄` section + 2 marked rows | heading `⇄ devon-laptop` |
| S3 | **Several peers** | 2-3 `⇄` sections + the collapsed `Peers` group | counts per peer |
| S4 | **Peer unreachable (rows cached)** | the section with `· unreachable`, marks in `dim`, the `Peers` row leading with `unreachable` | row `title`: `… on devon-laptop — unreachable: link down 4m ago` |
| S5 | **Peer refuses** (a create/move was rejected) | rows unchanged; a `role="alert"` notice above the list, with the peer's own reason | `devon-laptop refused: not permitted to open sessions` |
| S6 | **Session moving** | the row stays **in place**, `aria-busy`, dimmed (e.g. `opacity-60`), its `title` saying `Moving to devon-laptop…`; the peer heading's count is `aria-live="polite"` | no spinner in the status slot (it is the backend's) and no second trailing statement |
| S7 | **Move failed** | the row un-dims in place, the notice is `role="alert"` | `Could not move “<title>”: <reason>. Nothing changed.` |

Notes that matter to a designer:

* **S6/S7 must not move the row.** A row that jumps to another section before
  the operation lands, and back on failure, is two lies in a row; the section
  change happens once, on success, as part of the same store update that carries
  the new `locality`.
* **S4's dim must not read as disabled.** `dim` is the app's "not live"
  treatment; on an unreachable peer's *cached* row the state the row is telling
  is its truth as of the last sync, so its status glyph keeps its own ink and
  only the remote mark dims.
* **The `⇄` mark and the group heading must agree in every state**, including
  S4, where both say unreachable. A frame where they disagree is a defect in the
  frame, not a nuance.

### 2.6 The contract the UI needs for these states to exist

The field names and route shapes are **`mesh-session-mobility.md` §9.2-9.3's**,
adopted here verbatim; this section states what each one has to be for the states
above to render, and adds the two capability keys that document does not name.
All of it is additive, and it follows the repo's pinned rule for row fields:
**the new fields are present with both values on every row**, for the reason
`pinned` documents at length in `src/shared/desktop-session-contract.ts` (the
store's row merge is `{...current, ...incoming}` under "an absent key is not a
claim", so an omitted key leaves a stale value immortal). The transport mirror is
`local_operator/server/models/desktop_sessions.py`'s `SessionRow` (`:14`), whose
`pinned` comment (`:28-42`) states the same rule from the backend side.

**Capability keys** (added to the `features` map in
`local_operator/server/routes/capabilities.py`, the negotiation point
`docs/DESKTOP_API.md` describes):

```jsonc
"features": {
  …,
  // mesh-session-mobility §9.3's key, unchanged: the peer catalogue route, the
  // `locality`/`owner_device*` fields on every catalogue row, and the
  // `include_peers` parameter on the session list. ABSENT ⇒ no peer sections, no
  // remote marks, and the `Peers` group is NOT MOUNTED (not mounted-disabled: a
  // reserved empty section advertises a feature the user does not have, which is
  // the argument `session_pins` already makes).
  "peers": 1,
  // THIS DOCUMENT'S ADDITION, and it needs stating because §9.3's route table
  // names the transfer route without naming a key. A backend can show a peer's
  // sessions and be unable to move one; on `peers` alone the renderer would draw
  // the `Move a chat here…` row and 404. Own key, not a bump, by the rule
  // `session_search`/`session_interrupt` state: absent ⇒ the control is not
  // mounted at all.
  "session_transfer": 1
}
```

**The peer catalogue** (`GET /v1/desktop/peers`, required by §2.4's `Peers`
group and by the offline-safe completion cache of §1.4). One row per member, and
the four fields the UI cannot do without are marked:

```jsonc
// GET /v1/desktop/peers  → CRUDResponse<PeerList>
{"peers": [
   {"device_id": "9c02…",        // REQUIRED: the group's React key and the store's identity
    "name": "devon-laptop",      // REQUIRED: the group heading's label ("" ⇒ render the id's tail)
    "kind": "device",            // "device" | "pool"
    "lifecycle": "active",       // transport's vocabulary: provisioning|joining|active|draining|expired
    "reachable": true,           // REQUIRED: drives the `· unreachable` suffix and the mark's ink
    "rtt_ms": 24,                // null when unreachable (the row shows `—`, never `0ms`)
    "role": "drive",             // the grant summary, for the row's detail line
    "session_count": 2,          // REQUIRED: the heading's count and the empty-section decision
    "last_seen_at": 1789400123.4,// null if never — the `last seen 4m ago` sentence
    "unreachable_reason": "",    // the peer's own words; shown in the group's tooltip and in S5
    "size_class": "m",           // pool members only, else ""
    "expires_at": 1789412400.0}  // pool members only, else null
 ],
 "degraded": []}                 // the same `degraded` vocabulary the session list uses
```

**The session row delta** (both `SessionRow` and `SessionSearchRow`, and their
TypeScript mirrors), exactly `mesh-session-mobility.md` §9.2's list:

```jsonc
{
  "id": "b71e…", "name": "…", "mtime": 1789400000.0, "preview": "…",
  "live_state": "idle", "pending": null, "active": true, "pinned": false,
  "locality": "remote",            // "local"|"remote" — REQUIRED, both values, never inferred
  "owner_device": "9c02…",         // "" when local — the group's React key
  "owner_device_name": "devon-laptop", // "" when local — the heading label and the tooltip
  "reachable": true,               // true for local; for remote, whether the owner answered THIS poll
  "unreachable_reason": "",        // one sentence when reachable is false
  "last_synced_at": null,          // for a --keep copy at this device: when it last pulled (R22's visible half)
  "placement": {"mode": "peer", "network_id": "n_4a1c", "home_device": "9c02…", "policy": "pinned"},
  "origin": null                   // {"kind": "moved"|"fork", "source_device": "…", "source_session_id": "…"}
}
```

Three rules the renderer depends on, each with a failure it prevents:

* **`locality` is a field, never a derivation.** The UI must not infer remoteness
  from an id shape, a title prefix, or a cwd (spine §8). The one place it reads
  it is the row, so the mark and the group cannot disagree.
* **The peers' `device_id` is the key, the `name` is the label.** Two devices may
  present the same human name (a user can name both laptops "macbook"); grouping
  on the label would merge them, and the merged section would then be wrong about
  both.
* **Absent means a pre-mesh backend**, and the tree renders exactly what it
  renders today — which is the S1 story and the `peers`-absent path in the same
  code.

**Creation and movement:**

```jsonc
// POST /v1/desktop/sessions    — `CreateSession` gains ONE optional field, exactly
// as `model` did ("omitted or null ⇒ today's behaviour, byte for byte")
{"request_id": "…", "cwd": "/Users/damian/work", "peer": "9c02…"}

// POST /v1/desktop/sessions/{id}/transfer      (session_transfer: 1)
// the route `mesh-session-mobility.md` §9.3 names; its body there is
// {peer, keep, wait_s} and it answers with a PHASE TRANSCRIPT, because a move is
// seconds long and must show progress rather than park the messages endpoint.
// This document's additions are the two fields the sidebar renders from it:
//   each phase carries {"phase": "…", "peer": "9c02…", "progress": 0.0..1.0}
//   the terminal phase carries {"locality": "remote", "owner_device": "9c02…",
//                               "source_retired": true}
// A refusal is a 409 with the reason ("fenced"|"in_flight_turn"|"unreachable"|
// "occupied", mobility §6), and it leaves nothing changed — the sentence S7
// shows is the route's own `message`, never a paraphrase.
```

Both mutating routes are **idempotent by `request_id`** like every other desktop
write, and the transfer route's phase transcript is what makes S6 provable: the
row's `aria-busy` dimming is driven by the phase stream, not by an optimistic
guess, so S6 and S7 are two ends of one mechanism rather than two invented states.

### 2.7 What is deliberately not on the desktop

* **Panic and disconnect.** R17 needs a straightforward way to stop the mesh;
  it gets one on the TUI and the CLI, where the operation is typed and audited.
  A desktop panic button is a one-click irreversible act (it rotates the network
  secret) sitting behind no confirmation flow, in a window the user may have
  opened by mistake; the desktop is also the surface most likely to be driven by
  an agent or a screen-share. Rejected in this pass; if it is ever wanted it
  wants its own design round and its own typed confirmation.
* **A per-row "move" control** — §2.4's `Peers` row owns the gesture.
* **`/network` in the desktop command palette** — `desktop_destination` is left
  unset on every entry of the family (§1.1), following `/mobile`'s precedent and
  its reason ("offered-but-broken is the worst of the three options"): there is
  no proxy route behind `/v1/desktop` for mesh lifecycle in this pass, so the row
  would be discoverable and dead-end.

  **Delta to record in `mesh-incident-response.md`.** Its module table lists
  "`/network disconnect`, `/network panic`, `/network log`, `/network trust`
  entries with `desktop_destination`s (surface detail in `mesh-ui.md`)". This
  document is that surface detail, and it withholds all four: a destination the
  renderer has no adapter for is the exact defect `/mobile`'s entry records at
  length. The read-only verbs `ls`/`status`/`peers`/`log` are the ones a later
  desktop panel would want, and *that* PR sets their destinations — with the
  adapter in the same change. Withholding them now costs a later edit to the
  registry; offering them now costs a user a dead row in the palette.

---

### 2.8 The networks-and-devices tab (added 2026-09-21, R6's second surface)

**The requirement, as the operator stated it:** networks *and* devices must be
listable, and the desktop gets **a tab showing networks and devices with an
infrastructure-style view** — a Grafana-like node graph where the nodes of each
network are visible, **hovering a node shows its details and status**, a device
can be **removed from a network**, and a device can be **moved into one or more
networks**. `§2.1-§2.7` above are the chat sidebar's brief; this section is the
second surface R6 owns, and it is recorded here — not in a new document — because
R6's detail lives here and a requirement split across two files is a requirement
one of them will lose.

**Decision 1 — a TAB, not a section of the chat sidebar.** The node graph answers
a question the sidebar cannot ("what is the shape of my mesh"), it needs a canvas
the 240-360 px rail does not have (§2.1), and its natural gesture is a POINTER
hover — a sidebar row's hover is a tooltip over a list that already owns the
vertical space. One tab, one canvas, no change to `§2.1`'s measured budget.

**Decision 2 — the graph's nodes are networks and devices; an edge is a
membership.** So a device in two networks is ONE node with TWO edges, which is
exactly the "one or more networks" the operator asked for and the reason the
model question below is not academic.

| Node | Identity (React key) | Label | Detail card on hover |
|---|---|---|---|
| network | `network_id` | `name`, else the id's tail | epoch, trust, member count, `membership.table.sentence` (the table's provenance, as `lop network show` prints it) |
| device | `device_id` | `name`, else the id's tail | role, capabilities, `active`, `suspect`, `endpoints` and `last_seen_at` (from `net_show`'s per-network `members_detail`), plus `reachable` and its `reason` (from `net_peer_ls`) |

**What the hover card CANNOT show today, stated so nobody designs around a
field that is not there.** A latency figure (`rtt_ms`) and a per-device session
count are the two obvious candidates a Grafana-shaped view invites, and neither
is published: `PeerFacts` (``network/projection.py``) carries
``device_id|name|network_id|reachable|reason|age_s`` and nothing else, and the
member row adds endpoints and the last-seen stamp. Both are additions to the
transport's answer rather than to this tab, so the card is designed around what
`net_show`/`net_peer_ls` already return and the two extras ride the same
follow-up as the per-device `networks: [...]` list of §2.8.1.

**Decision 3 — removal is `member rm`, and it is destructive.** "Remove a device
from a network" is the CLI's `lop network member rm <network> <device>`, which
revokes the member, tombstones it and **rotates the network secret** (bumping the
epoch). It is the one act in this tab that changes other devices' state, so it
takes the typed confirmation §1.5 specifies for `panic` and never a one-click
button; and D11 stands unchanged — `panic`/`disconnect` stay off the desktop
entirely.

**Decision 4 — "move a device into a network" is an INVITE, not an add.** The
model has no unilateral "add a member" operation, and that is deliberate (R2/R5):
admission is two-sided, and the joining device is the one that proves the SAS
(§1.4.3). The tab's gesture is therefore **"invite this device to &lt;network&gt;"**
— `lop network invite --network <n> [--device <id>]` — which yields a single-use
token that must be delivered out of band and redeemed on the device. A tab that
offered "move" as a drag would promise an act the protocol refuses; the affordance
is a token, the copy says so, and the hover card for the target device shows
whether it is already a member of the network under the cursor.

#### 2.8.1 Can a device be in several networks? Yes — but the catalogue cannot say so

**The membership model expresses it.** A network is one `NetworkRecord`, and
membership is a `MemberRecord` **inside** it (`network/types.py:588`, `:402`;
`store.list_networks` returns one record per membership). Two networks on one
device are two records, each with its own member list, so the same `device_id`
appearing in both lists is the ordinary case — that is what "in one or more
networks" means here, and no migration or schema change is needed for it.

**The peer catalogue cannot report it, and this is the fact a UI implementer will
otherwise discover at render time.** The route the graph would read is
`net_peer_ls`/`federated_rows` (`relay.py`, `_fan_out_catalog`), whose peer block
map is keyed by a flat string:

```python
# relay.py, _fan_out_catalog — iterating networks × active_members()
peers.setdefault(member.device_id, block)      # unreachable path
peers[member.device_id] = block                # answered path
```

so a device in two networks is reported **once**, carrying whichever network's
block won the iteration, and the second membership is silent. `PeerFacts`
(`network/projection.py:136`) inherits the same collapse: one `network_id` per
device.

**What has to change, and the two routes to it.** For the *read* path — which is
all this tab needs — the graph should build its edges from `net_show --json`,
whose `members_detail` is already per-network (`{device_id, name, role,
capabilities, active, endpoints, suspect}`) and whose provenance sentence is the
one the tab's network node shows. That is one call per network, needs no wire
change, and cannot mis-file a membership. The alternative — a per-(device,
network) pairing or a `networks: [...]` list on the peer block — is what a later
surface needs if it wants **per-network reachability and rtt for one device** in
a single call; that is a transport-slice change and a route of its own
(`net_topology` is the shape), so it is recorded here as the follow-up rather
than attempted in this pass. The tab ships on N calls first and the collapse is
named in the code beside the join.

#### 2.8.2 Where this tab's data comes from, and what the tree does not have yet

| Need | Owner today | Note |
|---|---|---|
| networks on this device | `lop network ls --json` | `networks[]` with `name`/`network_id`/`epoch`/`role`/`members`/`trust`/`stale` |
| one network's members | `lop network show <net> --json` | `members_detail` is already per-device (see §2.8.1) |
| a device's live status | `lop network peers --json` | flat, collapsed by device — usable for the hover card's `reachable`/`reason`, not for the edges |
| invite / remove | `lop network invite`, `lop network member rm` | both already exist; removal takes the typed confirmation |

**Two assumptions of §1.1/§1.4 that this tree does not satisfy, recorded so they
are not re-invented per implementer.** (1) `peers.known_peer_names()`
(§1.1.2, §1.4.1) **does not exist**; the offline vocabulary the TUI's `/new`
autofill reads is `network/peers.py`'s `known_peer_names()`, a disk-only read of
the member lists (it must not dial: the picker opens on a keystroke, and §1.4.2
requires a STALE list to still be usable). (2) The `pool` row in §1.1.1 has no
CLI parser behind it — `mesh-compute-pool.md` is not implemented — so it is not in
`NETWORK_SUBCOMMANDS`; a picker row for a verb the CLI lacks is the
offered-but-broken defect `/mobile` records.

#### 2.8.3 The two predicate corrections this document needed

Both were found while implementing §1.1.2/§1.4.1, and both are corrections to a
rule this document had written down, not new decisions:

1. **`/new`'s shape must still accept the single word.** §1.1.2's `_is_remote_peer`
   is stated as "exactly two tokens", which would drop `/new <word>` — §1.4.1's
   own first grammar row, "today's behaviour", and the desktop picker's
   `selected=args` — out of `command_argument_is_used`, i.e. plan `/new foo` as
   PROSE. The predicate is a superset: one token (the legacy selection, exactly
   `WORD`'s rule) **or** two tokens with the first `remote` case-folded.

   **The second token is NOT resolved, and this sentence used to say it was**
   ("the second a name in the vocabulary" — corrected in review round 4, MINOR
   4). `_is_remote_peer` deliberately never asks whether the name exists: a name
   this device has not paired with is the HANDLER's refusal, with its own
   sentence naming `/network peers`, while a predicate that demanded vocabulary
   membership would put a disk read of every member list on the admission path —
   the path every slash command crosses. A later reader reconciling the code with
   the old sentence would add that check and turn a good refusal into prose.
2. **A declared subcommand vocabulary keeps `SUBCOMMAND`'s own token rule, and
   that rule's boundary is worth stating exactly.** `command_argument_words`
   returns `NETWORK_SUBCOMMANDS` when a row declares one (falling back to
   `MCP_SUBCOMMANDS` when it does not, so `/mcp` is unchanged by one byte), and
   `_is_mcp_invocation` still decides. Its boundary is NOT "one token": it is
   **at most two, the second name-shaped** (`SERVER_NAME_RE`), so
   `/network ls` and `/network doctor devon-laptop` are the command's argument
   while `/network disconnect please` also is — the second token is a plausible
   server name, and this shape reuses that predicate rather than growing a second
   one for one family. **The prose boundary is therefore the THIRD token** (or a
   punctuation-shaped second one): `/network member rm <network> <device>` is
   PROSE to the desktop's admission rule, which is harmless while the desktop has
   no `/network` query behind `/v1/desktop` (§2.7) and is the thing to revisit in
   the same PR that gives this family a destination — with the choice stated
   there: one shape of its own, or a vocabulary that publishes its arity.

---

## 3. The agent-facing half of R19

### 3.1 `local_operator/guides/network/GUIDE.md`

Path and discovery: guides are directories with a `GUIDE.md` carrying
`name:`/`description:` frontmatter, auto-discovered by
`local_operator/guides/discovery.py` (which skips a directory with no `GUIDE.md`
or no description, and never breaks a session over a malformed one). The
description is the routing signal the model sees; the body stays out of context
until `guide://network` is read. Template: `local_operator/guides/mobile/GUIDE.md`.

```yaml
---
name: network
description: Join this device to a lop mesh network, list and drive sessions on peers, move a session between devices, and respond to a mesh incident (disconnect or panic). Use when the user asks to pair machines, work on another computer's session, or shed load onto a peer.
---
```

Section outline, in this order (each heading is a working section, not a
reference — the mobile guide's shape, which opens with "When the user asks to set
it up" and *does the work*):

1. **`# Network: run and reach sessions on your other devices`** — two sentences
   of what a network is and the one rule that matters (`lop network` is about
   *this* device; a session's owner is the device it runs on).
2. **`## When the user asks to set it up`** — the numbered default sequence:
   check `lop network status --json`; `lop network init <name> --json` on one
   device; `lop network invite --role drive --json` and hand the token over out
   of band; on the other device `lop network join <token> --json` **and stop
   there** — the SAS must be read by the human on both devices, so phase two is
   `lop network join --confirm <sas> --json` run *by the user* or after they have
   read it back. Ends with the verification step (`lop network peers --json` →
   `reachable: true` for both).
3. **`## Which device should run this session`** — the decision the agent is
   actually being asked to make: create remotely (`/new remote <peer>`, or
   `lop exec --peer <peer> "…"`), or move an existing one
   (`lop sessions move <session> --to <peer>`; `--to local` brings it home;
   `--keep` leaves the source). Includes the sentence that a session with a
   strong local dependency (a repo only on this machine, an attached browser)
   should stay put.
4. **`## Driving a session on a peer`** — the commands that work identically
   against a remote session (`lop sessions --all-peers --json`, `lop send --peer
   <peer> <session> …`, and the TUI's slash commands, which run on the owner).
   States explicitly that quitting the local TUI does not stop it (R9).
5. **`## Credentials on a peer`** — the A5 rule in operational words: the agent
   never copies a token, never runs a login on a peer, and never re-authenticates
   to "fix" an expiry — a refresh is requested from the owning device.
6. **`## When something looks wrong`** — the diagnostic order:
   `lop network doctor --json` → `lop network log --since 1h --json` →
   `lop network peers --json` → `lop network status --json`, with what each
   answers and the two failures that are *not* errors (a peer unreachable: the
   sessions are still listed; a session marked unreachable: not deleted).
7. **`## Incident: stop the network`** — `lop network disconnect` and
   `lop network panic`, what each does, that panic rotates the secret so every
   other device must be re-invited, and the **rule for the agent**: run neither
   without an explicit instruction that names the network and the action, and
   never as a retry after a failed command.
8. **`## What never to do`** — do not edit `~/.local-operator` mesh files by
   hand; do not run `join --confirm` on a code the user has not read back; do not
   `--force` a full re-sync; do not add a peer to a network by editing a member
   list.
9. **`## Reference`** — the `--json` shapes (§3.3), the file locations
   (`<config>/peers/…`), and the exit codes (`3` = a human decision is required).

The guide instructs the agent to drive the CLI **with `--json`** in every step,
because the agent path parses it — the same contract the mobile group states
("Every action takes `--json`, following the mobile group's contract, because the
agent path drives the CLI and parses it", `mesh-network.md` §6).

### 3.2 The agent tool: `network`

**The tool's existence is `mesh-transport-identity.md` §12.5's decision, and it is
the right one: `network` is an UNCONDITIONAL entry** in `TOOL_BUILDERS` /
`DEFAULT_TOOL_NAMES` (`tools/registry.py:32-97`), not a `createIf`-gated factory —
because an agent must be able to *create the first network*, and a gate on "a
relay exists" would strip the tool from exactly the session that has to run
`lop network init`. It shells the CLI with `--json` (§12.5), and its actions
mirror that document's op list.

```python
# local_operator/tools/registry.py — TOOL_BUILDERS (:32) and DEFAULT_TOOL_NAMES (:70)
"network": lambda context: build_network_tool(context),
```

**When `build_network_tool` returns `None`: never — and that is a decision, not an
omission.** The ladder's rung-3 shape (`wake` with no scheduler, `browser` with no
browser surface, `secret` with an unreachable store) gates a tool on a
prerequisite whose absence makes the tool *unusable*. Here there is no such
prerequisite: `lop network init` is how the first network comes to exist, so a
gate on "a relay is configured" would remove the tool from precisely the session
that needs it, and the operator's R19 brief ("an agent can set a network up from
a verbal request") would be unsatisfiable on a fresh machine. The only condition
that would justify a gate — no `lop` CLI reachable at all — cannot obtain inside a
session that already has `bash`, and if it somehow did, `bash` would report it.
The cost of that choice is stated honestly: it is one more tool schema in every
session's prefix (the ladder's tax), and the mitigation is that the tool's
`description` and its twelve-action enum are kept as small as the CLI's own
surface allows.

```python
class NetworkParams(BaseModel):
    action: Literal[
        "status", "init", "invite", "join", "ls", "show", "peers",
        "member_rm", "disconnect", "panic", "log", "doctor",
    ]
    network: str = ""      # a network name or id
    token: str = ""        # for `join`; never echoed back
    sas: str = ""          # for the join's confirm step, read from the OTHER device by the user
    role: Literal["read", "drive", "admin"] = "read"
    device: str = ""       # for `member_rm`
    since: str = ""        # for `log`
```

**This document's contribution is the rule that makes that list safe**, and it is
one rule rather than a second guard implementation: *the tool may invoke any of
these, because the CLI's own guards are what stop the dangerous ones* —

| Action | Why it cannot complete from a tool alone |
|---|---|
| `join` | R3's human step is the CLI's own two-phase contract (§1.4.3): phase one exits `3` with `status: "awaiting_confirmation"` and the SAS, and `--confirm <sas>` is what finishes it. **The tool must never synthesise `--confirm`** — it returns the code and the sentence `Ask the user to read back the code K7QF-2M4D from the other device, then confirm it.` |
| `panic`, `disconnect`, `member_rm` | R17's controls require the typed confirmation (§1.5) — the network's name, or a `y/n` at a TTY. **The tool must never pass `--yes`**, and a non-TTY invocation returns the confirmation sentence rather than acting |
| `--force` of any kind, and any credential entry | the CLI refuses it; the tool has no flag for it, and the guide's §5/§8 sections say why |

Two further invariants, both from §12.5 and both worth restating because they are
easy to break in an implementation that "just shells the CLI":

* **The invite token and the SAS never appear in the whole result set**, in any
  field, in any log line, and in any error message — the token is single-use
  network credential material, and a tool result is the most-copied text in the
  system (it goes into the transcript, the analytics ledger's sizes, and the
  session's own history).
* **Output is parsed, never passed through.** The tool returns structured fields
  from the CLI's `--json`; a raw stdout blob would put a token in the transcript
  the moment a command printed one.

`approval_tier="write"` for `init`, `invite`, `join`, `member_rm`, `disconnect`,
`panic`; `"read"` for `status`, `ls`, `show`, `peers`, `log`, `doctor` (the tier is
per call, as `send` states its own — `tools/builtin.py:8307`).
`concurrency="exclusive"` for `panic`, `disconnect` and `member_rm`: each mutates
the epoch or the trust state, and two of them racing is a state nobody designed.
`interruptible=True` for the reads and for `join` (a bounded wait on the handshake);
`False` for nothing else.

The tool's `description` (the schema string the model reads) states the boundary
in its first sentence, so the refusals above are expected rather than surprising:
*"Read and drive a lop mesh network from this device: peers, their sessions,
creating a session on a peer, moving one there, and the network's own lifecycle.
Pairing and incident controls need a human — this tool reports what the CLI
refused and why."*

### 3.3 The `--json` shapes the guide depends on

Agent-facing, so they are contract rather than incidental output. The session and
peer shapes are **`mesh-transport-identity.md` §9.2–9.3's**, adopted verbatim —
this section only points at them and names the two exit-code facts the guide needs.

```jsonc
// lop network status --json     (the shape transport §2.5's net_status produces)
{"installed": true, "healthy": true, "relay_pid": 48213, "device_id": "d_6c1f…",
 "device_name": "damian-mbp", "trust": "active",
 "networks": [{"network_id": "n_7Yb3kQ", "name": "devmesh", "self_role": "admin",
               "epoch": 7, "members": 3, "trust": "active"}],
 "log_path": "…/network/audit.jsonl"}

// THE AUDIT BLOCK, `relay`, when the relay ANSWERS (`relay_answering`). These are the
// fields the human surfaces print — via `relay.audit_status_words`, one renderer for
// `lop network status`, the TUI's /network panel and the agent digest — and the pair a
// reader compares after finding a row missing from `audit.jsonl`:
{"relay": {"pid": 48213,
           "audit_recorded_through": 13,   // this writer has recorded 13 rows
           "audit_published_through": 13,  // high-water mark of rows written; a row
                                           // retention PRUNED keeps its number here
           "audit_degraded": false,        // a write failed: a LOSS signal
           "audit_degraded_reason": "",
           "audit_path": "…/network/audit.jsonl"},
 "relay_answering": true}
// With no relay answering, `relay` is null and NO `audit*` key is present. Absence is
// not zero: a reader that defaulted these to 0 would render every row "not yet
// written", so the human line prints a sentence instead (`D40`), and the pair is out
// of reach until the relay answers again.

// lop network peers --json / lop sessions --all-peers --json
// transport §9.3's aggregation, verbatim:
{"sessions": [
   {"session_id": "b71e…", "conversation_name": "mesh design", "locality": "remote",
    "peer": {"device_id": "d_9c02…", "name": "devon-laptop", "network_id": "n_7Yb3kQ",
             "reachable": true, "age_s": 1.2},
    "busy": false, "state": "live", "model_label": "qwen3-coder"}],
 "peers": [ /* transport §9.4's peers block */ ],
 "degraded": []}

// lop sessions move <id> --to <peer> [--keep] --json   (mesh-session-mobility §6)
{"session_id": "b71e…", "from": "local", "to": "d_9c02…", "keep": false,
 "phases": [{"phase": "quiesced"}, {"phase": "copied", "bytes": 212345},
            {"phase": "promoted"}, {"phase": "retired"}],
 "source_retired": true, "duration_ms": 812}
//  refusal → {"status": "refused", "reason": "fenced", "sentence": "…"} (exit 1)
```

Two facts the guide must state, because an agent that does not know them will
retry forever:

* **exit code 3 means a human decision is required** (`join`'s phase one, and any
  command that needs a typed confirmation in a non-TTY). Retrying without the
  human is the failure mode.
* **`degraded` is not an error.** An unreachable peer yields a `degraded` entry
  and its sessions may be absent; the agent's next step is `lop network doctor`,
  not a retry of the same list.

## 4. The evidence plan

The rule this plan obeys, from `~/local-operator/AGENTS.md`:
**a green test is not visual evidence**, every visual change needs a rendered
frame **before and after**, and the numbers behind the frame must be checked as
well as the pixels. "Evidence goes on the PR, never into the repository" (this
repo); in `local-operator-ui` the evidence set is that repo's own committed
convention (`docs/evidence/…`, written by its capture tool) and the frames ride
the PR.

### 4.1 TUI — commands, exactly

```sh
# the gallery first: never write another sample script before checking this
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/visual_gallery.py --list

# S1 zero peers — the BEFORE frame, and the after-change regression frame.
# `scripts/sidebar_shot.py` renders from a FIXED catalog, so the same seed with
# no peer fields IS the before tree's frame.
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/sidebar_shot.py before.svg 100x30
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/sidebar_shot.py after.svg  100x30

# S2/S3/S4 — the peer cases. `sidebar_shot.py` seeds a fixed catalog of
# SessionRows; the change adds three rows carrying locality/owner_* (one live
# peer's two rows, one unreachable peer's row) so the sections, the `⇄` slot and
# the heading suffix are all in one frame. `peers` is the VARIANT argument, and
# 100x30 is the size the gallery runs it at.
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/sidebar_shot.py peers.svg peers 100x30

# S5 — the SETTLING frame for the caret/locality interaction (round 1, D4) and
# the only size at which the UNREACHABLE peer's section is on screen at all: the
# peer tier costs three chrome lines per device, so at 100x30 the window stops
# before `⇄ pixel-8 (unreachable)` and that heading was in no artifact of the
# round (review round 4, MINOR 5). `peers-focus` is the variant; it sets the
# list focused and seeds the cursor on a remote row, which is the pair the
# design asked for. It is a GALLERY CASE (`sidebar_shot-peers-focus`), not a
# one-off with two environment variables, because a knob the gallery cannot
# reach is not evidence.
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/sidebar_shot.py peers-focus.svg peers-focus 100x45

# the network panel: BOTH phases from ONE boot, so the pair differs by nothing
# but the relay's answer. The script takes a directory and writes
# network-loading.svg and network-loaded.svg (plus their .geometry.json). The
# first frame is captured while the CLI stub is parked, so it is the real first
# paint (`checking…` rows) rather than a fast worker's result.
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/network_shot.py net-panel 100x30
```

Each capture writes `<name>.svg` **plus** `<name>.geometry.json`
(`scripts/visual_capture.py`'s `save_capture`), which is where the numbers come
from; the sidebar cases also write `<name>.list-state.json` (``entries``,
``offset``, ``page_size``, ``visible``, ``sections``), because a frame that
paints part of a list cannot be read without them (round 1, D5b). The checks
each frame must pass, from the repo's own list:

* **the geometry describes the screen the SVG shows.** `save_capture` records
  `screen.class` and walks the ACTIVE screen, because `App.query` resolves to the
  DEFAULT one while `export_screenshot()` composites the active one — which is
  how two panel frames shipped beside the geometry of the transcript underneath
  them (round 1, D1). A modal case's geometry must name `NetworkScreen` and
  carry the modal's own widgets;
* `app.screen.virtual_size == app.screen.size` (a virtual size larger than the
  actual is always a bug on this app);
* `show_vertical_scrollbar` is **False** in both before and after — a scrollbar
  appearing is a silent two-cell width loss, and the sidebar's title budget is
  the thing it would eat;
* the sidebar's own `size.width` is identical before/after at the same terminal
  size (the row must not grow: that is the whole point of decision 2 in §1.3);
* in the peer frame, the `⇄` mark is in the prefix slot (cell 1) and the title's
  first cell is column 4 — i.e. `render_lines_for_test()`-style string
  assertions plus the frame.

And the four tests a reviewer should expect on the PR, because the pixels cannot
assert them:

| Test | Asserts |
|---|---|
| `tests/unit/tui/test_sidebar_peer_sections.py` | section keys and order; a peer's rows are contiguous; `⇄` is in the prefix slot and column 2 is untouched; the zero-peer frame's `_display_rows()` equals the base revision's |
| `tests/unit/tui/test_slash_network.py` | `/network` is in `_FRONTEND_LOCAL_SLASHES`; every `NETWORK_SUBCOMMANDS` word is accepted and nothing else is; the picker's offered words equal the handler's accepted words (one list, both readers) |
| `tests/unit/tui/test_new_remote.py` | the grammatical forms of §1.4.1 and each refusal sentence; the autocomplete rows come from the peer catalogue, and an unreachable peer is offered but refused with its reason |
| `tests/unit/tui/test_slash_prefixes_text.py` (existing, extended) | the two new registry declarations (`/network` SUBCOMMAND with its vocabulary; `/new` REMOTE_PEER) are stated entry-by-entry rather than defaulted |

### 4.2 Desktop — commands, exactly

```sh
cd ~/local-operator-ui-worktrees/mesh-network
# Storybook, then the capture tool over the private headless Chromium
pnpm storybook                       # or the repo's own dev command
node scripts/capture-evidence.mjs --only=chat-sidebar-peers --themes=dark,light
node scripts/check-evidence.mjs      # asserts every captured frame paints
```

* One story per state S1-S7 (`chat-sidebar-peers*`), plus the `Peers` group and
  the move-refusal notice.
* `--only=` narrows the sweep to the new stories rather than re-photographing
  ~474 frames (the tool's own documented remediation mode, and it records
  `partialCapture` in the manifest so a narrowed set is not read as a sweep).
* **Both widths**: at least one capture at the 240px floor and one at the 360px
  cap, because the whole budget argument of §2.1 is about what happens at the
  floor — a frame at 280 alone cannot show the failure the design is guarding
  against.
* Frames are looked at **per theme** (that repo's review contract, judged
  visually), and the before/after pair for S1 must be *identical* — the
  zero-peer regression, proven by pixels rather than by reading the diff.

### 4.3 What none of this proves, and what covers it instead

The stills prove the surfaces paint correctly in each state. They do **not**
prove: that a peer's session list is really federated (spine §10's 1-peer
topology, driven end to end); that quitting the local TUI leaves the remote
runtime alive (R9); or that a move is safe under concurrency (R12). Those are
`mesh-session-mobility.md`'s QA matrix and the member/placement tests in
`mesh-compute-pool.md` §9 — and, per the operator's standing rules, the QA gate
drives the real application (worktree binary, `tests/e2e -m e2e -n0`, an
isolated config dir), never the stills.

---

## 5. Decisions this document makes, in one list (for the reviewer)

| # | Decision | Alternative rejected |
|---|---|---|
| D1 | One `/network` family with a declared subcommand vocabulary | separate top-level commands per verb (`/peers`, `/mesh`, …) |
| D2 | `SlashCommand.subcommands` + `ArgumentShape.REMOTE_PEER` (two small registry additions) | overloading `WORD`; a second command registry; a `:`-delimited `/new remote:peer` |
| D3 | `/network*` is `FRONTEND_LOCAL` | routing lifecycle to the session's owner |
| D4 | `NetworkScreen` modelled on `InfoScreen`, two-phase | a notice listing; a `ReportView` |
| D5 | Peer = a new section axis (rank 3), existing tiers intact | nesting tiers inside peers; per-row marks only; every row under a device |
| D6 | Locality mark in the **cursor-prefix slot** (columns 0-1) | the spine's suggested status-glyph column (owned by the urgency ladder); a reserved leading cell |
| D7 | Mobility keeps the spine's spelling — `/move <id> --to <peer\|local>` in the TUI, `lop sessions move … --to` in the CLI — with `--to` as the discriminant and an explicit ambiguity refusal | a separate `/handoff`: cleaner grammar, but the spine names `/move` and mobility §4.2 landed it, so a third spelling would be the divergence |
| D8 | Panel `d`/`shift+P`, selection-scoped, with typed confirmation for panic | a global chord; a bare `p`; yes/no for panic |
| D9 | Desktop: remote mark **only on remote rows**, peer sections, collapsed `Peers` group | a reserved second leading icon; the trailing statement slot; a second list |
| D10 | `features.peers` (mobility §9.3's) + `features.session_transfer` (this document's addition, for the route mobility names without a key) | bumping `session_catalogue`; gating transfer on `peers` alone |
| D11 | No panic/disconnect on the desktop | a one-click irreversible control in an Electron window |
| D12 | `network` is unconditional and shells the CLI with `--json` (transport §12.5), with the dangerous completions left to the CLI's own guards (never `--confirm`, never `--yes`) and no token in any result | a `createIf` tool gated on relay reachability (which could never create the first network); a tool that synthesises `--confirm` or `--yes` |
| D13 | The networks-and-devices surface is a desktop TAB with a node graph (§2.8), driven by `net_show`'s per-network `members_detail`; removal is `member rm` behind a typed confirmation and "move into a network" is an INVITE | a section of the chat sidebar (no canvas, and a hover already spent on the tooltip); a drag-to-move gesture the protocol cannot honour either direction |

## 6. Open questions, each with my recommendation

1. **Does `/network` need a `--json`-equivalent for the agent to use the TUI's
   screen?** *Recommend no.* The agent drives the CLI (§3), the screen is for a
   human, and a machine-readable screen would be a second rendering of the same
   data with no consumer.
2. **Does the desktop need the mesh's *audit* in v1?** *Recommend no* — a log
   viewer is its own surface, and the CLI's `lop network log --json` answers the
   support question today. Evidence that would change it: a support scenario
   where the user is only ever in the desktop app.
3. **Should the sidebar's remote rows be *sorted* inside their peer section by
   the backend's order (as now) or by `active` first?** *Recommend the backend's
   order, unchanged.* The catalogue's contract says the backend owns order and
   the client must not infer it; re-sorting per peer would make two surfaces
   disagree about the same list.
4. **Does `/new remote <peer> <prompt>` send the prompt in the same turn as the
   creation, or create and then send?** *Recommend create-then-send through the
   ordinary prompt path*, because a creation that fails after a prompt has been
   admitted is a half-session the user did not ask for — and R9's rule (nothing
   succeeds silently) would be broken by the other order.
5. **Is `peers` the right capability key name, or should it be `mesh`?**
   *Recommend `peers`*: it names the thing the sidebar renders (a peer list and
   the locality of rows), while `mesh` names the whole subsystem, and a key that
   names a subsystem invites gating non-peer surfaces on it later.
6. **Should the TUI's `⇄` mark carry the peer's initial (`⇄d`) when there is
   room?** *Recommend no.* One cell of a two-cell slot is not enough to
   disambiguate two peer labels, and a wrong guess is worse than the tooltip;
   the heading and the tooltip already name the peer in full.
7. **A Storybook story for the *moving* state (S6) needs the optimistic store
   state; does that pull store changes into the design round?** *Recommend yes,
   and it should be the real store path* — a story that fakes the dimming proves
   nothing about the store's optimistic update, and the repo's review process
   explicitly judges stories built from fixtures against the live path.
8. **Does the node graph need a fan-out of its own?** *Recommend no for this
   pass, and one next.* §2.8.1 gives the two routes to a device's memberships,
   and the read path (`net_show` per network) needs no wire change: N networks
   is N calls, each on the relay's own listing budget, and the graph's node set
   is small by construction (a network is a device group, not a fleet). The
   single-call route (`net_topology`, or a `networks: [...]` list on the peer
   block) is what per-network reachability in the hover card would need — it is
   a transport-slice change, so it is the follow-up rather than a v1
   dependency. Evidence that would promote it: a measured graph whose first
   paint waits on the Nth call at a member count the operator actually runs.

---

## 7. Convergence round 1 — what changed here

Three of this document's own positions were settled in round 1, and one was
already settled by the author before the round landed (recorded here so the
reviewer does not re-open it):

1. **The mobility spelling is the spine's** (`/move <id> --to <peer|local>`, §1.7)
   with `--to` as the discriminant and an explicit ambiguity refusal. This
   document's earlier `/handoff` recommendation is kept as the *rejected*
   alternative with the cost the overload pays (D7, §1.7's table) — a third
   spelling beside the spine's and §4.2's would be the divergence, not the fix.
2. **The four `/network` slash entries keep their `desktop_destination` unset**
   (§2.7), and §2.7 now states the delta against
   `mesh-incident-response.md`'s module table, which said the opposite. That
   document's entry is corrected in the same round.
3. **The installation verbs stay out of `NETWORK_SUBCOMMANDS`** (§1.1.1):
   `serve`/`start`/`stop`/`restart`/`uninstall` are the CLI's, for the reason
   §1.5 refuses a desktop panic button — a lifecycle verb whose failure mode is
   "the supervisor is gone" must not sit one picker row from the composer.
4. **`features.session_transfer: 1` is this document's key** (§2.6) and
   `mesh-session-mobility.md` §9.3 now lists it against the transfer route it
   gates, so the route table and the `features` map are one list rather than two.
5. **No visual surface changed in this round.** The sidebar annotation, the peer
   group, the states and the evidence plan (§4) are untouched, so any design round
   taken on them remains valid against this revision.
