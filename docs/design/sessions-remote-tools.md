# `sessions-remote` — cross-device session collaboration as a tool call

Worktree: `/Users/damian/local-operator-worktrees/sessions-remote-0b39` · branch `feat/sessions-remote-tools` @ `a8b315614` (= origin/main, v0.67.17). Every cite is to that tree. Status: design note, no code written.

## §1 Problem, spec, acceptance

Spec (operator, relayed): (1) the `send` tool reaches a REMOTE session, and the remote session can reply back — reply visible locally; (2) `sessions` works for ALL its ops against a peer: list, info, peek, spawn, resume, stop; (3) `list` covers local AND remote together, with an optional local-vs-remote filter; (4) every running session the user can be referring to is queryable through that interface. Constraints: robust, simple, controlled, token-lean; targeted tests + CI only.

Proven live (Aida): `lop network sessions --peer cloud-node-1 --send <session> <text>` → the owner ran the turn and its reply came back to the CLI. The outbound viewer-act and the reply path work TODAY; what is missing is the tool surface, name addressing, and the union listing.

Trap to design around: her probe exited rc=124 at a 60 s outer bound. The CLI's own budgets are 120/300/60 → one act bound of 540 s (`network/cli.py:4502-4536`); any bound UNDER 540 pre-empts the CLI's honest `still running` / `session_unreachable ... within 540s` report — the tool's bound must sit above it (§3A).

## §2 What exists (ground truth)

| Piece | Where | What it gives |
|---|---|---|
| Pilot trio `--send/--steer/--slash` | `network/cli.py:476-519`, `4714-5121` | one act on a peer's session, driving the SAME viewer the TUI sidebar opens (`4787-4935`); text is a REMAINDER positional, `--` escape, flag-boundary documented (`4547-4589`); receipts via `_emit` (`833-844`) |
| Reply path (per-act) | `network/cli.py:4938-4991`, `4592-4609` | `prompt_and_wait` → owner's terminal outcome; `_pilot_last_reply` reads the owner's own last assistant row from `viewer.display_history_window()`; `reply` rides the receipt; test `test_sessions_pilot.py:441-452` |
| Bounds | `network/cli.py:4502-4536` | BIND 120 / TURN 300 / REPLY 60; `PILOT_ACT_TIMEOUT_S = 540`; expiry REPORTED (`4955-4962`), never a silent 0 |
| TUI child budget | `tui/network_cli.py:69` | `PILOT_CALL_TIMEOUT_S = PILOT_ACT_TIMEOUT_S + 60` = 600 |
| Remote viewer | `session/remote_open.py:54-70`, `73-141`; `network/projection.py:1078-1097`, `1112+` | cold viewer; `RemoteSessionClient` inherits every frame incl. `history_page` (`mobile/attach_client.py:2128-2130`) |
| Network tool | `network/tool.py:79-137`, `300-471`, `491-517`, `1118-1271` | subprocess of THIS build's CLI with `--json`, parsed + scrubbed, never passed through; per-call tier; `--yes/--force/--confirm` unreachable |
| Session-plane ops | `network/relay.py:3859-3863`, `8842-8846`; `network/types.py:365-369`, `537-539` | owner side `net_session_create/engage/stop/lifecycle/receipt`; local side `peer_session_rows/facts/create/engage/stop`; caps: create→prompt, engage→view, stop→stop; session ops must name an owned session (`authorizer.py:308-359`) |
| Federated listing | `cli.py:1059-1075`, `4996+`, `5687-5756`; `relay.py:9846-9890` | `--peer`/`--all-peers` merge; EVERY row carries `locality`/`peer` (`info/collect.py:880-895`); three-way honesty `relay_unavailable`/`relay_refused`/`peer_unreachable`; unreachable peers named in notes |
| Sessions tool | `tools/builtin.py:13536-13651`, `13726-13754`, `13860-13942`, `14132-14146`, `14434+`, `16166-16238` | ops list/info/spawn/resume/stop/peek/help; derived description + on-demand `help`; per-op tiers; exclusive; subprocess precedent `_SESSIONS_CLI` (`13469`), env strip (`13481`, `15546-15551`), `_sessions_launch` (`15554-15586`), `python_argv` (`interpreter.py:79-91`) |
| Send tool / `lop send` | `tools/builtin.py:12785-12962`, `13120+`; `cli.py:914-966`; `mobile/peer_send.py:746+`, `895` | LOCAL ONLY: no `--peer` in the parser; resolver scans the local registry + local store; peer_send imports no network module at all |
| Guides | `guides/network/GUIDE.md:257-354`; `guides/sessions/GUIDE.md:19-107`; `guides/peer-messaging/GUIDE.md:39-141`, `514-531` | pilot verbs fully documented for humans; `lop send --peer`/`lop exec --peer` explicitly "NOT in this build" (`network GUIDE:385-393`, `876-888`) |
| Budget rulers | `scripts/bench_context_budget.py:68` (2.78 chars/billed-token), `:1210` (`BUDGET_BILLED_TOKENS = 36_552`); `tests/unit/tools/test_sessions_tool.py:952-1000` (schema pin, desc 230, cl100k) | sessions-tool `docs/design/sessions-tool.md:175-186` §8.4: list ~600 tok, peek ~1,800 |

## §3 Design

### A. `network` tool: `send` / `steer` / `slash` on a peer's session

Fields (add to `NetworkParams`, `network/tool.py:184-252`; names mirror `engage`/`stop`):

| field | type | meaning |
|---|---|---|
| `send` | `str = ""` | deliver a turn to this session (id-or-NAME) on `peer`, wait for its outcome |
| `steer` | `str = ""` | inject into that session's running turn |
| `slash` | `str = ""` | run a slash command there |
| `text` | `str = ""` | the payload for whichever act |

Why `text` (not message/prompt): `--prompt` already means "`--create`'s first turn" (`network/cli.py:469-475`); `text` matches the CLI's own positional name.

- Tier + argv share ONE predicate: extend `_SESSION_MUTATIONS` (`tool.py:111`) to `("create","engage","stop","delete","send","steer","slash")`. Tier = **write** for all three (a send starts a turn on another device; slash mutates the owner's record — same commitment as the send tool's write, `builtin.py:12938-12941`). Extend `test_tool.py:326` (tier and argv read one set).
- Guards, mirroring the CLI (`network/cli.py:5196-5241`): exactly ONE act; an act beside `create/engage/stop/delete` refused with the CLI's own wording; `peer` required (`tool.py:416-420`); `all_peers` beside an act refused (`421-432`).
- argv: `["network","sessions","--json","--peer",P,"--send",TARGET,"--",TEXT]`. `--json` goes BEFORE the act flag and text ALWAYS after `--`; the tail `+ ["--json"]` (`tool.py:471`) must NOT apply to pilot acts — after the first non-option token everything is REMAINDER (`network/cli.py:494-519`), so a trailing `--json` would be delivered as payload. `--` is dropped by the CLI (`4569-4589`), so leading-dash text is safe; a multi-line body passes as one argv element.
- Timeouts (chosen where `execute_network` picks one, `tool.py:1153`): pilot **600** = `PILOT_ACT_TIMEOUT_S`(540) + 60, the TUI's own derivation (`tui/network_cli.py:69`); stop **300** (CLI relay wait 240, `network/cli.py:5300`, + margin — a graceful stop can wait out a drain); create/engage **180** (CLI 120, `5321`/`5415`); join 120 (existing); default 30. A test pins `_PILOT_TIMEOUT_S > net_cli.PILOT_ACT_TIMEOUT_S` and the derivation (mirror `test_sessions_pilot.py:913-930`). This is the rc=124 fix: the tool never pre-empts the CLI's "still running".
- Receipts (parsed, never passed through; refusals keep the family shape `{ok:false, code, message}`, handled at `tool.py:1194-1220`): send → `{ok, outcome: finished|running|queued|failed|lost, reply?, code?, error?, peer_named?}`; steer → `{outcome:"steered", receipt}`; slash → `{command, ok, outcome ran|refused, text, style}`. Add a pilot branch to `_render` beside `930-969`; raw payload → `details`; `_scrub` (`281-291`) unchanged.
- Text edge cases: empty `text` is pre-refused in `_argv_for` with the sentence shape `action='sessions' with 'send' needs 'text': the words to deliver (this tool cannot pipe a body in).` — `_run_cli` DEVNULLs stdin (`tool.py:502`), so the CLI's stdin fallback (`network/cli.py:4564-4566`) is unreachable and its own missing-text sentence (`4746-4749`, rc 2) would be the fallback path.
- `create`-side fields stay as-is (only `prompt` today); spawn-with-fields is the sessions tool's job (§3C). No new wire op in this whole note.

### B. `send` tool: remote sessions (and the reply path)

Finding: **`lop send` does NOT reach a peer today.** No `--peer` on the parser (`cli.py:914-966`; the CLI's only `--peer` is the listing flag at `1063`); the resolver scans `registry.scan` + the local store with zero network imports (`mobile/peer_send.py:895`; verified by grep — no relay/catalogue/peer references); the guide says it in words (`network GUIDE:385-393`, `884`). The design's `lop send --peer` (`docs/design/mesh-ui.md:1190-1193`; mailbox route designed at `docs/design/mesh-session-mobility.md:748`, `787-794` as a forwarded `peer_message`) is unbuilt and would be NEW wire (no such op exists).

Recommended smallest path — build around the proven round trip:

- `SendParams.peer: str = ""` (`builtin.py:12785+`). With `peer`: `message` only; `wake`/`now`/`patience`/`model` REFUSED when given (fields-set based, the `_SESSIONS_OP_FIELDS` discipline) with a sentence naming the local-only modes.
- Delivery = subprocess `lop network sessions --peer P --send <target> -- <text> --json` (timeout 600) through the child-CLI helper of §D. Result renders the outcome and the owner's `reply`; `running`/`queued` are honest non-completions (`network/cli.py:4955-4985`).
- Target-by-name on that peer: the pilot verbs gain id-or-name resolution in `_pilot_act`'s row read (`network/cli.py:4806-4816`): exact session id → exact conversation name/cwd basename → case-insensitive substring; ambiguous → refusal listing the candidate rows; none → the existing `session_unknown` (`4672-4676`). ONE implementation next to `peer_session_row` in `session/peer_rows.py:166`; pure over fetched rows, so testable with no mesh. (Local resolver's semantics, `mobile/peer_send.py:777-800`, minus pid/stored-fallback tiers — those name local rows only.)
- Reply path mechanism (cite): viewer open (`remote_open.py:73-141`) → `prompt_and_wait` (`network/cli.py:4938-4954`) → `_pilot_last_reply` reads the owner's own last assistant row from the viewer's synced display window (`4592-4609`) → `reply` in the receipt (`4986-4990`). That is what "the reply is visible locally" means: it lands in the TOOL RESULT, per act. There is no cross-device mailbox, and no path that posts it into the caller's own transcript (peer messaging is loopback-only by construction, `peer_send.py:1-12`, `peer-messaging GUIDE:516-517`) — see §7.
- Tier: send stays **write** (`builtin.py:12938-12941`); `describe_approval` gains the peer ("peer <device> / …", label `12852-12867`).

### C. `sessions` tool: every op against a peer

| op | exact wire/proc | fields | tier | refusal semantics (new ones named) |
|---|---|---|---|---|
| `list` | local half unchanged (`builtin.py:14441-14451`) + subprocess `lop network sessions --all-peers --json` (or `--peer P`); drop `locality=="local"` rows from the federated doc (own rows; `relay.py:9846-9871`), merge by `session_id` (local wins, mirror `cli.py:5737-5752`), cap merged output at `limit` | add `scope` ∈ `all|local|remote` (default `all`), `peer` narrows remote rows to one device | read | no-mesh → local rows + ONE note (the CLI's `relay_unavailable` doc, `cli.py:5712-5728`); unreachable peers → named notes (`5731-5734`), never a silent drop |
| `info` | same federated fetch; select via the shared resolver; render existing fields + device/state | `peer`, `session`/`target` | read | `session_unknown`; ambiguity lists candidates |
| `peek` | FEASIBLE with NO new wire op: the viewer protocol's bounded owner-served read — synced `display_history_window()` (`attached.py:8725`) + `history_page` paging with signed tokens (`attached.py:5276`, `attach_client.py:2128-2130`; capability `"view"`, `types.py:665`; `before_token`/`snapshot_token`/`has_more`, `history_window.py:65-95`). v1 = tail window (`steps`, default 12, max 50) via a new read flag on the CLI pilot family; `query`/`regex`/`digest` refused for remote (local-only reads) | `peer`, `session`/`target`, `steps` | read | cold session: the peek WARMS it (viewer bind engages; `network/cli.py:4883-4885`) — stated, not hidden (decide §7) |
| `spawn` | `lop network sessions --peer P --create …` → `net_session_create` (`relay.py:6374-6392`; CLI flags `5335-5416`) | `peer`, `prompt` (required), `name`, `team`, `profile`, `model` (split `provider/model-id` → `--hosting`/`--model`) | write | `visibility`/`background` given → refused (local concepts); `yolo`/unattended never forwarded (no field) |
| `resume` | prompt given → `--send` (engages implicitly, runs the turn, returns the reply); no prompt → `--engage` (warm; receipt `network/cli.py:5311-5333`) | `peer`, `session`/`target`, `prompt` | write | the SET form (`paused`/`failed`/`all`) refused with `peer` — it selects over the LOCAL store |
| `stop` | `--stop` → `peer_session_stop` → `net_session_stop` → the owner's own ladder (`relay.py:6746-6789`; CLI `5277-5309`) | `peer`, `session`/`target` | exec (unchanged) | no force in v1; the remedy line names `lop network sessions --stop <id> --force` |

- Resolution for info/peek/resume/stop = §3B's helper (same id-or-name semantics; one spelling).
- Row marking needs NO new work: every row already carries `locality` + `peer` (`collect.py:880-895`); the tool passes them through.
- Infrastructure already fits: `peer_session_facts` covers one session's pid/facts (`relay.py:9688-9721`) but the catalogue row carries the same fields — info needs no new op.
- Table/schema/doc surface: `_SESSIONS_OP_FIELDS` (`builtin.py:13625-13651`) gains `peer`/`scope` per op; the description builder (`13726-13749`) gains ONE clause; the on-demand `help` reference (`13860-13906`) gains the per-op sentences (lazy, free at start).

### D. Where the remote code lives — one recommendation

**Subprocess-CLI reuse for every remote effect; no direct network imports in tool code; pure decision helpers may be shared by direct import.**

- Reasons: the tool header's single-surface rule (`network/tool.py:1-7` — "the agent drives the mesh through ONE surface: the authenticated `lop network` CLI"); no second writer of store/identity; the CLI keeps the bounds, tiers, refusals and receipts; and both precedents already exist (`network/tool.py:491-517` `_run_cli`; `builtin.py:15554-15586` `_sessions_launch` with `_SESSIONS_CLI` `13469`, env strip `13481`, `python_argv` `interpreter.py:79-91`).
- Implementation: ONE new helper in `builtin.py` beside `_sessions_launch` — run `python_argv(*_SESSIONS_CLI, "network", …, "--json")` with the same stripped env, DEVNULL stdin, process-group reap, the §3A caps (600/300/180/30); parse the JSON; scrub with the SAME marker set as `network/tool.py:135` (lazy import or lift `_clean`/`_scrub`).
- The name selector is I/O-free → may live in `session/peer_rows.py` and be imported directly by both the CLI and the tools; that is NOT a second I/O path.
- Rejected: direct `local_operator.network.*` orchestration in tools — duplicates viewer/relay/bounds in the session process and invites a second writer.

## §4 Guides + advertising (E), with measured budgets

Exact strings (measured now, char count → billed at the repo's 2.78 chars/token; the FULL provider-array delta is re-measured with `scripts/bench_context_budget.py` at implementation, incl. JSON scaffolding — the recorded examples show ~+100 chars of scaffold per new optional property):

**network tool description** — replace the tail `peers, their sessions, creating a session on a peer,` (85 ch) with `peers, their sessions (list, create, engage, stop, and send/steer/slash a conversation),` (121 ch): **+36 ch ≈ +13 billed**. Keep the rest byte-for-byte.

**`NetworkParams` field descriptions** (4 new): send `For sessions: deliver a turn to this session on \`peer\` (an id or name), and wait for its outcome.` (97 ch); steer `For sessions: inject into the turn that session is running there.` (65); slash ``For sessions: run a slash command in that session (e.g. '/rename a better name').`` (81); text `For send/steer/slash: the words to deliver, taken as the CLI's positional text.` (79) — **+322 ch ≈ +116 billed** + scaffold.

**send tool** — new field description `Message a session on ANOTHER device over the mesh (a peer name or id); drives a turn there and returns the owner's reply.` (121 ch ≈ +44) and one description clause ` With \`peer\`, the target addresses a session on that device (mesh): the send drives a turn there and returns the owner's reply; \`wake\`/\`now\`/\`patience\`/\`model\` are local-only and refused.` (187 ch ≈ +67). Total **+308 ch ≈ +111 billed** + scaffold.

**sessions tool** — field `peer`: `Remote (mesh): the device holding the session — a name or id; on \`list\`, narrows remote rows to it.` (99 ch ≈ +36); field `scope`: `list: which homes to show — all (default), local, or remote.` (60 ch ≈ +22); description clause `; \`peer\` acts on another device's session over the mesh (\`list\` shows those rows beside local ones; \`scope\` filters them)` (121 ch ≈ +44). Total **+280 ch ≈ +102 billed** + scaffold (the schema pin `test_sessions_tool.py:952-1000` is re-measured and moved; the description figure moves by +~44's token share).

**Guides (lazy — bodies are read on demand, not in the start context; measured for the read):**

- `guide://network`, after the pilot table (`GUIDE.md:276`): ~267 ch insert (≈96 billed when read) — "A PILOT VERB TARGETS AN ID **OR A NAME** on that device: an exact id or conversation name wins, then a case-insensitive substring of the name; an ambiguous name is refused with the candidate rows rather than guessed. `--stop` and `--engage` accept the same spellings." — plus the tool-first line ~244 ch: "INSIDE A SESSION the same acts are one tool call: the `network` tool's `sessions` action carries `send`/`steer`/`slash` (a session id or name) with `text`, and returns the owner's receipt — for `--send`, the `reply` field is the owner's answer."
- `guide://sessions`: one new section (~866 ch ≈ 312 billed when read), "Remote sessions (mesh)": list shows local+remote by default with the holding device on each row, `scope`/`peer` filter; no-mesh lists local with one note; `info`/`peek`/`spawn`/`resume`/`stop` take `peer`; spawn mints on that device; resume warms (prompt runs a turn and returns the reply); stop is the owner's usable ladder with no force; peek is a tail window (query/digest local-only).
- `guide://peer-messaging`: `send`-tool bullet `peer` (~275 ch) + Limits amendment (~306 ch): the local mailbox is loopback-only; `send(peer=…)` is the mesh path — a different trust boundary and a driven turn with the reply in the result.

**Budgets, stated:** start-context ceiling **36,552 billed** (`bench_context_budget.py:1210`; head 36,497 on the last entry) — tool-side additions move it by the MEASURED tree delta (the script's own method, clean arm; never hand-derived). `sessions` ruler: schema pin (~1,099 cl100k) + description 230 (`test_sessions_tool.py:952-1000`) — re-pin with measurement. Output budgets: list ~600 tok, peek ~1,800 (`sessions GUIDE:88-92`; `sessions-tool.md:175-186`); the merged listing caps at `limit` post-merge so mesh rows cannot unbounded-grow it. The three tool descriptions stay ≤2 lines each as drafted.

## §5 Safety / control (F)

- Tiers unchanged except the new verbs: network tool `send/steer/slash` → write (per-call, same predicate as argv); sessions remote list/info/peek read, spawn/resume write, stop exec; send tool write. No new `exec` tiers.
- No `--yes`/`--force`/`--confirm` is ever spelled — property of `_argv_for` (`tool.py:344-355`, `448-450`) and must stay one for the new branches; text after `--` is DATA, so a payload beginning `--force` cannot become a flag. Remote stop's force escalation stays the human CLI (`lop network sessions --stop --force`), named in the remedy line.
- Secret scrubbing: every payload through the network tool's `_scrub` marker set (`tool.py:135`, `281-291`) / the child-CLI helper's own scrub; stderr capped (`_STDERR_CAP:127`).
- DEVNULL stdin on every child; empty `text` pre-refused (sentence shape §3A); no stdin fallback reachable from a tool.
- Concurrency: both tools keep `concurrency="exclusive"` (`tool.py:1259-1264`; `builtin.py:16227-16231`); the network tool stays `interruptible=False` (a static field; aborting one act mid-rotation is worse than a bounded wait — the act is capped at 540 by the CLI, 600 by the tool).
- Refusals, not silent drops: one-act rule; ambiguity lists candidates; unreachable peers named; `running`/`queued` reported as non-completions.

## §6 PR split, tests, evidence (G)

- **PR-A `feat(network): reach a peer's session by name`** — CLI: id-or-name resolution for `--send/--steer/--slash/--stop/--engage` + the pure selector in `session/peer_rows.py` + `--peek` (tail window, read tier) + guide://network inserts. Tests: `tests/unit/network/test_sessions_pilot.py`, `test_sessions_pilot_cli.py`, `test_cli.py`, `test_session_plane.py`.
- **PR-B `feat(tools): sessions and send reach a peer`** — network tool fields/render/timeouts; sessions tool ops + union list; send tool `peer`; guides (sessions, peer-messaging, network tool-first line); bench entry; docs fix for the `lop send --peer` promise (`docs/design/mesh-ui.md:1190-1193` — mark built-via-tool, CLI flag still unbuilt). Tests: `tests/unit/tools/test_sessions_tool.py`, `test_send_tool.py`, `tests/unit/network/test_tool.py`, `tests/unit/guides/test_guides.py`, `tests/unit/info/test_sessions_set_filter.py`.
- Targeted command (iterate; CI remains the whole-tree gate): `.venv/bin/python -m pytest tests/unit/network tests/unit/tools/test_sessions_tool.py tests/unit/tools/test_send_tool.py tests/unit/guides -q`.
- Can-fail proofs to write (each currently fail-able by construction): (1) argv — text `--json is the field` delivered whole and `--json` precedes the act (extend `test_tool.py:392`); (2) bound — `_PILOT_TIMEOUT_S > net_cli.PILOT_ACT_TIMEOUT_S` and `== act + 60` (mirror `test_sessions_pilot.py:913-930`); (3) ambiguity refusal lists candidates; (4) no-mesh `list` degrades to local + one note; (5) new argv spells no `--yes/--force` (extend `test_tool.py:249`); (6) union merge marks `locality` and caps.
- Acceptance script for Aida's three cells (one live agent session + cloud-node-1): (i) ONE call: network tool `sessions(action='sessions', peer='cloud-node-1', send='<name>', text='…')` → expect `ok`, `outcome`, `peer` naming the holding device; same via the shell isolate `lop network sessions --peer cloud-node-1 --send '<name>' -- '…'`. (ii) the same call's result carries `reply` (the owner's answer) — or `running` with the resume hint. (iii) sessions tool `list` → local+remote rows with `locality`/`peer`; `scope='remote'` and `peer=…` filter; with the relay stopped it degrades to local + note. Failable cells: send by wrong name (candidates listed), send with `wake=True` explicitly given (refused), `list` with no relay (local + note, rc 0 in the tool).
- Node skew: cloud-node-1 @ 0.67.11 has everything the peer side needs (pilot + viewer + display window existed by v0.63.9; no new wire op anywhere in this design), which the live round trip already proves. The one thing never exercised against it is `--peek`'s `history_page` paging — QA should run one peek cell against that node before calling the skew benign.

## §7 Manager decisions needed (H)

1. **Reply semantics.** The built path returns the owner's `reply` in the act's receipt (proven round trip). "Reply BACK" as a peer MESSAGE into the caller's own transcript would be new cross-device messaging (loopback-only today) — confirm receipt-carried is the acceptance reading.
2. **`lop send --peer` (mailbox route).** The mobility design sketches it as a forwarded `peer_message` into the owner's inbox (`mesh-session-mobility.md:748`, `787-794`); unbuilt, and it is new wire. Recommend: keep unbuilt this wave (the tool's `peer` is the path; docs corrected), revisit only if quiet-spool/cold-inbox semantics are wanted.
3. **Remote `peek` warms a cold session.** Reading a stored session engages a runtime on that device (same as opening it in the TUI). Recommend: accept + document; the alternative is gating peek to live sessions (refuse cold with a sentence) or deferring peek — say which.
4. **Remote stop force.** v1 keeps the tool's no-force rule and names the CLI `--force` escalation. Confirm, or the tool grows a `force` field (and its ladder-wait budget of 300 s).

## §8 Manager decisions (locked 2026-10-06)

1. **Reply semantics — LOCKED (receipt-carried).** The one-call acceptance is the act's receipt carrying the owner's `reply`. The async inbound leg observed live (a remote side putting a new turn/message into the caller's session ~a minute later) is kept as-is and documented truthfully; the mesh lane has been asked to confirm the exact route (reverse pilot / viewer-prompt is the working reading). No new wire either way.
2. **`lop send --peer` mailbox route — LOCKED: unbuilt this wave.** The tool's `peer` field is the path; `docs/design/mesh-ui.md:1190-1193` corrected to say so. Revisit only if quiet-spool/cold-inbox semantics are wanted (v2 candidate).
3. **Remote `peek` on a cold session — LOCKED: GATE it.** v1 remote peek reads LIVE sessions only; a stored session refuses with a sentence naming `resume`/`engage`. Rationale: "controlled" — a peek must never start a runtime on a device nobody is watching, and the operator's own framing is "all RUNNING sessions". Documented in `guide://sessions`; Aida is flagged on this call.
4. **Remote stop `force` — LOCKED: none in v1**; the remedy line names the CLI `--force` escalation.
5. **PR split — LOCKED as §6:** PR A (CLI id-or-name resolution + `--peek` + guide://network), then PR B (three tool surfaces + guides + bench + mesh-ui docs fix). This design note rides PR A as `docs/design/sessions-remote-tools.md`.
6. **Budgets — LOCKED as drafted;** implementation re-measures with the repo rulers (`bench_context_budget.py`, the sessions-tool schema pin) and re-pins with the measured numbers.

## §9 Field confirmations (mesh lane, 2026-10-06 — binding)

1. **Inbound leg — use this wording verbatim in guide copy (PR-B):** "a send from another device is delivered through the owner's own prompt admission (the attached-viewer path, `viewer.prompt_and_wait`); the owner decides prompt vs steer exactly as it does for a local composer; there is no separate inbound wire op." Citations: `network/cli.py:_pilot_send`, `_pilot_steer` (idle-owner refusal), `session/session.py` append path (producer_command_id). Do NOT describe it as a general "viewer protocol" mechanism.
2. **Fresh-created peer sessions carry a resolution window today** (`peer_rows.peer_session_rows` TTL short-circuits `remote_row_for`'s live read; create-on-peer does not seed the cache). The mesh lane's fix is in flight (`ttl_s=0` on the miss + seed on create). Surfaces and tests must NOT encode the window; add a regression cell once the fix lands (PR-B/QA).
3. **Approvals stay out of tool surfaces** (remote park allow/deny is an origin-authority slice in flight); do not document allow-from-origin anywhere in this workstream.
