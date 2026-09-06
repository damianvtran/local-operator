# Design: build-skew detection and completion between viewers and runtimes

Status: proposal (architect). Scope: one PR. No `pyproject.toml` version bump —
the release owner handles that.

## 1. The problem as found in the code

Since 0.46.0 every fresh `lop` is a **viewer**: the TUI holds a cold
`RemoteSession` and a separate runtime process
(`python -m local_operator.session.runtime.process`, spawned from
`sys.executable` — `local_operator/session/runtime/launch.py:210-211`) owns the
real `Session`. Two long-lived populations therefore coexist on one host —
TUIs and runtimes — while `lop-update` replaces the on-disk uv-tool install
under them, often several times a day. Skew happens in both directions and
nothing today detects it.

The Sep 5 incident is the concrete shape:

1. TUI (pid 87909) ran in-memory code ≤ v0.46.23, i.e. **pre-#624** (e1db6e8da
   landed in v0.46.25; `git tag --contains e1db6e8da` starts at v0.46.25).
2. `/new` built a cold `RemoteSession`; `_engage_runtime_eagerly`
   (`tui/app.py:8774`) spawned a runtime **from the new on-disk build**
   (0.48.0), because the spawn resolves `sys.executable` — the uv-tool path —
   fresh at spawn time.
3. `/team lopdev <request>` routed to that runtime.
   `OwnedSessionHandle._team_attach_slash`
   (`local_operator/session/runtime/owned.py:1817-1891`) attached the team and
   returned `SlashResult(kind="notice", text="sending to lopdev. manager is
   coordinating.", data={"type": "team_attached", "request": ...})`.
4. Since #624 the **viewer** is expected to consume that receipt:
   `OperatorApp._render_authoritative_slash` (`tui/app.py:14674-14685`) syncs
   the band and calls `_submit_command_prompt(request, attachments)`. The
   pre-#624 viewer prints `text` and has no consumer for `data.request`
   (verified: `git show v0.46.23:local_operator/tui/app.py` — the renderer
   ends at `if text:`; and v0.46.23's `owned.py` still returned
   `noop {"type": "team_mutate"}`, so the receipt type was brand new to it).
5. Result: notice printed, request silently dropped. No user row, no turn.

The generalisation the manager stated is exact: routed slash results that
carry an **action for the invoker** — today exactly `team_attached` and
`agent_attached` with a non-empty `request` — fail silently under skew, and
skew is undetectable from either side today.

There is also a third, quieter shape: **new viewer + pre-#624 runtime**. A
runtime resident since Sep 4 (there are several on this host; the live records
under `~/.local-operator/run/mobile/` show protocol 5 daemons started
Sep 4 18:26 and Sep 4 22:35) returns `noop {"type": "team_mutate"}` for
`/team <name>`. The current renderer's noop branch (`app.py:14629-14645`)
handles only `agent_list`, so that command *still* vanishes — the same defect
shape `tests/unit/tui/test_noop_consumers.py` exists to prevent, reachable
through the version dimension the audit cannot see.

## 2. Options considered

**Do nothing / document.** Rejected: the failure is silent and this host runs
~16 resident runtimes plus several TUIs with many releases per day. Skew is
the steady state, not the edge.

**Bump `PROTOCOL_VERSION` and refuse mismatched peers.** Rejected.
`types.py:27-49` documents why bumps are reserved for breaking changes: a bump
makes every older peer *refuse a record it can in fact use*, and it does
nothing for the TUIs already running — they would simply be unable to attach
at all. The failure would become loud but the host would be unusable during
every release window.

**A. Local build-drift detection in the TUI (no protocol change).**
Snapshot what this process loaded; re-read at the seams that spawn or bind a
runtime; warn with `/reload` as the remedy.

**B. Runtime-side completion for skewed viewers (backward compatible).**
The attach auth frame gains an opt-in declaring which action-carrying receipt
types the client consumes. When a routed result carries a request the
connected client did not declare, the runtime admits the request itself
through the same admission path as the `prompt` op
(`owned.py:687-732`, `_PromptCommand` queue).

**C. Version exchange.** Additive `version`/`source_ref` on `SessionRecord`
(the record file *is* the hello channel an attach client reads before and at
dial — `RemoteSession._bind_to(record)` at `session/remote.py:1002`); the
viewer compares with its own build on bind.

**Recommendation: ship A + B + C in one PR.** They are one coherent change —
"skew between a viewer and a runtime is detected and the known-silent class is
repaired" — and each covers a failure the others cannot:

| Case | A (disk drift) | B (runtime completion) | C (version exchange) |
|---|---|---|---|
| old TUI spawns NEW runtime, `/team` request (the incident) | warns, offers `/reload` | **turn runs** for every already-running stale TUI | — |
| new TUI attaches to OLD resident runtime | — | — | detects, warns, names the remedy |
| new TUI + pre-#624 runtime (`noop team_mutate`) | — | — | **renderer degrades loudly** (paired viewer-side change) |
| same-version rebuilds between releases (`lop-update` on this host) | detects via `.lop-source` ref | — | detects via `source_ref` |
| every *future* silent-skew shape | detects the precondition | — | detects the precondition |

B is the load-bearing piece: it repairs the incident class for every stale TUI
**the moment the new runtime is on disk**, with zero user action. A and C are
each ~tens of lines sharing one new helper; cutting them to shrink the PR
would leave both skew directions undiagnosed for every future shape.

If the reviewer insists on splitting: B first (it is the fix), C second, A
third. I do not recommend splitting.

## 3. Empirical result the task asked for

**`importlib.metadata.version()` re-reads disk in a running process.** Probed
on this host with the repo interpreter (Python 3.14.3): installed a synthetic
`skewpkg-0.46.23.dist-info`, read `version("skewpkg")` → `0.46.23`; replaced
the dist-info with `skewpkg-0.49.0.dist-info`; re-read in the same process →
`0.49.0`, both with and without a sleep between (the dist-info directory
*name* changes with the version, so even a path-keyed cache misses).

**Caveat that shapes the design:** `lop-update` between releases rebuilds from
`main` while `pyproject.toml` still says the *last released* version. Then the
dist-info name is identical and only content changes — `version()` cannot
distinguish the builds, but `~/.local/share/uv/tools/local-operator/.lop-source`
can: `lop-update` records `<git-sha> <tag>` there (live example:
`4d3ce1d1a48f4f3b799efdfabb014979e70e0630 v0.49.0`). So the drift token is a
**pair**: `(version, source_ref)`, with `version` the fallback when
`.lop-source` is absent (PyPI/pipx installs, editable checkouts).

## 4. The change, file by file

### 4.0 New shared helper — `local_operator/update.py`

`update.py` is already stdlib-only (verified: imports are `time`,
`dataclasses`, `enum`, `importlib.metadata`, `pathlib`, `typing`) and already
owns `installed_version()` (update.py:98) and `is_git_snapshot()`
(update.py:395). Add:

```python
@dataclass(frozen=True)
class BuildStamp:
    """One comparable build token: distribution version + git ref of the
    install, when lop-update recorded one. Same-version rebuilds between
    releases are this host's common drift, so the ref is the primary key and
    the version the fallback; comparing on version alone misses them."""

    version: str
    source_ref: str = ""

def installed_build() -> BuildStamp: ...
```

Reads `.lop-source` at `sys.prefix` (the same root `is_git_snapshot` uses),
first whitespace-separated token as `source_ref`, `""` when absent. One small
file read; called rarely (adopt/engage/bind), never on a hot path.

### 4.1 B — runtime-side completion

**Wire field.** The attach auth frame gains an optional
`"slash_consumers": ["team_attached", "agent_attached"]`. Semantics follow the
documented precedent (`types.py:56-61`, `803-808` in server.py): **absent
field = old client**; an old runtime ignores unknown auth fields; the field is
advisory and never gates the connection. **No `PROTOCOL_VERSION` bump** — this
is exactly the additive shape v4/v5 were.

**Single source of truth** in `local_operator/session/runtime/types.py`
(import-light by contract, already imported by both ends):

```python
#: SlashResult data.types that carry an ACTION for the invoking terminal: the
#: owner attached, and the invoker is expected to submit data["request"] as a
#: user turn. A client that renders these declares them in its auth frame's
#: "slash_consumers"; an undeclared (old) client means the RUNTIME admits the
#: request itself. Absent-means-old, like ClientKind.
SLASH_ACTION_RECEIPTS: tuple[str, ...] = ("team_attached", "agent_attached")
```

**Client side.** `AttachClient.__init__`
(`local_operator/mobile/attach_client.py:128`) gains
`slash_consumers: Sequence[str] | None = None`; `connect` adds
`auth["slash_consumers"] = list(...)` when set (next to the `events`/
`frontend_state` flags at attach_client.py:195-201). `RemoteSession._dial`
(`session/remote.py:1042`) passes `SLASH_ACTION_RECEIPTS` — the full-TUI
viewer is the client that consumes them (at `app.py:14674`).

**Server side.** `RuntimeServer` auth handling
(`session/runtime/server.py:796-811`):

- Parse once: `slash_consumers = frame.get("slash_consumers")` — `None` when
  absent, else a `frozenset` of strings. Store on `_ClientConn`
  (server.py:162) as `slash_consumers: frozenset[str] | None = None`.
- `_dispatch_payload` (`server.py:1576-1598`) passes it to the handle exactly
  the way `locality` is passed today: generalise `_accepts_locality`
  (server.py:123) to a keyword probe (`_accepts_kw(fn, name)`), so existing
  handle implementations and test doubles that lack the parameter keep working
  unchanged (`tests/unit/session/runtime/test_server.py:135` has one).

**Handle side.** `OwnedSessionHandle.run_slash_authoritative`
(`owned.py:1550-1587`) gains `consumers: Iterable[str] | None = None`. After
`result = await self._slash_result(...)`, one call:

```python
result = await self._complete_unconsumed_action(result, images, consumers)
```

`_complete_unconsumed_action(result, images, consumers)`:

1. Return `result` unchanged unless `result.kind == "notice"`,
   `(result.data or {}).get("type") in SLASH_ACTION_RECEIPTS`, and
   `data.get("request")` is non-empty. (`/agent clear` returns
   `agent_attached` with `request: ""` — no action, no admission.)
2. Return unchanged when `data["type"] in (consumers or ())`. This is the
   double-submission guard: a client that declared the type submits the
   request itself; the runtime must **never** also run it. Both `None`
   (absent field, old viewer) and `[]` (declared but consumes nothing) mean
   "admit here" — the rule is `type not in declared`, never "declared is
   None".
3. Otherwise admit the request as a user turn, mirroring the viewer's own
   `_submit_prompt` split (`app.py:12229-12241`):
   - `self._session.is_streaming` → `await self.steer(request, images=images,
     command_id=<fresh uuid4>)` (owned.py:877);
   - else → `await self.prompt(request, images=images, command_id=<fresh
     uuid4>)` (owned.py:687). This is the same `_PromptCommand` admission the
     `prompt` op uses: durable append resolves `admitted`, the drain emits the
     user `MessageStartEvent`, and the old viewer — which since 0.46.x
     subscribes with `events=True` (verified in v0.46.23's remote.py) — paints
     the user row from that event through the existing mobile→TUI echo path
     (`app.py:23109-23141`, unmatched echo → `UserBlock`). No renderer change
     is needed on the old viewer.
   - Images are already on the wire: the viewer sends
     `resolve_markers(arg, attachments)` in the `slash_result` frame
     (`app.py:14852-14853`, present in 0.46.23 as well), and
     `run_slash_authoritative` already receives them as its `images`
     parameter. They go into the admission unchanged.
4. On admission failure (session closing, queue full, prompt rejected):
   return a **warning** notice instead —
   `f"{text} — but the request was not sent: {exc}"` — so the old viewer at
   least prints that the turn did not start. The attach itself already
   happened and stays.

The receipt text is unchanged ("sending to lopdev. manager is coordinating.")
because it is now true.

**Known, documented degradation:** the viewer expands collapsed pastes into
the request text at submit time (`_submit_command_prompt`,
`app.py:7025-7029`); the paste payloads live in the *viewer's* composer and
cannot cross the wire retroactively. A runtime-admitted request that cites
`<[Paste #1, 240 lines]>` reaches the manager as the chip label. Images
survive; paste bodies degrade. That is strictly better than the current
silent drop, and the skew notice (A/C) tells the user how to get the full
fidelity back (`/reload`). This limitation goes in the method's docstring.

**TUI-hosted owner mirror.** A session can also be owned by a TUI process
(`TuiSessionHandle`, `mobile/tui_handle.py:84`; `app.py:11815` constructs it
for in-process sessions). Both hosts must agree, per the existing mirror
contract documented at `app.py:21291-21293`:

- `TuiSessionHandle.run_slash_authoritative` (tui_handle.py:~495) accepts and
  forwards `consumers` (same signature-probe tolerance on the server side).
- `OperatorApp.run_slash_authoritative` (`app.py:20889`) gains
  `consumers: Iterable[str] | None = None` and applies the same predicate to
  its `_slash_result` outcome. Its completion submits through the app's own
  user-row path so the row paints once and the echo registry stays
  consistent: `images = _image_blocks(wire_images)` (the same helper
  owned.py/tui_handle.py use), then
  `self._submit_prompt(request, images, None, typed=request)`
  (`app.py:12163`) — **not** `_submit_command_prompt`, which would re-resolve
  markers against a composer map that does not exist owner-side and drop the
  wire images. `_submit_prompt`'s own streaming branch steers when busy, so
  the mirror inherits the busy behaviour for free.

**Static guard.** `tests/unit/tui/test_noop_consumers.py` audits that every
produced `noop` type has a consumer. Extend it with the action-carrying
twin: every `SlashResult` producer in `owned.py` and `app.py` whose
`SlashResult(...)` call passes a `request` key in `data` must have its
`data["type"]` listed in `SLASH_ACTION_RECEIPTS`, and every
`SLASH_ACTION_RECEIPTS` entry must be consumed in
`_render_authoritative_slash`. Same AST-walk technique the file already uses —
a future action-carrying receipt then fails CI without anyone remembering this
incident.

### 4.2 C — version exchange over the existing record

The discovery record **is** the version channel: an attach client reads it
before dial (`find_owner_record`) and holds it at bind
(`RemoteSession._bind_to(record)`, remote.py:1002); the daemon reads records
the same way. No frame change is needed — the "ready/hello" for a v5 attach
client effectively arrives with the record. (The task suggested the frontend
sync frame; the record is strictly earlier, needs no pydantic change, and
covers the daemon too. A runtime's version cannot change while it lives, so
one comparison at bind is complete.)

- `local_operator/session/runtime/types.py` — `SessionRecord` gains two
  additive fields **in the additive block** (types.py:142-161 documents the
  contract: `from_json` drops unknown keys for old readers, new readers
  default missing keys, and `PROTOCOL_VERSION` deliberately does not move):

  ```python
  #: What build this runtime is running. "" for a runtime older than the
  #: field; an attach client compares it with its own build to name skew
  #: instead of failing silently under it (the /team build-skew incident).
  version: str = ""
  source_ref: str = ""
  ```

- `local_operator/session/runtime/server.py` — stamp at record construction
  (server.py:372-387): `version=..., source_ref=...` from
  `installed_build()`. Import `local_operator.update` **function-locally** in
  `__init__` — `update.py` is stdlib-only so there is no real cost, but
  server.py sits near the CLI startup path and the house style
  (`app.py:5564,5610`) is lazy import for it. The heartbeat rewrite republishes
  the dataclass via `to_json()`, so the fields ride every rewrite for free
  (coder: confirm `RecordPublisher` rewrites from the live record object, and
  pin it in a test).
- `local_operator/session/remote.py` — at bind, stash the owner's stamp on
  the facade: `self.owner_version = record.version`,
  `self.owner_source_ref = record.source_ref` in `_bind_to` (and the
  equivalent in `connect`).
- `local_operator/tui/app.py` — compare and warn (§4.3).
- `local_operator/cli.py` — `lop sessions --json` rows gain `"version"` and
  `"source_ref"` via `getattr(rec, ..., "")` next to the existing
  getattr-defaulted live-state fields (cli.py:~1990). The fixed-width table is
  **unchanged** — it is already eight columns, and leaving it alone keeps this
  PR off the band/layout design-review surface.

### 4.3 A — drift detection and skew notices in the TUI

**Snapshot.** `OperatorApp.__init__` (`app.py:2102`):
`self._loaded_build = update.installed_build()` (lazy import, as above). App
init is process start for a TUI, so the snapshot is "what this process
loaded". Debounce state: `self._skew_notice_shown: set[tuple]`.

**Check points** (one helper, `_check_build_skew(*, reason: str)`):

1. `_adopt_session` (`app.py:3137`) — covers `/new`, `/resume`, `/fork`,
   `/login`, initial mount, all of which adopt. Two comparisons here:
   - **A (disk drift):** `update.installed_build() != self._loaded_build` →
     warning notice (copy below).
   - **C (owner skew):** if the adopted session is already attached and
     exposes `owner_version`, compare with `self._loaded_build`. An empty
     `owner_version` means the runtime predates the field, i.e. it is older
     than this window by construction — warn once per session with the
     absent-version copy.
2. `_start_runtime_engage` (`app.py:8848`) and the success tail of
   `_bind_then_dispatch` (`app.py:9004`) — re-check A immediately before a
   spawn, and re-check C after a fresh bind (`ensure()` resolved
   `owner_version`). Both are debounced, so the mount engage, the draft
   warm-up, and the slash-command engage cost one comparison each and paint at
   most one notice per distinct (kind, from, to, scope) key — scope being the
   session id for the owner notices and empty for disk drift, which is a fact
   about the process rather than about any one session.

**Where the notice lands:** `_system_notice(body, "warning")` — its contract
(`app.py:14527-14538`) is exactly this class ("infrastructure the user did not
ask about") and it preserves the boot composition. Deliberately **not** a
status-band element: a band change is a persistent layout surface and triggers
the design-review path for marginal benefit over a notice the user reads at
the moment it matters (when they are about to engage a runtime).

**Notice copy** (rewritten wholesale by design review round 1 — D1/D2/D3.
The originals used our vocabulary, not the user's: "routed commands",
"predates build reporting", and a runtime/terminal split the notice never
explains, so to a reader those two words collapse into the one thing they can
see. Every string below was measured through the real `NoticeBlock` at
120/100/80 columns and is the same height or shorter than what it replaced.):

- A: `local-operator was updated after this window opened (0.49.5, cf2b854 →
  a1b2c3d) — this window is still on the old version. /reload updates it and
  picks this session back up.` The parenthetical names a SHARED version once
  and lets the refs carry the difference (D3), because the headline case is
  `lop-update` rebuilding `main` with no version bump — the two-arm
  `0.48.0@a1b2c3d → 0.48.0@4d3ce1d` spent 22 characters restating one version.
  When the versions genuinely differ, both arms are shown in full. This is a
  call-site branch (`app.py::_build_change`), not a change to
  `BuildStamp.label()`, which other callers share.
- C, known versions: `“<conversation name>” is running 0.46.23 but this window
  is 0.49.5 — some commands may not work. /stop, then send again to restart
  it.`
  (Bare `/stop` on a follower sends the stop op to the owner,
  `app.py:15097-15102`; the next prompt or `/team` engages a fresh runtime
  from the current disk build. Both runtimes keep working in the meantime —
  this is advisory, not a block.)
- C, absent version: `“<conversation name>” is running an older version than
  this window — some commands may not work. /stop, then send again to restart
  it.` Once per session per process: right after this release ships, every
  resident runtime legitimately triggers it once, and it is telling the truth
  each time.

Both C notices NAME the session, falling back to `this session` only when the
title is empty (D1). The subject is load-bearing rather than decorative: two
stale sessions in one terminal each earn a notice, and with a deictic subject
in both they render as two byte-identical paragraphs — which reads as the app
printing one warning twice, i.e. exactly the duplicate-notice bug the
per-session debounce key exists to avoid, and leaves `/stop` ambiguous about
which session it acts on.

**Old-runtime noop degradation (viewer-side, pairs with C).**
`_render_authoritative_slash` (`app.py:14629`): a `noop` whose `data.type` is
`"team_mutate"` or `"agent_mutate"` can only come from a pre-#624 owner.
Render a warning instead of returning silently:
`this session is running a version too old to attach a team (before
0.46.25); nothing was attached. /stop, then send again to restart it.` Two lines of renderer that close the last silent quadrant.
(Adding a *consumer* for a type no current producer emits does not disturb the
`test_noop_consumers` audit, which maps producers → consumers.)

### 4.4 Four-quadrant behaviour (the matrix the tests pin)

| Viewer \ Runtime | old runtime (no field, pre-B) | new runtime |
|---|---|---|
| **old viewer** (pre-#624, no `slash_consumers`) | status quo (old/ old: both sides pre-#624 behave as that pair always did — noop `team_mutate`, silent; unchanged by this PR) | **B: runtime admits the request; turn runs; viewer paints the row from the relay.** A warns at adopt/engage. |
| **new viewer** (declares receipts) | renderer degrades loudly (noop `team_mutate` → warning); C warns | declared ⇒ runtime never admits; viewer submits exactly as today. **Single turn, no double-submission.** |

## 5. Tests

Unit — runtime (new file
`tests/unit/session/runtime/test_action_receipt_completion.py`):

1. `run_slash_authoritative("team", "viewerteam do the thing", images,
   consumers=None)` on a handle with a scripted session → receipt is
   `team_attached` **and** the session received prompt `"do the thing"` with
   the same images. (`consumers=None` simulates the incident viewer.)
2. `consumers=[]` → also admits (declared-nothing is not a consumer).
3. `consumers=["team_attached"]` → receipt returned, **no prompt admitted**
   (the double-submission guard).
4. `agent_attached` with `request=""` (the `/agent clear` shape) → no
   admission.
5. Busy session (`is_streaming=True`) → `steer` called, not `prompt`.
6. Admission raising (session closing) → warning notice naming the failure,
   attach still reported.
7. Server auth parse: `slash_consumers` lands on `_ClientConn`; absent →
   `None`; `_dispatch_payload` passes it to handles that accept it and not to
   those that do not (the existing `test_server.py` doubles prove the probe).

Unit — update/record:

8. `installed_build()`: with a `.lop-source` fixture → `(version, ref)`;
   without → `(version, "")`.
9. `SessionRecord` round-trip with the new fields; `from_json` on a payload
   lacking them → `""` defaults; a payload with extra unknown keys still
   parses (existing forward-compat test pattern).

Unit — TUI (new `tests/unit/tui/test_build_skew.py`, plus one audit edit):

10. Drift: monkeypatch `installed_build` to a new token after app init →
    `_adopt_session`/engage posts exactly one warning naming both builds and
    `/reload`; a second adopt with the same token posts nothing (debounce).
11. Owner skew: facade with `owner_version` older/absent → the C notice with
    `/stop` copy; matching version → silence.
12. Renderer: `noop {"type": "team_mutate"}` → warning notice, no exception,
    no submit; `team_attached` with request on a normal session still calls
    `_submit_command_prompt` (regression guard that the declaration work did
    not break the consumer).
13. `test_noop_consumers.py` extension: every producer `data` dict containing
    a `request` key has its type in `SLASH_ACTION_RECEIPTS`; every
    `SLASH_ACTION_RECEIPTS` entry is consumed by the renderer.

E2E (`tests/e2e/test_viewer_attach_e2e.py`, same in-process-runtime harness as
`test_a_viewer_runs_a_team_and_holds_a_credential`, isolated config dir):

14. **Skew simulation:** raw `AttachClient` constructed **without**
    `slash_consumers` (that *is* the old viewer on the wire) →
    `slash_result("team", "viewerteam do the thing", [])` → assert the
    receipt is `team_attached` and the session ran the turn (the scripted
    stream's reply was consumed exactly once — the user row exists in the
    transcript).
15. **No double admission:** a client **with** `slash_consumers =
    SLASH_ACTION_RECEIPTS` → same command → assert the session received **no**
    prompt (the runtime deferred to the client). This is the guard against
    double-submission.
16. Record: the published record carries `version == installed_version()` and,
    under a fake `.lop-source`, the matching `source_ref`.

QA matrix notes for the qa-tester (real execution, isolated `HOME` per cell —
see AGENTS.md): cell 1 = e2e cell 14 driven by hand against a worktree-built
runtime; cell 2 = a real two-install drift: `uv tool install --force` the repo
into an isolated `UV_TOOL_DIR` at two different refs, launch the TUI from
ref 1, flip the dir to ref 2, `/new` + `/team <name> <request>` → expect the
drift notice and a running turn; cell 3 = attach a current TUI to a runtime
started from the older ref → expect the C notice; cell 4 = `/reload` from the
drifted TUI resumes the same session on the new build.

## 6. Risks to watch during rollout

1. **Double-submission.** The invariant is "admit iff the type was not
   declared". Test 3 and e2e cell 15 pin it. The residual human-audit point:
   the declaration list must always equal the renderer's consumed set — the
   extended `test_noop_consumers` audit makes that a CI property, not a
   remembered one.
2. **Mid-turn attach.** A `/team` request routed while a turn runs becomes a
   *steer* under completion, matching what the new viewer does today
   (`app.py:12235`). It is delivered at the next boundary; that is existing
   behaviour, not a new queue.
3. **Admission failure after a successful attach** leaves the team attached
   with no turn. The warning receipt names it; the retry is re-sending the
   request. Acceptable, and strictly louder than today.
4. **Paste-payload degradation** under runtime completion (chip label instead
   of body; §4.1). Documented in the method docstring; the skew notice points
   at `/reload` for full fidelity.
5. **`importlib.metadata` behaviour** was verified empirically on 3.14.3
   (§3); the `.lop-source` ref is the primary token precisely because the
   common drift on this host (same-version rebuilds) is invisible to
   `version()`. Editable checkouts have no `.lop-source` and share one
   version string — dev-tree skew is out of scope by design.
6. **Lazy-import Frankenstein.** A days-old TUI that lazily imports a module
   *after* `lop-update` mixes builds inside one process. A detects the
   precondition at the next adopt/engage and names `/reload`; nothing can make
   the already-loaded modules young again. Not made worse by this PR.
7. **Post-release noise.** Every resident runtime predates the `version`
   field when this ships, so the C "predates build reporting" notice fires
   once per (session, TUI process) during the first attach window. It is
   accurate each time; debounce keeps it to once.
8. **Phone/daemon path.** The daemon never sends `slash_consumers`, and the
   only routed slash it drives is `/mcp reauth` (daemon.py:655-668), which is
   not action-carrying — no completion can trigger for it. `SessionRecord`
   additive fields are invisible to older daemons by the existing
   `from_json` filter.

## 7. PR shape

One PR, conventional commit
`fix(runtime): detect and complete across viewer/runtime build skew`. Files:
`local_operator/update.py`, `local_operator/session/runtime/types.py`,
`local_operator/session/runtime/server.py`,
`local_operator/session/runtime/owned.py`,
`local_operator/mobile/attach_client.py`,
`local_operator/mobile/tui_handle.py`, `local_operator/session/remote.py`,
`local_operator/tui/app.py`, `local_operator/cli.py`; tests as in §5. No
`pyproject.toml` bump. Gates per AGENTS.md: flake8 / black==26.1.0 /
isort==5.13.2 / pyright over the whole tree, the unit suite, the e2e stage
(`-m e2e -n0`), then the standing reviewer + QA rounds. User-visible surface
is notice text only — no band or layout change.
