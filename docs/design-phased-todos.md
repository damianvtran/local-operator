# Design: phased todos for local-operator

Status: proposal (design only — no source changed by this document)
Branch: `dev-phased-todos` (cut from `origin/main` @ f5dadc69)
Author: architect (lopdev team)
Scope: the `todo` tool (`local_operator/tools/builtin.py`), the session
continuation guardrail (`local_operator/session/session.py`), and the dock-band
`TodoPanel` (`local_operator/tui/widgets/todo_panel.py` + `app.py` wiring).

---

## 1. Problem

The `todo` tool today keeps a **flat** list per owner: the store is
`dict[owner -> list[{"text","status"[,"reason"]}]]`
(`builtin.py:3680`, `3738`). The dock panel renders it as one `Todos · n/total
resolved` header followed by up to eight `- [ ] text` rows
(`todo_panel.py:_build`, `210-291`). There is no grouping: a twelve-item plan
across three semantic stages reads as one undifferentiated column, the panel
either shows the first eight and hides the rest behind `… N more todos`
(`todo_panel.py:249-271`), and completed work stays on screen forever, crowding
out the rows the user actually needs to follow.

The user wants:

1. Todos grouped under named **phases**, each phase a header with its items
   indented beneath it and each item's completion status shown.
2. Completed phases removed from the panel view **after a delay**, so the panel
   tracks *current* work.
3. An interactive control (**hotkey and/or click**) to expand to show ALL
   todos, or collapse back to the current-work view.

omp (`~/oss/oh-my-pi/packages/coding-agent`) already ships exactly this. This
design mines its proven data model and collapse policy and ports the
*behaviour* — not the TypeScript idioms — onto local-operator's existing store,
guardrail, and row-budget machinery.

### What must not regress

Two functions are a hard contract the session guardrail depends on
(`builtin.py:3755`, `3778`; consumed at `session.py:3315-3331` and
`1232-1236`):

- `open_todos(session_id)` — the *single* definition of "open" the stop-time
  continuation guardrail (`Session._todo_continuation`) fires on. Excludes
  `blocked` deliberately.
- `todo_fingerprint(session_id)` — the `(text,status)` tuple over EVERY item in
  order that the no-progress latch compares between two yields.

And the panel/receipt mirror is load-bearing: `_TODO_MARKS`
(`builtin.py:3702`) and `STATUS_MARKS` (`todo_panel.py:108`) must spell one
list identically, because the transcript `view` receipt and the dock band
describe the same data (AGENTS.md, and the comment at `todo_panel.py:102-108`).

---

## 2. Guiding decisions (summary, then detail)

| # | Decision | Recommendation |
|---|----------|----------------|
| A | Store shape | Phased: `list[{"name", "items":[{text,status[,reason]}]}]`. Flat `init` maps to one implicit phase named `"Todos"`, rendered **headerless**. |
| B | Ops | Keep the existing six op names. Add an optional `phase` field for addressing; add a `phased` init payload alongside flat `items`. Do **not** add `start`/`unblock`/`rm`/`append` — local-operator has no `in_progress` status and no per-item removal today. |
| C | Guardrail | `open_todos` = pending items across all phases (flattened, unchanged shape). `todo_fingerprint` grows to a **3-tuple** `(phase_name, text, status)` per item in phase-then-item order. `_stamped_todo_fingerprint` must widen to match — a coupled edit. |
| D | Panel | Phase-header rows with per-phase `done/total`, items indented; single-phase lists stay headerless (byte-identical to today). Reuses the existing `_body_rows`/`_row_cells` budget. |
| E | Auto-remove | **Hide** settled phases in the panel after a delay (default 60s), never mutate the store. Only fully-settled phases are hidden; never scrub closed rows while a phase has open work. Timer lives app-level. |
| F | Expand/collapse | Hotkey **`ctrl+t`** (audited free) toggles a panel `expanded` flag. Hotkey-only for round 1; a clickable affordance is feasible (band siblings already handle `on_click`) and is proposed as a fast-follow, not a blocker. |

The smallest coherent change is **B without new ops**: local-operator's status
vocabulary is `pending/done/blocked/dropped` (`builtin.py:3702`), with no
`in_progress` and no literal removal. omp's `start`/`unblock`/`rm` exist to
serve a richer status model this harness does not have. Adding them would be a
second, larger feature bolted onto this one; this design deliberately declines
them and says so.

---

## 3. Store item / phase model (decision A)

### 3.1 Shape

```python
# builtin.py — the new persisted shape
TodoItem  = dict[str, str]            # {"text", "status"[, "reason"]}  (UNCHANGED)
TodoPhase = dict[str, Any]            # {"name": str, "items": list[TodoItem]}
TodoStore = dict[str, list[TodoPhase]]   # owner -> phases
```

Every phase holds the **same item dicts** as today. That is the key
back-compat lever: `_match_todos`, `_todo_rows`, `_TODO_MARKS`, the blocked
`reason` handling, and the panel's `_item_row` all operate on an item dict and
are reused verbatim once a phase's `items` list is in hand.

### 3.2 Flat → phased mapping

A flat `init` (`items=[...]`, no phases) creates **one implicit phase**:

```python
[{"name": "Todos", "items": [{"text": t, "status": "pending"} for t in items]}]
```

`"Todos"` is the sentinel implicit-phase name. The panel renders a
single-phase list **without a header** (§6.3), so an existing caller that
never mentions phases sees the identical panel it sees today. This mirrors
omp's single-phase case (`interactive-mode.ts:2210`, `multiPhase = phases.length
> 1`; `renderPhase` uses the bare `phase.name` when not multi-phase, and the
root header drops its `· n/total` fraction).

> **Why a wrapper phase rather than a `phases | None` union.** Keeping the
> store a *uniform* `list[phase]` means every reader (panel, fingerprint,
> guardrail, `view`) has exactly one shape to walk. A `None`-or-`phases` union
> pushes a branch into all four readers and is where the omp/local-operator
> style ("one builder, one shape") says the bug will live. The cost is a
> one-time migration read (§9).

### 3.3 Migration of a persisted flat store

The store is **process-global and in-memory** (`builtin.py:3680`,
"In-memory todo lists"; nothing serialises it to disk — verified: no
`TODO_STORE` persistence path exists). So there is **no on-disk migration**.
The only compatibility surface is *code that reads the store mid-process*:

- `todo_items()` in the panel (`todo_panel.py:116`) reads `TODO_STORE.get(id)`.
- `open_todos`/`todo_fingerprint` read `TODO_STORE.get(id)`.

All three must tolerate a list whose elements are **either** old flat item
dicts (`{"text",...}`) **or** new phase dicts (`{"name","items"}`), because a
long-lived session could hold a flat list written by the pre-upgrade tool and
then be read by post-upgrade code within the same process across a `lop-update`
… **no** — `lop-update` restarts the process, so the store is empty on the new
binary. The realistic mixed-read case is only *within one process*, and within
one process only the new tool writes. **Therefore a normaliser is defensive,
not required**, but we add one anyway (cheap, one function) so a hand-attached
`ToolContext.todos` holding legacy data cannot crash a reader:

```python
def _as_phases(raw: list[Any]) -> list[TodoPhase]:
    """Coerce a stored owner-list to phases. A legacy flat list (items at the
    top level) becomes one implicit 'Todos' phase; a already-phased list is
    returned as-is. The one shape every reader walks."""
    if raw and isinstance(raw[0], dict) and "items" in raw[0]:
        return raw  # already phased
    return [{"name": "Todos", "items": list(raw)}]  # legacy flat
```

---

## 4. Tool schema & ops (decision B)

### 4.1 `TodoParams` additions

```python
class InitPhase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phase: str = Field(description="phase name — a short noun phrase, e.g. "
                                   "'Foundation', 'Auth', 'Verification'. No "
                                   "'1.'/'Phase 1:' prefixes.")
    items: list[str] = Field(description="task texts for this phase, in order")

class TodoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["init", "add", "done", "block", "drop", "view"] = Field(...)
    items: list[str] = Field(default_factory=list, description=...)   # UNCHANGED
    # NEW — phased init payload; mutually-exclusive with `items` on init:
    phases: list[InitPhase] = Field(
        default_factory=list,
        description="Phased task list for 'init': each entry groups task texts "
                    "under a named phase. Use this OR 'items' (flat), not both. "
                    "Phases render as headers with their items indented beneath.",
    )
    # NEW — phase target for add/done/drop/block:
    phase: str = Field(
        default="",
        description="Phase to address. For 'add', append into this phase "
                    "(lazily created if new). For 'done'/'drop'/'block' with no "
                    "'items', resolve EVERY open item in this phase. Omit to "
                    "target the whole list (add → implicit 'Todos' phase).",
    )
    reason: str = Field(default="", description=...)                  # UNCHANGED
```

`extra="forbid"` stays: unknown keys still fail loud. The **field
descriptions are the contract the model reads** — they are written to spell out
the phased-vs-flat choice and the "phase target resolves every open item" rule,
matching local-operator's descriptions-discipline (the existing `op`/`items`
descriptions do the same work).

### 4.2 Per-op behaviour

| op | Input | Behaviour |
|----|-------|-----------|
| `init` | `phases:[{phase,items}]` | Replace the whole store with those phases. |
| `init` | `items:[...]` (flat) | Replace with one implicit `"Todos"` phase (§3.2). |
| `init` | both set | Error: "init takes `phases` OR `items`, not both." |
| `add` | `items`, optional `phase` | Append pending items into `phase` (default `"Todos"`, lazily created — mirrors omp `append`). Existing dedupe-by-open-text rule (`builtin.py:3902`) applies **within the target phase**. |
| `done`/`drop` | `items` | Resolve those texts wherever they live (search all phases). |
| `done`/`drop` | `phase` (no `items`) | Resolve every currently-open item in that phase. |
| `block` | `items`, `reason` | As today, search all phases; `reason` required. |
| `block` | `phase`, `reason` | Block every open item in that phase. |
| `view` | — | Echo the list grouped by phase (§4.4). |

**`_match_todos` change (`builtin.py:3815`).** Today it scans one flat
`current`. The minimal change is to have callers pass the **flattened item
list** across all phases, preserving the "first item not already in target
status wins" idempotency rule unchanged. Because the item dicts are shared by
reference between the flat view and their phase, mutating `item["status"]` in
place updates the phase too — no re-indexing needed. Concretely:

```python
def _all_items(phases: list[TodoPhase]) -> list[TodoItem]:
    """Every item across phases in phase-then-item order — the flat view the
    match/progress/fingerprint helpers walk. Items are shared by reference with
    their phase, so mutating one here updates its phase in place."""
    return [item for phase in phases for item in phase["items"]]
```

`done`/`block`/`drop` with a `phase` target select
`[i for i in phase["items"] if i["status"] in ("pending","blocked")]` and apply
the status directly (still idempotent: a fully-resolved phase resolves nothing
and reports it).

### 4.3 Ops deliberately NOT added

- **`start`** — local-operator has no `in_progress` status. omp promotes the
  earliest open task to `in_progress` (`normalizeInProgressTask`,
  `todo.ts:146`) to drive its "current work" highlight. Local-operator's panel
  has no such highlight and the guardrail treats every non-resolved item as
  open. Adding `in_progress` is a separate feature (it changes what
  `open_todos` and the marks mean); out of scope.
- **`unblock`** — no caller need surfaced; a blocked item is re-opened today by
  re-`init` or left blocked. Adding it is safe but unnecessary for the stated
  goal; defer.
- **`rm`** — local-operator resolves via `drop` (status `dropped`), never
  literal removal, so the transcript keeps the record. Keep that; do not add
  hard removal.

Declining these keeps the op surface at six names and the diff small.

### 4.4 `view` output

Grouped, so the receipt mirrors the panel. Single implicit phase → today's
flat output (no header). Multi-phase → a header line per phase:

```
Foundation (2/3)
- [x] scaffold the module
- [x] wire the config
- [ ] add the migration
Verification (0/2)
- [ ] run the gate
- [ ] capture the frame
```

`_todo_rows` (`builtin.py:3802`) is reused per phase. The `(done/total)` count
uses `_TODO_RESOLVED` exactly like `_todo_progress`.

---

## 5. Guardrail contract (decision C) — the riskiest coupling

### 5.1 `open_todos`

Unchanged **shape** (`list[dict]` of pending item copies), new **source**:

```python
def open_todos(session_id: str) -> list[dict[str, str]]:
    raw = TODO_STORE.get(session_id)
    if not raw:
        return []
    phases = _as_phases(raw)
    return [dict(item)
            for phase in phases
            for item in phase["items"]
            if item.get("status") == "pending"]
```

Open = pending items across ALL phases, phase-then-item order. `blocked` stays
excluded (the honest-stop escape hatch, `builtin.py:3761`). The returned dicts
are plain item dicts, so `_todo_reminder_text` (`session.py:370`, lists
`item['text']`) needs **no change**.

> Should the reminder name the phase? Optional and *not* recommended for round
> 1: the reminder lists open item texts so the model can echo them into
> `todo done` (`session.py:375`), and phase names are not needed to match a
> text. Leave the reminder text alone to keep the guardrail diff minimal;
> revisit only if a model conflates same-text items in different phases (rare;
> texts are unique identifiers by the tool's own contract).

### 5.2 `todo_fingerprint` — grows to a 3-tuple

The latch must detect movement, and with phases a *phase rename* or an item
moving between phases is movement the current 2-tuple is blind to. Include
phase identity:

```python
def todo_fingerprint(session_id: str) -> tuple[tuple[str, str, str], ...]:
    """(phase_name, text, status) for EVERY item, phase-then-item order."""
    raw = TODO_STORE.get(session_id) or ()
    phases = _as_phases(list(raw)) if raw else ()
    return tuple(
        (phase["name"], str(item.get("text", "")), str(item.get("status", "pending")))
        for phase in phases
        for item in phase["items"]
    )
```

### 5.3 The coupled edit nobody must miss

`_stamped_todo_fingerprint` (`session.py:499`) normalises the fingerprint
stamped in a reminder's `details` back for comparison, and it **filters to
`len(item) == 2`** (`session.py:513`). If `todo_fingerprint` starts emitting
3-tuples while this normaliser still keeps only 2-tuples, the stamped side
becomes empty and `expired()` (`session.py:1234`) is `True` for every reminder
on every render — the latch still *functions* (it errs to expiring, the safe
direction per that function's own docstring) but the no-progress suppression is
defeated: the model gets nudged every yield. **This must change in lockstep:**

```python
# session.py:_stamped_todo_fingerprint — widen to 3-tuples
return tuple(
    (str(item[0]), str(item[1]), str(item[2]))
    for item in stamped
    if isinstance(item, (list, tuple)) and len(item) == 3
)
```

This is the single most important line in the whole change to get right, and
it is in a *different file* from the tool. It is called out again in the test
plan (§8) and the workstream split (§9): **workstream A owns both `builtin.py`
and this `session.py` line as one atomic unit.**

### 5.4 What the guardrail does with hidden phases

Panel auto-hide (§7) is **view-only**: it never touches the store, so
`open_todos` and `todo_fingerprint` are unaffected. A hidden phase whose items
are still pending is still open work the guardrail nudges on. Hidden ≠
resolved — stated explicitly because the opposite (hiding = clearing) would let
a stalled plan read as finished to the guardrail, exactly the failure the
`blocked`-exclusion comments guard against.

---

## 6. Panel rendering (decision D)

### 6.1 What the panel reads

`todo_items()` (`todo_panel.py:116`) returns copies today. It changes to return
**phases** (each a `{"name", "items":[copied dicts]}`), via `_as_phases` so a
legacy flat store still yields one implicit phase. The panel's `sync`
fingerprint (`todo_panel.py:180-187`) grows phase identity so a repaint fires
on phase-level change:

```python
fingerprint = tuple(
    (phase["name"],
     str(item.get("text","")), str(item.get("status","pending")),
     str(item.get("reason","")))
    for phase in phases for item in phase["items"]
)
```

The `(fingerprint, budget)` equality guard (`todo_panel.py:197`) is otherwise
unchanged — it still repaints only when the list or its row budget moves, and
now also when the collapse/expand flag or the set of hidden phases moves (both
folded into a widened state tuple, §7.3).

### 6.2 Row model

A phase becomes:

- one **header row**: `PhaseName · done/total` (per-phase resolved count via
  `RESOLVED_STATUSES`, `todo_panel.py:113`), styled `muted` name + `dim`
  progress — the same treatment the current `Todos · n/total resolved` header
  uses (`todo_panel.py:229-234`) and the same omp uses
  (`interactive-mode.ts:2256-2262`).
- its items, each rendered by the **existing** `_item_row`
  (`todo_panel.py:423`) — status marks, strike, blocked/dropped tags all
  unchanged — but **indented one gutter** (a two-space lead before the
  `- [ ]`). Indentation is added in `_build`, not in `_item_row`, so
  `_item_row` stays the single mark authority the receipt mirrors.

The top summary line stays: `Todos · <active-phase-index>/<phase-count>` in
multi-phase mode (omp's root header, `interactive-mode.ts:2280-2283`), or the
existing `Todos · n/total resolved · N dropped` line in single-phase mode.

### 6.3 Single-phase = headerless = unchanged look

When `len(phases) == 1` and the phase is the implicit `"Todos"`, the panel
skips the per-phase header and the item indent, and renders exactly today's
output. This is the back-compat guarantee: the `test_band_panels.py` tests that
assert `"Todos · 5/5 resolved · 4 dropped"` (`:392`) and flat row content
(`:301`) keep passing untouched.

### 6.4 Fitting the row budget

The panel's budget machinery (`_body_rows`, `_row_cells`, `_DOCK_ROWS`,
`MAX_TODO_ROWS`, the `… N more` marker) is **preserved**. Phase headers are
rows too, so they consume from the same `cap` (`todo_panel.py:243-259`). Two
adjustments:

1. **Row accounting counts headers.** The `cap`/`marker` arithmetic operates on
   a *pre-flattened list of render rows* (headers + items) rather than raw
   items, so a header can never push the composer off screen. Build the row
   list first (headers interleaved with items, per §7's collapse policy), then
   apply the existing `cap`/marker/clip logic to that list unchanged.

2. **The `… N more` marker counts hidden items, not hidden rows** — keep it
   counting todos (not headers) so it stays truthful to the user
   (`todo_panel.py:269` "Counts what the reader cannot see"). Compute it from
   the item count, not the row count.

This keeps the `Screen { overflow: hidden }` clipping guarantee
(`todo_panel.py:194-195`) intact: the panel still sizes against `_body_rows()`
and never asks for more.

---

## 7. Collapse policy + auto-remove (decisions E, F) — port omp's walking viewport

### 7.1 The two visibility layers

omp has two orthogonal mechanisms; local-operator needs both, kept distinct:

1. **Auto-clear timer** (`#syncTodoAutoClearTimer`, `interactive-mode.ts:2112`)
   — after `todoClearDelay` seconds with a **fully-settled** list, the HUD
   stops rendering it. omp clears the whole list; we **hide settled phases**
   (§7.4).
2. **Collapse/expand** (`todoExpanded`, `toggleTodoExpansion`,
   `interactive-mode.ts:5166`) — collapsed applies the walking-viewport cap and
   shows only the active phase + a few following; expanded shows everything.

### 7.2 Port `selectCollapsedTodos` (the walking viewport)

omp's `selectCollapsedTodos` (`todo.ts:332`) + `selectWithinCap`
(`todo.ts:286`) + `isClosedTodo` (`todo.ts:242`) are the proven policy. Ported
to Python (no `isMatched`/subagent-highlight axis — local-operator's panel has
no subagent-todo matching, so drop that parameter):

```python
def _is_closed(item: TodoItem) -> bool:
    return item.get("status") in ("done", "dropped")

# COLLAPSED_CLOSED_CONTEXT: keep the last N closed rows directly above the open
# window so a just-completed item stays visible as it settles (omp keeps 1).
_COLLAPSED_CLOSED_CONTEXT = 1

def select_collapsed(items: list[TodoItem], cap: int) -> tuple[list[TodoItem], int]:
    """Return (rows_to_show, hidden_open_count) for one phase's collapsed
    preview. Open items fill the cap; the last closed item is kept as context
    so a completion is visible as it happens; a settled phase selects over its
    own closed rows. Direct port of omp selectCollapsedTodos (todo.ts:332),
    minus the subagent-match axis this panel does not have."""
    open_items = [i for i in items if not _is_closed(i)]
    if not open_items:
        shown = items[-cap:] if len(items) > cap else items
        return shown, 0
    lead = [i for i in items if _is_closed(i)][-_COLLAPSED_CLOSED_CONTEXT:]
    within = open_items[:cap]
    hidden = len(open_items) - len(within)
    return [*lead, *within], hidden
```

The per-phase item cap mirrors omp's `activeTaskCap = 5`
(`interactive-mode.ts:2214`); the number of phases shown after the active one
mirrors `subsequentStageCap = 4` (`:2213`). The **active phase** is the
earliest phase still holding open work (omp `formatSummary`'s `currentIdx`,
`todo.ts:732`) — port that as `_active_phase_index`.

> **This is where the omp collapse-policy test cases port directly** (§8): the
> "active work exceeds cap", "out-of-order completion keeps a closed lead", and
> "settled phase selects over itself" cases are exactly `selectWithinCap` /
> `selectCollapsedTodos`'s documented behaviours (`todo.ts:273-344`).

### 7.3 Panel state

Add to `TodoPanel`:

```python
self._expanded: bool = False              # collapse/expand flag (ctrl+t)
self._settled_since: dict[str, float] = {} # phase-name -> monotonic time it
                                           #   first became fully settled
```

`_settled_since` drives the auto-hide (§7.4). The `sync` equality guard's state
tuple widens to `(fingerprint, budget, expanded, hidden_phase_names)` so a
collapse toggle or a phase crossing the hide threshold forces a repaint.

### 7.4 Auto-hide settled phases (decision E: hide, not delete)

**Recommendation: hide in the panel view; leave the store intact.** Rationale:

- `view` and `todo_fingerprint` must still see the full list (a resolved item
  is a record the transcript and the latch both need).
- Deleting from the store would make the guardrail's fingerprint jump (the
  latch would read deletion as movement and re-nudge), and would lose the
  `view` receipt's record.

The rule, ported from omp's `#isTodoListSettled` safety invariant
(`interactive-mode.ts:2101`, the comment there is the whole reason this is
delicate):

- A phase is **hideable** only when **all** its items are closed
  (`done`/`dropped`) — never while it holds a pending or blocked item.
  Hiding a phase mid-flight would reset its `done/total` counter and drop the
  closed rows the progress count is computed from.
- A hideable phase is hidden once it has been continuously settled for
  `TODO_PHASE_HIDE_DELAY_S` (**propose 60s**, matching omp's default
  `todoClearDelay`). `_settled_since[name]` records when it first settled; it
  is cleared if the phase gains open work again (re-`init`/`add`), so a phase
  that reopens restarts its clock.
- When **every** phase is hidden, the panel shows only the collapse affordance
  line (§7.6), or hides entirely if there is nothing to expand to — but see the
  guardrail note: the *store* may still hold pending items, so "panel empty"
  never implies "work done".

### 7.5 Where the timer lives — app-level, riding the existing poll

**Recommendation: no new timer object.** The panel already repaints on the 1 Hz
job poll (`app.py:1589`, `JOB_POLL_INTERVAL_S = 1.0`; `_refresh_band` →
`todo_panel.sync`, `app.py:6668-6715`) and on `tool_execution_end` for the todo
tool (`app.py:11327-11328`). The auto-hide is a **time comparison inside
`sync`**, not a scheduled callback:

```python
now = time.monotonic()
for phase in phases:
    if _phase_settled(phase):
        self._settled_since.setdefault(phase["name"], now)
    else:
        self._settled_since.pop(phase["name"], None)
hidden = {name for name, t in self._settled_since.items()
          if now - t >= TODO_PHASE_HIDE_DELAY_S}
```

Because the panel repaints every second, a phase crossing 60s is hidden within
~1s of the threshold — visually indistinguishable from a dedicated timer, and
it inherits the equality-guard's cheapness (the guard's state tuple includes
`hidden`, so the second the set changes it repaints, and otherwise it does
not). This avoids adding a `set_timer`/`set_interval` and the teardown
bookkeeping that comes with it, and it means no timer can fire against a torn-
down panel. The 1 Hz poll is already unconditional, so there is no "list
settled but poll stopped" gap (omp needs its own timer precisely because its
HUD is event-driven, not polled — local-operator's poll removes that need).

> Evidence that would change this: if profiling showed the per-second phase
> scan is measurable on a large list, move the threshold check behind a
> `set_timer` armed only when a phase first settles. Given the list is at most
> a few dozen items and the guard already runs every second, this is very
> unlikely.

### 7.6 Expand/collapse control (decision F)

**Hotkey: `ctrl+t`.** Audited free — no `ctrl+t` in `local_operator/tui/`
(grep clean) and not in Textual's `TextArea.BINDINGS` (verified:
`ctrl+a/e/w/d/x/k/f/u/left/right` are bound, `ctrl+t` is not, so the composer
keeps every editing key). Mnemonic: **t**odos. Add:

```python
# app.py BINDINGS
Binding("ctrl+t", "toggle_todos", "Expand/collapse todos", show=False),

def action_toggle_todos(self) -> None:
    if self._todo_panel is not None:
        self._todo_panel.toggle_expanded()   # flips _expanded, forces a repaint
        self._refresh_band()                  # settle inset + budget in one tick
```

`show=False` matches every other binding in this app (`app.py:917-960`). Not
`priority=True`: like Esc (`app.py:937`), it should bubble so a focused picker
keeps first refusal on the key; unlike `shift+tab` there is no Screen-level
binding on `ctrl+t` to jump ahead of.

**Collapsed vs expanded:**

- **Collapsed** (default): hidden settled phases are omitted; remaining phases
  run the walking-viewport cap (§7.2); a trailing affordance line names what is
  hidden and how to see it — e.g.
  `+3 done · ctrl+t to expand` (dim), analogous to the existing `… N more
  todos` marker but pointing at the control. Count = items in hidden/settled
  phases plus items dropped by the per-phase cap.
- **Expanded**: every phase, every item, no per-phase cap, no auto-hide — the
  affordance line reads `ctrl+t to collapse`. Still bounded by `_body_rows()`
  and the `… N more` marker so it can never overflow the composer; expanded
  means "show all phases from the top", not "ignore the row budget".

**Clickable affordance — feasible, proposed as fast-follow, not a blocker.**
Band siblings already handle mouse: `SubagentPanel`'s rows implement
`on_click` (`subagent_panel.py:771`, calls `event.stop()` then an action). A
`TodoPanel.on_click` could toggle `_expanded` the same way. The obstacle is
**target precision**: the panel is one `Static` body, so a naive `on_click`
toggles on a click *anywhere* in the list, which is surprising (a user clicking
to place a selection would toggle). Making only the affordance *line* clickable
needs either a separate focusable widget for that row or hit-testing the click
`event.y` against the affordance row's offset. That is real work and orthogonal
to the hotkey. **Recommendation:** ship **hotkey-only** in round 1 (the user
asked for "hotkey and/or button"); add a clickable affordance as a scoped
fast-follow once the row model is settled, hit-testing the affordance row's `y`
and calling `event.stop()` (the band's mouse-isolation rule, AGENTS.md
"Overlays float"). Do not gate the feature on the click.

---

## 8. Test impact

### 8.1 Existing tests that change

| Test | File | Change |
|------|------|--------|
| `test_steering_mid_turn_does_not_end_the_turn_with_open_todos` | `session/test_todo_steering_e2e.py:128` | Asserts `TODO_STORE["e2e"]` item texts at `:178` — the store is now phased; update to walk phases (or assert via `open_todos`). |
| the `open_todos`/`TODO_STORE[...]["status"]` assertions | `test_todo_steering_e2e.py:183,281,282` | `open_todos` shape is unchanged (still item dicts) — most assertions pass as-is; only the raw `TODO_STORE["e2e"][0]` index (`:282`) must become a phase walk. |
| `test_todo_guardrail.py`, `test_todo_compaction.py` | `session/` | Re-run; any that construct `TODO_STORE` directly must build phases. Those going through the tool are unaffected. |
| `test_todo_panel_renders_*`, header/marks tests | `tui/test_band_panels.py:285-577` | Single-phase (implicit `"Todos"`) path must render **identically** — these are the back-compat guard and should pass unchanged. Verify each seeds a flat `init`. |

The guardrail's `_stamped_todo_fingerprint` widening (§5.3) is covered by the
existing steering e2e tests **only if** one of them exercises the
no-second-nudge path with a phased list — add that (§8.2) because the current
suite builds flat lists and would not catch a 2-vs-3-tuple mismatch.

### 8.2 New tests

**Tool (`builtin.py`), unit:**
- `init` with `phases:[...]` builds the phased store; `init` with flat `items`
  builds one implicit `"Todos"` phase.
- `init` with both `phases` and `items` errors.
- `add` into an existing phase; `add` into a new phase (lazy create); `add`
  with no `phase` → implicit `"Todos"`.
- `done`/`drop`/`block` by `phase` target resolves every open item in it and is
  idempotent (re-issue resolves nothing, reports cleanly).
- `done`/`drop` by `items` still finds items across phases; the
  first-not-in-target idempotency rule survives.
- `view` groups by phase with per-phase `(done/total)`; single implicit phase
  renders headerless.
- `_as_phases` coerces a legacy flat list.

**Guardrail (`session.py`), unit/e2e:**
- `open_todos` flattens pending across phases, still excludes `blocked`.
- `todo_fingerprint` is a 3-tuple including phase name; a phase rename moves the
  fingerprint; an item moving phases moves it.
- **`_stamped_todo_fingerprint` round-trips a 3-tuple** (the JSON list→tuple
  coercion at `session.py:509`), and a phased no-progress list is nudged exactly
  once (mirrors `test_a_model_that_cannot_proceed_is_nudged_once`
  `:230`, but phased) — this is the test that catches the §5.3 coupling.

**Panel (`todo_panel.py`), tui render:**
- Multi-phase list renders headers with `done/total`, items indented, marks
  unchanged.
- Single-phase implicit `"Todos"` renders headerless and byte-identical to the
  pre-upgrade output (assert against the existing goldens/strings).
- `select_collapsed` cases ported from omp `selectCollapsedTodos`
  (`todo.ts:332`): active work under cap; active work over cap (hidden count);
  out-of-order completion keeps one closed lead row; settled phase selects over
  its own closed rows.
- Auto-hide: a fully-settled phase is hidden after the delay (drive
  `time.monotonic` via a seam or a settable `_settled_since`), a phase with one
  pending item is **never** hidden, hiding does not change `open_todos`/
  `todo_fingerprint`.
- `ctrl+t` toggles expanded/collapsed; expanded shows a settled phase the
  collapsed view hid; the panel never exceeds `_body_rows()` in either state.
- Row budget: headers count toward `cap`; the `… N more` marker counts hidden
  *items* not rows; a short terminal (100x14) still fits the composer.

**Visual validation (AGENTS.md §Visual validation — required, not optional):**
capture before/after SVGs of the panel via the **real `OperatorApp`** (not
`_PanelHost`, which loads no CSS) at 100x30 and 100x14, in: multi-phase
collapsed, multi-phase expanded, a phase mid-hide vs after-hide, and the
single-phase headerless case (must match the before-frame pixel-for-pixel).
Check the geometry numbers (`app.screen.virtual_size` vs `size`, no scrollbar)
per AGENTS.md §4.

---

## 9. Migration/compat risks & implementation order

### 9.1 Risks to watch during rollout

1. **The `_stamped_todo_fingerprint` len-2→len-3 coupling (§5.3).** Highest
   risk: it is in a different file from the tool, fails *silently* (the latch
   errs safe, so tests that only check "does it nudge" pass while "does it stop
   nudging" quietly breaks), and re-nudges every yield if missed — burning the
   loop's `max_paused_turn_continuations` budget and spamming the model. The
   §8.2 no-second-nudge phased test is the guard; **both edits ship together**.
2. **Header rows in the row budget (§6.4).** A phase header is a row; if the
   `cap`/marker arithmetic still counts only items, a header can push the
   composer off screen on a short terminal — exactly the clip
   `_DOCK_ROWS`/`_body_rows` exist to prevent (`todo_panel.py:52-88`). Must
   flatten to render-rows *before* the cap.
3. **Auto-hide scrubbing a live phase (§7.4).** If the "fully settled" gate is
   wrong (e.g. treats `blocked` as closed), a phase with open work vanishes and
   its progress counter resets — omp's `#isTodoListSettled` comment
   (`interactive-mode.ts:2093`) documents this exact bug. `blocked` is **not**
   closed; only `done`/`dropped` are (`_TODO_RESOLVED`).
4. **Single-phase headerless drift.** If the implicit-`"Todos"` path ever grows
   a header, every existing panel test and the user's muscle memory break.
   Guarded by the byte-identical back-compat test (§8.2).
5. **Fingerprint churn from auto-hide.** Guarded by design: auto-hide is
   view-only, so `todo_fingerprint` never sees it. Watch that no one
   "optimises" it into a store mutation.

### 9.2 Two coder workstreams + the shared contract

The two halves meet at exactly one artifact: **the store shape** (§3.1) and the
**fingerprint tuple arity** (§5.2). Agree those two up front, in writing, before
either coder starts — they are the whole interface.

**Workstream A — tool + guardrail (backend, `coder` + `reviewer`).**
Owns `builtin.py` **and** the `session.py:_stamped_todo_fingerprint` line as
one atomic unit (§5.3). Deliverables: the phased store shape, `TodoParams`
additions, per-op behaviour, `_as_phases`, `open_todos`/`todo_fingerprint`
changes, the coupled `session.py` edit, and all §8 tool + guardrail tests.
No TUI. This workstream is where correctness lives and it must land — or at
least merge its shape — first, because B reads the store it defines.

**Workstream B — panel + app wiring (frontend, `coder` + `reviewer` +
`designer` + `ux-reviewer`).**
Owns `todo_panel.py` and the `app.py` binding/action + `_refresh_band` wiring.
Deliverables: `select_collapsed` port, phase-header render, indentation,
per-phase progress, the row-budget flattening, auto-hide via the poll,
`_expanded` + `ctrl+t`, the affordance line, all §8 panel tests, and the
required before/after visual validation. It is **user-visible and changes an
interaction** (a new hotkey and a new collapse flow), so it needs a **design
round** (`designer`, D-findings, on the rendered frames) and a **UX round**
(`ux-reviewer`, U-findings, walking the collapse/expand + auto-hide flow),
per the team's review gate.

**Ordering.** A first (or at minimum, A's store-shape and fingerprint-arity
merged/frozen), then B against the frozen shape. B can begin against a stub
that returns the phased shape as soon as §3.1/§5.2 are agreed, but must not
merge before A, or the panel reads a store the tool does not yet write.

**Both PRs** carry the standing review gate: an `### Agent review — round N`
from a `reviewer` other than the author, answered by remediation, fresh against
head; real execution evidence (run the tool, drive the panel), not a green
suite; B additionally the design + UX rounds. Conventional-commit messages;
comments explain the *why/constraints* (the auto-hide safety invariant and the
§5.3 coupling especially).

---

## 10. Recommendation in one paragraph

Adopt omp's proven model with the smallest coherent surface: a uniform
`list[phase]` store where a flat `init` becomes one implicit headerless
`"Todos"` phase (§3), the existing six ops plus an optional `phase` addressing
field and a `phases` init payload (§4, **no** new `start`/`unblock`/`rm` — this
harness has no `in_progress` status to justify them), `open_todos` unchanged in
shape and `todo_fingerprint` widened to a phase-aware 3-tuple with its
`session.py` normaliser widened in lockstep (§5), a panel that renders phase
headers with per-phase progress and indented items while single-phase lists
stay byte-identical to today (§6), auto-hide of fully-settled phases as a
view-only time comparison riding the existing 1 Hz poll rather than a new timer
(§7.4-7.5), and a `ctrl+t` collapse/expand hotkey with a clickable affordance
deferred to a scoped fast-follow (§7.6). The three riskiest decisions to watch:
**(1)** the cross-file `_stamped_todo_fingerprint` len-2→len-3 coupling that
fails silently; **(2)** flattening phase headers into render-rows before the
row-budget cap so a header cannot clip the composer; **(3)** the auto-hide
"fully settled" gate treating only `done`/`dropped` as closed so a live phase
is never scrubbed. Split into workstream A (tool + guardrail, lands first) and
workstream B (panel + hotkey + auto-clear + visual validation), meeting at the
frozen store shape and fingerprint arity.
