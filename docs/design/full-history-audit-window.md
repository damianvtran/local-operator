# Design brief: durable audit history behind the display window

Status: proposal (architect). Not implemented. Scope: `local-operator` @ `f1ab7d346`.

All line citations are against `f1ab7d346`, read in a clean worktree
(`/private/tmp/arch-hist/wt`) because the primary checkout has uncommitted edits
to `remote.py` and `app.py` from a sibling session.

---

## 1. The problem, as found

### 1.1 Confirmed: the display window inherits an LLM-context cut

`_capture_display_window` builds every page from the model's replay:

- `history_window.py:261` — `history = transcript.build_llm_history(through_id=through_id)`
- `transcript.py:1294-1366` — `replay_entries` finds the **latest** `ENTRY_COMPACTION`,
  emits `[compaction_summary marker, *preserved_user_turns]`, then replays only
  `entries[start:]` where `start` is the index of `first_kept_entry_id`.

Everything downstream is therefore computed over the post-compaction replay:
`total_message_count` (`history_window.py:351`), the `before_token` chain
(`:341-343`, `None` once `start == 0`), and `has_more` (`:350`). The bytes are
intact on disk; the reading surface simply cannot address them.

Re-measured independently across the operator's store (`Transcript(dir)`,
message rows on disk vs `len(build_llm_history())`):

| session | MB | msg rows | compactions | replayed | unreachable |
|---|---|---|---|---|---|
| `bda7b76d34e0` | 231 | 17,345 | 48 | 514 | 97.0% |
| `835fbcafdc27` | 60 | 14,324 | 26 | 630 | 95.6% |
| `9f8e5b652ac7` | 75 | 11,344 | 22 | 411 | 96.4% |
| `4140ee201ce1` | 78 | 10,548 | 14 | 464 | 95.6% |
| `28a800c6a783` | 15 | 4,372 | 9 | 638 | 85.4% |

Twelve of the operator's sessions exceed 80% unreachable. This is the defect.

### 1.2 The head notice is a faithful reporter, not the bug

`_reconcile_head_notice` (`app.py:7267-7269`) computes
`more = bool(self._resume_pending_head) or bool(session.history_before_token)`,
and `RemoteSession.history_before_token` (`remote.py:2603-2606`) returns the
window's `before_token` — `None` once the truncated replay is drained. So
`RESUME_START_NOTICE` (`app.py:1077`) is the honest rendering of a history layer
that already discarded the older rows. **Do not fix the copy.** Fix the layer.

### 1.3 The suspected second defect: not confirmed on the reported session

I drove `display_window` against `59cbf6dfc63f` directly. It has 294 entries, 281
message rows, **zero compactions**, 281 replayed. Its backward chain:

```
start=201 (80 msgs, has_more=True, total=281)
start=199 (2)   → start=113 (86)   → start=0 (113, has_more=False)
```

It terminates at 0 correctly, and 0 is the true beginning. **On that session
"start of conversation" was correct**, and there is no separable paging defect
demonstrated by it. The `_resume_pending_head = []` resets (`app.py:4972`,
`7076`, `15282`, `31546`) each guard a real transition (a snapshot that shrank, a
fresh resume, `/clear`, a canonical reset) and each is followed by a
re-projection of the *current* history, so none of them strands rows while
claiming exhaustion. `load_older_display_page`'s `full_required` branch
(`remote.py:2618-2620`) does not drop a page either — it escalates to
`materialize_history()` and returns the id-deduped difference.

What that session *does* show is a different, real, and much smaller thing worth
recording: 281 replayed rows project to only ~216 visible blocks (149 of the 281
are `role == "tool"` rows, skipped at `session_presentation.py:751-752`, and 11
are `session_state` customs with no renderer). A reader on a tall viewport
therefore drains the head faster than the message count suggests. That is
correct behaviour, not a defect.

**Conclusion for the coder: treat this as ONE defect, not two.** If the operator
saw "start of conversation" above genuinely unseen rows on a *compacted*
session, §1.1 explains it completely. I found no evidence for an independent
non-compacted paging bug, and I would not have a coder chase one on suspicion.
Evidence that would change my mind: a reproduction on a session with **zero**
compaction entries where `display_window`'s chain terminates at `start=0` while
`total_message_count < ` the count of message rows on disk. That comparison is
two lines of Python and belongs in the QA matrix (§7) precisely so the question
is settled by data rather than left open.

---

## 2. The seam

**Stop overloading one function.** `build_llm_history` answers "what may the
model see". Nothing should ask it "what did the conversation contain". Introduce
a second replay mode alongside it, in the same module, sharing the same row
rehydration.

### 2.1 `transcript.py` — the new primitive

```python
def replay_entries(
    entries, attachments, *, through_id=None, mode: Literal["context", "audit"] = "context"
) -> list[AgentMessage]: ...
```

- `mode="context"` — today's behaviour, byte for byte. Default. Untouched.
- `mode="audit"` — ignore the compaction cut entirely: replay **every**
  `ENTRY_MESSAGE` from index 0 through the cut, still applying prunes, and emit
  a `compaction_summary` `CustomMessage` **in place** at each compaction row
  (not just the latest, and not hoisted to the front).

Two consequences make audit mode strictly simpler than context mode:

- `preserved_user_turns` must **not** be re-injected in audit mode. They are
  verbatim copies of user rows that already exist before the cut
  (`transcript.py:1316-1343`), and audit mode replays those rows. Re-injecting
  would double every preserved turn, and would do so under the *original* ids
  (`:1339-1341`), producing duplicate ids in one list — which the TUI's
  `_resume_mounted_ids` dedupe (`app.py:7726`) would then silently swallow,
  hiding a real row. Skip them; the originals are right there.
- The `first_kept_entry_id` fallback (`:1344-1365`) is irrelevant: audit mode
  starts at 0 unconditionally, so the "silent amnesia" hazard that comment
  guards against cannot arise.

Add a windowed reader beside it — this is what keeps the design inside budget B:

```python
def audit_slice(entries, attachments, *, end_index: int, limit: int) -> list[AgentMessage]
```

Replays only `entries[lo:end_index]`, where `lo` walks back `limit` *message*
rows. Prunes are the only cross-row dependency and a prune is always appended
**after** its target (`transcript.py:376-380` documents exactly this), so a
prune targeting a row inside the slice may sit after `end_index`. Collect the
prune map from the whole `_entries` list (a cheap `type ==` scan over
already-parsed objects, no rehydration) and pass it in. Do not re-derive it from
the slice.

### 2.2 `history_window.py` — the field and the flag

`DisplayHistoryWindow` gains:

```python
audit: bool = False          # this page is pre-compaction history
audit_available: bool = False  # older rows exist behind the context cut
```

`display_window`/`_capture_display_window` gain `audit: bool = False`. Paging
becomes a two-phase chain over one continuous coordinate space:

1. **Context phase** — unchanged. Pages walk back through the replay to
   `start == 0`.
2. At `start == 0`, instead of `before_token = None`, mint a token carrying
   `{"phase": "audit", "end_index": <journal index of the cut>}` **iff** audit
   rows exist. Set `audit_available=True`.
3. **Audit phase** — tokens carry a journal `end_index`. Each page replays
   `audit_slice(...)` backward, sets `audit=True`, and mints the next token at
   the new `end_index`. `before_token` becomes `None` only when `end_index`
   reaches the first message row in the journal. **That is where paging
   terminates, and it is the only place it may.**

Both phases keep signing through `transcript._history_page_key` with the same
`conversation_id`/`owner_epoch`/`history_generation` claims (`:250-257`), so
compaction during a scroll still invalidates outstanding tokens into `reset` and
the existing reconciliation runs. `end_index` is a **journal index, resolved to
an entry id in the token** — sign the id, not the integer, for the same reason
`TranscriptPage` uses ids (`transcript.py:269-272`): `compact_file` replaces the
file and integer offsets become lies. On resolve-failure, return `status="reset"`.

`total_message_count` **stays the context total.** Do not inflate it. It is read
by `_sidebar_presentation_current` (`app.py:3922-3924`) as a monotonic growth
signal and by `history_message_count` (`remote.py:4280-4287`); redefining it
mid-chain would make cached presentations miss forever. Audit pages report their
extent through `start`/`audit` only. Expose the audit depth, if it is ever
needed, as a separate `audit_message_count` — but I would not add it until a
surface asks.

`theme_turn_count` and `opener_text` likewise stay context-derived.
`history_theme_turn_count` feeds `should_refresh_theme` (`naming.py:703-720`),
whose growth gate would re-fire across the board if the count jumped by 17,000.

---

## 3. Performance (budget B)

Measured, not assumed, on the operator's real journals:

| operation | `bda7b76d34e0` (231 MB) | `9f8e5b652ac7` (75 MB) |
|---|---|---|
| owner `Transcript` ctor (already paid) | 1.37 s | 0.34 s |
| naive full audit replay | 0.29 s, **47 MB peak** | 0.18 s, 33 MB |
| windowed audit slice (120 rows, tail) | **<1 ms** | 1 ms |
| windowed audit slice (120 rows, mid-file) | **<1 ms** | <1 ms |
| `read_transcript_page` backward page | 1.03–1.13 s | 0.31–0.42 s |

Three conclusions the coder must not relitigate:

1. **Serve audit pages from the owner's resident `_entries`, never from a disk
   re-read.** `Transcript.__init__` already parses the whole file into
   `self._entries` (`transcript.py:557-566`). The rows are in RAM before any
   page is requested; a windowed replay over them is ~1 ms. Using
   `read_transcript_page` instead would cost ~1 s per page on the 231 MB
   journal — a thousandfold regression for no benefit.
2. **Never call full audit replay.** 0.29 s and 47 MB of peak allocation per
   call is exactly what budget B forbids. `audit_slice` is the only entry point.
3. `read_replay_suffix` (`remote.py:99`, `transcript.py:360`) is for the
   *attach* path and stays exactly as it is. Audit paging is a post-attach,
   reader-driven gesture on an owner that already holds the journal. **The
   resume path's cost does not change at all.**

Page residency is bounded by the existing machinery, unchanged:
`DISPLAY_HISTORY_MESSAGES`/`DISPLAY_HISTORY_BYTES` (`history_window.py:27-28`)
bound each page; `_DisplayWindowCache` and `_retained_size` (`:82-162`) bound
per-owner retention; `RESUME_RENDER_MESSAGES` (`app.py:1049`) bounds first
paint; `SessionPresentation.retainable` (`session_presentation.py:317`) bounds a
parked view. Audit pages flow through all four untouched — they are ordinary
`DisplayHistoryWindow` pages carrying an extra boolean.

One thing to watch (§8): audit pages enter `_resume_pending_head`, which
`retainable()` measures directly (`session_presentation.py:386`), and its
docstring records the retained-text figure as 127–357 KiB against a 1 MiB
budget. A reader who scrolls far into a 17,000-row audit history will push a
parked presentation past `RETAIN_TEXT_BYTES` and it will stop being cached.
That is the budget **working as designed** — the presentation is re-prepared,
not corrupted — but it will look like sidebar lag on exactly the sessions this
feature targets. Do not "fix" it by raising the budget; that docstring records
two prior mis-tunings in that direction.

---

## 4. Honest presentation (requirement C)

Pre-compaction rows are real history the model can no longer see. Both facts
must be on screen.

**Reuse the existing marker.** `replay_entries` already emits a
`compaction_summary` `CustomMessage` (`transcript.py:1308-1315`); audit mode
emits one *per* compaction, in place. But note: `project_settled_rows` has **no
branch for `compaction_summary`** — verified, the custom-type branches are
`wake_prompt`, `peer_message`, `gate_timeout` and `COMPACTION_REFUSED_TYPE`
(`session_presentation.py:613-668`), and anything else falls through to
`role`-based handling and is dropped. So the marker renders as nothing today.
Add one branch:

```
context compacted — earlier history above the agent no longer sees
```

`NoticeBlock`, kind `"note"` — same reasoning as `RESUME_UNREACHABLE_NOTICE`
(`app.py:1084-1091`): it answers "where did my history go", and `info` maps to
`dim` at 3.77:1 on the light theme, below the AA floor. It is not an error and
not a warning; nothing went wrong.

**The direction is ABOVE.** An earlier draft of this table said "older messages
below", and design review round 1 (D1) measured that against real mounted block
positions: a transcript paints oldest-at-top, so the pre-compaction rows sit
*above* the marker. Compared against `mode="context"` membership, the rows below
a marker were 100% still in the model's context — precisely what the sentence
claimed the agent could not see — while the rows above were 0-1%. Both halves of
the original sentence were wrong.

**Hold the string at 66 characters or fewer, verified in a rendered 80-column
frame.** The original 82-character string wrapped and orphaned its last two
words at every compaction (D2). Arithmetic alone is not enough to settle this:
round 1 derived a "~76 character" budget and the 71-character string it produced
*still wrapped*. The budget is not the terminal width — an 80-column terminal
gives a 78-column screen, the notice block's padding takes more, and the `· `
glyph costs 2, leaving 68 columns of text and so 66 for the string. Check it
with `scripts/audit_history_shot.py <dir> marker 80x30`, which is the only
instrument that answers the question.

Head notice copy, by state — this is the full state table for
`_reconcile_head_notice`:

| state | copy |
|---|---|
| context rows remain, scrollable | `RESUME_OLDER_NOTICE` (unchanged) |
| context rows remain, not scrollable | `RESUME_UNREACHABLE_NOTICE` (unchanged) |
| context drained, `audit_available` | **new:** `earlier history above — scroll up to load` |
| audit rows remain, scrollable | same new string |
| audit rows remain, not scrollable | **new:** `earlier history above — select to load` |
| audit exhausted (`end_index` at first row) | `RESUME_START_NOTICE` — now *true* |
| no compaction ever, head drained | `RESUME_START_NOTICE` (unchanged, correct) |

The head notice stays interactive in every audit state and drops interactivity
only at true exhaustion — `_restate_head_notice` (`app.py:7337-7341`) keys that
off `text != RESUME_START_NOTICE`, which already produces the right answer for
the new strings with no change. Keep that predicate as-is.

Deliberate wording choice: **"earlier history", not "older messages"**. The two
phases are different in kind, and a reader who crosses the compaction marker
should see the vocabulary change with it.

---

## 5. Changes per file

**`local_operator/session/transcript.py`**
- `replay_entries`: add `mode` kwarg; audit path skips the compaction cut,
  skips `preserved_user_turns`, emits an in-place marker per compaction row.
- New `audit_slice(entries, attachments, *, end_index, limit, prunes)`.
- New `first_message_index(entries)` helper (termination test).
- `build_llm_history`: **unchanged signature and behaviour.**

**`local_operator/session/history_window.py`**
- `DisplayHistoryWindow`: `+audit: bool`, `+audit_available: bool`.
- `display_window`/`_capture_display_window`: accept and thread `audit`; add the
  audit phase to token minting and the `before_token`/`has_more` chain; add the
  phase to the cache key tuple (`:187-198`).
- Grouping/boundary logic (`:285-297`) applies unchanged to audit pages —
  tool call/result groups must not split there either.

**`local_operator/session/session.py`**
- `history_page` (`:5797-5809`): pass the token's phase through. The token
  already carries it; this is a thread-through, not a new parameter.

**`local_operator/session/remote.py`**
- `history_before_token` (`:2603-2606`): return the audit token once context is
  drained. Today it is gated on `not self._history_hydrated`; that flag means
  "the context replay is fully loaded" and must **not** be repurposed. Add a
  separate `_audit_exhausted` flag, and gate on `not (self._history_hydrated and
  self._audit_exhausted)`.
- `load_older_display_page` (`:2608-2634`): the contiguity assertion
  `page.start + len(page.messages) != window.start` (`:2621`) is a
  context-coordinate check and **will fire on the first audit page**. Skip it
  when `page.audit`; audit pages are contiguous in journal order, not in replay
  coordinates. Set `_history_hydrated`/`_audit_exhausted` from the phase.
- `materialize_history` (`:2659-2691`): **must not** walk into the audit phase.
  It exists to produce the *model's* history for naming
  (`app.py:18490`) and `history()`. Stop its loop when the context phase ends.
  This is load-bearing: letting it drain audit pages would hand `generate_retitle`
  17,000 rows.
- `full_required` fallback (`:2672-2683`, requirement D): its inline `replay()`
  builds a whole `Transcript` and calls `build_llm_history`. For an audit page
  that must instead call `audit_slice` over the same bounded window — a
  `full_required` audit page means *one group* exceeded the byte budget, not
  that the reader asked for 231 MB. Concretely: retry the same `end_index` with
  `max_messages=1` so the oversized group is delivered alone. If a single group
  still exceeds the frame limit, surface it as a page carrying a
  `NoticeBlock`-shaped custom row rather than escalating to a full replay.

**`local_operator/tui/app.py`**
- Three new copy constants beside `RESUME_START_NOTICE` (`:1077`).
- `_reconcile_head_notice` (`:7220-7295`): implement the §4 table. Its
  `more` computation (`:7267`) needs the audit token, which
  `history_before_token` now supplies — so the shape of that expression is
  unchanged.
- `_mount_older_resume_page` (`:7666`): no change. Its docstring's "No disk
  read" claim stays true — audit pages arrive through the same RPC into
  `_resume_pending_head` (`:7493`).

**`local_operator/tui/session_presentation.py`**
- `project_settled_rows`: add the `compaction_summary` branch (§4), beside the
  existing custom-type branches at `:613-668`.

**`local_operator/mobile/durable.py`** — see §6. No change in this PR.

---

## 6. Capability and back-compat

**Add a new capability string: `"display-history-audit-v1"`.**

`DISPLAY_HISTORY_CAPABILITY = "display-history-window-v1"`
(`history_window.py:26`) is advertised at `runtime/server.py:659` on the mere
presence of `history_page`, and negotiated at `mobile/attach_client.py:245`.
It is a *presence* flag with no version handshake, so it cannot express "this
owner also pages audit history". Two ways it breaks without a new string:

- **New viewer, old owner.** The viewer receives `audit_available` absent →
  Pydantic default `False` → it never asks for an audit page. Safe by
  construction. But `extra="forbid"` on the model means the reverse direction
  is not safe:
- **Old viewer, new owner.** `DisplayHistoryWindow` sets
  `model_config = ConfigDict(extra="forbid")` (`history_window.py:39`), so an
  old viewer validating a payload carrying `audit`/`audit_available` **raises**,
  and `_fetch_history_page` (`remote.py:2490`) turns that into a failed attach.
  This is a hard cross-version break, and it is the single most likely way this
  change causes an incident during a staged rollout.

So: the owner advertises `display-history-audit-v1` and **only emits the two new
fields when the attaching viewer negotiated it**, exactly as `display_window` is
negotiated today (`runtime/server.py:1253`, `attach_client.py:245`). Mixed
versions on one machine are the norm here — the global `lop` runtime is a
separate uv tool install updated only by `lop-update`, so a repo `.venv` session
and a global session routinely run different builds against the same sessions
directory.

**Existing transcripts need no migration.** Audit mode reads rows that are
already on disk; `compact_file` (`transcript.py:1093-1198`) only folds prune
journals and superseded `subagent_roster` customs, so a folded transcript's
message rows are still all present — a folded row simply replays with its notice
text, which is the honest record of what the model saw. No backfill, no version
stamp, no format change. **The journal format does not change at all.**

**The mobile surface has the same defect and is out of scope.**
`mobile/durable.py:122-125` and `_replay` (`:325-346`) reimplement
`build_llm_history` semantics for the phone's fold cache, so
`daemon._history_page` pages only post-compaction rows too. Fixing it means
teaching the fold cache an audit mode, and the fold cache's incremental cursor
design (offset-into-inode) is a separate problem. Record it in the PR as a
known follow-up; do not widen this PR into it.

---

## 7. Test surface

**Unit — `tests/unit/session/test_transcript.py`**
- `mode="context"` output is byte-identical to today's `build_llm_history` on a
  compacted fixture. This is the regression guard for the whole change.
- `mode="audit"` returns every message row; count equals the on-disk message
  row count.
- Preserved user turns appear **exactly once** in audit mode (the duplicate-id
  hazard, §2.1).
- A prune targeting a row inside an `audit_slice` whose prune entry sits after
  `end_index` still blanks that row.
- Multi-compaction fixture: one marker per compaction, in journal position.

**Unit — `tests/unit/session/test_history_window.py`**
- Chain terminates at the first message row, not at the compaction cut.
- Page count over a 3-compaction fixture equals `ceil(rows / page)`, and every
  row appears exactly once across the chain (the audit-completeness assertion).
- `total_message_count` and `theme_turn_count` are unchanged by audit paging.
- An audit token minted at generation N returns `reset` after a compaction
  bumps `_history_generation` (`transcript.py:841`).
- A tool call and its result never split across an audit page boundary.

**Unit — `tests/unit/session/test_display_page_cache.py`**
- Audit pages are cached and evicted under the existing byte budget; the phase
  is part of the cache key (a context and an audit page at the same nominal
  position must not collide).

**Unit — `tests/unit/tui/test_rendered_history_paging.py`**
- The §4 state table, one assertion per row. Existing coverage already asserts
  `!= "start of conversation"` (`:258`), so extend rather than duplicate.
- A compacted session reaches `RESUME_START_NOTICE` **only** after the audit
  chain drains.
- The compaction marker renders as a `NoticeBlock` mid-transcript.

**Cross-version — `tests/unit/session/runtime/`**
- A viewer that did **not** negotiate `display-history-audit-v1` receives a
  payload with no `audit` keys and validates clean under `extra="forbid"`.
  This is the §6 break, and it needs an explicit test.

**QA (independent, per the team's gate)** — on a **copy** of the operator's real
sessions in an isolated config dir, never the live store:
- The §1.3 comparison, run across every session: for each, assert the audit
  chain's total row count equals the on-disk message row count. This is what
  finally settles whether a second defect exists.
- Page latency on `bda7b76d34e0` (231 MB): assert per-page wall time stays in
  the single-digit-ms range and that no page allocates on the order of the
  47 MB full replay.
- Resume timing on that session is unchanged versus base — the attach path is
  not supposed to move at all.
- Visual: rendered before/after frames of the head notice in each state and of
  the compaction marker row, per AGENTS.md "Visual validation".

**Quality gates** before the PR: flake8, `black --check`, `isort --check-only`,
pyright, and the unit suite via `.venv/bin/python` over the whole tree; TUI
tests under `env -u NO_COLOR TERM=xterm-256color`.

---

## 8. Risks to watch during rollout

1. **Cross-version payload rejection (§6)** — the `extra="forbid"` break. Highest
   severity, and entirely preventable by gating the fields on the negotiated
   capability. Verify with a real old-binary viewer against a new owner, not
   only with a unit test.
2. **`materialize_history` walking into audit** — would feed 17,000 rows to the
   retitle sampler and to `history()`. Guard it explicitly (§5) and assert it.
3. **Parked-presentation cache eviction (§3)** — deep audit scrolling pushes
   `retainable()` over budget and re-prepares the presentation on each sidebar
   poll. Expected, but it presents as UI lag. Watch for it; do not raise the
   budget.
4. **`total_message_count` drift** — any future contributor "fixing" it to
   include audit rows breaks `_sidebar_presentation_current`'s monotonic growth
   check (`app.py:3922-3924`). Put the reason in a comment at the field, not
   only in this document.
5. **Contiguity assertion** (`remote.py:2621`) — if the audit exemption is
   missed, the first audit page raises `ConnectionError` and the reader sees a
   failed page rather than history. Cheap to get wrong, loud when it happens.

---

## 9. Explicitly NOT changing

- **LLM context.** `build_llm_history` keeps its signature and its semantics;
  compaction keeps bounding what the model sees. `mode="context"` is the
  default and every existing caller keeps it.
- **The transcript file format.** No new row types, no migration, no backfill.
- **The attach/resume path.** `read_replay_suffix`, `_read_transcript`,
  `_load_frontend_history` and `RESUME_RENDER_MESSAGES` are untouched; resume
  cost does not move.
- **The head-notice mechanism.** `_restate_head_notice`, `OlderHistoryNotice`
  and the interactivity rule stay as-is; only the copy table grows.
- **`compact_file`.** Its folding rules are correct and orthogonal.
- **The mobile fold cache** (§6) — same defect, deliberate follow-up.
- **`RESUME_START_NOTICE` itself.** It stays exactly as worded. After this
  change it is finally always true, which is the point.

---

## 10. Choices I made where two options were reasonable

**Second replay mode vs. a second index.** A parallel display index (a
`display.jsonl` or a sidecar offset map) would make audit paging O(1) instead of
O(window). Rejected: it is a second source of truth for the same bytes, it needs
a migration and a backfill for every existing transcript, and it must be kept
consistent with `compact_file`'s atomic replacement. The measured cost of the
windowed replay is ~1 ms against a resident entry list, so the index buys
nothing that matters. Fix the existing function; do not add a second one beside
it.

**Extending the existing paging chain vs. a separate audit RPC.** A separate
`audit_page` op would keep the two concerns visibly apart. Rejected: the TUI
would need a second cursor, a second pending buffer, and a second dedupe path,
and the reader's gesture is *identical* — scroll up. One chain with a phase flag
means `load_older_display_page`, `_resume_pending_head`, the lease machinery and
the cache all work unmodified. The seam belongs in the replay layer, which is
where the two semantics actually diverge, not in the transport.

**Emitting one marker per compaction vs. only the boundary the reader crosses.**
Chose per-compaction. `bda7b76d34e0` has 48 of them; a reader auditing that
session needs to see where each one fell, and a single marker at the context
boundary would misrepresent 47 others as ordinary history.

*Per-compaction means per surviving boundary, not per compaction ROW.* Repeated
compaction against a short kept suffix leaves earlier compaction rows sitting at
or above the current cut, and those describe boundaries the latest compaction has
already moved — the seam they name no longer exists in the rendered transcript.
Every compaction row below the cut is emitted in place by the audit phase, and
the latest is emitted at the context head as the live seam; on `2f95e374dd22`
that is 56 markers for 56 surviving boundaries out of 61 compaction rows.

Review round 1 (R2) read the remaining 5 as unreachable markers and suggested
emitting them. Rejected on measurement, and recorded here because it is the
natural thing to try:

- They would be **false where they landed.** Each sits 3-15 live context rows
  below the seam, so its notice would claim the rows above it are outside the
  model's context while the model can still see them — the D1 inversion, back
  again.
- Emitting them from the **context phase** breaks this change's central
  regression guard (`mode="context"` byte-identical to `build_llm_history`) and
  pushes each marker's `preserve_data` into the model's own context: measured
  +3,380,677 bytes, roughly 845k tokens, on that one journal.
- Extending the audit window's first `end_index` **past the cut** re-delivers 16
  message rows the context page already sent.

Both routes are pinned by tests in `tests/unit/session/test_history_window.py`
so the argument does not have to be rediscovered from this document.

**Keeping `total_message_count` context-scoped.** The alternative — making it
the true total — is arguably the more honest field name. Rejected on evidence:
two consumers read it as a monotonic context-growth signal (`app.py:3922`,
`remote.py:4287`), and redefining it silently breaks presentation caching. A
separate `audit_message_count` is available if a surface ever needs it; I did
not add it speculatively.

**One defect, not two (§1.3).** I chose to report the second defect as *not
demonstrated* rather than hedge. The session named in the report has no
compaction and pages correctly to a true zero. I have specified the exact
measurement that would prove me wrong and put it in the QA matrix rather than
leaving it as a suspicion for the coder to chase.
