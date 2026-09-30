# Design: the generic wake-trigger layer, and its first trigger (project staleness)

Status: implemented (v1). The source of truth for behaviour is the code —
`local_operator/wakes/triggers/` (registry, records, bounds, snapshot) and its
first source, `wakes/triggers/sources/project_staleness.py`; the consuming
engine side lives in `local_operator/aida/proactive.py` (`consume_triggers`).
This document is the condensed record of the decisions; docstrings carry the
detail.

## 1. What a wake trigger is

A named **source** that periodically evaluates a condition over local state
and, when the condition holds, causes a **wake** in a **target session**
(default: Aida) whose message is labelled with the trigger. The wake is an
ordinary wake row (`WakeSchedule`) in the target's schedule list — triggers
add no new firing path and no new transcript writer.

## 2. Where it runs

The evaluation pass rides the existing wake supervisor loop
(`wakes/supervisor.py`), throttled at `TRIGGER_EVAL_INTERVAL_S` (300 s), never
a second daemon: the supervisor is the one always-on process whose job is
"make a runtime exist for this session", which is what a trigger needs. The
*possibility* of a future trigger deliberately does not keep the supervisor
resident (that would convert it into a cron); a **pending record** does count
as fireable work while it is owed, exactly like a spooled turn.

## 3. The registry and the source protocol

```
wakes/triggers/
  __init__.py                    registry + record/state IO + bounds + snapshot
  sources/__init__.py            the ONE place to add a source
  sources/project_staleness.py   v1 source
```

```python
@dataclass(frozen=True)
class TriggerInstance:
    source: str; key: str
    fingerprint: tuple  # identity of the CONDITION STATE
    payload: Mapping; age_s: float

class TriggerSource(Protocol):
    name: str
    def enabled(self, values) -> bool: ...
    def evaluate(self, ctx) -> Sequence[TriggerInstance]: ...

def register(source): ...
def evaluate_all(config_dir, now_ms, values): ...
def commit(instances, config_dir, now_ms, values): ...  # the ONLY record writer
```

Everything is stdlib-only at module scope, pinned by
`tests/unit/test_import_graph.py` alongside `store`/`spooled`/`deliveries`:
the supervisor must not pay for the harness. Heavier reads (the runtime
registry for liveness, the projects store for the re-verify) are function-local
imports inside the functions that need them.

## 4. Records, dedupe, bounds, state

```
<config>/wakes/triggers/
  settings.json               published snapshot of the trigger settings
  state.json                  dedupe fingerprints + rolling fire timestamps
  pending/<session_id>.json   ONE record per target — the owed check-ins
```

- **Dedupe**: one wake per condition instance, identity
  `(source, key, fingerprint)`. The project source's fingerprint is the same
  `(id, status, int(progress_updated_at))` shape the completion-time check
  latches on, so "what counts as a new episode" cannot mean two things. No
  re-nudge for an unchanged stale record (the daily cadence backstops).
- **Bounds**: `wakes.triggers.max_per_day` (default 6, `0` disables) as a
  per-target rolling-24 h window, and `wakes.triggers.min_gap_minutes`
  (default 60) measured from the last fire. Blocked instances stay candidates
  and are not marked notified until a record is actually written. Internal
  constants: `RECORD_TTL_S = 72 h`, retry walk 15 s → ×2 → 1 h cap, per-record
  instance cap 20 (older-first; extras render as "and N more").
- **Snapshots**: the supervisor cannot read `config.yml` (no YAML), so
  config-aware writers publish `settings.json`
  (`triggers.publish_settings` — settings write/reset, Aida's boot, ensure and
  reconcile) and the pass reads it (`triggers.read_settings`), falling back to
  the module defaults. A hand-edited `config.yml` converges at the next
  publish point (≤ her next reconcile).

## 5. Suppression matrix (fail-closed; checked every pass, at creation AND delivery)

1. `LOCAL_OPERATOR_NO_AIDA` truthy ⇒ skip.
2. `<config>/aida/state.json` absent/unreadable/nameless ⇒ skip (reads only;
   zero footprint).
3. `wakes.triggers.enabled` false, or the source's own `enabled()` false ⇒ skip.
4. Paused (`/aida pause` sets `held_at` on her wake-index entry;
   `store.is_held`) ⇒ no record and no engagement.
5. Reactive class ⇒ skip (a minimal, fail-closed mirror of
   `action_class.session_action_class`; any doubt reads reactive). The engine
   re-checks authoritatively at consume.
6. Disable-with-leftovers costs at most one spurious engage; her load then
   drops the armed rows.

## 6. Supervisor integration

- `_load_and_reconcile_state` returns a fourth mapping (pending records) and
  reconciles it: TTL-expired and ghost (target session gone) records are
  dropped.
- `_due_sessions` appends `(target, cwd, due_ms)` for each pending record
  whose target is not scheduled/dormant, honouring `next_attempt_ms`.
- `_has_fireable_wakes` counts pending records as work (skipping dormant
  targets), so a lone owed check-in keeps the process resident.
- Failed/wedged engagements feed `triggers.note_attempt` (the spooled
  vocabulary: `wedged`/`failed`/`raised` move the walk; `started` does not).
- `serve()` kicks the pass detached at the top of each iteration; `--once`
  awaits it inline so its own state load sees the record and engages it.

## 7. The project-staleness source

The rule mirrors the store's `progress_is_stale` — status in
`planning`/`active`/`qa`/`validation` and (no progress text, or no
`progress_updated_at`, or older than the window) — and reads the SAME threshold
every rendered reader does: `projects.stale_after_hours` (int hours, default
4, bounds 1–168), resolved by `projects.stale_after_s()` and, on the trigger
side, through the published snapshot. Payload per instance:
`{display_name, status, progress_age_s, sessions: [{id, last_activity_age_s,
live}]}` where `live` is `live|wedged` from the runtime registry when it
answers, else `cold`/`missing` — a floor, not a verdict; "stalled" is the
engine's call.

## 8. The behaviour contract (Aida)

On a project-staleness wake she: messages the linked sessions (or the
project's manager) for a status update and a `project` progress refresh —
**never doing the update work herself**; makes ONE bounded resume attempt
(`lop exec --resume`) for a dead/stalled session, never force-stopping a
wedged runtime; surfaces a sessionless project to the operator instead of
spawning unbounded work; skips settled projects; and reports briefly. The
consume path arms ONE `aida-trigger-<8hex>` row (id over the sorted
fingerprints, idempotent), journals an `aida_trigger` receipt, and settles the
record only after the persist — the crash windows hang off that ordering.

## 9. Known residuals (accepted, watch-listed)

- **Live-target latency**: a record written while the target's runtime is
  live-idle waits for her next in-session seam (after-turn runs only when
  something wants the engine; the serving-start drain covers boot). Follow-ups
  if it bites: hand the record to the live runtime as a peer errand, or a
  cheap stat in the runtime's beat loop.
- **≤ one spurious engage** after a config-disable with leftover armed rows.
- **Triple-crash duplicate**: a millisecond-scale window can re-arm one row
  once; bounded by the budget.
- **Snapshot lag** on a hand-edited `config.yml` until the next publish point.
- **Cap-at-consume**: at the 16-schedule ceiling the check-in is skipped with
  a note and the record stays pending.
