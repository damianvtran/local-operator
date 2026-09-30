"""Generic wake triggers: conditions that cause a wake when nothing else would.

A **wake trigger** is a named SOURCE that periodically evaluates a condition
over local state and, when the condition holds, causes a wake in a TARGET
session whose message is labelled with the trigger. The first (and today only)
source is :mod:`~local_operator.wakes.triggers.sources.project_staleness`; the
target is Aida, resolved from her own ``state.json``.

**One mechanism, no new firing path.** A trigger does not deliver anything
itself. The evaluation pass records a *pending check-in* under
``<config>/wakes/triggers/pending/<target>.json``; the wake supervisor treats
a pending record exactly as it treats a spooled turn (``wakes/spooled.py``) —
as a reason a runtime must exist — and the TARGET's own engine consumes the
record and arms an ordinary wake row through the one schedule writer it
already has. Triggers add no new firing path and no new transcript writer.

**Where it runs.** The pass rides the existing wake supervisor loop, throttled
(``supervisor.TRIGGER_EVAL_INTERVAL_S``), never a second daemon: that process
is the one always-on process whose job is "make a runtime exist for this
session", which is exactly what a trigger needs. The *possibility* of a future
trigger deliberately does NOT keep the supervisor resident — on an aida-enabled
install her daily cadence row is fireable work, and a pending record counts as
fireable work while it is owed. A disabled install has no state file and this
module then reads nothing and writes nothing (see the suppression matrix).

**The suppression matrix** (checked in this order every pass; any failure
skips the target, and every arm is fail-closed):

1. ``LOCAL_OPERATOR_NO_AIDA`` truthy in the process environment ⇒ skip.
2. ``<config>/aida/state.json`` absent/unreadable/nameless ⇒ skip — she was
   never enabled, so an evaluation pass performs READS ONLY and creates
   nothing (the zero-footprint property, pinned by tests).
3. ``wakes.triggers.enabled`` false (or the source's own ``enabled()``) from
   the published settings snapshot ⇒ skip.
4. Paused: her wake-index entry exists and is held (``store.is_held`` — the
   ``held_at`` the ``/aida pause`` path stamps) ⇒ no record and no engagement.
5. Reactive class: fail-closed — see :func:`_class_reactive`.
6. Disable-with-leftovers: her index entry may still hold armed rows until her
   next load drops them; the first trigger attempt engages her, the load drops
   them, and gates (2)/(4) hold from then on (≤ one spurious engage).

**Dedupe and bounds live in one place.** One wake per *condition instance*,
identified by ``(source, key, fingerprint)`` — the fingerprint is the same
``(id, status, int(progress_updated_at))`` shape the project completion check
already latches on, so "what counts as a new episode" cannot mean two things.
A per-target rolling-24 h budget and a minimum gap between check-ins bound the
wakes (``state.json``); instances blocked by a bound stay candidates and fire
when it clears — they are never marked notified until a record is written.

**Why stdlib-only at module scope**, pinned by
``tests/unit/test_import_graph.py``: this module sits on the supervisor's
resident set (a ~40 MB process whose whole justification is that it does not
carry the harness). Everything heavier than the stdlib is imported inside the
functions that need it — and the supervisor-side functions never do.

**The settings you cannot import come from a published snapshot.** The
supervisor cannot read ``config.yml`` (no YAML), but the pass must evaluate
against THE configured values, so config-aware writers publish
``<config>/wakes/triggers/settings.json`` (:func:`publish_settings` — the
settings write/reset paths, Aida's boot and reconcile) and the pass reads the
snapshot (:func:`read_settings`), falling back to the module defaults — which
are identical to the registry's, so a missing snapshot cannot change behaviour,
only miss a recent edit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

logger = logging.getLogger(__name__)

#: Subdirectory of the wakes dir holding the trigger layer's own files. A
#: SUBdirectory for the reason ``deliveries``/``spooled`` give: the wake-index
#: scan lists ``wakes/`` and skips anything that is not a ``.json`` file, so the
#: layer can never be read as a schedule.
TRIGGERS_DIRNAME = "triggers"

#: One pending record per target session.
PENDING_DIRNAME = "pending"

#: The supervisor-owned dedupe + budget ledger (see :func:`commit`).
STATE_NAME = "state.json"

#: The published snapshot of the trigger settings (see the module docstring).
SETTINGS_NAME = "settings.json"

#: Bumped only on an incompatible change to a record's shape.
RECORD_SCHEMA = 1
#: ... to ``state.json``'s shape.
STATE_SCHEMA = 1
#: ... to ``settings.json``'s shape.
SETTINGS_SCHEMA = 1

#: The keys the snapshot carries, in one place. The settings registry owns the
#: rows; these are the names the supervisor-side reader resolves.
SNAPSHOT_KEYS: tuple[str, ...] = (
    "projects.stale_after_hours",
    "wakes.triggers.enabled",
    "wakes.triggers.max_per_day",
    "wakes.triggers.min_gap_minutes",
    "wakes.triggers.project_staleness.enabled",
)

#: The master switch's default. Registered in ``settings_io`` as a literal (the
#: CLI's settings layer must not import this package), and pinned against this
#: constant by ``tests/unit/test_settings_io.py`` so the two cannot drift.
DEFAULT_ENABLED = True

#: Per-target rolling-24 h budget on check-in wakes. ``0`` disables.
DEFAULT_MAX_PER_DAY = 6

#: Minimum spacing between two trigger wakes to the same target.
DEFAULT_MIN_GAP_MINUTES = 60

#: Every default the snapshot can fall back to — the same numbers the registry
#: ships, so an absent snapshot cannot change behaviour. Kept as literals with
#: the consumer-side constants above/in the source module, pinned by tests.
DEFAULTS: Mapping[str, Any] = {
    "projects.stale_after_hours": 4,
    "wakes.triggers.enabled": DEFAULT_ENABLED,
    "wakes.triggers.max_per_day": DEFAULT_MAX_PER_DAY,
    "wakes.triggers.min_gap_minutes": DEFAULT_MIN_GAP_MINUTES,
    "wakes.triggers.project_staleness.enabled": True,
}

#: A pending record older than this is dropped by :func:`reconcile`: a check-in
#: about NOW, delivered three days late, is noise — the target's daily cadence
#: has covered the interval by then.
RECORD_TTL_S = 72 * 3600.0

#: Retry walk for a target the supervisor could not raise, mirroring
#: ``spooled``: quick first retry, then back off so one unraisable target cannot
#: occupy an engagement slot on every pass.
RETRY_BASE_S = 15.0
RETRY_FACTOR = 2.0
RETRY_CAP_S = 3600.0

#: Per-record instance cap; extras ride the record's ``overflow`` count and are
#: summarised as "and N more" by the message composer.
INSTANCE_CAP = 20

#: Which engagement outcomes COUNT against a pending record's retry walk. Same
#: vocabulary and same reasoning as ``supervisor._SPOOLED_ATTEMPT_REASONS``:
#: ``live`` is absent because nothing was tried (the target is served and will
#: consume the record in-session), ``ghost`` because the reconciler drops such a
#: record, and ``started`` because the runtime that booted is the party that
#: consumes the record — the supervisor must not decide it was delivered.
ATTEMPT_REASONS = frozenset({"wedged", "failed", "raised"})

#: The environment kill switch. Spelled locally (with the same truthiness rule)
#: rather than importing ``local_operator.aida.state``: this module must stay
#: stdlib-only, and the rule is one line — ``0``/``false``/``no``/``off`` mean
#: "on".
_ENV_DISABLE = "LOCAL_OPERATOR_NO_AIDA"
_FALSY = {"0", "false", "no", "off", ""}

#: The transcript file name, spelled locally for the same reason ``spooled``
#: spells its spool fields locally: the ghost rule ("a record whose target
#: session is gone") must not import ``local_operator.resume`` onto the
#: supervisor's resident set.
_TRANSCRIPT_NAME = "transcript.jsonl"


@dataclass(frozen=True)
class TriggerInstance:
    """One live condition, as a source reports it.

    ``fingerprint`` identifies the CONDITION STATE (not the row): when it moves,
    a new episode begins and the dedupe will let a fresh record through. It must
    be JSON-safe. ``age_s`` is how long the condition has held, used only to
    order candidates (older first) and to compose the message.
    """

    source: str
    key: str
    fingerprint: tuple[Any, ...]
    payload: Mapping[str, Any]
    age_s: float


@dataclass(frozen=True)
class TriggerContext:
    """What a light source may read: the config root, a clock, the settings.

    Deliberately small. A source is expected to read its own local state
    directly (see ``sources/project_staleness.py``) rather than through the
    application layers, because it runs inside the supervisor's process.
    """

    config_dir: Path
    now_ms: int
    values: Mapping[str, Any]


class TriggerSource(Protocol):
    """The source protocol: a name, an enable switch, one evaluation."""

    name: str

    def enabled(self, values: Mapping[str, Any]) -> bool: ...

    def evaluate(self, ctx: TriggerContext) -> Sequence[TriggerInstance]: ...


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_SOURCES: dict[str, TriggerSource] = {}
_BUILTINS_LOADED = False


def register(source: TriggerSource) -> None:
    """Add ``source`` under its own name; a name identifies ONE source.

    FIRST registration wins, and a duplicate is ignored rather than replacing
    or raising: every legitimate caller registers a name exactly once (one
    module per source), and the builtin loader runs LAZILY — at the first
    evaluation — so first-wins is what lets a registration that happens before
    the load (a test standing one up, or an embedder overriding a name
    deliberately) keep the name it claimed, while a second builtin load can
    never silently replace a live source.
    """
    name = str(getattr(source, "name", "") or "").strip()
    if not name:
        raise ValueError("a trigger source must carry a non-empty name")
    _SOURCES.setdefault(name, source)


def registered_sources() -> list[TriggerSource]:
    """Every registered source, in deterministic (name-sorted) order."""
    return [_SOURCES[name] for name in sorted(_SOURCES)]


def _reset_registry() -> None:
    """Drop every registration AND every load latch. For tests.

    Clearing the flags too is what lets a test that reset the registry get the
    built-in sources back through :func:`evaluate_all`'s lazy loader — without
    it, the first load anywhere in the process latches and every later test
    would silently evaluate against an empty registry.
    """
    global _BUILTINS_LOADED
    _SOURCES.clear()
    _BUILTINS_LOADED = False
    try:
        from local_operator.wakes.triggers import sources as trigger_sources

        trigger_sources.reset_loaded()
    except Exception:  # noqa: BLE001 — best-effort test support
        logger.warning("trigger registry reset could not reach the sources loader", exc_info=True)


def _load_builtin_sources() -> None:
    """Import the packaged sources once per process.

    The import lives here rather than at module scope so the package imports
    without its sources and the sources can import ``register`` back without a
    cycle. A failure is logged once — the flag is set FIRST, because a broken
    packaging is not something a retry every 300 s fixes.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    try:
        from local_operator.wakes.triggers import sources

        sources.load_builtin()
    except Exception:  # noqa: BLE001 — a broken source must not kill the sweep
        logger.warning("could not load the built-in trigger sources", exc_info=True)


def evaluate_all(config_dir: Path, now_ms: int, values: Mapping[str, Any]) -> list[TriggerInstance]:
    """Every enabled source's instances, in name-sorted source order.

    One source raising costs that source's instances, never the pass.
    """
    _load_builtin_sources()
    ctx = TriggerContext(config_dir=Path(config_dir), now_ms=int(now_ms), values=values)
    out: list[TriggerInstance] = []
    for source in registered_sources():
        try:
            if not source.enabled(values):
                continue
        except Exception:  # noqa: BLE001 — doubt about enabled must not run it
            logger.warning(
                "trigger source %s could not resolve 'enabled'", source.name, exc_info=True
            )
            continue
        try:
            out.extend(source.evaluate(ctx))
        except Exception:  # noqa: BLE001 — one source never fails the pass
            logger.warning("trigger source %s failed to evaluate", source.name, exc_info=True)
    return out


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def triggers_dir(config_dir: Path | str) -> Path:
    return Path(config_dir) / "wakes" / TRIGGERS_DIRNAME


def pending_dir(config_dir: Path | str) -> Path:
    return triggers_dir(config_dir) / PENDING_DIRNAME


def pending_path(config_dir: Path | str, session_id: str) -> Path:
    """Where ``session_id``'s pending check-ins live. The id is never
    sanitised: it is a session directory name read off disk, exactly as
    ``store``/``spooled`` treat theirs."""
    return pending_dir(config_dir) / f"{session_id}.json"


def state_path(config_dir: Path | str) -> Path:
    return triggers_dir(config_dir) / STATE_NAME


def settings_path(config_dir: Path | str) -> Path:
    return triggers_dir(config_dir) / SETTINGS_NAME


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Record IO
# ---------------------------------------------------------------------------


def read_pending(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Every readable pending record, keyed by target session id.

    Never raises. One unreadable file (a torn write, a hand edit) costs one
    target, never the sweep — the same contract ``read_spooled`` states.
    """
    out: dict[str, dict[str, Any]] = {}
    try:
        candidates = sorted(pending_dir(config_dir).glob("*.json"))
    except OSError:
        return out
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("schema_version") != RECORD_SCHEMA:
            continue
        target = data.get("target") or path.stem
        if not isinstance(target, str) or not target:
            continue
        out[target] = data
    return out


def read_pending_record(config_dir: Path | str, session_id: str) -> dict[str, Any] | None:
    """One target's pending record, or ``None``."""
    if not session_id:
        return None
    try:
        data = json.loads(pending_path(config_dir, session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema_version") != RECORD_SCHEMA:
        return None
    return data


def has_pending(config_dir: Path | str, session_id: str) -> bool:
    """Whether ``session_id`` has a pending record. One stat; never raises.

    The cheap probe the session's after-turn seam uses so an owed check-in is
    consumed at the end of a turn rather than waiting for the next unrelated
    persist.
    """
    try:
        return pending_path(config_dir, session_id).is_file()
    except OSError:
        return False


def _fingerprint_list(value: Any) -> list[Any] | None:
    """A stored fingerprint as a JSON list, or ``None`` when unusable."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _instance_key(entry: Mapping[str, Any]) -> str:
    return f"{entry.get('source')}::{entry.get('key')}"


def next_attempt_at_ms(record: Mapping[str, Any]) -> int:
    """When this record may be engaged again. Never raises, never "never".

    Defensive on purpose, exactly like ``spooled.next_attempt_at_ms``: the
    record is written by the supervisor and read on its hot path, so a
    hand-edited field must cost one target's retry, not the whole pass. An
    unreadable wait reads as DUE NOW — the direction that cannot lose work;
    the backoff bounds what that costs.
    """
    recorded = record.get("next_attempt_ms")
    if isinstance(recorded, int) and not isinstance(recorded, bool):
        return recorded
    return 0


def backoff_s(attempts: int) -> float:
    """Seconds to wait after ``attempts`` consecutive failures. Mirrors
    ``spooled.backoff_s`` and is public for the same reason: tests drive it."""
    exponent = max(0, int(attempts) - 1)
    return min(RETRY_CAP_S, RETRY_BASE_S * (RETRY_FACTOR**exponent))


def _attempts_of(record: Mapping[str, Any]) -> int:
    raw = record.get("attempts")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    return 0


def note_attempt(config_dir: Path, session_id: str, *, reason: str) -> None:
    """Record a failed attempt to raise this target, with its backoff.

    The supervisor's own bookkeeping, the same shape ``spooled.note_attempt``
    uses: the record is a claim that the check-in is still owed, so a failure
    must not erase it — it must make the next attempt later. A no-op when the
    reason does not move the walk or when no record exists (the overwhelmingly
    common case: reading first is cheaper than writing).
    """
    if reason not in ATTEMPT_REASONS:
        return
    record = read_pending_record(config_dir, session_id)
    if record is None:
        return
    moment = _now_ms()
    attempts = _attempts_of(record) + 1
    record["attempts"] = attempts
    record["last_attempt_ms"] = moment
    record["last_error"] = str(reason)[:400]
    record["next_attempt_ms"] = moment + int(backoff_s(attempts) * 1000)
    record["updated_at_ms"] = moment
    _write_json(pending_path(config_dir, session_id), record)


def settle(
    config_dir: Path,
    session_id: str,
    keys: Sequence[tuple[str, str]],
    *,
    expected_updated_at_ms: int | None = None,
) -> bool:
    """Settle the named instances of a pending record. Returns whether the
    record moved.

    Called by the target's engine AFTER its schedule persist succeeded (the
    receipt/settle ordering in the design is what makes crash windows
    duplicate-light). Compare-and-delete, the spooled pattern: the record is
    written by another process, so if its ``updated_at_ms`` moved since the
    caller read it — a merge of new instances — the WHOLE record is left
    alone; the next consume re-reads it and settles then (and if the armed row
    is already in the schedule list, the next consume finds it and settles
    without arming a duplicate for the instances it covers).

    Instances named in ``keys`` are removed; when none remain the record is
    unlinked; when others remain the record is rewritten with them. Unknown
    ``keys`` are ignored.
    """
    record = read_pending_record(config_dir, session_id)
    if record is None:
        return False
    if expected_updated_at_ms is not None and record.get("updated_at_ms") != expected_updated_at_ms:
        return False
    wanted = {(str(source), str(key)) for source, key in keys}
    instances = [entry for entry in record.get("instances") or [] if isinstance(entry, dict)]
    remaining = [
        entry
        for entry in instances
        if (str(entry.get("source")), str(entry.get("key"))) not in wanted
    ]
    if len(remaining) == len(instances):
        return False
    path = pending_path(config_dir, session_id)
    if not remaining:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError:
            logger.warning("could not clear the trigger record for %s", session_id, exc_info=True)
            return False
    record["instances"] = remaining
    record["updated_at_ms"] = _now_ms()
    return _write_json(path, record)


def _write_json(path: Path, payload: Mapping[str, Any]) -> bool:
    """Stage and replace one small JSON file. Never raises.

    Atomic (temp file plus ``os.replace``) like ``store``/``spooled``: the
    supervisor may read this directory at any moment and a torn file would be
    read as no obligation at all — the failure direction that loses work.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        with handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
        return True
    except OSError:
        logger.warning("could not write the trigger file %s", path, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# State (dedupe + budget)
# ---------------------------------------------------------------------------


def _read_state(config_dir: Path) -> dict[str, Any]:
    """The dedupe/budget ledger, or an empty skeleton. Never raises."""
    try:
        data = json.loads(state_path(config_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict) or data.get("schema_version") != STATE_SCHEMA:
        return {"schema_version": STATE_SCHEMA, "instances": {}, "targets": {}}
    instances = data.get("instances")
    targets = data.get("targets")
    return {
        "schema_version": STATE_SCHEMA,
        "instances": dict(instances) if isinstance(instances, Mapping) else {},
        "targets": dict(targets) if isinstance(targets, Mapping) else {},
    }


def _target_state(state: Mapping[str, Any], target: str) -> dict[str, Any]:
    raw = (state.get("targets") or {}).get(target)
    if not isinstance(raw, Mapping):
        return {"fires": [], "last_fire_ms": None}
    fires = [f for f in (raw.get("fires") or []) if isinstance(f, int) and not isinstance(f, bool)]
    last = raw.get("last_fire_ms")
    return {
        "fires": fires,
        "last_fire_ms": last if isinstance(last, int) and not isinstance(last, bool) else None,
    }


def _int_value(values: Mapping[str, Any], key: str, default: int) -> int:
    """An int setting, or its default. Never raises (the snapshot is JSON)."""
    raw = values.get(key)
    if isinstance(raw, bool) or raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _strict_bool(value: Any, default: bool) -> bool:
    """A REAL boolean or ``default`` — the same rule ``settings_io.strict_bool``
    states, mirrored because this module must not import the CLI's settings
    layer. The snapshot is written by that layer, so the mirror is about
    hand-edits and future drift, not about normal operation."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
    return default


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def _env_disabled() -> bool:
    return os.environ.get(_ENV_DISABLE, "").strip().lower() not in _FALSY


def _default_config_dir() -> Path | None:
    from local_operator.paths import config_dir

    try:
        return Path(config_dir())
    except Exception:  # noqa: BLE001 — no root, no pass
        logger.debug("could not resolve the config dir for the trigger sweep", exc_info=True)
        return None


def _aida_target(config_dir: Path) -> str | None:
    """The trigger target: the session ``aida/state.json`` names.

    Read directly (stdlib) — the supervisor-side reader must not import the
    aida package. Absent/unreadable/nameless all answer ``None``, which the
    pass treats as "she was never enabled: nothing to do", reads only.
    """
    try:
        data = json.loads((config_dir / "aida" / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != 1:
        return None
    session_id = data.get("session_id")
    return session_id if isinstance(session_id, str) and session_id else None


def _target_held(config_dir: Path, session_id: str) -> bool:
    """Whether her wake-index entry is parked (pause/stop). Fail-closed.

    ``store.is_held`` covers both levers (``stopped_at`` and Aida's
    ``held_at``); an UNREADABLE answer suppresses, because the matrix's job is
    to keep a paused assistant silent and the cost of the wrong reading in that
    direction is one deferred check-in.
    """
    try:
        from local_operator.wakes.store import is_held, read_entry

        return is_held(read_entry(config_dir, session_id))
    except Exception:  # noqa: BLE001 — fail closed, see docstring
        logger.debug("could not resolve the trigger hold for %s", session_id, exc_info=True)
        return True


def _class_reactive(config_dir: Path, session_id: str) -> bool:
    """Whether the target's attached profile is outside the proactive class.

    A minimal, fail-closed mirror of ``action_class.session_action_class``
    limited to "reactive?" — the supervisor-side pass cannot import the class
    module (stdlib-only at module scope, and the class module drags the agent
    registry in). Same places, read in the same order: the attachment sidecar
    names the profile; a ROLE row in ``agents/`` shadows the packaged seed
    (``resolve_profile``'s own precedence), so the row's tags are consulted
    first; else the packaged seed's frontmatter. ANY doubt — no attachment, an
    unreadable or ambiguous row, a seed with no class — reads reactive, which
    matches the class module's own posture and the engine's authoritative
    re-check at consume. A wrong "reactive" defers a check-in to the target's
    daily cadence; a wrong "proactive" would raise a runtime for a stopped
    assistant, which is the failure this gate exists to prevent.
    """
    try:
        name = _attached_profile_name(config_dir, session_id)
        if not name:
            return True
        found, token = _registry_class_token(config_dir, name)
        if found:
            return token != "proactive"
        seed = _seed_class_token(name)
        if seed is None:
            return True
        return seed != "proactive"
    except Exception:  # noqa: BLE001 — documented fail-closed
        logger.debug("could not resolve the trigger target's class", exc_info=True)
        return True


def _attached_profile_name(config_dir: Path, session_id: str) -> str:
    try:
        payload = json.loads(
            (config_dir / "sessions" / session_id / "attachment.json").read_text(
                encoding="utf-8", errors="replace"
            )
        )
    except (OSError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    value = payload.get("agent")
    return value.strip() if isinstance(value, str) else ""


def _tag_lines(text: str) -> list[str]:
    """The block-list items PyYAML emits for a ``tags:`` list, unquoted.

    ``agents.save_agent`` dumps with ``default_flow_style=False``, so a tag
    list is one ``- <tag>`` line per item; anything else in the file (other
    fields' prose included) is ignored unless it LOOKS like a list item, which
    is why this reads lines rather than the whole text.
    """
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        item = stripped[2:].strip()
        if len(item) >= 2 and item[0] == item[-1] and item[0] in "\"'":
            item = item[1:-1]
        items.append(item)
    return items


def _registry_class_token(config_dir: Path, name: str) -> tuple[bool, str | None]:
    """``(found, class)`` for a role row shadowing ``name``.

    ``found=False`` means no row resolves for the name (the seed is
    authoritative). ``found=True`` with ``class=None`` means a row DOES shadow
    the seed but its class could not be read unambiguously — the caller reads
    reactive. Multiple matching rows, or a matching row that cannot be shown to
    be a role, are both doubt.
    """
    wanted = name.casefold()
    matches: list[tuple[bool, str | None]] = []
    try:
        rows = sorted((Path(config_dir) / "agents").glob("*/agent.yml"))
    except OSError:
        return (False, None)
    for path in rows:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # AN UNREADABLE CANDIDATE IS DOUBT, and doubt reads reactive. The
            # file might be the row that shadows the seed (reactive), so the
            # design's fail-closed direction applies even though the app's own
            # loader would have dropped the row and used the seed; the cost is
            # suppressed check-ins while the file stays unreadable, backstopped
            # by the daily cadence — the same trade the class module's own
            # posture makes.
            return (True, None)
        row_name = ""
        for line in text.splitlines():
            if not line.lstrip().startswith("name:"):
                continue
            raw = line.split(":", 1)[1].strip()
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
                raw = raw[1:-1]
            row_name = raw
            break
        if not row_name or row_name.casefold() != wanted:
            continue
        tags = _tag_lines(text)
        is_role_row = any(tag == "role" for tag in tags)
        if not is_role_row:
            continue  # resolve_profile ignores non-role rows; the seed stands
        tokens = {
            tag.split(":", 1)[1].strip().casefold() for tag in tags if tag.startswith("class:")
        }
        if len(tokens) == 1:
            matches.append((True, tokens.pop()))
        else:
            matches.append((True, None))  # absent or conflicting: doubt
    if not matches:
        return (False, None)
    if len(matches) > 1:
        return (True, None)  # ambiguous shadowing: doubt
    return matches[0]


def _seed_class_token(name: str) -> str | None:
    """The packaged seed's ``class:`` frontmatter value, or ``None``.

    ``None`` means "could not read it" — which the caller reads as reactive.
    The seeds directory sits beside this package (``local_operator/agent_seeds``
    relative to ``local_operator/wakes/triggers``), exactly where
    ``agent_profiles.SEEDS_DIR`` looks.
    """
    key = str(name).strip().casefold()
    if not key:
        return None
    seed = Path(__file__).resolve().parents[2] / "agent_seeds" / f"{key}.md"
    try:
        text = seed.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    in_frontmatter = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
            continue
        if not in_frontmatter:
            continue
        if stripped.startswith("class:"):
            return stripped.split(":", 1)[1].strip().strip("\"'").casefold() or None
    return None


def sweep(config_dir: Path | str | None = None, *, now_ms: int | None = None) -> list[str]:
    """One trigger evaluation pass. Returns the targets whose records were
    written or merged.

    The whole suppression matrix, in order (see the module docstring). In the
    suppressed states this performs READS ONLY — one env read and one stat/read
    of ``aida/state.json`` — so a machine that never enabled the target never
    grows a ``wakes/triggers/`` directory from this path.
    """
    root = Path(config_dir) if config_dir is not None else _default_config_dir()
    if root is None:
        return []
    if _env_disabled():
        return []
    target = _aida_target(root)
    if not target:
        return []
    values = read_settings(root)
    if not _strict_bool(values.get("wakes.triggers.enabled"), DEFAULT_ENABLED):
        return []
    if _target_held(root, target):
        return []
    if _class_reactive(root, target):
        return []
    moment = _now_ms() if now_ms is None else int(now_ms)
    instances = evaluate_all(root, moment, values)
    return commit(instances, root, moment, values)


def commit(
    instances: Sequence[TriggerInstance],
    config_dir: Path,
    now_ms: int,
    values: Mapping[str, Any],
) -> list[str]:
    """Merge ``instances`` into the target's pending record. Returns targets
    written/merged.

    The ONLY writer of pending records. Dedupe compares each instance's
    fingerprint against ``state.json``; instances whose fingerprint has not
    moved since they were notified are not candidates. Budgets are per target
    over a rolling 24 h window: ``wakes.triggers.max_per_day`` caps record
    creations, ``wakes.triggers.min_gap_minutes`` spaces them, and a blocked
    instance stays a candidate — nothing is marked notified until a record is
    actually written. Only the CREATION of a record counts as a fire; a merge
    into an existing record rides that record's single wake.
    """
    if not instances:
        return []
    root = Path(config_dir)
    target = _aida_target(root)
    if not target:
        return []
    max_per_day = _int_value(values, "wakes.triggers.max_per_day", DEFAULT_MAX_PER_DAY)
    gap_minutes = _int_value(values, "wakes.triggers.min_gap_minutes", DEFAULT_MIN_GAP_MINUTES)
    state = _read_state(root)
    budget = _target_state(state, target)
    window_start = now_ms - 86_400_000
    fires = [f for f in budget["fires"] if f > window_start]

    # -- dedupe ---------------------------------------------------------------
    known = state["instances"]
    candidates = [
        instance
        for instance in instances
        if _fingerprint_list(known.get(_instance_key(_instance_as_entry(instance))))
        != list(instance.fingerprint)
    ]
    if not candidates:
        return []

    # -- bounds ---------------------------------------------------------------
    # Checked BEFORE any marking: a blocked instance must stay a candidate.
    if max_per_day <= 0:
        return []
    if len(fires) >= max_per_day:
        return []
    last_fire = budget["last_fire_ms"]
    if gap_minutes > 0 and last_fire is not None and now_ms - last_fire < gap_minutes * 60_000:
        return []

    # -- merge ----------------------------------------------------------------
    existing = read_pending_record(root, target)
    record = dict(existing) if existing else {}
    merged: list[dict[str, Any]] = [
        dict(entry) for entry in (record.get("instances") or []) if isinstance(entry, dict)
    ]
    for instance in candidates:
        entry = _instance_as_entry(instance)
        for position, prior in enumerate(merged):
            if (prior.get("source"), prior.get("key")) == (entry["source"], entry["key"]):
                merged[position] = entry  # same instance, newer episode state
                break
        else:
            merged.append(entry)
    merged.sort(key=lambda item: (-_age_of(item), str(item.get("source")), str(item.get("key"))))
    overflow = max(0, len(merged) - INSTANCE_CAP)
    record["schema_version"] = RECORD_SCHEMA
    record["target"] = target
    record["noted_at_ms"] = (
        record.get("noted_at_ms") if isinstance(record.get("noted_at_ms"), int) else now_ms
    )
    record["updated_at_ms"] = now_ms
    record["instances"] = merged[:INSTANCE_CAP]
    record["overflow"] = overflow
    # A merge that brings new work pulls the retry walk no later than one base
    # interval from now — the same reasoning spooled states for a re-note: the
    # record's promise ("this gets acted on") has just grown, without letting a
    # stream of merges restart an already-short wait.
    if existing is not None:
        prior_next = next_attempt_at_ms(record)
        record["next_attempt_ms"] = (
            min(prior_next, now_ms + int(RETRY_BASE_S * 1000)) if prior_next else 0
        )
    else:
        record.setdefault("attempts", 0)
        record.setdefault("next_attempt_ms", 0)

    if not _write_json(pending_path(root, target), record):
        return []

    # -- state: notified + budget ---------------------------------------------
    for instance in candidates:
        known[_instance_key(_instance_as_entry(instance))] = list(instance.fingerprint)
    if existing is None:
        fires.append(now_ms)
        last_fire = now_ms
    state["instances"] = known
    state["targets"] = dict(state.get("targets") or {})
    # The PRUNED window is what gets written back on both paths — a merge that
    # appended nothing must still shed fires that fell out of the 24 h window,
    # or the list would never shrink on a install whose records only ever merge.
    state["targets"][target] = {"fires": fires, "last_fire_ms": last_fire}
    _write_json(state_path(root), state)
    logger.info(
        "trigger record for %s: %d instance(s) (%d new), overflow %d, attempts %s",
        target,
        len(record["instances"]),
        len(candidates),
        overflow,
        record.get("attempts") or 0,
    )
    return [target]


def _instance_as_entry(instance: TriggerInstance) -> dict[str, Any]:
    return {
        "source": str(instance.source),
        "key": str(instance.key),
        "fingerprint": list(instance.fingerprint),
        "age_s": _coerce_age(instance.age_s),
        "payload": dict(instance.payload),
    }


def _coerce_age(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _age_of(entry: Mapping[str, Any]) -> float:
    return _coerce_age(entry.get("age_s"))


def reconcile(config_dir: Path, pending: dict[str, dict[str, Any]]) -> None:
    """Drop pending records whose cause is gone. Mutates the caller's copy.

    Two drops, both derived state:
    * expired — older than :data:`RECORD_TTL_S`; a check-in about three days
      ago is noise, and the target's own cadence has covered the interval;
    * ghost — the target's session no longer exists on disk (a reap, a
      hand-deleted directory), so no runtime can ever consume it. The
      transcript is the existence test, matching ``supervisor._session_exists``
      (a directory without a transcript is not a session anything can resume).
    """
    moment = _now_ms()
    for session_id, record in list(pending.items()):
        if not isinstance(record, dict):
            del pending[session_id]
            continue
        updated = record.get("updated_at_ms")
        updated_ms = updated if isinstance(updated, int) and not isinstance(updated, bool) else 0
        expired = updated_ms > 0 and (moment - updated_ms) / 1000.0 > RECORD_TTL_S
        transcript = Path(config_dir) / "sessions" / session_id / _TRANSCRIPT_NAME
        if expired or not transcript.is_file():
            try:
                pending_path(config_dir, session_id).unlink()
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning(
                    "could not drop the trigger record for %s", session_id, exc_info=True
                )
                continue
            del pending[session_id]
            if expired:
                logger.info("trigger record for %s expired (older than 72 h)", session_id)


# ---------------------------------------------------------------------------
# The published settings snapshot
# ---------------------------------------------------------------------------


def read_settings(config_dir: Path | str) -> dict[str, Any]:
    """The trigger settings: the published snapshot merged over the defaults.

    Never raises, never returns a partial mapping: an absent, unreadable or
    future-schema snapshot yields exactly the module defaults, which are the
    registry's own — so a missing snapshot cannot change behaviour, only miss a
    recent edit. Values of the five keys the snapshot carries are the only ones
    taken from it; a hand-written extra key is ignored.
    """
    values = dict(DEFAULTS)
    try:
        data = json.loads(settings_path(config_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return values
    if not isinstance(data, dict) or data.get("schema_version") != SETTINGS_SCHEMA:
        return values
    raw = data.get("values")
    if isinstance(raw, Mapping):
        for key in SNAPSHOT_KEYS:
            if key in raw:
                values[key] = raw[key]
    return values


def publish_settings(config_dir: Path | str | None = None, *, manager: Any = None) -> bool:
    """Write the supervisor-readable snapshot of the trigger settings.

    The ONE publisher, called by every config-aware writer (the settings
    write/reset paths, Aida's boot and reconcile). The values come from the
    settings registry's own reader over a ``ConfigManager`` — the same reader
    the settings page paints from, so the snapshot can never disagree with a
    value a writer wrote, which a hand-rolled YAML scan could. Best-effort by
    contract: a failure logs and returns False, never raises, because every
    caller is already mid-write for something else.
    """
    try:
        from local_operator import settings_io

        if manager is None:
            from local_operator.config import ConfigManager

            root = Path(config_dir) if config_dir is not None else _default_config_dir()
            if root is None:
                return False
            manager = ConfigManager(config_dir=root)
        else:
            root = Path(config_dir) if config_dir is not None else Path(manager.config_dir)
        values: dict[str, Any] = {}
        for key in SNAPSHOT_KEYS:
            setting = settings_io.resolve_key(key)
            if setting is None:
                return False
            values[key] = settings_io.read_setting(manager, setting)
        moment = _now_ms()
        path = settings_path(root)
        # Skip a write whose VALUES did not move: reconcile runs on every one
        # of the target's persists, and rewriting an identical file (and, worse,
        # moving updated_at_ms) on each would be churn with no reader.
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = None
        if isinstance(current, dict) and current.get("values") == values:
            return True
        payload = {
            "schema_version": SETTINGS_SCHEMA,
            "values": values,
            "updated_at_ms": moment,
        }
        return _write_json(path, payload)
    except Exception:  # noqa: BLE001 — a snapshot must never fail a write
        logger.warning("could not publish the trigger settings snapshot", exc_info=True)
        return False


def trigger_row_id(instances: Sequence[Mapping[str, Any]]) -> str:
    """The deterministic id for the check-in row covering ``instances``:
    ``aida-trigger-<8 hex>`` over the sorted instance fingerprints.

    Deterministic so a re-consume of the same record re-arms the same id and
    the engine can diff it against the resident list; a moved fingerprint (a
    new episode) is a new id by construction.
    """
    fingerprints = sorted(
        json.dumps(list(entry.get("fingerprint") or []), separators=(",", ":"), sort_keys=True)
        for entry in instances
    )
    digest = hashlib.sha1("\n".join(fingerprints).encode("utf-8")).hexdigest()
    return f"aida-trigger-{digest[:8]}"
