"""The monitor spec — one armed monitor's durable shape (contract §10.1).

Pydantic + stdlib only at module scope, deliberately: ``harness.types``
annotates ``MonitorSchedulerProtocol`` with :class:`MonitorSpec` the way it
annotates ``WakeSchedulerProtocol`` with ``WakeSchedule``, and the wake stack
paid for that lesson once (``harness/wake_types.py``'s docstring records the
~203 ms the wake scheduler used to drag onto every ``harness.types`` import).
The duration/ISO parsing helpers are imported lazily inside
:func:`build_monitor_spec` for the same reason — ``harness.wake`` must not
enter this module's import graph through the back door.

Reading the contract:

- IDs are ``m1``… ``mN``, stable per session and NEVER reused after a cancel.
  That last property is why allocation is a high-water sequence (persisted in
  the transcript entry beside the rows, ``next_seq``) rather than "first free
  slot": the CLI's ``w{n}`` bug (``wakes/arm.py``'s docstring) came of
  ``len(existing) + 1`` reissuing a live id, and a first-free-slot allocator
  reissues a CANCELLED id — which is exactly what ``next_seq`` exists to
  prevent.
- ``arguments`` are validated by the caller against the same schema the loop
  validates calls with AND against the read-only evaluator before a spec is
  ever built; the validator callback parameter is that seam.
- ``every_ms`` carries the floor at the FIELD level (the ``WakeSchedule``
  precedent): ``load()`` adopts rows straight from a hand-editable transcript,
  and a 0 there would make the scheduler divide by zero.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: The floor on a monitor interval (§11.1). A tick is one read-only call with
#: no turn and no tokens; 30 s bounds the call frequency and matches the
#: shortest useful web-cache window. The wake floor stays 60 s because a wake
#: opens a turn.
MIN_MONITOR_INTERVAL_MS = 30_000

#: A name bound so list rows and index entries stay one line. Not a contract
#: number: the spec's only demand is that the name is "short".
MAX_MONITOR_NAME_CHARS = 120

#: At most this many drop-regexes per monitor (§7.1).
MAX_MONITOR_IGNORE = 8

#: Transcript custom-entry type for the schedule list, and the custom type of
#: a delivered delta — the wake pair (``wake_schedules`` / ``wake_prompt``).
MONITOR_SCHEDULES_CUSTOM_TYPE = "monitor_schedules"
MONITOR_PROMPT_MESSAGE_TYPE = "monitor_prompt"

_ID_RE = re.compile(r"^m(\d+)$")


class MonitorSpec(BaseModel):
    """One armed monitor. ``id`` is a stable per-session handle (``m1``…)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    tool: str  # session tool name, e.g. "bash"
    arguments: dict[str, Any]  # validated against the tool's schema at arm time
    every_ms: int = Field(ge=MIN_MONITOR_INTERVAL_MS)
    until_at: int | None = None  # None = durable
    description: str = ""
    notify: bool = False  # §14 — rides the delivery's details; quiet stays quiet
    sort_lines: bool = False
    ignore: list[str] = Field(default_factory=list)
    cwd: str = ""  # captured at arm; the check runs with it
    created_at: int = 0

    @field_validator("name")
    @classmethod
    def _name_bounded(cls, value: str) -> str:
        if len(value) > MAX_MONITOR_NAME_CHARS:
            raise ValueError(f"name must be at most {MAX_MONITOR_NAME_CHARS} characters")
        return value

    @field_validator("ignore")
    @classmethod
    def _ignore_bounded(cls, value: list[str]) -> list[str]:
        if len(value) > MAX_MONITOR_IGNORE:
            raise ValueError(f"at most {MAX_MONITOR_IGNORE} ignore patterns are allowed")
        for pattern in value:
            re.compile(pattern)  # a bad regex is a refused row, not a broken tick
        return value


def spec_identity(tool: str, arguments: Mapping[str, Any]) -> str:
    """The dedupe identity: ``sha256(tool + canonical-JSON(arguments))``.

    Canonical JSON (sorted keys, compact separators) so two spellings of the
    same call hash alike. NOT the name or the interval (§11.4): the thing a
    duplicate monitor duplicates is the polled call, and two rows polling one
    source is what this settles.
    """
    try:
        payload = json.dumps(dict(arguments), sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        # Unserialisable arguments never reach here from the tool path (pydantic
        # validates the schema first), but the identity function itself must not
        # be a failure mode — fall back to repr.
        payload = repr(arguments)
    return hashlib.sha256(f"{tool}\n{payload}".encode()).hexdigest()


def next_monitor_seq(ids: Sequence[str], stored_next_seq: Any = None) -> int:
    """The next free sequence number, never below any id already issued.

    ``stored_next_seq`` is the persisted high-water mark (the transcript entry
    carries it so an emptied list cannot reissue ``m1``); it is validated here,
    not trusted, because the transcript is hand-editable.
    """
    high = 0
    if isinstance(stored_next_seq, int) and not isinstance(stored_next_seq, bool):
        high = max(stored_next_seq, 0)
    for monitor_id in ids:
        match = _ID_RE.match(str(monitor_id))
        if match:
            high = max(high, int(match.group(1)) + 1)
    return max(high, 1)


def allocate_monitor_id(existing: Sequence[str], *, high_water: int) -> tuple[str, int]:
    """Allocate the next id and the bumped high-water mark.

    Returns ``(id, next_high_water)``. The caller persists the new mark in the
    same write that persists the row, so a crash cannot reissue the id.
    """
    seq = next_monitor_seq(existing, high_water)
    return f"m{seq}", seq + 1


class MonitorBuilt(TypedDict):
    """A validated spec, ready to hand to the scheduler."""

    spec: MonitorSpec


class MonitorBuildFailed(TypedDict):
    """Why the request was rejected, phrased for the model to act on.

    ``malformed`` separates the two reasons a request can be refused, exactly
    like ``WakeBuildFailed``: ``True`` means an argument could never have been
    valid (a bad duration, a bad regex — the model's fault, and the tool
    reports it as ``InvalidToolArgumentsError`` semantics), ``False`` means the
    request was well-formed but unsupportable (a write-tier target, a past
    ``until``).
    """

    error: str
    malformed: bool


def build_monitor_spec(
    request: Mapping[str, Any],
    *,
    monitor_id: str,
    now_ms: int,
    settings: Any,
    cwd: str,
    validate: Callable[[str, Mapping[str, Any]], str | None],
) -> MonitorBuilt | MonitorBuildFailed:
    """Validate one create request into a :class:`MonitorSpec`, or a sentence.

    Returns the error as TEXT rather than raising (the ``build_wake_schedule``
    posture), so the tool's failure path is a sentence the model can act on.

    ``validate`` is the arm-time read-only gate: ``(tool, arguments) ->
    refusal | None`` — the scheduler passes the session's own resolver, so the
    spec a caller gets back is one the harness just proved monitorable. This
    function performs the SHAPE validation; caps, dedupe and id allocation
    belong to the caller (the scheduler), which owns the live list.
    """
    tool = request.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        return {"error": "'create' requires 'tool' and 'arguments'", "malformed": True}
    tool = tool.strip()

    arguments_raw = request.get("arguments")
    if arguments_raw is None:
        arguments: dict[str, Any] = {}
    elif isinstance(arguments_raw, Mapping):
        arguments = dict(arguments_raw)
    else:
        return {"error": "'arguments' must be a JSON object.", "malformed": True}

    name = str(request.get("name") or "").strip()
    if not name:
        # The spec requires a name so a row is never anonymous; deriving it
        # from the tool costs the model no extra round trip when it omits one.
        name = tool
    if len(name) > MAX_MONITOR_NAME_CHARS:
        return {
            "error": f"monitor name must be at most {MAX_MONITOR_NAME_CHARS} characters.",
            "malformed": True,
        }

    from local_operator.harness.wake import parse_wake_at, parse_wake_duration

    every_raw = request.get("every")
    if every_raw is None:
        every_ms = int(settings.default_interval_ms)
    else:
        parsed = parse_wake_duration(str(every_raw))
        if parsed is None:
            return {
                "error": f"invalid 'every' duration '{every_raw}'; use e.g. 30s, 60s, 5m, 1h.",
                "malformed": True,
            }
        every_ms = parsed
    if every_ms < MIN_MONITOR_INTERVAL_MS:
        # NOT malformed: "every 5s" is a real duration this feature declines to
        # honour, not a spelling mistake — same split the wake floor uses.
        return {
            "error": f"monitor interval must be at least {MIN_MONITOR_INTERVAL_MS // 1000}s.",
            "malformed": False,
        }

    until_at: int | None = None
    until_raw = request.get("until")
    if until_raw is not None:
        parsed_until = parse_wake_at(str(until_raw), now_ms)
        if parsed_until is None:
            return {
                "error": f"invalid 'until' time '{until_raw}'; use an ISO datetime.",
                "malformed": True,
            }
        if parsed_until <= now_ms:
            return {
                "error": "'until' is in the past — the monitor would never check.",
                "malformed": False,
            }
        until_at = parsed_until

    description = request.get("description") or ""
    if not isinstance(description, str):
        description = str(description)

    ignore_raw = request.get("ignore")
    if ignore_raw is None:
        ignore: list[str] = []
    elif isinstance(ignore_raw, (list, tuple)):
        ignore = [str(item) for item in ignore_raw]
    else:
        return {"error": "'ignore' must be a list of regexes.", "malformed": True}
    if len(ignore) > MAX_MONITOR_IGNORE:
        return {
            "error": f"at most {MAX_MONITOR_IGNORE} ignore patterns are allowed.",
            "malformed": True,
        }
    for pattern in ignore:
        try:
            re.compile(pattern)
        except re.error as exc:
            return {"error": f"invalid ignore regex '{pattern}': {exc}.", "malformed": True}

    def as_bool(key: str) -> bool:
        raw = request.get(key)
        if raw is None:
            return False
        if isinstance(raw, bool):
            return raw
        from local_operator.settings_io import strict_bool

        return strict_bool(raw, False)

    # The read-only gate LAST: it is the expensive check (walks the session's
    # tool set, parses bash), and a request that fails shape validation should
    # not pay for it. Fail loudly, never arm-then-skip (§6.7).
    refusal = validate(tool, arguments)
    if refusal is not None:
        return {"error": refusal, "malformed": False}

    spec = MonitorSpec(
        id=monitor_id,
        name=name,
        tool=tool,
        arguments=arguments,
        every_ms=every_ms,
        until_at=until_at,
        description=description,
        notify=as_bool("notify"),
        sort_lines=as_bool("sort_lines"),
        ignore=ignore,
        cwd=cwd,
        created_at=now_ms,
    )
    return {"spec": spec}
