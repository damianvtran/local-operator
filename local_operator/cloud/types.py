"""The cloud compute seam's records: the intent, and what became of it.

WHY THIS EXISTS. Two tiers of auto-provisioned compute are planned — one ECS
Fargate task per run, and dedicated workloads on a two-pool EKS cluster
(``docs/design/cloud-compute-tiers.md`` §5) — and both are described by ONE
request record and ONE lifecycle record. Two shapes that agree today is how two
readers come to disagree, which is the argument ``session/placement.py`` makes
for ``SessionPlacement`` ("ONE dataclass serialised in both places rather than
two shapes that agree today"), so the pool contract's request keys and this
slice's cloud delta live in one dataclass, and the Fargate provider, the
Kubernetes provider and the fake all speak it.

WHAT LIVES WHERE. :class:`PoolRequest` is IMMUTABLE INTENT — written once, never
re-opened. :class:`ComputeTask` is the MUTABLE LIFECYCLE, whose one writer is
``ledger.CloudLedger``. ``state`` and ``grant_id`` sit on the task rather than on
the request for that reason: the pool contract puts both on its request because
its reader is a remote control plane polling a directory, and two mutable fields
on an immutable record is how a record gets two writers.

SERIALISATION (§3.5). Hand-written ``to_json``/``from_json``, never ``asdict``:
the house pattern for a record that has to survive a version skew is the LENIENT
reader (``SessionPlacement.from_json``, ``MemberRecord.from_json``'s ``_known``)
and ``asdict`` cannot be lenient. The rules, so two slices cannot diverge:

* keys are ``snake_case``, exactly the dataclass field names;
* ABSENT IS NOT A CLAIM — an absent key takes its default, and a ``version: 1``
  request from ``mesh-compute-pool.md`` §4.1 (whose ``size_class`` was the one
  spelling this slice renamed, and whose ``state``/``grant_id`` moved to the
  task) parses with the cloud delta defaulted;
* a key that is PRESENT but names something this build does not know — a version,
  a tier, a size class, a state — is REFUSED by name rather than coerced. That is
  the one direction that matters: silently reading an unknown ``size_class`` as
  the default ``m`` provisions the wrong machine, and the reader that guesses is
  the reason ``network/types.py``'s ``trust_state`` refuses instead of coercing;
* no secret ever appears: no token, no key, no ARN-with-credentials.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from secrets import token_hex
from typing import Any, Literal, NoReturn, cast

from local_operator.cloud.provider import CloudRefused
from local_operator.network.types import MemberLifecycle

Tier = Literal["fargate", "k8s"]
SizeClass = Literal["s", "m", "l"]
Hosting = Literal["hosted", "byo"]

#: The vocabularies as runtime sets, mirrored by the readers below — the one
#: place a string becomes one of the Literals above.
TIERS: frozenset[str] = frozenset({"fargate", "k8s"})
HOSTINGS: frozenset[str] = frozenset({"hosted", "byo"})


@dataclass(frozen=True)
class Size:
    """One size class's compute shape. The PRICE of a class is a rate, not a shape."""

    vcpu: float
    memory_gib: float


#: The size classes, which ``mesh-compute-pool.md`` §3.3 leaves opaque ("compute
#: size class, opaque to the mesh"). The ladder is the POC's ``m`` (2 vCPU /
#: 4 GiB, the only class Slice 0 ever ran) with one step either side.
SIZES: dict[SizeClass, Size] = {"s": Size(1.0, 2.0), "m": Size(2.0, 4.0), "l": Size(4.0, 8.0)}

#: ``version: 1`` was ``mesh-compute-pool.md`` §4.1's shape; ``2`` adds the cloud
#: delta. A superset, so both are readable and only a LATER version is refused.
POOL_REQUEST_VERSION = 2
POOL_REQUEST_VERSIONS: frozenset[int] = frozenset({1, POOL_REQUEST_VERSION})

#: The task record's shape version, bumped only for a change an older reader
#: could MISREAD (adding a field with a default is not one).
COMPUTE_TASK_VERSION = 1
COMPUTE_TASK_VERSIONS: frozenset[int] = frozenset({COMPUTE_TASK_VERSION})

TaskState = Literal[
    "queued",
    "provisioning",
    "running",
    "draining",  # live
    "expired",
    "failed",
    "cancelled",
    "lost",  # terminal
]
#: Terminal states. WRITE-ONCE: the first terminal transition wins and nothing may
#: leave a terminal state. That is ``network/move_queue.py``'s
#: ``QUEUE_TERMINAL_PHASES`` discipline ("each win exactly once and no later
#: transition may overwrite them, which is what makes 'cancel vs a relay
#: transition' race-safe without either side re-reading a stale copy"), applied to
#: this record.
TERMINAL_STATES: frozenset[str] = frozenset({"expired", "failed", "cancelled", "lost"})

#: Every state, live ones first — the reader's vocabulary and the parametrisation
#: the mapping test walks.
TASK_STATES: tuple[str, ...] = (
    "queued",
    "provisioning",
    "running",
    "draining",
    "expired",
    "failed",
    "cancelled",
    "lost",
)

#: The transitions the state machine allows, keyed by the state they leave. Every
#: edge here is in §3.4's table; a pair that is not in this map is a NO-OP rather
#: than an error, because the caller is usually a poll that re-observed the world
#: and the world had moved on.
#:
#: ``lost`` is reachable from every live state because §3.4's row is "any live":
#: the trigger is an OBSERVATION (``status().phase == "absent"`` with no stop
#: record), and an observation is not something the state machine can pre-empt.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"provisioning", "cancelled", "failed", "lost"}),
    "provisioning": frozenset({"running", "cancelled", "failed", "lost"}),
    "running": frozenset({"draining", "cancelled", "failed", "lost"}),
    "draining": frozenset({"expired", "failed", "lost"}),
}

#: The ``stop_reason`` a terminal state defaults to when the caller names none.
#: ``expired`` folds to ``drained`` because §3.4's own vocabulary for this field
#: ("drained | lost | cancelled | failed | over_deadline") has no ``expired``
#: member: an expiry that reached its checkpoint IS the drain. A drain whose
#: checkpoint never arrived passes ``reason="drained_unconfirmed"`` explicitly,
#: because the record must not read as success.
DEFAULT_STOP_REASON: dict[str, str] = {
    "expired": "drained",
    "failed": "failed",
    "cancelled": "cancelled",
    "lost": "lost",
}

#: The ``extra`` keys a transition writes onto the record rather than only into the
#: event's ``detail``. Closed on purpose: a key that is not here is journaled and
#: nothing else, so a mistyped field name cannot look like it was recorded. See
#: ``ComputeTask._apply_record_fields`` for why each of these arrives with a
#: transition rather than needing a writer of its own.
RECORD_FIELDS: tuple[str, ...] = (
    "provider",
    "provider_ref",
    "session_id",
    "exit_code",
    "artifacts",
    "checkpoint",
)


def _as_str(value: Any) -> str:
    """``value`` as a string, with ``None`` and any container read as absent."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value)


def _as_float(value: Any, *, default: float = 0.0) -> float:
    """``value`` as a float, or ``default`` — never an exception on a bad record."""
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> float | None:
    """``value`` as a float, or ``None`` when absent or unreadable.

    ``None`` rather than the epoch: these fields say "no timestamp yet", and a
    zero would read as 1970 to anything that formats them.
    """
    if value is None:
        return None
    parsed = _as_float(value, default=-1.0)
    return None if parsed < 0.0 else parsed


def _as_int(value: Any, *, default: int = 0) -> int:
    """``value`` as an int, or ``default`` — via the same total helper as floats.

    A version or an exit code arriving as a JSON string ("1") is ordinary; a
    401-digit integer is not, and it must not win a comparison.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    parsed = _as_float(value, default=float(default))
    return int(parsed)


def _as_optional_int(value: Any) -> int | None:
    """``value`` as an int, or ``None`` when absent or unreadable."""
    return None if value is None else _as_int(value, default=-1)


def _as_bool(value: Any) -> bool:
    """``value`` as a bool, reading the ``"1"``/``"0"`` an env override uses.

    Plain ``bool(value)`` would make the string ``"0"`` true, and the container
    contract passes ``POC_MOCK`` as exactly that.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _as_str_map(value: Any) -> dict[str, str]:
    """``value`` as a ``{str: str}`` map, dropping anything that is not a mapping."""
    if not isinstance(value, dict):
        return {}
    return {str(key): _as_str(item) for key, item in value.items()}


def _refuse(code: str, sentence: str) -> NoReturn:
    """Refuse by name. One helper so every reader below refuses the same way."""
    raise CloudRefused(code, sentence)


def _read_version(data: dict[str, Any], *, readable: frozenset[int], what: str) -> int:
    """The record's shape version, refusing a version this build cannot read.

    ABSENT is the current version (an absent key is not a claim, which is the
    discipline ``SessionPlacement.from_json`` applies); a version that is PRESENT
    and outside ``readable`` is refused by name rather than guessed at, because a
    record written by a newer build can carry a field this one would misread.
    """
    raw = data.get("version")
    if raw is None:
        return max(readable)
    version = _as_int(raw, default=-1)
    if version not in readable:
        readable_list = ", ".join(str(item) for item in sorted(readable))
        _refuse(
            "version_unknown",
            f"cloud: this {what} is version {raw!r} and this build reads {readable_list}",
        )
    return version


def _read_choice(value: Any, *, allowed: frozenset[str], default: str, code: str, what: str) -> str:
    """A closed vocabulary field: absent takes ``default``, unknown is refused.

    The asymmetry is deliberate and is ``network/types.py``'s ``trust_state``
    rule: a value that cannot be read must never be quietly promoted to the
    default, because the default here is a MACHINE (``m`` is 2 vCPU / 4 GiB) and
    the refusal is the only thing standing between a renamed field and a run on
    the wrong size.
    """
    if value is None:
        return default
    text = _as_str(value)
    if text not in allowed:
        _refuse(code, f"cloud: unknown {what} {value!r}; this build knows {sorted(allowed)}")
    return text


def _read_size(value: Any) -> SizeClass:
    """A size class: absent takes the default, an unknown one is refused by name."""
    return cast(
        SizeClass,
        _read_choice(
            value, allowed=frozenset(SIZES), default="m", code="size_unknown", what="size class"
        ),
    )


def _read_tier(value: Any) -> Tier:
    """A tier: absent takes the default, an unknown one is refused by name."""
    return cast(
        Tier,
        _read_choice(value, allowed=TIERS, default="fargate", code="tier_unknown", what="tier"),
    )


def _read_hosting(value: Any) -> Hosting:
    """A hosting mode: absent takes the default, an unknown one is refused by name."""
    return cast(
        Hosting,
        _read_choice(
            value, allowed=HOSTINGS, default="hosted", code="hosting_unknown", what="hosting mode"
        ),
    )


def _read_state(value: Any) -> TaskState:
    """A task state: absent is ``queued``; a state this build does not know is refused.

    The refusal code is ``version_unknown`` rather than a state-specific one: a
    record carrying a state outside :data:`TASK_STATES` was written by a build
    whose vocabulary this one does not have, which is what the code means.
    """
    return cast(
        TaskState,
        _read_choice(
            value,
            allowed=frozenset(TASK_STATES),
            default="queued",
            code="version_unknown",
            what="task state",
        ),
    )


def new_request_id() -> str:
    """``req_<8 hex>``: the ledger filename stem, and the branch's own suffix."""
    return f"req_{token_hex(4)}"


@dataclass(frozen=True)
class PoolRequest:
    """What was asked for. Immutable intent: written once and never re-opened.

    The first seven fields are ``mesh-compute-pool.md`` §4.1's keys verbatim; the
    rest is the cloud delta. One dataclass rather than two shapes that agree
    today (see the module docstring).
    """

    # --- mesh-compute-pool §4.1 keys, kept verbatim ---------------------------------
    version: int = POOL_REQUEST_VERSION
    request_id: str = ""
    network_id: str = ""  # "" = unbound (this slice; §9.3)
    size: SizeClass = "m"
    max_hours: float = 2.0
    requested_at: float = 0.0
    requested_by: str = ""  # device_id of the requester ("" when there is no network)
    # --- the cloud delta -----------------------------------------------------------
    tier: Tier = "fargate"
    budget_usd: float = 0.0  # model+compute cap; submit refuses 0
    spot_ok: bool = False
    repo_url: str = ""
    repo_sha: str = ""  # "" = clone the default branch tip
    prompt: str = ""
    hosting: Hosting = "hosted"
    model: str = ""  # provider model id; "" = the profile default
    labels: dict[str, str] = field(default_factory=dict)
    max_sessions: int = 1  # one member per task (remote-cloud-agents.md §3.3)
    include_local: bool = False  # upload a git bundle of unpushed commits
    mock: bool = False  # the POC's --mock, for a key-free end-to-end run

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "network_id": self.network_id,
            "size": self.size,
            "max_hours": self.max_hours,
            "requested_at": self.requested_at,
            "requested_by": self.requested_by,
            "tier": self.tier,
            "budget_usd": self.budget_usd,
            "spot_ok": self.spot_ok,
            "repo_url": self.repo_url,
            "repo_sha": self.repo_sha,
            "prompt": self.prompt,
            "hosting": self.hosting,
            "model": self.model,
            "labels": dict(self.labels),
            "max_sessions": self.max_sessions,
            "include_local": self.include_local,
            "mock": self.mock,
        }

    @staticmethod
    def from_json(data: Any) -> PoolRequest:
        """Parse a request, leniently about absence and strict about unknowns.

        Reads ``size_class`` as well as ``size``: that is §4.1's own spelling and
        a version 1 file really does carry it. Ignoring it would not fail — the
        default happens to be ``m`` — which is exactly why it is read explicitly
        rather than left to a coincidence. The pool contract's ``state`` and
        ``grant_id`` are deliberately NOT read: they belong to the task record.
        """
        if not isinstance(data, dict):
            _refuse("version_unknown", "cloud: a pool request must be a JSON object")
        version = _read_version(data, readable=POOL_REQUEST_VERSIONS, what="pool request")
        raw_size = data.get("size", data.get("size_class"))
        return PoolRequest(
            version=version,
            request_id=_as_str(data.get("request_id")),
            network_id=_as_str(data.get("network_id")),
            size=_read_size(raw_size),
            max_hours=_as_float(data.get("max_hours"), default=2.0),
            requested_at=_as_float(data.get("requested_at")),
            requested_by=_as_str(data.get("requested_by")),
            tier=_read_tier(data.get("tier")),
            budget_usd=_as_float(data.get("budget_usd")),
            spot_ok=_as_bool(data.get("spot_ok")),
            repo_url=_as_str(data.get("repo_url")),
            repo_sha=_as_str(data.get("repo_sha")),
            prompt=_as_str(data.get("prompt")),
            hosting=_read_hosting(data.get("hosting")),
            model=_as_str(data.get("model")),
            labels=_as_str_map(data.get("labels")),
            max_sessions=_as_int(data.get("max_sessions"), default=1),
            include_local=_as_bool(data.get("include_local")),
            mock=_as_bool(data.get("mock")),
        )


@dataclass
class ComputeTask:
    """One request's lifecycle: the mutable half, and the ledger's record.

    Every timestamp here is OBSERVED (the provider said so, or the ledger watched
    the phase change) — none is inferred from a deadline, which is why
    ``deadline_at`` and ``expires_at`` are separate fields from ``stopped_at``.
    """

    version: int = COMPUTE_TASK_VERSION
    request_id: str = ""
    tier: Tier = "fargate"
    provider: str = ""  # "" until submit
    provider_ref: str = ""  # an ECS task ARN, or "<namespace>/<statefulset>"
    state: TaskState = "queued"
    grant_id: str = ""  # the pool grant, once there is one (absent this slice)
    size: SizeClass = "m"
    spot: bool = False  # what we asked for
    spot_fallback: bool = False  # whether the static fallback fired (§5.4)
    created_at: float = 0.0
    submitted_at: float | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    deadline_at: float | None = None  # created_at + max_hours * 3600
    expires_at: float | None = None  # the member-side hard stop (pool contract §3.3)
    last_seen_at: float | None = None
    exit_code: int | None = None
    stop_reason: str = ""  # drained | lost | cancelled | failed | over_deadline
    budget_usd: float = 0.0
    repo_url: str = ""
    repo_sha: str = ""
    branch: str = ""  # "lop/<request_id>"
    session_id: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> path or URL
    checkpoint: dict[str, str] = field(default_factory=dict)  # what drain() returned
    #: ``[{"at": float, "phase": str, "detail": {...}}]``. APPEND-ONLY: a later
    #: reader must be able to see the whole sequence, so nothing ever edits an
    #: entry that is already there.
    events: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, state: TaskState, *, detail: str = "", **extra: Any) -> bool:
        """Advance to ``state``, or report that nothing advanced.

        Returns ``True`` only when the state CHANGED. A re-observation of the
        state the record is already in returns ``False`` and appends nothing:
        the poll that reads ``running`` ten times is one transition, and an event
        journal that grew by ten would hide the sequence it exists to show. An
        accepted reading stamps ``last_seen_at`` either way; a refused one
        changes nothing at all.

        Terminal states are WRITE-ONCE (§3.4, and ``move_queue``'s
        ``QUEUE_TERMINAL_PHASES`` rule): once the record is terminal this method
        writes nothing at all — not the state, not the reason, not the journal —
        which is what makes "cancel vs the provider's own stop" resolve to
        whichever the ledger saw first.

        ``detail`` is a human sentence, ``extra`` carries the machine fields the
        transition observed (``reason``, ``exit_code``, ``artifacts``,
        ``checkpoint``), and both land in the appended event's ``detail`` map.
        """
        now = time.time()
        if self.state in TERMINAL_STATES:
            return False
        if state == self.state:
            self.last_seen_at = now
            return False
        if state not in ALLOWED_TRANSITIONS.get(self.state, frozenset()):
            # Not this record's machine (§3.4's table). Refused rather than
            # coerced: a caller asking a queued task to be ``running`` before it
            # was ever submitted is a caller bug, and recording it would put a
            # state in the journal that nothing observed.
            return False
        self.last_seen_at = now
        self.state = state
        event_detail: dict[str, Any] = dict(extra)
        if detail:
            event_detail["message"] = detail
        self.events.append({"at": now, "phase": state, "detail": event_detail})
        if state == "provisioning" and self.submitted_at is None:
            self.submitted_at = now
        if state == "running" and self.started_at is None:
            self.started_at = now
        if state in TERMINAL_STATES:
            self.stopped_at = now
            self.stop_reason = _as_str(extra.get("reason")) or DEFAULT_STOP_REASON[state]
        self._apply_record_fields(extra)
        return True

    def _apply_record_fields(self, extra: dict[str, Any]) -> None:
        """Write the ``extra`` keys that name a RECORD field, not only an annotation.

        A CLOSED list (:data:`RECORD_FIELDS`) on purpose: an ``extra`` key that is
        not on it lands in the event's ``detail`` alone, so a typo cannot look like
        it was recorded. Each of these becomes true exactly when a transition does
        — the provider reference arrives WITH ``queued → provisioning`` (that
        transition IS "submit() returned a non-empty ref") and the artifacts and
        checkpoint arrive with the drain that produced them — so they are written
        by the same writer, at the same moment, rather than by a second one.
        """
        for name in RECORD_FIELDS:
            if name not in extra:
                continue
            if name == "exit_code":
                self.exit_code = _as_optional_int(extra[name])
            elif name in ("artifacts", "checkpoint"):
                setattr(self, name, _as_str_map(extra[name]))
            else:
                setattr(self, name, _as_str(extra[name]))

    @property
    def member_state(self) -> MemberLifecycle | None:
        """This task as the mesh's four-value lifecycle, if it has a member row.

        ``None`` is not "unknown": it is "there is no member row", which is the
        answer for a task that was never submitted or never reached ``running``
        (``member_lifecycle_for``).
        """
        return member_lifecycle_for(self.state, ever_ran=self.started_at is not None)

    @property
    def terminal(self) -> bool:
        """Whether nothing more can happen to this record."""
        return self.state in TERMINAL_STATES

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "request_id": self.request_id,
            "tier": self.tier,
            "provider": self.provider,
            "provider_ref": self.provider_ref,
            "state": self.state,
            "grant_id": self.grant_id,
            "size": self.size,
            "spot": self.spot,
            "spot_fallback": self.spot_fallback,
            "created_at": self.created_at,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "deadline_at": self.deadline_at,
            "expires_at": self.expires_at,
            "last_seen_at": self.last_seen_at,
            "exit_code": self.exit_code,
            "stop_reason": self.stop_reason,
            "budget_usd": self.budget_usd,
            "repo_url": self.repo_url,
            "repo_sha": self.repo_sha,
            "branch": self.branch,
            "session_id": self.session_id,
            "artifacts": dict(self.artifacts),
            "checkpoint": dict(self.checkpoint),
            # Copied entry by entry: the journal is append-only, and a reader that
            # could hold a reference into it could edit history.
            "events": [
                {
                    "at": _as_float(event.get("at")),
                    "phase": _as_str(event.get("phase")),
                    "detail": dict(event.get("detail") or {}),
                }
                for event in self.events
                if isinstance(event, dict)
            ],
        }

    @staticmethod
    def from_json(data: Any) -> ComputeTask:
        """Parse a task record, leniently about absence and strict about unknowns."""
        if not isinstance(data, dict):
            _refuse("version_unknown", "cloud: a compute task must be a JSON object")
        version = _read_version(data, readable=COMPUTE_TASK_VERSIONS, what="compute task")
        state = _read_state(data.get("state"))
        return ComputeTask(
            version=version,
            request_id=_as_str(data.get("request_id")),
            tier=_read_tier(data.get("tier")),
            provider=_as_str(data.get("provider")),
            provider_ref=_as_str(data.get("provider_ref")),
            state=state,
            grant_id=_as_str(data.get("grant_id")),
            size=_read_size(data.get("size")),
            spot=_as_bool(data.get("spot")),
            spot_fallback=_as_bool(data.get("spot_fallback")),
            created_at=_as_float(data.get("created_at")),
            submitted_at=_as_optional_float(data.get("submitted_at")),
            started_at=_as_optional_float(data.get("started_at")),
            stopped_at=_as_optional_float(data.get("stopped_at")),
            deadline_at=_as_optional_float(data.get("deadline_at")),
            expires_at=_as_optional_float(data.get("expires_at")),
            last_seen_at=_as_optional_float(data.get("last_seen_at")),
            exit_code=_as_optional_int(data.get("exit_code")),
            stop_reason=_as_str(data.get("stop_reason")),
            budget_usd=_as_float(data.get("budget_usd")),
            repo_url=_as_str(data.get("repo_url")),
            repo_sha=_as_str(data.get("repo_sha")),
            branch=_as_str(data.get("branch")),
            session_id=_as_str(data.get("session_id")),
            artifacts=_as_str_map(data.get("artifacts")),
            checkpoint=_as_str_map(data.get("checkpoint")),
            events=[
                {
                    "at": _as_float(event.get("at")),
                    "phase": _as_str(event.get("phase")),
                    "detail": dict(event.get("detail") or {}),
                }
                for event in (data.get("events") or [])
                if isinstance(event, dict)
            ],
        )


def member_lifecycle_for(state: str, *, ever_ran: bool = False) -> MemberLifecycle | None:
    """``state`` as the mesh's four-value ``MemberLifecycle``, or ``None``.

    The mesh vocabulary has four states and this one has eight (§3.4), so the
    three collapse states are decided by whether the task EVER reached
    ``running``: a task that never ran never produced a member row, so there is
    nothing to expire and the answer is ``None`` rather than a lifecycle a reader
    would read as "there was a member and it is gone".

    The REASON distinguishes the collapse states
    (``pool_member_expired {reason: drained | lost | revoked}`` is already the
    design's vocabulary), which is why this function does not try to.
    """
    if state == "queued":
        return None
    if state == "provisioning":
        return "provisioning"
    if state == "running":
        return "active"
    if state == "draining":
        return "draining"
    if state == "expired":
        return "expired"
    if state in TERMINAL_STATES:
        return "expired" if ever_ran else None
    raise ValueError(f"unknown task state {state!r}")
