"""Whether the organization Agent Hub may be shown at all: the Radient Team gate.

WHY THIS EXISTS. The organization (private) Agent Hub is a Radient Team
feature. The operator's constraint is explicit: it must not be advertised to
anyone who is not signed in to Radient, or whose Radient account belongs to no
team-plan organization. Every org surface — the ``org-hub`` guide, the
recommendation trigger, the ``lop hub org`` help, the agent-tool ops, the
settings section, auto-enrol — asks THIS module, and nothing else, whether the
gate is open. One answer, one rule.

THE RULE. An organization is eligible when the membership is ``active`` AND the
tenant plan's status is ``active`` or ``past_due``. That is byte-for-byte the
server's own entitlement predicate (agent-server ``planStatusEntitlesOrgAccess``)
so the client can never show a catalogue the server would refuse. An owner of a
no-plan organization is deliberately NOT eligible: the server lets owners read,
but the operator's constraint is "team plan". The Team Bridge availability
record is NOT reused: it is process-local (only the daemon writes it) and it
ANDs a deployment scope and kill switch that must not hide the hub.

ZERO NETWORK ON THE READ PATH. :func:`eligible_orgs`/:func:`is_eligible` run on
every session build and settings render, so they only ever read one on-disk
record, ``<config>/hub/org_catalogue.json`` (0600, atomic, never a token),
memoised on the record's and ``config.yml``'s stat. :func:`refresh` is the only
network path; it runs at boundaries — sign-in, daemon/TUI start, the hub
runner's periodic tick (6 h TTL), an explicit ``lop hub org`` (``force``) and a
server ``403 team_plan_required`` — never per message.

FAIL CLOSED. The reader returns ``[]`` for no record, a record written for a
different Radient base, a record older than :data:`MAX_AGE_SECONDS`, or a
corrupt one. A wrong absence costs one refresh; a wrong presence is the up-sell
the constraint forbids and nothing can undo on screen.

The same record also carries the auto-enrol bookkeeping (``seen``,
``next_catalogue_at``, ``last_enrol``) owned by :mod:`local_operator.hub_org.enrol`; writers go
through :func:`update_record` under :func:`refresh_lock` so neither side drops
the other's keys.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

logger = logging.getLogger(__name__)

SCHEMA = 1
RECORD_NAME = "org_catalogue.json"
LOCK_NAME = "org_catalogue.lock"

#: A record older than this reads as closed: a plan lapse must not keep the
#: hint up indefinitely on a machine that never reaches a refresh boundary.
MAX_AGE_SECONDS = 7 * 24 * 3600
#: The periodic (non-forced) refresh cadence. A lapse shows the hint for at most
#: this long, and any server ``403 team_plan_required`` closes it sooner.
REFRESH_TTL_SECONDS = 6 * 3600
#: After a failed refresh, non-forced callers wait this long before retrying so
#: an outage does not turn every boundary into a network call.
ERROR_BACKOFF_SECONDS = 30 * 60

#: Membership/plan states that entitle org access (mirrors the server).
_ENTITLED_PLAN_STATUSES = frozenset({"active", "past_due"})
_ROLES = frozenset({"owner", "admin", "member"})


@dataclass(frozen=True)
class OrgRef:
    """One eligible organization, as the catalogue and pull layers address it."""

    tenant_id: str
    tenant_name: str
    role: str  # owner|admin|member


def record_path(config_dir: Path) -> Path:
    return Path(config_dir) / "hub" / RECORD_NAME


def _lock_path(config_dir: Path) -> Path:
    return Path(config_dir) / "hub" / LOCK_NAME


def membership_entitles(membership: Mapping[str, Any]) -> bool:
    """The server's entitlement predicate for one ``/me/memberships`` entry."""

    if not isinstance(membership, Mapping):
        return False
    if membership.get("status") != "active":
        return False
    plan = membership.get("plan")
    if not isinstance(plan, Mapping):
        return False
    return plan.get("status") in _ENTITLED_PLAN_STATUSES


def _configured_base_url(config_dir: Path) -> str:
    """The Radient base this config root resolves to (the record's binding).

    The same rule as ``configured_radient_base_url`` (configured value, else
    ``RADIENT_API_BASE_URL``/canonical default), but read through
    ``read_config_values`` rather than a ``ConfigManager``: this runs from the
    hub runner's timer and every session build, and constructing a manager
    over a torn ``config.yml`` moves it aside — a side effect a gate read must
    never have. An unreadable file raises, which every caller turns into
    "closed".
    """

    from local_operator.config import read_config_values  # lazy: YAML stack
    from local_operator.env import resolve_radient_api_base_url

    values = read_config_values(Path(config_dir))
    if values is None:
        raise OSError("config.yml is unreadable right now")
    configured = values.get("radient_base_url")
    return resolve_radient_api_base_url(configured if isinstance(configured, str) else None)


def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


# ``config root -> (memo key, parsed orgs, checked_at)``. The key folds in the
# record's stat, config.yml's stat and RADIENT_API_BASE_URL, so a base change or
# a rewrite invalidates without a clock; age is re-checked on every read.
_MEMO: dict[str, tuple[tuple[Any, ...], tuple[OrgRef, ...], float]] = {}
_MEMO_LOCK = threading.Lock()


def read_record(config_dir: Path) -> dict[str, Any] | None:
    """The raw record, or ``None`` when absent or unreadable. Never raises."""

    path = record_path(config_dir)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError, UnicodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    return data


def _parse_orgs(data: Mapping[str, Any]) -> tuple[OrgRef, ...] | None:
    orgs = data.get("orgs")
    if not isinstance(orgs, list):
        return None
    out: list[OrgRef] = []
    for entry in orgs:
        if not isinstance(entry, Mapping):
            return None
        tenant_id = entry.get("tenant_id")
        name = entry.get("tenant_name")
        role = entry.get("role")
        if not isinstance(tenant_id, str) or not tenant_id:
            return None
        out.append(
            OrgRef(
                tenant_id=tenant_id,
                tenant_name=name if isinstance(name, str) and name else tenant_id,
                role=role if role in _ROLES else "member",
            )
        )
    return tuple(out)


def eligible_orgs(config_dir: Path) -> list[OrgRef]:
    """The eligible organizations from the cached record; ``[]`` = gate closed.

    Zero network, never raises. See the module docstring for the fail-closed
    cases.
    """

    try:
        return list(_eligible_orgs(Path(config_dir)))
    except Exception:  # noqa: BLE001 - a gate read must never break a caller
        logger.debug("org hub eligibility read failed", exc_info=True)
        return []


def _eligible_orgs(config_dir: Path) -> tuple[OrgRef, ...]:
    path = record_path(config_dir)
    record_stat = _stat_key(path)
    if record_stat is None:
        return ()
    key = (
        record_stat,
        _stat_key(config_dir / "config.yml"),
        os.environ.get("RADIENT_API_BASE_URL", ""),
    )
    memo_key = str(config_dir)
    with _MEMO_LOCK:
        cached = _MEMO.get(memo_key)
    if cached is not None and cached[0] == key:
        _, orgs, checked_at = cached
    else:
        data = read_record(config_dir)
        orgs, checked_at = (), 0.0
        if data is not None:
            parsed = _parse_orgs(data)
            stamp = data.get("checked_at")
            if (
                parsed is not None
                and isinstance(stamp, (int, float))
                and data.get("base_url") == _configured_base_url(config_dir)
            ):
                orgs, checked_at = parsed, float(stamp)
        with _MEMO_LOCK:
            _MEMO[memo_key] = (key, orgs, checked_at)
    if not orgs or time.time() - checked_at > MAX_AGE_SECONDS:
        return ()
    return orgs


def is_eligible(config_dir: Path) -> bool:
    """Whether any org surface may render for this config root. Zero network."""

    return bool(eligible_orgs(config_dir))


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    """0600 temp file + ``os.replace``; the record never holds a token."""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def refresh_lock(config_dir: Path) -> Iterator[bool]:
    """Single-flight, non-blocking: yields ``True`` to the holder, ``False`` to a loser.

    A loser skips its refresh/enrol rather than queueing — the holder is already
    doing the same work, and a TUI start must never wait on a daemon's pull.
    """

    path = _lock_path(config_dir)
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        yield False
        return
    try:
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # Windows: no flock; best-effort, still serial per process
            pass
        except OSError:
            yield False
            return
        yield True
    finally:
        os.close(fd)


def update_record(config_dir: Path, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Read-modify-write the record, preserving keys the caller does not touch.

    Callers hold :func:`refresh_lock`. A missing or corrupt record starts from
    an empty, closed one (no orgs), so a mutation can never open the gate by
    accident.
    """

    data = read_record(config_dir) or {}
    data.setdefault("orgs", [])
    data.setdefault("seen", {})
    data.setdefault("next_catalogue_at", {})
    data["schema"] = SCHEMA
    mutate(data)
    _atomic_write(record_path(config_dir), data)
    return data


def _api_error_status(exc: BaseException) -> tuple[int | None, str | None]:
    return getattr(exc, "status_code", None), getattr(exc, "code", None)


def refresh(config_dir: Path, *, boundary: str, force: bool = False) -> list[OrgRef]:
    """Re-read ``/me/memberships`` and rewrite the record. Network; never raises.

    ``boundary`` names the trigger for the debug log (``sign-in``, ``daemon-start``,
    ``tui-start``, ``timer``, ``explicit``, ``team-plan-required``). Without
    ``force`` a fresh record (younger than :data:`REFRESH_TTL_SECONDS`, or a
    failure inside :data:`ERROR_BACKOFF_SECONDS`) is returned as-is. A lock loser
    returns the cached answer. Signed out — or a destination that may not
    receive the person's bearer — writes an empty (closed) record. A failed
    call keeps the previous orgs and stamps ``last_error_at``, except a
    ``401``/``403`` which closes the gate: the server has just said no.
    """

    config_dir = Path(config_dir)
    try:
        return _refresh(config_dir, boundary=boundary, force=force)
    except Exception:  # noqa: BLE001 - boundaries must never fail on the gate
        logger.debug("org hub eligibility refresh failed (%s)", boundary, exc_info=True)
        return eligible_orgs(config_dir)


def _refresh(config_dir: Path, *, boundary: str, force: bool) -> list[OrgRef]:
    if not force:
        data = read_record(config_dir)
        if data is not None:
            now = time.time()
            checked = data.get("checked_at")
            failed = data.get("last_error_at")
            if isinstance(checked, (int, float)) and now - checked < REFRESH_TTL_SECONDS:
                if data.get("base_url") == _configured_base_url(config_dir):
                    return eligible_orgs(config_dir)
            if isinstance(failed, (int, float)) and now - failed < ERROR_BACKOFF_SECONDS:
                return eligible_orgs(config_dir)
    with refresh_lock(config_dir) as held:
        if not held:
            return eligible_orgs(config_dir)
        return _refresh_locked(config_dir, boundary=boundary)


def person_client(config_dir: Path, base_url: str | None = None) -> tuple[Any, str | None] | None:
    """``(RadientClient, account_id)`` acting as the signed-in PERSON, or ``None``.

    Org calls authenticate with the stored Radient OAuth access token, never the
    tenant API key (an API key proves an application tenant, not a person's
    membership). The resolver also refuses a destination that may not receive
    the person's bearer, which reads as ``None`` here exactly like a sign-out.
    Runs the async resolver on a private loop so it works from a worker thread
    inside a host whose own loop is running; never call it ON a running loop.
    """

    import asyncio

    from pydantic import SecretStr  # lazy: pydantic stays off the startup path

    from local_operator.clients.radient import RadientClient  # lazy: HTTP stack
    from local_operator.providers.radient_credentials import (
        resolve_radient_oauth_access,
    )

    base_url = base_url or _configured_base_url(config_dir)
    access = asyncio.run(resolve_radient_oauth_access(config_dir, base_url))
    if access is None:
        return None
    client = RadientClient(api_key=SecretStr(access.access_token), base_url=base_url)
    # The account id lets the record tell an account switch (reset the enrol
    # bookkeeping) from a re-sign-in of the same account (keep it).
    return client, access.account_id


def _fetch_memberships(config_dir: Path, base_url: str) -> tuple[str | None, list[Any]] | None:
    """``(account_id, memberships)`` for the signed-in person, or ``None`` signed out."""

    resolved = person_client(config_dir, base_url)
    if resolved is None:
        return None
    client, account_id = resolved
    return account_id, client.list_memberships()


def _refresh_locked(config_dir: Path, *, boundary: str) -> list[OrgRef]:
    base_url = _configured_base_url(config_dir)
    now = time.time()
    try:
        fetched = _fetch_memberships(config_dir, base_url)
    except Exception as exc:  # noqa: BLE001 - classified below, never re-raised
        status, _code = _api_error_status(exc)
        if status in (401, 403):
            # The hub refused this person outright: close the gate now.
            def _close(data: dict[str, Any]) -> None:
                data.update(base_url=base_url, checked_at=now, orgs=[], last_error_at=None)

            update_record(config_dir, _close)
        else:

            def _mark_failed(data: dict[str, Any]) -> None:
                data["last_error_at"] = now

            update_record(config_dir, _mark_failed)
        logger.debug("org hub memberships read failed (%s)", boundary, exc_info=True)
        return eligible_orgs(config_dir)

    account_id: str | None = None
    orgs: list[dict[str, str]] = []
    if fetched is not None:
        account_id, memberships = fetched
        for membership in memberships:
            if not membership_entitles(membership):
                continue
            tenant_id = membership.get("tenant_id")
            if not isinstance(tenant_id, str) or not tenant_id:
                continue
            role = membership.get("role")
            orgs.append(
                {
                    "tenant_id": tenant_id,
                    "tenant_name": str(membership.get("tenant_name") or tenant_id),
                    "role": role if role in _ROLES else "member",
                }
            )

    def _write(data: dict[str, Any]) -> None:
        if data.get("account_id") != account_id:
            # A different account (or a sign-out) must not inherit the previous
            # account's enrol bookkeeping.
            data["seen"] = {}
            data["next_catalogue_at"] = {}
            data["last_enrol"] = {}
        data.update(
            account_id=account_id,
            base_url=base_url,
            checked_at=now,
            orgs=orgs,
            last_error_at=None,
        )

    update_record(config_dir, _write)
    logger.debug("org hub eligibility refreshed (%s): %d org(s)", boundary, len(orgs))
    return eligible_orgs(config_dir)


def clear(config_dir: Path) -> None:
    """Forget the record (sign-out). Never raises; the gate reads closed after."""

    with contextlib.suppress(OSError):
        record_path(Path(config_dir)).unlink()
    with _MEMO_LOCK:
        _MEMO.pop(str(Path(config_dir)), None)


def reset_memo() -> None:
    """Drop the in-process memo (tests)."""

    with _MEMO_LOCK:
        _MEMO.clear()
