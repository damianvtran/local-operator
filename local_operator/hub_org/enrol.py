"""Auto-enrol: pull an organization's new agents, skills and teams without being asked.

WHY. The operator wants agents to "pull the org's latest teams, agents and
skills and subscribe automatically to stay up to date". Updates to items this
machine already links are the existing hub runner's job (its subscription
policy, ``hub.auto_update.*``); what nothing did before is notice an item the
organization ADDED. This module is that one missing step, and nothing else —
there is deliberately no second update path and no new poller.

WHEN. :func:`on_boundary` runs at the boundaries the eligibility gate already
uses — sign-in, daemon start and the hub runner's tick — and only for a
Radient Team member (:mod:`.eligibility`). Per organization the catalogue is
listed at most once per :data:`CATALOGUE_FLOOR_SECONDS` (three list GETs; the
list routes carry no ETag, so the floor is the cost bound). A machine with zero
links still enrols: the runner calls this BEFORE its ``no-linked-items`` early
return.

SAFETY. Automatic runs use ``on_conflict="adopt-link"`` and ``auto_push=False``.
An unlinked local copy sharing a name with an org item is backed up and LINKED
(subscribed), never silently overwritten: identical content needs nothing
more, differing content goes through the existing reconcile path, and when no
merge is possible the local content stays linked so drift shows in
``skills sync --check``. That closes the failure the operator saw — skills on
disk with no hub link drifting where no check could see them. What was adopted
is recorded under ``last_enrol`` in the gate record so ``lop hub org`` can show
it. Hub-wins adopt stays an explicit ``lop hub org pull`` choice. A brand-new organization — one
absent from ``seen`` — is a full first enrol. ``seen`` advances only for items
that did not fail, so a transient failure retries at the next floor.
Single-flight under the gate's lock: a loser skips. Every entry point is
never-raising and blocking work stays off any event loop.

CONTRACT WITH SLICE A. ``list_catalogue`` (``hub_org.catalogue``) and
``pull_all`` (``hub_org.pull``) are imported lazily and GUARDED: until they
exist on this install an enrol is a no-op, never an error.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from local_operator.hub_org import eligibility as el
from local_operator.hub_org.settings import auto_enrol_enabled

logger = logging.getLogger(__name__)

#: The per-organization catalogue listing floor.
CATALOGUE_FLOOR_SECONDS = 60 * 60
#: Order matters to the pull layer (team rosters name agents); listed as such.
KINDS: tuple[str, ...] = ("agent", "skill", "team")
#: Outcomes that must NOT mark an item seen, so the next floor retries it.
_RETRY_OUTCOMES = frozenset({"failed"})


def _slice_a() -> tuple[Any, Any] | None:
    """``(list_catalogue, pull_all)``, or ``None`` before slice A is installed."""

    # Slice A's modules, resolved by name so this file type-checks and imports
    # on a branch/install that predates them (an absent module = no-op enrol).
    import importlib

    try:
        list_catalogue = importlib.import_module("local_operator.hub_org.catalogue").list_catalogue
        pull_all = importlib.import_module("local_operator.hub_org.pull").pull_all
    except (ImportError, AttributeError):
        return None
    return list_catalogue, pull_all


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _adopted_rows(report: Any) -> list[dict[str, Any]]:
    """The report's ``adopted`` entries as plain JSON rows (name, kind, hub_id, result, backup)."""

    rows: list[dict[str, Any]] = []
    for entry in getattr(report, "adopted", None) or []:
        rows.append(
            {
                key: (str(value) if value is not None else None)
                for key in ("name", "kind", "hub_id", "result", "backup")
                for value in (_field(entry, key),)
            }
        )
    return rows


def last_enrol(config_dir: Path) -> dict[str, Any]:
    """``tenant -> {at, pulled, adopted}`` from the last automatic runs (``lop hub org``)."""

    record = el.read_record(Path(config_dir)) or {}
    value = record.get("last_enrol")
    return value if isinstance(value, dict) else {}


def on_boundary(
    config_dir: Path, boundary: str, *, force: bool = False, background: bool = True
) -> threading.Thread | None:
    """Refresh the gate, then enrol new items. Never raises.

    ``background`` (the default) runs the whole thing on a daemon thread and
    returns it, so a login route or daemon lifespan never waits on the network;
    ``False`` runs inline (the hub runner, which is already on a worker).
    """

    if not background:
        run_boundary(config_dir, boundary, force=force)
        return None
    try:
        thread = threading.Thread(
            target=run_boundary,
            args=(Path(config_dir), boundary),
            kwargs={"force": force},
            name=f"hub-org-enrol-{boundary}",
            daemon=True,
        )
        thread.start()
        return thread
    except Exception:  # noqa: BLE001 - a thread that cannot start costs only the enrol
        logger.debug("org hub enrol thread could not start (%s)", boundary, exc_info=True)
        return None


def run_boundary(config_dir: Path, boundary: str, *, force: bool = False) -> None:
    """The boundary body: gate refresh, then :func:`sync_new_items`. Never raises."""

    try:
        orgs = el.refresh(Path(config_dir), boundary=boundary, force=force)
        if orgs and auto_enrol_enabled(Path(config_dir)):
            sync_new_items(Path(config_dir), orgs)
    except Exception:  # noqa: BLE001 - see module docstring
        logger.debug("org hub enrol failed (%s)", boundary, exc_info=True)


def sync_new_items(config_dir: Path, orgs: Iterable[el.OrgRef] | None = None) -> None:
    """Pull every catalogue item absent from ``seen``, per org past its floor."""

    config_dir = Path(config_dir)
    orgs = list(el.eligible_orgs(config_dir) if orgs is None else orgs)
    if not orgs:
        return
    seam = _slice_a()
    if seam is None:
        return
    list_catalogue, pull_all = seam
    with el.refresh_lock(config_dir) as held:
        if not held:
            return
        record = el.read_record(config_dir) or {}
        due = record.get("next_catalogue_at") or {}
        now = time.time()
        for org in orgs:
            stamp = due.get(org.tenant_id) if isinstance(due, dict) else None
            if isinstance(stamp, (int, float)) and stamp > now:
                continue
            try:
                _enrol_org(config_dir, org, record, list_catalogue, pull_all)
            except Exception as exc:  # noqa: BLE001 - one org never stops the next
                if getattr(exc, "code", None) == "team_plan_required" or (
                    getattr(exc, "status_code", None) == 403
                ):
                    # The server just said this account lost its team plan:
                    # close the gate now instead of after the 6 h TTL. Inline
                    # (not ``refresh``): this thread already holds the lock.
                    el._refresh_locked(config_dir, boundary="team-plan-required")
                    return
                logger.debug("org hub enrol failed for %s", org.tenant_id, exc_info=True)


def _enrol_org(
    config_dir: Path,
    org: el.OrgRef,
    record: dict[str, Any],
    list_catalogue: Any,
    pull_all: Any,
) -> None:
    resolved = el.person_client(config_dir)
    if resolved is None:
        return
    client, _account = resolved
    items = list(list_catalogue(client, org.tenant_id, KINDS))
    seen_all = record.get("seen")
    seen = seen_all.get(org.tenant_id) if isinstance(seen_all, dict) else None
    seen = seen if isinstance(seen, dict) else {}

    def _is_seen(item: Any) -> bool:
        ids = seen.get(_field(item, "kind"))
        return isinstance(ids, list) and _field(item, "hub_id") in ids

    # Linked items belong to the runner (its subscription keeps them current),
    # so only ABSENT items and unlinked same-name CONFLICTS are handed to the
    # pull, the latter to be adopt-linked (see the module docstring).
    wanted = [i for i in items if not _is_seen(i) and _field(i, "local") in ("absent", "conflict")]
    retry: set[tuple[Any, Any]] = set()
    adopted: list[dict[str, Any]] = []
    if wanted:
        from local_operator.config import ConfigManager  # lazy: YAML stack

        report = pull_all(
            ConfigManager(config_dir),
            org.tenant_id,
            kinds=KINDS,
            names=[_field(i, "hub_id") for i in wanted],
            subscribe=True,
            auto_push=False,
            on_conflict="adopt-link",
            allow_llm=True,
        )
        for outcome in getattr(report, "items", None) or []:
            if _field(outcome, "outcome") in _RETRY_OUTCOMES:
                retry.add((_field(outcome, "kind"), _field(outcome, "hub_id")))
        adopted = _adopted_rows(report)

    next_seen: dict[str, list[str]] = {k: list(v) for k, v in seen.items() if isinstance(v, list)}
    for item in items:
        key = (_field(item, "kind"), _field(item, "hub_id"))
        if key in retry or not isinstance(key[0], str) or not isinstance(key[1], str):
            continue
        bucket = next_seen.setdefault(key[0], [])
        if key[1] not in bucket:
            bucket.append(key[1])
    floor = time.time() + CATALOGUE_FLOOR_SECONDS

    def _write(data: dict[str, Any]) -> None:
        data.setdefault("seen", {})[org.tenant_id] = next_seen
        data.setdefault("next_catalogue_at", {})[org.tenant_id] = floor
        if wanted:
            # The last automatic run's adoptions, per org, for ``lop hub org``.
            # Replaced (not appended) each run so the record stays bounded.
            data.setdefault("last_enrol", {})[org.tenant_id] = {
                "at": time.time(),
                "pulled": len(wanted) - len(retry) - len(adopted),
                "adopted": adopted,
            }

    written = el.update_record(config_dir, _write)
    record.update(seen=written.get("seen"), next_catalogue_at=written.get("next_catalogue_at"))
    if wanted:
        logger.info(
            "org hub: enrolled %d new item(s) from %s", len(wanted) - len(retry), org.tenant_name
        )
