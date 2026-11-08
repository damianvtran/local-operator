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

SAFETY. Automatic runs use ``on_conflict="adopt-link"``, ``allow_llm=False``
and ``auto_push=False``. An unlinked local copy sharing a name with an org
item is backed up and LINKED (subscribed) with its content byte-unchanged:
identical content reads up-to-date, differing content is
``kept-local-linked`` (the hub revision is its baseline, so the drift shows as
local-only in ``sync --check``). An automatic run never merges, never calls a
model (no tokens spent unasked) and never writes authored bytes; merging is the
explicit ``lop hub org pull --on-conflict reconcile``, hub-wins the explicit
``--on-conflict adopt``. A local item that already follows a different hub
item (``linked-elsewhere``) is left alone. That closes the failure the operator
saw — skills on disk with no hub link drifting where no check could see them.
What was adopted is recorded under ``last_enrol`` in the gate record so
``lop hub org`` can show it. A brand-new organization — one
absent from ``seen`` — is a full first enrol. ``seen`` advances only for items
that did not fail, so a transient failure retries at the next floor.
Single-flight under the gate's lock: a loser skips. Every entry point is
never-raising and blocking work stays off any event loop.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from local_operator.hub_org import eligibility as el
from local_operator.hub_org.catalogue import list_catalogue
from local_operator.hub_org.pull import pull_all
from local_operator.hub_org.settings import auto_enrol_enabled

logger = logging.getLogger(__name__)

#: The per-organization catalogue listing floor.
CATALOGUE_FLOOR_SECONDS = 60 * 60
#: Order matters to the pull layer (team rosters name agents); listed as such.
KINDS: tuple[str, ...] = ("agent", "skill", "team")
#: Outcomes that must NOT mark an item seen, so the next floor retries it.
_RETRY_OUTCOMES = frozenset({"failed"})


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


class BoundaryThread(threading.Thread):
    """The background boundary, with an event set once the GATE refresh is done.

    WHY. A CLI-only ``lop login radient`` returns as soon as the credential is
    committed, and interpreter exit kills a daemon thread, so the gate used to
    open only at the next TUI/daemon start (agent review R3). The login caller
    waits on :attr:`gate_refreshed` for a short bound -- the memberships read
    only, never the catalogue pull that follows, which stays in the background.
    """

    def __init__(self, config_dir: Path, boundary: str, *, force: bool) -> None:
        super().__init__(name=f"hub-org-enrol-{boundary}", daemon=True)
        self.gate_refreshed = threading.Event()
        self._args = (Path(config_dir), boundary, force)

    def run(self) -> None:
        config_dir, boundary, force = self._args
        run_boundary(config_dir, boundary, force=force, refreshed=self.gate_refreshed)


def on_boundary(
    config_dir: Path, boundary: str, *, force: bool = False, background: bool = True
) -> BoundaryThread | None:
    """Refresh the gate, then enrol new items. Never raises.

    ``background`` (the default) runs the whole thing on a daemon thread and
    returns it, so a login route or daemon lifespan never waits on the network;
    ``False`` runs inline (the hub runner, which is already on a worker). A
    caller that needs the gate answer before it exits waits on the thread's
    ``gate_refreshed`` event (:class:`BoundaryThread`).
    """

    if not background:
        run_boundary(config_dir, boundary, force=force)
        return None
    try:
        thread = BoundaryThread(Path(config_dir), boundary, force=force)
        thread.start()
        return thread
    except Exception:  # noqa: BLE001 - a thread that cannot start costs only the enrol
        logger.debug("org hub enrol thread could not start (%s)", boundary, exc_info=True)
        return None


def run_boundary(
    config_dir: Path,
    boundary: str,
    *,
    force: bool = False,
    refreshed: threading.Event | None = None,
) -> None:
    """The boundary body: gate refresh, then :func:`sync_new_items`. Never raises.

    ``refreshed`` is set as soon as the gate refresh has finished (or failed),
    before the enrol starts.
    """

    try:
        try:
            orgs = el.refresh(Path(config_dir), boundary=boundary, force=force)
        finally:
            if refreshed is not None:
                refreshed.set()
        if orgs:
            # The Radient Vault agent gate rides the same boundaries (sign-in,
            # daemon start, the 6 h timer): it is meaningful only for a
            # team-plan account, and its own TTL keeps this a no-op most ticks.
            _refresh_vault_gate(Path(config_dir), boundary, force=force)
        if orgs and auto_enrol_enabled(Path(config_dir)):
            sync_new_items(Path(config_dir), orgs)
    except Exception:  # noqa: BLE001 - see module docstring
        logger.debug("org hub enrol failed (%s)", boundary, exc_info=True)


def _refresh_vault_gate(config_dir: Path, boundary: str, *, force: bool) -> None:
    """Refresh ``vault_agent.eligibility`` at this boundary. Never raises."""

    try:
        from local_operator.vault_agent import eligibility as vault_gate

        vault_gate.refresh(config_dir, boundary=boundary, force=force)
    except Exception:  # noqa: BLE001 - a gate refresh never fails a boundary
        logger.debug("vault agent gate refresh failed (%s)", boundary, exc_info=True)


def sync_new_items(config_dir: Path, orgs: Iterable[el.OrgRef] | None = None) -> None:
    """Pull every catalogue item absent from ``seen``, per org past its floor."""

    config_dir = Path(config_dir)
    orgs = list(el.eligible_orgs(config_dir) if orgs is None else orgs)
    if not orgs:
        return
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
                _enrol_org(config_dir, org, record)
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
    # pull, the latter to be adopt-linked (see the module docstring). A
    # ``linked-elsewhere`` row follows another hub item and is never handed over.
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
            # Unattended: no model call, no tokens spent without consent (R4).
            allow_llm=False,
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
