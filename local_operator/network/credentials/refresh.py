"""The NODE's copy of an owner-signed refresh delegation, and its epoch ledger.

WHY THIS IS A SEPARATE DOCUMENT AND NOT A PLACEMENT FIELD. A placement entry says
who may *borrow a bearer*; this says that one named node may *present the refresh
token* — the one secret the broker protocol is built never to move
(``mesh-credentials.md`` §3.3). Keeping it apart is what lets the deny watermark
(``held_policy``), the placement document and this one each stay a single
concern, and it is why ``placement.py`` does not grow a second custody rule
(design §3.9).

WHAT IT MUST BE ABLE TO SAY, and why each answer is a refusal rather than a
guess (design §3.4/§3.5):

* ``active`` — a statement that verifies against the operator anchor right now,
  names THIS device and THIS key, is inside its window, and has budget left.
  Anything else is ``None``, and the caller must not POST.
* unprovable — a missing, corrupt, uninspectable or wrong-mode file raises
  :class:`RefreshUnknown`. That is the same rule as ``held_policy.PolicyUnknown``:
  missing evidence is never permission.
* refused — a revocation (or a failed family) is written into this document and
  the material is wiped. It is written BEFORE the report is published, so a
  crash cannot leave a "still active" record behind the deny.

THE VERIFICATION RUNS ON EVERY READ, not once at install time. A statement is
re-verified immediately before each POST because that is the only way
"before every POST, not merely at grant time" (§3.4) can be a property of the
code rather than of a caller's diligence; the cached copy is an optimisation
nobody is allowed to add, and the anchor is read through the one reader every
operator-signed surface uses (``operator.trust.load_local_anchor``).

THE EPOCH is a monotonic integer per family: incremented on every successful
refresh, recorded here, and carried by the handback so the owner can tell a
newer family state from an older one without ever comparing tokens (§3.3 Rule
B). :func:`handback_holds` is the whole comparison, in one place, because two
copies of it could disagree about which direction is "newer".
"""

from __future__ import annotations

import copy
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from local_operator.network import store
from local_operator.network.credentials import placement

#: The document's byte bound. A statement is at most
#: ``operator.verify.MAX_REFRESH_GRANT_BYTES``; the rest is bookkeeping, so this
#: is generous by construction rather than tuned. Mirrors ``held_policy.LIMIT``'s
#: role: a bound exists so a hostile file is refused before it is parsed.
LIMIT = 65536

#: The worst case number of refreshes a window can need, from the SHORTEST bearer
#: any provider in this tree mints (55 minutes, measured for Radient in the
#: store's own docstring). It is a CEILING for ``max_refreshes``, never a promise:
#: §6 item 5 says the provider's own expiry has not been measured per provider, so
#: a delegation that needs more than this must be re-minted rather than extended
#: (§7: a node never extends its own authority).
WORST_CASE_BEARER_S = 55 * 60.0
WORST_CASE_REFRESHES = 256


def max_refreshes_for(window_s: float) -> int:
    """A delegation's refresh budget for a window, clamped to the ceiling.

    Derived from the window rather than chosen: a 72 h window at the shortest
    measured bearer is ~79 refreshes, and anything the caller asks for beyond
    :data:`WORST_CASE_REFRESHES` is not a plan this design can honour.
    """
    if window_s <= 0:
        return 1
    return max(1, min(int(window_s / WORST_CASE_BEARER_S) + 2, WORST_CASE_REFRESHES))


SCHEMA = 1

DOC_FILENAME = "refresh.json"

#: The lock file beside the document. Dotted and never unlinked while it may be
#: contended — the same reasoning as ``placement.PLACEMENT_LOCK_FILENAME``.
LOCK_FILENAME = ".refresh.lock"

#: The states a document can be in. ``active`` is the only one that may POST;
#: every other value is a terminal answer until the owner mints again.
STATES = frozenset({"active", "refused", "returned", "failed"})

_CAUSES = frozenset(
    {
        "",
        "revoked",
        "expired",
        "returned",
        "family_moved",
        "grant_invalid",
        "already_delegated",
    }
)


class RefreshUnknown(RuntimeError):
    """Missing, corrupt or unprovable evidence about a delegation.

    The caller's rule is the same one ``PolicyUnknown`` carries: this is NOT
    permission. A node that cannot prove it is delegated does not refresh.
    """


class RefreshRefused(RuntimeError):
    """A decision, not an absence: the delegation exists and does not allow a POST."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code
        self.message = message or code


def refresh_path(network_id: str, root: Path | None = None) -> Path:
    """``<config>/network/credentials/<network_id>/refresh.json`` — never creates it.

    Deliberately the NON-CREATING spelling (``credentials_root``) for the read
    side: this path is asked on every resolve (``live_for``), and a read that
    mkdirs would leave a network directory on a device that never refreshed
    anything — ``placement.credentials_root``'s own argument.
    """
    store.safe_component(network_id)
    return placement.credentials_root(root) / network_id / DOC_FILENAME


def _lock_path(network_id: str, root: Path | None = None) -> Path:
    return refresh_path(network_id, root).with_name(LOCK_FILENAME)


def _checked_read(path: Path) -> dict[str, Any]:
    """Read one bounded 0600 document, refusing anything unprovable."""
    if not hasattr(path, "lstat"):
        raise RefreshUnknown("invalid refresh document path")
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise RefreshUnknown("no refresh delegation on this device") from exc
    except OSError as exc:
        raise RefreshUnknown("uninspectable refresh document") from exc
    placement._checked_info(before, directory=False)
    import os
    import stat

    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:  # a symlink, a directory, an unreadable file
        raise RefreshUnknown("unopenable refresh document") from exc
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        placement._checked_info(info, directory=False)
        if (
            not placement._same_node(before, info)
            or not placement._same_node(info, path.lstat())
            or (os.name != "nt" and stat.S_IMODE(info.st_mode) != 0o600)
        ):
            raise RefreshUnknown("unprovable refresh document")
        raw = stream.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise RefreshUnknown("refresh document bound exceeded")
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise RefreshUnknown("invalid refresh document") from exc
    if not isinstance(doc, dict):
        raise RefreshUnknown("invalid refresh document")
    return doc


def _validate(doc: dict[str, Any], *, network_id: str, recipient: str) -> None:
    if set(doc) != {
        "schema",
        "network_id",
        "recipient",
        "statement",
        "state",
        "epoch",
        "refreshes_used",
        "updated_at",
        "cause",
    }:
        raise RefreshUnknown("invalid refresh document schema")
    if (
        type(doc["schema"]) is not int
        or doc["schema"] != SCHEMA
        or doc["network_id"] != network_id
        or doc["recipient"] != recipient
        or not isinstance(doc["statement"], str)
        or not doc["statement"]
        or len(doc["statement"]) > 16384
    ):
        raise RefreshUnknown("invalid refresh document binding")
    if not isinstance(doc["state"], str) or doc["state"] not in STATES:
        raise RefreshUnknown("invalid refresh state")
    if not isinstance(doc["cause"], str) or doc["cause"] not in _CAUSES:
        raise RefreshUnknown("invalid refresh cause")
    for field in ("epoch", "refreshes_used"):
        if type(doc[field]) is not int or not 0 <= doc[field] < 2**31:
            raise RefreshUnknown("invalid refresh counter")
    if not isinstance(doc["updated_at"], (int, float)) or isinstance(doc["updated_at"], bool):
        raise RefreshUnknown("invalid refresh timestamp")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", network_id):
        raise RefreshUnknown("invalid refresh namespace")


def load(network_id: str, root: Path | None = None, *, recipient: str) -> dict[str, Any] | None:
    """The parsed document, or ``None`` when this device holds none.

    ``None`` is "no delegation" and is NOT a refusal: it is today's behaviour
    (§4.3). A file that exists but cannot be proven raises :class:`RefreshUnknown`,
    which every caller must treat as "do not POST".
    """
    path = refresh_path(network_id, root)
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RefreshUnknown("uninspectable refresh document") from exc
    doc = _checked_read(path)
    _validate(doc, network_id=network_id, recipient=recipient)
    return doc


def _publish(network_id: str, root: Path | None, doc: dict[str, Any]) -> None:
    # The WRITE path: only here is the network directory created (see
    # ``refresh_path``), and only into a 0700 root owned by this uid.
    path = placement.credentials_dir_for(network_id, root) / DOC_FILENAME
    store._write_private_json(path, doc)
    import os

    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if _checked_read(path) != doc:
        raise RefreshUnknown("refresh document write unconfirmed")


@contextmanager
def transaction(
    network_id: str,
    root: Path | None = None,
    *,
    recipient: str,
    create: Callable[[], dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any] | None]:
    """Read-modify-write under both locks; ``None`` when there is no document.

    ``create`` is the genesis factory, and it exists for one bug class: a caller
    that must WRITE and returns from inside the block closes this generator at its
    ``yield``, so the publish below never runs and the caller holds a document that
    exists only in memory. ``install`` is the writer-side caller; a caller that only
    reads (or that tolerates ``None``) passes no factory.

    The lock order is ``held_policy.transaction``'s (thread lock, then the
    ``flock`` beside the file) for the same reproduced reason: a CLI process and
    the relay are different processes, and an in-process lock serialises nothing
    between them.
    """
    path = refresh_path(network_id, root)
    with (
        store._write_lock(path),
        placement._cross_process_lock(
            path.with_name(LOCK_FILENAME), root=placement._config_root(root)
        ),
    ):
        current = load(network_id, root, recipient=recipient)
        existed = current is not None
        if current is None:
            if create is None:
                yield None
                return
            current = create()
        before = copy.deepcopy(current)
        yield current
        if current is None:
            return
        if current == before and existed:
            return
        _validate(current, network_id=network_id, recipient=recipient)
        current["updated_at"] = time.time()
        _publish(network_id, root, current)


def _anchor_spki() -> Any:
    """The operator anchor's public point, or ``None``. Never raises."""
    try:
        from local_operator.operator.trust import load_local_anchor

        anchor = load_local_anchor()
    except Exception:  # noqa: BLE001 — no readable anchor means nothing verifies
        return None
    return None if anchor is None else anchor.spki


def anchor_spki_for(root: Path | None) -> Any:
    """The anchor for an explicit config root — the seam a test drives."""
    if root is None:
        return _anchor_spki()
    try:
        from local_operator.operator.trust import load_local_anchor

        anchor = load_local_anchor(Path(root))
    except Exception:  # noqa: BLE001
        return None
    return None if anchor is None else anchor.spki


def install(
    network_id: str,
    root: Path | None,
    *,
    recipient: str,
    statement: str,
    operator_spki: Any = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Verify ``statement`` and adopt it as THIS device's delegation.

    Verification happens here as well as on every read: an install that trusted
    an unverified statement would make the read-time check the only fence, and a
    caller holding the lock between them could POST in the gap.
    """
    from local_operator.operator.verify import verify_refresh_grant

    spki = operator_spki if operator_spki is not None else anchor_spki_for(root)
    moment = time.time() if now is None else now
    grant = verify_refresh_grant(
        statement,
        operator_spki=spki,
        now=moment,
        network_id=network_id,
        node_device=recipient,
    )
    if grant is None or grant.relay_deny:
        raise RefreshRefused(
            "grant_invalid", "the delegation statement did not verify for this device"
        )

    def genesis() -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "network_id": network_id,
            "recipient": recipient,
            "statement": statement,
            "state": "active",
            "epoch": grant.family_epoch_at_issue,
            "refreshes_used": 0,
            "updated_at": moment,
            "cause": "",
        }

    with transaction(network_id, root, recipient=recipient, create=genesis) as doc:
        if doc is not None and doc["state"] == "active" and doc["statement"] != statement:
            # A REPLACEMENT IS ALLOWED and is how a higher epoch arrives (§3.5:
            # "only a fresh signed delegation at a higher epoch re-enables it"),
            # but never a LOWER one: a replayed older statement must not roll the
            # ledger back and re-open a window that has already been closed.
            current = verify_refresh_grant(
                doc["statement"],
                operator_spki=spki,
                now=moment,
                network_id=network_id,
                node_device=recipient,
            )
            if current is not None and current.family_epoch_at_issue > grant.family_epoch_at_issue:
                raise RefreshRefused(
                    "already_delegated",
                    "this device already holds a newer delegation for that login",
                )
        assert doc is not None  # `create` above guarantees it
        doc["statement"] = statement
        doc["state"] = "active"
        doc["epoch"] = max(doc["epoch"], grant.family_epoch_at_issue)
        doc["refreshes_used"] = 0
        doc["cause"] = ""
        installed = copy.deepcopy(doc)
    return installed


def active(
    network_id: str,
    root: Path | None,
    *,
    recipient: str,
    key: str,
    now: float | None = None,
    operator_spki: Any = None,
    require_key: bool = True,
) -> Any:
    """The verified, unexpired, still-budgeted grant, or ``None``. Never raises.

    ``None`` covers every "no" — no document, wrong key, expired window, spent
    budget, revoked. The caller's only rule is the one §3.4 states: no POST.
    :class:`RefreshUnknown` (unprovable evidence) propagates instead, because a
    caller that cannot tell must stop *and say why* rather than read "no".
    """
    from local_operator.operator.verify import verify_refresh_grant

    doc = load(network_id, root, recipient=recipient)
    if doc is None or doc["state"] != "active":
        return None
    spki = operator_spki if operator_spki is not None else anchor_spki_for(root)
    moment = time.time() if now is None else now
    grant = verify_refresh_grant(
        doc["statement"],
        operator_spki=spki,
        now=moment,
        network_id=network_id,
        node_device=recipient,
        key=key if require_key else "",
    )
    if grant is None or grant.relay_deny:
        return None
    if doc["refreshes_used"] >= grant.max_refreshes:
        return None
    return grant


def record_post(
    network_id: str,
    root: Path | None,
    *,
    recipient: str,
    epoch: int,
) -> None:
    """Persist the incremented epoch AFTER a successful POST (§3.3).

    Atomic by construction (staged write, ``fsync``, replace — the store's own
    ``_write_private_json``). A crash between the POST and this write leaves the
    store's send-marker armed, which is exactly the recovery the owner already
    has: the next attempt defers instead of re-presenting.
    """
    with transaction(network_id, root, recipient=recipient) as doc:
        if doc is None:
            return
        doc["epoch"] = max(doc["epoch"] + 1, int(epoch))
        doc["refreshes_used"] += 1


def stop(
    network_id: str,
    root: Path | None,
    *,
    recipient: str,
    state: str,
    cause: str,
) -> bool:
    """Write a terminal state. Returns whether a live document was stopped.

    The state is written BEFORE any wipe or report by every caller, so a crash
    can never leave a document that still reads ``active`` behind a decision
    that already stopped it.
    """
    if state not in STATES or state == "active":
        raise RefreshRefused("internal", f"{state!r} is not a terminal refresh state")
    if cause not in _CAUSES:
        raise RefreshRefused("internal", f"{cause!r} is not a known refresh cause")
    stopped = False
    with transaction(network_id, root, recipient=recipient) as doc:
        if doc is None:
            pass
        elif doc["state"] == "active" or state == "failed":
            doc["state"] = state
            doc["cause"] = cause
            stopped = True
    return stopped


def wipe(network_id: str, root: Path | None = None) -> None:
    """Remove the delegation document entirely (handback/failure).

    Best-effort and idempotent: a node that already lost the file has nothing to
    wipe, and reporting that as an error would make a completed handback look
    failed. The material itself is removed by the caller through its own row.
    """
    path = refresh_path(network_id, root)
    with (
        store._write_lock(path),
        placement._cross_process_lock(
            path.with_name(LOCK_FILENAME), root=placement._config_root(root)
        ),
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            return


def note_deny(
    statement: str,
    *,
    network_id: str,
    root: Path | None,
    recipient: str,
    operator_spki: Any = None,
    now: float | None = None,
) -> bool:
    """Apply an operator-signed DENY statement to this device, or ignore it.

    Returns whether this device was the named node and is now stopped. A
    statement that does not verify, names another device, or names another key
    is ignored silently — it is not this device's business, and refusing loudly
    would let any member probe what a node holds.
    """
    from local_operator.operator.verify import verify_refresh_grant

    spki = operator_spki if operator_spki is not None else anchor_spki_for(root)
    moment = time.time() if now is None else now
    grant = verify_refresh_grant(
        statement,
        operator_spki=spki,
        now=moment,
        network_id=network_id,
        node_device=recipient,
    )
    if grant is None or not grant.relay_deny:
        return False
    doc = load(network_id, root, recipient=recipient)
    if doc is None:
        return False
    held = verify_refresh_grant(
        doc["statement"],
        operator_spki=spki,
        now=moment,
        network_id=network_id,
        node_device=recipient,
    )
    if held is None or held.grant_id != grant.grant_id or held.key != grant.key:
        return False
    return stop(
        network_id,
        root,
        recipient=recipient,
        state="refused",
        cause="revoked",
    )


def live_for(
    network_id: str,
    root: Path | None,
    *,
    recipient: str,
    owner: str,
    key: str,
    now: float | None = None,
    operator_spki: Any = None,
) -> bool:
    """Whether THIS device is delegated to refresh ``(owner, key)`` right now.

    The one question ``store._owner_policy_skips_local`` asks: a live delegation
    for this owner binding IS the local answer, so a moved session must use it
    rather than broker to an owner whose laptop is closed. Fail-closed: unprovable
    evidence answers ``False`` (no delegation), which is today's behaviour.
    """
    try:
        grant = active(
            network_id,
            root,
            recipient=recipient,
            key=key,
            now=now,
            operator_spki=operator_spki,
        )
    except RefreshUnknown:
        return False
    return grant is not None and grant.owner_device == owner


def handback_holds(
    epoch: int, *, epoch_at_issue: int, incarnation: str, login_incarnation: str
) -> bool:
    """Whether a handback may be adopted (§3.3 Rule B), in ONE place.

    Two conditions and no clock: the epoch must be strictly higher than the one
    the delegation was minted at (a lower epoch never overwrites), and the login
    lineage must be the same one — so a handback from a superseded family is
    refused by a value the owner already has.
    """
    return epoch > epoch_at_issue and incarnation == login_incarnation
