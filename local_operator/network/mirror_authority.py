"""Who may AUTHOR a mirrored setting on a mesh peer: the operator's own device.

WHY THIS EXISTS (PR #3905 review round 1, M1). Settings mirrored over the
definitions sync (failover routing here; subagent tiers in PR #3869 follow the same
mirror pattern) must flow ONE way: from the operator's device to the nodes it
hands work to. An admin capability does not say which way that is. The operator
pairs their own machines as ``admin`` in both directions, so "may write
definitions on me" holds node→head as much as head→node. Gating on it alone let a
node's own chain land on the operator's Mac, and on a third device it let whichever
peer arrived first win for good.

THE TRUST ROOT IS THE EXISTING OPERATOR ANCHOR, not a new one. Remote onboarding
installs the operator's anchor (public key, ``key_id``) on every node as
VERIFY-ONLY data, while the private half stays on the operator's own device
(``readiness.operator_fact``: ``verify_only``). So:

* a device AUTHORS when its installed anchor is usable and it can SIGN with it
  (it holds the private half). That is the operator's device;
* a device ACCEPTS a mirror only when it holds the SAME anchor (``key_id``) and
  cannot sign with it itself (it is a node under that operator), and only from a
  sender claiming to sign for that anchor.

The claim travels unsigned. That is the same trust the definitions sync already
extends: only an ``admin`` member reaches ``net_definitions`` at all, and an admin
may install agents and teams outright. What the claim adds is DIRECTION between
honest peers. A node (which cannot sign) never sends, and the head (which signs)
never installs. Proof of possession would need a signature per push, and a
presence-backed key prompts the operator on every one.

A device with no usable anchor neither authors nor accepts, so nothing is
mirrored to or from it. ``lop network definitions state`` and the sender's report
say so rather than leaving the node silently primary-only.

REUSE: :func:`may_send_to` (sender side, against the peer's ``state`` reply
carrying :func:`state_fact`) and :func:`accepts_from` (receiver side, against the
claim in the incoming row) are the whole policy. A second mirrored lane calls the
same two functions and carries ``state_fact()`` and ``claim()`` under its own key.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

#: Anchor levels that mean "an operator anchor is installed and usable here"
#: (``operator.authority_level_load``). The same pair ``readiness.operator_fact``
#: tests before asking whether this host can sign.
INSTALLED_LEVELS = ("operator-presence", "operator-file-only")

#: How long one host probe is reused. The answer changes only through
#: ``lop operator init/install`` or a key rotation, which are deliberate operator
#: acts. The cadence asks once per member per minute, and every ask would
#: otherwise reload the signer, a keychain/enclave lookup. A stale answer lasts at
#: most this long and fails closed in the meantime: a freshly installed node
#: accepts a minute late, and nothing is accepted that should not be.
_PROBE_TTL_S = 60.0


@dataclass(frozen=True)
class Authority:
    """This device's standing under the operator anchor."""

    #: The installed, usable anchor's key id, or ``""`` when none is usable.
    key_id: str = ""
    #: Whether THIS host holds the private half (can sign for that anchor).
    signs: bool = False


def _probe_host(root: Path) -> Authority:
    """The real answer, from the operator module and readiness's own signer probe.

    ``root`` is unused in production: the anchor is a property of the HOST (a
    root-owned file per uid), not of one config root. It is a parameter so a test
    running a head and a node in one process can resolve each root separately.
    """
    del root
    try:
        from local_operator.network.readiness import _can_sign_here
        from local_operator.operator import authority_level_load

        level, loaded = authority_level_load()
        if level not in INSTALLED_LEVELS or not loaded.anchor:
            return Authority()
        return Authority(key_id=str(loaded.anchor.key_id or ""), signs=_can_sign_here())
    except Exception:  # noqa: BLE001 — no authority is a fact here, not a crash
        logger.debug("mirror authority: could not read the operator anchor", exc_info=True)
        return Authority()


#: The resolver seam. Tests replace it with a per-root table. Production keeps the
#: host probe.
_resolve: Callable[[Path], Authority] = _probe_host
_cache: dict[Path, tuple[float, Authority]] = {}
_cache_lock = threading.Lock()


def local_authority(root: Path) -> Authority:
    """This device's :class:`Authority`, reused for :data:`_PROBE_TTL_S`."""
    now = time.monotonic()
    key = Path(root)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and _resolve is _probe_host and now - hit[0] < _PROBE_TTL_S:
            return hit[1]
    value = _resolve(key)
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def state_fact(root: Path) -> dict[str, Any]:
    """What a ``state`` reply carries so a sender can tell whether it may author here.

    The key id is public anchor data (``lop operator anchor export`` prints it as
    the value to compare). No key material or backend detail travels.
    """
    authority = local_authority(root)
    return {"key_id": authority.key_id, "signs": authority.signs}


def claim(root: Path) -> str:
    """The anchor key id this device authors for, or ``""`` when it may not author."""
    authority = local_authority(root)
    return authority.key_id if authority.key_id and authority.signs else ""


def may_send_to(root: Path, remote_fact: Any) -> tuple[bool, str]:
    """Sender side: ``(True, "")`` only when this device authors for the peer's anchor."""
    mine = local_authority(root)
    if not (mine.key_id and mine.signs):
        return False, (
            "this device does not sign for the operator, so its settings stay here "
            "(only the operator's device sends them)"
        )
    if not isinstance(remote_fact, Mapping):
        return False, "the peer did not say which operator it belongs to"
    if str(remote_fact.get("key_id") or "") != mine.key_id:
        return False, "the peer does not hold this operator's authority, so nothing is sent to it"
    if remote_fact.get("signs"):
        return False, "the peer signs for the operator itself, so it keeps its own settings"
    return True, ""


def accepts_from(root: Path, claimed_key_id: Any) -> tuple[bool, str]:
    """Receiver side: ``(True, "")`` only for this node's operator-authority device."""
    mine = local_authority(root)
    if not mine.key_id:
        return False, "this device has no operator authority installed, so it takes no mirror"
    if mine.signs:
        return False, "this device signs for the operator itself, so it takes no mirror"
    if not isinstance(claimed_key_id, str) or claimed_key_id != mine.key_id:
        return False, "the sender does not sign for this device's operator"
    return True, ""


def reset_cache() -> None:
    """Forget cached probes (tests, and after an anchor install in-process)."""
    with _cache_lock:
        _cache.clear()
