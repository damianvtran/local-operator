"""Which mesh device is the OPERATOR's: a binding derived from a signature they already gave.

WHY THIS EXISTS (2026-11-15). A session STARTED on a mesh node has no placement
stamp, so its asks, site cards and ``notify`` items qualified for no viewer
(``notifications.remote_attention.stamp_rows_for_viewer`` forwards only to the
stamped origin) and stayed on the node — where nobody was watching. Forwarding
them needs one fact the node did not have: which linked device is the operator's.

MESH ROLES CANNOT ANSWER THAT. A member's ``role`` lives in ``networks/<id>.json``,
node-local, unsigned and writable by anything running as the node's user — so
"the admin member" is whatever a same-uid process last wrote there. The only
operator-rooted trust a node holds is the ROOT-OWNED anchor the onboarding runner
installs (``operator.trust.load_anchor``), whose key's private half never leaves
the operator's Mac.

THE BINDING IS DERIVED, NOT MINTED. Every ``device_onboard`` /
``device_onboard_network_only`` approval the operator signs is an ES256 signature
by that key over a request digest that binds (``network.approvals._immutable_request``,
or ``network.enrollment.commitment`` for the network-only kind):

* ``requested_by.device_id`` — the FILING device's mesh id, computed by
  ``onboard_approvals.request_device`` from the local identity rather than taken
  from a caller (invariant R1, pinned by a test: a remote filing path for these
  kinds would silently let a co-member mint a binding);
* ``what.network_id`` — the network the node is being onboarded into.

So the signed decision already says "I approve this, filed on mesh device M, for
network N". The Mac presents that existing decision over the authenticated link
(``net_operator_binding``); the node re-derives the digest, verifies the
signature against its installed anchor, and from then on treats the link whose
handshake-authenticated ``device_id`` is M, in network N, as the operator's.
No new card, no new Touch ID, no signed-format change.

WHAT IS REFUSED, and where (every step fails closed — see :func:`verify_proof`):

* a forged admin role — roles are never read here;
* a proof replayed on another network — ``what.network_id`` is inside the digest;
* a device removed from (or burned in) the network — the live membership read on
  every use is THE revocation lever. The anchor's ``devices`` list is not one:
  it holds operator-device KEY ids (phones), a different namespace from mesh
  ``d_…`` ids, so it can never name the Mac (security review S3);
* a decision that is not an onboarding — ``device_onboard`` is also filed by the
  credential share/inheritance flows with a caller-chosen ``requested_by``, so a
  proof must also carry the onboarding SHAPE (:func:`_is_onboarding_shape`), not
  just the kind;
* an old signer — the signature must verify under the CURRENT anchor's key, so a
  node re-anchored to a new key drops every proof signed by the old one;
* a tampered stored proof — the store holds the raw proof and every use
  re-verifies it (through a cache keyed by the proof bytes, see R5 below);
* a node holding the anchor's private key itself — such a node's agent could sign
  its own "approval", so it is ``no_approver`` (the site-approval rule) and no
  proof counts;
* the staged, user-writable anchor (``trust.load_local_anchor``) is NEVER read:
  verifying against it would let a same-uid process plant its own key (R4).

RESIDUAL RISK, DISCLOSED. On a node whose user has passwordless sudo the anchor —
and so this binding — is advisory, exactly as site approvals are
(``browser_bridge/site_approval.py``): such a root can replace the anchor, but it
already owns every byte this forwards.

DISCLOSURE. The proof carries the onboarding request's preimage — the onboarded
node's SSH host/user, its host-key fingerprint and the credential REFERENCE name
(never material) — because the signature covers them: stripping any field breaks
verification. So the Mac presents a proof ONLY to the node that very onboarding
admitted, and only that node's own proof: the onboarding runner records the
admitted mesh id beside the approval in Mac-local metadata
(:func:`record_onboarded`, never in the signed record), and :func:`own_proofs`
matches it exactly. A peer the Mac did not onboard — a co-member, or a node
onboarded before this build recorded anything — is sent nothing at all
(security review S1: the earlier "newest proof for the network" fallback put
the operator's SSH host, user and credential-ref name on every co-member's link).

R5 — COST. ES256 verification stays out of the per-listing loop: a verdict is
cached keyed by the proof's bytes, the anchor's bytes and the local-key state,
so a catalogue listing pays one dict lookup plus one membership read; a changed
anchor or an edited proof misses the cache and is re-verified. (ECDSA is
malleable: ``(r, n-s)`` of a valid signature also verifies, so one signed
decision can have two byte forms and two cache entries. Benign — both bind the
identical signed content — so no low-s canonicalisation; security review S4.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from local_operator.network.relay import PeerLink, RelayServer

logger = logging.getLogger(__name__)

#: The peer op: the Mac presents, the node verifies and stores.
OP = "net_operator_binding"

#: ONLY these kinds are evidence — necessary, not sufficient. An ONBOARDING of
#: either kind is filed by ``onboard_approvals.request_device``, which computes
#: ``requested_by`` from this install's own identity. ``device_onboard`` is ALSO
#: filed by ``credentials/finite_share.py`` and ``credentials/inheritance.py``
#: with a CLI-chosen ``requested_by``; those are kept out by the onboarding shape
#: (:func:`_is_onboarding_shape`), checked on both the Mac's selection and the
#: node's verification, not by their happening to lack ``what.network_id``
#: (security review S2). Kinds a REMOTE peer files into the store
#: (``operator_device_pair`` via ``mobile/pair_requests``) or that name no network
#: (``profile_save``, ``local_authority``) are excluded by this allow-list, not by
#: a deny-list a new kind could slip past.
PROOF_KINDS: frozenset[str] = frozenset({"device_onboard", "device_onboard_network_only"})

#: Record states in which the operator's APPROVE signature stands. ``failed`` is
#: included: the run failing does not unsay what the operator signed.
_APPROVED_STATES: frozenset[str] = frozenset({"approved", "connecting", "connected", "failed"})

PROOF_VERSION = 1
#: Frame bounds: a presentation is a handful of small JSON objects; anything
#: larger is not one this build minted.
MAX_PROOFS = 4
MAX_PROOF_BYTES = 16 * 1024
#: Network ids, mesh device ids and approval ids all become FILE NAMES here.
_NETWORK_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}\Z")
#: Keys a credential share/inheritance card carries in ``what``; their presence
#: makes a ``device_onboard`` record a credential operation, never an onboarding
#: (the same predicate ``server/utils/desktop_onboard.py`` uses).
_CREDENTIAL_OPERATION_KEYS = ("credential_inheritance", "credential_finite_share")
_HEX = re.compile(r"[0-9a-f]{16,512}\Z")

#: The anchor read is a few syscalls, but a listing loop on a busy relay should
#: not repeat them per frame; a short TTL keeps a re-anchor visible within seconds.
ANCHOR_TTL_S = 5.0

_lock = threading.Lock()
_verdicts: dict[tuple[str, ...], tuple[bool, str]] = {}
_anchor_memo: tuple[float, Any] | None = None
#: The last link verdict written per ``(root, network, device)``, so a listing
#: loop rewrites the verdict file only when the verdict CHANGES.
_link_memo: dict[tuple[str, str, str], bool] = {}


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    """0600 write-then-replace with a tmp name unique PER WRITE.

    Presentations run inline on per-link reader threads, so two links of one
    process can write at once; a per-process tmp name let them interleave or
    lose the replace (review R1-m3).
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _is_onboarding_shape(request: Any) -> bool:
    """The record is a real SSH ONBOARDING, not a credential operation (S2).

    ``request_device`` always writes ``what.connect``/``what.install`` true and an
    ``ssh`` device block; the credential share/inheritance cards that reuse the
    ``device_onboard`` kind carry their operation key in ``what`` instead. All of
    these fields are inside the signed digest, so the shape cannot be dressed up
    after signing.
    """
    if not isinstance(request, Mapping):
        return False
    what = request.get("what")
    device = request.get("device")
    if not isinstance(what, Mapping) or not isinstance(device, Mapping):
        return False
    if device.get("transport") != "ssh":
        return False
    if what.get("connect") is not True or what.get("install") is not True:
        return False
    return not any(key in what for key in _CREDENTIAL_OPERATION_KEYS)


# ---------------------------------------------------------------------------
# The Mac's side: which signed decisions are this device's proofs
# ---------------------------------------------------------------------------


def _network_of(record: Mapping[str, Any]) -> str:
    """``what.network_id`` — the one path the signed digest binds the network at.

    Flat in the immutable request for BOTH kinds (the network-only kind's
    on-disk file wraps the record as ``{format, version, record}``, so a raw
    ``jq`` reads it at ``.record.what.network_id``; ``what.anchor`` carries only
    ``{key_id, spki_fp, statement_digest}``). A legacy ``device_onboard`` filed
    without a network has none, and is no proof for any network.
    """
    what = record.get("what")
    value = what.get("network_id") if isinstance(what, Mapping) else None
    return value if isinstance(value, str) else ""


def proof_from_record(record: Mapping[str, Any], *, own_device_id: str) -> dict[str, Any] | None:
    """The presentable proof for one approval record, or ``None`` if it is not one.

    Eligible only when the operator APPROVED it with the anchor key directly
    (``cert is None`` — never a phone-delegated decision), for an allow-listed
    kind, naming a network, and filed by THIS device. The last condition keeps a
    device from presenting a decision that binds someone else (the node would
    refuse it anyway: the bound id must equal the presenting link's).
    """
    from local_operator.network.approvals import _immutable_request

    kind = str(record.get("kind") or "")
    if kind not in PROOF_KINDS or record.get("state") not in _APPROVED_STATES:
        return None
    signature = record.get("signature")
    if not isinstance(signature, Mapping) or signature.get("cert") is not None:
        return None
    key_id = str(signature.get("key_id") or "")
    if not key_id or signature.get("by") != f"operator:{key_id}" or not signature.get("sig"):
        return None
    requested_by = record.get("requested_by")
    if not isinstance(requested_by, Mapping) or requested_by.get("device_id") != own_device_id:
        return None
    if not own_device_id or not _network_of(record) or not _is_onboarding_shape(record):
        return None
    decided_at = record.get("decided_at")
    if not isinstance(decided_at, (int, float)) or isinstance(decided_at, bool):
        return None
    return {
        "v": PROOF_VERSION,
        "kind": kind,
        "approval_id": str(record.get("approval_id") or ""),
        "request_id": str(record.get("request_id") or ""),
        "request": _immutable_request(record),
        "decided_at": float(decided_at),
        "sig": str(signature["sig"]),
        "key_id": key_id,
    }


def _own_device_id(root: Path | None) -> str:
    """This install's mesh id, or ``""`` — never minted by a reader."""
    from local_operator.network import identity

    try:
        mine = identity.load(root)
    except (OSError, ValueError):
        return ""
    return mine.device_id if mine is not None else ""


def _own_proofs_by_record(root: Path | None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """``(record, proof)`` for every retained decision that is a proof. Never raises."""
    from local_operator.network import approvals

    own = _own_device_id(root)
    if not own:
        return []
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for enrollment_only in (False, True):
        try:
            directory = approvals.approvals_dir(root, enrollment_only=enrollment_only)
        except Exception:  # noqa: BLE001 — a refused namespace has no proofs
            continue
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("apn_*.json" if enrollment_only else "ap_*.json")):
            try:
                record = approvals._load_raw(path.stem, root)  # noqa: SLF001
            except Exception:  # noqa: BLE001 — unreadable or refused: not a proof
                continue
            proof = proof_from_record(record, own_device_id=own)
            if proof is not None:
                out.append((record, proof))
    return out


def _onboarded_dir(root: Path | None) -> Path:
    from local_operator.network.identity import network_root

    return network_root(root) / "operator-onboarded"


def record_onboarded(
    root: Path | None, *, approval_id: str, network_id: str, device_id: str
) -> bool:
    """Mac side: note that onboarding ``approval_id`` admitted mesh device ``device_id``.

    WHY (security review S1, 2026-11-15). The signed card cannot say which mesh id
    the node will have: ``request_device`` files ``device.device_id`` empty (the
    node mints its identity during the run), so the Mac had no way to tell the
    node it onboarded from any co-member and presented to everyone. The onboarding
    RUNNER learns the id when it admits the node (the invite's ``redeemed_by``,
    which the relay wrote from the handshake-authenticated id), and records it
    here. Mac-LOCAL metadata beside the approval, deliberately NOT in the signed
    record: it only decides where this Mac discloses, never what a node trusts
    (the node still verifies the operator's signature and binds the FILER).

    Returns whether a record was written. Never raises.
    """
    for value in (approval_id, network_id, device_id):
        if not _NETWORK_ID.fullmatch(value or ""):
            return False
    try:
        _atomic_write(
            _onboarded_dir(root) / f"{approval_id}.json",
            {
                "approval_id": approval_id,
                "network_id": network_id,
                "device_id": device_id,
                "at": time.time(),
            },
        )
    except OSError:
        logger.debug("operator binding: could not record the onboarded node", exc_info=True)
        return False
    return True


def onboarded_devices(root: Path | None) -> dict[str, tuple[str, str]]:
    """``{approval_id: (network_id, device_id)}`` the runner recorded. Never raises."""
    out: dict[str, tuple[str, str]] = {}
    try:
        directory = _onboarded_dir(root)
        if not directory.is_dir():
            return out
        for path in directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, Mapping) or data.get("approval_id") != path.stem:
                continue
            network_id, device_id = data.get("network_id"), data.get("device_id")
            if isinstance(network_id, str) and isinstance(device_id, str):
                out[path.stem] = (network_id, device_id)
    except OSError:
        logger.debug("operator binding: onboarded records unreadable", exc_info=True)
    return out


def _presentable(root: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """The NEWEST own proof per ``(network, onboarded device)`` the runner recorded."""
    admitted = onboarded_devices(root)
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for _record, proof in _own_proofs_by_record(root):
        target = admitted.get(proof["approval_id"])
        if target is None or target[0] != _network_of(proof["request"]):
            continue
        current = best.get(target)
        if current is None or proof["decided_at"] > current["decided_at"]:
            best[target] = proof
    return best


def retained_approval_ids(root: Path | None) -> frozenset[str]:
    """Approval ids the 30-day prune must keep: the binding proofs in use.

    Manager decision M1 (2026-11-15): ``approvals._sweep_namespace`` deletes
    terminal records after 30 days, and a pruned decision is a binding nobody can
    present again without a fresh onboarding. Keep exactly the decision
    :func:`own_proofs` presents to each onboarded node — the newest per
    ``(network, node)``. Older decisions, and ones with no recorded node (never
    presented to anyone), prune as before: retention is bounded by the number
    of onboarded nodes.
    """
    return frozenset(proof["approval_id"] for proof in _presentable(root).values())


def own_proofs(root: Path | None, network_id: str, *, for_device: str = "") -> list[dict[str, Any]]:
    """The proof to present to ``for_device`` in ``network_id``: at most one.

    ONLY the decision whose onboarding admitted exactly that mesh device in
    exactly that network (:func:`record_onboarded`). Anything else — a
    co-member, a node this Mac did not onboard, an empty ``for_device`` — gets
    ``[]``: the preimage discloses the node's SSH host/user and credential-ref
    name, and M3 allows it only to the operator's own nodes, which the Mac can
    establish only from its own records (security review S1).
    """
    if not network_id or not for_device:
        return []
    proof = _presentable(root).get((network_id, for_device))
    return [proof] if proof is not None else []


# ---------------------------------------------------------------------------
# The node's side: verification against the installed anchor
# ---------------------------------------------------------------------------


def _installed_anchor() -> Any:
    """``trust.load_anchor()`` (the ROOT-OWNED file), memoized for a few seconds.

    NEVER ``load_local_anchor``: that falls back to the staged statement under the
    user's config dir, which any same-uid process can write (R4).
    """
    global _anchor_memo
    from local_operator.operator.trust import load_anchor

    now = time.monotonic()
    with _lock:
        memo = _anchor_memo
    if memo is not None and now - memo[0] < ANCHOR_TTL_S:
        return memo[1]
    loaded = load_anchor()
    with _lock:
        _anchor_memo = (now, loaded)
    return loaded


def _reset_caches() -> None:
    """Test seam: forget memoized anchor reads and verdicts."""
    global _anchor_memo
    with _lock:
        _anchor_memo = None
        _verdicts.clear()
        _link_memo.clear()


def _key_file_state(config_root: Path) -> str:
    """A cache-key fragment that changes when a local operator key file appears."""
    from local_operator.operator.keychain import default_file_path

    try:
        info = os.stat(default_file_path(config_root))
    except OSError:
        return "absent"
    return f"{info.st_ino}:{info.st_mtime_ns}:{info.st_size}"


def _digest_for(proof: Mapping[str, Any]) -> str:
    """Re-derive the request digest the signature covers, from the carried preimage."""
    from local_operator.network import approvals, enrollment

    request = dict(proof["request"])
    if proof["kind"] == enrollment.KIND:
        # The commitment validates a whole record; rebuild one around the signed
        # preimage. The lifecycle keys are not in the commitment, so their values
        # here are only what ``validate`` needs to accept the shape.
        record = {
            **request,
            "approval_id": proof["approval_id"],
            "request_id": proof["request_id"],
            "state": "approved",
            "receipts": [],
            "schema": 1,
            "authority_version": enrollment.VERSION,
        }
        return enrollment.commitment(record)
    # A ``device_onboard`` digest is over the immutable request alone; no
    # ``approval_id`` is passed, so ``enrollment.restricted`` cannot reroute it.
    return approvals.request_digest(request)


def _check_signature(
    config_root: Path, proof: Mapping[str, Any], loaded: Any
) -> tuple[bool, str, Any]:
    """Steps 1-5: anchor usable, no local key, same key, signature, kind/decision.

    ``loaded`` is the SAME anchor read the caller keyed its verdict cache on, so a
    re-anchor between the key and the check cannot file a verdict under the
    wrong anchor.
    """
    from local_operator.browser_bridge.site_approval import anchor_key_held_locally
    from local_operator.network import approvals
    from local_operator.operator.verify import verify_signature

    if not loaded.usable or loaded.anchor is None:
        return False, "no_installed_anchor", None
    anchor = loaded.anchor
    if anchor_key_held_locally(config_root, anchor.key_id):
        # This node could sign its own "decision": it has no independent approver.
        return False, "no_approver", anchor
    if proof.get("key_id") != anchor.key_id:
        return False, "old_signer", anchor
    if proof.get("kind") not in PROOF_KINDS:
        return False, "kind_not_allowed", anchor
    request = proof.get("request")
    if not isinstance(request, Mapping) or request.get("kind") != proof.get("kind"):
        return False, "malformed", anchor
    if not _is_onboarding_shape(request):
        # A credential share/inheritance card the operator signed is a
        # ``device_onboard`` too; its signature says nothing about which device
        # is the operator's (S2).
        return False, "not_onboarding", anchor
    try:
        digest = _digest_for(proof)
        signature = bytes.fromhex(str(proof.get("sig") or ""))
        message = approvals.signed_payload(
            kind=str(proof["kind"]),
            request_id=str(proof.get("request_id") or ""),
            request_digest=digest,
            decision="approve",
            decided_at=float(proof["decided_at"]),
        )
    except Exception:  # noqa: BLE001 — any shape the codec refuses is not a proof
        return False, "malformed", anchor
    if not verify_signature(spki=anchor.spki, message=message, signature=signature):
        return False, "signature_invalid", anchor
    return True, "", anchor


def _well_formed(proof: Any) -> bool:
    if not isinstance(proof, Mapping) or proof.get("v") != PROOF_VERSION:
        return False
    if not isinstance(proof.get("request"), Mapping):
        return False
    if not _HEX.fullmatch(str(proof.get("sig") or "")):
        return False
    decided = proof.get("decided_at")
    return isinstance(decided, (int, float)) and not isinstance(decided, bool)


def _proof_key(proof: Mapping[str, Any]) -> str:
    from local_operator.network.wire import canonical_json

    return hashlib.sha256(canonical_json(dict(proof))).hexdigest()


def bound_device(proof: Mapping[str, Any]) -> str:
    """The mesh device the signed decision names as its filer."""
    request = proof.get("request")
    requested_by = request.get("requested_by") if isinstance(request, Mapping) else None
    value = requested_by.get("device_id") if isinstance(requested_by, Mapping) else ""
    return value if isinstance(value, str) else ""


def verify_proof(
    config_root: Path, proof: Any, *, network_id: str, device_id: str
) -> tuple[bool, str]:
    """Is ``proof`` the operator's signed word that ``device_id`` is theirs in ``network_id``?

    Fail-closed, ``(ok, reason)``. The signature part (the only costly step) is
    cached by proof bytes + anchor bytes + local-key state; the bindings and
    revocation are re-read each call from those same inputs. Membership is the
    caller's (:func:`is_operator_link` / :func:`operator_device`), because it
    changes without the anchor changing.
    """
    from local_operator.operator.trust import anchor_bytes

    if not _well_formed(proof):
        return False, "malformed"
    if not network_id or not device_id:
        return False, "no_link_identity"
    loaded = _installed_anchor()
    anchor_key = (
        hashlib.sha256(anchor_bytes(loaded.anchor)).hexdigest()
        if loaded.usable and loaded.anchor is not None
        else "unusable"
    )
    key = (_proof_key(proof), anchor_key, _key_file_state(Path(config_root)), str(config_root))
    with _lock:
        cached = _verdicts.get(key)
    if cached is None:
        ok, reason, _anchor = _check_signature(Path(config_root), proof, loaded)
        cached = (ok, reason)
        with _lock:
            if len(_verdicts) > 256:
                _verdicts.clear()
            _verdicts[key] = cached
    if not cached[0]:
        return cached
    if _network_of(proof["request"]) != network_id:
        # The network is inside the signed digest, so this is a replay of a valid
        # proof from another network — a hard refusal.
        return False, "network_mismatch"
    if bound_device(proof) != device_id:
        return False, "device_mismatch"
    # NO anchor-revocation check here (security review S3): the anchor's
    # ``devices`` list holds operator-device KEY ids, never mesh ids, so it could
    # not match. Revoking the operator's device for this purpose is removing (or
    # burning) it in the network: :func:`_active_member`, read live on every use.
    return True, ""


def _bindings_dir(root: Path) -> Path:
    from local_operator.network.identity import network_root

    return network_root(root) / "operator-bindings"


def _stored_path(root: Path, network_id: str, device_id: str) -> Path | None:
    """``operator-bindings/<network>/<device>.json``: one slot PER BOUND DEVICE.

    One slot per network made two legitimately bound devices (an anchored node
    may onboard machines too) overwrite each other on every link-up, so which
    one got the asks depended on link order (review R1-m2).
    """
    if not _NETWORK_ID.fullmatch(network_id or "") or not _NETWORK_ID.fullmatch(device_id or ""):
        return None
    return _bindings_dir(root) / network_id / f"{device_id}.json"


def load_stored(root: Path, network_id: str, device_id: str) -> dict[str, Any] | None:
    """The RAW stored proof for one device in one network; untrusted until verified."""
    path = _stored_path(root, network_id, device_id)
    if path is None:
        return None
    try:
        if path.stat().st_size > MAX_PROOF_BYTES:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _store(root: Path, network_id: str, device_id: str, proof: Mapping[str, Any]) -> None:
    path = _stored_path(root, network_id, device_id)
    if path is None:
        return
    _atomic_write(path, proof)


def _links_path(root: Path) -> Path:
    from local_operator.network.identity import network_root

    return network_root(root) / "operator-binding-links.json"


def _link_verdicts(root: Path) -> dict[str, bool]:
    try:
        data = json.loads(_links_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, bool)}


def _record_link_verdict(root: Path, network_id: str, device_id: str, ok: bool) -> None:
    """Persist the listing leg's verdict for a BOUND device's link. Never raises.

    WHY (review R1-m1 / QA Q1). The ``notify`` sentence is decided in the
    session's process, which cannot see the relay's links; it trusted the stored
    proof alone, so a bound Mac that reconnected as an older build (no
    ``operator-binding-v1``, so the catalogue carries nothing) still got "will be
    forwarded ... the operator's device". The relay now records what its last
    listing for that device decided, and :func:`operator_device` requires it.
    Written only when the verdict CHANGES (memoized per process), so a listing
    loop costs a dict lookup, not a write.
    """
    memo_key = (str(root), network_id, device_id)
    with _lock:
        if _link_memo.get(memo_key) is ok:
            return
    try:
        with _lock:
            verdicts = _link_verdicts(root)
            entry = f"{network_id}/{device_id}"
            if verdicts.get(entry) is not ok:
                verdicts[entry] = ok
                _atomic_write(_links_path(root), verdicts)
            _link_memo[memo_key] = ok
    except OSError:
        logger.debug("operator binding: could not record the link verdict", exc_info=True)


def _active_member(root: Path, network_id: str, device_id: str) -> bool:
    """``device_id`` is an ACTIVE, un-burned member of a trusted network here.

    Exact id match only: ``MemberRecord.previous_ids`` is node-local and unsigned,
    so it is no rotation evidence.
    """
    from local_operator.network import store as net_store

    try:
        record = net_store.load(network_id, root)
    except Exception:  # noqa: BLE001 — an unreadable record binds nothing
        return False
    if record.trust != "active" or record.is_burned(device_id):
        return False
    return any(m.device_id == device_id for m in record.active_members())


def accept_presented(root: Path, link: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
    """Node side of ``net_operator_binding``: verify each proof, store the first good one.

    The presenter must BE the bound device on this link's network: a device can
    only ever present its own binding, which keeps the store honest even though
    use-time re-verification would make a cross-presentation inert.
    """
    network_id = str(getattr(link, "network_id", "") or "")
    device_id = str(getattr(link, "device_id", "") or "")
    proofs = frame.get("proofs")
    if not isinstance(proofs, list) or not proofs or len(proofs) > MAX_PROOFS:
        return {"accepted": False, "reason": "malformed"}
    reason = "malformed"
    for proof in proofs:
        try:
            size = len(json.dumps(proof))
        except (TypeError, ValueError):
            continue
        if size > MAX_PROOF_BYTES:
            continue
        ok, reason = verify_proof(root, proof, network_id=network_id, device_id=device_id)
        if ok:
            try:
                _store(root, network_id, device_id, proof)
            except OSError as exc:
                return {"accepted": False, "reason": f"store_failed: {exc.strerror}"}
            _audit(root, "operator_binding_accepted", network_id, device_id)
            return {"accepted": True, "reason": ""}
    _audit(root, "operator_binding_refused", network_id, device_id, reason)
    return {"accepted": False, "reason": reason}


def _audit(root: Path, event: str, network_id: str, device_id: str, reason: str = "") -> None:
    # The approvals store's writer: one never-raising mesh-audit seam, not a second.
    from local_operator.network.approvals import _audit as approvals_audit

    approvals_audit(
        event,
        actor=device_id,
        subject=network_id,
        root=root,
        detail={"reason": reason} if reason else {},
    )


def is_operator_link(root: Path, link: Any) -> bool:
    """Is this authenticated link the operator's device? Never raises.

    Requires the peer to advertise the capability (an old Mac never presented, and
    a downgraded one must not ride a proof it can no longer refresh), a stored
    proof that verifies for the link's own network and device id, and an active
    membership row for that id. For a device with a stored proof the verdict is
    recorded (:func:`_record_link_verdict`) so the ``notify`` sentence agrees.
    """
    try:
        from local_operator.network import wire

        network_id = str(getattr(link, "network_id", "") or "")
        device_id = str(getattr(link, "device_id", "") or "")
        proof = load_stored(Path(root), network_id, device_id)
        if proof is None:
            return False
        ok = wire.OPERATOR_BINDING_V1 in getattr(link, "capabilities", ())
        if ok:
            ok, _reason = verify_proof(
                Path(root), proof, network_id=network_id, device_id=device_id
            )
            ok = ok and _active_member(Path(root), network_id, device_id)
        _record_link_verdict(Path(root), network_id, device_id, ok)
        return ok
    except Exception:  # noqa: BLE001 — a listing never fails on this
        logger.debug("operator binding: link check failed", exc_info=True)
        return False


def operator_device(root: Path) -> tuple[str, str]:
    """``(device_id, network_id)`` of a verified operator device for this node, or ``("", "")``.

    The session-side half (``remote_attention.notice_forwarding``): a ``notify``
    result may say "forwarded" only when the catalogue leg would carry it: a
    stored proof that still verifies for an active member AND whose link's last
    listing verdict was positive (capability present; R1-m1). With several bound
    devices, the newest signed decision names the one the sentence reports.
    Never raises.
    """
    best: tuple[float, str, str] | None = None
    try:
        directory = _bindings_dir(Path(root))
        if not directory.is_dir():
            return "", ""
        verdicts = _link_verdicts(Path(root))
        for path in sorted(directory.glob("*/*.json")):
            network_id, device_id = path.parent.name, path.stem
            if verdicts.get(f"{network_id}/{device_id}") is not True:
                continue
            proof = load_stored(Path(root), network_id, device_id)
            if proof is None:
                continue
            ok, _reason = verify_proof(
                Path(root), proof, network_id=network_id, device_id=device_id
            )
            if ok and _active_member(Path(root), network_id, device_id):
                at = float(proof["decided_at"])
                if best is None or at > best[0]:
                    best = (at, device_id, network_id)
    except Exception:  # noqa: BLE001
        logger.debug("operator binding: device lookup failed", exc_info=True)
    return (best[1], best[2]) if best is not None else ("", "")


# ---------------------------------------------------------------------------
# Relay wiring
# ---------------------------------------------------------------------------


def make_handler(server: "RelayServer") -> Any:
    def _handle(link: "PeerLink", frame: dict[str, Any]) -> dict[str, Any]:
        return accept_presented(server.root, link, frame)

    return _handle


def present(server: "RelayServer", link: Any) -> dict[str, Any] | None:
    """Mac side: send the peer ITS OWN onboarding proof, or nothing at all.

    Nothing is sent — not even an empty frame — to a peer without the capability
    or one this Mac's records do not show it onboarded (:func:`own_proofs`).
    """
    from local_operator.network import wire

    if wire.OPERATOR_BINDING_V1 not in getattr(link, "capabilities", ()):
        return None
    proofs = own_proofs(
        server.root, str(link.network_id or ""), for_device=str(link.device_id or "")
    )
    if not proofs:
        return None
    return link.request(
        {"op": OP, "req": server._next_relay_req(), "proofs": proofs}  # noqa: SLF001
    )


def _on_link_up(server: "RelayServer", link: Any) -> None:
    """Link-up hook: present off-thread — a hook must only record and signal."""
    from local_operator.network import wire

    if wire.OPERATOR_BINDING_V1 not in getattr(link, "capabilities", ()):
        return

    def _run() -> None:
        try:
            reply = present(server, link)
            if reply is not None:
                detail = reply.get("detail") if isinstance(reply, Mapping) else None
                logger.info("operator binding: presented to %s: %s", link.device_id, detail)
        except Exception as exc:  # noqa: BLE001 — best effort; the next link-up retries
            logger.debug("operator binding: presentation failed (%s)", exc)

    threading.Thread(target=_run, name="operator-binding-present", daemon=True).start()


def install(server: "RelayServer") -> None:
    server.register_ops(
        {OP: make_handler(server)},
        on_link_up={"operator-binding": lambda link: _on_link_up(server, link)},
    )
