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
* a decision that is neither an onboarding nor an EXACT credential card —
  ``device_onboard`` is also filed by the credential share/inheritance flows
  (owner = the filing device, pinned by their shape predicates), so a proof
  must carry one of those signed SHAPES (:func:`_is_onboarding_shape`,
  :func:`_card_target`), not just the kind;
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

CREDENTIAL CARDS ARE PROOFS TOO (PR #4030 review R2-M1, security ruling
conditions (a)-(g)). A node joined by invite, or onboarded before the runner
recorded anything, has no Mac-side tie to an onboarding decision — but the
operator may have signed a credential share/inheritance card FOR it. Such a
card's digest binds ``requested_by.device_id == owner_device`` (the Mac),
``recipient_device`` and ``network_id`` inside ``what.credential_*``, all
proven by the flow's own exact predicate (``finite_share.exact_shape``,
``inheritance._parse``). So the target is IN the signed bytes: the Mac presents
a card only to its recipient on its network while that recipient is an active
member, and the node binds only a card whose recipient is ITSELF. No
onboarding record or backfill is involved. Accepted residual: revoking the
credential grant does not revoke the binding (it still truthfully says the Mac
is the operator's); removing or burning the Mac does.

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
from collections.abc import Mapping, Sequence
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
#: (owner = the filing device: their ``_subject`` refuses any other owner, and
#: ``exact_shape`` / ``_parse`` pin ``requested_by.device_id == owner_device``;
#: security review S7). Those count only in their EXACT signed shape
#: (:func:`_card_target`), whose target is the signed recipient; any other
#: ``device_onboard`` must be the onboarding shape (:func:`_is_onboarding_shape`).
#: Both are checked on the Mac's selection and the node's verification
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


def _card_target(request: Any) -> tuple[str, str, str] | None:
    """``(family, network_id, recipient_device)`` of an EXACT credential card, else ``None``.

    Security ruling condition (a): only each flow's OWN exact predicate decides,
    never a loose "has a credential key" test — ``finite_share.exact_shape`` and
    ``inheritance._parse`` are what prove ``requested_by.device_id ==
    owner_device`` and ``device.device_id == recipient_device`` from the signed
    bytes. Works on a full record or on its immutable request (both carry
    ``what``/``device``/``requested_by``/``credential_ref``). Never raises.
    """
    return _card_check(request)[0]


def _card_check(request: Any) -> tuple[tuple[str, str, str] | None, str]:
    """:func:`_card_target` plus WHY not, for the dry run's ``credential_cards_refused``.

    The reason is ``not_a_card``, ``not_exact_shape`` or ``check_failed:<Exception>``.
    The last one matters: ``inheritance._parse`` reaches ``reconcile.provider_ok``,
    which imports pydantic, so a dry run under an interpreter without the
    runtime's dependencies would otherwise drop an inheritance card silently.
    """
    from local_operator.network import approvals
    from local_operator.network.credentials import finite_share, inheritance
    from local_operator.network.types import MeshRefusal

    if not isinstance(request, Mapping) or request.get("kind") != approvals.KIND_DEVICE_ONBOARD:
        return None, "not_a_card"
    record = dict(request)
    what = record.get("what")
    if not isinstance(what, dict):
        return None, "not_a_card"
    try:
        if finite_share.OP in what:
            data = finite_share.intent(record) if finite_share.exact_shape(record) else None
            family = "finite_share"
        elif inheritance.OP in what:
            data = inheritance._parse(record)["data"]  # noqa: SLF001 — the flow's own gate
            family = "inheritance"
        else:
            return None, "not_a_card"
    except MeshRefusal:
        return None, "not_exact_shape"
    except Exception as exc:  # noqa: BLE001 — any check that cannot run is no proof
        return None, f"check_failed:{type(exc).__name__}"
    if not isinstance(data, Mapping):
        return None, "not_exact_shape"
    network_id, recipient = data.get("network_id"), data.get("recipient_device")
    owner = data.get("owner_device")
    requested_by = record.get("requested_by")
    if not all(isinstance(v, str) and _NETWORK_ID.fullmatch(v) for v in (network_id, recipient)):
        return None, "not_exact_shape"
    # Redundant with both predicates; kept so this module states its own rule.
    if not isinstance(requested_by, Mapping) or requested_by.get("device_id") != owner:
        return None, "not_exact_shape"
    return (family, str(network_id), str(recipient)), ""


def _network_of(record: Mapping[str, Any]) -> str:
    """The signed network, read at the ONE path each proof family binds it at.

    A credential card: ``what.credential_*.network_id`` via :func:`_card_target`
    (condition (c)); a record carrying a credential key that is NOT an exact card
    has no network at all — no fallback to any other field. An onboarding:
    ``what.network_id``, as follows.

    Flat in the immutable request for BOTH kinds (the network-only kind's
    on-disk file wraps the record as ``{format, version, record}``, so a raw
    ``jq`` reads it at ``.record.what.network_id``; ``what.anchor`` carries only
    ``{key_id, spki_fp, statement_digest}``). A legacy ``device_onboard`` filed
    without a network has none, and is no proof for any network.
    """
    what = record.get("what")
    if isinstance(what, Mapping) and any(key in what for key in _CREDENTIAL_OPERATION_KEYS):
        card = _card_target(record)
        return card[1] if card is not None else ""
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
    if not own_device_id or not _network_of(record):
        return None
    if not _is_onboarding_shape(record) and _card_target(record) is None:
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


#: The invite step's own receipt sentence (``onboard.OnboardRun.step_invite``):
#: "invite <id> minted for role <role>...". The id is THIS Mac's mint (Crockford,
#: lowercase), written into this Mac's approval record by this Mac's runner.
_INVITE_RECEIPT = re.compile(r"invite ([0-9a-z]{4,64}) minted for role ")

#: Last logged backfill outcome per approval, so a link-up storm logs each
#: resolution or skip once per process rather than once per link.
_backfill_logged: dict[str, str] = {}


def _invite_ids(record: Mapping[str, Any]) -> list[str]:
    """Invite ids this approval's runs minted, from its OK ``invite`` receipts."""
    out: list[str] = []
    for receipt in record.get("receipts") or ():
        if not isinstance(receipt, Mapping) or receipt.get("step") != "invite":
            continue
        if not receipt.get("ok"):
            continue
        match = _INVITE_RECEIPT.match(str(receipt.get("detail") or ""))
        if match and match.group(1) not in out:
            out.append(match.group(1))
    return out


def _backfill_one(root: Path | None, record: Mapping[str, Any], network_id: str) -> dict[str, Any]:
    """Resolve one pre-feature onboarding to the mesh id it admitted, or say why not.

    THE CHAIN, every link Mac-written and none node-writable: the approval's own
    ``invite`` receipt (this Mac's runner, naming the invite this Mac minted) →
    that invite in this Mac's network record → its ``redeemed_by``, which this
    Mac's relay wrote from the HANDSHAKE-authenticated device id, on an invite
    the relay consumed as ``admitted`` → still an active, unburned member. The
    node's own reports (the join receipt's "joined … as <id>", ``identity
    show``) are never read: a node could name a co-member there.
    """
    from local_operator.network import store as net_store

    candidates: set[str] = set()
    try:
        network = net_store.load(network_id, root)
    except Exception:  # noqa: BLE001 — no record of the network: nothing to tie
        return {"result": "skipped", "reason": "network_record_unreadable"}
    invites = _invite_ids(record)
    if not invites:
        return {"result": "skipped", "reason": "no_invite_receipt"}
    for invite_id in invites:
        redeemer = admitted_redeemer(network.invite(invite_id))
        if redeemer:
            candidates.add(redeemer)
    if not candidates:
        return {"result": "skipped", "reason": "no_admitted_invite"}
    if len(candidates) > 1:
        return {"result": "skipped", "reason": "ambiguous", "candidates": sorted(candidates)}
    [device_id] = candidates
    if network.is_burned(device_id) or not any(
        m.device_id == device_id for m in network.active_members()
    ):
        return {"result": "skipped", "reason": "not_an_active_member", "device_id": device_id}
    return {"result": "resolved", "device_id": device_id}


def admitted_redeemer(invite: Any) -> str:
    """``redeemed_by`` of an invite this relay consumed as ``admitted``, else ``""``.

    THE one definition of "this invite admitted that device", shared by the
    backfill, the runner's invite path (``onboard._record_operator_binding_target``,
    security review S8) and the invite prune's keep rule.
    """
    if invite is None or invite.state != "consumed" or invite.outcome != "admitted":
        return ""
    return str(invite.redeemed_by or "")


def admitted_device(
    root: Path | None, *, receipts: Sequence[Mapping[str, Any]], network_id: str
) -> str:
    """The mesh id one onboarding admitted, from Mac-written records only, or ``""``.

    The runner's already-active path (``onboard.OnboardRun._join_already_active``)
    cannot use the node's ``identity show`` id on its own — a node could name a
    co-member — so it resolves through :func:`_backfill_one`'s chain (this
    approval's ``invite`` receipts → ``redeemed_by`` on an ``admitted`` consume →
    still active, unburned). Unresolvable or ambiguous answers ``""``. Never raises.
    """
    try:
        derived = _backfill_one(root, {"receipts": list(receipts)}, network_id)
    except Exception:  # noqa: BLE001 — no id is the safe answer
        return ""
    if derived.get("result") != "resolved":
        return ""
    return str(derived.get("device_id") or "")


def backfill(root: Path | None, *, dry_run: bool = False) -> dict[str, Any]:
    """Record the onboarded node for decisions made before the runner recorded it.

    WHY (zero-friction rollout, 2026-11-15). :func:`own_proofs` presents only to
    the node a decision's ``operator-onboarded`` record names, and the runner
    writes that record only from this build on — so every node onboarded earlier
    would never be presented to, and its asks would stay node-only. This derives
    the record from handshake-authenticated records this Mac already holds
    (:func:`_backfill_one`), and records NOTHING when it cannot be sure: an
    unresolvable or ambiguous decision stays honestly node-only.

    Idempotent, and never on a caller's critical path: run from :func:`present`
    (link-up, off the link's thread), from :func:`retained_approval_ids` (the
    approvals prune, so a decision the backfill can tie to a node is not pruned
    first — which the relay runs OFF-THREAD AT EVERY START,
    ``relay._sweep_approvals_once``, once per namespace), and on the create
    path. A failure is caught here, so it can never fail a relay start. An
    existing record is NEVER overwritten: the same id is ``already_recorded``, a
    different one is a logged ``conflict`` and skipped.

    Credential cards need no backfill (their target is signed); they are listed
    under ``credential_cards`` as ``{network: {recipient: [entry]}}`` with
    ``result`` ``presented`` / ``superseded`` (an older card for the same
    recipient) / ``skipped`` + ``reason`` (``not_an_active_member``), so a dry run
    shows which nodes the cards alone will bind.

    ``dry_run`` writes nothing and logs nothing; the report is safe to print —
    approval ids, network ids, kinds, card family, mesh ids and reasons only,
    never the request's SSH host/user, credential reference or credential
    name/provider/account. Never raises.
    """
    report: dict[str, Any] = {
        "dry_run": dry_run,
        "networks": {},
        "credential_cards": {},
        "credential_cards_refused": [],
    }
    try:
        recorded = onboarded_devices(root)
        rows = _own_proofs_by_record(root)
    except Exception as exc:  # noqa: BLE001
        logger.debug("operator binding: backfill could not read records", exc_info=True)
        # Said in the report, so a dry run never mistakes "could not read" for
        # "nothing to backfill".
        report["error"] = type(exc).__name__
        return report
    try:
        report["credential_cards"] = _card_report(root, rows)
        report["credential_cards_refused"] = _refused_cards(root)
    except Exception as exc:  # noqa: BLE001
        report["credential_cards_error"] = type(exc).__name__
    for record, proof in rows:
        if _card_target(proof["request"]) is not None:
            continue  # signed target: nothing to backfill
        approval_id = proof["approval_id"]
        network_id = _network_of(proof["request"])
        entry: dict[str, Any] = {"approval_id": approval_id, "kind": proof["kind"]}
        derived = _backfill_one(root, record, network_id)
        entry.update(derived)
        existing = recorded.get(approval_id)
        if existing is not None:
            if derived.get("result") == "resolved" and existing[1] != derived["device_id"]:
                entry.update(result="conflict", recorded_device_id=existing[1])
            else:
                entry.update(result="already_recorded", device_id=existing[1])
                entry.pop("reason", None)
                entry.pop("candidates", None)
        elif derived.get("result") == "resolved" and not dry_run:
            if record_onboarded(
                root, approval_id=approval_id, network_id=network_id, device_id=derived["device_id"]
            ):
                entry["result"] = "recorded"
            else:
                entry.update(result="skipped", reason="write_failed")
        report["networks"].setdefault(network_id, []).append(entry)
        if not dry_run:
            _log_backfill(network_id, entry)
    return report


def _log_backfill(network_id: str, entry: Mapping[str, Any]) -> None:
    """One log line per approval per outcome, per process (see ``_backfill_logged``)."""
    outcome = f"{entry.get('result')}:{entry.get('reason', '')}:{entry.get('device_id', '')}"
    approval_id = str(entry.get("approval_id") or "")
    with _lock:
        if _backfill_logged.get(approval_id) == outcome:
            return
        _backfill_logged[approval_id] = outcome
    result = entry.get("result")
    if result == "already_recorded":
        return
    if result == "conflict":
        logger.warning(
            "operator binding: backfill conflict for %s in %s: recorded %s, derived %s; kept",
            approval_id,
            network_id,
            entry.get("recorded_device_id"),
            entry.get("device_id"),
        )
    elif result == "recorded":
        logger.info(
            "operator binding: backfilled %s in %s -> %s",
            approval_id,
            network_id,
            entry.get("device_id"),
        )
    else:
        logger.info(
            "operator binding: backfill skipped %s in %s: %s",
            approval_id,
            network_id,
            entry.get("reason"),
        )


def _target_of(proof: Mapping[str, Any], admitted: Mapping[str, tuple[str, str]]) -> Any:
    """``(family, (network, device))`` a proof may be presented to, or ``None``.

    An onboarding: the node the runner (or backfill) recorded it admitted. A
    credential card: its SIGNED ``(network_id, recipient_device)`` (condition (b)).
    """
    card = _card_target(proof["request"])
    if card is not None:
        return "card", (card[1], card[2])
    target = admitted.get(proof["approval_id"])
    if target is None or target[0] != _network_of(proof["request"]):
        return None
    return "onboarding", target


def _presentable(
    root: Path | None,
    rows: Sequence[tuple[Mapping[str, Any], dict[str, Any]]] | None = None,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Per ``(network, device)``: the NEWEST onboarding proof and the NEWEST card.

    At most one of each family per target (condition (f)), newest first, so a
    node that refuses one family still gets the other. A target that is no
    longer an ACTIVE, unburned member here gets nothing — and so nothing of it
    is retained by the prune either (security review S5: a removed node's proof
    is not kept forever, and presenting no longer leans on link admission).
    """
    admitted = onboarded_devices(root)
    best: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for _record, proof in _own_proofs_by_record(root) if rows is None else rows:
        found = _target_of(proof, admitted)
        if found is None:
            continue
        family, target = found
        slot = best.setdefault(target, {})
        current = slot.get(family)
        if current is None or proof["decided_at"] > current["decided_at"]:
            slot[family] = proof
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for target, slot in best.items():
        if not _active_member(root, target[0], target[1]):
            continue
        out[target] = sorted(slot.values(), key=lambda p: p["decided_at"], reverse=True)
    return out


def _refused_cards(root: Path | None) -> list[dict[str, Any]]:
    """Own, operator-signed credential cards that are NOT proofs, and why (ids only).

    So a dry run that finds no card for a node can tell "there is none" from
    "there is one this interpreter or this shape check refused"
    (:func:`_card_check`).
    """
    from local_operator.network import approvals

    own = _own_device_id(root)
    out: list[dict[str, Any]] = []
    try:
        directory = approvals.approvals_dir(root, enrollment_only=False)
    except Exception:  # noqa: BLE001
        return out
    if not own or not directory.is_dir():
        return out
    for path in sorted(directory.glob("ap_*.json")):
        try:
            record = approvals._load_raw(path.stem, root)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            continue
        what = record.get("what")
        family = next(
            (k for k in _CREDENTIAL_OPERATION_KEYS if isinstance(what, Mapping) and k in what), ""
        )
        signature = record.get("signature")
        requested_by = record.get("requested_by")
        if (
            not family
            or record.get("kind") != approvals.KIND_DEVICE_ONBOARD
            or record.get("state") not in _APPROVED_STATES
            or not isinstance(signature, Mapping)
            or signature.get("cert") is not None
            or not isinstance(requested_by, Mapping)
            or requested_by.get("device_id") != own
        ):
            continue
        target, reason = _card_check(record)
        if target is None:
            out.append(
                {
                    "approval_id": str(record.get("approval_id") or path.stem),
                    "family": family.removeprefix("credential_"),
                    "reason": reason,
                }
            )
    return out


def _card_report(
    root: Path | None, rows: Sequence[tuple[Mapping[str, Any], dict[str, Any]]]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """The dry run's ``credential_cards`` section: ids and verdicts only."""
    chosen = {
        proof["approval_id"] for proofs in _presentable(root, rows).values() for proof in proofs
    }
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for _record, proof in rows:
        card = _card_target(proof["request"])
        if card is None:
            continue
        family, network_id, recipient = card
        entry: dict[str, Any] = {"approval_id": proof["approval_id"], "family": family}
        if proof["approval_id"] in chosen:
            entry["result"] = "presented"
        elif _active_member(root, network_id, recipient):
            entry["result"] = "superseded"
        else:
            entry.update(result="skipped", reason="not_an_active_member")
        out.setdefault(network_id, {}).setdefault(recipient, []).append(entry)
    return out


def retained_approval_ids(root: Path | None) -> frozenset[str]:
    """Approval ids the 30-day prune must keep: the binding proofs in use.

    Manager decision M1 (2026-11-15): ``approvals._sweep_namespace`` deletes
    terminal records after 30 days, and a pruned decision is a binding nobody can
    present again without a fresh onboarding. Keep exactly the decision
    :func:`own_proofs` presents to each onboarded node — the newest per
    ``(network, node)``. Older decisions, and ones with no recorded node (never
    presented to anyone), prune as before: retention is bounded by the number
    of onboarded nodes. The same holds for credential cards (condition (f)):
    the newest card per ``(network, recipient)``. Only for targets that are
    still active members (S5, via :func:`_presentable`).
    """
    backfill(root)
    return frozenset(
        proof["approval_id"] for proofs in _presentable(root).values() for proof in proofs
    )


def retained_invite_ids(root: Path | None) -> frozenset[str]:
    """Invite ids named by a RETAINED own signed onboarding (``store.prune_invites``' keep).

    WHY (PR #4030 review R2-M1(b)). The backfill's chain runs through this
    Mac's invite row (``redeemed_by``), which the invite prune dropped 24 h
    after redemption — so a node onboarded more than a day before a missing
    ``operator-onboarded`` record was noticed could never be tied again. Every
    invite an own, still-stored onboarding decision's receipts name is kept;
    the prune itself keeps only the ones consumed as ``admitted`` by a device
    still an active, unburned member. Read from the approvals store only.
    """
    out: set[str] = set()
    for record, proof in _own_proofs_by_record(root):
        if _card_target(proof["request"]) is None:
            out.update(_invite_ids(record))
    return frozenset(out)


def own_proofs(root: Path | None, network_id: str, *, for_device: str = "") -> list[dict[str, Any]]:
    """The proofs to present to ``for_device`` in ``network_id``: at most two.

    ONLY the decision whose onboarding admitted exactly that mesh device in
    exactly that network (:func:`record_onboarded`), and the newest credential
    card whose SIGNED recipient and network are exactly these, while that
    device is an active member (:func:`_presentable`). Anything else — a
    co-member, a node this Mac did not onboard, an empty ``for_device`` — gets
    ``[]``: the preimage discloses the node's SSH host/user and credential-ref
    name, and M3 allows it only to the operator's own nodes, which the Mac can
    establish only from its own records (security review S1).
    """
    if not network_id or not for_device:
        return []
    return list(_presentable(root).get((network_id, for_device), ()))


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
        _backfill_logged.clear()


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
    if not _is_onboarding_shape(request) and _card_target(request) is None:
        # Neither a real onboarding nor an EXACT credential card (S2, ruling
        # condition (a)): a hybrid or loose shape says nothing about which
        # device is the operator's.
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
    card = _card_target(proof["request"])
    if card is not None and card[2] != _node_self_id(Path(config_root), network_id):
        # Ruling condition (c): a card binds only on ITS recipient. A card the
        # operator signed for a co-member, presented here, proves nothing here.
        return False, "not_recipient"
    # NO anchor-revocation check here (security review S3): the anchor's
    # ``devices`` list holds operator-device KEY ids, never mesh ids, so it could
    # not match. Revoking the operator's device for this purpose is removing (or
    # burning) it in the network: :func:`_active_member`, read live on every use.
    return True, ""


def _node_self_id(root: Path, network_id: str) -> str:
    """This node's mesh id in ``network_id``, or ``""`` when its records disagree.

    The network record's ``self_device_id`` (the id the credential flows check
    against too), and — when an identity is on disk — that identity's id must
    equal it. Uncached: the verdict cache is keyed on the proof, not on us.
    """
    from local_operator.network import store as net_store

    try:
        self_id = net_store.load(network_id, root).self_device_id
    except Exception:  # noqa: BLE001 — no record: no self, nothing binds
        return ""
    own = _own_device_id(root)
    if own and own != self_id:
        return ""
    return str(self_id or "")


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


def _active_member(root: Path | None, network_id: str, device_id: str) -> bool:
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
            # QA Q3: record the link verdict NOW. ``operator_device`` requires
            # it, and it was otherwise written only by the next catalogue
            # listing, so a ``notify`` in between under-claimed "node only"
            # for an item that listing then forwarded.
            from local_operator.network import wire

            _record_link_verdict(
                Path(root),
                network_id,
                device_id,
                wire.OPERATOR_BINDING_V1 in getattr(link, "capabilities", ())
                and _active_member(Path(root), network_id, device_id),
            )
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
    # Lazily tie pre-feature onboardings to their node before choosing (idempotent).
    backfill(server.root)
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
