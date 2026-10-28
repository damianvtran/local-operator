"""The registry of TYPED approval operations: the one signer path for every approval card.

WHY THIS EXISTS. Some approval records cannot be signed by a generic "Approve" because the
generic card cannot state what the signature authorises. A credential permission (a
finite-share recovery, a standing inheritance) rides the ``device_onboard`` storage, so
the bodiless desktop ``/approve`` would read its operator-key anchor as an install scope
and sign it blind. Until this module, that was contained by hand-written literal guards in
every generic approver, and each new typed operation needed another one -- the bug class
where the guard someone forgot is the blind signature.

This module turns those guards into DATA. Every typed operation registers an
:class:`ApprovalOp` here, and:

* the typed engine (``network.credentials.desktop_approval``) looks the operation up to
  find its intent, disclosure, ``expected`` apply block, apply and file-only policy, so
  the typed desktop routes (``/approve-operation``, ``/apply-operation`` and their
  ``-credential`` aliases) are one engine for every op;
* BOTH generic approvers -- the frozen bodiless ``POST /v1/desktop/approvals/{id}/approve``
  and ``lop network approvals approve`` -- run ONE check, :func:`require_generic_approvable`,
  before any signer call. A record carrying a registered typed op is refused there; deny and
  withdraw are never gated (they are the safe direction).

CONSTRAINTS.

* CLOSED AND STATIC. Registration happens only at import of the modules named in
  :data:`_PROVIDERS`; there are no entry points and nothing outside this repository can add
  an op. A typo'd or missing provider fails the import loudly rather than leaving an op
  unguarded.
* KEY PRESENCE IS THE BOUNDARY. An op claims a record when its key is present in the
  record's ``what`` block, whether or not the payload parses -- a malformed payload must
  fail closed, never fall through to a generic signer.
* FROZEN REFUSALS. The credential ops keep the exact refusal codes and sentences their
  literal guards answered (clients and tests key on them); :data:`TYPED_APPROVAL_REQUIRED`
  is the default only for ops that register no refusal of their own.
* NO NEW AUTHORITY. This module never signs and never reads the secret store; it only
  decides which path may sign. It imports nothing heavy at module load so the stdlib-only
  approvals store can consult it.

HOW TO ADD AN OP (S4a-pair-op is the first user): implement :class:`ApprovalOp` in the
op's own module, call :func:`register` at import, add that module to :data:`_PROVIDERS`,
and decide ``file_only_signing`` as an explicit operator decision -- never by default.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from local_operator.network.types import MeshRefusal

#: The generic refusal for a typed op that registers no frozen sentence of its own.
TYPED_APPROVAL_REQUIRED = "typed_approval_required"

#: The surfaces that run the generic check. The DESKTOP route has no disclosure at all, so
#: every typed op is refused there; the CLI verb prints an op's disclosure before signing,
#: so an op may opt in to it (``cli_approvable``) -- which is how the credential ops keep
#: the terminal approval the operator chose (asks a-b815 / a-4ba6).
SURFACE_DESKTOP = "desktop"
SURFACE_CLI = "cli"

#: Modules that register ops at import. CLOSED: the only way an op enters the registry.
_PROVIDERS: tuple[str, ...] = ("local_operator.network.credentials.desktop_approval",)


@runtime_checkable
class ApprovalOp(Protocol):
    """One typed approval operation: what it is, how it is disclosed, how it is applied.

    ``op`` is the ``what`` key the record carries; ``kind`` the approvals KIND it rides
    (metadata in S4a-core, enforced from S4a-pair-op). ``file_only_signing`` is the
    operator's decision whether a FILE-ONLY operator key may sign it from the desktop
    (a presence key always may). ``cli_approvable`` says whether ``lop network approvals
    approve`` may sign it because that verb prints this op's full disclosure first.
    """

    op: str
    kind: str
    file_only_signing: bool
    cli_approvable: bool

    def intent(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """The parsed payload, or ``None`` when the record does not carry this op validly."""

    def disclosure(self, record: dict[str, Any]) -> dict[str, Any]:
        """Every fact the signer must see, as data; raises ``MeshRefusal`` if undisclosable."""
        ...

    def expected(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """The apply ``expected`` block, derived from the SIGNED record."""
        ...

    def apply(self, approval_id: str, expected: dict[str, Any], root: Path) -> Any:
        """Apply a signed card; raises ``MeshRefusal`` on refusal."""

    def generic_refusal(self, record: Mapping[str, Any], approval_id: str) -> MeshRefusal | None:
        """The refusal a GENERIC approver answers for this record, or ``None`` if unclaimed.

        Return ``None`` when the record does not carry this op in a form this op owns, so
        the next op (or a family rule) can answer; the default answer for a claimed record
        is :func:`typed_required`.
        """


_LOCK = threading.Lock()
_OPS: dict[str, ApprovalOp] = {}
#: ``what``-key prefixes owned by a family of ops, with the refusal for an UNREGISTERED
#: member (e.g. an unknown ``credential_*`` op): undisclosable, so never signable.
_FAMILIES: dict[str, Any] = {}
_loaded = False


def register(operation: ApprovalOp) -> ApprovalOp:
    """Add ``operation``; registering the same op name twice is a programming error.

    Registration ORDER is the precedence of :func:`require_generic_approvable` when one
    record carries several op keys, so a provider registers in the order its old literal
    guards ran.
    """
    with _LOCK:
        if operation.op in _OPS and _OPS[operation.op] is not operation:
            raise ValueError(f"approval op {operation.op!r} is already registered")
        _OPS[operation.op] = operation
    return operation


def register_family(prefix: str, unsupported: Any) -> None:
    """Own every ``what`` key starting with ``prefix``.

    ``unsupported(record, approval_id) -> MeshRefusal`` answers for a key of the family that
    no op registered: the generic approver must not sign what no op can disclose.
    """
    with _LOCK:
        _FAMILIES[prefix] = unsupported


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    for name in _PROVIDERS:
        importlib.import_module(name)
    _loaded = True


def registered() -> tuple[ApprovalOp, ...]:
    """Every registered op, in registration order."""
    _ensure_loaded()
    with _LOCK:
        return tuple(_OPS.values())


def get(op: str) -> ApprovalOp | None:
    """The op registered under ``op``, or ``None``."""
    _ensure_loaded()
    with _LOCK:
        return _OPS.get(op)


def file_only_signing() -> dict[str, bool]:
    """``{op: may a FILE-ONLY key sign it from the desktop}``, derived from the registry."""
    return {operation.op: operation.file_only_signing for operation in registered()}


def typed_required(approval_id: str) -> MeshRefusal:
    """The default generic-approver refusal for a typed op with no frozen sentence."""
    return MeshRefusal(
        TYPED_APPROVAL_REQUIRED,
        "This approval can only be signed through its typed review, which shows exactly "
        f"what it authorises ({approval_id}). Nothing was approved; deny stays available.",
    )


def require_generic_approvable(
    record: Mapping[str, Any], approval_id: str, *, surface: str
) -> None:
    """THE ONE CHECK both generic approvers run BEFORE any signer call.

    Raises the claiming op's refusal when ``record`` carries a registered typed op that
    ``surface`` may not sign; on the desktop, an unregistered member of a registered
    family refuses too. Returns ``None`` when the record is the generic approver's to sign.
    Deny/withdraw never call this.
    """
    what = record.get("what")
    if not isinstance(what, dict):
        return
    for operation in registered():
        if operation.op not in what:
            continue
        if surface == SURFACE_CLI and operation.cli_approvable:
            # The CLI verb discloses this op itself before signing (operator decision).
            continue
        refusal = operation.generic_refusal(record, approval_id)
        if refusal is not None:
            raise refusal
    if surface != SURFACE_DESKTOP:
        # PURE REFACTOR: the CLI verb never refused unknown family keys; changing that is a
        # behaviour change for its own PR.
        return
    with _LOCK:
        families = dict(_FAMILIES)
    for prefix, unsupported in families.items():
        if any(isinstance(key, str) and key.startswith(prefix) for key in what):
            raise unsupported(record, approval_id)


# ---------------------------------------------------------------------------
# The credential ops' FROZEN generic-approver refusals
# ---------------------------------------------------------------------------
#
# WHY THEY LIVE HERE AND NOT IN ``network/credentials``. They are the literal guards the
# bodiless desktop ``/approve`` ran before S4a-core, moved verbatim (codes and sentences are
# a client contract). They belong to the GENERIC approver, not to the typed credential path:
# ``approval_copy.KNOWN_CODES`` is the typed card's refusal vocabulary and never listed them,
# and ``test_every_refusal_code_is_known`` walks ``network/credentials`` to keep it honest.
# The credential modules are imported lazily so this module stays importable by the store.


def finite_generic_refusal(record: Mapping[str, Any], approval_id: str) -> MeshRefusal | None:
    """The bodiless desktop ``/approve``'s FROZEN answer for a finite-share card.

    Moved verbatim from ``server.utils.desktop_approvals.approve`` (S4a-core): presence of
    the typed operation key is the boundary, NOT intent()'s successfully parsed value, so a
    malformed finite payload fails closed rather than falling through to the generic
    onboarding signer. Only the EXACT shape finite apply accepts (``set(what) == {OP,
    "anchor"}``) is sent to the CLI; an untyped value or any extra key -- an unknown
    ``credential_*`` operation or an install scope -- would be signed by the CLI without
    being disclosed, so both refuse with no signing remedy (agent review round 1, R1).
    """
    from local_operator.network.credentials import finite_share as _finite

    what = record.get("what")
    if not isinstance(what, dict) or _finite.OP not in what:
        return None
    if _finite.intent(dict(record)) is None or set(what) != {_finite.OP, "anchor"}:
        return MeshRefusal(
            "finite_disclosure_invalid",
            "The finite-share request is not the exact typed shape this build can "
            "validate and disclose (one finite-share operation and its anchor, nothing "
            "else). Nothing was approved; request a new finite-share recovery with the "
            "complete withdrawal disclosure.",
        )
    # Until the desktop's GENERIC route can render the full validated withdrawal disclosure,
    # the CLI (or the typed route) is the only signing surface. Deny remains available.
    return MeshRefusal(
        "finite_terminal_approval_required",
        "This finite-share recovery can only be approved from the command line on this "
        "machine for now, where the complete withdrawal disclosure is validated and shown "
        "before you sign: "
        f"lop network approvals approve {approval_id}. Nothing was approved.",
    )


def inheritance_generic_refusal(record: Mapping[str, Any], approval_id: str) -> MeshRefusal | None:
    """The bodiless desktop ``/approve``'s FROZEN answer for a credential permission.

    Claims only a record whose inheritance payload parses (today's ``intent() is not None``
    guard); a malformed one falls to the ``credential_*`` family refusal, exactly as before.
    The generic card would read the request's operator-key anchor as an "install anchor"
    scope and show a blind Approve, so it refuses before the signer; deny stays available.
    """
    from local_operator.network.credentials import inheritance as _inheritance

    if _inheritance.intent(dict(record)) is None:
        return None
    return MeshRefusal(
        "inherit_terminal_approval_required",
        "This credential permission can only be approved from the command line on this "
        "machine for now, where its account, recipient and duration are shown before you "
        "sign: "
        f"lop network approvals approve {approval_id}. Nothing was approved.",
    )


def credential_unsupported(record: Mapping[str, Any], approval_id: str) -> MeshRefusal:
    """An unregistered ``credential_*`` op: no validator, so no surface may sign it blind."""
    del record, approval_id
    return MeshRefusal(
        "credential_approval_unsupported",
        "This credential request uses an operation this build cannot validate and "
        "disclose for approval. Nothing was approved; a typed approval flow with full "
        "validation is required.",
    )
