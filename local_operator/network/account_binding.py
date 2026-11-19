"""Owner-process file-evidence adapter; deliberately not a new authority domain.

This module is not registered with any relay, tool or HTTP route. Existing signed
file approval can establish only its original file purpose, without another
prompt. A result records one freshly checked read context, NOT permission to
read/hash/serve content. P3 must still establish provenance, strict pre-content
audit, sealed content versions and a final writer/read-revoke fence. In particular,
checking this object and then enqueueing a mutable response is not that fence.

The trusted owner-process boundary matches file_device_approval: arbitrary code
running as the owner is not an adversary these Python value types can contain.
Private construction prevents accidental dict/bool/cached-subject substitution;
it is not a new bearer token or an in-process cryptographic security boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.network import file_device_approval as F
from local_operator.network.types import MeshRefusal
from local_operator.providers.auth_store import AuthStore


class _Value(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class CanonicalSubject(_Value):
    issuer: str = Field(min_length=1)
    opaque_id: str = Field(min_length=1)


class ExactNativeAccountSelection(_Value):
    credential_id: int = Field(gt=0)


class OwnerFileEvidenceReference(_Value):
    """Owner-local selector only; knowing the association does not authorize it."""

    association: str = Field(pattern=r"^[a-f0-9]{64}$")


class ReadBinding(_Value):
    """Exact intended read identity, not verification of the resource it names.

    file_version is an opaque equality constraint here, not a content commitment
    issued by this adapter. The future content guard must establish/verify that
    commitment before any response, including stat, verification and resume.
    """

    owner_key: F.MeshKey
    network_id: str = Field(min_length=1)
    network_epoch: int = Field(gt=0)
    link_id: str = Field(min_length=1)
    transcript_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    source_row_id: str = Field(min_length=1)
    provenance_id: str = Field(min_length=1)
    proof_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    file_version: str = Field(min_length=1)
    phase: Literal["stat", "fetch", "verify", "resume"]
    offset: int = Field(ge=0)
    length: int = Field(ge=0)
    transfer_nonce: str = Field(pattern=r"^[a-f0-9]{32}$")


def _deny(code: str) -> NoReturn:
    # Reuse the existing keyed refusal envelope; do not expose owner paths or
    # create a parallel error-copy/authorization domain for this internal API.
    raise F._refuse(code)


class BindingRefusal(MeshRefusal):
    """Typed failure of this adapter; never fall back to legacy file authority."""


@dataclass(frozen=True, slots=True, init=False)
class VerifiedAccountDeviceBinding:
    """Non-serializable per-read observation, privately built by verify_live.

    There is intentionally no allow(), wire token, dict projection, content API
    or response-commit method. A later read/phase must call verify_live again;
    this result does not close revocation races after that call returns.
    """

    subject: CanonicalSubject
    purpose: Literal["future_live_mentioned_file_access"]
    assurance: Literal["installed_presence"]
    owner_key: F.MeshKey
    requester_key: F.MeshKey
    native_account: F.NativeAccount
    anchor: F.Anchor
    network_id: str
    network_epoch: int
    owner_admission: tuple[float, str, str]
    requester_admission: tuple[float, str, str]
    association: str
    authority_epoch: str
    generation: int
    evidence_digest: str
    journal_revision: int
    deny_generation: int
    relay_instance: str
    read_binding: ReadBinding
    _incoming_link: Any = field(repr=False, compare=False)

    def __new__(cls, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("verified bindings are constructed only by verify_live")

    def __bool__(self) -> NoReturn:
        raise TypeError("a verified binding is not a boolean file grant")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError("verified bindings cannot leave the owner process")

    def require_read(self, incoming_link: Any, read_binding: ReadBinding) -> None:
        """Reject substitution, not authorize emission or refresh stale context."""
        if type(read_binding) is not ReadBinding:
            error = F._refuse("account_binding_read_mismatch")
            raise BindingRefusal(error.code, error.sentence)
        try:
            read_binding = ReadBinding.model_validate(read_binding.model_dump())
        except (ValidationError, ValueError, TypeError) as invalid:
            error = F._refuse("account_binding_read_mismatch")
            raise BindingRefusal(error.code, error.sentence) from invalid
        if (
            incoming_link is not self._incoming_link
            or read_binding != self.read_binding
            or incoming_link.server.instance_id != self.relay_instance
            or incoming_link.link_id != self.read_binding.link_id
            or incoming_link.result.link_id != self.read_binding.link_id
            or incoming_link.result.transcript_hash != self.read_binding.transcript_hash
        ):
            error = F._refuse("account_binding_read_mismatch")
            raise BindingRefusal(error.code, error.sentence)


async def verify_live(
    *,
    incoming_link: Any,
    auth: AuthStore,
    owner_account: ExactNativeAccountSelection,
    evidence: OwnerFileEvidenceReference,
    required_subject: CanonicalSubject,
    required_purpose: str,
    read_binding: ReadBinding,
) -> VerifiedAccountDeviceBinding:
    """Verify existing file evidence against the actual incoming listener.

    All asynchronous canonical-account IO occurs in the existing verifier before
    its file-domain lock. The immutable successful snapshot is built inside that
    lock, after signature/current membership/native row/anchor/negative checks.
    Subject and read inputs constrain the result; they cannot establish authority.
    No remote fields select a root, peer key, native account identity or consent.
    """
    try:
        if (
            type(owner_account) is not ExactNativeAccountSelection
            or type(evidence) is not OwnerFileEvidenceReference
            or type(required_subject) is not CanonicalSubject
            or type(read_binding) is not ReadBinding
            or type(required_purpose) is not str
            or required_purpose != F.PURPOSE
        ):
            _deny("account_binding_invalid_context")
        # Pydantic's model_construct/model_copy deliberately bypass validation.
        # Rebuild from scalar data at this trust boundary rather than trusting a
        # class name or retaining a caller's potentially malformed nested value.
        try:
            owner_account = ExactNativeAccountSelection.model_validate(owner_account.model_dump())
            evidence = OwnerFileEvidenceReference.model_validate(evidence.model_dump())
            required_subject = CanonicalSubject.model_validate(required_subject.model_dump())
            read_binding = ReadBinding.model_validate(read_binding.model_dump())
        except (ValidationError, ValueError, TypeError):
            _deny("account_binding_invalid_context")
        # Validate actual class/root/listener membership before capturing link
        # identity. Never accept a caller's shape double as a production PeerLink.
        root = F._link_root(incoming_link)
        mesh = F._mesh(incoming_link, root)
        link_identity = (
            incoming_link.server.instance_id,
            incoming_link.link_id,
            incoming_link.result.transcript_hash,
        )
        if (
            read_binding.owner_key != mesh["owner"]
            or read_binding.network_id != mesh["network_id"]
            or read_binding.network_epoch != mesh["network_epoch"]
            or (read_binding.link_id, read_binding.transcript_hash) != link_identity[1:]
            or incoming_link.result.link_id != read_binding.link_id
        ):
            _deny("account_binding_read_mismatch")
        context = await F._verified_context(incoming_link, auth, owner_account.credential_id)
        intent = context.intent
        subject = CanonicalSubject(issuer=intent.account.hub, opaque_id=intent.account.account_id)
        if subject != required_subject:
            _deny("account_binding_subject_mismatch")
        if evidence.association != context.association or intent.purpose != required_purpose:
            _deny("account_binding_evidence_mismatch")
        # A reconnect is valid for a standing association but cannot reuse a
        # particular in-flight read receipt. Capture the current link, not the
        # historical issuance link stored in FileIntent.
        if (
            F._mesh(incoming_link, root) != mesh
            or any(mesh[key] != getattr(intent, key) for key in mesh)
            or link_identity
            != (
                incoming_link.server.instance_id,
                incoming_link.link_id,
                incoming_link.result.transcript_hash,
            )
            or incoming_link.result.link_id != read_binding.link_id
        ):
            _deny("account_binding_read_mismatch")
        result = object.__new__(VerifiedAccountDeviceBinding)
        values = {
            "subject": subject,
            "purpose": intent.purpose,
            "assurance": intent.assurance,
            "owner_key": intent.owner,
            "requester_key": intent.requester,
            "native_account": intent.account,
            "anchor": intent.anchor,
            "network_id": intent.network_id,
            "network_epoch": intent.network_epoch,
            "owner_admission": (
                intent.owner_added_at,
                intent.owner_added_by,
                intent.owner_added_via,
            ),
            "requester_admission": (
                intent.requester_added_at,
                intent.requester_added_by,
                intent.requester_added_via,
            ),
            "association": context.association,
            "authority_epoch": context.authority_epoch,
            "generation": context.generation,
            "evidence_digest": context.request_digest,
            "journal_revision": context.journal_revision,
            "deny_generation": context.deny_generation,
            "relay_instance": link_identity[0],
            "read_binding": read_binding,
            "_incoming_link": incoming_link,
        }
        for name, value in values.items():
            object.__setattr__(result, name, value)
        return result
    except MeshRefusal as error:
        raise BindingRefusal(error.code, error.sentence) from error
