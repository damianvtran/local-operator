"""Source-scoped identities and strict independent link policies.

Names and the active organization selector are display conveniences, never
write authority. The complete normalized API origin and stored scope are part
of an immutable link key: otherwise equal remote ids on another API path or
organization can inherit an unrelated public publication grant. Persistence
remains in the existing baseline/config stores; these are pure value objects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

Kind = Literal["agent", "team", "skill"]
Visibility = Literal["public", "org"]
DEFAULT_SUBSCRIBE = True
DEFAULT_AUTO_PUSH = False
PUBLICATION_CONTRACT = 1
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LINK_ID = re.compile(r"^[a-f0-9]{64}$")


class LinkError(ValueError):
    """A named refusal whose message contains no document or credential bytes."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def normalize_source(source: str) -> str:
    """Canonicalize a credential-free FULL API base, without following redirects.

    This is identity validation, not authorization to send credentials. Network
    callers must additionally compare the result with their configured/approved
    source before selecting credentials; accepting a syntactically valid URL
    must never create an arbitrary-URL credential egress path.
    """
    if not isinstance(source, str) or not source or any(c.isspace() for c in source):
        raise LinkError("origin-conflict", "hub source must be a credential-free API base")
    try:
        url = urlsplit(source)
        port = url.port
        host = url.hostname
    except ValueError:
        raise LinkError("origin-conflict", "hub source is not a valid API base") from None
    if (
        url.scheme.lower() not in ("http", "https")
        or not host
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or "?" in source
        or "#" in source
        or "\\" in source
        or any(ord(c) < 32 or ord(c) == 127 for c in source)
    ):
        raise LinkError("origin-conflict", "hub source must be a credential-free API base")
    try:
        canonical_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise LinkError("origin-conflict", "hub source has an invalid host") from None
    if ":" in canonical_host:
        canonical_host = f"[{canonical_host}]"
    scheme = url.scheme.lower()
    if port is not None and (scheme, port) not in (("https", 443), ("http", 80)):
        canonical_host += f":{port}"
    path = url.path.rstrip("/")
    # Dot segments have server-dependent normalization; refuse rather than
    # permit two spellings of the same destination to acquire separate grants.
    if any(part in (".", "..") for part in path.split("/")):
        raise LinkError("origin-conflict", "hub source has an ambiguous API path")
    return urlunsplit((scheme, canonical_host, path, "", ""))


def approve_source(source: str, approved: tuple[str, ...]) -> str:
    """Complete the version segment, then require the configured origin exactly.

    THE SEGMENT IS COMPLETION, NOT GUESSING. The renderer composes its
    ``hub_source`` from the UNVERSIONED server base
    (``https://api.radienthq.com`` or the rig's ``http://127.0.0.1:8127``), while
    the configured base carries ``/v1``; the two spellings name the same API
    origin. ``normalize_radient_api_base_url`` is the ONE completion rule every
    producer already applies (``env`` for the configured value and the
    environment), so applying it to the INCOMING value here is what lets the
    desktop reach pull, subscribe and publish without teaching each surface a
    private spelling.

    It is not a relaxation: only a ROOT with no path gets ``/v1`` appended; a
    custom gateway path is untouched, and host/scheme are never rewritten. The
    equality below is still the authorization check, and a source that does not
    name the configured origin keeps the same nondisclosing refusal.

    The RAW value is validated by :func:`normalize_source` BEFORE completion:
    that function owns the refusal vocabulary for whitespace, credentials and
    malformed URLs (a bracketed host such as ``http://[::1`` raises ValueError
    from ``urlsplit``), and completion strips whitespace. Validating first keeps
    those inputs refused here with the same ``origin-conflict`` sentence instead
    of leaking a parser message or silently accepting a padded value.
    """

    # Imported here: `local_operator.env` loads `.env` at import time, and the
    # pure identity helpers must not pull that cost into every hub_sync import.
    from local_operator.env import normalize_radient_api_base_url

    def canonical(value: str) -> str:
        raw = normalize_source(value)
        return normalize_source(normalize_radient_api_base_url(raw))

    normalized = canonical(source)
    if normalized not in {canonical(candidate) for candidate in approved}:
        raise LinkError("origin-conflict", "hub source is not configured or approved")
    return normalized


@dataclass(frozen=True)
class LinkOrigin:
    """Where a link was served from: the base, and the scope it was served under.

    THE SCOPE IS A FACT ABOUT THE ARM THAT SERVED THE ROW, never about the
    document it served. A team document carries ``tenant_id`` and no visibility
    field, and a team in the PUBLIC catalogue carries its PUBLISHER's
    ``tenant_id`` (measured on the live hub on 2026-10-19: ``list_public_teams``
    answers ``software-development`` with ``tenant_id=5b72d9f4-...`` and
    ``visibility=None``). So "the document names a tenant" is not a scope: it
    would put a public row behind an organization identity and past the public
    egress gate in ``_eligibility``.

    The organization's tenant is the tenant the scope names -- ``None`` for a
    public scope, and REQUIRED for an organization one. It is also the tenant
    the RECORD has to carry: an origin and a record that disagree are a scope
    conflict, which the write path refuses rather than silently relinks.
    """

    hub_source: str
    visibility: Visibility
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        if self.visibility not in ("public", "org"):
            raise LinkError("origin-conflict", "hub link visibility is unsupported")
        if self.visibility == "public" and self.tenant_id:
            raise LinkError("origin-conflict", "a public origin must not name an organization")
        if self.visibility == "org" and not self.tenant_id:
            raise LinkError("origin-conflict", "an organization origin must name its organization")
        object.__setattr__(self, "hub_source", normalize_source(self.hub_source))


def link_origin(
    hub_source: str | None, visibility: str, tenant_id: str | None = None
) -> "LinkOrigin | None":
    """The origin a served row was read under, or ``None`` when it cannot be stated.

    A pull, an accepted publication and the origin repair all name their origin
    the same way, and none of them may FAIL over it: an origin that cannot be
    stated (no base, a base that is not a credential-free API base, a scope that
    does not match the tenant it names) yields ``None``, and the caller records
    the row WITHOUT an identity -- the honest "origin unknown" state the
    inventory reports as unresolved.
    """

    if not hub_source or visibility not in ("public", "org"):
        return None
    try:
        return LinkOrigin(
            hub_source=hub_source,
            visibility=cast(Visibility, visibility),
            tenant_id=tenant_id,
        )
    except LinkError:
        logger.warning(
            "could not state the hub origin for a %s scope; the link stays unresolved",
            visibility,
            exc_info=True,
        )
        return None


@dataclass(frozen=True)
class LinkIdentity:
    kind: Kind
    local_id: str
    hub_source: str
    visibility: Visibility
    tenant_id: str | None
    hub_id: str

    def __post_init__(self) -> None:
        if self.kind not in ("agent", "team", "skill"):
            raise LinkError("origin-conflict", "hub link kind is unsupported")
        for value in (self.local_id, self.hub_id):
            if not isinstance(value, str) or not _ID.fullmatch(value):
                raise LinkError("origin-conflict", "hub link identifier is invalid")
        if self.visibility == "public":
            if self.tenant_id is not None:
                raise LinkError("origin-conflict", "public hub links cannot name an organization")
        elif self.visibility == "org":
            if not isinstance(self.tenant_id, str) or not _ID.fullmatch(self.tenant_id):
                raise LinkError("origin-conflict", "organization hub links require a valid tenant")
        else:
            raise LinkError("origin-conflict", "hub link visibility is unsupported")
        object.__setattr__(self, "hub_source", normalize_source(self.hub_source))

    @property
    def link_id(self) -> str:
        return _digest(
            [
                "local-operator/hub-link/v1",
                self.kind,
                self.local_id,
                self.hub_source,
                self.visibility,
                self.tenant_id,
                self.hub_id,
            ]
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "link_id": self.link_id,
            "kind": self.kind,
            "local_id": self.local_id,
            "hub_source": self.hub_source,
            "visibility": self.visibility,
            "tenant_id": self.tenant_id,
            "hub_id": self.hub_id,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "LinkIdentity":
        try:
            identity = cls(
                kind=value["kind"],
                local_id=value["local_id"],
                hub_source=value["hub_source"],
                visibility=value["visibility"],
                tenant_id=value["tenant_id"],
                hub_id=value["hub_id"],
            )
        except (KeyError, TypeError):
            raise LinkError("origin-conflict", "hub link identity is incomplete") from None
        if value.get("link_id") != identity.link_id:
            raise LinkError("origin-conflict", "hub link identity digest does not match")
        return identity

    def require_same_origin(self, other: "LinkIdentity") -> None:
        if self.link_id != other.link_id:
            raise LinkError("origin-conflict", "create a separate source-scoped local copy")


@dataclass(frozen=True)
class PublicationTarget:
    """A draft destination before creation returns a remote id; never a fake link."""

    kind: Kind
    local_id: str
    hub_source: str
    visibility: Visibility
    tenant_id: str | None

    def __post_init__(self) -> None:
        if self.kind not in ("agent", "team", "skill"):
            raise LinkError("origin-conflict", "publication kind is unsupported")
        if not isinstance(self.local_id, str) or not _ID.fullmatch(self.local_id):
            raise LinkError("origin-conflict", "publication local id is invalid")
        if self.visibility == "public":
            if self.tenant_id is not None:
                raise LinkError(
                    "origin-conflict", "public publication cannot carry an organization"
                )
        elif self.visibility == "org":
            if not isinstance(self.tenant_id, str) or not _ID.fullmatch(self.tenant_id):
                raise LinkError(
                    "origin-conflict", "organization publication requires an explicit scope"
                )
        else:
            raise LinkError("origin-conflict", "publication visibility is unsupported")
        object.__setattr__(self, "hub_source", normalize_source(self.hub_source))

    @classmethod
    def from_identity(cls, identity: LinkIdentity) -> "PublicationTarget":
        return cls(
            identity.kind,
            identity.local_id,
            identity.hub_source,
            identity.visibility,
            identity.tenant_id,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "local_id": self.local_id,
            "hub_source": self.hub_source,
            "visibility": self.visibility,
            "tenant_id": self.tenant_id,
        }

    def bind(self, hub_id: str) -> LinkIdentity:
        return LinkIdentity(
            self.kind, self.local_id, self.hub_source, self.visibility, self.tenant_id, hub_id
        )


@dataclass(frozen=True)
class PublicEgressGrant:
    """Explicit future-publication consent, not proof of safe content lineage."""

    link_id: str
    publication_contract: int = PUBLICATION_CONTRACT

    def __post_init__(self) -> None:
        if not isinstance(self.link_id, str) or not _LINK_ID.fullmatch(self.link_id):
            raise LinkError("invalid-link-policy", "public egress grant has an invalid link")
        if type(self.publication_contract) is not int or self.publication_contract < 1:
            raise LinkError("invalid-link-policy", "public egress grant has an invalid contract")

    def to_json(self) -> dict[str, Any]:
        return {"link_id": self.link_id, "publication_contract": self.publication_contract}


@dataclass(frozen=True)
class LinkPolicy:
    subscribe: bool = DEFAULT_SUBSCRIBE
    auto_push: bool = DEFAULT_AUTO_PUSH

    def __post_init__(self) -> None:
        if type(self.subscribe) is not bool or type(self.auto_push) is not bool:
            raise LinkError("invalid-link-policy", "link policy values must be booleans")

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "LinkPolicy":
        if not isinstance(value, Mapping) or set(value) != {"subscribe", "auto_push"}:
            raise LinkError("invalid-link-policy", "link policy requires two independent values")
        return cls(subscribe=value["subscribe"], auto_push=value["auto_push"])

    def revision(self, identity: LinkIdentity, grant: PublicEgressGrant | None = None) -> str:
        if grant is not None and (
            identity.visibility != "public" or grant.link_id != identity.link_id
        ):
            raise LinkError("invalid-link-policy", "public egress grant does not match this origin")
        # A digest sees direct config edits too. A revision counter only works
        # if every external editor remembers to increment it, which they do not.
        return _digest(
            {
                "domain": "local-operator/hub-policy/v1",
                "identity": identity.to_json(),
                "policy": self.to_json(),
                "egress_grant": grant.to_json() if grant is not None else None,
            }
        )

    def to_json(self) -> dict[str, bool]:
        return {"subscribe": self.subscribe, "auto_push": self.auto_push}
