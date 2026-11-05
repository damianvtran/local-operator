"""The person-scoped Radient client every organization read and pull uses.

WHY THIS LIVES HERE. ``cli._resolve_org_client`` used to be the only builder of
an organization client, and it printed its own remedy. The org catalogue and
bulk pull run from three hosts -- the CLI, the ``agent`` tool and slice B's
background auto-enrol -- and two of them have no terminal to print to. The
decision (which credential, which destination may receive it) therefore lives
here once, returns a REASON instead of printing, and the CLI keeps rendering
its existing sentences from that reason. A second copy of the destination rule
beside the first is exactly how a bearer token ends up sent to the wrong host.

CONSTRAINT. Organization operations authenticate with the stored Radient OAuth
ACCESS token -- never the tenant API key, because an API key proves an
application tenant, not a person's membership. A non-canonical configured hub
is refused before any credential is attached
(``RADIENT_ORG_ALLOW_NONCANONICAL_BASE`` is the explicit local/QA opt-in).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.clients.radient import RadientClient

#: Why no client could be built. ``destination-refused`` and ``signed-out`` have
#: different fixes, so callers render them differently (review round 1 of the
#: original resolver, R1-2/S-1).
OrgClientReason = Literal["ok", "destination-refused", "signed-out"]


@dataclass(frozen=True)
class OrgClient:
    """A built client (or None) plus the reason and the base it would address."""

    client: "RadientClient | None"
    reason: OrgClientReason
    base_url: str


def build_org_client(config_manager: Any) -> OrgClient:
    """A Radient client acting as the signed-in PERSON, or the reason there is none.

    Never prints and never raises for the expected cases (no login, refused
    destination); an unexpected credential-store failure propagates so a real
    bug is not presented as "signed out".
    """

    from pydantic import SecretStr  # lazy: pydantic stays off the startup path

    from local_operator.clients.radient import RadientClient  # lazy: HTTP stack
    from local_operator.providers.radient_credentials import (  # lazy: auth stack
        configured_radient_base_url,
        org_oauth_destination_allowed,
        resolve_radient_oauth_access_sync,
    )

    base_url = configured_radient_base_url(config_manager)
    access = resolve_radient_oauth_access_sync(config_manager.config_dir, base_url)
    if access is None:
        reason: OrgClientReason = (
            "signed-out" if org_oauth_destination_allowed(base_url) else "destination-refused"
        )
        return OrgClient(None, reason, base_url)
    return OrgClient(
        RadientClient(api_key=SecretStr(access.access_token), base_url=base_url),
        "ok",
        base_url,
    )
