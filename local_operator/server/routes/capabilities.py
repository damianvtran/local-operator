"""Public feature negotiation: no credentials, and one resolved locale tag.

The map is safe to serve to any local client because it carries no credentials
and no configuration values — with exactly one deliberate exception: the
RESOLVED ``language`` tag (a shipped-locale string, not a raw config value;
RFC §2.5). Everything else is a build fact.
"""

from fastapi import APIRouter

from local_operator.i18n.resolve import resolve_language
from local_operator.server.desktop import desktop_posture
from local_operator.server.features import feature_flags
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["Capabilities"])


@router.get("/v1/capabilities", response_model=CRUDResponse)
async def capabilities():
    # The map itself lives in `local_operator.server.features.feature_flags` --
    # off this route because the mobile daemon imports it too, and the deferred
    # `references` import (with the startup-cost reasoning the guard in
    # `tests/unit/test_import_graph.py` enforces) moved there with it. The route
    # stays the HTTP face of the same function.
    return CRUDResponse(
        status=200,
        message="Backend capabilities retrieved.",
        result={
            "desktop_contract": 1,
            # The signal the UI needs to choose its path WITHOUT guessing, and
            # it has to answer the claim handshake too: false while the plane
            # is closed (including a daemon nobody has claimed yet, where the
            # UI's job is to claim it), true once the app's env capability or
            # an accepted claim governs it. Reading the env variable here
            # directly was how a claimed daemon reported "no desktop" while
            # its routes had already opened — see `server/desktop.py`.
            "desktop_available": desktop_posture().enabled,
            "desktop_auth": "bearer",
            # These version the HTTP subsystems, not renderer completion or
            # third-party authorization. No aggregate "full parity" claim.
            "features": feature_flags(),
            # The language this backend RESOLVES to (RFC §2.5): config
            # `language` -> LOP_LANG -> OS -> en, normalised and filtered to
            # shipped locales. Resolved per call so a config edit reaches the
            # client's next read without a restart; "en" until a translation
            # wave passes audit and widens the shipped set (§6).
            "language": resolve_language(),
        },
    )
