"""Public feature negotiation: no credentials, and one resolved locale tag.

The map is safe to serve to any local client because it carries no credentials
and no configuration values — with exactly one deliberate exception: the
RESOLVED ``language`` tag (a shipped-locale string, not a raw config value;
RFC §2.5). Authenticated desktop callers additionally receive the current
owner's fresh, cached hub conversation capabilities. This is observed support,
not a build fact; public callers receive only the ordinary runtime map.
"""

from fastapi import APIRouter, HTTPException, Request

from local_operator.i18n.resolve import locale_direction, resolve_language
from local_operator.server.desktop import desktop_posture
from local_operator.server.features import feature_flags
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["Capabilities"])


@router.get("/v1/capabilities", response_model=CRUDResponse)
async def capabilities(request: Request):
    # The map itself lives in `local_operator.server.features.feature_flags` --
    # off this route because the mobile daemon imports it too, and the deferred
    # `references` import (with the startup-cost reasoning the guard in
    # `tests/unit/test_import_graph.py` enforces) moved there with it. The route
    # stays the HTTP face of the same function.
    # Resolve once: a live config edit between two reads must not pair one
    # locale's copy with another's direction. Held preferences remain en/ltr.
    language = resolve_language()
    features = feature_flags()
    if request.headers.get("authorization"):
        # The public build negotiation stays public. Account-scoped hub facts
        # are added only for an authorized desktop caller, using the existing
        # status cache: polling capabilities must never enroll or publish.
        from local_operator.server.desktop import require_desktop
        from local_operator.team_bridge.client import (
            BridgeRefusal,
            BridgeSignInRequired,
            get_status_discipline,
        )

        try:
            require_desktop(request)
        except HTTPException:
            pass  # Public negotiation still works; no account facts are added.
        else:
            config_manager = getattr(request.app.state, "config_manager", None)
            if config_manager is not None:
                try:
                    features.update(
                        get_status_discipline(
                            config_manager.config_dir
                        ).cached_conversation_features()
                    )
                except (BridgeRefusal, BridgeSignInRequired):
                    pass  # Unknown/signed-out is absence, never invented support.
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
            # Runtime subsystem versions plus the guarded, observed hub flag
            # above; no renderer-completion or aggregate "full parity" claim.
            "features": features,
            # The language this backend RESOLVES to (RFC §2.5): config
            # `language` -> LOP_LANG -> OS -> en, normalised and filtered to
            # shipped locales. Resolved per call so a config edit reaches the
            # client's next read without a restart; "en" until a translation
            # wave passes audit and widens the shipped set (§6).
            "language": language,
            # Additive negotiation metadata; older clients keep reading the
            # existing language field. This is text direction, not a claim that
            # every client implements RTL layout or that Urdu is eligible.
            "language_direction": locale_direction(language),
        },
    )
