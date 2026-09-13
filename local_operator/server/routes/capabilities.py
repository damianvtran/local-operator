"""Public feature negotiation contains no credentials or configuration values."""

import os

from fastapi import APIRouter

from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["Capabilities"])


@router.get("/v1/capabilities", response_model=CRUDResponse)
async def capabilities():
    return CRUDResponse(
        status=200,
        message="Backend capabilities retrieved.",
        result={
            "desktop_contract": 1,
            "desktop_available": bool(os.environ.get("LOCAL_OPERATOR_DESKTOP_TOKEN")),
            "desktop_auth": "bearer",
            # These version the HTTP subsystems, not renderer completion or
            # third-party authorization. No aggregate "full parity" claim.
            "features": {
                "auth": 1,
                "settings": 1,
                "commands": 1,
                "catalogues": 1,
                "profile_catalogue": 1,
                "team_catalogue": 1,
                "session_catalogue": 2,
                # Searching past conversations by their CONTENT (name, id, exact
                # body, bounded soft match) rather than by the page a client
                # already holds. Its own key rather than a bump of
                # `session_catalogue`: a client can render the catalogue
                # perfectly well against a backend whose search route does not
                # exist, and gating the list on the search version would hide a
                # working surface because a newer one is missing.
                "session_search": 1,
                "lifecycle": 1,
                # Watch leases route notification delivery; they never mark read.
                # Named for this map's convention (`<subsystem>: <version>`); the
                # runtime record's capability LIST is a different namespace and
                # keeps its versioned `completion-ack-v1` string.
                "completion_ack": 1,
                # The `notification` frame's payload shape. A renderer that
                # sees this owns every completion banner and must stop toasting
                # on `agent_end`, or the user gets two for one turn; a renderer
                # that does not see it is talking to a backend that composes
                # nothing and keeps its legacy path. Absent is therefore a
                # meaningful answer, which is why this is a plain integer
                # rather than a boolean with a default.
                "notification_contract": 1,
                "mcp": 1,
                "radient": 1,
            },
        },
    )
