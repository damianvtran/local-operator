"""Public feature negotiation contains no credentials or configuration values."""

from fastapi import APIRouter

from local_operator.server.desktop import desktop_posture
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["Capabilities"])


@router.get("/v1/capabilities", response_model=CRUDResponse)
async def capabilities():
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
            "features": {
                "auth": 1,
                "settings": 1,
                "commands": 1,
                "catalogues": 1,
                "profile_catalogue": 1,
                "team_catalogue": 1,
                # 3 adds POST /v1/desktop/sessions/{id}/warm, which starts a
                # session's runtime without submitting work to it.
                #
                # A BUMP rather than a new key, and the rule is the one
                # `session_search` states below rather than "subsystems never
                # get a key per route" -- which that entry would contradict,
                # being a route in this very subsystem with its own key. The
                # real question is what a client must NOT be gated on: a
                # separate key exists so an EXISTING surface keeps working
                # against a backend that lacks the new route. Nothing here is
                # gated: warming is an optimisation on the send path a client
                # already has, so a renderer reading < 3 simply never calls it
                # and pays the cold engage on its first send, exactly as
                # before. Gating nothing, it needs no key of its own.
                "session_catalogue": 3,
                # Searching past conversations by their CONTENT (name, id, exact
                # body, bounded soft match) rather than by the page a client
                # already holds. Its own key rather than a bump of
                # `session_catalogue`: a client can render the catalogue
                # perfectly well against a backend whose search route does not
                # exist, and gating the list on the search version would hide a
                # working surface because a newer one is missing.
                "session_search": 1,
                # The readings a NEW-conversation pane may show, resolved by the
                # backend for a session that does not exist yet (a draft has no
                # session record, and the strip must be able to say what model
                # the first send will use). Its own key rather than a bump of
                # `session_catalogue`: a client renders the catalogue perfectly
                # well against a backend without this route, and gating the list
                # on it would hide a working surface because a newer one is
                # missing. The draft strip is the only thing that may not render
                # without it.
                "draft_preview": 1,
                # Whether a draft's model and effort chips can be made
                # ACTIONABLE. Its own key rather than a bump of `draft_preview`,
                # and it exists precisely because the two answer different
                # questions: `draft_preview` says the backend can REPORT the
                # readings a first turn will get, which a renderer shows inert;
                # this says it also accepts a `model` on create and preview, so
                # a client that does not see it must keep its chips inert rather
                # than let them dispatch into a 422 (an older backend ignores the
                # field, and a pick that silently did nothing would be worse than
                # a chip that never offered itself).
                "draft_selection": 1,
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
                # Reading a SUBAGENT's own transcript through its parent's
                # child route (design § 9.1) — the run sidebar's child reader.
                # Its own key rather than a bump of `session_catalogue`,
                # because the roster and the to-dos ship with the renderer and
                # work against any backend: only the reader is gated, and a
                # missing capability has to mean exactly "this backend cannot
                # serve a child transcript", not "this renderer cannot show a
                # roster".
                "subagent_transcript": 1,
            },
        },
    )
