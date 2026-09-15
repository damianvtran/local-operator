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
                "mcp_auth": 1,
                "radient": 1,
                # Session code memory: GET/POST/PATCH/DELETE on
                # `/v1/desktop/sessions/{id}/variables`, reading and writing a
                # session's LIVE eval-kernel namespace by session id. Its OWN key
                # rather than a bump of `session_catalogue`, by the same rule
                # `session_search` states above: a renderer that does not see it
                # is talking to a backend whose only variables surface is the
                # legacy agent-id route (which cannot resolve a session id at
                # all), and its panel says "update the backend" instead of
                # retrying a call that can never succeed. Everything else in the
                # renderer works against such a backend, so gating anything else
                # on this would hide a working surface.
                "session_variables": 1,
                # Reading a SUBAGENT's own transcript through its parent's
                # child route (design § 9.1) — the run sidebar's child reader.
                # Its own key rather than a bump of `session_catalogue`,
                # because the roster and the to-dos ship with the renderer and
                # work against any backend: only the reader is gated, and a
                # missing capability has to mean exactly "this backend cannot
                # serve a child transcript", not "this renderer cannot show a
                # roster".
                "subagent_transcript": 1,
                # Moving a live session's working directory
                # (POST /v1/desktop/sessions/{id}/working-directory).
                #
                # ITS OWN KEY rather than a bump, and the rule is the one
                # `session_search` and `draft_preview` state above: an EXISTING
                # surface must keep working against a backend that lacks the new
                # route. Here the existing surface is the read-only
                # working-directory chip, which is exactly what a renderer that
                # sees no `session_move` keeps rendering. Bumping `commands`
                # would be the wrong lever twice over -- a client renders the
                # command palette perfectly well without this route, and
                # `/move`'s presentation already exists on older backends (it
                # answers its native_action today).
                #
                # BUMPED TO 2 for the exclusivity fence: a move now refuses
                # while another actual attach is registered (review R3), so a
                # renderer must not promise the old unconditional behaviour. The
                # bump is of THIS FEATURE CONTRACT, never the package version,
                # and `1` remains readable by a renderer that gates on presence.
                "session_move": 2,
                # An explicit desktop-only `frontend.replace` frame on the event
                # stream, plus the additive `frontend_replace=1` subscription
                # flag that negotiates it (review R4).
                #
                # Its own key because the two are independently useful and must
                # gate independently: a renderer that cannot consume the
                # replacement must keep its move controls DISABLED even against
                # this backend (`session_move >= 2` AND `frontend_replace >= 1`),
                # because a move whose accepted directory no mounted viewer can
                # render is exactly the stale-paint defect the frame exists to
                # fix.
                "frontend_replace": 1,
                # ``/info``'s host read and ``/session``'s one-snapshot ledger
                # report. A NEW key rather than a bump of `catalogues`, and the
                # rule is the one `session_search` states above: a renderer
                # renders `/analytics` and `/failovers` perfectly well against a
                # backend whose two diagnostic routes do not exist, and gating
                # those working panels on this version would hide them because a
                # newer one is missing. Only the two new ops read it, and a
                # renderer that does not see it shows the backend update action
                # instead of calling them.
                "diagnostics": 1,
            },
        },
    )
