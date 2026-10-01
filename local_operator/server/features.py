"""Public feature negotiation, importable off the HTTP route.

The ``features`` map of ``GET /v1/capabilities`` lives here as a pure function
rather than inside the route body because it now has a reader that does not
speak HTTP: the mobile daemon negotiates attach capabilities with owner
runtimes and imports this function to answer the same question the desktop
renderer answers over the wire. ONE map for every reader -- a second copy
would drift the way every restated rule does.

The ``references`` import stays INSIDE :func:`feature_flags`, and that is
load-bearing rather than stylistic: ``local_operator.references`` reaches
``local_operator.tools.builtin`` (~160 ms), and a module-scope import here
would put that back on every ``lop serve`` startup before the first request --
the server-shape guard in ``tests/unit/test_import_graph.py`` fails outright
on it. Deferred, the cost is paid once per process, on the first caller, and
the surface it serves is cold negotiation, not a hot path.
"""

from __future__ import annotations

from typing import Any


def feature_flags() -> dict[str, Any]:
    """The versioned feature map, resolved per call.

    A pure function of PROCESS state -- no request, no session, no arguments --
    which is what lets a non-HTTP surface import it and read exactly what the
    route publishes: ``references`` follows its kill switch, everything else is
    a build fact.
    """
    # IMPORTED HERE, NOT AT MODULE SCOPE -- see the module docstring. The
    # precedents the guard names are ``routes.desktop_profiles`` (imports
    # ``tools.agent_tool`` per profile write) and ``routes.auth`` (imports
    # ``tunnels.report`` per request).
    from local_operator.references import at_references_enabled

    return {
        "auth": 1,
        "settings": 1,
        # 2 narrows the messages endpoint's slash policy: it now accepts a
        # message that merely BEGINS with a command word (`/mcp logout
        # seems to cause a crash` is prose), while a text that as a whole
        # IS a command is still refused so a control can never become
        # paid model chat. A BUMP rather than a new key, by the rule
        # `session_catalogue` states below: nothing is gated on it — no
        # surface is withheld — and its only consumer is the refusal
        # alert's remedy, which differs by whether the backend can reach
        # that refusal at all. A renderer on < 2 keeps the sentence and
        # offers the backend update; on >= 2 the refusal means a client
        # bug, so the sentence stands alone.
        "commands": 2,
        "catalogues": 1,
        # The SESSIONLESS skill catalogue: `GET /v1/desktop/skills` answers
        # for an explicit `cwd` (home when omitted) with no session at all,
        # and every answer — sessionless or session form — carries `version`,
        # the skill tree's own identity (a digest of the runtime's
        # `roots_fingerprint`), so a composer can cache the vocabulary per
        # folder and refetch exactly when the tree changes.
        #
        # ITS OWN KEY rather than a bump of `catalogues`, by the rule
        # `session_catalogue_page` states below — a client must be able to ASK
        # whether the daemon understands the sessionless form BEFORE it sends
        # one — and the half `session_search` adds: gating a working surface on
        # the newer sibling would hide it. On a backend without this key the
        # route REQUIRES `session_id`, so a sessionless call is a 422, and a
        # client that does not see this key must say "update the backend"
        # instead of attempting an operation whose failure it cannot classify
        # (the shape `radient_org` states below). `catalogues` itself also has
        # other consumers whose behaviour must not version-split: they render
        # the model and command catalogues perfectly well against a backend
        # that cannot serve a sessionless skill read, and bumping would hide a
        # working surface because a newer one is missing.
        "skill_catalogue": 1,
        "profile_catalogue": 1,
        "team_catalogue": 1,
        # Agent Hub update availability: GET /v1/desktop/hub/updates (a store read,
        # no network) plus check/apply/apply-all/retry. A renderer gates the
        # sidebar indicator on this so it never polls a backend that lacks it.
        "hub_updates": 1,
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
        # A catalogue that can be asked for ONE scope at a time, resumed
        # by an opaque cursor, and counted per group: the
        # `scope_kind`/`scope_name`/`cursor`/`with_counts` parameters on
        # GET /v1/desktop/sessions, and `next_cursor`/`cursor_missing`/
        # `scope`/`counts` on its answer.
        #
        # ITS OWN KEY, not a bump of `session_catalogue`, by the rule
        # `session_pins` states below: a key exists so an EXISTING surface
        # keeps working against a backend that lacks the new one, and the
        # existing surface here is the whole chats list. The failure mode
        # a bump would not fix is why this is not cosmetic: FastAPI
        # SILENTLY IGNORES unknown query parameters, so an un-gated client
        # that sent `scope_kind=team&scope_name=lopdev` to an older daemon
        # would receive the UNSCOPED page and draw other teams'
        # conversations under that team, and an un-gated `cursor` would
        # receive page one again and duplicate it. The client must be able
        # to ASK whether the daemon understands the parameters, before it
        # sends them — which is the one thing a capability key can answer
        # and a version bump of a working surface cannot.
        #
        # ONE KEY FOR THREE PROMISES (scoping, paging, the census),
        # because they are one contract revision: a client that had the
        # counts without the scope could not draw a group's count
        # consistently with that group's paged rows.
        #
        # Absent ⇒ the client makes ONE unscoped request exactly as today
        # (`limit=500`, no scope, no cursor, no counts) and renders exactly
        # as today, with today's latency. `session_catalogue` and
        # `session_pins` are untouched: no row changes shape, and the
        # pin's own promise is unchanged.
        "session_catalogue_page": 1,
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
        # The provider registry VIEW: `brand` / `capabilities` / `state` /
        # `identity` / `account_count` on the rows of
        # `GET /v1/auth/providers` (a.k.a. the census), AND the licence
        # for the composer's provider inline lists (`/login` and
        # `/logout` arguments). Its own key, not a bump of `auth`, by the
        # rule `session_search` states above: a renderer that does not
        # see it keeps rendering the census exactly as today (it already
        # ignores unknown row fields) and keeps opening the
        # LoginPicker/LogoutPicker dialogs on a pick — both are working
        # surfaces over a backend without this key, and gating them on
        # the newer LIST shape would hide them. With the key, the
        # optional row fields (`brand`, `capabilities`, `identity`) may
        # still be MISSING (an older backend that answers the census but
        # not the view) — every reader falls back to its local
        # derivation, so the key's presence never licenses a field that
        # is not on the row.
        "provider_catalogue": 1,
        # The new-chat pane's DRAFT PRE-ENGAGE: `POST
        # /v1/desktop/sessions/draft` mints an id the pane may subscribe,
        # watch and warm before any session exists, and `create` accepts
        # it back so the first send never pays the engage. Its own key
        # rather than a bump of `draft_selection`, by the rule that entry
        # states: this gates NOTHING a renderer already shows — warming
        # is an optimisation on a send path the client already has — so
        # a renderer that does not see it simply never mints and pays
        # the cold engage exactly as today (the same policy as
        # `session_catalogue`'s warm version). A renderer gated on this
        # key must also degrade silently on `sessions.draft` failure:
        # the warm is speculation, never a reason to block a send.
        "session_draft_warm": 1,
        "lifecycle": 1,
        # Watch leases route notification delivery; they never mark read.
        # Named for this map's convention (`<subsystem>: <version>`); the
        # runtime record's capability LIST is a different namespace and
        # keeps its versioned `completion-ack-v1` string.
        "completion_ack": 1,
        # The BULK read receipt: one request clearing the completions a
        # client RENDERED, token-bound (`POST /v1/desktop/attention/seen`).
        # Its OWN key rather than a bump of `completion_ack`, because the
        # per-session ack must keep working against a backend that lacks
        # the batch route -- the same rule `session_search` and
        # `draft_preview` above state. A renderer that does not see this
        # key must neither draw the control nor send the op (a 404 after
        # a click is a broken control), and nothing else is gated on
        # `completion_ack`'s version, so no other surface can be hidden
        # by the addition.
        "completion_ack_bulk": 1,
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
        # Sessionless MCP management: `GET/POST /v1/desktop/mcp` and
        # `POST /v1/desktop/mcp/credentials`, answering with no session
        # and no model configured, in the catalog vocabulary
        # (connected / needs_sign_in / not_started / connecting / error,
        # plus per-row `actions`). Its own key because a renderer that
        # does not see it must keep using the session route, which still
        # works — "update the backend" would be false there.
        #
        # 2 adds the document's `verbs` array (verb/description/
        # destructive/offers per `/mcp` subcommand) AND licences the
        # composer's MCP inline lists. A BUMP rather than a new key, by
        # the rule `commands` states above: the same route and the same
        # document gained a contract element, and a client reading >= 1
        # that ignores `verbs` loses nothing — Settings > Integrations
        # keeps working unchanged. Below 2 (an older backend) the
        # composer draws NO MCP list and the pick of `/mcp` runs the
        # existing McpPicker, which is today's path; it must not send a
        # `/mcp ` argument list against a document with no verbs, where
        # every row would be invented client-side.
        "mcp_catalog": 2,
        "radient": 1,
        # The organization surfaces on the Radient proxy (design §4.7):
        # the four closed operations `memberships.list`, `org_agents.list`,
        # `org_teams.list` and `org_team.get`, and the three frozen
        # membership refusals they carry through under their own codes
        # (`not_a_member`, `insufficient_role`, `team_plan_required`).
        # Its OWN key rather than a bump of `radient`, by the rule
        # `completion_ack_bulk` states above: a renderer that does not
        # see it is talking to a backend that answers the unknown
        # operations with a MASKED 422 ("The request has invalid
        # fields."), indistinguishable from a malformed call -- so the
        # org surface must say "update the backend" instead of
        # attempting an operation whose failure it cannot classify,
        # while everything else keeps working and is not gated on this.
        "radient_org": 1,
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
        # The child reader's LIVE half, and a NEW KEY rather than a bump
        # of `subagent_transcript` above (design § 1, D3.4).
        #
        # The reader's admission gate asks for `subagent_transcript` with
        # no minimum version, so bumping it would leave an older backend
        # still passing that gate — the gate could not tell "the reader
        # may open" apart from "the live part exists" without asking a
        # second question of a key that already answered one. A separate
        # key keeps the reader's floor exactly where it is and makes the
        # live question its own: absent ⇒ the reader is today's pager,
        # issues no watch, and looks exactly as it does now. That is the
        # POST/DELETE pair on
        # `/v1/desktop/sessions/{id}/children/{job}/trajectory` plus the
        # `job_trajectory_appends`/`job_trajectory_replacements` field
        # pair those ops turn on for that job.
        "subagent_trajectory": 1,
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
        # The machine-wide desktop event feed and its delivery-presence
        # lease. TWO keys rather than one, because each has a consumer
        # that can be absent independently:
        #
        # * `desktop_feed` gates `GET /v1/desktop/events` AND
        #   `POST /v1/desktop/presence`. Absent ⇒ the app opens no
        #   feed, beats no presence, and keeps its 5 s catalogue poll
        #   and its per-session notification path verbatim.
        # * `desktop_presence` gates the backend's READ of the
        #   presence: with it absent the backend composes no
        #   machine-wide expectation, so nothing is suppressed on the
        #   strength of a lease nobody publishes. An app that may not
        #   publish a viewer record should also not claim presence,
        #   which is why the two travel together from the client's
        #   side.
        #
        # NEITHER is a bump of `notification_contract`, which stays 1:
        # the `notification` payload is unchanged (modulo the derived
        # `focus_policy` routing field, which the client already
        # special-cases) and a renderer that ignores the new frame
        # types keeps working.
        #
        # NOR does the `session_status` frame on this same stream need a
        # key of its own, on the rule `session_search` states above: a
        # key exists so an EXISTING surface keeps working against a
        # backend that lacks the new one, and nothing here is gated. The
        # frame is a LATENCY OPTIMISATION on a path every client already
        # has — the list — and the two list keys it pairs with
        # (`status_epoch`/`status_revision`) are additive fields on a
        # model that is `extra="allow"` by design. A renderer that does
        # not branch on `session_status` ignores it and keeps its 30 s
        # safety poll; a renderer against an older backend sees a list
        # with no stamps, reads "no comparison available", and takes the
        # list's value, which is exactly what happens today. What WOULD
        # force a key is any client behaviour that depends on the
        # backend supporting the frame (relaxing a poll, dropping a
        # refetch, rendering an "unavailable" hint); the design that
        # added it does none of those, and this comment is where that
        # condition gets re-checked rather than rediscovered.
        "desktop_feed": 1,
        "desktop_presence": 1,
        # Stopping a session's CURRENT WORK without ending the session:
        # POST /v1/desktop/sessions/{id}/interrupt.
        #
        # ITS OWN KEY, not a bump of `lifecycle`, and the reason is the
        # one this whole map is written to: a backend that can stop a
        # session but cannot interrupt a turn must keep `/stop` working
        # and must NOT be told it can interrupt. The two are different
        # promises: `lifecycle` is the kill switch (deny gates, dispose,
        # release the writer lease, unpublish, exit the runtime), and a
        # renderer that read a bumped `lifecycle` as "I may interrupt"
        # would send a press that a pre-interrupt backend answers with a
        # 404 — or worse, wire it to `/stop` and end the user's session
        # under a button promising it would not. Absent ⇒ the renderer
        # HIDES its Stop control and fires nothing; falling back to the
        # old silent no-op would keep exactly the lie this route exists
        # to remove. Nothing else in the renderer is gated on it, so
        # gating anything else here would hide a working surface.
        "session_interrupt": 1,
        # A CONFIGURATION RUN can be started from the Agents/Teams page: an
        # additive ``purpose: "agents-config"`` on ``POST
        # /v1/desktop/sessions`` (stamped with the hidden ``agent-config``
        # origin) plus the server-owned tool declaration and preamble that give
        # the run its reach.
        #
        # ITS OWN KEY, not a bump of an existing one, by the rule
        # ``interrupt-turn`` states for ``session_interrupt`` above: an older
        # backend answers an unknown field on ``CreateSession`` with a 422
        # (``Input`` is ``extra="forbid"``), so a renderer that sent the body
        # ungated would turn "this backend cannot do it" into a broken create.
        # The page reads this key and, without it, keeps the structured editor
        # as the only path — which is exactly the release-skew property the
        # sibling UI PR is built to rely on.
        #
        # The run ALSO needs ``session_interrupt`` (to stop a turn) and the two
        # catalogue keys (to show what a run changed). Those are existing keys
        # and are deliberately not re-declared here: a client requires the
        # conjunction, it does not get a second spelling of the same fact.
        "agents_config": 1,
        # Durable conversation pins: the `pinned` flag on every catalogue
        # row, and POST /v1/desktop/sessions/{id}/pin.
        #
        # ITS OWN KEY, not a bump of `session_catalogue`, by the rule
        # `session_search` states above: an EXISTING surface must keep
        # working against a backend that lacks the new one, and here that
        # surface is the whole chats list. Bumping `session_catalogue` to
        # 4 would hide a working catalogue behind an update it does not
        # need, and no EXISTING key's shape changes — `sessions` gains
        # rows (see below) without gaining a field.
        #
        # STILL 1, AND THAT IS NOT AN OMISSION. `sessions` on this route
        # now also carries pinned conversations the page did not reach,
        # which a client MUST read to render a pin made on an older
        # conversation; that is a change to what the contract contains,
        # not to the contract's shape, and the number has never been
        # RELEASED — this key and the route that reads it are landing in
        # the same unreleased window, and #1200/#1214 are still open. A
        # bump to 2 would advertise a difference from a version no client
        # has ever talked to, which is a migration nobody can perform.
        # What the number gates is unchanged: absent ⇒ no affordance, no
        # slot, no handler; present ⇒ the row's `pinned` flag is readable
        # and settable, and `sessions` is complete for the pinned set.
        #
        # NOT gated on `desktop_feed` either, because the two answer
        # different questions: the pin is durable state the renderer has
        # to be able to READ, while the feed is only how fast it learns
        # that someone else changed it. A backend with the route and no
        # feed must still show pins — its ≤30 s safety poll picks them up.
        #
        # Absent ⇒ the renderer mounts NO affordance at all: no pin slot,
        # no hover reveal, no handler (`session_interrupt` above is the
        # precedent, and for the same reason). A DISABLED pin would be
        # worse than an absent one — the row's control slot exists, so the
        # user would read it as a feature they have not unlocked, and a
        # permanently reserved empty slot costs every row width to
        # advertise nothing.
        "session_pins": 1,
        # Whether a session can be ARCHIVED: hidden from every default
        # listing and from search, still resumable by id.
        #
        # ITS OWN KEY rather than a bump of `session_catalogue`, on the
        # rule that entry states: every catalogue key above publishes a
        # VERSION of a surface whose older form still works, while this
        # one is a capability that is either there or not. A renderer
        # reading it gets the archive control, the `archived` flag on
        # rows and the `include_archived` parameter; a renderer that does
        # not has an older backend, whose catalogue rows have no such
        # flag to merge and whose routes would 404 a control the user
        # could press. Absent => no affordance, no slot, no handler.
        #
        # A NEW KEY RATHER THAN A BUMP, and that is the fail-closed half:
        # the archive is not a shape change to the catalogue (the rows
        # are the same rows, one field richer, and that field is ADDITIVE
        # — an older renderer ignores it), so bumping would hide a
        # catalogue that works perfectly well from a client that predates
        # the feature.
        "session_archive": 1,
        # Whether a session can be PERMANENTLY DELETED.
        #
        # SEPARATE FROM `session_archive`, deliberately, even though
        # this change ships both: the two are not one capability wearing
        # two names. Archive is reversible and receipt-free; delete is
        # irreversible, carries a confirmation the client must render and
        # can be refused with a sentence (409) the client must be able to
        # SHOW. A client that could archive but not delete is a
        # representable product, and a single key would make that
        # impossible to express — the same argument `session_search`
        # makes for not riding `session_catalogue`.
        "session_delete": 1,
        # This machine's tunnel and Radient-login state:
        # `GET /v1/desktop/tunnel`, plus `radient_login` and
        # `tunnel_remedy` on `GET /v1/auth/status`.
        #
        # ITS OWN KEY, and it gates one narrow thing: whether the
        # account section may tell a user that their stored Radient
        # login is no longer accepted for remote access, and whether the
        # tunnel panel may show a connector state at all. The question
        # those answer is about THIS MACHINE, which the rest of the auth
        # surface cannot answer — a row can be `configured` with an
        # unexpired access token and still be refused by the identity
        # provider.
        #
        # Absent ⇒ the renderer shows no tunnel state and no sign-in
        # callout, and the account section keeps its current wording
        # (which is not wrong, only incomplete). It must NOT treat an
        # absent key as "the tunnel is fine": a backend that has never
        # heard of this route cannot be asked, and a green claim nobody
        # made is worse than the silence.
        #
        # NOT a bump of `auth`: nothing on that surface changes shape,
        # an old renderer ignores the extra fields, and bumping would
        # hide a working sign-in flow behind an update it does not need.
        "tunnel": 1,
        # THE MESH, and both keys gate AFFORDANCES rather than shapes.
        #
        # `peers` is the peer catalogue's key (mesh-session-mobility.md
        # §9.3, `mesh-ui.md` §2.6): the `GET /v1/desktop/peers` route, the
        # `include_peers` parameter on the session list, the flat
        # `locality`/`owner_device*` fields on every row, the peer `peer`
        # on create, and the Networks tab's data. ABSENT ⇒ no peer
        # sections and no remote marks at all — and the `Peers` group must
        # be NOT MOUNTED rather than mounted-empty, because a reserved
        # empty section advertises a feature the user does not have (the
        # argument `session_pins` already makes).
        #
        # `session_transfer` is ITS OWN KEY, not a bump of `peers`, by
        # the rule `session_search`/`session_delete` state above: a
        # backend can show a peer's sessions and be unable to move one,
        # and on `peers` alone the renderer would draw its
        # `Move a chat here…` row against a route that 404s. Absent ⇒ the
        # control is not mounted and the list is not offered.
        #
        # ADVERTISED UNCONDITIONALLY, including on a machine in no
        # network, because the KEYS answer "what can this backend do"
        # rather than "is this machine in a mesh": a renderer that had to
        # ask a second question would gate its own controls on a
        # configuration read it cannot make. A device in no network
        # answers an empty catalogue, and an empty catalogue mounts
        # nothing.
        "peers": 1,
        "session_transfer": 1,
        # ``@path`` in an ordinary submitted message, which
        # :meth:`Session.prompt` expands into an
        # ``<operator-references>`` block before the model sees the turn
        # (`references.expand_references`; `prompt` is the single
        # expansion site EVERY NON-ASIDE SURFACE reaches — the CLI,
        # headless runs, the server, the scheduler, the mobile service and
        # every subagent — and `references.py`'s docstring counts the
        # other expanding caller, the TUI's aside worker, beside it).
        #
        # WHAT THIS KEY DOES NOT COVER, stated because a client that
        # assumed otherwise would paint a chip over literal text: this
        # server's send route takes ``mode: "prompt" | "steer"``, and a
        # STEER bypasses `prompt` — :meth:`Session.steer` queues the
        # message and expands nothing. `session/session.py` lists the
        # three entry paths that predate the feature and leave an `@path`
        # as inert prose: `steer`, a wake delivery, and an aside FORK —
        # and that last one is true only of the adopted ROW, because the
        # TUI's own aside worker expands the text it hands the model
        # (`tui/app.py`'s `_expand_references` is that call, and
        # `references.py` names it as one of the TWO expanding callers).
        # So the key means "this backend expands a reference in a
        # PROMPT", never "in any text a client sends", and a composer
        # that can send either must withhold the affordance for the
        # steer — which is what the shipped one does, `!busy` beside
        # this gate.
        #
        # THIS IS THE WRITER THE RENDERER HAS BEEN WAITING FOR. The app
        # ships the `@` picker, the inline chips and the composer tip,
        # and withholds all three until a backend advertises this key —
        # so on every backend older than this line the affordance is
        # dark BY DESIGN, `@path` stays the plain text such a backend
        # sends, and the composer says why. The key is the whole switch;
        # nothing else about the app changed to light it up.
        #
        # CONDITIONAL, and it is the only key in this map whose PRESENCE
        # is a runtime fact rather than a build fact.
        # ``LOCAL_OPERATOR_AT_REFERENCES``
        # (:data:`local_operator.references.AT_REFERENCES_ENV`) is a kill
        # switch read per call, so a process told not to expand must not
        # advertise a picker that paints chips for an expansion it will
        # not perform: a client that believed the key would show the
        # user a reference the model never receives, which is the one
        # lie this gate exists to prevent. That variable has exactly ONE
        # reader, :func:`~local_operator.references.at_references_enabled`,
        # which is called here rather than re-read, so this payload and
        # the expansion THIS PROCESS performs cannot disagree.
        #
        # AND THAT LAST CLAUSE IS SCOPED TO THIS PROCESS, which is a real
        # boundary rather than a hedge: a turn is admitted over a socket
        # to whichever process OWNS the session, and a runtime's
        # environment is the snapshot it was spawned with. So a daemon
        # whose switch is on can advertise the key while an owner engaged
        # from a shell that exported the switch off sends the literal
        # `@path`. No value read here can fix that, and it needs no new
        # handling: the owner's behaviour is the same absent-key
        # behaviour a client already implements, and the case that
        # matters — every backend older than this line — is decided by
        # presence alone.
        #
        # ITS OWN KEY rather than a bump, by the rule `session_search`
        # states above: the composer renders perfectly well without it —
        # an unexpanded `@path` is prose the model reads as prose — so
        # gating that working composer on a newer version would take a
        # surface away to advertise nothing. A client that does not see
        # the key must send the draft as typed and offer no affordance.
        # The PROJECT primitive (`/v1/desktop/projects*`, the ``project``
        # tool and ``/project``). ONE key gating both the affordance and
        # the reads: the Projects tab, its CRUD dialogs and the
        # ``@project:`` picker section all hang off this surface, and an
        # older backend has no routes to answer any of them.
        #
        # Absent ⇒ no Projects destination and no project rows in the
        # ``@`` picker, which is the honest degradation: an older
        # backend does not know what a project is. Its own key rather
        # than a bump of anything above it, by the rule every neighbour
        # states — nothing on an existing surface changes shape here.
        # AIDA (the built-in chief of staff): the sidebar row, the
        # `/aida` command handling, and the `/v1/desktop/aida` routes
        # — design §4's frozen pair (capability + route contract).
        #
        # ONE KEY GATES BOTH HALVES, by the rule `projects` states
        # above: a row that opened a route an older backend does not
        # serve is a dead end, and a served route with no row is an
        # orphan. Absent ⇒ no row, no command handling, and the UI
        # must not call the route at all; a client that typed `/aida`
        # at such a backend keeps today's behaviour (the word is
        # prose to it — its registry has no such entry).
        #
        # `1` IS THE FEATURE'S FIRST VERSION, not the contract's: the
        # route shape is frozen in the design, so a later change to
        # it bumps this key. NOT a bump of `commands`: nothing on the
        # command surface changes shape — an older renderer simply
        # never hears of this word.
        "aida": 1,
        "projects": 1,
        # THE ONBOARDING APPROVAL SURFACE: GET /v1/desktop/approvals plus the two
        # decision posts (POST /v1/desktop/approvals/{id}/approve|deny). ONE key
        # for the family because it is one contract revision and one flow: a
        # renderer that can draw the card is the renderer that can answer it.
        #
        # WHY A KEY AT ALL: the routes are ADDITIVE (an old renderer never calls
        # one), so this gates nothing on an old client — it is how a renderer
        # learns the surface EXISTS before it builds a Mesh-tab affordance whose
        # POST would 404 on a backend without it. Absent, the tab simply has no
        # approval affordance, which is the pre-onboarding state.
        "approvals": 1,
        # THE ONE DIAL on the projects surface: `POST
        # /v1/desktop/projects/{key}/request-update`, which asks the linked
        # sessions for a progress update through the same peer-send core the
        # `send` tool uses (mailbox drop + wake). ITS OWN KEY, not a bump of
        # `projects`, by the rule that entry states above: a renderer that
        # does not see it keeps working against a backend without the route —
        # the tab, its CRUD and its picker rows are untouched — it simply
        # shows no request-update action. Absent ⇒ no action anywhere; a
        # client that shows one against an older backend would be offering a
        # route that 404s.
        "projects_request_update": 1,
        # THE SILENT INPUT-METADATA CARRIAGE: `input_mode`/`input_path` on a
        # message body (`POST /v1/desktop/sessions/{id}/messages`) and onto the
        # durable user row. ONE key un-gates BOTH fields, because they landed
        # as ONE contract revision (the mode axis plus the reserved route
        # slot): a client that may send one may send the other, and a second
        # key would only create a state where half the contract is negotiable.
        #
        # WHY A KEY AT ALL: `Prompt` is `extra="forbid"`, so an ungated send to
        # an older backend would 422 the user's own message. Gated, the
        # composer omits the fields and behaves exactly as today — absence is
        # the legacy path, not a degraded one.
        #
        # THE ATTACH-WIRE HALF OF THE SAME NEGOTIATION is `input-mode-v1` in
        # the owner record (the mobile stream reads that string; the desktop
        # renderer reads this key). Two faces, deliberately separate: they
        # gate two different wires, and neither implies the other.
        "input_mode": 1,
        # THE SPEECH SURFACE: `GET /v1/stt/paths` (the ordered cascade report)
        # and `POST /v1/stt/transcriptions` (the executor, with its 409
        # `stt_unavailable` structured refusal), plus the audio-accepting
        # message body the recording door rides (`Prompt.audio`, v1: at most
        # one block per send). ONE key for the family because they are one
        # contract revision: a client that can read the report is the client
        # that can send the recording, and half the contract is not a state
        # worth negotiating.
        #
        # WHY A KEY: `Prompt` is `extra="forbid"`, so an ungated audio send to
        # an older backend would 422 the user's own message; absent, the mic
        # stays off and the client keeps its text path. The LEGACY Radient
        # transcription route is untouched by all of this and remains the
        # ungated path for a client that has it.
        "stt": 1,
        **({"references": 1} if at_references_enabled() else {}),
    }
