"""Who answered a queued ask: the surface label the RUNTIME stamps on the answer.

WHY THIS EXISTS. An answered ask records ``answered_by.surface``, and two
readers treat some of those labels as the operator's own word: the guardrail's
dangerous-delete checkpoint (``guardrails.provenance.TRUSTED_ASK_SURFACES`` —
an approved ask that names the path settles the delete before any judge) and
the profile-trust classifier (``session.session._profile_input_class``). Until
this module the label came from the answering FRAME's ``by`` field, and the
control socket authenticates only with the record's ``control_key`` — a 0600
file any same-uid process can read, including a model-authored ``bash`` call.
So the agent could queue "Delete ~/X?", dial its own runtime and answer
``by="desktop"``; a mesh peer holding only ``prompt`` could do the same through
``net_forward``; and the legacy ``ask_answer`` mirror stamped ``"mirror"`` for
ANY caller. Each of those turned a forged answer into delete consent with zero
judge calls.

THE RULE. A trusted label is written only when the connection PROVED operator
authority on that very frame, the same proof ``approval_answer`` needs: the
per-connection request proof of the in-memory operator capability
(``harness.approval.request_proof_ok``), which a process that merely read the
record cannot produce. A proved answer gets the CONNECTION's surface (also
server-decided at auth time, ``desktop`` or ``terminal``) — or ``mirror`` on the
legacy card op — and an unproven one gets :data:`REMOTE` (a relayed/peer or
phone-daemon hop) or :data:`UNVERIFIED` (a local dial without the proof). Both
still SETTLE the ask, so the agent gets its answer; neither is consent.

WHO PRESENTS THE PROOF. ``AttachClient`` attaches it to an ask frame only when
the CALLER opts in (``operator_ask=True``): the desktop facade, the attached
pane and the TUI dock do; the phone daemon never does, although it can hold a
capability for a runtime it engaged, because what it relays is the portal's
(remote) answer. A relayed (non-local) or non-``attach`` connection is never
trusted, proof or not.

CONSTRAINTS.

* stdlib-only and import-light: the runtime dispatch, both serving handles and
  the session's profile classifier import it.
* Operator/device SIGNATURES are not accepted for asks here: a signature costs an
  OS prompt per action, and asking for one on every ask answer would be a UX
  change on every surface. A surface without the capability (a desktop or pane
  attached to a runtime it did not start, the phone) therefore records a
  non-consent label; its answer still reaches the agent, and a dangerous delete
  it was meant to authorise goes to the judge (or the card) like any other.
* Direct writes to ``asks.jsonl`` by a same-uid script are a separate,
  store-integrity problem this label cannot fix.
"""

from __future__ import annotations

#: An unproven answer that arrived over a relayed hop (mesh peer, phone daemon).
REMOTE = "remote"
#: An unproven answer from a local dial: someone who could read the record key.
UNVERIFIED = "unverified"
#: The legacy per-question card op (``ask_answer`` on ``"<ask_id>.<qidx>"``).
MIRROR = "mirror"

#: Labels the profile-trust classifier counts as a person's consent. ``tui`` and
#: ``desktop`` are sound ONLY because :func:`stamp_answer_surface` is the sole
#: writer of a wire answer's label (and the in-process TUI writes ``tui``/
#: ``terminal`` without crossing the socket). ``chat`` is unchanged pre-existing
#: behaviour (the agent's ``ask_withdraw answered_in_chat``) and out of scope here.
PROFILE_CONSENT_SURFACES = frozenset({"tui", "desktop", "chat"})


def stamp_answer_surface(
    *, proved: bool, surface: str, locality: str, kind: str, mirror: bool = False
) -> str:
    """The ``answered_by.surface`` the runtime records for one wire answer.

    ``proved`` is whether THIS frame carried a valid capability proof for THIS
    connection; ``surface``/``locality``/``kind`` are the connection's
    server-side facts. The frame's own ``by`` is never an input: that is the
    forgery this function closes.
    """
    if locality != "local":
        # A RELAYED frame (mesh ``net_forward``/``stream_send``) is another
        # device's answer whatever it carries: the relaying runtime may itself
        # hold a capability, so a proof on this hop proves nothing about the
        # person at the far end.
        return REMOTE
    if kind not in ("attach", ""):
        # Not an ``AttachClient`` dial (the phone daemon's registrar, a CLI
        # carrier): every local operator surface answers through one, so any
        # other kind speaks for something else. ``""`` is no connection at all
        # (a direct ``_dispatch``), which proves nothing and reads unverified.
        return REMOTE
    if proved:
        if mirror:
            return MIRROR
        return surface if surface in ("desktop", "terminal") else "terminal"
    return UNVERIFIED
