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

ONE ANSWER, MANY WRITES (agent review round 1, R1-1/R1-2). The label that
counts is not one frame's: an ask's effective answers can be shaped by several
writes — the first ``answered`` row and every ``revised`` row after it, or the
legacy mirror's one-question-at-a-time taps that settle as one row. Each write
is stamped on its own, and the record carries the WEAKEST of them
(:func:`weakest_surface`): a proved "No, keep it" revised to "Yes" by an
unproven dial, or an unproven tap on question 1 completed by a proved tap on
question 2, must read as the unproven write it contains, never as the proved
one around it. Consent is trusted only when EVERY contributing write was.

F2 (2026-11-16) ADDS TWO RULES. (1) A signature-verified desktop: the signed
app's main process attests the exact answer content over a per-runtime AF_UNIX
socket (``asks/app_attest``), and a frame carrying that content is stamped
``desktop`` whoever started the runtime. (2) Minted-here: a capability proof
vouches only for asks THIS runtime process minted (:data:`_MINTED_HERE`), so
a cold-starter cannot answer asks inherited from an earlier runtime.

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

from typing import Callable

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


#: Labels the dangerous-delete checkpoint trusts. A MIRROR of
#: ``guardrails.provenance.TRUSTED_ASK_SURFACES`` (that module is too heavy to
#: import from here); ``test_ask_answer_surface_stamp`` pins the two equal, so a
#: drift fails a test rather than weakening :func:`weakest_surface`.
GUARD_TRUSTED_SURFACES = frozenset({"terminal", "tui", "desktop", "mirror"})


def weakest_surface(*surfaces: str) -> str:
    """The least-trusted label among the writes that shaped one answer.

    WHY A LATTICE, NOT A RANK. The two consent readers trust DIFFERENT sets —
    the guard :data:`GUARD_TRUSTED_SURFACES`, the profile classifier
    :data:`PROFILE_CONSENT_SURFACES` — so the result must be trusted by a reader
    only when EVERY input is trusted by that reader. An input that already has
    exactly that membership is returned as-is (an untrusted input is always one,
    so ``remote``/``unverified`` survive verbatim); when no single input does
    (``chat`` + ``terminal``: each trusted by one reader only) the answer is
    :data:`UNVERIFIED`, which neither reader trusts. Fail-closed by construction:
    no combination can produce a label more trusted than any of its parts.
    An empty call (nothing recorded who wrote) reads :data:`UNVERIFIED` too.
    """
    labels = [str(s or "") for s in surfaces]
    if not labels:
        return UNVERIFIED
    readers = (GUARD_TRUSTED_SURFACES, PROFILE_CONSENT_SURFACES)
    wanted = tuple(all(label in reader for label in labels) for reader in readers)
    for label in labels:
        if tuple(label in reader for reader in readers) == wanted:
            return label
    return UNVERIFIED


#: ``(session_id, ask_id)`` pairs THIS PROCESS minted (F2 Part C, decision D1).
#:
#: WHY. Whoever starts a runtime holds its operator capability, and that has
#: included a peer, the wake supervisor and a model's ``bash`` engaging an idle
#: session: before this set, such a cold-starter could answer every ask that was
#: already pending (Aida's 24 h asks among them) and have it stamped as the
#: operator's. A capability proof now earns a trusted label only for an ask this
#: runtime process created, so the starter speaks for the questions it watched
#: being asked and for nothing older. Asks INHERITED from an earlier runtime need
#: a signature-verified surface (``asks/app_attest``), whoever started this one.
#:
#: In memory ONLY, never persisted: a store edit or a restart must not be able to
#: extend it. Keyed by session because ask ids are only session-unique
#: (``store.new_ask_id``). Filled by ``AskQueue.enqueue``; bounded by the asks one
#: runtime ever mints. Not closed (stated in the design note): a cold-starter
#: that prompts the session into a NEW ask can answer that one with its cap.
_MINTED_HERE: set[tuple[str, str]] = set()


def note_minted(session_id: str, ask_id: str) -> None:
    """Record that this process minted ``ask_id`` (``AskQueue.enqueue`` only)."""
    _MINTED_HERE.add((str(session_id), str(ask_id)))


def minted_here(session_id: str, ask_id: str) -> bool:
    """Whether this runtime process minted the ask (see :data:`_MINTED_HERE`)."""
    return (str(session_id), str(ask_id)) in _MINTED_HERE


class StampedReceipt(str):
    """An ask-answer receipt that also names the label the runtime stamped.

    A ``str`` subclass so every caller that treats the receipt as its sentence
    keeps working unchanged; ``surface`` is ``None`` when the runtime predates
    the reply field. The desktop's ``consent_bearing`` (PR #4127) must come from
    this - what the runtime actually recorded - not from whether the backend
    holds a capability: after F2 and the minted-here rule the two disagree in
    both directions (an attested answer is consent with no cap; a cap answer to
    an inherited ask is not).
    """

    surface: str | None

    def __new__(cls, detail: str, surface: str | None = None) -> "StampedReceipt":
        obj = super().__new__(cls, detail)
        obj.surface = surface
        return obj


def consent_bearing(surface: str | None) -> bool | None:
    """Whether a stamped label is delete consent; ``None`` when it is unknown."""
    if surface is None:
        return None
    return surface in GUARD_TRUSTED_SURFACES


#: The ``by.proof`` audit value written beside a signature-attested answer.
APP_SIGNATURE_PROOF = "app-signature"


def stamp_answer_surface(
    *,
    proved: bool,
    surface: str,
    locality: str,
    kind: str,
    mirror: bool = False,
    minted: bool = True,
    attested: Callable[[], bool] | None = None,
) -> str:
    """The ``answered_by.surface`` the runtime records for one wire answer.

    ``proved`` is whether THIS frame carried a valid capability proof for THIS
    connection; ``surface``/``locality``/``kind`` are the connection's
    server-side facts. The frame's own ``by`` is never an input: that is the
    forgery this function closes.

    ``attested`` (F2) consumes a matching signature-verified attestation and
    says whether one existed. It is a CALLABLE, invoked only after the locality
    and kind checks, so a relayed or non-attach frame can neither use nor BURN
    the operator's attestation (design row N15). ``minted`` is the Part C rule:
    a capability proof counts only for an ask this process minted; the
    attestation is exempt (a verified surface answers any ask).
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
    if kind == "attach" and attested is not None and attested():
        return "desktop"
    if proved and not minted:
        # An INHERITED ask answered with a cap proof: the cap proves who started
        # this runtime, not who is at the keyboard (Part C). Same label as any
        # unproven dial, so the guard's mirror sets stay untouched.
        return UNVERIFIED
    if proved:
        if mirror:
            return MIRROR
        return surface if surface in ("desktop", "terminal") else "terminal"
    return UNVERIFIED
