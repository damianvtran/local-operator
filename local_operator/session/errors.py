"""Known-safe admission rejections shared by owner, attach client and HTTP.

Only these categories cross the transport boundary as actionable user errors.
Never certify a message by its wording: an arbitrary RuntimeError may contain
socket addresses, credentials or another conversation's identity.
"""


class AttachmentUnavailable(ValueError):
    code = "unresolved_attachment"

    def __init__(self) -> None:
        super().__init__(
            "The attached profile or team could not be restored. "
            "Choose an available profile or detach it before sending."
        )


class RuntimeRetiring(ValueError, RuntimeError):
    """This runtime has committed to leaving; the message was not admitted.

    A DRAIN, not a failure: the runtime is handing over to a successor that
    boots the build now on disk, and it refuses new work while it finishes what
    is already in flight. The refusal is therefore transient and self-healing,
    which is what the wording and the app's notice register both have to say.

    The sentence is rebuilt HERE rather than crossing the wire, so the category
    and its copy cannot drift, and an older peer that does not know the code
    still receives it as the frame's ``message``. It is also the only thing a
    user reads about this whole mechanism, which is what makes the vocabulary
    the contract: the owner's previous wording named an internal log token
    (``runtime-retired``) and described the machinery ("the next engage runs
    the new build") rather than the situation the operator is now in (design
    round 1, D2/D3; UX round 1, U3).

    It says NOTHING about where the draft is, and that is load-bearing rather
    than modest: the same category is reached from the peer-send spool fallback
    (a sender whose message could not be spooled), where there is no composer at
    all. The viewer appends that claim where it IS true, the same way it does
    for the oversize refusal one branch over.

    BOTH BASES, deliberately. ``admission_error`` decodes this family as
    ``ValueError`` (its other two categories are), and the refusal has been
    raised as a bare ``RuntimeError`` since it existed — by the runtime's own
    admission gates, by the peer-send spool fallback, and by every test that
    pins them. A ``ValueError``-only class would have silently changed the catch
    shape of a refusal three call sites already handle as ``RuntimeError``, the
    same trap ``OwnerAckTimeout(ConnectionError, TimeoutError)`` records one
    module over.
    """

    code = "runtime_retiring"

    #: The sentence, in the halves a viewer needs. ``HEAD`` states the situation,
    #: ``TAIL`` names the one act left; the owner's own rendering keeps them
    #: joined by ``REFUSED``, and a viewer with a COMPOSER inserts its claim
    #: between them instead of bolting it on after the full stop. Measured on the
    #: appended form (design round 3, D1; UX round 3, U5): two em dashes in one
    #: paragraph, a fragment opening after a ``.``, a strand at 100 columns and a
    #: one-word last line (``composer``) at 60 — in the one sentence whose job is
    #: to say the operator's work is safe. Exposed rather than re-composed so the
    #: two ends cannot drift.
    HEAD = "This session is switching to a newer build; the one it loaded is gone from disk."
    REFUSED = "The message was not admitted"
    TAIL = "send it again once the session is running again."

    def __init__(self) -> None:
        super().__init__(f"{self.HEAD} {self.REFUSED} — {self.TAIL}")


class ProfileRegistryUnavailable(ValueError):
    code = "profile_registry_unavailable"

    def __init__(self, count: int | None = None) -> None:
        """Report HOW MANY definitions are unreadable, never WHICH.

        The offending paths are logged at warning by the raising site. They are
        deliberately not interpolated here: this message crosses the transport
        boundary (see the module docstring), and a path names the operator's
        home directory, while a basename is an agent id. A count is
        content-free but still actionable -- it tells the user whether to look
        for one bad definition or several, and confirms the number is not zero,
        which is what the original wording could not do when the true cause was
        a directory that is not an agent at all.

        ``count`` is optional because the raising site does not always have a
        number: a scan that died on an ``OSError`` never attributed the failure
        to specific directories, and an older peer sends no count at all.
        """
        # Kept so the transport can forward the integer itself rather than
        # re-parsing it out of the rendered sentence.
        self.count = count
        detail = ""
        if count is not None:
            noun = "definition" if count == 1 else "definitions"
            detail = f" {count} agent {noun} could not be read."
        super().__init__(
            "The agent registry could not be read completely." + detail + " "
            "Repair unreadable or invalid agent definitions, then retry. "
            "No packaged profile was substituted."
        )


class MoveIndeterminate(Exception):
    """A move whose owner outcome is UNKNOWN, so nothing may be rolled back.

    WHY THIS IS NOT A ``RuntimeError``. The move route maps ``RuntimeError`` to
    an ordinary 409 refusal and, on that path, the durable marker and the
    viewer's fields are restored — the correct story for a refusal, and the
    WRONG one here. This class is raised when the retire REQUEST reached the
    owner and the answer did not come back definately (a dropped socket, an ack
    timeout): the owner may already have retired and accepted the new
    directory, so restoring the old marker would overwrite a committed move
    with a stale one, and the successor could then spawn in the old path while
    the receipt says the session is there.

    The honest answer is "reconcile before claiming either directory", which is
    what the route turns into a 503 whose body is
    ``{"code": :data:`code`, "message": <the sentence>}`` — the same shape
    ``DaemonRetiring`` and ``SubagentChildUnavailable`` use in that ladder, and
    the shape the desktop client already reads (it takes ``detail.message`` when
    ``detail`` is an object, so a named condition and a plain sentence both
    render). A subsequent move must first finish that reconciliation under the
    per-session move lock rather than act on an optimistic ``_cwd``.

    :attr:`detail` is the underlying cause — transport errno, a marker path, the
    three copies that disagreed — and is deliberately NOT on the wire: it names
    sockets, control ports and directories. Both raise sites LOG it instead,
    because a 503 whose cause is recorded nowhere leaves an operator with a
    generic "reconcile" and no thread to pull: the transport/unknown-outcome
    raise logs the exception with ``exc_info`` where it is still live
    (``session/attached.py``, ``set_working_directory``), and the settlement logs
    all four readbacks plus the path and errno of a repair write that failed
    (``server/utils/desktop_sessions.py``, ``_settle_unconfirmed_move``).

    :attr:`message` is overridden by exactly two callers, and both sentences are
    deliberate. The publication failure
    (``AttachedSession._publish_working_directory``) is its own sentence because
    the move itself IS confirmed there and only the viewer's repaint is not. The
    settlement refusal (``_settle_unconfirmed_move``) is its own because the
    directory is genuinely unresolved — and it names the action (reconnect, then
    reconcile) rather than the transport. Both carry :data:`code`, so a renderer
    keys on the condition instead of on the prose.
    """

    code = "move_outcome_unknown"

    def __init__(self, detail: str = "", *, message: str | None = None) -> None:
        self.detail = detail
        super().__init__(
            message
            or (
                "The move's outcome could not be confirmed. The session may have "
                "moved; reconnect, then reconcile its working directory before "
                "moving again."
            )
        )


def admission_error(code: str, count: int | None = None) -> ValueError | None:
    """Decode only an enumerated category, never owner-supplied message text.

    ``count`` is carried as its own integer field rather than being recovered
    from the peer's message, which is the whole point: the wording is rebuilt
    locally from the category, so the only thing crossing the transport is a
    number. An integer cannot name a path, a socket address or another
    conversation's identity, so it does not widen what the module docstring
    admits -- unlike ``str(exc)``, which is why that is still never trusted.

    Anything that is not a plain non-negative ``int`` is dropped rather than
    rendered: the far side is untrusted input, and a caller that omits the
    field (an older runtime) must degrade to the countless wording, not raise.
    """
    if code == AttachmentUnavailable.code:
        return AttachmentUnavailable()
    if code == RuntimeRetiring.code:
        return RuntimeRetiring()
    if code == ProfileRegistryUnavailable.code:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            count = None
        return ProfileRegistryUnavailable(count=count)
    return None
