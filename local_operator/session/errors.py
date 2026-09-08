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
    if code == ProfileRegistryUnavailable.code:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            count = None
        return ProfileRegistryUnavailable(count=count)
    return None
