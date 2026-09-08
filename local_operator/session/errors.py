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

    def __init__(self) -> None:
        super().__init__(
            "The agent registry could not be read completely. "
            "Repair unreadable or invalid agent definitions, then retry. "
            "No packaged profile was substituted."
        )


def admission_error(code: str) -> ValueError | None:
    """Decode only an enumerated category, never owner-supplied message text."""
    if code == AttachmentUnavailable.code:
        return AttachmentUnavailable()
    if code == ProfileRegistryUnavailable.code:
        return ProfileRegistryUnavailable()
    return None
