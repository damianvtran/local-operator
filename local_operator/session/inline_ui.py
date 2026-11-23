"""Request-only discovery for negotiated inline widgets in an ordinary answer."""

from __future__ import annotations

# Request-only discovery pointer. Store schemas and examples in the packaged
# guide so adding a component doesn't expand the provider prompt.
INSTRUCTIONS = (
    "Inline charts and data grids are available on this turn. When useful, read "
    "guide://inline-ui for the component catalog, then emit a standalone lop-ui "
    "fenced JSON block in the answer. Keep the takeaway in normal prose."
)


def instructions_for(surface: str | None, *, viewer_capable: bool = False) -> str | None:
    """A local TUI can always render; attached desktop and mobile must negotiate."""
    if surface == "tui" or (surface in {"desktop", "mobile"} and viewer_capable):
        return INSTRUCTIONS
    return None
