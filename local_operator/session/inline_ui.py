"""Request-only discovery for negotiated inline widgets in an ordinary answer."""

from __future__ import annotations

from typing import TypedDict

# Request-only discovery pointer. Store schemas and examples in the packaged
# guide so adding a component doesn't expand the provider prompt.
INSTRUCTIONS = (
    "Inline charts and data grids are available on this turn. When useful, read "
    "guide://inline-ui for the component catalog, then emit a standalone lop-ui "
    "fenced JSON block in the answer. Keep the takeaway in normal prose."
)


def instructions_for(surface: str | None, *, viewer_capable: bool = False) -> str | None:
    """The in-process terminal stamps its default at admission; others negotiate."""
    if surface in {"tui", "desktop", "mobile"} and viewer_capable:
        return INSTRUCTIONS
    return None


class InlineUiOptIn(TypedDict, total=False):
    inline_ui: bool


def inline_ui_opt_in(enabled: bool | int) -> InlineUiOptIn:
    """Omit the key on legacy-compatible internal calls, without an untyped splat."""
    return {"inline_ui": True} if enabled else {}
