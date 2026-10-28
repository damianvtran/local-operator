"""The frozen LEGACY ``$skill`` payload, for tests that pin replay and naming.

The WRITER (``skills.invoke.render_invocation``) is gone; the READER
(``skills.invoke.typed_line_of``) is kept forever because every transcript
written before the tail-block shape still carries this form, and old
transcripts never migrate. A hand-spelled tag would pin a shape the product
never produced, so the attribute escaping below reproduces the deleted writer's
rules exactly: ``&`` first, then the two characters that could end the
attribute or the tag, and a newline as ``&#10;``.
"""

from __future__ import annotations

_DEFAULT_BODY = "# Research\n\nSKILL_BODY_MARKER\n"


def legacy_skill_payload(typed: str, *, body: str = _DEFAULT_BODY, name: str = "research") -> str:
    """One ``<skill … invocation="…">`` payload, as the deleted writer built it."""
    escaped = typed
    for old, new in (
        ("&", "&amp;"),
        ('"', "&quot;"),
        ("<", "&lt;"),
        (">", "&gt;"),
        ("\n", "&#10;"),
    ):
        escaped = escaped.replace(old, new)
    return (
        f"The user invoked the `{name}` skill directly. Follow it for this "
        "request. Its reference files, if any, are listed at the end of the body "
        "and are read with `skill://<name>/<path>`.\n"
        f'<skill name="{name}" invocation="{escaped}">\n{body}\n</skill>'
    )
