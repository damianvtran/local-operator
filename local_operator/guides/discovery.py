"""Discover the small, packaged Local Operator guide catalog.

Guides are harness-owned operational knowledge, not user-authored skills. Their
frontmatter descriptions are semantic routing signals and the short text shown
to the model; the markdown body stays out of context until a ``guide://`` read.
Keeping the catalog packaged makes it available to pip, desktop-UI, and source
installs without copying files into each user's configuration directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from local_operator.skills.discovery import Skill, parse_frontmatter
from local_operator.skills.protocol import resolve_resource_url


class Guide(Skill):
    """One immutable harness guide routed through the shared semantic index."""


def discover_guides() -> list[Guide]:
    """Load packaged ``<name>/GUIDE.md`` files in deterministic name order.

    A malformed guide is skipped instead of breaking every session. Package
    contents are trusted release artifacts, but graceful degradation still
    matters for editable installs while a contributor is changing a guide.
    """

    root = Path(__file__).parent
    guides: list[Guide] = []
    try:
        children = sorted(root.iterdir(), key=lambda path: (path.name.lower(), path.name))
    except OSError:
        return guides

    for child in children:
        if child.name.startswith(".") or not child.is_dir():
            continue
        guide_md = child / "GUIDE.md"
        if not guide_md.is_file():
            continue
        try:
            metadata = parse_frontmatter(guide_md.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        name = metadata.get("name")
        description = metadata.get("description")
        if not isinstance(name, str) or not name.strip():
            name = child.name
        if not isinstance(description, str) or not description.strip():
            continue
        guides.append(
            Guide(
                name=name.strip(),
                description=description.strip(),
                file_path=guide_md,
                base_dir=child,
                source=str(root),
                resource_type="guide",
            )
        )
    return guides


def make_guide_resolver(guides: Mapping[str, Skill]):
    """Return a read-tool adapter for ``guide://`` URLs.

    Resolver errors are returned as content so the model sees the available
    names and can self-correct in one tool round, matching ``skill://``.
    """

    def resolver(url: str) -> str | None:
        if not url.startswith("guide://"):
            return None
        try:
            return resolve_resource_url(url, guides, scheme="guide", label="guide")
        except ValueError as exc:
            return str(exc)

    return resolver
