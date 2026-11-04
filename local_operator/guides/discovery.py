"""Discover the small, packaged Local Operator guide catalog.

Guides are harness-owned operational knowledge, not user-authored skills. Their
frontmatter descriptions are semantic routing signals and the short text shown
to the model; the markdown body stays out of context until a ``guide://`` read.
Keeping the catalog packaged makes it available to pip, desktop-UI, and source
installs without copying files into each user's configuration directory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from local_operator.skills.discovery import Skill, parse_frontmatter
from local_operator.skills.protocol import resolve_resource_url


class Guide(Skill):
    """One immutable harness guide routed through the shared semantic index.

    ``requires`` names an availability predicate from the ``requires``
    frontmatter key (see :data:`GUIDE_PREDICATES`). Discovery itself stays
    static and network-free; the SESSION drops a guide whose predicate fails
    before it builds the router, the classification roster and the
    ``guide://`` resolver (:func:`guides_available`), so an ineligible session
    can neither see, recommend nor read it.
    """

    requires: str | None = None


def _radient_team(config_dir: Path | None) -> bool:
    # The org Agent Hub is a Radient Team feature and must not be advertised to
    # anyone signed out or outside a team-plan org (operator constraint). The
    # gate is the cached record: zero network, fail closed.
    if config_dir is None:
        return False
    from local_operator.hub_org.eligibility import is_eligible  # lazy: hub stack

    return is_eligible(config_dir)


#: ``requires:`` value -> predicate over the session's config root. An UNKNOWN
#: value fails closed: a typo must hide a gated guide, never publish it.
GUIDE_PREDICATES: dict[str, Callable[[Path | None], bool]] = {
    "radient-team": _radient_team,
}


def guides_available(guides: list[Guide], config_dir: Path | None) -> list[Guide]:
    """The guides whose ``requires`` predicate holds for ``config_dir``. Never raises."""

    kept: list[Guide] = []
    for guide in guides:
        if guide.requires is None:
            kept.append(guide)
            continue
        predicate = GUIDE_PREDICATES.get(guide.requires)
        try:
            if predicate is not None and predicate(config_dir):
                kept.append(guide)
        except Exception:  # noqa: BLE001 - a failing predicate hides its guide
            continue
    return kept


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
        requires = metadata.get("requires")
        if requires is not None and (not isinstance(requires, str) or not requires.strip()):
            # Present but unusable reads as an unknown predicate (hidden), not
            # as "no requirement" (published).
            requires = "invalid"
        guides.append(
            Guide(
                name=name.strip(),
                description=description.strip(),
                file_path=guide_md,
                base_dir=child,
                source=str(root),
                resource_type="guide",
                requires=requires.strip() if isinstance(requires, str) else None,
            )
        )
    return guides


def make_guide_resolver(guides: Mapping[str, Skill]):
    """Return a read-tool adapter for ``guide://`` URLs.

    Resolver errors are returned as content so the model sees the available
    names and can self-correct in one tool round, matching ``skill://``.

    ``OSError`` is caught for the same reason ``make_skill_resolver`` catches
    it, and the two must stay in step: both adapters share
    :func:`resolve_resource_url`, so any filesystem failure reachable through
    one is reachable through the other (a GUIDE.md deleted or chmod-000'd
    between discovery and the read). Catching only ``ValueError`` here let a
    ``PermissionError`` escape a resolver whose contract is "never raises",
    on the one surface that is packaged with the harness (review F7).
    """

    def resolver(url: str) -> str | None:
        if not url.startswith("guide://"):
            return None
        try:
            return resolve_resource_url(url, guides, scheme="guide", label="guide")
        except (ValueError, OSError) as exc:
            return str(exc)

    return resolver
