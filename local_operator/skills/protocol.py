"""The ``skill://`` URL protocol — progressive disclosure for skill content.

Contract shared with stream A's ``read`` tool (docs/REWRITE.md §C):
``resolve_skill_url(url, skills)`` returns ``None`` when ``url`` is not a
skill URL (so the caller falls through to normal paths), a ``str`` with the
resolved content, and raises ``ValueError`` for bad skill URLs.

Resolution ladder:

1. ``skill://<name>`` — exact-name lookup of the skill; miss raises with ALL
   available names listed. That error message is the model's self-correction
   path: it reads the list and retries with a real name.
2. No path → the full ``SKILL.md`` text (frontmatter included; the body is
   the payload, the frontmatter is cheap context).
3. With path → joined under ``base_dir`` after rejecting absolute paths and
   ``..`` segments, then re-checked with ``Path.resolve().is_relative_to`` —
   the second check is the symlink defense (a link inside the skill dir can
   point anywhere).
4. Directory target → a rendered listing (``name/ (dir)`` lines, capped at
   500 entries) so the model can discover ``references/`` content; file
   target → text read with ``errors="replace"``, capped at 200KB by
   bounding the READ itself, never loading the whole file first.

``hide`` skills resolve normally here — hiding only affects prompt listings.
Dotfiles are unlisted AND unreadable: directory listings skip any name
starting with ``.``, and a direct read of a dotfile path is rejected with an
explicit message, so "not listed" never silently means "still reachable".
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlsplit

from local_operator.skills.discovery import Skill

MAX_READ_BYTES = 200 * 1024
"""Files larger than this are truncated; skills ship text, not binaries."""

_MAX_LISTING_ENTRIES = 500
"""Directory listings never exceed this many entries (RS-14); the overflow is
summarized with a marker rather than streamed into context."""

_TRUNCATION_MARKER = "\n\n[... content truncated at 200KB ...]"


def _read_text_capped(path: Path) -> str:
    """Read a file as UTF-8 (lossy), truncating at MAX_READ_BYTES.

    The cap bounds the READ, not just the returned string: we read at most
    ``MAX_READ_BYTES + 1`` bytes so a multi-GB file inside a skill dir never
    gets fully loaded into memory before being cut.
    """
    with path.open("rb") as fh:
        raw = fh.read(MAX_READ_BYTES + 1)
    if len(raw) > MAX_READ_BYTES:
        return raw[:MAX_READ_BYTES].decode("utf-8", errors="replace") + _TRUNCATION_MARKER
    return raw.decode("utf-8", errors="replace")


def _list_directory(directory: Path) -> str:
    """Render a directory listing: ``name/ (dir)`` for dirs, plain names for
    files, deterministic alphabetical order. Dotfiles are skipped (they are
    unlisted and unreadable by design) and the listing is capped at
    ``_MAX_LISTING_ENTRIES`` with an overflow marker."""
    lines: list[str] = []
    try:
        entries = sorted(directory.iterdir(), key=lambda p: (p.name.lower(), p.name))
    except OSError:
        return "(unreadable directory)"
    shown = 0
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if shown >= _MAX_LISTING_ENTRIES:
            break
        lines.append(f"{entry.name}/ (dir)" if entry.is_dir() else entry.name)
        shown += 1
    hidden = sum(1 for entry in entries if not entry.name.startswith(".")) - shown
    if hidden > 0:
        lines.append(f"[... {hidden} more entries not shown ...]")
    return "\n".join(lines) if lines else "(empty directory)"


def _resolve_child(skill: Skill, raw_path: str) -> str:
    """Join ``raw_path`` under the skill's base_dir with containment checks.

    ``raw_path`` comes from ``urlsplit`` and always starts with ``/`` (the
    separator after the netloc), so that leading slash is the URL separator,
    not an absolute filesystem path. ``..`` segments are rejected outright,
    and the resolved target is re-checked against ``base_dir`` — that second
    check is what catches symlinks pointing out of the skill directory.
    """
    relative = unquote(raw_path.lstrip("/"))

    segments = relative.split("/")
    if relative.startswith("/") or any(part == ".." for part in segments):
        raise ValueError(
            f"Invalid skill path '{raw_path}': absolute paths and '..' segments " "are not allowed"
        )
    if any(part.startswith(".") and part != "." for part in segments):
        # "." alone just names the directory itself (the base-dir listing
        # probe); real dotfiles are unlisted and unreadable by design.
        raise ValueError(
            f"Invalid skill path '{raw_path}': dotfiles are not listed and " "cannot be read"
        )

    base = skill.base_dir.resolve()
    target = (skill.base_dir / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Invalid skill path '{raw_path}': escapes the skill directory")
    if not target.exists():
        raise ValueError(f"Skill path not found: skill://{skill.name}/{relative}")
    if target.is_dir():
        return _list_directory(target)
    return _read_text_capped(target)


def resolve_skill_url(url: str, skills: Mapping[str, Skill]) -> str | None:
    """Resolve a ``skill://`` URL to content, or None for non-skill URLs.

    ``skills`` maps skill name → Skill. Returns None immediately when the URL
    does not use the ``skill`` scheme so callers can chain resolvers. Raises
    ``ValueError`` for unknown skill names (listing all available names) and
    for unsafe paths — both are surfaced to the model as tool errors.
    """
    if not url.startswith("skill://"):
        return None

    parts = urlsplit(url)
    name = unquote(parts.netloc)
    if not name:
        raise ValueError("Skill URL missing a name: expected skill://<name>")

    skill = skills.get(name)
    if skill is None:
        available = ", ".join(sorted(skills.keys())) or "(none)"
        raise ValueError(f"Unknown skill: {name}\nAvailable: {available}")

    path = parts.path
    # No path, a bare "/", or nothing but slashes ("//") → the SKILL.md text.
    # Routing "//" here keeps _resolve_child's input non-empty (no dead branch).
    if not path or not path.strip("/"):
        return _read_text_capped(skill.file_path)
    return _resolve_child(skill, path)
