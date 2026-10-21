"""The desktop sidebar's destination layout: one small preference, two regions.

The sidebar is TWO ordered lists cut at one fold line (design §2): the rows
drawn above ``More`` (``visible``) and the rows drawn after it (``more``).
``visible ++ more`` is the single order a drag operates on; the boundary is
stored, not derived, because a drag across the fold changes both lists at once
and a round trip must not silently re-cut them.

**Why the backend stores it.** The renderer could keep this in
``localStorage``, and that is exactly what the design rejects: the app is
reinstalled and updated, the renderer's storage is cleared, and the operator's
arrangement is gone. Living under the config dir means it survives app updates,
there is ONE source of truth for every window, and a future terminal or phone
surface can read the same resource without a second preference (design "The
TUI decision"). The terminal does not read or write it today.

**Durability versus atomicity, both deliberate.**

- A missing file is not an error: the defaults ARE today's arrangement, so
  :func:`read_layout` answers them for an absent, unreadable, corrupt or
  invalid file. A corrupt preference must never cost the sidebar its column;
  the worst outcome is the shipped order.
- A write is staged in the config directory and published with
  ``os.replace``: a reader (this daemon, or the TUI's config reader) never sees
  a torn file, and a process that dies mid-write leaves the old layout intact.
  This is the same discipline every small index in this tree uses
  (``librarian.state.write_json``, ``monitors.state._atomic_write_json``).
  It is atomicity against torn READS, not an fsync-level durability promise.
- :func:`write_layout` raises ``OSError`` instead of swallowing it. A save the
  user was told succeeded and did not is worse than an error (design §4): the
  route turns the failure into a 500, and the client keeps the arrangement it
  was already showing rather than believing in a preference that will not
  survive the next launch.

**Why a SUBSET is valid and a duplicate is not.** The client may persist only
the destinations it knows about (the design has the client re-insert missing
ids from the default), so the backend accepts any subset of the allow-list.
Duplicates would make ``visible ++ more`` ambiguous — the same id would draw
twice and a drag would have two valid answers — so they are refused rather
than silently de-duplicated, and the 9-id bound falls out of uniqueness over
the allow-list.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

logger = logging.getLogger(__name__)

#: The stable wire ids of the destinations the sidebar can draw. The labels are
#: localised and are NOT ids (``team-bridge`` vs "Team Bridge"); a destination
#: missing from a build is removed from the RENDER, never from a stored layout,
#: so this list only grows.
SIDEBAR_DESTINATION_IDS: tuple[str, ...] = (
    "aida",
    "library",
    "projects",
    "agents",
    "schedules",
    "browser",
    "team-bridge",
    "agent-hub",
    "mesh",
)

#: Today's arrangement, exactly (design §2). This is the answer for a missing
#: file and the arrangement a fresh install shows, so it is also the shape every
#: reader can rely on.
DEFAULT_SIDEBAR_LAYOUT: dict[str, list[str]] = {
    "visible": ["aida", "library", "projects", "agents", "schedules", "browser"],
    "more": ["team-bridge", "agent-hub", "mesh"],
}

#: One file, beside ``config.yml``, inside the config directory. A separate
#: file rather than a ``config.yml`` key: the layout is written per drag, and a
#: whole-config rewrite on every drag would race the settings surface and the
#: renderer's own config writes for no benefit.
SIDEBAR_LAYOUT_FILE = "sidebar-layout.json"

#: The stored schema's version. A file from a FUTURE version is treated as
#: invalid (defaults) rather than guessed at: a newer writer may have re-cut
#: the semantics, and the shipped order is the safe answer.
SIDEBAR_LAYOUT_VERSION = 1


class SidebarLayout(BaseModel):
    """A whole stored preference: both regions, in draw order.

    Both fields are REQUIRED. A body that omits ``more`` must not silently
    mean "restore the default fold", because that is a destructive answer to
    what is far more likely a malformed request; the client always sends the
    whole preference (design §4), and a partial file on disk degrades to the
    defaults in :func:`read_layout` instead of being completed here.
    """

    # ``extra="forbid"`` at every layer: an unknown key is a client sending a
    # field this backend does not understand (a newer client, or a typo), and
    # accepting-and-ignoring it would report a save that dropped part of what
    # was asked for.
    model_config = ConfigDict(extra="forbid")

    visible: list[str]
    more: list[str]

    @model_validator(mode="after")
    def _known_unique_and_bounded(self) -> "SidebarLayout":
        """Refuse unknown ids, duplicates, and more ids than exist.

        Unknown ids are refused rather than dropped: dropping them would let a
        client persist a layout it believes in while the backend stored
        something else. Duplicates are refused for the reason the module
        docstring gives; the 9-id bound then follows, and is stated explicitly
        so the invariant does not depend on a reader noticing that it follows.
        """
        seen: set[str] = set()
        for region in ("visible", "more"):
            for destination in getattr(self, region):
                if destination not in SIDEBAR_DESTINATION_IDS:
                    raise ValueError(
                        f"unknown sidebar destination {destination!r}; "
                        f"known ids are {', '.join(SIDEBAR_DESTINATION_IDS)}"
                    )
                if destination in seen:
                    raise ValueError(f"sidebar destination {destination!r} appears twice")
                seen.add(destination)
        if len(seen) > len(SIDEBAR_DESTINATION_IDS):
            raise ValueError(
                f"a sidebar layout holds at most {len(SIDEBAR_DESTINATION_IDS)} destinations"
            )
        return self


def default_layout() -> dict[str, Any]:
    """The shipped arrangement, as a fresh copy.

    A copy rather than the module constant: :data:`DEFAULT_SIDEBAR_LAYOUT` is
    read by callers and models, and one caller mutating the answer in place
    would silently redefine every later request's defaults.
    """
    return {
        "version": SIDEBAR_LAYOUT_VERSION,
        "visible": list(DEFAULT_SIDEBAR_LAYOUT["visible"]),
        "more": list(DEFAULT_SIDEBAR_LAYOUT["more"]),
    }


def _layout_path(config_dir: Path) -> Path:
    return Path(config_dir) / SIDEBAR_LAYOUT_FILE


def read_layout(config_dir: Path) -> dict[str, Any]:
    """The stored layout, or the defaults when there is no trustworthy one.

    Best-effort by contract (design §4): every failure mode — no file, an
    unreadable file, malformed JSON, a non-object, an unknown version, a
    layout that fails validation — degrades to :func:`default_layout` and is
    logged at warning level so a persistent corruption is diagnosable. The
    route never 500s because a preference file is unreadable; it would turn a
    cosmetic problem into a sidebar that does not load.
    """
    path = _layout_path(config_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default_layout()
    except (OSError, ValueError):
        logger.warning("sidebar layout: unreadable %s; using defaults", path, exc_info=True)
        return default_layout()
    if not isinstance(payload, dict):
        logger.warning("sidebar layout: %s is not an object; using defaults", path)
        return default_layout()
    if payload.get("version") != SIDEBAR_LAYOUT_VERSION:
        logger.warning(
            "sidebar layout: %s has version %r, not %d; using defaults",
            path,
            payload.get("version"),
            SIDEBAR_LAYOUT_VERSION,
        )
        return default_layout()
    try:
        layout = SidebarLayout.model_validate(
            {"visible": payload.get("visible"), "more": payload.get("more")}
        )
    except ValidationError:
        logger.warning("sidebar layout: %s is invalid; using defaults", path, exc_info=True)
        return default_layout()
    return {
        "version": SIDEBAR_LAYOUT_VERSION,
        "visible": list(layout.visible),
        "more": list(layout.more),
    }


def write_layout(config_dir: Path, layout: dict[str, Any]) -> dict[str, Any]:
    """Validate, then atomically store one layout. Raises on failure.

    The caller supplies only the two regions; the VERSION is this module's to
    write, so a client can never pin a schema. Validation happens here as well
    as at the route so no other caller can persist a layout the reader would
    reject (a ``ValidationError`` escapes, rather than being stored).

    ``OSError`` is deliberately not caught: the route answers 500 on it, and a
    caller that does not expect one gets the real failure instead of a bool
    that lies. The temp file starts with ``.`` so nothing scanning the config
    directory mistakes it for a preference, and ``mkstemp`` creates it
    ``0600``, matching the config directory's own permissions.
    """
    validated = SidebarLayout.model_validate(
        {"visible": layout.get("visible"), "more": layout.get("more")}
    )
    stored: dict[str, Any] = {
        "version": SIDEBAR_LAYOUT_VERSION,
        "visible": list(validated.visible),
        "more": list(validated.more),
    }
    directory = Path(config_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SIDEBAR_LAYOUT_FILE
    descriptor, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{SIDEBAR_LAYOUT_FILE}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(stored, handle, separators=(",", ":"), sort_keys=True)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return stored
