"""Desktop fork auto-tagging: the child's intent sidecar and its one write.

LOH-11 V2-3. A desktop fork may ask the naming call to assign tags from the
same reply that names the conversation. The *intent* has to survive the gap
between the fork route (which is where the human enabled it) and the naming
worker (which runs on the child's first substantive turn, in whichever process
hosts that runtime), so the route writes a small sidecar into the child's
session directory and the worker reads it there. A missing sidecar means off —
the shipped state for every surface that does not opt in.

Two properties are load-bearing:

- The sidecar carries the child's **fork-time assignment revision**, not a
  value re-read at naming time. The write is conditional on that revision
  through the store's shipped ``expected_revision``/``TagRevisionConflict``
  path, so a human who assigns and then clears tags before the first turn has
  still edited the row (the revision moves on any real change) and auto-tagging
  skips. Eligibility is a snapshot, deliberately.
- Tags are **applied once**: the worker settles the intent after the naming
  attempt whose title was accepted, and the sidecar is removed. A later manual
  clear is not a trigger to re-run, and the title heal retry never reads the
  sidecar.

Nothing here creates a category. Model output is resolved against the contract
the naming call was given, then filtered against the live registry at write
time (a category deleted during the provider call is dropped); an empty result
assigns nothing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from local_operator.session import naming, session_tags

logger = logging.getLogger(__name__)

#: The per-child marker. Its presence is the only claim that auto-tagging was
#: requested for this fork; its removal is the apply-once receipt.
AUTO_TAG_FILENAME = "auto-tag.json"

#: Matches the desktop session-id contract, spelled here for the same reason
#: ``session_tags._entity_path`` spells it: this module must not import the
#: catalogue (and its runtime dependencies) to validate a path segment.
_SESSION_ID_RE = re.compile(r"[a-f0-9]{12}")


@dataclass(frozen=True)
class AutoTagIntent:
    """One fork's captured intent: which store, which child, which revision."""

    root: Path
    session_id: str
    revision: str


def _sidecar(root: Path, session_id: str) -> Path:
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("Invalid entity id")
    return Path(root) / "sessions" / session_id / AUTO_TAG_FILENAME


def record_fork(root: Path, session_id: str) -> bool:
    """Snapshot a fresh fork's auto-tag eligibility; True iff a sidecar landed.

    Called by the desktop fork route only when the request carried
    ``auto_tag``. A child that ALREADY carries tags gets no sidecar: auto-tag
    never runs over an assignment a human could have made, and recording an
    intent there would only manufacture a revision conflict later.

    Never raises. The fork has already succeeded by the time this runs, and the
    sidecar is an optional decoration: an unwritable session directory must not
    turn a successful fork into an error response the client would retry. A
    replay of the fork request never reaches here — the receipt journal returns
    the first answer — so the write is at-most-once per request id.
    """
    try:
        state = session_tags.read_entity_tags(root, "session", session_id)
        if state["tags"]:
            return False
        payload = json.dumps(
            {
                "version": 1,
                "auto_tag": True,
                "tags_revision": str(state["revision"]),
            }
        )
        path = _sidecar(root, session_id)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=".auto-tag-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(payload + "\n")
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
        return True
    except (OSError, ValueError):
        logger.warning("auto-tag: could not record fork intent for %s", session_id, exc_info=True)
        return False


def read(root: Path, session_id: str) -> AutoTagIntent | None:
    """The fork's intent, or ``None`` when auto-tagging is off or unreadable.

    A malformed sidecar is OFF rather than an error: the file can be truncated
    by a crash, and the naming call must fall back to its title-only prompt
    rather than fail a turn. It is left in place — inert, because nothing but
    this reader consults it — until a later fork overwrites the same path.
    """
    try:
        raw = json.loads(_sidecar(root, session_id).read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.debug("auto-tag: unreadable sidecar for %s", session_id, exc_info=True)
        return None
    if not isinstance(raw, dict) or raw.get("version") != 1 or raw.get("auto_tag") is not True:
        return None
    revision = raw.get("tags_revision")
    if not isinstance(revision, str) or not revision:
        return None
    return AutoTagIntent(Path(root), session_id, revision)


def clear(intent: AutoTagIntent) -> None:
    """Spend the intent. Best-effort: a stuck sidecar must not fail a turn."""
    try:
        _sidecar(intent.root, intent.session_id).unlink(missing_ok=True)
    except (OSError, ValueError):
        logger.debug("auto-tag: could not clear the sidecar for %s", intent.session_id)


def live_contract(root: Path) -> naming.TagContract | None:
    """The naming contract for the registry as it stands NOW, or None if empty.

    Read per naming attempt rather than captured at fork time: a category list
    frozen at the fork would let the prompt offer an id the registry has since
    deleted. The write is re-validated against the registry again for the same
    reason — the provider call takes long enough for the list to move.
    """
    categories = session_tags.read_categories(root)
    if not categories:
        return None
    return naming.TagContract(tuple((item["id"], item["name"]) for item in categories))


def settle(intent: AutoTagIntent, tag_ids: Sequence[str]) -> session_tags.EntityTags | None:
    """Write the accepted sample's tags once, then spend the intent.

    Conditional on the FORK-TIME revision: a human's edits — including
    add-then-clear, which moves the revision without changing the final value —
    win by making the store refuse with ``TagRevisionConflict``. The intent is
    spent either way, because the naming attempt has answered and auto-tagging
    must never reassert itself later.

    Never raises. Registry drift is resolved against the live categories at
    this instant: ids removed during the provider call are dropped, and when
    none survive there is no assignment. Model output never creates a category.
    """
    applied: session_tags.EntityTags | None = None
    try:
        known = {item["id"] for item in session_tags.read_categories(intent.root)}
        surviving = [key for key in tag_ids if key in known][: session_tags.MAX_TAGS]
        if surviving:
            applied = session_tags.set_entity_tags(
                intent.root,
                "session",
                intent.session_id,
                surviving,
                surviving[0],
                expected_revision=intent.revision,
            )
    except session_tags.TagRevisionConflict:
        # A manual edit landed first. That is the intended precedence, not a
        # failure: keep the human's assignment and stop ownership here.
        logger.debug("auto-tag: %s was edited during naming; human tags win", intent.session_id)
    except (OSError, ValueError):
        logger.warning("auto-tag: assignment failed for %s", intent.session_id, exc_info=True)
    finally:
        clear(intent)
    return applied
