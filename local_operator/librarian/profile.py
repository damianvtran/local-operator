"""The guarded write path for the Librarian's learned taxonomy.

WHAT THIS IS, AND WHAT IT IS NOT. The Librarian learns the operator's
vocabulary: which tag categories they use, how they name things, recurring
themes, and aliases worth avoiding. Those lessons are NOT profile notes and
must never land in ``system_prompt.md``: that file is the operator's own
standing instructions, shared by every session and subagent, and an agent
learning that "reports go under topic:weekly" would be polluting the
operator's custom instructions with catalog bookkeeping. (The Aida precedent
is :mod:`local_operator.aida.profile`, whose subject IS who the operator is;
this module's subject is how the LIBRARY is organised.)

So the durable home is the Library metadata store's ``conventions`` table
(design §5.1), written through :meth:`MetadataStore.set_convention`. That is
the store the ``library`` agent tool reads, the surface the desktop shows, and
the one place whose schema owns the row shapes.

WHY A GUARD RATHER THAN A RAW SQL WRITE. The store is the source of truth for
everything a person or the Librarian decided, and a second writer that opened
the database itself could write an unknown kind, an id that collides with a
tag name, or a row with no body. This module pins the small vocabulary and
rejects before touching the file: the caller supplies text, never SQL and
never a path.

THE REFUSALS, one word each (this function never raises; the seed and any
later CLI map each word to a sentence, and the tests pin every branch):

- ``empty``     — nothing to record after stripping.
- ``invalid``   — the id is not a safe identifier, or the kind is outside the
                  documented vocabulary. Refused rather than coerced: a typo'd
                  kind would silently create a section nothing reads.
- ``oversize``  — the body is past :data:`MAX_BODY_CHARS`. A convention is a
                  rule, not a document; a pasted transcript belongs in the
                  session, not in a row every read has to load.
- ``unavailable`` — the store refused (locked, corrupt, full disk). Logged;
                  the caller retries or reports, and nothing is half-written
                  because the store's own write is one transaction.

``actor`` defaults to ``"librarian"`` and is recorded as ``created_by``; a
later user-facing editor may pass ``"user"`` without changing this module.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The convention kinds the design names (§5.1): preferred tag categories,
#: naming rules, recurring themes and aliases to avoid. Small and closed on
#: purpose — every reader of this table renders the kinds, so an unknown one
#: is a row nothing shows.
CONVENTION_KINDS: tuple[str, ...] = ("tags", "naming", "themes", "aliases")

#: Ids are stable handles (``tags.weekly-reports``), not prose: lowercase
#: alnum plus dot/dash/underscore, so a reader can address a row from a URL,
#: a CLI flag or a seed instruction without quoting rules.
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

#: A rule, not a document. Roughly two paragraphs; past that the operator's
#: own instructions are the right home, not a catalog convention.
MAX_BODY_CHARS = 2_000


def _store(config_dir: Path | str | None):
    from local_operator.library.metadata import MetadataStore
    from local_operator.paths import config_dir as resolve_config_dir

    root = Path(config_dir) if config_dir is not None else resolve_config_dir()
    return MetadataStore(root)


def record(
    convention_id: str,
    kind: str,
    body: str,
    *,
    config_dir: Path | str | None = None,
    actor: str = "librarian",
) -> str:
    """Record (or replace) one convention row. Returns one word, never raises.

    Replacement is deliberate: a convention is a living rule, and the
    Librarian's daily pass refines it as the catalog grows. The caller keeps
    the id stable and rewrites the body; the store stamps ``updated_at``.
    """
    identifier = (convention_id or "").strip()
    if not _ID_RE.fullmatch(identifier):
        return "invalid"
    if kind not in CONVENTION_KINDS:
        return "invalid"
    text = (body or "").strip()
    if not text:
        return "empty"
    if len(text) > MAX_BODY_CHARS:
        return "oversize"
    try:
        _store(config_dir).set_convention(identifier, kind, text, actor=actor)
    except Exception:  # noqa: BLE001 — the caller renders the refusal word
        logger.warning("librarian: could not record convention %s", identifier, exc_info=True)
        return "unavailable"
    return "recorded"


def conventions(config_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Every recorded convention, tolerant: an unreadable store reads empty."""
    try:
        return _store(config_dir).conventions()
    except Exception:  # noqa: BLE001 — a read for a prompt/receipt never raises
        logger.warning("librarian: could not read conventions", exc_info=True)
        return []
