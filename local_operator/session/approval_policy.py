"""The CONVERSATION's own tool-approval mode, persisted beside its transcript.

Why this file exists: ``tool_approval_mode`` in ``config.yml`` is the machine's
default for NEW sessions, and a running conversation follows it through
``config_watch`` — but an explicit ``/approvals ask|auto`` typed into one
conversation lived only in the handle's memory (``ServingSessionHandle.
_explicit_approvals_mode``). Two consequences the operator hit directly:

* resuming that conversation lost the choice, so it reopened on whatever the
  file said — for a user who hardened a session to ``ask`` and later set the
  global default to ``auto``, the resume silently came back ungated;
* a session started before a global default changed kept prompting, because
  nothing in the running conversation outranked the file in the LOOSENING
  direction except a choice nobody had recorded anywhere durable.

So the choice is a SIDECAR of the conversation: it travels with the session
directory (``--resume`` finds it), it is frozen for a fork (``fork.py``
deliberately does not copy it — see its exclusion notes), and it outranks the
global default IN BOTH DIRECTIONS. The asymmetry the file reader implements is
NOT the same one ``config.yml`` writes implement (see
:func:`local_operator.harness.approval.loosening_is_authorised`, which still
governs how a FILE-originated change reaches a session that never chose):

* an explicit conversation choice is the human's, so it wins both ways;
* a session with no choice inherits the file, and a loosening still needs
  operator authority.

**Failure direction: malformed means ``ask``.** Dropping a protective override
and inheriting ``auto`` is the unsafe direction — the user who hardened a
session would silently be ungated by a truncated write. A corrupt document is
therefore read as ``ask`` WITH a warning rather than as "no choice", and the
warning is surfaced (the handle emits it as a notice) because a silent
downgrade to prompting is still a change the user should see.

**The write is atomic and mode 0600**, for the reasons the other session
sidecars document (``resume.write_session_attachment``): a live owner and a
resume that is about to be refused can both touch the directory, and a reader
must never see a half-written document. ``tempfile.mkstemp`` gives the unique
same-directory name AND the 0600 mode in one step, and the directory's mtime is
preserved because recency ranks by transcript activity — journalling a policy
is bookkeeping ABOUT a session, never activity IN it.

The module imports only the standard library: it sits on the resume path and
must not drag the engine (same rule ``resume.py``'s docstring states).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

#: The sidecar's name inside a session directory. A dotted-free name like the
#: other sidecars (``title.json``, ``goal.json``) so it sorts as a document,
#: not as tooling, beside the transcript.
OVERRIDE_FILENAME = "approvals.json"

#: Bumped only when the PAYLOAD shape changes incompatibly. A file carrying any
#: other value is refused to ``ask`` rather than guessed at: a future schema may
#: mean something this build cannot represent, and the safe reading of "I do not
#: understand this" is the gate, never the absence of one.
SCHEMA_VERSION = 1

#: The two modes a conversation may record. ``None`` (in the payload) is the
#: third valid state — "no choice, follow the file".
_MODES = ("ask", "auto")


def _refusal_warning(reason: str) -> str:
    """The one sentence a corrupt sidecar earns.

    Names the file, the reason and the direction taken, in the shape the other
    approval notices use (a state, an explanation, and what happens next). It
    deliberately does NOT print the exception's message: the file is data, and
    a JSON decoder's offset is noise to the person reading it.
    """
    return (
        f"tool approvals: ask — {OVERRIDE_FILENAME} in this conversation could not be "
        f"read ({reason}), and a protective choice is never dropped, so this session "
        "prompts before write and command tools"
    )


def read_override(session_dir: Path) -> tuple[str | None, str | None]:
    """This conversation's recorded mode and a warning, or ``(None, None)``.

    ``(None, None)`` means "no usable choice recorded": the file is absent, or
    it explicitly stores ``null`` (a valid state the writer emits when a choice
    is cleared). Every OTHER abnormality FAILS CLOSED to ``ask`` and returns a
    warning sentence for the caller to surface — see the module docstring for
    why the unsafe direction is the one that inherits ``auto``.
    """
    path = session_dir / OVERRIDE_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError as error:
        # Unreadable is not absent: permissions or a directory in the file's
        # place means a choice may exist and must not be assumed away.
        logger.debug("could not read %s", path, exc_info=True)
        return "ask", _refusal_warning(error.__class__.__name__)
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        # UnicodeDecodeError is a ValueError, named here for the reader: a
        # process killed mid-write can cut the file inside a multi-byte
        # character, the same hazard ``read_goal_record`` documents.
        logger.debug("%s is not valid JSON", path, exc_info=True)
        return "ask", _refusal_warning("not valid JSON")
    if not isinstance(payload, dict):
        return "ask", _refusal_warning("not a JSON object")
    schema = payload.get("schema_version")
    # ``type is int`` rather than ``isinstance``: ``True`` is an ``int`` in
    # Python, and a boolean schema version is a malformed document, not v1.
    if type(schema) is not int or schema != SCHEMA_VERSION:
        return "ask", _refusal_warning(f"unsupported schema_version {schema!r}")
    override = payload.get("override")
    if override is None:
        return None, None
    if override not in _MODES:
        return "ask", _refusal_warning(f"invalid override {override!r}")
    return str(override), None


def write_override(session_dir: Path, mode: str | None) -> None:
    """Record ``mode`` for the conversation in ``session_dir``.

    ``None`` writes the explicit "no choice" document rather than deleting the
    file, so the payload shape has one writer and one reader (a deletion and a
    missing file would be indistinguishable from a crash between two writes).

    Raises on failure — every caller has a user in front of it whose receipt
    must say the mode was NOT saved (see ``ServingSessionHandle._approvals_slash``);
    swallowing here would produce the "applied and forgotten" state this module
    was written to end.
    """
    if mode is not None and mode not in _MODES:
        raise ValueError(f"approval mode must be one of {_MODES + (None,)}, got {mode!r}")
    path = session_dir / OVERRIDE_FILENAME
    try:
        previous_mtime: float | None = session_dir.stat().st_mtime
    except OSError:
        previous_mtime = None
    session_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=session_dir, prefix=f".{OVERRIDE_FILENAME}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": SCHEMA_VERSION, "override": mode}, handle)
            handle.flush()
            # fsync BEFORE the rename: the replace may be reordered otherwise,
            # leaving a renamed-but-empty file after a power loss, which reads
            # as malformed and (here) silently hardens the session to `ask`.
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:  # already gone, or never created past mkstemp
            pass
        raise
    finally:
        # Restored even on failure: mkstemp and the unlink above change the
        # directory's mtime too, and a failed write must not reorder the
        # ``/resume`` picker either.
        if previous_mtime is not None:
            try:
                os.utime(session_dir, (previous_mtime, previous_mtime))
            except OSError:
                logger.debug("could not restore %s mtime", session_dir, exc_info=True)


def resolve_effective(
    override: str | None, global_mode: str | None, *, pinned_auto: bool | None = None
) -> tuple[str, str]:
    """The mode in force and where it came from: ``(mode, provenance)``.

    Precedence, highest first, and the reason for the order:

    * ``pinned_auto`` — ``--yolo`` on this run (or its symmetrical pin to
      ``ask``). A command-line flag applies to exactly this run and outranks
      everything durable, which is what "for this run" means;
    * ``override`` — the conversation's own recorded choice. It outranks the
      machine default in BOTH directions: it is the human's decision about this
      conversation, and the file is about the next one. (The operator hit the
      other precedence as a bug: set a global default to ``auto`` and a session
      hardened to ``ask`` must not be loosened by it, and a session chosen
      ``auto`` must not be re-armed by a later global ``ask``.)
    * ``global_mode`` — ``tool_approval_mode`` from ``config.yml``.

    Unknown spellings of either input land on ``ask`` rather than raising: this
    function is called on the boot path (``exec_control``, ``session_factory``)
    where a corrupt value must degrade to the safe gate, never to a crash.
    Provenance is returned so a receipt can say WHY a mode is in force instead
    of asserting a value the reader cannot place.

    THIS IS THE DEFINITION, not the only copy of the code. The two production
    readers state the same order inline — ``exec_control.start_exec_control``'s
    seed and ``session_factory``'s plain headless gate — because both already
    hold the inputs in their own typed form (a config bool there, a mode string
    here). The order they implement is this one, and
    ``tests/unit/session/test_approval_policy.py`` pins the table, so a third
    reader has a definition to follow rather than a new interpretation to
    invent.
    """
    if pinned_auto is not None:
        return ("auto" if pinned_auto else "ask"), "pin"
    if override is not None:
        if override in _MODES:
            return str(override), "session"
        # An unknown spelling that reached here (the reader already refuses a
        # corrupt file to `ask`) must NOT fall through to a global ``auto``:
        # that is the same unsafe direction the reader exists to prevent, one
        # layer up.
        return "ask", "session"
    if global_mode in _MODES:
        return str(global_mode), "global"
    return "ask", "global"
