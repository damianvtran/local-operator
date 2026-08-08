"""Which previous session ``--resume`` reopens.

A module of its own, and a deliberately tiny one: it imports nothing but
``pathlib``. The CLI has to resolve ``--resume`` before it starts anything (a
typo must be one line on stderr, not a full-screen app that launches, paints and
tears down to report it), and the CLI's startup path is guarded by tests that
FAIL if importing it drags in the engine, the providers, or even ``asyncio``.
Putting this policy in ``session_factory`` — the obvious home — is what broke
that guard: the sentinel alone pulled ``local_operator.harness`` and asyncio onto
every ``local-operator --help``.

Resuming is a filesystem question ("which transcript directory"), so nothing
here needs the engine. ``session_factory`` imports these same functions for the
transcript-directory decision, so the rule has one definition.
"""

from __future__ import annotations

from pathlib import Path

#: ``--resume`` with no id. A sentinel rather than a second boolean flag so the
#: whole "which session" decision stays ONE value threaded through one parameter.
RESUME_LATEST = "@latest"

#: The file whose presence makes a directory a resumable session. Also what the
#: recency ordering is read from: a directory's own mtime moves for reasons that
#: are not turns (retention sweeps touch it), so it is not the clock to use.
TRANSCRIPT_NAME = "transcript.jsonl"


class ResumeNotFound(Exception):
    """``--resume`` named a session that is not on disk (or none exist)."""


def resume_dir(config_dir: Path, requested: str) -> Path:
    """The session directory ``--resume`` names, or raise :class:`ResumeNotFound`.

    Resuming is deliberately CONFINED to ``sessions/``: an agent directory is
    that agent's own long-lived history, reached with ``--agent``/``--train``, and
    letting an id select one would silently append a throwaway session's turns
    onto it.

    Existence is checked HERE rather than left to the transcript reader, because
    a typo'd id would otherwise create an empty directory and start a session
    that looks resumed and has no history — the one failure a resume must never
    have.
    """
    sessions = config_dir / "sessions"
    if requested == RESUME_LATEST:
        candidates = [path for path in sessions.glob("*") if (path / TRANSCRIPT_NAME).is_file()]
        if not candidates:
            raise ResumeNotFound("no previous session to resume")
        return max(candidates, key=lambda path: (path / TRANSCRIPT_NAME).stat().st_mtime)
    # Reject anything that could escape the sessions directory: the id arrives
    # straight from argv and is used to build a path.
    if requested in ("", ".", "..") or "/" in requested or "\\" in requested:
        raise ResumeNotFound(f"not a session id: {requested!r}")
    candidate = sessions / requested
    if not (candidate / TRANSCRIPT_NAME).is_file():
        raise ResumeNotFound(f"no session {requested!r} to resume")
    return candidate


def resolve_resume_id(config_dir: Path, requested: str) -> str:
    """Validate ``--resume`` up front and return the CONCRETE session id.

    Returning the resolved id (never the ``@latest`` sentinel) means the session
    factory sees a real directory name, and the resume command the app prints on
    exit names the same id the user could pass back in.
    """
    return resume_dir(config_dir, requested).name


def recent_session_ids(config_dir: Path, limit: int = 10) -> list[str]:
    """Resumable session ids, newest first — what to offer after a bad ``--resume``.

    Best-effort: a directory that vanishes mid-scan (retention sweeps run
    concurrently) is skipped rather than raising out of an error path whose whole
    job is to be helpful.
    """
    rows: list[tuple[float, str]] = []
    for path in (config_dir / "sessions").glob("*"):
        try:
            rows.append(((path / TRANSCRIPT_NAME).stat().st_mtime, path.name))
        except OSError:
            continue
    rows.sort(reverse=True)
    return [name for _, name in rows[:limit]]
