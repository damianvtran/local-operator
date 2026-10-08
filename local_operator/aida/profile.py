"""The guarded write path for profile notes (R23).

WHAT THIS IS. During the first conversation Aida asks the operator a few
details about themselves (name, how they like to be addressed, an email if
they want it remembered, anything else set-up relevant) and records what they
agree to into ``<config root>/system_prompt.md`` — the custom-instructions file
every future session and subagent reads (``guide://configuration``). This
module is the ONE code path for that write, exposed to her as
``lop aida note`` so the recorded text cannot choose the file.

WHY A GUARD RATHER THAN AN EDIT. The instructions file is the operator's own
standing rules, shared by the desktop Settings box, ``GET``/``PATCH
/v1/config/system-prompt`` and the session loader. An agent told to "write it
down" with its normal file tools can target the wrong path (its cwd, another
config root, a writable copy in a sandbox), silently clobber rules it did not
read, or leave a half-written file if it dies mid-edit. So the write is
path-FREE by construction — the caller supplies text, never a path — and the
module resolves the file itself through the same ``paths.config_dir()`` the
route and the loader use, refuses anything that would leave the config root,
and replaces the file atomically.

THE REFUSALS, one word each (this function never raises; the CLI maps each word
to the sentence a reader sees, and the tests pin every branch):

- ``empty`` — nothing to record after stripping.
- ``invalid`` — the text contains this module's own section markers, which
  would corrupt the section for every later write; refused rather than
  silently stripped.
- ``oversize`` — the write would push the file past
  :data:`MAX_FILE_CHARS` (the loader's own budget: a file past it is
  truncated when assembled, so recording more would silently cost the operator
  rules they wrote earlier). Curate first.
- ``unsafe`` — the resolved file would escape the config root (a symlinked
  ``system_prompt.md`` pointing elsewhere, or a root that cannot be resolved);
  writing through it would reach a file the operator did not name. A symlink
  that stays INSIDE the root is written through to its target rather than
  replaced, so a dotfiles-style arrangement survives a recording.
- ``failed`` — an unexpected I/O error; logged, and the file is left as it
  was (the temp-file staging means there is no torn write to repair).

THE SECTION. Notes live in ONE marked section so her recordings cannot be
confused with the operator's own text and cannot grow forever:

    <!-- aida:profile -->
    ## About the operator
    - Name: …
    <!-- /aida:profile -->

Notes are stored as bullets, one per non-empty line, and a line already
present is not appended twice (that is what makes a greeting retry idempotent).
The operator stays free to edit anything OUTSIDE the markers — nothing here
touches those bytes — and deleting the section is a supported way to say "forget it".

SEQUENCING. This writer holds no cross-process lock: the file's other writers
(the desktop route, an editor) do not participate in one, so a lock here would
be theatre. What it does guarantee is atomicity (staged write + ``os.replace``,
the same discipline ``aida/state.py`` uses) and that reads-modify-write cycles
come from Aida's own serialized turn. Secrets are out of scope for the guard —
no heuristic can reliably tell a secret from a fact — and are covered by her
instructions ("never record secrets") and the operator's ability to edit the
section afterwards.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

#: The custom-instructions file, relative to the config root. The one constant
#: shared with ``server/routes/config.py`` and the session loader's docs.
SYSTEM_PROMPT_NAME = "system_prompt.md"

#: The loader's assembled-instructions budget (``session_factory``). Text past
#: it is truncated at READ time, so a larger file silently costs the operator
#: rules they wrote earlier; the writer refuses past it instead.
MAX_FILE_CHARS = 64_000

#: The section markers. HTML comments are invisible in every markdown renderer
#: the file is likely to meet, and the pair makes the section movable and
#: deletable with find-and-delete.
SECTION_START = "<!-- aida:profile -->"
SECTION_END = "<!-- /aida:profile -->"
SECTION_HEADING = "## About the operator"


def instruction_file(config_dir: Path | str | None = None) -> Path:
    """The operator's custom-instructions file, resolved per call.

    ``config_dir()`` is read from the environment on every call (its own
    docstring's rule), so an isolated run and a test each resolve the file
    they actually configured; callers that already resolved a root pass it in.
    """
    if config_dir is None:
        from local_operator.paths import config_dir as resolve_config_dir

        root = resolve_config_dir()
    else:
        root = Path(config_dir)
    return Path(root) / SYSTEM_PROMPT_NAME


def _bullets(text: str) -> list[str]:
    """The note as bullet lines: one per non-empty line, markers preserved."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ")):
            lines.append(line)
        else:
            lines.append(f"- {line}")
    return lines


def _section_bounds(content: str) -> tuple[int, int] | None:
    """``(start, end)`` character offsets of the marked section, or ``None``.

    ``str.find`` semantics are deliberate: the FIRST start marker and the
    FIRST end marker after it delimit the section, so a stray end marker in
    the operator's own prose cannot swallow the file.
    """
    start = content.find(SECTION_START)
    if start == -1:
        return None
    end = content.find(SECTION_END, start + len(SECTION_START))
    if end == -1:
        return None
    return start, end


def _render_section(bullets: list[str]) -> str:
    body = "\n".join(bullets)
    return f"{SECTION_START}\n{SECTION_HEADING}\n{body}\n{SECTION_END}"


def record_profile_note(
    text: str,
    *,
    config_dir: Path | str | None = None,
) -> str:
    """Record ``text`` into the instructions file's profile section.

    Returns one word: ``"recorded"``, ``"duplicate"``, or a refusal —
    ``"empty"``, ``"invalid"``, ``"oversize"``, ``"unsafe"``, ``"failed"``
    (see the module docstring for each). Never raises.
    """
    raw = text or ""
    if SECTION_START in raw or SECTION_END in raw:
        return "invalid"
    bullets = _bullets(raw)
    if not bullets:
        return "empty"

    path = instruction_file(config_dir)
    try:
        root_real = Path(os.path.realpath(path.parent))
    except OSError:
        return "unsafe"
    # A symlinked system_prompt.md whose target is outside the root is the one
    # way this path-free API could still write somewhere else; resolve before
    # trusting, and refuse rather than follow. An in-root symlink is followed
    # deliberately — the write targets the RESOLVED file so the link itself is
    # never replaced by `os.replace`.
    write_target = path
    try:
        # The resolve is UNCONDITIONAL, and that is the fix, not a style
        # choice: `path.exists()` follows the link, so a DANGLING symlink to
        # an out-of-root target read as "nothing there", the guard was
        # skipped, and the write then CREATED that target through the link
        # (review round 1, F2). `realpath` resolves a missing target fine.
        if not Path(os.path.realpath(path)).is_relative_to(root_real):
            return "unsafe"
        if path.is_symlink():
            write_target = Path(os.path.realpath(path))
    except OSError:
        return "unsafe"

    try:
        content = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        logger.warning("aida: could not read %s", path, exc_info=True)
        return "failed"

    bounds = _section_bounds(content)
    if bounds is not None:
        start, end = bounds
        body = content[start:end]
        # Dedupe by the bullet's TEXT (either marker counts), so a `* item`
        # already present is not re-added as `- item`.
        existing = set()
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ")):
                existing.add(stripped[2:].strip())
        fresh = [bullet for bullet in bullets if bullet[2:].strip() not in existing]
        if not fresh:
            return "duplicate"
        # Insert inside the section, immediately before its end marker.
        new_content = content[:end].rstrip("\n") + "\n" + "\n".join(fresh) + "\n" + content[end:]
    else:
        base = content.rstrip("\n") + "\n\n" if content.strip() else ""
        new_content = base + _render_section(bullets) + "\n"

    if len(new_content) > MAX_FILE_CHARS:
        return "oversize"

    try:
        _atomic_write(write_target, new_content)
    except OSError:
        logger.warning("aida: could not write %s", write_target, exc_info=True)
        return "failed"
    return "recorded"


def _atomic_write(path: Path, content: str) -> None:
    """Staged write + ``os.replace`` — a reader can never see a torn file.

    The same discipline ``aida/state.py``'s ``write_json`` uses; the temp name
    starts with ``.`` so nothing scanning the directory picks it up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


#: The suffix that marks a bullet the SIGN-IN wrote rather than one Aida
#: recorded with the operator's consent. It is what lets a re-login replace
#: its own lines (a renamed account, a different account) instead of
#: appending a second "Name:" beside the first, while never touching a line
#: she or the operator wrote.
RADIENT_ACCOUNT_TAG = "(Radient account)"


def record_account_identity(
    name: str,
    email: str,
    *,
    config_dir: Path | str | None = None,
    tag: str = RADIENT_ACCOUNT_TAG,
) -> str:
    """Write ``Name:``/``Email:`` bullets from a provider SIGN-IN. Idempotent.

    WHY THIS EXISTS. A Radient sign-in proves who the operator is (the
    ``id_token`` claims, see ``providers/oauth/radient.py``), and the
    instructions file is the stable prefix every session and subagent reads
    (``session_factory``'s custom-instructions block). Writing the identity
    there means every conversation knows the operator's name from the first
    turn, and Aida's first contact confirms it instead of asking for an email
    the login already recorded (audit A5/A6).

    IDEMPOTENT BY REPLACEMENT, not by dedupe: every bullet ending in ``tag`` is
    the sign-in's own, so they are dropped and rewritten as one pair. A second
    login with the same identity therefore answers ``"duplicate"`` and writes
    nothing; a changed name replaces the old line. Bullets without the tag —
    what she recorded, what the operator typed — are never touched.

    Same guards and refusal words as :func:`record_profile_note` (it routes
    through the same path checks and atomic write). Never raises.
    """
    wanted = []
    if name.strip():
        wanted.append(f"- Name: {name.strip()} {tag}")
    if email.strip():
        wanted.append(f"- Email: {email.strip()} {tag}")
    if not wanted:
        return "empty"
    if any(SECTION_START in line or SECTION_END in line for line in wanted):
        return "invalid"
    path = instruction_file(config_dir)
    try:
        root_real = Path(os.path.realpath(path.parent))
        if not Path(os.path.realpath(path)).is_relative_to(root_real):
            return "unsafe"
        write_target = Path(os.path.realpath(path)) if path.is_symlink() else path
    except OSError:
        return "unsafe"
    try:
        content = path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        logger.warning("aida: could not read %s", path, exc_info=True)
        return "failed"

    bounds = _section_bounds(content)
    if bounds is None:
        base = content.rstrip("\n") + "\n\n" if content.strip() else ""
        new_content = base + _render_section(wanted) + "\n"
    else:
        start, end = bounds
        lines = content[start:end].splitlines()
        kept = [line for line in lines if not line.strip().endswith(tag)]
        existing_tagged = [line.strip() for line in lines if line.strip().endswith(tag)]
        if existing_tagged == wanted:
            return "duplicate"
        # The tagged pair goes FIRST under the heading: it is the fact every
        # later note is about, and a stable position keeps the prefix stable
        # across re-logins (a moved line is a cache miss on every session).
        heading_at = next(
            (i for i, line in enumerate(kept) if line.strip() == SECTION_HEADING), None
        )
        insert_at = (heading_at + 1) if heading_at is not None else 1
        body_lines = kept[:insert_at] + wanted + kept[insert_at:]
        body = "\n".join(body_lines)
        new_content = content[:start] + body.rstrip("\n") + "\n" + content[end:]

    if len(new_content) > MAX_FILE_CHARS:
        return "oversize"
    try:
        _atomic_write(write_target, new_content)
    except OSError:
        logger.warning("aida: could not write %s", write_target, exc_info=True)
        return "failed"
    return "recorded"


def record_radient_login(credential: dict, *, config_dir: Path | str | None = None) -> str:
    """Record a fresh Radient OAuth credential's identity. Never raises.

    The ONE call both login hosts make (``providers.controller`` for the TUI
    and desktop, ``providers.auth_cli`` for ``lop login``), so the two cannot
    disagree about what lands in the file. A credential without claims (an
    older Radient backend, a pasted key) answers ``"empty"`` and writes nothing.
    """
    try:
        return record_account_identity(
            str(credential.get("name") or ""),
            str(credential.get("email") or ""),
            config_dir=config_dir,
        )
    except Exception:  # noqa: BLE001 — a label must never fail a login
        logger.warning("aida: could not record the Radient identity", exc_info=True)
        return "failed"
