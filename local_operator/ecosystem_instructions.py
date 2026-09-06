"""User-scope instruction files shared with other agent tools.

``<config_dir>/system_prompt.md`` is lop's own machine-wide instructions file
and the only one lop WRITES. But an operator running lop next to Claude Code,
Codex, opencode or droid keeps one set of standing preferences — output style,
commit conventions, safety rules — and before this module the only way to get
them into lop was to maintain a second copy under a lop-specific filename.

lop already imports two other categories from other tools' user-scope
directories: skills are scanned out of ``~/.omp/agent/skills``,
``~/.claude/skills``, ``~/.codex/skills`` and ``~/.agents/skills``
(``skills/api.py``), and MCP servers are read from ``~/.claude.json``,
``~/.cursor/mcp.json`` and ``~/.codex/config.toml`` (``mcp/config.py``).
Instructions were the one category left out — lop loaded skills from
``~/.agents/skills`` while ignoring an ``AGENTS.md`` sitting directly beside
them. This module closes that asymmetry.

Contract, mirroring the skills loader deliberately:

- A fixed-order tuple of paths (:data:`ECOSYSTEM_INSTRUCTION_PATHS`), read in
  order, so two machines with the same files compute the same result.
- **Read-only.** ``system_prompt.md`` remains the sole write target for
  Settings → Instructions and ``GET``/``PATCH /v1/config/system-prompt``, so
  those surfaces cannot drift and lop never writes into another tool's file.
- Placed BEFORE ``system_prompt.md`` in the prompt, so lop's own instructions
  are read last and win on conflict — the same ranking native skills have over
  imported ones, and the same reason the MCP loader appends Codex last.
- Identical content collapses (:func:`content_digest`). An operator whose
  ``system_prompt.md`` was generated from the ecosystem file by a sync script
  has two byte-identical sources; paying context for both, on every cached
  request of every session, would be the worst possible outcome of adding this.
- :data:`ECOSYSTEM_INSTRUCTIONS_ENV` replaces the default set with a
  colon-separated path list, and an EMPTY value disables the feature outright —
  the same override shape as ``LOCAL_OPERATOR_SKILL_EXTRA_ROOTS``, which gives
  "point it somewhere else" and "turn it off" in one variable instead of two.

Why ``~/.agents/AGENTS.md`` and nothing else by default. There is no standard
for user-scope agent instructions — the agents.md spec covers repository files
only — but the FILENAME has converged, and two tool-neutral directories have
more than one independent implementation: ``~/.agents/`` (Cline, Factory droid)
and ``~/.config/`` (Amp, Crush). ``~/.agents/`` is chosen because lop already
reads that exact directory for skills, so this requires no new relationship
with a path lop does not already know. A bare ``AGENTS.md`` at the XDG config
root is a materially bigger namespace claim and is left to the override; the
tuple exists so adding it later is one line rather than a refactor.

``~/.local-operator/AGENTS.md`` is deliberately NOT read. It would gain no
interoperability (no other tool reads that directory) and would create a
write-path ambiguity: with both it and ``system_prompt.md`` present, what
``GET /v1/config/system-prompt`` returns and where ``PATCH`` writes stop having
one answer. Keeping every ecosystem file read-only avoids the question.

Symlinks ARE followed here, unlike :mod:`local_operator.context_files`. The two
policies differ because the trust boundaries do: repo guidance is discovered by
walking into directories cloned from anywhere, so it refuses links under
``O_NOFOLLOW``, while these are fixed paths in the operator's own home
directory — the same trust domain as ``system_prompt.md``, whose loader follows
links for the express purpose of pointing the file at a dotfiles checkout.
Versioning shared instructions in dotfiles is the main reason this feature is
wanted at all, so refusing the link would refuse the use case.
"""

from __future__ import annotations

import hashlib
import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger("local_operator.ecosystem_instructions")

#: User-scope instruction files imported from the shared agent-tool namespace,
#: read in this fixed order. Relative to the home directory; see the module
#: docstring for why this is ``~/.agents/`` and why it is a tuple of one.
ECOSYSTEM_INSTRUCTION_PATHS: tuple[Path, ...] = (Path(".agents") / "AGENTS.md",)

#: Replaces the default set when set. Colon-separated paths (``~`` expanded);
#: an EMPTY value disables ecosystem instructions entirely, so an operator who
#: wants only ``system_prompt.md`` gets exactly that. Shaped after
#: ``LOCAL_OPERATOR_SKILL_EXTRA_ROOTS`` rather than the on/off
#: ``LOCAL_OPERATOR_CONTEXT_FILES``, because a path list needs to be
#: redirectable and not merely switchable.
ECOSYSTEM_INSTRUCTIONS_ENV = "LOCAL_OPERATOR_ECOSYSTEM_INSTRUCTIONS"

#: Per-file byte cap. The joined result is bounded again on the instructions
#: budget by the caller; this one exists so a single pathological file is never
#: read into memory whole just to be discarded a moment later.
MAX_FILE_BYTES = 64 * 1024


def content_digest(text: str) -> str:
    """Stable digest of instruction content, for collapsing duplicates.

    Whitespace-stripped before hashing so a trailing newline — the difference
    between a hand-edited file and one written by a sync script — does not
    defeat the dedup and charge the operator twice for one set of rules.
    """
    return hashlib.sha256(text.strip().encode("utf-8", errors="replace")).hexdigest()


def ecosystem_instruction_files() -> list[Path]:
    """The instruction paths to read, in order. Empty when disabled or absent.

    Missing files are filtered HERE rather than left for the reader to skip,
    because this list is also the user-facing answer to "what is lop reading",
    and an answer naming files that do not exist is not one. Override paths are
    filtered the same way so the override behaves like the set it replaces.
    """
    raw = os.environ.get(ECOSYSTEM_INSTRUCTIONS_ENV)
    if raw is None:
        candidates = [Path.home() / relative for relative in ECOSYSTEM_INSTRUCTION_PATHS]
    else:
        candidates = [Path(part).expanduser() for part in raw.split(":") if part.strip()]
    return [candidate for candidate in candidates if candidate.is_file()]


def _read_bounded(path: Path) -> str:
    """Read one regular file, following links, without exceeding the cap.

    Symlinks are followed (see the module docstring), but a non-regular target
    is still refused: this runs on the synchronous session-construction path
    with no timeout above it, so a fifo here would block startup forever —
    a strictly worse failure than having no instructions.

    ``ecosystem_instruction_files`` already drops non-regular paths via
    ``is_file()``, so in the ordinary arrangement this guard never fires. It
    exists for the TOCTOU window: a regular file swapped for a fifo between
    that listing and this open, which is precisely the case that would hang
    rather than raise.

    ``O_NONBLOCK`` is what makes the guard reachable at all, and is load-
    bearing rather than defensive tidiness. Opening a fifo ``O_RDONLY`` blocks
    in the kernel until a writer appears, so control would never reach the
    ``fstat`` below — the check would be ordered after the syscall it exists to
    protect. With the flag, the open returns a descriptor immediately and
    ``S_ISREG`` refuses it. Regular files are unaffected: ``O_NONBLOCK`` has no
    meaning for them, and the read below returns the same bytes either way.
    """
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            probe = stream.read(MAX_FILE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(probe) > MAX_FILE_BYTES:
        logger.warning(
            "ecosystem instructions at %s exceed %d bytes; reading the first %d",
            path,
            MAX_FILE_BYTES,
            MAX_FILE_BYTES,
        )
    # ``utf-8-sig`` for the same reason the native loader uses it: a BOM from a
    # Windows editor would otherwise survive into the prompt ahead of rule one.
    return probe[:MAX_FILE_BYTES].decode("utf-8-sig", errors="replace")


def load_ecosystem_instructions(skip_digests: frozenset[str] = frozenset()) -> str:
    """Imported user-scope instructions, joined in order. ``""`` when there are none.

    ``skip_digests`` carries the digests of content the caller has ALREADY
    taken — in practice ``system_prompt.md`` — so an operator who generates the
    native file from the ecosystem one does not pay for both copies on every
    cached request.

    Failures degrade rather than breaking startup, matching
    ``load_user_instructions``: an unreadable file is skipped and undecodable
    bytes are replaced, because a bad byte in a shared instructions file should
    cost one glyph, never a session.
    """
    parts: list[str] = []
    seen = set(skip_digests)
    for path in ecosystem_instruction_files():
        try:
            text = _read_bounded(path)
        except OSError:
            continue
        stripped = text.strip()
        if not stripped:
            continue
        digest = content_digest(stripped)
        if digest in seen:
            logger.debug("skipping %s: identical to instructions already loaded", path)
            continue
        seen.add(digest)
        # INFO, not DEBUG: this file is written by another tool and named in no
        # lop-owned setting, yet it changes the system prompt of every session
        # and every subagent — the one import whose provenance an operator
        # cannot otherwise recover. It is not startup chatter: the record is
        # emitted only when a file actually EXISTS and actually contributed, so
        # the default install (no ``~/.agents/AGENTS.md``) stays silent, and on
        # the TUI it lands in the rotating log file rather than on screen
        # because ``file_logging`` detaches the console handlers. The dedup
        # skip above stays at DEBUG: nothing reached the prompt, so there is
        # nothing to account for.
        logger.info("loaded ecosystem instructions from %s (%d chars)", path, len(stripped))
        parts.append(stripped)
    return "\n\n".join(parts)
