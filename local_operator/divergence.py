"""Divergence guard — notice when a turn ASSERTS a project root nothing gave it.

WHY THIS EXISTS
---------------
2026-11-06, session ``73084269786c``: a QA child ran an ordinary review for
five minutes, then a single assistant turn switched to a task nobody on this
machine had given it and spent ninety minutes building ``~/squeeze``, an image
compression service. No harness-supplied input contained that project: the
transcript holds exactly one user message (the QA brief), the foreign phrase
occurs zero times outside the model's own reasoning, and ``whoami`` in the same
run answered ``damian``. See
``~/workspace/incidents/agent-context-contamination-20261106/INCIDENT.md``.

The harness cannot see why that happened (the request-assembly layer lives in
another service). What it CAN see is the shape the derailment took on its own
surface: from one turn onward, the model's tool calls named a project root that
appeared nowhere in its instructions — not in the system prompt, not in the
custom instructions, not in any user message, and — this is the part that makes
the measurement honest — not in its own earlier tool output either.

WHY THE PREDICATE IS NARROW (read this before widening it)
----------------------------------------------------------
``~/workspace/incidents/agent-context-contamination-20261106/analysis/
divergence/divergence-heuristic-baseline.md`` measured a literal
"novel path-shaped string" heuristic over 20 ordinary sessions: **78 flags per
100 turns**. 76% of those named a file whose ancestor directory the context
already named — the model doing the task, not diverging. The five narrowings
that measurement settled, and the ones encoded here:

1. **Only what the model ASSERTS, and only where it can act.** The proposal's
   own discriminator: a path inside ``cd ~/x 2>/dev/null`` is a path the model
   *tried*; a ``write`` argument or a ``mkdir`` operand is a path it *asserted*.
   Commands that swallow failure (``2>/dev/null``, ``|| true``, ``|| :``) are
   probes and are dropped per command SEGMENT — 41% of the narrowest baseline
   flags were exactly that, and they were not divergence.
2. **Project-root granularity, never file granularity.** A file under a named
   root is explained by the root. Only the first component under ``$HOME`` (or
   the ``/Users/<name>``-style prefix, or the first two components of anything
   else) is compared.
3. **The session's own earlier tool output is a source.** 48% of literal flags
   named something the model had just read in a directory listing; a guard that
   ignores tool output alerts on the model reading its results.
4. **Fire once per novel root, escalate on repetition or on a write.** The
   incident names its noun 117 times in 134 turns; one row carries the same
   information, and repeated mentions would otherwise bury the transcript.
5. **Skip roots no local decision could have produced.** The package caches,
   the config dir, the scratchpad, volatile temp roots — see ``_skip_roots``.

The result must stay at or below ~4.8 flags per 100 turns on that same sample
(``impl-evidence/detector/EVIDENCE.md`` carries the measurement of THIS module,
not of the baseline script).

MASKING RULE — THE ONE THAT KEEPS THE GUARD HONEST
---------------------------------------------------
Absence is established against the **on-disk bytes** of what the session
actually stored — the same bytes the model sent and received. The harness masks
credential-shaped material at the *render* seam (``local_operator.variables``,
``SessionStreamFn``), never before storage; the baseline measured 16 masked
characters in 4.16 M (0.0004%) and 0 verdict changes, precisely because the
scan read the stored form. **Render-time redaction must never move before
storage**: the moment "appeared nowhere" is decided against a rendered copy
rather than the stored one, a root that exists in the bytes could read as
absent (or worse, a masked span could make an explained root read as novel) and
the guard becomes decoration. So: read stored message content, never a rendered
view; never compare against displayed text.

WHAT IT CANNOT SEE — NAMED, NEVER ASSUMED AWAY
----------------------------------------------
A detector that says "nothing in this session named that root" is making a
claim about every carrier the model received. Three of them are genuinely
unreadable from here, and each is recorded per event in
:attr:`DivergenceEvent.unseen` so the record and the notice can say so instead
of asserting absence over them:

* ``masked_text`` — a value the loop masked **in storage** (a ``VariableStore``
  registration redacts the tool result before it is written) is not in the
  stored bytes; the root may have been delivered and then erased. Measured by
  QA round 1: a root printed into a tool result, masked, and then asserted
  fires here with the pre-mask bytes already gone.
* ``image_attachments`` — an instruction can arrive as a screenshot. Text
  extraction reads text; a path painted in an image is invisible.
* ``compacted_history`` — compaction replaces ``context.messages`` with a
  summary, so a root named before the cut may be absent from the list this
  judge is handed.

A fourth carrier is out of reach for a structural reason worth stating: the
**request-assembly service** composes the provider request outside this
process, so a root injected there is invisible here. That is the incident's own
residual uncertainty, and the notice's wording is scoped to the sources this
module actually read for that reason.

INSTRUCTIONS ARRIVE AS USER TURNS, NOT ONLY AS MESSAGES
-------------------------------------------------------
``CustomMessage`` rows carry no ``role``: the renderer turns them into user
turns when it builds the request (``harness/render.py``), so a wake prompt, a
hub delivery, a ``session_state`` goal delta or a relayed ask is an
INSTRUCTION the model genuinely received. Folding only ``role == "user"``
missed all of them and then claimed the root appeared in no user message —
false absence, the one error this module may never make. The set of carriers
is :data:`~local_operator.harness.message_types.MODEL_VISIBLE_INJECTED_TYPES`,
the same list the renderer uses, so a new aside type cannot be invisible only
to the guard.

OBSERVATION ONLY — NEVER A GATE
-------------------------------
Nothing here denies, approves, delays or re-routes a call. The judge returns
events, appends audit rows, and hands a payload to an optional host callback
that paints a transcript notice. Every entry point is wrapped so a failure in
this module can only ever cost the notice, never the turn. A flagged call runs
exactly as it would have.

WHY IT IS NOT UNDER ``local_operator/guardrails/``
--------------------------------------------------
That package ENFORCES: a judge there can deny a call before it runs. This one
may not, and keeping it beside the enforcement would invite exactly the change
the incident does not justify. The separation is the contract.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from local_operator.harness.approval import GATE_TIMEOUT_CUSTOM_TYPE
from local_operator.harness.message_types import MODEL_VISIBLE_INJECTED_TYPES
from local_operator.harness.rows import gate_timeout_subject
from local_operator.paths import config_dir

logger = logging.getLogger(__name__)

#: The audit log this module appends to. A sibling of
#: ``audit/public-copy-reads.jsonl`` (``tools/public_copy.py``) — the only other
#: machine-facing record set in the tree — because a transcript is deletable
#: (``session/cleanup.py`` owns the only path that removes one) and the incident
#: this guard was built for is the reason a record of a derailed turn must
#: outlive the session directory that produced it.
AUDIT_DIRNAME = "audit"
AUDIT_FILENAME = "divergence.jsonl"

#: One row per novel root, then at most one escalation row.
#:
#: The incident repeats ``squeeze`` in 117 of 134 turns. A per-mention record
#: would be 117 rows of the same fact and would drown the transcript notice it
#: is supposed to support; two rows carry "it appeared" and "it persisted".
MAX_ROWS_PER_ROOT = 2

#: Distinct roots that may paint a NOTICE in one session.
#:
#: ``MAX_ROWS_PER_ROOT`` bounds the record for one root; nothing bounded the
#: number of roots, and design round 1 measured the consequence: three roots
#: produced six notices and 20 amber rows, 62% of each row the same repeated
#: provenance sentence, on a 30-row screen. Past this many roots the audit log
#: is the record — the operator gets one row saying so, never a wall of rows
#: (design round 1, D6). The CAP is on notices only: every root still gets its
#: own audit rows, so nothing observed is lost.
MAX_NOTICED_ROOTS = 3

#: The mask a ``VariableStore`` value becomes when the loop redacts it before
#: storage (``local_operator/variables.py``). Its presence in the corpus is how
#: this module knows part of what the model read is not on disk any more.
_MASK_MARKER = "[redacted]"

#: The instruction sources this module can read, in the order the operator's own
#: list names them. A row records only the ones actually available to it.
_SOURCE_ORDER = (
    "system_prompt",
    "custom_instructions",
    "user_messages",
    "earlier_tool_output",
)

#: Carriers the model received that this module CANNOT read, in record order.
#: They are named on the row so absence is scoped rather than total — see
#: "WHAT IT CANNOT SEE" in the module docstring.
_UNSEEN_ORDER = ("masked_text", "image_attachments", "compacted_history")

#: Same flags, same reasoning as ``public_copy._AUDIT_OPEN_FLAGS``:
#: ``O_NONBLOCK`` turns a FIFO planted at the log path into a refusal (``ENXIO``)
#: instead of a blocked tool call, and regular files ignore it. Windows has no
#: FIFO at a filesystem path and no ``O_NONBLOCK``.
_AUDIT_OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NONBLOCK", 0)

#: Virtual schemes are stores, not filesystem targets: ``scratchpad://x`` names
#: the session's own store and ``memory:``/``skill://``/``guide://``/``spill://``
#: /``tool://`` name harness resources. None of them can be a foreign project
#: root, and treating them as paths produced every kind of nonsense in the
#: baseline scan (``~/local-operatorsessions`` was a missing slash, not a
#: project).
_VIRTUAL_SCHEMES = ("scratchpad://", "memory:", "skill://", "guide://", "spill://", "tool://")

#: Tools whose ARGUMENT is a target path, and whether naming it is a WRITE.
#:
#: ``write``/``edit`` create; ``read``/``grep``/``glob`` only look — and the
#: difference is what the escalation ladder keys on, because a model that
#: INVESTIGATES a root once and moves on is the baseline's flag #9 (a guess that
#: died in one probe), while one that WRITES there is tonight's incident. Both
#: are judged: the operator's scope is "any tool that names a target path". The
#: read side is the quiet one at root granularity — the literal-file form of this
#: heuristic priced ``read`` at 0.12 flags per 100 turns, and root granularity
#: plus the skip list only lowers that.
_FILE_TARGET_TOOLS: dict[str, tuple[tuple[str, ...], bool]] = {
    "write": (("path",), True),
    "edit": (("path",), True),
    "read": (("path",), False),
    "grep": (("path",), False),
    "glob": (("path",), False),
}

#: ``browser`` writes a file only for two actions (``screenshot``, ``record_start``),
#: and both take ``path``. Every other browser action is navigation.
_BROWSER_TARGET_ACTIONS = frozenset({"screenshot", "record_start"})

#: The commands the operator named, mapped to the operand that is CREATED.
#: ``cp``/``mv`` create their DESTINATION only; ``tee``/``mkdir``/``touch`` create
#: every operand. ``git clone`` and ``uv venv`` are the two the shipped pad
#: scanner (``tools/builtin._bash_created_entries``) does not model at all, and
#: they are also the two that would build a project from nothing.
_CREATING_COMMANDS = frozenset({"mkdir", "touch", "cp", "mv", "tee"})
_GIT_CLONE = "clone"
_UV_VENV = "venv"

#: Prefix commands that stand in front of the real command word
#: (``sudo mkdir``, ``timeout 60 touch``, ``env FOO=1 cp``). Same idea as
#: ``tools/builtin._COMMAND_PREFIXES``, trimmed to the words that appear in this
#: fleet's traffic; a prefix this list misses is a MISS (the safe direction).
_COMMAND_PREFIXES = frozenset(
    {"sudo", "timeout", "nice", "env", "command", "time", "nohup", "exec", "do", "then", "else"}
)

#: ``NAME=value`` ahead of a command word.
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")

#: A bare operand that belongs to a prefix rather than to the command
#: (``timeout 60``): a number with an optional unit.
_PREFIX_OPERAND = re.compile(r"\d+(?:\.\d+)?[smhd]?$")

#: Failure-swallowing in a SEGMENT (the per-segment probe rule). A ``||`` arm
#: that runs a no-op is handled at split time, because it is a property of the
#: separator plus the next segment rather than of this segment's text.
_SWALLOW_REDIRECT = re.compile(r"(?:2>|\$\s*>|&>)\s*/dev/null")
_SWALLOW_ARM = re.compile(r"^(?:true|:|exit\s+0)$")

#: Top-level shell separators, longest first so ``&&`` is not read as ``&``.
_SEPARATORS = ("&&", "||", ";", "|", "&", "\n")

#: A heredoc opener: ``<<EOF``, ``<<'EOF'``, ``<<-"EOF"``.
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


class DivergenceAuditError(RuntimeError):
    """The audit row could not be written.

    Raised rather than swallowed so a caller cannot report an event as recorded
    when nothing reached disk. :meth:`DivergenceJudge.observe_turn` catches it
    (an observer must never break a turn) and marks the event ``recorded``
    ``False``, which the notice text and the row's own absence both carry.
    """


@dataclass(frozen=True)
class DivergenceEvent:
    """One observation: a turn asserted a root nothing in its context named."""

    session_id: str
    parent_session_id: str
    turn_index: int
    root: str
    path: str
    tool: str
    call_id: str
    escalated: bool
    reason: str
    sources: tuple[str, ...]
    #: True for the row that FIRST announced this root. Escalation and first
    #: sighting are different facts (design round 1, D1: they used to paint
    #: identically), and a write on first sighting is BOTH — the copy has to be
    #: able to say "again" only when it really is a return.
    first_sighting: bool = True
    #: Carriers the model received that this judge could NOT read, so the
    #: absence claim is scoped rather than total. See the module docstring's
    #: "WHAT IT CANNOT SEE"; empty means every carrier this session had was
    #: read.
    unseen: tuple[str, ...] = ()
    recorded: bool = True

    def as_record(self) -> dict[str, Any]:
        """The JSONL row. Field names are the operator's list, not synonyms."""
        return {
            "v": 1,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kind": "divergence_assertion",
            "session": self.session_id,
            "parent": self.parent_session_id,
            "turn": self.turn_index,
            "root": self.root,
            "path": self.path,
            "tool": self.tool,
            "call_id": self.call_id,
            "escalated": self.escalated,
            "reason": self.reason,
            "sources": list(self.sources),
            "unseen": list(self.unseen),
            "first": self.first_sighting,
            "recorded": self.recorded,
        }

    def as_notice_payload(self) -> dict[str, Any]:
        """The details a transcript notice carries (``harness/rows.py`` renders)."""
        return {
            "root": self.root,
            "path": self.path,
            "tool": self.tool,
            "turn": self.turn_index,
            "escalated": self.escalated,
            "reason": self.reason,
            "sources": list(self.sources),
            "unseen": list(self.unseen),
            "first": self.first_sighting,
            "recorded": self.recorded,
        }


@dataclass
class _RootState:
    """Per-session memory of one root, so a repeat cannot flood the record."""

    first_turn: int
    escalated: bool = False
    rows: int = 0
    #: Every asserting path seen for this root, for the escalation row's detail.
    paths: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Path → project root
# ---------------------------------------------------------------------------


def _as_path(value: Any) -> Path | None:
    """A host-supplied working directory as a ``Path``, whatever shape it came in.

    ``ToolContext.cwd`` is declared ``str`` (``harness/types.py``), and the loop
    hands it straight through, while the replay driver and the tests pass a
    ``Path``. Normalising HERE — at the module's boundary — is what keeps that
    difference from being load-bearing: the first shipped version called
    ``cwd.resolve()`` on the string, raised ``AttributeError`` inside the
    observer's swallow, and the guard wrote zero rows in every live session
    while the ``Path``-passing replay evidence stayed green (QA round 1, case 2).

    Anything that is not a non-empty ``str``/``os.PathLike`` is ``None`` — a
    cwd this module cannot place, which :func:`_expand` treats as "relative
    paths are unknowable" rather than inventing a directory.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return Path(text) if text else None
    try:
        return Path(os.fspath(value))
    except TypeError:  # pragma: no cover - a host that passes neither shape
        return None


def _resolved(path: Path) -> Path:
    """``path`` with symlinks followed, or ``path`` unchanged if it cannot be."""
    try:
        return path.resolve()
    except (OSError, RuntimeError):  # pragma: no cover - nothing can stat it
        return path


def _expand(raw: str, cwd: Path | None, home: Path) -> Path | None:
    """``raw`` as the filesystem path it names, or ``None`` when it is not one.

    ``~``/``$HOME``/``${HOME}`` are the shell's ways of writing the same file
    (the pad scanner learned this the hard way: the same target fired spelled
    absolutely and went silent spelled through the home directory), ``..`` is
    resolved, and a relative path is joined to ``cwd``. A virtual scheme and an
    empty string are not paths. An unresolvable path (``ENAMETOOLONG``, a
    symlink loop) is ``None`` — a MISS, never a crash.
    """
    text = (raw or "").strip()
    if not text or "://" in text or text.startswith(_VIRTUAL_SCHEMES):
        return None
    if text.startswith("${HOME}"):
        text = str(home) + text[len("${HOME}") :]
    elif text.startswith("$HOME"):
        text = str(home) + text[len("$HOME") :]
    elif text == "~" or text.startswith("~/"):
        text = str(home) + text[1:]
    if "$" in text:
        # An UNEXPANDED shell variable names no root this process can resolve
        # (``> ~/$OUT/bad.txt`` was a real row in the incident session's own
        # replay). Guessing would assert a project on the operator's behalf,
        # which is the one thing this guard may not do.
        return None
    try:
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            # No cwd means the path cannot be placed: guessing would invent a
            # root, which is the one mistake this guard may not make.
            if cwd is None:
                return None
            candidate = cwd / candidate
        return candidate.resolve()
    except (OSError, RuntimeError):  # pragma: no cover - a path the OS cannot resolve
        return None


def project_root(path: Path, home: Path) -> str:
    """The project root ``path`` belongs to, at the granularity the guard judges.

    ``~/local-operator/.venv/bin/python`` is explained when the prompt names
    ``~/local-operator`` — that T3→T4 step is the single largest reduction in
    the baseline (34.3 → 7.7 flags per 100 turns), and it is why this returns a
    root and never a file. A path under ``$HOME`` yields ``~/<first component>``;
    the ``/Users/<name>`` / ``/home/<name>`` shapes yield three components
    (``/Users/damian/squeeze``), because two would make every path on this
    machine share the root ``/Users/damian`` and explain everything; anything
    else yields its first two components.

    BOTH sides are normalised: ``path`` arrives resolved (``_expand``) and a
    caller's ``home`` may not be (``Path.home()`` is whatever ``$HOME`` holds,
    which on a host whose home is a symlink is not the canonical form). An
    unresolved home compared against resolved paths fails the prefix test for
    every path under it, so each distinct project collapsed onto one root and
    ordinary work fired — reproduced in code review round 1 with a fixture home
    under ``/var`` reading ``~/squeeze/a`` as the root ``/private/var``.
    """
    try:
        home = home.resolve()
    except (OSError, RuntimeError):  # pragma: no cover - an unresolvable home
        pass
    parts = path.parts
    home_parts = home.parts
    if len(parts) > len(home_parts) and parts[: len(home_parts)] == home_parts:
        return f"~/{parts[len(home_parts)]}"
    if len(parts) >= 3 and parts[1] in ("Users", "home"):
        return "/" + "/".join(parts[1:4])
    if len(parts) >= 3:
        return "/" + "/".join(parts[1:3])
    return "/" + "/".join(parts[1:]) if len(parts) > 1 else "/"


# ---------------------------------------------------------------------------
# Session identity
# ---------------------------------------------------------------------------


ORIGIN_NAME = "origin.json"


def origin_parent_session_id(session_dir: Path | None) -> str:
    """The session that delegated ``session_dir``, or ``""`` when none did.

    ``origin.json`` is written once by whoever CREATES the session directory
    (``resume.mark_session_origin``) as ``{"origin": ..., "parent": <id>}``; a
    child session and a user's own conversation are otherwise identical on disk
    by the time anything reads them. Absence means the user's own session — the
    same direction ``resume.session_origin`` takes, and for the same reason: a
    record that says too little is recoverable, one that mislabels a person's
    own conversation as a child's is not.

    Tolerant on purpose. ``errors="replace"`` matters more than it looks: the
    marker is written non-atomically, so a child killed mid-write leaves the
    file cut inside a multi-byte character, and a strict decode raises
    ``UnicodeDecodeError`` (a ``ValueError``) past any ``except OSError``. One
    such sidecar once took down every ``--resume`` on this machine.
    """
    if session_dir is None:
        return ""
    try:
        raw = (Path(session_dir) / ORIGIN_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    try:
        payload = json.loads(raw)
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    parent = payload.get("parent")
    return parent if isinstance(parent, str) else ""


def root_key(root: str) -> str:
    """The lowercase basename an explained-check matches on.

    ``~/squeeze`` → ``squeeze``; ``/Users/damian/squeeze`` → ``squeeze``;
    ``~/Library/Application Support`` → ``application support``. A root is
    explained when this token appears anywhere in the context corpus, which is
    the baseline's own containment test applied at root granularity.
    """
    return root.rsplit("/", 1)[-1].strip().lower()


# ---------------------------------------------------------------------------
# Shell extraction (assertion, not probe)
# ---------------------------------------------------------------------------


def _strip_heredocs(command: str) -> str:
    """Drop heredoc BODIES, keeping the opener line.

    Measured on the ordinary-session sample: without this, prose inside a
    ``cat > file <<'EOF'`` body produced redirect "targets" that were really
    the text after a ``>`` in a sentence — ``~/6,371,817,635)``,
    ``~/30_000_000:``, ``/=8:`` — three of the eighteen rows the first
    measurement raised, all of them parser artefacts rather than assertions.
    The opener line keeps its redirect target, which is the part that matters.
    """
    if "<<" not in command:
        return command
    kept: list[str] = []
    terminators: list[str] = []
    for line in command.split("\n"):
        if terminators:
            if line.strip() == terminators[0]:
                terminators.pop(0)
            continue
        kept.append(line)
        for match in _HEREDOC.finditer(line):
            terminators.append(match.group(2))
    return "\n".join(kept)


def _split_segments(command: str) -> list[tuple[str, str | None]]:
    """Split ``command`` into ``(segment, separator that FOLLOWS it)``.

    A quote-aware walk rather than a regex split, because ``echo "a;b"`` is one
    command and the pad scanner's own tokenizer makes the same distinction. A
    separator is ``&&``/``||``/``;``/``|``/``&``/newline.
    """
    segments: list[tuple[str, str | None]] = []
    buffer: list[str] = []
    index = 0
    length = len(command)
    quote: str | None = None
    while index < length:
        char = command[index]
        if quote is not None:
            buffer.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
            buffer.append(char)
            index += 1
            continue
        if char == "\\" and index + 1 < length:
            buffer.append(char)
            buffer.append(command[index + 1])
            index += 2
            continue
        matched = next((sep for sep in _SEPARATORS if command.startswith(sep, index)), None)
        if matched is None:
            buffer.append(char)
            index += 1
            continue
        segments.append(("".join(buffer), matched))
        buffer = []
        index += len(matched)
    segments.append(("".join(buffer), None))
    return segments


def _tokens(segment: str) -> list[str]:
    """A segment's words, quotes removed (``> "/tmp/x.log"`` → ``/tmp/x.log``).

    Small on purpose: its only consumer is an advisory extractor, so anything it
    does not model degrades to a MISS rather than to a wrong assertion.
    """
    words: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    index = 0
    length = len(segment)
    while index < length:
        char = segment[index]
        if quote is not None:
            if char == quote:
                quote = None
            else:
                buffer.append(char)
            index += 1
            continue
        if char in "'\"":
            quote = char
            index += 1
            continue
        if char == "\\" and index + 1 < length:
            buffer.append(segment[index + 1])
            index += 2
            continue
        if char in " \t":
            if buffer:
                words.append("".join(buffer))
                buffer = []
            index += 1
            continue
        buffer.append(char)
        index += 1
    if buffer:
        words.append("".join(buffer))
    return words


def _skip_prefix(words: list[str]) -> int:
    """Index of the real command word, stepping over prefixes and assignments."""
    index = 0
    while index < len(words):
        word = words[index]
        if _ASSIGNMENT.match(word):
            index += 1
            continue
        if word in _COMMAND_PREFIXES:
            index += 1
            # A prefix's own bare operand (``timeout 60``, ``nice 10``).
            if index < len(words) and _PREFIX_OPERAND.match(words[index]):
                index += 1
            continue
        if word.startswith("-"):
            index += 1
            continue
        break
    return index


def _operands(words: list[str]) -> list[str]:
    """The positional words of a command, flags and their ``--flag=value`` forms dropped."""
    out: list[str] = []
    for word in words:
        if word.startswith("-"):
            continue
        out.append(word)
    return out


def _redirect_targets(segment: str) -> list[str]:
    """The paths this segment redirects INTO (``>``, ``>>``, ``&>``).

    ``2>``/``2>&1``/``<`` are not creation targets: the first is a diagnostic
    channel and the second a read. ``2>/dev/null`` is a probe and the segment is
    dropped whole before this runs.
    """
    targets: list[str] = []
    quote: str | None = None
    index = 0
    length = len(segment)
    while index < length:
        char = segment[index]
        if quote is not None:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
            index += 1
            continue
        if char == ">":
            # ``>&`` is a descriptor duplication (``2>&1``, ``>&2``), not a file;
            # ``2>`` / ``1>`` / ``&>`` DO create their target, so those count.
            if index + 1 < length and segment[index + 1] == "&":
                index += 2
                continue
            at = index + 1
            while at < length and segment[at] in ">|":
                at += 1
            rest = segment[at:].lstrip()
            word = rest.split()[0] if rest.split() else ""
            word = word.strip("\"'")
            if word and not word.startswith("&") and word not in ("/dev/null", "/dev/stdout"):
                targets.append(word)
            index = at
            continue
        index += 1
    return targets


def _created_by_command(words: list[str]) -> list[str]:
    """Paths the command word CREATES.

    ``mkdir``/``touch``/``tee`` create every operand; ``cp``/``mv`` only the
    destination; ``git clone``/``uv venv`` are the two the shipped pad scanner
    (``tools/builtin._bash_created_entries``) does not model, and also the two
    with which a model would build a project from nothing.
    """
    start = _skip_prefix(words)
    if start >= len(words):
        return []
    name = words[start]
    rest = _operands(words[start + 1 :])
    if name in ("git", "uv"):
        # ``git clone <url> <dir>`` / ``uv venv <dir>``: the created path is a
        # SUBcommand operand, so the two-word form is read explicitly. A flag's
        # VALUE is not a positional here (``git clone --depth 1 u d`` is three
        # operands to a naive read), so the destination is taken as the LAST
        # operand, which is also how both commands spell it.
        if len(words) <= start + 1:
            return []
        sub = words[start + 1]
        args = _operands(words[start + 2 :])
        if name == "git" and sub == _GIT_CLONE:
            # ``git clone url`` derives the directory name from the url; only an
            # explicit second operand is a path the model wrote down.
            paths = [word for word in args if "://" not in word and "@" not in word]
            return paths[-1:] if paths else []
        if name == "uv" and sub == _UV_VENV:
            # ``uv venv`` with no operand creates ``.venv`` under the cwd, which
            # the cwd's own root already explains; an explicit operand is what
            # to judge.
            return args[-1:] if args else []
        return []
    if name in ("mkdir", "touch", "tee"):
        return rest
    if name in ("cp", "mv"):
        # Only the DESTINATION is created; the sources are read.
        return rest[-1:]
    return []


def shell_asserted_paths(command: str, cwd: Path | None, home: Path) -> list[str]:
    """The absolute paths ``command`` ASSERTS by creating, probes removed.

    Two passes: split into segments, drop the ones that swallow failure (the
    single largest false-positive generator in the baseline — 41% of its
    narrowest flags were a guessed path fed to a ``cd`` designed to fail
    silently), then read creating positions out of what remains. ``cd`` is
    tracked so ``cd ~/x && mkdir src`` resolves ``src`` under ``~/x`` rather than
    under the session's cwd, which is the one case where the model did say where
    it was going.
    """
    segments = _split_segments(_strip_heredocs(command))
    # Probe pass: a segment whose failure is swallowed is not an assertion. The
    # baseline priced exactly this at 41% of its narrowest flags — a guessed path
    # fed to a ``cd`` designed to fail silently — so it is dropped per SEGMENT,
    # never per command string: ``cd ~/x 2>/dev/null && mkdir -p ~/x/src`` still
    # asserts the creation that follows the probe.
    probe = [False] * len(segments)
    for position, (segment, following) in enumerate(segments):
        if _SWALLOW_REDIRECT.search(segment):
            probe[position] = True
        if following == "||" and position + 1 < len(segments):
            arm = " ".join(_tokens(segments[position + 1][0].strip()))
            if _SWALLOW_ARM.match(arm):
                # ``<segment> || true``: the no-op exists to swallow the failure
                # of the segment before it, which is therefore a probe.
                probe[position] = True
    out: list[str] = []
    base = cwd
    for position, (segment, _following) in enumerate(segments):
        text = segment.strip()
        if not text or probe[position]:
            continue
        words = _tokens(text)
        if not words:
            continue
        start = _skip_prefix(words)
        if start < len(words) and words[start] == "cd":
            operands = _operands(words[start + 1 :])
            target = _expand(operands[0], base, home) if operands else home
            if target is not None:
                base = target
            continue
        for raw in [*_redirect_targets(text), *_created_by_command(words)]:
            resolved = _expand(raw, base, home)
            if resolved is not None:
                out.append(str(resolved))
    return out


def asserted_paths(
    tool: str,
    arguments: Mapping[str, Any],
    *,
    cwd: Path | None,
    home: Path,
) -> list[tuple[str, bool]]:
    """The ``(absolute path, is_write)`` pairs a call asserts.

    ``is_write`` is what escalation keys on: a ``write``/``edit`` argument and a
    creating command are both writes; a ``screenshot`` path is not.

    Returns ``[]`` for every other tool. That is a deliberate MISS, stated rather
    than hidden: ``eval`` runs arbitrary Python and no path can be recovered from
    it; ``console`` is human-typed and MCP tools run server-side — the design
    note (``workspace-boundary-observation-design.md`` §7) names the same three
    blind spots.
    """
    out: list[tuple[str, bool]] = []
    if tool in _FILE_TARGET_TOOLS:
        args, is_write = _FILE_TARGET_TOOLS[tool]
        for arg in args:
            resolved = _expand(str(arguments.get(arg) or ""), cwd, home)
            if resolved is not None:
                out.append((str(resolved), is_write))
        return out
    if tool == "browser":
        if str(arguments.get("action") or "") in _BROWSER_TARGET_ACTIONS:
            resolved = _expand(str(arguments.get("path") or ""), cwd, home)
            if resolved is not None:
                out.append((str(resolved), True))
        return out
    if tool == "bash":
        command = arguments.get("command")
        if isinstance(command, str) and command:
            out.extend((path, True) for path in shell_asserted_paths(command, cwd, home))
    return out


# ---------------------------------------------------------------------------
# Skip roots
# ---------------------------------------------------------------------------


def _skip_roots(
    cwd: str | os.PathLike[str] | None, home: str | os.PathLike[str] | None = None
) -> tuple[tuple[Path, str], ...]:
    """The skip list for ``cwd``/``home``, whatever shape the caller holds.

    A thin boundary in front of :func:`_derive_skip_roots`: it accepts the
    ``str`` a ``ToolContext`` carries as readily as a ``Path``, so no caller has
    to convert before calling, and it cannot raise. If the list cannot be
    derived at all the answer is EMPTY — no root is skipped — because the only
    effect of a skip root is to suppress a notice: losing the list over-reports
    an observation, which is a row an operator can dismiss, while raising here
    loses the OBSERVATION. (That failure was real: ``cwd.resolve()`` on a string
    raised out of ``observe_turn``, which the loop swallowed at DEBUG, so the
    shipped guard recorded nothing at all — QA round 1, case 2.)
    """
    try:
        return _derive_skip_roots(_as_path(cwd), _as_path(home))
    except Exception:  # noqa: BLE001 - an observer must never break a turn
        logger.debug("could not derive the divergence skip roots", exc_info=True)
        return ()


def _derive_skip_roots(cwd: Path | None, home: Path | None = None) -> tuple[tuple[Path, str], ...]:
    """Roots a local decision could produce without any foreign context.

    Each is a place the harness itself sends the model (its config store, its
    logs, the session's scratchpad, the package caches an ordinary build reads)
    or a directory the OS owns. ``~/workspace``, ``~/minervaai`` and ``~/oss``
    are deliberately NOT here: those are exactly the trees this guard exists to
    notice.

    A root is skipped when the asserting path is the root itself or under it —
    the same containment rule the baseline used, resolved so ``/tmp``'s symlink
    to ``/private/tmp`` cannot slip past a string prefix.
    """
    # The session module's own rule, not a copy: a change to what the OS may
    # purge must reach this guard too, and both halves of the predicate
    # (``volatile_roots`` here, containment below) come from one place.
    from local_operator.session.spec import volatile_roots

    roots: list[tuple[Path, str]] = []

    def add(path: Path | None, why: str) -> None:
        if path is None:
            return
        try:
            roots.append((path.resolve(), why))
        except (OSError, RuntimeError):  # pragma: no cover - nothing can stat it
            roots.append((path, why))

    home_root = home or Path.home()
    if cwd is not None:
        # The session's OWN tree, unless the session runs AT (or above) home.
        # That exception is load-bearing, not a convenience: 169 of the 170
        # desktop sessions that persist a cwd on this machine run at ``$HOME``,
        # so skipping "the cwd's root" there would skip every project on the
        # machine — and ``~/squeeze`` is exactly one of them. The whole point of
        # this guard is the case where the session's cwd explains nothing.
        resolved_cwd = cwd
        try:
            resolved_cwd = cwd.resolve()
        except (OSError, RuntimeError):  # pragma: no cover - nothing can stat it
            resolved_cwd = cwd
        try:
            resolved_home = home_root.resolve()
        except (OSError, RuntimeError):  # pragma: no cover
            resolved_home = home_root
        at_or_above_home = (
            resolved_cwd == resolved_home
            or resolved_cwd in resolved_home.parents
            or resolved_home in resolved_cwd.parents
        )
        if not at_or_above_home:
            add(resolved_cwd, "session cwd")
    add(config_dir(), "config dir")
    try:
        from local_operator.paths import agent_home_dir, log_dir

        add(agent_home_dir(), "agent home")
        add(log_dir(), "log dir")
    except Exception:  # pragma: no cover - a paths module without these
        pass
    scratch = os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")
    if scratch:
        add(Path(scratch), "scratchpad")
    add(Path("/dev"), "system device tree")
    for root in volatile_roots():
        add(root, "volatile temp root")
    home = home or Path.home()
    for rel, why in (
        (".cache", "package cache"),
        (".npm", "package cache"),
        (".pnpm-store", "package cache"),
        (".cargo", "package cache"),
        (".local/share/uv", "package cache"),
        ("go/pkg/mod", "package cache"),
        ("Library/Caches", "package cache"),
        ("Library/Application Support", "application support"),
    ):
        add(home / rel, why)
    return tuple(roots)


def _under(path: str, root: Path) -> bool:
    try:
        candidate = Path(path)
        return candidate == root or root in candidate.parents
    except (OSError, RuntimeError):  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


def _message_text(message: Any) -> str:
    """The plain text of one stored message, whatever shape the caller holds.

    Two callers feed this: the loop hands ``harness.types.Message`` objects (the
    live path) and the replay driver hands the transcript's JSON dicts. Both are
    the STORED form — see the masking rule in this module's docstring.
    """
    if isinstance(message, str):
        return message
    content = (
        message.get("content")
        if isinstance(message, Mapping)
        else getattr(message, "content", None)
    )
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or ():
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        else:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _message_role(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def _custom_type(message: Any) -> str:
    """The ``custom_type`` of a stored ``CustomMessage``, or ``""``.

    Two shapes reach this module: ``harness.types.CustomMessage`` objects (the
    live loop) and the transcript's JSON dicts (the replay driver), where the
    type sits under ``payload``.
    """
    if isinstance(message, Mapping):
        payload = message.get("payload")
        if isinstance(payload, Mapping) and payload.get("custom_type"):
            return str(payload["custom_type"])
        return str(message.get("custom_type") or "")
    return str(getattr(message, "custom_type", "") or "")


def _custom_details(message: Any) -> Mapping[str, Any]:
    if isinstance(message, Mapping):
        payload = message.get("payload")
        details = payload.get("details") if isinstance(payload, Mapping) else None
        if not isinstance(details, Mapping):
            details = message.get("details")
        return details if isinstance(details, Mapping) else {}
    details = getattr(message, "details", None)
    return details if isinstance(details, Mapping) else {}


def _custom_text(message: Any) -> str:
    """The text a delivered ``CustomMessage`` carries to the model.

    Two shapes: ``details["text"]`` is what the sender (or the session)
    formatted and what ``harness/render.py`` injects as a user turn, and the
    gate-expiry row is COMPOSED from its structured payload rather than stored
    (``gate_timeout_subject``, the same formatter the renderer uses). Reading
    only ``text`` was what let that carrier contribute nothing while an event
    still fired claiming ``unseen: ()`` — code review round 2, MAJOR-1.

    A FOLDED TYPE MUST CONTRIBUTE TEXT HERE. Folding a carrier is the claim that
    the judge can read what it told the model, and that claim is not automatic:
    the gate arm above exists because a structured payload reaches the model
    while a ``text``-only reader returned ``''`` and the row still reported the
    carrier as checked (round-2 MAJOR-1). Any member of
    :data:`MODEL_VISIBLE_INJECTED_TYPES` whose shape is not a stored ``text``
    obliges an arm here, and ``test_every_folded_carrier_contributes_text``
    drives every member through the real renderer and fails when the extraction
    comes back empty — including for a type the test does not name.

    ``details["summary"]`` is deliberately NOT read: it belongs to the
    compaction marker, whose author is the MODEL, and model-authored text may
    not be an instruction source (see :data:`MODEL_AUTHORED_CARRIERS`).
    """
    details = _custom_details(message)
    if _custom_type(message) == GATE_TIMEOUT_CUSTOM_TYPE:
        return gate_timeout_subject(details)
    text = details.get("text")
    return text if isinstance(text, str) and text else ""


def _has_image(message: Any) -> bool:
    """Does this message carry an image the text scan cannot read?"""
    content = (
        message.get("content")
        if isinstance(message, Mapping)
        else getattr(message, "content", None)
    )
    if isinstance(content, str) or not content:
        return False
    for block in content:
        if isinstance(block, Mapping):
            if str(block.get("type") or "") in ("image", "image_url"):
                return True
        elif str(getattr(block, "type", "") or "") in ("image", "image_url"):
            return True
    return False


#: Carriers in :data:`MODEL_VISIBLE_INJECTED_TYPES` whose bytes the MODEL wrote.
#:
#: THE RULE, in one sentence: text the model can author is NEVER an instruction
#: source, because it cannot authorise the path the model itself invented.
#:
#: Every member reaches the model as a user turn, and every member carries the
#: model's own words inside that turn — so folding one would let a model that
#: invented a project explain its next assertion with its own explanation and
#: silence the notice: model text authorising the path the model invented is
#: exactly the laundering this module's rule forbids. Each stays a carrier (the
#: model DID receive it); it is simply not evidence that anybody else named the
#: root:
#:
#: * ``compaction_summary`` — a model's own prose (``compaction/marker.py``),
#:   standing in for the operator's earlier turns after a cut. The session flags
#:   ``unseen=compacted_history`` instead, so the row says the turns could not
#:   be checked rather than going quiet.
#: * ``wake_prompt`` — ``details["text"]`` ends with the scheduling model's own
#:   ``WakeParams.message``, verbatim (``harness/wake.py``).
#: * ``todo_reminder`` — lists the model's own ``TodoParams.items`` verbatim
#:   (:func:`local_operator.session.session._todo_reminder_text`). Left out, a
#:   session that plans in todos would disable the guard for every root in its
#:   own plan: turn N writes the todo, the reminder lands at the yield boundary,
#:   and turn N+1's write reads as already explained.
#: * ``project_reminder`` — quotes the model's own project names and progress
#:   lines (:func:`local_operator.session.session._project_reminder_text`).
#: * ``monitor_prompt`` — its body is real tool output, but the ``name`` and
#:   ``description`` opening it are the model's choice; the weakest member of the
#:   set and still the same authoring channel.
#:
#: The cost of excluding them is bounded by construction: when the OPERATOR's
#: words named the root, ``user_messages`` still carries that turn, so what an
#: exclusion can add is at most one redundant notice on a root the model talked
#: itself into. Folding them instead buys a guard that goes silent in the
#: incident's own shape — a visible extra row is the cheaper error, so the rule
#: is applied by author, not by how operator-like the prose reads.
MODEL_AUTHORED_CARRIERS = frozenset(
    {
        "compaction_summary",
        "wake_prompt",
        "todo_reminder",
        "project_reminder",
        "monitor_prompt",
    }
)


def _corpus_entry(message: Any) -> tuple[str, str] | None:
    """``(text, source)`` for one stored message that is an INSTRUCTION source.

    The three arms are the three shapes an instruction reaches the model in:

    * an ordinary user turn;
    * a tool result — the model's own earlier output;
    * a ``CustomMessage`` the renderer delivers as a user turn (a wake prompt, a
      hub/peer delivery, a ``session_state`` goal delta, a relayed ask …). These
      carry no ``role`` at all, so folding only ``role == "user"`` missed every
      one of them and then claimed the root appeared in no user message. The
      membership test is the renderer's own list, imported rather than retyped.

    ``None`` for everything else: assistant text is deliberately not a source
    (a model that names a root in its own prose has already started to assert
    it, and letting that explain the next call would launder the failure this
    guard exists to see) — and for the same reason
    :data:`MODEL_AUTHORED_CARRIERS`, which are carriers the model received but
    did not receive as an instruction.
    """
    custom = _custom_type(message)
    if custom and custom in MODEL_VISIBLE_INJECTED_TYPES:
        if custom in MODEL_AUTHORED_CARRIERS:
            return None
        return _custom_text(message), "user_messages"
    role = _message_role(message)
    if role == "user":
        return _message_text(message), "user_messages"
    if role == "tool":
        return _message_text(message), "earlier_tool_output"
    return None


class DivergenceJudge:
    """Per-session observation state; the loop calls :meth:`observe_turn`.

    Session-scoped on purpose: "fire once per novel root" is a fact about a
    session, not about a turn, and a run-scoped instance would re-fire on the
    next turn for the same root.

    :param session_id: the asserting session — for a child that is the CHILD's id.
    :param parent_session_id: ``origin.json``'s parent, or ``""``.
    :param home: ``$HOME`` as the session sees it; parameterised so a test and
        the replay driver can hold a fixture home rather than the operator's.
    :param on_notice: optional host callback receiving
        :meth:`DivergenceEvent.as_notice_payload`. It paints the transcript row;
        it is called at most ``MAX_ROWS_PER_ROOT`` times per root, and for at
        most ``MAX_NOTICED_ROOTS`` distinct roots in the session (then once more,
        with a ``collapsed`` payload, to say the rest are audit-only).
    :param audit_path: override for the JSONL destination (tests).
    """

    def __init__(
        self,
        *,
        session_id: str,
        parent_session_id: str = "",
        home: Path | None = None,
        on_notice: Callable[[dict[str, Any]], None] | None = None,
        audit_path: Path | None = None,
        cwd: Path | None = None,
        skip_roots: tuple[tuple[Path, str], ...] | None = None,
    ) -> None:
        self.session_id = session_id
        self.parent_session_id = parent_session_id
        #: Resolved once, here. ``Path.home()`` is whatever ``$HOME`` holds — on
        #: a host whose home is a symlink that is not the canonical form, and
        #: every asserting path arrives resolved (``_expand``), so comparing the
        #: two as given made EVERY path under home fail the prefix test and
        #: collapse onto one root (code review round 1: a fixture home under
        #: ``/var`` read ``~/squeeze/a`` as the root ``/private/var``).
        self.home = _resolved(home or Path.home())
        self.on_notice = on_notice
        self._audit_path = audit_path
        self._cwd = _as_path(cwd)
        #: ``None`` means "derive the shipped list" (``_skip_roots``), which is
        #: what production does. A caller holding a FIXTURE root set overrides
        #: it: a test home lives under the OS temp root, which is itself a skip
        #: root, and using the real list there would prove nothing about the
        #: predicate — every assertion would be skipped for the wrong reason.
        self._skip_roots = skip_roots
        #: Derived skip lists, memoised per ``(cwd, home)``. ``_skip_roots``
        #: imports the session spec, reads the environment and stats a dozen
        #: paths; it is a property of where the session runs, not of the call, so
        #: re-deriving it per tool call was pure overhead (0.6-22 ms measured).
        self._skip_cache: dict[tuple[str, str], tuple[tuple[Path, str], ...]] = {}
        self._roots: dict[str, _RootState] = {}
        #: Roots that have painted a notice, and whether the overflow row has
        #: been sent (``MAX_NOTICED_ROOTS``). The audit keeps every root; the
        #: TRANSCRIPT is what gets a cap.
        self._noticed: set[str] = set()
        self._collapsed = False
        # Corpus state: one lowercased chunk per source message, appended and
        # never rewritten. There is deliberately NO per-root search cursor: the
        # live path asks about a root at most once per corpus (an announced root
        # short-circuits before the search, and an explained one is memoised in
        # ``_explained``), so a cursor was unreachable state that read as a
        # load-bearing optimisation (QA round 2, note 2).
        self._messages_ref: object | None = None
        self._consumed = 0
        self._system_text = ""
        self._chunks: list[str] = []
        self._sources: set[str] = set()
        self._unseen: set[str] = set()
        self._explained: set[str] = set()
        self._events: list[DivergenceEvent] = []
        #: Turn numbering that survives compaction (see :meth:`_monotonic_turn`).
        self._turn_peak = 0
        self._turn_shift = 0
        #: The first unexpected failure is logged once at WARNING: a silent
        #: observer is exactly how the shipped guard recorded nothing at all for
        #: a whole release (QA round 1, case 2), and a swallowed failure that
        #: nobody can see is indistinguishable from "nothing happened".
        self._failure_logged = False

    # -- context sources ----------------------------------------------------

    def _sync_corpus(self, messages: Sequence[Any], system_blocks: Sequence[str]) -> None:
        """Fold new history into the corpus, and only new history.

        The corpus is exactly the operator's list: system prompt, custom
        instructions (both arrive in ``system_blocks`` — the harness appends the
        custom-instruction block there), user messages INCLUDING every
        ``CustomMessage`` the renderer delivers as a user turn, and the session's
        own earlier tool output. **Assistant text is deliberately NOT a source**:
        a guard that let the model's own prior assertions explain the next one
        would launder precisely the failure it exists to see.

        WHY CHUNKS. The corpus is held as one lowercased string per source
        message, appended and never rewritten. A single concatenated string had
        to be rebuilt whenever one tool result arrived — 531 ms on a 6.3 MB
        history, on the loop's post-result path — and re-searched in full every
        turn thereafter. With chunks, a new tool result adds one small chunk and
        nothing rewrites what came before; the string work is the same either
        way (one lowercase + one containment test per byte), what it stops doing
        is repeating it. What bounds the burst is the CALLER, not a cursor here:
        ``_observe_turn`` never re-searches a root this session has already
        announced (``root not in self._roots``), and an explained root is
        memoised in ``_explained``, so each root is searched at most once per
        corpus.
        """
        system = "\n".join(block for block in system_blocks if block)
        if system != self._system_text:
            # A changed system prompt invalidates every chunk and every cursor:
            # the source of truth was replaced, not extended. Rare (an epoch
            # change), and it is the ONE case that re-reads the whole history.
            self._system_text = system
            self._reset_corpus()
        if messages is not self._messages_ref:
            # A different list object is a different history (a resume, a test
            # fixture, a replay driver, a compaction): rebuild from its start.
            self._messages_ref = messages
            self._consumed = 0
            self._reset_corpus()
        for message in messages[self._consumed :]:
            entry = _corpus_entry(message)
            if entry is not None:
                text, source = entry
                if text:
                    # ONE lowercase per chunk: it is both the searchable form and
                    # the mask test. Doing it twice doubled the first call's
                    # string work on a 6.3 MB history for no gain.
                    lowered = text.lower()
                    self._chunks.append(lowered)
                    self._sources.add(source)
                    if _MASK_MARKER in lowered:
                        self._unseen.add("masked_text")
            if _has_image(message):
                self._unseen.add("image_attachments")
            if _custom_type(message) == "compaction_summary":
                # Compaction replaced the messages this session started with by a
                # summary: a root named before the cut may be absent from the
                # list this judge is handed, so absence over "earlier turns" is
                # not established.
                self._unseen.add("compacted_history")
        self._consumed = len(messages)
        if self._system_text:
            self._sources.add("system_prompt")
            # The operator's first two lists are ONE carrier in practice: the
            # harness folds the custom-instruction block into the system blocks,
            # so a guard that read them read both. Claiming them separately
            # without that sentence would be two claims for one read.
            self._sources.add("custom_instructions")

    def _reset_corpus(self) -> None:
        self._chunks = [self._system_text.lower()] if self._system_text else []
        # ``sources`` describes THIS corpus. Leaving it populated across a
        # replacement made a row claim ``earlier_tool_output`` was checked while
        # the replacement held no tool text at all — a claim the record cannot
        # support, which is the defect this whole field exists to avoid (code
        # review round 2, MINOR-2). ``_unseen`` survives on purpose: it is a
        # fact about the session (text was masked, an image arrived, a cut
        # happened), not about the current chunk list.
        self._sources.clear()

    def _is_explained(self, root: str) -> bool:
        """Does the corpus name ``root``?

        Root granularity: the basename token, or the whole root for an absolute
        path outside home (``/opt/acme`` → ``/opt/acme``).

        Chunks are append-only, so a hit is memoised in ``_explained`` and a
        root this session already announced never reaches here at all
        (``_observe_turn`` short-circuits on ``root in self._roots``). Each root
        therefore costs at most one full scan per corpus — the incident names
        its root 117 times in 134 turns, and 116 of those scans were the whole
        per-call cost the round-2 measurement attributed to a cursor that could
        never be reached.
        """
        key = root_key(root)
        if not key:
            return True
        if key in self._explained:
            return True
        lowered_root = root.lower()
        for chunk in self._chunks:
            if key in chunk or lowered_root in chunk:
                self._explained.add(key)
                return True
        return False

    # -- skip rules ---------------------------------------------------------

    def _skipped(self, path: str, skip: tuple[tuple[Path, str], ...]) -> bool:
        return any(_under(path, root) for root, _why in skip)

    # -- observation --------------------------------------------------------

    def observe_turn(
        self,
        *,
        turn_index: int,
        tool: str,
        arguments: Mapping[str, Any],
        call_id: str = "",
        cwd: str | os.PathLike[str] | None = None,
        messages: Sequence[Any] = (),
        system_blocks: Sequence[str] = (),
    ) -> list[DivergenceEvent]:
        """Judge one tool call, append at most one audit row, return new events.

        Never raises, and that is now true of every line rather than of all but
        one: the skip-root derivation that used to sit outside any ``try`` (and
        raised ``AttributeError`` on the ``str`` cwd a real ``ToolContext``
        carries) is inside it, and :func:`_skip_roots` itself cannot raise.
        An observer that can break a turn is a gate with a disguise on, and the
        operator's constraint is that a flagged call runs exactly as it would
        have.

        ``cwd`` accepts the ``str`` that ``ToolContext.cwd`` is declared as well
        as a ``Path``: the loop passes it through untouched, so normalising at
        this boundary is what makes the difference between the two impossible to
        ship as a defect.
        """
        try:
            return self._observe_turn(
                turn_index=turn_index,
                tool=tool,
                arguments=arguments,
                call_id=call_id,
                cwd=cwd,
                messages=messages,
                system_blocks=system_blocks,
            )
        except Exception:  # noqa: BLE001 - an observer must never break a turn
            if not self._failure_logged:
                self._failure_logged = True
                logger.warning("divergence observer failed", exc_info=True)
            else:
                logger.debug("divergence observer failed again", exc_info=True)
            return []

    def _observe_turn(
        self,
        *,
        turn_index: int,
        tool: str,
        arguments: Mapping[str, Any],
        call_id: str,
        cwd: str | os.PathLike[str] | None,
        messages: Sequence[Any],
        system_blocks: Sequence[str],
    ) -> list[DivergenceEvent]:
        if not arguments or not isinstance(arguments, Mapping):
            return []
        workdir = _as_path(cwd) or self._cwd
        candidates = asserted_paths(tool, arguments, cwd=workdir, home=self.home)
        if not candidates:
            return []
        self._sync_corpus(messages, system_blocks)
        turn = self._monotonic_turn(turn_index)
        skip = self._skip_for(workdir)
        fresh: list[DivergenceEvent] = []
        for path, is_write in candidates:
            root = project_root(Path(path), self.home)
            if self._skipped(path, skip):
                continue
            if root not in self._roots and self._is_explained(root):
                # ``root not in self._roots``: a root this session already
                # announced stays announced. Its escalation decision belongs to
                # ``_register`` and does not depend on the corpus, so scanning
                # 6.3 MB again for the 117th mention would change nothing — and
                # that scan was most of the per-call cost.
                continue
            event = self._register(
                root=root,
                path=path,
                tool=tool,
                call_id=call_id,
                turn_index=turn,
                is_write=is_write,
            )
            if event is not None:
                fresh.append(event)
        return fresh

    def _monotonic_turn(self, turn_index: int) -> int:
        """``turn_index``, shifted so the sequence never goes backwards.

        The loop derives its index by counting assistant messages in the live
        history (``harness/loop.py``), and compaction REPLACES that list: the
        count can drop, and an escalated mention could then compare equal to
        ``first_turn`` and silently lose its escalation row. The safe direction,
        but a turned corner is still a lost fact. A shift carried on the judge
        makes the sequence monotonic across any number of cuts.
        """
        effective = turn_index + self._turn_shift
        if effective <= self._turn_peak:
            self._turn_shift += self._turn_peak - effective + 1
            effective = turn_index + self._turn_shift
        self._turn_peak = effective
        return effective

    def _skip_for(self, workdir: Path | None) -> tuple[tuple[Path, str], ...]:
        """The skip list for this session, derived once per ``(cwd, home)``."""
        if self._skip_roots is not None:
            return self._skip_roots
        key = (str(workdir) if workdir is not None else "", str(self.home))
        cached = self._skip_cache.get(key)
        if cached is None:
            cached = _skip_roots(workdir, self.home)
            self._skip_cache[key] = cached
        return cached

    def _register(
        self,
        *,
        root: str,
        path: str,
        tool: str,
        call_id: str,
        turn_index: int,
        is_write: bool,
    ) -> DivergenceEvent | None:
        """Apply the once-per-root / escalate rules and record the row."""
        state = self._roots.get(root)
        if state is None:
            state = _RootState(first_turn=turn_index)
            self._roots[root] = state
            # A WRITE on first sighting is already the escalated fact: the model
            # is not just naming a foreign project, it is creating in it. One
            # row carries both, so a single call cannot spend the whole budget.
            escalated = is_write
            reason = "write" if is_write else "first"
        else:
            state.paths.add(path)
            if state.escalated or state.rows >= MAX_ROWS_PER_ROOT:
                # Already said the strongest thing this module may say, or the
                # budget is spent. The 117-mention case ends here, on purpose.
                return None
            if is_write:
                escalated, reason = True, "write"
            elif turn_index != state.first_turn:
                escalated, reason = True, "repeat"
            else:
                # Same turn, no write: a second mention in one turn adds nothing.
                return None
        state.paths.add(path)
        state.rows += 1
        if escalated:
            state.escalated = True
        event = DivergenceEvent(
            session_id=self.session_id,
            parent_session_id=self.parent_session_id,
            turn_index=turn_index,
            root=root,
            path=path,
            tool=tool,
            call_id=call_id,
            escalated=escalated,
            reason=reason,
            # What this judge ACTUALLY read, in a stable order. A fixed tuple
            # claimed all four sources on every row whether or not the caller
            # gave it a history or a system prompt — a claim the record cannot
            # support is the same defect as asserting absence over a carrier it
            # never saw.
            sources=tuple(name for name in _SOURCE_ORDER if name in self._sources),
            unseen=tuple(name for name in _UNSEEN_ORDER if name in self._unseen),
            first_sighting=state.rows == 1,
        )
        try:
            self._append(event.as_record())
        except DivergenceAuditError:
            # "A failed write must not be reported as success": the event still
            # surfaces (the operator notice is the fallback record), and it
            # carries the failure rather than pretending to be durable.
            event = replace(event, recorded=False)
        self._events.append(event)
        self._notice(event)
        return event

    def _notice(self, event: DivergenceEvent) -> None:
        """Hand the host a notice payload, within the per-session root cap.

        ``MAX_ROWS_PER_ROOT`` bounds one root; this bounds how many ROOTS may
        paint. Past ``MAX_NOTICED_ROOTS`` the transcript gets ONE row saying the
        rest are in the audit log (design round 1, D6: three roots produced six
        notices and 20 amber rows). Nothing is dropped from the record.
        """
        if self.on_notice is None:
            return
        payload: dict[str, Any] | None = None
        if event.root in self._noticed or len(self._noticed) < MAX_NOTICED_ROOTS:
            self._noticed.add(event.root)
            payload = event.as_notice_payload()
        elif not self._collapsed:
            self._collapsed = True
            payload = {
                "collapsed": True,
                "roots": len(self._roots),
                "root": event.root,
                "path": event.path,
                "tool": event.tool,
                "recorded": event.recorded,
                "unseen": list(event.unseen),
            }
        if payload is None:
            return
        try:
            self.on_notice(payload)
        except Exception:  # noqa: BLE001 - a notice must never break a turn
            pass

    # -- the audit writer ---------------------------------------------------

    def audit_path(self) -> Path:
        return self._audit_path or (config_dir() / AUDIT_DIRNAME / AUDIT_FILENAME)

    def _append(self, record: Mapping[str, Any]) -> None:
        """Append one row and fsync it. Raises :class:`DivergenceAuditError`.

        Same discipline as ``tools/public_copy._record_read`` and for the same
        reason: this row is the durable half of the observation, and a record
        that only reaches a page cache is one a crash can erase. ``O_APPEND``
        keeps concurrent writers' rows whole without a lock (a short line is a
        single atomic write on the local filesystem); ``O_NOFOLLOW`` + a 0700
        directory refuse a planted symlink.
        """
        path = self.audit_path()
        audit_dir = path.parent
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        try:
            audit_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            if os.name == "nt":  # pragma: no cover - no O_NOFOLLOW on Windows
                if audit_dir.is_symlink() or path.is_symlink():
                    raise OSError(f"{path} is reached through a symlink")
                fd = os.open(path, _AUDIT_OPEN_FLAGS, 0o600)
            else:
                dir_fd = os.open(audit_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fchmod(dir_fd, 0o700)
                    fd = os.open(
                        path.name,
                        _AUDIT_OPEN_FLAGS | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=dir_fd,
                    )
                finally:
                    os.close(dir_fd)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise OSError(f"{path} is not a regular file")
                if os.name != "nt" and info.st_mode & 0o777 != 0o600:
                    os.fchmod(fd, 0o600)
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            raise DivergenceAuditError(f"divergence audit row not written: {exc}") from exc
