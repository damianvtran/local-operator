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
import os
import re
import stat
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from local_operator.paths import config_dir

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
    """
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


def _skip_roots(cwd: Path | None, home: Path | None = None) -> tuple[tuple[Path, str], ...]:
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
        it is called at most ``MAX_ROWS_PER_ROOT`` times per root.
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
        self.home = home or Path.home()
        self.on_notice = on_notice
        self._audit_path = audit_path
        self._cwd = cwd
        #: ``None`` means "derive the shipped list" (``_skip_roots``), which is
        #: what production does. A caller holding a FIXTURE root set overrides
        #: it: a test home lives under the OS temp root, which is itself a skip
        #: root, and using the real list there would prove nothing about the
        #: predicate — every assertion would be skipped for the wrong reason.
        self._skip_roots = skip_roots
        self._roots: dict[str, _RootState] = {}
        # Explained-set cache. ``_corpus`` is built LAZILY — on the first
        # candidate that survives the skip rules — and then maintained
        # incrementally, because concatenating a long history on every tool call
        # is the cost that would make this feature unacceptable on a busy turn.
        self._messages_ref: object | None = None
        self._consumed = 0
        self._users: list[str] = []
        self._tool_output: list[str] = []
        self._system_text = ""
        self._corpus: str | None = None
        self._explained: set[str] = set()
        self._events: list[DivergenceEvent] = []

    # -- context sources ----------------------------------------------------

    def _sync_corpus(self, messages: Sequence[Any], system_blocks: Sequence[str]) -> None:
        """Fold new history into the explained-set, and only new history.

        The explained-set is exactly the operator's list: system prompt, custom
        instructions (both arrive in ``system_blocks`` — the harness appends the
        custom-instruction block there), user messages, and the session's own
        earlier tool output. **Assistant text is deliberately NOT a source**: a
        guard that let the model's own prior assertions explain the next one
        would launder precisely the failure it exists to see.
        """
        system = "\n".join(block for block in system_blocks if block)
        if system != self._system_text:
            self._system_text = system
            self._corpus = None
        if messages is not self._messages_ref:
            # A different list object is a different history (a resume, a test
            # fixture, a replay driver): rebuild from its start.
            self._messages_ref = messages
            self._consumed = 0
            self._users = []
            self._tool_output = []
            self._corpus = None
        fresh = False
        for message in messages[self._consumed :]:
            role = _message_role(message)
            if role == "user":
                self._users.append(_message_text(message))
                fresh = True
            elif role == "tool":
                # Strictly EARLIER output: a tool result is only visible to the
                # guard once it is in the stored history, which is what "the
                # model's own earlier tool output" means.
                self._tool_output.append(_message_text(message))
                fresh = True
        self._consumed = len(messages)
        if fresh:
            self._corpus = None

    def _corpus_lower(self) -> str:
        if self._corpus is None:
            self._corpus = (
                self._system_text
                + "\n"
                + "\n".join(self._users)
                + "\n"
                + "\n".join(self._tool_output)
            ).lower()
        return self._corpus

    def _is_explained(self, root: str) -> bool:
        key = root_key(root)
        if not key:
            return True
        if key in self._explained:
            return True
        corpus = self._corpus_lower()
        # Root granularity: the basename token, or the whole root for an
        # absolute path outside home (``/opt/acme`` → ``/opt/acme``).
        if key in corpus or root.lower() in corpus:
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
        cwd: Path | None = None,
        messages: Sequence[Any] = (),
        system_blocks: Sequence[str] = (),
    ) -> list[DivergenceEvent]:
        """Judge one tool call, append at most one audit row, return new events.

        Never raises: an observer that can break a turn is a gate with a
        disguise on, and the operator's constraint is that a flagged call runs
        exactly as it would have.
        """
        if not arguments or not isinstance(arguments, Mapping):
            return []
        workdir = cwd or self._cwd
        try:
            candidates = asserted_paths(tool, arguments, cwd=workdir, home=self.home)
        except Exception:  # noqa: BLE001 - extraction must never break a turn
            return []
        if not candidates:
            return []
        try:
            self._sync_corpus(messages, system_blocks)
        except Exception:  # noqa: BLE001 - an unreadable history is a MISS
            return []
        skip = self._skip_roots if self._skip_roots is not None else _skip_roots(workdir, self.home)
        fresh: list[DivergenceEvent] = []
        for path, is_write in candidates:
            try:
                root = project_root(Path(path), self.home)
            except (OSError, RuntimeError):  # pragma: no cover
                continue
            if self._skipped(path, skip):
                continue
            if self._is_explained(root):
                continue
            event = self._register(
                root=root,
                path=path,
                tool=tool,
                call_id=call_id,
                turn_index=turn_index,
                is_write=is_write,
            )
            if event is not None:
                fresh.append(event)
        return fresh

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
            sources=(
                "system_prompt",
                "custom_instructions",
                "user_messages",
                "earlier_tool_output",
            ),
        )
        try:
            self._append(event.as_record())
        except DivergenceAuditError:
            # "A failed write must not be reported as success": the event still
            # surfaces (the operator notice is the fallback record), and it
            # carries the failure rather than pretending to be durable.
            event = replace(event, recorded=False)
        self._events.append(event)
        if self.on_notice is not None:
            try:
                self.on_notice(event.as_notice_payload())
            except Exception:  # noqa: BLE001 - a notice must never break a turn
                pass
        return event

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
