"""Deliverable-file candidates and the deterministic pre-filter (memo §2.3 Step 1, §3.1 row 5).

WHY DETERMINISTIC. Most turns have nothing to call out, and the decision is a paid vendor
call. This module is the gate in front of it: pure functions over the logical turn's
messages, ``stat`` calls as the only I/O, target <= 5 ms. A turn with no candidate file (and
no structured data when graphics are on) never reaches a vendor at all.

ADMITTED ON EVIDENCE, NEVER ON RESEMBLANCE (the UI's ``mentioned-files.ts`` rule, ported
here as this feature's one Python home). A path becomes a candidate through exactly three
doors, ranked by how directly the turn says "I made this":

* tier 1 -- a ``write`` tool call's ``path`` (the turn created or replaced this file);
* tier 2 -- a ``bash``/``eval`` command under a recognised output flag (``-o``,
  ``--output``, ``> file``, ``tee``, ``to_csv(...)``/``savefig(...)``/``write_text(...)``);
* tier 3 -- an absolute or ``~/`` path with a known deliverable extension in the FINAL
  ANSWER's prose.

An ``edit`` call is deliberately NOT a door. Editing an existing file is the diff, and every
surface already shows it as a tool row; featuring ``foo.py`` after a code edit is the spam
this feature exists to avoid (memo §5.2 "Code edits"). A ``write`` of a SOURCE-code file is
admitted as a candidate (the decision can tell "write me a script" from "I refactored") but
:attr:`Candidate.deliverable` is False for it, which keeps it out of the no-vendor heuristic.

THEN EVERY CANDIDATE MUST SURVIVE, in order: it ``stat``s as a REGULAR file whose written and
resolved forms are both under the session cwd or ``~``; it is not scratch (scratchpad, /tmp,
``$TMPDIR``, the config dir), not a vendored or generated tree (``.git/``, ``node_modules/``,
``.venv/``, ``__pycache__/``, ``.pytest_cache/``, ``dist/``, ``build/``), not ``*.log``/
``*.lock``/``*.pyc``, not under an operator ``denyPrefixes`` entry; it is not SENSITIVE
(:mod:`~local_operator.supplements.denylist`, applied before anything is listed or sent); and
it is not ALREADY VISIBLE -- its path appears verbatim in the answer inside a code span or a
markdown link, which the UI keeps outside the fold. That last rule applies to tier 3 (a
prose-only mention) and to non-deliverable kinds. A tier 1/2 DELIVERABLE the answer links is
KEPT and flagged :attr:`Candidate.linked_in_answer`: "the turn wrote this and says so" is the
strongest evidence there is, and the answer's link sits inside the (often folded) prose, not
in the callout block -- dropping it made a turn that linked its own receipt show no
Highlights at all.

WHAT A SHELL COMMAND WROTE (:func:`bash_effects`). Tier 2 reads the command with its
working directory tracked: a ``cd ~/x; cp a.jsonl b.jsonl; python3 - <<PY ... open('r.md','w')``
names ``~/x/b.jsonl`` and ``~/x/r.md``, not session-cwd-relative strings (which stat the wrong
file or none). Heredoc bodies are lifted out BEFORE the shell is tokenised -- a python body
holds unbalanced quotes that made ``shlex`` raise and the whole command name nothing -- and are
scanned for python write literals only when the heredoc feeds an interpreter (a ``cat >
f <<EOF`` body is the FILE's text, not code). A ``cd`` whose target is not statically
knowable (``$VAR``, ``cd -``) makes later relative names UNRESOLVABLE and they are dropped,
never guessed against the session cwd. Files a writing call left beside its outputs under a
variable path (``json.dump(d, open(p, 'w'))``) are found by the bounded touched-directory
sweep (:func:`_sweep`), which is why it only runs for calls that show a write signal.

PATHS ARE NEVER ABSOLUTE ON THE WAY OUT. :attr:`Candidate.path` is cwd-relative or ``~/``-
relative so the operator's directory layout does not replicate to peers (memo S-R14);
:attr:`Candidate.absolute` is for the local ``stat`` and the file route only and is never
journaled or sent to a vendor. The decision's option text carries the BASENAME alone.

LEAF: standard library plus the sibling leaves.
"""

import os
import re
import shlex
import stat as stat_module
import tempfile
import time
from dataclasses import dataclass, field, replace
from itertools import islice
from typing import Final, Iterable, Sequence

from local_operator.ansi import sanitize_prompt_line
from local_operator.supplements import denylist
from local_operator.supplements.evidence import Evidence, extract
from local_operator.supplements.trigger import ToolCallItem, TurnItem

#: Candidates offered to the decision (the recommend layer's ``maxCandidates`` scale).
MAX_OFFERED: Final = 12
#: Hard ceiling on candidates considered per turn -- a loop that writes 5,000 files costs the
#: same as one that writes 50.
MAX_CONSIDERED: Final = 64
#: Pre-filter wall target (memo §2.3). A budget the tests measure, not an enforced deadline.
PREFILTER_TARGET_MS: Final = 5.0

TIER_WRITE: Final = 1
TIER_COMMAND: Final = 2
TIER_PROSE: Final = 3

#: Extension -> kind. ``kind`` is the callout's icon class on a surface and the decision's
#: "what is this" word. Anything not listed is ``other``.
_KIND_BY_EXT: Final[dict[str, str]] = {
    **dict.fromkeys((".md", ".markdown", ".rst"), "markdown"),
    **dict.fromkeys((".txt",), "text"),
    **dict.fromkeys((".csv", ".tsv"), "csv"),
    **dict.fromkeys((".json", ".jsonl", ".ndjson"), "json"),
    **dict.fromkeys((".html", ".htm"), "html"),
    **dict.fromkeys((".pdf",), "pdf"),
    **dict.fromkeys((".doc", ".docx", ".odt", ".rtf", ".pages"), "document"),
    **dict.fromkeys((".xls", ".xlsx", ".ods", ".numbers"), "spreadsheet"),
    **dict.fromkeys((".ppt", ".pptx", ".odp", ".key"), "presentation"),
    **dict.fromkeys(
        (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".tiff", ".heic"), "image"
    ),
    **dict.fromkeys((".mp3", ".wav", ".m4a", ".flac", ".ogg"), "audio"),
    **dict.fromkeys((".mp4", ".mov", ".webm", ".mkv"), "video"),
    **dict.fromkeys((".zip", ".tar", ".gz", ".tgz", ".7z"), "archive"),
    **dict.fromkeys(
        (
            ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".c", ".h",
            ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".sh", ".bash", ".zsh",
            ".sql", ".css", ".scss", ".vue", ".svelte", ".toml", ".yaml", ".yml", ".ini",
            ".cfg", ".xml", ".lua", ".r", ".ipynb",
        ),
        "code",
    ),
}  # fmt: skip
#: Checksum manifests (``shasum -a 256 *.csv > SHA256SUMS``) are the receipt of a delivery
#: package. They have no extension, so ``other`` would keep them out of every deliverable
#: rule; plain text is what they are and an icon class every surface already paints.
#: NARROW ON PURPOSE: only manifests -- ``LICENSE``/``Makefile``/``README`` style names are
#: ordinary extension-less files and stay ``other``.
_CHECKSUM_NAMES: Final = frozenset(
    {"sha256sums", "sha512sums", "sha1sums", "md5sums", "checksums", "checksums.txt"}
)
_CHECKSUM_SUFFIXES: Final = (".sha256", ".sha512", ".sha1", ".md5", ".sha256sum", ".md5sum")
#: Kinds that are a PRODUCT for a reader, as opposed to source (``code``) or unknown. The
#: no-vendor heuristic and tier 3 admit only these.
_DELIVERABLE_KINDS: Final = frozenset(
    {
        "markdown", "text", "csv", "json", "html", "pdf", "document", "spreadsheet",
        "presentation", "image", "audio", "video", "archive",
    }
)  # fmt: skip

_EXCLUDED_COMPONENTS: Final = frozenset(
    {
        ".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", "dist",
        "build", ".mypy_cache", ".ruff_cache", ".tox",
    }
)  # fmt: skip
_EXCLUDED_SUFFIXES: Final = (".log", ".lock", ".pyc", ".pyo", ".tmp", ".swp", ".DS_Store")
_SCRATCH_ROOTS: Final[tuple[str, ...]] = (
    "/tmp",
    "/private/tmp",
    "/var/tmp",
    "/private/var/tmp",
)

_OUTPUT_FLAGS: Final = frozenset(
    {"-o", "--output", "--out", "--outfile", "--out-file", "--output-file"}
)
_REDIRECT_TOKENS: Final = frozenset({">", ">>", "1>", "1>>", "&>", "&>>"})
_REDIRECT_ATTACHED: Final = re.compile(r"^(?:1|&)?>{1,2}(?P<path>[^>&\s].*)$")
_EVAL_WRITES: Final = re.compile(
    r"(?:\.to_(?:csv|excel|json|parquet|markdown|html)|\.savefig|\.write_text|\.write_bytes)"
    r"""\(\s*(?:f?r?)(?P<q>['"])(?P<path>[^'"\n]+)(?P=q)"""
    r"|\bopen\(\s*(?:f?r?)(?P<q2>['\"])(?P<path2>[^'\"\n]+)(?P=q2)\s*,\s*"
    r"""['"][wax]b?\+?['"]"""
    # ``Path('out.md').write_text(s)`` -- the literal is the Path's argument, not write_text's.
    r"|\bPath\(\s*(?:f?r?)(?P<q3>['\"])(?P<path3>[^'\"\n]+)(?P=q3)\s*\)"
    r"(?:\.expanduser\(\))?\.write_(?:text|bytes)\("
)
#: Tier-3 prose path: ``/abs/x.pdf`` or ``~/x.csv``. No spaces, so a sentence cannot swallow
#: its neighbour; ``(?<![\w:/.])`` rejects URLs and mid-token matches.
_PROSE_PATH: Final = re.compile(
    r"(?<![\w:/.\-])(?P<path>(?:(?:~|/|\./)?[\w.\-+@%]+/)*[\w.\-+@%]+\.[A-Za-z0-9]{1,8})(?![\w/])"
)
_CODE_SPAN: Final = re.compile(r"`+([^`\n]+)`+")
_MD_LINK: Final = re.compile(r"\[(?P<text>[^\]\n]*)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")

_INTENT_MAX_CHARS: Final = 80


@dataclass(frozen=True)
class Candidate:
    """One deliverable-file candidate. See the module docstring for the privacy rules."""

    path: str
    absolute: str
    name: str
    kind: str
    size_bytes: int
    mtime: float
    tier: int
    tool: str
    #: Position of the first mention in the turn (higher = later); recency tiebreak.
    order: int
    intent: str = ""
    #: The answer's own prose links or code-spans this path. Informational: the callout is
    #: still offered (see the module docstring); journaled additively as
    #: ``SupplementFile.linked_in_answer`` so a surface MAY fold it against its own link.
    linked_in_answer: bool = False

    @property
    def deliverable(self) -> bool:
        return self.kind in _DELIVERABLE_KINDS

    @property
    def why(self) -> str:
        """The row's ``why`` string (memo §2.4 example ``written by write``)."""
        if self.tool == "edit":
            return "updated by edit"
        if self.tier == TIER_PROSE:
            return "named in the answer"
        return f"written by {self.tool}"


@dataclass(frozen=True)
class Prefilter:
    candidates: tuple[Candidate, ...]
    evidence: Evidence
    #: ``"prefilter"`` when nothing survives (the memo's ``skipped`` spelling), else ``None``.
    skipped: str | None
    #: Wall milliseconds, for the DEBUG absorption line and the golden-set report.
    elapsed_ms: float = 0.0
    #: Why each rejected path was rejected (reason -> count), for the absorption line and for
    #: tests that prove each guard fires. Paths themselves are not kept (privacy).
    rejected: dict[str, int] = field(default_factory=dict)
    #: Content is private local promotion input, never part of decision options.
    prepared: tuple[tuple[Candidate, str], ...] = ()


def kind_of(name: str) -> str:
    base = os.path.basename(name).casefold()
    if base in _CHECKSUM_NAMES or base.endswith(_CHECKSUM_SUFFIXES):
        return "text"
    return _KIND_BY_EXT.get(os.path.splitext(name)[1].casefold(), "other")


# -- harvesting --------------------------------------------------------------------------


#: Heredoc operator (``<<EOF``, ``<<-'PY'``). ``(?<!<)``/``(?!<)`` keep a here-STRING (``<<<``)
#: and a shift (``1<<3``) out; the tag must start with a letter or underscore.
_HEREDOC: Final = re.compile(
    r"(?<!<)<<(?!<)(?P<dash>-?)\s*(?P<q>['\"]?)(?P<tag>[A-Za-z_]\w*)(?P=q)"
)
_HEREDOC_MARK: Final = "__HEREDOC_{}__"
_HEREDOC_MARK_RE: Final = re.compile(r"^__HEREDOC_(\d+)__$")
#: A shell word that makes a heredoc body PYTHON source (its stdin is the program).
_PYTHON_WORD: Final = re.compile(r"^(?:python[\d.]*|pypy\d*|ipython\d*)$")
#: The shell's separator and redirect punctuation, as ``shlex`` merges adjacent runs of it.
_PUNCT_PIECES: Final = re.compile(r"&>>|&>|>>|>&|>|<<|<|;;|;|&&|\|\||\||&|\(|\)|\n")
_SEPARATORS: Final = frozenset({";;", ";", "&&", "||", "|", "&", "\n"})
_COPY_COMMANDS: Final = frozenset({"cp", "mv", "install"})
#: A python write whose TARGET is not a string literal (``open(p, 'w')``, ``df.to_csv(path)``,
#: ``Path(x).write_text(..)`` on a variable): the one case no literal names, so the only case
#: the touched-directory sweep exists for. Literal targets are removed (``_EVAL_WRITES``)
#: before this runs, and a redirect/``cp``/``tee`` destination is always named, so none of
#: those arm the sweep -- a plain ``cd d; echo > a.txt`` must not list ``d``.
_PY_VARIABLE_WRITE: Final = re.compile(
    r"""\bopen\(\s*(?!f?r?['"])[^\s,)][^,()\n]{0,512},\s*['"][wax]b?\+?['"]"""
    r"""|\.to_(?:csv|excel|json|parquet|markdown|html)\(\s*(?!f?r?['"])[^\s)]"""
    r"""|\.savefig\(\s*(?!f?r?['"])[^\s)]"""
    r"|\bshutil\.(?:copy\w*|move)\("
)
SHELL_SCAN_MAX_CHARS: Final = 32_768
_LEADING_HOME: Final = re.compile(r"^\$(?:\{HOME\}|HOME)(?=/|$)")


@dataclass(frozen=True)
class BashEffects:
    """What one shell command wrote, with its ``cd`` working directory tracked.

    ``outputs`` are raw path spellings, already joined against the directory the shell was
    in at that point (``~/``-rooted, absolute or session-cwd-relative). ``sweep_dirs`` are the
    directories the command ``cd``'d into AND wrote something in: the only places the
    touched-directory sweep may look (see :func:`_sweep`).
    """

    outputs: tuple[str, ...] = ()
    sweep_dirs: tuple[str, ...] = ()


def _variable_write(code: str) -> bool:
    """Whether python ``code`` writes through a path no literal names (arms the sweep)."""
    return bool(_PY_VARIABLE_WRITE.search(_EVAL_WRITES.sub("", code)))


def _lift_heredocs(command: str) -> tuple[str, list[str]]:
    """The command with each heredoc body removed and its operator replaced by a marker.

    WHY: a python heredoc is full of quotes and apostrophes that make ``shlex`` raise on the
    whole command, which used to make ``cd d; python3 - <<PY ... open('r.md','w')`` name
    NOTHING. Bodies are returned in operator order for :func:`bash_effects` to scan.
    """
    lines = command.split("\n")
    kept: list[str] = []
    bodies: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        operators: list[tuple[str, bool]] = []

        def mark(match: "re.Match[str]") -> str:
            operators.append((match.group("tag"), bool(match.group("dash"))))
            return " " + _HEREDOC_MARK.format(len(bodies) + len(operators) - 1) + " "

        # Quoted/comment text is not shell syntax. Mask it for operator discovery,
        # preserving offsets so replacement keeps every original non-operator byte.
        masked = list(line)
        quote = ""
        escaped = False
        for pos, char in enumerate(line):
            if escaped:
                masked[pos] = " "
                escaped = False
            elif char == "\\" and quote != "'":
                masked[pos] = " "
                escaped = True
            elif quote:
                masked[pos] = " "
                if char == quote:
                    quote = ""
            elif char in {"'", '"'}:
                quote = char
                masked[pos] = " "
            elif char == "#" and (pos == 0 or line[pos - 1].isspace()):
                masked[pos:] = " " * (len(line) - pos)
                break
        # Only use the mask to test the operator start: quoted heredoc delimiters
        # themselves are valid and must be parsed from the original line.
        matches = [
            m for m in _HEREDOC.finditer(line) if "".join(masked[m.start() : m.start() + 2]) == "<<"
        ]
        pieces = []
        previous = 0
        for match in matches:
            pieces.extend((line[previous : match.start()], mark(match)))
            previous = match.end()
        pieces.append(line[previous:])
        kept.append("".join(pieces))
        for tag, dash in operators:
            body: list[str] = []
            while index < len(lines):
                candidate = lines[index]
                index += 1
                if (candidate.strip() if dash else candidate.rstrip()) == tag:
                    break
                body.append(candidate)
            bodies.append("\n".join(body))
    return "\n".join(kept), bodies


def _shell_tokens(text: str) -> list[str] | None:
    """Words and punctuation of a (heredoc-free) shell text; ``None`` when it will not lex."""
    lexer = shlex.shlex(text, posix=True, punctuation_chars=";()<>|&\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens: list[str] = []
    try:
        for token in lexer:
            if token and all(char in ";()<>|&\n" for char in token):
                tokens.extend(_PUNCT_PIECES.findall(token))
            else:
                tokens.append(token)
    except ValueError:
        return None
    return tokens


def _in_dir(raw: str, cwd: str | None) -> str | None:
    """``raw`` joined onto the shell's directory, or ``None`` when that is unknowable.

    ``cwd`` is ``""`` before any ``cd`` (names stay session-cwd-relative), a path after one,
    and ``None`` after a ``cd`` whose target the harness cannot know statically: a relative
    name is then DROPPED, never guessed against the session cwd (the wrong file, or none).
    """
    raw = _LEADING_HOME.sub("~", raw)
    if raw.startswith(("~", "/")):
        return raw
    if cwd is None:
        return None
    return os.path.join(cwd, raw) if cwd else raw


def bash_effects(command: str) -> BashEffects:
    """Output paths named by one shell command (tier 2), ``cd``-aware. Tolerant: bad lexing -> none.

    Walks the command statement by statement tracking the working directory (``cd a && cmd``,
    ``cd a; cmd``, ``( cd a; cmd )``), and finds: output flags (``-o``/``--output``),
    redirects, ``tee [-a]``, the destination of ``cp``/``mv``/``install`` (plus
    ``dest/<source name>`` for a directory destination) and the python write literals of a
    heredoc that feeds an interpreter. Anything with a ``$`` or a glob is dropped: it names no
    single file.
    """
    # Shared with FileScan, which reads unbounded journal arguments rather than clipped
    # supplement snapshots. Refuse oversized commands before lexing/regex work; truncating
    # a shell program can invent a different command (especially inside a heredoc).
    if len(command) > SHELL_SCAN_MAX_CHARS or "\x00" in command:
        return BashEffects()
    text, bodies = _lift_heredocs(command)
    tokens = _shell_tokens(text)
    if tokens is None:
        return BashEffects()
    outputs: list[str] = []
    sweep_dirs: list[str] = []
    cwd: str | None = ""
    stack: list[str | None] = []
    segment: list[str] = []

    def emit(raw: str, found: list[str]) -> None:
        if raw and not raw.startswith(("-", "&", "/dev/")) and "$" not in raw and "*" not in raw:
            joined = _in_dir(raw, cwd)
            if joined:
                found.append(joined)

    def flush() -> None:
        nonlocal cwd
        words, segment[:] = list(segment), []
        while words and "=" in words[0] and not words[0].startswith(("=", "-")):
            words.pop(0)  # ``FOO=bar cmd``
        if not words:
            return
        if words[0] == "cd":
            target = next((w for w in words[1:] if w != "--"), "~")
            target = _LEADING_HOME.sub("~", target)
            if "$" in target or "*" in target or target.startswith("-") or "`" in target:
                cwd = None
            elif target.startswith(("~", "/")):
                cwd = target
            else:
                cwd = None if cwd is None else os.path.join(cwd, target)
            return
        found: list[str] = []
        operands: list[str] = []
        redirected = False
        for index, word in enumerate(words):
            following = words[index + 1] if index + 1 < len(words) else ""
            if redirected:
                redirected = False
                continue
            if word in _REDIRECT_TOKENS:
                emit(following, found)
                redirected = True
                if operands and operands[-1].isdigit():
                    operands.pop()  # ``2>err.txt``: the lexer split the fd off the operator
            elif word in _OUTPUT_FLAGS:
                emit(following, found)
            elif word.startswith(("--output=", "--out=")):
                emit(word.split("=", 1)[1], found)
            elif _REDIRECT_ATTACHED.match(word):
                attached = _REDIRECT_ATTACHED.match(word)
                emit(attached.group("path") if attached else "", found)
            elif word in _HEREDOC_OPERANDS or word.startswith(("<", ">")):
                # ``2>&1`` / ``<in`` / ``<<MARK``: the word after is a descriptor, an INPUT or
                # a heredoc marker, never an output; the digit before ``>&`` is its fd.
                redirected = True
                if word == ">&" and operands and operands[-1].isdigit():
                    operands.pop()
                continue
            elif not word.startswith("-") and not _HEREDOC_MARK_RE.match(word):
                operands.append(word)
        command_word = os.path.basename(words[0])
        if command_word == "tee":
            for word in operands[1:]:
                emit(word, found)
        elif command_word in _COPY_COMMANDS and len(operands) >= 3:
            # GNU target-directory forms move the destination before the sources. Until
            # their option grammar is supported, refuse rather than feature a source.
            if any(
                w in {"-t", "--target-directory"}
                or w.startswith("--target-directory=")
                or (w.startswith("-t") and len(w) > 2)
                for w in words[1:]
            ):
                return
            destination = operands[-1]
            if not destination.endswith("/"):  # a directory is not itself an output
                emit(destination, found)
            for source in operands[1:-1]:
                if "*" not in source and "$" not in source:
                    emit(destination.rstrip("/") + "/" + os.path.basename(source), found)
        unnamed_write = False
        if any(_PYTHON_WORD.match(os.path.basename(w)) for w in words):
            for index, word in enumerate(words[:-1]):
                if word == "-c":  # ``python3 -c "open('r.md','w')..."``: the code is one word
                    literals = _eval_outputs(words[index + 1])
                    for raw in literals:
                        emit(raw, found)
                    unnamed_write = unnamed_write or _variable_write(words[index + 1])
            for word in words:
                mark = _HEREDOC_MARK_RE.match(word)
                if mark and int(mark.group(1)) < len(bodies):
                    body = bodies[int(mark.group(1))]
                    literals = _eval_outputs(body)
                    for raw in literals:
                        emit(raw, found)
                    unnamed_write = unnamed_write or _variable_write(body)
        outputs.extend(found)
        if unnamed_write and cwd:
            sweep_dirs.append(cwd)

    for token in tokens:
        if token in _SEPARATORS:
            flush()
        elif token == "(":
            flush()
            stack.append(cwd)
        elif token == ")":
            flush()
            if stack:
                cwd = stack.pop()
        else:
            segment.append(token)
    flush()
    return BashEffects(tuple(dict.fromkeys(outputs)), tuple(dict.fromkeys(sweep_dirs)))


#: ``<`` / ``<<`` tokens and the word after a redirect that is an INPUT, never an output.
_HEREDOC_OPERANDS: Final = frozenset({"<", "<<", "<<-", ">&"})


def _bash_outputs(command: str) -> list[str]:
    """Output paths named by one shell command (tier 2); see :func:`bash_effects`.

    Kept under its old name because ``session_files.scan`` imports it: one parser for "what
    did this command write", so the Files inventory gains the ``cd`` tracking too.
    """
    return list(bash_effects(command).outputs)


def _eval_outputs(code: str) -> list[str]:
    # FileScan feeds full eval arguments too; keep the shared grammar's work bounded
    # regardless of which tool supplied Python, without truncating a literal into a path.
    if len(code) > SHELL_SCAN_MAX_CHARS or "\x00" in code:
        return []
    found: list[str] = []
    for match in _EVAL_WRITES.finditer(code):
        item = match.group("path") or match.group("path2") or match.group("path3")
        if item and "{" not in item:
            found.append(item)
    return found


@dataclass(frozen=True)
class Mention:
    """One door-1/2/3 mention of a path, before it is admitted as a :class:`Candidate`."""

    raw: str
    tier: int
    tool: str
    intent: str
    order: int


def _harvest(items: Sequence[TurnItem]) -> tuple[list[Mention], list[tuple[str, str, int]]]:
    """Every door-1 and door-2 mention, plus the ``(dir, intent, order)`` sweep candidates.

    A call whose tool RESULT errored wrote nothing, so it is not evidence: results are
    matched back to calls by ``tool_call_id``.
    """
    failed = {item.tool_call_id for item in items if item.role == "tool" and item.is_error}
    harvested: list[Mention] = []
    sweep: list[tuple[str, str, int]] = []
    order = 0
    for item in items:
        if item.role != "assistant":
            continue
        for call in item.tool_calls:
            order += 1
            if call.id in failed:
                continue
            intent = sanitize_prompt_line(call.args.get("i", ""), limit=_INTENT_MAX_CHARS)
            if call.name == "bash":
                effects = bash_effects(call.args.get("command", ""))
                harvested.extend(
                    Mention(p, TIER_COMMAND, "bash", intent, order) for p in effects.outputs
                )
                sweep.extend((d, intent, order) for d in effects.sweep_dirs)
            else:
                harvested.extend(_from_call(call, intent, order))
    return harvested, sweep


def _from_call(call: ToolCallItem, intent: str, order: int) -> Iterable[Mention]:
    if call.name == "write" or (
        call.name == "edit"
        and kind_of(call.args.get("path", "")) in {"markdown", "text", "html", "document", "pdf"}
    ):
        path = call.args.get("path", "")
        if path:
            yield Mention(path, TIER_WRITE, call.name, intent, order)
    elif call.name == "eval":
        for path in _eval_outputs(call.args.get("code", "")):
            yield Mention(path, TIER_COMMAND, "eval", intent, order)


#: Touched-directory sweep bounds. The sweep exists for the write a command makes through a
#: VARIABLE path (``p = ...; json.dump(d, open(p, 'w'))``) or a glob-fed tool
#: (``shasum *.csv > SHA256SUMS`` is a redirect and is named, but its sibling outputs are
#: not), where no literal names the file. It lists only a directory the command ``cd``'d into
#: and wrote in, one level, at most this many entries, and keeps at most this many files.
SWEEP_MAX_ENTRIES: Final = 200
SWEEP_MAX_FILES: Final = 8
#: A file counts as touched this turn if its mtime is within this slack of the turn start
#: (coarse filesystem timestamps; a clock read a hair after the first write).
SWEEP_SLACK_S: Final = 2.0


def _sweep(directory: str, since: float) -> list[str]:
    """Regular, visible files of a deliverable kind in ``directory`` modified since ``since``.

    Non-recursive and bounded (:data:`SWEEP_MAX_ENTRIES` entries read). Dotfiles are skipped
    outright (``.env`` and friends never need the denylist to catch them); extension-less names
    qualify only as checksum manifests (:func:`kind_of`). Returns names, newest first.
    """
    found: list[tuple[float, str]] = []
    try:
        with os.scandir(directory) as entries:
            for entry in islice(entries, SWEEP_MAX_ENTRIES):
                if entry.name.startswith(".") or kind_of(entry.name) not in _DELIVERABLE_KINDS:
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    modified = entry.stat(follow_symlinks=False).st_mtime
                except OSError:
                    continue
                if modified >= since - SWEEP_SLACK_S:
                    found.append((modified, entry.name))
    except (OSError, ValueError):
        return []
    found.sort(reverse=True)
    return [name for _modified, name in found[:SWEEP_MAX_FILES]]


def _spellings(raw: str, absolute: str, display: str, home: str) -> set[str]:
    """Every way the answer could have spelled this path, comparable with _answer_spans."""
    spellings = {raw, absolute, display}
    if _under(absolute, home):
        spellings.add("~/" + os.path.relpath(absolute, home))
    return {item[2:] if item.startswith("./") else item for item in spellings}


def _answer_spans(answer: str) -> set[str]:
    """Every token the answer shows in a code span or a markdown link (target and text)."""
    spans: set[str] = set()
    for match in _CODE_SPAN.finditer(answer):
        spans.update(match.group(1).split())
    for match in _MD_LINK.finditer(answer):
        spans.add(match.group("target"))
        spans.update(match.group("text").split())
    cleaned: set[str] = set()
    for span in spans:
        span = span.strip(".,;:()[]\"'")
        if span.startswith("file://"):
            span = span[len("file://") :]
        cleaned.add(span[2:] if span.startswith("./") else span)
    return cleaned


# -- admission ---------------------------------------------------------------------------


def _under(path: str, root: str) -> bool:
    root = root.rstrip(os.sep) or os.sep
    return path == root or path.startswith(root + os.sep)


def _under_any(path: str, roots: Sequence[str]) -> bool:
    return any(_under(path, root) for root in roots)


def _root_forms(root: str) -> list[str]:
    """A root as written AND resolved.

    Both, because macOS temp roots are symlinks (``/var`` -> ``/private/var``): a candidate's
    ``realpath`` lands under the resolved form while the configured cwd carries the written
    one, and comparing only one pair silently rejects every file in that tree
    (``resolves-outside-roots``) -- measured on this host with a ``/var/folders`` workdir.
    """
    forms = [root]
    resolved = os.path.realpath(root)
    if resolved != root:
        forms.append(resolved)
    return forms


def _scratch_roots() -> list[str]:
    # Explicit annotation: ``_SCRATCH_ROOTS`` is a ``Final`` tuple of literals and pyright
    # narrows its element type to the literal strings, which then refuses ``append``.
    roots: list[str] = list(_SCRATCH_ROOTS)
    roots.append(tempfile.gettempdir())
    roots.extend(os.path.realpath(root) for root in list(roots))
    return roots


def _excluded(absolute: str, resolved: str, deny: Sequence[str], cwd: str) -> str:
    """The exclusion reason for a path, or ``""``. Deliverables only (memo §2.3).

    The temp-dir exclusion yields to the session cwd: a project that LIVES under a temp root
    (a throwaway checkout in ``/tmp``, a test sandbox) is where the operator is working, so
    its files are not scratch. The scratchpad and the config dir are denied absolutely, by
    the denylist, whatever the cwd.
    """
    for form in {absolute, resolved}:
        if not _under_any(form, _root_forms(cwd)) and any(
            _under(form, root) for root in _scratch_roots()
        ):
            return "scratch"
        parts = form.split(os.sep)[:-1]
        if any(part in _EXCLUDED_COMPONENTS for part in parts):
            return "generated-tree"
        if form.endswith(_EXCLUDED_SUFFIXES):
            return "excluded-suffix"
        if any(_under(form, prefix) for prefix in deny):
            return "deny-prefix"
    return ""


def _normalise_deny(prefixes: Iterable[str], cwd: str) -> list[str]:
    roots: list[str] = []
    for prefix in prefixes:
        expanded = os.path.expanduser(prefix)
        if not os.path.isabs(expanded):
            expanded = os.path.join(cwd, expanded)
        expanded = os.path.normpath(expanded)
        roots.append(expanded)
        resolved = os.path.realpath(expanded)
        if resolved != expanded:
            roots.append(resolved)
    return roots


def _expand_home(raw: str, home: str) -> str:
    """``~``/``~/x`` against the pre-filter's ``home`` (not the process's), like
    ``session_files.scan.resolve_path``: one pinned home for every path in a pass. In
    production the two are the same directory; a test or a reader that pins ``home`` gets
    the file it meant. ``~user`` forms are left to ``expanduser``."""
    if raw == "~":
        return home
    if raw.startswith("~/"):
        return os.path.join(home, raw[2:])
    return os.path.expanduser(raw)


def _display_path(absolute: str, cwd: str, home: str) -> str:
    """cwd-relative if under cwd, else ``~/``-relative; never absolute (memo S-R14)."""
    if _under(absolute, cwd):
        return os.path.relpath(absolute, cwd)
    return "~/" + os.path.relpath(absolute, home)


def prefilter(
    items: Sequence[TurnItem],
    answer_text: str,
    *,
    cwd: str,
    deny_prefixes: Sequence[str] = (),
    home: str | None = None,
    want_files: bool = True,
    want_graphics: bool = True,
    since: float = 0.0,
) -> Prefilter:
    """Run the whole deterministic pre-filter over the logical turn.

    ``want_files``/``want_graphics`` are the operator's sub-switches: a half that is off
    contributes nothing, and the gate (``candidates == [] and not structured``) is evaluated
    over what is left, so ``files: false`` plus a table in the answer still stops when
    ``graphics`` is also off.

    ``since`` is the logical turn's start (epoch seconds) and arms the touched-directory
    sweep (:func:`_sweep`). ``0`` -- the default, and what a provenance that never recorded
    a start carries -- DISABLES the sweep: without a start every file in the directory would
    look "written this turn", which is the over-matching failure, so absent means off.
    """
    started = time.perf_counter()
    home = os.path.abspath(home or os.path.expanduser("~"))
    cwd = os.path.abspath(cwd or ".")
    cwd_roots = _root_forms(cwd)
    home_roots = _root_forms(home)
    deny = _normalise_deny(deny_prefixes, cwd)
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    candidates: dict[str, Candidate] = {}
    prepared_content: dict[str, tuple[str, str]] = {}
    failed_calls = {item.tool_call_id for item in items if item.role == "tool" and item.is_error}
    if want_graphics:
        for item in items:
            for call in item.tool_calls:
                if call.id in failed_calls:
                    continue
                if call.name in {"bash", "eval"}:
                    prepared_content.clear()
                if call.name in {"write", "edit"}:
                    raw = os.path.expanduser(call.args.get("path", ""))
                    absolute = os.path.normpath(
                        raw if os.path.isabs(raw) else os.path.join(cwd, raw)
                    )
                    identity = os.path.realpath(absolute)
                    prepared_content.pop(identity, None)
                    if call.prepared_html and call.name == "write" and len(prepared_content) < 3:
                        prepared_content[identity] = (absolute, call.prepared_html)
    prepared: list[tuple[Candidate, str]] = []
    if want_files or prepared_content:
        mentions, sweep_dirs = _harvest(items)
        if since > 0:
            for directory, intent, order in sweep_dirs:
                base = _expand_home(directory, home) if directory else cwd
                base = os.path.normpath(base if os.path.isabs(base) else os.path.join(cwd, base))
                if not (_under_any(base, cwd_roots) or _under_any(base, home_roots)):
                    reject("outside-roots")
                    continue
                mentions.extend(
                    Mention(os.path.join(base, name), TIER_COMMAND, "bash", intent, order)
                    for name in _sweep(base, since)
                )
        # Links participate in admission too; a tier-3 mention of a file the answer already
        # shows is deduped below, a tier-1/2 deliverable is kept and flagged. Metadata
        # discovery must never turn read source files into products.
        for match in _MD_LINK.finditer(answer_text):
            target = match.group("target")
            if kind_of(target) in _DELIVERABLE_KINDS and ":" not in target:
                mentions.append(Mention(target, TIER_PROSE, "answer", "", 10_000))
        for match in _PROSE_PATH.finditer(answer_text):
            if kind_of(match.group("path")) in _DELIVERABLE_KINDS:
                mentions.append(Mention(match.group("path"), TIER_PROSE, "answer", "", 10_000))
        visible = _answer_spans(answer_text)
        for mention in mentions[: MAX_CONSIDERED * 4]:
            raw, tier, tool, intent, order = (
                mention.raw,
                mention.tier,
                mention.tool,
                mention.intent,
                mention.order,
            )
            if len(candidates) >= MAX_CONSIDERED:
                break
            expanded = _expand_home(raw.strip(), home)
            if not expanded:
                continue
            absolute = os.path.normpath(
                expanded if os.path.isabs(expanded) else os.path.join(cwd, expanded)
            )
            if not (_under_any(absolute, cwd_roots) or _under_any(absolute, home_roots)):
                reject("outside-roots")
                continue
            try:
                resolved = os.path.realpath(absolute)
                info = os.stat(absolute)
            except (OSError, ValueError):
                reject("missing")
                continue
            if not stat_module.S_ISREG(info.st_mode):
                reject("not-regular")
                continue
            if not (_under_any(resolved, cwd_roots) or _under_any(resolved, home_roots)):
                reject("resolves-outside-roots")
                continue
            reason = _excluded(absolute, resolved, deny, cwd)
            if reason:
                reject(reason)
                continue
            # SENSITIVE is checked before the candidate can be listed, sent or previewed.
            if denylist.is_sensitive(absolute, cwd=cwd):
                reject("sensitive")
                continue
            display = _display_path(absolute, cwd, home)
            captured = prepared_content.get(os.path.realpath(absolute))
            held = candidates.get(absolute)
            kind = kind_of(absolute)
            # Linked in the answer: its path in a code span / link, in any spelling, or --
            # for the flag only -- its bare base name (``RUN-RECORD.md`` in backticks beside
            # ``in ~/workspace/x/``). The base-name form never DEDUPS: two files can share a name.
            linked = not _spellings(raw.strip(), absolute, display, home).isdisjoint(visible)
            linked_name = linked or os.path.basename(absolute) in visible
            if (captured is None or captured[0] != absolute) and linked:
                # Dedup stays for what the turn only MENTIONED (tier 3) and for non-deliverable
                # kinds (a linked script). A tier 1/2 deliverable is kept, flagged below.
                if held is not None and held.tier <= TIER_COMMAND:
                    candidates[absolute] = replace(held, linked_in_answer=True)
                    continue
                if tier == TIER_PROSE or kind not in _DELIVERABLE_KINDS:
                    reject("already-visible")
                    continue
            if held is not None and (held.tier, -held.order) <= (tier, -order):
                if linked_name and not held.linked_in_answer:
                    candidates[absolute] = replace(held, linked_in_answer=True)
                continue
            candidates[absolute] = Candidate(
                path=display,
                absolute=absolute,
                name=os.path.basename(absolute),
                kind=kind,
                size_bytes=int(info.st_size),
                mtime=float(info.st_mtime),
                tier=tier,
                tool=tool,
                order=order,
                intent=intent,
                linked_in_answer=linked_name or (held is not None and held.linked_in_answer),
            )
    prepared = [
        (candidate, captured[1])
        for candidate in candidates.values()
        if (captured := prepared_content.get(os.path.realpath(candidate.absolute))) is not None
        and captured[0] == candidate.absolute
    ][:3]
    ordered = (
        tuple(sorted(candidates.values(), key=lambda c: (c.tier, -c.order, c.path)))
        if want_files
        else ()
    )
    evidence = extract(items, answer_text) if want_graphics else Evidence(structured=False)
    return Prefilter(
        candidates=ordered,
        evidence=evidence,
        skipped=None if (ordered or evidence.structured or prepared) else "prefilter",
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        rejected=rejected,
        prepared=tuple(prepared),
    )
