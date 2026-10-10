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
markdown link, which the UI keeps outside the fold.

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
from dataclasses import dataclass, field
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
)
#: Tier-3 prose path: ``/abs/x.pdf`` or ``~/x.csv``. No spaces, so a sentence cannot swallow
#: its neighbour; ``(?<![\w:/.])`` rejects URLs and mid-token matches.
_PROSE_PATH: Final = re.compile(
    r"(?<![\w:/.\-])(?P<path>(?:~|/)[\w.\-+@%/~]*\.[A-Za-z0-9]{1,8})(?![\w/])"
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

    @property
    def deliverable(self) -> bool:
        return self.kind in _DELIVERABLE_KINDS

    @property
    def why(self) -> str:
        """The row's ``why`` string (memo §2.4 example ``written by write``)."""
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


def kind_of(name: str) -> str:
    return _KIND_BY_EXT.get(os.path.splitext(name)[1].casefold(), "other")


# -- harvesting --------------------------------------------------------------------------


def _bash_outputs(command: str) -> list[str]:
    """Output paths named by one shell command (tier 2). Tolerant: bad quoting -> none."""
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        return []
    found: list[str] = []
    for index, token in enumerate(tokens):
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token in _OUTPUT_FLAGS or token in _REDIRECT_TOKENS or token == "tee":
            found.append(following)
        elif token.startswith("--output=") or token.startswith("--out="):
            found.append(token.split("=", 1)[1])
        elif token == "-a" and index and tokens[index - 1] == "tee":
            found.append(following)
        else:
            attached = _REDIRECT_ATTACHED.match(token)
            if attached:
                found.append(attached.group("path"))
    return [
        item
        for item in found
        if item and not item.startswith(("-", "&", "/dev/")) and "$" not in item and "*" not in item
    ]


def _eval_outputs(code: str) -> list[str]:
    found: list[str] = []
    for match in _EVAL_WRITES.finditer(code):
        item = match.group("path") or match.group("path2")
        if item and "{" not in item:
            found.append(item)
    return found


def _harvest(
    items: Sequence[TurnItem],
) -> list[tuple[str, int, str, str, int]]:
    """``(raw path, tier, tool, intent, order)`` for every door-1 and door-2 mention.

    A call whose tool RESULT errored wrote nothing, so it is not evidence: results are
    matched back to calls by ``tool_call_id``.
    """
    failed = {item.tool_call_id for item in items if item.role == "tool" and item.is_error}
    harvested: list[tuple[str, int, str, str, int]] = []
    order = 0
    for item in items:
        if item.role != "assistant":
            continue
        for call in item.tool_calls:
            order += 1
            if call.id in failed:
                continue
            intent = sanitize_prompt_line(call.args.get("i", ""), limit=_INTENT_MAX_CHARS)
            harvested.extend(_from_call(call, intent, order))
    return harvested


def _from_call(
    call: ToolCallItem, intent: str, order: int
) -> Iterable[tuple[str, int, str, str, int]]:
    if call.name == "write":
        path = call.args.get("path", "")
        if path:
            yield (path, TIER_WRITE, "write", intent, order)
    elif call.name == "bash":
        for path in _bash_outputs(call.args.get("command", "")):
            yield (path, TIER_COMMAND, "bash", intent, order)
    elif call.name == "eval":
        for path in _eval_outputs(call.args.get("code", "")):
            yield (path, TIER_COMMAND, "eval", intent, order)


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
) -> Prefilter:
    """Run the whole deterministic pre-filter over the logical turn.

    ``want_files``/``want_graphics`` are the operator's sub-switches: a half that is off
    contributes nothing, and the gate (``candidates == [] and not structured``) is evaluated
    over what is left, so ``files: false`` plus a table in the answer still stops when
    ``graphics`` is also off.
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
    if want_files:
        mentions = _harvest(items)
        for match in _PROSE_PATH.finditer(answer_text):
            if kind_of(match.group("path")) in _DELIVERABLE_KINDS:
                mentions.append((match.group("path"), TIER_PROSE, "answer", "", 10_000))
        visible = _answer_spans(answer_text)
        for raw, tier, tool, intent, order in mentions[: MAX_CONSIDERED * 4]:
            if len(candidates) >= MAX_CONSIDERED:
                break
            expanded = os.path.expanduser(raw.strip())
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
            if not _spellings(raw.strip(), absolute, display, home).isdisjoint(visible):
                reject("already-visible")
                continue
            held = candidates.get(absolute)
            if held is not None and (held.tier, -held.order) <= (tier, -order):
                continue
            candidates[absolute] = Candidate(
                path=display,
                absolute=absolute,
                name=os.path.basename(absolute),
                kind=kind_of(absolute),
                size_bytes=int(info.st_size),
                mtime=float(info.st_mtime),
                tier=tier,
                tool=tool,
                order=order,
                intent=intent,
            )
    ordered = tuple(sorted(candidates.values(), key=lambda c: (c.tier, -c.order, c.path)))
    evidence = extract(items, answer_text) if want_graphics else Evidence(structured=False)
    return Prefilter(
        candidates=ordered,
        evidence=evidence,
        skipped=None if (ordered or evidence.structured) else "prefilter",
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        rejected=rejected,
    )
