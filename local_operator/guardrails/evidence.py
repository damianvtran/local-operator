"""Evidence that THIS session (or a subagent it launched) created a delete target.

WHY. The creation ledger (``provenance.py``) only knows what was created after
it shipped, and only through the shapes the resolver can parse (``mkdir``,
``git worktree add`` …). The cleanups that stalled the operator's lanes were
made earlier and by rig scripts: a QA child wrote ``qa_r5_rig.py`` (``RIG =
REAL_HOME / f"qa-r5-dg-{os.getpid()}"``) and ran it with ``python``. Nothing in
the command names the directory it made. The session's own HISTORY does,
though, and it is on disk: every tool call and its real output, with
timestamps, in ``<state>/sessions/<id>/transcript.jsonl``.

So when a delete ranks HIGH, the checkpoint extracts an evidence list from that
history — deterministically, before any model sees it — and the forked judge
may only ALLOW by citing entries of it. The checkpoint then re-verifies each
cited entry (:func:`verify`). That is what stops a judge persuaded by injected
text in the forked prefix: it can only pass with a real creating call.

WHAT COUNTS. Only the agent's own TOOL CALLS and their real results: the
arguments it sent and the output the tool returned. Never prose (assistant
text, a subagent's final report), never a user-role row (Team Bridge, peer,
inbox, hub), never a fetched page or file content returned by ``read`` —
those are text anyone can author, and a fake "tool output" inside them is
exactly the injection this module must not believe. Only ``bash``/``eval``
calls can create; a ``write`` counts only as the SCRIPT a later bash/eval call
of the same session ran.

A call covers a target ``T`` when (all of):

* the call is not read-only (``ls``, ``cat``, ``find`` without ``-delete``,
  ``git status``/``log`` … never create anything);
* ``T``, or an ancestor ``A`` of ``T`` below the protected roots, is a
  directory whose birth time (``st_birthtime``; ``st_ctime`` where the
  platform has none) falls inside the call's window;
* and EITHER (B1) re-resolving the call's command yields a creation target
  equal to ``A`` or an ancestor of it, OR (B2) the call's own command or a
  script the same session wrote and the call ran contains ``A``'s literal
  basename or the literal name prefix that produced it (``qa-r5-dg-`` in
  ``f"qa-r5-dg-{os.getpid()}"``), OR the call's real output contains ``A``'s
  FULL path (absolute or ``~/``). A bare name in output is a listing or a
  log, not a creation (review R1-MAJOR-2).

The window and the name together rule out a directory another session
created meanwhile. CRITICAL targets and primary checkouts never reach this
module (``classify`` ranks them first; the caller re-checks).

WHERE THE CALLS COME FROM. The CURRENT session's calls come from the
harness's in-memory rows (``extract_live``), never its own transcript file.
Descendants' calls come from their on-disk transcripts, which ``classify``
ranks CRITICAL to write, append, move or delete, as it does ``origin.json``
and the ask store. An interpreter writing them (``python -c``, ``node -e``)
is not modelled by the shell resolver: that is the documented ceiling, the
same one as ``python -c 'shutil.rmtree(...)'``.

DESCENDANTS. Found, recursively, from the harness-written lineage registry
(``provenance.read_lineage``, under ``<state>/guardrails/``) — not from the
parent's roster, which is pruned. ``origin.json`` parent pointers are a
MIGRATION fallback only, for child directories born before the registry file
itself was (sessions that predate this change, such as the 2026-11-07 QA
field case): a directory planted later is born after the registry existed,
so it cannot use the fallback. That is a narrowing, not a seal — macOS
``utimes`` can move a birth time backwards, and an interpreter write is the
same ceiling as above. Symlinked session directories, transcripts and
``origin.json`` files are never read.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TRANSCRIPT_FILENAME = "transcript.jsonl"
ORIGIN_FILENAME = "origin.json"
#: Clock slack around a call's window. The window opens at the row written
#: before the call's assistant row and closes at its result row (see
#: ``extract``), both from the same host clock as the filesystem's; 2 s absorbs
#: write ordering without admitting a neighbouring batch's work.
_SLACK_S = 2.0
#: Creating tools. ``write`` is not here: it creates a FILE, and it counts only
#: as the script a bash/eval call ran.
_CREATING_TOOLS = frozenset({"bash", "eval"})
#: Per-row text kept for the name check, and the excerpt shown to the judge.
_MAX_TEXT = 200_000
_EXCERPT = 160
#: Bounds on the descendant walk and on how many evidence rows reach the judge.
_MAX_DESCENDANTS = 400
#: One cleanup can cover dozens of rig directories, each made by its own run
#: (the field case: 21). Each target cites at most one call, so this bounds the
#: brief at ~64 short lines.
_MAX_LISTED = 64
#: Process cache for the ``origin.json`` scan (2.7k session dirs measured at
#: ~0.5 s); HIGH-ranked deletes are rare, a stale child list for 30 s is not.
_TREE_TTL_S = 30.0
_TREE_CACHE: dict[str, tuple[float, dict[str, list[str]]]] = {}


@dataclass(frozen=True)
class Evidence:
    """One tool call from the session tree, as the transcript recorded it."""

    session_id: str
    call_id: str
    tool: str
    #: The bash command / eval code / write path.
    command: str
    started: float
    finished: float | None
    exit_status: str
    output: str
    root_session: str = ""
    #: ``write`` only: the file content.
    content: str = ""

    @property
    def label(self) -> str:
        if self.session_id == self.root_session:
            return f"call {self.call_id}"
        return f"subagent {self.session_id} call {self.call_id}"


@dataclass
class EvidenceSet:
    """The tree's calls plus, per target, the calls that verifiably cover it."""

    calls: list[Evidence] = field(default_factory=list)
    #: ``e1``… → the covering call, in the order shown to the judge.
    listed: dict[str, Evidence] = field(default_factory=dict)
    #: target path → evidence ids that cover it.
    covers: dict[str, list[str]] = field(default_factory=dict)

    def describe(self) -> tuple[str, ...]:
        lines = []
        for eid, ev in self.listed.items():
            out = ev.output.strip().replace("\n", " ")[:_EXCERPT]
            lines.append(
                f"{eid}: {ev.label} ({ev.tool}, exit {ev.exit_status or '?'}, "
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ev.started))}): "
                f"{ev.command.strip()[:_EXCERPT]} -> {out}"
            )
        return tuple(lines)


# -- session tree ------------------------------------------------------------------


def _children_index(sessions: Path) -> dict[str, list[str]]:
    """LEGACY: ``parent → [child]`` from ``origin.json`` files (see module doc)."""
    key = str(sessions)
    cached = _TREE_CACHE.get(key)
    now = time.monotonic()
    if cached is not None and now - cached[0] < _TREE_TTL_S:
        return cached[1]
    index: dict[str, list[str]] = {}
    try:
        entries = list(os.scandir(sessions))
    except OSError:
        entries = []
    for entry in entries:
        try:
            if entry.is_symlink():
                continue
            fd = os.open(os.path.join(entry.path, ORIGIN_FILENAME), os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, encoding="utf-8") as fh:
                origin = json.load(fh)
        except (OSError, ValueError):
            continue
        parent = origin.get("parent") if isinstance(origin, dict) else None
        if isinstance(parent, str) and parent and origin.get("origin") == "subagent":
            index.setdefault(parent, []).append(entry.name)
    _TREE_CACHE[key] = (now, index)
    return index


def _session_created(sessions: Path, session_id: str) -> float | None:
    """When ``session_id`` was created: ``created_at.json``, else its dir's birth."""
    directory = sessions / session_id
    try:
        value = json.loads((directory / "created_at.json").read_text(encoding="utf-8"))
        if isinstance(value, (int, float)):
            return float(value)
    except (OSError, ValueError):
        pass
    return _born(str(directory))


def session_tree(state_root: str | os.PathLike[str], session_id: str) -> list[str]:
    """``session_id`` followed by every descendant.

    Registry children are trusted as recorded. A legacy ``origin.json`` child
    is adopted only when its directory was born BEFORE the lineage registry
    was (the migration window) AND no earlier than the parent's
    ``created_at`` (slack allowed). A session whose own creation time cannot
    be read adopts no legacy children (fail closed: no evidence, the old
    ranking). Forgery is narrowed elsewhere (``classify._touch_reason``): a
    write-shaped shell touch of ``sessions/`` (a redirection, a known
    writer's destination, git ``-C`` into it) ranks CRITICAL, a path handed
    to any other program ranks HIGH, and an interpreter that computes the
    path itself is the documented ceiling (review R4-MINOR-1).
    """
    if not session_id:
        return []
    from local_operator.guardrails.provenance import (
        ensure_lineage_registry,
        read_lineage,
    )

    sessions = Path(state_root) / "sessions"
    cutoff = ensure_lineage_registry(state_root)
    registry: dict[str, list[str]] = {}
    for child, row in read_lineage(state_root).items():
        for parent in (row.get("parent"), row.get("opener")):
            if isinstance(parent, str) and parent:
                registry.setdefault(parent, []).append(child)
    index = _children_index(sessions) if cutoff is not None else {}
    out = [session_id]
    seen = {session_id}
    created = {session_id: _session_created(sessions, session_id)}
    i = 0
    while i < len(out) and len(out) < _MAX_DESCENDANTS:
        parent = out[i]
        parent_created = created.get(parent)
        for child in registry.get(parent, ()):
            if child in seen or os.path.islink(sessions / child):
                continue
            seen.add(child)
            created[child] = _session_created(sessions, child)
            out.append(child)
        for child in index.get(parent, ()):
            if child in seen or parent_created is None or cutoff is None:
                continue
            born = _born(str(sessions / child))
            if born is None or born + _SLACK_S < parent_created or born >= cutoff:
                continue
            seen.add(child)
            created[child] = _session_created(sessions, child)
            out.append(child)
        i += 1
    return out


# -- extraction --------------------------------------------------------------------


def _result_text(payload: Mapping[str, object]) -> str:
    parts = []
    for block in payload.get("content") or []:  # type: ignore[union-attr]
        if isinstance(block, Mapping) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)[:_MAX_TEXT]


_EXIT_RE = re.compile(r"exit code: (-?\d+)")
#: Rows whose ``ts`` differ by less than this were written in one flush.
_FLUSH_S = 0.05
_TS_RE = re.compile(r'"ts":\s*([0-9]+(?:\.[0-9]+)?)')


def extract(
    transcript: Path, session_id: str, root_session: str, *, not_before: float = 0.0
) -> list[Evidence]:
    """The tool calls (and their results) recorded in one transcript.

    Reads ONLY ``role == "assistant"`` rows' ``tool_calls`` and ``role ==
    "tool"`` rows paired to them by ``tool_call_id``. Every other row —
    prose, user rows, custom deliveries — is skipped by construction.
    """
    calls: dict[str, dict[str, object]] = {}
    order: list[str] = []
    try:
        if transcript.stat().st_mtime + _SLACK_S < not_before:
            return []  # last written before the target existed: cannot hold its creation
        # A symlinked transcript (or session dir) is not this tree's record.
        if os.path.islink(os.path.dirname(transcript)):
            return []
        fh = os.fdopen(os.open(transcript, os.O_RDONLY | os.O_NOFOLLOW), encoding="utf-8")
    except OSError:
        return []
    previous_ts = 0.0
    flush_ts = 0.0
    with fh:
        for line in fh:
            # A row's ``ts`` is when it was WRITTEN, and the loop writes an
            # assistant row together with its results at the end of the tool
            # batch — so the call's real start is bounded by the row written
            # before it (the spend record of the model call that produced it,
            # or the previous batch), not by its own ``ts``. Measured on the
            # QA lane this module exists for: assistant and tool rows carried
            # the same ``ts`` while the rig had run for ~24 s before it.
            # Rows flushed together (a harness notice, the assistant row and
            # its results) share one ``ts``; the window opens at the last row
            # of the PREVIOUS flush, or the notice would close it too late.
            m = _TS_RE.search(line)
            if m is not None:
                stamp = float(m.group(1))
                if stamp > flush_ts + _FLUSH_S:
                    previous_ts, flush_ts = flush_ts, stamp
            row_start = previous_ts
            if '"tool' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            payload = row.get("payload") if isinstance(row, dict) else None
            if not isinstance(payload, dict) or row.get("type") != "message":
                continue
            ts = row.get("ts")
            ts = float(ts) if isinstance(ts, (int, float)) else 0.0
            role = payload.get("role")
            if role == "assistant":
                for call in payload.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    cid = str(call.get("id") or "")
                    name = str(call.get("name") or "")
                    args = call.get("arguments")
                    if not cid or name not in (*_CREATING_TOOLS, "write"):
                        continue
                    if not isinstance(args, dict):
                        continue
                    calls[cid] = {"tool": name, "args": args, "started": row_start or ts}
                    order.append(cid)
            elif role == "tool":
                cid = str(payload.get("tool_call_id") or "")
                if cid in calls:
                    calls[cid]["finished"] = ts
                    calls[cid]["output"] = _result_text(payload)
    return _to_evidence(calls, order, session_id, root_session)


def _to_evidence(
    calls: Mapping[str, Mapping[str, object]],
    order: Sequence[str],
    session_id: str,
    root_session: str,
) -> list[Evidence]:
    out: list[Evidence] = []
    for cid in order:
        c = calls[cid]
        args = c["args"]
        assert isinstance(args, dict)
        tool = str(c["tool"])
        if tool == "bash":
            command = str(args.get("command") or "")
        elif tool == "eval":
            command = str(args.get("code") or "")
        else:
            command = str(args.get("path") or "")
        output = str(c.get("output") or "")
        match = _EXIT_RE.search(output[:200])
        out.append(
            Evidence(
                session_id=session_id,
                call_id=cid,
                tool=tool,
                command=command[:_MAX_TEXT],
                started=float(c["started"]),  # type: ignore[arg-type]
                finished=c.get("finished"),  # type: ignore[arg-type]
                exit_status=match.group(1) if match else ("" if tool == "eval" else "?"),
                output=output,
                root_session=root_session,
                content=str(args.get("content") or "")[:_MAX_TEXT] if tool == "write" else "",
            )
        )
    return out


def extract_live(messages: Sequence[object], session_id: str) -> list[Evidence]:
    """The CURRENT session's calls, from the harness's in-memory rows.

    WHY NOT THE DISK TRANSCRIPT. Its own transcript is a file the session can
    reach with its shell; the in-memory list is what the loop itself appended
    (review R1-MAJOR-1). Rows carry no timestamp, so the window comes from the
    provider usage clock: an assistant row's ``usage.at_ms`` is when the model
    call that produced the tool call finished (the call cannot have started
    earlier), and the NEXT assistant row's ``usage.at_ms`` bounds when its
    results were already in (else "now"). That end is deliberately WIDE — it
    includes the next model call's latency — because rows carry no finer
    clock; the name/path check, not the window, is what rules out another
    session's concurrent directory (review R2-NIT-2). A call without a usage clock is not
    evidence. Compacted-away rows are gone from this list, and so is their
    evidence — the ledger still covers what was created through it.
    """
    calls: dict[str, dict[str, object]] = {}
    order: list[str] = []
    pending: list[str] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role == "assistant":
            usage = getattr(message, "usage", None)
            at_ms = getattr(usage, "at_ms", None) if usage is not None else None
            stamp = float(at_ms) / 1000.0 if isinstance(at_ms, (int, float)) else None
            if stamp is not None:
                for cid in pending:
                    calls[cid].setdefault("finished", stamp)
                pending = []
            for call in getattr(message, "tool_calls", None) or []:
                cid = str(getattr(call, "id", "") or "")
                name = str(getattr(call, "name", "") or "")
                args = getattr(call, "arguments", None)
                if not cid or stamp is None or name not in (*_CREATING_TOOLS, "write"):
                    continue
                if not isinstance(args, dict):
                    continue
                calls[cid] = {"tool": name, "args": args, "started": stamp}
                order.append(cid)
                pending.append(cid)
        elif role in ("tool", "toolResult"):
            cid = str(getattr(message, "tool_call_id", "") or "")
            if cid in calls:
                parts = [
                    getattr(block, "text", "")
                    for block in getattr(message, "content", None) or []
                    if isinstance(getattr(block, "text", None), str)
                ]
                calls[cid]["output"] = "\n".join(parts)[:_MAX_TEXT]
    now = time.time()
    for cid in pending:
        calls[cid].setdefault("finished", now)
    return _to_evidence(calls, order, session_id, session_id)


# -- verification ------------------------------------------------------------------


def _born(path: str) -> float | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    birth = getattr(st, "st_birthtime", None)
    return float(birth) if birth is not None else float(st.st_ctime)


def _in_window(born: float, ev: Evidence) -> bool:
    end = ev.finished if ev.finished is not None else ev.started
    return ev.started - _SLACK_S <= born <= end + _SLACK_S


#: A literal name prefix followed by something the call computed: ``qa-r5-dg-``
#: then ``{``/``$``/``%``/a quote/``+``. At least 5 characters ending in a
#: separator, so ``tmp`` alone never vouches for every ``tmp…`` directory.
_PREFIX_TAIL = "{$%'\"+`("


def _names(text: str, name: str) -> bool:
    if not text or not name:
        return False
    if re.search(r"(?<![\w.-])" + re.escape(name) + r"(?![\w.-])", text):
        return True
    for i in range(len(name) - 1, 4, -1):
        if name[i - 1] not in "-_.":
            continue
        prefix, rest = name[:i], name[i:]
        if not re.fullmatch(r"[\w]+", rest):
            continue
        for m in re.finditer(r"(?<![\w.-])" + re.escape(prefix), text):
            nxt = text[m.end() : m.end() + 1]
            if nxt and nxt in _PREFIX_TAIL:
                return True
    return False


def _scripts_named(ev: Evidence, writes: Sequence[Evidence]) -> list[str]:
    """Content of ``write`` calls (same session, earlier) whose file the call names."""
    out = []
    for w in writes:
        if w.session_id != ev.session_id or w.started > ev.started:
            continue
        base = os.path.basename(w.command)
        # ``python qa_r5_rig.py`` names the file; ``from qa_r5_rig import …``
        # (a probe driving the rig) names the module stem.
        names = [base, base[:-3]] if base.endswith(".py") and len(base) > 6 else [base]
        if base and any(
            re.search(r"(?<![\w.-])" + re.escape(n) + r"(?![\w-])", ev.command) for n in names
        ):
            out.append(w.content)
    return out


def _b1(ev: Evidence, anchor: str, env: Mapping[str, str]) -> bool:
    if ev.tool != "bash":
        return False
    from local_operator.guardrails.resolve import resolve_command

    home = env.get("HOME") or os.path.expanduser("~")
    try:
        res = resolve_command(ev.command, cwd=None, env=env, home=home)
    except Exception:  # noqa: BLE001
        return False
    for created in res.created:
        if created.path is None:
            continue
        real = os.path.realpath(created.path)
        # The created directory is ``anchor`` or one of its ancestors.
        if anchor == real or anchor.startswith(real.rstrip("/") + "/"):
            return True
    return False


def _anchors(target: str, stop: Iterable[str]) -> list[str]:
    """``target`` and its ancestors, stopping BEFORE any protected root."""
    stops = {s.rstrip("/") or "/" for s in stop}
    out = []
    cur = os.path.realpath(target)
    while cur not in stops and cur != "/" and len(out) < 32:
        out.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return out


def covers(
    ev: Evidence,
    target: str,
    *,
    writes: Sequence[Evidence],
    env: Mapping[str, str],
    stop: Iterable[str],
) -> bool:
    """Does ``ev`` verifiably cover ``target`` (see the module docstring)?"""
    if ev.tool not in _CREATING_TOOLS:
        return False
    if ev.tool == "bash" and _read_only(ev.command):
        return False
    texts: list[str] | None = None
    for anchor in _anchors(target, stop):
        if not os.path.isdir(anchor) or os.path.isdir(os.path.join(anchor, ".git")):
            # A primary checkout is never vouched for by history.
            if os.path.isdir(os.path.join(anchor, ".git")):
                return False
            continue
        born = _born(anchor)
        if born is None or not _in_window(born, ev):
            continue
        if _b1(ev, anchor, env):
            return True
        if texts is None:
            # Basename/prefix matches count only where THIS call said what to
            # make: its own arguments, or a script this session wrote and the
            # call ran. Output only confirms with the FULL path — a listing or
            # a test log naming ``data`` is not a creation (R1-MAJOR-2 / Q2).
            texts = [ev.command, *_scripts_named(ev, writes)]
        name = os.path.basename(anchor)
        if any(_names(t, name) for t in texts) or _names_full_path(ev.output, anchor, env):
            return True
    return False


def _names_full_path(output: str, anchor: str, env: Mapping[str, str]) -> bool:
    if not output:
        return False
    home = (env.get("HOME") or os.path.expanduser("~")).rstrip("/")
    spellings = {anchor}
    if anchor.startswith(home + "/"):
        spellings.add("~" + anchor[len(home) :])
    return any(
        re.search(r"(?<![\w./~-])" + re.escape(s) + r"(?![\w.-])", output) for s in spellings
    )


#: Programs that only read. A call made of nothing but these (and ``cd``) never
#: creates anything, whatever its window or output (Q2: an ``ls ~`` overlapping
#: another writer's ``mkdir`` was attributed the directory).
_READ_ONLY = frozenset(
    {
        "ls",
        "cat",
        "stat",
        "du",
        "df",
        "grep",
        "rg",
        "egrep",
        "head",
        "tail",
        "wc",
        "less",
        "file",
        "pwd",
        "which",
        "echo",
        "printf",
        "true",
        "sleep",
        "readlink",
        "realpath",
        "basename",
        "dirname",
        "sort",
        "uniq",
        "cut",
        "jq",
        "test",
        "[",
    }
)
_READ_ONLY_GIT = frozenset(
    {"status", "log", "diff", "show", "branch", "worktree list", "rev-parse"}
)


def _read_only(command: str) -> bool:
    """Is every segment of a bash ``command`` a read-only program? Conservative."""
    if not command.strip() or "$(" in command or "`" in command or ">" in command:
        return False
    for segment in re.split(r"&&|\|\||;|\||\n", command):
        words = segment.split()
        while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=\S*", words[0]):
            words = words[1:]
        if not words or words[0] in ("cd", "pushd", "popd"):
            continue
        name = os.path.basename(words[0])
        if name == "find":
            if any(
                w in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint") for w in words
            ):
                return False
            continue
        if name == "git":
            rest = [w for w in words[1:] if not w.startswith("-")]
            sub = " ".join(rest[:2]) if rest[:1] == ["worktree"] else (rest[0] if rest else "")
            if sub in _READ_ONLY_GIT:
                continue
            return False
        if name not in _READ_ONLY:
            return False
    return True


def gather(
    targets: Sequence[str],
    *,
    session_id: str,
    env: Mapping[str, str],
    stop: Iterable[str],
    state_root: str | os.PathLike[str] | None = None,
    messages: Sequence[object] | None = None,
) -> EvidenceSet:
    """Extract the session tree's calls and map each target to covering calls.

    ``messages`` is the CURRENT session's in-memory history; when given, its
    calls come from there and its disk transcript is not read. Without it
    (preview, which never reads the conversation) the disk copy stands in.
    """
    result = EvidenceSet()
    if not session_id or not targets:
        return result
    try:
        from local_operator.paths import config_dir

        root = Path(state_root) if state_root is not None else config_dir()
        stop = list(stop)
        # A transcript last written before the OLDEST candidate directory was
        # born cannot hold the call that created it: skip reading it.
        births = [b for t in targets for a in _anchors(t, stop) if (b := _born(a)) is not None]
        not_before = min(births) if births else 0.0
        for sid in session_tree(root, session_id):
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sid):
                continue
            if sid == session_id and messages is not None:
                result.calls.extend(extract_live(messages, session_id))
                continue
            path = root / "sessions" / sid / TRANSCRIPT_FILENAME
            result.calls.extend(extract(path, sid, session_id, not_before=not_before))
        writes = [c for c in result.calls if c.tool == "write"]
        by_label: dict[str, str] = {}
        for target in targets:
            # Prefer a call already listed for another target, so a batch of
            # directories made by one call shares one evidence id.
            ordered = sorted(result.calls, key=lambda c: c.label not in by_label)
            for ev in ordered:
                if ev.label not in by_label and len(result.listed) >= _MAX_LISTED:
                    continue
                if not covers(ev, target, writes=writes, env=env, stop=stop):
                    continue
                eid = by_label.get(ev.label)
                if eid is None:
                    eid = f"e{len(result.listed) + 1}"
                    by_label[ev.label] = eid
                    result.listed[eid] = ev
                result.covers[target] = [eid]
                break
            else:
                result.covers[target] = []
    except Exception:  # noqa: BLE001 — no evidence is the safe reading
        logger.warning("guardrail evidence extraction failed", exc_info=True)
    return result


def verify(
    cited: Sequence[str],
    targets: Sequence[str],
    evidence: EvidenceSet,
    *,
    env: Mapping[str, str],
    stop: Iterable[str],
) -> dict[str, str]:
    """``target -> covering call label`` for targets a CITED id re-verifies.

    Re-runs :func:`covers` rather than trusting ``gather``'s map, so a target
    swapped between gather and the answer (a new inode, a new birth time)
    no longer verifies.
    """
    writes = [c for c in evidence.calls if c.tool == "write"]
    stop = list(stop)
    out: dict[str, str] = {}
    for target in targets:
        for eid in cited:
            ev = evidence.listed.get(eid)
            if ev is not None and covers(ev, target, writes=writes, env=env, stop=stop):
                out[target] = ev.label
                break
    return out
