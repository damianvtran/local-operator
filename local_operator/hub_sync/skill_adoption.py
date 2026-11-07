"""Bounded raw-folder adoption into the strict Hub document contract.

Snapshots retain original layout and bytes before proposal/model work. Text root
files relocate into references; non-document assets stay local, explicitly listed.
Unknown ancestry uses the shared merge engine; conflicts never imply consent to
prefer the Hub. The service owns receipt authorization and storage owns CAS/commit.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import posixpath
import re
import stat
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from markdown_it import MarkdownIt
from markdown_it.rules_inline.backticks import backtick
from markdown_it.rules_inline.state_inline import StateInline

from local_operator.hub_sync import segment as seg
from local_operator.hub_sync.merge import (
    FieldInput,
    MergeOptions,
    MergeResult,
    Resolver,
    merge_field,
)
from local_operator.hub_sync.skill_document import (
    MAX_DESCRIPTION_CHARS,
    MAX_FILE_BYTES,
    MAX_FILES,
    SkillDocument,
    SkillFile,
    SkillValidationError,
    parse_frontmatter,
    validate_name,
    validate_path,
)

#: How the proposal resolves the collision. ``equal`` and ``merge`` come from the
#: content; ``keep-local``/``take-hub`` exist only as an explicit operator choice.
Strategy = Literal["equal", "merge", "keep-local", "take-hub"]
#: What a preview tells the caller to do next.
ProposalState = Literal["equal", "ready", "needs-choice"]

#: Bounds for reading the occupying folder. Deliberately ABOVE the manifest's own
#: bounds (64 files / 200 KiB): an oversized or binary file is not a reason to
#: refuse the adoption, it is a file to snapshot and list as set aside. The walk
#: is still bounded so a pathological tree cannot stall the request.
MAX_TREE_ENTRIES = 2048
MAX_TREE_BYTES = 64 * 1024 * 1024

#: A root-level text file is the only kind the relocation rule moves: the hub
#: manifest permits only SKILL.md at the root, and a local skill that keeps its
#: rubric beside SKILL.md is an ordinary authoring layout, not an error.
_RELOCATABLE_SUFFIXES = (".md", ".markdown", ".txt", ".yaml", ".yml", ".json", ".csv")


class AdoptionError(ValueError):
    """A named refusal while reading or proposing; ``code`` is the wire category."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


# -- reading the occupying folder ------------------------------------------------------


@dataclass(frozen=True)
class TreeEntry:
    """One file of the occupying folder, exactly as it is on disk."""

    path: str
    data: bytes
    mode: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def text(self) -> str | None:
        """UTF-8 text, or ``None`` for bytes no hub document can carry."""

        if b"\x00" in self.data:
            return None
        try:
            return self.data.decode("utf-8", errors="strict")
        except UnicodeError:
            return None


@dataclass(frozen=True)
class LocalTree:
    """The complete occupying folder: every file, binary and hidden included."""

    root: Path
    entries: tuple[TreeEntry, ...]

    @property
    def tree_hash(self) -> str:
        """A content hash over paths, bytes and executable bits, order-independent.

        This is what a preview receipt is bound to: an edit, an added or removed
        file, or a chmod between preview and apply refuses the apply.
        """

        digest = hashlib.sha256()
        for entry in sorted(self.entries, key=lambda e: e.path):
            digest.update(entry.path.encode("utf-8", "surrogatepass"))
            digest.update(b"\0")
            # CAS includes all permission bits: a chmod is a local edit too.
            digest.update(str(entry.mode).encode("ascii"))
            digest.update(entry.sha256.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {"path": e.path, "sha256": e.sha256, "bytes": len(e.data), "mode": e.mode}
            for e in sorted(self.entries, key=lambda e: e.path)
        ]


def read_tree(root: Path) -> LocalTree:
    """Read bounded original bytes without following links, including during races."""
    root = Path(root)
    entries: list[TreeEntry] = []
    total = seen = 0

    def walk(fd: int, prefix: str = "") -> None:
        nonlocal total, seen
        for name in sorted(os.listdir(fd)):
            seen += 1
            if seen > MAX_TREE_ENTRIES:
                raise AdoptionError("skill-too-large", "local tree has too many entries")
            relative = prefix + name
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    walk(child, relative + "/")
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
                with os.fdopen(child, "rb") as stream:
                    actual = os.fstat(stream.fileno())
                    if not stat.S_ISREG(actual.st_mode):
                        raise AdoptionError("skill-validation", "local file changed type")
                    data = stream.read(MAX_TREE_BYTES - total + 1)
                    total += len(data)
                    if total > MAX_TREE_BYTES:
                        raise AdoptionError("skill-too-large", "local tree exceeds byte limit")
                    entries.append(TreeEntry(relative, data, stat.S_IMODE(actual.st_mode)))
            else:
                raise AdoptionError(
                    "skill-validation",
                    "local tree contains a link or special file",
                )

    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            walk(fd)
        finally:
            os.close(fd)
    except OSError as error:
        raise AdoptionError("skill-validation", "local tree is unsafe or unreadable") from error
    return LocalTree(root, tuple(sorted(entries, key=lambda item: item.path)))


def durable_snapshot(config_dir: Path, alias: str, tree: LocalTree) -> dict[str, Any]:
    """Persist ORIGINAL layout before any resolver call; never reuse a mutable copy."""
    from local_operator.config_writer import config_write_lock
    from local_operator.hub_sync.transactions import (
        _sync_ancestors,
        _sync_directory,
        atomic_json,
    )

    validate_name(alias)
    config_dir = Path(config_dir).resolve()
    with config_write_lock(config_dir):
        if (
            tree.root != config_dir / "skills" / alias
            or read_tree(tree.root).tree_hash != tree.tree_hash
        ):
            raise AdoptionError("local-concurrent-edit", "local tree changed before snapshot")
        folder = config_dir / "hub" / "backups" / f"skill-adopt-{alias}-{uuid.uuid4().hex}"
        for parent in (config_dir / "hub", folder.parent, folder):
            if parent.is_symlink():
                raise AdoptionError("skill-validation", "snapshot path is unsafe")
            parent.mkdir(mode=0o700, exist_ok=True)
        content = folder / "content"
        content.mkdir(mode=0o700)
        for entry in tree.entries:
            target = content / entry.path
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(entry.data)
                stream.flush()
                os.fchmod(stream.fileno(), entry.mode)
                os.fsync(stream.fileno())
        for directory, _, _ in os.walk(content, topdown=False):
            _sync_directory(Path(directory))
        snapshot = {"path": str(folder), "tree_hash": tree.tree_hash, "manifest": tree.manifest()}
        atomic_json(folder / "manifest.json", snapshot)
        _sync_ancestors(folder, config_dir / "hub")
        if (
            read_tree(content).tree_hash != tree.tree_hash
            or read_tree(tree.root).tree_hash != tree.tree_hash
        ):
            raise AdoptionError("local-concurrent-edit", "local tree changed during snapshot")
        return snapshot


# -- the proposal ------------------------------------------------------------------------


@dataclass(frozen=True)
class FilePlan:
    """What the proposal does with ONE path of the result (or of the local folder).

    ``action`` vocabulary, stable on the wire:

    * ``same`` -- both sides agree (after normalization); the hub bytes are used.
    * ``hub-only`` / ``local-only`` -- one side has the file; it is kept.
    * ``relocated`` -- a local root file whose content lands at ``path`` (the
      hub's ``references/`` spelling, or ``references/<name>`` when the hub has
      none); ``local_path`` names where it was.
    * ``merged`` -- both sides differ; the per-file merge combined them.
    * ``needs-choice`` -- the merge could not combine them without a model or the
      validator refused the model's text; nothing is proposed for this path.
    * ``carried`` -- a recorded local-only asset preserved beside the document.
    * ``set-aside`` -- omitted only by an explicit take-hub choice; backed up.
    * ``not-installed`` -- Hub-only text excluded by an explicit keep-local choice.
    """

    path: str
    action: str
    local_path: str | None = None
    local: str | None = None
    remote: str | None = None
    proposed: str | None = None
    reason: str = ""
    result: MergeResult | None = None

    def to_json(self, *, include_text: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": self.path,
            "action": self.action,
            "local_path": self.local_path,
            "reason": self.reason,
        }
        if include_text:
            out["proposed"] = self.proposed
            out["diff_vs_local"] = _diff(self.local, self.proposed, "local", "proposed", self.path)
            out["diff_vs_hub"] = _diff(self.remote, self.proposed, "hub", "proposed", self.path)
        if self.result is not None:
            out["kept"] = _kept(self.result)
            out["engine"] = {
                "mode": self.result.engine.mode,
                "model": self.result.engine.model,
                "failure_class": self.result.engine.failure_class,
            }
        return out


@dataclass(frozen=True)
class AdoptionProposal:
    """The reviewable answer to "what would adopting this folder write?"."""

    alias: str
    state: ProposalState
    strategy: Strategy | None
    local_tree_hash: str
    local_manifest: tuple[Mapping[str, Any], ...]
    files: tuple[FilePlan, ...]
    document: SkillDocument | None
    description_source: str
    model: str | None = None
    failure_class: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def set_aside(self) -> tuple[FilePlan, ...]:
        return tuple(f for f in self.files if f.action == "set-aside")

    @property
    def unresolved(self) -> tuple[FilePlan, ...]:
        return tuple(f for f in self.files if f.action == "needs-choice")

    def to_json(self, *, include_text: bool = True) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "state": self.state,
            "strategy": self.strategy,
            "local_tree_hash": self.local_tree_hash,
            "proposed_fingerprint": (
                self.document.fingerprint if self.document is not None else None
            ),
            "files": [f.to_json(include_text=include_text) for f in self.files],
            "set_aside": [f.local_path or f.path for f in self.set_aside],
            "unresolved": [f.path for f in self.unresolved],
            "description_source": self.description_source,
            "model": self.model,
            "failure_class": self.failure_class,
            "warnings": list(self.warnings),
            "local_files": [dict(row) for row in self.local_manifest],
        }


def _diff(before: str | None, after: str | None, a: str, b: str, path: str) -> str:
    if before == after:
        return ""
    lines = difflib.unified_diff(
        (before or "").splitlines(keepends=True),
        (after or "").splitlines(keepends=True),
        fromfile=f"{a}/{path}" if before is not None else "/dev/null",
        tofile=f"{b}/{path}" if after is not None else "/dev/null",
    )
    text = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    return text


def _kept(result: MergeResult) -> dict[str, list[str]]:
    """Which sections came from where -- the "what was kept from each side" line.

    Region names are the headings the engine reported; the preamble is ``(top)``.
    """

    out: dict[str, list[str]] = {"local": [], "hub": [], "combined": [], "unresolved": []}
    bucket = {
        "kept-local": "local",
        "taken-remote": "hub",
        "combined": "combined",
        "unresolved": "unresolved",
    }
    for region in result.regions:
        key = bucket.get(region.provenance)
        if key is None:
            continue
        out[key].append(region.name or "(top)")
    return out


def _prose_path(path: str, mode: int = 0) -> bool:
    """A relocation must never promote an original data/executable payload to prose."""
    return (
        path.lower().endswith((".md", ".markdown"))
        and not path.startswith(("scripts/", "assets/"))
        and not mode & 0o111
    )


def _equivalent(source: str, local: str, target: str, remote: str, mode: int = 0) -> bool:
    if not (_prose_path(source, mode) and _prose_path(target)):
        return local == remote
    return _norm(local) == _norm(remote)


@dataclass(frozen=True)
class _MarkdownSource:
    exact: tuple[tuple[int, int], ...]
    inline: tuple[tuple[int, int, int, int], ...]


def _markdown_source(text: str) -> _MarkdownSource:
    """Default EXACT; CommonMark maps only grant supported top-level prose.

    Container prefixes, HTML, code and unknown physical slices never come from
    reconstructed token content. Inline offsets are usable only when the installed
    parser's actual source equals the raw paragraph, apart from its terminal EOL.
    """
    md = MarkdownIt("commonmark").disable("inline")
    blocks = md.parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    exact: list[tuple[int, int]] = []
    inline: list[tuple[int, int, int, int]] = []
    covered = [False] * len(lines)
    captured: list[tuple[int, int, int, int]] = []
    active: StateInline | None = None
    ambiguous_inline = False

    def protect(first: int, last: int) -> None:
        # CommonMark maps omit trailing blank lines from indented/container blocks.
        # Those raw gaps belong to the exact block, not to a prose normalizer.
        while last < len(lines) and not covered[last] and not lines[last].strip():
            covered[last] = True
            last += 1
        exact.append((offsets[first], offsets[last]))

    def observe_backticks(state: StateInline, silent: bool) -> bool:
        nonlocal ambiguous_inline
        start, count = state.pos, len(state.tokens)
        handled = backtick(state, silent)
        # The installed rule returns True for unmatched delimiters too. Only a
        # newly emitted code_inline proves that this is a raw code slice.
        if not silent and any(token.type == "code_inline" for token in state.tokens[count:]):
            if state is not active:
                ambiguous_inline = True
                return handled
            width = 0
            while start + width < state.pos and state.src[start + width] == "`":
                width += 1
            captured.append((start, state.pos, start + width, state.pos - width))
        return handled

    md.inline.ruler.at("backticks", observe_backticks)
    for index, block in enumerate(blocks):
        if block.level != 0 or block.map is None:
            continue
        first, last = block.map
        if not (0 <= first < last <= len(lines)):
            return _MarkdownSource(((0, len(text)),) if text else (), ())
        if any(covered[first:last]):
            continue
        covered[first:last] = [True] * (last - first)
        start, end = offsets[first], offsets[last]
        raw = text[start:end]
        candidate = blocks[index + 1] if index + 1 < len(blocks) else None
        if (
            block.type != "paragraph_open"
            or candidate is None
            or candidate.type != "inline"
            or candidate.map != block.map
        ):
            protect(first, last)
            continue
        # token.content is merely a parser-input candidate, NEVER byte authority.
        # No substring search can recover offsets after indentation/CRLF stripping.
        state = StateInline(candidate.content, md, {}, [])
        body = raw[:-2] if raw.endswith("\r\n") else raw[:-1] if raw.endswith("\n") else raw
        if state.src != body:
            protect(first, last)
            continue
        captured.clear()
        active, ambiguous_inline = state, False
        md.inline.tokenize(state)
        unsupported = any(
            token.type
            not in {
                "text",
                "softbreak",
                "code_inline",
                "em_open",
                "em_close",
                "strong_open",
                "strong_close",
                "link_open",
                "link_close",
            }
            for token in state.tokens
        )
        # Unmatched or escaped backticks are not positively identified code/prose.
        unproved_ticks = any(
            char == "`" and not any(a <= pos < b for a, b, _, _ in captured)
            for pos, char in enumerate(body)
        )
        if unsupported or unproved_ticks or ambiguous_inline:
            protect(first, last)
            continue
        for a, b, payload_start, payload_end in captured:
            exact.append((start + a, start + b))
            inline.append((start + a, start + b, start + payload_start, start + payload_end))
    index = 0
    while index < len(lines):
        if covered[index]:
            index += 1
            continue
        last = index + 1
        while last < len(lines) and not covered[last]:
            last += 1
        # A whole unmapped gap is unknown unless it consists ONLY of plain LF
        # separators. Definitions/unknown containers keep all their physical bytes.
        if any(line != "\n" for line in lines[index:last]):
            exact.append((offsets[index], offsets[last]))
        index = last
    return _MarkdownSource(tuple(sorted(exact)), tuple(sorted(inline)))


def _code_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Raw exact islands, including every unproven/container slice, not just code."""
    return _markdown_source(text).exact


def _opaque_separator(raw: str) -> str:
    return "\n\n" if raw.endswith(("\n\n", "\r\n\r\n")) else "\n" if raw.endswith("\n") else ""


def _shield_code(text: str) -> tuple[str, dict[str, str]]:
    payloads: dict[str, str] = {}
    parts: list[str] = []
    cursor = 0
    for start, end in _code_spans(text):
        raw = text[start:end]
        token = "LOP_PROTECTED_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if token in text:
            raise AdoptionError("adoption-needs-choice", "protected code token is ambiguous")
        payloads[token] = raw
        # Keep block boundaries visible to the segmenter, without exposing code.
        parts.extend((text[cursor:start], token + _opaque_separator(raw)))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), payloads


def _restore_code(text: str, payloads: Mapping[str, str]) -> str:
    for token, raw in payloads.items():
        text = text.replace(token + _opaque_separator(raw), raw)
        text = text.replace(token, raw)  # segmenters may strip the final prose newline
    return text


def _norm(text: str) -> str:
    """Normalize only source-proven prose; every other physical slice is opaque."""

    text, _ = _shield_code(text)
    return "\n".join(" ".join(paragraph.split()) for paragraph in seg.norm(text).split("\n\n"))


def _rewrite_relocated_refs(
    text: str,
    moves: Mapping[str, str],
    *,
    source_path: str = "SKILL.md",
    destination_path: str | None = None,
    known_paths: set[str] | None = None,
    on_ambiguous: Literal["refuse", "verbatim"] = "refuse",
) -> str:
    """Resolve within the captured source tree, then relativize at the destination.

    Basename substitution cannot distinguish ../requirements.md from a different
    file beside the source. Unknown/traversing references to moved names refuse
    rather than leave a ready proposal with a silently broken link.

    ONE case is not ambiguous and is accepted verbatim: a file that does not move
    itself (``source_path == destination_path``) naming a path that resolves to a
    relocation DESTINATION. The operator's real municipal tree is the witness:
    ``references/record-contract.md`` says ``rubric.md`` while rubric.md sits at
    the skill root. Dangling locally, that sibling name becomes correct the moment
    relocation lands rubric.md at ``references/rubric.md``, so refusing it was a
    dead end on exactly the case adoption exists for. Anything else that merely
    shares a moved basename still refuses under merge.

    ``on_ambiguous="verbatim"`` is for callers that must never dead-end on a
    rewrite question they do not need answered. take-hub: the local text it
    rewrites is never installed (shared paths install the Hub's text, local-only
    and relocated files are set aside into the snapshot). keep-local projection:
    the folder on disk is untouched, but the PROJECTED document (fingerprinted
    and pushed) keeps an ambiguous reference exactly as authored instead of
    refusing -- a link that was already broken locally stays broken in the
    projection, unrewritten and unguessed, rather than bricking reads of the
    managed skill. Only cases that previously raised change output; every
    resolvable rewrite is unchanged.
    """
    if not moves:
        return text
    destination_path = destination_path or moves.get(source_path, source_path)
    known = known_paths if known_paths is not None else set(moves)
    stationary = source_path == destination_path
    destinations = set(moves.values())

    def ambiguous(reason: str) -> None:
        if on_ambiguous == "refuse":
            raise AdoptionError("adoption-needs-choice", reason)

    pattern = re.compile(r"(?<![\w/.:~-])(?:[\w.@~+-]+/)*[\w@~+-]+\.[\w.~+-]*[\w~+-]")

    def rewrite(match: re.Match[str]) -> str:
        token = match.group()
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), token))
        if resolved == ".." or resolved.startswith("../"):
            ambiguous("reference escapes captured skill root")
            return token
        if resolved not in known:
            if stationary and resolved in destinations:
                return token  # valid once the relocation lands (see docstring)
            if posixpath.basename(token) in {posixpath.basename(old) for old in moves}:
                ambiguous("ambiguous reference to relocated file")
            return token
        target = moves.get(resolved, resolved)
        relative = posixpath.relpath(target, posixpath.dirname(destination_path) or ".")
        if token.startswith("./") and not relative.startswith("."):
            relative = "./" + relative
        return relative

    if _prose_path(source_path):
        replacements: list[tuple[int, int, str]] = []
        for _, _, start, end in _markdown_source(text).inline:
            literal = text[start:end]
            # An entire raw literal path is the only allowed code rewrite. Code
            # expressions, escapes, URIs, whitespace and absolute paths stay exact.
            if not re.fullmatch(r"(?:[A-Za-z0-9_.@+-]+/)*[A-Za-z0-9_.@+-]+", literal):
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), literal))
            if resolved in moves and resolved in known:
                relative = posixpath.relpath(
                    moves[resolved], posixpath.dirname(destination_path) or "."
                )
                if literal.startswith("./") and not relative.startswith("."):
                    relative = "./" + relative
                replacements.append((start, end, relative))
            elif stationary and resolved in destinations and resolved not in known:
                # ``resolved not in known`` is implied today (a destination that
                # collides with a local file drops the move upstream) and kept as
                # an explicit guard: an existing local file is never shadowed.
                continue  # valid once the relocation lands (see docstring)
            elif posixpath.basename(literal) in {posixpath.basename(old) for old in moves}:
                if resolved.startswith("../") or resolved not in known:
                    ambiguous("ambiguous inline filepath")
        for start, end, relative in reversed(replacements):
            text = text[:start] + relative + text[end:]
    masked, payloads = _shield_code(text)
    return _restore_code(pattern.sub(rewrite, masked), payloads)


def _hub_text_files(remote: SkillDocument) -> dict[str, str]:
    return {f.path: f.content for f in remote.files}


def _relocation_target(
    local_path: str,
    entry_text: str,
    remote_files: Mapping[str, str],
    taken: set[str],
    *,
    mode: int = 0,
) -> str | None:
    """Where a root-level local file lands in a hub-shaped tree, or ``None``.

    Preference order: the hub path with the SAME basename under ``references/``
    (the municipal case: ``rubric.md`` -> ``references/rubric.md``); else a hub
    path whose normalized content equals this file; else a fresh
    ``references/<basename>`` when that path is free on both sides.
    """

    name = local_path.rsplit("/", 1)[-1]
    if "/" in local_path or local_path == "SKILL.md":
        return None
    if not name.lower().endswith(_RELOCATABLE_SUFFIXES) or name.startswith("."):
        return None
    for prefix in ("references/", "assets/"):
        candidate = prefix + name
        if candidate in remote_files and candidate not in taken:
            return candidate
    for path, content in remote_files.items():
        if (
            path not in taken
            and path != "SKILL.md"
            and _prose_path(local_path, mode)
            and _prose_path(path)
            and _equivalent(local_path, entry_text, path, content)
        ):
            return path
    candidate = "references/" + name
    try:
        validate_path(candidate)
    except SkillValidationError:
        return None
    if candidate in taken:
        return None
    return candidate


def _carriable(path: str, entry: TreeEntry) -> tuple[str | None, str]:
    """``(text, "")`` when the hub manifest can carry this file, else ``(None, reason)``."""

    try:
        validate_path(path)
    except SkillValidationError as error:
        return None, f"path not carried by hub skills ({error})"
    if len(entry.data) > MAX_FILE_BYTES:
        return None, "larger than the 200 KiB per-file limit"
    text = entry.text()
    if text is None:
        return None, "binary content"
    return text, ""


def _merge_text(
    path: str,
    local: str,
    remote: str,
    *,
    resolver: Resolver | None,
    allow_llm: bool,
) -> MergeResult:
    """Two-way merge of one file with NO common ancestor (there is no baseline)."""

    local_codes = tuple(local[a:b] for a, b in _code_spans(local))
    remote_codes = tuple(remote[a:b] for a, b in _code_spans(remote))
    if local_codes != remote_codes:
        return MergeResult(path, "refused", local, refusal="protected code differs")
    masked_local, payloads = _shield_code(local)
    masked_remote, _ = _shield_code(remote)
    result = merge_field(
        FieldInput(
            field=path, kind="markdown", base=None, local=masked_local, remote=masked_remote
        ),
        MergeOptions(
            allow_llm=allow_llm,
            acknowledge_unknown_baseline=True,
            resolver=resolver if allow_llm else None,
        ),
    )
    if result.outcome not in ("merged", "unchanged"):
        return replace(result, merged=local)
    proposed = str(result.merged)
    if any(proposed.count(token) != masked_local.count(token) for token in payloads):
        return replace(result, outcome="refused", merged=local, refusal="protected code missing")
    restored = _restore_code(proposed, payloads)
    if tuple(restored[a:b] for a, b in _code_spans(restored)) != local_codes:
        return replace(result, outcome="refused", merged=local, refusal="protected code changed")
    return replace(result, merged=restored)


def build_proposal(
    alias: str,
    tree: LocalTree,
    remote: SkillDocument,
    *,
    resolver: Resolver | None,
    allow_llm: bool = True,
    strategy: Literal["merge", "keep-local", "take-hub"] = "merge",
) -> AdoptionProposal:
    """Compute (never write) what adopting ``tree`` under ``remote`` would install.

    ``remote`` is the hub's CANONICAL document (its own name); the result is
    rendered under the hub name too and the caller re-renders the alias, exactly
    as ``install_skill`` does for a fresh pull. ``strategy`` other than ``merge``
    is the operator's explicit choice after a ``needs-choice`` preview.
    """

    remote_files = _hub_text_files(remote)
    if strategy == "keep-local":
        # The receipt must describe the tree apply actually keeps, not the
        # hypothetical union with Hub-only files which will never be written.
        extras = []
        plans = []
        for entry in tree.entries:
            text, _ = _carriable(entry.path, entry)
            relocatable = (
                entry.text() is not None
                and len(entry.data) <= MAX_FILE_BYTES
                and _relocation_target(entry.path, entry.text() or "", {}, set()) is not None
            )
            extra = text is None and not relocatable
            if extra:
                extras.append(entry.path)
            plans.append(
                FilePlan(
                    entry.path,
                    "carried" if extra else "kept-local",
                    local_path=entry.path,
                    local=entry.text(),
                    remote=remote_files.get(entry.path),
                    proposed=entry.text(),
                )
            )
        local_paths = {entry.path for entry in tree.entries}
        plans.extend(
            FilePlan(
                path,
                "not-installed",
                remote=text,
                proposed=None,
                reason="keep-local leaves the original folder unchanged",
            )
            for path, text in remote_files.items()
            if path not in local_paths
        )
        projected = local_projection(tree, carried_paths=extras).with_name(remote.name)
        return AdoptionProposal(
            alias,
            "ready",
            "keep-local",
            tree.tree_hash,
            tuple(tree.manifest()),
            tuple(plans),
            projected,
            "keep-local",
        )
    local_by_path = {e.path: e for e in tree.entries}

    # 1. Classify every local file: carriable text, relocatable root text, or
    #    set aside. Relocation is decided BEFORE any comparison so the reference
    #    rewrite below can see the full move map.
    carried: dict[str, tuple[str, str]] = {}  # target path -> (local path, text)
    moves: dict[str, str] = {}
    set_aside: list[FilePlan] = []
    for entry in tree.entries:
        text, reason = _carriable(entry.path, entry)
        if text is not None:
            carried[entry.path] = (entry.path, text)
            continue
        raw = entry.text()
        if raw is not None and len(entry.data) <= MAX_FILE_BYTES:
            target = _relocation_target(
                entry.path,
                raw,
                remote_files,
                set(carried) | set(moves.values()) | set(local_by_path),
                mode=entry.mode,
            )
            if target is not None and target not in local_by_path:
                moves[entry.path] = target
                carried[target] = (entry.path, raw)
                continue
            local_collision = target in local_by_path or (
                "/" not in entry.path and "references/" + entry.path in local_by_path
            )
            if local_collision and strategy == "merge":
                set_aside.append(
                    FilePlan(
                        entry.path,
                        "needs-choice",
                        local_path=entry.path,
                        reason="relocation collides with another local file",
                    )
                )
                continue
        action = "set-aside" if strategy == "take-hub" else "carried"
        if strategy == "merge" and entry.path in remote_files:
            action = "needs-choice"
        set_aside.append(
            FilePlan(entry.path, action, local_path=entry.path, reason=reason or "not carriable")
        )

    if "SKILL.md" not in carried:
        raise AdoptionError(
            "skill-validation", "the local skill folder has no readable SKILL.md to adopt"
        )

    # 2. Rewrite references to relocated files inside every carried LOCAL text.
    #    Deterministic, never a model call, and recorded so the preview can say so.
    if moves:
        carried = {
            target: (
                src,
                (
                    _rewrite_relocated_refs(
                        text,
                        moves,
                        source_path=src,
                        destination_path=target,
                        known_paths=set(local_by_path),
                        # take-hub installs the Hub text; local text is display only.
                        on_ambiguous="verbatim" if strategy == "take-hub" else "refuse",
                    )
                    if _prose_path(src, local_by_path[src].mode) and _prose_path(target)
                    else text
                ),
            )
            for target, (src, text) in carried.items()
        }

    # Description routing is authored guidance too; differing values need a
    # validated merge or an explicit choice, never an implicit Hub preference.
    local_skill = carried["SKILL.md"][1]
    try:
        local_meta = parse_frontmatter(local_skill)
    except SkillValidationError:
        local_meta = {}
    local_description = str(local_meta.get("description") or "")
    if strategy in ("keep-local", "take-hub"):
        description = local_description if strategy == "keep-local" else remote.description
        description_source = strategy
    else:
        description, description_source = _merge_description(
            local_description, remote.description, resolver=resolver, allow_llm=allow_llm
        )

    # The body uses the shared merger. Metadata is retained by key or refused
    # on conflict, since flattening YAML through a prose model loses structure.
    plans: list[FilePlan] = []
    model: str | None = None
    failure: str | None = None
    for path in sorted(set(carried) | set(remote_files)):
        local_text = carried[path][1] if path in carried else None
        source = carried[path][0] if path in carried else None
        remote_text = remote_files.get(path)
        relocated_from = source if source is not None and source != path else None
        if path == "SKILL.md":
            assert local_text is not None and remote_text is not None
            if (
                strategy == "merge"
                and source
                and local_by_path[source].mode & 0o111
                and local_text != remote_text
            ):
                plan = FilePlan(
                    path,
                    "needs-choice",
                    local=local_text,
                    remote=remote_text,
                    reason="executable source differs",
                )
            else:
                plan = _plan_skill_md(
                    local_text,
                    remote_text,
                    resolver=resolver,
                    allow_llm=allow_llm,
                    strategy=strategy,
                )
        elif local_text is None:
            plan = FilePlan(path, "hub-only", remote=remote_text, proposed=remote_text)
        elif remote_text is None:
            action = "relocated" if relocated_from else "local-only"
            plan = FilePlan(
                path,
                action,
                local_path=relocated_from,
                local=local_text,
                proposed=None if strategy == "take-hub" else local_text,
                reason=(
                    "kept from the local copy"
                    if strategy != "take-hub"
                    else "not in the hub version"
                ),
            )
            if strategy == "take-hub":
                # Taking the hub version does not install a local-only file, so
                # it is LISTED (and stays in the snapshot), never silently gone.
                plan = FilePlan(
                    path,
                    "set-aside",
                    local_path=source,
                    local=local_text,
                    reason="not in the hub version; kept in the snapshot",
                )
        elif _equivalent(
            source or path,
            local_text,
            path,
            remote_text,
            local_by_path[source].mode if source else 0,
        ):
            plan = FilePlan(
                path,
                "relocated" if relocated_from else "same",
                local_path=relocated_from,
                local=local_text,
                remote=remote_text,
                proposed=remote_text,
            )
        elif strategy == "keep-local":
            plan = FilePlan(
                path,
                "kept-local",
                local_path=relocated_from,
                local=local_text,
                remote=remote_text,
                proposed=local_text,
            )
        elif strategy == "take-hub":
            plan = FilePlan(
                path,
                "taken-hub",
                local_path=relocated_from,
                local=local_text,
                remote=remote_text,
                proposed=remote_text,
            )
        elif not _prose_path(
            source or path, local_by_path[source].mode if source else 0
        ) or not _prose_path(path):
            # Assets are payloads, even when UTF-8. The resolver must never see
            # or invent their bytes; differing versions require an explicit choice.
            plan = FilePlan(
                path,
                "needs-choice",
                local=local_text,
                remote=remote_text,
                reason="asset contents differ",
            )
        elif not allow_llm:
            plan = FilePlan(
                path,
                "needs-choice",
                local=local_text,
                remote=remote_text,
                reason="model merge disabled",
            )
        else:
            result = _merge_text(
                path,
                local_text,
                remote_text,
                resolver=resolver,
                allow_llm=allow_llm,
            )
            model = model or result.engine.model
            failure = failure or result.engine.failure_class
            if result.outcome in ("merged", "unchanged"):
                plan = FilePlan(
                    path,
                    "merged",
                    local_path=relocated_from,
                    local=local_text,
                    remote=remote_text,
                    proposed=str(result.merged),
                    result=result,
                )
            else:
                plan = FilePlan(
                    path,
                    "needs-choice",
                    local_path=relocated_from,
                    local=local_text,
                    remote=remote_text,
                    reason=result.refusal or (result.engine.failure_class or "conflict"),
                    result=result,
                )
        if plan.result is not None:
            model = model or plan.result.engine.model
            failure = failure or plan.result.engine.failure_class
        if source is not None and plan.local is not None:
            # The frozen rewrite is part of the proposal, not the original tree.
            # Review diffs must expose inline-path changes against captured bytes.
            plan = replace(plan, local=local_by_path[source].text())
        plans.append(plan)
    plans.extend(set_aside)

    unresolved = [p for p in plans if p.action == "needs-choice"]
    if description is None:
        unresolved.append(FilePlan("description", "needs-choice", reason="descriptions differ"))
        plans.append(unresolved[-1])

    if unresolved:
        return AdoptionProposal(
            alias,
            "needs-choice",
            None,
            tree.tree_hash,
            tuple(tree.manifest()),
            tuple(plans),
            None,
            description_source,
            model=model,
            failure_class=failure,
        )

    files = tuple(
        SkillFile(p.path, p.proposed)
        for p in plans
        if p.proposed is not None and p.action != "set-aside"
    )
    warnings: list[str] = []
    if len(files) > MAX_FILES:
        raise AdoptionError("skill-too-large", "the consolidated skill exceeds 64 files")
    assert description is not None
    # Align the frontmatter with the chosen name/description by replacing only
    # the scalar spans, and only when the parsed value actually differs. A full
    # ``yaml.safe_dump`` re-render folds long descriptions at width 80 and
    # re-wraps Hub-authored folding, so a take-hub install no longer matched the
    # Hub bytes and its link classified local-only (#3423). The fingerprint is
    # over exact bytes, so the fix keeps the bytes rather than normalizing both
    # sides before comparison.
    skill_meta = parse_frontmatter(next(f.content for f in files if f.path == "SKILL.md"))
    document = SkillDocument(skill_meta["name"], skill_meta["description"], files)
    if document.description != description:
        document = document.with_description(description)
    if document.name != remote.name:
        document = document.with_name(remote.name)

    # Equal = every carried path matched the hub after normalization, nothing set
    # aside, nothing relocated into a NEW path, description unchanged. Then the
    # adoption links the folder WITHOUT writing content.
    remote_meta = parse_frontmatter(remote_files["SKILL.md"])
    local_meta = parse_frontmatter(local_skill)
    equal = (
        not set_aside
        and {k: v for k, v in local_meta.items() if k != "name"}
        == {k: v for k, v in remote_meta.items() if k != "name"}
        and description == remote.description
        and all(p.action in ("same", "relocated") and p.path in remote_files for p in plans)
        and all(p.path in carried for p in plans if p.action != "set-aside")
    )
    if strategy == "merge":
        chosen: Strategy = "equal" if equal else "merge"
    else:
        chosen = strategy
    return AdoptionProposal(
        alias,
        "equal" if equal else "ready",
        chosen,
        tree.tree_hash,
        tuple(tree.manifest()),
        tuple(plans),
        document,
        description_source,
        model=model,
        failure_class=failure,
        warnings=tuple(warnings),
    )


def _split_frontmatter(text: str) -> tuple[str, str]:
    """``(frontmatter block incl. fences, body)``; body is everything after it."""

    if not text.startswith("---"):
        return "", text
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            head = "".join(lines[: index + 1])
            return head, text[len(head) :]
    return "", text


def _plan_skill_md(
    local: str,
    remote: str,
    *,
    resolver: Resolver | None,
    allow_llm: bool,
    strategy: str,
) -> FilePlan:
    """Preserve metadata by key; prose merging cannot safely reconcile YAML conflicts."""

    import yaml

    remote_head, remote_body = _split_frontmatter(remote)
    _local_head, local_body = _split_frontmatter(local)
    local_meta, remote_meta = parse_frontmatter(local), parse_frontmatter(remote)
    extras = {key: value for key, value in local_meta.items() if key not in ("name", "description")}
    conflict = any(
        key in remote_meta and remote_meta[key] != value for key, value in extras.items()
    )
    if strategy == "merge" and conflict:
        return FilePlan(
            "SKILL.md",
            "needs-choice",
            local=local,
            remote=remote,
            reason="frontmatter differs",
        )
    meta = dict(remote_meta)
    if strategy != "take-hub":
        meta.update(extras)
        if strategy == "keep-local":
            meta["description"] = local_meta["description"]
    if meta != remote_meta:
        # Only a real metadata change (local extras, kept-local description) is
        # re-rendered; otherwise the Hub's frontmatter bytes are carried verbatim
        # so same/taken-hub/equal installs fingerprint-match the Hub (#3423).
        remote_head = "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n"
    if strategy == "merge" and not allow_llm and _norm(local_body) != _norm(remote_body):
        return FilePlan(
            "SKILL.md",
            "needs-choice",
            local=local,
            remote=remote,
            reason="model merge disabled",
        )
    if _norm(local_body) == _norm(remote_body):
        return FilePlan(
            "SKILL.md",
            "same",
            local=local,
            remote=remote,
            proposed=remote_head + remote_body,
        )
    if strategy == "keep-local":
        return FilePlan(
            "SKILL.md",
            "kept-local",
            local=local,
            remote=remote,
            proposed=remote_head + local_body,
        )
    if strategy == "take-hub":
        return FilePlan("SKILL.md", "taken-hub", local=local, remote=remote, proposed=remote)
    result = _merge_text(
        "SKILL.md",
        local_body,
        remote_body,
        resolver=resolver,
        allow_llm=allow_llm,
    )
    if result.outcome in ("merged", "unchanged"):
        body = str(result.merged)
        # The paragraph merger joins segments without the file's terminal
        # newline. When BOTH authored sides end with one, keep it: otherwise the
        # installed SKILL.md differs from both by a missing EOF newline and the
        # next push shows a spurious diff (QA round 1, Q1 on PR 3390).
        if local_body.endswith("\n") and remote_body.endswith("\n") and not body.endswith("\n"):
            body += "\n"
        lead = "\n" if remote_body.startswith("\n") else ""
        proposed = remote_head + lead + body
        return FilePlan(
            "SKILL.md",
            "merged",
            local=local,
            remote=remote,
            proposed=proposed,
            result=result,
        )
    return FilePlan(
        "SKILL.md",
        "needs-choice",
        local=local,
        remote=remote,
        reason=result.refusal or (result.engine.failure_class or "conflict"),
        result=result,
    )


def _merge_description(
    local: str, remote: str, *, resolver: Resolver | None, allow_llm: bool
) -> tuple[str | None, str]:
    """Merge routing guidance without silently preferring either authored side."""

    if not local.strip() or seg.collapse_ws(local) == seg.collapse_ws(remote):
        return remote, "hub"
    if not allow_llm or resolver is None:
        return None, "needs-choice"
    result = merge_field(
        FieldInput("description", "text", None, local, remote),
        MergeOptions(
            allow_llm=True,
            acknowledge_unknown_baseline=True,
            resolver=resolver,
            max_chars=MAX_DESCRIPTION_CHARS,
        ),
    )
    if result.outcome == "merged" and str(result.merged).strip():
        return seg.collapse_ws(str(result.merged)), "merged"
    return None, "needs-choice"


def rendered_document(proposal: AdoptionProposal, alias: str) -> SkillDocument:
    """The proposal's document rendered under the LOCAL alias (the bytes written)."""

    if proposal.document is None:
        raise AdoptionError("adoption-needs-choice", "this proposal has unresolved files")
    return proposal.document.with_name(alias)


def equal_tree(tree: LocalTree, document: SkillDocument) -> bool:
    """Whether the folder already IS ``document`` byte for byte (no write needed)."""

    wanted = {f.path: f.content.encode("utf-8") for f in document.files}
    have = {e.path: e.data for e in tree.entries}
    return wanted == have


def iter_text_paths(files: Iterable[FilePlan]) -> Sequence[str]:
    return [f.path for f in files if f.action != "set-aside"]


def local_projection(tree: LocalTree, *, carried_paths: Sequence[str] = ()) -> SkillDocument:
    """Project a specifically authorized raw tree, never weaken the Hub parser.

    Only receipt-recorded local assets are excluded. All other paths and metadata
    must survive the same strict document validation as a downloaded skill.
    """
    excluded = set(carried_paths)
    files: dict[str, str] = {}
    moves: dict[str, str] = {}
    original = {entry.path for entry in tree.entries}
    for entry in tree.entries:
        if entry.path in excluded:
            continue
        text, _ = _carriable(entry.path, entry)
        target = entry.path
        if text is None:
            text = entry.text()
            if text is None:
                raise AdoptionError("skill-validation", "unrecorded non-document local asset")
            target = _relocation_target(entry.path, text, {}, original | set(files))
            if target is None and "/" not in entry.path and not entry.path.startswith("."):
                candidate = "references/local-root-" + entry.path
                validate_path(candidate)
                if candidate not in original and candidate not in files:
                    target = candidate
            if target is None or len(entry.data) > MAX_FILE_BYTES:
                raise AdoptionError("skill-validation", "local path cannot be projected")
            moves[entry.path] = target
        if target in files:
            raise AdoptionError("adoption-needs-choice", "local relocation collides")
        files[target] = text
    inverse = {target: source for source, target in moves.items()}
    modes = {entry.path: entry.mode for entry in tree.entries}
    files = {
        path: (
            _rewrite_relocated_refs(
                text,
                moves,
                source_path=inverse.get(path, path),
                destination_path=path,
                known_paths=original,
                # The projected folder is kept byte for byte; never refuse a read.
                on_ambiguous="verbatim",
            )
            if (
                _prose_path(inverse.get(path, path), modes[inverse.get(path, path)])
                and _prose_path(path)
            )
            else text
        )
        for path, text in files.items()
    }
    meta = parse_frontmatter(files.get("SKILL.md", ""))
    return SkillDocument(
        meta["name"],
        meta["description"],
        tuple(SkillFile(p, t) for p, t in files.items()),
    )
