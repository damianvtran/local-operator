"""Bounded, lossless text skill manifests shared by publication and sync.

A folder is not a safe publication boundary: references and inert scripts can
carry precisely the private material omitted from its short description. V1
therefore validates and hashes EVERY transferable path and byte, refuses files
it cannot represent, and never silently turns a bundle into a partial skill.
The server-owned schema is deliberately a strict subset of ecosystem YAML;
unknown keys fail rather than disappear during a round trip.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

MAX_FILES = 64
MAX_FILE_BYTES = 200 * 1024
MAX_CONTENT_BYTES = 1024 * 1024
MAX_WIRE_BYTES = 2 * 1024 * 1024
MAX_PATH_BYTES = 240
MAX_PATH_SEGMENTS = 5
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 1024
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ALLOWED_KEYS = {
    "name",
    "description",
    "license",
    "compatibility",
    "allowed-tools",
    "metadata",
    "targets",
    "version",
    "enabled",
    "hide",
    "disable-model-invocation",
}


class SkillValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _invalid(message: str) -> SkillValidationError:
    # Diagnostics intentionally name the constraint, never private file bytes.
    return SkillValidationError("skill-validation", message)


def _large(message: str) -> SkillValidationError:
    return SkillValidationError("skill-too-large", message)


def _text_bytes(value: Any) -> bytes:
    if not isinstance(value, str):
        raise _invalid("skill content must be UTF-8 text")
    try:
        result = value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise _invalid("skill content must be UTF-8 text") from None
    if any(ord(c) < 32 and c not in "\n\r\t" or ord(c) == 127 for c in value):
        raise _invalid("binary or control content is unsupported in skill v1")
    return result


def validate_name(name: Any) -> str:
    if not isinstance(name, str) or len(name) > MAX_NAME_CHARS or not _NAME.fullmatch(name):
        raise _invalid("skill name must be lower kebab-case, between 1 and 64 characters")
    return name


def validate_path(path: Any) -> str:
    if not isinstance(path, str) or not path:
        raise _invalid("skill path must be a relative text path")
    raw = _text_bytes(path)
    parts = path.split("/")
    if (
        path.startswith("/")
        or "\\" in path
        or ":" in path
        or any(not part or part.startswith(".") for part in parts)
        or any(c.isspace() and c not in " " for c in path)
    ):
        raise _invalid("skill path is unsafe")
    # Format characters (Unicode category Cf, e.g. U+202E RIGHT-TO-LEFT OVERRIDE)
    # render invisibly while carrying bytes: they can smuggle a second spelling of
    # a path past the NFC/casefold collision rule (and past a reviewer's eyes).
    # Unpaired surrogates are already refused by ``_text_bytes`` (strict UTF-8).
    import unicodedata as _ud

    if any(_ud.category(c) == "Cf" for c in path):
        raise _invalid("skill path must not contain Unicode format characters")
    if len(raw) > MAX_PATH_BYTES or len(parts) > MAX_PATH_SEGMENTS:
        raise _large("skill path exceeds v1 bounds")
    if path != "SKILL.md" and (
        len(parts) < 2 or parts[0] not in ("references", "scripts", "assets")
    ):
        raise _invalid("skill files must be SKILL.md or references/scripts/assets text")
    return path


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader alone silently accepts duplicate keys, which loses authored bytes."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        mapping: dict[Hashable, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or not key.strip() or key in mapping:
                raise _invalid("skill frontmatter mapping keys must be unique nonblank strings")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _frontmatter_span(text: str) -> tuple[str, int, int]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---" or not text.startswith("---"):
        raise _invalid("SKILL.md requires strict YAML frontmatter")
    start = len(lines[0])
    for index, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end = sum(len(part) for part in lines[:index])
            return text[start:end], start, end
    raise _invalid("SKILL.md frontmatter is unterminated")


def _scalar(value: Any) -> bool:
    if isinstance(value, str) or type(value) is bool:
        return True
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # Python can parse integers that overflow the server's JSON numeric
        # representation. Refuse them by name, not an uncaught conversion error.
        return False


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def parse_frontmatter(text: str) -> dict[str, Any]:
    block, _, _ = _frontmatter_span(text)
    try:
        # Even safe YAML aliases create alternate byte/semantic projections.
        # V1 does not transfer anchors, aliases, tags or merge-key expansion.
        depth = 0
        for token in yaml.scan(block):
            if isinstance(
                token,
                (
                    yaml.tokens.FlowSequenceStartToken,
                    yaml.tokens.FlowMappingStartToken,
                    yaml.tokens.BlockSequenceStartToken,
                    yaml.tokens.BlockMappingStartToken,
                ),
            ):
                depth += 1
                # Valid v1 metadata is flat and needs at most three collection
                # levels. Bound invalid nesting BEFORE recursive construction.
                if depth > 8:
                    raise _invalid("skill frontmatter is too deeply nested")
            elif isinstance(
                token,
                (
                    yaml.tokens.FlowSequenceEndToken,
                    yaml.tokens.FlowMappingEndToken,
                    yaml.tokens.BlockEndToken,
                ),
            ):
                depth -= 1
            if isinstance(
                token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken, yaml.tokens.TagToken)
            ):
                raise _invalid("skill frontmatter does not support YAML tags or aliases")
        meta = yaml.load(block, Loader=_StrictLoader)
    except (yaml.YAMLError, RecursionError):
        raise _invalid("SKILL.md has invalid YAML frontmatter") from None
    if not isinstance(meta, dict) or set(meta) - _ALLOWED_KEYS:
        raise _invalid("SKILL.md has unsupported frontmatter keys")
    validate_name(meta.get("name"))
    description = meta.get("description")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > MAX_DESCRIPTION_CHARS
    ):
        raise _invalid("skill description must be nonblank and at most 1024 characters")
    for key in ("license", "compatibility", "version"):
        if key in meta and not isinstance(meta[key], str):
            raise _invalid("optional skill prose fields must be strings")
    for key in ("enabled", "hide", "disable-model-invocation"):
        if key in meta and type(meta[key]) is not bool:
            raise _invalid("skill discovery flags must be booleans")
    if "targets" in meta and not _strings(meta["targets"]):
        raise _invalid("skill targets must be a list of strings")
    if "allowed-tools" in meta and not (
        isinstance(meta["allowed-tools"], str) or _strings(meta["allowed-tools"])
    ):
        raise _invalid("skill allowed-tools must be a string or list of strings")
    metadata = meta.get("metadata", {})
    if not isinstance(metadata, dict) or any(
        not (_scalar(value) or isinstance(value, list) and all(_scalar(item) for item in value))
        for value in metadata.values()
    ):
        raise _invalid("skill metadata must contain safe scalars or flat scalar lists")
    return meta


@dataclass(frozen=True)
class SkillFile:
    path: str
    content: str


@dataclass(frozen=True)
class SkillDocument:
    name: str
    description: str
    files: tuple[SkillFile, ...]
    document_version: int = 1

    def __post_init__(self) -> None:
        if type(self.document_version) is not int or self.document_version != 1:
            raise _invalid("unsupported skill document version")
        validate_name(self.name)
        if not isinstance(self.description, str):
            raise _invalid("skill description must be a string")
        if not isinstance(self.files, tuple) or not self.files or len(self.files) > MAX_FILES:
            raise _large("skill must contain between 1 and 64 files")
        total = 0
        seen: set[str] = set()
        for file in self.files:
            if not isinstance(file, SkillFile):
                raise _invalid("skill manifest file is invalid")
            validate_path(file.path)
            # Preserve submitted case/Unicode spelling, but reject filesystem
            # aliases before APFS normalization can overwrite another entry.
            key = unicodedata.normalize("NFC", file.path).casefold()
            if key in seen or any(
                key.startswith(previous + "/") or previous.startswith(key + "/")
                for previous in seen
            ):
                # Files cannot also be ancestor directories, including aliases
                # after APFS-style case/NFC normalization. Validate the complete
                # manifest before any staged mkdir can encounter FileExistsError.
                raise _invalid("skill paths collide after Unicode/case normalization")
            seen.add(key)
            size = len(_text_bytes(file.content))
            if size > MAX_FILE_BYTES:
                raise _large("skill file exceeds 200 KiB")
            total += size
        if total > MAX_CONTENT_BYTES:
            raise _large("skill content exceeds 1 MiB")
        content = next((file.content for file in self.files if file.path == "SKILL.md"), None)
        if content is None:
            raise _invalid("skill requires root SKILL.md")
        meta = parse_frontmatter(content)
        if meta["name"] != self.name or meta["description"] != self.description:
            raise _invalid("skill manifest name/description must match SKILL.md frontmatter")
        object.__setattr__(self, "files", tuple(sorted(self.files, key=lambda item: item.path)))
        if len(self.wire_bytes()) > MAX_WIRE_BYTES:
            raise _large("skill encoded JSON exceeds 2 MiB")

    #: The manifest's OWN fields: everything the CLIENT's SkillDocument model
    #: carries. A hub DETAIL response (server source, agent-server@bdd03ab,
    #: internal/responses/skills.go HubSkillResponse) wraps the manifest in
    #: METADATA keys (content_revision, revision, sync_contract, id, tenant_id,
    #: account_id, account_metadata, visibility, moderation, created_date,
    #: updated_at) and may also carry a server-side `version` the client model
    #: does not store; the manifest is the subset below. The server documents
    #: this shape ONLY in that struct and docs/swagger.json --
    #: docs/HUB-REVISIONS.md predates the skill routes and does not claim them.
    MANIFEST_KEYS = ("document_version", "name", "description", "files")

    @classmethod
    def from_remote(cls, value: Mapping[str, Any]) -> "SkillDocument":
        """Build a manifest from a hub DETAIL payload, ignoring METADATA keys.

        The manifest itself stays STRICT (``from_json`` on the filtered subset):
        a wrong document_version, a bad file, an oversized set still refuses.
        What is tolerated is the response ENVELOPE, so one extra server-side key
        can never raise out of a check and take other kinds' checks down with it
        (round-2 QN1).
        """

        if not isinstance(value, Mapping):
            raise _invalid("the remote skill payload is not an object")
        subset = {key: value[key] for key in cls.MANIFEST_KEYS if key in value}
        return cls.from_json(subset)

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "SkillDocument":
        if not isinstance(value, Mapping) or set(value) != {
            "document_version",
            "name",
            "description",
            "files",
        }:
            raise _invalid("skill manifest has missing or unsupported keys")
        rows = value["files"]
        if not isinstance(rows, list):
            raise _invalid("skill files must be a list")
        files: list[SkillFile] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"path", "content"}:
                raise _invalid("skill file has missing or unsupported keys")
            files.append(SkillFile(row["path"], row["content"]))
        return cls(value["name"], value["description"], tuple(files), value["document_version"])

    @classmethod
    def from_bytes(cls, value: bytes) -> "SkillDocument":
        if len(value) > MAX_WIRE_BYTES:
            raise _large("skill encoded JSON exceeds 2 MiB")
        try:

            def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, item in pairs:
                    if key in result:
                        raise _invalid("skill JSON contains duplicate keys")
                    result[key] = item
                return result

            raw = json.loads(value.decode("utf-8", errors="strict"), object_pairs_hook=unique)
        except (UnicodeError, ValueError, RecursionError) as exc:
            if isinstance(exc, SkillValidationError):
                raise
            raise _invalid("skill manifest must be valid UTF-8 JSON") from None
        return cls.from_json(raw)

    def to_json(self) -> dict[str, Any]:
        return {
            "document_version": self.document_version,
            "name": self.name,
            "description": self.description,
            "files": [{"path": file.path, "content": file.content} for file in self.files],
        }

    def wire_bytes(self) -> bytes:
        return json.dumps(self.to_json(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.wire_bytes()).hexdigest()

    def with_name(self, name: str) -> "SkillDocument":
        """Local alias rendering only; callers restore the source name for ancestry.

        Replace just the YAML scalar span, not the entire dumped mapping. Safe
        optional metadata, comments, scripts and prose remain byte-for-byte;
        changing the local display alias must not become a remote rename.
        """
        return self._with_frontmatter_scalar("name", name, allow_description=False)

    def with_description(self, description: str) -> "SkillDocument":
        """Re-render the frontmatter ``description`` scalar from the merged field.

        Name and description are stored TWICE -- as manifest fields and as the
        SKILL.md frontmatter -- and construction refuses a document where the two
        disagree. A merge that edits the description field must therefore render
        the frontmatter copy in the same write, or the merged document would be
        invalid; only the scalar span is replaced, so prose, comments and
        metadata stay byte-for-byte.
        """
        return self._with_frontmatter_scalar("description", description, allow_description=True)

    def _with_frontmatter_scalar(
        self, key: str, value: str, *, allow_description: bool
    ) -> "SkillDocument":
        if key == "name":
            validate_name(value)
        elif len(value) > MAX_DESCRIPTION_CHARS:
            raise _invalid("skill description exceeds v1 bounds")
        files: list[SkillFile] = []
        for file in self.files:
            content = file.content
            if file.path == "SKILL.md":
                block, start, _ = _frontmatter_span(content)
                node = yaml.compose(block, Loader=_StrictLoader)
                assert isinstance(node, yaml.MappingNode)  # validated by construction
                scalar = next(item for k, item in node.value if k.value == key)
                left, right = start + scalar.start_mark.index, start + scalar.end_mark.index
                replaced = content[left:right]
                # Block-style YAML scalar spans own their terminal newline;
                # retain it so the next metadata key cannot join the alias.
                separator = (
                    "\r\n" if replaced.endswith("\r\n") else "\n" if replaced.endswith("\n") else ""
                )
                content = content[:left] + value + separator + content[right:]
            files.append(SkillFile(file.path, content))
        name = value if key == "name" else self.name
        description = value if key == "description" else self.description
        return SkillDocument(name, description, tuple(files))

    @classmethod
    def from_directory(cls, root: Path) -> "SkillDocument":
        """Read an explicitly selected source; never execute or omit its attachments.

        Storage/read locking is supplied by the shared managed-skill discipline.
        This bounded reader also applies to read-only ecosystem/project sources.
        """
        root = Path(root)
        if root.is_symlink() or not root.is_dir():
            raise _invalid("skill source must be a real directory")
        files: list[SkillFile] = []
        total = 0
        pending = [root]
        entries: list[Path] = []
        seen_entries = 0
        while pending:
            directory = pending.pop()
            # Do not sort/rglob an unbounded tree before checking its limits.
            # Empty directories spend the read budget too; unlimited traversal
            # must not hide behind the 64-file publication ceiling.
            with os.scandir(directory) as children:
                for child in children:
                    seen_entries += 1
                    if seen_entries > MAX_FILES * MAX_PATH_SEGMENTS + 3:
                        raise _large("skill source tree exceeds bounded traversal")
                    entry = Path(child.path)
                    path = entry.relative_to(root).as_posix()
                    if child.is_symlink():
                        raise _invalid("skill source contains a symlink")
                    if child.is_dir(follow_symlinks=False):
                        # Refuse hidden/unsupported directories even if empty.
                        validate_path(path + "/placeholder")
                        pending.append(entry)
                    else:
                        entries.append(entry)
        for entry in sorted(entries):
            path = entry.relative_to(root).as_posix()
            validate_path(path)
            if not entry.is_file():
                raise _invalid("skill source contains an unsupported filesystem entry")
            if len(files) >= MAX_FILES:
                raise _large("skill source exceeds 64 files")
            with entry.open("rb") as handle:
                data = handle.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise _large("skill source file exceeds 200 KiB")
            total += len(data)
            if total > MAX_CONTENT_BYTES:
                raise _large("skill source exceeds 1 MiB")
            try:
                content = data.decode("utf-8", errors="strict")
            except UnicodeError:
                raise _invalid("skill source contains non-UTF-8 binary content") from None
            files.append(SkillFile(path, content))
        skill = next((file for file in files if file.path == "SKILL.md"), None)
        if skill is None:
            raise _invalid("skill source requires SKILL.md")
        meta = parse_frontmatter(skill.content)
        return cls(meta["name"], meta["description"], tuple(files))
