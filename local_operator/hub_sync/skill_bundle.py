"""Content-addressed multi-file skill bundles: build, validate, extract.

WHY A ZIP AT ALL. A document skill is text only (``skill_document``); a real
skill directory carries binaries, examples and UI kits. A bundle is the same
skill published as one archive plus a small manifest, and the archive is treated
as UNTRUSTED on both ends: the builder is deterministic so the operator's
examples are stable, and the validator refuses by name instead of sanitizing.

TWO IDENTITIES, NEVER ONE. ``bundle.hash`` is sha256 over the archive BYTES: it
is the transport/storage identity and depends on the deflate implementation.
``bundle.tree_hash`` is sha256 over the decoded tree's paths, sizes and content
hashes: it is recomputable from a local directory, so it is the sync
fingerprint (L/B/R classification) and the value the journal and the baseline
carry. A client must never classify a link through the archive hash -- a Python
upgrade can change deflate output for identical content (design §2.2).

VALIDATION IS SEMANTIC (§12.3), NOT "byte-identical to our builder". Any archive
that is safe to unpack is accepted: the EOCD frame must account for every byte
(no trailing region), no ZIP64, no encryption or data descriptors, method stored
or deflate, no directory entries, entry modes regular-or-absent with no
exec/setuid/setgid/sticky bit, the path grammar below, no Unicode-casefold or
file-vs-ancestor collision, and streamed decompression counted against both the
declared size and the absolute budget with the CRC checked at EOF. Extraction is
our own code -- never ``ZipFile.extract``/``extractall``, which sanitize names
rather than refuse them -- and it writes 0600 files into a 0700 stage, never
applies a mode from the archive and never runs anything.

WHAT A REFUSAL MAY SAY. Diagnostics name the step, the rule and the offending
PATH; never file contents (this error surface is user-visible and may be logged).
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import shutil
import stat
import struct
import unicodedata
import zipfile
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from local_operator.hub_sync.skill_document import (
    MAX_DESCRIPTION_CHARS,
    SkillValidationError,
    parse_frontmatter,
    validate_name,
)

#: §12.1 bounds. The canonical example is ``minerva-design``: 141 files,
#: 25,485,405 uncompressed bytes, 24,773,425 as a deterministic archive -- a
#: 16 MiB/256-file/32 MiB carrier cannot hold it. Every one of these is a
#: STREAMED count; a declared size is only a pre-check.
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 96 * 1024 * 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_FILES = 1024
MAX_PATH_SEGMENTS = 8
MAX_PATH_BYTES = 240
MAX_SKILL_MD_BYTES = 200 * 1024

ENTRYPOINT = "SKILL.md"
FORMAT = "zip"
TREE_HASH_PREFIX = "skill-tree-v1\n"
_CHUNK = 64 * 1024
#: EOCD: signature, disk, cd disk, entries-on-disk, entries-total, cd size,
#: cd offset, comment length. 22 bytes, no comment (§3.1).
_EOCD = struct.Struct("<4s4H2LH")
_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_LOCATOR = b"PK\x06\x07"
#: Bytes Python's ``zipfile`` writes for a deflate entry built here; pinned so a
#: determinism test can compare two builds byte for byte.
_STORED = zipfile.ZIP_STORED
_DEFLATED = zipfile.ZIP_DEFLATED
_METHODS = (_STORED, _DEFLATED)

_REF_KEYS = (
    "format",
    "hash",
    "size",
    "uncompressed_size",
    "file_count",
    "entrypoint",
    "tree_hash",
)
_MANIFEST_KEYS = ("document_version", "name", "description", "bundle")
_MANIFEST_OPTIONAL = ("version",)


class SkillBundleError(SkillValidationError):
    """A named bundle refusal: ``code`` (§3.4), a human message, and ``step``.

    ``code`` is one of ``bundle-invalid``, ``bundle-hash-mismatch`` or
    ``bundle-too-large``; ``step`` names WHERE the refusal happened so the CLI
    and the publish path can render one message per rule rather than a generic
    "invalid archive". It subclasses ``SkillValidationError`` because existing
    catch sites (a per-item ``unavailable`` in a check tick, a refused install)
    must treat a bundle manifest exactly like a refused document manifest: an
    old caller must degrade, never crash, on the new form.
    """

    def __init__(self, code: str, message: str, *, step: str) -> None:
        self.step = step
        super().__init__(code, message)


def _refuse(code: str, step: str, message: str) -> SkillBundleError:
    return SkillBundleError(code, message, step=step)


def _refuse_too_large(step: str, message: str) -> SkillBundleError:
    return _refuse("bundle-too-large", step, message)


# -- paths ----------------------------------------------------------------------


def validate_bundle_path(path: Any) -> str:
    """The §12.2 bundle path grammar; raises ``bundle-invalid``/``entry``.

    Deliberately NOT ``skill_document.validate_path``: a bundle may use any
    relative directory (``ui_kits/``, ``components/``, ...) while a document is
    restricted to ``references|scripts|assets``. Every SAFETY rule is shared:
    relative, ``/``-separated, no backslash/colon (drive letters and Windows
    separators), no empty/dot/``..`` segment, no dot-prefixed segment, UTF-8,
    NFC-stable, no control or Unicode format (``Cf``) character, no segment
    ending in a dot or space, and the byte/depth bounds. ``SKILL.md`` must sit at
    the root; the builder skips hidden entries rather than failing.
    """
    if not isinstance(path, str) or not path:
        raise _refuse("bundle-invalid", "entry", "bundle path must be a relative text path")
    try:
        raw = path.encode("utf-8", errors="strict")
    except UnicodeError:
        raise _refuse("bundle-invalid", "entry", "bundle path must be UTF-8 text") from None
    parts = path.split("/")
    if (
        path.startswith("/")
        or "\\" in path
        or ":" in path
        or any(not part or part in (".", "..") or part.startswith(".") for part in parts)
        or any(ord(c) < 32 or ord(c) == 127 for c in path)
        or any(c.isspace() and c != " " for c in path)
    ):
        raise _refuse("bundle-invalid", "entry", f"bundle path {path!r} is unsafe")
    if any(unicodedata.category(c) == "Cf" for c in path):
        raise _refuse("bundle-invalid", "entry", f"bundle path {path!r} is unsafe")
    if path != unicodedata.normalize("NFC", path):
        raise _refuse("bundle-invalid", "entry", "bundle path must be NFC-normalized")
    if len(raw) > MAX_PATH_BYTES or len(parts) > MAX_PATH_SEGMENTS:
        raise _refuse("bundle-invalid", "entry", f"bundle path {path!r} exceeds its bounds")
    if any(part.endswith(".") or part.endswith(" ") for part in parts):
        # APFS and Windows silently strip these, so two spellings could name
        # one file: refuse the spelling at the source rather than normalize it.
        raise _refuse("bundle-invalid", "entry", f"bundle path {path!r} ends in a dot or space")
    return path


def _check_collisions(paths: Iterable[str]) -> None:
    """Refuse names that one filesystem would fold together (NFC + casefold).

    A file may not also be an ancestor DIRECTORY: staged ``mkdir`` would then
    meet ``FileExistsError`` after some content was already written.
    """
    seen: set[str] = set()
    for path in paths:
        key = unicodedata.normalize("NFC", path).casefold()
        if key in seen or any(
            key.startswith(previous + "/") or previous.startswith(key + "/") for previous in seen
        ):
            raise _refuse(
                "bundle-invalid",
                "entry",
                f"bundle paths collide after Unicode/case normalization: {path!r}",
            )
        seen.add(key)


# -- manifest -------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    # ``bool`` is an int subclass and a JSON ``true``/``1.0`` must not pass a
    # byte-count field: the manifest is the dedupe key, so its numbers are exact.
    return type(value) is int


@dataclass(frozen=True)
class SkillBundleRef:
    """The ``bundle`` sub-object: archive identity plus its declared inventory."""

    format: str
    hash: str
    size: int
    uncompressed_size: int
    file_count: int
    entrypoint: str
    tree_hash: str

    def __post_init__(self) -> None:
        if self.format != FORMAT:
            raise _refuse("bundle-invalid", "content", "bundle format must be 'zip'")
        for field_name, value in (("hash", self.hash), ("tree_hash", self.tree_hash)):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise _refuse(
                    "bundle-invalid", "content", f"bundle {field_name} must be 64 lowercase hex"
                )
        if self.entrypoint != ENTRYPOINT:
            raise _refuse("bundle-invalid", "entrypoint", "bundle entrypoint must be SKILL.md")
        if not _is_int(self.size) or not 1 <= self.size <= MAX_ARCHIVE_BYTES:
            raise _refuse_too_large("size", "bundle archive size is outside its bounds")
        if (
            not _is_int(self.uncompressed_size)
            or not 1 <= self.uncompressed_size <= MAX_UNCOMPRESSED_BYTES
        ):
            raise _refuse_too_large("size", "bundle uncompressed size is outside its bounds")
        if not _is_int(self.file_count) or not 1 <= self.file_count <= MAX_FILES:
            raise _refuse_too_large("entry", "bundle file count is outside its bounds")

    def to_json(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _REF_KEYS}

    @classmethod
    def from_json(cls, value: Any) -> SkillBundleRef:
        if not isinstance(value, Mapping) or set(value) != set(_REF_KEYS):
            raise _refuse("bundle-invalid", "content", "bundle reference has unsupported keys")
        return cls(
            format=value["format"],
            hash=value["hash"],
            size=value["size"],
            uncompressed_size=value["uncompressed_size"],
            file_count=value["file_count"],
            entrypoint=value["entrypoint"],
            tree_hash=value["tree_hash"],
        )


@dataclass(frozen=True)
class SkillBundleManifest:
    """One published bundle: the operator's fields plus the archive inventory.

    ``fingerprint`` is the TREE hash, not a hash of this JSON: it is the value
    ``classify`` compares against a local directory (design §3.2/§12) and the
    value a baseline record stores. Hashing the manifest JSON would make L==B
    unreachable -- the manifest is not recomputable from a directory.
    """

    name: str
    description: str
    bundle: SkillBundleRef
    document_version: int = 1
    version: str | None = None

    def __post_init__(self) -> None:
        if type(self.document_version) is not int or self.document_version != 1:
            raise _refuse("bundle-invalid", "content", "unsupported bundle document version")
        validate_name(self.name)
        if (
            not isinstance(self.description, str)
            or not self.description.strip()
            or len(self.description) > MAX_DESCRIPTION_CHARS
        ):
            raise _refuse(
                "bundle-invalid",
                "content",
                "bundle description must be nonblank and at most 1024 characters",
            )
        if not isinstance(self.bundle, SkillBundleRef):
            raise _refuse("bundle-invalid", "content", "bundle reference is invalid")
        if self.version is not None and not isinstance(self.version, str):
            raise _refuse("bundle-invalid", "content", "bundle version must be text")

    @property
    def fingerprint(self) -> str:
        return self.bundle.tree_hash

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "document_version": self.document_version,
            "name": self.name,
            "description": self.description,
            "bundle": self.bundle.to_json(),
        }
        if self.version is not None:
            payload["version"] = self.version
        return payload

    def wire_bytes(self) -> bytes:
        return json.dumps(self.to_json(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, value: Any) -> SkillBundleManifest:
        """Strict: the closed key set is exactly §3.2's, ``version`` optional."""

        if not isinstance(value, Mapping):
            raise _refuse("bundle-invalid", "content", "the bundle manifest is not an object")
        keys = set(value)
        if not set(_MANIFEST_KEYS) <= keys or keys - set(_MANIFEST_KEYS) - set(_MANIFEST_OPTIONAL):
            raise _refuse("bundle-invalid", "content", "bundle manifest has unsupported keys")
        return cls(
            name=value["name"],
            description=value["description"],
            bundle=SkillBundleRef.from_json(value["bundle"]),
            document_version=value["document_version"],
            version=value.get("version"),
        )

    @classmethod
    def from_remote(cls, value: Mapping[str, Any]) -> SkillBundleManifest:
        """Tolerant ENVELOPE, strict manifest -- the ``SkillDocument.from_remote`` rule.

        A hub DETAIL payload wraps the manifest in metadata (revision, id,
        sync_contract, publisher, ...). Filtering to the manifest's own keys
        keeps one extra server key from taking a check tick down, while the
        manifest subset still refuses by name.
        """

        if not isinstance(value, Mapping):
            raise _refuse("bundle-invalid", "content", "the remote bundle payload is not an object")
        subset = {key: value[key] for key in (*_MANIFEST_KEYS, *_MANIFEST_OPTIONAL) if key in value}
        return cls.from_json(subset)


@dataclass(frozen=True)
class BundleFile:
    """One decoded file: what review and previews name without shipping bytes."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class BundleInventory:
    """What validation measured, from the DECODED content (never headers)."""

    entries: tuple[BundleFile, ...]
    name: str
    description: str
    archive_hash: str
    tree_hash: str
    size: int
    uncompressed_size: int
    file_count: int

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(entry.path for entry in self.entries)


@dataclass(frozen=True)
class BuiltBundle:
    """A deterministic archive plus the manifest that describes it."""

    data: bytes
    archive_path: Path | None
    archive_hash: str
    tree_hash: str
    size: int
    uncompressed_size: int
    file_count: int
    name: str
    description: str

    @property
    def bundle(self) -> SkillBundleRef:
        return SkillBundleRef(
            format=FORMAT,
            hash=self.archive_hash,
            size=self.size,
            uncompressed_size=self.uncompressed_size,
            file_count=self.file_count,
            entrypoint=ENTRYPOINT,
            tree_hash=self.tree_hash,
        )

    @property
    def manifest(self) -> SkillBundleManifest:
        return SkillBundleManifest(name=self.name, description=self.description, bundle=self.bundle)


# -- tree hashing ---------------------------------------------------------------


def _tree_digest(rows: Iterable[tuple[str, int, str]]) -> str:
    """§3.1 ``tree_hash``: paths by UTF-8 bytes, then NUL size NUL sha256 LF."""

    digest = hashlib.sha256()
    digest.update(TREE_HASH_PREFIX.encode("utf-8"))
    for path, size, content_hash in sorted(rows, key=lambda row: row[0].encode("utf-8")):
        digest.update(f"{path}\x00{size}\x00{content_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def _stat_source(path: Path, name: str) -> os.stat_result:
    """``lstat`` one source entry and apply every "no executables, ever" rule."""

    try:
        info = os.lstat(path)
    except OSError:
        raise _refuse("bundle-invalid", "source", f"cannot read source entry {name!r}") from None
    mode = info.st_mode
    if stat.S_ISLNK(mode):
        raise _refuse("bundle-invalid", "source", f"source entry {name!r} is a symlink")
    if stat.S_ISDIR(mode):
        raise _refuse("bundle-invalid", "source", f"source entry {name!r} changed while walking")
    if not stat.S_ISREG(mode):
        raise _refuse(
            "bundle-invalid",
            "source",
            f"source entry {name!r} is not a regular file (device/FIFO/socket)",
        )
    if info.st_nlink > 1:
        # A hardlink shares bytes with a file outside the published tree: the
        # operator never chose to publish that other path, but its content would
        # ship. Refuse and ask for a copy instead (design §6.1).
        raise _refuse("bundle-invalid", "source", f"source entry {name!r} is a hardlink")
    if stat.S_IMODE(mode) & 0o7111:
        # exec/setuid/setgid/sticky; bundles are data, never programs (§6.1).
        raise _refuse(
            "bundle-invalid",
            "source",
            f"source entry {name!r} is executable or carries a special mode bit",
        )
    return info


def _iter_source_files(root: Path) -> Iterator[tuple[str, Path, os.stat_result]]:
    """Walk a source tree with the SAME rules ``tree_hash_of`` and the builder use.

    Dot-prefixed files AND directories are skipped (never published); every
    other entry must pass ``_stat_source`` and the bundle path grammar.
    """

    try:
        info = os.lstat(root)
    except OSError:
        raise _refuse("bundle-invalid", "source", "the skill directory does not exist") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise _refuse("bundle-invalid", "source", "the skill source must be a real directory")

    def walk(directory: Path, prefix: str) -> Iterator[tuple[str, Path, os.stat_result]]:
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            raise _refuse("bundle-invalid", "source", "the skill directory is unreadable") from None
        for child in children:
            if child.name.startswith("."):
                continue
            relative = f"{prefix}/{child.name}" if prefix else child.name
            entry = Path(child.path)
            try:
                child_info = child.stat(follow_symlinks=False)
            except OSError:
                raise _refuse(
                    "bundle-invalid", "source", f"cannot read source entry {relative!r}"
                ) from None
            if stat.S_ISDIR(child_info.st_mode):
                # ``lstat`` semantics: a symlinked directory is NOT S_ISDIR, so it
                # reaches ``_stat_source`` and is refused there.
                validate_bundle_path(relative)
                yield from walk(entry, relative)
                continue
            validate_bundle_path(relative)
            yield relative, entry, _stat_source(entry, relative)

    yield from walk(root, "")


def _read_source_file(path: Path, info: os.stat_result, name: str) -> bytes:
    """Read one already-vetted source file, re-checking what ``lstat`` promised."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise _refuse("bundle-invalid", "source", f"cannot read source entry {name!r}") from None
    with os.fdopen(fd, "rb") as stream:
        fresh = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(fresh.st_mode)
            or fresh.st_nlink > 1
            or stat.S_IMODE(fresh.st_mode) & 0o7111
            or (fresh.st_dev, fresh.st_ino) != (info.st_dev, info.st_ino)
        ):
            # The path changed between the walk and the read: refuse rather than
            # publish bytes nobody vetted (the same TOCTOU discipline as fork.py).
            raise _refuse(
                "bundle-invalid", "source", f"source entry {name!r} changed while reading"
            )
        content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise _refuse_too_large("entry", f"source entry {name!r} exceeds 16 MiB")
    return content


def _hash_source_file(path: Path, info: os.stat_result, name: str) -> tuple[int, str]:
    """Stream one source file to its (size, sha256), bounded by MAX_FILE_BYTES."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise _refuse("bundle-invalid", "source", f"cannot read source entry {name!r}") from None
    size = 0
    digest = hashlib.sha256()
    with os.fdopen(fd, "rb") as stream:
        fresh = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(fresh.st_mode)
            or fresh.st_nlink > 1
            or stat.S_IMODE(fresh.st_mode) & 0o7111
            or (fresh.st_dev, fresh.st_ino) != (info.st_dev, info.st_ino)
        ):
            raise _refuse(
                "bundle-invalid", "source", f"source entry {name!r} changed while reading"
            )
        while True:
            chunk = stream.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_BYTES:
                raise _refuse_too_large("entry", f"source entry {name!r} exceeds 16 MiB")
            digest.update(chunk)
    return size, digest.hexdigest()


def tree_hash_of(root: Path) -> str:
    """The §3.1 hash of a local directory, with the builder's exact refusals.

    This is the LOCAL side of classification: it refuses exactly what
    ``build_bundle`` refuses (symlinks, hardlinks, special/executable modes, bad
    paths) so a tree cannot build fine and then read as a different artifact.
    It deliberately does NOT require ``SKILL.md``: a bundle link whose entrypoint
    a user deleted is a LOCAL CHANGE to classify, not a crash, and the hash of
    the remaining tree is what says so.
    """

    rows: list[tuple[str, int, str]] = []
    total = 0
    for relative, path, info in _iter_source_files(Path(root)):
        if len(rows) >= MAX_FILES:
            raise _refuse_too_large("entry", "the skill directory holds more than 1024 files")
        size, content_hash = _hash_source_file(path, info, relative)
        total += size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise _refuse_too_large("size", "the skill directory exceeds 96 MiB")
        rows.append((relative, size, content_hash))
    if not rows:
        raise _refuse("bundle-invalid", "entrypoint", "the skill directory is empty")
    _check_collisions(row[0] for row in rows)
    return _tree_digest(rows)


# -- building -------------------------------------------------------------------


def _zip_bytes(rows: list[tuple[str, bytes]]) -> bytes:
    """The deterministic §3.1 archive: sorted paths, fixed metadata, deflate 6."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=_DEFLATED, compresslevel=6, strict_timestamps=True
    ) as archive:
        for path, content in sorted(rows, key=lambda row: row[0].encode("utf-8")):
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            # Pinned EXPLICITLY: Windows' ZipInfo defaults to create_system 0,
            # which changes the external_attr byte layout and therefore the hash
            # of an identical tree (design §1.3).
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compress_type=_DEFLATED, compresslevel=6)
    return buffer.getvalue()


def _write_archive(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise _refuse("bundle-invalid", "source", "the archive destination is a symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def build_bundle(root: Path, *, dest: Path | None = None) -> BuiltBundle:
    """Build the deterministic archive and manifest for one skill directory.

    A source may not contain a symlink, hardlink, device, FIFO/socket or any
    executable/setuid/setgid/sticky-mode file: bundles are inert data and the
    refusal names the entry. ``dest`` (optional) receives the same bytes with
    0600 permissions; the bytes are always returned so a publisher can upload
    without a second read.
    """

    source_root = Path(root).expanduser()
    rows: list[tuple[str, bytes]] = []
    total = 0
    for relative, path, info in _iter_source_files(source_root):
        if len(rows) >= MAX_FILES:
            raise _refuse_too_large("entry", "the skill directory holds more than 1024 files")
        content = _read_source_file(path, info, relative)
        total += len(content)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise _refuse_too_large("size", "the skill directory exceeds 96 MiB")
        rows.append((relative, content))
    _check_collisions(row[0] for row in rows)
    paths = [row[0] for row in rows]
    if paths.count(ENTRYPOINT) != 1:
        raise _refuse("bundle-invalid", "entrypoint", "a bundle requires exactly one root SKILL.md")
    skill_md = dict(rows)[ENTRYPOINT]
    if len(skill_md) > MAX_SKILL_MD_BYTES:
        raise _refuse_too_large("entrypoint", "SKILL.md exceeds 200 KiB")
    try:
        text = skill_md.decode("utf-8", errors="strict")
    except UnicodeError:
        raise _refuse("bundle-invalid", "entrypoint", "SKILL.md must be UTF-8 text") from None
    try:
        meta = parse_frontmatter(text)
    except SkillValidationError as exc:
        raise _refuse(
            "bundle-invalid", "entrypoint", f"SKILL.md frontmatter is invalid: {exc}"
        ) from None
    data = _zip_bytes(rows)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise _refuse_too_large("size", "the built bundle archive exceeds 64 MiB")
    archive_path: Path | None = None
    if dest is not None:
        archive_path = Path(dest).expanduser()
        _write_archive(archive_path, data)
    tree_hash = _tree_digest(
        (path, len(content), hashlib.sha256(content).hexdigest()) for path, content in rows
    )
    return BuiltBundle(
        data=data,
        archive_path=archive_path,
        archive_hash=hashlib.sha256(data).hexdigest(),
        tree_hash=tree_hash,
        size=len(data),
        uncompressed_size=total,
        file_count=len(rows),
        name=str(meta["name"]),
        description=str(meta["description"]),
    )


def bundle_manifest_from_directory(
    root: Path, *, name: str | None = None, description: str | None = None
) -> SkillBundleManifest:
    """Manifest for a directory that is ABOUT to be published or compared.

    The archive is built in memory: it is bounded at 64 MiB and a temporary file
    would be one more thing that can be left behind on a crash. ``name`` /
    ``description`` default to the SKILL.md frontmatter (the source of truth a
    reviewer reads).
    """

    built = build_bundle(Path(root))
    manifest = built.manifest
    if name is None and description is None:
        return manifest
    return SkillBundleManifest(
        name=manifest.name if name is None else name,
        description=manifest.description if description is None else description,
        bundle=manifest.bundle,
    )


# -- reading --------------------------------------------------------------------


def _read_bounded(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise _refuse("bundle-invalid", "size", "the bundle archive cannot be read") from None
    with os.fdopen(fd, "rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise _refuse_too_large("size", "the bundle archive exceeds 64 MiB")
    return data


def _check_frame(data: bytes) -> int:
    """§4 step 2: the EOCD frame must account for EVERY byte of the file.

    Done by hand, before any zip library, so trailing/prepended regions and an
    absurd declared entry count are refused before the library allocates a
    record per entry.
    """

    if len(data) < _EOCD.size:
        raise _refuse("bundle-invalid", "container", "the bundle archive is too short")
    if len(data) >= _EOCD.size + 20 and data[-_EOCD.size - 20 : -_EOCD.size][:4] == _ZIP64_LOCATOR:
        raise _refuse("bundle-invalid", "container", "ZIP64 bundles are not supported")
    (
        signature,
        disk,
        cd_disk,
        entries_disk,
        entries_total,
        cd_size,
        cd_offset,
        comment_len,
    ) = _EOCD.unpack(data[-_EOCD.size :])
    if (
        signature != _EOCD_SIGNATURE
        or comment_len != 0
        or disk != 0
        or cd_disk != 0
        or entries_disk != entries_total
    ):
        # A comment moves the frame we validated; a nonzero disk field means a
        # spanned archive we do not read. Both are refusals, not sanitisation.
        raise _refuse("bundle-invalid", "container", "the bundle is not a single-disk zip")
    if entries_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        raise _refuse("bundle-invalid", "container", "ZIP64 bundles are not supported")
    if entries_total > MAX_FILES:
        raise _refuse_too_large("entry", "the bundle declares more than 1024 entries")
    if cd_offset + cd_size + _EOCD.size != len(data):
        raise _refuse("bundle-invalid", "container", "the bundle has trailing or prepended bytes")
    return entries_total


def _entry_mode(info: zipfile.ZipInfo) -> int:
    # The high 16 bits of external_attr are the Unix mode when create_system is
    # 3; a FAT/Windows writer leaves them 0 (absent), which is allowed.
    return info.external_attr >> 16


def _open_archive(data: bytes) -> zipfile.ZipFile:
    """Open the archive ONCE from the bytes already read (`_validate_bytes` shares it)."""

    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, ValueError):
        raise _refuse(
            "bundle-invalid", "container", "the bundle is not a readable zip archive"
        ) from None


def _validate_bytes(
    data: bytes,
    *,
    expected: SkillBundleRef | None,
    expected_name: str | None,
    expected_description: str | None,
    opened: zipfile.ZipFile | None = None,
    archive_hash: str | None = None,
) -> BundleInventory:
    """Validate the archive BYTES, optionally through an already-open archive.

    THE BYTES ARE THE INPUT, never a path: a caller that validates a path and
    then re-opens it can have the file swapped in between (the TOCTOU
    ``extract_archive`` used to have). ``opened`` lets extraction validate and
    unpack through one and the same ``ZipFile`` object.
    """

    digest = archive_hash if archive_hash is not None else hashlib.sha256(data).hexdigest()
    entries_total = _check_frame(data)
    own = opened is None
    archive = opened if opened is not None else _open_archive(data)
    try:
        infos = archive.infolist()
        if len(infos) != entries_total:
            raise _refuse("bundle-invalid", "container", "the bundle's entry count is inconsistent")

        total_declared = 0
        names: list[str] = []
        for info in infos:
            # ``orig_filename`` keeps the raw spelling; ``filename`` has already
            # rewritten backslashes to slashes, which would make a ``..\x`` entry
            # look like an ordinary relative path. Validate the ORIGINAL.
            name = info.orig_filename
            if name != info.filename:
                raise _refuse("bundle-invalid", "entry", "bundle entry name is ambiguous")
            if not name or name.endswith("/"):
                raise _refuse(
                    "bundle-invalid", "entry", "bundle archives must not carry directories"
                )
            try:
                validate_bundle_path(name)
            except SkillBundleError as exc:
                raise _refuse("bundle-invalid", exc.step, str(exc)) from None
            if info.flag_bits & ~0x0800:
                # Encryption (bit 0), data descriptors (bit 3), patched data (5),
                # strong encryption (6): none is a shape this carrier accepts.
                raise _refuse("bundle-invalid", "entry", f"bundle entry {name!r} uses zip flags")
            if info.compress_type not in _METHODS:
                raise _refuse(
                    "bundle-invalid", "entry", f"bundle entry {name!r} uses an unsupported method"
                )
            mode = _entry_mode(info)
            file_type = stat.S_IFMT(mode)
            if file_type not in (0, stat.S_IFREG):
                # symlink, device, FIFO, socket or directory MODE from the central
                # directory. This is the single check §12.3 keeps from the old
                # canonical-form equality.
                raise _refuse(
                    "bundle-invalid", "entry", f"bundle entry {name!r} is not a regular file"
                )
            if mode & 0o7111:
                raise _refuse(
                    "bundle-invalid",
                    "entry",
                    f"bundle entry {name!r} is executable or special-mode",
                )
            if info.file_size > MAX_FILE_BYTES:
                raise _refuse_too_large("entry", f"bundle entry {name!r} exceeds 16 MiB")
            if info.compress_type == _STORED and info.compress_size != info.file_size:
                # Stored data cannot be smaller or larger than its declared size; a
                # mismatch is either a bomb shape or a parser differential, and
                # ``zipfile`` would silently truncate the read to the declared size.
                raise _refuse(
                    "bundle-invalid", "content", f"bundle entry {name!r} has inconsistent sizes"
                )
            total_declared += info.file_size
            if total_declared > MAX_UNCOMPRESSED_BYTES:
                raise _refuse_too_large("size", "the bundle declares more than 96 MiB")
            names.append(name)
        _check_collisions(names)
        if ENTRYPOINT not in names:
            raise _refuse("bundle-invalid", "entrypoint", "a bundle requires a root SKILL.md")

        entries: list[BundleFile] = []
        total_actual = 0
        for info in infos:
            name = info.orig_filename
            size = 0
            file_digest = hashlib.sha256()
            try:
                with archive.open(info) as stream:
                    while True:
                        chunk = stream.read(_CHUNK)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > info.file_size:
                            # Declared sizes are a claim; count what we decode.
                            raise _refuse(
                                "bundle-invalid",
                                "content",
                                f"bundle entry {name!r} is larger than its declared size",
                            )
                        if size > MAX_FILE_BYTES:
                            raise _refuse_too_large(
                                "content", f"bundle entry {name!r} exceeds 16 MiB"
                            )
                        total_actual += len(chunk)
                        if total_actual > MAX_UNCOMPRESSED_BYTES:
                            raise _refuse_too_large("content", "the bundle expands beyond 96 MiB")
                        file_digest.update(chunk)
            except (zipfile.BadZipFile, zlib.error):
                # ``zipfile`` wraps a bad CRC as BadZipFile at EOF, but a corrupt
                # DEFLATE stream escapes as ``zlib.error`` -- an uncaught
                # exception here would crash an install instead of refusing it,
                # so both are this one named refusal. Reading to EOF is what
                # makes the library check the CRC at all.
                raise _refuse(
                    "bundle-invalid", "content", f"bundle entry {name!r} failed its integrity check"
                ) from None
            if size != info.file_size:
                raise _refuse(
                    "bundle-invalid",
                    "content",
                    f"bundle entry {name!r} does not match its declared size",
                )
            entries.append(BundleFile(name, size, file_digest.hexdigest()))

        by_path = {entry.path: entry for entry in entries}
        skill_md = by_path[ENTRYPOINT]
        if skill_md.size > MAX_SKILL_MD_BYTES:
            raise _refuse_too_large("entrypoint", "SKILL.md exceeds 200 KiB")
        # Read SKILL.md from the SAME open archive; it is ≤200 KiB and already
        # CRC-checked above.
        try:
            content = archive.read(ENTRYPOINT)
        except (zipfile.BadZipFile, OSError, KeyError):
            raise _refuse("bundle-invalid", "entrypoint", "SKILL.md cannot be read") from None
    finally:
        if own:
            archive.close()

    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeError:
        raise _refuse("bundle-invalid", "entrypoint", "SKILL.md must be UTF-8 text") from None
    try:
        meta = parse_frontmatter(text)
    except SkillValidationError as exc:
        raise _refuse(
            "bundle-invalid", "entrypoint", f"SKILL.md frontmatter is invalid: {exc}"
        ) from None
    if expected_name is not None and meta["name"] != expected_name:
        raise _refuse("bundle-invalid", "entrypoint", "SKILL.md name does not match the manifest")
    if expected_description is not None and meta["description"] != expected_description:
        raise _refuse(
            "bundle-invalid", "entrypoint", "SKILL.md description does not match the manifest"
        )

    tree_hash = _tree_digest((entry.path, entry.size, entry.sha256) for entry in entries)
    uncompressed_size = sum(entry.size for entry in entries)
    if expected is not None:
        if len(data) != expected.size:
            raise _refuse("bundle-invalid", "size", "the bundle archive size does not match")
        if uncompressed_size != expected.uncompressed_size:
            raise _refuse("bundle-invalid", "size", "the bundle uncompressed size does not match")
        if len(entries) != expected.file_count:
            raise _refuse("bundle-invalid", "entry", "the bundle file count does not match")
        if expected.entrypoint not in by_path:
            raise _refuse("bundle-invalid", "entrypoint", "the bundle entrypoint is missing")
        if not hmac.compare_digest(tree_hash, expected.tree_hash):
            raise _refuse("bundle-invalid", "tree", "the bundle tree hash does not match")
    return BundleInventory(
        entries=tuple(entries),
        name=str(meta["name"]),
        description=str(meta["description"]),
        archive_hash=digest,
        tree_hash=tree_hash,
        size=len(data),
        uncompressed_size=uncompressed_size,
        file_count=len(entries),
    )


def validate_archive(
    archive: Path,
    *,
    expected: SkillBundleRef | None = None,
    expected_name: str | None = None,
    expected_description: str | None = None,
) -> BundleInventory:
    """Validate one archive semantically (§4, as amended by §12.3).

    Order matters: the tree is hashed BEFORE anything is parsed, so a
    ``bundle-hash-mismatch`` is reported without opening the archive; the
    manifest's declared size/count/hash are compared before decompression; and
    every refusal names its step. ``expected_name`` / ``expected_description``
    are the manifest's operator fields (they live beside the reference, not
    inside it) and are checked against ``SKILL.md`` when supplied.

    The path is read exactly ONCE: everything after that works on those bytes
    (``_validate_bytes``), so a swap between validation and use cannot exist.
    """

    archive = Path(archive)
    try:
        info = os.lstat(archive)
    except OSError:
        raise _refuse("bundle-invalid", "size", "the bundle archive is missing") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise _refuse("bundle-invalid", "size", "the bundle archive is not a regular file")
    if info.st_size == 0:
        raise _refuse("bundle-invalid", "size", "the bundle archive is empty")
    if info.st_size > MAX_ARCHIVE_BYTES:
        raise _refuse_too_large("size", "the bundle archive exceeds 64 MiB")
    data = _read_bounded(archive, MAX_ARCHIVE_BYTES)
    archive_hash = hashlib.sha256(data).hexdigest()
    if expected is not None and not hmac.compare_digest(archive_hash, expected.hash):
        # Hash first, parse never: the manifest's identity is the bytes, so a
        # substituted archive is refused before a zip parser sees it.
        raise _refuse("bundle-hash-mismatch", "hash", "the bundle archive hash does not match")
    with _open_archive(data) as opened:
        return _validate_bytes(
            data,
            expected=expected,
            expected_name=expected_name,
            expected_description=expected_description,
            opened=opened,
            archive_hash=archive_hash,
        )


def extract_archive(
    archive: Path,
    dest: Path,
    *,
    expected: SkillBundleRef,
    expected_name: str | None = None,
    expected_description: str | None = None,
) -> str:
    """Validate and unpack one bundle into an EMPTY ``dest``; return its tree hash.

    ``dest`` must not exist and is created 0700; every file is created 0600 with
    ``O_EXCL|O_NOFOLLOW``, no mode from the archive is applied and nothing is
    executed (design §5 step 4).

    ONE READ, ONE OPENED ARCHIVE. The file is read once, hashed against the
    manifest, validated and unpacked through the SAME ``zipfile.ZipFile`` over
    those bytes -- there is no second read of the path to swap, and every
    written path is the VALIDATED relative spelling (never ``orig_filename``
    used raw). The extracted tree is re-hashed from disk, so a write that landed
    somewhere unexpected cannot pass as the expected tree.
    """

    archive = Path(archive)
    dest = Path(dest)
    if dest.exists() or dest.is_symlink():
        raise _refuse("bundle-invalid", "content", "the bundle staging directory already exists")
    try:
        info = os.lstat(archive)
    except OSError:
        raise _refuse("bundle-invalid", "size", "the bundle archive is missing") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise _refuse("bundle-invalid", "size", "the bundle archive is not a regular file")
    if info.st_size == 0:
        raise _refuse("bundle-invalid", "size", "the bundle archive is empty")
    if info.st_size > MAX_ARCHIVE_BYTES:
        raise _refuse_too_large("size", "the bundle archive exceeds 64 MiB")
    data = _read_bounded(archive, MAX_ARCHIVE_BYTES)
    archive_hash = hashlib.sha256(data).hexdigest()
    if not hmac.compare_digest(archive_hash, expected.hash):
        # The manifest's identity IS the bytes: refuse before a parser sees them.
        raise _refuse("bundle-hash-mismatch", "hash", "the bundle archive hash does not match")
    staged = ""
    try:
        os.mkdir(dest, 0o700)
        total = 0
        with _open_archive(data) as opened:
            inventory = _validate_bytes(
                data,
                expected=expected,
                expected_name=expected_name,
                expected_description=expected_description,
                opened=opened,
                archive_hash=archive_hash,
            )
            for entry in inventory.entries:
                parts = entry.path.split("/")
                target = dest.joinpath(*parts)
                # Defence in depth: the path grammar already refused traversal,
                # and this refuses a joined target that left the stage anyway.
                try:
                    target.relative_to(dest)
                except ValueError:
                    raise _refuse(
                        "bundle-invalid", "entry", f"bundle entry {entry.path!r} is unsafe"
                    ) from None
                if len(parts) > 1:
                    _ensure_parents(dest, Path(*parts[:-1]))
                flags = (
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_BINARY", 0)
                )
                fd = os.open(target, flags, 0o600)
                size = 0
                try:
                    with os.fdopen(fd, "wb") as out, opened.open(entry.path) as stream:
                        while True:
                            chunk = stream.read(_CHUNK)
                            if not chunk:
                                break
                            size += len(chunk)
                            total += len(chunk)
                            if size > entry.size or size > MAX_FILE_BYTES:
                                raise _refuse_too_large(
                                    "content", f"bundle entry {entry.path!r} exceeds its bounds"
                                )
                            if total > MAX_UNCOMPRESSED_BYTES:
                                raise _refuse_too_large(
                                    "content", "the bundle expands beyond 96 MiB"
                                )
                            out.write(chunk)
                        out.flush()
                        os.fsync(out.fileno())
                except (zipfile.BadZipFile, zlib.error):
                    # Same named refusal as the validator's streamed pass: a
                    # corrupt DEFLATE stream must not escape as a raw library
                    # exception out of an install.
                    raise _refuse(
                        "bundle-invalid",
                        "content",
                        f"bundle entry {entry.path!r} failed its integrity check",
                    ) from None
        staged = tree_hash_of(dest)
        if not hmac.compare_digest(staged, expected.tree_hash):
            raise _refuse("bundle-invalid", "tree", "the extracted bundle does not match its tree")
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return staged


def _ensure_parents(root: Path, relative: Path) -> None:
    """Create 0700 parents for a VALIDATED relative path, refusing anything unsafe.

    Independently defensive: it does not trust the caller's validation. Absolute
    paths, drive letters, empty/``.``/``..`` components, separators inside a
    component are refused; an existing component is accepted only when it is a
    real directory (never a symlink), and ``FileExistsError`` is verified rather
    than swallowed.
    """

    if relative.is_absolute() or relative.drive:
        raise _refuse("bundle-invalid", "content", "the bundle staging path is unsafe")
    root = Path(root)
    current = root
    for part in relative.parts:
        if part in ("", ".", "..") or "/" in part or "\\" in part or ":" in part:
            raise _refuse("bundle-invalid", "content", "the bundle staging path is unsafe")
        current = current / part
        try:
            os.mkdir(current, 0o700)
        except FileExistsError:
            # An existing component is fine only when it really is a directory:
            # a symlink or a file here would redirect (or collide with) the write.
            if current.is_symlink() or not current.is_dir():
                raise _refuse(
                    "bundle-invalid", "content", "the bundle staging path is unsafe"
                ) from None
        except OSError:
            raise _refuse(
                "bundle-invalid", "content", "the bundle staging path is unsafe"
            ) from None
        if current.is_symlink() or not current.is_dir():
            raise _refuse("bundle-invalid", "content", "the bundle staging path is unsafe")
