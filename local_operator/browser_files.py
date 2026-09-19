"""File policy for the browser tool: what may be downloaded, what may be uploaded.

ONE module for both hosts and for the tool, because the rules must be identical
on every host and a second implementation is how two hosts start disagreeing
(design §10.1). It is deliberately a plain-function module — no state, no socket,
no import of ``browser_bridge`` — so the desktop app's code does not have to
import the extension bridge to learn what a PDF is.

Three jobs, and nothing else:

* :func:`safe_name` — the only door a page-supplied string may pass through
  before it becomes a path component, a log line, an audit row or an approval
  prompt (design §10.2, §9.2.6).
* :func:`classify_download` — content-first classification of a file that has
  LANDED, returning the verdict the model is told (design §5.3, §10.2).
* :func:`check_upload` — the unconditional pre-flight gate a local path must
  pass before its bytes are handed to a browser (design §9.2).

**Content wins over the name, in one direction only.** A ``.txt`` whose bytes
are a PE executable is DENIED; a ``.exe`` name over PDF bytes is allowed *as a
PDF*, renamed. That asymmetry is the design (§10.2's table), not an oversight.

**The tables are data, and they are the source of truth for the generated
TypeScript** (`extension/src/driver/file-transfer.tables.gen.ts`): the deny/allow
lists, the caps, and a hand-written conformance fixture whose expectations the
generator re-derives from :func:`classify_bytes`. A generator that merely
serialised Python's output would be a tautology; one that must reproduce
hand-written expectations is a gate (design §10.4).

**Sniffing is a hand-rolled signature table, not a library.** The design
recommends PyPI ``filetype`` (MIT, dependency-free); this build takes the
documented fallback instead because PR A is under a no-new-default-dependency
constraint. The cost is named where it bites: without reading the archive's
first entries a hand table cannot tell a ``.docx`` from a plain ``.zip``, so
both are one "ZIP container" class and the NAME decides which of the two the
file is reported as. Nothing is looser as a result — the class is on the
allow-list either way — but it is strictly weaker than the library, exactly as
the design's fallback paragraph says.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import re
import secrets
import stat
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, NamedTuple

from local_operator.paths import config_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caps (design §10.3)
# ---------------------------------------------------------------------------
#
# Module constants rather than settings, deliberately: AGENTS.md requires every
# new configuration key to be registered in `settings_io.py` with a section, a
# scope and a consumer binding, and to be covered by
# `test_every_default_matches_its_consumer`. Five keys for five numbers nobody
# has yet wanted to change is a `/settings` tax for a hypothetical need, so the
# numbers live here and a settings follow-up can add them if the operator ever
# wants to raise one (design §16.3).

#: Per downloaded file. A page that streams gigabytes is a denial of service on
#: the agent's own turn, and the browser process reads the bytes into memory.
DOWNLOAD_MAX_BYTES = 256 * 1024 * 1024
#: Per `download` call — one page can start many downloads from one click.
DOWNLOAD_MAX_FILES_PER_CALL = 20
#: Per session, over the whole directory. Checked BEFORE a call rather than
#: mid-flight, so an over-quota session is refused without arming anything.
#: Stated precisely because the number reads like a bound on the directory:
#: it bounds the NEXT call ("refuse when the directory already exceeds it"),
#: so a session can sit up to one call's worth above it (review round 1, R5).
DOWNLOAD_MAX_TOTAL_BYTES_PER_SESSION = 2 * 1024**3
#: How long a `download` waits for the page to start one. The tool may raise it
#: through ``timeout_s`` up to the ceiling below; the ceiling exists because a
#: browser command cannot hold a tab's lock indefinitely.
DOWNLOAD_TIMEOUT_S = 120.0
DOWNLOAD_TIMEOUT_MAX_S = 600.0
#: Per uploaded file, and per `upload` call.
UPLOAD_MAX_BYTES = 256 * 1024 * 1024
UPLOAD_MAX_FILES = 10

#: Fallback basename when a page-supplied name sanitises to nothing usable.
_FALLBACK_STEM = "download"
#: Names are capped in BYTES, not characters: the filesystem limit is 255 bytes
#: and a uniquifying suffix has to fit beside it.
MAX_NAME_BYTES = 200


@dataclass(frozen=True)
class Policy:
    """The tunable part of the policy: the caps, as one injectable object.

    The signature/extension tables are deliberately NOT on this object. They are
    emitted into TypeScript (see the module docstring), so an instance-level copy
    would be a second source of truth for the same data — the thing §10.4 exists
    to prevent. The caps have no TS consumer today, so they can be instance data
    and a test can shrink them without monkeypatching module globals.
    """

    download_max_bytes: int = DOWNLOAD_MAX_BYTES
    download_max_files: int = DOWNLOAD_MAX_FILES_PER_CALL
    download_max_session_bytes: int = DOWNLOAD_MAX_TOTAL_BYTES_PER_SESSION
    upload_max_bytes: int = UPLOAD_MAX_BYTES
    upload_max_files: int = UPLOAD_MAX_FILES


DEFAULT = Policy()

DownloadClass = Literal["allow", "deny", "unknown"]


#: A classification, and the exact words the model is given for it.
#:
#: ``reason`` is "" when there is nothing to say (an allowed file whose name
#: already matches its content). ``sniffed`` is the CLASS name the content
#: matched, or "" when nothing matched — the extension is derived from the class
#: (``sniffed_ext``) rather than being a second field, so the two can never
#: disagree.
class Verdict(NamedTuple):
    kind: DownloadClass
    reason: str
    sniffed: str
    safe_name: str


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentClass:
    """One signature class: what the bytes are, and what names fit it.

    ``exts`` is the set of extensions the class is *name-consistent* with. A file
    whose name is outside that set is still allowed when the class is on the
    allow-list — it is RENAMED to ``ext`` and the rename is reported (design
    §10.2's third row). A deny class has no such repair: content wins, and the
    file is deleted.
    """

    name: str
    label: str
    ext: str
    exts: frozenset[str]
    #: (offset, bytes) pairs, any one of which identifies the class. Offsets
    #: other than 0 are what makes RIFF containers and TAR distinguishable.
    signatures: tuple[tuple[int, bytes], ...] = ()
    #: Signatures that live at the END of the file (``koly`` closes a DMG).
    footer: tuple[bytes, ...] = ()


#: Classes whose CONTENT is refused outright. A name is not consulted: the class
#: list is what catches a novel extension over executable bytes (design §10.2).
DENY_CLASSES: tuple[ContentClass, ...] = (
    ContentClass(
        name="pe",
        label="a Windows executable (PE)",
        ext="exe",
        exts=frozenset({"exe", "dll", "sys", "scr", "com", "cpl", "ocx", "drv", "efi"}),
        signatures=((0, b"MZ"),),
    ),
    ContentClass(
        name="elf",
        label="a Linux executable (ELF)",
        ext="elf",
        exts=frozenset({"elf", "so", "bin"}),
        signatures=((0, b"\x7fELF"),),
    ),
    ContentClass(
        name="macho",
        label="a compiled binary (Mach-O or Java bytecode)",
        ext="bin",
        exts=frozenset({"dylib", "bin"}),
        # `CAFEBABE` is BOTH a Mach-O universal binary and Java bytecode; both are
        # denied, so distinguishing them would only produce a nicer label.
        signatures=(
            (0, b"\xfe\xed\xfa\xce"),
            (0, b"\xce\xfa\xed\xfe"),
            (0, b"\xfe\xed\xfa\xcf"),
            (0, b"\xcf\xfa\xed\xfe"),
            (0, b"\xca\xfe\xba\xbe"),
        ),
    ),
    ContentClass(
        name="script",
        label="a script (shebang)",
        ext="sh",
        exts=frozenset({"sh", "bash", "zsh", "command", "scpt"}),
        signatures=((0, b"#!"),),
    ),
    ContentClass(
        name="dex",
        label="Android bytecode (DEX)",
        ext="dex",
        exts=frozenset({"dex", "apk"}),
        signatures=((0, b"dex\n"),),
    ),
    ContentClass(
        name="wasm",
        label="a WebAssembly module",
        ext="wasm",
        exts=frozenset({"wasm"}),
        signatures=((0, b"\x00asm"),),
    ),
    ContentClass(
        name="crx",
        label="a Chrome extension package (CRX)",
        ext="crx",
        exts=frozenset({"crx"}),
        signatures=((0, b"Cr24"),),
    ),
    ContentClass(
        name="dmg",
        label="a macOS disk image",
        ext="dmg",
        exts=frozenset({"dmg"}),
        footer=(b"koly",),
    ),
    ContentClass(
        name="xar",
        label="a macOS installer package (XAR)",
        ext="pkg",
        exts=frozenset({"pkg"}),
        signatures=((0, b"xar!"),),
    ),
)

#: Classes whose content is KEPT. The name decides which of two names for the
#: same bytes is right (``zip`` covers both an archive and an OOXML document).
ALLOW_CLASSES: tuple[ContentClass, ...] = (
    ContentClass(
        name="pdf",
        label="a PDF document",
        ext="pdf",
        exts=frozenset({"pdf"}),
        signatures=((0, b"%PDF"),),
    ),
    ContentClass(
        name="zip",
        label="a ZIP container (archive or an OOXML/OpenDocument document)",
        ext="zip",
        exts=frozenset({"zip", "docx", "xlsx", "pptx", "odt", "ods", "odp", "epub", "pages"}),
        signatures=((0, b"PK\x03\x04"), (0, b"PK\x05\x06"), (0, b"PK\x07\x08")),
    ),
    ContentClass(
        name="ole2",
        label="a legacy Office document (OLE2)",
        ext="doc",
        exts=frozenset({"doc", "xls", "ppt", "msg"}),
        signatures=((0, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"),),
    ),
    ContentClass(
        name="gzip",
        label="a gzip archive",
        ext="gz",
        exts=frozenset({"gz", "tgz"}),
        signatures=((0, b"\x1f\x8b"),),
    ),
    ContentClass(
        name="bzip2",
        label="a bzip2 archive",
        ext="bz2",
        exts=frozenset({"bz2", "tbz2"}),
        signatures=((0, b"BZh"),),
    ),
    ContentClass(
        name="xz",
        label="an xz archive",
        ext="xz",
        exts=frozenset({"xz", "txz"}),
        signatures=((0, b"\xfd7zXZ\x00"),),
    ),
    ContentClass(
        name="7z",
        label="a 7-Zip archive",
        ext="7z",
        exts=frozenset({"7z"}),
        signatures=((0, b"7z\xbc\xaf\x27\x1c"),),
    ),
    ContentClass(
        name="rar",
        label="a RAR archive",
        ext="rar",
        exts=frozenset({"rar"}),
        signatures=((0, b"Rar!\x1a\x07"),),
    ),
    ContentClass(
        name="tar",
        label="a TAR archive",
        ext="tar",
        exts=frozenset({"tar"}),
        signatures=((257, b"ustar"),),
    ),
    ContentClass(
        name="png",
        label="a PNG image",
        ext="png",
        exts=frozenset({"png"}),
        signatures=((0, b"\x89PNG\r\n\x1a\n"),),
    ),
    ContentClass(
        name="jpeg",
        label="a JPEG image",
        ext="jpg",
        exts=frozenset({"jpg", "jpeg", "jpe"}),
        signatures=((0, b"\xff\xd8\xff"),),
    ),
    ContentClass(
        name="gif",
        label="a GIF image",
        ext="gif",
        exts=frozenset({"gif"}),
        signatures=((0, b"GIF87a"), (0, b"GIF89a")),
    ),
    ContentClass(
        name="webp",
        label="a WebP image",
        ext="webp",
        exts=frozenset({"webp"}),
        signatures=((8, b"WEBP"),),
    ),
    ContentClass(
        name="bmp",
        label="a BMP image",
        ext="bmp",
        exts=frozenset({"bmp"}),
        signatures=((0, b"BM"),),
    ),
    ContentClass(
        name="tiff",
        label="a TIFF image",
        ext="tiff",
        exts=frozenset({"tif", "tiff"}),
        signatures=((0, b"II*\x00"), (0, b"MM\x00*")),
    ),
    ContentClass(
        name="heic",
        label="a HEIF image",
        ext="heic",
        exts=frozenset({"heic", "heif"}),
        # `ftyp` at 4 with a HEIF brand at 8, the same discrimination
        # `media.sniff_image` makes for the image pipeline.
        signatures=(
            (8, b"heic"),
            (8, b"heix"),
            (8, b"hevc"),
            (8, b"mif1"),
            (8, b"msf1"),
        ),
    ),
    ContentClass(
        name="mp4",
        label="an MP4/QuickTime movie",
        ext="mp4",
        exts=frozenset({"mp4", "m4a", "m4v", "mov"}),
        signatures=((4, b"ftyp"),),
    ),
    ContentClass(
        name="webm",
        label="a WebM/Matroska movie",
        ext="webm",
        exts=frozenset({"webm", "mkv"}),
        signatures=((0, b"\x1aE\xdf\xa3"),),
    ),
    ContentClass(
        name="ogg",
        label="an Ogg stream",
        ext="ogg",
        exts=frozenset({"ogg", "oga", "ogv", "opus"}),
        signatures=((0, b"OggS"),),
    ),
    ContentClass(
        name="wav",
        label="a WAV audio file",
        ext="wav",
        exts=frozenset({"wav"}),
        signatures=((8, b"WAVE"),),
    ),
    ContentClass(
        name="avi",
        label="an AVI movie",
        ext="avi",
        exts=frozenset({"avi"}),
        signatures=((8, b"AVI "),),
    ),
    ContentClass(
        name="flac",
        label="a FLAC audio file",
        ext="flac",
        exts=frozenset({"flac"}),
        signatures=((0, b"fLaC"),),
    ),
    ContentClass(
        name="mp3",
        label="an MP3 audio file",
        ext="mp3",
        exts=frozenset({"mp3"}),
        signatures=((0, b"ID3"), (0, b"\xff\xfb"), (0, b"\xff\xf3"), (0, b"\xff\xf2")),
    ),
    ContentClass(
        name="rtf",
        label="an RTF document",
        ext="rtf",
        exts=frozenset({"rtf"}),
        signatures=((0, b"{\\rtf"),),
    ),
)

#: Extensions that are refused on the NAME alone, over any content.
#:
#: This is the half a content table cannot cover: a Windows shortcut, a `.reg`
#: import, a macOS `.command` or a `.jar` is identified by its extension, and the
#: bytes of several of them (`.url`, `.reg`) are plain text. Content is still
#: checked FIRST, so a `.txt` holding a PE is caught by the class above.
DENY_EXTS: frozenset[str] = frozenset(
    {
        "apk",
        "app",
        "bat",
        "bash",
        "cmd",
        "com",
        "command",
        "cpl",
        "crx",
        "dex",
        "dll",
        "dmg",
        "drv",
        "exe",
        "hta",
        "jar",
        "lnk",
        "msp",
        "mst",
        "msi",
        "ocx",
        "pkg",
        "ps1",
        "psm1",
        "reg",
        "scpt",
        "scr",
        "sh",
        "sys",
        "url",
        "vbe",
        "vbs",
        "wasm",
        "wsf",
        "zsh",
    }
)

#: Extensions that are allowed on the NAME alone, because the format has no
#: signature to match. SVG is deliberately ABSENT: it is script-bearing markup,
#: so it lands in `unknown` — kept, flagged, never opened (design §10.2).
TEXT_EXTS: frozenset[str] = frozenset(
    {"txt", "md", "markdown", "csv", "tsv", "json", "xml", "log", "yaml", "yml"}
)

#: Credential-bearing basenames, matched case-insensitively with `fnmatch`
#: (design §9.2.4). A DENY list rather than an allow list: the operator asked for
#: documents and a long tail (a `.drawio`, an `.stl`, a `.msg`), and the secret
#: classes are enumerable while the safe ones are not.
CREDENTIAL_NAME_PATTERNS: tuple[str, ...] = (
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "id_dsa*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    ".netrc",
    "_netrc",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
    ".pgpass",
    ".my.cnf",
    ".dockercfg",
    ".env",
    ".env.*",
    "credentials",
    "credentials.json",
    "service-account*.json",
    "*.keychain",
    "*.keychain-db",
)

#: Path COMPONENTS (anywhere in the resolved path) that refuse an upload.
CREDENTIAL_COMPONENTS: frozenset[str] = frozenset(
    {".ssh", ".gnupg", ".aws", ".azure", ".kube", "keychains", "secrets"}
)

_SNIFF_HEAD_BYTES = 65_536
#: Enough for `koly` and for a DMG's trailing structures; DMG is the only class
#: whose signature lives at the end.
_SNIFF_FOOTER_BYTES = 4096


# ---------------------------------------------------------------------------
# Name sanitising
# ---------------------------------------------------------------------------

#: C0 and C1 control characters, which are terminal escapes waiting to happen:
#: an allow-listed filename is displayed in an approval prompt and in the
#: session's scrollback, so this is an injection into the operator's card, the
#: same threat `_display_target` exists for on paths.
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
#: Bidi and zero-width overrides: `evil.exe` written with an RTL override
#: DISPLAYS as `evilexe.pdf`, which is the whole point of including them.
_BIDI_ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069]")
_WINDOWS_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _fallback_name(raw: str) -> str:
    """A generated basename for a name that cannot be used as one.

    The design says ``download-<8 hex of sha256(url+stamp)>``; this hashes the
    RAW name instead, because ``safe_name`` is the single door and has no url —
    and it uses FNV-1a, not sha256, because the value is a NAME rather than an
    integrity claim and the extension must compute the same one synchronously
    (``crypto.subtle`` is async, and a hand-rolled SHA-256 in a vendored policy
    module would be a second implementation of a security primitive for no gain).
    Determinism from the input is what makes it testable; a collision between two
    different hostile names is what the per-call uniquifying suffix is for.
    """
    digest = 0x811C9DC5
    for byte in raw.encode("utf-8", "replace"):
        digest = ((digest ^ byte) * 0x01000193) & 0xFFFFFFFF
    return f"{_FALLBACK_STEM}-{digest:08x}"


def _truncate_bytes(name: str, limit: int) -> str:
    """Cut ``name`` to at most ``limit`` UTF-8 bytes, keeping its extension.

    Two steps rather than one: the stem is truncated on a character boundary
    (never mid-codepoint) and the extension is preserved, because a long name
    that loses its extension is a file the user cannot open.
    """
    head, dot, ext = name.rpartition(".")
    if not dot or len(ext) > 12:
        head, ext = name, ""
    # The extension wins the budget when it cannot fit beside a minimal stem.
    if len(ext.encode("utf-8")) > limit - 8:
        return _truncate_bytes(name[: len(name) // 2], limit)
    budget = limit - (len(ext.encode("utf-8")) + 1 if ext else 0)
    while len(head.encode("utf-8")) > budget:
        head = head[: len(head) - 1]
    return f"{head}.{ext}" if ext else head


def _generated_name(raw: str, sniffed_ext: str) -> str:
    """The generated basename, carrying the sniffed extension when there is one.

    The sniffed extension is applied here too, not only on the ordinary path:
    a name like ``"  "`` or ``CON.txt`` sanitises to nothing usable, and the
    content is then the ONLY thing that can name the file. Without this the
    corrected-name rule would silently skip exactly the files whose name was
    least trustworthy.
    """
    fallback = _fallback_name(raw)
    return f"{fallback}.{sniffed_ext}" if sniffed_ext else fallback


def safe_name(raw: str, *, sniffed_ext: str = "") -> str:
    """The sanitised basename a page-supplied (or caller-supplied) name may use.

    Everything here is a hostile-name defence, and each clause is one row of the
    design's enumeration (§10.2): both separator flavours, NUL and C0/C1, the
    bidi/zero-width overrides, `.`/`..`, the Windows reserved stems, trailing
    dots and spaces (which Windows silently drops, making `evil.exe ` and
    `evil.exe` the same file there and different files here), and the byte cap.

    ``sniffed_ext`` corrects the extension to what the CONTENT turned out to be,
    which is the only case where a name is changed rather than merely cleaned.
    """
    name = raw.replace("\\", "/")
    name = name.rsplit("/", 1)[-1]
    name = _CONTROL_RE.sub("", name)
    name = _BIDI_ZERO_WIDTH_RE.sub("", name)
    name = unicodedata.normalize("NFC", name).strip().rstrip(". ")
    if not name or name in {".", ".."}:
        return _generated_name(raw, sniffed_ext)
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    if stem.lower() in _WINDOWS_RESERVED_STEMS:
        # `CON.txt` is as reserved as `CON` on Windows, which is why the STEM is
        # what is tested and the whole name is what is replaced.
        return _generated_name(raw, sniffed_ext)
    if sniffed_ext and ext.lower() != sniffed_ext.lower():
        stem = stem or name
        ext = sniffed_ext
    return _truncate_bytes(f"{stem}.{ext}" if ext else stem, MAX_NAME_BYTES)


#: Declared types that carry no information, so quoting one would read as a
#: signal the server never sent. Compared lowercased.
_GENERIC_MIMES = frozenset(
    {"", "application/octet-stream", "binary/octet-stream", "application/binary", "unknown"}
)
#: The declared type lands in model-facing text AND in an audit row, so it is
#: capped like a name is; 80 bytes is far past any real `Content-Type`.
MAX_MIME_BYTES = 80
#: The host's read-back marker is a composed SENTENCE rather than a type: the
#: extension builds `"unavailable — the read-back failed (" + describeError(e) +
#: ")"`, and `describeError` caps the ERROR TEXT at 120 *characters* — so an
#: honest marker reaches ~158-159 bytes for an ASCII error, and more when the
#: error text is multibyte (120 characters is not 120 bytes). This ceiling sits
#: above that composed bound with headroom, and :func:`readback_label` marks any
#: clip visibly, because no fixed ceiling can bound an honest marker whose message
#: is multibyte (review round 3's MINOR-1 / QA's Q-2).
MAX_READBACK_BYTES = 200
#: Appended when a host-supplied diagnostic had to be clipped, so a truncated
#: value is never read as the whole one (review round 3's Q-2).
CLIP_MARK = "\u2026"


def redact_name(name: str) -> str:
    """A refused file's name, for the audit row: first character plus an ellipsis.

    The model is told the real name (its context is the user's own transcript);
    the audit file is not, because a log that maps where the secrets are is a
    second copy of the thing the refusal exists to protect (design §9.4).
    """
    return f"{name[:1]}\u2026" if name else ""


def _outside_text(raw: str) -> str:
    """A string from OUTSIDE, with the characters that would let it lie removed.

    One door for every host-supplied label that is interpolated into
    model-facing text and into an audit row, so there is a single answer to "is
    this safe to print": C0/C1 controls (a terminal escape, and a `\r\n` that
    grows the tool result by a line the host chose) and the bidi/zero-width
    overrides (`evil.exe` written with an RTL override DISPLAYS as
    `evilexe.pdf`). Removed rather than replaced by a space: these are not
    whitespace to preserve, and a substitution would insert a separator the host
    never sent.
    """
    return _BIDI_ZERO_WIDTH_RE.sub("", _CONTROL_RE.sub("", str(raw or ""))).strip()


def declared_mime_label(raw: str) -> str:
    """The host's declared `Content-Type`, sanitised, or "" when it says nothing.

    A HINT and never a decision: §5.3 keeps the filesystem as the truth, so this
    is only quoted in the rename sentence and carried into the audit row. It is
    sanitised because it is a string from OUTSIDE landing in the transcript and
    in the log — the class `safe_name` exists for, and the same one that makes
    `redact_name` redact — and dropped when it is generic, so the copy can never
    imply a signal the server did not send.
    """
    label = _outside_text(raw)
    if label.lower() in _GENERIC_MIMES:
        return ""
    return _truncate_bytes(label, MAX_MIME_BYTES)


def _clip_text(text: str, limit: int) -> str:
    """``text`` cut to at most ``limit`` UTF-8 bytes, keeping its HEAD, marked.

    Deliberately not :func:`_truncate_bytes`: that one preserves a filename's
    EXTENSION, which a diagnostic sentence does not have — run over a marker it
    keeps a fragment of the tail and drops the middle, so
    `... failed (timeout waiting for promise.js)` would come back as
    `... promise.js`. A sentence loses its tail instead, on a character boundary
    (never mid-codepoint), and says it was cut.
    """
    if len(text.encode("utf-8")) <= limit:
        return text
    room = max(limit - len(CLIP_MARK.encode("utf-8")), 0)
    return text.encode("utf-8")[:room].decode("utf-8", "ignore") + CLIP_MARK


def readback_label(raw: str) -> str:
    """The host's read-back marker, sanitised and capped, or "" when absent.

    The same discipline as :func:`declared_mime_label`, for the same reason: it
    is a string from outside that is interpolated into the model-facing note and
    into the audit row's `reason`, and it is the one field of the upload result a
    HOST chooses the length and content of. The extension caps the error TEXT it
    embeds, not the marker it composes around it, so the harness is the boundary
    that must not take either on trust (review round 2, R7) — and a clip is shown
    rather than silent (review round 3's MINOR-1 / QA's Q-2).
    """
    return _clip_text(_outside_text(raw), MAX_READBACK_BYTES)


# ---------------------------------------------------------------------------
# Content classification
# ---------------------------------------------------------------------------


def _match(content_class: ContentClass, head: bytes, footer: bytes) -> bool:
    for offset, magic in content_class.signatures:
        if head[offset : offset + len(magic)] == magic:
            return True
    return any(footer.endswith(magic) for magic in content_class.footer)


def sniff(head: bytes, footer: bytes = b"") -> ContentClass | None:
    """The content class of these bytes, or ``None`` when nothing matched.

    Deny classes are tested FIRST, so a class list collision (a ZIP-shaped
    container that is also, say, a CRX) resolves to the refusal.
    """
    for content_class in DENY_CLASSES:
        if _match(content_class, head, footer):
            return content_class
    for content_class in ALLOW_CLASSES:
        if _match(content_class, head, footer):
            return content_class
    return None


def classify_bytes(
    raw_name: str, head: bytes, *, declared_mime: str = "", footer: bytes = b""
) -> Verdict:
    """The verdict for content and a name that are already in memory.

    Split out of :func:`classify_download` so the conformance fixture can be
    re-derived by the generator without writing files (§10.4) — and so the
    classifier is exercised on exact bytes rather than on whatever a temp
    filesystem produced.

    A deny REASON is the rule and nothing else — no "refused and deleted:"
    prefix and no claim about the entry (review round 2, N7). Only the caller
    knows whether the entry was actually removed (it is the one that deletes), so
    only the caller states the verdict and the outcome, in the same words for
    every refusal reason (`builtin._delete_outcome`). A reason that pre-empted it
    could only be right half the time, which is what made the tail of the refusal
    sentence asymmetric between the rules.
    """
    # An unreadable or empty artifact is not a deliverable, and a zero-byte file
    # would otherwise fall through every content check to `unknown`, i.e. be
    # kept and reported as a possible document.
    if not head:
        return Verdict(
            "deny",
            f"{safe_name(raw_name)} is empty (0 bytes) — an empty download is not a file "
            "you can use",
            "",
            safe_name(raw_name),
        )

    content_class = sniff(head, footer)
    if content_class is not None and content_class in DENY_CLASSES:
        return Verdict(
            "deny",
            f"the file at {safe_name(raw_name)} is {content_class.label}; nothing "
            "executable is ever kept",
            content_class.name,
            safe_name(raw_name),
        )

    name_ext = _name_ext(raw_name)
    if name_ext and name_ext in DENY_EXTS:
        # Name-only refusal for the formats a content table cannot cover (a
        # `.url` or a `.reg` is text). It fires even when the content matched an
        # ALLOW class: a `.dmg` that is really a PDF is still a file whose name
        # promises an installer.
        return Verdict(
            "deny",
            f"{safe_name(raw_name)} is an executable or script type ('{name_ext}'); "
            "nothing was saved",
            content_class.name if content_class is not None else "",
            safe_name(raw_name),
        )

    if content_class is not None:
        if name_ext in content_class.exts:
            return Verdict("allow", "", content_class.name, safe_name(raw_name))
        corrected = safe_name(raw_name, sniffed_ext=content_class.ext)
        # The server's declared type is quoted when it has one to quote (design
        # §7.4's "the server called it …"): the rename is the moment the model is
        # told which signals disagreed, and the declared type is the one signal
        # this function otherwise drops on the floor. It never decides anything —
        # content does (§5.3) — so a lying `Content-Type` changes wording, not
        # outcome.
        declared = declared_mime_label(declared_mime)
        server_said = f"the server said '{declared}'; " if declared else ""
        return Verdict(
            "allow",
            f"saved as '{corrected}' (the name said '{name_ext or '(none)'}'; {server_said}"
            f"the content is {content_class.label})",
            content_class.name,
            corrected,
        )

    if name_ext in TEXT_EXTS:
        # No signature exists for plain text, Markdown, CSV, JSON or XML, so the
        # name is the only signal there is. `sniffed` stays "" — claiming a
        # content class here would be a lie the audit row would carry.
        return Verdict("allow", "", "", safe_name(raw_name))

    return Verdict(
        "unknown",
        "kept but unverified: the content matched no known signature, so it has NOT been "
        "opened or executed and its type is unknown",
        "",
        safe_name(raw_name),
    )


def _name_ext(raw: str) -> str:
    """The lowercased extension of the SANITISED name, without the dot.

    Applied to the sanitised form on purpose: the class rule must see the name
    that will actually exist on disk, so `report.pdf.exe` is an executable (design
    §10.2's last sanitiser clause).
    """
    name = safe_name(raw).lower()
    _, dot, ext = name.rpartition(".")
    if not dot or len(ext) > 12 or "/" in ext:
        return ""
    return ext


def classify_download(path: Path, *, declared_mime: str = "", policy: Policy = DEFAULT) -> Verdict:
    """Classify a file that has LANDED, reading it from disk.

    The host's word is a hint; the filesystem is the truth (design §5.3). This
    reads only the head and a small footer, never the whole file.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return Verdict(
            "deny",
            f"{safe_name(path.name)} could not be read ({exc.strerror})",
            "",
            safe_name(path.name),
        )
    if size > policy.download_max_bytes:
        return Verdict(
            "deny",
            f"{safe_name(path.name)} is {size} bytes, over the "
            f"{policy.download_max_bytes} byte limit",
            "",
            safe_name(path.name),
        )
    try:
        with path.open("rb") as handle:
            head = handle.read(_SNIFF_HEAD_BYTES)
            if size > _SNIFF_HEAD_BYTES:
                handle.seek(max(0, size - _SNIFF_FOOTER_BYTES))
                footer = handle.read(_SNIFF_FOOTER_BYTES)
            else:
                footer = head
    except OSError as exc:
        return Verdict(
            "deny",
            f"{safe_name(path.name)} could not be read ({exc.strerror})",
            "",
            safe_name(path.name),
        )
    return classify_bytes(path.name, head, declared_mime=declared_mime, footer=footer)


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------


def _credential_refusal(path: Path) -> str:
    """Why this resolved path may not be uploaded, or "" when it may.

    Both halves are checked because either alone is escapable: a basename can be
    innocent inside a `secrets/` directory, and a path can leave `~/.ssh/` by
    being copied out. The BASENAME patterns are globs, so `id_rsa.pub` is as
    refused as `id_rsa` — a public key names its private twin's location.
    """
    lowered = path.name.lower()
    for pattern in CREDENTIAL_NAME_PATTERNS:
        if fnmatch.fnmatchcase(lowered, pattern):
            return f"refused: '{path.name}' matches the credential deny-list ({pattern})"
    for component in path.parts:
        if component.lower() in CREDENTIAL_COMPONENTS:
            return (
                f"refused: '{path.name}' is inside a '{component}' directory, which holds "
                "credentials"
            )
    return ""


def check_upload(raw: str, *, cwd: str, policy: Policy = DEFAULT) -> tuple[Path | None, str]:
    """``(resolved_path, "")`` or ``(None, reason)`` — one call site, one seam.

    Unconditional: it consults no approval policy and no tier, because the
    adversary it exists for is a confused-deputy agent whose call may be
    auto-approved (§9.1, §7.3). Everything is judged on the RESOLVED path, so a
    symlink is judged by its target — which is what closes the "a symlink named
    `handout.pdf` pointing at `~/.ssh/id_rsa`" case.
    """
    if not (raw or "").strip():
        return None, "refused: no path given"
    raw_path = Path(raw.strip()).expanduser()
    if not raw_path.is_absolute():
        raw_path = Path(cwd) / raw_path
    try:
        # strict: the file must exist, and resolving first is the rule the whole
        # check list is applied to (design §9.2's step 2).
        resolved = raw_path.resolve(strict=True)
    except OSError:
        return None, f"refused: '{safe_name(str(raw))}' does not exist"
    config_root = config_dir().resolve()
    if is_within(resolved, config_root):
        # Unconditional and first: this is where the encrypted secret store and
        # config.yml live (§9.2's step 1, "the cheapest high-value rule").
        return None, (
            "refused: that file is inside Local Operator's own config directory, which holds "
            "credentials"
        )
    try:
        mode = resolved.stat().st_mode
    except OSError:
        return None, f"refused: '{safe_name(resolved.name)}' could not be read"
    if not stat.S_ISREG(mode):
        return None, (
            f"refused: '{safe_name(resolved.name)}' is not a regular file "
            "(directories, devices and sockets cannot be attached)"
        )
    credential = _credential_refusal(resolved)
    if credential:
        return None, credential
    try:
        size = resolved.stat().st_size
    except OSError:
        return None, f"refused: '{safe_name(resolved.name)}' could not be read"
    if size > policy.upload_max_bytes:
        return None, (
            f"refused: '{safe_name(resolved.name)}' is {size} bytes, over the "
            f"{policy.upload_max_bytes} byte limit"
        )
    if size == 0:
        return None, f"refused: '{safe_name(resolved.name)}' is empty (0 bytes)"
    return resolved, ""


def is_within(path: Path, root: Path) -> bool:
    """Whether ``path`` is ``root`` or lives under it, on normalised paths.

    Public, and the ONE spelling of containment: the upload gate applies it to
    the config root and the download half applies it to the quarantine root, and
    a second private copy in the caller is how the two would drift apart (review
    round 1, N4). Both paths must already be resolved — this compares normalised
    paths, it does not resolve them.
    """
    return path == root or root in path.parents


# ---------------------------------------------------------------------------
# The quarantine root, and the audit trail
# ---------------------------------------------------------------------------

DOWNLOADS_DIRNAME = "browser/downloads"
AUDIT_FILENAME = "audit.jsonl"


def downloads_root(root: Path | None = None) -> Path:
    """``<config_dir>/browser/downloads`` — pure path arithmetic, creates nothing."""
    return (root or config_dir()) / DOWNLOADS_DIRNAME


def session_dir(session_id: str, *, root: Path | None = None, stamp: str | None = None) -> Path:
    """``<config>/browser/downloads/<stamp>-<session8>/``, created 0700.

    A directory per session because that is what makes the before/after DIRECTORY
    DIFF a sound way to learn what a call landed with several sessions on one
    machine (§5.3), and a timestamp in the name so a later `ls` explains itself.

    UNIQUE, not merely stamped: the stamp has one-second resolution, so two calls
    inside the same second would otherwise share a directory — and then the diff
    for the second call would be silently compared against the first call's
    files. A `-2`, `-3`... suffix (the same rule the content-corrected rename
    uses) keeps each call's directory its own without changing the documented
    name shape.
    """
    session8 = "".join(ch for ch in (session_id or "nosession"))[:8] or "nosession"
    label = f"{stamp or time.strftime('%Y%m%d-%H%M%S')}-{_safe_component(session8)}"
    parent = downloads_root(root)
    parent.mkdir(parents=True, exist_ok=True)
    os.chmod(parent, 0o700)
    directory = parent / label
    index = 2
    while directory.exists():
        directory = parent / f"{label}-{index}"
        index += 1
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


def _safe_component(value: str) -> str:
    """A filename-safe, identity-safe component for a directory name."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)[:32] or "session"


#: What a KEPT artifact must be, and what §4.1 promises it is. The host writes
#: the bytes and therefore owns the mode it lands with — an Electron/Chromium
#: write lands 0644 by umask — so the harness tightens it once the file is the
#: one it is going to report. Best-effort: a chmod that fails must not turn a
#: saved file into a failed call, exactly like the audit write (§10.5).
PRIVATE_FILE_MODE = 0o600


def chmod_private(path: Path) -> bool:
    """Tighten a kept artifact to 0600 (§4.1). ``False`` if it could not be done.

    Bounded by the 0700 session directory either way, so this is defence in
    depth rather than the only thing standing between a page's download and
    another local user; it is still done, because the design states it and a
    claim the code does not enforce is the defect class review round 1 caught.

    The mode change must land on the ENTRY this call reports, which is why a
    symlink entry takes ``lchmod`` and is refused where the platform has none
    (review round 2, N8): ``os.chmod`` FOLLOWS the link, so on an in-root symlink
    artifact it tightened the target — a file this call neither landed nor names,
    and one the page chose. The containment check bounds the target to the
    session root, so this was never an escape; it is the same "the artifact it is
    about to report" rule R1 applies to the delete.

    ``False`` therefore means two different things, and the caller reports both:
    the mode could not be set at all, or this platform cannot set a SYMLINK's own
    mode (Linux has no ``lchmod``; macOS does). Neither is a reason to fall back
    to ``chmod`` — that would tighten whatever the link points at, which is the
    bug this branch exists for.
    """
    if path.is_symlink():
        lchmod = getattr(os, "lchmod", None)
        if lchmod is None:
            # Linux. The entry keeps the mode the host wrote; the 0700 session
            # directory is still the bound, and the caller says so in the result.
            return False
        try:
            lchmod(path, PRIVATE_FILE_MODE)
            return True
        except OSError:
            return False
    try:
        os.chmod(path, PRIVATE_FILE_MODE)
        return True
    except OSError:
        return False


def session_bytes(session_id: str, *, root: Path | None = None) -> int:
    """Total bytes in every stamped download directory belonging to this session.

    Per SESSION rather than per call, because each call gets its own stamped
    directory (design §4.1) — so the quota has to span them, and the session id is
    what they share. Best-effort like :func:`dir_size`: this is a ceiling, and a
    ceiling that can fail a turn is worse than one that is occasionally optimistic.
    """
    session8 = "".join(ch for ch in (session_id or "nosession"))[:8] or "nosession"
    component = _safe_component(session8)
    total = 0
    try:
        # Both spellings: a directory whose stamp collided inside one second gets a
        # `-2` suffix (see `session_dir`), and it belongs to the same session.
        directories = list(downloads_root(root).glob(f"*-{component}")) + list(
            downloads_root(root).glob(f"*-{component}-*")
        )
    except OSError:
        return total
    for entry in directories:
        try:
            if entry.is_dir():
                total += dir_size(entry)
        except OSError:
            continue
    return total


def dir_size(directory: Path) -> int:
    """Total bytes in a directory tree, or 0 when it does not exist.

    Best-effort and never raising: this is a quota check, and a quota check that
    can fail a turn is worse than a quota that is occasionally optimistic.
    """
    total = 0
    try:
        for entry in directory.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


#: The audit writer logs ONE line per process on failure, not one per call: a
#: full or unwritable audit path is a standing condition, and a per-call warning
#: would flood the session log it is supposed to explain.
_audit_warned = False


def audit(record: Mapping[str, Any], *, root: Path | None = None) -> None:
    """Append one JSONL decision row, 0600, best-effort.

    Nothing here may raise into a turn (§10.5): the file's bytes are the user's,
    the audit row is ours, and a failed append must not cost them the download.
    """
    global _audit_warned
    try:
        path = downloads_root(root) / AUDIT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        # 0700 on the DIRECTORY too, not only on the file: a 0755 parent would
        # let any local user LIST the download names the row's redaction exists
        # to keep out of the log.
        os.chmod(path.parent, 0o700)
        line = json.dumps({"ts_ms": int(time.time() * 1000), **record}, default=str)
        # O_APPEND + 0600 at creation: append-only by construction, private by
        # construction, and never truncated by a concurrent writer.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            os.close(descriptor)
            raise
    except Exception as exc:  # noqa: BLE001 - the audit may never break a turn
        if not _audit_warned:
            _audit_warned = True
            logger.warning("browser file audit row could not be written: %s", exc)


def new_call_id() -> str:
    """An id shared by the host's row and Python's row for one call (§10.5)."""
    return f"bf-{secrets.token_hex(6)}"


def stat_fact(name: str, path: Path, *, declared_mime: str = "") -> dict[str, Any]:
    """A ``FileFact``'s Python-computed half for a file on disk (§6.1).

    ``sha256`` is computed HERE, never taken from a host: a host that reports a
    hash it did not compute is a host whose word we would be trusting, which is
    the property the post-hoc verification exists to remove.
    """
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        return {
            "name": name,
            "path": str(path),
            "bytes": 0,
            "mime": declared_mime,
            "sniffed": "",
            "sha256": "",
        }
    return {
        "name": name,
        "path": str(path),
        "bytes": size,
        "mime": declared_mime,
        "sniffed": "",
        "sha256": digest.hexdigest(),
    }


def snapshot(directory: Path) -> dict[str, int]:
    """``{name: size}`` for a directory: one half of the before/after diff (§5.3)."""
    found: dict[str, int] = {}
    try:
        for entry in directory.iterdir():
            try:
                if entry.is_file():
                    found[entry.name] = entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return found
    return found


# ---------------------------------------------------------------------------
# The conformance fixture (hand-written expectations, re-derived by gen_ts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConformanceCase:
    """One ``(name, content) → expected verdict`` row of the shared fixture.

    The EXPECTATION is hand-written. That is what makes it a gate: `gen_ts`
    re-derives the verdict from :func:`classify_bytes` and FAILS generation if
    the two disagree, so a classifier change that alters a decision cannot reach
    the extension's tables without someone editing this table on purpose.
    """

    name: str
    head: bytes
    expected_kind: DownloadClass
    expected_sniffed: str
    expected_safe_name: str
    declared_mime: str = ""
    #: Bytes that close the file. Only DMG needs one (its signature is `koly`,
    #: written LAST), so it is a fixture field rather than a second head.
    footer: bytes = b""
    #: The sniffed extension the EXPECTED name was corrected with, or "" when the
    #: name is reported uncorrected. Explicit rather than inferred, because the
    #: TypeScript half must replay exactly this call and cannot tell from the
    #: verdict alone which of the two produced the name it is comparing against.
    sniffed_ext_for_name: str = ""
    #: Whether the sanitised name's own extension is on the deny list. The
    #: TypeScript policy port can compute exactly this much, so it is the half of
    #: the fixture the extension's own test can reproduce (§10.4).
    name_is_deny_listed: bool = False


_PDF_HEAD = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n"
_ZIP_HEAD = b"PK\x03\x04\x14\x00\x00\x00\x08\x00"

CONFORMANCE_CASES: tuple[ConformanceCase, ...] = (
    # --- content decides, and it wins in the direction that matters ----------
    ConformanceCase(
        name="receipt.pdf",
        head=_PDF_HEAD,
        expected_kind="allow",
        expected_sniffed="pdf",
        expected_safe_name="receipt.pdf",
    ),
    ConformanceCase(
        # A PDF served as a ZIP: kept, renamed, and the rename is reported.
        name="invoice.zip",
        head=_PDF_HEAD,
        declared_mime="application/zip",
        expected_kind="allow",
        expected_sniffed="pdf",
        expected_safe_name="invoice.pdf",
        sniffed_ext_for_name="pdf",
    ),
    ConformanceCase(
        # A PE executable under an innocent name: content wins, deleted.
        name="holiday-photo.jpg",
        head=b"MZ\x90\x00\x03\x00\x00\x00",
        expected_kind="deny",
        expected_sniffed="pe",
        expected_safe_name="holiday-photo.jpg",
    ),
    ConformanceCase(
        # The reverse asymmetry: an .exe NAME over PDF bytes is a PDF.
        name="report.pdf.exe",
        head=_PDF_HEAD,
        expected_kind="deny",
        expected_sniffed="pdf",
        expected_safe_name="report.pdf.exe",
        name_is_deny_listed=True,
    ),
    ConformanceCase(
        name="setup.exe",
        head=b"MZ\x90\x00\x03\x00\x00\x00",
        expected_kind="deny",
        expected_sniffed="pe",
        expected_safe_name="setup.exe",
        name_is_deny_listed=True,
    ),
    ConformanceCase(
        name="installer.dmg",
        head=b"\x00" * 512,
        footer=b"\x00" * 120 + b"koly",
        expected_kind="deny",
        expected_sniffed="dmg",
        expected_safe_name="installer.dmg",
        name_is_deny_listed=True,
    ),
    ConformanceCase(
        # An ELF named as a document: the CLASS is what is reported.
        name="notes.txt",
        head=b"\x7fELF\x02\x01\x01\x00",
        expected_kind="deny",
        expected_sniffed="elf",
        expected_safe_name="notes.txt",
    ),
    # --- content decides, and the name is corrected ------------------------
    ConformanceCase(
        name="deck.pptx",
        head=_ZIP_HEAD,
        expected_kind="allow",
        expected_sniffed="zip",
        expected_safe_name="deck.pptx",
    ),
    ConformanceCase(
        name="archive.zip",
        head=_ZIP_HEAD,
        expected_kind="allow",
        expected_sniffed="zip",
        expected_safe_name="archive.zip",
    ),
    ConformanceCase(
        # A ZIP under a name with no extension at all: allowed, named for what
        # it is.
        name="handout",
        head=_ZIP_HEAD,
        expected_kind="allow",
        expected_sniffed="zip",
        expected_safe_name="handout.zip",
        sniffed_ext_for_name="zip",
    ),
    ConformanceCase(
        name="photo.png",
        head=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
        expected_kind="allow",
        expected_sniffed="png",
        expected_safe_name="photo.png",
    ),
    ConformanceCase(
        name="scan.JPEG",
        head=b"\xff\xd8\xff\xe0\x00\x10JFIF",
        expected_kind="allow",
        expected_sniffed="jpeg",
        expected_safe_name="scan.JPEG",
    ),
    # --- the name is the only signal for text ------------------------------
    ConformanceCase(
        name="notes.md",
        head=b"# Notes\n\nnothing executable here\n",
        expected_kind="allow",
        expected_sniffed="",
        expected_safe_name="notes.md",
    ),
    ConformanceCase(
        # SVG is script-bearing markup, so it is NOT on the text allow-list:
        # kept, flagged unverified, never opened.
        name="logo.svg",
        head=b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
        expected_kind="unknown",
        expected_sniffed="",
        expected_safe_name="logo.svg",
    ),
    ConformanceCase(
        name="payload.bin",
        head=b"\x00\x01random bytes\xff",
        expected_kind="unknown",
        expected_sniffed="",
        expected_safe_name="payload.bin",
    ),
    # --- an empty or unreadable artifact is not a deliverable --------------
    ConformanceCase(
        name="empty.pdf",
        head=b"",
        expected_kind="deny",
        expected_sniffed="",
        expected_safe_name="empty.pdf",
    ),
    # --- hostile names (design §10.2's enumerations) -----------------------
    ConformanceCase(
        name="../../.ssh/authorized_keys",
        head=_PDF_HEAD,
        expected_kind="allow",
        expected_sniffed="pdf",
        expected_safe_name="authorized_keys.pdf",
        sniffed_ext_for_name="pdf",
    ),
    ConformanceCase(
        name="..\\..\\windows\\evil.txt",
        head=b"plain text\n",
        expected_kind="allow",
        expected_sniffed="",
        expected_safe_name="evil.txt",
    ),
    ConformanceCase(
        # Bidi overrides do not change what is stored, only what a human reads:
        # this DISPLAYS as `photojpg.exe` reversed into `exe.jpg`-looking text.
        name="photo\u202ejpg.exe",
        head=b"MZ\x90\x00",
        expected_kind="deny",
        expected_sniffed="pe",
        expected_safe_name="photojpg.exe",
        name_is_deny_listed=True,
    ),
    ConformanceCase(
        # The mirror case, and the reason CONTENT is the primary signal: a
        # zero-width isolate makes this read as `payload.pdf`, and the extension
        # after sanitising IS `pdf` — so the name check would pass it. The PE
        # class inside is what refuses it.
        name="payload.exe\u2066.pdf",
        head=b"MZ\x90\x00",
        expected_kind="deny",
        expected_sniffed="pe",
        expected_safe_name="payload.exe.pdf",
        name_is_deny_listed=False,
    ),
    ConformanceCase(
        # A Windows reserved stem is replaced wholesale, so the name carries no
        # extension and nothing about it can be trusted — not even that it is
        # text, which is why this lands in `unknown` rather than `allow`.
        name="CON.txt",
        head=b"plain text\n",
        expected_kind="unknown",
        expected_sniffed="",
        expected_safe_name=_fallback_name("CON.txt"),
    ),
    ConformanceCase(
        name="report . .pdf",
        head=_PDF_HEAD,
        expected_kind="allow",
        expected_sniffed="pdf",
        expected_safe_name="report . .pdf",
    ),
    ConformanceCase(
        name="  ",
        head=_PDF_HEAD,
        expected_kind="allow",
        expected_sniffed="pdf",
        expected_safe_name=_fallback_name("  ") + ".pdf",
        sniffed_ext_for_name="pdf",
    ),
)


def tables_for_ts() -> dict[str, Any]:
    """Everything the TypeScript half needs, as plain JSON-able data (§10.4).

    One function rather than a walk over module globals, so the emitted set is a
    decision someone made rather than whatever happened to be public.
    """

    def class_row(item: ContentClass) -> dict[str, Any]:
        return {
            "name": item.name,
            "label": item.label,
            "ext": item.ext,
            "exts": sorted(item.exts),
            "deny": item in DENY_CLASSES,
        }

    return {
        "denyClasses": [class_row(item) for item in DENY_CLASSES],
        "allowClasses": [class_row(item) for item in ALLOW_CLASSES],
        "denyExts": sorted(DENY_EXTS),
        "textExts": sorted(TEXT_EXTS),
        "credentialNamePatterns": list(CREDENTIAL_NAME_PATTERNS),
        "credentialComponents": sorted(CREDENTIAL_COMPONENTS),
        "caps": {
            "downloadMaxBytes": DOWNLOAD_MAX_BYTES,
            "downloadMaxFilesPerCall": DOWNLOAD_MAX_FILES_PER_CALL,
            "downloadMaxTotalBytesPerSession": DOWNLOAD_MAX_TOTAL_BYTES_PER_SESSION,
            "downloadTimeoutS": DOWNLOAD_TIMEOUT_S,
            "downloadTimeoutMaxS": DOWNLOAD_TIMEOUT_MAX_S,
            "uploadMaxBytes": UPLOAD_MAX_BYTES,
            "uploadMaxFiles": UPLOAD_MAX_FILES,
        },
        "cases": [
            {
                "name": case.name,
                "headBase64": _b64(case.head),
                "declaredMime": case.declared_mime,
                "kind": case.expected_kind,
                "sniffed": case.expected_sniffed,
                "safeName": case.expected_safe_name,
                "safeNameSniffedExt": case.sniffed_ext_for_name,
                "nameIsDenyListed": case.name_is_deny_listed,
            }
            for case in CONFORMANCE_CASES
        ],
    }


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def fixture_mismatches() -> list[str]:
    """Cases whose hand-written expectation the classifier no longer reproduces.

    Called by `gen_ts` BEFORE it writes anything, so a policy change that alters
    a decision fails generation instead of silently shipping a table whose
    expectations describe the old behaviour (§10.4).
    """
    problems: list[str] = []
    for case in CONFORMANCE_CASES:
        verdict = classify_bytes(
            case.name, case.head, declared_mime=case.declared_mime, footer=case.footer
        )
        if verdict.kind != case.expected_kind:
            problems.append(f"{case.name!r}: kind {verdict.kind} != {case.expected_kind}")
        if verdict.sniffed != case.expected_sniffed:
            problems.append(
                f"{case.name!r}: sniffed {verdict.sniffed!r} != {case.expected_sniffed!r}"
            )
        if verdict.safe_name != case.expected_safe_name:
            problems.append(
                f"{case.name!r}: safe_name {verdict.safe_name!r} != {case.expected_safe_name!r}"
            )
        # The TypeScript half replays exactly this call, so the fixture has to
        # name the correction explicitly — and proving it here means the shared
        # table cannot describe a sanitiser call the two languages disagree about.
        replayed = safe_name(case.name, sniffed_ext=case.sniffed_ext_for_name)
        if replayed != case.expected_safe_name:
            problems.append(
                f"{case.name!r}: safe_name({case.name!r}, sniffed_ext="
                f"{case.sniffed_ext_for_name!r}) is {replayed!r}, not "
                f"{case.expected_safe_name!r}"
            )
        if (_name_ext(case.name) in DENY_EXTS) != case.name_is_deny_listed:
            problems.append(
                f"{case.name!r}: name_is_deny_listed {_name_ext(case.name) in DENY_EXTS} "
                f"!= {case.name_is_deny_listed}"
            )
    return problems
