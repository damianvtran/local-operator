"""The directory-anchored listing grammar: bare file names listed under a rooted directory.

THE IDIOM. An agent closing a turn names its deliverables as a directory plus a list of bare
names ("Files are in `~/Documents/Radient/`:\n- Report.pdf\n- Report (editable).docx — notes").
The full path is never spelled out, so a path-shaped prose scan cannot see it, and the command
that wrote the files often used shell variables (`"$D/$N.pdf"`) a bash grammar may not resolve.

ONE PARSER FOR EVERY PYTHON READER. The session Files ledger
(:mod:`local_operator.session_files.scan`) admits these names as assistant-prose mentions, and
a later reader (the Highlights pre-filter, PR #3044) reads a final answer through the same
grammar. Each imports :func:`directory_listing` from here: a second copy is how the inventory
and a callout block come to disagree about which files an answer named.

THE CONTRACT is the desktop renderer's Files-panel walker, so the ledger and the renderer's
transcript links / old-daemon fallback list the same files. It is a PORT, not a shared
implementation, of `src/renderer/src/features/chat/utils/link-grammar.ts` in local-operator-ui
on branch `fix/mentioned-files-dir-listing` (PR #1451): `listingDirectoryIn` :1117,
`listedTargetFor` :1165, `LISTED_ANNOTATION_SEPARATORS` :1214, `listedEntryFor` :1281,
`codeLinesOf` :1371, `itemText` :1438, `listedTargetsIn` :1475, and the directory token's
scanner `targetsIn` :825 with `normalizeCandidate` :395 / `normalizeFileUrl` :487. The
line numbers and the copied extension set were read at that branch's `c1798b33b0`, which is a
POINT-IN-TIME reference and not a pin: the branch is unreleased work that has already moved on
(`12f1e462d0` when this was written), so treat any commit id here as "what I last compared
against", never as the contract. WHAT GUARANTEES PARITY is the tests, not this paragraph or that
commit: tests/unit/session_files/test_dir_listing.py carries the renderer suite's cases, its
round-1 PARITY table, the 13 differential cases review round 1 found against `a895cf571b`, and
`test_known_extensions_match_the_renderer_source_when_checked_out`, which re-reads the
renderer's own `file-kind.ts` from a checkout. A renderer change on that branch is a change
here, re-checked by running both on the same inputs. The renderer is evaluated under
`MENTION_POLICY` plus an evidence oracle (the panel's known-extension rule; the ledger's stat
is the oracle). Two deliberate deltas, both from #1451's round-2 review and mirrored ahead of
it: an ordered list straight under an introducer must start with the LITERAL marker `1.`/`1)`
(`01.` is refused), and an item line inside a blockquote is never an item.

Why so narrow: a bare name ANYWHERE else in prose is a guess about which directory was meant,
and a wrong guess is a row pointing at nothing. The introducer (a rooted directory, then ":")
is what turns the guess into a statement; resolving only the IMMEDIATE list is what keeps a
later, unrelated bullet list from inheriting it. This module returns UNRESOLVED candidates
only: each caller's own stat is the renderer's evidence oracle (every listed name, whole item
or annotated head, is admitted only when `<dir>/<name>` exists as a regular file), and its own
denylist and dedupe apply.

JAVASCRIPT SEMANTICS, ported explicitly. JS `\\s`, `trim()` and `.` differ from Python's: JS
whitespace includes U+FEFF and excludes U+001C-U+001F/U+0085, and JS `.` stops at U+2028/U+2029
(so a list line holding one is not an item there). Every class below spells the JS set out;
Python's own `\\s`/`strip()`/`.` would silently diverge on those code points. String lengths
are code points here and UTF-16 units there, which only matters past the 4096-char cap.

PURITY. No filesystem, no clock, stdlib only, so importing it from any reader costs nothing.
"""

from __future__ import annotations

import re
from typing import Final
from urllib.parse import unquote, urlsplit

# -- JS character classes -------------------------------------------------------------------
#: ECMAScript WhiteSpace + LineTerminator: exactly what JS `\s` and `String.prototype.trim` use.
_JS_SPACE_CHARS: Final = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009"
    "\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)
_S: Final = "[" + re.escape(_JS_SPACE_CHARS) + "]"
#: JS `.` without the `s` flag: anything but a line terminator.
_DOT: Final = "[^\n\r\u2028\u2029]"


def _js_trim(text: str) -> str:
    return text.strip(_JS_SPACE_CHARS)


def _js_trim_start(text: str) -> str:
    return text.lstrip(_JS_SPACE_CHARS)


# -- the renderer's known-extension set -----------------------------------------------------
#: The renderer's `KNOWN_EXTENSIONS` (src/renderer/src/features/chat/utils/file-kind.ts at
#: c1798b33b0), copied rather than borrowed from :func:`local_operator.library.classify.
#: is_known_extension`. WHY A LOCAL SET: review round 1 (F4) diffed the two and found
#: `zip gz tar 7z rar key odt odp xml conf env jsonc ...` only in the renderer's and `ipynb
#: jsonl numbers ... makefile` only in the Library's, so `- export.zip` under an introducer
#: linked in the transcript and was missing from the ledger. This rule's contract is the
#: renderer's, so it uses the renderer's set; the Library classifier's own set is a separate
#: decision (what the catalog can classify) and is deliberately NOT widened for this.
#: tests/unit/session_files/test_dir_listing.py checks this set against the renderer source.
KNOWN_EXTENSIONS: Final = frozenset(
    {
        # image
        "jpg", "jpeg", "png", "gif", "webp", "bmp", "svg", "tiff", "tif", "ico", "heic",
        "heif", "avif", "jfif", "pjpeg", "pjp",
        # video
        "mp4", "webm", "ogg", "mov", "avi", "wmv", "flv", "mkv", "m4v", "3gp", "3g2",
        # audio
        "mp3", "wav", "aac", "flac", "m4a", "oga",
        # spreadsheet
        "xls", "xlsx", "ods", "csv", "tsv",
        # document
        "doc", "docx", "odt", "rtf", "pages",
        # presentation
        "ppt", "pptx", "odp", "key",
        # archive
        "zip", "tar", "gz", "rar", "7z",
        # code
        "astro", "bash", "c", "cfg", "cjs", "clj", "conf", "cpp", "cs", "css", "dart", "env",
        "ex", "exs", "fish", "go", "gradle", "h", "ini", "java", "js", "json", "jsonc", "jsx",
        "kt", "less", "lua", "mjs", "php", "pl", "py", "r", "rb", "rs", "scala", "scss", "sh",
        "sql", "svelte", "swift", "toml", "ts", "tsx", "vue", "xml", "yaml", "yml", "zsh",
        # text
        "txt", "log", "text",
        # markdown
        "md", "markdown", "mdown", "mkd", "mkdn", "mdx",
        # html
        "html", "htm", "xhtml",
        # pdf
        "pdf",
    }
)  # fmt: skip


def _extension_of(path: str) -> str | None:
    """``extensionOf``: lowercase suffix, ``None`` for none/dotfile/trailing dot."""
    name = path[max(path.rfind("/"), path.rfind("\\")) + 1 :]
    dot = name.rfind(".")
    if dot <= 0 or dot == len(name) - 1:
        return None
    return name[dot + 1 :].lower()


# -- the directory token: `targetsIn` under DISCOVERY_POLICY, one line ----------------------
_PLACEHOLDER_MARKERS: Final = re.compile(r"[$*?{}<>\u2026]")
_ELLIPSIS_SEGMENT: Final = re.compile(r"(?:^|[/\\])\.{3,}(?=[/\\]|$)")
_URL_QUERY: Final = re.compile(r"[^&#]+=[^&#]*(?:&[^&#]+=[^&#]*)*")
_LINE_REFERENCE: Final = re.compile(r":[0-9]+(?::[0-9]+)?\Z")
_PROSE_TOKEN_END: Final = re.compile(_S[:-1] + "\"'`()\\[\\]{}<>,;*|]")
_HTTP_URL: Final = re.compile(r"https?://[^" + re.escape(_JS_SPACE_CHARS) + "]+")
_LINE_BREAK: Final = re.compile(r"[\n\r]")
_WHITESPACE: Final = re.compile(_S)
_URL_TRUNCATION: Final = frozenset("*}>")
_FILE_URL: Final = re.compile(
    r"file://(?:localhost)?/[^" + re.escape(_JS_SPACE_CHARS) + "\"'`>*,;}]+"
)
_PROSE_PATH: Final = re.compile(
    r"(?:~/|/)[^" + re.escape(_JS_SPACE_CHARS) + "\"'`()\\[\\]{}<>,;*|]+"
)
_TRAILING_PUNCTUATION: Final = re.compile(r"[.,;:)\]}\"'`]+\Z")
#: What may sit right before a prose path: `x_/tmp/a` is the tail of a relative name, and an
#: underscore is therefore NOT here - so `_~/out/_:` introduces nothing, on both sides.
_ALLOWED_PREFIX: Final = frozenset("\n\t `\"'([{<>=,;:|*\u2014")
_MAX_CANDIDATE_LENGTH: Final = 4096
_API_PATH_PREFIXES: Final = ("/v1/", "/api/")
_PATH_TRUNCATION: Final = frozenset("*}>|")
_FRAGMENT_OPENERS: Final = {"(": ")", "[": "]", "{": "}", "<": ">"}
_TRAILING_SLASHES: Final = re.compile(r"[/\\]+\Z")


def _first_token(text: str, end: re.Pattern[str]) -> str:
    """JS ``text.split(end, 1)[0]``."""
    match = end.search(text)
    return text[: match.start()] if match else text


def _trimmed_tail_length(token: str) -> int:
    trimmed = _TRAILING_PUNCTUATION.sub("", token)
    return len(token) - len(_LINE_REFERENCE.sub("", trimmed))


def _normalize_candidate(raw: str) -> str | None:
    """``normalizeCandidate``: a prose token as a local rooted path, or None."""
    candidate = _TRAILING_PUNCTUATION.sub("", _js_trim(raw))
    if not candidate or len(candidate) > _MAX_CANDIDATE_LENGTH:
        return None
    candidate = _LINE_REFERENCE.sub("", candidate)
    if not candidate or "://" in candidate or candidate.startswith("//"):
        return None
    if candidate.startswith(_API_PATH_PREFIXES):
        return None
    if not candidate.startswith(("/", "~/")):
        return None
    if _WHITESPACE.search(candidate) or _PLACEHOLDER_MARKERS.search(candidate):
        return None
    if _ELLIPSIS_SEGMENT.search(candidate):
        return None
    return candidate


def _normalize_file_url(raw: str) -> str | None:
    """``normalizeFileUrl``: a local ``file://`` URL's decoded path, or None."""
    raw_path = raw.split("#", 1)[0]
    query_at = raw_path.find("?")
    if query_at != -1 and not _URL_QUERY.fullmatch(raw_path[query_at + 1 :]):
        return None
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return None
    candidate = _TRAILING_PUNCTUATION.sub("", unquote(parsed.path))
    if not candidate or len(candidate) > _MAX_CANDIDATE_LENGTH:
        return None
    candidate = _LINE_REFERENCE.sub("", candidate)
    if not candidate or candidate.startswith(_API_PATH_PREFIXES):
        return None
    if _PLACEHOLDER_MARKERS.search(candidate) or _ELLIPSIS_SEGMENT.search(candidate):
        return None
    return candidate if candidate.startswith("/") else None


def _continues_path(token: str) -> bool:
    return "/" in token or "." in token


def _is_fragment(rest: str) -> bool:
    """``isFragment``: the token stopped at a character that continues the same path."""
    if not rest:
        return False
    closer = _FRAGMENT_OPENERS.get(rest[0])
    if closer:
        return closer in _first_token(rest[1:], _WHITESPACE)
    if rest[0] not in _PATH_TRUNCATION:
        return False
    return _continues_path(_first_token(rest[1:], _PROSE_TOKEN_END))


def _is_placeholder_tail(text: str, index: int) -> bool:
    """``isPlaceholderTail``: the path starts inside a ``dir/<name>``-style placeholder."""
    head = text[:index]
    word = head[len(head.rstrip("".join(set(head) - set(_JS_SPACE_CHARS)))) :] if head else ""
    if not word.endswith(">"):
        return False
    bracket = word.find("<")
    return bracket > 0 and "/" in word[:bracket]


def _last_directory_span(line: str) -> tuple[str, int, bool] | None:
    """The LAST rooted target on ``line`` as ``targetsIn`` sees it: (target, end, is_url).

    Evidence is suspended (``DISCOVERY_POLICY``): an extensionless token is a candidate
    without asking the disk, because the introducer is a ROOT, never itself a row.
    """
    spans: list[tuple[int, int, str, bool]] = []
    for match in _FILE_URL.finditer(line):
        end = match.end()
        rest = line[end:]
        trimmed = _js_trim_start(rest)
        gap = rest[: len(rest) - len(trimmed)]
        if gap and not _LINE_BREAK.search(gap):
            if _continues_path(_first_token(trimmed, _PROSE_TOKEN_END)):
                continue
        if rest and rest[0] in _URL_TRUNCATION:
            if _continues_path(_first_token(rest[1:], _PROSE_TOKEN_END)):
                continue
        candidate = _normalize_file_url(match.group(0))
        if candidate:
            spans.append(
                (match.start(), end - _trimmed_tail_length(match.group(0)), candidate, True)
            )
    masked = _HTTP_URL.sub(lambda url: " " * len(url.group(0)), line)
    for match in _PROSE_PATH.finditer(masked):
        index = match.start()
        if index > 0 and masked[index - 1] not in _ALLOWED_PREFIX:
            continue
        if _is_placeholder_tail(masked, index):
            continue
        raw = match.group(0)
        written = raw.split("#", 1)[0]
        if not written or _is_fragment(masked[index + len(written) :]):
            continue
        candidate = _normalize_candidate(written)
        if candidate is None:
            continue
        spans.append(
            (index, index + len(written) - _trimmed_tail_length(written), candidate, False)
        )
    if not spans:
        return None
    _, end, target, is_url = max(spans, key=lambda span: span[0])
    return target, end, is_url


# -- the listing rule -----------------------------------------------------------------------
_ENDS_WITH_COLON: Final = re.compile(":" + _S + r"*\Z")
_LISTING_INTRODUCER_TAIL: Final = re.compile("[`*_]*" + _S + "*:" + _S + "*")
_LISTED_EXTENSION: Final = re.compile(r"[a-z0-9]{1,10}")
_LISTED_TRAILING_PUNCTUATION: Final = re.compile(r"[.,;:]+\Z")
_LIST_ITEM_LINE: Final = re.compile(
    "(" + _S + r"*)([-*+]|[0-9]{1,9}[.)])[ \t]+(?:\[[ xX]\][ \t]+)?(" + _DOT + "*)"
)
_CONTENT_INDENT: Final = re.compile(_S + r"*(?:(?:[-*+]|[0-9]{1,9}[.)])[ \t]+)?")
_QUOTE_OR_HEADING: Final = re.compile(r"(?:>|#{1,6}(?:[ \t]|\Z))")
_FENCE_LINE: Final = re.compile(_S + "*(`{3,}|~{3,})(" + _DOT + "*)")
#: CommonMark lets only an ordered list starting at 1 interrupt a paragraph. The LITERAL
#: digit, not the value: `01.` straight under an introducer is refused (#1451 round 2). The
#: refusal ends the list before any item is recorded, so `01.` first drops EVERY item under
#: that introducer, not just the one on its own line (`1. a.pdf` after `01. b.pdf` is dropped
#: with it) - the fail-safe direction: CommonMark lets only a `1` interrupt a paragraph, so the
#: mdast holds no such list there, and resolving its items anyway would be inventing a list
#: the renderer never sees.
_ORDERED_INTERRUPT: Final = re.compile(r"1[.)]")
_WHOLE_CODE_SPAN: Final = re.compile(r"`([^`]+)`")
_LEADING_CODE_SPAN: Final = re.compile("`([^`]+)`(" + _DOT + "*)")
_MARKDOWN_ESCAPE: Final = re.compile(r"\\([!-/:-@\[-`{-~])")
#: Spaces included on both sides of every dash, so a hyphen inside a filename ("2026-10-27")
#: never splits it; ": " needs only the trailing space because that is how a note is written.
LISTED_ANNOTATION_SEPARATORS: Final = (" \u2014 ", " \u2013 ", " - ", ": ")


def listing_directory(line: str) -> str | None:
    """``listingDirectoryIn``: the rooted directory ``line`` introduces a list under, or None.

    Refused: a heading or blockquote line; a token that is not directory-shaped (a trailing
    slash, or no dot past the first character of its last segment - so ``~/.config/app:``
    introduces and ``/etc/nginx/conf.d:`` / ``~/x/Next.js:`` do not); and a one-segment
    ``/word``, as likely a slash command (``Run /new:``) as a folder, so ``/tmp:`` stays
    unresolved by design. Emphasis or a code span closing after the directory is allowed
    (``**~/Documents/Radient/**:``) because the scanner's token stops at ``*`` and backtick.
    """
    if not _ENDS_WITH_COLON.search(line):
        return None
    indent = _CONTENT_INDENT.match(line)
    if _QUOTE_OR_HEADING.match(line[indent.end() if indent else 0 :]):
        return None
    last = _last_directory_span(line)
    if last is None:
        return None
    directory, end, is_url = last
    if not _LISTING_INTRODUCER_TAIL.fullmatch(line[end:]):
        return None
    last_segment = directory[directory.rfind("/") + 1 :]
    if not (directory.endswith("/") or last_segment.find(".", 1) == -1):
        return None
    rooted = is_url or directory.startswith("~/") or directory.find("/", 1) != -1
    return directory if rooted else None


def _listed_target(directory: str, item: str) -> str | None:
    """``listedTargetFor`` under the panel's known-extension policy: ``<dir>/<name>`` or None."""
    name = _LISTED_TRAILING_PUNCTUATION.sub("", _js_trim(item))
    if not name or len(name) > _MAX_CANDIDATE_LENGTH:
        return None
    if _LINE_BREAK.search(name) or "/" in name or "\\" in name:
        return None
    if _PLACEHOLDER_MARKERS.search(name):
        return None
    extension = _extension_of(name)
    if not extension or not _LISTED_EXTENSION.fullmatch(extension):
        return None
    if extension not in KNOWN_EXTENSIONS:
        return None
    return f"{_TRAILING_SLASHES.sub('', directory)}/{name}"


def _listed_entry(directory: str, item: str) -> str | None:
    """``listedEntryFor``: the ONE candidate a list item names, unresolved, or None.

    An annotated head (text before the FIRST separator by position) that qualifies by shape is
    final: when the disk refuses it, the item is not retried as a whole name, because the whole
    item is then a name plus prose. The renderer asks its evidence oracle here; this returns the
    candidate and the caller's stat is that oracle, for a head and a whole item alike.
    """
    text = _js_trim(item)
    cuts = [at for at in (text.find(sep) for sep in LISTED_ANNOTATION_SEPARATORS) if at > 0]
    if cuts:
        head = text[: min(cuts)]
        extension = _extension_of(head)
        if extension is not None and extension in KNOWN_EXTENSIONS:
            target = _listed_target(directory, head)
            if target is not None and target.endswith(f"/{head}"):
                return target
    return _listed_target(directory, text)


def _item_text(content: str) -> str:
    """``itemText``: a list item's name text, code span unwrapped, markdown escapes decoded."""
    trimmed = _js_trim(content)
    whole = _WHOLE_CODE_SPAN.fullmatch(trimmed)
    if whole:
        return whole.group(1)
    leading = _LEADING_CODE_SPAN.fullmatch(trimmed)
    if leading:
        return leading.group(1) + _MARKDOWN_ESCAPE.sub(r"\1", leading.group(2))
    return _MARKDOWN_ESCAPE.sub(r"\1", trimmed)


def _marker_kind(marker: str) -> str:
    """One CommonMark list: the bullet character, or an ordered list's delimiter."""
    return f"ordered{marker[-1]}" if marker[0].isdigit() else marker


def _indent_columns(line: str) -> int:
    columns = 0
    for char in line:
        if char == " ":
            columns += 1
        elif char == "\t":
            columns += 4 - columns % 4
        else:
            break
    return columns


def _code_lines(lines: list[str]) -> list[bool]:
    """``codeLinesOf``: which lines sit in a fenced (fences included) or indented code block.

    The renderer's approximation of CommonMark, erring toward "code" only where that drops a
    candidate: a fence closes on a run of its character at least as long, alone on its line
    (an unclosed fence runs to the end); an indented block is a line four columns in after a
    blank line, outside a list, running over blank lines until a shallower line.
    """
    code: list[bool] = []
    fence: tuple[str, int] | None = None
    previous_blank = True
    in_list = False
    in_indented = False
    for raw in lines:
        line = raw.removesuffix("\r")
        blank = _js_trim(line) == ""
        if fence is not None:
            code.append(True)
            trimmed = _js_trim(line)
            run = len(trimmed) - len(trimmed.lstrip(fence[0]))
            if run >= fence[1] and run == len(trimmed):
                fence = None
            previous_blank = False
            continue
        opened = _FENCE_LINE.fullmatch(line)
        marker = opened.group(1) if opened else ""
        # A backtick fence's info string may not hold a backtick: that is an inline span.
        if opened and not (marker[0] == "`" and "`" in opened.group(2)):
            fence = (marker[0], len(marker))
            code.append(True)
            in_indented = False
            previous_blank = False
            continue
        indent = _indent_columns(line)
        if not blank and indent >= 4 and (in_indented or (previous_blank and not in_list)):
            code.append(True)
            in_indented = True
            previous_blank = False
            continue
        if not blank:
            in_indented = False
            in_list = bool(_LIST_ITEM_LINE.fullmatch(line)) or (
                in_list and (indent > 0 or not previous_blank)
            )
        code.append(False)
        previous_blank = blank
    return code


def directory_listing(text: str) -> list[str]:
    """``listedTargetsIn``: unresolved ``<dir>/<name>`` candidates, in document order.

    Follows CommonMark list structure as the renderer's mdast sees it: blank lines may separate
    the introducer and the items; the first item sits inside the introducer's block (its content
    indent, under four more); all items share one marker kind; a deeper marker is a nested list
    and is skipped; a non-marker line straight after an item continues it, so that item is
    dropped. Code (fenced or indented) never introduces or yields an item; an ordered list
    straight under the introducer must start at a literal ``1``. A line inside a blockquote is
    never an item (it does not match the item shape), and a quoted introducer is refused by
    :func:`listing_directory`, so a quoted listing resolves nothing.
    """
    if ":" not in text:
        return []
    found: list[str] = []
    lines = text.split("\n")
    code = _code_lines(lines)
    index = 0
    while index < len(lines):
        if code[index]:
            index += 1
            continue
        introducer = lines[index]
        directory = listing_directory(introducer)
        if directory is None:
            index += 1
            continue
        indent_match = _CONTENT_INDENT.match(introducer)
        floor = indent_match.end() if indent_match else 0
        base: int | None = None
        marker = ""
        pending: str | None = None
        blank = False
        cursor = index + 1
        while cursor < len(lines):
            row = lines[cursor].removesuffix("\r")
            if _js_trim(row) == "":
                blank = True
                cursor += 1
                continue
            item = _LIST_ITEM_LINE.fullmatch(row)
            indent = len(row) - len(_js_trim_start(row))
            if code[cursor]:
                # A fence at the list's level ends it; one indented under an item makes that
                # item more than one name.
                if base is not None and indent > base:
                    pending = None
                    cursor += 1
                    continue
                break
            if item is None:
                if base is None:
                    break
                if not blank or indent > base:
                    pending = None
                    cursor += 1
                    continue
                break
            after_blank = blank
            blank = False
            kind = _marker_kind(item.group(2))
            if base is None:
                if indent < floor or indent > floor + 3:
                    break
                if not after_blank and kind.startswith("ordered"):
                    if not _ORDERED_INTERRUPT.fullmatch(item.group(2)):
                        break
                base, marker = indent, kind
            if indent > base:
                cursor += 1
                continue
            if indent < base or kind != marker:
                break
            if pending is not None:
                found.append(pending)
            pending = _listed_entry(directory, _item_text(item.group(3)))
            cursor += 1
        if pending is not None:
            found.append(pending)
        # The list's own lines cannot introduce another list at this level.
        index = max(index + 1, cursor)
    return found
