"""The generator-output parser and the honesty validator (memo §2.5, §2.10, §4.2).

WHAT IT GUARDS, AND WHAT IT DOES NOT. The sandbox is the security boundary (memo §4.1); the
validator exists to fail fast, cut noise and keep honesty checkable. A string scan is evadable
by obfuscation and that is accepted, because nothing it guards is reachable under the sandbox
and CSP. The check that is NOT best-effort is the honesty one: a component may plot only the
numbers the turn's evidence holds, and every number it prints statically must be reproducible
from its own declared data by the prelude's own formatting rules (``supplements/fmt.py``).

THE GRAMMAR (memo App. A). The fork answers with 0-3 blocks, or the single word ``NONE``::

    <component title="…" source="…">
    <data>{"<dataId>": {"title": "…", "columns": ["…"], "rows": [[…]]}}</data>
    <html>…body…</html>
    </component>

A block that fails ANY check is rejected with its precise errors. The caller keeps the blocks
that passed and feeds the errors back as ONE repair turn (memo §2.5); a block that fails again
is dropped, never rendered (memo §2.5 "a component that fails validation is dropped").

THE STORED FORM. A valid block becomes one :class:`Component` whose ``blob`` is the
content-addressed bytes ``AttachmentStore.put_bytes(raw, "text/html")`` holds: the leading
``<data>`` block(s), canonicalised through ``document.data_json``, then the body. That keeps
one extractor (``document.split_component``) able to rebuild the exact document the validator
checked -- the HTML never has its data lifted out of it into a second representation.

WHY THE CHECKS ARE ORDERED AS THEY ARE. Size and well-formedness first (cheap, and a
malformed document cannot be scanned meaningfully), then the §4.2 scan, then provenance: the
scan's reject reasons are about markup the model wrote, while a provenance failure is about
numbers it invented, and the repair message reads better with the markup settled first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Final, Iterable, Mapping, Sequence

from local_operator.supplements import fmt as fmt_mod
from local_operator.supplements.contract import HEIGHT_HINT_MAX, HEIGHT_HINT_MIN
from local_operator.supplements.document import data_json, numeric_cell, split_component
from local_operator.supplements.evidence import Dataset

#: Memo §4.2: a component over this size is rejected outright.
MAX_COMPONENT_BYTES: Final = 48 * 1024
#: The reserved box's height when nothing has told us better. There is no render probe in v1
#: (memo §8 F8), so the writer states the figure it can defend -- the barchart fixtures' own
#: working height -- and clamps it into the memo's range. A later render probe replaces this.
DEFAULT_HEIGHT_HINT: Final = 320

_COMPONENT_RE: Final = re.compile(r"<component\b([^>]*)>(.*?)</component\s*>", re.DOTALL | re.I)
_ATTR_RE: Final = re.compile(r"([a-zA-Z:-]+)\s*=\s*\"([^\"]*)\"")
_HTML_RE: Final = re.compile(r"<html\b[^>]*>(.*?)</html\s*>", re.DOTALL | re.I)

#: Memo §4.2, plus round-1 review S-R9's WebRTC/worker class. Compared case-insensitively on
#: the body, and the whole list is here so a reader can see the scan's actual coverage.
_BANNED: Final[tuple[str, ...]] = (
    "<iframe",
    "<object",
    "<embed",
    "<base",
    "<link",
    "<meta",
    "<form",
    "<frame",
    "<portal",
    "javascript:",
    "srcdoc",
    "http://",
    "https://",
    "fetch(",
    "XMLHttpRequest",
    "WebSocket",
    "EventSource",
    "import(",
    "importScripts",
    "eval(",
    "Function(",
    "document.cookie",
    "localStorage",
    "sessionStorage",
    "indexedDB",
    "navigator.sendBeacon",
    "window.open",
    "top.",
    "parent.",
    "location",
    "RTCPeerConnection",
    "RTCDataChannel",
    "WebTransport",
    "SharedWorker",
    "BroadcastChannel",
    "new Worker",
)

#: A numeric literal of three or more digits is a reject reason outside the STRUCTURAL
#: attribute values (:data:`_STRUCTURAL_ATTRS` -- ``viewBox``, SVG geometry, sizes, ``style``):
#: memo §4.2, whose viewBox/style-only exemption round-2 review R2-1 widened to the whole
#: table (a 640x480 ``viewBox`` or ``points="0,480 …"`` was refused). Plotted values must
#: come from ``<data>``, and a hard-coded 3-digit number in a script is how a fabricated
#: series rides in.
_INLINE_NUMBER_RE: Final = re.compile(r"\d{3,}")

#: ``"… (derived: a/b)"``: the only way a computed column may appear (memo §2.10). The
#: expression is row-wise over two columns of the same dataset, one operator, and an optional
#: ``*100`` for a percentage.
_DERIVED_RE: Final = re.compile(r"^(?P<label>.+?)\s*\(derived:\s*(?P<expr>[^)]+)\)\s*$", re.I)
_DERIVED_EXPR_RE: Final = re.compile(
    r"^\s*(?P<a>[^*/+\-]+?)\s*(?P<op>[+\-/])\s*(?P<b>[^*]+?)\s*(?P<pct>\*\s*100)?\s*$"
)

#: Numeric-looking tokens in STATIC text (never inside ``<script>``/``<style>``/``<data>``):
#: an optional sign, grouped digits and decimals, an optional short unit. Everything the
#: component prints literally is matched against the values its own ``<data>`` supports.
_STATIC_NUMBER_RE: Final = re.compile(
    r"(?<![\w.\-/:])(?P<num>[-+]?\d[\d,]*(?:\.\d+)?)"
    r"(?:\s?(?P<unit>%|[A-Za-zµ][A-Za-zµ/]{0,5}))?(?![\w\-/:])"
)
_STATIC_SKIP_RE: Final = re.compile(r"<(script|style|data)\b.*?</\1\s*>", re.DOTALL | re.I)

#: Attribute VALUES that are structural/geometry, never displayed claims (§4.2's own
#: ``viewBox``/style carve-out, applied to BOTH scans since round-2 review R2-1; round-1
#: review R3 introduced it for the rendered-values check).
#: A raw DOM/SVG component draws WITH these numbers -- a ``viewBox``, a path's ``d``, a
#: polyline's ``points``, a ``transform``, coordinates, sizes, stroke geometry -- and the
#: blessed raw-SVG path (App. A: "no fixed widths, use viewBox SVG") must not be refused
#: for them. Deliberately NAME-keyed: anything not listed (``aria-label``, ``title``, ...)
#: stays scanned, because those attributes carry text a user can read. The list is the
#: markup vocabulary the prelude and the fixtures build charts with.
#:
#: THE BOUNDARY THIS BUYS (round-2 review R2-2): blanking geometry means the scan verifies
#: READABLE numerals -- element text and data-bearing attributes -- not drawn geometry: a
#: chart's SHAPE may encode a value its text does not state (a bar ``height`` of 42 against
#: data that says 98.5 is accepted, by construction), and proportion-level deception is the
#: golden-set QA spot-check's territory (memo §5), not this check's. The table is applied by
#: the tag walk (:func:`_blank_structural`), which blanks a value only when its tag parses
#: cleanly -- what cannot be proven structural stays scannable (round-4 review R4-1).
_STRUCTURAL_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "viewbox",
        "d",
        "points",
        "transform",
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "r",
        "rx",
        "ry",
        "width",
        "height",
        "offset",
        "stroke-width",
        "stroke-dasharray",
        "stroke-dashoffset",
        "opacity",
        "fill-opacity",
        "stroke-opacity",
        "style",
    }
)
#: The characters that end a tag or attribute NAME in the walk (:func:`_scan_start_tag`):
#: whitespace, ``/``, ``<``, ``>``, ``=`` and the quote characters. A name stopping where no
#: name can continue is the ambiguous shape the walk refuses to read (round-4 review R4-1)
#: -- the span is scanned instead.
_NAME_END: Final = " \t\n\r\f/<>=\"'"
#: The characters that end an UNQUOTED value: HTML's attribute grammar, which has no
#: whitespace, quotes, ``=``, ``<``, ``>`` or backtick inside one (round-3 review R3-2
#: keeps the unquoted form; the backtick stayed out of values under the pair regex too).
_BARE_END: Final = " \t\n\r\f<>=\"'`"

_VOID: Final[frozenset[str]] = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


@dataclass(frozen=True)
class Component:
    """One validated component, ready to store and to put on a row."""

    title: str
    source: str
    #: The stored bytes (``<data>`` blocks + body) -- what ``AttachmentStore`` digests.
    blob: str
    #: The merged datasets, for the row's own readers and for the tests' provenance checks.
    data: dict[str, Any]
    height_hint: int


@dataclass(frozen=True)
class Rejected:
    """One block that failed, with the errors the repair turn is handed."""

    title: str
    errors: tuple[str, ...]

    def describe(self) -> str:
        """One line for the repair message: which block, and why (memo App. A).

        The title is how the fork names the block back to itself -- a repair turn that only
        listed errors would leave the model guessing which of two charts went wrong -- and the
        errors are joined with ``; `` so the message stays one line per block.
        """
        name = self.title or "(untitled component)"
        return f"{name}: {'; '.join(self.errors)}"


@dataclass(frozen=True)
class ValidationResult:
    components: tuple[Component, ...] = ()
    rejected: tuple[Rejected, ...] = ()
    #: The fork answered ``NONE``: a legitimate, complete answer meaning "no graphic here".
    none: bool = False

    @property
    def repair_errors(self) -> list[str]:
        """Every rejection's errors, flattened -- the repair turn's list (memo App. A)."""
        return [error for block in self.rejected for error in block.errors]


class _WellFormed(HTMLParser):
    """A tag-stack checker: the first mismatch is the reason, and it names the tag.

    ``html.parser`` is deliberately lenient; the memo's first reject reason is "not
    well-formed (``html.parser``, stdlib)", so the stack is ours. Void elements and
    self-closing tags never push. A stray close tag is a mismatch, not ignored: a component
    that closes a tag it never opened is markup the host would repair differently than the
    model intended.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._open: list[str] = []
        self.problem: str = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID:
            return
        self._open.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        if not self._open:
            self.problem = self.problem or f"stray </{tag}>"
            return
        if self._open[-1] != tag:
            self.problem = self.problem or f"</{tag}> closes <{self._open[-1]}>"
            return
        self._open.pop()

    def close(self) -> None:  # noqa: D102 - the base class' own name
        super().close()
        if not self.problem and self._open:
            self.problem = f"unclosed <{self._open[-1]}>"


def strip_data(blob: str) -> str:
    """The body of a stored blob, minus its leading ``<data>`` blocks."""
    body, _ = split_component(blob)
    return body


def _component_attrs(raw: str) -> dict[str, str]:
    return {key.lower(): value for key, value in _ATTR_RE.findall(raw)}


def _numeric(value: Any) -> float | None:
    # The prompt emits plain numbers, but legacy components/evidence may spell the
    # same datum "$0.15". Use the assembler's conservative reading on BOTH sides;
    # keep stored cells untouched so provenance never depends on serialisation order.
    value = numeric_cell(value)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", ""))
        except ValueError:
            return None
    return None


#: Markdown emphasis glyphs around or inside a table cell: ``**bold**``, ``*italic*``,
#: Emphasis glyphs around a cell's content: ``**bold**``, ``*italic*``, ``__alt__``. A run
#: FLANKED BY WORD CHARACTERS on both sides is DATA, never emphasis -- ``5*12`` is arithmetic
#: and ``foo__bar`` an identifier (round-1 review R1 / QA Q-1: the first cut removed every
#: asterisk run, so a ``5*12`` cell canonicalised to ``512`` and a component claiming 512
#: passed provenance; both probes are pinned in ``test_validate.py``).
_EMPHASIS_RE: Final = re.compile(
    r"(?<![A-Za-z0-9_*])\*+|\*+(?![A-Za-z0-9_*])|(?<![A-Za-z0-9_])_{2,}|_{2,}(?![A-Za-z0-9_])"
)


def _strip_emphasis(text: str) -> str:
    """``**42.3**`` -> ``42.3``: emphasis is formatting, not the datum (memo §2.10).

    A markdown table that bolds its winners yields extracted cells like ``**77.0**``; the
    value it holds is 77.0, and the model's faithful emission of 77.0 must be admitted. The
    2026-10-17 investigation (session 8197f86e639d) found the opposite: every bolded cell
    failed provenance as "not in the evidence", and a bolded LABEL as "an invented label is
    a fabrication" against glyphs that were sitting in the evidence itself.

    The same word-adjacency rule guards both delimiters: a glyph run flanked by word
    characters stays verbatim (``5*12``, ``foo__bar``), so tightening emphasis handling can
    never fuse two datums into a third.
    """
    return _EMPHASIS_RE.sub("", text).strip()


class _EvidenceIndex:
    """The evidence's own values, canonicalised once per output (memo §2.10).

    ``numbers`` is float equality after the source's own precision; ``strings`` is exact match
    for identifiers. An evidence cell that parses as a number contributes to BOTH: a component
    may copy ``"1,204.50"`` verbatim as an identifier, and it may plot ``1204.5``.
    """

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.numbers: set[float] = set()
        self.strings: set[str] = set()
        self.units: set[str] = set()
        for dataset in datasets:
            for row in dataset.rows:
                for cell in row:
                    text = str(cell).strip()
                    if not text:
                        continue
                    # The cell's CONTENT, with markdown emphasis canonicalised away: the
                    # verbatim spelling stays admissible too (a component may copy the cell
                    # exactly as it reads), and the plain form is what the value checks use.
                    plain = _strip_emphasis(text) or text
                    self.strings.add(text)
                    self.strings.add(plain)
                    number = _numeric(plain)
                    if number is not None:
                        self.numbers.add(number)
                    else:
                        # ``"12 ms"``: the evidence's own cell, unit included.
                        from local_operator.supplements.evidence import (  # noqa: PLC0415
                            _number,
                        )

                        parsed = _number(plain)
                        if parsed is not None:
                            self.numbers.add(parsed)
            for column in dataset.columns:
                self.units.update(_units_in(column))
            self.units.update(_units_in(dataset.title))
            self.strings.update(dataset.columns)

    def accepts_number(self, value: float) -> bool:
        return value in self.numbers

    def accepts_string(self, text: str) -> bool:
        return text in self.strings


def _units_in(text: str) -> set[str]:
    """Unit tokens inside a column name or title (``Latency (ms)`` -> ``ms``)."""
    return {
        match.group("unit") for match in _STATIC_NUMBER_RE.finditer(text) if match.group("unit")
    }


def _derived_ok(columns: Sequence[str], rows: Sequence[Sequence[Any]], index: int) -> list[str]:
    """Recompute one ``(derived: …)`` column; returns the errors it produced (memo §2.10).

    Only row-wise sums, differences and ratios of two OTHER columns of the same dataset are
    allowed, with an optional ``*100`` for a percentage. Anything else -- a mean, a
    percentage change, a value from another dataset -- is not recomputable and is refused
    rather than trusted.
    """
    match = _DERIVED_RE.match(columns[index])
    assert match is not None
    expression = _DERIVED_EXPR_RE.match(match.group("expr"))
    if expression is None:
        return [
            f"column {columns[index]!r}: a derived column must be `a op b` with op one of "
            "+ - / (and an optional `* 100`), row-wise over two columns of the same dataset"
        ]
    left, right = expression.group("a").strip(), expression.group("b").strip()
    op = expression.group("op")
    scale = 100.0 if expression.group("pct") else 1.0
    if left not in columns or right not in columns:
        return [
            f"column {columns[index]!r}: {left!r}/{right!r} must name columns of its own dataset"
        ]
    left_index, right_index = columns.index(left), columns.index(right)
    if index in (left_index, right_index):
        return [f"column {columns[index]!r}: a derived column cannot derive from itself"]
    errors: list[str] = []
    for position, row in enumerate(rows):
        a = _numeric(row[left_index]) if left_index < len(row) else None
        b = _numeric(row[right_index]) if right_index < len(row) else None
        declared = _numeric(row[index]) if index < len(row) else None
        if a is None or b is None:
            errors.append(
                f"column {columns[index]!r} row {position + 1}: a source cell is not numeric"
            )
            continue
        if op == "+":
            expected = a + b
        elif op == "-":
            expected = a - b
        else:
            if b == 0:
                errors.append(f"column {columns[index]!r} row {position + 1}: division by zero")
                continue
            expected = a / b * scale
        if declared is None:
            errors.append(
                f"column {columns[index]!r} row {position + 1}: {row[index]!r} is not numeric"
            )
        elif abs(declared - expected) > max(abs(expected), 1.0) * 1e-6:
            errors.append(
                f"column {columns[index]!r} row {position + 1}: {declared!r} is not "
                f"{expected!r} from {left}/{right}"
            )
    return errors


def _check_data(data: Mapping[str, Any], evidence: _EvidenceIndex) -> tuple[list[str], set[str]]:
    """Provenance for the whole ``<data>`` block (memo §2.10). Returns errors and the units."""
    errors: list[str] = []
    units: set[str] = set()
    if not isinstance(data, Mapping):
        return ["<data> must be a JSON object of datasets"], units
    for data_id, dataset in data.items():
        if not isinstance(dataset, Mapping):
            errors.append(f"dataset {data_id!r} is not an object")
            continue
        columns = dataset.get("columns")
        rows = dataset.get("rows")
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
            errors.append(f"dataset {data_id!r}: `columns` must be a list of strings")
            continue
        if not isinstance(rows, list) or not all(isinstance(row, list) for row in rows):
            errors.append(f"dataset {data_id!r}: `rows` must be a list of lists")
            continue
        title = dataset.get("title")
        units.update(_units_in(str(title or "")))
        units.update(_units_in(" ".join(columns)))
        derived: set[int] = set()
        for index, column in enumerate(columns):
            if _DERIVED_RE.match(column):
                derived.add(index)
        for index, column in enumerate(columns):
            if index in derived:
                errors.extend(_derived_ok(columns, rows, index))
        for position, row in enumerate(rows):
            if len(row) != len(columns):
                errors.append(
                    f"dataset {data_id!r} row {position + 1}: {len(row)} cells for "
                    f"{len(columns)} columns"
                )
                continue
            for index, cell in enumerate(row):
                if index in derived:
                    continue
                number = _numeric(cell)
                if number is not None:
                    if not evidence.accepts_number(number):
                        errors.append(
                            f"dataset {data_id!r} row {position + 1} column "
                            f"{columns[index]!r}: {cell!r} is not in the evidence"
                        )
                    continue
                text = str(cell).strip()
                if text and not evidence.accepts_string(text):
                    errors.append(
                        f"dataset {data_id!r} row {position + 1} column {columns[index]!r}: "
                        f"{text!r} is not in the evidence (an invented label is a fabrication)"
                    )
    return errors, units


def _blank_structural(text: str) -> str:
    """Every structural attribute value a CLEAN tag parse proves, replaced by one space.

    THE INVARIANT (round-4 review R4-1). If a span cannot be PROVEN structural geometry,
    its numerals must stay scannable: refuse an honest geometry number rather than hide a
    fabricated claim (fail-closed -- over-refuse, never under-refuse). The pair regex this
    replaced paired quotes by proximity, so a parity-balanced skew (``x="0 width=640>``
    closing at a LATER quote) blanked ``<text>7000 ms</text>`` as one attribute value --
    hidden from both scans while the tag stack stayed balanced.

    WHAT "PROVE IT IS STRUCTURAL" MEANS. A tag blanks its :data:`_STRUCTURAL_ATTRS` values
    only when it parses cleanly to its closing ``>``: quoting balanced, attribute
    boundaries unambiguous, and no ``<``/``>`` inside a quoted value (a legitimate
    attribute value never contains either -- one that does is evidence the quote pairing
    is a skew, not a value). Any imbalance, unterminated quote, cross-tag span or
    truncated tag leaves everything from that tag on scannable. A ``<`` that cannot open a
    tag (``a < b``) is plain text. A quoted value may run over lines; an UNQUOTED value
    cannot contain whitespace (HTML's attribute grammar), so a spaced coordinate list
    still has to be quoted to be blanked (round-3 review R3-2).
    """
    out: list[str] = []
    index = 0
    size = len(text)
    while index < size:
        start = text.find("<", index)
        if start < 0:
            out.append(text[index:])
            break
        out.append(text[index:start])
        second = text[start + 1 : start + 2]
        opens = second == "/" or second == "!" or second == "?"
        if opens or (second.isascii() and second.isalpha()):
            tag = _scan_tag(text, start)
            if tag is None:
                # Not provable: every numeral from here stays scannable.
                out.append(text[start:])
                break
            end, replacement = tag
            out.append(replacement)
            index = end
        else:
            # A "<" that cannot open a tag ("a < b"): text, nothing to blank.
            out.append("<")
            index = start + 1
    return "".join(out)


def _scan_tag(text: str, start: int) -> tuple[int, str] | None:
    """The tag at ``text[start] == "<"`` -- ``(end, replacement)``, or ``None`` if unprovable.

    Recognises the constructs an inline component uses: start tags (through
    :func:`_scan_start_tag`), end tags, comments (``<!-- … -->``), CDATA, declarations
    (``<!doctype …>``) and processing instructions. Everything except a start tag comes
    back unchanged -- its extent just has to be found so the walk resumes after it; nothing
    inside it is blanked, so a numeral there stays scannable like anything else the walk
    cannot read.
    """
    size = len(text)
    second = text[start + 1 : start + 2]
    if second == "!":
        if text.startswith("<!--", start):
            end = text.find("-->", start + 4)
            return None if end < 0 else (end + 3, text[start : end + 3])
        if text[start : start + 9].upper() == "<![CDATA[":
            end = text.find("]]>", start + 9)
            return None if end < 0 else (end + 3, text[start : end + 3])
        end = text.find(">", start + 2)
        return None if end < 0 else (end + 1, text[start : end + 1])
    if second == "?":
        end = text.find(">", start + 2)
        return None if end < 0 else (end + 1, text[start : end + 1])
    if second == "/":
        index = start + 2
        name = index
        while index < size and text[index] not in _NAME_END:
            index += 1
        if index == name:
            return None
        index = _skip_space(text, index)
        if index < size and text[index] == ">":
            return index + 1, text[start : index + 1]
        return None
    return _scan_start_tag(text, start)


def _scan_start_tag(text: str, start: int) -> tuple[int, str] | None:
    """The attribute walk for ONE start tag; ``None`` when it cannot be proven clean.

    Boundaries follow HTML's own grammar where it is unambiguous: a name runs to
    :data:`_NAME_END`, ``=`` opens a value, quoted (must close before ``<``, ``>`` or the
    end) or bare (stops at :data:`_BARE_END`). A name stopping where a name cannot
    continue, a ``/`` that is not the self-closing end, a quoted value reaching
    ``<``/``>`` or the end, an ``=`` with no value -- each is the ambiguous class the walk
    refuses to read: ``None``, and the caller leaves the span scannable (R4-1). The span
    blanked for a :data:`_STRUCTURAL_ATTRS` name is its pair, name through value.
    """
    size = len(text)
    index = start + 1
    while index < size and text[index] not in _NAME_END:
        index += 1
    spans: list[tuple[int, int]] = []
    while True:
        index = _skip_space(text, index)
        if index >= size:
            return None  # EOF inside the tag
        char = text[index]
        if char == ">":
            return index + 1, _render_tag(text, start, index + 1, spans)
        if char == "/":
            if text[index + 1 : index + 2] == ">":
                return index + 2, _render_tag(text, start, index + 2, spans)
            return None  # a "/" that is not the self-closing end
        if char == "<":
            return None  # a tag boundary inside a tag: not provable
        span = index
        while index < size and text[index] not in _NAME_END:
            index += 1
        if index == span:
            return None  # '=', a quote or '/' where an attribute name must start
        attribute = text[span:index]
        index = _skip_space(text, index)
        if index < size and text[index] == "=":
            index = _skip_space(text, index + 1)
            if index >= size:
                return None
            quote = text[index]
            if quote in "\"'":
                index += 1
                while index < size and text[index] != quote:
                    if text[index] in "<>":
                        return None  # the skew this walk exists for (R4-1)
                    index += 1
                if index >= size:
                    return None  # unterminated quote
                index += 1
            else:
                value = index
                while index < size and text[index] not in _BARE_END:
                    index += 1
                if index == value:
                    return None  # "name=" with no value
            if attribute.lower() in _STRUCTURAL_ATTRS:
                spans.append((span, index))


def _render_tag(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> str:
    """The tag's own text with each structural pair's span replaced by one space (R4-1)."""
    if not spans:
        return text[start:end]
    parts: list[str] = []
    cursor = start
    for span, stop in spans:
        parts.append(text[cursor:span])
        parts.append(" ")
        cursor = stop
    parts.append(text[cursor:end])
    return "".join(parts)


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\n\r\f":
        index += 1
    return index


def _check_static_numbers(body: str, data: Mapping[str, Any], units: set[str]) -> list[str]:
    """Every number PRINTED in static markup must be reproducible from the component's data.

    Static text is markup outside ``<script>``/``<style>``/``<data>``: a helper call's output
    is decided at runtime by the prelude (and by ``LO.fmt``), while a literal in markup is the
    model's own claim. The comparison is against ``LO.fmt``'s own rules via the Python mirror
    (:mod:`local_operator.supplements.fmt`) at DEFAULT precision plus the units the data
    carries, because static text has no place to put a ``digits`` option -- a baked-in rounded
    literal is exactly the failure this catches (memo §2.10, round-1 D4).

    Structural attribute values are BLANKED before the scan (round-1 review R3): ``viewBox``,
    a path's ``d``, ``points``, ``transform``, coordinates, sizes and stroke geometry are
    layout the component draws WITH, not claims about data -- §4.2's carve-out, one shared
    rule for both scans since round-2 review R2-1 -- while element TEXT and data-bearing
    attributes (``aria-label`` and friends) keep being validated, because those are what a
    user can actually read.
    """
    allowed: set[str] = set()
    for dataset in data.values() if isinstance(data, Mapping) else []:
        if not isinstance(dataset, Mapping):
            continue
        for row in dataset.get("rows") or []:
            if not isinstance(row, list):
                continue
            for cell in row:
                number = _numeric(cell)
                if number is None:
                    continue
                allowed.add(fmt_mod.fmt(number))
                for unit in units:
                    allowed.add(fmt_mod.fmt(number, unit=unit))
    errors: list[str] = []
    text = _STATIC_SKIP_RE.sub(" ", body)
    # Blank the proven structural attributes' values first (see _blank_structural): layout
    # numbers are not displayed values, and only a clean tag parse is blanked (R4-1).
    text = _blank_structural(text)
    for match in _STATIC_NUMBER_RE.finditer(text):
        printed = match.group(0).strip()
        number = _numeric(match.group("num"))
        if number is None:
            continue
        if printed in allowed or fmt_mod.fmt(number) in allowed:
            continue
        if match.group("unit"):
            candidate = fmt_mod.fmt(number, unit=match.group("unit"))
            if candidate in allowed:
                continue
        errors.append(f"static text {printed!r} is not a value the component's own <data> supports")
    return errors


class _Resources(HTMLParser):
    """HTMLParser decodes entities and recognizes quoted/unquoted attributes equally."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []
        self.tags: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.add(tag)
        for key, raw in attrs:
            value = (raw or "").strip().casefold()
            if key == "srcset":
                self.errors.append("srcset dependencies are unsupported")
            if key in {"src", "href", "xlink:href", "poster", "action", "background", "data"}:
                if not value.startswith(("#", "data:image/")) or tag == "script":
                    self.errors.append("resource dependencies are unavailable in an inline preview")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _check_scan(blob: str, *, authored: bool = False) -> list[str]:
    """The §4.2 string scan plus the 3-digit literal rule, on the body (data excluded).

    The literal rule blanks the structural attribute values the walk can prove before
    searching (round-2 review R2-1; only a clean tag parse since round-4 review R4-1): a
    chart drawn at a 640x480 ``viewBox`` or carrying ``points="0,480 …"`` draws with those
    numbers and must not be refused for them -- the same table :func:`_check_static_numbers`
    uses, so both scans read readable numerals only (R2-2; the boundary it buys is at
    :data:`_STRUCTURAL_ATTRS`).
    """
    body = strip_data(blob)
    lowered = body.lower()
    nodes = _Resources()
    nodes.feed(body)
    errors: list[str] = [
        f"rejected string {needle!r}"
        for needle in _BANNED
        if (needle[1:] in nodes.tags if needle.startswith("<") else needle.lower() in lowered)
    ]
    # CSP intentionally cannot fetch dependencies. Fail honestly before storing a
    # frame that could only render half a document; data images remain self-contained.
    resources = _Resources()
    resources.feed(body)
    errors.extend(resources.errors)
    dependency_text = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    dependency_text = re.sub(
        r"\\([0-9a-fA-F]{1,6})\s?",
        lambda match: chr(int(match.group(1), 16)) if int(match.group(1), 16) <= 0x10FFFF else "",
        dependency_text,
    )
    if re.search(
        r"@import\b|\bimport\s*(?:\(|[\"'\{*])|\bexport\s+.*?\bfrom\s*[\"']",
        dependency_text,
        re.I | re.S,
    ):
        errors.append("module and CSS imports are unavailable in an inline preview")
    for match in re.finditer(r"url\s*\(\s*([\"']?)(.*?)\1\s*\)", dependency_text, re.I | re.S):
        if not match.group(2).strip().casefold().startswith(("data:image/", "#")):
            errors.append("CSS resource URLs must be inline data images or local fragments")
    # The banned-string pass above reads the RAW body (attributes and all); this scan reads
    # the walked text -- §4.2's viewBox/style carve-out widened to the structural table.
    scannable = _blank_structural(re.sub(r"<style\b.*?</style\s*>", "", body, flags=re.I | re.S))
    if authored:
        # Authored scripts retain timing/layout literals; displayed claims below
        # still require exact evidence values. Generated-chart rules stay strict.
        scannable = _STATIC_SKIP_RE.sub(" ", scannable)
    match = _INLINE_NUMBER_RE.search(scannable)
    if match is not None and not authored:
        errors.append(
            f"inline numeric literal {match.group(0)!r} (3+ digits) outside structural "
            "attribute values (viewBox, geometry, style): plotted values belong in <data>"
        )
    return errors


def parse_output(text: str) -> tuple[list[tuple[dict[str, str], str]], list[str]]:
    """``[(attrs, inner), …]`` from the fork's answer, plus the parse errors.

    A ``NONE`` answer (the memo's own word, case-insensitive, alone on its line) is the
    caller's to recognise via :func:`validate_output`; this function reports it as zero blocks
    and no errors.
    """
    stripped = text.strip()
    if not stripped or stripped.upper() == "NONE":
        return [], []
    blocks = [
        (_component_attrs(match.group(1)), match.group(2).strip())
        for match in _COMPONENT_RE.finditer(text)
    ]
    errors: list[str] = []
    if not blocks:
        errors.append("no <component> block and not the word NONE")
    # Anything outside the blocks is the memo's "nothing outside it" rule, minus the fence
    # prose a model may wrap the blocks in (```xml … ```): the outer text is not rendered, so
    # it is not a reject -- only the ABSENCE of any block is.
    return blocks, errors


def _split_inner(inner: str) -> tuple[str, dict[str, Any]]:
    """``(body, data)`` for one block: ``<html>…</html>`` and the leading ``<data>`` blocks."""
    html_match = _HTML_RE.search(inner)
    if html_match is None:
        raise ValueError("<html>…</html> is missing")
    body = html_match.group(1)
    head = inner[: html_match.start()]
    try:
        split_body, data = split_component(head + "\n" + body)
    except ValueError as error:
        raise ValueError(f"<data> is malformed: {error}") from None
    if split_body != body.strip("\n"):
        raise ValueError("<data> blocks may only precede the body")
    return body, data


def _without_timer_delays(script: str) -> str:
    """Blank only a literal second argument; callback/source claims remain inspectable.

    This is deliberately not a JavaScript parser. Ambiguous regex literals or
    unbalanced syntax receive no exemption; refusing a delay is safer than
    accidentally erasing a callback. Limit nested walks to keep the work bounded.
    """
    spans: list[tuple[int, int]] = []
    for match in list(re.finditer(r"\b(?:setTimeout|setInterval)\s*\(", script))[:128]:
        stack = ["("]
        commas: list[int] = []
        quote = ""
        index = match.end()
        close: int | None = None
        while index < len(script):
            char = script[index]
            if quote:
                if char == "\\":
                    index += 2
                    continue
                if char == quote:
                    quote = ""
            elif char in {"'", '"', "`"}:
                quote = char
            elif script.startswith("//", index):
                newline = script.find("\n", index + 2)
                if newline < 0:
                    break
                index = newline
            elif script.startswith("/*", index):
                end = script.find("*/", index + 2)
                if end < 0:
                    break
                index = end + 2
                continue
            elif char == "/":
                break  # Can't prove division vs regex without a full JS parser.
            elif char in "([{":
                stack.append(char)
            elif char in ")]}":
                if not stack or {"(": ")", "[": "]", "{": "}"}[stack[-1]] != char:
                    break
                stack.pop()
                if not stack:
                    close = index
                    break
            elif char == "," and stack == ["("]:
                commas.append(index)
            index += 1
        if close is None or not commas:
            continue
        begin = commas[0] + 1
        end = commas[1] if len(commas) > 1 else close
        argument = script[begin:end]
        if re.fullmatch(r"\s*\d+\s*", argument):
            digits = re.search(r"\d+", argument)
            assert digits is not None
            spans.append((begin + digits.start(), begin + digits.end()))
    for begin, end in sorted(set(spans), reverse=True):
        script = script[:begin] + " " * (end - begin) + script[end:]
    return script


def validate_output(
    text: str, evidence: Sequence[Dataset], *, authored: bool = False
) -> ValidationResult:
    """Parse, scan and provenance-check one fork answer (memo §2.5's turn-1/turn-2 gate)."""
    if text.strip().upper() == "NONE":
        return ValidationResult(none=True)
    blocks, parse_errors = parse_output(text)
    if parse_errors:
        return ValidationResult(rejected=(Rejected(title="", errors=tuple(parse_errors)),))
    index = _EvidenceIndex(evidence)
    components: list[Component] = []
    rejected: list[Rejected] = []
    for attrs, inner in blocks:
        title = attrs.get("title", "").strip()
        source = attrs.get("source", "").strip()
        errors: list[str] = []
        if not source:
            errors.append("missing source=… (every component says which evidence it shows)")
        try:
            body, data = _split_inner(inner)
        except ValueError as error:
            rejected.append(Rejected(title=title, errors=(str(error),)))
            continue
        if authored:
            # Standard authored HTML has no proprietary wrapper. Its admissible
            # value set is copied from supplied turn evidence, never inferred from HTML.
            data = {
                f"e{index}": {
                    "title": item.title,
                    "columns": list(item.columns),
                    "rows": [list(row) for row in item.rows],
                }
                for index, item in enumerate(evidence)
            }
        blob = "\n".join(f"<data>{data_json({key: value})}</data>" for key, value in data.items())
        blob = f"{blob}\n{body}" if blob else body
        if len(blob.encode("utf-8")) > MAX_COMPONENT_BYTES:
            errors.append(f"component is over {MAX_COMPONENT_BYTES} bytes")
        checker = _WellFormed()
        checker.feed(body)
        checker.close()
        if checker.problem:
            errors.append(f"not well-formed: {checker.problem}")
        errors.extend(_check_scan(blob, authored=authored))
        data_errors, units = _check_data(data, index)
        errors.extend(data_errors)
        claims = (
            re.sub(r"\b(?:Step|Stage|Phase)\s+\d+\b", "", body, flags=re.I) if authored else body
        )
        errors.extend(_check_static_numbers(claims, data, units))
        if authored:
            supported = {
                _numeric(cell)
                for item in evidence
                for row in item.rows
                for cell in row
                if _numeric(cell) is not None
            }
            for script in re.findall(r"<script\b[^>]*>(.*?)</script\s*>", body, re.I | re.S):
                # Timer delay is infrastructure, not a displayed data claim.
                script = _without_timer_delays(script)
                script = re.sub(
                    r"\.style\.(?:width|height|padding|margin|gap|fontSize)\s*=\s*"
                    r"([\"'])[\d.]+(?:px|em|rem|%|vh|vw)\1",
                    "",
                    script,
                )
                for literal in _INLINE_NUMBER_RE.finditer(script):
                    if _numeric(literal.group(0)) not in supported:
                        errors.append("authored script numeric value lacks turn evidence")
        if errors:
            rejected.append(Rejected(title=title, errors=tuple(errors)))
            continue
        components.append(
            Component(
                title=title,
                source=source,
                blob=blob,
                data=dict(data),
                height_hint=min(HEIGHT_HINT_MAX, max(HEIGHT_HINT_MIN, DEFAULT_HEIGHT_HINT)),
            )
        )
    return ValidationResult(components=tuple(components), rejected=tuple(rejected))


def component_refs(components: Iterable[Component], digests: Iterable[str]) -> list[dict[str, Any]]:
    """The row's ``components[]`` entries for stored components (memo §2.4's shape)."""
    return [
        {
            "attachment": digest,
            "title": component.title,
            "source": component.source,
            "mime": "text/html",
            "height_hint": component.height_hint,
        }
        for component, digest in zip(components, digests, strict=True)
    ]
