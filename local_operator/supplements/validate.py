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
from local_operator.supplements.document import data_json, split_component
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

#: A numeric literal of three or more digits is a reject reason OUTSIDE ``viewBox``/``style``
#: (memo §4.2): layout constants live in those two attributes, plotted values must come from
#: ``<data>``, and a hard-coded 3-digit number in a script is how a fabricated series rides in.
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
_VIEWBOX_RE: Final = re.compile(r"viewBox\s*=\s*\"[^\"]*\"", re.I)
_STYLE_ATTR_RE: Final = re.compile(r"style\s*=\s*\"[^\"]*\"", re.I)

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
                    self.strings.add(text)
                    number = _numeric(text)
                    if number is not None:
                        self.numbers.add(number)
                    else:
                        # ``"12 ms"``: the evidence's own cell, unit included.
                        from local_operator.supplements.evidence import (  # noqa: PLC0415
                            _number,
                        )

                        parsed = _number(text)
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


def _check_static_numbers(body: str, data: Mapping[str, Any], units: set[str]) -> list[str]:
    """Every number PRINTED in static markup must be reproducible from the component's data.

    Static text is markup outside ``<script>``/``<style>``/``<data>``: a helper call's output
    is decided at runtime by the prelude (and by ``LO.fmt``), while a literal in markup is the
    model's own claim. The comparison is against ``LO.fmt``'s own rules via the Python mirror
    (:mod:`local_operator.supplements.fmt`) at DEFAULT precision plus the units the data
    carries, because static text has no place to put a ``digits`` option -- a baked-in rounded
    literal is exactly the failure this catches (memo §2.10, round-1 D4).
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


def _check_scan(blob: str) -> list[str]:
    """The §4.2 string scan plus the 3-digit literal rule, on the body (data excluded)."""
    body = strip_data(blob)
    lowered = body.lower()
    errors: list[str] = [
        f"rejected string {needle!r}" for needle in _BANNED if needle.lower() in lowered
    ]
    scannable = _STYLE_ATTR_RE.sub(" ", _VIEWBOX_RE.sub(" ", body))
    match = _INLINE_NUMBER_RE.search(scannable)
    if match is not None:
        errors.append(
            f"inline numeric literal {match.group(0)!r} (3+ digits) outside viewBox/style: "
            "plotted values belong in <data>"
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


def validate_output(text: str, evidence: Sequence[Dataset]) -> ValidationResult:
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
        blob = "\n".join(f"<data>{data_json({key: value})}</data>" for key, value in data.items())
        blob = f"{blob}\n{body}" if blob else body
        if len(blob.encode("utf-8")) > MAX_COMPONENT_BYTES:
            errors.append(f"component is over {MAX_COMPONENT_BYTES} bytes")
        checker = _WellFormed()
        checker.feed(body)
        checker.close()
        if checker.problem:
            errors.append(f"not well-formed: {checker.problem}")
        errors.extend(_check_scan(blob))
        data_errors, units = _check_data(data, index)
        errors.extend(data_errors)
        errors.extend(_check_static_numbers(body, data, units))
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
