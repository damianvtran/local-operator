"""The structured-data signal and the evidence datasets (memo §2.3 "Structured-data signal", §2.10).

WHY A SIGNAL AT ALL. The graphics half of the decision is a vendor call that costs money and
whose rubric ("would a figure help?") is only meaningful when the conversation actually
HOLDS numbers to draw. So the pre-filter asks a deterministic question first -- does the
logical turn's tool output or final answer contain structured numeric data -- and a model
"yes" without that evidence is overruled (memo §2.3 Graphics). The same extractor yields the
**evidence datasets** the generator (lane C1b) is limited to: it may plot only what it
declares, and the validator requires every numeric cell to appear here (§2.10).

THE THREE FORMS (a Boolean OR; any one makes the signal true):

* (a) a markdown / CSV / TSV table with >= 3 data rows and a NUMERIC column;
* (b) a JSON array of >= 3 objects sharing >= 1 numeric key;
* (c) >= 4 numbers sharing one unit token inside one paragraph of the final answer.

A "numeric column" excludes a strictly sequential 1..n index: a row-numbered list of names is
not data to chart. The answer's prose is not otherwise trusted -- (c) needs a shared unit, so
"improved across 3 regions" does not qualify while "12 ms, 15 ms, 9 ms, 22 ms" does.

COST. The scan is linear in the text it reads and bounded three ways (per item, per turn,
rows kept per dataset) so a 50 MB tool output costs what a 64 KiB one does. It runs on the
pre-filter's worker thread beside the ``stat`` calls, inside the memo's <= 5 ms target.

LEAF: standard library plus ``ansi`` (itself stdlib-only).
"""

import csv
import json
import re
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Sequence

from local_operator.ansi import sanitize_prompt_line
from local_operator.supplements.trigger import TurnItem

#: Minimum data rows (excluding the header) for a table / array to count as a dataset.
MIN_ROWS: Final = 3
#: Minimum numbers sharing a unit within one paragraph for form (c).
MIN_UNIT_NUMBERS: Final = 4
#: Datasets kept per turn and rows kept per dataset -- the generator's whole evidence budget.
MAX_DATASETS: Final = 6
MAX_ROWS_KEPT: Final = 200
#: Characters of tool output / answer scanned per turn, summed across items.
SCAN_BUDGET_CHARS: Final = 400_000
#: A JSON document larger than this is not parsed (a parse is the one super-linear-ish cost).
JSON_MAX_CHARS: Final = 262_144
_TITLE_MAX_CHARS: Final = 80

#: One numeric cell: optional sign / currency, grouped digits and decimals, an optional short
#: unit or percent sign. "42", "-3.5", "$1,204.50", "87%", "12 ms", "3.2GB", "4.1k".
_NUMBER: Final = re.compile(
    r"^\s*[-+]?[$€£]?\s?(?P<num>\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"\s*(?P<unit>%|[A-Za-zµ/]{1,6}(?:/[A-Za-z]{1,4})?)?\s*$"
)
_SEPARATOR_ROW: Final = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_FENCED_JSON: Final = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.DOTALL)
#: ``12 ms`` / ``3.5%`` / ``$40`` inside prose. The lookarounds reject version strings
#: (``1.2.3``), dates (``2026-10-09``) and digits glued to identifiers (``v2``, ``x86``).
_UNIT_NUMBER: Final = re.compile(
    r"(?<![\w.\-/:])(?P<cur>[$€£])?(?P<num>\d[\d,]*(?:\.\d+)?)(?![\d.]*\.\d)"
    r"(?:\s?(?P<unit>%|[A-Za-zµ]{1,10}))?(?![\w\-/:])"
)
#: Unit tokens that are not units of measure (articles, connectives, ordinals' tails).
_NOT_UNITS: Final = frozenset(
    {
        "a", "an", "and", "or", "of", "in", "on", "to", "the", "is", "are", "was", "were",
        "at", "by", "for", "as", "vs", "than", "from", "with", "st", "nd", "rd", "th",
        "i", "e", "g",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Dataset:
    """One evidence dataset: where it came from, its shape, and (bounded) rows.

    ``title`` and ``source`` are harness-derived and sanitised single lines. ``rows`` are
    kept for the generator lane and are NEVER sent to the decision vendor (the state carries
    :meth:`shape` only, memo §2.3 State).
    """

    title: str
    source: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    n_rows: int
    numeric_columns: tuple[str, ...]

    def shape(self) -> str:
        """``12 rows x 4 cols (region, p50, p99, ...)`` -- the vendor-safe description."""
        shown = ", ".join(self.columns[:6]) + (", ..." if len(self.columns) > 6 else "")
        return f"{self.n_rows} rows x {len(self.columns)} cols ({shown})"


@dataclass(frozen=True)
class Evidence:
    structured: bool
    datasets: tuple[Dataset, ...] = ()
    #: Which of the three forms fired, for the DEBUG absorption line and the tests.
    forms: tuple[str, ...] = field(default_factory=tuple)


def _number(cell: str) -> float | None:
    match = _NUMBER.match(cell)
    if match is None:
        return None
    try:
        return float(match.group("num").replace(",", ""))
    except ValueError:
        return None


def _numeric_columns(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> tuple[str, ...]:
    """Columns whose non-empty cells are (almost) all numbers, minus a sequential index."""
    found: list[str] = []
    for index, name in enumerate(columns):
        cells = [row[index].strip() for row in rows if index < len(row) and row[index].strip()]
        values = [_number(cell) for cell in cells]
        numeric = [value for value in values if value is not None]
        if len(numeric) < MIN_ROWS or len(numeric) < 0.8 * len(cells):
            continue
        if _sequential_index(numeric):
            continue
        found.append(name)
    return tuple(found)


def _sequential_index(values: Sequence[float]) -> bool:
    """1..n or 0..n-1 step 1: a row number, not a measurement."""
    if len(values) < 2:
        return False
    first = values[0]
    return first in (0.0, 1.0) and all(b - a == 1.0 for a, b in zip(values, values[1:]))


def _title(lines: Sequence[str], start: int, fallback: str) -> str:
    """The nearest non-table line above ``start`` when it reads like a heading, else fallback."""
    for index in range(start - 1, max(start - 4, -1), -1):
        text = lines[index].strip().lstrip("#*_ ").rstrip(":*_ ").strip()
        if not text:
            continue
        if "|" in text or len(text) > _TITLE_MAX_CHARS:
            break
        return sanitize_prompt_line(text, limit=_TITLE_MAX_CHARS)
    return fallback


def _dataset(
    title: str, source: str, columns: Sequence[str], rows: Sequence[Sequence[str]]
) -> Dataset | None:
    if len(rows) < MIN_ROWS:
        return None
    numeric = _numeric_columns(columns, rows)
    if not numeric:
        return None
    clean_columns = tuple(
        sanitize_prompt_line(str(c), limit=60) or f"col{i + 1}" for i, c in enumerate(columns)
    )
    clean_numeric = tuple(clean_columns[list(columns).index(c)] for c in numeric)
    kept = tuple(tuple(str(cell).strip() for cell in row) for row in rows[:MAX_ROWS_KEPT])
    return Dataset(
        title=sanitize_prompt_line(title, limit=_TITLE_MAX_CHARS) or "table",
        source=sanitize_prompt_line(source, limit=_TITLE_MAX_CHARS),
        columns=clean_columns,
        rows=kept,
        n_rows=len(rows),
        numeric_columns=clean_numeric,
    )


def _markdown_tables(text: str, source: str) -> list[Dataset]:
    lines = text.split("\n")
    found: list[Dataset] = []
    index = 0
    while index < len(lines) - 1:
        header, separator = lines[index], lines[index + 1]
        if "|" in header and _SEPARATOR_ROW.match(separator) and "-" in separator:
            columns = _split_row(header)
            rows: list[list[str]] = []
            cursor = index + 2
            while cursor < len(lines) and "|" in lines[cursor] and lines[cursor].strip():
                rows.append(_split_row(lines[cursor]))
                cursor += 1
            built = _dataset(_title(lines, index, source), source, columns, rows)
            if built is not None:
                found.append(built)
            index = cursor
            continue
        index += 1
    return found


def _split_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _delimited_blocks(text: str, source: str) -> list[Dataset]:
    """CSV / TSV: a run of >= MIN_ROWS + 1 lines with one consistent field count >= 2."""
    lines = text.split("\n")
    found: list[Dataset] = []
    for delimiter in (",", "\t"):
        index = 0
        while index < len(lines):
            width = _field_count(lines[index], delimiter)
            if width < 2:
                index += 1
                continue
            end = index
            while end < len(lines) and _field_count(lines[end], delimiter) == width:
                end += 1
            block = lines[index:end]
            if len(block) >= MIN_ROWS + 1 and "|" not in block[0]:
                parsed = list(csv.reader(block, delimiter=delimiter))
                # A header row is text; a block whose first row is all numbers has none.
                if parsed and all(_number(cell) is None for cell in parsed[0] if cell.strip()):
                    built = _dataset(_title(lines, index, source), source, parsed[0], parsed[1:])
                    if built is not None:
                        found.append(built)
            index = max(end, index + 1)
    return found


def _field_count(line: str, delimiter: str) -> int:
    if not line.strip() or delimiter not in line or len(line) > 2_000:
        return 0
    try:
        return len(next(csv.reader([line], delimiter=delimiter)))
    except (csv.Error, StopIteration):
        return 0


def _json_arrays(text: str, source: str) -> list[Dataset]:
    """Form (b): a JSON array (or an object holding one) of >= 3 uniform-ish objects."""
    candidates: list[str] = []
    stripped = text.strip()
    if stripped[:1] in "[{":
        candidates.append(stripped)
    candidates.extend(match.group(1) for match in _FENCED_JSON.finditer(text))
    found: list[Dataset] = []
    for raw in candidates:
        if len(raw) > JSON_MAX_CHARS:
            continue
        try:
            document = json.loads(raw)
        except ValueError:
            continue
        for label, array in _arrays_in(document):
            built = _array_dataset(array, f"{source}{label}")
            if built is not None:
                found.append(built)
    return found


def _arrays_in(document: Any) -> Iterable[tuple[str, list[Any]]]:
    if isinstance(document, list):
        yield "", document
    elif isinstance(document, dict):
        for key, value in list(document.items())[:12]:
            if isinstance(value, list):
                yield f".{key}", value


def _array_dataset(array: list[Any], source: str) -> Dataset | None:
    objects = [item for item in array if isinstance(item, dict)]
    if len(objects) < MIN_ROWS or len(objects) < 0.8 * len(array):
        return None
    shared = set(objects[0])
    for item in objects[1:]:
        shared &= set(item)
    numeric_keys = [
        key
        for key in objects[0]
        if key in shared
        and all(
            isinstance(item[key], (int, float)) and not isinstance(item[key], bool)
            for item in objects
        )
    ]
    if not numeric_keys:
        return None
    columns = [key for key in objects[0] if key in shared]
    rows = [[str(item[key]) for key in columns] for item in objects]
    return _dataset(source, source, [str(c) for c in columns], rows)


def _unit_paragraph(text: str) -> bool:
    """Form (c): >= MIN_UNIT_NUMBERS numbers sharing one unit token inside a paragraph."""
    for paragraph in re.split(r"\n\s*\n", text):
        counts: dict[str, int] = {}
        for match in _UNIT_NUMBER.finditer(paragraph):
            unit = match.group("unit")
            token = (
                unit.casefold()
                if unit and unit.casefold() not in _NOT_UNITS
                else match.group("cur") if match.group("cur") else None
            )
            if token:
                counts[token] = counts.get(token, 0) + 1
        if counts and max(counts.values()) >= MIN_UNIT_NUMBERS:
            return True
    return False


def extract(items: Iterable[TurnItem], answer_text: str) -> Evidence:
    """The structured-data signal and datasets over the logical turn.

    ``items`` is the accumulated turn (tool results are the ``tool`` role); ``answer_text``
    the final answer. Never raises: an exception in a heuristic is "no signal", because the
    caller's fail-open is to show nothing, and a crash here would only cost the whole job.
    """
    datasets: list[Dataset] = []
    forms: list[str] = []
    budget = SCAN_BUDGET_CHARS
    try:
        sources: list[tuple[str, str]] = [
            (item.tool_name or "tool", item.text)
            for item in items
            if item.role == "tool" and item.text and not item.is_error
        ]
        sources.append(("answer", answer_text))
        for source, text in sources:
            if budget <= 0:
                break
            text = text[:budget]
            budget -= len(text)
            for form, built in (
                ("table", _markdown_tables(text, source)),
                ("delimited", _delimited_blocks(text, source)),
                ("json", _json_arrays(text, source)),
            ):
                if built:
                    forms.append(form)
                    datasets.extend(built)
        if _unit_paragraph(answer_text):
            forms.append("units")
    except Exception:  # noqa: BLE001 — a heuristic's failure is "no signal", never a crash
        return Evidence(structured=False)
    return Evidence(
        structured=bool(datasets) or "units" in forms,
        datasets=tuple(datasets[:MAX_DATASETS]),
        forms=tuple(dict.fromkeys(forms)),
    )
