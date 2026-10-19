"""Pure state and formatting for the ``/library`` page.

WHY THIS MODULE EXISTS. The page itself is a Textual widget with a worker and a
number of mounted children; the parts that decide *what the library shows* —
which filters are in force, which query the service receives, how a size or a
timestamp reads — are worth testing without a terminal, a config root or a
worker. That is the same split ``projects_render`` makes for the projects page
and ``tui.review``/``tui.schedule`` make for their ledgers: the widget paints,
this module decides.

The service owns the ANSWER. ``LibraryState.query`` only translates the page's
controls into :class:`~local_operator.library.service.Query`; the service still
filters, sorts and pages, so the list can never disagree with the row values a
page displays. That is why search and sort live here as query fields rather
than as client-side list comprehensions over a payload.

NO I/O, NO TEXTUAL, NO I18N. Every function is pure; the page passes catalogue
strings in where a sentence is needed, so this module stays importable in a
headless test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Final, Iterable, Mapping, Sequence

from local_operator.library import classify
from local_operator.library.metadata import MAX_TAGS_PER_ASSET, TAG_CATEGORIES
from local_operator.library.service import MAX_LIMIT, MAX_QUERY_CHARS, Query

#: The page's sort vocabulary, in the order ``s`` cycles it. Three of the four
#: names are the service's own; ``type`` is the reader's word for sorting by
#: kind (``chart.png`` and ``notes.md`` group by their kind before name), so the
#: translation is explicit in :func:`service_sort`.
SORT_CYCLE: Final = ("recent", "name", "size", "type")

#: The three status filters the rail offers. Each maps onto a different
#: ``Query`` field; ``unorganized`` is the inverse of an "organized" filter
#: because the interesting set is the one the Librarian has not reached.
STATUS_FILTERS: Final = ("highlighted", "unorganized", "favorites")

#: One page's worth of rows. The service caps at 200; the page asks for that
#: cap and paints what it gets, because a terminal cannot usefully scroll more
#: and the footer says how many matched in total.
PAGE_LIMIT: Final = MAX_LIMIT

#: The category rail's vocabulary in display order; ``all`` is the empty
#: category in the state, never a value the service sees.
CATEGORY_VALUES: Final = tuple(classify.CATEGORIES)


def service_sort(sort: str) -> str:
    """The service sort behind one page sort word.

    ``type`` is ``kind`` on the wire. Kept as one table so a future service
    sort cannot be invented by a typo in the page.
    """
    return "kind" if sort == "type" else sort


@dataclass
class LibraryState:
    """Every control the page owns, and the query they add up to.

    The cursor is the LIST's row index and is clamped by the page against the
    payload it last painted (full-page lists clamp rather than wrap, the
    documented exception shared with ``/settings`` and ``/project show``).
    """

    q: str = ""
    #: ``""`` is All; any other value is one of ``CATEGORY_VALUES``.
    category: str = ""
    #: ``""`` is All; any other value is one of ``STATUS_FILTERS``.
    status: str = ""
    #: The STABLE tag id sent to the service. The store compares ids, not
    #: labels: a name here selected zero rows because the service found no
    #: tag whose id was "q3" (QA round 1). `tag_label` is what the page paints.
    tag: str | None = None
    tag_label: str | None = None
    sort: str = "recent"
    cursor: int = 0

    # -- controls ----------------------------------------------------------

    def cycle_sort(self) -> str:
        """Advance ``s`` and return the sort now in force."""
        index = SORT_CYCLE.index(self.sort) if self.sort in SORT_CYCLE else 0
        self.sort = SORT_CYCLE[(index + 1) % len(SORT_CYCLE)]
        self.cursor = 0
        return self.sort

    def select_category(self, value: str) -> None:
        """Apply a category, or clear it when the same row is chosen twice.

        Choosing the active row again is the rail's clear gesture: a filter
        strip whose only way back to All is a separate row makes "show
        everything" cost an aim, and the All row is already one.
        """
        self.category = "" if value == self.category else value
        self.cursor = 0

    def select_status(self, value: str) -> None:
        self.status = "" if value == self.status else value
        self.cursor = 0

    def select_tag(self, tag_id: str, name: str | None = None) -> None:
        """Select by ID, carry the human label beside it for painting."""
        if self.tag == tag_id:
            self.tag = None
            self.tag_label = None
        else:
            self.tag = tag_id
            self.tag_label = name or tag_id
        self.cursor = 0

    def clear_filters(self) -> None:
        self.category = ""
        self.status = ""
        self.tag = None
        self.tag_label = None
        self.cursor = 0

    def set_query(self, text: str) -> None:
        self.q = text
        self.cursor = 0

    def move(self, delta: int, row_count: int) -> int:
        """Move the cursor by ``delta`` and clamp it; returns the new index."""
        last = max(row_count - 1, 0)
        self.cursor = max(0, min(self.cursor + delta, last))
        return self.cursor

    def clamp(self, row_count: int) -> int:
        self.cursor = max(0, min(self.cursor, max(row_count - 1, 0)))
        return self.cursor

    def filtered(self) -> bool:
        return bool(self.q.strip() or self.category or self.status or self.tag)

    # -- the service query -------------------------------------------------

    def query(self, *, limit: int = PAGE_LIMIT) -> Query:
        """The one Query the worker sends for the state on screen."""
        return Query(
            q=self.q.strip()[:MAX_QUERY_CHARS],
            categories=frozenset({self.category}) if self.category else frozenset(),
            highlighted=True if self.status == "highlighted" else None,
            organized=False if self.status == "unorganized" else None,
            favorite=True if self.status == "favorites" else None,
            tag=self.tag,
            sort=service_sort(self.sort),
            limit=limit,
        )


def category_options(counts: Mapping[str, Any] | None) -> list[tuple[str, int]]:
    """``(value, count)`` for the category rail, All first.

    Counts are catalog-wide (the service says so on purpose: the rail must not
    empty itself while a search narrows the list), so All is their sum rather
    than the filtered total.
    """
    bucket = (counts or {}).get("category", {}) if isinstance(counts, Mapping) else {}
    total = sum(int(value) for value in bucket.values()) if isinstance(bucket, Mapping) else 0
    rows: list[tuple[str, int]] = [("", total)]
    for value in CATEGORY_VALUES:
        rows.append((value, int(bucket.get(value, 0)) if isinstance(bucket, Mapping) else 0))
    return rows


def tag_groups(tags: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """The rail's tag rows grouped by their category, in the frozen order.

    A tag row without a category the vocabulary knows is filed under ``custom``
    rather than dropped: it exists in the store, so hiding it would make a tag
    the user can see in an editor un-findable in the only filter that lists it.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in TAG_CATEGORIES}
    for tag in tags:
        category = str(tag.get("category") or "custom")
        if category not in grouped:
            category = "custom"
        grouped[category].append(tag)
    return [(name, grouped[name]) for name in TAG_CATEGORIES if grouped.get(name)]


def format_size(size: int | float | None) -> str:
    """A byte count as a compact human string; ``—`` when the row has none."""
    if size is None:
        return "—"
    value = float(size)
    if value < 1024:
        return f"{int(value)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024.0
        if value < 1024 or unit == "TB":
            if value >= 100:
                return f"{value:.0f} {unit}"
            if value >= 10:
                return f"{value:.1f} {unit}"
            return f"{value:.2f} {unit}"
    return f"{value:.2f} TB"


def format_when(stamp: float | None, *, now: float | None = None, now_label: str = "now") -> str:
    """Relative time for a row or a footer; a date once it is a week old.

    Sub-minute stamps read ``now`` rather than ``0m``, and a missing stamp is
    an empty string: the caller decides what an unknown date says, because
    that is catalogue copy.
    """
    if not stamp:
        return ""
    current = time.time() if now is None else now
    delta = max(0.0, current - float(stamp))
    if delta < 60:
        return now_label
    if delta < 3600:
        return f"{int(delta // 60)}m"
    if delta < 86_400:
        return f"{int(delta // 3600)}h"
    if delta < 7 * 86_400:
        return f"{int(delta // 86_400)}d"
    return time.strftime("%Y-%m-%d", time.localtime(float(stamp)))


def ask_reference(row: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Which catalogue sentence names ``row``, and with what values.

    A digest-only asset has no path to name — the index deliberately keeps one
    only once a surface materialises it — so the id is the reference, and the
    agent can resolve it through the Library's own content route rather than a
    made-up path. This function only chooses the SHAPE; the page renders the
    chosen catalogue message so the sentence stays translated (an ICU template
    is not a `str.format` string: its placeholders are typed arguments).
    """
    name = str(row.get("name") or row.get("derived_name") or row.get("id") or "")
    kind = str(row.get("kind") or "other")
    ref = row.get("ref") if isinstance(row.get("ref"), Mapping) else {}
    path = str(ref.get("path") or "") if isinstance(ref, Mapping) else ""
    sessions = row.get("occurrences") or []
    session = ""
    if isinstance(sessions, Sequence) and sessions:
        first = sessions[0] if isinstance(sessions[0], Mapping) else {}
        session = str(first.get("session_title") or first.get("session_id") or "")
    if path:
        return "path", {"name": name, "path": path, "kind": kind, "session": session}
    return "digest", {"name": name, "id": str(row.get("id") or ""), "kind": kind}


def tag_names(raw: str) -> list[str]:
    """Split the form's comma grammar into unique names, order preserved.

    The store is the authority on what a tag may be called; this only removes
    the separators a person typed. More names than the store allows is the
    form's refusal to make, so the list is returned whole rather than
    silently truncated (a truncation would drop the one tag the user added).
    """
    seen: set[str] = set()
    names: list[str] = []
    for part in raw.replace("\n", ",").split(","):
        name = part.strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def too_many_tags(names: Iterable[str]) -> bool:
    return len(list(names)) > MAX_TAGS_PER_ASSET
