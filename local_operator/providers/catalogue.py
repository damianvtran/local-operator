"""Catalogue entries → picker rows: the shaping every model picker shares.

Two surfaces offer the same catalogue and must offer it the same way: the TUI's
``/model`` and the phone's model sheet. They diverged, and the divergence was
not cosmetic — ``GET /api/models`` served 962 rows in registry order, so ~445
aggregated Radient rows rendered ahead of the first direct provider and the
phone's sheet looked like it only knew Radient and OpenRouter. The ordering was
never the phone's to invent; it already existed, in the TUI.

What lives here is the part of that shaping which is a function of the CATALOGUE
alone: which providers the user can actually run on, OpenAI's context-window
choice, and the rank. What deliberately does NOT live here is everything that is
a function of a live TUI SESSION — the follower/runtime-catalogue merge, the
sticky-account context rewrite, the rescue of a current row an authoritative
listing pruned, and the footer's wording. Those stay in ``tui/app.py`` because a
daemon has no session to ask, and pulling them down here would mean inventing
stubs for concepts the caller does not have.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Collection, Iterable

from local_operator.model.ranking import ModelRow, rank_rows

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.providers.controller import CatalogueEntry


def picker_rows(
    entries: Iterable["CatalogueEntry"],
    *,
    usable: Collection[str] | None,
    current: str | None = None,
    query: str | None = None,
    use_max_context: bool = False,
) -> tuple[list[ModelRow], int]:
    """``(rows, hidden)`` — the models this user can run, best first, and what was cut.

    HIDDEN, not demoted. A picker is a list of choices, and a row that cannot be
    chosen is not one: ``/model opus`` filling a fourteen-row window with four
    providers' opus rows of which exactly one can run costs a keystroke per miss.
    The count comes back so the caller can say "42 hidden — /login <provider>"
    instead of silently shortening the list.

    ``usable=None`` means the credential store could not be READ, which is a
    different answer from "no providers are usable" and the only honest one when
    SQLite is locked. Everything shows in that case: an empty picker claims the
    user owns no models, a claim the app has precisely failed to establish. The
    session's ``current`` model is likewise exempt from the filter, because its
    marker is what answers "what am I on" and dropping it makes a broken
    configuration invisible rather than obvious.

    ``use_max_context`` selects which window an OpenAI row advertises. The
    provider publishes a maximum the Responses API only reaches with the opt-in,
    so a caller that has not opted in must show the DEFAULT window or the picker
    promises capacity the turn will not get.

    ``hidden`` counts what the filter dropped, and is computed before any caller
    appends a rescue row — subtracting the rescue made one real hidden entry
    vanish from the footer, and flooring at zero hid the arithmetic rather than
    fixing it.
    """
    listed = list(entries)
    rows = [
        ModelRow(
            provider=entry.provider,
            model_id=entry.model_id,
            label=entry.label,
            context_window=(
                entry.default_context_window
                if entry.provider == "openai"
                and not use_max_context
                and entry.default_context_window
                else entry.context_window
            ),
            default_context_window=entry.default_context_window,
            max_context_window=entry.max_context_window,
            input_price=entry.input_price,
            output_price=entry.output_price,
            connected=entry.connected,
            aggregated=entry.aggregated,
            routed=entry.routed,
        )
        for entry in listed
        if usable is None or entry.provider in usable or entry.selector == current
    ]
    hidden = len(listed) - len(rows)
    return rank_rows(rows, query or ""), hidden
