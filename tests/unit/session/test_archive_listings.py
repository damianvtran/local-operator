"""ONE archive predicate, and the four surfaces that must agree about it.

``resume._scan_sessions`` is where the filter lives, and that is the whole
design: the picker, the sidebar catalogue, the desktop catalogue and the search
digests all reach their rows through it, so one predicate is what makes them
agree about which conversations exist to be OFFERED. A second filter at a second
call site is how the sidebar and the phone came to disagree about subagent
visibility.

Four properties are load-bearing here, and each is a defect this feature was
pictured shipping:

* **Narrowing what is OFFERED must never narrow what EXISTS.** An archived
  conversation still resolves by explicit id — ``resume_dir`` is the resolver
  ``lop --resume <id>`` and the desktop's ``GET /v1/desktop/sessions/{id}`` both
  use — which is the rule ``recent_sessions`` already states for the subagent
  axis of visibility.
* **The pinned-and-archived case is settled, not left to fall out.** An archived
  session that is PINNED must not appear in a sidebar's pinned section, and it
  must not become a PHANTOM in the catalogue's off-page pinned resolution
  (a pinned id the catalogue cannot resolve and therefore renders as nothing).
* **The archive flag survives the row cache.** ``cached_session_rows`` serves
  rows out of a cache keyed on the transcript's stat, and archiving writes
  nothing to the transcript — so a cached row would otherwise keep claiming
  ``archived=False`` for as long as the conversation is not appended to, which
  for the picker's reveal toggle is precisely the row it exists to reveal.
* **An archived id is never handed to the search index.** That is what makes
  "archived conversations are not found by the standard search" true without an
  index change, and it is asserted rather than assumed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from local_operator.resume import (
    ORIGIN_SUBAGENT,
    is_user_session,
    mark_session_origin,
    recent_session_rows,
    recent_sessions,
    resume_dir,
)
from local_operator.session import session_search
from local_operator.session.archived import set_archived
from local_operator.session.catalog import cached_session_rows, load_catalog
from local_operator.session.search_index import build_index

A = "a" * 12
B = "b" * 12
C = "c" * 12


def _session(root: Path, session_id: str, *, age: float = 0.0) -> Path:
    """A conversation the store lists, with a searchable transcript.

    Written in the PERSISTED shape (``{"type": "message", "payload": {...}}``)
    rather than as a bare role/content pair, because the body digest is built
    from that shape and a fixture that is not the real thing would let the
    search assertions below pass against a store no search can read.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = directory / "transcript.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "message",
                "payload": {"role": "user", "content": f"about {session_id}"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if age:
        import os

        stamp = time.time() - age
        os.utime(transcript, (stamp, stamp))
    return directory


def _ids(rows: list[object]) -> list[str]:
    return [getattr(row, "id", rows) for row in rows]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The predicate itself
# ---------------------------------------------------------------------------


def test_the_default_listing_hides_an_archived_session(tmp_path: Path) -> None:
    _session(tmp_path, A)
    _session(tmp_path, B)
    set_archived(tmp_path, B, True)

    # SORTED, because the listing is newest-first and this fixture's ordering is
    # an artefact of which directory was written last — the ORDER of the
    # listing is pinned by the scan's own tests, and asserting it here would be
    # a second, weaker copy of that.
    assert sorted(name for name, _ in recent_sessions(tmp_path, limit=None)) == [A]
    assert sorted(
        name for name, _ in recent_sessions(tmp_path, limit=None, include_archived=True)
    ) == [A, B]


def test_a_row_states_its_archive_state_truthfully_in_both_modes(tmp_path: Path) -> None:
    """``archived`` is stamped rather than left to the field's default.

    A caller must never have to ask which mode produced the list it is holding,
    and the renderer's merge treats an absent key as no claim — so a row that is
    archived has to SAY so, not merely fail to deny it.
    """
    _session(tmp_path, A)
    _session(tmp_path, B)
    set_archived(tmp_path, B, True)

    default = recent_session_rows(tmp_path, limit=None)
    assert [(row.id, row.archived) for row in default] == [(A, False)]

    revealed = recent_session_rows(tmp_path, limit=None, include_archived=True)
    assert sorted((row.id, row.archived) for row in revealed) == [(A, False), (B, True)]


def test_an_archived_session_is_still_resolvable_by_explicit_id(tmp_path: Path) -> None:
    """A LISTING narrows what is OFFERED, never what exists.

    ``resume_dir`` is the explicit-id resolver behind ``lop --resume <id>`` and
    the desktop snapshot route, and it must not consult the archive at all: the
    whole point of an archive is that the conversation is still there.
    """
    directory = _session(tmp_path, A)
    set_archived(tmp_path, A, True)

    assert recent_sessions(tmp_path, limit=None) == []
    assert resume_dir(tmp_path, A) == directory
    assert is_user_session(directory)
    # And the session the user is standing in is still the user's own session,
    # which is what ``cleanup.delete_session``'s admission reads.
    assert (directory / "transcript.jsonl").is_file()


def test_the_picker_can_ask_for_them_and_the_other_listings_cannot(tmp_path: Path) -> None:
    """``include_archived`` is opt-in per caller, and the default is the safe one."""
    _session(tmp_path, A)
    _session(tmp_path, B)
    set_archived(tmp_path, B, True)

    assert [entry.id for entry in load_catalog(tmp_path)] == [A]
    revealed = load_catalog(tmp_path, include_archived=True)
    assert {entry.id: entry.row.archived for entry in revealed} == {A: False, B: True}


# ---------------------------------------------------------------------------
# The pinned-and-archived case
# ---------------------------------------------------------------------------


def test_a_pinned_archived_session_is_not_in_the_catalogue_and_is_not_a_phantom(
    tmp_path: Path,
) -> None:
    """Both halves of the pinned-and-archived rule, in one place because they are one rule.

    NOT LISTED: the pins store is untouched and the row is filtered with
    everything else, so it cannot reach a pinned SECTION.

    NOT A PHANTOM: ``pinned_off_page`` exists to resolve a pinned conversation
    the page did not carry, and a client asking for it gets ``None`` back for
    this id — the catalogue reports what it can render, not an id it cannot.
    The failure this guards is a client that renders a section header for a
    conversation with no row under it, off a pin it can see in the store.

    Un-archiving restores the row WITHOUT the pin being rewritten, which is what
    makes the pin store's independence from the archive store a tested fact
    rather than a hope.
    """
    _session(tmp_path, A)
    _session(tmp_path, B)
    (tmp_path / "sidebar-pins.json").write_text(json.dumps([B, A]), encoding="utf-8")
    set_archived(tmp_path, B, True)

    default = load_catalog(tmp_path, pinned_off_page=(B, A))
    assert {entry.id for entry in default} == {A}
    assert all(entry.id != B for entry in default), "not in the pinned section, not anywhere"

    # The page size is what makes the off-page resolution matter; with the pin
    # off the page it must still not resolve.
    off_page = load_catalog(tmp_path, limit=1, pinned_off_page=(B,))
    assert {entry.id for entry in off_page} == {A}

    set_archived(tmp_path, B, False)
    assert {entry.id for entry in load_catalog(tmp_path, pinned_off_page=(B,))} == {A, B}


def test_unarchiving_does_not_need_the_pin_store_to_be_rewritten(tmp_path: Path) -> None:
    """The catalogue resolves an off-page pin from the SAME ranked list.

    Asserted on the id set rather than on a ``pinned`` field: the catalogue does
    not carry one (the desktop route adds it from the pins store when it
    projects a row), so the observable claim here is that the conversation is
    back in the list it was filtered out of — with the pin store byte-identical
    throughout.
    """
    _session(tmp_path, A)
    pins = tmp_path / "sidebar-pins.json"
    pins.write_text(json.dumps([A]), encoding="utf-8")
    set_archived(tmp_path, A, True)
    assert load_catalog(tmp_path, pinned_off_page=(A,)) == []
    set_archived(tmp_path, A, False)
    assert [entry.id for entry in load_catalog(tmp_path, pinned_off_page=(A,))] == [A]
    assert json.loads(pins.read_text()) == [A]


# ---------------------------------------------------------------------------
# The row cache
# ---------------------------------------------------------------------------


def test_the_archive_flag_is_not_served_stale_from_the_row_cache(tmp_path: Path) -> None:
    """Archiving writes NOTHING the cache key observes, so the flag is re-stamped.

    The cache key is the transcript's ``(mtime, size)``. Archiving a
    conversation does not touch it, so the second call below hits the cache —
    and would serve ``archived=False`` for a row the store now holds as
    archived, which is exactly the row the picker's reveal toggle exists to
    show.
    """
    _session(tmp_path, A)
    warm = cached_session_rows(tmp_path)
    assert [(row.id, row.archived) for row in warm] == [(A, False)]

    set_archived(tmp_path, A, True)

    # Default mode: the row is gone, because the scan filtered it.
    assert cached_session_rows(tmp_path) == []
    # Revealed mode: the row is back and it SAYS it is archived, from the cache.
    revealed = cached_session_rows(tmp_path, include_archived=True)
    assert [(row.id, row.archived) for row in revealed] == [(A, True)]


# ---------------------------------------------------------------------------
# The hidden (subagent) layer
# ---------------------------------------------------------------------------


def test_an_archived_subagent_run_is_not_offered_by_the_hidden_layer(tmp_path: Path) -> None:
    """The roster layer is the same listing, so it answers the same question.

    A reveal toggle cannot reach this section, so a run offered here is a row
    the user archived with no way back from inside the surface that shows it.
    """
    _session(tmp_path, A)
    child = _session(tmp_path, B)
    mark_session_origin(child, ORIGIN_SUBAGENT)

    assert {entry.id for entry in load_catalog(tmp_path, include_subagents=True)} == {A, B}
    set_archived(tmp_path, B, True)
    assert {entry.id for entry in load_catalog(tmp_path, include_subagents=True)} == {A}
    # The two visibility axes are INDEPENDENT: revealing archives does not
    # un-hide the subagent layer (that is its own opt-in), and asking for both
    # returns the archived run with its flag set — which is the only way a
    # client learns the row it is now looking at is archived.
    assert {entry.id for entry in load_catalog(tmp_path, include_archived=True)} == {A}
    both = load_catalog(tmp_path, include_archived=True, include_subagents=True)
    assert {entry.id: entry.row.archived for entry in both} == {A: False, B: True}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_the_default_search_does_not_return_an_archived_conversation(tmp_path: Path) -> None:
    _session(tmp_path, A)
    _session(tmp_path, B)
    set_archived(tmp_path, B, True)

    default = session_search.search_store(tmp_path, "about")
    assert [match.row.id for match in default] == [A]

    revealed = session_search.search_store(tmp_path, "about", include_archived=True)
    assert sorted((match.row.id, match.row.archived) for match in revealed) == [
        (A, False),
        (B, True),
    ]


def test_an_archived_id_is_never_handed_to_the_search_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index needs no change for this feature, and that is asserted, not assumed.

    ``build_index`` is built from exactly the ids its caller listed, so the
    filter upstream of it is the whole mechanism: if an archived id ever
    reached it, an archived conversation would be reachable by a BODY match
    through a surface that does not offer it.
    """
    _session(tmp_path, A)
    _session(tmp_path, B)
    set_archived(tmp_path, B, True)

    seen: list[list[str]] = []
    real = build_index

    def spy(config_dir: Path, session_ids: list[str]) -> dict[str, str]:
        seen.append(list(session_ids))
        return real(config_dir, session_ids)

    monkeypatch.setattr(session_search, "build_index", spy)
    session_search.search_store(tmp_path, "about")
    assert seen and all(B not in ids for ids in seen), seen

    # And when a caller DOES reveal them, they are indexed like any other row —
    # the toggle's search is not a second, weaker search.
    seen.clear()
    session_search.search_store(tmp_path, "about", include_archived=True)
    assert seen and any(B in ids for ids in seen), seen


# ---------------------------------------------------------------------------
# The one caller that must opt IN: the retention guard
# ---------------------------------------------------------------------------


def test_the_retention_guard_still_ranks_an_archived_session(tmp_path: Path) -> None:
    """The single place in the codebase that overrides the listing's default.

    ``cleanup._picker_rows`` is the deletion authority's ranking: the first
    ``RECENT_KEEP`` entries are the guard against a sweep taking work someone
    still wants. With the default filter an archived conversation would drop
    out of it the moment it was archived — and the sessions a user archives are
    the OLDER ones, so it would drop straight into the ranked set every limit
    draws from, and a routine sweep would delete the archive.
    """
    from local_operator.session.cleanup import _picker_rows

    _session(tmp_path, A)
    _session(tmp_path, B)
    set_archived(tmp_path, B, True)

    assert set(_picker_rows(tmp_path)) == {A, B}


# ---------------------------------------------------------------------------
# The LIVE row: the second place the predicate has to be asked (QA round 1, Q1)
# ---------------------------------------------------------------------------


def _publish_live(root: Path, session_id: str, name: str = "synthetic owner record") -> None:
    """A running owner's discovery record for ``session_id``.

    Through the real writer with the real model, because ``decorate_rows`` reads
    the registry's own ``scan``/``classify`` and a hand-written JSON file would
    agree with a wrong implementation. ``os.getpid()`` is alive and the heartbeat
    is stamped by ``publish``, which is what makes the record classify ``live``.
    """
    import os

    from local_operator.session.runtime.registry import publish
    from local_operator.session.runtime.types import SessionRecord

    publish(
        SessionRecord(
            pid=os.getpid(),
            kind="tui",
            session_id=session_id,
            conversation_name=name,
            cwd=str(root),
            model_label="test/model",
            control_port=0,
            control_key="synthetic",
        ),
        root,
    )


def _transcript_less_session(root: Path, session_id: str) -> Path:
    """A user's session directory the SCAN cannot carry, so a live row is needed.

    No transcript and no inbox means no activity, so ``_scan_sessions`` emits no
    row for it — the exact population ``decorate_rows(include_live=True)``
    re-adds from the registry, which is where the archive predicate was missing.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "created_at.json").write_text("1700000000", encoding="utf-8")
    return directory


def test_a_live_archived_session_is_not_offered_and_is_stamped_when_revealed(
    tmp_path: Path,
) -> None:
    """THE DEFECT QA FOUND (Q1), at the catalogue it was found in.

    The live row is appended from the REGISTRY, which knows nothing about
    archives, so an archived conversation that is still running was re-added
    with the dataclass default — offered by the sidebar and reported
    ``archived: false`` while the store held it archived. That is the flow the
    command exists for: ``/archive`` acts on the CURRENT conversation.
    """
    _transcript_less_session(tmp_path, A)
    _transcript_less_session(tmp_path, B)
    _publish_live(tmp_path, A)

    # The live row is listed while it is not archived — the mechanism still works.
    assert [entry.id for entry in load_catalog(tmp_path)] == [A]

    set_archived(tmp_path, A, True)

    assert load_catalog(tmp_path) == [], "a live archived row is not offered"
    [entry] = load_catalog(tmp_path, include_archived=True)
    assert entry.id == A
    assert entry.row.archived is True, "the re-added row must carry the store's answer"

    # ...and the other live-shaped surface's sibling, the row with a transcript,
    # is unaffected: that one comes from the scan and was always stamped.
    _session(tmp_path, B)
    set_archived(tmp_path, B, True)
    assert {entry.id for entry in load_catalog(tmp_path)} == set()
    assert {
        entry.id: entry.row.archived for entry in load_catalog(tmp_path, include_archived=True)
    } == {A: True, B: True}


def _speculative_residue(root: Path, session_id: str) -> Path:
    """A speculative engage's directory as measured: the runtime's bookkeeping only.

    No ``desktop.json`` marker, no ``created_at.json`` birth sidecar, no
    transcript and no inbox — exactly what a draft's warm leaves behind while
    the id is not a session yet (measured on the real engage:
    ``.execution-lease`` + ``.session.pid``, nothing else).
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".execution-lease").write_text('{"pid": 1}', encoding="utf-8")
    (directory / ".session.pid").write_text("1", encoding="utf-8")
    return directory


def test_a_speculative_engages_residue_is_not_appended_as_a_live_row(tmp_path: Path) -> None:
    """C2 (round-2 review): the shared catalogue must not carry a draft id.

    The live-append branch re-adds rows the scan cannot carry — including a
    user's transcript-less session, which MUST stay listed (pinned above by
    ``_transcript_less_session``, which carries the ``created_at.json`` a real
    one gets at construction) — but a speculative engage's directory holds no
    marker, no birth sidecar, no transcript and no spool, so it is not a
    session yet. Without this every consumer of this function (the TUI
    sidebar included) painted the warm draft as an "Untitled conversation"
    row while its own doors called the id unknown (spec §1.6).
    """
    _transcript_less_session(tmp_path, A)
    _speculative_residue(tmp_path, B)

    # B's record — the phantom row's source — is skipped; A has no record yet,
    # so nothing is listed.
    _publish_live(tmp_path, B)
    assert load_catalog(tmp_path) == [], "a speculative engage's residue was listed"

    # The same machinery over a user session's id still appends its row (one
    # record per pid: this publish replaces B's). Without this half the cell
    # would pass on a predicate that broke the append entirely.
    _publish_live(tmp_path, A)
    assert [entry.id for entry in load_catalog(tmp_path)] == [A]
