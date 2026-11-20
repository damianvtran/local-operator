"""Live-state row labels render through ``wire.session_status`` keys.

WHY THIS FILE EXISTS. After the terminal labels were keyed (PR #4312) the
LIVE rungs of ``CatalogEntry.status`` — approval/answer needed, not answering,
working, open, ready, headless exec, waiting and delegating — were still English
literals, so every locale's desktop sidebar, mobile relay and TUI sidebar showed
English on a running session. Two properties are pinned here:

* the English source is BYTE-IDENTICAL to the pre-extraction literals (clients,
  feed dedupe keys and other tests carry these spellings), including the
  heartbeat suffix and the 1/2 plural forms;
* each label is DISPATCHED through its key: an isolated fixture locale with
  opaque markers (not a translation contribution) replaces every label, so a
  literal left behind would show up as English here.
"""

from __future__ import annotations

import json

import pytest

from local_operator.i18n import catalogues
from local_operator.resume import SessionRow
from local_operator.session import catalog

ROW = SessionRow("synthetic001", 1.0, "Synthetic session")


def status(**fields) -> str:
    return catalog.CatalogEntry(ROW._replace(**fields)).status


#: (row fields, pre-extraction English label). Copied from the literals this
#: change removed, so a wording drift in the catalogue fails here.
EN_CASES = [
    (dict(pending="approval", live_state="busy"), "Approval needed"),
    (dict(pending="ask", live_state="busy"), "Answer needed"),
    (dict(live_state="wedged"), "Not answering · process alive"),
    (
        dict(live_state="wedged", heartbeat_age_s=240.0),
        "Not answering · process alive (last heartbeat 4m ago)",
    ),
    (dict(live_state="busy"), "Working"),
    (dict(live_state="attached"), "Open"),
    (dict(live_state="idle"), "Ready"),
    (dict(live_state="idle", kind="exec"), "Running headless (exec)"),
    (dict(live_state="busy", waiting_on=1), "Waiting on 1 job"),
    (dict(live_state="busy", waiting_on=2), "Waiting on 2 jobs"),
    (
        dict(live_state="busy", waiting_on=1, subagents_running=1),
        "Waiting on 1 job · 1 subagent running",
    ),
    (
        dict(live_state="busy", waiting_on=2, subagents_running=2),
        "Waiting on 2 jobs · 2 subagents running",
    ),
    (
        dict(live_state="busy", waiting_on=1, subagents_running=2),
        "Waiting on 1 job · 2 subagents running",
    ),
]

#: (running, queued, jobs) -> the pre-extraction ``delegating_label`` output.
DELEGATING_EN = [
    ((1, 0, 0), "1 subagent running"),
    ((2, 0, 0), "2 subagents running"),
    ((0, 1, 0), "1 subagent queued"),
    ((0, 2, 0), "2 subagents queued"),
    ((1, 1, 0), "1 subagent running · 1 queued"),
    ((2, 1, 0), "2 subagents running · 1 queued"),
    ((0, 0, 1), "1 job running"),
    ((0, 0, 2), "2 jobs running"),
    ((1, 2, 3), "3 jobs running · 1 subagent running · 2 queued"),
    ((0, 1, 2), "2 jobs running · 1 subagent queued"),
]


@pytest.fixture
def english(monkeypatch):
    monkeypatch.setattr(catalog, "_label_locale", lambda: "en")


@pytest.mark.usefixtures("english")
@pytest.mark.parametrize(("fields", "label"), EN_CASES)
def test_english_live_labels_are_byte_identical(fields, label):
    assert status(**fields) == label


@pytest.mark.usefixtures("english")
@pytest.mark.parametrize(("counts", "label"), DELEGATING_EN)
def test_english_delegating_label_is_byte_identical(counts, label):
    assert catalog.delegating_label(*counts) == label


@pytest.mark.usefixtures("english")
def test_wedged_status_is_the_bare_wedged_label():
    assert catalog.wedged_status() == "Not answering · process alive"
    assert status(live_state="wedged") == catalog.wedged_status()


#: Opaque fixture messages for the isolated ``fr`` catalogue: every one names
#: its key and exposes its parameters, and the plural ones route through ICU
#: so French's own plural arms (``one`` covers 0 and 1) are what select them.
FIXTURE = {
    "approval_needed": "fx:approval_needed",
    "answer_needed": "fx:answer_needed",
    "wedged": "fx:wedged",
    "wedged_heartbeat": "fx:wedged_heartbeat[{duration}]",
    "working": "fx:working",
    "attached": "fx:attached",
    "ready": "fx:ready",
    "exec_headless": "fx:exec_headless",
    "waiting": "fx:waiting[{jobs, plural, one {one:#} other {other:#}}]",
    "waiting_subagents": (
        "fx:waiting_subagents[{jobs, plural, one {one:#} other {other:#}}"
        "|{subagents, plural, one {one:#} other {other:#}}]"
    ),
    "delegating_jobs": "fx:delegating_jobs[{count, plural, one {one:#} other {other:#}}]",
    "delegating_running": "fx:delegating_running[{count, plural, one {one:#} other {other:#}}]",
    "delegating_queued": "fx:delegating_queued[{count, plural, one {one:#} other {other:#}}]",
    "delegating_queued_addend": (
        "fx:delegating_queued_addend[{count, plural, one {one:#} other {other:#}}]"
    ),
}


@pytest.fixture
def fixture_fr(tmp_path, monkeypatch):
    source = catalogues._CATALOGUES / "en" / "wire.session_status.json"
    messages = json.loads(source.read_text())
    prefix = "wire.session_status."
    # The fixture must cover every key this change added, or a missing entry
    # would silently fall back to English and hide an unkeyed label.
    assert {prefix + name for name in FIXTURE} <= set(messages)
    localized = dict(messages)
    localized.update({prefix + name: text for name, text in FIXTURE.items()})
    for locale, payload in (("en", messages), ("fr", localized)):
        (tmp_path / locale).mkdir()
        (tmp_path / locale / "wire.session_status.json").write_text(json.dumps(payload))
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    from local_operator.i18n import resolve

    # Eligibility is the fixture's too: never alter the real shipped set.
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
    monkeypatch.setattr(catalog, "_label_locale", lambda: "fr")


@pytest.mark.usefixtures("fixture_fr")
@pytest.mark.parametrize(
    ("fields", "label"),
    [
        (dict(pending="approval", live_state="busy"), "fx:approval_needed"),
        (dict(pending="ask", live_state="busy"), "fx:answer_needed"),
        (dict(live_state="wedged"), "fx:wedged"),
        (dict(live_state="wedged", heartbeat_age_s=240.0), "fx:wedged_heartbeat[4m]"),
        (dict(live_state="busy"), "fx:working"),
        (dict(live_state="attached"), "fx:attached"),
        (dict(live_state="idle"), "fx:ready"),
        (dict(live_state="idle", kind="exec"), "fx:exec_headless"),
        (dict(live_state="busy", waiting_on=1), "fx:waiting[one:1]"),
        (dict(live_state="busy", waiting_on=2), "fx:waiting[other:2]"),
        (
            dict(live_state="busy", waiting_on=2, subagents_running=1),
            "fx:waiting_subagents[other:2|one:1]",
        ),
    ],
)
def test_every_live_label_dispatches_through_its_key(fields, label):
    item = catalog.CatalogEntry(ROW._replace(**fields))
    assert item.status == label


@pytest.mark.usefixtures("fixture_fr")
@pytest.mark.parametrize(
    ("counts", "label"),
    [
        ((2, 0, 0), "fx:delegating_running[other:2]"),
        ((0, 1, 0), "fx:delegating_queued[one:1]"),
        (
            (2, 1, 3),
            "fx:delegating_jobs[other:3] · fx:delegating_running[other:2]"
            " · fx:delegating_queued_addend[one:1]",
        ),
    ],
)
def test_delegating_segments_dispatch_through_keys(counts, label):
    assert catalog.delegating_label(*counts) == label


@pytest.mark.usefixtures("fixture_fr")
def test_wedged_status_follows_the_label_locale():
    assert catalog.wedged_status() == "fx:wedged"
