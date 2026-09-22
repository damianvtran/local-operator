"""An agent-opened run is HIDDEN by default and LISTED when the operator asked for it.

Two halves of one defect, both reproduced on a real isolated store before this
module existed:

* an agent that fans out long-lived parallel work with ``lop exec`` produces a
  session that is invisible in the desktop sidebar, the ``/resume`` picker, the
  phone list and search — because visibility here is an opt-in ALLOW-LIST
  (``resume.USER_ORIGINS``) and every one of those surfaces funnels through the
  same scan and the same predicate. That default is right for a throwaway run and
  wrong for a workstream the operator asked for;
* the 2026-09-18 incident was not "a session was listed" — it was that a
  machine-started session was indistinguishable from one the operator had
  opened. So the value that un-hides a run must also carry WHO opened it, and
  the record has to be written at the only moment that fact exists: creation.

The fix is therefore one new origin value, minted only by an explicit intent
(``lop exec --workstream``), registered in the allow-list — plus the opener's
identity written beside it. This module pins the value, the registration, the
attribution, the surfaces that must AGREE about it, and the guards that must not
move because of it (``created_here``, the nested-session escape, the refusals).

Nothing here writes a marker the product would not: the store is a tmp_path,
``LOCAL_OPERATOR_CONFIG_DIR`` names it, and every read goes through the real
readers (``session.catalog.load_catalog``, ``resume.recent_session_rows``,
``session_search.search_store``, the desktop route).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator import resume
from local_operator.agent_shell import (
    AGENT_SHELL_ENV,
    ALLOW_NESTED_SESSION_ENV,
    MAY_DELEGATE_ENV,
    exec_session_refusal,
    interactive_session_refusal,
    stamp_agent_shell_session,
)
from local_operator.resume import (
    OPENED_BY_KEYS,
    ORIGIN_AGENT_SHELL,
    ORIGIN_AGENT_WORKSTREAM,
    ORIGIN_FORK,
    ORIGIN_NAME,
    is_user_session,
    mark_session_origin,
    recent_session_rows,
    session_origin,
    workstream_opened_by,
    write_session_title,
)
from local_operator.scratchpad import SCRATCHPAD_PATH_ENV
from local_operator.session import catalog
from local_operator.session.session_search import search_store

REQUESTER_ID = "req000000001"


def _store(tmp_path: Path, name: str) -> Path:
    """An isolated config root with a ``sessions/`` directory."""
    root = tmp_path / name
    (root / "sessions").mkdir(parents=True)
    return root


def _session(root: Path, session_id: str, name: str) -> Path:
    """A real conversation directory: transcript, birth stamp and stored title.

    Written through the same readers the listings use (``write_session_title``
    rather than a hand-rolled sidecar), so a row that appears here appears
    because the product can name it, not because a fixture made it nameable.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "created_at.json").write_text("1700000000.0", encoding="utf-8")
    (directory / "transcript.jsonl").write_text(
        '{"id":"e1","ts":1,"type":"message",'
        '"payload":{"kind":"message","role":"user","content":[{"text":"go"}]}}\n',
        encoding="utf-8",
    )
    write_session_title(directory, name, user_set=False, past_names=[])
    return directory


def _surfaces(root: Path) -> dict[str, set[str]]:
    """Every listing in the tree that decides membership, asked of the REAL reader.

    Named one by one because the property under test IS agreement: the desktop
    sidebar and the TUI's both go through ``session.catalog``, the picker
    through ``recent_session_rows``, the phone's history through the same reader
    with ``strict=True``, search through ``search_store``, and the retention
    policy's recent-N guard through ``cleanup._picker_rows`` — the listing that
    decides what may be DELETED. A new surface that asks its own question is
    what this dict is here to catch.
    """
    from local_operator.session.cleanup import _picker_rows

    store = root / "sessions"
    return {
        "resume.is_user_session": {
            entry.name for entry in store.iterdir() if is_user_session(entry)
        },
        "desktop+TUI sidebar": {entry.id for entry in catalog.load_catalog(root)},
        "/resume picker": {row.id for row in recent_session_rows(root)},
        "phone history": {row.id for row in recent_session_rows(root, strict=True)},
        "session search": {match.row.id for match in search_store(root, "go")},
        "cleanup recent-N guard": set(_picker_rows(root)),
    }


@pytest.fixture
def agent_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shape an agent's ``bash`` tool hands to a child, both markers set."""
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")
    monkeypatch.setenv(MAY_DELEGATE_ENV, "1")


@pytest.fixture
def requester(tmp_path: Path, agent_shell: None, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The requesting session, as a fan-out actually presents itself.

    A DELEGATED child ("on whose behalf"): it carries ``label``/``agent`` in its
    own marker, exactly as ``harness.subagent`` writes them, and its scratchpad
    path is what the ``bash`` tool exports to name the session a child belongs
    to — which is the only source the stamp has for the opener's identity.
    """
    root = _store(tmp_path, "store")
    requester = _session(root, REQUESTER_ID, "lo-1428-review")
    mark_session_origin(requester, resume.ORIGIN_SUBAGENT, label="1428 review", agent="coder")
    (requester / "scratchpad").mkdir()
    monkeypatch.setenv(SCRATCHPAD_PATH_ENV, str(requester / "scratchpad"))
    return requester


# --- the value, and the registration that makes it visible everywhere ----------


def test_the_workstream_origin_is_the_users_own_and_is_a_value_of_its_own() -> None:
    """One registration is the whole visibility change, and it must not be a reuse.

    ``USER_ORIGINS`` is consulted by ``is_user_session`` AND by the scan's
    ``_is_hidden_origin``, which are the two spellings every listing reaches, so
    registering the value here is what lists it in all six surfaces at once.

    It is a NEW value rather than ``agent-shell`` on purpose: that constant
    answers one question — did a command from an agent's shell open this? — and
    every listing filtering on it wants exactly that answer. Overloading it would
    make the hidden case unhideable, because the two populations would share a
    value; this asserts the two cannot be conflated.
    """
    assert ORIGIN_AGENT_WORKSTREAM in resume.USER_ORIGINS
    assert ORIGIN_AGENT_WORKSTREAM != ORIGIN_AGENT_SHELL
    assert ORIGIN_AGENT_SHELL not in resume.USER_ORIGINS
    # The predicate agrees for both spellings of the decision it makes.
    assert resume._is_hidden_origin(ORIGIN_AGENT_SHELL) is True
    assert resume._is_hidden_origin(ORIGIN_AGENT_WORKSTREAM) is False


def test_an_unflagged_agent_run_is_hidden_in_every_surface(
    tmp_path: Path, agent_shell: None
) -> None:
    """The property the operator's question turns on: the default is UNCHANGED.

    A `lop exec` from a delegating agent shell that asks for nothing is stamped
    ``agent-shell`` and must stay out of every listing — that is the whole point
    of the opt-in default, and a change that published these would refill the
    sidebar with throwaway runs.
    """
    root = _store(tmp_path, "store")
    theirs = _session(root, "aaaa00000001", "The operator's own work")
    ephemeral = _session(root, "bbbb00000001", "Throwaway review run")
    assert stamp_agent_shell_session(ephemeral, created_here=True) is True
    assert session_origin(ephemeral) == ORIGIN_AGENT_SHELL

    for surface, ids in _surfaces(root).items():
        assert theirs.name in ids, (surface, "the control session is missing")
        assert ephemeral.name not in ids, (surface, "a hidden run leaked into a listing")


def test_a_workstream_run_is_listed_in_every_surface_and_carries_its_opener(
    tmp_path: Path, requester: Path
) -> None:
    """The surfaces AGREE, which is the property the single registration buys.

    One assertion per surface, from one row, so a future surface that asks its
    own question about visibility fails here rather than silently disagreeing
    with the sidebar — the shape that produced the bug this change fixes.
    """
    root = requester.parents[1]
    workstream = _session(root, "cccc00000001", "Fan-out audit")
    assert (
        stamp_agent_shell_session(workstream, created_here=True, delegated_workstream=True) is True
    )
    assert session_origin(workstream) == ORIGIN_AGENT_WORKSTREAM

    for surface, ids in _surfaces(root).items():
        assert workstream.name in ids, (surface, "a workstream is missing from a listing")

    expected = {"agent": "coder", "label": "1428 review", "session": REQUESTER_ID}
    for row in recent_session_rows(root):
        assert row.id == workstream.name
        assert row.opened_by == expected
    assert workstream_opened_by(workstream) == expected


# --- the flag chooses the value, and only where the stamp already runs ---------


def test_the_stamp_writes_the_workstream_value_only_when_asked(
    tmp_path: Path, agent_shell: None
) -> None:
    """Two arms, one guard: the flag changes WHICH value, never WHETHER to stamp."""
    asked = tmp_path / "sessions" / "asked0000001"
    unasked = tmp_path / "sessions" / "unasked000001"
    assert stamp_agent_shell_session(asked, created_here=True, delegated_workstream=True) is True
    assert stamp_agent_shell_session(unasked, created_here=True) is True
    assert session_origin(asked) == ORIGIN_AGENT_WORKSTREAM
    assert session_origin(unasked) == ORIGIN_AGENT_SHELL
    assert is_user_session(asked) is True
    assert is_user_session(unasked) is False


def test_the_nested_session_escape_still_stamps_as_agent_shell_unless_asked(
    tmp_path: Path, agent_shell: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented QA escape keeps stamping, and defaults to the hidden value.

    A harness's child is nobody's workstream unless it says so, so the escape
    stays a hidden stamp — while ``--workstream`` is honoured through the same
    route, because the flag is what the operator asked for and the escape is
    merely how the run was allowed to start.
    """
    monkeypatch.setenv(ALLOW_NESTED_SESSION_ENV, "1")
    escaped = tmp_path / "sessions" / "escaped000001"
    asked = tmp_path / "sessions" / "escaped000002"
    assert stamp_agent_shell_session(escaped, created_here=True) is True
    assert stamp_agent_shell_session(asked, created_here=True, delegated_workstream=True) is True
    assert session_origin(escaped) == ORIGIN_AGENT_SHELL
    assert session_origin(asked) == ORIGIN_AGENT_WORKSTREAM


def test_a_conversation_this_call_did_not_create_is_never_stamped(
    tmp_path: Path, agent_shell: None
) -> None:
    """``--resume`` adopts the operator's own conversation — the mirror bug.

    Hiding their chat is as wrong as listing a machine's run, and the flag must
    not become a way to do it: ``created_here`` outranks both values.
    """
    existing = _session(tmp_path, "theirs000000", "The operator's own work")
    assert stamp_agent_shell_session(existing, created_here=False) is False
    assert (
        stamp_agent_shell_session(existing, created_here=False, delegated_workstream=True) is False
    )
    assert not (existing / ORIGIN_NAME).exists()
    assert is_user_session(existing) is True


def test_an_ordinary_terminal_run_is_not_stamped_even_with_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's own terminal: no agent-shell marker, so the flag is a no-op.

    Documented behaviour rather than an accident — the flag only has meaning
    where the stamp does, and their own `lop exec --workstream` is an ordinary
    session they opened.
    """
    monkeypatch.delenv(AGENT_SHELL_ENV, raising=False)
    monkeypatch.delenv(MAY_DELEGATE_ENV, raising=False)
    directory = tmp_path / "sessions" / "ordinary0001"
    assert (
        stamp_agent_shell_session(directory, created_here=True, delegated_workstream=True) is False
    )
    assert not (directory / ORIGIN_NAME).exists()
    assert is_user_session(directory) is True


def test_a_resumed_workstream_keeps_its_visibility(tmp_path: Path, requester: Path) -> None:
    """``lop exec --resume`` of a workstream does not re-stamp, so it stays listed.

    The rule is the same one every other origin obeys — the marker is written
    once, at creation, and a resumed run must neither hide a listed session nor
    un-hide a hidden one. Both directions are asserted here because they are the
    same code path.
    """
    root = requester.parents[1]
    listed = _session(root, "dddd00000001", "Long-running fan-out")
    mark_session_origin(listed, ORIGIN_AGENT_WORKSTREAM, opened_by={"session": REQUESTER_ID})
    hidden = _session(root, "eeee00000001", "Throwaway")
    mark_session_origin(hidden, ORIGIN_AGENT_SHELL)

    assert stamp_agent_shell_session(listed, created_here=False, delegated_workstream=True) is False
    assert stamp_agent_shell_session(hidden, created_here=False, delegated_workstream=True) is False
    assert session_origin(listed) == ORIGIN_AGENT_WORKSTREAM
    assert session_origin(hidden) == ORIGIN_AGENT_SHELL
    assert is_user_session(listed) is True
    assert is_user_session(hidden) is False


# --- attribution: WHO asked, and on whose behalf ---------------------------------


def test_the_marker_records_the_requesting_session_and_its_role(
    tmp_path: Path, requester: Path
) -> None:
    """The durable record names the opener, its task label and its session id.

    Read from the two sources a child actually has — the requesting session's
    directory (from the scratchpad path its shell exported) and that session's
    own marker — and never inferred from a neighbouring field.
    """
    import json

    directory = tmp_path / "sessions" / "attrib0000001"
    assert (
        stamp_agent_shell_session(directory, created_here=True, delegated_workstream=True) is True
    )
    payload = json.loads((directory / ORIGIN_NAME).read_text(encoding="utf-8"))
    assert payload["origin"] == ORIGIN_AGENT_WORKSTREAM
    assert payload["opened_by"] == {
        "agent": "coder",
        "label": "1428 review",
        "session": REQUESTER_ID,
        "name": "lo-1428-review",
    }
    # The WIRE object is the frozen subset: three keys, all nullable. The name
    # rides the marker for the durable record and is deliberately not published.
    assert tuple(workstream_opened_by(directory) or {}) == OPENED_BY_KEYS


def test_an_opener_that_cannot_be_read_is_recorded_as_null_not_invented(
    tmp_path: Path, agent_shell: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing source is ``None`` on every member, never a guess.

    The attribution decorates the row the operator uses to tell whose work this
    is, so a fabricated answer is worse than none. Two shapes: no scratchpad at
    all (a run whose parent has no session directory), and a scratchpad whose
    parent carries no marker of its own.
    """
    monkeypatch.delenv(SCRATCHPAD_PATH_ENV, raising=False)
    anonymous = tmp_path / "sessions" / "anon00000001"
    assert (
        stamp_agent_shell_session(anonymous, created_here=True, delegated_workstream=True) is True
    )
    assert workstream_opened_by(anonymous) == {"agent": None, "label": None, "session": None}

    top_level = _session(tmp_path, "toplevel0001", "The operator's manager")
    monkeypatch.setenv(SCRATCHPAD_PATH_ENV, str(top_level / "scratchpad"))
    (top_level / "scratchpad").mkdir()
    named = tmp_path / "sessions" / "named0000001"
    assert stamp_agent_shell_session(named, created_here=True, delegated_workstream=True) is True
    assert workstream_opened_by(named) == {
        "agent": None,
        "label": None,
        "session": top_level.name,
    }


def test_opened_by_is_read_only_for_a_workstream_marker(tmp_path: Path) -> None:
    """Every other origin, and every broken marker, reads as "nobody machine-opened"."""
    plain = _session(tmp_path, "plain0000001", "Their own work")
    assert workstream_opened_by(plain) is None

    for session_id, origin in (
        ("shell00000001", ORIGIN_AGENT_SHELL),
        ("fork000000001", ORIGIN_FORK),
        ("sub0000000001", resume.ORIGIN_SUBAGENT),
    ):
        directory = _session(tmp_path, session_id, session_id)
        mark_session_origin(directory, origin)
        assert workstream_opened_by(directory) is None

    torn = _session(tmp_path, "torn00000001", "Torn marker")
    (torn / ORIGIN_NAME).write_text('{"origin": "agent-workstream", "opened_b', encoding="utf-8")
    assert workstream_opened_by(torn) is None

    half = _session(tmp_path, "half00000001", "Half marker")
    (half / ORIGIN_NAME).write_text(
        '{"origin": "agent-workstream", "opened_by": "not an object"}', encoding="utf-8"
    )
    assert workstream_opened_by(half) == {"agent": None, "label": None, "session": None}


# --- the guards that must not move ----------------------------------------------


def test_a_session_without_task_is_refused_exactly_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--workstream`` does not soften the refusal: it is not consulted by it.

    The guard answers from the calling session's live inventory, so a role that
    does not delegate cannot reach the workstream route either — the flag adds a
    disposition to an already-allowed run, never an allowance.
    """
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")
    monkeypatch.setenv(MAY_DELEGATE_ENV, "")
    assert exec_session_refusal() is not None

    monkeypatch.setenv(MAY_DELEGATE_ENV, "1")
    assert exec_session_refusal() is None


def test_the_interactive_path_stays_refused_for_every_agent_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both halves of the guard, checked together: `--workstream` changes neither.

    A change that relaxed one predicate and not the other is the defect the
    AGENTS.md paragraph exists to prevent, and an agent has no terminal however
    the run is labelled.
    """
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")
    monkeypatch.setenv(MAY_DELEGATE_ENV, "1")
    assert interactive_session_refusal() is not None
    monkeypatch.setenv(MAY_DELEGATE_ENV, "")
    assert interactive_session_refusal() is not None


def test_the_desktop_row_publishes_the_frozen_opened_by_object(
    tmp_path: Path, requester: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wire contract the sidebar is written against, read off the REAL route.

    ``opened_by`` is ``{agent, label, session}`` — all nullable, nothing renamed
    and nothing added — on a workstream row, and ``None`` on every other row. The
    UI is being written against exactly this, so it is asserted at the route
    rather than at the row builder, and through the bearer-gated endpoint the app
    actually calls. ``resume.OPENED_BY_KEYS`` owns the three names.
    """
    import os

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from local_operator.config import ConfigManager
    from local_operator.server.routes import desktop_sessions as route_module

    root = requester.parents[1]
    workstream = _session(root, "ffff00000001", "Fan-out audit")
    mark_session_origin(
        workstream,
        ORIGIN_AGENT_WORKSTREAM,
        opened_by={"agent": "coder", "label": "1428 review", "session": REQUESTER_ID},
    )
    _session(root, "111100000001", "The operator's own work")

    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "token")
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.include_router(route_module.router)

    with TestClient(app) as client:
        answer = client.get(
            "/v1/desktop/sessions?limit=50", headers={"Authorization": "Bearer token"}
        )
    assert answer.status_code == 200, answer.text
    rows = {row["id"]: row for row in answer.json()["result"]["sessions"]}
    assert rows["ffff00000001"]["opened_by"] == {
        "agent": "coder",
        "label": "1428 review",
        "session": REQUESTER_ID,
    }
    # Not an omission on the other row: the key is present and null, so a client
    # cannot read its absence as a claim.
    assert "opened_by" in rows["111100000001"]
    assert rows["111100000001"]["opened_by"] is None
