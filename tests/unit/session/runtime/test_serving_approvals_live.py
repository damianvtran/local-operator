"""``ServingSessionHandle``'s approval gate follows ``tool_approval_mode`` live.

The gate the RUNTIME's tools actually consult is ``_auto_approve`` on the
handle (``_install_gates``), read per decision. ``follow_config`` hangs a
listener on the process ``ConfigWatcher`` so a ``config.yml`` write moves it
within a poll — TIGHTENING always, and LOOSENING only when this process made
the write through the operator's own settings facade (issue #1282: the party
being gated must not be the authority that may lower its own gate, so a
model-run shell command rewriting ``config.yml`` is refused, as is an editor,
a second pane, and ``lop config edit``). Two further limits hold: a card
already PARKED is never auto-answered or auto-denied, and an explicit
``--yolo`` pin ignores the key.

``poll_now()`` is the tick; nothing here waits on the clock.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import ConfigWatcher, _reset_for_tests, process_watcher
from local_operator.harness.approval import (
    LOOSENING_REFUSED_NOTICE,
    REARM_KEPT_BY_AUTO_NOTICE,
)
from local_operator.session.approval_policy import OVERRIDE_FILENAME
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.runtime.test_serving import FakeSession

#: A model-authored shell write: another process, the raw file, no facade. This
#: is the write shape the issue names (``python - <<EOF`` setting the key), kept
#: as its own program so the test can run it as a real second process.
_RAW_APPROVAL_WRITE = (
    "import os, pathlib, sys, yaml\n"
    "path = pathlib.Path(sys.argv[1])\n"
    "doc = yaml.safe_load(path.read_text()) or {}\n"
    "doc.setdefault('values', {})['tool_approval_mode'] = 'auto'\n"
    "tmp = path.with_name(path.name + '.tmp')\n"
    "tmp.write_text(yaml.safe_dump(doc))\n"
    "os.replace(tmp, path)\n"
)


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _write_elsewhere(config_dir, key: str, value: Any) -> None:
    """A write shaped like another process's: below the notify hook."""
    setting = settings_io.resolve_key(key)
    assert setting is not None, key
    settings_io._store(ConfigManager(config_dir), setting.path, value)


def _write_here(config_dir, key: str, value: Any) -> None:
    """A write shaped like an OPERATOR's in this process: the settings facade.

    ``write_setting`` lands the value and then notifies this process's watcher
    with ``source="local"``, which is the one delivery the gate may loosen on.
    """
    setting = settings_io.resolve_key(key)
    assert setting is not None, key
    settings_io.write_setting(ConfigManager(config_dir), setting, value)


def _handle(
    tmp_path,
    *,
    auto_approve: bool,
    pinned: bool = False,
    session_dir: Path | None = None,
) -> tuple[ServingSessionHandle, FakeSession, ConfigWatcher, list[Any]]:
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    ConfigManager(config_dir).set_config_value(
        "tool_approval_mode", "auto" if auto_approve else "ask"
    )
    session = FakeSession()
    if session_dir is not None:
        # The handle reads the sidecar through ``_session_directory()``, which
        # is the two-hop ``transcript.directory`` read a real session answers.
        session.transcript = SimpleNamespace(directory=session_dir)
    emitted: list[Any] = []

    async def _emit(event: object) -> None:
        emitted.append(event)

    session._emit = _emit
    handle = ServingSessionHandle(
        session,
        asyncio.get_running_loop(),
        cwd=str(tmp_path),
        auto_approve=auto_approve,
        approval_pinned=pinned,
    )
    watcher = process_watcher(config_dir)
    handle.follow_config(watcher)
    return handle, session, watcher, emitted


@pytest.mark.asyncio
async def test_a_disk_write_cannot_loosen_the_gate_at_the_next_decision(tmp_path) -> None:
    """THE #1282 RULE, runtime half (the issue's reproduction, pinned).

    ``tool_approval_mode: auto`` arriving as a write this process did not make
    is unattributable from here, so the gate does not move: the party being
    gated — a model-run shell command can write that file — must not be the
    authority that lowers its own gate, and this process cannot tell such a
    write apart from an editor's or a second pane's. The opposite of this
    assertion is what the file pinned before #1282, deliberately, and the
    positive control for the attributed path is right below it.
    """
    handle, _session, watcher, emitted = _handle(tmp_path, auto_approve=False)
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    change = watcher.poll_now()
    assert change is not None and "tool_approval_mode" in change.changed_keys
    assert change.source == "disk", "the fixture did not deliver a disk write"

    assert handle._auto_approve is False
    # The gate really is still ARMED, not merely flagged: the next decision
    # parks a card the human has to settle.
    parked = asyncio.ensure_future(handle._approval_gate("bash", "rm -rf build/"))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None, "an unattributed loosening opened the gate"
    await handle.approval_answer(pending.request_id, False, False)
    assert await parked is False

    # The refusal names itself and the command that DOES loosen this session,
    # and no `tool approvals: auto` receipt was emitted for the write.
    await asyncio.sleep(0)
    texts = [getattr(e, "text", "") for e in emitted]
    assert any("keeping tool approvals: ask" in t and "/approvals auto" in t for t in texts), texts
    assert not any("tool approvals: auto" in t for t in texts), texts
    # The sentence is the SHARED constant, so the runtime and the embedded pane
    # cannot describe one refusal two ways (UX round 1, U5's class).
    assert LOOSENING_REFUSED_NOTICE in texts, texts
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_model_authored_shell_write_cannot_loosen_the_gate(tmp_path) -> None:
    """The issue's repro with the write made by a REAL second process.

    A model-run shell command rewriting ``config.yml`` is the path the issue
    names, and it is a different process from this runtime. Everything below the
    write is production: the watcher stats the file, parses it, and fans out as
    ``"disk"``. The in-process variant above proves the rule; this one proves
    the rule holds for the shape the rule exists for.
    """
    handle, _session, watcher, emitted = _handle(tmp_path, auto_approve=False)
    config_file = watcher.config_dir / "config.yml"
    subprocess.run(
        [sys.executable, "-c", _RAW_APPROVAL_WRITE, str(config_file)],
        check=True,
        capture_output=True,
    )
    change = watcher.poll_now()
    assert change is not None and "tool_approval_mode" in change.changed_keys
    assert str(watcher.values.get("tool_approval_mode")) == "auto"

    assert handle._auto_approve is False, "a second process's write loosened this gate"
    parked = asyncio.ensure_future(handle._approval_gate("write", "a file"))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None, "the gate opened on an unattributed write"
    await handle.approval_answer(pending.request_id, True, False)
    assert await parked is True  # the human's own yes still works, of course
    await asyncio.sleep(0)
    assert not any("tool approvals: auto" in getattr(e, "text", "") for e in emitted)
    await handle.dispose()


@pytest.mark.asyncio
async def test_an_attributed_write_in_this_process_still_loosens_the_gate(tmp_path) -> None:
    """The POSITIVE CONTROL for the rule above.

    Without it, "the unattributed write did not loosen the gate" is satisfiable
    by freezing the key and ignoring the operator. This is the same transition
    delivered the one way that IS attributable — the settings facade in this
    process, which is what ``source="local"`` marks — and it must still move the
    gate and emit the receipt.
    """
    handle, _session, watcher, emitted = _handle(tmp_path, auto_approve=False)
    _write_here(watcher.config_dir, "tool_approval_mode", "auto")

    assert handle._auto_approve is True
    # The NEXT decision answers inline — no card parked.
    assert await handle._approval_gate("bash", "rm -rf build/") is True
    assert handle._fold.projection.pending is None
    await asyncio.sleep(0)
    texts = [getattr(e, "text", "") for e in emitted]
    assert any("tool approvals: auto" in t and "config.yml changed" in t for t in texts), texts
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_disk_write_tightens_the_gate_and_the_next_decision_parks(tmp_path) -> None:
    handle, _session, watcher, emitted = _handle(tmp_path, auto_approve=True)
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "ask")
    watcher.poll_now()

    assert handle._auto_approve is False
    pending = asyncio.ensure_future(handle._approval_gate("write", "a file"))
    await asyncio.sleep(0)
    parked_card = handle._fold.projection.pending
    assert parked_card is not None, "the tightened gate did not park a card"
    request_id = parked_card.request_id
    await handle.approval_answer(request_id, False, False)
    assert await pending is False
    await asyncio.sleep(0)
    texts = [getattr(e, "text", "") for e in emitted]
    assert any("tool approvals: ask" in t and "prompt again" in t for t in texts), texts
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_loosening_does_not_revoke_an_explicit_per_session_ask(tmp_path) -> None:
    """THE ASYMMETRIC RULE, loosening half (review R1, UX U1).

    A human who typed ``/approvals ask`` in this session keeps that gate when
    the FILE later says ``auto``, and reads a keep notice naming the way to
    adopt the file. The operator asked for settings to REACH running sessions;
    they did not ask for a file write to revoke a hardening a human typed into
    a specific pane. This mirrors the model half of the same change
    (``Session._explicit_model_choice`` and its ``keeping …`` notice) on the
    more dangerous of the two keys.
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    handle, session, watcher, emitted = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )

    from local_operator.session.frontend_state import SlashResult

    handle._approvals_slash(session, "ask", SlashResult)
    assert handle._auto_approve is False
    # The choice was written BESIDE the conversation, so a resume can find it.
    assert json.loads((session_dir / OVERRIDE_FILENAME).read_text())["override"] == "ask"
    emitted.clear()

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    await asyncio.sleep(0)

    assert handle._auto_approve is False, "a file write revoked a hardening the human typed"
    # The gate really is still armed, not merely flagged: a decision parks.
    parked = asyncio.ensure_future(handle._approval_gate("bash", "rm -rf build/"))
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None
    await handle.approval_answer(handle._fold.projection.pending.request_id, False, False)
    assert await parked is False

    texts = [getattr(e, "text", "") for e in emitted]
    assert any(
        "keeping tool approvals: ask" in t and "/approvals auto adopts it" in t for t in texts
    ), texts
    # And the named way out works in one step.
    handle._approvals_slash(session, "auto", SlashResult)
    assert handle._auto_approve is True
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_global_tightening_does_not_rearm_an_explicit_auto(tmp_path) -> None:
    """A GLOBAL CHANGE DOES NOT MOVE A SESSION THAT CHOSE — either direction.

    This replaces the previous rule, which let a tightening through "because
    safety propagates without exception". It does not propagate through a
    conversation's own decision: a session its human deliberately set to
    ``auto`` (a long unattended run, a CI-shaped task) was re-armed by a
    default about OTHER sessions, mid-turn and silently. Since 2026-10-23 the
    recorded choice outranks the file both ways, the notice says so, and it
    names the command that follows the file.

    The asymmetry that remains is the FILE's, not the choice's: a session that
    never chose still follows a tightening unconditionally (the test below).
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    handle, session, watcher, emitted = _handle(
        tmp_path, auto_approve=True, session_dir=session_dir
    )

    from local_operator.session.frontend_state import SlashResult

    handle._approvals_slash(session, "auto", SlashResult)
    assert handle._auto_approve is True
    assert handle._explicit_approvals_mode == "auto"
    # The human's choice is durable, not just in memory.
    assert json.loads((session_dir / OVERRIDE_FILENAME).read_text())["override"] == "auto"
    emitted.clear()

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "ask")
    watcher.poll_now()
    await asyncio.sleep(0)

    assert handle._auto_approve is True, "a global default re-armed a conversation that chose auto"
    assert handle._explicit_approvals_mode == "auto"
    texts = [getattr(e, "text", "") for e in emitted]
    assert REARM_KEPT_BY_AUTO_NOTICE in texts, texts
    # The gate really is still loose: the next decision answers inline, and no
    # card is parked for a human who is not there.
    assert await handle._approval_gate("bash", "rm -rf build/") is True
    assert handle._fold.projection.pending is None
    # ...and the file's direction did not rewrite the conversation's record.
    assert json.loads((session_dir / OVERRIDE_FILENAME).read_text())["override"] == "auto"
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_session_that_never_chose_follows_a_tightening_and_refuses_a_loosening(
    tmp_path,
) -> None:
    """The operator's own case, in the direction this change still delivers it.

    A session nobody typed ``/approvals`` into is exactly the "goes into effect
    for all my agents" one, and a config write still reaches it: a TIGHTENING
    unconditionally, because safety propagates without exception. The loosening
    half is the #1282 boundary — unattributed, it is refused and the notice
    names the command that does loosen this session, so the operator's route is
    one keystroke away rather than gone. The attributed half is the positive
    control above.
    """
    handle, _session, watcher, emitted = _handle(tmp_path, auto_approve=True)

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "ask")
    watcher.poll_now()
    assert handle._auto_approve is False, "a tightening was refused; safety must always propagate"

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    await asyncio.sleep(0)
    assert handle._auto_approve is False, "an unattributed loosening moved the gate"
    texts = [getattr(e, "text", "") for e in emitted]
    assert any("keeping tool approvals: ask" in t and "/approvals auto" in t for t in texts), texts
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_bare_approvals_reports_a_divergence_against_the_file(
    tmp_path, monkeypatch
) -> None:
    """The reporting half (UX U1 step 3 / U2), plus the order of the two keep rules.

    A session can hold a mode the file disagrees with for either reason the
    keeping rule now has: a human typed ``ask`` here and the file says ``auto``,
    or an unattributed loosening arrived and was refused. Either way the one
    surface whose job is "what is in effect and why" compares against the FILE
    rather than a cached default — otherwise it reports a matched pair for
    exactly the state it exists to disclose.

    The explicit-choice keep branch is asserted to be the one that SPEAKS here,
    because it is checked first: with two refusal reasons now reachable, a
    session that hardened itself must keep reading the sentence about its own
    typed choice rather than the attribution one.
    """
    handle, session, watcher, emitted = _handle(tmp_path, auto_approve=False)

    from local_operator.session.frontend_state import SlashResult

    # The report resolves the watcher through `paths.config_dir()`, so the env
    # var has to name this scratch dir. `_handle` already listens on that
    # directory's PROCESS watcher, which is the one `notify_local` and this
    # report both resolve — there is no second watcher to keep in step.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(watcher.config_dir))

    handle._approvals_slash(session, "ask", SlashResult)
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()  # the explicit-`ask` keep rule holds this gate at `ask`
    await asyncio.sleep(0)

    texts = [getattr(e, "text", "") for e in emitted]
    assert any("set with /approvals in this session" in t for t in texts), texts

    # ``may_loosen=True`` explicitly: the DEFAULT is now "not said", which the
    # sentence builders read conservatively (agent review round 4, R4-3), and
    # every production caller passes the connection's own answer.
    reported = handle._approvals_slash(session, "", SlashResult, may_loosen=True)
    text = getattr(reported, "text", "")
    assert "tool approvals: ask (this session)" in text, text
    assert "config.yml says auto" in text, text
    # ...and the remedy is named, in the direction that MATCHES the file (UX
    # round 1, U3): the runtime's report used to stop at the divergence, leaving
    # the one surface whose job is "what is in effect and why" to describe a
    # problem without its answer.
    #
    # The default answers for a connection that COULD loosen: only a caller that
    # does not know its connection omits the flag, and every production caller
    # (``RuntimeServer``'s payload dispatch) passes it.
    assert text.endswith("/approvals auto adopts it in this session"), text
    assert "typed in the terminal" not in text, text

    # ...AND WHEN THE CONNECTION MAY NOT LOOSEN, IT NAMES THE REMEDY THAT WORKS
    # FROM WHERE THE READER IS (design round 1 D3, UX round 1 U1/U2; design round 2
    # D10, UX round 2 U9; revision 2 §5).
    #
    # BOTH HALVES OF THE OLD SENTENCE ARE NOW PINNED ABSENT, and this assertion is
    # the one that held the deleted remedy in place. It required "typed in the
    # terminal or app window that started this session" AND "let this session's
    # runtime retire and reopen the session here" to be PRESENT, while
    # `test_approval_authority_seam.py` asserted the same clause absent from the
    # refusal — the suite holding both ends of a contradiction one reader met as
    # two rules (agent review round 6 R6-3 = design round 6 D1 = UX round 6 U4).
    # The report now names the levers that exist under this model, and the clause
    # it used to append is the missing-anchor state instead (UX round 6, U1/U2),
    # which on a host WITH an anchor is the empty string — asserted below.
    follower = handle._approvals_slash(session, "", SlashResult, may_loosen=False)
    follower_text = getattr(follower, "text", "")
    assert (
        "/approvals auto adopts it with the operator's own consent — authorise it from this "
        "machine (Touch ID) or from a paired phone" in follower_text
    ), follower_text
    assert "typed in the terminal" not in follower_text, follower_text
    assert "started this session" not in follower_text, follower_text
    assert "retire and reopen" not in follower_text, follower_text
    assert "the window that opens a runtime owns its gate" not in follower_text, follower_text

    # `/approvals default <mode>` USED TO BE REFUSED here and is now HONOURED:
    # the persistence path is exercised in
    # ``test_the_runtime_default_saves_the_file_and_reaches_an_inheriting_handle``
    # below, which is the one place the receipt, the file and the sibling
    # handle are asserted together.
    await handle.dispose()


@pytest.mark.asyncio
async def test_the_routed_receipt_discloses_the_dark_marker(tmp_path, monkeypatch) -> None:
    """UX round 2, U6 and U10 — what the runtime tells the operator, in two places.

    U6: after #1282 a routed ``/approvals auto`` is the ONLY route that loosens a
    running session, and a routed command cannot move the pane's own
    ``_approve_all`` — the marker's only input — so the persistent ``!`` stays
    dark for exactly the state that route creates. Fixing the marker is a change
    of its own (deferred on the PR with the measurements); telling the operator is
    one clause, and the surface where the state changes is this receipt.

    U10: the app-local matched-pair report has always ended "new sessions open the
    same way" and the runtime's stopped one clause short — the same divergence U5
    closed for the receipts. Both sentences are asserted verbatim, and the clause
    that names the FILE is asserted absent when no watcher has read it, because
    the runtime may not vouch for a file it cannot see.
    """
    handle, session, watcher, _emitted = _handle(tmp_path, auto_approve=False)

    from local_operator.session.frontend_state import SlashResult

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(watcher.config_dir))
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    await asyncio.sleep(0)

    receipt = getattr(handle._approvals_slash(session, "auto", SlashResult), "text", "")
    assert receipt == (
        "tool approvals: auto — every tool runs without asking (this session); "
        "the band's ! will not follow this — /approvals re-reports the gate"
    ), receipt

    report = getattr(handle._approvals_slash(session, "", SlashResult), "text", "")
    assert report == (
        "tool approvals: auto — every tool runs without asking; new sessions open the "
        "same way; the band's ! will not follow this — /approvals re-reports the gate"
    ), report

    # A routed TIGHTENING leaves the marker correctly dark, so the clause would be
    # noise there — pinned so it cannot spread.
    tighten = getattr(handle._approvals_slash(session, "ask", SlashResult), "text", "")
    assert "the band's !" not in tighten, tighten

    # No watcher snapshot: the runtime has not read the file, so it says nothing
    # about it (the clause is conditional on `on_disk == live`).
    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)
    bare = getattr(handle._approvals_slash(session, "", SlashResult), "text", "")
    assert "new sessions" not in bare, bare
    assert bare == "tool approvals: ask — write and command tools prompt before running", bare
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_parked_prompt_is_left_for_the_human(tmp_path) -> None:
    """The gate reads the flag when a DECISION is made. A card already on
    screen when the file loosens is neither auto-approved nor dismissed; a
    card on screen when it tightens is not auto-denied. The human answers.

    The loosening is delivered ATTRIBUTED (the settings facade in this process;
    an unattributed one is refused outright since #1282), because the parked-card
    rule has to be pinned on a loosening that actually moves the gate — on a
    refused write it would pass by never being exercised.
    """
    handle, _session, watcher, _emitted = _handle(tmp_path, auto_approve=False)
    parked = asyncio.ensure_future(handle._approval_gate("bash", "touch /tmp/x"))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None

    _write_here(watcher.config_dir, "tool_approval_mode", "auto")
    await asyncio.sleep(0)
    assert handle._auto_approve is True
    assert not parked.done(), "a loosening auto-answered a card the human was looking at"
    front = handle._fold.projection.pending
    assert front is not None
    assert front.request_id == pending.request_id

    # The human decides, and only then does the future settle.
    await handle.approval_answer(pending.request_id, False, False)
    assert await parked is False
    # A brand-new decision after the human's answer follows the file.
    assert await handle._approval_gate("bash", "next") is True
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_yolo_pin_ignores_the_key(tmp_path) -> None:
    """``lop exec --control --yolo``: an explicit flag on this run outranks a
    default in a file. Nothing moves, nothing is announced."""
    handle, _session, watcher, emitted = _handle(tmp_path, auto_approve=True, pinned=True)
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "ask")
    watcher.poll_now()
    assert handle._auto_approve is True
    assert await handle._approval_gate("bash", "anything") is True
    await asyncio.sleep(0)
    assert emitted == []
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_no_op_write_and_an_unknown_mode_leave_the_gate_alone(tmp_path) -> None:
    handle, _session, watcher, emitted = _handle(tmp_path, auto_approve=False)
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "sometimes")
    watcher.poll_now()
    assert handle._auto_approve is False
    await asyncio.sleep(0)
    assert emitted == []
    await handle.dispose()


@pytest.mark.asyncio
async def test_dispose_unsubscribes_and_follow_config_is_idempotent(tmp_path) -> None:
    handle, _session, watcher, _emitted = _handle(tmp_path, auto_approve=False)
    handle.follow_config(watcher)
    assert len(watcher._listeners) == 1
    await handle.dispose()
    assert watcher._listeners == []
    # A tick after dispose reaches nothing.
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    assert handle._auto_approve is False


@pytest.mark.asyncio
async def test_a_repeated_answer_is_a_settlement_not_a_refusal(tmp_path) -> None:
    """The desktop answer path retries ONCE under the same id when its ack is lost.

    ``AttachedSession.answer_gate`` re-issues rather than report a transport
    failure over a card the operator already cleared, and that is only safe
    because a repeat here is a SETTLEMENT: without this the owner answered "that
    prompt is no longer waiting" to the very press it had already honoured, and the
    retrying caller read a refusal where the truth was a receipt.
    """
    handle, _session, _watcher, _emitted = _handle(tmp_path, auto_approve=False)
    parked = asyncio.ensure_future(handle._approval_gate("bash", "rm -rf build/"))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None, "the gate did not park, so there is nothing to repeat"

    first = await handle.approval_answer(pending.request_id, True, False)
    repeat = await handle.approval_answer(pending.request_id, True, False)

    assert repeat == first, (first, repeat)
    assert await parked is True
    await handle.dispose()


@pytest.mark.asyncio
async def test_an_id_that_never_settled_is_still_refused(tmp_path) -> None:
    """The idempotence is keyed on THIS id having settled, not on the gate being open.

    An unknown request id has nothing recorded against it, so it keeps the
    developer-worded refusal it always had — the retry cannot turn a wrong id into
    a success.
    """
    handle, _session, _watcher, _emitted = _handle(tmp_path, auto_approve=False)

    with pytest.raises(ValueError, match="no longer waiting"):
        await handle.approval_answer("never-issued", True, False)

    await handle.dispose()


@pytest.mark.asyncio
async def test_a_repeat_with_a_different_decision_is_still_refused(tmp_path) -> None:
    """The retry is idempotent under the SAME value, and only under it.

    A second front end that answers the same gate differently is not a repeat of
    anything — it is a competing decision — so it keeps the refusal it always had.
    Reporting success for a choice the owner did not make would be a worse lie
    than the lost-acknowledgement refusal this change fixes.
    """
    handle, _session, _watcher, _emitted = _handle(tmp_path, auto_approve=False)
    parked = asyncio.ensure_future(handle._approval_gate("bash", "rm -rf build/"))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None

    assert await handle.approval_answer(pending.request_id, True, False) == "approved"
    with pytest.raises(ValueError, match="no longer waiting"):
        await handle.approval_answer(pending.request_id, False, False)

    assert await parked is True
    await handle.dispose()


# ---------------------------------------------------------------------------
# The conversation's own recorded choice, and the machine-default sink
# ---------------------------------------------------------------------------
#
# Two live bugs this section pins, both from the operator's report: a session
# started in `ask` kept prompting after the global default was set to `auto`
# (the runtime refused to persist a default from a control connection, and its
# in-memory choice was not written down), and an explicit `/approvals` pick
# vanished on resume. The rule now: the choice is PERSISTED with the
# conversation and outranks the file in BOTH directions; every session that
# never chose still follows the file, with a loosening still needing authority.


def test_the_file_name_used_by_the_tests_is_the_module_constant() -> None:
    """One spelling: the sidecar name is imported, never copied into a test."""
    assert OVERRIDE_FILENAME == "approvals.json"


@pytest.mark.asyncio
async def test_a_resumed_conversation_starts_in_its_recorded_mode(tmp_path) -> None:
    """RESUME reads the choice back from the session directory.

    The machine default says ``ask`` and the handle is CONSTRUCTED with that
    value — exactly what ``exec_control``/``spawn_owned_session`` do — so the
    only way the gate can start loose is by adopting the record. Before this
    the handle went back to asking and the user's `/approvals auto` was gone.
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    original, original_session, _watcher, _emitted = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )
    from local_operator.session.frontend_state import SlashResult

    original._approvals_slash(original_session, "auto", SlashResult)
    await original.dispose()

    resumed, _session, _watcher, _emitted = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )
    assert resumed._explicit_approvals_mode == "auto"
    assert resumed._auto_approve is True
    # The FIRST decision is already loose: no card is parked and then answered.
    assert await resumed._approval_gate("bash", "make test") is True
    assert resumed._fold.projection.pending is None
    await resumed.dispose()


@pytest.mark.asyncio
async def test_an_unreadable_record_warns_and_starts_in_ask(tmp_path) -> None:
    """FAIL CLOSED: a corrupt record is `ask`, and the user is told.

    The unsafe reading is "no choice" → inherit the config's ``auto``, which
    would silently un-gate a conversation whose human had hardened it.
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / OVERRIDE_FILENAME).write_text('{"schema_version": 1, "overr', encoding="utf-8")
    handle, _session, _watcher, emitted = _handle(
        tmp_path, auto_approve=True, session_dir=session_dir
    )
    await asyncio.sleep(0)
    assert handle._explicit_approvals_mode is None
    assert handle._auto_approve is False, "a corrupt record inherited the config's auto"
    texts = [getattr(e, "text", "") for e in emitted]
    assert any("could not be read" in t for t in texts), texts
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_persist_failure_is_disclosed_and_the_session_still_switches(tmp_path) -> None:
    """The live switch happens; the receipt says the resume will not keep it.

    Not raising is deliberate: refusing the switch after the human typed it
    would leave the gate disagreeing with the answer just given. The clause
    names the error CLASS, never a stack — this text lands in a transcript.
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    handle, session, _watcher, _emitted = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )

    def _refuse(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch_target = "local_operator.session.runtime.serving._write_approval_override"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(monkeypatch_target, _refuse)
        from local_operator.session.frontend_state import SlashResult

        receipt = handle._approvals_slash(session, "auto", SlashResult)

    assert handle._auto_approve is True, "the live switch must still apply"
    text = getattr(receipt, "text", "")
    assert "applied for this session but NOT saved for resume (OSError)" in text, text
    assert not (session_dir / OVERRIDE_FILENAME).exists()
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_global_tightening_leaves_a_parked_prompt_for_the_human(tmp_path) -> None:
    """A card on screen is never auto-denied by a tightening at the next tick.

    The gate reads the flag when a DECISION is made, so an in-flight card keeps
    waiting; only the decision after it sees the new mode. The card has to be
    opened while the gate is ARMED (an auto gate never parks one), and the
    loosening follows so the tightening is a real transition on this handle.
    """
    handle, _session, watcher, _emitted = _handle(tmp_path, auto_approve=False)
    parked = asyncio.ensure_future(handle._approval_gate("bash", "touch /tmp/x"))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None

    # Attributed, so it actually moves this gate (an unattributed write cannot).
    _write_here(watcher.config_dir, "tool_approval_mode", "auto")
    await asyncio.sleep(0)
    assert handle._auto_approve is True
    assert not parked.done()
    front = handle._fold.projection.pending
    assert front is not None
    assert front.request_id == pending.request_id

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "ask")
    watcher.poll_now()
    await asyncio.sleep(0)
    assert handle._auto_approve is False
    assert not parked.done(), "a tightening auto-denied a card the human was looking at"
    assert handle._fold.projection.pending is not None

    # The human's own yes still settles the parked card, and the NEXT decision
    # is made under the tightened mode.
    await handle.approval_answer(pending.request_id, True, False)
    assert await parked is True
    next_card = asyncio.ensure_future(handle._approval_gate("write", "later"))
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None, "the next decision ignored the file"
    await handle.approval_answer(handle._fold.projection.pending.request_id, False, False)
    assert await next_card is False
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_recorded_choice_leaves_a_parked_prompt_alone(tmp_path) -> None:
    """A parked card is untouched by a global change even when the gate is kept.

    Both rules meet here: the conversation's choice means the file does not
    move its gate, and the card was already open, so nothing about this tick
    may answer it. The card is opened under the recorded `ask` (an auto gate
    never parks one); the global loosening is then kept, and the human answers.
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    handle, session, watcher, emitted = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )
    from local_operator.session.frontend_state import SlashResult

    handle._approvals_slash(session, "ask", SlashResult)
    parked = asyncio.ensure_future(handle._approval_gate("bash", "touch /tmp/x"))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None
    emitted.clear()

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    await asyncio.sleep(0)
    assert handle._auto_approve is False
    assert not parked.done()
    front = handle._fold.projection.pending
    assert front is not None
    assert front.request_id == pending.request_id
    assert any(
        "keeping tool approvals: ask" in t for t in [getattr(e, "text", "") for e in emitted]
    )
    await handle.approval_answer(pending.request_id, True, False)
    assert await parked is True
    # The next decision still parks, because the conversation's choice holds.
    next_card = asyncio.ensure_future(handle._approval_gate("bash", "next"))
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None
    await handle.approval_answer(handle._fold.projection.pending.request_id, True, False)
    assert await next_card is True
    await handle.dispose()


@pytest.mark.asyncio
async def test_the_runtime_default_saves_the_file_and_reaches_an_inheriting_handle(
    tmp_path, monkeypatch
) -> None:
    """`/approvals default auto` from a control connection now PERSISTS (#sink).

    It writes ``tool_approval_mode`` through this process's settings facade, so
    the file changes AND every inheriting handle in this process follows within
    the same delivery — the sibling below is the operator's second session,
    which used to keep prompting. The receipt names the file, the key and what
    applied, because "saved" alone does not say where.
    """
    maker, maker_session, watcher, _emitted = _handle(tmp_path, auto_approve=False)
    sibling, _sibling_session, _same_watcher, _sibling_emitted = _handle(
        tmp_path, auto_approve=False
    )
    assert sibling._auto_approve is False

    # The facade resolves the config dir through ``paths.config_dir()``, so the
    # env has to name this scratch dir — the same wiring the divergence report
    # test uses. Without it the write would land in the isolated HOME's config
    # and the watcher below would (correctly) see nothing.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(watcher.config_dir))

    from local_operator.session.frontend_state import SlashResult

    receipt = maker._approvals_slash(maker_session, "default auto", SlashResult)

    assert ConfigManager(watcher.config_dir).get_config_value("tool_approval_mode") == "auto"
    text = getattr(receipt, "text", "")
    assert str(ConfigManager(watcher.config_dir).config_file) in text, text
    assert "tool_approval_mode" in text, text
    assert "this conversation now follows it" in text, text
    assert "other conversations with their own choice keep it" in text, text
    # The sibling followed through the watcher — no direct flag write needed.
    assert sibling._auto_approve is True
    assert await sibling._approval_gate("bash", "make test") is True
    assert sibling._fold.projection.pending is None
    # ...and the issuing handle follows the command it just issued.
    assert maker._auto_approve is True
    assert maker._explicit_approvals_mode is None

    # The tightening direction of the same op is honoured too.
    back = maker._approvals_slash(maker_session, "default ask", SlashResult)
    assert ConfigManager(watcher.config_dir).get_config_value("tool_approval_mode") == "ask"
    assert "this conversation now follows it" in getattr(back, "text", ""), getattr(
        back, "text", ""
    )
    assert sibling._auto_approve is False
    assert maker._auto_approve is False
    await maker.dispose()
    await sibling.dispose()


@pytest.mark.asyncio
async def test_the_runtime_default_clears_only_the_issuing_conversations_choice(
    tmp_path, monkeypatch
) -> None:
    """The issuing conversation adopts the global; NOBODY else's record moves.

    This is the design round's refinement on top of the sink: the command has
    always meant "this session and every new one", so the issuing conversation
    must stop keeping a choice that now contradicts the default it just set.
    Every OTHER conversation keeps its own record — a global preference is a
    default, not a forced rewrite, and revoking a hardening elsewhere is the
    consent expansion this whole change refuses.
    """
    maker_dir = tmp_path / "maker-session"
    maker_dir.mkdir()
    maker, maker_session, watcher, _emitted = _handle(
        tmp_path, auto_approve=True, session_dir=maker_dir
    )
    inheriting, _inheriting_session, _same_watcher, _inherited = _handle(
        tmp_path, auto_approve=True
    )
    keeper_dir = tmp_path / "keeper-session"
    keeper_dir.mkdir()
    keeper, keeper_session, _keeper_watcher, _keeper_emitted = _handle(
        tmp_path, auto_approve=True, session_dir=keeper_dir
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(watcher.config_dir))
    from local_operator.session.frontend_state import SlashResult

    maker._approvals_slash(maker_session, "ask", SlashResult)  # the issuer chose ask
    keeper._approvals_slash(keeper_session, "auto", SlashResult)  # someone else chose auto
    assert maker._auto_approve is False
    assert keeper._auto_approve is True

    receipt = maker._approvals_slash(maker_session, "default ask", SlashResult)

    assert ConfigManager(watcher.config_dir).get_config_value("tool_approval_mode") == "ask"
    # The issuer follows the default it set, and its stale record is cleared.
    assert maker._auto_approve is False
    assert maker._explicit_approvals_mode is None
    # The persisted record was cleared too, not merely forgotten in memory.
    assert json.loads((maker_dir / OVERRIDE_FILENAME).read_text())["override"] is None
    assert "this conversation now follows it" in getattr(receipt, "text", ""), getattr(
        receipt, "text", ""
    )
    # An inheriting session follows through the watcher...
    assert inheriting._auto_approve is False
    # ...and a session with its own recorded choice is untouched.
    assert keeper._auto_approve is True, "a global default revoked another conversation's choice"
    assert keeper._explicit_approvals_mode == "auto"
    assert json.loads((keeper_dir / OVERRIDE_FILENAME).read_text())["override"] == "auto"
    await maker.dispose()
    await inheriting.dispose()
    await keeper.dispose()


@pytest.mark.asyncio
async def test_the_runtime_default_write_failure_says_nothing_was_saved(
    tmp_path, monkeypatch
) -> None:
    """A failed write is reported as a failure — never as a propagation.

    The old receipt claimed ``auto`` "now"; the new one claims exactly what the
    facade did. If the write raises (a read-only config directory, a full
    disk), the receipt names the error class and says no other session changed.
    """
    handle, session, watcher, _emitted = _handle(tmp_path, auto_approve=False)

    def _refuse(*_args, **_kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr("local_operator.settings_io.write_setting", _refuse)
    from local_operator.session.frontend_state import SlashResult

    receipt = handle._approvals_slash(session, "default auto", SlashResult)
    text = getattr(receipt, "text", "")
    assert "NOT saved (PermissionError)" in text, text
    assert "no other session was told" in text, text
    assert "every session" not in text, text
    assert ConfigManager(watcher.config_dir).get_config_value("tool_approval_mode") == "ask"
    assert handle._auto_approve is False
    await handle.dispose()


@pytest.mark.asyncio
async def test_the_published_gate_carries_the_policy_and_its_provenance(tmp_path) -> None:
    """The phone/desktop contract must say WHO set the rule, not just the mode.

    A viewer cannot derive the effective policy: the conversation's own record
    outranks config.yml, so "Ask — session override; global Auto" is only
    possible if the publisher stamps both halves. This drives the real
    publication seam with a stub store and checks the payload a viewer folds.
    """
    handle, session, _watcher, _emitted = _handle(tmp_path, auto_approve=False)

    class _Store:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def mutate(self, **changes: Any) -> None:
            self.calls.append(changes)

    store = _Store()
    session._frontend_state_store = store  # type: ignore[attr-defined]
    parked = asyncio.ensure_future(handle._approval_gate("bash", "probe"))
    await asyncio.sleep(0)
    handle._publish_pending_gate()

    payload = store.calls[-1]["pending_gate"]
    assert payload["approval_mode"] == "ask", payload
    assert payload["approval_provenance"] == "global", payload
    assert payload["approval_global_mode"] == "ask", payload

    # GLOBAL ASK + SESSION ASK: the two words agree, and the payload must still
    # carry the global half instead of letting a viewer derive the complement
    # (the U4 defect said "global Auto" here).
    handle._explicit_approvals_mode = "ask"
    handle._publish_pending_gate()
    payload = store.calls[-1]["pending_gate"]
    assert payload["approval_mode"] == "ask", payload
    assert payload["approval_provenance"] == "session", payload
    assert payload["approval_global_mode"] == "ask", payload

    # Session AUTO over a global ASK: the interesting cross direction.
    handle._explicit_approvals_mode = "auto"
    handle._publish_pending_gate()
    payload = store.calls[-1]["pending_gate"]
    assert payload["approval_mode"] == "auto", payload
    assert payload["approval_provenance"] == "session", payload
    assert payload["approval_global_mode"] == "ask", payload

    pending = handle._fold.projection.pending
    assert pending is not None
    await handle.approval_answer(pending.request_id, False, False)
    assert await parked is False
    await handle.dispose()

    # GLOBAL AUTO + SESSION AUTO, the mirror. A gate can be PARKED under auto
    # only because it was raised before the mode moved (that is the saved-state
    # card), so drive the real writes in that order rather than hand-setting the
    # fields: default auto first, then this conversation's own auto.
    from local_operator.session.frontend_state import SlashResult

    (tmp_path / "mirror").mkdir()
    auto_handle, auto_session, _auto_watcher, _auto_emitted = _handle(
        tmp_path / "mirror", auto_approve=False
    )
    auto_store = _Store()
    auto_session._frontend_state_store = auto_store  # type: ignore[attr-defined]
    auto_parked = asyncio.ensure_future(auto_handle._approval_gate("bash", "probe"))
    await asyncio.sleep(0)
    auto_handle._approvals_default_slash("default auto", SlashResult)
    auto_handle._approvals_slash(auto_handle._session, "auto", SlashResult)
    auto_handle._publish_pending_gate()
    auto_payload = auto_store.calls[-1]["pending_gate"]
    assert auto_payload["approval_mode"] == "auto", auto_payload
    assert auto_payload["approval_provenance"] == "session", auto_payload
    assert auto_payload["approval_global_mode"] == "auto", auto_payload
    auto_pending = auto_handle._fold.projection.pending
    assert auto_pending is not None
    await auto_handle.approval_answer(auto_pending.request_id, False, False)
    assert await auto_parked is False
    await auto_handle.dispose()


@pytest.mark.asyncio
async def test_the_phone_reads_a_successful_loosening_as_success(tmp_path) -> None:
    """A successful loosening is tinted ``warning`` but is not a refusal.

    ``_slash_receipt`` used to classify on the tint, so the phone showed every
    successful ``auto`` / ``default auto`` as refused and kept the draft. The
    structured ``data.ok`` is the verdict; the tint rule survives only for an
    older runtime that sends none (agent review round 1, B1).
    """
    from local_operator.mobile.daemon import _slash_receipt
    from local_operator.session.frontend_state import SlashResult

    handle, _session, _watcher, _emitted = _handle(tmp_path, auto_approve=False)

    live = handle._approvals_slash(handle._session, "auto", SlashResult)
    detail, _tone, refused = _slash_receipt(live.model_dump(mode="json"), "approvals")
    assert live.data["ok"] is True
    assert refused is False, detail

    default = handle._approvals_default_slash("default auto", SlashResult)
    detail, _tone, refused = _slash_receipt(default.model_dump(mode="json"), "approvals")
    assert default.data["ok"] is True
    assert refused is False, detail

    # Legacy owner with no verdict keeps the old rule: a warning-shaped outcome
    # with no structured success is a refusal.
    assert _slash_receipt({"text": "x", "style": "warning"}, "approvals")[2] is True
    await handle.dispose()
