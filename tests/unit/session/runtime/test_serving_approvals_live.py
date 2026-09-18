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
import subprocess
import sys
from typing import Any

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import ConfigWatcher, _reset_for_tests, process_watcher
from local_operator.harness.approval import LOOSENING_REFUSED_NOTICE
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
    tmp_path, *, auto_approve: bool, pinned: bool = False
) -> tuple[ServingSessionHandle, FakeSession, ConfigWatcher, list[Any]]:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    ConfigManager(config_dir).set_config_value(
        "tool_approval_mode", "auto" if auto_approve else "ask"
    )
    session = FakeSession()
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
    assert handle._fold.projection.pending is not None, "the tightened gate did not park a card"
    request_id = handle._fold.projection.pending.request_id
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
    handle, session, watcher, emitted = _handle(tmp_path, auto_approve=False)

    from local_operator.session.frontend_state import SlashResult

    handle._approvals_slash(session, "ask", SlashResult)
    assert handle._auto_approve is False
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
async def test_a_tightening_follows_the_file_even_over_an_explicit_choice(tmp_path) -> None:
    """THE ASYMMETRIC RULE, tightening half. Safety propagates without
    exception: a session that explicitly chose ``auto`` still follows the file
    to ``ask``, because a user who ends up safer than they asked is never the
    wrong surprise. This is the direction the rule does NOT make conditional."""
    # File AND gate start at `auto`, so the disk write below is a real
    # transition; the session's own `/approvals auto` is what records the
    # explicit choice the tightening then has to override.
    handle, session, watcher, emitted = _handle(tmp_path, auto_approve=True)

    from local_operator.session.frontend_state import SlashResult

    handle._approvals_slash(session, "auto", SlashResult)
    assert handle._auto_approve is True
    assert handle._explicit_approvals_mode == "auto"
    emitted.clear()

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "ask")
    watcher.poll_now()
    await asyncio.sleep(0)
    assert handle._auto_approve is False, "a tightening was refused; safety must always propagate"
    texts = [getattr(e, "text", "") for e in emitted]
    assert any("tool approvals: ask" in t and "config.yml changed" in t for t in texts), texts

    # ...AND THE SESSION CAN STILL FOLLOW THE FILE BACK (review round 2, R6).
    # The runtime half of the same regression: the only mode this human ever
    # typed is `auto`, so there is no hardening for the explicit-`ask` keep
    # branch to protect, and a guard that read merely "this session chose
    # something" pinned it to `ask` permanently. The FILE owns the value once it
    # moves the gate, which is why the recorded mode is cleared — that is also
    # what keeps the keep notice's "set with /approvals in this session" a true
    # statement rather than a claim about a value the file chose.
    assert (
        handle._explicit_approvals_mode is None
    ), "a file write left the session claiming the human had typed the mode"
    # Since #1282 the way back is the ATTRIBUTED write, not any file write: an
    # unattributed loosening is refused, and refused for the ATTRIBUTION reason
    # rather than by blaming a choice this human never made.
    emitted.clear()
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    await asyncio.sleep(0)
    assert handle._auto_approve is False, "an unattributed loosening moved this gate"
    texts = [getattr(e, "text", "") for e in emitted]
    assert any("keeping tool approvals" in t for t in texts), texts
    assert not any(
        "set with /approvals in this session" in t for t in texts
    ), "the refusal blamed a choice this human never made (R6)"
    _write_here(watcher.config_dir, "tool_approval_mode", "auto")
    await asyncio.sleep(0)
    assert handle._auto_approve is False, (
        "the file already said auto when the loosening was refused, so re-writing the "
        "same value is not even a delivery -- there is no value change to announce"
    )
    # ...AND THE SESSION IS NOT PINNED: the human's own command still loosens it,
    # which is the property R6 protected. Under #1282 this is the ONLY route from
    # the divergence the refusal leaves behind, and `/approvals` says as much.
    from local_operator.session.frontend_state import SlashResult

    handle._approvals_slash(session, "auto", SlashResult)
    assert handle._auto_approve is True
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

    reported = handle._approvals_slash(session, "", SlashResult)
    text = getattr(reported, "text", "")
    assert "tool approvals: ask (this session)" in text, text
    assert "config.yml says auto" in text, text
    # ...and the remedy is named, in the direction that MATCHES the file (UX
    # round 1, U3): the runtime's report used to stop at the divergence, leaving
    # the one surface whose job is "what is in effect and why" to describe a
    # problem without its answer.
    assert "/approvals auto adopts it in this session" in text, text
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
    assert handle._fold.projection.pending is not None
    assert handle._fold.projection.pending.request_id == pending.request_id

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
