"""Exercise the actual slash worker/busy guard with synthetic setup outcomes."""

import threading
from unittest.mock import AsyncMock

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeKind
from local_operator.tunnels import setup
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.waiting import settle


@pytest.mark.asyncio
async def test_mobile_status_selected_owner_is_read_only(monkeypatch):
    observe = AsyncMock(
        return_value={
            "step": "apply",
            "verified": False,
            "receipts": [],
            "remedy": "Run lop tunnel setup apply",
        }
    )
    apply = AsyncMock()
    monkeypatch.setattr(setup, "observe", observe)
    monkeypatch.setattr(setup, "apply", apply)
    app = OperatorApp(lambda: _factory(FakeSession()))
    notices: list[str] = []

    def notice(body: str, kind: NoticeKind = "info") -> None:
        notices.append(body)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_system_notice", notice)
        app._cmd_mobile("status --credential-id 17", notice)
        await pilot.pause(0.2)
        observe.assert_awaited_once_with(17)
        apply.assert_not_awaited()
        assert not app._tunnel_action_busy
        assert any("/mobile enable" in line for line in notices)


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/mobile", "/mobile "])
async def test_mobile_composer_does_not_offer_provider_arguments(monkeypatch, typed):
    from local_operator.tui.widgets.editor import Editor

    observe = AsyncMock(
        return_value={
            "step": "login",
            "verified": False,
            "receipts": [],
            "credential_choices": [],
            "remedy": "Run lop login radient",
        }
    )
    monkeypatch.setattr(setup, "observe", observe)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        for char in typed:
            await pilot.press("slash" if char == "/" else "space" if char == " " else char)
        await pilot.pause()
        assert editor.picker.highlighted_name() not in {"radient", "openai", "anthropic"}
        await pilot.press("enter")
        await pilot.pause(0.3)
        observe.assert_awaited_once_with(None)
        assert not app._tunnel_action_busy
        assert session.prompts == []


@pytest.mark.asyncio
async def test_mobile_enable_progress_and_busy_guard(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    seen = []

    async def apply(selected=None, accepted=None, *, progress=None):
        seen.append((selected, accepted))
        assert progress is not None
        progress("Synthetic reservation saved")
        started.set()
        # A hang guard only: the test releases this in its ``finally``. It is
        # long so a descheduled test loop cannot outwait it and clear the busy
        # flag before the guard assertion runs (the old 3 s was a clock bet).
        release.wait(30)
        return {
            "step": "failed",
            "verified": False,
            "receipts": ["Synthetic reservation saved"],
            "remedy": "Retry /mobile status",
        }

    monkeypatch.setattr(setup, "apply", apply)
    app = OperatorApp(lambda: _factory(FakeSession()))
    notices: list[str] = []

    def notice(body: str, kind: NoticeKind = "info") -> None:
        notices.append(body)

    # Event-driven, not timed: the fake apply runs on a worker thread behind
    # run_worker -> asyncio.to_thread -> asyncio.run, and a fixed
    # ``pilot.pause(0.2)`` lost that race on loaded CI shards (PR #3881, test
    # (3.12, 5), twice: ``started`` still unset). ``settle`` waits on the
    # thread's own ``started`` edge under a deadline that only catches a hang.
    # The first wait must NOT join the "mobile-setup" worker: it is parked on
    # ``release`` and asserting its intermediate busy state is the point.
    async with app.run_test(size=(100, 30)) as pilot:
        try:
            await pilot.pause()
            monkeypatch.setattr(app, "_system_notice", notice)
            app._cmd_mobile("enable 0.05 --credential-id 17", notice)
            await settle(pilot, started.is_set, what="fake mobile apply started")
            assert started.is_set() and app._tunnel_action_busy
            app._cmd_mobile("enable 0.05", notice)
            assert seen == [(17, "0.05")]
            assert any("already running" in line for line in notices)
            assert "Synthetic reservation saved" in notices
            release.set()
            await settle(
                pilot,
                lambda: not app._tunnel_action_busy,
                what="mobile setup worker finished",
                workers=("mobile-setup",),
            )
            assert not app._tunnel_action_busy
            assert any("Phone access could not finish setup" in line for line in notices)
            assert sum("Synthetic reservation saved" in line for line in notices) == 1
        finally:
            # Unpark the fake thread before app teardown, even when an
            # assertion above fails, so it can neither leak nor stall exit.
            release.set()


@pytest.mark.asyncio
async def test_mobile_approvals_on_runs_the_shared_gesture_setup(monkeypatch):
    """`/mobile approvals on` (design v2 §6): the SAME setup_authority as the desktop,
    with actor `tui` and terminal sudo allowed (the TUI owns a TTY), and the N5
    success sentence; anything else after `approvals` prints the usage line.

    Notice shape (design round 1 on #3950): the progress line comes only from the
    runner, i.e. after the prechecks (D5); the outcome is one `info` notice and the
    file-only caveat its OWN `warning` notice (D1/D8)."""
    import asyncio

    from local_operator.operator import authority_setup, elevate

    seen: dict[str, object] = {}

    def fake_setup(root, **kwargs):
        seen.update(kwargs)
        seen["runner_result"] = kwargs["runner"]([["sudo", "true"]])
        return {"ok": True, "outcome": "on", "file_only": True}

    monkeypatch.setattr(authority_setup, "setup_authority", fake_setup)
    elevated: list[object] = []
    monkeypatch.setattr(
        elevate,
        "run_elevated",
        lambda command, **k: elevated.append(k) or elevate.ElevationResult(0),
    )
    monkeypatch.setattr("sys.platform", "darwin")  # a GUI prompt: no suspend needed
    app = OperatorApp(lambda: _factory(FakeSession()))
    notices: list[tuple[str, str]] = []

    def notice(body: str, kind: NoticeKind = "info") -> None:
        notices.append((body, kind))

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_system_notice", notice)
        app._cmd_mobile("approvals", notice)
        assert notices[-1] == (
            "To let phones approve requests on this computer, type /mobile approvals on",
            "info",
        )
        app._cmd_mobile("approvals on", notice)
        for _ in range(50):
            await pilot.pause(0.05)
            if seen and not app._operator_setup_busy:
                break
        await asyncio.sleep(0)
    assert seen["actor"] == "tui" and seen["terminal_sudo"] is True
    assert callable(seen["can_prompt"]) and seen["can_prompt"]() is True  # GUI prompt
    assert elevated == [{"purpose": "phone_approvals", "terminal_sudo": True}]
    assert notices[1][0] == "Turning on phone approvals…"
    assert notices[-2] == (
        "Phone approvals are on. Each new phone will still ask for your computer's password once.",
        "info",
    )
    assert notices[-1] == (
        "Note: on this computer itself, approvals won't ask for a fingerprint or password.",
        "warning",
    )


@pytest.mark.asyncio
async def test_mobile_approvals_already_on_prints_no_progress_line(monkeypatch):
    """D5: a run that never reaches a prompt goes straight to its outcome."""
    from local_operator.operator import authority_setup

    monkeypatch.setattr(
        authority_setup,
        "setup_authority",
        lambda root, **k: {"ok": True, "outcome": "already_on", "file_only": False},
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    notices: list[tuple[str, str]] = []

    def notice(body: str, kind: NoticeKind = "info") -> None:
        notices.append((body, kind))

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_system_notice", notice)
        app._cmd_mobile("approvals on", notice)
        for _ in range(50):
            await pilot.pause(0.05)
            if notices and not app._operator_setup_busy:
                break
    assert notices == [("Phone approvals are already on for this computer.", "info")]


@pytest.mark.asyncio
async def test_mobile_approvals_on_linux_without_a_suspendable_terminal_refuses(
    monkeypatch, tmp_path
):
    """On Linux the prompt needs the real terminal (sudo), so it runs under
    App.suspend(); a driver that cannot suspend is refused in setup_authority's
    PRECHECK (QA Q3) — no consent row, no key, nothing staged — with a sentence
    naming this terminal app as the cause (QA Q4), as a warning (D8), and with no
    progress line before it (D5). The REAL setup_authority runs here."""
    import json

    from local_operator.operator import elevate, handlers, trust

    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", tmp_path / "anchors")
    monkeypatch.setattr(handlers, "_is_privileged", lambda: False)
    monkeypatch.setattr(elevate.shutil, "which", lambda name, *a, **k: "/usr/bin/sudo")
    launched: list[object] = []
    monkeypatch.setattr(elevate, "run_elevated", lambda *a, **k: launched.append(a))
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr(trust, "_on_windows", lambda: False)
    app = OperatorApp(lambda: _factory(FakeSession()))
    notices: list[tuple[str, str]] = []

    def notice(body: str, kind: NoticeKind = "info") -> None:
        notices.append((body, kind))

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_system_notice", notice)
        assert app._driver is not None and app._driver.can_suspend is False  # headless
        app._cmd_mobile("approvals on", notice)
        for _ in range(50):
            await pilot.pause(0.05)
            if notices and not app._operator_setup_busy:
                break
        # QA Q1: a second attempt refused by the pane's own busy guard is audited.
        app._operator_setup_busy = True
        app._cmd_mobile("approvals on", notice)
        app._operator_setup_busy = False
    assert launched == []
    assert notices[0] == (
        "This terminal app can't show a password prompt, so approvals can't be turned on "
        "from it. Use Settings → Phone access in the desktop app instead.",
        "warning",
    )
    assert notices[1][1] == "warning" and "already running" in notices[1][0]
    rows = [
        json.loads(line)
        for line in (config / "operator" / "authority-setup-audit.jsonl").read_text().splitlines()
    ]
    assert [(r["step"], r["ok"], r["code"]) for r in rows] == [
        ("proposed", True, "phone_approvals"),
        ("consent", False, "terminal_cannot_prompt"),
        ("proposed", False, "in_progress"),
    ]
    assert not trust.staging_path(config).exists()
    assert handlers._existing_key(config, "auto") is None, "no key was created"


def test_approvals_report_without_approvals_set_up_is_one_keyed_sentence(monkeypatch):
    """D2/D3 (design round 1 on #3950): the report line for a host with no usable
    anchor is ONE keyed message, so a non-English locale never gets a sentence that
    switches language halfway, and the TUI and runtime mirrors cannot drift."""
    from local_operator.i18n import text
    from local_operator.session.runtime.serving import ServingSessionHandle

    monkeypatch.setattr("local_operator.operator.operator_authority_unusable", lambda: True)
    tui = OperatorApp._uninstalled_anchor_report("ask", "auto")
    assert tui == ServingSessionHandle._uninstalled_anchor_report("ask", "auto")
    assert tui is not None and tui.startswith("tool approvals: ask (this session)")
    # D2: "phone approvals" names the switch; the blocker is approvals not set up.
    assert "approvals aren't set up on this computer yet" in tui
    assert "Settings → Phone access" in tui and "/mobile approvals on" in tui

    # The catalogue, not shipped eligibility, is under test: `es` is made selectable
    # so the line renders even while its audit receipt is pending.
    from local_operator.i18n import resolve

    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "es"))
    text.set_active_locale("es")
    es = OperatorApp._uninstalled_anchor_report("ask", "auto")
    assert es is not None and es.startswith("aprobaciones de herramientas: ask")
    for english in ("tool approvals", "this session", "config.yml says", "but approvals"):
        assert english not in es

    monkeypatch.setattr("local_operator.operator.operator_authority_unusable", lambda: False)
    assert OperatorApp._uninstalled_anchor_report("ask", "auto") is None
