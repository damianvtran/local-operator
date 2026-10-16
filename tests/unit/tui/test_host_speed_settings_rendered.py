"""Real-App release evidence for the new numeric Settings row.

These are ordinary CI tests, NOT a locally admitted lightweight capture rig.
The fake owns only the session boundary; Settings, config writes, validation,
production CSS and capture geometry remain real. Image review is still required.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SETTING_KEY = "providers.openrouter.retire_below_tps"
SOURCE_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session", autouse=True)
def _prepay_session_import_warmup() -> None:
    """Shadow the parent SESSION fixture, before function monkeypatch setup.

    This module tests an injected facade, not owner construction. Paying the
    real owner's provider/MCP warmup defeats that boundary; its ordering remains
    covered by the dedicated startup tests. No production default is changed.
    """


@pytest.fixture
def ui_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Reject external effects and check counters outside swallowed App errors."""
    root = tmp_path / "synthetic-ui"
    root.mkdir()
    config = root / "config"
    config.mkdir()
    for key in tuple(os.environ):
        if key.startswith(("CMUX_", "LOP_", "LOCAL_OPERATOR_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(root)
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_DESKTOP_LAUNCH", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_SHIMMER", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    violations: list[str] = []

    def refuse(*_args, **_kwargs):
        violations.append("external-effect-or-provider")
        raise AssertionError("UI proof attempted an external effect or provider")

    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, refuse)
    for name in ("system", "posix_spawn", "posix_spawnp"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)

    import asyncio

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", refuse)

    # No App import occurs at module scope. These traps precede this module's
    # product imports, not pytest's already-collected parent conftest imports.
    # That remaining import closure is why this is remote-CI-only, not LIGHT.
    from local_operator.tui import app as app_module
    from local_operator.tui import quota_notice
    from scripts import visual_capture
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    assert Path(app_module.__file__).resolve() == (
        SOURCE_ROOT / "local_operator/tui/app.py"
    )
    assert app_module.OperatorApp.CSS_PATH == "local_operator.tcss"
    assert (SOURCE_ROOT / "local_operator/tui/local_operator.tcss").is_file()

    async def no_quota_prewarm():
        return False

    monkeypatch.setattr(app_module, "start_prewarm", lambda: None)
    monkeypatch.setattr(app_module, "await_prewarm", no_quota_prewarm)
    monkeypatch.setattr(quota_notice, "_prewarm_started", True)

    async def no_attended_onboarding(_self):
        return None

    # A fresh synthetic HOME is deliberately not a first attended install.
    # Only this unrelated onboarding route is replaced, not boot/adoption,
    # on_mount, polling, Settings handlers or their real ConfigManager writes.
    monkeypatch.setattr(
        app_module.OperatorApp, "_route_first_run_boot", no_attended_onboarding
    )
    for name in (
        "_check_for_update",
        "_start_terminal_title",
        "_start_multiplexer_broadcast",
        "_start_herdr_reporter",
    ):
        monkeypatch.setattr(app_module.OperatorApp, name, lambda _self: None)

    # save_capture's optional fontconfig metadata normally spawns fc-match.
    # Record UNMEASURED rather than allow an incidental child or invent font
    # metrics. The real SVG exporter/CSS/geometry are untouched; the independent
    # rasterizer later owns actual font selection and viewed-pixel evidence.
    monkeypatch.setattr(
        visual_capture,
        "font_provenance",
        lambda _profile: {"status": "unmeasured: no font subprocess in UI proof"},
    )

    class BoundSession(FakeSession):
        async def prompt(self, *_args, **_kwargs):
            refuse()

        async def complete_once(self, *_args, **_kwargs):
            refuse()

    session = BoundSession()
    factory_calls: list[str] = []

    async def factory():
        factory_calls.append("synthetic-facade")
        return await _factory(session)

    app = app_module.OperatorApp(factory, warm_session_imports=False)
    proof = SimpleNamespace(
        app=app,
        session=session,
        config=config,
        root=root,
        output=tmp_path / "proof",
        violations=violations,
        factory_calls=factory_calls,
    )
    try:
        yield proof
    finally:
        assert violations == [], "caught App exceptions must not hide tripwire calls"
        assert factory_calls == ["synthetic-facade"]
        assert not session.prompts and not session.completions
        assert getattr(app, "_notifier", None) is None


async def capture_pair(proof, pilot, case: str, phase: str) -> set[str]:
    """Use the existing active-screen exporter; keep unviewed evidence labelled."""
    from scripts.visual_capture import save_capture

    proof.output.mkdir(exist_ok=True)
    generated: set[str] = set()
    previous = None
    for frame in ("immediate", "settled"):
        await pilot.pause()
        stem = f"{case}-{phase}-{frame}"
        target = proof.output / f"{stem}.svg"
        save_capture(proof.app, target)
        geometry_path = target.with_suffix(".geometry.json")
        geometry = json.loads(geometry_path.read_text())
        assert geometry["grid"] == list(proof.app.size)
        assert geometry["widgets"] and geometry["css_path"]
        measured = {key: geometry[key] for key in ("grid", "screen", "widgets")}
        if previous is not None:
            assert previous == measured, "consecutive frames changed geometry"
        previous = measured
        for path in (target, geometry_path):
            data = path.read_bytes()
            # Fail closed rather than rewrite visible pixels or conceal a leak.
            for private_prefix in (proof.root, SOURCE_ROOT):
                assert str(private_prefix).encode() not in data
            generated.add(path.name)
    publish_proof(proof, case, phase, generated)
    return generated


def publish_proof(proof, case: str, phase: str, names: set[str]) -> None:
    """Extend only the existing unit artifact; never collect arbitrary files."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    # Unit CI is POSIX. A lock covers all xdist workers in THIS job; neither a
    # partial overwrite nor a per-worker quota may hide a per-job overflow.
    import fcntl

    target = SOURCE_ROOT / "test-artifacts" / "tui-proof"
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".manifest.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = target / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        )
        head = (SOURCE_ROOT / ".git" / "HEAD").read_text().strip()
        assert len(head) == 40 and all(c in "0123456789abcdef" for c in head)
        assert phase in proof.expected_phases
        incoming = {}
        for name in names:
            assert name.startswith(case + "-")
            assert name.endswith((".svg", ".geometry.json"))
            incoming[name] = (proof.output / name).read_bytes()
        existing = {
            p.name: p.stat().st_size
            for p in target.iterdir()
            if p.suffix in (".svg", ".json")
        }
        assert sum(n.endswith(".svg") for n in existing.keys() | names) <= 48
        sources = (
            "local_operator/tui/local_operator.tcss",
            "local_operator/tui/app.py",
            "local_operator/tui/widgets/settings_view.py",
            "local_operator/tui/widgets/tool_card.py",
            "local_operator/tools/tool_docs.py",
            "tests/unit/tui/test_host_speed_settings_rendered.py",
            "tests/unit/tui/test_ownership_inspection_rendered.py",
        )
        manifest.update(
            checkout_revision=head,
            run_id=os.environ.get("GITHUB_RUN_ID"),
            python=sys.version,
            node=None,
            node_note="Python/Textual capture; Node not used or measured",
            font_note="unmeasured here; independent rasterization/view still required",
            source_sha256={
                name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
                for name in sources
            },
        )
        manifest.setdefault("expected_phases", {})[case] = sorted(
            proof.expected_phases
        )
        manifest.setdefault("cases", {}).setdefault(case, {})[phase] = {
            "grid": list(proof.app.size),
            "files": {
                name: hashlib.sha256(incoming[name]).hexdigest()
                for name in sorted(names)
            },
            "external_attempts_at_capture": len(proof.violations),
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode()
        projected = (
            sum(
                size
                for name, size in existing.items()
                if name not in incoming and name != "manifest.json"
            )
            + sum(map(len, incoming.values()))
            + len(manifest_bytes)
        )
        assert projected <= 12 * 1024 * 1024, "proof quota exceeded before copying"
        for name, data in incoming.items():
            (target / name).write_bytes(data)
        manifest_path.write_bytes(manifest_bytes)


async def replace_inline(pilot, view, value: str) -> None:
    # The row paints an inline buffer, not an Input widget. Exercise its actual
    # end/backspace/printable key handlers rather than writing _buffer directly.
    await pilot.press("end", *(["backspace"] * len(view._buffer)), *tuple(value))


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [100, 60])
async def test_retirement_float_row_real_app(ui_proof, width: int) -> None:
    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.tui.widgets.editor import Editor
    from local_operator.tui.widgets.settings_view import SettingsView
    from tests.unit.tui.conftest import await_session_boot

    proof = ui_proof
    app = proof.app
    case = f"host-speed-{width}x30"
    phases: set[str] = set()
    proof.expected_phases = {"default", "editor", "saved", "reset", "invalid", "closed"}
    setting = settings_io.resolve_key(SETTING_KEY)
    assert setting is not None
    assert setting.kind is settings_io.Kind.FLOAT
    assert (setting.default, setting.minimum, setting.maximum) == (0.0, 0.0, 100000.0)
    async with app.run_test(size=(width, 30)) as pilot:
        await await_session_boot(app)
        assert app._session is proof.session
        editor = app.query_one(Editor)
        editor.focus()
        editor.load_text("/settings")
        await pilot.press("enter")
        view = app.query_one(SettingsView)
        indices = [
            i
            for i, row in enumerate(view._rows)
            if row.kind == "setting" and row.setting and row.setting.key == SETTING_KEY
        ]
        assert len(indices) == 1
        await pilot.press("home")
        # Section headers are not selectable, so a row index is not a number
        # of Down keys. Navigate through real focus/selection events instead.
        for _ in range(len(view._rows)):
            if view.selected_key == SETTING_KEY:
                break
            await pilot.press("down")
        assert view.selected_key == SETTING_KEY
        assert view.has_focus and setting.help
        manager = ConfigManager(proof.config)

        async def shot(phase):
            phases.add(phase)
            await capture_pair(proof, pilot, case, phase)

        assert settings_io.read_setting(manager, setting) == 0.0
        await shot("default")
        await pilot.press("enter")
        assert view.editing_key == SETTING_KEY
        await replace_inline(pilot, view, "25")
        await shot("editor")
        await pilot.press("enter")
        assert view.editing_key is None and not view.error_text
        assert settings_io.read_setting(ConfigManager(proof.config), setting) == 25.0
        await shot("saved")
        await pilot.press("r")
        assert settings_io.read_setting(ConfigManager(proof.config), setting) == 0.0
        await shot("reset")
        config_path = proof.config / "config.yml"
        before = config_path.read_bytes()
        for invalid in ("-1", "100001", "not-a-number"):
            if view.editing_key is None:
                await pilot.press("enter")
            await replace_inline(pilot, view, invalid)
            await pilot.press("enter")
            assert view.editing_key == SETTING_KEY and view.error_text
            assert config_path.read_bytes() == before
        await shot("invalid")
        await pilot.press("escape")
        assert view.editing_key is None and view.selected_key == SETTING_KEY
        await pilot.press("escape")
        assert app._settings_view is None and editor.has_focus
        assert config_path.read_bytes() == before
        await shot("closed")
        assert phases == proof.expected_phases
