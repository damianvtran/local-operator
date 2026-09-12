"""The developer capture must measure the app, not redesign its cell layout."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from rich.console import Console

from scripts.visual_capture import (
    CaptureProfile,
    save_capture,
    settle_status_line,
    terminal_svg,
)

NS = {"s": "http://www.w3.org/2000/svg"}


def _svg() -> str:
    console = Console(width=10, height=2, record=True)
    console.print("ab 界 e\u0301", end="\n")
    console.print("0123456789", end="")
    return console.export_svg()


def test_geometry_and_local_fonts_preserve_cells() -> None:
    svg = terminal_svg(_svg(), 10, 2, CaptureProfile())
    root = ET.fromstring(svg)
    assert root.get("viewBox") == "0 0 80 34"
    assert root.get("width") == "80"
    assert root.get("height") == "34"
    assert "@font-face" not in svg
    assert "cdnjs" not in svg
    assert "textLength" not in svg
    assert not root.findall("s:circle", NS)
    assert root.find("s:g", NS).get("transform") is None  # type: ignore[union-attr]
    text = root.find(".//s:g/s:g/s:text", NS)
    assert text is not None
    assert [s.get("x") for s in text] == ["0", "8", "16", "24", "40", "48"]
    assert [s.text for s in text][-1] == "e\u0301"
    assert "Menlo, DejaVu Sans Mono, monospace" in svg


@pytest.mark.parametrize("cluster", ["👩‍💻", "👨‍👩‍👧‍👦", "界\u0301", "❤️"])
def test_grapheme_shaping_and_following_ascii_origin(cluster: str) -> None:
    from rich.cells import cell_len

    console = Console(width=10, height=1, record=True)
    console.print(cluster + "X", end="")
    root = ET.fromstring(terminal_svg(console.export_svg(), 10, 1, CaptureProfile()))
    spans = root.findall(".//s:tspan", NS)
    assert spans[0].text == cluster
    assert spans[0].get("x") == "0"
    assert spans[1].text == "X"
    assert spans[1].get("x") == str(8 * cell_len(cluster))


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_dimensions_fail(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        CaptureProfile(cell_width=value)


def test_invalid_font_and_upstream_geometry_fail_loudly() -> None:
    with pytest.raises(ValueError, match="plain CSS"):
        CaptureProfile(font_family="foo; color:red")
    with pytest.raises(ValueError, match="font size"):
        CaptureProfile(font_size=18)
    with pytest.raises(ValueError, match="unsupported Rich"):
        terminal_svg(
            _svg().replace("translate(9, 41)", "translate(8, 40)"), 10, 2, CaptureProfile()
        )


@pytest.mark.asyncio
async def test_real_app_export_and_widget_geometry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        before = {id(w): w.region for w in app.query("*")}
        target = tmp_path / "capture.svg"
        with pytest.raises(ValueError, match="destination must end"):
            save_capture(app, tmp_path / "wrong.png")
        assert not (tmp_path / "wrong.png").exists()
        save_capture(app, target)
        assert before == {id(w): w.region for w in app.query("*")}
        data = json.loads(target.with_suffix(".geometry.json").read_text())
        assert data["grid"] == [100, 30]
        assert data["native_pixels"] == [800, 510]
        assert data["widgets"]
        assert data["css_path"]
        # The public Textual export is intentionally still its legacy format.
        assert "translate(9, 41)" in app.export_screenshot()


def test_cssless_host_cannot_be_visual_evidence(tmp_path: Path) -> None:
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="production CSS"):
        save_capture(SimpleNamespace(CSS_PATH=None), tmp_path / "bare.svg")
    assert not (tmp_path / "bare.svg").exists()


def test_all_current_sample_writers_adopt_helper() -> None:
    import ast

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    writers = []
    for path in scripts.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "save_screenshot", f"legacy capture in {path.name}"
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "save_capture":
                    writers.append(path.name)
        if path.name in writers:
            # Either sandbox is acceptable; ``probe_isolation`` is the stricter
            # (import-time, refuses a late import) and is what new scripts use.
            text = path.read_text()
            assert (
                "isolate_capture()" in text or "import scripts.probe_isolation" in text
            ), f"{path.name} writes captures without re-homing HOME/config first"
    assert len(set(writers)) >= 23


def test_probe_isolation_refuses_a_late_import(tmp_path: Path) -> None:
    """The helper is only useful FIRST: an app module already imported has
    already resolved HOME. Importing it late must raise, not silently
    re-home a process that has the real config in memory."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    late = subprocess.run(
        [
            sys.executable,
            "-c",
            "import local_operator.paths; import scripts.probe_isolation",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert late.returncode != 0 and "must be imported BEFORE" in late.stderr, late.stderr
    early = subprocess.run(
        [
            sys.executable,
            "-c",
            # The CMUX assertion is the N1 pin: an inherited live workspace id
            # must not survive the import, however the caller set it. A headless
            # pilot that kept one has renamed the operator's real workspaces.
            "import os; os.environ['CMUX_WORKSPACE_ID'] = 'live-workspace'; "
            "import scripts.probe_isolation as p, os; "
            "from local_operator.paths import config_dir; "
            "assert str(config_dir()).startswith(str(p.SANDBOX)), config_dir(); "
            "assert os.environ['HOME'] == str(p.SANDBOX); "
            "assert 'CMUX_WORKSPACE_ID' not in os.environ, os.environ.get('CMUX_WORKSPACE_ID'); "
            "print('ok')",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert early.returncode == 0 and early.stdout.strip() == "ok", early.stderr


# -- the capture must not race the status band (review round 2, M1/Q1) ---------
#
# `settle_status_line` exists so a before/after pair differs in the change under
# test and nothing else, so what these pin is the WAIT: that the pending
# sentinel is not mistaken for a settled band, that a settled band costs no
# frames, and that neither a missing band nor a wedged one can fail a capture.
# The first of the three fails on the implementation that shipped in round 1 —
# it tested `_model_label` for truthiness, and `MODEL_PENDING` is a truthy
# string, so it returned after one frame with the band still pending.


class _FakePilot:
    """Counts frames, so a wait is asserted instead of waited out for real.

    ``settle_after`` is the number of frames the band needs before it carries a
    real label; ``None`` means it never gets there.
    """

    def __init__(self, band: object, *, settle_after: int | None = None) -> None:
        self.pauses = 0
        self._band = band
        self._settle_after = settle_after

    async def pause(self) -> None:
        self.pauses += 1
        if self._settle_after is not None and self.pauses >= self._settle_after:
            setattr(self._band, "_model_label", "test/model")


def _pending_band() -> SimpleNamespace:
    from local_operator.tui.widgets.welcome import MODEL_PENDING

    return SimpleNamespace(_model_label=MODEL_PENDING)


@pytest.mark.asyncio
async def test_settle_status_line_does_not_treat_the_pending_sentinel_as_settled(capsys) -> None:
    """THE regression pin for M1: `connecting…` is a NON-EMPTY label.

    A band stuck on the sentinel must be waited out (the bounded tries) and
    reported, never handed back as settled. On the round-1 implementation this
    fails twice over: it returns after a single frame, and it says nothing.
    """
    band = _pending_band()
    pilot = _FakePilot(band)

    await settle_status_line(pilot, SimpleNamespace(_status=band), tries=4)

    captured = capsys.readouterr()
    assert pilot.pauses == 4, "the pending sentinel was mistaken for a settled band"
    assert "still reads" in captured.err and "connecting" in captured.err


@pytest.mark.asyncio
async def test_settle_status_line_waits_exactly_until_the_label_lands() -> None:
    """A band that settles on the third frame is waited for, and no longer."""
    band = _pending_band()
    pilot = _FakePilot(band, settle_after=3)

    await settle_status_line(pilot, SimpleNamespace(_status=band), tries=200)

    assert pilot.pauses == 3
    assert band._model_label == "test/model"


@pytest.mark.asyncio
async def test_settle_status_line_costs_no_frames_when_already_settled() -> None:
    """The common case: the caller's own pauses already settled the band."""
    band = SimpleNamespace(_model_label="test/model")
    pilot = _FakePilot(band)

    started = time.monotonic()
    await settle_status_line(pilot, SimpleNamespace(_status=band), tries=200)
    elapsed = time.monotonic() - started

    assert pilot.pauses == 0
    # Generous, because this asserts "no waiting happened", not a benchmark: the
    # round-1 helper would have spent tries * pause() here for a band it could
    # not read at all.
    assert elapsed < 1.0, elapsed


@pytest.mark.asyncio
async def test_settle_status_line_no_ops_without_a_status_band() -> None:
    """No band (or an unreadable one) is not a reason to fail a capture.

    Neither branch may raise: `settle_status_line` is exported from the shared
    capture module, and a status line is not worth losing a frame over.
    """
    for app in (SimpleNamespace(), SimpleNamespace(_status=SimpleNamespace())):
        pilot = _FakePilot(app)
        started = time.monotonic()
        await settle_status_line(pilot, app, tries=200)
        elapsed = time.monotonic() - started
        assert pilot.pauses == 0, app
        assert elapsed < 1.0, (app, elapsed)
