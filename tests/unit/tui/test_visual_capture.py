"""The developer capture must measure the app, not redesign its cell layout."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from rich.console import Console

from scripts.visual_capture import CaptureProfile, save_capture, terminal_svg

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
        save_capture(app, target)
        assert before == {id(w): w.region for w in app.query("*")}
        data = json.loads(target.with_suffix(".geometry.json").read_text())
        assert data["grid"] == [100, 30]
        assert data["native_pixels"] == [800, 510]
        assert data["widgets"]
        assert data["css_path"]
        # The public Textual export is intentionally still its legacy format.
        assert "translate(9, 41)" in app.export_screenshot()


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
            assert "isolate_capture()" in path.read_text()
    assert len(set(writers)) >= 23
