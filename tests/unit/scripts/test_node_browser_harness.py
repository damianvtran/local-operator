"""The node-browser harness's launch guards (``scripts/e2e/node_browser_harness.py``).

The harness writes operator keys, Chrome policy and profiles. These cells pin
the three refusals that keep it off the operator's real node: it runs only with
``--harness``, only in a fresh subdirectory of the session scratchpad (never in
or above the real config root), and the inner run aborts when the real managed
policy is visible (i.e. it is not inside its mount namespace). No Chrome, daemon
or namespace is started here: the guards are pure functions on purpose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.e2e import node_browser_harness as harness


@pytest.fixture()
def layout(tmp_path: Path) -> dict[str, Path]:
    real = tmp_path / "home" / ".local-operator"
    pad = real / "sessions" / "abc123" / "scratchpad"
    pad.mkdir(parents=True)
    return {"real": real, "pad": pad}


def test_no_harness_flag_is_refused(layout: dict[str, Path]) -> None:
    reason = harness.refusal(
        harness_flag=False,
        root=layout["pad"] / "run",
        scratchpad=str(layout["pad"]),
        real_config=layout["real"],
    )
    assert reason is not None and "--harness" in reason


def test_the_real_config_root_is_refused(layout: dict[str, Path]) -> None:
    # Even with the scratchpad pointed at the real root, the root itself is refused.
    reason = harness.refusal(
        harness_flag=True,
        root=layout["real"],
        scratchpad=str(layout["real"]),
        real_config=layout["real"],
    )
    assert reason is not None and "real config root" in reason


def test_a_root_outside_the_scratchpad_is_refused(layout: dict[str, Path], tmp_path: Path) -> None:
    for root in (tmp_path / "elsewhere", layout["real"] / "sessions", layout["real"]):
        reason = harness.refusal(
            harness_flag=True,
            root=root,
            scratchpad=str(layout["pad"]),
            real_config=layout["real"],
        )
        assert reason is not None and "not inside the scratchpad" in reason, root


def test_a_missing_scratchpad_and_the_bare_scratchpad_are_refused(layout: dict[str, Path]) -> None:
    assert "not set" in str(
        harness.refusal(
            harness_flag=True,
            root=layout["pad"] / "run",
            scratchpad=None,
            real_config=layout["real"],
        )
    )
    assert "fresh subdirectory" in str(
        harness.refusal(
            harness_flag=True,
            root=layout["pad"],
            scratchpad=str(layout["pad"]),
            real_config=layout["real"],
        )
    )


def test_a_fresh_scratchpad_subdirectory_is_accepted(layout: dict[str, Path]) -> None:
    assert (
        harness.refusal(
            harness_flag=True,
            root=layout["pad"] / "node-browser-harness-1",
            scratchpad=str(layout["pad"]),
            real_config=layout["real"],
        )
        is None
    )


def test_a_visible_real_policy_aborts_the_inner_run(tmp_path: Path) -> None:
    policy = tmp_path / "local-operator.json"
    assert harness.namespace_refusal(policy) is None
    policy.write_text("{}")
    reason = harness.namespace_refusal(policy)
    assert reason is not None and "NOT inside the harness mount namespace" in reason


def test_main_without_the_flag_exits_2_and_starts_nothing(
    layout: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(layout["pad"]))
    monkeypatch.setattr(harness, "outer", lambda _a: pytest.fail("outer must not run"))
    assert harness.main(["--root", str(layout["pad"] / "run")]) == 2
    assert "--harness" in capsys.readouterr().err


def test_a_loading_frame_does_not_count_as_a_rendered_theme(tmp_path: Path) -> None:
    """Run 6's false pass: a spinner frame differs by hash but shows no story.

    The light/dark row now also needs each frame to carry content. A frame that
    is background plus a small spinner sits far below the bound; one with a
    panel of text-like content sits above it.
    """
    from PIL import Image, ImageDraw

    spinner = Image.new("RGB", (780, 493), (255, 255, 255))
    ImageDraw.Draw(spinner).ellipse((375, 231, 405, 261), outline=(200, 200, 200), width=2)
    spinner_path = tmp_path / "spinner.png"
    spinner.save(spinner_path)

    story = Image.new("RGB", (780, 493), (242, 237, 227))
    draw = ImageDraw.Draw(story)
    draw.rectangle((16, 16, 336, 493), fill=(248, 245, 238))
    for top in range(40, 480, 20):
        draw.rectangle((30, top, 300, top + 8), fill=(40, 40, 40))
    story_path = tmp_path / "story.png"
    story.save(story_path)

    assert harness.rendered_fraction(spinner_path) < harness.RENDERED_MIN_FRACTION
    assert harness.rendered_fraction(story_path) >= harness.RENDERED_MIN_FRACTION


def _frame(path: Path, background: tuple[int, int, int], *, content: bool) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (780, 493), background)
    if content:
        draw = ImageDraw.Draw(image)
        for top in range(40, 480, 20):
            draw.rectangle((30, top, 300, top + 8), fill=(128, 64, 200))
    image.save(path)
    return path


def _request_with_tool_text(text: str) -> object:
    from types import SimpleNamespace

    block = SimpleNamespace(text=text)
    return SimpleNamespace(messages=[SimpleNamespace(role="tool", content=[block])])


def _tool_call_name_and_args(turn: list[object]) -> tuple[str, dict[str, object]]:
    import json

    call = next(event for event in turn if getattr(event, "name", None))
    return str(getattr(call, "id")), json.loads(getattr(call, "argument_delta"))


def test_a_capture_snapshots_then_screenshots_once_the_story_root_is_there(
    tmp_path: Path,
) -> None:
    """Run 6's spinner fix: no screenshot counts until the story has rendered."""
    import asyncio

    shot = tmp_path / "dark.png"
    capture = harness.Capture("dark", str(shot))

    async def drive() -> list[str]:
        ids = []
        turn = await capture.next_turn(_request_with_tool_text(""))
        assert turn is not None
        ids.append(_tool_call_name_and_args(turn)[0])
        # The snapshot found the story root: the next call is the screenshot.
        turn = await capture.next_turn(_request_with_tool_text("heading 'Approvals'"))
        assert turn is not None
        call_id, args = _tool_call_name_and_args(turn)
        ids.append(call_id)
        assert args == {"action": "screenshot", "path": str(shot)}
        _frame(shot, (20, 22, 28), content=True)
        assert await capture.next_turn(_request_with_tool_text("saved")) is None
        return ids

    ids = asyncio.run(drive())
    assert ids[0].startswith("snap_dark_") and ids[1].startswith("shot_dark_")
    assert capture.outcome["rendered"] is True
    assert capture.outcome["background_luminance"] < harness.DARK_MAX_LUMINANCE


def test_a_capture_that_never_renders_its_theme_records_not_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A light page (or a spinner) captured for the DARK story never counts."""
    import asyncio

    monkeypatch.setattr(harness, "RENDER_WAIT_S", 0.0)
    shot = _frame(tmp_path / "dark.png", (242, 237, 227), content=True)
    capture = harness.Capture("dark", str(shot))
    capture._phase = "shot"  # a screenshot was just taken
    capture._started = 1.0  # long ago: the wait is spent

    assert asyncio.run(capture.next_turn(_request_with_tool_text("saved"))) is None
    assert capture.outcome["rendered"] is False
    assert "not rendered" in capture.outcome["error"]
    assert capture.outcome["background_luminance"] > harness.LIGHT_MIN_LUMINANCE
