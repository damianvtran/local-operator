"""The ``/move --to`` working-set card (docs/design/move-working-set.md §11.3).

The operator's requirement (2026-11-18): the remote move must warn that moving
mid-conversation involves the working files and require the user's explicit approval
in the dialog. Pinned on the REAL ``OperatorApp`` (so ``local_operator.tcss`` applies):

* the card names the destination device and renders the CLI's summary — worktree,
  counts, the withheld secret, what is never copied;
* Move is disabled until the checkbox is ticked, and focus starts on the checkbox;
* Esc / Cancel dismiss ``""``; a ticked Move dismisses ``"confirm"``;
* an ``unknown`` / ``unsupported`` manifest offers the explicit "Move without the
  working files" (``"without"``) behind the same checkbox (review D2/U2, round 1);
* the checkbox and buttons stay ON SCREEN at 80x24 / 100x24 with a long summary,
  which scrolls (review D1/U1); wrapped lines keep their indent (D3); the label
  wraps rather than truncating the device (D4); the empty box is >= 3:1 (D5);
* untrusted text is never parsed as markup (security S2).
"""

from __future__ import annotations

from typing import Any

import pytest
from textual.widgets import Button, Checkbox, Static

from local_operator.i18n.text import set_active_locale
from local_operator.network import workset_text
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.move_workset_confirm import MoveWorkingSetConfirmScreen
from tests.unit.tui.test_app_pilot import FakeSession, _factory

SESSION = "9f3ac1e0b7d2"

BLOCK: dict[str, Any] = {
    "version": 1,
    "scope_digest": "sha256:" + "a" * 64,
    "state": "preflight",
    "carries_files": False,
    "destination": {"device_id": "d_b", "name": "lop-remote-node-2"},
    # The OWNER, named: on a recall this card runs on the destination, so the files
    # staying on "this device" would name the wrong machine.
    "source": {"device_id": "d_a", "name": "laptop"},
    "totals": {"worktrees": 1, "unpushed_commits": 1, "dirty_files": 6, "bytes": 3_200_000},
    "worktrees": [
        {
            "path": "~/proj/.worktrees/feat",
            "repo": "~/proj",
            "head": "0" * 40,
            "state": "planned",
            "branch": "feat",
            "unpushed_commits": 1,
            "staged": 2,
            "unstaged": 2,
            "untracked": 2,
        }
    ],
    "excluded": [{"path": "~/proj/.worktrees/feat/.env.local", "why": "secret_path"}],
    "left_behind": [
        {
            "path": "~/proj/.worktrees/feat/node_modules",
            "kind": "dependency_tree",
            "recipe": {"kind": "pnpm", "run": "pnpm install --frozen-lockfile", "cwd": "~"},
        }
    ],
}


@pytest.fixture(autouse=True)
def _english() -> Any:
    set_active_locale("en")
    yield
    set_active_locale("en")


def _summary(screen: MoveWorkingSetConfirmScreen) -> str:
    """What the summary rows actually render, joined (one Static per line, D3)."""
    return "\n".join(
        str(row.render()) for row in screen.query(".move-workset-line").results(Static)
    )


async def _open(app: OperatorApp, block: dict[str, Any], results: list[Any]) -> Any:
    screen = MoveWorkingSetConfirmScreen(block, session_id=SESSION, device="peer")
    app.push_screen(screen, callback=results.append)
    return screen


@pytest.mark.asyncio
async def test_the_card_renders_the_summary_and_names_the_destination() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    results: list[str] = []
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = await _open(app, BLOCK, results)
        await pilot.pause()
        title = str(screen.query_one("#move-workset-title", Static).render())
        summary = _summary(screen)
        check = screen.query_one("#move-workset-check", Checkbox)
    assert title == "Move without the working files?"
    assert f"Moving {SESSION} to lop-remote-node-2 mid-conversation" in summary
    assert "~/proj/.worktrees/feat  feat (1 unpushed, 2 staged, 2 unstaged, 2 untracked)" in summary
    assert "withheld (looks like a secret): ~/proj/.worktrees/feat/.env.local" in summary
    assert "pnpm install --frozen-lockfile" in summary
    assert "They stay on laptop:" in summary
    assert str(check.label) == "I understand these working files stay on laptop"


@pytest.mark.asyncio
async def test_move_is_disabled_until_the_checkbox_is_ticked() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    results: list[str] = []
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = await _open(app, BLOCK, results)
        await pilot.pause()
        confirm = screen.query_one("#move-workset-confirm", Button)
        check = screen.query_one("#move-workset-check", Checkbox)
        assert confirm.disabled is True
        assert app.focused is check  # the safe default: nothing that moves has focus
        await pilot.press("space")  # tick the focused checkbox
        await pilot.pause()
        assert check.value is True
        assert confirm.disabled is False
        await pilot.click("#move-workset-confirm")
        await pilot.pause()
    assert results == ["confirm"]


@pytest.mark.asyncio
async def test_escape_cancels_and_nothing_is_confirmed() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    results: list[str] = []
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        await _open(app, BLOCK, results)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert results == [""]


@pytest.mark.asyncio
async def test_a_sends_manifest_asks_for_confidence_in_the_destination() -> None:
    sends = {**BLOCK, "carries_files": True}
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = await _open(app, sends, [])
        await pilot.pause()
        title = str(screen.query_one("#move-workset-title", Static).render())
        label = str(screen.query_one("#move-workset-check", Checkbox).label)
    assert title == "Send working files to lop-remote-node-2?"
    assert label == "I'm confident lop-remote-node-2 should receive these files"


UNKNOWN: dict[str, Any] = {
    "version": 1,
    "state": "unknown",
    "destination": {"device_id": "d_b", "name": "lop-remote-node-2"},
    "source": {"device_id": "d_a", "name": "laptop"},
    "failure": {"step": "status", "repo": "~/proj/.worktrees/feat", "cause": "index_corrupt"},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["unknown", "unsupported"])
async def test_an_unreadable_working_set_offers_move_without_behind_the_checkbox(
    state: str,
) -> None:
    """D2/U2: the card used to offer only Cancel, a dead end during every rollout."""
    block = {**UNKNOWN, "state": state, "owner": {"device_id": "d_a", "name": "laptop"}}
    results: list[str] = []
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = await _open(app, block, results)
        await pilot.pause()
        summary = _summary(screen)
        without = screen.query_one("#move-workset-without", Button)
        assert not screen.query("#move-workset-confirm")
        assert without.disabled is True
        assert str(without.label) == "Move without the working files"
        assert app.focused is screen.query_one("#move-workset-check", Checkbox)
        await pilot.press("space")
        await pilot.pause()
        assert without.disabled is False
        await pilot.click("#move-workset-without")
        await pilot.pause()
    assert results == ["without"]
    assert "--without-working-set" in summary
    if state == "unknown":
        # U6: the cause is named in words, with the command that shows it.
        assert "its index file is damaged" in summary
        assert "git -C ~/proj/.worktrees/feat status" in summary


@pytest.mark.asyncio
async def test_an_outdated_destination_offers_only_cancel() -> None:
    block = {
        **UNKNOWN,
        "state": "unsupported",
        "update_device": {"device_id": "d_b", "name": "lop-remote-node-2", "role": "destination"},
    }
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = await _open(app, block, [])
        await pilot.pause()
        summary = _summary(screen)
        assert not screen.query("#move-workset-check")
        assert not screen.query("#move-workset-without")
    assert "Update lop on lop-remote-node-2" in summary
    assert "--without-working-set" not in summary


def _probed_block(count: int = 3) -> dict[str, Any]:
    """BLOCK with ``count`` worktrees on one SSH remote the destination cannot fetch:
    the shape a real Mac -> node2 move produced (design frames, round 1)."""
    rows = []
    for index in range(count):
        rows.append(
            {
                **BLOCK["worktrees"][0],
                "path": f"~/proj/.worktrees/feature-DT-1234-a-long-branch-name-{index}",
                "branch": f"feature/DT-1234-a-long-branch-name-{index}",
                "remote_url": "git@github.com:owner/proj.git",
                "clone_check": "destination_cannot_fetch",
                "https_rewrite": {
                    "url": "https://github.com/owner/proj.git",
                    "fetch": True,
                    "push": False,
                },
                "remedy": {
                    "run_on": "lending_device",
                    "command": "lop network credential share github --with lop-remote-node-2",
                },
            }
        )
    return {
        **BLOCK,
        "worktrees": rows,
        "destination_probe": {
            "state": "answered",
            "missing_tools": ["go", "pnpm"],
            "team_bridge": {"needed": True, "enrolled": False},
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 24), (60, 32)])
async def test_the_consent_controls_stay_on_screen_with_a_long_summary(
    size: tuple[int, int],
) -> None:
    """D1/U1: at 80x24 the buttons were painted below the card; with three probed
    worktrees the checkbox too, and the user ticked and confirmed blind."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = await _open(app, _probed_block(), [])
        await pilot.pause()
        region = screen.region
        card = screen.query_one(".move-workset").region
        scroll = screen.query_one("#move-workset-summary")
        for selector in ("#move-workset-title", "#move-workset-check", "#move-workset-confirm",
                         "#move-workset-cancel"):
            widget = screen.query_one(selector)
            box = widget.region
            assert box.height > 0, (selector, box)
            assert region.contains_region(box), (selector, box, region)
            assert card.contains_region(box), (selector, box, card)
        # The summary is what gives way, and it scrolls rather than clipping.
        assert scroll.virtual_size.height > scroll.size.height, (scroll.virtual_size, scroll.size)
        assert scroll.allow_vertical_scroll is True


@pytest.mark.asyncio
async def test_wrapped_lines_keep_their_indent_and_probe_lines_print_once_per_remote() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        screen = await _open(app, _probed_block(), [])
        await pilot.pause()
        rows = list(screen.query(".move-workset-line").results(Static))
        indents = {str(row.render()): row.styles.padding.left for row in rows}
        regions = {str(row.render()): row.content_region for row in rows}
    worktree = next(text for text in indents if text.startswith("~/proj/.worktrees/feature"))
    assert indents[worktree] == 4  # D3: the hanging indent is the row's own padding
    assert regions[worktree].x > min(r.x for r in regions.values())
    # U7: the destination's answer is per remote, printed once for three worktrees.
    summary = "\n".join(indents)
    assert summary.count("cannot fetch git@github.com:owner/proj.git") == 1
    assert summary.count("share github --with lop-remote-node-2") == 1


@pytest.mark.asyncio
async def test_the_checkbox_label_wraps_and_never_truncates_the_device() -> None:
    sends = {**BLOCK, "carries_files": True}
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(60, 32)) as pilot:
        await pilot.pause()
        screen = await _open(app, sends, [])
        await pilot.pause()
        check = screen.query_one("#move-workset-check", Checkbox)
        lines = [
            "".join(segment.text for segment in strip)
            for strip in check.render_lines(check.region.reset_offset)
        ]
        height = check.region.height
    painted = " ".join(" ".join(line.split()) for line in lines)
    assert "lop-remote-node-2" in painted, lines  # D4: the name the consent rests on
    assert "…" not in painted
    assert height >= 2  # it wrapped instead of being cut


@pytest.mark.asyncio
async def test_the_empty_checkbox_has_three_to_one_contrast() -> None:
    """D5: the empty box measured 1.37:1 against the card."""
    from local_operator.tui import theme

    variables = theme.tcss_variable_map("dark")

    def luminance(hex_value: str) -> float:
        channels = [int(hex_value.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        screen = await _open(app, BLOCK, [])
        await pilot.pause()
        check = screen.query_one("#move-workset-check", Checkbox)
        box = check.get_component_styles("toggle--button").background.hex
    card = variables["lo-overlay"]
    lighter, darker = sorted((luminance(box), luminance(card)), reverse=True)
    assert (lighter + 0.05) / (darker + 0.05) >= 3.0, (box, card)


HOSTILE = [
    "[@click=screen.dismiss('confirm')]XXXX (details)[/]",
    "~/proj/notes[/]draft.env",
    "~/proj/app/[tenant]/.env.local",
]


@pytest.mark.asyncio
async def test_untrusted_text_is_rendered_verbatim_and_cannot_dismiss() -> None:
    """S2: a destination-supplied ``[@click=…]`` got past the checkbox in one click,
    ``[/]`` crashed the app, and ``[tenant]`` vanished from a withheld path."""
    block = {
        **BLOCK,
        "destination": {"device_id": "d_b", "name": "node[/]2"},
        "excluded": [{"path": path, "why": "secret_path"} for path in HOSTILE],
        "worktrees": [
            {
                **BLOCK["worktrees"][0],
                "remote_url": "git@github.com:owner/proj.git",
                "clone_check": "destination_cannot_push",
                "remedy": {"run_on": "lending_device", "command": HOSTILE[0]},
            }
        ],
    }
    results: list[str] = []
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = await _open(app, block, results)
        await pilot.pause()
        summary = _summary(screen)
        rows = list(screen.query(".move-workset-line").results(Static))
        target = next(row for row in rows if "XXXX" in str(row.render()))
        await pilot.click(target)
        await pilot.pause()
        assert app.screen is screen  # the click did nothing
        label = str(screen.query_one("#move-workset-check", Checkbox).label)
    for text in HOSTILE:
        assert text in summary, (text, summary)
    assert results == []
    assert "node[/]2" in summary and "laptop" in label


def test_terminal_escapes_are_stripped_from_every_rendered_line() -> None:
    """S3: an ESC sequence in a remedy rewrote the lines above the CLI's y/N."""
    block = {
        **BLOCK,
        "worktrees": [
            {**BLOCK["worktrees"][0], "path": "~/proj/x\x1b[2K\r\x1b[1AALL CLEAR"}
        ],
    }
    lines = workset_text.summary_lines(block, session=SESSION, device="peer")
    joined = "\n".join(lines)
    assert "\x1b" not in joined and "\r" not in joined
    assert "ALL CLEAR" in joined  # the text stays; only the control bytes go


def test_move_accepts_without_working_set_and_passes_it_to_the_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D2/U2: the card and the refusal copy name ``--without-working-set``; ``/move``
    refused it as an unknown flag, so the advice was a dead end in this surface."""
    from local_operator.tui import session_move

    parsed = session_move.parse_move_to(f"{SESSION} --to peer --without-working-set")
    assert parsed is not None and parsed.error == "", parsed
    assert parsed.without_working_set is True

    seen: list[list[str]] = []
    monkeypatch.setattr(
        session_move, "_run_cli_json", lambda sid, to, argv: seen.append(argv) or {"ok": True}
    )
    session_move.run_session_move(SESSION, "peer", without_working_set=True)
    session_move.run_session_move(SESSION, "peer", confirm_working_set="sha256:x")
    assert "--without-working-set" in seen[0]
    assert "--without-working-set" not in seen[1]  # a digest is the consent there
