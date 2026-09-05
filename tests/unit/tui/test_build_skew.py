"""Skew between what this terminal loaded and the code around it, made visible.

**This file exists because a build mismatch between a viewer and its runtime
was undetectable from either side, and its most expensive symptom was
silence.**

The shape, concretely: a TUI keeps running the code it imported at launch,
``lop-update`` replaces the on-disk install under it several times a day, and
the runtime that TUI spawns resolves ``sys.executable`` fresh — so it is built
from the NEW install. An old terminal therefore drives a new runtime as a
matter of routine, and when the two disagreed about who submits a ``/team``
request, the request vanished with no user row, no turn and no error.

Three notices close that, and each is a different fact:

* **disk drift** — the install moved under this process, so anything spawned
  from here will be newer than this terminal. ``/reload`` is the remedy.
* **owner skew** — the bound runtime reports a different build. ``/stop`` then
  resend is the remedy.
* **owner predates reporting** — the runtime cannot say what it runs, which by
  construction makes it older than a terminal that can read the field.

All three are ADVISORY. Nothing here may refuse a command or an attach: both
builds keep working, and a diagnostic that blocks work is worse than the skew
it reports.
"""

from __future__ import annotations

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock
from local_operator.update import BuildStamp
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _notices(app: OperatorApp) -> list[str]:
    return [block._text for block in app.query(NoticeBlock)]


class _BoundViewer(FakeSession):
    """A follower facade that has already bound to a runtime.

    ``is_remote`` plus a resolved ``owner_version`` is what a real
    ``RemoteSession`` looks like after ``_dial``; ``is_cold`` False is what
    makes the owner comparison meaningful, since a cold viewer has not dialled
    anything and its empty stamp would otherwise read as a prehistoric
    runtime.
    """

    is_remote = True
    is_cold = False

    def __init__(self, owner_version: str = "", owner_source_ref: str = "") -> None:
        super().__init__()
        self.owner_version = owner_version
        self.owner_source_ref = owner_source_ref


@pytest.mark.asyncio
async def test_a_moved_install_warns_once_and_names_reload(monkeypatch, tmp_path) -> None:
    """Disk drift: the install is no longer the one this process loaded.

    The notice has to name BOTH builds — a warning that says only "something
    changed" leaves the user unable to tell a routine rebuild from the
    several-release gap that actually breaks routed commands — and it has to
    name ``/reload``, which is the one action that fixes it without losing the
    session.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.46.23")
        app._skew_notice_shown.clear()

        import local_operator.update as update_mod

        monkeypatch.setattr(
            update_mod, "installed_build", lambda *_a, **_k: BuildStamp(version="0.49.0")
        )
        app._check_build_skew(reason="test")
        await pilot.pause()
        first = _notices(app)

        # A SECOND check with the same pair must stay quiet: the mount engage,
        # the draft warm-up and the slash engage all call this, and three
        # copies of one warning is noise the user learns to skip.
        app._check_build_skew(reason="test-again")
        await pilot.pause()
        second = _notices(app)

    drift = [n for n in first if "install on disk changed" in n]
    assert len(drift) == 1, first
    assert "0.46.23" in drift[0] and "0.49.0" in drift[0], "both builds must be named"
    assert "/reload" in drift[0]
    assert second == first, "the notice is debounced per (kind, from, to)"


@pytest.mark.asyncio
async def test_a_second_distinct_drift_is_still_announced(monkeypatch, tmp_path) -> None:
    """Debounce on the TRIPLE, not on the kind.

    Two ``lop-update`` runs while one terminal lives is normal here. Keying
    the debounce on "have we warned about drift" would report the first and
    swallow every later one, so a terminal that has fallen two releases behind
    would look exactly like one that is current.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.46.23")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(
            update_mod, "installed_build", lambda *_a, **_k: BuildStamp(version="0.49.0")
        )
        app._check_build_skew(reason="one")
        monkeypatch.setattr(
            update_mod, "installed_build", lambda *_a, **_k: BuildStamp(version="0.50.0")
        )
        app._check_build_skew(reason="two")
        await pilot.pause()
        notices = _notices(app)

    drift = [n for n in notices if "install on disk changed" in n]
    assert len(drift) == 2, drift
    assert any("0.49.0" in n for n in drift) and any("0.50.0" in n for n in drift)


@pytest.mark.asyncio
async def test_a_same_version_rebuild_is_detected_through_its_ref(monkeypatch, tmp_path) -> None:
    """The drift this host produces most often, and the reason for the ref.

    ``lop-update`` builds from ``main`` while ``pyproject.toml`` still names
    the last released version, so both sides report the same version string
    and only the recorded commit differs. Version-only comparison reports "no
    drift" for precisely the case that is drifting.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.49.0", source_ref="aaaaaaa1111")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(
            update_mod,
            "installed_build",
            lambda *_a, **_k: BuildStamp(version="0.49.0", source_ref="bbbbbbb2222"),
        )
        app._check_build_skew(reason="rebuild")
        await pilot.pause()
        notices = _notices(app)

    drift = [n for n in notices if "install on disk changed" in n]
    assert len(drift) == 1, notices
    assert "0.49.0@aaaaaaa" in drift[0] and "0.49.0@bbbbbbb" in drift[0]


@pytest.mark.asyncio
async def test_a_matching_build_says_nothing(monkeypatch, tmp_path) -> None:
    """The quiet case, which is almost every check.

    This runs at every adopt and every engage, so a false positive would
    become a notice on every `/new` — the fastest way to teach a user to stop
    reading warnings.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        stamp = BuildStamp(version="0.49.0", source_ref="abc1234")
        app._loaded_build = stamp
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: stamp)
        app._check_build_skew(reason="same")
        await pilot.pause()
        notices = _notices(app)

    assert not [n for n in notices if "install on disk changed" in n]
    assert not [n for n in notices if "runtime is running" in n]


@pytest.mark.asyncio
async def test_an_older_runtime_is_named_with_the_stop_remedy(monkeypatch, tmp_path) -> None:
    """Owner skew: the bound runtime reports a build this terminal is not on.

    ``/stop`` then resend is the remedy rather than ``/reload``: the terminal
    is fine, it is the runtime that is stale, and a bare ``/stop`` on a
    follower asks the owner to stop so the next prompt engages a fresh one
    from the current install.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        stamp = BuildStamp(version="0.49.0")
        app._loaded_build = stamp
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: stamp)
        app._session = _BoundViewer(owner_version="0.46.23")
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)

    skew = [n for n in notices if "runtime is running 0.46.23" in n]
    assert len(skew) == 1, notices
    assert "0.49.0" in skew[0]
    assert "/stop" in skew[0]


@pytest.mark.asyncio
async def test_a_runtime_without_a_stamp_is_reported_as_predating_it(monkeypatch, tmp_path) -> None:
    """An absent version is itself informative, not a missing value.

    The field ships in this build, so a runtime that does not publish it
    predates this terminal by construction. Every resident runtime trips this
    exactly once in the first window after release, and it is telling the
    truth each time.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        stamp = BuildStamp(version="0.49.0")
        app._loaded_build = stamp
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: stamp)
        app._session = _BoundViewer(owner_version="")
        app._check_build_skew(reason="bind")
        app._check_build_skew(reason="bind-again")
        await pilot.pause()
        notices = _notices(app)

    predates = [n for n in notices if "predates build reporting" in n]
    assert len(predates) == 1, "once per session per process, not once per seam"
    assert "/stop" in predates[0]


@pytest.mark.asyncio
async def test_a_matching_runtime_is_silent(monkeypatch, tmp_path) -> None:
    """Same build on both ends: nothing to say."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        stamp = BuildStamp(version="0.49.0", source_ref="abc1234")
        app._loaded_build = stamp
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: stamp)
        app._session = _BoundViewer(owner_version="0.49.0", owner_source_ref="abc1234")
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)

    assert not [n for n in notices if "runtime is running" in n]
    assert not [n for n in notices if "predates" in n]


@pytest.mark.asyncio
async def test_a_cold_viewer_is_not_reported_as_a_prehistoric_runtime(
    monkeypatch, tmp_path
) -> None:
    """A cold viewer has not dialled anything, so it has no owner to compare.

    Its ``owner_version`` is empty for the trivial reason that no runtime
    exists yet. Reading that as "the runtime predates the field" would fire
    the notice on every fresh `lop`, which is the false-positive that would
    make the whole mechanism ignorable.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        stamp = BuildStamp(version="0.49.0")
        app._loaded_build = stamp
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: stamp)
        cold = _BoundViewer(owner_version="")
        cold.is_cold = True
        app._session = cold
        app._check_build_skew(reason="cold")
        await pilot.pause()
        notices = _notices(app)

    assert not [n for n in notices if "predates build reporting" in n]


@pytest.mark.asyncio
async def test_an_unreadable_own_build_disables_the_check(monkeypatch, tmp_path) -> None:
    """No snapshot means no comparison, not a comparison against unknown.

    A TUI that could not read its own version at startup would otherwise
    report drift against every runtime it ever meets.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = None
        app._skew_notice_shown.clear()
        app._session = _BoundViewer(owner_version="0.1.0")
        app._check_build_skew(reason="unknown")
        await pilot.pause()
        notices = _notices(app)

    assert not [n for n in notices if "runtime is running" in n]
    assert not [n for n in notices if "install on disk changed" in n]


@pytest.mark.asyncio
async def test_a_failing_disk_read_does_not_break_the_seam(monkeypatch, tmp_path) -> None:
    """This runs inside adopt and engage; it must never raise into them.

    A skew notice is a diagnostic. An exception here would take down the
    session adoption it was called from, converting a cosmetic problem into a
    broken terminal.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.49.0")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        def explode(*_a, **_k):
            raise OSError("permission denied")

        monkeypatch.setattr(update_mod, "installed_build", explode)
        app._check_build_skew(reason="boom")  # must not raise
        await pilot.pause()
        notices = _notices(app)

    assert not [n for n in notices if "install on disk changed" in n]


@pytest.mark.asyncio
async def test_a_pre_attach_runtime_noop_degrades_loudly(monkeypatch, tmp_path) -> None:
    """The last silent quadrant: a NEW viewer against a PRE-#624 runtime.

    Such a runtime answers ``/team <name>`` with ``noop {"type":
    "team_mutate"}``. The renderer's noop branch handled only ``agent_list``,
    so the command vanished — the same defect the static audit exists to
    prevent, arriving through the version dimension the audit cannot see. No
    current producer emits this type; only a resident older process does.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._render_authoritative_slash(
            "team", "lopdev go", {"kind": "noop", "data": {"type": "team_mutate"}}
        )
        await pilot.pause()
        notices = _notices(app)

    stale = [n for n in notices if "older than /team attach" in n]
    assert len(stale) == 1, notices
    assert "nothing was attached" in stale[0], "the user must learn the attach did NOT happen"
    assert "/stop" in stale[0]


@pytest.mark.asyncio
async def test_an_attach_receipt_still_submits_its_request(monkeypatch, tmp_path) -> None:
    """Regression guard: the declaration work must not break the consumer.

    This viewer DECLARES that it consumes ``team_attached``, which is what
    stops the runtime from completing the request. If the renderer then failed
    to submit it, the request would vanish on the NEW client — the original
    bug, restored, in the place nobody would think to look.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        submitted: list[str] = []
        monkeypatch.setattr(
            app, "_submit_command_prompt", lambda text, attachments=None: submitted.append(text)
        )
        app._render_authoritative_slash(
            "team",
            "lopdev go",
            {
                "kind": "notice",
                "text": "sending to lopdev. manager is coordinating.",
                "data": {"type": "team_attached", "team": "lopdev", "request": "do the thing"},
            },
        )
        await pilot.pause()

    assert submitted == ["do the thing"], (
        "a declaring viewer promises the runtime it will submit this itself; "
        "breaking that promise restores the silent drop on the new client"
    )
