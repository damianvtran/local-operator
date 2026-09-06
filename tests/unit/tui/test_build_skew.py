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
* **owner skew** — the bound runtime reports a different build. The remedy
  is the RUNTIME's: an idle stale runtime is asked to retire now and the
  viewer re-engages a fresh one silently; a busy one earns one info line
  saying it will move over when its work finishes. No notice ever tells the
  user to ``/stop`` (design-runtime-autorefresh §3.3/§3.5).
* **owner predates reporting** — the runtime cannot say what it runs, which by
  construction makes it older than a terminal that can read the field.

All three are ADVISORY. Nothing here may refuse a command or an attach: both
builds keep working, and a diagnostic that blocks work is worse than the skew
it reports.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock
from local_operator.update import BuildStamp
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _notices(app: OperatorApp) -> list[str]:
    return [block._text for block in app.query(NoticeBlock)]


def _wrapped_rows(block: NoticeBlock) -> list[str]:
    """The notice's lines AS WRAPPED by the block at its current width.

    ``_text`` is the unwrapped sentence, so a wrap defect is invisible to it;
    this renders what the compositor laid out, which is what the reader sees.
    """
    from rich.console import Console

    console = Console(width=block.content_size.width or 80, no_color=True)
    with console.capture() as capture:
        console.print(block.renderable)
    return [line.rstrip() for line in capture.get().splitlines() if line.strip()]


# Match on a STABLE fragment of each notice rather than a whole sentence: the
# copy is a design surface and was rewritten once already (design review round
# 1, D1/D2), so cells that quote whole strings turn every wording change into a
# batch of unrelated test failures. These name the one phrase that identifies
# each notice.
DRIFT = "was updated after this window opened"
OWNER_SKEW = "\u2192"  # the old \u2192 new build pair, kept on one row (design round 1, D1)
OWNER_UNKNOWN = "running an older version than this window"
MOVES_OVER = "will move to the new version when its current work finishes"


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

    def __init__(
        self,
        owner_version: str = "",
        owner_source_ref: str = "",
        session_id: str = "",
        conversation_name: str = "",
        idle: bool = False,
        refresh_answer: str = "retiring",
    ) -> None:
        super().__init__()
        self.owner_version = owner_version
        self.owner_source_ref = owner_source_ref
        # ``idle`` is what a real ``RemoteSession.owner_idle`` reads off the
        # canonical snapshot. The default is BUSY so every pre-existing cell
        # keeps exercising the notice path; the refresh cells opt in.
        self._skew_idle = idle
        # What the RUNTIME answers ``refresh_if_idle`` with. ``retiring`` is
        # the happy path; a ``kept: \u2026`` answer (busy again, or an old runtime
        # that does not know the op) is the case R1-1 painted nothing for.
        # ``raise`` makes the ask itself fail.
        self._refresh_answer = refresh_answer
        self.refresh_requests = 0
        # ``session_id`` and ``conversation_name`` are read-only properties on
        # the base double. The debounce is keyed by the first and the notice
        # names the session with the second, so a test that needs two DISTINCT
        # named sessions has to override both here rather than assign them.
        self._skew_session_id = session_id
        self._skew_conversation_name = conversation_name

    @property
    def session_id(self) -> str:  # type: ignore[override]
        return self._skew_session_id or super().session_id

    @property
    def conversation_name(self) -> str:  # type: ignore[override]
        return self._skew_conversation_name

    def owner_idle(self) -> bool:
        return self._skew_idle

    async def request_refresh(self) -> str:
        self.refresh_requests += 1
        if self._refresh_answer == "raise":
            raise ConnectionError("owner went away mid-ask")
        return self._refresh_answer


@pytest.mark.asyncio
async def test_a_moved_install_warns_once_and_names_reload(monkeypatch, tmp_path) -> None:
    """Disk drift: the install is no longer the one this process loaded.

    The notice has to name BOTH builds — a warning that says only "something
    changed" leaves the user unable to tell a routine rebuild from the
    several-release gap that actually breaks commands — and it has to
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

    drift = [n for n in first if DRIFT in n]
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

    drift = [n for n in notices if DRIFT in n]
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

    drift = [n for n in notices if DRIFT in n]
    assert len(drift) == 1, notices
    # The refs are the only distinguishing fact, so both must appear — but the
    # shared version is named ONCE rather than repeated on both arms, which is
    # the whole of D3: `0.49.0@aaaaaaa → 0.49.0@bbbbbbb` spent 22 characters
    # restating one version in the sentence's most prominent parenthetical.
    assert "0.49.0, aaaaaaa \u2192 bbbbbbb" in drift[0], drift[0]
    assert "0.49.0@" not in drift[0], "the version must not be repeated per arm"


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

    assert not [n for n in notices if DRIFT in n]
    assert not [n for n in notices if OWNER_SKEW in n]


@pytest.mark.asyncio
async def test_a_busy_older_runtime_is_told_it_will_move_over(monkeypatch, tmp_path) -> None:
    """Owner skew on a BUSY runtime: one info line, and no chore for the user.

    The runtime's own reaper retires it when its work finishes, so the notice
    only says that. ``note`` rather than ``warning`` because nothing is wrong
    and nothing is asked. The ``/stop`` sentence that used to end this notice
    is the chore the operator refused ("the user should never need to run
    /stop to refresh or update a runtime"), so its absence is asserted on the
    whole ledger, not just this row.
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
        viewer = _BoundViewer(owner_version="0.46.23", idle=False)
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)
        tokens = [block._token for block in app.query(NoticeBlock) if MOVES_OVER in block._text]

    skew = [n for n in notices if "running 0.46.23" in n and OWNER_SKEW in n]
    assert len(skew) == 1, notices
    assert "0.49.0" in skew[0]
    assert MOVES_OVER in skew[0]
    assert tokens == ["muted"], "a note, never a warning: nothing is wrong and nothing is asked"
    assert viewer.refresh_requests == 0, "a busy runtime is never asked to retire"
    assert not any("/stop" in n for n in notices), notices


@pytest.mark.asyncio
async def test_an_idle_older_runtime_is_refreshed_silently(monkeypatch, tmp_path) -> None:
    """Owner skew on an IDLE runtime: request the refresh, paint nothing.

    The belt for the reaper (design-runtime-autorefresh §3.3): a resume in
    the seconds after ``lop-update`` binds to a stale idle owner before its
    reaper has noticed. The viewer asks it to retire now and re-engages on
    its ``retiring`` frame; the user sees the band's ``starting…`` for a
    second and no prose at all.
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
        viewer = _BoundViewer(owner_version="0.46.23", idle=True)
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        notices = _notices(app)

    assert viewer.refresh_requests == 1, "an idle stale owner is asked to retire"
    assert not [n for n in notices if OWNER_SKEW in n or MOVES_OVER in n], notices
    assert not any("/stop" in n for n in notices), notices


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        "kept: busy",
        "kept: unknown op: 'refresh_if_idle'",
        "raise",
    ],
)
async def test_an_idle_owner_that_stays_still_gets_the_notice(
    monkeypatch, tmp_path, answer
) -> None:
    """R1-1: silence is earned by ``retiring``, never by asking.

    The upgrade-window population is a resume onto a runtime built BEFORE this
    PR: it answers the unknown op with an error, cannot self-refresh, and used
    to leave the user with nothing at all \u2014 the old ``/stop`` notice gone and
    no replacement, on every later idle seam too (the debounce is not the
    reason; that branch simply never announced). A ``kept: busy`` answer and a
    failed ask are the same shape: the runtime is staying, so say so.
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
        viewer = _BoundViewer(owner_version="0.46.23", idle=True, refresh_answer=answer)
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        notices = _notices(app)
        tokens = [block._token for block in app.query(NoticeBlock) if MOVES_OVER in block._text]

    assert viewer.refresh_requests == 1, "the refresh is still attempted first"
    skew = [n for n in notices if MOVES_OVER in n]
    assert len(skew) == 1, f"a runtime that stays must say so: {notices}"
    assert "0.46.23" in skew[0] and "0.49.0" in skew[0], "both builds are named"
    assert tokens == ["muted"], "still a note: the runtime repairs itself when it can"
    assert not any("/stop" in n for n in notices), notices


@pytest.mark.asyncio
async def test_an_unstamped_idle_owner_that_stays_gets_the_unknown_copy(
    monkeypatch, tmp_path
) -> None:
    """The same rule for a runtime too old to report its build at all.

    It cannot know the op either, so this is the commonest shape of R1-1 in
    the wild: no stamp AND no ``refresh_if_idle``.
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
        viewer = _BoundViewer(owner_version="", idle=True, refresh_answer="raise")
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        notices = _notices(app)

    predates = [n for n in notices if OWNER_UNKNOWN in n]
    assert len(predates) == 1, notices
    assert MOVES_OVER in predates[0]
    assert "/stop" not in predates[0]


@pytest.mark.asyncio
async def test_the_version_pair_survives_a_narrow_splash(monkeypatch, tmp_path) -> None:
    """D1: the two stamps are ONE fact and must not wrap apart.

    The parenthetical form split between ``(this window is`` and the second
    stamp on an 80- and 100-column splash \u2014 which is exactly where a resume
    paints C\u2032. Asserted on the block the real app wrapped, at the width the
    finding was reported at, rather than on the unwrapped string.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for width in (80, 100, 120):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(width, 24)) as pilot:
            await pilot.pause()
            stamp = BuildStamp(version="0.49.9", source_ref="f4a70b991234567")
            app._loaded_build = stamp
            app._skew_notice_shown.clear()
            import local_operator.update as update_mod

            monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: stamp)
            app._session = _BoundViewer(
                owner_version="0.49.8",
                owner_source_ref="46a4e9b1234567",
                conversation_name="Runtime refresh notes",
            )
            app._check_build_skew(reason="bind")
            await pilot.pause()
            await pilot.pause()
            block = next(b for b in app.query(NoticeBlock) if MOVES_OVER in b._text)
            rows = _wrapped_rows(block)

        joined = "\n".join(rows)
        assert "0.49.8@46a4e9b" in joined and "0.49.9@f4a70b9" in joined, (width, rows)
        pair_rows = [r for r in rows if "0.49.8@46a4e9b" in r or "0.49.9@f4a70b9" in r]
        assert len(pair_rows) == 1, f"the version pair wrapped apart at {width}: {rows}"


@pytest.mark.asyncio
async def test_the_refresh_callback_re_engages_eagerly(monkeypatch, tmp_path) -> None:
    """The ``retiring`` frame lands as a re-engage, not as a cold band.

    ``_warm_engage_started`` was latched by the engage that bound the runtime
    which just left; without the reset the next engage would be a no-op and
    the first keystroke after ``lop-update`` would pay a cold start in the
    foreground.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        reasons: list[str] = []
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: reasons.append(reason))
        app._warm_engage_started = True
        app._on_runtime_refreshed()
        await pilot.pause()

    assert reasons == ["refresh"]
    assert app._warm_engage_started is False


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

    predates = [n for n in notices if OWNER_UNKNOWN in n]
    assert len(predates) == 1, "one session, repeatedly checked, speaks once"
    assert MOVES_OVER in predates[0]
    assert "/stop" not in predates[0]


@pytest.mark.asyncio
async def test_a_second_stale_session_gets_its_own_notice(monkeypatch, tmp_path) -> None:
    """ "Once per SESSION per process" — the claim the debounce key must honour.

    A terminal that adopts one stale runtime, then `/resume`s onto another, is
    looking at two different stale runtimes. Keying the debounce on the notice
    kind alone silently swallows the second, so the user is told about one of
    the two and has no way to know the other is also stale. Design §6.7 and
    this method's contract both say per-session; this cell is what makes that
    true rather than merely written down (review round 1, R1-5, NIT-1).
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

        first = _BoundViewer(owner_version="", session_id="sessionaaaa1")
        app._session = first
        app._check_build_skew(reason="bind")
        app._check_build_skew(reason="bind-again")  # same session: still once

        second = _BoundViewer(owner_version="", session_id="sessionbbbb2")
        app._session = second
        app._check_build_skew(reason="resume")
        await pilot.pause()
        notices = _notices(app)

    predates = [n for n in notices if OWNER_UNKNOWN in n]
    assert len(predates) == 2, (
        "each stale session must be reported once; a per-process key hides "
        "every runtime after the first"
    )


@pytest.mark.asyncio
async def test_disk_drift_is_not_rescoped_by_a_session_swap(monkeypatch, tmp_path) -> None:
    """Drift is a fact about THIS PROCESS, so it must not repeat per session.

    The counterpart to the cell above: scoping every notice by session would
    make an unchanged disk drift re-announce on each `/new`, which is the
    notice-fatigue failure. Only the owner notices carry a scope.
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
        first = _BoundViewer(owner_version="0.49.0", session_id="sessionaaaa1")
        app._session = first
        app._check_build_skew(reason="adopt")

        second = _BoundViewer(owner_version="0.49.0", session_id="sessionbbbb2")
        app._session = second
        app._check_build_skew(reason="adopt-2")
        await pilot.pause()
        notices = _notices(app)

    drift = [n for n in notices if DRIFT in n]
    assert len(drift) == 1, "disk drift is per-process, not per-session"


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

    assert not [n for n in notices if OWNER_SKEW in n]
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

    assert not [n for n in notices if OWNER_UNKNOWN in n]


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

    assert not [n for n in notices if OWNER_SKEW in n]
    assert not [n for n in notices if DRIFT in n]


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

    assert not [n for n in notices if DRIFT in n]


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

    stale = [n for n in notices if "too old to attach a team" in n]
    assert len(stale) == 1, notices
    assert "nothing was attached" in stale[0], "the user must learn the attach did NOT happen"
    assert "/stop" not in stale[0], "the runtime refreshes itself; the user is never told to /stop"
    # D2: the tail states the automatic repair rather than handing back a
    # chore. "send the request again then" was the last remaining instruction
    # on this surface, in the same ink as notice A's genuine /reload action.
    assert "on its own" in stale[0]
    assert "send the request again" not in stale[0]


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


# ---------------------------------------------------------------------------
# The TUI-hosted owner mirror
#
# A session is owned either by a detached runtime or by THIS app, and a
# follower routing `/team <name> <request>` must get the same answer from
# both. Every completion cell in
# ``tests/unit/session/runtime/test_action_receipt_completion.py`` drives
# ``OwnedSessionHandle``, so before these cells the app-side copy of the
# predicate was executed by nothing in CI: an edit drifting it toward
# ``declared is None`` would double-submit on the TUI-owner path with no test
# noticing (review round 1, R1-3).
# ---------------------------------------------------------------------------


async def _app_with_team(app: OperatorApp, tmp_path, name: str = "lopdev") -> None:
    """Give the app's session a real registry holding one attachable team."""
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    registry = TeamRegistry(tmp_path / "teams")
    registry.create_team(
        TeamEditFields(
            name=name,
            description="d",
            manager="manager",
            members=[TeamMember(role="coder")],
        )
    )
    # ``SessionProtocol`` does not declare ``team_registry`` (it is session
    # state the real Session carries and the pilot double mirrors), so the
    # assignment is narrowed for the type checker rather than the protocol
    # being widened for a test.
    setattr(app._session, "team_registry", registry)


@pytest.mark.asyncio
async def test_the_tui_owner_completes_for_an_undeclaring_client(monkeypatch, tmp_path) -> None:
    """The incident shape when the OWNER is a TUI rather than a runtime.

    Same contract, second host: a client that did not declare the receipt type
    cannot submit the request, so this app must.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        for _ in range(200):
            if app._session is not None:
                break
            await asyncio.sleep(0.01)
        await _app_with_team(app, tmp_path)
        submitted: list[str] = []
        monkeypatch.setattr(
            app,
            "_submit_prompt",
            lambda text, images=None, attachments=None, **kw: submitted.append(text),
        )

        outcome = await app.run_slash_authoritative(
            "team", "lopdev do the thing", [], consumers=None
        )

    assert outcome["data"]["type"] == "team_attached"
    assert submitted == ["do the thing"], (
        "the TUI-hosted owner must complete an undeclared client's request, "
        "exactly as the runtime host does"
    )


@pytest.mark.asyncio
async def test_the_tui_owner_defers_to_a_declaring_client(monkeypatch, tmp_path) -> None:
    """The double-submission guard on the second host.

    This is the branch with no CI coverage before this cell: a drift here runs
    the user's command twice whenever the session happens to be TUI-owned.
    """
    from local_operator.session.runtime.types import SLASH_ACTION_RECEIPTS

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        for _ in range(200):
            if app._session is not None:
                break
            await asyncio.sleep(0.01)
        await _app_with_team(app, tmp_path)
        submitted: list[str] = []
        monkeypatch.setattr(
            app,
            "_submit_prompt",
            lambda text, images=None, attachments=None, **kw: submitted.append(text),
        )

        outcome = await app.run_slash_authoritative(
            "team", "lopdev do the thing", [], consumers=list(SLASH_ACTION_RECEIPTS)
        )

    assert outcome["data"]["request"] == "do the thing"
    assert submitted == [], "a declaring client submits it itself; the owner must not"


@pytest.mark.asyncio
async def test_the_tui_owner_treats_declaring_nothing_as_undeclared(monkeypatch, tmp_path) -> None:
    """``[]`` admits here too — the rule is ``type not in declared``.

    The tempting shortcut ("complete when the field was absent") passes the
    undeclared cell and fails this one, on this host only.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        for _ in range(200):
            if app._session is not None:
                break
            await asyncio.sleep(0.01)
        await _app_with_team(app, tmp_path)
        submitted: list[str] = []
        monkeypatch.setattr(
            app,
            "_submit_prompt",
            lambda text, images=None, attachments=None, **kw: submitted.append(text),
        )

        await app.run_slash_authoritative("team", "lopdev do the thing", [], consumers=[])

    assert submitted == ["do the thing"]


# ---------------------------------------------------------------------------
# Notice COPY (design review round 1)
#
# The copy is a reviewed design surface, so these cells pin the properties the
# review bought rather than whole sentences: the subject is named, the shared
# version is not repeated, and the internal vocabulary stays out.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_stale_sessions_are_told_apart_by_name(monkeypatch, tmp_path) -> None:
    """D1: correct per-session behaviour must not RENDER as a duplicate bug.

    Two stale sessions each legitimately earn a notice — that is what the
    per-session debounce buys. With a deictic "this session" in both, the two
    paragraphs came out byte-identical, which reads as the app printing one
    warning twice and leaves ``/stop`` ambiguous about which session it acts
    on. Naming the subject is what makes two notices an inventory rather than
    a malfunction.
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

        first = _BoundViewer(
            owner_version="", session_id="sessionaaaa1", conversation_name="ingest pipeline"
        )
        app._session = first
        app._check_build_skew(reason="bind")

        second = _BoundViewer(
            owner_version="", session_id="sessionbbbb2", conversation_name="release notes"
        )
        app._session = second
        app._check_build_skew(reason="resume")
        await pilot.pause()
        notices = _notices(app)

    stale = [n for n in notices if OWNER_UNKNOWN in n]
    assert len(stale) == 2, notices
    assert stale[0] != stale[1], (
        "two stale sessions must not produce byte-identical paragraphs; that "
        "is indistinguishable from the duplicate-notice bug the re-key fixed"
    )
    assert "\u201cingest pipeline\u201d" in stale[0]
    assert "\u201crelease notes\u201d" in stale[1]


@pytest.mark.asyncio
async def test_an_unnamed_session_falls_back_to_the_deictic(monkeypatch, tmp_path) -> None:
    """A fork or a fresh session has no title yet; the notice still has to work.

    The fallback is the ONLY case where the old deictic wording survives, so
    an empty title must not render as empty quotes.
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
        unnamed = _BoundViewer(owner_version="", session_id="sessionccccc")
        app._session = unnamed
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)

    stale = [n for n in notices if OWNER_UNKNOWN in n]
    assert len(stale) == 1, notices
    assert stale[0].startswith("this session is running an older version")
    assert "\u201c\u201d" not in stale[0], "an empty title must not render as empty quotes"


@pytest.mark.asyncio
async def test_differing_versions_keep_the_two_arm_form(monkeypatch, tmp_path) -> None:
    """D3 factors out a SHARED version only; a real version change still shows both.

    Collapsing here would hide the fact the user most needs — that the version
    itself moved, not just the commit.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.46.23", source_ref="aaaaaaa1111")
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

    drift = [n for n in notices if DRIFT in n]
    assert len(drift) == 1, notices
    assert "0.46.23@aaaaaaa \u2192 0.49.0@bbbbbbb" in drift[0], drift[0]


def test_the_notices_carry_no_internal_vocabulary() -> None:
    """D2: the words we use for ourselves must not reach the user.

    "routed", "runtime"/"terminal" and "build reporting" are all real and
    load-bearing internally, but the notice never explains the runtime/terminal
    split, so to a reader those words collapse into the one thing they can see.

    Walks the AST and inspects STRING LITERALS only. A regex over the source
    was the first attempt and is unsound: it reads through the surrounding
    comments, which legitimately discuss runtimes and terminals and must keep
    being able to — it reported a failure for the word "routed" appearing in a
    code comment.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(OperatorApp._check_build_skew)))
    literals = [
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    # The docstring is prose for maintainers, not user copy.
    body = " ".join(literals[1:]) if literals else ""
    for word in ("routed", "predates build reporting", "this terminal", "resend"):
        assert word not in body, f"internal vocabulary reached the notice copy: {word!r}"
