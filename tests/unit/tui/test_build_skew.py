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

from local_operator.session.protocol import RuntimeLocality
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
# Unique to C\u2032. NOT the bare arrow: ``_build_change`` puts one in the DRIFT
# notice too, so the four NEGATIVE assertions below would have fired on a drift
# row and quietly stopped meaning "no owner notice" (review round 2, R2-3).
OWNER_SKEW = "is running "
OWNER_UNKNOWN = "running an older version than this window"
MOVES_OVER = "will switch to the new version when it is next idle"


class _BoundViewer(FakeSession):
    """A follower facade that has already bound to a runtime.

    Not owning the runtime, plus a resolved ``runtime_version``, is what a real
    ``AttachedSession`` looks like after ``_dial``; ``is_cold`` False is what
    makes the owner comparison meaningful, since a cold viewer has not dialled
    anything and its empty stamp would otherwise read as a prehistoric
    runtime.
    """

    # Runtime role (SessionProtocol): this fake emulates an ATTACHED
    # viewer: it owns no loop and learns outcomes over the wire.
    owns_runtime = False
    outcome_is_synchronous = False
    runtime_locality: RuntimeLocality = "this-machine"
    is_cold = False

    def __init__(
        self,
        runtime_version: str = "",
        runtime_source_ref: str = "",
        session_id: str = "",
        conversation_name: str = "",
        idle: bool = False,
        refresh_answer: str = "retiring",
    ) -> None:
        super().__init__()
        self.runtime_version = runtime_version
        self.runtime_source_ref = runtime_source_ref
        # ``idle`` is what a real ``AttachedSession.runtime_idle`` reads off the
        # canonical snapshot. The default is BUSY so every pre-existing cell
        # keeps exercising the notice path; the refresh cells opt in.
        self._skew_idle = idle
        # What the RUNTIME answers ``refresh_if_idle`` with. ``retiring`` is
        # the happy path; a ``kept: \u2026`` answer (busy again, or an old runtime
        # that does not know the op) is the case R1-1 painted nothing for.
        # ``raise`` makes the ask itself fail. Anything else is returned AS
        # GIVEN \u2014 including a non-string, which is the shape that crashed the
        # worker in R2-1 and which a duck-typed host may legitimately send.
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

    def runtime_idle(self) -> bool:
        return self._skew_idle

    async def request_refresh(self):  # deliberately unannotated: see below
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
        viewer = _BoundViewer(runtime_version="0.46.23", idle=False)
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
        viewer = _BoundViewer(runtime_version="0.46.23", idle=True)
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
        viewer = _BoundViewer(runtime_version="0.46.23", idle=True, refresh_answer=answer)
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
        viewer = _BoundViewer(runtime_version="", idle=True, refresh_answer="raise")
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
@pytest.mark.parametrize("answer", [None, object(), b"retiring", 0])
async def test_a_non_string_answer_cannot_take_the_window_down(
    monkeypatch, tmp_path, answer
) -> None:
    """R2-1: a build diagnostic must never close the window it is diagnosing.

    ``request_refresh`` is reached through a duck-typed ``getattr`` probe, so
    the answer's SHAPE is whatever an older runtime facade, an embedder or a
    reduced double sends \u2014 exactly the population this notice exists for.
    Calling ``.strip()`` on it raised inside a Textual worker, which runs with
    ``exit_on_error=True``: the session ended with ``WorkerFailed`` instead of
    painting anything. Every other probe on this seam degrades; so does this.

    ``b"retiring"`` is the trap case: it *looks* like the silence answer but is
    not a ``str``, and it must be treated as "the runtime stays" rather than
    matched.
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
        app._session = _BoundViewer(runtime_version="0.46.23", idle=True, refresh_answer=answer)
        app._check_build_skew(reason="bind")
        await pilot.pause()
        # Raises WorkerFailed on the unfixed tree, before any assertion below.
        await app.workers.wait_for_complete()
        await pilot.pause()
        notices = _notices(app)
        # Sampled INSIDE the pilot: outside it the app has been shut down by
        # the context manager and this is False for every run.
        alive = app.is_running

    assert alive, "the app must survive an answer it did not expect"
    skew = [n for n in notices if MOVES_OVER in n]
    assert len(skew) == 1, f"an unreadable answer is a runtime that stays: {notices}"
    assert not any("/stop" in n for n in notices), notices


@pytest.mark.asyncio
async def test_a_same_version_rebuild_names_only_the_refs(monkeypatch, tmp_path) -> None:
    """R2-2 / D3: C\u2032 and notice A must name one fact the same way.

    A ``lop-update`` rebuild from ``main`` with no version bump is this host's
    headline drift, and both sides then carry the same version. Hand-rolling
    the arrow rendered it ``0.49.0@aaaaaaa \u2192 0.49.0@bbbbbbb`` \u2014 the version
    twice, with the seven characters that actually differ buried \u2014 while the
    drift notice one block higher already collapsed it. ``_build_change`` is
    that formatter; this pins C\u2032 to it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        stamp = BuildStamp(version="0.49.0", source_ref="bbbbbbb2222")
        app._loaded_build = stamp
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: stamp)
        app._session = _BoundViewer(runtime_version="0.49.0", runtime_source_ref="aaaaaaa1111")
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)

    skew = [n for n in notices if MOVES_OVER in n]
    assert len(skew) == 1, notices
    assert "0.49.0, aaaaaaa \u2192 bbbbbbb" in skew[0], skew[0]
    assert "0.49.0@aaaaaaa" not in skew[0], "the shared version must not be stated twice"


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
                runtime_version="0.49.8",
                runtime_source_ref="46a4e9b1234567",
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
        app._session = _BoundViewer(runtime_version="")
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

        first = _BoundViewer(runtime_version="", session_id="sessionaaaa1")
        app._session = first
        app._check_build_skew(reason="bind")
        app._check_build_skew(reason="bind-again")  # same session: still once

        second = _BoundViewer(runtime_version="", session_id="sessionbbbb2")
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
        first = _BoundViewer(runtime_version="0.49.0", session_id="sessionaaaa1")
        app._session = first
        app._check_build_skew(reason="adopt")

        second = _BoundViewer(runtime_version="0.49.0", session_id="sessionbbbb2")
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
        app._session = _BoundViewer(runtime_version="0.49.0", runtime_source_ref="abc1234")
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

    Its ``runtime_version`` is empty for the trivial reason that no runtime
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
        cold = _BoundViewer(runtime_version="")
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
        app._session = _BoundViewer(runtime_version="0.1.0")
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
# ``ServingSessionHandle``, so before these cells the app-side copy of the
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
            runtime_version="", session_id="sessionaaaa1", conversation_name="ingest pipeline"
        )
        app._session = first
        app._check_build_skew(reason="bind")

        second = _BoundViewer(
            runtime_version="", session_id="sessionbbbb2", conversation_name="release notes"
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
        unnamed = _BoundViewer(runtime_version="", session_id="sessionccccc")
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


# --- Which side is stale: the direction the notice must not get backwards ----
#
# The defect these pin was reported from a live session sitting IDLE, which
# painted `"<name>" is running 0.51.30@d7f12d3 -> 0.51.29@2412b1d - it will
# switch to the new version when it is next idle.` Both halves are wrong: the
# arrow reads as a downgrade when the runtime is the NEWER side, and there was
# nothing to say at all. `scripts/build_skew_direction_repro.py` prints the
# sentence itself; these cells pin the behaviour behind it.
#
# The shape is routine here rather than exotic. A window keeps the build it
# imported at launch forever; `lop-update` runs several times a day; a runtime
# spawned after it resolves `sys.executable` fresh. An OLD window driving a NEW
# runtime is therefore the steady state, and the old code, seeing only
# `owner != loaded`, assumed the runtime was always the stale one.


@pytest.mark.asyncio
async def test_a_runtime_newer_than_this_window_is_completely_silent(monkeypatch, tmp_path) -> None:
    """The reported case: window behind disk, runtime ON disk, idle.

    Nothing may be asked of the runtime (it is already current, and its own
    check answers ``kept`` because it compares against the same disk), and no
    owner notice may be painted. Disk drift has ALREADY told the user the
    install moved and named ``/reload``; a second line about the same fact
    pointing the other way is the bug.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.29", source_ref="2412b1daf")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        on_disk = BuildStamp(version="0.51.30", source_ref="d7f12d3a7")
        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: on_disk)
        viewer = _BoundViewer(
            runtime_version="0.51.30",
            runtime_source_ref="d7f12d3a7",
            idle=True,
            refresh_answer="kept: build on disk matches (or has not settled)",
        )
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        notices = _notices(app)

    assert viewer.refresh_requests == 0, "a runtime that matches disk is already current"
    assert not [n for n in notices if OWNER_SKEW in n or MOVES_OVER in n], notices
    # The window IS behind, and that fact keeps its notice - with the arrow the
    # right way round and the remedy that actually applies to a window.
    drift = [n for n in notices if DRIFT in n]
    assert len(drift) == 1, notices
    assert "0.51.29@2412b1d \u2192 0.51.30@d7f12d3" in drift[0], drift[0]
    assert "/reload" in drift[0]


@pytest.mark.asyncio
async def test_a_runtime_newer_than_this_window_is_silent_while_busy(monkeypatch, tmp_path) -> None:
    """Same shape, BUSY runtime: still nothing to say.

    The busy branch paints C\u2032 without asking anything, so it is a second,
    independent way into the reversed notice - and the complaint is not limited
    to idle sessions. Whether the runtime is working says nothing about which
    side is stale.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.29", source_ref="2412b1daf")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        on_disk = BuildStamp(version="0.51.30", source_ref="d7f12d3a7")
        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: on_disk)
        viewer = _BoundViewer(runtime_version="0.51.30", runtime_source_ref="d7f12d3a7", idle=False)
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)

    assert not [n for n in notices if OWNER_SKEW in n or MOVES_OVER in n], notices
    assert viewer.refresh_requests == 0


@pytest.mark.asyncio
async def test_a_same_version_rebuild_is_directional_both_ways(monkeypatch, tmp_path) -> None:
    """Ref-only drift, in both directions - the case that forces the disk compare.

    Both builds say ``0.51.30`` and differ only in the recorded commit, so NO
    ordering of version numbers can say which is newer. Matching disk is the
    only fact that does, and it resolves both directions exactly.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    import local_operator.update as update_mod

    on_disk = BuildStamp(version="0.51.30", source_ref="bbbbbbb2222")

    # (a) runtime is the rebuild that is on disk: silent.
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.30", source_ref="aaaaaaa1111")
        app._skew_notice_shown.clear()
        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: on_disk)
        viewer = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="bbbbbbb2222", idle=False
        )
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        current_side = _notices(app)

    assert not [n for n in current_side if OWNER_SKEW in n or MOVES_OVER in n], current_side

    # (b) the window is the one on disk and the runtime is the older rebuild:
    #     genuinely stale, and it earns the notice.
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = on_disk
        app._skew_notice_shown.clear()
        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: on_disk)
        viewer = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="aaaaaaa1111", idle=False
        )
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        stale_side = _notices(app)

    skew = [n for n in stale_side if MOVES_OVER in n]
    assert len(skew) == 1, stale_side
    assert "aaaaaaa \u2192 bbbbbbb" in skew[0], skew[0]


@pytest.mark.asyncio
async def test_an_unreadable_disk_falls_back_to_version_order(monkeypatch, tmp_path) -> None:
    """No disk reference: a strictly newer runtime is still recognised.

    ``installed_build()`` raising skips check A too, so the discriminator has
    nothing to compare against. Version ordering is the fallback - weaker (it
    cannot see a ref-only rebuild) but decisive when the versions differ.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.29")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        def _boom(*_a, **_k):
            raise OSError("no install metadata")

        monkeypatch.setattr(update_mod, "installed_build", _boom)
        viewer = _BoundViewer(runtime_version="0.51.30", idle=True)
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        notices = _notices(app)

    assert notices == [], notices
    assert viewer.refresh_requests == 0


@pytest.mark.asyncio
async def test_an_inconclusive_order_keeps_todays_notice(monkeypatch, tmp_path) -> None:
    """Unreadable disk AND an undecidable pair: keep the advisory line.

    Same version, differing refs, with no disk to appeal to - nothing here can
    say which side is behind. Silence would regress R1-1 (a resume onto a
    pre-refresh runtime explained by nothing, with no later seam to repair it),
    so an undiagnosable skew is still worth exactly one line.

    The line it gets is the UNDIRECTED one, since this is exactly the shape
    where no term can rank the pair (QA round 1, Q-1). The cell's claim has
    always been non-silence rather than a particular wording, so it asserts a
    notice paints and that the wording makes no claim it cannot support.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.30", source_ref="aaaaaaa1111")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        def _boom(*_a, **_k):
            raise OSError("no install metadata")

        monkeypatch.setattr(update_mod, "installed_build", _boom)
        viewer = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="bbbbbbb2222", idle=False
        )
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)

    owner = [n for n in notices if "is running" in n]
    assert len(owner) == 1, notices
    assert "\u2192" not in owner[0], f"nothing here can rank the pair: {owner[0]}"
    # The remedy is the reason the line is worth painting at all; a mutation
    # that strips it must not stay green (QA round 2, Q2-1).
    assert "switch" in owner[0], owner[0]


@pytest.mark.asyncio
async def test_an_unparseable_version_is_not_guessed_at(monkeypatch, tmp_path) -> None:
    """Fallback ordering answers only when BOTH sides parse.

    A local ``0.51.30rc1`` is not evidence of anything; treating it as newer
    would silence a skew on a guess. ``parse_version`` returns ``None`` and the
    advisory line stands.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.29")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        def _boom(*_a, **_k):
            raise OSError("no install metadata")

        monkeypatch.setattr(update_mod, "installed_build", _boom)
        app._session = _BoundViewer(runtime_version="0.51.30rc1", idle=False)
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)

    # parse_version cannot rank the pair, so neither predicate claims a
    # direction and the notice is the undirected one — a full-label, arrow-free
    # line. Non-silence is the claim this cell has always made.
    owner = [n for n in notices if "is running" in n]
    assert len(owner) == 1, notices
    assert "\u2192" not in owner[0], owner[0]
    assert "switch" in owner[0], owner[0]


# --- The completed self-refresh, said once ----------------------------------


@pytest.mark.asyncio
async def test_a_completed_refresh_names_the_version_it_moved_to(monkeypatch, tmp_path) -> None:
    """The operator asked for this: an update that happens on its own says so.

    The line is painted only after the successor BINDS, so it reports a
    completed fact. The pair is captured across the re-engage: the retiring
    stamp exists only before the re-bind overwrites it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._skew_notice_shown.clear()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
        # The runtime that is leaving.
        app._session = _BoundViewer(
            runtime_version="0.51.29",
            runtime_source_ref="2412b1daf",
            session_id="s1",
            conversation_name="Investigating suspicious pwned notification source",
        )
        app._on_runtime_refreshed()
        # The successor, resolved from the build now on disk, is the SAME
        # session rebound — the refresh keeps the id, which is exactly what the
        # belt keys on.
        app._session = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="d7f12d3a7", session_id="s1"
        )
        app._announce_refresh_completed()
        await pilot.pause()
        notices = _notices(app)
        tokens = [block._token for block in app.query(NoticeBlock)]

        # Consumed once: a second engage tail must not repeat it.
        app._announce_refresh_completed()
        await pilot.pause()
        again = _notices(app)

    assert len(notices) == 1, notices
    assert "0.51.29@2412b1d \u2192 0.51.30@d7f12d3" in notices[0], notices[0]
    assert "Investigating suspicious pwned" in notices[0], "the line names its session"
    assert tokens == ["muted"], "a note: nothing is wrong and nothing is asked"
    assert again == notices, "the pending stamp is cleared on read"


@pytest.mark.asyncio
async def test_an_unchanged_build_after_a_refresh_says_nothing(monkeypatch, tmp_path) -> None:
    """A re-engage that did not change the build is not an update.

    A runtime that came back on the same build (a restart that was not an
    update, a successor resolving the same install) changed nothing the user
    could act on, and announcing it would turn ordinary re-engagement into a
    stream of notices.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._skew_notice_shown.clear()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
        app._session = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="d7f12d3a7", session_id="s1"
        )
        app._on_runtime_refreshed()
        app._session = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="d7f12d3a7", session_id="s1"
        )
        app._announce_refresh_completed()
        await pilot.pause()
        notices = _notices(app)

    assert notices == [], notices


@pytest.mark.asyncio
async def test_an_ordinary_engage_never_announces_a_refresh(monkeypatch, tmp_path) -> None:
    """No pending stamp, no line. The engage tail calls this unconditionally."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._skew_notice_shown.clear()
        app._session = _BoundViewer(runtime_version="0.51.30", runtime_source_ref="d7f12d3a7")
        app._announce_refresh_completed()
        await pilot.pause()
        notices = _notices(app)

    assert notices == [], notices


@pytest.mark.asyncio
async def test_a_runtime_too_old_to_name_its_build_is_not_half_announced(
    monkeypatch, tmp_path
) -> None:
    """No "from" side means no sentence.

    "updated to X" with nothing it came from reads as an update out of
    nowhere, which is less informative than saying nothing.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._skew_notice_shown.clear()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
        app._session = _BoundViewer(runtime_version="", session_id="s1")
        app._on_runtime_refreshed()
        assert app._refreshed_from is None
        app._session = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="d7f12d3a7", session_id="s1"
        )
        app._announce_refresh_completed()
        await pilot.pause()
        notices = _notices(app)

    assert notices == [], notices


# --- Three builds alive at once: the quadrant round 1 found still broken -----
#
# `_owner_matches_disk` answered only `owner == on_disk`, which is an equality
# and therefore only ever describes a TWO-build host. With three builds alive
# the runtime can be newer than the window while being unequal to disk, and the
# old code fell through and painted the reversed arrow at it. That is not a
# corner: a runtime lives across updates precisely because it refuses to retire
# while busy, so on a host that runs `lop-update` several times a day the triple
# is the steady state (review round 1 R1-1 / QA round 1 Q-1, both reproduced
# against the previous head using this machine's own live build population).


@pytest.mark.asyncio
async def test_a_newer_runtime_is_silent_when_disk_has_moved_again(monkeypatch, tmp_path) -> None:
    """Window 0.51.29, runtime 0.51.30, disk 0.52.0 — the live triple.

    The runtime is newer than the window and older than disk. It is not this
    window's problem to report: the window is the stale side, the drift notice
    already says so, and asking a runtime that is ahead of us to retire is the
    original defect wearing a third version number.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.29", source_ref="2412b1daf")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(
            update_mod,
            "installed_build",
            lambda *_a, **_k: BuildStamp(version="0.52.0", source_ref="5db5f6d65"),
        )
        viewer = _BoundViewer(
            runtime_version="0.51.30",
            runtime_source_ref="d7f12d3a7",
            idle=True,
            refresh_answer="kept: build on disk matches (or has not settled)",
        )
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        notices = _notices(app)

    assert viewer.refresh_requests == 0, "a runtime ahead of this window is never asked to retire"
    assert not [n for n in notices if OWNER_SKEW in n or MOVES_OVER in n], notices
    # The window is behind DISK, and that is the fact worth painting.
    drift = [n for n in notices if DRIFT in n]
    assert len(drift) == 1, notices
    assert "0.52.0@5db5f6d" in drift[0] and "/reload" in drift[0]


@pytest.mark.asyncio
async def test_a_newer_runtime_stays_silent_while_busy_in_the_triple(monkeypatch, tmp_path) -> None:
    """The same triple on the BUSY branch, which paints without asking first."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.29", source_ref="2412b1daf")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(
            update_mod,
            "installed_build",
            lambda *_a, **_k: BuildStamp(version="0.52.0", source_ref="5db5f6d65"),
        )
        viewer = _BoundViewer(runtime_version="0.51.30", runtime_source_ref="d7f12d3a7", idle=False)
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)

    assert not [n for n in notices if OWNER_SKEW in n or MOVES_OVER in n], notices
    assert viewer.refresh_requests == 0


@pytest.mark.asyncio
async def test_a_long_lived_window_does_not_start_speaking_when_disk_moves_again(
    monkeypatch, tmp_path
) -> None:
    """The second route into the triple, which looks like a different bug.

    A window correctly silenced about a runtime at t0 must not start reporting
    that SAME runtime at t1 merely because `lop-update` ran again. Nothing
    about the pair changed; only a third build appeared.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.29", source_ref="2412b1daf")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        viewer = _BoundViewer(
            runtime_version="0.51.30",
            runtime_source_ref="d7f12d3a7",
            idle=True,
            refresh_answer="kept: build on disk matches (or has not settled)",
        )
        app._session = viewer
        # t0: disk == the runtime's build. Silent, and pinned elsewhere too.
        monkeypatch.setattr(
            update_mod,
            "installed_build",
            lambda *_a, **_k: BuildStamp(version="0.51.30", source_ref="d7f12d3a7"),
        )
        app._check_build_skew(reason="t0")
        await pilot.pause()
        await app.workers.wait_for_complete()
        # t1: a second lop-update. The runtime has not moved.
        monkeypatch.setattr(
            update_mod,
            "installed_build",
            lambda *_a, **_k: BuildStamp(version="0.52.0", source_ref="5db5f6d65"),
        )
        app._check_build_skew(reason="t1")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        notices = _notices(app)

    assert not [n for n in notices if OWNER_SKEW in n or MOVES_OVER in n], notices
    assert viewer.refresh_requests == 0
    assert len([n for n in notices if DRIFT in n]) == 2, "each install move is its own fact"


@pytest.mark.asyncio
async def test_an_older_runtime_still_speaks_when_disk_has_moved_again(
    monkeypatch, tmp_path
) -> None:
    """The over-silencing guard: adding the ordering term must not mute a
    genuinely older runtime just because disk is a third build.

    Without this cell the R1-1 fix could be "return True more often" and every
    other cell would still pass.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.29", source_ref="2412b1daf")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(
            update_mod,
            "installed_build",
            lambda *_a, **_k: BuildStamp(version="0.52.0", source_ref="5db5f6d65"),
        )
        viewer = _BoundViewer(runtime_version="0.51.28", runtime_source_ref="8c2015f11", idle=False)
        app._session = viewer
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)

    skew = [n for n in notices if MOVES_OVER in n]
    assert len(skew) == 1, notices
    assert "0.51.28@8c2015f \u2192 0.51.29@2412b1d" in skew[0], skew[0]


@pytest.mark.asyncio
async def test_three_refs_at_one_version_claim_no_direction(monkeypatch, tmp_path) -> None:
    """Window, runtime and disk are three rebuilds of ONE version.

    Nothing can rank them: the versions are equal so ordering is silent, and
    all three refs differ so the disk equality cannot fire either. The notice
    still paints — an undiagnosable skew is worth one line, and silence would
    regress the predecessor's R1-1 — but it must NOT use the arrow, because an
    arrow asserts an order that no term here derived (QA round 1, Q-1 all-refs
    sub-case).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.30", source_ref="aaaaaaa11")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(
            update_mod,
            "installed_build",
            lambda *_a, **_k: BuildStamp(version="0.51.30", source_ref="ccccccc33"),
        )
        app._session = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="bbbbbbb22", idle=False
        )
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)

    owner = [n for n in notices if "is running" in n]
    assert len(owner) == 1, notices
    assert "\u2192" not in owner[0], f"no arrow may claim a direction here: {owner[0]}"
    # Collapsed: the versions are equal by construction here, so the pair must
    # be stated once and the "vs" pivot must show which stamp is whose (design
    # round 2, D2-2/D2-3).
    assert "0.51.30, bbbbbbb vs this window\u2019s aaaaaaa" in owner[0], owner[0]
    assert "0.51.30@bbbbbbb" not in owner[0], "the shared version must not be stated twice"
    assert "switch" in owner[0], owner[0]


@pytest.mark.asyncio
async def test_a_ref_only_pair_still_uses_the_arrow_when_disk_ranks_it(
    monkeypatch, tmp_path
) -> None:
    """Direction IS knowable when disk equals one of the two, so the arrow stays.

    Guards the undirected branch from swallowing the same-version case the disk
    comparison exists to resolve — that reasoning is the docstring's core claim
    and must not be undone by the fix for the all-refs shape.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        on_disk = BuildStamp(version="0.51.30", source_ref="bbbbbbb22")
        app._loaded_build = on_disk
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: on_disk)
        app._session = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="aaaaaaa11", idle=False
        )
        app._check_build_skew(reason="bind")
        await pilot.pause()
        notices = _notices(app)

    skew = [n for n in notices if MOVES_OVER in n]
    assert len(skew) == 1, notices
    assert "aaaaaaa \u2192 bbbbbbb" in skew[0], skew[0]


# --- The fourth exit from an engage, and a title that must not cost one ------


@pytest.mark.asyncio
async def test_a_cancelled_engage_cannot_announce_the_previous_session(
    monkeypatch, tmp_path
) -> None:
    """`/resume` and `/new` cancel the engage worker before its tail runs.

    The three clears inside `_start_runtime_engage` enumerate the exits IT can
    see; cancellation happens from outside it, so the pending stamp survived
    and the next session's ordinary engage consumed it — naming a session the
    user had already left, against an unrelated runtime's build, with the pair
    running backwards under the word "updated" (R1-2 / Q-2).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._skew_notice_shown.clear()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
        app._session = _BoundViewer(
            runtime_version="0.51.29",
            runtime_source_ref="2412b1daf",
            session_id="sess-a",
            conversation_name="Session A",
        )
        app._on_runtime_refreshed()
        assert app._refreshed_from is not None, "the retire must arm the announcement"

        # The real teardown `/resume` and `/new` run.
        app._cancel_runtime_engage()
        assert app._refreshed_from is None, "the swap must disarm it"

        # Session B binds normally. Nothing about session A may be said.
        app._session = _BoundViewer(
            runtime_version="0.51.19",
            runtime_source_ref="dc6bec0aa",
            session_id="sess-b",
            conversation_name="Session B",
        )
        app._announce_refresh_completed()
        await pilot.pause()
        notices = _notices(app)

    assert notices == [], notices


@pytest.mark.asyncio
async def test_a_sidebar_switch_cannot_announce_the_previous_session(monkeypatch, tmp_path) -> None:
    """The belt, not the route. The sidebar switch swaps ``self._session``
    through the ``"session"`` worker group — NOT ``"warm-engage"`` — so the
    round-1 fix in ``_cancel_runtime_engage`` never ran for it, and the
    in-flight engage worker's own tail fired the announcement against the
    session switched TO. The guard is a comparison at the paint point, which
    closes the CLASS rather than enumerating swap routes (review round 2,
    R2-1). Driven through the real methods, not a hand-run sequence.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._skew_notice_shown.clear()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
        app._session = _BoundViewer(
            runtime_version="0.51.29",
            runtime_source_ref="2412b1daf",
            session_id="sess-a",
            conversation_name="Session A",
        )
        app._on_runtime_refreshed()
        assert app._refreshed_from is not None

        # The sidebar swap: the pending slot SURVIVES this, which is the
        # route round 1 missed. Session B then binds and its engage tail runs.
        app._session = _BoundViewer(
            runtime_version="0.51.19",
            runtime_source_ref="dc6bec0aa",
            session_id="sess-b",
            conversation_name="Session B",
        )
        app._announce_refresh_completed()
        await pilot.pause()
        notices = _notices(app)

    assert notices == [], f"session A's stamp answered against session B: {notices}"


@pytest.mark.asyncio
async def test_a_refresh_rebind_to_the_same_session_still_announces(monkeypatch, tmp_path) -> None:
    """The belt must not over-fire: a refresh keeps the session id, so the
    successor binding the SAME session paints the note. Without this the
    R2-1 guard could be `return` and every silent cell would still pass.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._skew_notice_shown.clear()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
        app._session = _BoundViewer(
            runtime_version="0.51.29",
            runtime_source_ref="2412b1daf",
            session_id="s1",
            conversation_name="Session A",
        )
        app._on_runtime_refreshed()
        app._session = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="d7f12d3a7", session_id="s1"
        )
        app._announce_refresh_completed()
        await pilot.pause()
        notices = _notices(app)

    assert len(notices) == 1, notices
    assert "0.51.29@2412b1d \u2192 0.51.30@d7f12d3" in notices[0], notices[0]


@pytest.mark.asyncio
async def test_one_build_pair_is_announced_once_across_a_disk_move(monkeypatch, tmp_path) -> None:
    """R2-2: the two skew wordings share ONE debounce key.

    A same-version pair can paint directed C\u2032 at adopt (disk == the window)
    and, after `lop-update` moves disk to a third ref, qualify as undirected
    on the next check. Two kinds meant ONE unchanged build pair painted BOTH —
    the second apparently withdrawing the direction the first asserted. The
    pair is one fact in one session and is announced once, whichever wording
    won the race.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._loaded_build = BuildStamp(version="0.51.30", source_ref="aaaaaaa11")
        app._skew_notice_shown.clear()
        import local_operator.update as update_mod

        # First check: disk == the window, so the pair is rankable and the
        # directed copy paints.
        monkeypatch.setattr(
            update_mod,
            "installed_build",
            lambda *_a, **_k: BuildStamp(version="0.51.30", source_ref="aaaaaaa11"),
        )
        app._session = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="bbbbbbb22", session_id="s1", idle=False
        )
        app._check_build_skew(reason="adopt")
        await pilot.pause()
        first = _notices(app)
        # Disk moves to a third ref; the SAME pair now qualifies as undirected.
        monkeypatch.setattr(
            update_mod,
            "installed_build",
            lambda *_a, **_k: BuildStamp(version="0.51.30", source_ref="ccccccc33"),
        )
        app._check_build_skew(reason="engage")
        await pilot.pause()
        second = _notices(app)

    owner = [n for n in second if "is running" in n]
    assert len(owner) == 1, f"one pair must paint once: {second}"
    # The DRIFT notice re-paints on the disk move — that is its own fact, not
    # the pair being repeated. The claim under test is narrower than
    # `second == first`: the owner notice that already painted at adopt does
    # not fire again, in EITHER wording, when the direction becomes unknowable.
    owner_first = [n for n in first if "is running" in n]
    assert owner == owner_first, (owner_first, owner)
    assert "\u2192" in owner[0], "the directed form won the race and is the one shown"


@pytest.mark.asyncio
async def test_a_title_that_raises_does_not_cost_the_re_engage(monkeypatch, tmp_path) -> None:
    """A pre-sync `conversation_name` raises; the eager re-engage must survive.

    `AttachedSession.conversation_name` reads `frontend_state`, which raises
    until the first sync completes — and the refresh callback reads the title
    BEFORE re-engaging. `_go_cold` swallows the exception, so the cost was a
    silently lost re-engage: the viewer stays cold until the next keystroke,
    which is the guarantee this feature exists to provide (R1-3).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    class _Raising(_BoundViewer):
        @property
        def conversation_name(self) -> str:  # type: ignore[override]
            raise RuntimeError("frontend state has not synchronized")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        reasons: list[str] = []
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: reasons.append(reason))
        app._warm_engage_started = True
        app._session = _Raising(runtime_version="0.51.29", runtime_source_ref="2412b1daf")
        app._on_runtime_refreshed()
        await pilot.pause()

    assert reasons == ["refresh"], "the re-engage must happen even with no readable title"
    assert app._warm_engage_started is False
    assert app._refreshed_from is not None
    assert app._refreshed_from[1] == "this session", "an unreadable title degrades to the deictic"


@pytest.mark.asyncio
async def test_the_refresh_note_keeps_its_version_pair_on_one_row(monkeypatch, tmp_path) -> None:
    """D1: the pair must not wrap, INCLUDING for an unnamed session.

    The parenthetical form split it at 6 of 19 name lengths at both 80 and 100
    columns — length 0, the unnamed default, among them. The reported 51-char
    name happened to be a non-splitting length, which is why the original still
    looked clean. Asserted through the real block's wrapped rows rather than
    the raw string, since the defect is invisible to `_text`.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for width in (80, 100):
        for name in ("", "Debugging the reaper", "x" * 72):
            app = OperatorApp(lambda: _factory(FakeSession()))
            async with app.run_test(size=(width, 24)) as pilot:
                await pilot.pause()
                app._skew_notice_shown.clear()
                monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
                app._session = _BoundViewer(
                    runtime_version="0.51.29",
                    runtime_source_ref="2412b1daf",
                    session_id="s1",
                    conversation_name=name,
                )
                app._on_runtime_refreshed()
                app._session = _BoundViewer(
                    runtime_version="0.51.30", runtime_source_ref="d7f12d3a7", session_id="s1"
                )
                app._announce_refresh_completed()
                await pilot.pause()
                blocks = list(app.query(NoticeBlock))
                assert len(blocks) == 1, (width, name)
                rows = _wrapped_rows(blocks[0])

            pair_rows = [r for r in rows if "0.51.29@2412b1d" in r or "0.51.30@d7f12d3" in r]
            assert len(pair_rows) == 1, f"the pair wrapped apart at {width} for {name!r}: {rows}"


@pytest.mark.asyncio
async def test_the_refresh_note_does_not_restate_the_version_twice(monkeypatch, tmp_path) -> None:
    """D3: on a same-version rebuild the copy must not claim novelty in prose.

    "a newer version" was the only thing asserting a change beside two stamps
    that read alike; `_build_change`'s collapsed form puts the seven characters
    that actually differ where the reader looks.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._skew_notice_shown.clear()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
        app._session = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="aaaaaaa11", session_id="s1"
        )
        app._on_runtime_refreshed()
        app._session = _BoundViewer(
            runtime_version="0.51.30", runtime_source_ref="bbbbbbb22", session_id="s1"
        )
        app._announce_refresh_completed()
        await pilot.pause()
        notices = _notices(app)

    assert len(notices) == 1, notices
    assert "0.51.30, aaaaaaa \u2192 bbbbbbb" in notices[0], notices[0]
    assert "newer version" not in notices[0], notices[0]
