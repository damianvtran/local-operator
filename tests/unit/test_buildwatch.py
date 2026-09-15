"""The build-watch rule is defined ONCE, and both watchers read that one copy.

The runtime and the ``serve`` daemon must agree about when a build on disk has
replaced the one they loaded: the timings and the changed-and-settled comparison
live in :mod:`local_operator.buildwatch`, the runtime re-exports them under the
private names it always published, and the daemon imports the same functions
rather than restating them. A second copy of the settle window is a torn-tree
race in whichever copy a later edit forgets, so these tests pin the identity of
the shared objects rather than their values.

``tests/unit/session/runtime/test_process_refresh.py`` still pins the runtime's
behaviour through the re-exports; this file guards the seam they now share.
"""

from __future__ import annotations

import pytest

from local_operator import buildwatch
from local_operator import update as update_mod
from local_operator.server import retire as serve_retire
from local_operator.session.runtime import process as process_mod
from local_operator.update import BuildStamp

OLD = BuildStamp(version="0.54.30", source_ref="1111111")
NEW = BuildStamp(version="0.54.31", source_ref="2222222")


class TestOneDefinition:
    """Both watchers resolve the same objects, not lookalikes."""

    def test_the_runtime_reexports_the_shared_timings(self) -> None:
        assert process_mod.BUILD_CHECK_S is buildwatch.BUILD_CHECK_S
        assert process_mod.BUILD_SETTLE_S is buildwatch.BUILD_SETTLE_S
        assert process_mod.BUILD_STAGGER_S is buildwatch.BUILD_STAGGER_S

    def test_the_runtime_reexports_the_shared_rule_and_readers(self) -> None:
        assert process_mod._build_changed is buildwatch.build_changed
        assert process_mod._build_pair is buildwatch.build_pair
        assert process_mod._build_prefix is buildwatch.build_prefix
        assert process_mod._build_settle_seconds is buildwatch.build_settle_seconds
        assert process_mod._build_stagger_seconds is buildwatch.build_stagger_seconds

    def test_the_daemon_carries_no_copy_of_its_own(self) -> None:
        """A local constant in ``retire`` would be the drift this module exists
        to prevent, and it would shadow silently — nothing would fail."""
        for name in ("BUILD_CHECK_S", "BUILD_SETTLE_S", "BUILD_STAGGER_S"):
            assert name not in vars(serve_retire), f"serve/retire must read the shared {name}"

    def test_the_daemon_watches_with_the_shared_rule(self) -> None:
        assert serve_retire.buildwatch is buildwatch


class TestEnvOverrides:
    """``LOP_BUILD_SETTLE_S`` / ``LOP_BUILD_STAGGER_S`` are the e2e stage's only
    way to shorten the watch, and a malformed value must fall back to the
    constant rather than disarm the guard it names."""

    def test_unset_means_the_constant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOP_BUILD_SETTLE_S", raising=False)
        monkeypatch.delenv("LOP_BUILD_STAGGER_S", raising=False)
        assert buildwatch.build_settle_seconds() == buildwatch.BUILD_SETTLE_S == 10.0
        assert buildwatch.build_stagger_seconds() == buildwatch.BUILD_STAGGER_S == 20.0

    @pytest.mark.parametrize("value", ["0.2", "3"])
    def test_a_positive_override_is_honoured(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("LOP_BUILD_SETTLE_S", value)
        monkeypatch.setenv("LOP_BUILD_STAGGER_S", value)
        assert buildwatch.build_settle_seconds() == pytest.approx(float(value))
        assert buildwatch.build_stagger_seconds() == pytest.approx(float(value))

    @pytest.mark.parametrize("value", ["0", "-1", "nope", ""])
    def test_an_unusable_override_keeps_the_constant(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("LOP_BUILD_SETTLE_S", value)
        monkeypatch.setenv("LOP_BUILD_STAGGER_S", value)
        assert (
            buildwatch.build_settle_seconds() == buildwatch.BUILD_SETTLE_S
        ), "zero is the torn-tree race; the settle must refuse it"
        assert buildwatch.build_stagger_seconds() == buildwatch.BUILD_STAGGER_S


class TestTheWatch:
    """The shared rule's own semantics, at the seam both watchers call."""

    @pytest.fixture(autouse=True)
    def disk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.state: dict[str, object] = {"build": OLD, "age": 999.0}
        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: self.state["build"])
        monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: self.state["age"])
        monkeypatch.delenv("LOP_BUILD_SETTLE_S", raising=False)
        monkeypatch.delenv("LOP_BUILD_PREFIX", raising=False)

    def test_no_baseline_is_never_a_change(self) -> None:
        assert buildwatch.build_changed(None) is None

    def test_the_same_build_is_no_change(self) -> None:
        assert buildwatch.build_changed(OLD) is None

    def test_a_moved_build_inside_the_settle_waits(self) -> None:
        self.state["build"] = NEW
        self.state["age"] = buildwatch.BUILD_SETTLE_S / 2
        assert buildwatch.build_changed(OLD) is None

    def test_a_moved_build_past_the_settle_is_a_change(self) -> None:
        self.state["build"] = NEW
        self.state["age"] = buildwatch.BUILD_SETTLE_S + 1
        assert buildwatch.build_changed(OLD) == NEW

    def test_a_version_only_stamp_is_not_a_move_when_the_version_stands_still(self) -> None:
        """QA round 2, OBS-1: a stamp nobody could read is an absence of evidence.

        ``update.source_ref`` answers ``""`` for a marker that is missing,
        unreadable, empty, truncated or not a commit, so the stamp on disk for an
        aged ``chmod 000`` marker is ``0.54.30`` against a boot stamp of
        ``0.54.30@1111111`` — different, and different only by the ref that could
        not be read. QA measured the consequence on a real daemon (announced,
        latched, exited, record removed, onto a build it could not read). The ref is
        the primary key here precisely because two builds share one version string.
        """
        self.state["build"] = BuildStamp(version=OLD.version)
        self.state["age"] = buildwatch.BUILD_SETTLE_S + 1
        assert buildwatch.build_changed(OLD) is None
        assert buildwatch.proves_a_move(OLD, BuildStamp(version=OLD.version)) is False

    def test_a_version_move_with_no_ref_is_still_a_move(self) -> None:
        """The other direction, so the OBS-1 guard cannot close over every PyPI install.

        A wheel upgrade writes ``pypi <version>``: no commit to record, and the
        version is the thing that moved. Refusing version-only stamps outright
        would turn this guard into "the feature never fires on a PyPI install".
        """
        moved = BuildStamp(version="0.54.31")
        self.state["build"] = moved
        self.state["age"] = buildwatch.BUILD_SETTLE_S + 1
        assert buildwatch.build_changed(OLD) == moved
        assert buildwatch.proves_a_move(OLD, moved) is True

    def test_a_stamp_with_nothing_in_it_is_not_a_build_to_leave_for(self) -> None:
        """Both halves unreadable — no dist-info version and no marker — is not a move."""
        self.state["build"] = BuildStamp(version="")
        self.state["age"] = buildwatch.BUILD_SETTLE_S + 1
        assert buildwatch.build_changed(OLD) is None

    def test_handover_build_is_the_move_without_the_settle(self) -> None:
        """The announced phase's own read: a move does not settle a second time.

        ``build_changed`` waits out ``BUILD_SETTLE_S`` because a detection may not
        act on a half-written install. A handover that has ALREADY been announced
        is not a detection any more: it is the question "is this still true?", and
        answering it through the settle would delay the withdrawal — and the
        re-announcement — by up to a settle window for no gain.
        """
        self.state["build"] = NEW
        self.state["age"] = 0.0
        assert buildwatch.build_changed(OLD) is None
        assert buildwatch.handover_build(OLD) == NEW

    def test_handover_build_withdraws_on_a_reverted_install(self) -> None:
        """MINOR-2 at the rule's seam: back on ``boot`` is not a handover."""
        self.state["build"] = NEW
        assert buildwatch.handover_build(OLD) == NEW
        self.state["build"] = OLD
        assert buildwatch.handover_build(OLD) is None

    def test_handover_build_withdraws_on_an_unreadable_install(self) -> None:
        self.state["build"] = BuildStamp(version=OLD.version)
        assert buildwatch.handover_build(OLD) is None

    def test_handover_build_needs_a_baseline(self) -> None:
        self.state["build"] = NEW
        assert buildwatch.handover_build(None) is None

    def test_the_override_shortens_what_counts_as_settled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOP_BUILD_SETTLE_S", "0.5")
        self.state["build"] = NEW
        self.state["age"] = 1.0
        assert buildwatch.build_changed(OLD) == NEW

    def test_the_prefix_override_names_what_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``LOP_BUILD_PREFIX`` is how the e2e stage points a real watcher (the
        runtime AND the daemon) at a fake install root."""
        seen: list[object] = []

        def _build(prefix: object = None, **_k: object) -> BuildStamp:
            seen.append(prefix)
            return OLD

        monkeypatch.setattr(update_mod, "installed_build", _build)
        monkeypatch.setenv("LOP_BUILD_PREFIX", "/tmp/fake-install")
        assert buildwatch.boot_build() == OLD
        assert seen == ["/tmp/fake-install"]

    def test_an_unreadable_boot_stamp_disables_the_watch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_a: object, **_k: object) -> BuildStamp:
            raise RuntimeError("no dist-info")

        monkeypatch.setattr(update_mod, "installed_build", _boom)
        assert buildwatch.boot_build() is None

    def test_an_unreadable_settle_is_no_change_rather_than_a_dead_watcher(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The marker-age read must not take the caller's task down with it.

        Both callers of this function are background watchers — the daemon's
        retirement poll and the runtime's refresh check — and a raise here used
        to leave that task dead with nothing logged. For the runtime that meant
        silently keeping an old build; for the daemon, with the latch on the
        other side of it, a process that never rolled forward at all (review
        round 1, MINOR-3). "Not settled" is the safe direction: this process
        stays, and the next check asks again.
        """

        def _boom(*_a: object, **_k: object) -> float:
            raise OSError("the marker is unreadable")

        monkeypatch.setattr(update_mod, "build_marker_age_s", _boom)
        self.state["build"] = NEW
        assert buildwatch.build_changed(OLD) is None

    def test_the_pair_names_both_builds(self) -> None:
        assert buildwatch.build_pair(OLD, NEW) == f" ({OLD.label()} → {NEW.label()})"
        assert buildwatch.build_pair(None, NEW) == ""


class TestTheWireSentences:
    """The rotation's answers are a CONTRACT between two processes.

    The runtime picks the sentence and ``control.refresh_session`` routes on it,
    so a reword at either end used to fail SILENTLY: the matcher missed and fell
    through to its generic ``kept`` branch, reporting "was not moved: …" for a
    state that has a precise, actionable diagnosis (review round 2, NIT-1).
    """

    def test_the_retired_hedge_still_routes(self) -> None:
        """A runtime started before #1141 answers this until build skew retires it.

        ``lop refresh``'s own docstring says its first run is ``lop-update``, so
        the fleet that exists at update time is BY CONSTRUCTION made of runtimes
        executing the previous build's code — the one build that cannot know the
        two precise sentences. The matcher therefore keeps accepting the retired
        hedge, and it does so by matching ``KEPT_MATCHES`` as a PREFIX of it,
        which is a cross-version contract rather than a tidiness accident.
        """
        assert buildwatch.KEPT_MATCHES_OR_UNSETTLED.startswith(buildwatch.KEPT_MATCHES)
        assert "has not settled" in buildwatch.KEPT_MATCHES_OR_UNSETTLED

    def test_the_two_live_answers_are_distinct_and_neither_is_the_hedge(self) -> None:
        assert buildwatch.KEPT_MATCHES != buildwatch.KEPT_UNSETTLED
        for live in (buildwatch.KEPT_MATCHES, buildwatch.KEPT_UNSETTLED):
            assert live.startswith("kept: "), live
            assert live != buildwatch.KEPT_MATCHES_OR_UNSETTLED, live


class TestMovedAndUnsettled:
    """The settle question asked about a stamp somebody ELSE published.

    ``pending_build`` asks it about the stamp this process booted from;
    ``lop refresh`` holds a DISCOVERY RECORD instead, and that is the case this
    reader exists for (design round 2 D1 / UX round 2 U6 / QA round 2 O1).
    """

    @pytest.fixture(autouse=True)
    def disk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.state: dict[str, object] = {"build": NEW, "age": 0.0}
        monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: self.state["build"])
        monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: self.state["age"])
        monkeypatch.delenv("LOP_BUILD_PREFIX", raising=False)

    def test_a_record_with_no_stamp_asks_nothing(self) -> None:
        """With nothing to compare, the marker cannot be said to have moved PAST it.

        The runtime's own "matches" then stands rather than being second-guessed,
        which is also what keeps an old record from being reported as an
        unsettled install on the strength of a comparison it never took part in.
        """
        assert buildwatch.moved_and_unsettled("", "") is False

    def test_a_moved_marker_inside_the_settle_is_unsettled(self) -> None:
        assert buildwatch.moved_and_unsettled(OLD.version, OLD.source_ref) is True

    def test_a_settled_move_is_not_unsettled(self) -> None:
        self.state["age"] = buildwatch.BUILD_SETTLE_S + 1
        assert buildwatch.moved_and_unsettled(OLD.version, OLD.source_ref) is False

    def test_a_matching_install_is_not_unsettled(self) -> None:
        self.state["build"] = OLD
        assert buildwatch.moved_and_unsettled(OLD.version, OLD.source_ref) is False

    def test_a_raising_probe_is_no_evidence_rather_than_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This is read while composing a receipt for a person, mid-scan."""

        def _boom(*_a: object, **_k: object) -> BuildStamp:
            raise RuntimeError("no dist-info")

        monkeypatch.setattr(update_mod, "installed_build", _boom)
        assert buildwatch.moved_and_unsettled(OLD.version, OLD.source_ref) is False
