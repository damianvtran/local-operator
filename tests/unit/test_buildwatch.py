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

    def test_the_pair_names_both_builds(self) -> None:
        assert buildwatch.build_pair(OLD, NEW) == f" ({OLD.label()} → {NEW.label()})"
        assert buildwatch.build_pair(None, NEW) == ""
