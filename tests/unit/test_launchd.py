"""Guards for the shared LaunchAgent helpers — addressability and repair.

WHAT IS AT RISK HERE, and therefore what these tests pin:

1. **The addressability test is IDENTITY, not containment.** ``launchctl``
   addresses the calling user's live session whatever ``Path.home()`` says, so a
   test or sandbox that redirects ``HOME`` into a tmpdir must not reach the real
   LaunchAgents. The containment shortcut fails on the shape that is NOT exotic:
   a redirected home that lands *inside* the real one. One such run evicted
   ``com.local-operator.browser`` from the operator's live session (recorded in
   ``browser_bridge/install.py``).
2. **A repair must not change anything it was not asked about.** Byte content is
   compared, not existence or mtime, and a rewrite preserves the on-disk mode —
   the tunnel plist is written 0600 and must stay there.
3. **Every function is no-raise by contract.** A stale plist is decoration on a
   process listing; an upgrade that has already succeeded must never fail on it.
4. **A reload is a SEQUENCE, not a pair.** Measured on macOS (2026-09-17): a
   ``bootout`` immediately followed by a ``bootstrap`` fails with ``Bootstrap
   failed: 5: Input/output error`` 8 times out of 8, because launchd is still
   tearing the previous job down. The bootout has already succeeded, so that
   failure leaves the daemon DOWN. ``reload_job`` waits for the label to be
   released, retries the bootstrap to a deadline, and verifies the job is
   registered — and its failure carries launchd's own stderr rather than a
   summary, because that sentence is what the operator reads.

The end-to-end behaviour (a stale plist really being rewritten and the daemon
restarted on the operator's machine) is on the PR as live ``ps``/``plutil``
captures: it cannot be asserted here without touching a real launchd session.
"""

from __future__ import annotations

import os
import plistlib
from pathlib import Path

import pytest

from local_operator import launchd
from local_operator.paths import CONFIG_DIR_ENV

PLIST = "com.local-operator.mobile"


def _plist_path(home: Path) -> Path:
    return home / "Library" / "LaunchAgents" / f"{PLIST}.plist"


class TestAddressability:
    """``is_own_plist`` — the guard that keeps a sandbox off the real session."""

    def test_the_real_home_path_is_addressable(self) -> None:
        home = launchd.real_home()
        assert home is not None
        assert launchd.is_own_plist(_plist_path(home), PLIST) is True

    def test_a_redirected_home_is_refused(self, tmp_path: Path) -> None:
        assert launchd.is_own_plist(_plist_path(tmp_path), PLIST) is False

    def test_containment_would_have_passed_and_identity_does_not(self, monkeypatch) -> None:
        """A home nested inside the real one is the failure this pins.

        ``TMPDIR`` under ``$HOME`` is ordinary on macOS, so "is the plist inside
        the real home?" answers *yes* for a sandbox — and then the guard lets a
        sandbox rewrite the operator's daemon.
        """
        home = launchd.real_home()
        assert home is not None
        nested = home / "sandbox-home"
        monkeypatch.setattr(launchd, "real_home", lambda: nested)
        # The path the REAL home produces (what launchctl would actually act on)
        # is refused once the identity is taken from the redirected home.
        assert launchd.is_own_plist(_plist_path(home), PLIST) is False
        # ... and the nested home's own path is addressable, which is only ever
        # reachable when it is genuinely the passwd home.
        assert launchd.is_own_plist(_plist_path(nested), PLIST) is True

    def test_unreadable_passwd_refuses(self, monkeypatch) -> None:
        monkeypatch.setattr(launchd, "real_home", lambda: None)
        assert launchd.is_own_plist(Path("/anywhere") / PLIST, PLIST) is False


class TestConfigDirContainment:
    """``config_lives_in_real_home`` — the store-outlives-us guard."""

    def test_a_sandbox_store_is_refused(self, tmp_path: Path) -> None:
        assert launchd.config_lives_in_real_home(tmp_path / "sandbox-cfg") is False

    def test_a_store_under_the_real_home_is_accepted(self) -> None:
        home = launchd.real_home()
        assert home is not None
        assert launchd.config_lives_in_real_home(home / ".local-operator") is True


class TestArgValue:
    """Reading a plist's argv for its ARGUMENTS, never for an interpreter."""

    def test_separate_and_equals_forms(self) -> None:
        assert launchd.arg_value({"ProgramArguments": ["x", "--port", "4098"]}, "--port") == "4098"
        assert launchd.arg_value({"ProgramArguments": ["x", "--port=5000"]}, "--port") == "5000"

    @pytest.mark.parametrize(
        "plist",
        [
            None,
            {},
            {"ProgramArguments": "not-a-list"},
            {"ProgramArguments": ["x", "--other", "1"]},
            {"ProgramArguments": ["x", "--port"]},
            {"ProgramArguments": ["x", "--port", 4098]},
        ],
    )
    def test_absent_or_unusable_is_none(self, plist) -> None:
        assert launchd.arg_value(plist, "--port") is None

    def test_int_arg_keeps_the_configured_port(self) -> None:
        """A repair must not move a daemon off a non-default port."""
        assert launchd.int_arg({"ProgramArguments": ["x", "--port=5000"]}, "--port", 4098) == 5000

    def test_int_arg_falls_back_rather_than_guessing(self) -> None:
        assert launchd.int_arg({"ProgramArguments": ["x", "--port", "abc"]}, "--port", 4098) == 4098
        assert launchd.int_arg(None, "--port", 4098) == 4098


class TestConfigDirFromPlist:
    def test_reads_the_recorded_store(self, tmp_path: Path) -> None:
        plist: dict[str, object] = {"EnvironmentVariables": {CONFIG_DIR_ENV: str(tmp_path)}}
        assert launchd.config_dir_from_plist(plist) == tmp_path

    @pytest.mark.parametrize(
        "plist", [None, {}, {"EnvironmentVariables": "nope"}, {"EnvironmentVariables": {}}]
    )
    def test_absent_is_none(self, plist) -> None:
        assert launchd.config_dir_from_plist(plist) is None


class TestRewriteIfStale:
    """Content, not existence — and nothing else touched."""

    def test_absent_is_not_installed(self, tmp_path: Path) -> None:
        outcome = launchd.rewrite_if_stale(
            name="mobile", path=tmp_path / "missing.plist", rendered={"Label": "x"}
        )
        assert outcome == launchd.PlistRefresh(name="mobile", kind="not-installed")
        assert outcome.summary() == ""
        assert outcome.warning() == ""

    def test_current_is_left_untouched(self, tmp_path: Path) -> None:
        rendered = {"Label": "com.local-operator.mobile", "ProgramArguments": ["a", "-m", "b"]}
        path = tmp_path / PLIST
        path.write_bytes(plistlib.dumps(rendered))
        before = path.stat().st_mtime_ns
        outcome = launchd.rewrite_if_stale(name="mobile", path=path, rendered=dict(rendered))
        assert outcome.kind == "current"
        assert path.stat().st_mtime_ns == before, "an up-to-date plist was rewritten anyway"

    def test_stale_is_rewritten_and_says_so(self, tmp_path: Path) -> None:
        path = tmp_path / PLIST
        path.write_bytes(plistlib.dumps({"Label": "com.local-operator.mobile", "Legacy": True}))
        rendered = {
            "Label": "com.local-operator.mobile",
            "Program": "/prefix/bin/Local Operator",
            "ProgramArguments": ["Local Operator [mobile daemon] port=4098", "-m", "svc"],
        }
        outcome = launchd.rewrite_if_stale(name="mobile", path=path, rendered=rendered)
        assert outcome.kind == "repaired"
        assert plistlib.loads(path.read_bytes()) == rendered
        assert "mobile" in outcome.summary()

    def test_a_rewrite_preserves_the_mode(self, tmp_path: Path) -> None:
        """The tunnel plist is 0600; a repair must not widen it."""
        path = tmp_path / PLIST
        path.write_bytes(plistlib.dumps({"Label": "old"}))
        path.chmod(0o600)
        launchd.rewrite_if_stale(name="tunnel", path=path, rendered={"Label": "new"})
        assert path.stat().st_mode & 0o777 == 0o600

    def test_a_corrupt_plist_is_repaired_rather_than_crashing(self, tmp_path: Path) -> None:
        path = tmp_path / PLIST
        path.write_bytes(b"not a plist at all")
        outcome = launchd.rewrite_if_stale(name="mobile", path=path, rendered={"Label": "new"})
        assert outcome.kind == "repaired"

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the file mode")
    def test_an_unwritable_plist_fails_without_raising(self, tmp_path: Path) -> None:
        path = tmp_path / f"{PLIST}.plist"
        path.write_bytes(plistlib.dumps({"Label": "old"}))
        path.chmod(0o400)
        try:
            outcome = launchd.rewrite_if_stale(name="mobile", path=path, rendered={"Label": "new"})
        finally:
            path.chmod(0o600)
        assert outcome.kind == "failed"
        assert outcome.warning().startswith("warning: ")

    def test_the_recorded_install_prefix_reads_both_plist_shapes(self) -> None:
        """The test the repair's identity guard runs on.

        Prefix equality rather than path equality is the point: a stale plist
        recording ``<prefix>/bin/python3`` and the branded shape recording
        ``<prefix>/bin/Local Operator`` are the SAME install, so a repair that
        upgrades one into the other must not read as "another installation".
        """
        branded = {"Program": "/opt/tool/bin/Local Operator"}
        legacy = {
            "ProgramArguments": ["/opt/tool/bin/python3", "-m", "local_operator.wakes.supervisor"]
        }
        assert launchd.recorded_install_prefix(branded) == Path("/opt/tool")
        assert launchd.recorded_install_prefix(legacy) == Path("/opt/tool")

    def test_the_recorded_install_prefix_gives_up_rather_than_guessing(self) -> None:
        """``None`` is "cannot tell", which callers must treat as no objection.

        The branded shape carries the image in ``Program``, so a LABEL in
        ``ProgramArguments[0]`` is not a path and must not be read as one — a
        guess here would refuse a legitimate repair on a machine whose plist
        this code simply does not understand.
        """
        assert launchd.recorded_install_prefix({}) is None
        assert launchd.recorded_install_prefix(None) is None
        assert launchd.recorded_install_prefix({"ProgramArguments": []}) is None
        assert (
            launchd.recorded_install_prefix({"ProgramArguments": ["Local Operator [wakes]"]})
            is None
        )


#: launchd's own words for the race this helper exists to absorb, taken
#: verbatim from the measured runs on 2026-09-17.
BOOTSTRAP_EIO = "Bootstrap failed: 5: Input/output error"


class _Result:
    """One ``launchctl`` answer, in the shape the installers' helpers return."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeLaunchd:
    """A launchd stand-in that models the LABEL, not just a return code.

    ``print`` resolves only while the label is loaded, ``bootout`` unloads it,
    and ``bootstrap`` loads it back (unless asked to fail, or to fail a bounded
    number of times first so the retry path is the thing under test). Scripted
    at the verb level because that is the level the defect lives at: the old
    fake — every call returns 0 — is exactly what let an un-sequenced
    bootout+bootstrap pair look correct.
    """

    def __init__(
        self,
        *,
        loaded: bool = False,
        bootout_rc: int = 0,
        bootout_stderr: str = "",
        bootstrap_failures: int = 0,
        bootstrap_always_fails: bool = False,
        register_on_bootstrap: bool = True,
        bytes_stderr: bool = False,
        release_after_polls: int = 0,
    ) -> None:
        self.loaded = loaded
        self.calls: list[tuple[str, ...]] = []
        self._bootout_rc = bootout_rc
        self._bootout_stderr = bootout_stderr
        self._failures_left = bootstrap_failures
        self._always_fails = bootstrap_always_fails
        self._register = register_on_bootstrap
        self._bytes = bytes_stderr
        #: How many ``print`` probes after the bootout still resolve, which is
        #: the teardown the reload is supposed to wait out.
        self._polls_left = release_after_polls

    def __call__(self, *args: str) -> _Result:
        self.calls.append(args)
        verb = args[0]
        if verb == "print":
            if self._polls_left:
                self._polls_left -= 1
                return _Result(0, stdout="state = not running")
            return _Result(0 if self.loaded else 1, stdout="state = running" if self.loaded else "")
        if verb == "bootout":
            self.loaded = False
            return _Result(self._bootout_rc, stderr=self._bootout_stderr)
        if verb == "bootstrap":
            if self._always_fails or self._failures_left:
                self._failures_left = max(0, self._failures_left - 1)
                stderr: object = BOOTSTRAP_EIO.encode() if self._bytes else BOOTSTRAP_EIO
                return _Result(5, stderr=stderr)  # type: ignore[arg-type]
            if self._register:
                self.loaded = True
            return _Result(0)
        return _Result(0)

    @property
    def verbs(self) -> list[str]:
        return [call[0] for call in self.calls]

    def attempts(self, verb: str) -> int:
        return self.verbs.count(verb)


class _FrozenClock:
    """``launchd``'s clock, advanced only by the sleeps the reload asks for.

    The release wait and the bootstrap budget are REAL deadlines. Against a
    launchctl stub that never recovers, an honest implementation spends them in
    wall clock — seconds per test — so the tests move the clock instead and
    assert on the retry count and its spacing.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class TestReloadJob:
    """``bootout`` → wait for the label → ``bootstrap`` → verify.

    Every test here passes a fake runner, so no test in this file can reach a
    real launchd session (see the module docstring's addressability note).
    """

    @pytest.fixture
    def clock(self, monkeypatch: pytest.MonkeyPatch) -> _FrozenClock:
        clock = _FrozenClock()
        monkeypatch.setattr(launchd, "_monotonic", clock.monotonic)
        monkeypatch.setattr(launchd, "_sleep", clock.sleep)
        return clock

    def _path(self, label: str = PLIST) -> Path:
        """The plist path the REAL passwd home owns, which is what the guard checks."""
        home = launchd.real_home()
        assert home is not None
        return home / "Library" / "LaunchAgents" / f"{label}.plist"

    @property
    def _target(self) -> str:
        return f"gui/{os.getuid()}/{PLIST}"

    def test_first_attempt_success_does_exactly_one_bootstrap(self, clock: _FrozenClock) -> None:
        fake = _FakeLaunchd()
        result = launchd.reload_job(label=PLIST, path=self._path(), runner=fake)

        assert result.ok is True
        assert result.outcome == "reloaded"
        assert result.attempts == 1
        assert fake.attempts("bootstrap") == 1, fake.calls
        # An absent job needs no release wait: the first probe is immediate, so
        # the happy path never sleeps at all.
        assert clock.sleeps == []
        assert fake.calls[0] == ("bootout", self._target), fake.calls

    def test_a_transient_input_output_error_is_retried(self, clock: _FrozenClock) -> None:
        """The measured race itself: the FIRST bootstrap fails with EIO."""
        fake = _FakeLaunchd(bootstrap_failures=2)
        result = launchd.reload_job(label=PLIST, path=self._path(), runner=fake)

        assert result.ok is True, result
        assert result.attempts == 3
        assert fake.attempts("bootstrap") == 3
        # Backoff, not a hot loop: the retries are spaced and each is longer
        # than the last until the cap.
        assert clock.sleeps[:2] == [0.1, 0.2], clock.sleeps

    def test_it_waits_for_the_label_to_be_released_before_bootstrapping(
        self, clock: _FrozenClock
    ) -> None:
        """``print`` still resolving means launchd has not let go of the label."""
        fake = _FakeLaunchd(loaded=True, release_after_polls=2)

        result = launchd.reload_job(label=PLIST, path=self._path(), runner=fake)

        assert result.ok is True, result
        # Two probes that still resolved, then the one that did not — and NOT
        # ONE bootstrap issued while launchd still knew the label.
        first_bootstrap = fake.verbs.index("bootstrap")
        assert fake.verbs[:first_bootstrap] == ["bootout", "print", "print", "print"], fake.verbs
        assert clock.sleeps[:2] == [0.05, 0.05], clock.sleeps
        assert result.attempts == 1, result

    def test_the_wait_and_the_retries_are_bounded(self, clock: _FrozenClock) -> None:
        """A launchd that never recovers must not hold the caller forever.

        Two separate bounds: the release wait gives up on the label, and the
        bootstrap budget gives up on the load. Both are finite, and the reload
        still reports launchd's own reason.
        """
        fake = _FakeLaunchd(loaded=True, bootstrap_always_fails=True)
        result = launchd.reload_job(label=PLIST, path=self._path(), runner=fake)

        assert result.outcome == "failed"
        assert 2 <= result.attempts <= 20, result.attempts
        assert fake.attempts("bootstrap") == result.attempts
        # The release wait was bounded rather than unbounded...
        release_polls = fake.verbs.count("print")
        assert release_polls <= 60, release_polls
        # ... and every sleep it took was one of the two named intervals.
        assert set(clock.sleeps) <= {0.05, 0.1, 0.2, 0.4, 0.8, 1.0}, clock.sleeps

    def test_a_hard_failure_carries_launchd_stderr_and_the_recovery(
        self, clock: _FrozenClock
    ) -> None:
        """The sentence the operator reads must be launchd's, plus the fix."""
        fake = _FakeLaunchd(bootstrap_always_fails=True)
        result = launchd.reload_job(label=PLIST, path=self._path(), runner=fake)

        assert result.outcome == "failed"
        assert BOOTSTRAP_EIO in result.detail, result.detail
        failure = result.as_refresh_failure(
            name="mobile", path=self._path(), recovery="lop mobile install"
        )
        assert failure.kind == "failed"
        assert "launchctl could not load it" in failure.detail
        assert "STOPPED" in failure.detail
        assert "run `lop mobile install` to reinstall it" in failure.detail

    def test_bootout_tolerates_an_absent_job(self, clock: _FrozenClock) -> None:
        """A first install bootouts a job that does not exist; not a failure."""
        fake = _FakeLaunchd(
            bootout_rc=3, bootout_stderr=f'Could not find service "{PLIST}" in domain for user'
        )
        result = launchd.reload_job(label=PLIST, path=self._path(), runner=fake)

        assert result.ok is True, result
        assert result.detail == ""

    def test_a_success_that_leaves_no_registered_job_is_a_failure(
        self, clock: _FrozenClock
    ) -> None:
        """A zero exit is launchd's claim, and the caller is about to repeat it."""
        fake = _FakeLaunchd(register_on_bootstrap=False)
        result = launchd.reload_job(label=PLIST, path=self._path(), runner=fake)

        assert result.outcome == "failed"
        assert "is not registered" in result.detail, result.detail

    def test_stderr_captured_as_bytes_is_still_reported(self, clock: _FrozenClock) -> None:
        """The tunnel's helper decodes bytes; a caller may hand over either."""
        fake = _FakeLaunchd(bootstrap_always_fails=True, bytes_stderr=True)
        result = launchd.reload_job(label=PLIST, path=self._path(), runner=fake)

        assert result.outcome == "failed"
        assert BOOTSTRAP_EIO in result.detail, result.detail

    def test_a_sandbox_never_reaches_launchd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard is a PRECONDITION here, so a caller cannot skip it.

        ``launchctl`` addresses the real user's session whatever ``HOME`` says,
        so a redirected home must be refused before anything is booted out —
        the incident this codebase has already paid for once.
        """
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/tmp/sandbox-home")))

        def explode(*args: object, **kwargs: object) -> object:
            raise AssertionError("a sandbox reload must not reach launchd")

        result = launchd.reload_job(
            label=PLIST,
            path=Path("/tmp/sandbox-home/Library/LaunchAgents") / f"{PLIST}.plist",
            runner=explode,
        )

        assert result.outcome == "not-addressable"
        assert result.ok is False
        # And it must NOT be reported as a stopped daemon: nothing was booted
        # out, so `reload_failure`'s sentence would be a lie about the machine.
        failure = result.as_refresh_failure(
            name="mobile", path=self._path(), recovery="lop mobile install"
        )
        assert failure.kind == "not-addressable"
        assert failure.warning() == ""
