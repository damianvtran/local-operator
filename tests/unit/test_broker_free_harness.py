"""Synthetic controls for the opt-in broker-free contract; no daemon or child.

Each control owns a separate state so intentionally denied attempts do not poison
an enclosing broker-free pytest run. Artifact fixtures remove only files they
created; no status response is ever permission to signal an operating-system PID.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pluggy
import pytest

import tests.conftest as harness
from local_operator.secrets import client, protocol


@pytest.fixture
def state(tmp_path, monkeypatch):
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    # tempfile caches its effective directory independently of TMPDIR. A narrow
    # synthetic owner must derive fallbacks inside the same private root too.
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    result = harness._BrokerFree((tmp_path,))
    try:
        yield result
    finally:
        result.patch.undo()


def test_clean_candidate_is_audited_without_status_or_signal(state, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("audit reached a status/PID/signal/deletion path")

    monkeypatch.setattr(client, "broker_status", forbidden)
    monkeypatch.setattr(client, "is_running", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)
    state.remember(tmp_path / "config")
    state.audit()
    assert state.candidates == {tmp_path / "config"}
    assert state.violations == []


def test_spawn_denial_is_retained_even_when_caught(state, tmp_path):
    state.install()
    with pytest.raises(RuntimeError, match="denied _spawn_broker"):
        client._spawn_broker(tmp_path / "config")
    assert "broker-free denied _spawn_broker" in state.violations
    assert tmp_path / "config" in state.candidates


def test_popen_refusal_preserves_class_contract_without_launch(state):
    state.install()
    assert issubclass(subprocess.Popen, object)
    assert subprocess.Popen[bytes] is not None
    with pytest.raises(RuntimeError, match="denied subprocess.Popen"):
        subprocess.Popen(["must-not-run"])
    assert "broker-free denied subprocess.Popen" in state.violations


def test_foreign_candidate_rejected_before_filesystem_access(state, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("foreign candidate was inspected")

    calls = []

    def inspected(*args, **kwargs):
        calls.append(args)
        return forbidden(*args, **kwargs)

    # End the global double before the real call-wrapper/teardown audit; those
    # audits must see the actual filesystem, not this intentional refusal.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", inspected)
        state.remember(tmp_path.parent / "foreign")
        assert calls == []
        assert any("foreign candidate" in problem for problem in state.violations)
        assert state.candidates == set()


def test_symlink_escape_is_not_followed(state, tmp_path):
    escape = tmp_path / "escape"
    escape.symlink_to(tmp_path.parent, target_is_directory=True)
    state.remember(escape / "foreign")
    assert any("symlink" in problem for problem in state.violations)
    assert state.candidates == set()
    escape.unlink()


def test_inspection_error_is_not_an_empty_success(state, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise PermissionError("synthetic inspection refusal")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", broken)
        state.remember(tmp_path / "config")
        assert any("inspection failed" in problem for problem in state.violations)


def test_discovery_limit_is_visible(state, tmp_path, monkeypatch):
    monkeypatch.setattr(state, "LIMIT", 1)
    state.discover((tmp_path,))
    assert any("limit reached" in problem for problem in state.violations)


@pytest.mark.parametrize("artifact_kind", ["socket", "lock"])
def test_unexpected_artifact_never_reaches_status_or_signal(
    state, tmp_path, monkeypatch, artifact_kind
):
    state.install()
    base = tmp_path / "config"
    artifact = protocol.socket_path(base) if artifact_kind == "socket" else protocol.lock_path(base)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.touch()

    def forbidden(*args, **kwargs):
        # No PID cases are processed: this sentinel proves the status/signal
        # seam is never entered, regardless of what a broker might claim.
        pytest.fail("audit reached a forbidden status/PID/signal seam")

    outer = harness._BROKER_FREE
    monkeypatch.setattr(harness, "_BROKER_FREE", state)
    monkeypatch.setattr(client, "broker_status", forbidden)
    monkeypatch.setattr(client, "is_running", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)
    try:
        assert harness._stop_brokers_in([base]) == set()
        assert any("unexpected broker artifact" in problem for problem in state.violations)
        assert artifact.exists(), "audit must preserve evidence, not pretend cleanup"
    finally:
        artifact.unlink()
        monkeypatch.setattr(harness, "_BROKER_FREE", outer)


def test_parent_symlink_swap_is_rejected_before_artifact_inspection(state, tmp_path):
    state.install()
    base = tmp_path / "config"
    artifact = protocol.socket_path(base)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    # Rename the owned parent, then replace it by a symlink. Audit must reject
    # the parent before lstat could reach a foreign leaf through that symlink.
    parent = artifact.parent
    saved = parent.with_name(parent.name + "-saved")
    parent.rename(saved)
    parent.symlink_to(saved, target_is_directory=True)
    try:
        state.audit()
        assert any("symlink" in problem for problem in state.violations)
    finally:
        parent.unlink()
        saved.rename(parent)


def test_client_alias_retains_deleted_config_and_fallback(state, tmp_path):
    state.install()
    base = tmp_path / ("deep" * 35) / "config"
    base.mkdir(parents=True)
    artifact = client.socket_path(base)
    assert artifact == protocol.socket_path(base)
    assert not artifact.is_relative_to(base), "control must exercise fallback layout"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.touch()
    base.rmdir()
    try:
        state.audit()
        assert base in state.candidates
        assert artifact in state.artifacts
        assert any("unexpected broker artifact" in problem for problem in state.violations)
    finally:
        artifact.unlink()


def test_terminal_audit_fails_caught_denial_without_replacing_prior_failure(
    state, tmp_path, monkeypatch
):
    outer = harness._BROKER_FREE
    monkeypatch.setattr(harness, "_BROKER_FREE", state)
    monkeypatch.setattr(harness, "_fail_session_on_runtime_root_escapes", lambda session: None)
    try:
        state.fail("synthetic caught denial")
        for initial in (pytest.ExitCode.OK, pytest.ExitCode.INTERRUPTED):
            session = SimpleNamespace(exitstatus=initial)
            harness.pytest_sessionfinish(cast(pytest.Session, session))
            expected = pytest.ExitCode.TESTS_FAILED if initial == pytest.ExitCode.OK else initial
            assert session.exitstatus == expected
        assert state.violations == ["synthetic caught denial"]
    finally:
        monkeypatch.setattr(harness, "_BROKER_FREE", outer)


@pytest.mark.parametrize("entrypoint", ["call", "teardown", "session"])
def test_each_cleanup_entrypoint_audits_before_legacy_action(
    state, tmp_path, monkeypatch, entrypoint
):
    state.install()
    base = tmp_path / "config"
    artifact = protocol.lock_path(base)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.touch()
    outer = harness._BROKER_FREE
    monkeypatch.setattr(harness, "_BROKER_FREE", state)
    monkeypatch.setattr(harness, "_fail_session_on_runtime_root_escapes", lambda session: None)

    def forbidden(*args, **kwargs):
        pytest.fail("broker-free cleanup entered forgiving legacy discovery or signalling")

    for name in ("_temp_roots", "_secret_config_dirs", "_session_sweep_candidates"):
        monkeypatch.setattr(harness, name, forbidden)
    monkeypatch.setattr(client, "broker_status", forbidden)
    monkeypatch.setattr(client, "is_running", forbidden)
    monkeypatch.setattr(os, "kill", forbidden)
    item = SimpleNamespace(funcargs={"isolate_environment": tmp_path})
    try:
        if entrypoint == "call":
            hook = harness.pytest_runtest_call(cast(pytest.Item, item))
            next(hook)
            with pytest.raises(StopIteration):
                next(hook)
        elif entrypoint == "teardown":
            fixture = harness.stop_secret_brokers_started_by_this_test.__wrapped__(item, tmp_path)
            next(fixture)
            with pytest.raises(StopIteration):
                next(fixture)
        else:
            session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
            harness.pytest_sessionfinish(cast(pytest.Session, session))
            assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
        assert any("unexpected broker artifact" in problem for problem in state.violations)
        assert artifact.exists()
    finally:
        artifact.unlink()
        monkeypatch.setattr(harness, "_BROKER_FREE", outer)


def test_late_finalizer_deleted_config_surviving_fallback_fails_terminal_audit(
    state, tmp_path, monkeypatch
):
    state.install()
    outer = harness._BROKER_FREE
    monkeypatch.setattr(harness, "_BROKER_FREE", state)
    monkeypatch.setattr(harness, "_fail_session_on_runtime_root_escapes", lambda session: None)
    artifact = None
    try:
        session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
        # Simulate successful call + test teardown before a module/session
        # finalizer derives a rendezvous and removes its config directory.
        item = SimpleNamespace(funcargs={"isolate_environment": tmp_path})
        hook = harness.pytest_runtest_call(cast(pytest.Item, item))
        next(hook)
        with pytest.raises(StopIteration):
            next(hook)
        fixture = harness.stop_secret_brokers_started_by_this_test.__wrapped__(item, tmp_path)
        next(fixture)
        with pytest.raises(StopIteration):
            next(fixture)
        assert state.violations == []
        base = tmp_path / ("late" * 35) / "config"
        base.mkdir(parents=True)
        artifact = client.lock_path(base)
        assert not artifact.is_relative_to(base)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.touch()
        base.rmdir()
        harness.pytest_sessionfinish(cast(pytest.Session, session))
        assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
        assert artifact in state.artifacts
        assert artifact.exists()
    finally:
        if artifact is not None:
            artifact.unlink(missing_ok=True)
        monkeypatch.setattr(harness, "_BROKER_FREE", outer)


@pytest.mark.parametrize("initial", [pytest.ExitCode.OK, pytest.ExitCode.INTERRUPTED])
def test_caught_denial_after_sessionfinish_fails_at_guard_removal(
    state, tmp_path, monkeypatch, initial
):
    state.install()
    outer = harness._BROKER_FREE
    monkeypatch.setattr(harness, "_BROKER_FREE", state)
    monkeypatch.setattr(harness, "_fail_session_on_runtime_root_escapes", lambda session: None)
    try:
        session = SimpleNamespace(exitstatus=initial)
        harness.pytest_sessionfinish(cast(pytest.Session, session))
        assert session.exitstatus == initial
        with pytest.raises(RuntimeError, match="denied _spawn_broker"):
            client._spawn_broker(tmp_path / "late-config")
        harness.pytest_unconfigure(cast(pytest.Config, SimpleNamespace()))
        expected = pytest.ExitCode.TESTS_FAILED if initial == pytest.ExitCode.OK else initial
        assert session.exitstatus == expected
        assert harness._BROKER_FREE is None
        assert "broker-free denied _spawn_broker" in state.violations
    finally:
        monkeypatch.setattr(harness, "_BROKER_FREE", outer)


@pytest.mark.parametrize("workers", [None, "auto", 1])
def test_configure_requires_explicit_serial_mode(monkeypatch, workers):
    config = SimpleNamespace(
        getoption=lambda name, default=None: {"broker_free": True, "numprocesses": workers}.get(
            name, default
        )
    )
    monkeypatch.setattr(harness, "arm_process_desktop_gates", lambda: None)
    monkeypatch.setattr(harness, "scrub_import_time_environment", lambda: None)
    with pytest.raises(pytest.UsageError, match="explicit -n0"):
        harness.pytest_broker_free_prepare(cast(pytest.Config, config))


@pytest.mark.parametrize("invalid", ["missing-temp", "foreign-config"])
def test_configure_rejects_incomplete_or_unowned_roots(tmp_path, monkeypatch, invalid):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "home" / "config"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "home" / "temporary"))
    if invalid == "missing-temp":
        monkeypatch.delenv("TMPDIR")
    else:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path.parent / "foreign"))
    config = SimpleNamespace(
        getoption=lambda name, default=None: {
            "broker_free": True,
            "numprocesses": 0,
            "basetemp": str(tmp_path / "pytest"),
        }.get(name, default)
    )
    monkeypatch.setattr(harness, "arm_process_desktop_gates", lambda: None)
    monkeypatch.setattr(harness, "scrub_import_time_environment", lambda: None)
    with pytest.raises(pytest.UsageError, match="isolated|foreign"):
        harness.pytest_broker_free_prepare(cast(pytest.Config, config))


@pytest.mark.parametrize("registration_order", ["boundary-first", "normal-first"])
@pytest.mark.parametrize("scenario", ["pass", "caught-broker", "identity-refusal", "disabled"])
def test_real_pluggy_configure_orders_identity_before_subprocess_sealing(
    tmp_path, monkeypatch, request, registration_order, scenario
):
    """Use historic hook dispatch, not hand-ordered calls to the phase helpers.

    The real root identity refusal helper stays in the normal hook. Its actual
    child probe is replaced by an in-process sentinel, so these controls neither
    launch children nor pretend to establish real interpreter-source identity.
    """
    from _pytest import hookspec

    home = tmp_path / "home"
    temporary = home / "temporary"
    temporary.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home / "config"))
    monkeypatch.setenv("TMPDIR", str(temporary))
    monkeypatch.setattr(harness, "arm_process_desktop_gates", lambda: None)
    monkeypatch.setattr(harness, "scrub_import_time_environment", lambda: None)
    monkeypatch.setattr(harness, "_fail_session_on_runtime_root_escapes", lambda session: None)
    outer = harness._BROKER_FREE
    monkeypatch.setattr(harness, "_BROKER_FREE", None)
    original_popen = subprocess.Popen
    original_spawn = client._spawn_broker
    enabled = scenario != "disabled"
    events = []
    config = SimpleNamespace(
        rootpath=request.config.rootpath,
        getoption=lambda name, default=None: {
            "broker_free": enabled,
            "numprocesses": 0,
            "basetemp": str(tmp_path / "pytest"),
        }.get(name, default),
    )
    root = request.config.pluginmanager.getplugin(str(request.config.rootpath / "conftest.py"))
    assert root is not None, "the actual repository root identity hook must be loaded"

    def identity_probe(rootpath):
        assert rootpath == request.config.rootpath
        assert subprocess.Popen is original_popen, "identity runs before general sealing"
        if enabled:
            assert harness._BROKER_FREE is not None, "broker prevention must already be active"
            assert client._spawn_broker is not original_spawn
        else:
            assert harness._BROKER_FREE is None
            assert client._spawn_broker is original_spawn
        events.append("identity")
        if scenario == "caught-broker":
            with pytest.raises(RuntimeError, match="denied _spawn_broker"):
                client._spawn_broker(home / "config")
        return "synthetic identity refusal" if scenario == "identity-refusal" else None

    monkeypatch.setattr(root.tree_identity, "mismatch", identity_probe)

    class Normal:
        @pytest.hookimpl
        def pytest_configure(self, config):
            root._refuse_a_split_tree(config)

        @pytest.hookimpl
        def pytest_collection(self, session):
            events.append("collection")

    boundary = SimpleNamespace(
        pytest_broker_free_prepare=harness.pytest_broker_free_prepare,
        pytest_broker_free_seal=harness.pytest_broker_free_seal,
    )
    manager = pluggy.PluginManager("pytest")
    manager.add_hookspecs(hookspec)
    plugins = (
        (boundary, Normal()) if registration_order == "boundary-first" else (Normal(), boundary)
    )
    try:
        for plugin in plugins:
            manager.register(plugin)
        if scenario == "identity-refusal":
            with pytest.raises(pytest.UsageError, match="synthetic identity refusal"):
                manager.hook.pytest_configure.call_historic(kwargs={"config": config})
                manager.hook.pytest_collection(session=None)
            assert events == ["identity"]
            assert subprocess.Popen is original_popen, "a failed configure must not reach sealing"
        else:
            manager.hook.pytest_configure.call_historic(kwargs={"config": config})
            if enabled:
                state = harness._BROKER_FREE
                assert state is not None
                if scenario == "caught-broker":
                    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
                    harness.pytest_sessionfinish(cast(pytest.Session, session))
                    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
                else:
                    assert state.violations == []
                with pytest.raises(RuntimeError, match="denied subprocess.Popen"):
                    subprocess.Popen(["must-not-run"])
                assert "broker-free denied subprocess.Popen" in state.violations
            else:
                assert harness._BROKER_FREE is None
                assert subprocess.Popen is original_popen
                assert client._spawn_broker is original_spawn
            manager.hook.pytest_collection(session=None)
            assert events == ["identity", "collection"]
    finally:
        if harness._BROKER_FREE is not None:
            harness._BROKER_FREE.patch.undo()
        monkeypatch.setattr(harness, "_BROKER_FREE", outer)


def test_synthetic_cache_environment_and_effective_directory_agree(state, tmp_path):
    expected = tmp_path / "temporary"
    assert Path(os.environ["TMPDIR"]) == expected
    assert Path(tempfile.gettempdir()) == expected
    state.remember(tmp_path / "config")
    state.audit()
    assert state.violations == []
    assert all(path.is_relative_to(tmp_path) for path in state.artifacts)


def test_scoped_inspection_double_leaves_real_outer_audit_clean(state, tmp_path, monkeypatch):
    observer = harness._BROKER_FREE or harness._BrokerFree((tmp_path,))
    assert observer.violations == []

    def broken(*args, **kwargs):
        raise PermissionError("synthetic only")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", broken)
        state.remember(tmp_path / "config")
        assert any("inspection failed" in value for value in state.violations)
    # The enclosing session audits before ordinary fixture patch teardown.
    # Exercise that real observer only after the intentional double is undone.
    observer.discover((tmp_path,))
    observer.audit()
    assert observer.violations == []


@pytest.fixture
def numbered(state, tmp_path):
    from _pytest.tmpdir import TempPathFactory

    base = tmp_path / "numbered"
    base.mkdir()
    state.install_broker_guards()
    return TempPathFactory(
        given_basetemp=base,
        basetemp=base,
        retention_count=3,
        retention_policy="all",
        trace=lambda *args, **kwargs: None,
        _ispytest=True,
    )


def test_exact_pytest_alias_provenance_keeps_every_numbered_target(state, numbered):
    first = numbered.mktemp("record-")
    second = numbered.mktemp("record-")
    alias = first.parent / "record-current"
    assert state.pytest_aliases == {alias: second}
    assert first in state.candidates and second in state.candidates
    assert os.readlink(alias) == str(second)
    state.discover((first.parent,))
    state.audit()
    assert state.violations == []


def test_unrecorded_current_lookalike_remains_a_failure(state, numbered):
    target = numbered.mktemp("record-")
    alias = target.parent / "unrecordedcurrent"
    alias.symlink_to(target, target_is_directory=True)
    try:
        state.discover((target.parent,))
        assert any("symlink candidate" in value for value in state.violations)
    finally:
        alias.unlink()


@pytest.mark.parametrize("destination", ["owned", "foreign"])
def test_recorded_alias_retarget_fails_without_adopting_target(state, numbered, destination):
    target = numbered.mktemp("record-")
    alias = target.parent / "record-current"
    alternate = target.parent / "other" if destination == "owned" else target.parent.parent.parent
    alias.unlink()
    alias.symlink_to(alternate, target_is_directory=True)
    try:
        state.audit_pytest_alias(alias)
        assert any("target changed" in value for value in state.violations)
        assert state.pytest_aliases[alias] == target
        assert alternate not in state.candidates
    finally:
        alias.unlink()
        alias.symlink_to(target, target_is_directory=True)


def test_recorded_alias_parent_escape_is_rejected(state, numbered):
    target = numbered.mktemp("record-")
    parent = target.parent
    saved = parent.with_name("numbered-saved")
    parent.rename(saved)
    parent.symlink_to(parent.parent.parent, target_is_directory=True)
    try:
        state.audit()
        assert any("symlink candidate" in value for value in state.violations)
    finally:
        parent.unlink()
        saved.rename(parent)


def test_recorded_alias_inspection_error_is_visible(state, numbered, monkeypatch):
    target = numbered.mktemp("record-")
    alias = target.parent / "record-current"

    def broken(*args, **kwargs):
        raise PermissionError("synthetic alias refusal")

    with monkeypatch.context() as patch:
        patch.setattr(os, "readlink", broken)
        state.audit_pytest_alias(alias)
        assert any("alias inspection failed" in value for value in state.violations)


def test_pytest_alias_provenance_ledger_is_bounded(state, numbered, monkeypatch):
    monkeypatch.setattr(state, "LIMIT", 1)
    numbered.mktemp("first-")
    numbered.mktemp("second-")
    assert len(state.pytest_aliases) == 1
    assert any("alias limit reached" in value for value in state.violations)


def test_deleted_numbered_target_retains_surviving_fallback(state, numbered):
    target = numbered.mktemp("record-")
    artifact = client.socket_path(target)
    assert not artifact.is_relative_to(target), "control must exercise fallback layout"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.touch()
    target.rmdir()
    try:
        state.discover((target.parent,))
        state.audit()
        assert target in state.candidates
        assert artifact in state.artifacts
        assert any("unexpected broker artifact" in value for value in state.violations)
    finally:
        artifact.unlink()


@pytest.mark.parametrize("metadata", ["uninitialized", "unsupported", "foreign", "stale"])
def test_mktemp_rejects_unsafe_metadata_before_mutating_accessor(
    state, tmp_path, monkeypatch, metadata
):
    from _pytest.tmpdir import TempPathFactory

    calls = []
    factory = TempPathFactory(
        given_basetemp=tmp_path.parent / "foreign-root",
        retention_count=3,
        retention_policy="all",
        trace=lambda *args, **kwargs: None,
        _ispytest=True,
    )
    values = {
        "uninitialized": None,
        "unsupported": "not-a-Path",
        "foreign": tmp_path.parent / "foreign-root",
        "stale": tmp_path / "missing-root",
    }
    monkeypatch.setattr(factory, "_basetemp", values[metadata])

    def mutating_accessor():
        calls.append("would-delete-or-create")
        pytest.fail("unsafe getbasetemp accessor was called")

    monkeypatch.setattr(factory, "getbasetemp", mutating_accessor)
    state.install_broker_guards()
    with pytest.raises(RuntimeError, match="broker-free"):
        factory.mktemp("record-")
    assert calls == []
    assert state.violations


@pytest.mark.parametrize("preexisting", [False, True])
def test_session_factory_initialization_never_calls_destructive_accessor(
    tmp_path, monkeypatch, preexisting
):
    fresh = tmp_path / "fresh-session"
    marker = fresh / "must-survive"
    if preexisting:
        fresh.mkdir()
        marker.write_text("owned test marker")
    owner = harness._BrokerFree((fresh, tmp_path))
    calls = []

    def mutating_accessor():
        calls.append("would-delete-or-create")
        pytest.fail("session initialization used destructive getbasetemp")

    factory = SimpleNamespace(_basetemp=None, _given_basetemp=fresh, getbasetemp=mutating_accessor)
    with monkeypatch.context() as patch:
        patch.setattr(harness, "_BROKER_FREE", owner)
        patch.setattr(harness, "_SESSION_BASETEMP", harness._SESSION_BASETEMP)
        if preexisting:
            with pytest.raises(RuntimeError, match="initial root creation failed"):
                harness._capture_session_basetemp.__wrapped__(factory)
            assert marker.read_text() == "owned test marker"
            assert owner.violations
        else:
            harness._capture_session_basetemp.__wrapped__(factory)
            assert factory._basetemp == fresh
            assert fresh.is_dir()
            assert harness._SESSION_BASETEMP == fresh
            assert owner.violations == []
    assert calls == []


def test_best_effort_alias_failure_keeps_expected_return_not_leftover_link(
    state, numbered, monkeypatch
):
    first = numbered.mktemp("record-")
    alias = first.parent / "record-current"
    original_unlink = Path.unlink
    original_symlink = Path.symlink_to
    attempts = []
    second = None

    def blocked_unlink(path, *args, **kwargs):
        if path == alias:
            attempts.append("unlink")
            raise PermissionError("synthetic best-effort unlink refusal")
        return original_unlink(path, *args, **kwargs)

    def blocked_symlink(path, *args, **kwargs):
        if path == alias:
            attempts.append("symlink")
            raise PermissionError("synthetic best-effort creation refusal")
        return original_symlink(path, *args, **kwargs)

    try:
        # Drive the real pytest helper: it suppresses both failures and returns
        # a new numbered directory while the OLD, now-wrong alias survives.
        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", blocked_unlink)
            patch.setattr(Path, "symlink_to", blocked_symlink)
            second = numbered.mktemp("record-")
        assert attempts == ["unlink", "symlink"]
        assert os.readlink(alias) == str(first)
        assert second != first
        assert state.pytest_aliases[alias] == second
        state.audit_pytest_alias(alias)
        assert any("target changed" in value for value in state.violations)
        assert state.pytest_aliases[alias] != Path(os.readlink(alias))
    finally:
        # Only this synthetic private alias is restored for the enclosing
        # observer; never touch a foreign target or erase a real ledger entry.
        if second is not None:
            alias.unlink()
            alias.symlink_to(second, target_is_directory=True)


def test_mode_rejects_broker_opt_in_at_collection():
    config = SimpleNamespace(getoption=lambda name: True)
    item = SimpleNamespace(get_closest_marker=lambda name: object())
    with pytest.raises(pytest.UsageError, match="conflicts"):
        harness.pytest_collection_modifyitems(
            cast(pytest.Config, config), [cast(pytest.Item, item)]
        )
