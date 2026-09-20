"""The tunnel installer must not write or churn launchd when nothing changed.

WHY THIS IS A FILE OF ITS OWN, next to ``test_tunnel_install.py`` (which covers
the platform arms and the refusal messages): the behaviour added here is the
opposite of "do more". With the generation shim the plist path is stable, so a
re-install usually renders the SAME bytes, and rewriting them plus a
bootout/bootstrap is exactly what an EDR reads as "Persistence: launchd job /
plist file modification" (MITRE T1543.001) with no functional change behind it.
``wakes.install`` has compared content before acting since it shipped; these
cells pin the same shape here, in both directions, with the launchctl call log
asserted directly.

Darwin-only: off macOS this daemon is a systemd unit or a scheduled task, and
neither is re-created the same way.
"""

from __future__ import annotations

import argparse
import http.server
import json
import plistlib
import sys
import threading
from pathlib import Path

import pytest

from local_operator import launchd, supervisors
from local_operator.tunnels import install

#: The REAL reload, kept because the rig replaces it (its plist lives under
#: ``tmp_path``) and the sandbox cell has to put the real one back to prove the
#: refusal still happens — the identity test it uses is already the real one.
_REAL_RELOAD_JOB = launchd.reload_job

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the plist lifecycle is macOS-only; Linux/Windows have their own arms",
)


class FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Rig:
    """``install()`` with the file system pointed at ``tmp_path`` and calls recorded.

    ``own`` says whether this tmpdir is the home launchd would have taken the
    plist from, and it is spelled by pointing ``real_home`` at it rather than by
    stubbing ``is_own_plist``: a stubbed guard restored the original over a
    runtime mutation and hid it, which is what made these cells
    un-mutation-verifiable (review round 2, R-10). ``own=False`` pins nothing, so
    the real operator home is compared against and the refusal is genuine.

    ``answering`` is the connector's own gateway, faked because the real probe
    is exercised against a real loopback listener by its own cell below.
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, own: bool = True
    ) -> None:
        home = tmp_path
        home.joinpath("Library", "LaunchAgents").mkdir(parents=True, exist_ok=True)
        self.path = home / "Library" / "LaunchAgents" / f"{install.LABEL}.plist"
        self.calls: list[list[str]] = []
        self.loaded = True
        self.answering = True
        self.reloads: list[Path] = []
        # Stated rather than inherited from the runner's OS, like the mobile
        # installer's own fixture does.
        monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.LAUNCHCTL)
        monkeypatch.setattr(install, "service_path", lambda: self.path)
        monkeypatch.setattr(install.config, "directory", lambda base=None: tmp_path)
        monkeypatch.setattr(install, "_launchctl", self._launchctl)
        monkeypatch.setattr(install.launchd, "reload_job", self._reload_job)
        monkeypatch.setattr(install, "gateway_answers", lambda *a, **k: self.answering)
        if own:
            monkeypatch.setattr(install.launchd, "real_home", lambda: home)

    def _launchctl(self, *cmd: str) -> FakeProc:
        self.calls.append(list(cmd))
        if cmd[:1] == ("print",):
            # `pid = …` is what launchd prints only while a process is alive:
            # this is the reading `launchd.job_running` parses.
            if self.loaded:
                return FakeProc(returncode=0, stdout="\tpid = 4242\n")
            return FakeProc(returncode=113, stdout="Bad request.", stderr="Could not find service")
        if cmd[:2] == ("kickstart", "-k"):
            self.loaded = True
        return FakeProc(returncode=0)

    def _reload_job(self, **kwargs: object) -> launchd.JobReload:
        self.reloads.append(kwargs["path"])  # type: ignore[arg-type]
        return launchd.JobReload(label=str(kwargs["label"]), outcome="reloaded")

    def write_current_plist(self) -> None:
        payload = plistlib.dumps(install.render_plist())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(payload)
        self.path.chmod(0o600)


def test_a_sandboxed_repair_never_reaches_the_real_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-1 for the tunnel arm, whose LABEL is a fixed constant.

    The reviewer's reproduction: a re-install under a redirected HOME that finds
    the job loaded-but-stopped used to issue `kickstart -k
    gui/<uid>/com.local-operator.tunnel` — the operator's real connector. The
    identity guard refuses both helpers now, and the reload that follows
    refuses in its own words, so no launchctl call is issued at all and the
    decline is reported rather than silent.
    """
    rig = _Rig(monkeypatch, tmp_path, own=False)
    monkeypatch.setattr(install.launchd, "reload_job", _REAL_RELOAD_JOB)
    rig.write_current_plist()
    rig.loaded = False  # launchd knows the label but is running nothing

    with pytest.raises(ValueError) as raised:
        install.install()

    assert rig.calls == [], f"a sandboxed install reached launchctl: {rig.calls}"
    assert "not the LaunchAgent the real home owns" in str(raised.value)


def test_a_reinstall_that_changes_nothing_does_not_write_or_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Equal bytes and a live job: no write, no bootout/bootstrap, no kickstart.

    The probe is the ONLY call allowed, and it is the call that makes the skip
    safe — launchd is the authority on whether this job is running.
    """
    rig = _Rig(monkeypatch, tmp_path)
    rig.write_current_plist()
    before = rig.path.stat().st_mtime_ns

    install.install()

    assert rig.path.stat().st_mtime_ns == before, "the plist was rewritten"
    assert rig.reloads == [], "an unchanged plist must not be reloaded"
    assert rig.calls == [["print", f"{launchd.job_domain()}/{install.LABEL}"]], rig.calls


def test_a_reinstall_still_writes_and_reloads_a_changed_plist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plist from an older build is still replaced and loaded."""
    rig = _Rig(monkeypatch, tmp_path)
    rig.path.parent.mkdir(parents=True, exist_ok=True)
    rig.path.write_bytes(plistlib.dumps({"Label": install.LABEL, "stale": True}))
    rig.path.chmod(0o600)

    install.install()

    assert plistlib.loads(rig.path.read_bytes()) == install.render_plist()
    assert rig.reloads == [rig.path], "a changed plist must still be reloaded"


def test_a_wrong_mode_is_still_repaired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Content AND mode make "current": this installer is the one that sets 0600.

    A file whose bytes match but whose mode drifted is a real state (an archive
    restore, a hand edit) and the old unconditional write repaired it. Treating
    "bytes match" as enough would quietly stop doing that.
    """
    rig = _Rig(monkeypatch, tmp_path)
    rig.write_current_plist()
    rig.path.chmod(0o644)

    install.install()

    assert (rig.path.stat().st_mode & 0o777) == 0o600
    assert rig.reloads == [rig.path]


def test_a_loaded_but_dead_job_is_restarted_without_rewriting_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The state the old unconditional reload repaired, still repaired.

    ``kickstart -k`` rather than bootout+bootstrap: the file is already correct,
    so the narrower operation is right here (the bootout pair is for a REWRITE,
    where launchd's in-memory definition is what has to be replaced).
    """
    rig = _Rig(monkeypatch, tmp_path)
    rig.write_current_plist()
    rig.loaded = False
    before = rig.path.stat().st_mtime_ns

    install.install()

    assert rig.path.stat().st_mtime_ns == before, "the file was already correct"
    assert ["kickstart", "-k", f"{launchd.job_domain()}/{install.LABEL}"] in rig.calls
    assert rig.reloads == [], "a stopped job needs a restart, not a rewrite"


def test_a_label_launchd_forgot_falls_through_to_the_full_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing registered ⇒ the reload IS the registration; skipping it would leave it dead."""
    rig = _Rig(monkeypatch, tmp_path)
    rig.write_current_plist()

    def unregistered(*cmd: str) -> FakeProc:
        rig.calls.append(list(cmd))
        # `print` fails for an unknown label AND the kickstart that follows it,
        # which is the real sequence: the only way back is bootout+bootstrap.
        return FakeProc(returncode=113, stdout="Bad request.", stderr='Could not find service "x"')

    monkeypatch.setattr(install, "_launchctl", unregistered)

    install.install()

    assert rig.reloads == [rig.path]


def test_the_log_file_is_still_prepared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The skip is about the PLIST and launchd; the log path is still made ready.

    Pinned because "install became a no-op" is the natural over-reading of this
    change: the daemon writes its output there, and an install on a fresh store
    must still leave it writable.
    """
    rig = _Rig(monkeypatch, tmp_path)
    rig.write_current_plist()

    install.install()

    assert (tmp_path / "service.log").exists()


def test_a_loaded_but_unanswering_connector_is_repaired_not_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-7: the connector's own gateway is the local surface the skip must ask.

    The old gate skipped on liveness alone, so `lop tunnel install` reported
    success over a connector that launchd was holding a live pid for and that
    could not relay — and `lop tunnel install` is the very command
    `tunnels/cli.py` names as the repair for that state. The plist is already
    correct here, so the repair is the narrow one.
    """
    rig = _Rig(monkeypatch, tmp_path)
    rig.write_current_plist()
    rig.loaded = True
    rig.answering = False
    before = rig.path.stat().st_mtime_ns

    install.install()

    assert rig.path.stat().st_mtime_ns == before, "the file was already correct"
    assert ["kickstart", "-k", f"{launchd.job_domain()}/{install.LABEL}"] in rig.calls, rig.calls
    assert rig.reloads == [], "kickstart is the narrower repair and it succeeded"


def test_the_gateway_probe_answers_only_for_a_serving_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe against a REAL loopback listener, in both directions.

    A fake would prove nothing about this finding: the whole point of R-7 is that
    the surface exists and is worth asking. The record is pointed at a port this
    test binds, so the answer comes from a real socket.

    THIS HANDLER IS ITSELF THE STRAY LISTENER the docstring names as a limit: it
    is not the gateway, and it passes the probe. That is the measured behaviour
    the docstring now states instead of denying (review round 3, QA Q-1/N2); the
    composite gate is safe because `job_running` is asked first.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — the stdlib's spelling
            if self.path != "/_lop_tunnel/health":
                self.send_response(404)
                self.end_headers()
                return
            payload = b'{"ok": false, "connected": false}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return  # an access log per probe would be noise in the pytest output

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(install.config, "load", lambda: {"gateway_port": server.server_port})
        # `ok: false` is the deliberate part: that is a LIVE connector whose
        # relay authorization lapsed, and this gate's question is only "did my
        # own gateway answer on my port" — the same statement the twins'
        # `health(port) is not None` makes. `lop tunnel status` owns the other
        # question.
        assert install.gateway_answers() is True
    finally:
        server.shutdown()
        server.server_close()

    # A port nothing listens on: refused, closed or hung all mean repair.
    monkeypatch.setattr(install.config, "load", lambda: {"gateway_port": 1})
    assert install.gateway_answers(timeout=0.5) is False


def test_the_stop_verb_refuses_a_redirected_home_and_acts_for_the_owned_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R-8: `stop` was a bare `bootout gui/<uid>/<label>` from anywhere.

    The label is a fixed module constant here while `service_path()` moves with
    `$HOME`, so a redirected home's stop took the OPERATOR's connector down. The
    mirrored half keeps the real-home verb working — through `_run`, which is
    where this arm's bootout goes rather than through `_launchctl`.
    """
    sandbox = _Rig(monkeypatch, tmp_path / "sandbox", own=False)
    sandbox.write_current_plist()

    with pytest.raises(ValueError) as raised:
        install.action("stop")

    assert sandbox.calls == [], f"a redirected home reached launchd: {sandbox.calls}"
    assert "not the LaunchAgent the real home owns" in str(raised.value)

    owned = _Rig(monkeypatch, tmp_path / "home")
    owned.write_current_plist()
    runs: list[list[str]] = []
    monkeypatch.setattr(install, "_run", lambda args, **kwargs: runs.append(list(args)))

    install.action("stop")

    assert runs == [["launchctl", "bootout", f"{launchd.job_domain()}/{install.LABEL}"]], runs


def test_an_unreadable_record_answers_not_answering_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """M2: the probe's never-raises contract covers the FILESYSTEM, not just config.

    The reviewer's reproduction: with the record present but unreadable as a file
    — `config.json` a directory, or a permission the sandbox lacks — `load()`
    raises `OSError`, which is not a `ValueError`. Catching only that let it
    escape `install()` and turn a repair into a traceback. Not answering is the
    answer, because it is the direction that repairs.
    """
    store = tmp_path / "store" / "tunnel"
    (store / "config.json").mkdir(parents=True)
    monkeypatch.setattr(install.config, "directory", lambda base=None: store)

    assert install.gateway_answers(timeout=0.2) is False


def test_a_redirected_home_uninstall_still_removes_its_own_plist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """N3: the supervisor half may refuse; the FILE half must still work.

    `uninstall()` calls `action("stop")`, which applies the identity guard, so
    from a redirected home it declines — and that refusal escaped, leaving the
    sandbox's own plist on disk behind a traceback. Nothing of the operator's is
    touched either way: the refused stop issues no call at all.
    """
    rig = _Rig(monkeypatch, tmp_path / "sandbox", own=False)
    rig.write_current_plist()
    monkeypatch.setattr(install, "_run", lambda args, **kwargs: rig.calls.append(list(args)))

    install.uninstall()  # must not raise

    assert not rig.path.exists(), "the sandbox's own plist must still be removed"
    assert rig.calls == [], f"a redirected home reached launchd: {rig.calls}"


def test_a_refused_stop_is_reported_as_a_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA Q-2: a refused stop must not read as success.

    `lop tunnel stop` treats a `ValueError` from `action("stop")` as "there is no
    supervised job — the connector runs in the foreground". That is right for the
    foreground case and was a false success for a refusal: the identity guard
    raised the same type, nothing was called, and the CLI printed
    "Stop requested …" with exit 0. The refusal now reaches the operator as one —
    its own sentence on stderr, non-zero exit — which is what the type
    (`launchd.JobNotOurs`) exists for.
    """
    from local_operator.tunnels import cli as tunnel_cli

    rig = _Rig(monkeypatch, tmp_path / "sandbox", own=False)
    rig.write_current_plist()
    # The rig's home IS the store the CLI reads (`config.directory` is patched to
    # it), so the record has to live there, not one level up.
    (tmp_path / "sandbox" / "config.json").write_text(
        json.dumps({"stopped": False, "gateway_port": 4100})
    )

    code = tunnel_cli.main(argparse.Namespace(tunnel_command="stop"))

    captured = capsys.readouterr()
    assert code == 1, captured
    assert "not the LaunchAgent the real home owns" in captured.err
    assert "Stop requested" not in captured.out
    assert rig.calls == [], f"a redirected home reached launchd: {rig.calls}"


def test_a_stop_with_no_supervisor_still_reports_the_foreground_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other direction: the cheerful branch is right where it still applies.

    A machine with no supervisor at all (a foreground connector, `lop tunnel
    serve`) is not a refusal, and the sentence that branch exists for must
    survive the new type.
    """
    from local_operator.tunnels import cli as tunnel_cli

    monkeypatch.setattr(install.supervisors, "supervisor", lambda: None)
    monkeypatch.setattr(install.config, "directory", lambda base=None: tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"stopped": False, "gateway_port": 4100}))

    code = tunnel_cli.main(argparse.Namespace(tunnel_command="stop"))

    captured = capsys.readouterr()
    assert code == 0, captured
    assert "Stop requested" in captured.out
