"""The address axis, and the one command that can end a serve daemon.

Two harms shape every assertion here, because this is the only code in the
product that signals a process the user did not start in this shell:

* **Never end a daemon that is SERVING.** ``reclaim`` exists for a daemon that
  holds an address and answers nothing, or that holds an address another record
  claims. A working plane is not a thing to reclaim, and the verdict that says
  "serving" must refuse before any signal — the incident's own daemon (pid 1276)
  was NOT the one to end; the two strangers squatting its port were.
* **Never signal a pid that was not re-identified at signal time.** A pid is
  recyclable, so the proof (this product's serve daemon, same uid) is re-read
  immediately before the signal, exactly as the session sweep re-reads its
  targets. A snapshot is not evidence about this instant.

The third subject is the REPORTING half: a daemon that is alive and answering
nothing used to print as "none running" and had no name in the product at all,
which is why the operator spent twelve minutes reconstructing "who is on 1111"
from ``lsof`` by hand on 2026-09-23.
"""

from __future__ import annotations

import urllib.error
from typing import Any

import pytest

from local_operator import services
from local_operator.server import registry as serve_registry

SERVE_ARGV = (
    "/opt/lop/bin/Local Operator [serve] port=1111 -P -m local_operator.cli "
    "serve --host 127.0.0.1 --port 1111"
)


def _record(**over: Any) -> serve_registry.ServeRecord:
    fields: dict[str, Any] = {
        "pid": 4242,
        "host": "127.0.0.1",
        "port": 1111,
        "instance_id": "instance-one",
        "version": "0.62.24",
        "source_ref": "a" * 40,
        "prefix": "/opt/gen/tools/local-operator",
        "install_kind": "uv-tool",
        "desktop": True,
        "reloadable": True,
    }
    fields.update(over)
    return serve_registry.ServeRecord(**fields)


class _Response:
    """One ``/health`` answer, in the shape ``urlopen`` yields."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


@pytest.fixture
def probe_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Control the probe's HTTP answer and record every URL it asked."""
    state: dict[str, Any] = {"answer": _Response(b'{"result":{"instance_id":"instance-one"}}')}
    asked: list[str] = []

    def _open(url: str, timeout: float = 0) -> Any:
        asked.append(url)
        answer = state["answer"]
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            return answer(url)
        return answer

    monkeypatch.setattr("urllib.request.urlopen", _open)
    return {"state": state, "asked": asked}


# --------------------------------------------------------------------------- #
# The address axis
# --------------------------------------------------------------------------- #


def test_a_matching_instance_is_serving(probe_env: dict[str, Any]) -> None:
    """The only verdict that is allowed to refuse a reclaim on its own."""
    probe = services.probe_address(_record(instance_id="instance-one"))
    assert probe.verdict == services.SERVING
    # ``detail`` empty is the shape `_answers_as_record` has always rendered,
    # which is why the verdict is a separate field rather than a prefix.
    assert probe.detail == ""


def test_a_foreign_answer_is_squatted_and_names_the_stranger(
    probe_env: dict[str, Any],
) -> None:
    """The incident's ``identity-mismatch``: somebody else is on the address."""
    probe_env["state"]["answer"] = _Response(
        b'{"result":{"instance_id":"a-different-process"}}'
    )
    probe = services.probe_address(_record(instance_id="instance-one"))
    assert probe.verdict == services.SQUATTED
    assert probe.answered_as == "a-different-process"
    assert "is answering as a-different-process" in probe.detail


def test_an_error_response_is_not_reported_as_nobody(probe_env: dict[str, Any]) -> None:
    """A 500 is an ANSWER, and calling it silence hides the process from the reader.

    Measured on 2026-09-23: a stray rig's daemon answered ``500`` on port 8080 and
    no surface in the product named it. ``HTTPError`` is a response class, so it
    must not fall into the "did not identify itself" arm with refusals.
    """
    probe_env["state"]["answer"] = urllib.error.HTTPError(
        "http://127.0.0.1:1111/health",  # the URL the probe asked
        500,  # the status it answered with
        "Internal Server Error",
        {},  # headers
        None,  # fp
    )
    probe = services.probe_address(_record())
    assert probe.verdict == services.SQUATTED
    assert "answered 500" in probe.detail


def test_a_silent_address_is_deaf_never_serving(probe_env: dict[str, Any]) -> None:
    """FAIL-CLOSED. An unreadable answer may not authorize anything."""
    probe_env["state"]["answer"] = OSError("Connection refused")
    probe = services.probe_address(_record())
    assert probe.verdict == services.DEAF
    assert "did not identify itself" in probe.detail


def test_the_probe_url_brackets_an_ipv6_host(probe_env: dict[str, Any]) -> None:
    """Inherited from ``_answers_as_record``: an unbracketed literal is not a URL."""
    services.probe_address(_record(host="::1", port=1111, instance_id="instance-one"))
    assert probe_env["asked"] == ["http://[::1]:1111/health"]


def test_reports_compose_the_shared_state_with_the_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``wedged`` and ``stale`` come from the shared classifier and cost no probe.

    The address axis is ADDITIVE: a reader that asks "is the owner there" keeps
    the answer it had, and the records whose owner is not reporting are not worth
    a loopback round trip.
    """
    live = _record(pid=1, instance_id="instance-one")
    deaf = _record(pid=2, instance_id="two")
    wedged = _record(pid=3)
    stale = _record(pid=4)
    monkeypatch.setattr(
        services.serve_registry,
        "scan",
        lambda root=None: [(live, "live"), (deaf, "live"), (wedged, "wedged"), (stale, "stale")],
    )
    asked: list[int] = []

    def _probe(record: Any) -> services.AddressProbe:
        asked.append(record.pid)
        if record.pid == 2:
            return services.AddressProbe(services.DEAF, "nothing answered")
        return services.AddressProbe(services.SERVING)

    reports = services.serve_daemon_reports(probe=_probe)
    assert [(report.record.pid, report.verdict) for report in reports] == [
        (1, services.SERVING),
        (2, services.DEAF),
        (3, services.WEDGED),
        (4, services.STALE),
    ]
    # ``wedged`` IS probed: its owner stopped reporting, but its address can still
    # have been taken by somebody else, and "your wedged daemon's port is now held
    # by a stranger" is a different thing to tell an operator than "nothing is
    # answering there". ``stale`` alone is not probed — its pid is gone and its
    # record is on its way to ``reaped/``.
    assert asked == [1, 2, 3]


def test_the_spawn_contract_is_matched_as_words_not_a_substring() -> None:
    """A `grep` of the module name is not a serve daemon.

    The census this proves against reads the process table, where a person (or a
    test, or a rig) grepping for ``local_operator.cli serve`` looks exactly like
    the daemon to a substring match — and this command ends processes.
    """
    assert services.is_serve_command(SERVE_ARGV)
    assert services.is_serve_command("/Users/damian/.local/bin/lop serve --port 1111")
    assert not services.is_serve_command("grep -rn local_operator.cli serve src/")
    assert not services.is_serve_command(
        "/bin/zsh -c python -m local_operator.cli services status"
    )


# --------------------------------------------------------------------------- #
# Reporting: a daemon that is alive and not answering
# --------------------------------------------------------------------------- #


def test_status_names_a_daemon_that_is_not_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The twelve-minute gap, closed: the stuck daemon has a name and a remedy."""
    record = _record(pid=1276, port=1111)
    monkeypatch.setattr(services, "live_serve_daemons", lambda: [record])
    monkeypatch.setattr(
        services,
        "serve_daemon_reports",
        lambda *a, **k: [
            services.ServeDaemonReport(
                record=record,
                state="live",
                probe=services.AddressProbe(
                    services.DEAF, "127.0.0.1:1111 did not identify itself"
                ),
            )
        ],
    )
    lines = services.status_lines()
    rendered = "\n".join(lines)
    assert "127.0.0.1:1111" in rendered
    # The state is named in words, not as the token: every operator-facing surface
    # in this repo says what it means (``row_state_mark``'s vocabulary is the one
    # place tokens are painted), so the assertion is on the sentence.
    assert "but nothing is answering there" in rendered
    assert "lop services reclaim 1276" in rendered


def test_status_says_none_serving_rather_than_none_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``none running`` was a false statement about a live, recorded daemon."""
    record = _record(pid=99)
    monkeypatch.setattr(services, "live_serve_daemons", lambda: [])
    monkeypatch.setattr(
        services,
        "serve_daemon_reports",
        lambda *a, **k: [
            services.ServeDaemonReport(
                record=record, state="wedged", probe=None
            )
        ],
    )
    rendered = "\n".join(services.status_lines())
    assert "serve daemons: none serving (1 recorded and not answering)" in rendered


def test_a_stale_record_is_not_offered_a_reclaim() -> None:
    """The remedy is named only where it is real.

    A stale record's pid is gone; telling a reader to reclaim it is advice that
    cannot work, and this repo refuses to print that class of advice.
    """
    report = services.ServeDaemonReport(record=_record(pid=5), state="stale", probe=None)
    rendered = "\n".join(services.not_serving_lines(report))
    assert "reclaim" not in rendered


# --------------------------------------------------------------------------- #
# reclaim: the proof, the verdict, and the ladder
# --------------------------------------------------------------------------- #


def _reclaim(
    pid: int = 4242,
    *,
    verdict: str = services.DEAF,
    command: str | None = SERVE_ARGV,
    uid: int | None = None,
    alive: list[bool] | None = None,
    records: list[services.ServeDaemonReport] | None = None,
    **over: Any,
) -> tuple[services.ReclaimReport, list[tuple[int, int]]]:
    """Run ``reclaim_serve_daemon`` against injected evidence.

    ``alive`` is a script of answers the ladder consumes in order, so a test can
    say "it leaves on SIGTERM" or "it ignores SIGTERM and dies on SIGKILL"
    without a real process.
    """
    import os as os_mod

    kills: list[tuple[int, int]] = []
    # ``alive`` is a script consumed in order, and the LAST entry repeats once it
    # is exhausted: ``[True, False]`` is "it leaves on SIGTERM", ``[True]`` is "it
    # never leaves", and neither test has to know how many polls the ladder pays.
    answers = list(alive) if alive is not None else []

    def _alive(_pid: int) -> bool | None:
        if not answers:
            return True
        value = answers.pop(0)
        if not answers:
            answers.append(value)
        return value

    reports = records if records is not None else [
        services.ServeDaemonReport(
            record=_record(pid=pid),
            state="live",
            probe=services.AddressProbe(verdict, "" if verdict == services.SERVING else "silent"),
        )
    ]

    def _probe(_record: Any) -> services.AddressProbe:
        return services.AddressProbe(verdict, "" if verdict == services.SERVING else "silent")

    outcome = services.reclaim_serve_daemon(
        pid,
        probe=_probe,
        reports=lambda: reports,
        read_command=lambda _pid: command,
        read_uid=lambda _pid: (uid if uid is not None else os_mod.getuid()),
        kill=lambda target, sig: kills.append((target, sig)),
        alive=_alive,
        sleep=lambda _s: None,
        term_grace_s=0.001,
        kill_confirm_s=0.001,
        poll_s=0.0005,
        confirm_gap_s=0.0,
        **over,
    )
    return outcome, kills


def test_a_serving_daemon_is_never_reclaimed() -> None:
    """THE REFUSAL THAT MATTERS MOST: a working plane is not a stray."""
    outcome, kills = _reclaim(verdict=services.SERVING)
    assert kills == []
    assert outcome.refused
    assert outcome.problem == "serving"
    assert "IS the daemon serving" in "\n".join(outcome.lines)


def test_a_pid_that_is_not_a_serve_daemon_is_refused() -> None:
    """Proof of brand, so a recycled or mistyped pid is refused, not signalled."""
    outcome, kills = _reclaim(command="/Applications/Some App/Some App --chatty")
    assert kills == []
    assert outcome.problem == "not-a-serve-daemon"


def test_an_unreadable_pid_is_refused() -> None:
    """``None`` from the re-read is DOUBT, and doubt sends nothing.

    It is reported as ``not-running`` because that is what the reader can verify:
    the process table does not describe this pid now. Doubt is never upgraded into
    a signal, and the verdict carries the shared word for a pid that is not there.
    """
    outcome, kills = _reclaim(command=None)
    assert kills == []
    assert outcome.problem == "not-running"
    assert outcome.verdict == services.STALE


def test_another_users_process_is_refused() -> None:
    """The uid is the security boundary this proof leans on."""
    outcome, kills = _reclaim(uid=0)
    assert kills == []
    assert outcome.problem == "foreign-user"
    assert "sudo" in "\n".join(outcome.lines)


def test_the_verdict_must_hold_across_probes() -> None:
    """One refused connection is not evidence a daemon is down.

    The probe is asked ``PROBE_CONFIRMATIONS`` times and any disagreement aborts
    before a signal: this host ran at a load average of 130 for hours, and a
    single sample taken through that is how a healthy daemon gets ended.
    """
    calls: list[int] = []

    def _probe(_record: Any) -> services.AddressProbe:
        calls.append(1)
        if len(calls) == 1:
            return services.AddressProbe(services.DEAF, "silent")
        return services.AddressProbe(services.SERVING)

    kills: list[tuple[int, int]] = []
    outcome = services.reclaim_serve_daemon(
        4242,
        probe=_probe,
        reports=lambda: [
            services.ServeDaemonReport(
                record=_record(pid=4242),
                state="live",
                probe=services.AddressProbe(services.DEAF, "silent"),
            )
        ],
        read_command=lambda _pid: SERVE_ARGV,
        read_uid=lambda _pid: __import__("os").getuid(),
        kill=lambda target, sig: kills.append((target, sig)),
        alive=lambda _pid: True,
        sleep=lambda _s: None,
        confirm_gap_s=0.0,
    )
    assert kills == []
    assert outcome.problem == "unconfirmed"
    assert len(calls) == 2


def test_a_deaf_daemon_leaves_on_sigterm() -> None:
    """The ordinary reclaim: SIGTERM, it goes, and the receipt says so."""
    outcome, kills = _reclaim(alive=[True, False])
    assert [sig for _pid, sig in kills] == [__import__("signal").SIGTERM]
    assert outcome.acted and not outcome.refused
    assert "ended on SIGTERM" in "\n".join(outcome.lines)


def test_the_ladder_escalates_to_sigkill_at_the_bound() -> None:
    """A daemon that ignores SIGTERM is still ended — the incident's case."""
    import signal as signal_mod

    outcome, kills = _reclaim(alive=[True, True, True, False])
    assert [sig for _pid, sig in kills] == [signal_mod.SIGTERM, signal_mod.SIGKILL]
    assert "sending SIGKILL" in "\n".join(outcome.lines)
    assert outcome.acted


def test_a_daemon_that_survives_sigkill_is_reported_not_hidden() -> None:
    """Wedged in the kernel is a real outcome, and the operator is told."""
    outcome, kills = _reclaim(alive=[True])
    assert len(kills) == 2
    assert outcome.problem == "survived-sigkill"
    assert "STILL RUNNING" in "\n".join(outcome.lines)


def test_a_stray_with_no_record_of_ours_is_named_as_stray() -> None:
    """The incident's process: nothing of ours describes it, argv names it."""
    outcome, kills = _reclaim(
        command="/other/install/bin/lop serve --host 127.0.0.1 --port 1111",
        records=[],
        alive=[True, False],
    )
    assert outcome.verdict == services.STRAY
    assert outcome.acted
    assert "127.0.0.1:1111" in "\n".join(outcome.lines)


def test_a_stray_without_a_readable_address_is_refused() -> None:
    """An adopted listener names no port, so there is nothing to measure."""
    outcome, kills = _reclaim(
        command="/x/lop serve --host 127.0.0.1 --listener-fd 3",
        records=[],
    )
    assert kills == []
    assert outcome.problem == "no-address"


def test_the_process_is_re_identified_immediately_before_signalling() -> None:
    """A pid that stopped being the serve daemon is not the one we measured."""
    seen: list[int] = []

    def _read(pid: int) -> str | None:
        seen.append(pid)
        # First read: the proof. Second read (signal time): a stranger.
        return SERVE_ARGV if len(seen) == 1 else "/usr/bin/some-daemon --idle"

    kills: list[tuple[int, int]] = []
    outcome = services.reclaim_serve_daemon(
        4242,
        probe=lambda _r: services.AddressProbe(services.DEAF, "silent"),
        reports=lambda: [
            services.ServeDaemonReport(
                record=_record(pid=4242),
                state="live",
                probe=services.AddressProbe(services.DEAF, "silent"),
            )
        ],
        read_command=_read,
        read_uid=lambda _pid: __import__("os").getuid(),
        kill=lambda target, sig: kills.append((target, sig)),
        alive=lambda _pid: True,
        sleep=lambda _s: None,
        confirm_gap_s=0.0,
    )
    assert kills == []
    assert outcome.problem == "changed"
    assert len(seen) == 2


def test_a_probe_that_cannot_run_sends_nothing() -> None:
    """FAIL-CLOSED: an exception on the evidence path is not a verdict."""

    def _boom(_record: Any) -> services.AddressProbe:
        raise OSError("the process table could not be read")

    kills: list[tuple[int, int]] = []
    with pytest.raises(OSError):
        services.reclaim_serve_daemon(
            4242,
            probe=_boom,
            reports=lambda: [
                services.ServeDaemonReport(
                    record=_record(pid=4242),
                    state="live",
                    probe=services.AddressProbe(services.DEAF, "silent"),
                )
            ],
            read_command=lambda _pid: SERVE_ARGV,
            read_uid=lambda _pid: __import__("os").getuid(),
            kill=lambda target, sig: kills.append((target, sig)),
            alive=lambda _pid: True,
            sleep=lambda _s: None,
        )
    assert kills == []


# --------------------------------------------------------------------------- #
# The update path reports what it cannot move
# --------------------------------------------------------------------------- #


def test_the_update_path_reports_a_daemon_it_cannot_move() -> None:
    """Silence was the other half of the incident: the install said nothing.

    ``reload_serve_daemons`` iterates LIVE records to ask them to move, so a
    daemon that is alive and not answering was passed over without a word — and
    it is precisely the one that blocks the app's address.
    """
    record = _record(pid=1276, version="0.62.24", source_ref="a" * 40)
    kills: list[tuple[int, int]] = []
    stuck = [
        services.ServeDaemonReport(
            record=record,
            state="live",
            probe=services.AddressProbe(services.DEAF, "did not identify itself"),
        )
    ]
    out = services.reload_serve_daemons(
        scan=lambda: [record],
        kill=lambda pid, sig: kills.append((pid, sig)),
        probe=lambda _r: None,
        stuck=lambda: stuck,
    )
    warnings = [warning for refresh in out for warning in refresh.warnings]
    assert any("lop services reclaim 1276" in warning for warning in warnings)
    assert kills == [], "a daemon that cannot answer must still not be signalled"


def test_a_serving_daemon_is_not_reported_as_stuck() -> None:
    """The report is about daemons that are NOT serving; the rest keep their silence."""
    record = _record(pid=7)
    out = services.reload_serve_daemons(
        scan=lambda: [],
        kill=lambda pid, sig: None,
        stuck=lambda: [
            services.ServeDaemonReport(
                record=record, state="live", probe=services.AddressProbe(services.SERVING)
            )
        ],
    )
    assert out == []
