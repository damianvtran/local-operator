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


def _no_sleep(_seconds: float) -> None:
    """A sleep the ladder does not have to pay for."""


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
    probe_env["state"]["answer"] = _Response(b'{"result":{"instance_id":"a-different-process"}}')
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
    assert "did not answer" in probe.detail


def test_the_probe_url_brackets_an_ipv6_host(probe_env: dict[str, Any]) -> None:
    """Inherited from ``_answers_as_record``: an unbracketed literal is not a URL."""
    services.probe_address(_record(host="::1", port=1111, instance_id="instance-one"))
    assert probe_env["asked"] == ["http://[::1]:1111/health"]


def test_reports_compose_the_shared_state_with_the_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe outranks the beat where a probe ran; the beat is the story without one.

    Both halves of that composition matter, and they pull in opposite directions:

    * ``wedged`` is NOT a verdict on the process — the shared classifier says a
      stale beat is a long turn or a starved loop — so a wedged record whose
      address answers as its own instance is SERVING, and the report has to say so
      or ``reclaim`` has no guard left against ending a working plane (review
      round 1, R1-1: it had none for every state but ``live``).
    * a wedged record the address ALSO cannot reach keeps its own, truer word:
      "the daemon stopped reporting" says more than "nothing is answering".
    * ``stale`` is not probed at all — its pid is gone and the record is on its way
      to ``reaped/`` — so the shared state is the whole answer there.
    """
    live = _record(pid=1, instance_id="instance-one")
    deaf = _record(pid=2, instance_id="two")
    wedged_serving = _record(pid=3)
    wedged_deaf = _record(pid=5)
    stale = _record(pid=4)
    monkeypatch.setattr(
        services.serve_registry,
        "scan",
        lambda root=None: [
            (live, "live"),
            (deaf, "live"),
            (wedged_serving, "wedged"),
            (wedged_deaf, "wedged"),
            (stale, "stale"),
        ],
    )
    asked: list[int] = []

    def _probe(record: Any) -> services.AddressProbe:
        asked.append(record.pid)
        if record.pid in (2, 5):
            return services.AddressProbe(services.DEAF, "nothing answered")
        return services.AddressProbe(services.SERVING)

    reports = services.serve_daemon_reports(probe=_probe)
    assert [(report.record.pid, report.verdict) for report in reports] == [
        (1, services.SERVING),
        (2, services.DEAF),
        (3, services.SERVING),
        (5, services.WEDGED),
        (4, services.STALE),
    ]
    # ``wedged`` IS probed: its owner stopped reporting, but its address can still
    # have been taken by somebody else, and "your wedged daemon's port is now held
    # by a stranger" is a different thing to tell an operator than "nothing is
    # answering there". ``stale`` alone is not probed — its pid is gone and its
    # record is on its way to ``reaped/``.
    assert asked == [1, 2, 3, 5]


def test_the_spawn_contract_is_matched_as_words_not_a_substring() -> None:
    """A `grep` of the module name is not a serve daemon.

    The census this proves against reads the process table, where a person (or a
    test, or a rig) grepping for ``local_operator.cli serve`` looks exactly like
    the daemon to a substring match — and this command ends processes.
    """
    assert services.is_serve_command(SERVE_ARGV)
    assert services.is_serve_command("/Users/damian/.local/bin/lop serve --port 1111")
    assert not services.is_serve_command("grep -rn local_operator.cli serve src/")
    assert not services.is_serve_command("/bin/zsh -c python -m local_operator.cli services status")


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
        lambda *a, **k: [services.ServeDaemonReport(record=record, state="wedged", probe=None)],
    )
    rendered = "\n".join(services.status_lines())
    assert "serve daemons: none serving (1 recorded, not serving its address)" in rendered


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
    probe_script: list[str] | None = None,
    kill_lookup_error: bool = False,
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

    reports = (
        records
        if records is not None
        else [
            services.ServeDaemonReport(
                record=_record(pid=pid),
                state="live",
                probe=services.AddressProbe(
                    verdict, "" if verdict == services.SERVING else "silent"
                ),
            )
        ]
    )

    def _probe(_record: Any) -> services.AddressProbe:
        return services.AddressProbe(verdict, "" if verdict == services.SERVING else "silent")

    # ``probe_script`` is the same idea for the ADDRESS: a script of readings
    # consumed in order (last repeats), so a test can say "deaf, deaf, and then it
    # began serving" without a real listener.
    script = list(probe_script) if probe_script is not None else []

    def _scripted(_record: Any) -> services.AddressProbe:
        value = script.pop(0)
        if not script:
            script.append(value)
        return services.AddressProbe(value, "" if value == services.SERVING else "silent")

    def _kill(target: int, sig: int) -> None:
        if kill_lookup_error and sig == 9:
            # The daemon left between the last liveness read and the escalation.
            raise ProcessLookupError(target)
        kills.append((target, sig))

    outcome = services.reclaim_serve_daemon(
        pid,
        probe=_scripted if probe_script is not None else _probe,
        reports=lambda: reports,
        read_command=lambda _pid: command,
        read_uid=lambda _pid: (uid if uid is not None else os_mod.getuid()),
        kill=_kill,
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


# ---------------------------------------------------------------------------
# Review round 1. Each test below fails on the pre-remediation commit.
# ---------------------------------------------------------------------------


def test_a_wedged_owner_that_answers_as_itself_is_serving() -> None:
    """R1-1: the probe outranks the beat, so the "never end a working plane" guard
    is reachable for a daemon whose owner stopped reporting but is still serving.

    Before the fix ``ServeDaemonReport.verdict`` returned the shared state for every
    non-``live`` state and DISCARDED the probe it had paid for, so this record read
    ``wedged``, was signalled with 0 address probes and no confirmation, and its
    receipt claimed it was not serving.
    """
    wedged_but_serving = services.ServeDaemonReport(
        record=_record(pid=7), state="wedged", probe=services.AddressProbe(services.SERVING)
    )
    assert wedged_but_serving.verdict == services.SERVING
    outcome, kills = _reclaim(7, records=[wedged_but_serving])
    assert kills == []
    assert outcome.refused
    assert "IS the daemon serving" in "\n".join(outcome.lines)


def test_a_wedged_owner_that_answers_nothing_keeps_its_own_word() -> None:
    """R1-1's other half: the shared word survives where it says more."""
    wedged_deaf = services.ServeDaemonReport(
        record=_record(pid=8),
        state="wedged",
        probe=services.AddressProbe(services.DEAF, "nothing answered"),
    )
    assert wedged_deaf.verdict == services.WEDGED
    # And it is still actionable: the ADDRESS reading is the evidence that gets
    # confirmed, which is why a wedged record now costs probes it never used to.
    outcome, kills = _reclaim(8, records=[wedged_deaf], alive=[False])
    assert kills == [(8, 15)], "SIGTERM, once, after the address was confirmed"
    assert "ending wedged serve daemon on 127.0.0.1:1111" in "\n".join(outcome.lines)


def test_a_non_http_listener_is_classified_rather_than_raising() -> None:
    """R1-2: ``http.client.HTTPException`` is neither an ``OSError`` nor a ``ValueError``.

    A bare TCP listener (or a proxy speaking something else) on the port raises
    ``BadStatusLine`` out of the response parser. It used to escape ``probe_address``
    — so ``status_lines``, which made no network calls at all before this change,
    and ``reload_serve_daemons``, whose docstring promises NEVER RAISES, both died
    with a traceback instead of classifying the address.
    """
    import socket
    import threading

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def _serve_once() -> None:
        try:
            conn, _ = server.accept()
            conn.sendall(b"this is not HTTP at all\r\n\r\n")
            conn.close()
        finally:
            server.close()

    threading.Thread(target=_serve_once, daemon=True).start()
    probe = services.probe_address(_record(port=port))
    # SQUATTED, not DEAF (review round 2, R2-3): the port ANSWERED. Filing a talking
    # port under "nothing is answering there" is the same defect class as R1-3 one
    # bucket over, and catching the exception is what made it printable.
    assert probe.verdict == services.SQUATTED
    assert "answered with something that is not this product's health endpoint" in probe.detail
    # And the other direction, so the two buckets cannot collapse into one: a port
    # nobody is listening on is DEAF, and is the only arm that may say so.
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent_port = silent.getsockname()[1]
    silent.close()
    quiet = services.probe_address(_record(port=silent_port))
    assert quiet.verdict == services.DEAF
    assert "did not answer" in quiet.detail


def test_a_wrapped_mention_of_the_launcher_is_not_a_serve_daemon() -> None:
    """R1-8: the launcher form is adjacent, at argv[0] — the process IS `lop serve`.

    The stray path signals on this proof ALONE (no record of ours describes the
    pid), and matching the pair anywhere in argv accepted a shell wrapper or a
    ``grep`` that merely mentions the command.
    """
    assert services.is_serve_command("/Users/me/.local/bin/lop serve --port 1111")
    assert services.is_serve_command("/Users/me/.local/bin/local-operator serve")
    assert services.is_serve_command(SERVE_ARGV)
    assert not services.is_serve_command('/bin/zsh -c "lop serve --port 1111"')
    assert not services.is_serve_command("grep -rn lop serve src/")
    assert not services.is_serve_command("/Users/me/.local/bin/lop stop --all")


def test_a_live_but_deaf_daemon_prints_exactly_one_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1-6: one scan, one row per record.

    Two scans printed the incident's own daemon twice — an ordinary `serve daemon`
    row AND a stuck one — breaking the "exactly one line per record" property
    ``_fleet_action_lines`` documents and `grep`/`awk` counting relies on.
    """
    report = services.ServeDaemonReport(
        record=_record(pid=9),
        state="live",
        probe=services.AddressProbe(services.DEAF, "nothing answered"),
    )
    monkeypatch.setattr(services, "serve_daemon_reports", lambda **_kwargs: [report])
    # The OLD reader's seam, patched TOO. The duplicate this test exists for came
    # from `status_lines` reading `live_serve_daemons()` for the ordinary rows and a
    # second scan for the stuck ones, so a pin that patches only the new seam passes
    # on the pre-fix tree without ever reproducing it (review round 2, R2-5).
    monkeypatch.setattr(services, "live_serve_daemons", lambda: [report.record])
    monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
    lines = services.status_lines()
    assert sum(1 for line in lines if "pid 9" in line) == 1, (
        "one record, one row: an ordinary `serve daemon` row claims the daemon "
        "serves its address, and nothing answered, so the duplicate is the bug"
    )
    assert any("nothing is answering" in line for line in lines)
    assert any("lop services reclaim 9" in line for line in lines)


def test_a_serving_record_is_listed_as_serving_even_if_the_beat_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1-6's other face: the probe decides the ordinary row too."""
    report = services.ServeDaemonReport(
        record=_record(pid=10, version="0.62.24"),
        state="wedged",
        probe=services.AddressProbe(services.SERVING),
    )
    monkeypatch.setattr(services, "serve_daemon_reports", lambda **_kwargs: [report])
    monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
    lines = services.status_lines()
    assert any(line.startswith("serve daemon pid 10") for line in lines)
    assert not any("reclaim" in line for line in lines)


def test_an_unreadable_process_table_after_sigkill_is_doubt() -> None:
    """R1-7: three-valued liveness, and the third value is not a diagnosis.

    ``alive=None`` (unreadable) used to be folded into "STILL RUNNING — it is wedged
    in the kernel; a reboot is the only way left", which is a diagnosis made from a
    reading that never arrived.
    """
    outcome, kills = _reclaim(alive=[None])
    assert len(kills) == 2, "the escalation still runs: doubt is not a reason to stop"
    assert outcome.problem == "unreadable-after-signal"
    text = "\n".join(outcome.lines)
    assert "DOUBT" in text
    assert "STILL RUNNING" not in text


def test_a_daemon_that_begins_serving_is_refused_at_the_last_reading() -> None:
    """R1-5: the last reading before the signal is an ADDRESS reading.

    The brand re-read cannot see a daemon that starts answering between the
    confirmation and the signal — the recovery this command must not punish.
    """
    outcome, kills = _reclaim(probe_script=[services.DEAF, services.DEAF, services.SERVING])
    assert kills == []
    assert outcome.problem == "serving"
    assert "began answering" in "\n".join(outcome.lines)


def test_a_stray_that_is_serving_its_own_plane_says_so() -> None:
    """R1-4: the stray arm ASKS the address instead of claiming nothing answers.

    A rig's stray can perfectly well be serving a plane of its own — its records
    are simply not this install's — and that is a different thing to tell an
    operator than "it is not serving that address", which nothing had earned.
    """
    command = "/opt/rig/bin/local-operator serve --host 127.0.0.1 --port 11331"
    outcome, kills = _reclaim(
        pid=555, command=command, records=[], probe_script=[services.SQUATTED], alive=[False]
    )
    assert len(kills) == 1
    text = "\n".join(outcome.lines)
    assert "under records that are not this install's" in text
    assert outcome.verdict == services.STRAY


def test_a_stray_is_refused_when_the_records_address_is_served() -> None:
    """R1-5's fail-closed face, in the claimant arm."""
    claimant = services.ServeDaemonReport(
        record=_record(pid=600, host="127.0.0.1", port=11331),
        state="live",
        probe=services.AddressProbe(services.SERVING),
    )
    outcome, kills = _reclaim(
        pid=555,
        command="/opt/rig/bin/local-operator serve --host 127.0.0.1 --port 11331",
        records=[claimant],
        probe_script=[services.SERVING],
    )
    assert kills == []
    assert outcome.problem == "serving"
    assert "IS being served" in "\n".join(outcome.lines)


def test_every_row_verdict_has_one_sentence_and_one_adjective() -> None:
    """N1: ONE vocabulary, and it is read with ``[]`` so a missing row is loud."""
    assert set(services.VERDICTS) == {
        services.DEAF,
        services.SQUATTED,
        services.WEDGED,
        services.STALE,
        services.STRAY,
    }
    assert (
        services.SERVING not in services.VERDICTS
    ), "a serving daemon gets the ordinary `serve daemon` row, not a stuck one"
    for entry in services.VERDICTS.values():
        assert "{address}" in entry.row
        assert entry.phrase and " " not in entry.phrase


def test_the_update_warning_reuses_the_axis_sentence_and_real_remedies() -> None:
    """R1-3: one sentence for both surfaces, and a remedy only where one exists.

    The update path's first version typed a second sentence, and it said "nothing
    is answering there" about a SQUATTED address (something was), told the reader to
    reclaim a STALE record's pid (which is gone), and printed the raw verdict token.
    """
    squatted = services.ServeDaemonReport(
        record=_record(pid=11),
        state="live",
        probe=services.AddressProbe(services.SQUATTED, "answered as somebody else", "other"),
    )
    text = "\n".join(services.stuck_report_lines(squatted))
    assert "what answers there is not this record's daemon" in text
    assert "nothing is answering" not in text
    assert "lop services reclaim 11" in text

    stale = services.ServeDaemonReport(record=_record(pid=12), state="stale")
    text = "\n".join(services.stuck_report_lines(stale))
    assert "its process has exited" in text
    assert "reclaim" not in text, "there is no pid left to reclaim"


def test_an_address_that_changes_occupant_is_refused() -> None:
    """R2-6: the `changed` refusal — the one gate that can refuse a legitimate reclaim.

    A deaf address that becomes a different kind of unserved address while the
    command is confirming is not the address the verdict was measured on. The
    operator is told which two readings disagreed and told to run it again, rather
    than being sent into the signal on a verdict the address no longer supports.
    """
    outcome, kills = _reclaim(probe_script=[services.DEAF, services.DEAF, services.SQUATTED])
    assert kills == []
    assert outcome.problem == "changed"
    assert "changed while this was being confirmed" in "\n".join(outcome.lines)


def test_a_stale_record_is_refused_by_the_allow_list() -> None:
    """R2-6: `not-actionable` — a state whose remedy is not this command.

    ``stale`` means the pid is gone: the brand re-read would refuse it a moment
    later, and the allow-list says so up front instead of relying on that.
    """
    stale = services.ServeDaemonReport(record=_record(pid=13), state="stale")
    outcome, kills = _reclaim(13, records=[stale])
    assert kills == []
    assert outcome.problem == "not-actionable"
    assert "nothing was sent" in "\n".join(outcome.lines)


def test_a_pid_that_leaves_before_the_escalation_is_still_ended() -> None:
    """R2-6: SIGKILL's ``ProcessLookupError`` is the outcome asked for, not a crash.

    The daemon can exit between the last liveness read and the escalation signal;
    ``os.kill`` then raises, and letting that out would report a successful reclaim
    as a traceback.
    """
    outcome, kills = _reclaim(alive=[True], kill_lookup_error=True)
    assert kills == [(4242, 15)], "SIGTERM only: the escalation found no process"
    assert outcome.acted
    assert "ended on SIGKILL" in "\n".join(outcome.lines)


def test_the_launcher_proof_survives_the_procname_label_and_the_app_entrypoint() -> None:
    """R3-2/R3-1: the launcher arm, pinned by inputs that do NOT carry the marker.

    Every other fixture in this file uses ``SERVE_ARGV``, which contains
    ``-m local_operator.cli serve`` — so the MARKER arm answers them and the
    launcher arm could be reverted to ``argv[0]``-only with the suite still green
    (review round 3, R3-2, measured: 39/39 green after reverting it). These three
    inputs can only be accepted by the launcher and entry-point arms:

    * the real labelled daemon shape, measured on this host with ``ps -o command=``
      — ``procname`` REPLACES argv[0], so the launcher is four words in;
    * the desktop app's managed backend — ``<interpreter> -c "<entrypoint>" serve``
      — which is deliberately never branded and which ``reclaim`` refused until
      this round (R3-1: it is the daemon that held 1111 in the incident).
    """
    labelled = (
        "Local Operator [serve] port=18490 /Users/damian/.local/bin/local-operator serve "
        "--host 127.0.0.1 --port 18490"
    )
    app_backend = (
        "/Users/me/.local/share/lop/generations/2026/tools/local-operator/bin/python "
        "-c from local_operator.cli import main; main() serve --port 1111"
    )
    assert services.is_serve_command(labelled)
    assert services.is_serve_command(app_backend)
    assert services.is_serve_command("/Users/me/.local/bin/lop serve")
    # And the false positives R1-8 closed stay closed.
    assert not services.is_serve_command('/bin/zsh -c "lop serve --port 1111"')
    assert not services.is_serve_command("grep -rn lop serve src/")
    # A `-c` that merely PRINTS the entry point is not a daemon, and neither is the
    # app's own identity probe (which imports the same module for a different job).
    assert not services.is_serve_command('/usr/bin/python3 -c "print(1)" serve')
    assert not services.is_serve_command(
        "/usr/bin/python3 -c import json, sys; from local_operator.cli import main;"
        " print(json.dumps([sys.executable]))"
    )
    assert not services.is_serve_command(
        "/Users/me/.local/share/lop/generations/2026/tools/local-operator/bin/python "
        "-c from local_operator.cli import main; main() --version"
    )
    # A HOME WITH A SPACE IN IT (review round 4, R4-1): the interpreter is reached by
    # a home-derived path, so it can be two `ps` words — and the round-3 version,
    # which assumed `words[1]` was the `-c`, refused the app's own backend for exactly
    # those users.
    assert services.is_serve_command(
        "/Users/John Doe/.local/share/lop/generations/2026/tools/local-operator/bin/python "
        "-c from local_operator.cli import main; main() serve --port 1111"
    )
    # And the app's pre-exec wrapper, which the search admits: no nameable pid is
    # ever that bash (the plan execs), and its argv still carries the entry point.
    assert services.is_serve_command(
        'bash -c exec "$@" owned-serve /Users/me/.local/bin/python '
        "-c from local_operator.cli import main; main() serve --port 1111"
    )


def test_the_wait_answers_from_its_last_read() -> None:
    """R3-2/N-4: the three-valued wait describes its FINAL read, not a blip.

    Readings ``unreadable, unreadable, still there`` — the process table blipped for
    both polls and then answered that the process is running. The wait must answer
    "still there". The round-1 version kept a sticky "something was unreadable" flag,
    so this exact input returned DOUBT and the report printed "the process table
    could not be read" over a reading that had already answered.
    """

    def _scripted(values: list[bool | None]):
        queue = list(values)

        def _read(_pid: int) -> bool | None:
            value = queue.pop(0)
            if not queue:
                queue.append(value)
            return value

        return _read

    blipped = services._await_exit(4242, _scripted([None, None, True]), _no_sleep, 0.001, 0.0005)
    assert blipped is False, "the last read said still-there, so the answer is still-there"
    # And the other two answers still come through, so this is three-valued rather
    # than merely "never None".
    assert services._await_exit(4242, _scripted([None, False]), _no_sleep, 0.001, 0.0005) is True
    assert services._await_exit(4242, _scripted([None]), _no_sleep, 0.001, 0.0005) is None
