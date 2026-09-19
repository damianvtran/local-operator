"""Bringing the non-runtime fleet onto the current build.

The assertions here are about the two ways this command can do HARM, which is why
they are so specific:

* **Never signal a daemon that cannot take it.** The request is ``SIGUSR1``, whose
  default disposition is to terminate, so a daemon built before the capability
  existed would be KILLED by a well-meaning update. The record's ``reloadable``
  field is the guard, and one of these tests exists to keep it a guard.
* **Never claim a move that did not happen.** ``instance_id`` is minted once per
  process, so a changed one PROVES the daemon was replaced; the version does not
  (this host's ordinary update is a same-version rebuild). Reading the version
  here would report success for a daemon that never moved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator import services
from local_operator.server import registry as serve_registry
from local_operator.server.reload import RELOAD_SIGNAL
from local_operator.update import BuildStamp

NEW = BuildStamp(version="0.59.0", source_ref="")
OLD = BuildStamp(version="0.56.14", source_ref="")


def _record(**over: Any) -> serve_registry.ServeRecord:
    fields: dict[str, Any] = {
        "pid": 4242,
        "host": "127.0.0.1",
        "port": 1111,
        "instance_id": "instance-one",
        "version": OLD.version,
        "source_ref": OLD.source_ref,
        "prefix": "/opt/gen/old/tools/local-operator",
        "install_kind": "uv-tool",
        "desktop": True,
        "reloadable": True,
    }
    fields.update(over)
    return serve_registry.ServeRecord(**fields)


@pytest.fixture
def pointer(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """What the install pointer names. The SUBJECT of every comparison below."""
    state: dict[str, Any] = {"stamp": NEW}
    monkeypatch.setattr("local_operator.update.disk_build", lambda *_a, **_k: state["stamp"])
    return state


def _run(
    records: list[Any],
    *,
    kills: list[tuple[int, int]],
    after: dict[int, Any] | None = None,
    wait_s: float = 1.0,
    clock: list[float] | None = None,
    probe: Any = None,
) -> list[services.ServiceRefresh]:
    """Drive one reload pass with every seam injected.

    ``scan`` answers differently AFTER the signal, which is how a real
    replacement announces itself: the same pid, a new ``instance_id``.

    ``probe`` defaults to ASSENT, because the identity check is its own test and
    every other test here is about something else: without that default each of
    them would open a real loopback connection to the port in its own fixture and
    fail for a reason that has nothing to do with what it asserts.
    """
    after = after or {record.pid: record for record in records}
    ticks = clock if clock is not None else [0.0]
    answers = probe if probe is not None else (lambda record: None)

    def scan() -> list[Any]:
        return records if not kills else list(after.values())

    def kill(pid: int, sig: int) -> None:
        kills.append((pid, sig))

    def sleep(_seconds: float) -> None:
        ticks[0] += 0.05

    return services.reload_serve_daemons(
        wait_s=wait_s,
        sleep=sleep,
        scan=scan,
        kill=kill,
        probe=answers,
        monotonic=lambda: ticks[0],
    )


def test_an_unproven_process_is_never_signalled(pointer: dict[str, Any]) -> None:
    """THE IDENTITY GUARD (serve-reload review round 1, R1-4).

    A record is a file whose name is a pid, and a pid is recycled: review
    constructed a live `sleep` named by a hand-written live, reloadable record and
    this command killed it (rc -30). Nothing is signalled unless the address
    answers AS the instance the record names — and the refusal is a warning that
    says nothing was sent, not a silence.
    """
    record = _record()
    kills: list[tuple[int, int]] = []
    out = _run(
        [record],
        kills=kills,
        wait_s=0.0,
        probe=lambda _record: "127.0.0.1:1111 is answering as nobody, not the instance one…",
    )
    assert kills == []
    warnings = [warning for refresh in out for warning in refresh.warnings]
    assert any("was not asked to reload" in warning for warning in warnings)
    assert any("Nothing was signalled" in warning for warning in warnings)


def test_the_identity_check_is_a_real_health_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default probe reads /health and compares `instance_id`, not the port."""
    record = _record(port=61111, host="127.0.0.1", instance_id="expected-instance")

    class _Response:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"status":200,"result":{"instance_id":"a-different-process"}}'

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=0: _Response())
    mismatch = services._answers_as_record(record)
    assert mismatch is not None and "a-different-process" in mismatch

    class _Matching(_Response):
        def read(self) -> bytes:
            return b'{"status":200,"result":{"instance_id":"expected-instance"}}'

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=0: _Matching())
    assert services._answers_as_record(record) is None


def test_a_missing_stamp_moves_nothing_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """serve-reload R2-1's fence: a comparison against an absent right-hand side is not a
    verdict.

    With no readable stamp, `_serves_current_build` answers False for EVERY
    daemon — so without this guard the whole fleet looks stale to a caller that
    has no build to move anything onto. Review demonstrated it end to end with a
    fabricated-root daemon, and the reachable caller is a source checkout, whose
    `disk_build()` is None.
    """
    monkeypatch.setattr("local_operator.update.disk_build", lambda *_a, **_k: None)
    kills: list[tuple[int, int]] = []
    records = [_record(pid=1), _record(pid=2)]
    out = services.reload_serve_daemons(
        scan=lambda: records, kill=lambda pid, sig: kills.append((pid, sig))
    )
    assert kills == []
    assert len(out) == 1
    assert "NOTHING was signalled" in out[0].warnings[0]


def test_the_probe_url_brackets_an_ipv6_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bracket join is load-bearing and was untested (serve-reload review round 2, R2-6).

    `http://::1:1111/health` is not a URL: an IPv6 literal must be bracketed, and
    a probe that cannot parse its own URL is a probe that answers "did not
    identify itself" for a daemon that is answering perfectly.
    """
    import json as json_mod

    seen: list[str] = []

    class _Response:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def read(self) -> bytes:
            return json_mod.dumps({"result": {"instance_id": "v6"}}).encode()

    def _open(url: str, timeout: float = 0) -> Any:
        seen.append(url)
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _open)
    assert services._answers_as_record(_record(host="::1", port=1111, instance_id="v6")) is None
    assert seen == ["http://[::1]:1111/health"]


def test_a_daemon_already_on_the_current_build_is_left_alone(pointer: dict[str, Any]) -> None:
    """Nothing to move, so nothing is interrupted — no signal, no stream cut."""
    record = _record(version=NEW.version, source_ref=NEW.source_ref)
    kills: list[tuple[int, int]] = []
    out = _run([record], kills=kills)
    assert kills == []
    assert out[0].lines == (f"serve pid 4242 is already on {NEW.label()}",)


def test_a_daemon_that_cannot_reload_is_reported_and_never_signalled(
    pointer: dict[str, Any],
) -> None:
    """THE guard. ``SIGUSR1`` kills a process that never installed a handler."""
    record = _record(reloadable=False)
    kills: list[tuple[int, int]] = []
    out = _run([record], kills=kills, wait_s=0.0)
    assert kills == []
    assert len(out) == 1
    assert out[0].lines == ()
    # The sentence names a REASON, and which reason depends on the platform — see the
    # sibling test below, which is what covers the no-SIGUSR1 arm. Asserting the
    # POSIX-only wording here made this test fail on a platform that has no such
    # signal (serve-reload review round 9, R9-1).
    assert "cannot move itself" in out[0].warnings[0]
    assert "restart that server by hand" in out[0].warnings[0]


def test_a_platform_without_the_signal_still_reports_every_daemon(
    monkeypatch: pytest.MonkeyPatch,
    pointer: dict[str, Any],
) -> None:
    """serve-reload R8-1's arm, with the only coverage it has.

    On a platform with no SIGUSR1 the capability is absent for EVERY daemon, so the
    whole fleet looks unmovable. The early return that used to replace the per-daemon
    report with a single sentence is gone: the operator must still be told which
    daemons are running and what each needs, and the reason must be the true one —
    the platform, not "it predates in-place reload".
    """
    monkeypatch.setattr(services.serve_reload, "RELOAD_SIGNAL", None)
    stale = _record()
    already = _record(version=NEW.version, source_ref=NEW.source_ref, instance_id="instance-two")
    kills: list[tuple[int, int]] = []
    out = _run([stale, already], kills=kills, wait_s=0.0)
    assert kills == [], "nothing may be signalled where the signal does not exist"
    warnings = " ".join(warning for refresh in out for warning in refresh.warnings)
    assert "this platform has no SIGUSR1" in warnings
    lines = " ".join(line for refresh in out for line in refresh.lines)
    assert "is already on" in lines, "the daemons that need nothing must still be reported"
    assert any("cannot move itself" in w for refresh in out for w in refresh.warnings)


def test_a_stale_daemon_is_asked_and_reported_once_it_republishes(pointer: dict[str, Any]) -> None:
    """The happy path, and the reason ``instance_id`` is the proof."""
    record = _record()
    replacement = _record(
        version=NEW.version, source_ref=NEW.source_ref, instance_id="instance-two"
    )
    kills: list[tuple[int, int]] = []
    out = _run([record], kills=kills, after={record.pid: replacement})
    assert kills == [(record.pid, RELOAD_SIGNAL)]
    assert any("is now serving 0.59.0" in line for refresh in out for line in refresh.lines)


def test_a_same_version_rebuild_is_not_read_as_a_no_op(pointer: dict[str, Any]) -> None:
    """This host's ORDINARY update: same version, new ``source_ref``.

    A version-only comparison would call the daemon current and never move it,
    which is the whole failure this feature exists to fix.
    """
    pointer["stamp"] = BuildStamp(version="0.59.0", source_ref="d808fc67")
    record = _record(version="0.59.0", source_ref="")
    kills: list[tuple[int, int]] = []
    _run([record], kills=kills, wait_s=0.0)
    assert kills == [(record.pid, RELOAD_SIGNAL)]


def test_a_daemon_that_never_comes_back_is_a_warning_not_a_failure(pointer: dict[str, Any]) -> None:
    """The install already succeeded; a nudge that did not land is a warning."""
    record = _record()
    kills: list[tuple[int, int]] = []
    out = _run([record], kills=kills, after={record.pid: record}, wait_s=0.25)
    warnings = [warning for refresh in out for warning in refresh.warnings]
    assert kills == [(record.pid, RELOAD_SIGNAL)]
    assert any("did not come back on the current build" in warning for warning in warnings)
    assert any("lop services restart" in warning for warning in warnings)


def test_a_daemon_that_vanishes_while_waiting_is_not_reported_as_moved(
    pointer: dict[str, Any],
) -> None:
    """A pid that is gone may have exited for its own reasons; that is not a move."""
    record = _record()
    kills: list[tuple[int, int]] = []
    out = _run([record], kills=kills, after={}, wait_s=0.25)
    assert all("is now serving" not in line for refresh in out for line in refresh.lines)


def test_one_deadline_covers_the_whole_fleet(pointer: dict[str, Any]) -> None:
    """Three daemons must not cost three budgets: the wait is shared.

    Serialising the waits would add their drains together and report the last
    daemon's answer as if it were the fleet's.
    """
    records = [_record(pid=pid) for pid in (11, 22, 33)]
    kills: list[tuple[int, int]] = []
    ticks = [0.0]
    out = _run(
        records,
        kills=kills,
        after={record.pid: record for record in records},
        wait_s=0.2,
        clock=ticks,
    )
    assert len(kills) == 3
    assert ticks[0] <= 0.2 + 0.05
    assert sum(len(refresh.warnings) for refresh in out) == 3


def test_live_serve_daemons_drops_everything_that_is_not_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``wedged`` is a stuck process, and a stuck process is not signalled."""
    live, wedged, stale = _record(pid=1), _record(pid=2), _record(pid=3)
    monkeypatch.setattr(
        serve_registry,
        "scan",
        lambda root=None: [(wedged, "wedged"), (stale, "stale"), (live, "live")],
    )
    assert [record.pid for record in services.live_serve_daemons()] == [1]


def test_status_lines_name_the_drift_and_the_capability(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader is an operator who was just told "older build than the install".

    BOTH OPERANDS ON EVERY LINE (design review D1). The version alone made the
    ordinary case — a same-version rebuild — read as ``STALE (0.59.2)`` beside an
    install that was also 0.59.2, with no ref anywhere to explain the verdict.
    """
    # The pointer names a build WITH a ref, so the ref half of every comparison is
    # visible in the assertions below.
    pointer["stamp"] = BuildStamp(version="0.59.0", source_ref="4d3ce1d")
    monkeypatch.setattr(
        services,
        "live_serve_daemons",
        lambda: [
            _record(source_ref="9f2c1ab"),
            _record(pid=9, version="0.59.0", source_ref="4d3ce1d", reloadable=False),
        ],
    )
    monkeypatch.setattr(
        services,
        "_supervised_daemon_plists",
        lambda: [Path("/tmp/com.local-operator.mobile.plist")],
    )
    # A checkout's own guard refuses the move (`install_kind` is EDITABLE here), so the
    # promise sentences are asserted against a caller the guard WOULD permit.
    monkeypatch.setattr(services, "_install_may_repoint_daemons", lambda: True)
    lines = services.status_lines()
    assert "install: 0.59.0@4d3ce1d" in lines, "the build the reader is comparing against"
    # Stale: identity, then the verdict leading an indented line, then what to DO about
    # it — not the word "reloadable", which is not reader vocabulary. Three lines
    # because folding them into one took the line to 137 columns (D6).
    at = lines.index("serve daemon pid 4242 on 127.0.0.1:1111")
    assert lines[at + 1] == "  STALE: serving 0.56.14@9f2c1ab, current is 0.59.0@4d3ce1d"
    assert lines[at + 2] == "  `lop services restart` will move it"
    assert "serve daemon pid 9 on 127.0.0.1:1111 — current (0.59.0@4d3ce1d)" in lines
    assert "supervised daemon: com.local-operator.mobile" in lines
    # D3: their build is not in the plist, and the reader is told so rather than left
    # to assume it is current.
    assert any("resolves the install when it starts" in line for line in lines)
    assert any("run `lop services restart`" in line for line in lines)
    # D6: the longest line in the whole output must fit a plain 80-column terminal —
    # the first version of this fix reached 137 and soft-wrapped mid-word.
    longest = max(len(line) for line in lines)
    assert longest <= 80, f"longest line is {longest}: {max(lines, key=len)!r}"


def test_status_lines_name_the_true_reason_a_daemon_cannot_move(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daemon that cannot be asked is told apart from one that can (D1)."""
    monkeypatch.setattr(services, "live_serve_daemons", lambda: [_record(reloadable=False)])
    monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
    lines = services.status_lines()
    assert "  cannot move itself; restart it by hand" in lines


def test_status_lines_bracket_an_ipv6_authority(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``::1:56569`` is ambiguous, and past the point the eye finds the boundary (D4)."""
    monkeypatch.setattr(services, "live_serve_daemons", lambda: [_record(host="::1")])
    monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
    line = next(line for line in services.status_lines() if line.startswith("serve daemon"))
    assert "on [::1]:1111" in line


def test_status_lines_says_it_cannot_compare_when_the_stamp_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D1's fix must not turn an unanswerable comparison into a tautology.

    ``_label(None)`` falls back to a phrase, so the first version printed "current is
    the current build" — a sentence that reads like an answer on the one line whose job
    is to name both operands. It also advised `lop services restart` when there is no
    build to move anything onto, which is advice the tool cannot carry out.

    This is not a corner case: it is what a DEVELOPER sees, because a checkout's own
    install on disk is its working tree and ``disk_build()`` is None there by design.
    """
    monkeypatch.setattr("local_operator.update.disk_build", lambda *a, **k: None)
    monkeypatch.setattr(services, "live_serve_daemons", lambda: [_record()])
    monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
    lines = services.status_lines()
    assert lines[0] == "install: no build the pointer can name, so nothing can be compared"
    # D13: the per-daemon line does not restate that reason — it says what the daemon
    # serves and leaves the WHY to the install line directly above it.
    assert "  serving 0.56.14" in lines
    assert not any(
        "pointer names no build" in line for line in lines[1:]
    ), "D13: the reason lives on the install line, not restated per daemon"
    joined = " ".join(lines)
    assert "current is the current build" not in joined
    assert "run `lop services restart`" not in joined
    # D7: the supervised note carries the SAME promise as that action line, and it was
    # the clause left ungated — `update._repair_refusal` refuses an editable caller, so
    # on a checkout `restart` moves none of these and the note was a lie.
    assert not any("puts them on the current build" in line for line in lines)
    monkeypatch.setattr("local_operator.update.disk_build", lambda *a, **k: NEW)
    monkeypatch.setattr(
        services,
        "_supervised_daemon_plists",
        lambda: [Path("/tmp/com.local-operator.mobile.plist")],
    )
    monkeypatch.setattr(services, "_install_may_repoint_daemons", lambda: True)
    backed = services.status_lines()
    assert any("puts them on the current build" in line for line in backed)


def test_services_defines_what_a_service_is() -> None:
    """D9: the page a reader opens to find out what a "service" is defined nothing."""
    from local_operator import cli

    parser = cli.build_cli_parser()
    subparsers = next(
        action for action in parser._actions if getattr(action, "dest", None) == "subcommand"
    )
    choices: dict[str, Any] = getattr(subparsers, "choices", {})
    text: str | None = choices["services"].description
    assert text, "the group must describe itself"
    assert "not a conversation" in text
    assert "serve" in text and "runtimes" in text.lower()


def test_the_fleet_advice_is_not_printed_when_restart_would_skip_them(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The advice must not name a command that skips the daemons it is about.

    ``reloadable`` defaults to False and records written before the capability existed
    do not carry it, so "every stale daemon is unmovable" is the COMMON case rather
    than an edge one (design review D11, round 11 R11-2 — the same defect D7 fixed one
    line over).
    """
    monkeypatch.setattr(services, "live_serve_daemons", lambda: [_record(reloadable=False)])
    monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
    monkeypatch.setattr(services, "_install_may_repoint_daemons", lambda: True)
    lines = services.status_lines()
    assert "the stale daemon cannot be moved by `lop services restart`; restart it by hand" in lines
    assert not any("run `lop services restart`" in line for line in lines)


def test_the_fleet_advice_counts_the_two_kinds_apart(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mixed fleet is told which half the verb will move and which it will not."""
    monkeypatch.setattr(
        services,
        "live_serve_daemons",
        lambda: [_record(source_ref="a"), _record(pid=3, source_ref="b", reloadable=False)],
    )
    monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
    monkeypatch.setattr(services, "_install_may_repoint_daemons", lambda: True)
    lines = services.status_lines()
    assert "run `lop services restart` for the 1 it can move; 1 needs restarting by hand" in lines


def test_a_guard_that_refuses_silences_every_promise(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 11 R11-3: the guard has TWO questions, and both gate the promises.

    Keying on the stamp alone covered only "is this an installation at all"; a durable
    install that is not the one the plists run is refused by the second question, and
    both sentences were still printed there.
    """
    monkeypatch.setattr(services, "live_serve_daemons", lambda: [_record()])
    monkeypatch.setattr(
        services,
        "_supervised_daemon_plists",
        lambda: [Path("/tmp/com.local-operator.browser.plist")],
    )
    # The guard says NO to the plist repoint. It says nothing about the mobile bounce,
    # which happens whatever this caller is (measured by bouncing the real daemon from
    # a checkout), nor about the serve half, which never consults the guard.
    monkeypatch.setattr(services, "_install_may_repoint_daemons", lambda: False)
    # Only BROWSER is installed here, so nothing bounces: `refresh_mobile_after_upgrade`
    # returns without a bounce when there is no mobile plist, and naming a bounce would
    # over-warn about a daemon the reader does not have (round 13 R13-1, design D21).
    lines = services.status_lines()
    joined = " ".join(lines)
    assert "puts them on the current build" not in joined
    assert not any("bounces the mobile relay" in line for line in lines)
    assert any("repoints them only from the" in line for line in lines)

    # With the mobile plist present the bounce DOES happen whatever this caller is
    # (measured by bouncing the real daemon from a checkout), so the note says so.
    monkeypatch.setattr(
        services,
        "_supervised_daemon_plists",
        lambda: [
            Path("/tmp/com.local-operator.mobile.plist"),
            Path("/tmp/com.local-operator.browser.plist"),
        ],
    )
    lines = services.status_lines()
    assert any("still bounces the mobile relay" in line for line in lines)
    assert any("repointed only by the install that owns them" in line for line in lines)

    monkeypatch.setattr(services, "_install_may_repoint_daemons", lambda: True)
    permitted = " ".join(services.status_lines())
    assert "puts them on the current build" in permitted

    # And the serve-daemon advice is NOT gated on the guard: `reload_serve_daemons`
    # never asks it, so a durable second install gets correct advice even though the
    # plist repoint is refused.
    monkeypatch.setattr(services, "_install_may_repoint_daemons", lambda: False)
    assert any("run `lop services restart`" in line for line in services.status_lines())


def test_no_line_exceeds_the_budget_with_a_long_authority(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 11 R11-1: the joined `— current` line overflowed on a long IPv6 host.

    A link-local address is 25 characters of authority, which took
    `serve daemon pid N on [host]:port — current (X@y)` past 80 while every branch
    D6 measured stayed inside it — and the `longest <= 80` test only covered
    ``127.0.0.1``, so it could not catch that. The joined form is now width-checked.
    """
    pointer["stamp"] = BuildStamp(version="0.59.0", source_ref="4d3ce1d")
    for host in ("127.0.0.1", "::1", "fe80::1c2d:3e4f:5a6b:7c8d", "2001:db8:85a3::8a2e:370:7334"):
        monkeypatch.setattr(
            services,
            "live_serve_daemons",
            lambda host=host: [
                _record(host=host, version="0.59.0", source_ref="4d3ce1d"),
                _record(pid=5, host=host, version="0.56.14", source_ref="9f2c1ab"),
            ],
        )
        monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
        monkeypatch.setattr(services, "_install_may_repoint_daemons", lambda: True)
        lines = services.status_lines()
        longest = max(len(line) for line in lines)
        assert longest <= 80, f"{host}: {longest} columns: {max(lines, key=len)!r}"


def test_the_predicate_body_is_the_guard_and_only_the_guard(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Code round 12, R12-4: the other tests replace this wholesale, so its BODY was
    untested — reverting it to a stamp-only check would leave them all green.

    It must be exactly the repair guard's question: a durable install that is not the
    one the plists run is refused (question 2), and an unanswerable guard is not
    permission.
    """
    from local_operator import update

    # NO POINTER, which is the only state where the old stamp-only body and this one
    # differ — with `pointer`'s stamp pinned to NEW they agree, so the test passed
    # against the reverted body (round 13, R13-2).
    pointer["stamp"] = None
    monkeypatch.setattr(update, "_repair_refusal", lambda: None)
    assert services._install_may_repoint_daemons() is True

    monkeypatch.setattr(update, "_repair_refusal", lambda: "a second tool env")
    assert services._install_may_repoint_daemons() is False

    def _explode() -> str | None:
        raise RuntimeError("unreadable")

    monkeypatch.setattr(update, "_repair_refusal", _explode)
    assert services._install_may_repoint_daemons() is False


def test_no_pointer_never_calls_the_fleet_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Design review D16: the fallback classified the fleet on no evidence.

    On a checkout — `disk_build()` is None by design, so this is the DEFAULT developer
    render — the line called the daemons "the stale ones" and prescribed a HAND restart
    while the install line said the comparison was impossible. Nothing in that render
    was labelled STALE:, and the action prescribed is the destructive one: a hand
    restart drops the runtimes the in-place reload exists to keep.
    """
    monkeypatch.setattr("local_operator.update.disk_build", lambda *a, **k: None)
    monkeypatch.setattr(services, "live_serve_daemons", lambda: [_record()])
    monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
    lines = services.status_lines()
    assert "no build to compare against, so no serve daemon can be moved automatically" in lines
    joined = " ".join(lines)
    assert "the stale ones" not in joined
    assert "restart them by hand" not in joined


def test_the_hand_restart_count_agrees_with_itself(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Code round 12, R12-3: `1 need restarting by hand` is right at two, wrong at one."""
    pointer["stamp"] = BuildStamp(version="0.59.0", source_ref="4d3ce1d")
    monkeypatch.setattr(
        services,
        "live_serve_daemons",
        lambda: [
            _record(source_ref="a"),
            _record(pid=4, source_ref="b", reloadable=False),
        ],
    )
    monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
    assert (
        "run `lop services restart` for the 1 it can move; 1 needs restarting by hand"
        in services.status_lines()
    )


def test_status_lines_says_so_when_there_are_no_daemons(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(services, "live_serve_daemons", lambda: [])
    monkeypatch.setattr(services, "_supervised_daemon_plists", lambda: [])
    assert services.status_lines() == ["install: 0.59.0", "serve daemons: none running"]


def test_the_documented_wait_default_is_the_one_used(
    pointer: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--help`` must not name a default the tool does not use (design review D2).

    The literal is duplicated because ``cli.py`` may not import this module (its
    startup path is asserted stdlib-light and this one reaches asyncio), so this test
    IS the guard that keeps the two from drifting.
    """
    from local_operator import cli

    assert cli.DEFAULT_SERVICES_WAIT_S == services.RELOAD_WAIT_S


def test_a_non_positive_wait_is_a_usage_error() -> None:
    """``--wait -1`` ran and reported "within -1s"; ``--wait 0.5`` said "within 0s" (D2).

    Refused by the PARSER, before anything is touched, so this asserts on
    ``build_cli_parser`` rather than on ``main`` (which brands the process and
    configures logging as side effects).
    """
    from local_operator import cli

    parser = cli.build_cli_parser()
    for value in ("-1", "0", "-0.5"):
        with pytest.raises(SystemExit) as caught:
            parser.parse_args(["services", "restart", "--wait", value])
        assert caught.value.code == 2, value
    # A positive one is accepted, and the documented default is the one used.
    assert parser.parse_args(["services", "restart", "--wait", "0.5"]).wait == 0.5
    assert parser.parse_args(["services", "restart"]).wait is None
