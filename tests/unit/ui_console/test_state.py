"""The console capability, read off the app host's shared discovery record.

Every test here is about the ONE thing this module adds to the browser host's
record: the ``console`` capability bit, which is what the tool's `createIf` gate
reads. The file, the modes, the heartbeat and the three-state classifier belong
to the browser host's module and are tested there; what is tested here is that
the console's reader agrees with them (same path, same modes) and that the
capability clause does not become a second, divergent liveness rule.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from local_operator.browser_bridge.protocol import PROTO_VERSION
from local_operator.ui_browser import state as host_state
from local_operator.ui_console import state


def record(**updates: object) -> state.ConsoleHostState:
    values: dict[str, object] = {
        "pid": os.getpid(),
        "port": 52133,
        "session_key": "k" * 32,
        "proto": PROTO_VERSION,
        "app_version": "0.26.0",
        "console": True,
        "console_surfaces": 2,
        "console_agent_surfaces": 1,
    }
    values.update(updates)
    return state.ConsoleHostState.model_validate(values)


def test_the_console_reads_the_apps_own_record(tmp_path: Path) -> None:
    """One endpoint, one record, one key (design §10.1).

    A second discovery file would mean a second key, a second heartbeat and a
    second set of the four loopback safety rules to keep true, for a lifecycle
    that does not differ — same process, same writer, same teardown. So the
    console's namespace is a MODEL, and the path must be the app host's own.
    """
    assert state.state_path(tmp_path) == host_state.state_path(tmp_path)
    assert state.RUN_DIRNAME == host_state.RUN_DIRNAME
    assert state.STATE_FILENAME == host_state.STATE_FILENAME


def test_the_record_is_private_and_atomic_like_the_hosts(tmp_path: Path) -> None:
    target = state.publish(record(), tmp_path)
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700


def test_a_read_creates_nothing(tmp_path: Path) -> None:
    """`state_path` is pure arithmetic and `read` must not initialise a file.

    The app owns this record's lifecycle; a session-side read that created it
    would hand every session on the machine a file the app then has to explain,
    and the gate would start answering about a record nothing is writing.
    """
    path = state.state_path(tmp_path)
    assert state.read(tmp_path) is None
    assert not path.exists()
    assert not path.parent.exists()


def test_the_console_fields_survive_the_read(tmp_path: Path) -> None:
    """Reading through the host's model would drop them silently.

    `UiHostState` declares `extra="ignore"`, so a console reader that reused it
    would parse this record cleanly and lose `console` with no error at all —
    the gate would then be permanently False and the tool would never appear.
    """
    state.publish(record(console_surfaces=3, console_agent_surfaces=2, console_proto=1), tmp_path)
    current = state.read(tmp_path)
    assert current is not None
    assert current.console is True
    assert current.console_surfaces == 3
    assert current.console_agent_surfaces == 2
    assert current.console_proto == 1


def test_an_older_record_without_console_fields_reads_as_no_console(tmp_path: Path) -> None:
    """Additive fields with defaults, and the default is the honest one.

    An app that predates the console writes no such key. The reader must treat
    that as "no console" rather than raising (which would make session startup
    fail on every machine with an older app) or defaulting to True (which would
    advertise a tool whose every call refuses).
    """
    host_state.publish(
        host_state.UiHostState.model_validate(
            {
                "pid": os.getpid(),
                "port": 52133,
                "session_key": "k" * 32,
                "proto": PROTO_VERSION,
            }
        ),
        tmp_path,
    )
    current = state.read(tmp_path)
    assert current is not None and current.console is False
    assert state.available(tmp_path) is False
    assert state.advertisable(tmp_path) is False


def test_the_capability_bit_is_required_and_is_not_a_liveness_question(tmp_path: Path) -> None:
    """The one clause this module adds to the browser host's predicate.

    A running app whose console feature is off (disabled in Settings, the
    `LOCAL_OPERATOR_UI_CONSOLE_HOST=0` launch flag, or a failed native load)
    must offer no tool at all: advertising a tool whose every action refuses is
    worse than offering none. The distinction is deliberately NOT folded into
    `liveness`, which keeps answering about the process — "there is no app" and
    "the app says its console is off" are different facts with different
    remedies, and the tool reports them with different sentences.
    """
    state.publish(record(console=False), tmp_path)
    status, current = state.liveness(tmp_path)
    assert status is state.Liveness.FRESH
    assert current is not None and current.console is False
    assert state.available(tmp_path) is False
    assert state.advertisable(tmp_path) is False

    state.publish(record(console=True), tmp_path)
    assert state.available(tmp_path) is True
    assert state.advertisable(tmp_path) is True


def test_the_three_state_classifier_is_the_hosts_own(tmp_path: Path) -> None:
    """FRESH advertises and executes; STALE advertises only; ABSENT neither.

    The weaker `advertisable` commitment is the browser's rule and is kept
    deliberately: hiding the tool from an app whose heartbeat writer stopped
    leaves the agent with no console and no explanation for a host that is
    running and would answer.
    """
    assert state.liveness(tmp_path)[0] is state.Liveness.ABSENT

    # A dead pid is ABSENT even with a live-looking heartbeat.
    state.publish(record(pid=999_999_999), tmp_path)
    assert state.liveness(tmp_path)[0] is state.Liveness.ABSENT
    assert state.available(tmp_path) is False
    assert state.advertisable(tmp_path) is False

    state.publish(record(), tmp_path)
    assert state.liveness(tmp_path)[0] is state.Liveness.FRESH

    path = state.state_path(tmp_path)
    stale = state.read(tmp_path)
    assert stale is not None
    stale.heartbeat_at = time.time() - state.HEARTBEAT_TIMEOUT_S - 1
    path.write_text(stale.model_dump_json())
    assert state.liveness(tmp_path)[0] is state.Liveness.STALE
    assert state.available(tmp_path) is False
    assert state.advertisable(tmp_path) is True


def test_the_browsers_own_reader_still_parses_the_console_record(tmp_path: Path) -> None:
    """Two readers, one file, and neither may break the other.

    This is the invariant that makes the shared record safe: the console's extra
    fields are additive, so the browser's reader (`extra="ignore"`) keeps working
    byte-for-byte on a record the console wrote, and the console's reader keeps
    every browser field.
    """
    state.publish(record(tabs=4, agent_tabs=1), tmp_path)
    browser_view = host_state.read(tmp_path)
    assert browser_view is not None
    assert browser_view.tabs == 4 and browser_view.agent_tabs == 1
    console_view = state.read(tmp_path)
    assert console_view is not None
    assert console_view.console is True
