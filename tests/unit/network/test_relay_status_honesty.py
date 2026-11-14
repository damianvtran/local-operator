"""``lop network status`` reports the RUNNING relay, not the reader's guess.

Two rows misled the operator on 2026-11-10 while a mesh delivery was being
diagnosed:

* **log:** printed ``<caller's LOCAL_OPERATOR_CONFIG_DIR>/logs/network.log``, a
  path the reader derived from its own environment. The launchd-run relay was
  appending to ``~/Library/Logs/local-operator/network.log`` (its plist's
  ``StandardOutPath``), so the operator was pointed at a file nothing writes.
* **build:** said "behind the install; restart the relay" for a relay whose
  generation was installed from the SAME wheel as the install's: generations
  are compared by directory name, and a reinstall makes a new name.
"""

from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import update
from local_operator.network import relay
from local_operator.network.types import PeerRecord

# --------------------------------------------------------------------------
# log: the relay's own fd, carried in its status and its record
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform == "darwin"),
    reason="the kernel query exists on Linux (/proc) and macOS (F_GETPATH) only",
)
def test_a_process_names_the_file_its_stdout_appends_to(tmp_path: Path) -> None:
    """The REAL kernel answer, in a child whose stdout a 'supervisor' opened."""
    log = tmp_path / "supervisor-chose-this" / "network.log"
    log.parent.mkdir()
    probe = (
        "import sys; from local_operator.network.relay import _own_output_path; "
        "sys.stderr.write(_own_output_path())"
    )
    with log.open("ab") as out:
        done = subprocess.run(
            [sys.executable, "-c", probe], stdout=out, stderr=subprocess.PIPE, timeout=60
        )
    assert done.returncode == 0, done.stderr
    assert Path(done.stderr.decode()).resolve() == log.resolve()

    # Not a file (a pipe here; a terminal or the journal's socket in life): no path.
    piped = subprocess.run(
        [sys.executable, "-c", probe], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
    )
    assert piped.returncode == 0, piped.stderr
    assert piped.stderr == b""


def test_the_record_carries_the_log_path_and_an_older_record_still_parses() -> None:
    record = PeerRecord(pid=1, log_path="/var/log/elsewhere/network.log")
    assert PeerRecord.from_json(record.to_json()).log_path == "/var/log/elsewhere/network.log"
    older = record.to_json()
    older.pop("log_path")
    assert PeerRecord.from_json(older).log_path == ""


def _status_with(
    monkeypatch: pytest.MonkeyPatch, caller_root: Path, live: Any, record: Any
) -> dict[str, Any]:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(caller_root))
    monkeypatch.setattr(relay.store, "scan_own_relay", lambda root=None: (record, "live"))
    monkeypatch.setattr(relay, "health", lambda *a, **k: live)
    return relay.status(refresh=True)


def test_status_prints_the_running_relays_log_not_the_callers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    caller = tmp_path / "isolated-caller"
    relays_own = "/Users/op/Library/Logs/local-operator/network.log"
    record = SimpleNamespace(pid=4242, log_path=relays_own, to_json=lambda: {"pid": 4242})

    answering = _status_with(monkeypatch, caller, {"log_path": relays_own}, record)
    assert answering["log"] == relays_own, answering["log"]

    # A WEDGED relay (no answer) still has its record, and the record knows.
    wedged = _status_with(monkeypatch, caller, None, record)
    assert wedged["log"] == relays_own, wedged["log"]


def test_with_nothing_reported_status_keeps_todays_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An older relay (no field) and no relay at all keep the reader's path."""
    caller = tmp_path / "caller"
    older = SimpleNamespace(pid=4242, to_json=lambda: {"pid": 4242})
    assert _status_with(monkeypatch, caller, {}, older)["log"] == str(relay.log_path())
    assert _status_with(monkeypatch, caller, None, None)["log"] == str(relay.log_path())


# --------------------------------------------------------------------------
# build: one wheel in two generations is not "behind the install"
# --------------------------------------------------------------------------


def _hash(content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
    return digest.decode()


def _generation(root: Path, name: str, version: str, files: dict[str, bytes]) -> Path:
    """A generation tree with real dist-info METADATA and RECORD (installer-shaped)."""
    generation = root / "generations" / name
    site = generation / "tools" / "local-operator" / "lib" / "python3.12" / "site-packages"
    dist = site / f"local_operator-{version}.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: local-operator\nVersion: {version}\n", encoding="utf-8"
    )
    rows = [f"{path},sha256={_hash(body)},{len(body)}" for path, body in files.items()]
    # Install-specific rows differ between two installs of ONE wheel and must not count.
    rows.append(
        f"local_operator-{version}.dist-info/direct_url.json,sha256={_hash(name.encode())},9"
    )
    rows.append(f"local_operator-{version}.dist-info/RECORD,,")
    (dist / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return generation


WHEEL = {"local_operator/__init__.py": b"VERSION = '0.96.6'\n", "local_operator/x.py": b"x = 1\n"}


def _reading(monkeypatch: pytest.MonkeyPatch, running: Path, current: Path) -> dict[str, Any]:
    """``generation_reading`` with only the PROCESS and POINTER lookups pinned."""
    shim = current / "bin" / "lop"
    monkeypatch.setattr(update, "generation_of_process", lambda pid: running)
    monkeypatch.setattr(update, "current_generation", lambda: current)
    monkeypatch.setattr(relay, "_restart_target", lambda: (current, shim, True))
    return relay.generation_reading(4242)


def test_a_reinstall_of_the_same_wheel_is_current_not_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    running = _generation(tmp_path, "20261110T080000Z-0.96.6", "0.96.6", WHEEL)
    current = _generation(tmp_path, "20261110T090000Z-0.96.6", "0.96.6", WHEEL)
    facts = _reading(monkeypatch, running, current)
    assert facts["relay_generation"] != facts["installed_generation"], facts
    assert facts["relay_generation_stale"] is False, facts
    words = relay.generation_words(facts)
    assert "behind" not in words and "restart" not in words, words


def test_different_code_under_the_same_version_is_still_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    running = _generation(tmp_path, "20261110T080000Z-0.96.6", "0.96.6", WHEEL)
    changed = {**WHEEL, "local_operator/x.py": b"x = 2\n"}
    current = _generation(tmp_path, "20261110T090000Z-0.96.6", "0.96.6", changed)
    facts = _reading(monkeypatch, running, current)
    assert facts["relay_generation_stale"] is True, facts
    assert "restart the relay" in relay.generation_words(facts)


def test_an_unprovable_identity_never_silences_staleness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No hashes (an editable install) is "unproven", never "the same"."""
    running = _generation(tmp_path, "20261110T080000Z-0.96.6", "0.96.6", {})
    current = _generation(tmp_path, "20261110T090000Z-0.96.6", "0.96.6", {})
    assert _reading(monkeypatch, running, current)["relay_generation_stale"] is True
