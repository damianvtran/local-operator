"""The secret store on a platform with no ``fcntl`` and no peer identity.

The audit ids these cover: D1/B18/A11 (module-level ``fcntl`` on the primary
read path), D2 (``os.getuid`` in the rendezvous helpers), D6/C7 (the broker's
peer-identity and AF_UNIX assumptions), D7 (``harden`` deleting the only
key that opens the store) and D9 (``chmod`` as the whole confidentiality
argument).

**How the platform is simulated, and why each test simulates it that way.**
This host is macOS, so nothing here may claim an execution it did not do:

* where the defect is *"the module cannot be imported at all"*, the POSIX-only
  module is blocked in a FRESH interpreter and the import is attempted there —
  a module already in ``sys.modules`` would make an in-process test pass
  vacuously (D1, D5);
* where the defect is a *branch*, the named platform constant is
  monkeypatched and the assertion is about the branch, never about a codec or
  a path separator this host does not have. ``os.name`` itself is never
  patched: ``pathlib`` reads it at call time, so a test that sets it to
  ``"nt"`` makes the next ``Path(...)`` in the process a ``WindowsPath`` and
  fails with ``UnsupportedOperation`` — a dead end that looks like a
  product bug.

Every test that stubs a branch also asserts the POSIX behaviour it replaced is
unchanged, because "macOS behaviour must not change" is a constraint of the
change rather than a nicety: the closed half is what makes the open half safe.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.secrets import broker as broker_module
from local_operator.secrets import client as client_module
from local_operator.secrets import handlers
from local_operator.secrets import keys as keys_module
from local_operator.secrets import peer as peer_module
from local_operator.secrets import protocol as protocol_module
from local_operator.secrets.broker import BrokerError
from local_operator.secrets.keys import key_mode, key_path, load_master_key

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def windows_peer_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for a platform with no peer-identity probe.

    The predicate is a module-level constant frozen at import — the convention
    this tree already uses for platform decisions (``teams._DIR_FD_READS``,
    ``group_reaper._REAPING_IS_SUPPORTED``) — so each module holds its own copy
    and this patches every copy the code under test reads. Patching only the
    one in ``peer`` would leave ``ensure_broker`` still believing it can spawn.
    """
    for module in (peer_module, client_module, broker_module):
        monkeypatch.setattr(module, "PEER_AUTHENTICATION_SUPPORTED", False)


#: A fresh interpreter that behaves as if ``fcntl`` did not exist, which is
#: what a Windows process sees. Installed as a meta-path finder rather than by
#: deleting an attribute, because the import machinery is what has to fail — a
#: module that simply sets ``fcntl = None`` would not reproduce the
#: ``ModuleNotFoundError`` the audit recorded.
_BLOCK_FCNTL = """
import sys

class _NoFcntl:
    def find_spec(self, name, path=None, target=None):
        if name == "fcntl":
            raise ModuleNotFoundError("No module named 'fcntl'")
        return None

sys.meta_path.insert(0, _NoFcntl())
sys.modules.pop("fcntl", None)
"""


def _import_without_fcntl(tmp_path: Path, statement: str) -> subprocess.CompletedProcess[str]:
    """Import ``statement`` in a fresh interpreter with ``fcntl`` unimportable."""
    script = _BLOCK_FCNTL + statement
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    environment.update(
        HOME=str(tmp_path / "home"),
        LOCAL_OPERATOR_CONFIG_DIR=str(tmp_path / "config"),
        PYTHONPATH=str(REPO_ROOT),
    )
    (tmp_path / "home").mkdir(exist_ok=True)
    (tmp_path / "config").mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )


# --- D1 / B18 / A11: the client, and through it every credential read -------


def test_the_client_imports_where_fcntl_does_not_exist(tmp_path: Path) -> None:
    """The blocker: ``import fcntl`` at module scope made every read a crash.

    Reachability, not tidiness: ``secrets.access`` imports this module inside
    its retrieval path, so a Windows process raised ``ModuleNotFoundError``
    out of a property of the agent loop instead of reading the key file it had
    every right to read.
    """
    result = _import_without_fcntl(
        tmp_path,
        "import local_operator.secrets.client as client\nprint('imported', client.__name__)",
    )
    assert result.returncode == 0, result.stderr
    assert "imported local_operator.secrets.client" in result.stdout


#: A fresh interpreter whose ``socket`` module has no ``AF_UNIX``, which is what
#: a Windows process sees. The attribute is DELETED rather than shadowed, because
#: that is how the real failure arrives: ``socket.socket(socket.AF_UNIX, ...)``
#: resolves the module global at call time, so the platform raises
#: ``AttributeError: module 'socket' has no attribute 'AF_UNIX'`` — measured on
#: the Windows runner as ``secret.roundtrip``'s failure, taking ``lop secret
#: status``, ``get`` and ``set`` down with it.
#:
#: The peer-identity probe is switched off in the same preamble, because the two
#: absences travel together on the platform being simulated
#: (``peer.PEER_AUTHENTICATION_SUPPORTED`` is derived from ``sys.platform``, and
#: nothing off darwin/linux has ``AF_UNIX``). Setting only ``AF_UNIX`` would make
#: this host reach ``ensure_broker``'s SPAWN arm and start a REAL broker daemon
#: inside the test, which is a side effect rather than a simulation.
_STRIP_AF_UNIX = """
import socket

assert hasattr(socket, "AF_UNIX"), "harness premise: this host has AF_UNIX"
del socket.AF_UNIX
assert not hasattr(socket, "AF_UNIX"), "harness failed to remove socket.AF_UNIX"

from local_operator.secrets import broker as _broker
from local_operator.secrets import client as _client
from local_operator.secrets import peer as _peer

for _module in (_peer, _client, _broker):
    _module.PEER_AUTHENTICATION_SUPPORTED = False
"""


def _run_without_af_unix(
    tmp_path: Path,
    statement: str,
    *,
    argv: list[str] | None = None,
    stdin: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run ``statement`` in a fresh interpreter with no ``socket.AF_UNIX``."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    environment.update(
        HOME=str(tmp_path / "home"),
        LOCAL_OPERATOR_CONFIG_DIR=str(tmp_path / "config"),
        PYTHONPATH=str(REPO_ROOT),
    )
    (tmp_path / "home").mkdir(exist_ok=True)
    (tmp_path / "config").mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, "-c", _STRIP_AF_UNIX + statement, *(argv or [])],
        capture_output=True,
        text=True,
        input=stdin,
        env=environment,
        timeout=180,
    )


def _cli_without_af_unix(
    tmp_path: Path, *argv: str, stdin: str = ""
) -> subprocess.CompletedProcess[str]:
    """One ``lop secret ...`` invocation on a host with no ``socket.AF_UNIX``.

    The real dispatch chain (``build_cli_parser`` → the ``secret`` subcommand's
    own ``main``), not a handle on the handlers: the probe that found this defect
    drove the CLI, and the surface a user meets is the one that has to work.
    """
    return _run_without_af_unix(
        tmp_path,
        "import sys\n"
        "from local_operator.cli import build_cli_parser\n"
        "from local_operator.secrets.cli import main as secret_main\n"
        "sys.exit(secret_main(build_cli_parser().parse_args(sys.argv[1:])))\n",
        argv=list(argv),
        stdin=stdin,
    )


def test_every_platform_layer_answers_no_broker_without_af_unix(tmp_path: Path) -> None:
    """The graded answers, not an ``AttributeError`` out of a liveness probe.

    Each of these three is documented as returning a VALUE when there is no
    broker — ``is_running`` a bool, ``broker_status`` ``None``,
    ``ensure_broker`` ``False`` — and the tree's keyfile fallback is wired to
    exactly those values. The ``AttributeError`` escaped all three, so the
    fallback was never reached and the store was simply dead on Windows.
    """
    result = _run_without_af_unix(
        tmp_path,
        "import socket\n"
        "from local_operator.secrets import client\n"
        "assert not hasattr(socket, 'AF_UNIX')\n"
        "assert client.is_running() is False, 'is_running must answer False, not raise'\n"
        "assert client.broker_status() is None, 'broker_status must answer None, not raise'\n"
        "assert client.ensure_broker() is False, 'no broker may be spawned here'\n"
        "assert client._AF_UNIX_AVAILABLE is False\n"
        "print('graded')\n",
    )
    assert result.returncode == 0, result.stderr
    assert "graded" in result.stdout


def test_the_store_round_trips_without_af_unix(tmp_path: Path) -> None:
    """``status``/``set``/``get`` end to end, which is what the probe measures.

    The keyfile tier needs no broker, so every one of these must work on a
    platform with no ``AF_UNIX`` — and ``get``'s stdout is asserted byte-exact,
    because a value on that stream crosses into a child's argv and a corrupted
    credential fails as a confusing 401 rather than as an error here.
    """
    before = _cli_without_af_unix(tmp_path, "secret", "status")
    assert before.returncode == 0, before.stderr
    assert "keyfile" in before.stdout, before.stdout
    assert "AF_UNIX" not in (before.stdout + before.stderr)

    stored = _cli_without_af_unix(tmp_path, "secret", "set", "xplat_probe", stdin="probe-value")
    assert stored.returncode == 0, stored.stderr

    fetched = _cli_without_af_unix(tmp_path, "secret", "get", "xplat_probe")
    assert fetched.returncode == 0, fetched.stderr
    # ``get``'s stdout is byte-exact: a value on that stream crosses into a child's
    # argv, and one stray byte fails as a confusing 401 rather than here.
    assert fetched.stdout == "probe-value", fetched.stdout

    after = _cli_without_af_unix(tmp_path, "secret", "status")
    assert after.returncode == 0, after.stderr
    assert "1" in after.stdout, "the store did not report the record just written"


def test_the_posix_connect_path_is_unchanged(tmp_path: Path) -> None:
    """The open half: the new guard must not touch a platform that has AF_UNIX.

    Asserted on the failure the call always produced — a ``BrokerUnavailable``
    naming the socket path — so a guard that swallowed the connect on POSIX
    (or replaced the real error with the platform one) fails here.
    """
    assert client_module._AF_UNIX_AVAILABLE is True
    missing = tmp_path / "no-broker-here.sock"
    with pytest.raises(client_module.BrokerUnavailable, match="could not reach the secret broker"):
        client_module._connect(missing)


def test_the_lazy_start_lock_is_only_reached_where_fcntl_exists(
    config_root: Path, monkeypatch: pytest.MonkeyPatch, windows_peer_authentication: None
) -> None:
    """The import moved inside ``ensure_broker``; the lock itself is unchanged.

    Asserted by making the function take its refusal path with the socket
    unstarted, so the ``flock`` call site is never reached on a platform
    without it — the reason a ``None``-guard fallback would have been
    unreachable code rather than a fallback.
    """
    spawned: list[Path | None] = []
    monkeypatch.setattr(client_module, "is_running", lambda base=None: False)
    monkeypatch.setattr(client_module, "_spawn_broker", lambda base: spawned.append(base))

    assert client_module.ensure_broker(config_root) is False
    assert spawned == [], "a broker was spawned where none could authenticate a peer"


def test_ensure_broker_still_starts_one_where_peer_authentication_works(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The POSIX half: the refusal must not leak into a platform that has one."""
    spawned: list[Path | None] = []
    monkeypatch.setattr(client_module, "PEER_AUTHENTICATION_SUPPORTED", True)
    monkeypatch.setattr(client_module, "is_running", lambda base=None: False)
    monkeypatch.setattr(client_module, "_spawn_broker", lambda base: spawned.append(base))

    # Nothing the stub spawns ever starts listening, so the bounded poll runs
    # out — with a zero deadline, since the default is bound at def time and
    # this test is about reaching the spawn, not about the wait.
    assert client_module.ensure_broker(config_root, timeout=0.0) is False
    assert spawned == [config_root], "the lazy start did not reach the spawn on POSIX"


# --- D6 / C7: the broker cannot attribute a caller off darwin/linux ---------


def test_the_broker_refuses_to_bind_where_it_could_not_authenticate(
    config_root: Path, windows_peer_authentication: None
) -> None:
    """A daemon that can serve nobody is refused by name, not started and denied.

    ``_serve_connection`` already fails closed per connection, so the
    observable behaviour without this is the same denial — reached only after
    an operator has started a daemon, watched it listen, and watched it say no.
    """
    broker = broker_module.SecretBroker(config_root)
    with pytest.raises(BrokerError, match=re.escape(peer_module._PLATFORM_LABEL)) as refusal:
        broker.start()
    assert "keyfile" in str(refusal.value), "the refusal must name the tier that does work"


def test_the_platform_refusal_names_the_os_and_the_alternative(
    windows_peer_authentication: None,
) -> None:
    reason = peer_module.broker_unsupported_reason()
    assert reason is not None
    assert peer_module._PLATFORM_LABEL in reason
    assert "keyfile" in reason


def test_the_platform_refusal_never_shows_the_user_a_cpython_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D5: the refusal interpolated ``sys.platform``, so a Windows user read
    "cannot run on win32" — a spelling that appears nowhere else they can see.

    ``_PLATFORM_LABEL`` is patched rather than ``os.name``: ``pathlib`` reads
    ``os.name`` at call time, so setting it to ``nt`` makes the next ``Path(...)``
    in this process a ``WindowsPath`` and the test dies on an
    ``UnsupportedOperation`` that looks like a product bug (the convention this
    file's docstring records).
    """
    monkeypatch.setattr(peer_module, "PEER_AUTHENTICATION_SUPPORTED", False)
    monkeypatch.setattr(peer_module, "_PLATFORM_LABEL", "Windows")

    reason = peer_module.broker_unsupported_reason()

    assert reason is not None
    assert "on Windows" in reason
    assert "win32" not in reason, "the raw sys.platform value reached the operator"


def test_the_platform_refusal_is_absent_where_a_broker_can_run() -> None:
    """POSIX behaviour unchanged: no message, so nothing prints one."""
    assert peer_module.PEER_AUTHENTICATION_SUPPORTED is True
    assert peer_module.broker_unsupported_reason() is None


# --- D7: harden must not delete the only key that opens the store ----------


def test_harden_refuses_where_the_wrapped_key_could_never_be_unwrapped(
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    windows_peer_authentication: None,
) -> None:
    """The data-loss blocker, asserted on the KEY rather than on the message.

    ``harden`` deletes the plaintext master key and leaves the unwrap to the
    broker, which is the only code that can perform it. Where the broker
    cannot authenticate anyone, that is unrecoverable: the wrapped file is
    still on disk but no reachable path opens it. So the assertion that
    matters is that the plaintext key SURVIVED, and that no passphrase was
    typed for a change that is not going to happen.
    """
    load_master_key(config_root, create=True)

    def _must_not_prompt(_prompt: str) -> str:
        raise AssertionError("harden prompted for a passphrase it could not use")

    monkeypatch.setattr(handlers, "_read_passphrase", _must_not_prompt)

    assert handlers._harden(argparse.Namespace()) == 2
    captured = capsys.readouterr()
    assert "Cannot harden" in captured.err
    assert sys.platform in captured.err
    assert key_path(config_root).exists(), "the plaintext key was destroyed by a refused harden"
    assert key_mode(config_root) == "keyfile"


# --- D2: os.getuid in the rendezvous helpers --------------------------------


def test_the_rendezvous_helpers_work_where_os_getuid_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``getuid`` is Unix-only; the name it builds and the check it makes are not.

    Reached whenever the config path is too deep for ``sun_path``, which
    ``LOCAL_OPERATOR_CONFIG_DIR`` and pytest's ``tmp_path`` make routine — so
    this was a live crash on the platform, not a latent one.
    """
    assert protocol_module._UID_IS_MEANINGFUL is True
    assert protocol_module._uid_token() == os.getuid()

    monkeypatch.setattr(protocol_module, "_UID_IS_MEANINGFUL", False)
    monkeypatch.delattr(os, "getuid", raising=False)

    assert protocol_module._uid_token() == 0
    fallback = protocol_module._runtime_fallback_dir(Path("/some/config/secrets"))
    assert "-0-" in fallback.name

    deep = tmp_path / ("d" * 90) / "config"
    deep.mkdir(parents=True)
    assert len(str(deep)) > protocol_module.MAX_SOCKET_PATH
    relocated = protocol_module.socket_path(deep)
    assert str(relocated).startswith(str(Path(protocol_module.tempfile.gettempdir())))

    runtime_dir = tmp_path / "rt"
    assert protocol_module.ensure_runtime_dir(runtime_dir) == runtime_dir


def test_the_runtime_dir_ownership_check_still_fires_on_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check this change makes conditional must stay live where it means something."""
    monkeypatch.setattr(protocol_module, "_uid_token", lambda: 987_654)
    with pytest.raises(protocol_module.ProtocolError, match="owned by uid"):
        protocol_module.ensure_runtime_dir(tmp_path / "rt")


def test_the_runtime_dir_ownership_check_is_skipped_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipped BY NAME, not left to compare a field Windows does not populate.

    ``st_uid`` is 0 there, so comparing it against a fabricated uid would pass
    every time — a check that cannot fail while reading like one that does.
    """
    monkeypatch.setattr(protocol_module, "_UID_IS_MEANINGFUL", False)
    runtime_dir = tmp_path / "rt"
    assert protocol_module.ensure_runtime_dir(runtime_dir) == runtime_dir


# --- D9: chmod is not the confidentiality mechanism on Windows --------------


def test_the_at_rest_note_is_silent_on_posix_and_explicit_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The skip in ``check_mode`` stops being silent without becoming an error.

    Confirmed rather than skipped: the note says what IS protecting the store
    (the inherited ACL), that lop does not create or verify it, and which
    placement defeats it.
    """
    assert keys_module.at_rest_protection_note() is None
    monkeypatch.setattr(keys_module, "_MODE_BITS_ARE_CONFIDENTIALITY", False)
    note = keys_module.at_rest_protection_note()
    assert note is not None
    assert "access control list" in note
    assert "chmod" in note


def test_secret_status_reports_the_at_rest_model(
    config_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``status`` is where an operator forms the mental model, so it carries the note."""
    assert handlers._status(argparse.Namespace(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["at_rest_protection"] is None

    monkeypatch.setattr(keys_module, "_MODE_BITS_ARE_CONFIDENTIALITY", False)
    assert handlers._status(argparse.Namespace(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["at_rest_protection"] is not None
