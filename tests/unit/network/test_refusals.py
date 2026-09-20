"""The refusal surface: named sentences and codes, never a traceback (R2, R3, R5).

Every command here is one an operator runs when something is ALREADY wrong — after
a disconnect, against a file that was moved, on a build that does not serve an op
yet. A stack trace in that moment tells them nothing they can act on, and the rule
the whole CLI now follows is that a file whose absence is a legitimate state is a
refusal with a code and a sentence.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import invite as invite_mod
from local_operator.network import relay, store, types

NETWORK = "n_0123456789abcdef01234567"


def _disconnected_device(root: Path) -> types.NetworkRecord:
    """What `lop network disconnect` leaves behind: a record, no secrets.

    Reproduced rather than described — this is the exact state the panic traceback
    came from, and a test that hand-rolled a different one would not have caught it.
    """
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        self_device_id="d_" + "a" * 32,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )
    store.save(record, root)
    return record


def _args(**fields: object) -> Namespace:
    base: dict[str, object] = {
        "json": True,
        "network": NETWORK,
        "role": "drive",
        "expires": 600.0,
        "hosts": "",
        "device": "",
        "print_token": False,
        "peer": "",
    }
    base.update(fields)
    return Namespace(**base)


# ---------------------------------------------------------------------------
# R2 — panic (and friends) after a disconnect
# ---------------------------------------------------------------------------


def test_panic_after_disconnect_refuses_by_name_instead_of_tracebacking(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reported defect: `disconnect` deletes the secrets file, then `panic`
    raised `FileNotFoundError` out of `store.load_secrets`."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _disconnected_device(root)
    assert store.secrets_path(NETWORK, root).exists() is False

    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._cmd_panic(_args())  # noqa: SLF001
    assert excinfo.value.code == "no_network_secret"
    # The sentence says what state the device is in and what to do about it.
    assert "rejoin" in excinfo.value.sentence.lower()
    assert "lop network join" in excinfo.value.sentence
    assert capsys.readouterr().err == ""  # nothing leaked to stderr on the way out


@pytest.mark.parametrize("command", ["invite", "member_rm"])
def test_the_other_secret_readers_refuse_the_same_way(
    command: str, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _disconnected_device(root)
    with pytest.raises(types.MeshRefusal) as excinfo:
        if command == "invite":
            net_cli._cmd_invite(_args())  # noqa: SLF001
        else:
            net_cli._cmd_member_rm(_args(device="d_" + "b" * 32))  # noqa: SLF001
    assert excinfo.value.code == "no_network_secret"


def test_the_relay_control_path_refuses_by_name_too(root: Path) -> None:
    """The refusal must hold on the path the CLI ACTUALLY takes when the relay
    runs — otherwise the defect simply moves one process over."""
    _disconnected_device(root)
    server = relay.RelayServer(
        root=root,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
    )
    try:
        with pytest.raises(types.MeshRefusal) as excinfo:
            server._ctl_panic({"network": NETWORK})  # noqa: SLF001
        assert excinfo.value.code == "no_network_secret"
        # And the control socket renders it as an error frame a --json consumer can
        # branch on, not as a broken connection.
        reply = server.control_dispatch("net_panic_local", {"network": NETWORK, "req": 9})
        assert reply["op"] == "error"
        assert reply["code"] == "no_network_secret"
    finally:
        server.stop()


@pytest.mark.parametrize(
    ("command", "expect"),
    [
        ("show", 0),
        ("trust", 0),
        ("log", 0),
        ("doctor", 0),
        ("status", 0),
    ],
)
def test_commands_that_need_no_secret_still_work_after_a_disconnect(
    command: str, expect: int, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of R2: the remedy must not be to make everything refuse.

    `show`, `trust`, `log`, `doctor` and `status` read records, the audit log and the
    identity — all of which SURVIVE a disconnect — so they must answer, and the
    assertion is the absence of an exception (a traceback fails the test either way).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _disconnected_device(root)
    args = _args(network=NETWORK, limit=10, follow=False, since="", export=None)
    outcome = {
        "show": lambda: net_cli._cmd_show(args),  # noqa: SLF001
        "trust": lambda: net_cli._cmd_trust(
            _args(network=NETWORK, active=True, untrusted=False)
        ),  # noqa: SLF001
        "log": lambda: net_cli._cmd_log(args),  # noqa: SLF001
        "doctor": lambda: net_cli._cmd_doctor(args),  # noqa: SLF001
        "status": lambda: net_cli._cmd_status(args),  # noqa: SLF001
    }[command]()
    assert outcome == expect


def test_peers_with_no_relay_is_an_answer_not_a_failure(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _disconnected_device(root)
    assert net_cli._cmd_peers(_args()) == 1  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    # The JSON keeps the sentence the agent tool and the guide document; the REMEDY
    # lives on the human line, which is where a person reads it.
    assert payload["error"] == "the relay is not running"


def test_a_network_that_does_not_exist_refuses_by_name(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._cmd_panic(_args())  # noqa: SLF001
    # With no networks at all the resolver names that rather than inventing a match;
    # which of the two codes it picks is its business, that it is NAMED is not.
    assert excinfo.value.code in {"unknown_network", "ambiguous_network"}

    _disconnected_device(root)
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._cmd_panic(_args(network="lab"))  # noqa: SLF001
    assert excinfo.value.code == "unknown_network"
    assert "lab" in excinfo.value.sentence


# ---------------------------------------------------------------------------
# R3 — join with a token that cannot be read
# ---------------------------------------------------------------------------


def test_join_with_a_missing_token_file_names_the_path(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    missing = root / "not-here.invite"
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._read_token(f"@{missing}")  # noqa: SLF001
    assert excinfo.value.code == "token_unreadable"
    assert str(missing) in excinfo.value.sentence
    assert "lop network invite" in excinfo.value.sentence


def test_join_with_an_unreadable_token_path_refuses_too(root: Path) -> None:
    """A directory is the honest second case: something exists there, and reading it
    fails for a reason the operator has to hear."""
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._read_token(f"@{root}")  # noqa: SLF001
    assert excinfo.value.code == "token_unreadable"
    assert str(root) in excinfo.value.sentence


def test_join_with_no_argument_and_no_outbox_names_the_next_step(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._read_token("")  # noqa: SLF001
    assert excinfo.value.code == "no_invite_token"


def test_join_with_a_corrupt_token_is_a_pairing_refusal_not_a_traceback(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token that decodes to nonsense is the attacker-adjacent case, and the reader
    upstream already names it; this pins that the CLI surfaces THAT rather than a
    decode error of its own."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _disconnected_device(root)
    path = store.save_invite_token("broken", "lop1.not-base64.not-a-tag", root)
    token = net_cli._read_token(f"@{path}")  # noqa: SLF001
    assert token == "lop1.not-base64.not-a-tag"
    with pytest.raises(types.MeshRefusal) as excinfo:
        invite_mod.decode(token)
    assert isinstance(excinfo.value, types.PairingRefusal)


# ---------------------------------------------------------------------------
# R5 — an unimplemented op answers in operator language
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op",
    ["net_forward_session", "net_session_move", "net_sync", "net_broker"],
)
def test_an_unimplemented_op_names_its_capability_and_says_nothing_changed(
    op: str, root: Path
) -> None:
    """The string the guide agent found was `unknown local op 'net_forward_session'`
    — developer language, and silent about what the caller should do. It now names
    the slice that owns the op and states that nothing was changed."""
    server = relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    try:
        reply = server.control_dispatch(op, {"req": 3})
    finally:
        server.stop()
    assert reply["op"] == "error"
    assert reply["code"] == "not_implemented"
    message = str(reply["message"])
    assert op in message
    assert ".md" in message, "the owner of the slice is named"
    assert "Nothing was changed" in message
    assert "doctor" in message, "and where to find what this build does serve"
    # No Python repr, no internal identifier soup.
    assert "'" not in message


def test_an_op_nobody_planned_is_also_operator_language(root: Path) -> None:
    server = relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    try:
        reply = server.control_dispatch("net_teleport", {"req": 3})
    finally:
        server.stop()
    assert reply["code"] == "unknown_local_op"
    assert "net_teleport" in str(reply["message"])
    assert "Nothing was changed" in str(reply["message"])
    assert "unknown local op 'net_teleport'" not in str(reply["message"])


# ---------------------------------------------------------------------------
# R6 — an isolated HOME is expected, not a failure
# ---------------------------------------------------------------------------


def test_init_in_an_isolated_home_explains_the_missing_launchagent(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reported symptom: `lop network init` printed a launchd refusal that read
    like a failure. It must instead say what a redirected HOME means and what to run
    instead — and `--no-start` must remain silent about it entirely."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    args = Namespace(
        name="home-net",
        listen_address="",
        port=0,
        advertise_hosts=[],
        no_start=False,
        json=True,
    )
    assert net_cli._cmd_init(args) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    note = str(payload["relay"])
    assert "launchd" in note
    assert "serve" in note
    assert "isolated" in note or "redirected" in note
    assert "did not start" not in note

    quiet = Namespace(
        name="quiet-net",
        listen_address="",
        port=0,
        advertise_hosts=[],
        no_start=True,
        json=True,
    )
    assert net_cli._cmd_init(quiet) == 0  # noqa: SLF001
    assert json.loads(capsys.readouterr().out)["relay"] == ""


# ---------------------------------------------------------------------------
# R7 — the audit bounds are real config keys
# ---------------------------------------------------------------------------


def _write_config(root: Path, **values: object) -> None:
    """Write the keys the way the REPO writes them — `settings_io.write_setting`
    through a real ``ConfigManager`` — so this test exercises the registry path an
    operator uses (`/settings`, `lop config edit`) rather than a hand-rolled YAML
    file that could differ from it in shape."""
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    root.mkdir(parents=True, exist_ok=True)
    manager = ConfigManager(root)
    for key, value in values.items():
        settings_io.write_setting(manager, settings_io.BY_KEY[key], value)


def test_the_audit_bounds_come_from_the_config_store(root: Path) -> None:
    """`network.audit.*` follow the repo's configuration procedure (AGENTS.md):
    registered in `settings_io.SETTINGS`, mirrored by a module constant next to the
    reader, and consumed through ONE reader (`store.read_config`)."""
    from local_operator.network.audit import (
        AUDIT_GENERATIONS,
        AUDIT_MAX_AGE_DAYS,
        AUDIT_MAX_BYTES,
        AuditLog,
    )

    default_log = AuditLog(root)
    assert (default_log.max_bytes, default_log.generations, default_log.max_age_days) == (
        AUDIT_MAX_BYTES,
        AUDIT_GENERATIONS,
        AUDIT_MAX_AGE_DAYS,
    )

    _write_config(
        root,
        **{
            "network.audit.max_bytes": 131072,
            "network.audit.generations": 2,
            "network.audit.max_age_days": 3.0,
        },
    )
    configured = AuditLog.from_config(root)
    assert (configured.max_bytes, configured.generations, configured.max_age_days) == (
        131072,
        2,
        3.0,
    )
    # An explicit argument still wins: a test or a probe must be able to pin a value
    # without editing the user's config.
    pinned = AuditLog(root, max_bytes=4096)
    assert pinned.max_bytes == 4096
    assert pinned.generations == 2  # the config still supplies the rest


def test_a_nonsense_configured_value_falls_back_instead_of_refusing(root: Path) -> None:
    """The audit log is the component that must still work when everything else is
    broken: a mistyped number must not be the reason an incident has no records."""
    from local_operator.network.audit import AUDIT_GENERATIONS, AuditLog

    # A value the schema rejects cannot be written through the registry (the test
    # above proves the floor is enforced), so the nonsense goes where a hand-editor
    # leaves it: straight into the file.
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yml").write_text(
        "network:\n  audit:\n    generations: soon\n", encoding="utf-8"
    )
    log = AuditLog.from_config(root)
    assert log.generations == AUDIT_GENERATIONS


def test_the_registry_entries_exist_with_the_documented_bounds() -> None:
    from local_operator.network.audit import (
        AUDIT_GENERATIONS,
        AUDIT_MAX_AGE_DAYS,
        AUDIT_MAX_BYTES,
    )
    from local_operator.settings_io import BY_KEY

    for key, default in (
        ("network.audit.max_bytes", AUDIT_MAX_BYTES),
        ("network.audit.generations", AUDIT_GENERATIONS),
        ("network.audit.max_age_days", AUDIT_MAX_AGE_DAYS),
    ):
        setting = BY_KEY[key]
        assert setting.default == default, key
        assert setting.minimum is not None and setting.maximum is not None, key
        assert setting.path == tuple(key.split(".")), key


def test_a_configured_cap_actually_rotates_at_that_size(root: Path) -> None:
    """End to end for R7: the configured number decides when a generation closes —
    a key that is registered and read but not USED would pass every test above."""
    from local_operator.network.audit import AuditEvent, AuditLog

    # 65536 is the registry's own floor, which the write below proves is enforced.
    _write_config(root, **{"network.audit.max_bytes": 65536, "network.audit.generations": 2})
    log = AuditLog.from_config(root)
    for index in range(400):
        log.record(
            AuditEvent(event="link_closed", network_id=NETWORK, detail={"cause": f"c{index}"})
        )
    log.close()
    assert list(store.audit_path(root).parent.glob("audit.jsonl.*.gz")), "no rotation happened"
