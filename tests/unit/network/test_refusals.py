"""The refusal surface: named sentences and codes, never a traceback (R2, R3, R5).

Every command here is one an operator runs when something is ALREADY wrong — after
a disconnect, against a file that was moved, on a build that does not serve an op
yet. A stack trace in that moment tells them nothing they can act on, and the rule
the whole CLI now follows is that a file whose absence is a legitimate state is a
refusal with a code and a sentence.
"""

from __future__ import annotations

import json
import uuid
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


def test_sessions_peer_json_refuses_with_a_document(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``lop sessions --peer X --json`` answers with JSON, not with silence (F-5).

    Every `lop network` verb emits ``{"ok": false, "code": ..., "message": ...}``
    when it refuses, and the design says the agent path PARSES the JSON. This one
    printed its sentence on stderr and left stdout EMPTY with rc=1, so a parser
    could not tell a refusal from a crash. The other front end for the same rows is
    `lop network sessions`, which already answered with a document.

    WHICH refusal it is here is the Q-R5-1 half and is asserted in its own section
    below: with no relay on this device the reason is this device's own relay, so
    the code is the family's ``relay_unavailable`` rather than a claim about the
    peer. THIS test is about the document existing at all, which is why its
    assertions are the keys every refusal carries.
    """
    from local_operator import cli as main_cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    args = Namespace(
        json=True,
        sessions_command=None,
        all=False,
        limit=None,
        peer="d_absent",
        all_peers=False,
    )
    assert main_cli.sessions_command(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["code"] == "relay_unavailable"
    assert payload["message"]


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
        # ``doctor`` exits non-zero because a DISCONNECTED device has a failing
        # check: ``ok`` now means "the mesh is healthy", not "the command ran",
        # so rc follows the checks (QA round 1).
        ("doctor", 1),
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
    # ONE SHAPE FOR THE FAMILY: ``code`` + ``message``, the same keys every other
    # `lop network` refusal carries. This one used to answer with a bare ``error``
    # key, so a consumer that read ``code`` found nothing (QA round 1, F-6).
    assert payload["code"] == "relay_unavailable"
    assert "relay" in payload["message"]


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


# ---------------------------------------------------------------------------
# R4 — Q-R4-1: a missing answer is not an outcome
# ---------------------------------------------------------------------------

#: `lop network sessions`'s own flag set — `_args` above builds `ls`'s, and a
#: Namespace missing one of these would fail on `getattr` inside the handler
#: rather than on the condition under test.
_SESSIONS_FLAGS: dict[str, object] = {
    "json": True,
    "network_command": "sessions",
    "peer": "",
    "all_peers": False,
    "create": False,
    "engage": "",
    "stop": "",
    "force": False,
    "cwd": "",
    "name": "",
    "prompt": "",
}
PEER = "d_" + "b" * 32


def _sessions_args(**fields: object) -> Namespace:
    flags = dict(_SESSIONS_FLAGS)
    flags.update(fields)
    return Namespace(**flags)


def _relay_down(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The condition QA round 4 reproduced: an admitted member, no relay running.

    A record on disk and nothing answering `find_own_relay` — which is the state
    `qa-c2` was in, and NOT the same as a device with no network at all.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _disconnected_device(root)


@pytest.mark.parametrize(
    "fields",
    [
        {"peer": PEER, "stop": "deadbeefcafe"},
        {"peer": PEER, "engage": "deadbeefcafe"},
        {"peer": PEER, "create": True},
        {"all_peers": True},
        {"peer": PEER},
    ],
    ids=["stop", "engage", "create", "all-peers", "peer-list"],
)
def test_the_session_family_refuses_by_name_when_this_devices_relay_is_down(
    fields: dict[str, object],
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every verb on the piloting surface, through `main` — the path a shell takes.

    QA round 4's Q-R4-1: with this device's own relay down, `--stop` answered
    ``{"ok": true}`` with rc 0 (a success receipt for a stop that never left the
    machine), ``--engage`` answered a bare ``{"ok": false}`` with no code and no
    sentence, and ``--all-peers`` presented an empty peer set as a result. The
    refusal is now raised where the answer is lost, so each of these is the same
    document `peers` has always shipped.
    """
    _relay_down(root, monkeypatch)
    assert net_cli.main(_sessions_args(**fields)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["code"] == "relay_unavailable"
    assert "relay" in payload["message"]


def test_the_whole_family_refuses_in_the_same_words_as_peers(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ONE VOICE: the family's refusal is byte-identical to the sibling's.

    `peers` is the verb the guide documents the shape on, and it was already
    right; the finding was that the session verbs next to it were not. Comparing
    the two here is what keeps them from drifting apart again — the code is a
    shared constant and the sentence is one function, and this fails if either
    grows a second spelling.
    """
    _relay_down(root, monkeypatch)
    assert net_cli._cmd_peers(_args()) == 1  # noqa: SLF001
    reference = json.loads(capsys.readouterr().out)

    for fields in (
        {"peer": PEER, "stop": "deadbeefcafe"},
        {"peer": PEER, "engage": "deadbeefcafe"},
        {"all_peers": True},
    ):
        assert net_cli.main(_sessions_args(**fields)) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["code"] == reference["code"]
        assert payload["message"] == reference["message"]


@pytest.mark.parametrize(
    ("outcome", "ended"),
    [
        ("stopped", True),
        ("killed", True),
        ("already-gone", True),
        # The peer holds no runtime for that id, and the relay emits that as an
        # OUTCOME rather than an error (`relay._op_session_stop`) — so it is the
        # answer to "did it stop", not a missing one.
        ("not_running", True),
        ("refused", False),
        # A busy target: the ladder declined to signal and the runtime is still up
        # (`control.LEFT_ALONE_METHODS`). The old derivation called this success.
        ("skipped", False),
        ("unknown", False),
    ],
)
def test_a_stop_reports_success_only_for_an_outcome_that_ended_the_session(
    outcome: str,
    ended: bool,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`ok` follows a NAMED outcome, and the receipt still carries it verbatim.

    The relay answered, so this is the relay-UP half: the word comes from the peer
    and this side must not re-derive it. `skipped` is the case the old
    ``outcome not in ("", "refused")`` got wrong even when the relay WAS running:
    a stop that did not act must not report success (Q-R4-1).
    """
    _relay_down(root, monkeypatch)
    answer = {"outcome": outcome, "rung": "socket", "session_id": "deadbeefcafe", "detail": ""}
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: answer)

    assert net_cli.main(_sessions_args(peer=PEER, stop="deadbeefcafe")) == (0 if ended else 1)
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is ended
    assert payload["outcome"] == outcome  # the peer's own word, unaltered


def test_a_stop_answer_without_an_outcome_refuses_instead_of_reporting_success(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The residual half of Q-R4-1: ``None not in ("", "refused")`` was TRUE.

    A relay that answers without the field this verb exists to report is not a
    stop that succeeded — it is a relay whose answer cannot be read, and the old
    derivation turned it into ``{"ok": true}``. Both the empty dict the `or {}`
    used to manufacture and the non-mapping wrap ``_relay_call`` makes
    (``{"value": None}``) are covered.
    """
    for answer in ({}, {"value": None}):
        _relay_down(root, monkeypatch)
        monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: answer)
        assert net_cli.main(_sessions_args(peer=PEER, stop="deadbeefcafe")) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["code"] == "stop_unreported"
        assert "outcome" in payload["message"]


@pytest.mark.parametrize(
    ("fields", "code"),
    [
        ({"peer": PEER, "engage": "deadbeefcafe"}, "engage_unreported"),
        ({"peer": PEER, "create": True}, "create_unreported"),
    ],
)
def test_engage_and_create_name_their_missing_field_too(
    fields: dict[str, object],
    code: str,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same silence, one verb over: a bare ``{"ok": false}`` is not an answer.

    QA round 4 saw ``{"ok": false}`` with no code and no sentence from `--engage`;
    the missing-answer cause is fixed at the source, and this pins the other half
    — an ANSWER without ``engaged``/``session_id`` — so neither verb can go back to
    reporting an unreadable receipt as a plain failure.
    """
    _relay_down(root, monkeypatch)
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: {"ok": True})
    assert net_cli.main(_sessions_args(**fields)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["code"] == code


def test_the_stop_word_map_never_turns_a_method_it_does_not_know_into_success() -> None:
    """Every ladder method has a word, and the fallback word is not a success.

    ``.get(method, "stopped")`` reported ``draining`` — alive and already leaving,
    which `control.LEFT_ALONE_METHODS` groups with ``busy`` — as ``stopped``: a
    receipt claiming this device ended a runtime it never signalled. Both halves
    are pinned here because either one alone re-opens the hole: a method with no
    word, or a default that reads as ended.
    """
    from local_operator.network.relay import _STOP_OUTCOME_DEFAULT, _STOP_OUTCOME_WORD

    for method in ("socket", "sigterm", "sigkill", "gone", "refused", "busy", "draining"):
        assert method in _STOP_OUTCOME_WORD, method
    assert _STOP_OUTCOME_WORD["draining"] == "skipped"
    assert _STOP_OUTCOME_DEFAULT not in net_cli._STOP_ENDED_OUTCOMES  # noqa: SLF001


# ---------------------------------------------------------------------------
# Q-R5-1 — the ordinary session list, and WHICH component it blames
# ---------------------------------------------------------------------------
#
# The session verbs inside `network.cli` refuse by name when this device's relay
# is down (Q-R4-1). The ORDINARY listing — `lop sessions --all-peers` / `--peer`,
# `local_operator/cli.py::_remote_listing`, the guide's "the same rows through the
# ordinary session list" — was the one consumer the sweep missed: `--all-peers`
# answered `[]` with rc 0 and printed "no active lop sessions" (a lying emptiness),
# and `--peer` answered `peer_unreachable`, blaming the peer for a fault here.
#
# The tests below pin the three outcomes apart, because the rule QA round 5 states
# is about the PAIR: a missing outcome may not render as success or as silence, and
# a refusal must not blame the wrong component. A test for either half alone goes
# green on the bug (an empty table IS silence; `peer_unreachable` IS a refusal).


def _ordinary_sessions_args(**fields: object) -> Namespace:
    """`lop sessions`'s own flag set, as `main` hands it to `sessions_command`."""
    flags: dict[str, object] = {
        "json": True,
        "sessions_command": None,
        "all": False,
        "limit": None,
        "peer": "",
        "all_peers": False,
    }
    flags.update(fields)
    return Namespace(**flags)


def _live_relay(root: Path, monkeypatch: pytest.MonkeyPatch) -> relay.RelayServer:
    """A REAL relay on this root, reached the way the CLI reaches it: its record.

    The relay is the only process that speaks the mesh, so "the relay answered"
    cannot be unit-tested at a lesser boundary. It publishes its own record on
    ``start()``, and the CLI finds that record through the ambient config dir and
    dials loopback — which is the path an operator's `lop sessions --peer` takes.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    server = relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    server.start()
    return server


def _admit_a_member_that_cannot_answer(
    server: relay.RelayServer, *, name: str, endpoints: list[str]
) -> types.NetworkRecord:
    """A network this device admins, holding ONE member nothing can dial.

    Built from the RELAY's own identity, because ``_fan_out_catalog`` skips the
    member whose id is its own: a record naming a different self would make the
    relay dial itself, and the test would then be about a different failure than
    the one it claims.
    """
    from local_operator.network import wire

    record = types.NetworkRecord(
        network_id=store.new_network_id(),
        name="home-net",
        created_by=server.identity.device_id,
        self_device_id=server.identity.device_id,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )
    relay.admit(
        record,
        device_id=server.identity.device_id,
        public_key=server.identity.public_key,
        name=server.identity.name,
        role="admin",
        added_by=server.identity.device_id,
        added_via="self",
        root=server.root,
        persist=False,
    )
    relay.admit(
        record,
        device_id="d_" + "b" * 32,
        public_key=wire.b64u(bytes(range(32))),
        name=name,
        role="drive",
        endpoints=list(endpoints),
        root=server.root,
        persist=False,
    )
    store.save(record, server.root)
    store.save_secrets(
        types.SecretState(
            network_id=record.network_id, epoch=1, secret=wire.b64u(bytes(reversed(range(32))))
        ),
        server.root,
    )
    return record


@pytest.mark.parametrize(
    "fields", [{"all_peers": True}, {"peer": "lop-mesh-peer-b"}], ids=["all-peers", "peer"]
)
def test_the_ordinary_session_list_refuses_locally_when_this_devices_relay_is_down(
    fields: dict[str, object],
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q-R5-1, and it is the SAME document `peers` ships, compared field by field.

    ``_relay_down`` is QA's own condition: an admitted member whose own relay was
    never started (`relay_running=false relay_answering=false relay_state=stopped`).
    Both flags must refuse — an `--all-peers` that printed `[]` here told an
    operator "no remote sessions" about a question this device never asked anybody
    — and the refusal must be the FAMILY's: one code and one sentence, checked
    against `_cmd_peers` rather than against a literal, so a second spelling of the
    same incident fails this test instead of shipping beside the first.
    """
    from local_operator import cli as main_cli

    _relay_down(root, monkeypatch)
    assert net_cli._cmd_peers(_args()) == 1  # noqa: SLF001
    reference = json.loads(capsys.readouterr().out)

    assert main_cli.sessions_command(_ordinary_sessions_args(**fields)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["code"] == reference["code"] == "relay_unavailable"
    assert payload["message"] == reference["message"]
    assert "relay" in payload["message"]


@pytest.mark.parametrize(
    "fields", [{"all_peers": True}, {"peer": "lop-mesh-peer-b"}], ids=["all-peers", "peer"]
)
def test_the_human_form_refuses_on_stderr_rather_than_printing_no_sessions(
    fields: dict[str, object],
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The human half of the same finding: rc 0 and a sentence, not rc 0 and none.

    QA round 5 ran this without `--json` and got "no active lop sessions" with an
    empty stderr — the shape an operator reads as "there is nothing out there".
    """
    from local_operator import cli as main_cli

    _relay_down(root, monkeypatch)
    assert main_cli.sessions_command(_ordinary_sessions_args(json=False, **fields)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "relay" in captured.err


def test_with_the_relay_up_an_unknown_peer_is_still_blamed_on_the_peer(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other side of the pair: the relay ANSWERED, so the peer is the reason.

    An empty peer set with a live relay is a fact ("asked, and it holds nothing"),
    not a missing answer: `--all-peers` exits 0 with no rows, and a NAMED device
    that is not in the network keeps the peer's own code and sentence.
    """
    from local_operator import cli as main_cli

    server = _live_relay(root, monkeypatch)
    try:
        assert main_cli.sessions_command(_ordinary_sessions_args(peer="d_absent")) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["code"] == "peer_unreachable"
        assert "reachable" in payload["message"]

        assert main_cli.sessions_command(_ordinary_sessions_args(all_peers=True)) == 0
        assert json.loads(capsys.readouterr().out) == []
    finally:
        server.stop()


def test_with_the_relay_up_a_member_that_does_not_answer_is_blamed_on_that_member(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reachable:false branch, which is what "the peer is genuinely the reason" means.

    A member with an endpoint nothing listens on is asked and fails by name: the
    named form refuses with THAT device's name and the observed reason (not with
    the relay's code), and the merged form still lists with a note on stderr —
    the F-2 contract, which an over-refusal would have broken.
    """
    from local_operator import cli as main_cli

    server = _live_relay(root, monkeypatch)
    try:
        _admit_a_member_that_cannot_answer(
            server, name="lop-mesh-peer-b", endpoints=["127.0.0.1:1"]
        )

        assert main_cli.sessions_command(_ordinary_sessions_args(peer="lop-mesh-peer-b")) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["code"] == "peer_unreachable"
        assert "lop-mesh-peer-b" in payload["message"]
        assert "cannot be reached" in payload["message"]

        assert main_cli.sessions_command(_ordinary_sessions_args(all_peers=True)) == 0
        captured = capsys.readouterr()
        assert json.loads(captured.out) == []
        assert "lop-mesh-peer-b: unreachable" in captured.err
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# QA round 11 / UX round 2 — the merged listing's own three silences
# ---------------------------------------------------------------------------


def test_a_listing_that_names_an_unknown_peer_refuses_in_the_family_s_words(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA round 11, Q-R11-3: a typo'd device must not be answered with a global.

    ``--peer ghost`` used to filter the listing down to nothing and print "no
    sessions are held by other devices right now" with rc 0 — a claim about EVERY
    device, made false by the peer that was holding sessions at that moment —
    while ``--create --peer ghost`` on the same token refused correctly. Both
    halves now name the same condition in the same words, and this asserts the
    REACHABLE peer is still listed normally beside it (a refusal that refused too
    much would pass the first half of this test).
    """
    server = _live_relay(root, monkeypatch)
    try:
        _admit_a_member_that_cannot_answer(
            server, name="lop-mesh-peer-b", endpoints=["127.0.0.1:1"]
        )
        assert net_cli.main(_sessions_args(peer="ghost", json=False)) == 1
        captured = capsys.readouterr()
        assert not captured.out
        assert "not in a network with anything called 'ghost'" in captured.err

        # …and the same token on the WRITE half, so the two sentences cannot
        # drift apart the way they did.
        assert net_cli.main(_sessions_args(peer="ghost", create=True, json=False)) == 1
        write_half = capsys.readouterr().err
        assert "not in a network with anything called 'ghost'" in write_half
    finally:
        server.stop()


def test_a_row_this_device_holds_is_not_attributed_to_nobody(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """UX round 2, U13: ``--all-peers`` includes this machine's own rows.

    The federated listing merges this device's rows in beside the peers', and the
    device column rendered the absent peer block as ``?`` — so the ONE column that
    answers "which device holds what" said "unknown" about the device the reader
    is sitting at, and the row's own help text ("sessions on other devices") was
    false about its first line.
    """
    server = _live_relay(root, monkeypatch)
    try:
        _a_stored_session_with_an_unread_completion(root)
        assert net_cli.main(_sessions_args(all_peers=True, json=False)) == 0
        out = capsys.readouterr().out
        row = [line for line in out.splitlines() if _UNSEEN_ROW in line]
        assert row, out
        assert "?" not in row[0], row[0]
        # The relay's own name for this device when it has one, and the honest
        # fallback when it does not — never a question mark.
        assert server.identity.name in row[0] or "this device" in row[0], row[0]
    finally:
        server.stop()


def test_an_unasked_device_is_named_in_every_empty_listing(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """UX round 2, U14: with a peer's relay down the listing stated a fact it
    had not established.

    ``--all-peers`` answered "no sessions are held by other devices right now"
    while one device had never been asked — a complete-looking answer about a
    set the command had not reached, and the sibling family
    (``lop sessions --all-peers``) already prints ``<device>: unreachable
    (<reason>)`` for exactly that. Both spellings are asserted: the merged one
    names the missing device, and the NAMED one does not follow the failure line
    with a claim about a device it could not ask.
    """
    server = _live_relay(root, monkeypatch)
    try:
        _admit_a_member_that_cannot_answer(
            server, name="lop-mesh-peer-b", endpoints=["127.0.0.1:1"]
        )

        assert net_cli.main(_sessions_args(all_peers=True, json=False)) == 0
        merged = capsys.readouterr().out
        assert "lop-mesh-peer-b: unreachable" in merged, merged
        assert "no sessions are held by other devices right now" not in merged, merged

        assert net_cli.main(_sessions_args(peer="lop-mesh-peer-b", json=False)) == 0
        named = capsys.readouterr().out
        assert "lop-mesh-peer-b: unreachable" in named, named
        assert "no sessions are held by" not in named, named
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Q-R7-1 — the NEEDS column's one shape, in the HUMAN spelling
# ---------------------------------------------------------------------------


#: The id of the stored row these tests render. It has to be a name the catalogue
#: ACCEPTS, not one it parses: what is under test is the row the producer derives
#: from the directory, not the directory's own spelling.
_UNSEEN_ROW = "qr7-1-stored-unseen"


def _a_stored_session_with_an_unread_completion(root: Path) -> None:
    """One stored session, plus an unread completion for it.

    Written through the REAL stores — the directory the catalogue walks and the
    attention store the sidebar reads — because the row under test is the one the
    relay derives from exactly those two.
    """
    from local_operator.session.attention import AttentionStore

    directory = root / "sessions" / _UNSEEN_ROW
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    AttentionStore().publish(
        f"session/{_UNSEEN_ROW}",
        token=str(uuid.uuid4()),
        anchor="qr7-1-anchor",
        kind="complete",
        baseline_seen=False,
    )


def _rendered_row(table: str, session_id: str) -> str:
    """The rendered line for one row, or a failure naming what WAS rendered."""
    for line in table.splitlines():
        if session_id in line:
            return line
    raise AssertionError(f"no row for {session_id} in the listing:\n{table}")


def test_the_human_listing_renders_a_row_whose_unseen_is_true(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Q-R7-1: the NEEDS column survives a row carrying an unread completion.

    ``pending`` is rendered through ``rich.cells.cell_len``, so the one shape that
    killed the whole command was a truthy NON-string: the peer half of a catalogue
    published ``bool(entry.unseen)`` while the local half published
    ``SessionRecord.pending``, and `lop sessions --all-peers` died with
    ``TypeError: object of type 'bool' has no len()`` on any reachable peer holding
    a stored session with ``unseen=True``.

    THE HUMAN SPELLING IS THE POINT. Rounds 5 and 6 both drove ``--json`` — where a
    bool is a value that crosses the wire perfectly well, so the crash could not be
    seen from there — and this test renders the TABLE, for both spellings an
    operator has: ``--all-peers`` (real relay, real producer, real renderer) and
    ``--peer <dev>`` (the same producer's row with the hop stubbed, because the row
    is what crashed, not the hop).
    """
    from local_operator import cli as main_cli

    server = _live_relay(root, monkeypatch)
    try:
        _a_stored_session_with_an_unread_completion(root)

        assert main_cli.sessions_command(_ordinary_sessions_args(all_peers=True, json=False)) == 0
        merged = capsys.readouterr()
        assert "ask" in _rendered_row(merged.out, _UNSEEN_ROW), merged.out

        produced = [
            {**main_cli._REMOTE_ROW_FILL, **row, "locality": "remote"}
            for row in server.local_session_rows()
        ]
        monkeypatch.setattr(
            main_cli, "_remote_listing", lambda **_: main_cli._RemoteListing(produced, [])
        )
        assert (
            main_cli.sessions_command(
                _ordinary_sessions_args(peer=server.identity.device_id, json=False)
            )
            == 0
        )
        named = capsys.readouterr()
        assert "ask" in _rendered_row(named.out, _UNSEEN_ROW), named.out
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Q-R5-2 — the flag the busy refusal names is one this verb must accept
# ---------------------------------------------------------------------------


def test_the_stop_verb_takes_force_and_sends_the_owners_own_mode(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--force` on the mesh verb is the OWNER's `--force`, carried by ``mode``.

    The ladder's busy refusal names ``--force`` as the way past a turn in flight,
    and only the owner's own ``lop stop`` had it: the sentence therefore offered an
    action this verb could not accept (the defect UX round 2 called U7, QA round 5,
    Q-R5-2). The flag is now real here, and it must keep the owner's semantics
    rather than grow a second meaning — so the assertion is the FRAME: ``mode`` is
    the ladder's own spelling, and the relay maps ``immediate`` to
    ``control.stop_session(force=True)`` (``test_session_plane`` owns that hop; the
    flag's meaning is pinned in ``tests/unit/session/runtime/test_control.py``).
    """
    frames: list[dict[str, object]] = []

    def _capture(op: str, **fields: object) -> dict[str, object]:
        frames.append({"op": op, **fields})
        return {"rung": "socket", "outcome": "stopped", "pid": 1, "detail": "stopped"}

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(net_cli, "_relay_answer", _capture)
    assert net_cli.main(_sessions_args(peer=PEER, stop="deadbeefcafe", force=False)) == 0
    assert net_cli.main(_sessions_args(peer=PEER, stop="deadbeefcafe", force=True)) == 0
    capsys.readouterr()
    assert [frame["op"] for frame in frames] == ["peer_session_stop", "peer_session_stop"]
    assert [frame["mode"] for frame in frames] == ["graceful", "immediate"]


def test_force_without_a_stop_is_a_usage_error(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A flag accepted and then quietly dropped is the same untruth as one invented.

    ``--force`` means one thing — the ladder's way past a turn in flight — so a
    caller who passes it without ``--stop`` has asked for an act this verb is not
    performing. The guide's exit-code table gives a usage error its own rc (2), and
    a refusal (rc 1, the relay's code) would misreport it as a relay condition.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    assert net_cli.main(_sessions_args(peer=PEER, force=True)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--force applies to --stop only" in captured.err
