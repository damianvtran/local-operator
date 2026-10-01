"""``lop mobile devices`` — the operator surface ADR 0006 §4 names.

WHAT THIS FILE IS FOR, and what it deliberately does not do. The route contract
is pinned in ``test_push_devices.py`` (codes, statuses, the store's side
effects); this file pins the SURFACE an operator actually drives. Three of its
properties are slice requirements rather than conveniences:

* a device id this computer never registered is refused with its own exit
  status and an error object on **stderr**, before any request is sent — the
  route is idempotent by contract and answers ``{"ok": true}`` for an id it does
  not hold, which is right for the app's retry and wrong for a human's command;
* ``unrevoke`` presents the machine's operator key, which is the whole
  distinction between the machine's surface and a phone's (the daemon refuses
  the route without it), and the CLI **never mints** that key (review round 1,
  R5) — it reports a distinct code instead;
* the rendered text is the product's vocabulary: every state description comes
  from ``push_devices.STATE_DESCRIPTIONS`` (design round 1, D2), and no secret —
  no push token (never stored) and no ``device_key`` — can appear in it.

``cli._mobile_api_call`` is replaced with a recorder rather than a live daemon:
the HTTP hop itself is exercised end-to-end in the slice's loopback evidence and
in QA, and repeating it here would test ``urllib`` rather than this command.
What is NOT stubbed is anything the command decides — the key it reads and
sends, the verb it chooses from the pre-verb state, the id it refuses, and every
character it prints.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from local_operator import cli
from local_operator.mobile import push_devices
from local_operator.paths import config_dir

#: A label long enough that leading with the id pushed the row past column 80 in
#: the round-1 review (123 columns measured). Every render test uses it, because
#: an 80-column terminal is the shape the layout has to survive (D7).
LONG_LABEL = "Damian's iPhone 17 Pro Max (work, replaced battery 2026)"


class _Api:
    """A stand-in for one loopback call, recording everything it was asked for."""

    def __init__(self, responses: list[tuple[int, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[int, str, str, dict[str, str]]] = []

    def __call__(
        self, port: int, method: str, path: str, *, headers: dict[str, str] | None = None
    ) -> tuple[int, Any]:
        self.calls.append((port, method, path, dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected extra call: {method} {path}")
        return self.responses.pop(0)


def _args(command: str | None = None, device_id: str | None = None) -> Any:
    import argparse

    return argparse.Namespace(devices_command=command, device_id=device_id, port=4123)


def _device(device_id: str, state: str, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "device_id": device_id,
        "platform": "ios",
        "app_version": "1.0.0 (12)",
        "registered_at": 1_759_000_000,
        "last_seen_at": 1_759_000_600,
        "state": state,
    }
    entry.update(extra)
    return entry


def _registry(*devices: dict[str, Any]) -> tuple[int, Any]:
    return (200, {"devices": list(devices), "precedence": push_devices.PRECEDENCE})


def _register_a_device_so_the_store_holds_a_key() -> str:
    """Mint the machine's operator key the way the daemon does — by registering."""
    registered = push_devices.register(
        config_dir(),
        {
            "platform": "ios",
            "token": "t",
            "environment": "production",
            "app_version": "1.0.0 (12)",
            "install_id": "9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",
        },
    )
    return str(registered["device_id"])


# ---------------------------------------------------------------------------
# The list
# ---------------------------------------------------------------------------


def test_the_list_leads_with_the_label_and_fits_eighty_columns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D7: the recognisable word first, the id on its own line, and no wrap.

    Measured before this round: a minimal row was 81 columns and a real-world
    label 123, so on an 80-column terminal every row wrapped — mid-label, since
    the label was last. The id is kept in full (it is the argument ``revoke``
    needs); it just does not lead.
    """
    device_id = "a" * 32
    api = _Api([_registry(_device(device_id, push_devices.STATE_LIVE, name=LONG_LABEL))])
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args()) == cli.MOBILE_DEVICES_OK
    out = capsys.readouterr().out
    lines = out.splitlines()
    widest = max(len(line) for line in lines)
    assert widest <= 80, f"widest line is {widest}: {lines!r}"

    # The label opens its row; the id is indented under it, in full.
    assert f"  {LONG_LABEL}" in lines
    assert f"      id {device_id}" in lines
    # The count is a human fact, not the wire's precedence notation (D6).
    assert "1 push device on this computer" in out


def test_the_list_renders_the_modules_own_state_copy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D2/D4: every description, including ``absent``, comes from the module.

    Asserted against ``STATE_DESCRIPTIONS`` rather than against literals: a copy
    change in the module must move the legend with it, which is the whole point
    of lifting the sentences out of the renderer.
    """
    api = _Api(
        [
            _registry(
                _device("a" * 32, push_devices.STATE_LIVE, name="Damian's iPhone"),
                _device("b" * 32, push_devices.STATE_EXPIRED),
            )
        ]
    )
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args()) == cli.MOBILE_DEVICES_OK
    out = capsys.readouterr().out
    for state, description in push_devices.STATE_DESCRIPTIONS.items():
        assert description in out, state
    # …and the renderer holds no copy of its own, which is what makes this cell
    # discriminating rather than tautological: comparing the output against the
    # module passes even if the legend INLINED the same sentence, because the
    # text would be identical today and diverge on the first edit. The sentences
    # are not spelled anywhere in cli.py, so they cannot drift from the module's.
    source = inspect.getsource(cli)
    for description in push_devices.STATE_DESCRIPTIONS.values():
        assert description not in source, description
    # The precedence sentence is WRAPPED, so the assertion collapses whitespace
    # rather than looking for an unwrapped line: the claim is that the sentence
    # the module defines is the sentence rendered, not that it fits one line.
    assert push_devices.PRECEDENCE_SENTENCE in " ".join(out.split())
    # D12: a nameless device is not identified by its platform alone.
    assert "unnamed ios device" in out
    assert "2 push devices on this computer" in out


def test_the_list_shows_the_credential_reading_and_the_authenticated_time(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D11/R3: the label names the field it printed, and the flag is shown.

    ``live`` + ``credential_live: false`` is "registered but not receiving" — the
    state this slice's own ``unrevoke`` test produces — and it must not render as
    a bare ``live``. A row with no ``last_authenticated_at`` (one an earlier build
    wrote, or one S4c has not seen yet) says ``last seen``, because that is the
    field it is actually printing.
    """
    api = _Api(
        [
            _registry(
                _device(
                    "a" * 32,
                    push_devices.STATE_LIVE,
                    name="lapsed phone",
                    credential_live=False,
                    last_authenticated_at=1_754_000_000,
                ),
                _device("b" * 32, push_devices.STATE_LIVE, name="legacy phone"),
            )
        ]
    )
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args()) == cli.MOBILE_DEVICES_OK
    out = capsys.readouterr().out
    assert "credential lapsed" in out
    assert "credential live" not in out
    assert "last authenticated 2025-" in out
    assert "last seen 2025-" in out
    # The credential flag is only rendered when the machine holds a reading: a
    # row with none says nothing rather than defaulting to "live".
    lapsed, legacy = out.split("legacy phone")
    assert "credential" not in legacy


def test_an_empty_registry_says_what_to_do_and_drops_the_legend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D3: the empty render is a next step, not eleven lines about nothing."""
    api = _Api([_registry()])
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args()) == cli.MOBILE_DEVICES_OK
    out = capsys.readouterr().out
    assert "no push devices registered on this computer yet" in out
    assert "install the mobile app and sign in with your portal password" in out
    assert "states:" not in out
    assert push_devices.STATE_DESCRIPTIONS[push_devices.STATE_LIVE] not in out


# ---------------------------------------------------------------------------
# The verbs
# ---------------------------------------------------------------------------


def test_unrevoke_names_the_marker_it_actually_cleared(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D1, the MAJOR: no verb may claim a revoke that never happened.

    ADR §4 rule 1 says the unpair case must say *which computer* left "instead of
    claiming a revoke nobody performed". ``unrevoke`` clears ``revoked_at`` OR
    ``unpaired_at``, so the result line is chosen from the state the row was in
    BEFORE the verb ran — this command's own observation, not a claim.
    """
    device_id = "a" * 32
    cases = [
        # (pre-verb state, the line that must lead the output)
        (push_devices.STATE_REVOKED, f"unrevoked {LONG_LABEL}"),
        (push_devices.STATE_UNPAIRED, f"cleared the unpaired marker on {LONG_LABEL}"),
        (push_devices.STATE_EXPIRED, f"nothing to clear on {LONG_LABEL}"),
        (push_devices.STATE_LIVE, f"nothing to clear on {LONG_LABEL}"),
    ]
    for before, expected in cases:
        api = _Api(
            [
                _registry(_device(device_id, before, name=LONG_LABEL)),
                (200, {"ok": True, "device_id": device_id}),
                _registry(_device(device_id, push_devices.STATE_LIVE, name=LONG_LABEL)),
            ]
        )
        monkeypatch.setattr(cli, "_mobile_api_call", api)
        monkeypatch.setattr(cli, "_mobile_api_call", api, raising=False)
        push_devices.register(
            config_dir(),
            {
                "platform": "ios",
                "token": "t",
                "environment": "production",
                "app_version": "1.0.0 (12)",
                "install_id": "9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",
            },
        )
        assert cli._mobile_devices_command(_args("unrevoke", device_id)) == cli.MOBILE_DEVICES_OK
        out = capsys.readouterr().out
        assert out.splitlines()[0] == expected, (before, out)
        assert "id " + device_id + " — state: live" in out
        # The ADR's own sentence about the token survives every branch.
        assert "no token or credential was restored" in out
        # …and the expired case explains WHICH state it is (D1's wording).
        if before == push_devices.STATE_EXPIRED:
            assert "it is expired, not revoked" in out


def test_the_unrevoke_explanation_is_only_printed_when_it_applies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The expired explanation must not ride along on a real revoke."""
    device_id = "a" * 32
    _register_a_device_so_the_store_holds_a_key()
    api = _Api(
        [
            _registry(_device(device_id, push_devices.STATE_REVOKED)),
            (200, {"ok": True, "device_id": device_id}),
            _registry(_device(device_id, push_devices.STATE_LIVE)),
        ]
    )
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args("unrevoke", device_id)) == cli.MOBILE_DEVICES_OK
    out = capsys.readouterr().out
    assert "it is expired, not revoked" not in out


def test_a_read_back_that_did_not_answer_is_a_sentence_not_a_placeholder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D4: ``state: ?`` is a variable name shown to a person.

    The write succeeded — the refusal path never reaches here — so the line still
    reports what happened; only the state it could not read is said to be unread.
    """
    device_id = "a" * 32
    calls: list[str] = []

    def dying(port: int, method: str, path: str, *, headers: Any = None) -> Any:
        calls.append(method)
        if method == "GET" and len(calls) > 1:
            # The write landed; the READ-BACK is what cannot be made.
            raise cli._MobileApiUnavailable("daemon_unreachable", "boom")
        if method == "DELETE":
            return (200, {"ok": True})
        return _registry(_device(device_id, push_devices.STATE_LIVE))

    monkeypatch.setattr(cli, "_mobile_api_call", dying)
    assert cli._mobile_devices_command(_args("revoke", device_id)) == cli.MOBILE_DEVICES_OK
    out = capsys.readouterr().out
    assert "?" not in out
    assert "state not read back: the daemon did not answer the follow-up list" in out
    assert "delivery stops for it" in out


def test_a_row_gone_from_the_refreshed_registry_reads_as_absent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``absent`` is a real state of the ADR's table, and the legend explains it."""
    device_id = "a" * 32
    api = _Api(
        [
            _registry(_device(device_id, push_devices.STATE_REVOKED)),
            (200, {"ok": True}),
            _registry(),
        ]
    )
    monkeypatch.setattr(cli, "_mobile_api_call", api)
    assert cli._mobile_devices_command(_args("revoke", device_id)) == cli.MOBILE_DEVICES_OK
    assert f"state: {push_devices.STATE_ABSENT}" in capsys.readouterr().out


def test_revoke_reports_the_state_the_device_is_left_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A revoke is not a removal, and the copy says what the device keeps."""
    device_id = "a" * 32
    api = _Api(
        [
            _registry(_device(device_id, push_devices.STATE_LIVE, name="evidence phone")),
            (200, {"ok": True}),
            _registry(_device(device_id, push_devices.STATE_REVOKED, name="evidence phone")),
        ]
    )
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args("revoke", device_id)) == cli.MOBILE_DEVICES_OK
    _port, method, path, headers = api.calls[1]
    assert (method, path) == ("DELETE", f"/api/push/devices/{device_id}")
    assert headers == {}, "revoke is not operators-only: a device may revoke itself"
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "revoked evidence phone"
    assert f"state: {push_devices.STATE_REVOKED}" in out


# ---------------------------------------------------------------------------
# The failures
# ---------------------------------------------------------------------------


def test_unrevoke_presents_this_machines_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The header IS the distinction: a phone's session can never send it.

    The tunnel gateway rebuilds a device's request headers from a fixed allowlist,
    so a device cannot carry a header of our choosing at all — and even if one
    could, it would not have this machine's key. Asserted here on the call the
    command makes, with the key read from this computer's own store.
    """
    device_id = _register_a_device_so_the_store_holds_a_key()
    api = _Api(
        [
            _registry(_device(device_id, push_devices.STATE_REVOKED)),
            (200, {"ok": True, "device_id": device_id}),
            _registry(_device(device_id, push_devices.STATE_LIVE)),
        ]
    )
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args("unrevoke", device_id)) == cli.MOBILE_DEVICES_OK
    _port, method, path, headers = api.calls[1]
    assert (method, path) == ("POST", f"/api/push/devices/{device_id}/unrevoke")
    assert headers == {push_devices.OPERATOR_KEY_HEADER: push_devices.operator_key(config_dir())}


def test_a_store_with_no_operator_key_is_a_distinct_refusal_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """R5: the CLI reads the key, and never becomes a second writer.

    A store written before this build holds devices and no key. The old code
    minted one here, which made the CLI a second writer of a file the daemon owns
    — a read-modify-write that can lose a concurrent registration. Now the CLI
    reports ``operator_key_missing`` and leaves the store byte-identical.
    """
    device_id = "a" * 32
    store = config_dir() / push_devices.PUSH_DEVICES_STORE_NAME
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        '{"devices":[{"device_id":"' + device_id + '","platform":"ios",'
        '"environment":"production","app_version":"1.0.0 (12)",'
        '"install_id":"9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",'
        '"registered_at":1759000000,"last_seen_at":1759000000}]}'
    )
    before = store.read_bytes()
    api = _Api([_registry(_device(device_id, push_devices.STATE_REVOKED))])
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args("unrevoke", device_id)) == cli.MOBILE_DEVICES_FAILED
    err = capsys.readouterr().err
    assert "error: operator_key_missing:" in err
    assert "mints it when a device registers" in err
    assert store.read_bytes() == before, "the CLI must not write this store"
    assert [call[1] for call in api.calls] == ["GET"], "no mutation may be attempted"


def test_a_corrupt_store_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """N2: the one un-guarded call in this command.

    ``operator_key`` reads the store, so a store that went unreadable between the
    ``GET`` and the ``POST`` raised out of ``mobile_command`` into ``main``'s
    generic handler, which prints a traceback and no code.
    """
    device_id = "a" * 32
    store = config_dir() / push_devices.PUSH_DEVICES_STORE_NAME
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("{not json")
    api = _Api([_registry(_device(device_id, push_devices.STATE_REVOKED))])
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args("unrevoke", device_id)) == cli.MOBILE_DEVICES_FAILED
    err = capsys.readouterr().err
    assert "error: registry_corrupt:" in err
    assert "Traceback" not in err


def test_an_unknown_id_exits_three_before_any_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal is resolved against the registry, not inferred from a route.

    The route is idempotent by contract and answers ``{"ok": true}`` for an id it
    does not hold — right for the app's retry, wrong for a human's command, which
    would report success for a revoke of nothing.
    """
    api = _Api([_registry(_device("a" * 32, push_devices.STATE_LIVE))])
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args("revoke", "not-a-real-id")) == (
        cli.MOBILE_DEVICES_NO_SUCH_DEVICE
    )
    err = capsys.readouterr().err
    assert f"error: {push_devices.DEVICE_ABSENT_CODE}:" in err
    assert "not-a-real-id" in err
    assert [call[1] for call in api.calls] == ["GET"], "no mutation may be attempted"


def test_a_missing_device_id_is_a_usage_failure_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D9/N1: every failure of this command goes to stderr with ``error:``.

    The missing-id case used to print to stdout, which a script capturing the
    streams separately reads as normal output, and it returned a status the help
    text did not document.
    """
    api = _Api([])
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args("revoke", None)) == cli.MOBILE_DEVICES_USAGE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: device_id_required:" in captured.err
    assert "lop mobile devices revoke <device_id>" in captured.err
    assert api.calls == [], "usage is decided before any request"


def test_machine_only_on_the_operator_surface_gets_the_machine_side_remedy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """D5: the ADR's sentence is written for a phone, and this reader is not one.

    Rendered by ``lop`` on the computer, "use the computer or your account" names
    the thing the reader is already using and the account they may not have. The
    code stays (``machine_only``); the remedy becomes the mismatch it actually is.
    """
    device_id = _register_a_device_so_the_store_holds_a_key()
    api = _Api(
        [
            _registry(_device(device_id, push_devices.STATE_REVOKED)),
            (
                403,
                {
                    "code": push_devices.MACHINE_ONLY_CODE,
                    "error": push_devices.MACHINE_ONLY_MESSAGE,
                },
            ),
        ]
    )
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args("unrevoke", device_id)) == cli.MOBILE_DEVICES_FAILED
    err = capsys.readouterr().err
    assert push_devices.MACHINE_ONLY_CODE in err
    assert "does not match the daemon's registry" in err
    assert "check --port or restart the daemon" in err
    assert push_devices.MACHINE_ONLY_MESSAGE not in err, "the device-facing sentence is the app's"


def test_no_password_is_its_own_failure_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`credential_missing` is not `daemon_unreachable`: the remedies differ."""

    def missing(port: int, method: str, path: str, *, headers: Any = None) -> Any:
        raise cli._MobileApiUnavailable(
            "credential_missing",
            "no mobile password is set on this computer — run `lop mobile install`",
        )

    monkeypatch.setattr(cli, "_mobile_api_call", missing)
    assert cli._mobile_devices_command(_args("revoke", "a" * 32)) == cli.MOBILE_DEVICES_FAILED
    err = capsys.readouterr().err
    assert "credential_missing" in err
    assert "lop mobile install" in err


def test_an_unreachable_daemon_is_reported_rather_than_guessed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No answer is not an empty registry: the command fails and says which.

    The distinction matters more here than anywhere else in this file — a phone's
    ``revoke`` command that silently did nothing is worse than one that reports
    the daemon is down, because the user believes the device is cut off.
    """

    def unavailable(port: int, method: str, path: str, *, headers: Any = None) -> Any:
        raise cli._MobileApiUnavailable(
            "daemon_unreachable", f"the mobile daemon on port {port} did not answer (boom)"
        )

    monkeypatch.setattr(cli, "_mobile_api_call", unavailable)
    assert cli._mobile_devices_command(_args()) == cli.MOBILE_DEVICES_FAILED
    err = capsys.readouterr().err
    assert "daemon_unreachable" in err
    assert "did not answer" in err


def test_the_help_documents_every_exit_status(capsys: pytest.CaptureFixture[str]) -> None:
    """D8/N1: the epilog is rendered from the constants the code returns.

    It used to promise "1 any other failure" while a missing device id returned
    2 — a help text that names a status the command does not use is worse than no
    help text, because it is believed.
    """
    parser = cli.build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["mobile", "devices", "--help"])
    help_text = capsys.readouterr().out
    for status in (
        cli.MOBILE_DEVICES_OK,
        cli.MOBILE_DEVICES_FAILED,
        cli.MOBILE_DEVICES_USAGE,
        cli.MOBILE_DEVICES_NO_SUCH_DEVICE,
    ):
        assert f"{status} " in help_text or f"{status} ok" in help_text
    assert "missing device id" in help_text
