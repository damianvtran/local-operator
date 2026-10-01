"""``lop mobile devices`` — the operator surface ADR 0006 §4 names.

WHAT THIS FILE IS FOR, and what it deliberately does not do. The route contract
is pinned in ``test_push_devices.py`` (codes, statuses, the store's side
effects); this file pins the SURFACE an operator actually drives, because three
of its properties are the slice's requirements rather than conveniences:

* a device id this computer never registered is refused with its own exit
  status and an error OBJECT, before any request is sent — the route is
  idempotent by contract and answers ``{"ok": true}`` for an id it does not
  hold, which is right for the app's retry and wrong for a human's command;
* ``unrevoke`` presents the machine's operator key, which is the whole
  distinction between the machine's surface and a phone's (the daemon refuses
  the route without it);
* the list renders the module's state vocabulary and prints no secret — no push
  token (never stored) and no ``device_key`` (returned once, at registration).

``cli._mobile_api_call`` is replaced with a recorder rather than a live daemon:
the HTTP hop itself is exercised end-to-end against a real daemon in the slice's
loopback evidence, and repeating it here would test ``urllib`` rather than this
command. What is NOT stubbed is anything the command decides — the key it reads
and sends, the id it refuses, and the text it prints.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator import cli
from local_operator.mobile import push_devices
from local_operator.paths import config_dir


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
        "credential_live": state == push_devices.STATE_LIVE,
        "last_authenticated_at": 1_759_000_600,
    }
    entry.update(extra)
    return entry


def test_the_list_renders_the_states_and_prints_no_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every state in the module's vocabulary, and the one constant that must not drift.

    The credential sentence is rendered from ``CLOUD_IDLE_DROP_DAYS`` rather than
    spelled, so the number the operator reads is the number the module defines —
    and the sentence says the machine CANNOT see that drop, because inventing a
    label for it would tell the user their device was removed when nothing here
    can know that.
    """
    # A machine that has registered a device holds an operator key; it must not
    # reach stdout, and neither may a device's own key.
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
    operator_key = push_devices.operator_key(config_dir())
    device_key = "device-key-that-must-not-be-printed"
    devices = [
        _device("a" * 32, push_devices.STATE_LIVE, name="Damian's iPhone"),
        _device("b" * 32, push_devices.STATE_EXPIRED),
        _device("c" * 32, push_devices.STATE_UNPAIRED),
        _device("d" * 32, push_devices.STATE_REVOKED),
        {**_device("e" * 32, push_devices.STATE_LIVE), "device_key": device_key},
    ]
    api = _Api([(200, {"devices": devices, "precedence": push_devices.PRECEDENCE})])
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args()) == 0
    out = capsys.readouterr().out
    for state in push_devices.DEVICE_STATES:
        assert state in out, state
    assert push_devices.PRECEDENCE in out
    assert f"{push_devices.CLOUD_IDLE_DROP_DAYS} days" in out
    assert device_key not in out
    assert operator_key not in out
    assert api.calls == [(4123, "GET", "/api/push/devices", {})]


def test_a_never_registered_id_is_refused_before_any_request_is_sent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No such device is its own exit status and an error object, not a success.

    The check happens against the list the command already read: the DELETE route
    answers ``{"ok": true}`` for an id it does not hold (the app's retry contract),
    so a human's command must resolve the id itself rather than report a revoke
    the registry never had. Nothing is sent — the recorded calls say so.
    """
    api = _Api([(200, {"devices": [_device("a" * 32, push_devices.STATE_LIVE)]})])
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    status = cli._mobile_devices_command(_args("revoke", "not-a-real-id"))
    assert status == cli.MOBILE_DEVICES_NO_SUCH_DEVICE
    assert status != 0 and status != 1, "it must be distinguishable from a generic failure"
    err = capsys.readouterr().err
    assert push_devices.DEVICE_ABSENT_CODE in err
    assert "not-a-real-id" in err
    assert [call[1] for call in api.calls] == ["GET"], "no mutation may be attempted"


def test_unrevoke_presents_this_machines_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The header IS the distinction: a phone's session can never send it.

    The tunnel gateway rebuilds a device's request headers from a fixed allowlist,
    so a device cannot carry a header of our choosing at all — and even if one
    could, it would not have this machine's key. Asserted here on the call the
    command makes, with the key read from this computer's own store.
    """
    device_id = "a" * 32
    api = _Api(
        [
            (200, {"devices": [_device(device_id, push_devices.STATE_REVOKED)]}),
            (200, {"ok": True, "device_id": device_id}),
            (200, {"devices": [_device(device_id, push_devices.STATE_LIVE)]}),
        ]
    )
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args("unrevoke", device_id)) == 0
    _port, method, path, headers = api.calls[1]
    assert (method, path) == ("POST", f"/api/push/devices/{device_id}/unrevoke")
    assert headers == {push_devices.OPERATOR_KEY_HEADER: push_devices.operator_key(config_dir())}
    out = capsys.readouterr().out
    # The RESULTING state, read back: unrevoke restores no token, so what the
    # device can do next is the state after the marker cleared.
    assert f"state: {push_devices.STATE_LIVE}" in out
    assert "must register again" in out


def test_revoke_reports_the_state_the_device_is_actually_left_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A revoke is not a removal, and the copy says what the device keeps."""
    device_id = "a" * 32
    api = _Api(
        [
            (200, {"devices": [_device(device_id, push_devices.STATE_LIVE)]}),
            (200, {"ok": True}),
            (200, {"devices": [_device(device_id, push_devices.STATE_REVOKED)]}),
        ]
    )
    monkeypatch.setattr(cli, "_mobile_api_call", api)

    assert cli._mobile_devices_command(_args("revoke", device_id)) == 0
    _port, method, path, headers = api.calls[1]
    assert (method, path) == ("DELETE", f"/api/push/devices/{device_id}")
    assert headers == {}, "revoke is not operators-only: a device may revoke itself"
    assert f"state: {push_devices.STATE_REVOKED}" in capsys.readouterr().out


def test_a_refused_route_is_reported_with_its_own_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Whatever the route refuses with, the command prints that and fails."""
    device_id = "a" * 32
    api = _Api(
        [
            (200, {"devices": [_device(device_id, push_devices.STATE_REVOKED)]}),
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

    assert cli._mobile_devices_command(_args("unrevoke", device_id)) == 1
    err = capsys.readouterr().err
    assert push_devices.MACHINE_ONLY_CODE in err
    assert push_devices.MACHINE_ONLY_MESSAGE in err


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
    assert cli._mobile_devices_command(_args("revoke", "a" * 32)) == 1
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
    assert cli._mobile_devices_command(_args()) == 1
    err = capsys.readouterr().err
    assert "daemon_unreachable" in err
    assert "did not answer" in err
