"""Explicit owner refresh uses the existing broker grant, never new authority."""

from __future__ import annotations

import argparse
import json
from typing import Any

import pytest

from local_operator.network import cli, readiness


@pytest.mark.parametrize("admin", [False, True])
@pytest.mark.parametrize("borrower_role", ["admin", "drive", "read", "", None])
@pytest.mark.parametrize("state", ["expired", "withdrawn", "owner_unavailable", "pending"])
def test_explicit_refresh_remedy_is_only_for_admin_expired_grant(
    admin: bool, borrower_role: str | None, state: str
) -> None:
    from types import SimpleNamespace

    member = SimpleNamespace(device_id="synthetic-peer")
    if borrower_role is not None:
        member.role = borrower_role
    row = readiness.selected_model_row(
        member,
        {"provider": "openai", "owner_device": "synthetic-owner", "state": state},
        selector="openai/gpt-test",
        session_id="synthetic-session",
        peer_label="synthetic-peer",
        observer_is_admin=admin,
    )
    command = "lop network credential refresh openai --json"
    offered = admin and borrower_role == "admin" and state == "expired"
    assert any(command in remedy for remedy in row["remedies"]) is offered
    if state == "expired":
        assert "automatically" in row["remedies"][0]


@pytest.mark.parametrize("flag", ["--with", "--owner", "--scope", "--url", "--path"])
def test_refresh_cannot_select_authority_or_destination(flag: str) -> None:
    parser = argparse.ArgumentParser(prog="lop")
    cli.add_parser(parser.add_subparsers(dest="subcommand"))
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["network", "credential", "refresh", "openai", flag, "synthetic"])
    assert error.value.code == 2


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lop")
    cli.add_parser(parser.add_subparsers(dest="subcommand"))
    return parser.parse_args(["network", "credential", "refresh", "openai", "--json"])


@pytest.mark.parametrize(
    "code",
    ["not_authorised", "grant_invalid", "interactive_required", "not_a_holder", "refresh_failed"],
)
def test_refresh_preserves_owner_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], code: str
) -> None:
    args = _args()
    calls: list[tuple[str, dict[str, Any]]] = []

    def relay(op: str, **fields: Any) -> dict[str, Any]:
        calls.append((op, fields))
        return {"kind": "error", "code": code, "message": "synthetic owner refusal"}

    monkeypatch.setattr(cli, "_relay_call", relay)
    assert cli.main(args) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["code"] == code
    assert calls[0][0] == "credential_grant"
    assert calls[0][1]["force_refresh"] is True
    assert calls[0][1]["credential_key"] == "openai"


def test_refresh_never_prints_grant_material(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args()
    monkeypatch.setattr(
        cli,
        "_relay_call",
        lambda *a, **kw: {"kind": "grant", "access_token": "synthetic-never-print"},
    )
    assert cli.main(args) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["ok"] is True
    assert "synthetic-never-print" not in output
