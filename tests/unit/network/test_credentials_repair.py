"""The owner-side repair notice (decision memo item D): DERIVED, and shown everywhere.

The predicate, the row shape and the two surfaces that carry it are pinned here:

* ``open_reports`` is the pure derivation — a ``credential.report`` with
  ``failure=interactive_required`` for key K is open unless a LATER
  ``credential.grant`` for K follows it in the read window;
* ``repair_checks`` turns one open report into the ``credential_repair`` check row
  that both ``lop network doctor`` paths and the ``/network`` panel consume;
* the doctor integration cells drive the REAL relay doctor and the REAL local
  fallback, because the e2e expectation ("runA's doctor shows the row") is a claim
  about those two call sites, not about the helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import cli as net_cli
from local_operator.network import relay, store, types, wire
from local_operator.network.credentials import repair as repair_mod
from local_operator.network.credentials.messages import render_repair_notice

SELF = "d_" + "1" * 32
PEER = "d_" + "2" * 32
NETWORK = "n_0123456789abcdef01234567"
OTHER_NETWORK = "n_ffffffffffffffffffffffff"
MCP_KEY = "mcp:https://mcp.example.com"
PROVIDER_KEY = "openai"


def _record() -> types.NetworkRecord:
    record = types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        epoch=1,
        self_device_id=SELF,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )
    record.members.append(
        types.MemberRecord(
            device_id=SELF,
            public_key=wire.b64u(b"a" * 32),
            role="admin",
            capabilities=sorted(types.capabilities_for_role("admin")),
            added_via="self",
        )
    )
    record.members.append(
        types.MemberRecord(
            device_id=PEER,
            public_key=wire.b64u(b"b" * 32),
            role="drive",
            capabilities=sorted(types.capabilities_for_role("drive")),
            added_via="invite",
            name="laptop",
        )
    )
    return record


def _report(
    log: audit_mod.AuditLog,
    *,
    key: str = MCP_KEY,
    failure: str = "interactive_required",
    network_id: str = NETWORK,
    sub: str = PEER,
) -> None:
    log.record(
        audit_mod.AuditEvent(
            event="credential.report",
            actor=SELF,
            network_id=network_id,
            subject=sub,
            detail={"credential_key": key, "act": SELF, "sub": sub, "failure": failure},
        )
    )


def _grant(
    log: audit_mod.AuditLog, *, key: str = MCP_KEY, network_id: str = NETWORK, sub: str = PEER
) -> None:
    log.record(
        audit_mod.AuditEvent(
            event="credential.grant",
            actor=SELF,
            network_id=network_id,
            subject=sub,
            detail={"credential_key": key, "act": SELF, "sub": sub},
        )
    )


# ---------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------


def test_an_open_report_becomes_the_check_row_the_surfaces_read(root: Path) -> None:
    """The row's whole shape: the key, the asker, the producer's sentence, the remedy."""
    record = _record()
    log = audit_mod.AuditLog(root)
    _report(log)

    rows = repair_mod.repair_checks(record, log=log)

    assert len(rows) == 1, rows
    row = rows[0]
    assert row["check"] == "credential_repair"
    assert row["ok"] is False
    assert row["network_id"] == NETWORK
    assert row["credential_name"] == MCP_KEY
    assert row["device_id"] == PEER
    assert row["device_name"] == "laptop"
    # The wire diagnostic's own wording, recomposed for the owner's operator.
    assert row["detail"] == render_repair_notice("laptop", MCP_KEY)
    assert f"/mcp login {MCP_KEY[4:]}" in row["detail"]
    assert row["remedies"] == [f"run `/mcp login {MCP_KEY[4:]}` on this device"]


def test_a_later_grant_closes_the_row(root: Path) -> None:
    """The repair landing IS the clear: the owner's next successful borrow writes it."""
    record = _record()
    log = audit_mod.AuditLog(root)
    _report(log)
    _grant(log)

    assert repair_mod.repair_checks(record, log=log) == []


def test_a_grant_older_than_the_report_does_not_close_it(root: Path) -> None:
    """Only a LATER grant clears: an old success says nothing about this failure."""
    record = _record()
    log = audit_mod.AuditLog(root)
    _grant(log)
    _report(log)

    assert len(repair_mod.repair_checks(record, log=log)) == 1


@pytest.mark.parametrize("failure", ["", "quota", "invalid", "unauthorized"])
def test_only_interactive_required_reports_are_repairs(failure: str, root: Path) -> None:
    """The failure matrix has nine codes; exactly one of them wants a login HERE."""
    record = _record()
    log = audit_mod.AuditLog(root)
    _report(log, failure=failure)

    assert repair_mod.repair_checks(record, log=log) == []


def test_multiple_keys_are_separately_derived_and_cleared(root: Path) -> None:
    """Two dead logins are two rows; repairing one leaves the other standing."""
    record = _record()
    log = audit_mod.AuditLog(root)
    _report(log, key=MCP_KEY)
    _report(log, key=PROVIDER_KEY)

    rows = repair_mod.repair_checks(record, log=log)
    assert [row["credential_name"] for row in rows] == [MCP_KEY, PROVIDER_KEY]
    # A provider key's remedy is the provider login, spelled once (messages.py).
    assert rows[1]["remedies"] == [f"run `/login {PROVIDER_KEY}` on this device"]

    _grant(log, key=MCP_KEY)
    rows = repair_mod.repair_checks(record, log=log)
    assert [row["credential_name"] for row in rows] == [PROVIDER_KEY]


def test_a_report_for_another_network_is_not_this_records(root: Path) -> None:
    record = _record()
    log = audit_mod.AuditLog(root)
    _report(log, network_id=OTHER_NETWORK)

    assert repair_mod.repair_checks(record, log=log) == []


def test_the_tail_window_bounds_the_derivation(root: Path) -> None:
    """The window is the audit tail, and its bound only ever errs toward silence.

    With a window that no longer holds the report, no row is claimed; with the
    shipped window it is. (The converse miss is impossible by construction: a
    grant later than the report is always still inside any window that holds the
    report, so a repaired key can never read as open.)
    """
    record = _record()
    log = audit_mod.AuditLog(root)
    _report(log, key=MCP_KEY)
    _report(log, key=PROVIDER_KEY)
    _grant(log, key=PROVIDER_KEY)

    assert repair_mod.repair_checks(record, log=log, limit=2) == []
    rows = repair_mod.repair_checks(record, log=log, limit=3)
    assert [row["credential_name"] for row in rows] == [MCP_KEY]


def test_the_rows_survive_the_agent_tools_scrubber(root: Path) -> None:
    """No field NAME may carry a secret marker, or the tool drops the field.

    The credentials listing already paid for this once (``network/cli.py``): the
    agent tool drops any field whose name contains "key", and the first name this
    fix reached for was ``credential_key`` — which would have silently rendered
    ``None`` where the credential's name belongs on every agent-facing digest.
    """
    from local_operator.network.tool import _SECRET_KEY_MARKERS

    record = _record()
    log = audit_mod.AuditLog(root)
    _report(log)
    row = repair_mod.repair_checks(record, log=log)[0]

    for field in row:
        assert not any(marker in field.lower() for marker in _SECRET_KEY_MARKERS), field


# ---------------------------------------------------------------------------
# the two doctor paths (the surfaces the e2e expectation names)
# ---------------------------------------------------------------------------


def test_the_local_doctor_carries_the_row_and_the_repair_clears_it(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """``_doctor_locally`` — the relay-down path — derives it from the same audit."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    record = _record()
    store.save(record, root)
    log = audit_mod.AuditLog(root)
    _report(log)
    log.flush()

    payload = net_cli._doctor_locally(argparse.Namespace())  # noqa: SLF001
    repairs = [check for check in payload["checks"] if check["check"] == "credential_repair"]
    assert len(repairs) == 1, payload["checks"]
    assert repairs[0]["device_id"] == PEER

    # THE OWNER RUNS THE LOGIN (the next borrow grants), and the row is gone.
    _grant(log)
    log.flush()
    payload = net_cli._doctor_locally(argparse.Namespace())  # noqa: SLF001
    assert [c for c in payload["checks"] if c["check"] == "credential_repair"] == []


def test_the_relays_doctor_carries_the_row_and_a_later_grant_clears_it(root: Path) -> None:
    """The relay path — the one a normally-running device actually reads."""
    record = _record()
    store.save(record, root)
    log = audit_mod.AuditLog(root)
    server = relay.RelayServer(root=root, audit=log)
    _report(log)

    findings = server.doctor()["checks"]
    repairs = [check for check in findings if check["check"] == "credential_repair"]
    assert len(repairs) == 1, findings
    assert repairs[0]["credential_name"] == MCP_KEY
    assert repairs[0]["device_id"] == PEER

    _grant(log)
    findings = server.doctor()["checks"]
    assert [c for c in findings if c["check"] == "credential_repair"] == []


def test_the_doctor_summary_turns_unhealthy_for_an_open_repair(
    monkeypatch: pytest.MonkeyPatch, root: Path, capsys: Any
) -> None:
    """The row is not decoration: an open repair makes ``doctor`` answer unhealthy.

    ``ok`` is derived from the rows (QA round 1), so a repair row must flip it —
    and the human line must carry the sentence, not a raw token: the detail is
    prose, and ``doctor_detail_words`` passes it through whole.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    record = _record()
    store.save(record, root)
    log = audit_mod.AuditLog(root)
    _report(log)
    log.flush()

    assert net_cli._cmd_doctor(argparse.Namespace(json=True, peer="")) == 1  # noqa: SLF001
    machine = capsys.readouterr().out
    assert '"check": "credential_repair"' in machine

    assert net_cli._cmd_doctor(argparse.Namespace(json=False, peer="")) == 1  # noqa: SLF001
    human = capsys.readouterr().out
    # The human line carries the sentence WHOLE — the detail is prose, and
    # ``doctor_detail_words`` passes prose through rather than tokenising it.
    assert "FAIL credential_repair" in human
    assert "laptop needs '/mcp login https://mcp.example.com'" in human
    assert "its borrowed credential cannot be refreshed" in human
