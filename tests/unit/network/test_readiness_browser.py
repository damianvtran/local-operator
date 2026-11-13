"""(l) The ``browser`` readiness row: would a session on that peer get a browser?

Non-gating equipment. Pins the four readings the operator acts on — ``n/a``
(a user's browser host serves the device, or it is not Linux), ``ok``, ``warn``
not installed (remedy names the command AND the operator's authorisation) and
``warn`` installed-but-unusable (remedy is ``--repair``) — plus the older-peer
fallback, the row's place in the peer-side set, and that it never gates.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import readiness

PEER = "d_peer"
LABEL = "lop-remote-node-2"


def _row(fact: Any) -> dict[str, Any]:
    facts = {} if fact is None else {"browser": fact}
    member = SimpleNamespace(device_id=PEER, name=LABEL)
    row = readiness.browser_row(member, facts, peer_label=LABEL)
    row["state"] = readiness.row_state(row)
    return row


NODE_OK = {
    "host": "node",
    "chrome_version": "Google Chrome 157.0.8103.78",
    "extension_version": "0.1.27",
    "unit_active": True,
    "unit_state": "active",
    "daemon_healthy": True,
    "paired": True,
}


def test_ok_names_chrome_extension_and_pairing() -> None:
    row = _row(NODE_OK)
    assert row["state"] == "ok"
    assert row["detail"] == "Chrome 157.0.8103.78 (node-managed, headless), extension 0.1.27 paired"
    assert row["remedies"] == []


def test_not_installed_warns_with_the_authorised_setup_remedy() -> None:
    row = _row({"host": "none"})
    assert row["state"] == "warn"
    assert row["code"] == readiness.CODE_NOT_INSTALLED
    assert row["detail"] == f"sessions on {LABEL} have no browser tool"
    (remedy,) = row["remedies"]
    assert f"on {LABEL}: `lop browser setup --node --authorised-by '<who>'`" in remedy
    assert "operator authorisation required" in remedy
    assert "Headless nodes" in remedy


@pytest.mark.parametrize(
    ("change", "clause"),
    [
        ({"unit_active": False, "unit_state": "inactive"}, "its Chrome unit is inactive"),
        ({"daemon_healthy": False}, "its browser bridge daemon is not answering"),
        ({"paired": False}, "its extension is not paired"),
    ],
)
def test_installed_but_unusable_warns_with_the_repair_remedy(
    change: dict[str, Any], clause: str
) -> None:
    row = _row({**NODE_OK, **change})
    assert row["state"] == "warn"
    assert row["code"] == readiness.CODE_UNUSABLE
    assert clause in row["detail"]
    assert row["detail"].endswith("sessions there have no browser tool")
    assert row["remedies"] == [
        f"on {LABEL}: `lop browser setup --node --repair --authorised-by '<who>'`"
    ]


@pytest.mark.parametrize("fact", [{"host": "user"}, {"host": "unsupported_platform"}])
def test_user_hosts_and_other_platforms_are_not_applicable(fact: dict[str, Any]) -> None:
    row = _row(fact)
    assert row["state"] == "n/a"
    assert row["ok"] is True
    assert row["detail"].startswith("n/a:")
    assert row["remedies"] == []


def test_an_older_peer_reads_peer_too_old_not_no_browser() -> None:
    row = _row(None)
    assert row["code"] == readiness.CODE_PEER_TOO_OLD
    assert row["capability"] == readiness.CAPABILITY_BROWSER


def test_an_unreadable_fact_is_unknown() -> None:
    row = _row({"host": "unknown", "error": "boom"})
    assert row["code"] == readiness.CODE_UNKNOWN
    assert "boom" in row["detail"]


def test_the_row_never_gates_onboarding() -> None:
    assert readiness.CAPABILITY_BROWSER in readiness.NON_GATING_EQUIPMENT
    assert readiness.CAPABILITY_BROWSER in readiness.PEER_SIDE_CHECKS
    failing = [_row({"host": "none"}), _row({**NODE_OK, "paired": False})]
    assert readiness.onboarding_failures(failing) == []


def test_the_fact_rides_the_peer_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setattr(readiness, "browser_fact", lambda: {"host": "none"})
    facts = readiness.collect_peer_facts(tmp_path)
    assert facts["browser"] == {"host": "none"}


def test_browser_fact_wrapper_degrades_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.browser_bridge import node_host

    def boom() -> dict[str, Any]:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(node_host, "browser_fact", boom)
    assert readiness.browser_fact() == {"host": "unknown", "error": "probe failed"}
