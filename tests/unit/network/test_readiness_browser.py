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


def test_a_node_that_holds_its_operator_key_warns() -> None:  # PR #3938 review R2-3
    """Usable browser, but no person can approve a new site: the node could sign
    its own approvals, so the row says so and names the remedy."""
    row = _row({**NODE_OK, "operator_key_held_locally": True})
    assert row["state"] == "warn"
    assert "holds the operator key's private half" in row["detail"]
    assert "--authorised-by" in row["remedies"][0]
    assert _row({**NODE_OK, "operator_key_held_locally": False})["state"] == "ok"


def test_not_installed_warns_with_the_authorised_setup_remedy() -> None:
    row = _row({"host": "none"})
    assert row["state"] == "warn"
    assert row["code"] == readiness.CODE_NOT_INSTALLED
    assert row["detail"] == f"sessions on {LABEL} have no browser tool"
    (remedy,) = row["remedies"]
    assert f"on {LABEL}: `lop browser setup --node --authorised-by '<who, when>'`" in remedy
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
        f"on {LABEL}: `lop browser setup --node --repair --authorised-by '<who, when>'`"
    ]


@pytest.mark.parametrize(
    ("fact", "clause"),
    [
        ({"host": "user"}, "a user's browser host"),
        ({"host": "unsupported_platform", "platform": "darwin"}, "darwin"),
        # R1-6: a Linux node setup would refuse reads n/a, never "run setup".
        ({"host": "unsupported_platform", "platform": "linux aarch64"}, "linux aarch64"),
        ({"host": "unsupported_distro"}, "dnf/rpm"),
    ],
)
def test_user_hosts_and_unsupported_nodes_are_not_applicable(
    fact: dict[str, Any], clause: str
) -> None:
    row = _row(fact)
    assert row["state"] == "n/a"
    assert row["ok"] is True
    assert clause in row["detail"]
    # U10: the state column already says n/a; the detail must not repeat it.
    assert not row["detail"].startswith("n/a")
    assert row["remedies"] == []


def test_an_older_peer_reads_like_its_neighbouring_rows() -> None:
    """U6: "predates readiness reporting" was untrue of every 0.102.x peer, which
    answers readiness and simply lacks this one fact."""
    row = _row(None)
    assert row["capability"] == readiness.CAPABILITY_BROWSER
    assert row["code"] == readiness.CODE_UNKNOWN
    assert row["detail"] == "the peer's answer did not carry this check"
    assert "predates" not in row["detail"]
    (remedy,) = row["remedies"]
    assert remedy.startswith(f"update lop on {LABEL}")


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
