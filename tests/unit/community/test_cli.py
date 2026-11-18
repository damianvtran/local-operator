"""``lop community`` through the real argparse tree and the real client over the fake hub.

Pins the documented exit codes (TUI doc, "Commands for scripts"): 0 posted,
3 refused, 4 rate limited, 5 moderation unavailable, 6 offline, 7 restricted,
8 guidelines acceptance needed; ``cancel`` 0 cancelled / 9 already posted; and
that ``join`` sends nothing unless the named version is the current one.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from local_operator.cli import build_cli_parser
from local_operator.community import cli
from tests.unit.community.fakehub import (
    COMMUNITY,
    JOINED,
    KEY,
    FakeHub,
    make_client,
    outcome,
)

SUBMIT = "/community/v1/communities/local-operator/conversations/conv_general/submissions"
JOIN = "/community/v1/communities/local-operator/join"
CANCEL = f"/community/v1/communities/local-operator/submissions/{KEY}:cancel"


def _hub(
    joined: bool = True,
    submit: tuple[int, Any] | None = None,
    cancel: tuple[int, Any] | None = None,
) -> FakeHub:
    hub = FakeHub()

    def override(sent: Any) -> httpx.Response | None:
        if sent.method == "GET" and sent.path == "/community/v1/communities":
            body = {
                "joined": [COMMUNITY] if joined else [],
                "featured": [] if joined else [COMMUNITY],
            }
            return httpx.Response(200, json=body)
        if sent.path.endswith("/bootstrap") and joined:
            return httpx.Response(200, json=JOINED)
        if sent.path == SUBMIT and submit is not None:
            return httpx.Response(submit[0], json=submit[1])
        if sent.path == CANCEL and cancel is not None:
            return httpx.Response(cancel[0], json=cancel[1])
        return None

    hub.override = override
    return hub


def _run(argv: list[str], hub: FakeHub, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict]:
    args = build_cli_parser().parse_args(["community", *argv, "--json"])
    code = cli.main(args, client_factory=lambda _cfg: make_client(hub))
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip().startswith("{") else {}


@pytest.fixture(autouse=True)
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_COMMUNITY", "1")


def test_dark_refuses_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LOCAL_OPERATOR_COMMUNITY")
    hub = _hub()
    code, payload = _run(["status"], hub, capsys)
    assert code == 1 and payload["code"] == "community_disabled" and hub.sent == []


def test_hidden_from_top_level_help(capsys: pytest.CaptureFixture[str]) -> None:
    # The `resume-click` idiom: no help row (the choices set still names it,
    # exactly as it names resume-click).
    with pytest.raises(SystemExit):
        build_cli_parser().parse_args(["--help"])
    rows = [
        line.strip().split()[0]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("    ") and line.strip()
    ]
    assert "community" not in rows and "resume-click" not in rows


def test_join_needs_the_current_version(capsys: pytest.CaptureFixture[str]) -> None:
    hub = _hub(joined=False)
    code, payload = _run(["join", "--accept-guidelines", "2025-01"], hub, capsys)
    assert code == 8 and payload["version"] == "2026-11-01"
    assert not any(s.path == JOIN for s in hub.sent)
    code, payload = _run(["join", "--accept-guidelines", "2026-11-01"], hub, capsys)
    assert code == 0
    (sent,) = [s for s in hub.sent if s.path == JOIN]
    assert json.loads(sent.body) == {"guidelines_version": "2026-11-01", "accept_guidelines": True}


@pytest.mark.parametrize(
    ("status", "body", "expected", "key"),
    [
        (200, outcome(), 0, "message_id"),
        (
            422,
            {"status": "refused", "category": "spam_or_promotion", "reference": "MOD-AAAA-BBBB"},
            3,
            "reference",
        ),
        (429, {"retry_after": 42, "scope": "account_burst"}, 4, "scope"),
        (503, {"code": "moderation_unavailable"}, 5, "kind"),
        (403, {"message": "locked"}, 7, "kind"),
        (409, {"code": "guidelines_changed", "guidelines_version": "2026-12"}, 8, "version"),
    ],
)
def test_post_exit_codes(
    status: int, body: Any, expected: int, key: str, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = _hub(submit=(status, body))
    code, payload = _run(["post", "#general", "--text", "hello"], hub, capsys)
    assert code == expected, payload
    assert key in payload


def test_cancel_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = _run(["cancel", KEY], _hub(cancel=(200, outcome("cancelled"))), capsys)
    assert code == 0
    code, payload = _run(["cancel", KEY], _hub(cancel=(200, outcome("published"))), capsys)
    assert code == 9 and payload["kind"] == "too_late"


def test_read_prints_the_channel(capsys: pytest.CaptureFixture[str]) -> None:
    code, payload = _run(["read", "#general"], _hub(), capsys)
    assert code == 0 and payload["messages"][0]["text"] == "hello"


def test_leave_needs_yes(capsys: pytest.CaptureFixture[str]) -> None:
    hub = _hub()
    code, _ = _run(["leave"], hub, capsys)
    assert code == 1 and not any(s.path.endswith("/leave") for s in hub.sent)
    code, payload = _run(["leave", "--yes"], hub, capsys)
    assert code == 0 and payload["state"] == "left"
