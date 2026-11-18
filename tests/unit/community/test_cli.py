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


def _run(
    argv: list[str], hub: FakeHub, capsys: pytest.CaptureFixture[str]
) -> tuple[int, dict[str, Any]]:
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
        # R1-9: the documented offline row — the transport never reaches the hub.
        (None, None, 6, "kind"),
    ],
)
def test_post_exit_codes(
    status: int | None, body: Any, expected: int, key: str, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = _hub(submit=None if status is None else (status, body))
    if status is None:
        previous = hub.override
        assert previous is not None

        def unreachable(sent: Any) -> httpx.Response | None:
            if sent.path == SUBMIT:
                raise httpx.ConnectError("unreachable")
            return previous(sent)

        hub.override = unreachable
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


@pytest.mark.parametrize(
    "argv",
    [
        ["post", "#general", "--text", "run lop community post"],
        ["join", "--accept-guidelines", "2026-11-01"],
        ["leave", "--yes"],
        ["cancel", KEY],
        ["read", "#general"],
    ],
)
def test_agent_shell_refuses_acting_and_reading_verbs_before_any_request(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # R1-1: the bash tool marks every command it runs; an injected "run
    # `lop community post …`" must not publish as the owner unapproved.
    monkeypatch.setenv("LOCAL_OPERATOR_AGENT_SHELL", "1")
    hub = _hub()
    code, payload = _run(argv, hub, capsys)
    assert code == 10 and payload["code"] == "agent_shell", payload
    assert "bridge" in payload["use"] or "owner" in payload["use"]
    assert hub.sent == [], "refused before any request"


def test_agent_shell_text_mode_names_the_bridge_route(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_AGENT_SHELL", "1")
    args = build_cli_parser().parse_args(["community", "post", "#general", "--text", "x"])
    assert cli.main(args, client_factory=lambda _cfg: make_client(_hub())) == 10
    out = capsys.readouterr().out
    assert "op=community_post" in out and "exit 10" in out


def test_agent_shell_still_allows_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_AGENT_SHELL", "1")
    code, payload = _run(["status"], _hub(), capsys)
    assert code == 0 and payload["joined"] is True


def test_text_mode_strips_control_sequences_and_json_escapes_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # R1-3: post text and display names are stranger-authored.
    hostile = "hi\x1b]52;c;cm0gLXJmIH4K\x1b\\ \x1b[2J\x1b]0;PWNED\x1b\\ \x9b31mthere"
    hub = _hub()
    previous = hub.override
    assert previous is not None
    history = "/community/v1/communities/local-operator/conversations/conv_general/messages"

    def evil(sent: Any) -> httpx.Response | None:
        if sent.path == history:
            item = {
                "id": "m1",
                "conversation_id": "conv_general",
                "seq": 1,
                "author": {"kind": "human", "display_name": "Mallory\x1b[5m"},
                "created_at": "2026-11-18T10:00:00Z",
                "text": hostile,
            }
            return httpx.Response(200, json={"items": [item], "next_cursor": None})
        return previous(sent)

    hub.override = evil
    args = build_cli_parser().parse_args(["community", "read", "#general"])
    assert cli.main(args, client_factory=lambda _cfg: make_client(hub)) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x9b" not in out and "PWNED" not in out
    assert "Mallory" in out and "hi  there" in out
    code, payload = _run(["read", "#general"], hub, capsys)
    assert code == 0 and payload["messages"][0]["text"] == hostile, "--json stays exact"


def _text(argv: list[str], hub: FakeHub, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    args = build_cli_parser().parse_args(["community", *argv])
    code = cli.main(args, client_factory=lambda _cfg: make_client(hub))
    return code, capsys.readouterr().out


@pytest.mark.parametrize(
    ("status", "body", "expected", "absent"),
    [
        (200, outcome(), "Posted to #general\n", "message_id"),
        (
            422,
            {"status": "refused", "category": "spam_or_promotion", "reference": "MOD-AAAA-BBBB"},
            "Not posted: Spam or undisclosed promotion · reference MOD-AAAA-BBBB\n",
            "category",
        ),
        (
            429,
            {"retry_after": 42, "scope": "account_burst"},
            "Slow down: 10 a minute for you · send in 0:42\n",
            "scope",
        ),
        (
            409,
            {"code": "guidelines_changed", "guidelines_version": "2026-12"},
            "Guidelines updated to 2026-12 · accept them first: "
            "lop community join --accept-guidelines 2026-12\n",
            "version",
        ),
    ],
)
def test_text_mode_speaks_human_copy_and_json_keeps_the_codes(
    status: int, body: Any, expected: str, absent: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # Design review D4: category and scope labels, no message ids, the exact
    # acceptance command; the codes and ids stay in --json for scripts.
    hub = _hub(submit=(status, body))
    _code, out = _text(["post", "#general", "--text", "hello"], hub, capsys)
    assert out == expected
    for code_word in ("msg_2", "spam_or_promotion", "account_burst", "guidelines_changed"):
        assert code_word not in out
    _code, payload = _run(
        ["post", "#general", "--text", "hello"], _hub(submit=(status, body)), capsys
    )
    assert absent in payload, payload


def test_read_prints_local_times_not_utc_stamps(capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import datetime

    code, out = _text(["read", "#general"], _hub(), capsys)
    local = datetime.fromisoformat("2026-11-18T10:00:00+00:00").astimezone()
    assert code == 0 and "T10:00:00Z" not in out
    assert local.strftime("%H:%M") in out.splitlines()[0]


def test_join_re_accepts_newer_guidelines_for_a_member(capsys: pytest.CaptureFixture[str]) -> None:
    # The refusal above names `join --accept-guidelines`; for a member on an
    # older version that must re-accept, not answer "already joined".
    hub = _hub()
    previous = hub.override
    assert previous is not None
    newer = {**JOINED, "guidelines": {"version": "2026-12", "text": "Be kinder."}}

    def updated(sent: Any) -> httpx.Response | None:
        if sent.path.endswith("/bootstrap"):
            return httpx.Response(200, json=newer)
        if sent.path == JOIN:
            return httpx.Response(200, json=newer)
        return previous(sent)

    hub.override = updated
    code, payload = _run(["join", "--accept-guidelines", "2026-12"], hub, capsys)
    assert code == 0 and payload["state"] == "joined"
    (sent,) = [s for s in hub.sent if s.path == JOIN]
    assert json.loads(sent.body) == {"guidelines_version": "2026-12", "accept_guidelines": True}
    # Same version already accepted: still nothing sent.
    code, payload = _run(["join", "--accept-guidelines", "2026-11-01"], _hub(), capsys)
    assert code == 0 and payload["state"] == "already_joined"
