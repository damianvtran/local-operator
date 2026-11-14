"""The invoke client's SSE decoding, rendering and argument guards (offline)."""

from __future__ import annotations

import io

import invoke
import pytest


def test_parse_sse_decodes_json_frames_and_passes_text_through() -> None:
    lines = [
        b'data: {"type": "text", "text": "hi"}',
        b"",
        b": keep-alive comment",
        b'data: "a bare string"',
        b"data: not json",
    ]
    assert list(invoke.parse_sse(lines)) == [
        {"type": "text", "text": "hi"},
        {"type": "text", "text": "a bare string"},
        {"type": "text", "text": "not json"},
    ]


def test_render_shows_tools_and_turn_errors() -> None:
    out = io.StringIO()
    invoke.render({"type": "tool_start", "tool": "bash", "args": {"command": "ls"}}, out)
    invoke.render({"type": "tool_end", "tool": "bash", "is_error": False, "result": "a\n"}, out)
    invoke.render({"type": "end", "error": "403 AccessDenied"}, out)
    text = out.getvalue()
    assert '[tool] bash {"command": "ls"}' in text
    assert "[tool ok] a" in text
    assert "[turn error] 403 AccessDenied" in text


def test_render_shows_an_ignored_model_notice() -> None:
    out = io.StringIO()
    invoke.render(
        {
            "type": "session",
            "session_id": "s",
            "model": "m.a",
            "notice": "requested model 'm.b' ignored",
        },
        out,
    )
    assert out.getvalue() == "[session s model=m.a]\n[notice] requested model 'm.b' ignored\n"


def test_new_session_ids_meet_the_agentcore_minimum() -> None:
    session_id = invoke.new_session_id()
    assert len(session_id) >= invoke.MIN_SESSION_ID_LENGTH
    assert session_id != invoke.new_session_id()


def test_profile_is_required() -> None:
    with pytest.raises(SystemExit) as caught:
        invoke.main(["hello"])
    assert caught.value.code == 2  # argparse usage error


def test_short_session_id_is_refused_before_any_call() -> None:
    with pytest.raises(SystemExit, match="at least 33"):
        invoke.main(
            [
                "hi",
                "--profile",
                "p",
                "--region",
                "us-east-1",
                "--runtime-arn",
                "arn",
                "--session-id",
                "short",
            ]
        )
