"""The scanner: what a session's rows are, derived from transcript rows alone.

The rules under test are the ones that decide whether a reader is told the truth:

* a ref mentioned only in TOOL OUTPUT is collapsed (session ``439818272d84`` holds 158
  such URLs, most of them from audit scripts);
* a script-created URL is ``unknown`` only if the session had not seen the ref before;
* an event wins over a mention of the same ref (a ``gh pr merge`` after a create keeps
  the row opened AND records the merge);
* a fork's inherited rows say so, and are never claimed as opens performed here.
"""

from __future__ import annotations

from typing import Any

from local_operator.code_requests.refs import HostContext, Remote
from local_operator.code_requests.scan import (
    EVENT_CUSTOM_TYPE,
    RELATION_ACTED,
    RELATION_INHERITED,
    RELATION_MENTIONED,
    RELATION_OPENED,
    RELATION_UNKNOWN,
    SOURCE_ASSISTANT,
    SOURCE_PEER,
    SOURCE_TOOL,
    SOURCE_USER,
    scan_rows,
)

CWD = HostContext(remotes=(Remote("origin", "github.com", "damianvtran/local-operator"),))
PR = "https://github.com/damianvtran/local-operator/pull/1904"


def _assistant(
    text: str, *, calls: list[dict[str, Any]] | None = None, ts: float = 1.0
) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": "message", "role": "assistant", "content": [{"text": text}]}
    if calls:
        payload["tool_calls"] = calls
    return {"id": f"a{ts}", "ts": ts, "type": "message", "payload": payload}


def _tool(text: str, call_id: str, *, ts: float = 2.0) -> dict[str, Any]:
    return {
        "id": f"t{ts}",
        "ts": ts,
        "type": "message",
        "payload": {
            "kind": "message",
            "role": "tool",
            "tool_name": "bash",
            "tool_call_id": call_id,
            "content": [{"text": text}],
        },
    }


def _user(text: str, ts: float = 0.5) -> dict[str, Any]:
    return {
        "id": f"u{ts}",
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "user", "content": [{"text": text}]},
    }


def _peer(text: str, ts: float = 3.0) -> dict[str, Any]:
    return {
        "id": f"p{ts}",
        "ts": ts,
        "type": "message",
        "payload": {"kind": "custom", "custom_type": "peer_message", "details": {"text": text}},
    }


def _event(kind: str, *, at: float = 10.0, number: int = 1904) -> dict[str, Any]:
    return {
        "id": f"e{at}",
        "ts": at,
        "type": "custom",
        "payload": {
            "custom_type": EVENT_CUSTOM_TYPE,
            "details": {
                "v": 1,
                "kind": kind,
                "ref": {
                    "key": f"github.com/damianvtran/local-operator#{number}",
                    "forge": "github",
                    "host": "github.com",
                    "project": "damianvtran/local-operator",
                    "number": number,
                    "url": f"https://github.com/damianvtran/local-operator/pull/{number}",
                    "full": True,
                },
                "evidence": {"tool": "bash", "rule": "gh-pr-create-stdout", "verb": "gh pr create"},
                "at": at,
            },
        },
    }


def _create_command() -> list[dict[str, Any]]:
    return [
        {
            "id": "call_1",
            "name": "bash",
            "arguments": {"command": "gh pr create --title t --body-file b.md"},
        }
    ]


def test_a_create_result_becomes_an_opened_row_with_evidence():
    rows = [
        _assistant("Opening the PR.", calls=_create_command()),
        _tool(f"exit code: 0\n--- stdout ---\n{PR}\n\n--- stderr ---\n(empty)", "call_1"),
    ]
    result = scan_rows(rows, CWD)
    assert [row.relation for row in result.rows] == [RELATION_OPENED]
    row = result.rows[0]
    assert row.ref.number == 1904
    assert row.evidence and row.evidence[-1]["rule"] == "gh-pr-create-stdout"
    assert row.evidence[-1]["kind"] == RELATION_OPENED


def test_tool_only_mentions_collapse_but_are_counted():
    rows = [
        _assistant("running an audit"),
        _tool(
            (
                "exit code: 0\n--- stdout ---\n"
                "https://github.com/o/r/pull/1\nhttps://github.com/o/r/pull/2"
            ),
            "call_9",
        ),
    ]
    result = scan_rows(rows, CWD)
    assert result.rows == []
    assert result.tool_output_only == 2
    assert {row.ref.number for row in result.tool_only_rows} == {1, 2}


def test_a_ref_the_user_named_is_visible_even_if_tool_output_later_mentions_it():
    rows = [
        _user(f"please merge {PR}"),
        _assistant("looking", calls=_create_command()),
        _tool(f"exit code: 0\n--- stdout ---\n{PR}\n", "call_1"),
    ]
    result = scan_rows(rows, CWD)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert set(row.mentions) == {SOURCE_USER, SOURCE_TOOL}
    assert row.mentions[SOURCE_USER].count == 1
    assert result.tool_output_only == 0


def test_a_script_url_the_session_never_saw_is_unknown_with_its_reason():
    rows = [
        _assistant("running the helper"),
        _tool("exit code: 0\n--- stdout ---\nhttps://github.com/o/r/pull/77\n", "call_9"),
    ]
    # The assistant row carries no tool call with a script command, so the classifier sees
    # the script through the tool row's own provider payload below.
    rows[0]["payload"]["tool_calls"] = [
        {"id": "call_9", "name": "bash", "arguments": {"command": "./scripts/open-pr.sh"}}
    ]
    result = scan_rows(rows, CWD)
    assert [row.relation for row in result.rows] == [RELATION_UNKNOWN]
    assert "possibly opened by this call" in (result.rows[0].unknown_reason or "")


def test_an_event_beats_a_mention_and_keeps_the_acts():
    rows = [
        _user(f"have a look at {PR}"),
        _event(RELATION_OPENED, at=10.0),
        _event(RELATION_ACTED, at=20.0),
        _assistant(f"merged {PR}"),
    ]
    result = scan_rows(rows, CWD)
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.relation == RELATION_OPENED
    assert set(row.relations) >= {RELATION_OPENED, RELATION_ACTED}
    assert SOURCE_USER in row.mentions and SOURCE_ASSISTANT in row.mentions


def test_an_event_older_than_the_fork_is_inherited():
    rows = [_event(RELATION_OPENED, at=5.0)]
    result = scan_rows(rows, CWD, forked_at=100.0, parent_id="abcdef123456")
    assert [row.relation for row in result.rows] == [RELATION_INHERITED]
    assert result.rows[0].inherited_from == "abcdef123456"
    # An event AFTER the fork is this session's own work.
    fresh = scan_rows([_event(RELATION_OPENED, at=200.0)], CWD, forked_at=100.0)
    assert [row.relation for row in fresh.rows] == [RELATION_OPENED]


def test_peer_and_wake_text_are_mention_sources():
    rows = [_peer(f"the other session opened {PR}"), _user(f"and I pasted {PR}")]
    result = scan_rows(rows, CWD)
    assert len(result.rows) == 1
    assert set(result.rows[0].mentions) == {SOURCE_PEER, SOURCE_USER}


def test_a_git_push_hint_is_recorded_and_never_a_row():
    rows = [
        _assistant(
            "pushing",
            calls=[
                {"id": "c", "name": "bash", "arguments": {"command": "git push -u origin feat/x"}}
            ],
        ),
        _tool(
            "exit code: 0\n--- stdout ---\nremote: Create a pull request for 'feat/x':\n"
            "remote:   https://github.com/o/r/pull/new/feat/x\n",
            "c",
        ),
    ]
    result = scan_rows(rows, CWD)
    assert result.rows == []
    assert result.hints and result.hints[0]["rule"] == "git-push-hint"


def test_rows_sort_opened_first_then_acted_then_mentions():
    rows = [
        _user("look at https://github.com/o/r/pull/9"),
        _event(RELATION_OPENED, at=10.0, number=1904),
    ]
    rows[0]["payload"]["content"] = [{"text": "look at https://github.com/o/r/pull/9 please"}]
    result = scan_rows(rows, CWD)
    assert [row.relation for row in result.rows] == [RELATION_OPENED, RELATION_MENTIONED]
