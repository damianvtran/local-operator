"""An owner outage with no saved GitHub access is said once, and lends nothing.

WHY THIS FILE EXISTS. On 2026-11-21 remote review/QA subagents ran ``gh`` with
no ``GH_TOKEN`` while the device lending GitHub was offline: the session held no
grant of its own, ``borrowed_git_env`` returned ``({}, "")`` in silence, and
``gh`` said only "set the GH_TOKEN environment variable". The bash tool result
now carries ONE stderr line naming the lender and the outage. It carries the
owner's display label and nothing else (never a token), is absent on success
and on a denial, and changes nothing about what enters the child.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials import github
from local_operator.network.credentials.types import BorrowTimeout, BrokerError
from local_operator.tools import builtin
from tests.unit.network.test_github_borrow_timeout_note import _run_bash

NOTE = (
    "GitHub login is lent by synthetic-owner, which is offline; "
    "this session has no saved access to use meanwhile"
)


def test_an_offline_owner_with_no_saved_access_adds_one_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text, outcome, _ = _run_bash(tmp_path, monkeypatch, "close")
    assert type(outcome) is BrokerError and outcome.code == "owner_offline"
    assert "TOKEN_PRESENT False" in text
    assert "ordinary stderr" in text
    assert text.count(NOTE) == 1, text


def test_a_denial_is_not_an_outage_and_shows_no_offline_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text, outcome, _ = _run_bash(tmp_path, monkeypatch, "deny")
    assert type(outcome) is BrokerError and outcome.code == "not_a_holder"
    assert "which is offline" not in text, text


class _Client:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def should_borrow(self, key: str) -> bool:
        return True

    def request_grant_sync(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.outcome


@pytest.mark.parametrize("code", sorted(github.OUTAGE_REFUSALS))
def test_the_door_reports_each_outage_code_only_with_a_sink(code: str) -> None:
    refusal = BrokerError(code=code, key="github", owner_device_name="synthetic-owner")
    seen: list[Any] = []
    client = _Client(refusal)
    assert github.lent_github_env(
        session_id="s", register=None, client=client, on_refusal=seen.append
    ) == ({}, "")
    assert seen == [] and client.calls == [], "no scrubber must not even trigger a borrow"
    assert github.lent_github_env(
        session_id="s", register=lambda t: None, client=client, on_refusal=seen.append
    ) == ({}, "")
    assert seen == [refusal]
    assert builtin._github_owner_offline_note(refusal) == NOTE  # noqa: SLF001


@pytest.mark.parametrize(
    "outcome",
    [
        BrokerError(code="not_a_holder", key="github"),
        BrokerError(code="consent_withdrawn", key="github"),
        BorrowTimeout(code="owner_offline", key="github", timeout_s=8.0),
    ],
)
def test_the_door_keeps_other_answers_off_the_outage_note(outcome: Any) -> None:
    seen: list[Any] = []
    assert github.lent_github_env(
        session_id="s", register=lambda t: None, client=_Client(outcome), on_refusal=seen.append
    ) == ({}, "")
    assert seen == [], "a deny or a timeout keeps its own copy"


def test_ancestors_reach_the_client_only_when_the_harness_set_them() -> None:
    """The door forwards ``hold_fallback_session_ids`` verbatim, and omits it when empty."""
    client = _Client(BrokerError(code="owner_offline", key="github"))
    github.lent_github_env(session_id="s", register=lambda t: None, client=client)
    github.lent_github_env(
        session_id="s",
        register=lambda t: None,
        client=client,
        hold_fallback_session_ids=("parent",),
    )
    assert "hold_fallback_session_ids" not in client.calls[0]
    assert client.calls[1]["hold_fallback_session_ids"] == ("parent",)


def test_only_a_tuple_of_harness_ancestors_is_read_from_the_context() -> None:
    """A context whose field was filled with anything but the harness's tuple bridges nothing."""
    from types import SimpleNamespace

    read = builtin._hold_fallback_session_ids  # noqa: SLF001
    assert read(SimpleNamespace(ancestor_session_ids=("p", "", "g"))) == ("p", "g")
    assert read(SimpleNamespace(ancestor_session_ids=["p"])) == ()
    assert read(SimpleNamespace()) == ()
