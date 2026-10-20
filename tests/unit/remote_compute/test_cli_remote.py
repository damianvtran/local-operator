"""``lop exec --remote`` driven through the real argparse entry point.

WHAT THESE TESTS ARE. Every case parses an actual command line through
``cli.main`` with an isolated config root, an isolated credential store and a
fake origin behind ``httpx.MockTransport``, so what is asserted is the surface
an operator uses: exit codes, the sentence on stderr, and the requests that did
or did not leave. Nothing here reaches Radient — the autouse fixture makes the
real transports raise if anything ever escapes the mock.

CTRL-C IS PART OF THE CONTRACT. The watch loop's sleep is the seam a Ctrl-C
lands in, so raising ``KeyboardInterrupt`` there proves the three things that
matter together: the process exits 130, the farewell names ``--remote-cancel``,
and NO cancel request was sent — an interrupted watch must not kill work.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator import cli as cli_module
from local_operator.network import store
from local_operator.network.types import MemberRecord, NetworkRecord
from local_operator.providers.auth_store import AuthStore
from local_operator.remote_compute import cli as remote_cli
from local_operator.remote_compute.arguments import LOCAL_ONLY_FLAGS

RUN_ID = "run_01HZX"
KEY = "key-fixed"


class _Origin:
    """A fake Radient run endpoint that records every request it receives."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        return self.responses.pop(0)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def paths(self) -> list[str]:
        return [request.url.raw_path.decode() for request in self.requests]


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Fail loudly if a request leaves the mock transport."""
    escaped: list[str] = []

    def refuse(self: Any, request: httpx.Request) -> httpx.Response:
        escaped.append(f"{request.method} {request.url}")
        raise AssertionError(f"a test attempted a real network call: {request.method}")

    async def arefuse(self: Any, request: httpx.Request) -> httpx.Response:
        return refuse(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse, raising=True)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", arefuse, raising=True)
    yield escaped
    assert escaped == [], f"tests made real network calls: {escaped}"


@pytest.fixture(autouse=True)
def isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


@pytest.fixture()
def origin() -> _Origin:
    return _Origin()


@pytest.fixture(autouse=True)
def fake_origin(origin: _Origin, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        remote_cli, "_http_client", lambda: httpx.AsyncClient(transport=origin.transport)
    )


@pytest.fixture()
def enabled(isolated_roots: Path) -> None:
    (isolated_roots / "config.yml").write_text(
        "values:\n  remote_compute:\n    enabled: true\n", encoding="utf-8"
    )


@pytest.fixture()
def network() -> NetworkRecord:
    record = NetworkRecord(
        network_id=store.new_network_id(),
        name="home",
        self_device_id="d_self",
        members=[MemberRecord(device_id="d_self", name="this device")],
    )
    store.save(record)
    return record


@pytest.fixture()
def signed_in() -> int:
    with closing(AuthStore()) as store_:
        row = store_.upsert_credential(
            "radient", {"type": "oauth", "account_id": "acct", "access": "token-abc"}
        )
        return row.id


def _run(command: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(sys, "argv", ["lop", *command])
    return cli_module.main()


BASE = ["exec", "--remote", "--repo", "https://example.invalid/repo.git", "--sha", "deadbeef"]
HOSTING = ["--hosting", "openai", "--model", "gpt-5"]


def _submitted(**extra: Any) -> dict[str, Any]:
    return {"result": {"id": RUN_ID, "status": "queued", **extra}}


# ---------------------------------------------------------------------------
# Refusals: one sentence, non-zero, nothing sent
# ---------------------------------------------------------------------------


def test_the_switch_off_refuses_and_names_the_setting(
    origin: _Origin, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(BASE + ["do the thing"], monkeypatch) == 1
    err = capsys.readouterr().err
    assert "remote_compute.enabled" in err
    assert "lop config edit remote_compute.enabled true" in err
    assert origin.requests == []


def test_no_network_surfaces_the_resolution_message(
    enabled: None,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(BASE + ["do the thing"], monkeypatch) == 1
    err = capsys.readouterr().err
    assert "lop network init" in err
    assert origin.requests == []


def test_an_ambiguous_network_refuses_rather_than_picking_a_tenant(
    enabled: None,
    signed_in: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for _ in range(2):
        store.save(
            NetworkRecord(
                network_id=store.new_network_id(),
                name="home",
                self_device_id="d_self",
                members=[MemberRecord(device_id="d_self", name="this device")],
            )
        )
    assert _run(BASE + ["do the thing"], monkeypatch) == 1
    # Both candidates are NAMED, so the reader can choose without `lop network ls`.
    err = capsys.readouterr().err
    assert "name a network for this remote run" in err
    assert err.count("(n_") == 2


def test_a_missing_repository_refuses_and_says_nothing_is_uploaded(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(["exec", "--remote", "do the thing"], monkeypatch) == 1
    err = capsys.readouterr().err
    assert "--repo" in err and "--sha" in err
    assert "never uploads local files" in err
    assert origin.requests == []


def test_signing_in_is_required_and_the_refusal_names_login(
    enabled: None,
    network: NetworkRecord,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(BASE + ["do the thing"], monkeypatch) == 1
    assert "lop login radient" in capsys.readouterr().err
    assert origin.requests == []


@pytest.mark.parametrize("flag", ["--size", "--detach", "--network", "--idempotency-key"])
def test_a_remote_flag_without_remote_is_refused(
    flag: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    value = [] if flag in ("--detach",) else ["s" if flag == "--size" else "x"]
    assert _run(["exec", flag, *value, "do the thing"], monkeypatch) == 1
    assert f"{flag} only applies with --remote." in capsys.readouterr().err


@pytest.mark.parametrize("name,flag", [pair for pair in LOCAL_ONLY_FLAGS if pair[0] != "resume"])
def test_every_locally_only_flag_is_refused_by_name(
    name: str,
    flag: str,
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The whole table, not a sample: a flag that is silently ignored is the bug."""
    value = {
        "--loop": ["2"],
        "--loop-goal": ["x"],
        "--run-in": ["."],
        "--supervisor-fd": ["3"],
        "--tools": ["bash"],
        "--team": ["release"],
        "--agent": ["helper"],
        "--agent-id": ["a_1"],
        "--profile": ["reviewer"],
        "--effort": ["high"],
        "--status": ["job-1"],
        "--output-format": ["json"],
        "--output-schema": ["schema.json"],
        "--output-retries": ["1"],
        "--goal": ["finish"],
    }.get(flag, [])
    assert _run(BASE + [flag, *value, "do the thing"], monkeypatch) == 1
    assert f"{flag} only makes sense for a local run" in capsys.readouterr().err
    assert origin.requests == []


def test_a_resume_never_reaches_the_remote_command(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--resume`` is resolved globally BEFORE the exec branch, so that refusal
    is the one an operator sees. What this pins is the part that matters here:
    no remote request is sent for a command the CLI already refused."""
    assert _run(BASE + ["--resume", "missing-session", "do the thing"], monkeypatch) == 1
    assert origin.requests == []


def test_a_remote_command_cannot_be_mixed_with_a_run(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(["exec", "--remote", "--remote-status", RUN_ID], monkeypatch) == 1
    assert "--remote-status and --remote-cancel are commands of their own" in (
        capsys.readouterr().err
    )
    assert origin.requests == []


# ---------------------------------------------------------------------------
# Submitting
# ---------------------------------------------------------------------------


def test_detach_prints_the_run_and_the_key_and_sends_the_contract(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    origin.responses.append(httpx.Response(201, json=_submitted()))
    assert (
        _run(
            BASE
            + HOSTING
            + ["--detach", "--size", "l", "--max-hours", "3", "--budget-usd", "4", "do it"],
            monkeypatch,
        )
        == 0
    )
    out = capsys.readouterr().out
    assert RUN_ID in out
    assert "Idempotency key: " in out
    assert f"lop exec --remote-cancel {RUN_ID}" in out
    (sent,) = origin.requests
    body = json.loads(sent.content)
    assert body["network_id"] == network.network_id
    assert (body["size"], body["max_hours"], body["budget_usd"]) == ("l", 3.0, 4.0)
    assert (body["hosting"], body["model"]) == ("openai", "gpt-5")
    assert body["prompt"] == "do it"
    assert sent.headers["idempotency-key"]


def test_an_oversize_idempotency_key_is_refused_before_the_request(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(BASE + HOSTING + ["--idempotency-key", "x" * 129, "do it"], monkeypatch) == 1
    assert "128 bytes" in capsys.readouterr().err
    assert origin.requests == []


def test_the_defaults_are_size_m_and_two_hours(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin.responses.append(httpx.Response(201, json=_submitted()))
    assert _run(BASE + HOSTING + ["--detach", "do it"], monkeypatch) == 0
    body = json.loads(origin.requests[0].content)
    assert (body["size"], body["max_hours"]) == ("m", 2.0)
    assert "budget_usd" not in body


def test_a_given_idempotency_key_is_reused(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin.responses.append(httpx.Response(201, json=_submitted()))
    _run(BASE + HOSTING + ["--detach", "--idempotency-key", KEY, "do it"], monkeypatch)
    assert origin.requests[0].headers["idempotency-key"] == KEY


def test_polling_prints_transitions_until_a_terminal_status(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(remote_cli, "_sleep", no_sleep)
    origin.responses.extend(
        [
            httpx.Response(201, json=_submitted()),
            httpx.Response(200, json={"result": {"id": RUN_ID, "status": "running"}}),
            httpx.Response(200, json={"result": {"id": RUN_ID, "status": "succeeded"}}),
        ]
    )
    assert _run(BASE + HOSTING + ["do it"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert out.split().count("status:") == 3
    assert "status: running" in out and "status: succeeded" in out
    assert f"Remote run {RUN_ID} finished: succeeded." in out


def test_a_failed_run_exits_non_zero(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(remote_cli, "_sleep", no_sleep)
    origin.responses.extend(
        [
            httpx.Response(201, json=_submitted()),
            httpx.Response(200, json={"result": {"id": RUN_ID, "status": "failed"}}),
        ]
    )
    assert _run(BASE + HOSTING + ["do it"], monkeypatch) == 1
    assert "finished: failed" in capsys.readouterr().out


def test_ctrl_c_stops_watching_without_cancelling(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(remote_cli, "_sleep", interrupt)
    origin.responses.extend(
        [
            httpx.Response(201, json=_submitted()),
            # A second read is armed to prove none happens after the interrupt.
            httpx.Response(200, json={"result": {"id": RUN_ID, "status": "running"}}),
        ]
    )
    assert _run(BASE + HOSTING + ["do it"], monkeypatch) == 130
    captured = capsys.readouterr()
    assert f"lop exec --remote-cancel {RUN_ID}" in captured.out
    assert "still running remotely" in captured.out
    # ONE request: the submission. No poll, and above all no cancel.
    assert len(origin.requests) == 1
    assert not any(path.endswith("/cancel") for path in origin.paths())


# ---------------------------------------------------------------------------
# The two remote commands
# ---------------------------------------------------------------------------


def test_remote_status_prints_the_run(
    enabled: None,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    origin.responses.append(
        httpx.Response(200, json={"result": {"id": RUN_ID, "status": "running"}})
    )
    assert _run(["exec", "--remote-status", RUN_ID], monkeypatch) == 0
    assert json.loads(capsys.readouterr().out) == {"id": RUN_ID, "status": "running"}
    assert origin.paths() == [f"/v1/remote-compute/runs/{RUN_ID}"]


def test_remote_cancel_posts_and_works_with_the_switch_off(
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cancelling must not be gated on the preview switch: work already running
    has to remain cancellable after the switch is turned off."""
    origin.responses.append(
        httpx.Response(200, json={"result": {"id": RUN_ID, "status": "cancelled"}})
    )
    assert _run(["exec", "--remote-cancel", RUN_ID], monkeypatch) == 0
    assert f"Remote run {RUN_ID} is cancelled." in capsys.readouterr().out
    assert origin.paths() == [f"/v1/remote-compute/runs/{RUN_ID}/cancel"]
    assert origin.requests[0].method == "POST"


def test_remote_status_reports_an_unknown_run(
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    origin.responses.append(httpx.Response(404, json={"error": "no such run"}))
    assert _run(["exec", "--remote-status", "run_missing"], monkeypatch) == 1
    assert "no such remote run" in capsys.readouterr().err


def test_a_conflict_names_the_key_flag(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    origin.responses.append(httpx.Response(409, json={"error": "conflict"}))
    assert _run(BASE + HOSTING + ["--detach", "do it"], monkeypatch) == 1
    assert "--idempotency-key" in capsys.readouterr().err


def test_capacity_and_credit_have_their_own_sentences(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for status, sentence in (
        (402, "no compute credit"),
        (503, "no remote-compute capacity"),
    ):
        origin_ = _Origin(httpx.Response(status, json={"error": "x"}))
        monkeypatch.setattr(
            remote_cli, "_http_client", lambda o=origin_: httpx.AsyncClient(transport=o.transport)
        )
        assert _run(BASE + HOSTING + ["--detach", "do it"], monkeypatch) == 1
        assert sentence in capsys.readouterr().err
