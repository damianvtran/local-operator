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


SHA = "0123456789abcdef0123456789abcdef01234567"
BASE = ["exec", "--remote", "--repo", "https://example.invalid/repo.git", "--sha", SHA]
HOSTING = ["--hosting", "openai", "--model", "gpt-5"]

#: Sentinels for the pod's secrets. A create returns them once, for the pod, and
#: nothing on this side may print, log or store them.
RUN_TOKEN = "zq-run-1"
CLOUDFLARED_TOKEN = "zq-tunnel-2"
ORIGIN_AUTH = "zq-origin-3"


def _with_secrets(wrapper: dict[str, Any]) -> dict[str, Any]:
    """The create WRAPPER (``result``) as a 201 really sends it."""
    wrapper["run_token"] = RUN_TOKEN
    wrapper["tunnel"] = {
        "tunnel_id": "tun_1",
        "hostname": "run-1.example.invalid",
        "gateway_port": 8443,
        "cloudflared_token": CLOUDFLARED_TOKEN,
        "origin_auth": ORIGIN_AUTH,
    }
    return wrapper


def _submitted_with_secrets(network_id: str) -> dict[str, Any]:
    """A whole 201: the envelope around a wrapper that carries the pod's secrets."""
    return {
        "msg": "Run created",
        "result": _with_secrets({"run": _wire_run("pending", network_id)}),
    }


def _wire_run(status: str, network_id: str, **extra: Any) -> dict[str, Any]:
    """A run as the server returns it."""
    return {"id": RUN_ID, "status": status, "network_id": network_id, **extra}


def _submitted(network_id: str, **extra: Any) -> dict[str, Any]:
    """The create body: the envelope around the run's own wrapper.

    ``network_id`` is REQUIRED because the client checks the nested run's tenant
    and refuses a receipt it cannot attribute.
    """
    return {
        "msg": "Run created",
        "result": {"run": _wire_run("pending", network_id, **extra)},
    }


def _poll(run: dict[str, Any]) -> httpx.Response:
    """One ``GET /runs/:id`` answer: the envelope around the run itself."""
    return httpx.Response(200, json={"msg": "Run", "result": run})


def _stopped(network_id: str, outcome: str | None = "succeeded", **extra: Any) -> dict[str, Any]:
    """A finished run. ``outcome`` is separate from the status, by contract."""
    run = _wire_run("stopped", network_id, **extra)
    if outcome is not None:
        run["outcome"] = outcome
    return run


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
# Local validation: the argument, never the network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sha",
    ["deadbeef", SHA[:39], SHA + "a", "HEAD", "z" * 40],
)
def test_a_sha_that_is_not_a_full_commit_is_refused_locally(
    sha: str,
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SE4 — a ref or a short hash is resolved later, by the far end, out of sight."""
    command = ["exec", "--remote", "--repo", "https://example.invalid/repo.git", "--sha", sha]
    assert _run(command + HOSTING + ["do it"], monkeypatch) == 1
    assert "full 40-character commit SHA" in capsys.readouterr().err
    assert origin.requests == []


@pytest.mark.parametrize(
    "repo",
    ["./repo", "../repo", "git://example.invalid/repo.git", "file:///srv/repo"],
)
def test_a_repo_the_far_end_cannot_clone_is_refused_locally(
    repo: str,
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = ["exec", "--remote", "--repo", repo, "--sha", SHA, *HOSTING, "do it"]
    assert _run(command, monkeypatch) == 1
    assert "clones the repository from the far end" in capsys.readouterr().err
    assert origin.requests == []


def test_an_empty_idempotency_key_is_refused_not_replaced(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E2 — the operator asked for a collision with an earlier submission."""
    assert _run(BASE + HOSTING + ["--detach", "--idempotency-key", "", "do it"], monkeypatch) == 1
    assert "refused rather than replaced" in capsys.readouterr().err
    assert origin.requests == []


@pytest.mark.parametrize("key", ["with space\tand tab", "ké", "line\nbreak"])
def test_a_key_outside_printable_ascii_is_refused_locally(
    key: str,
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E3 — a control byte is not a header value, and no request is composed."""
    assert _run(BASE + HOSTING + ["--detach", "--idempotency-key", key, "do it"], monkeypatch) == 1
    assert "printable ASCII" in capsys.readouterr().err
    assert origin.requests == []


def test_the_key_is_validated_before_the_credential_is_read(
    enabled: None,
    network: NetworkRecord,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E8 — no ``signed_in`` fixture: the key refusal must come FIRST."""
    assert _run(BASE + HOSTING + ["--detach", "--idempotency-key", "", "do it"], monkeypatch) == 1
    err = capsys.readouterr().err
    assert "refused rather than replaced" in err
    assert "lop login radient" not in err


def test_a_resume_is_refused_before_the_session_store_is_touched(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E6 — ``--resume`` is resolved globally, but a remote run is not a session.

    The patch fails the test if the resume branch reaches the store at all, which
    is the part a refusal sentence alone would not prove.
    """
    from local_operator import resume as resume_module

    def explode(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a remote invocation touched the session store")

    monkeypatch.setattr(resume_module, "backfill_session_origins", explode)
    assert _run(BASE + ["--resume", "missing-session", "do it"], monkeypatch) == 1
    assert "--resume only makes sense for a local run" in capsys.readouterr().err
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
    origin.responses.append(httpx.Response(201, json=_submitted(network.network_id)))
    assert (
        _run(
            BASE
            + HOSTING
            + ["--detach", "--size", "l", "--max-hours", "3", "--budget-usd", "4", "do it"],
            monkeypatch,
        )
        == 0
    )
    captured = capsys.readouterr()
    # SE5 — the run id ALONE on stdout, so `id=$(lop exec --remote --detach ...)`
    # captures it; the receipt, key included, is on stderr.
    assert captured.out.strip() == RUN_ID
    assert "Idempotency key: " in captured.err
    # SE1 — the receipt states the network the SERVER returned, labelled as
    # confirmed, never the client's own expectation.
    assert f"network (confirmed by Radient): {network.network_id}" in captured.err
    assert f"lop exec --remote-cancel {RUN_ID}" in captured.err
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
    origin.responses.append(httpx.Response(201, json=_submitted(network.network_id)))
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
    origin.responses.append(httpx.Response(201, json=_submitted(network.network_id)))
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
            httpx.Response(201, json=_submitted(network.network_id)),
            _poll(_wire_run("running", network.network_id)),
            _poll(_stopped(network.network_id)),
        ]
    )
    assert _run(BASE + HOSTING + ["do it"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert out.split().count("status:") == 3
    assert "status: pending" in out and "status: running" in out and "status: stopped" in out
    assert f"network (confirmed by Radient): {network.network_id}" in out
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
            httpx.Response(201, json=_submitted(network.network_id)),
            _poll(_stopped(network.network_id, "failed", exit_code=127)),
        ]
    )
    assert _run(BASE + HOSTING + ["do it"], monkeypatch) == 1
    out = capsys.readouterr().out
    assert f"Remote run {RUN_ID} finished: failed." in out
    assert "exit code: 127" in out


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
            httpx.Response(201, json=_submitted(network.network_id)),
            # A second read is armed to prove none happens after the interrupt.
            _poll(_wire_run("running", network.network_id)),
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
# The tenant is confirmed, and the watch is bounded
# ---------------------------------------------------------------------------


def test_a_run_from_another_network_is_never_reported_as_this_one(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SE1 — a receipt may only claim what the server confirmed."""
    origin.responses.append(httpx.Response(201, json=_submitted("n_" + "b" * 24)))
    assert _run(BASE + HOSTING + ["--detach", "do it"], monkeypatch) == 1
    captured = capsys.readouterr()
    assert "another network" in captured.err
    assert captured.out.strip() == ""


def test_a_poll_from_another_network_stops_the_watch(
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
            httpx.Response(201, json=_submitted(network.network_id)),
            _poll(_wire_run("running", "n_" + "c" * 24)),
        ]
    )
    assert _run(BASE + HOSTING + ["do it"], monkeypatch) == 1
    assert "another network" in capsys.readouterr().err


def test_a_bodiless_cancel_is_success(
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E4 — 204 is the documented success shape of a cancel."""
    origin.responses.append(httpx.Response(204))
    assert _run(["exec", "--remote-cancel", RUN_ID], monkeypatch) == 0
    assert f"Remote run {RUN_ID} is cancelled." in capsys.readouterr().out


def test_the_watch_wait_backs_off_to_its_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """E5 — growth is monotone, capped, and jittered inside the cap."""
    monkeypatch.setattr(remote_cli.random, "uniform", lambda low, high: high)
    interval = remote_cli.POLL_INTERVAL_SECONDS
    waits = []
    for _ in range(8):
        interval = remote_cli._poll_delay(interval)
        waits.append(interval)
    assert waits[0] == pytest.approx(remote_cli.POLL_INTERVAL_SECONDS * 1.5 * 1.1)
    assert waits == sorted(waits)
    assert waits[-1] == remote_cli.POLL_MAX_INTERVAL_SECONDS


def test_the_watch_stops_at_the_bound_and_says_how_to_check_the_run(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E5 — the bound is REPORTED and exits 0; it is never a silent stop.

    The bound is the SERVER's ``max_hours``, not ``--max-hours``: this run is
    submitted with ``--max-hours 3`` and comes back admitting 1, so a watch that
    trusted the flag would wait two hours past a run that can no longer change.
    """
    clock = {"now": 0.0}

    async def advance(seconds: float) -> None:
        # Past the run's own deadline plus the margin: nothing can change now.
        clock["now"] = 10_000_000.0

    monkeypatch.setattr(remote_cli, "_sleep", advance)
    monkeypatch.setattr(remote_cli, "_monotonic", lambda: clock["now"])
    origin.responses.extend(
        [
            httpx.Response(201, json=_submitted(network.network_id, max_hours=1.0)),
            _poll(_wire_run("running", network.network_id)),
        ]
    )
    assert _run(BASE + HOSTING + ["--max-hours", "3", "do it"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert "still running past the 1-hour limit Radient reported" in out
    assert f"lop exec --remote-status {RUN_ID}" in out
    assert "finished" not in out
    # The submission and ONE poll: stopping is the point of the bound.
    assert len(origin.requests) == 2


def test_an_unknown_status_is_printed_once_and_does_not_end_the_watch(
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
            httpx.Response(201, json=_submitted(network.network_id)),
            _poll(_wire_run("warming_up", network.network_id)),
            _poll(_wire_run("warming_up", network.network_id)),
            _poll(_stopped(network.network_id)),
        ]
    )
    assert _run(BASE + HOSTING + ["do it"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert out.count("not a status this build knows") == 1
    assert f"Remote run {RUN_ID} finished: succeeded." in out


# ---------------------------------------------------------------------------
# The two remote commands
# ---------------------------------------------------------------------------


def test_remote_status_prints_the_run(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The JSON is the run alone; the network verdict is the line after it."""
    origin.responses.append(
        httpx.Response(200, json={"msg": "Run", "result": _wire_run("running", network.network_id)})
    )
    assert _run(["exec", "--remote-status", RUN_ID], monkeypatch) == 0
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0]) == _wire_run("running", network.network_id)
    assert "this device's network" in lines[1]
    assert origin.paths() == [f"/v1/remote-compute/runs/{RUN_ID}"]


def test_remote_status_refuses_a_run_it_did_not_ask_for(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A body about another run is not this run's state, and is not printed."""
    other = {"id": "run_OTHER", "status": "running", "network_id": network.network_id}
    origin.responses.append(httpx.Response(200, json={"msg": "Run", "result": other}))
    assert _run(["exec", "--remote-status", RUN_ID], monkeypatch) == 1
    captured = capsys.readouterr()
    assert "run_OTHER" not in captured.out
    assert "not the run" in captured.err


def test_remote_status_says_so_when_the_network_does_not_resolve(
    enabled: None,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No network on this device is not a failure to read the run."""
    origin.responses.append(
        httpx.Response(200, json={"msg": "Run", "result": {"id": RUN_ID, "status": "running"}})
    )
    assert _run(["exec", "--remote-status", RUN_ID], monkeypatch) == 0
    out = capsys.readouterr().out
    assert "network unconfirmed" in out
    assert json.loads(out.splitlines()[0]) == {"id": RUN_ID, "status": "running"}


def test_remote_status_names_a_mismatch_it_can_see(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--network`` is allowed beside ``--remote-status``: it is a question."""
    other = "n_" + "d" * 24
    origin.responses.append(
        httpx.Response(200, json={"msg": "Run", "result": _wire_run("running", other)})
    )
    assert _run(["exec", "--remote-status", RUN_ID, "--network", "home"], monkeypatch) == 0
    out = capsys.readouterr().out
    assert other in out and "not this device's network" in out


def test_remote_status_drops_the_secret_fields_a_get_should_never_carry(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Defensive: the print is a whole run, so the two secret fields are filtered."""
    run = _wire_run("running", network.network_id)
    run.update({"run_token": RUN_TOKEN, "tunnel": {"cloudflared_token": CLOUDFLARED_TOKEN}})
    origin.responses.append(httpx.Response(200, json={"msg": "Run", "result": run}))
    assert _run(["exec", "--remote-status", RUN_ID], monkeypatch) == 0
    out = capsys.readouterr().out
    assert RUN_TOKEN not in out and CLOUDFLARED_TOKEN not in out


def test_remote_cancel_posts_and_works_with_the_switch_off(
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Cancelling must not be gated on the preview switch: work already running
    has to remain cancellable after the switch is turned off."""
    origin.responses.append(
        httpx.Response(200, json={"msg": "Run", "result": {"id": RUN_ID, "status": "stopping"}})
    )
    assert _run(["exec", "--remote-cancel", RUN_ID], monkeypatch) == 0
    assert f"Remote run {RUN_ID} is cancelled." in capsys.readouterr().out
    assert origin.paths() == [f"/v1/remote-compute/runs/{RUN_ID}/cancel"]
    assert origin.requests[0].method == "POST"


def test_a_cancel_answering_for_another_run_is_refused(
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reporting "cancelled" from another run's body would be a lie about THIS run."""
    origin.responses.append(
        httpx.Response(
            200, json={"msg": "Run", "result": {"id": "run_OTHER", "status": "stopping"}}
        )
    )
    assert _run(["exec", "--remote-cancel", RUN_ID], monkeypatch) == 1
    assert "is cancelled" not in capsys.readouterr().out


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


# ---------------------------------------------------------------------------
# The pod's secrets, the outcome vocabulary and the exit codes
# ---------------------------------------------------------------------------


def _detach(
    network: NetworkRecord,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, str]:
    origin.responses.append(httpx.Response(201, json=_submitted_with_secrets(network.network_id)))
    code = _run(BASE + HOSTING + ["--detach", "do it"], monkeypatch)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_create_secrets_never_reach_stdout_stderr_or_the_config_root(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    isolated_roots: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--detach: the pod's token and tunnel are dropped the moment the run is read."""
    code, out, err = _detach(network, origin, monkeypatch, capsys)
    assert code == 0
    for sentinel in (RUN_TOKEN, CLOUDFLARED_TOKEN, ORIGIN_AUTH):
        assert sentinel not in out and sentinel not in err
        assert sentinel not in repr(capsys.readouterr())
    stored = [
        path.read_text(errors="replace") for path in isolated_roots.rglob("*") if path.is_file()
    ]
    for sentinel in (RUN_TOKEN, CLOUDFLARED_TOKEN, ORIGIN_AUTH):
        assert all(sentinel not in body for body in stored)


def test_the_watch_path_never_prints_the_create_secrets(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same 201, watched to a stop: nothing secret is printed either way."""

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(remote_cli, "_sleep", no_sleep)
    origin.responses.extend(
        [
            httpx.Response(201, json=_submitted_with_secrets(network.network_id)),
            _poll(_stopped(network.network_id)),
        ]
    )
    assert _run(BASE + HOSTING + ["do it"], monkeypatch) == 0
    captured = capsys.readouterr()
    assert f"Remote run {RUN_ID} finished: succeeded." in captured.out
    for sentinel in (RUN_TOKEN, CLOUDFLARED_TOKEN, ORIGIN_AUTH):
        assert sentinel not in captured.out and sentinel not in captured.err


def _watched(
    network: NetworkRecord,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *polls: dict[str, Any],
) -> tuple[int, str]:
    """Submit, then answer each poll; return the exit code and stdout."""

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(remote_cli, "_sleep", no_sleep)
    origin.responses.append(httpx.Response(201, json=_submitted(network.network_id)))
    origin.responses.extend(_poll(poll) for poll in polls)
    code = _run(BASE + HOSTING + ["do it"], monkeypatch)
    return code, capsys.readouterr().out


@pytest.mark.parametrize(
    ("outcome", "code", "sentence"),
    [
        ("succeeded", 0, "finished: succeeded."),
        ("failed", 1, "finished: failed."),
        ("expired", 3, "finished: expired"),
        ("cancelled", 4, "finished: cancelled."),
    ],
)
def test_each_outcome_has_its_own_sentence_and_exit_code(
    outcome: str,
    code: int,
    sentence: str,
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The OUTCOME decides the exit code, not the status: stopped is one word."""
    exit_code, out = _watched(
        network, origin, monkeypatch, capsys, _stopped(network.network_id, outcome)
    )
    assert exit_code == code
    assert f"Remote run {RUN_ID} {sentence}" in out


def test_a_deleted_run_with_no_outcome_is_a_failure(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``deleted`` is terminal; with no outcome there is nothing to claim success for."""
    exit_code, out = _watched(
        network, origin, monkeypatch, capsys, _wire_run("deleted", network.network_id)
    )
    assert exit_code == 1
    assert "no outcome this build can read" in out


def test_the_exit_code_and_the_charge_are_reported_when_the_server_sends_them(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code, out = _watched(
        network,
        origin,
        monkeypatch,
        capsys,
        _stopped(network.network_id, "failed", exit_code=2, charged_usd=0.42),
    )
    assert exit_code == 1
    assert "exit code: 2" in out
    # Quoted as the server sent it: this client computes no prices.
    assert "charged: 0.42" in out


def test_an_errored_run_is_announced_as_retrying_once_and_the_watch_goes_on(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``error`` is NOT terminal: the server retries it, and the watch waits."""
    exit_code, out = _watched(
        network,
        origin,
        monkeypatch,
        capsys,
        _wire_run("error", network.network_id),
        _wire_run("error", network.network_id),
        _stopped(network.network_id),
    )
    assert exit_code == 0
    assert out.count("retries it, so this watch continues") == 1
    assert out.count("status: error") == 1
    assert f"Remote run {RUN_ID} finished: succeeded." in out


def test_a_poll_for_another_run_stops_the_watch_without_reporting_it(
    enabled: None,
    network: NetworkRecord,
    signed_in: int,
    origin: _Origin,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A watch that started answering about another run must not report that run."""
    other = {"id": "run_OTHER", "status": "stopped", "outcome": "succeeded"}
    exit_code, out = _watched(network, origin, monkeypatch, capsys, other)
    assert exit_code == 1
    assert "finished" not in out
    assert "run_OTHER" not in out
