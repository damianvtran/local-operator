"""A fast benchmark is evidence only if its tree and treatment are verified."""

import subprocess

import httpx
import pytest

from scripts import bench_cold_send_http, bench_tree


def test_measured_tree_rejects_changed_and_residual_source(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.email", "benchmark@example.invalid")
    git("config", "user.name", "Benchmark fixture")
    source = tmp_path / "local_operator"
    source.mkdir()
    module = source / "module.py"
    module.write_text("before = True\n")
    git("add", ".")
    git("commit", "-qm", "test: seed before tree")
    before = git("rev-parse", "HEAD")
    monkeypatch.setattr(bench_tree, "REPO_ROOT", tmp_path)
    assert bench_tree.describe()["rev"] == before
    module.write_text("before = False\n")
    with pytest.raises(bench_tree.MeasuredTreeError, match="does not match"):
        bench_tree.describe(before)
    git("add", ".")
    git("commit", "-qm", "test: seed after tree")
    after = git("rev-parse", "HEAD")
    module.write_text("before = True\n")
    result = bench_tree.describe(before)
    assert result["rev"] == before
    assert result["worktree_head"] == after
    (source / "residual.py").write_text("unexpected = True\n")
    with pytest.raises(bench_tree.MeasuredTreeError, match="untracked"):
        bench_tree.describe(before)


@pytest.mark.parametrize(
    "status,data,expected",
    [
        (200, {"cold": True, "servers": [{"name": "bench-slow"}]}, ["bench-slow"]),
        (200, {"cold": True, "servers": []}, []),
        (409, {}, None),
        (200, {"servers": []}, None),
        (200, {"cold": True}, None),
    ],
)
def test_http_declaration_requires_valid_cold_response(status, data, expected):
    def response(request):
        assert request.method == "GET"
        assert request.url.path == "/v1/desktop/sessions/synthetic/mcp"
        return httpx.Response(status, json={"result": {"data": data}})

    with httpx.Client(
        base_url="http://benchmark.invalid", transport=httpx.MockTransport(response)
    ) as client:
        assert bench_cold_send_http._declared_servers(client, "synthetic") == expected


@pytest.mark.parametrize(
    "status,data,allow_live,expected",
    [
        # The control arm's rule, unchanged: the cold page only.
        (200, {"cold": True, "servers": [{"name": "bench-slow"}]}, False, ["bench-slow"]),
        (200, {"cold": True, "servers": []}, False, []),
        (409, {}, False, None),
        (200, {"servers": []}, False, None),
        (200, {"cold": True}, False, None),
        # QA round 1, Q-1: the draft arms ALSO accept the live page the route
        # serves once the warm has bound — refusing it graded the runs where the
        # warm worked best as unreadable and exited 3.
        (200, {"operations": [], "servers": []}, True, []),
        (200, {"operations": [], "servers": [{"name": "bench-slow"}]}, True, ["bench-slow"]),
        (200, {"cold": True, "servers": []}, True, []),
        (409, {}, True, None),
        (200, {"operations": []}, True, None),
    ],
)
def test_the_declaration_reader_takes_the_cold_and_draft_live_pages(
    status, data, allow_live, expected
):
    def response(request):
        assert request.method == "GET"
        assert request.url.path == "/v1/desktop/sessions/synthetic/mcp"
        return httpx.Response(status, json={"result": {"data": data}})

    report: dict[str, str | None] = {}
    with httpx.Client(
        base_url="http://benchmark.invalid", transport=httpx.MockTransport(response)
    ) as client:
        assert (
            bench_cold_send_http._declared_servers(
                client, "synthetic", allow_live=allow_live, report=report
            )
            == expected
        )
    if expected is None:
        assert "mcp_declaration_page" not in report
    else:
        assert report["mcp_declaration_page"] == (
            "live" if data.get("cold") is not True else "cold"
        )


def test_the_declaration_read_settles_past_a_mid_boot_refusal():
    """The draft arms' create lands while the warm may still be booting.

    Round-1 Q-1 observed the read answering a reconcile-refusal in that window;
    the settle loop retries an UNREADABLE answer (not only an exception) until
    the deadline, so the same config read a second later is the verdict.
    """
    answers = [
        httpx.Response(409, json={"detail": "attaching"}),
        httpx.Response(200, json={"result": {"data": {"operations": [], "servers": []}}}),
    ]

    def response(request):
        return answers.pop(0) if len(answers) > 1 else answers[0]

    with httpx.Client(
        base_url="http://benchmark.invalid", transport=httpx.MockTransport(response)
    ) as client:
        assert (
            bench_cold_send_http._declared_servers(
                client, "synthetic", settle_s=5.0, allow_live=True
            )
            == []
        )


def test_the_control_read_settles_past_a_refusal_too():
    """The control arm's settle keeps the COLD rule: a refusal then a cold page."""
    answers = [
        httpx.Response(409, json={"detail": "attaching"}),
        httpx.Response(
            200,
            json={"result": {"data": {"cold": True, "servers": [{"name": "bench-slow"}]}}},
        ),
    ]

    def response(request):
        return answers.pop(0) if len(answers) > 1 else answers[0]

    with httpx.Client(
        base_url="http://benchmark.invalid", transport=httpx.MockTransport(response)
    ) as client:
        assert bench_cold_send_http._declared_servers(client, "synthetic", settle_s=5.0) == [
            "bench-slow"
        ]


def test_declaration_grading_is_tri_state():
    """``declaration_correct`` False only on a MISMATCH; unreadable is its own list.

    The pre-Q-1 comparison graded ``None != []`` as a mismatch, so an unreadable
    draft row exited 3 with a sentence about a declaration that had not
    mismatched. The three verdicts now stay distinct in the summary.
    """
    unreadable = [
        {
            "run": 0,
            "first_send_ms": 1.0,
            "mcp_declaration_correct": None,
            "mcp_declared_servers": None,
        }
    ]
    summary = bench_cold_send_http._summarize(unreadable)
    assert summary["declaration_correct"] is None
    assert summary["declaration_unreadable"] == [0]

    verified = [{**unreadable[0], "mcp_declaration_correct": True, "mcp_declared_servers": []}]
    summary = bench_cold_send_http._summarize(verified)
    assert summary["declaration_correct"] is True
    assert summary["declaration_unreadable"] == []

    mismatched = [
        {**verified[0], "mcp_declaration_correct": False, "mcp_declared_servers": ["bench-slow"]}
    ]
    summary = bench_cold_send_http._summarize(mismatched)
    assert summary["declaration_correct"] is False
    assert summary["declaration_unreadable"] == []
