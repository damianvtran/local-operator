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
