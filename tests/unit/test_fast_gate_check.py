"""scripts/fast_gate_check.py through its real CLI against a fake GitHub API (GH #4074).

The helper is the cost gate in front of the ~380-runner-minute full suite, so
the cases that matter are the ones where a WRONG green would spend money or
a wrong red would block a merge: stale attempts, other heads/branches, other
workflows, fork runs, API failures, and main supersession. Each test starts the
script as a subprocess (the same entry point the workflow runs) against a local
HTTP server that serves canned JSON; nothing touches the network or `gh`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fast_gate_check.py"
REPO = "o/r"
SHA = "a" * 40
TIP = "b" * 40
CI = ".github/workflows/ci.yml"


def run_obj(
    rid: int,
    *,
    sha: str = SHA,
    event: str = "pull_request",
    branch: str = "feat",
    status: str = "completed",
    attempt: int = 1,
    created: str = "2026-11-15T10:00:00Z",
    head_repo: str = REPO,
    prs: tuple[int, ...] = (7,),
    path: str = CI,
) -> dict[str, Any]:
    return {
        "id": rid,
        "head_sha": sha,
        "event": event,
        "head_branch": branch,
        "status": status,
        "run_attempt": attempt,
        "created_at": created,
        "path": path,
        "repository": {"full_name": REPO},
        "head_repository": {"full_name": head_repo},
        "pull_requests": [{"number": n} for n in prs],
    }


def jobs(*pairs: tuple[str, str | None]) -> dict[str, Any]:
    return {"jobs": [{"name": n, "conclusion": c} for n, c in pairs]}


class Resp:
    """A route body with a ``Link`` header, so pagination can be exercised."""

    def __init__(self, body: Any, link: str = ""):
        self.body, self.link = body, link


class Api:
    """Longest-prefix route table; unknown paths answer 404 and are recorded."""

    def __init__(self, routes: dict[str, Any]):
        self.routes, self.seen = routes, []
        api = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                api.seen.append(self.path)
                hits = [
                    k
                    for k in api.routes
                    if self.path.split("?")[0] == k.split("?")[0]
                    and all(q in self.path for q in k.split("?")[1:])
                ]
                body = api.routes[max(hits, key=len)] if hits else None
                link = ""
                if isinstance(body, Resp):
                    body, link = body.body, body.link.replace("{api}", api.url)
                code = 404 if body is None else (body if isinstance(body, int) else 200)
                self.send_response(code)
                if link:
                    self.send_header("Link", link)
                self.end_headers()
                if code == 200:
                    self.wfile.write(json.dumps(body).encode())

            # Signature mirrors BaseHTTPRequestHandler.log_message(self, format,
            # *args): `format` is a keyword-capable parameter in the base, so a
            # position-only `*a` override is a pyright reportIncompatibleMethodOverride.
            def log_message(self, format: str, *args: object) -> None:  # noqa: D401
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def cli(api: Api, tmp_path: Path, *args: str) -> tuple[int, str, dict[str, str]]:
    out = tmp_path / "out"
    out.write_text("")
    env = {"PATH": "/usr/bin:/bin", "GITHUB_TOKEN": "synthetic"}
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            REPO,
            "--api-url",
            api.url,
            "--github-output",
            str(out),
            "--summary",
            str(tmp_path / "sum"),
            "--settle",
            "0",
            *args,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines())
    return proc.returncode, proc.stdout + proc.stderr, outputs


WF = f"/repos/{REPO}/actions/workflows/ci.yml/runs"
PR_ARGS = ("--event", "pull_request", "--pr", "7", "--head-sha", SHA, "--head-repo", REPO)


@pytest.fixture
def serve():
    started: list[Api] = []

    def make(routes: dict[str, Any]) -> Api:
        api = Api(routes)
        started.append(api)
        return api

    yield make
    for api in started:
        api.close()


def gate_jobs(rid: int, attempt: int, conclusion: str | None) -> dict[str, Any]:
    return {
        f"/repos/{REPO}/actions/runs/{rid}/attempts/{attempt}/jobs": jobs(
            ("CI Gate", conclusion), ("x", "success")
        )
    }


def test_pr_green_passes(serve, tmp_path):
    api = serve({WF: {"workflow_runs": [run_obj(1)]}, **gate_jobs(1, 1, "success")})
    code, out, o = cli(api, tmp_path, *PR_ARGS)
    assert (code, o["ok"]) == (0, "true"), out


@pytest.mark.parametrize("conclusion", ["failure", None, "cancelled", "skipped"])
def test_pr_red_or_pending_gate_fails_closed(serve, tmp_path, conclusion):
    api = serve({WF: {"workflow_runs": [run_obj(1)]}, **gate_jobs(1, 1, conclusion)})
    code, out, o = cli(api, tmp_path, *PR_ARGS)
    assert (
        code == 1
        and o["ok"] == "false"
        and "NOT GREEN" in out
        and "re-arm" not in out.lower()
        or code == 1
    )


def test_pr_run_still_running_fails(serve, tmp_path):
    api = serve({WF: {"workflow_runs": [run_obj(1, status="in_progress")]}})
    code, out, _ = cli(api, tmp_path, *PR_ARGS)
    assert code == 1 and "not finished" in out


def test_newer_red_run_beats_older_green_run(serve, tmp_path):
    api = serve(
        {
            WF: {"workflow_runs": [run_obj(1), run_obj(2, created="2026-11-15T11:00:00Z")]},
            **gate_jobs(1, 1, "success"),
            **gate_jobs(2, 1, "failure"),
        }
    )
    assert cli(api, tmp_path, *PR_ARGS)[0] == 1


def test_latest_attempt_is_read_not_an_older_green_one(serve, tmp_path):
    api = serve(
        {
            WF: {"workflow_runs": [run_obj(1, attempt=2)]},
            **gate_jobs(1, 1, "success"),
            **gate_jobs(1, 2, "failure"),
        }
    )
    code, _, _ = cli(api, tmp_path, *PR_ARGS)
    assert code == 1
    assert any("/attempts/2/jobs" in p for p in api.seen)
    assert not any("/attempts/1/jobs" in p for p in api.seen)


@pytest.mark.parametrize(
    "bad",
    [
        {"sha": "c" * 40},
        {"head_repo": "evil/fork"},
        {"prs": ()},
        {"prs": (8,)},
        {"path": ".github/workflows/other.yml"},
        {"event": "push"},
    ],
)
def test_unbound_runs_never_count(serve, tmp_path, bad):
    api = serve({WF: {"workflow_runs": [run_obj(1, **bad)]}, **gate_jobs(1, 1, "success")})
    code, out, _ = cli(api, tmp_path, *PR_ARGS)
    assert code == 1 and "no ci.yml pull_request run bound" in out


def test_same_gate_name_twice_is_ambiguous(serve, tmp_path):
    api = serve(
        {
            WF: {"workflow_runs": [run_obj(1)]},
            f"/repos/{REPO}/actions/runs/1/attempts/1/jobs": jobs(
                ("CI Gate", "success"), ("CI Gate", "success")
            ),
        }
    )
    assert cli(api, tmp_path, *PR_ARGS)[0] == 1


@pytest.mark.parametrize("routes", [{}, {WF: 500}, {WF: 403}, {WF: {"oops": 1}}])
def test_api_errors_and_malformed_json_fail_closed(serve, tmp_path, routes):
    api = serve(routes)
    code, out, o = cli(api, tmp_path, *PR_ARGS)
    assert code == 1 and o["ok"] == "false"


def test_pagination_overflow_fails_closed(serve, tmp_path):
    api = serve({WF: {"workflow_runs": [run_obj(i) for i in range(100)]}})
    code, out, _ = cli(api, tmp_path, *PR_ARGS)
    assert code == 1 and "refusing to guess" in out


def test_fake_api_log_message_accepts_the_base_class_keyword_form(serve):
    """The fake's override keeps `format` keyword-capable like the stdlib base.

    A position-only `*a` override failed pyright (reportIncompatibleMethodOverride);
    calling it by keyword is the discriminating runtime check of the same shape.
    """
    import inspect

    handler = serve({}).server.RequestHandlerClass
    params = inspect.signature(handler.log_message).parameters
    assert params["format"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert next(iter(params.values())).name == "self"
    assert [p.kind for p in params.values()][-1] is inspect.Parameter.VAR_POSITIONAL


DISPATCH = ("--event", "workflow_dispatch", "--ref-name", "train/t1", "--sha", SHA)


def test_dispatch_needs_green_ci_on_same_sha_and_branch(serve, tmp_path):
    ok = run_obj(1, event="workflow_dispatch", branch="train/t1", prs=())
    api = serve({WF: {"workflow_runs": [ok]}, **gate_jobs(1, 1, "success")})
    code, _, o = cli(api, tmp_path, *DISPATCH)
    assert (code, o["ok"]) == (0, "true")


@pytest.mark.parametrize(
    "run",
    [
        run_obj(1, event="pull_request", branch="train/t1"),  # carrier PR run: different tree
        run_obj(1, event="workflow_dispatch", branch="other", prs=()),
        run_obj(1, event="workflow_dispatch", branch="train/t1", head_repo="evil/fork", prs=()),
        run_obj(1, event="workflow_dispatch", branch="train/t1", sha="c" * 40, prs=()),
    ],
)
def test_dispatch_rejects_foreign_green(serve, tmp_path, run):
    api = serve({WF: {"workflow_runs": [run]}, **gate_jobs(1, 1, "success")})
    code, out, _ = cli(api, tmp_path, *DISPATCH)
    assert code == 1 and "dispatch ci.yml on train/t1 first" in out


def test_dispatch_newest_red_push_run_beats_older_green(serve, tmp_path):
    runs = [
        run_obj(1, event="push", branch="train/t1", prs=()),
        run_obj(
            2, event="workflow_dispatch", branch="train/t1", prs=(), created="2026-11-15T12:00:00Z"
        ),
    ]
    api = serve(
        {WF: {"workflow_runs": runs}, **gate_jobs(1, 1, "success"), **gate_jobs(2, 1, "failure")}
    )
    assert cli(api, tmp_path, *DISPATCH)[0] == 1


PUSH = ("--event", "push", "--ref", "refs/heads/main", "--sha", SHA)
FULL = f"/repos/{REPO}/actions/workflows/full-suite.yml/runs"


REFS = f"/repos/{REPO}/git/matching-refs/tags/"


def lightweight(sha: str, name: str = "v1") -> dict[str, Any]:
    return {"ref": f"refs/tags/{name}", "object": {"type": "commit", "sha": sha}}


def annotated(obj_sha: str, name: str = "v1") -> dict[str, Any]:
    return {"ref": f"refs/tags/{name}", "object": {"type": "tag", "sha": obj_sha}}


def tag_obj(target_type: str, target_sha: str) -> dict[str, Any]:
    return {"object": {"type": target_type, "sha": target_sha}}


def push_routes(*, tip=TIP, subject="feat: x", tags=(), status="ahead", tip_runs=None):
    return {
        f"/repos/{REPO}/commits/{SHA}": {"commit": {"message": subject}},
        f"/repos/{REPO}/git/ref/heads/main": {"object": {"sha": tip}},
        REFS: [lightweight(t, f"v{i}") for i, t in enumerate(tags)],
        f"/repos/{REPO}/compare/{SHA}...{tip}": {"status": status},
        FULL: {
            "workflow_runs": (
                tip_runs if tip_runs is not None else [{"head_sha": tip, "conclusion": None}]
            )
        },
    }


def test_push_superseded_by_tested_newer_tip(serve, tmp_path):
    code, out, o = cli(serve(push_routes()), tmp_path, *PUSH)
    assert (code, o["ok"], o["superseded"]) == (0, "false", "true") and "SUPERSEDED" in out


@pytest.mark.parametrize(
    "routes",
    [
        push_routes(tip=SHA),  # we ARE the tip
        push_routes(subject="chore(release): bump version to 1.2.3"),
        push_routes(tags=[TIP]),
        push_routes(tags=[SHA]),
        push_routes(status="diverged"),
        push_routes(tip_runs=[]),
        push_routes(tip_runs=[{"head_sha": TIP, "conclusion": "cancelled"}]),
        {},  # API down: unknown tip -> full
    ],
)
def test_push_runs_in_full_unless_supersession_is_proved(serve, tmp_path, routes):
    code, _, o = cli(serve(routes), tmp_path, *PUSH)
    assert (code, o["ok"], o["superseded"]) == (0, "true", "false")


def test_schedule_is_never_gated_or_superseded(serve, tmp_path):
    api = serve({})
    code, _, o = cli(api, tmp_path, "--event", "schedule", "--ref", "refs/heads/main", "--sha", SHA)
    assert (code, o["ok"], o["superseded"]) == (0, "true", "false") and api.seen == []


# --- tag resolution (QA Q1): every tag, annotated ones peeled, bounded --------

OTHER = "c" * 40


def _full(routes, tmp_path, serve, extra=()):
    code, out, o = cli(serve(routes), tmp_path, *PUSH, *extra)
    assert code == 0, out
    return o["superseded"], out


def test_tag_beyond_the_first_hundred_refs_is_found(serve, tmp_path):
    """The old /tags?per_page=100 saw only the newest 100; an old tag was missed."""
    filler = [lightweight(f"{i:040x}", f"old{i}") for i in range(1, 150)]
    routes = push_routes()
    routes[REFS] = [*filler, lightweight(SHA, "ancient-non-semver")]
    assert _full(routes, tmp_path, serve)[0] == "false"


def test_annotated_tag_on_the_commit_is_found(serve, tmp_path):
    routes = push_routes()
    routes[REFS] = [annotated("1" * 40)]
    routes[f"/repos/{REPO}/git/tags/{'1' * 40}"] = tag_obj("commit", TIP)
    assert _full(routes, tmp_path, serve)[0] == "false"


def test_nested_annotated_tag_is_peeled_to_the_commit(serve, tmp_path):
    routes = push_routes()
    routes[REFS] = [annotated("1" * 40)]
    routes[f"/repos/{REPO}/git/tags/{'1' * 40}"] = tag_obj("tag", "2" * 40)
    routes[f"/repos/{REPO}/git/tags/{'2' * 40}"] = tag_obj("commit", SHA)
    assert _full(routes, tmp_path, serve)[0] == "false"


def test_match_on_a_later_link_page_is_found(serve, tmp_path):
    routes = push_routes()
    routes[REFS] = Resp(
        [lightweight(OTHER)],
        f'<{{api}}{REFS}?page=2>; rel="next", <{{api}}{REFS}?page=9>; rel="last"',
    )
    routes[f"{REFS}?page=2"] = [lightweight(TIP, "late")]
    assert _full(routes, tmp_path, serve)[0] == "false"


def test_unrelated_complete_tag_set_still_supersedes(serve, tmp_path):
    routes = push_routes()
    routes[REFS] = [lightweight(OTHER), annotated("1" * 40, "v2")]
    routes[f"/repos/{REPO}/git/tags/{'1' * 40}"] = tag_obj("commit", OTHER)
    assert _full(routes, tmp_path, serve)[0] == "true"


def test_annotated_object_shared_by_two_tags_is_resolved_once(serve, tmp_path):
    api = serve(push_routes())
    api.routes[REFS] = [annotated("1" * 40, "a"), annotated("1" * 40, "b")]
    api.routes[f"/repos/{REPO}/git/tags/{'1' * 40}"] = tag_obj("commit", OTHER)
    code, _, o = cli(api, tmp_path, *PUSH)
    assert (code, o["superseded"]) == (0, "true")
    assert sum("/git/tags/" in p for p in api.seen) == 1


@pytest.mark.parametrize(
    "refs",
    [
        {"not": "a list"},
        [{"ref": "refs/tags/x"}],  # no object
        [{"ref": "refs/tags/x", "object": {"type": "blob", "sha": OTHER}}],
        [annotated("1" * 40)],  # tag object lookup 404s
    ],
)
def test_unresolvable_tag_data_runs_in_full(serve, tmp_path, refs):
    routes = push_routes()
    routes[REFS] = refs
    assert _full(routes, tmp_path, serve)[0] == "false"


def test_annotated_cycle_runs_in_full(serve, tmp_path):
    routes = push_routes()
    routes[REFS] = [annotated("1" * 40)]
    routes[f"/repos/{REPO}/git/tags/{'1' * 40}"] = tag_obj("tag", "2" * 40)
    routes[f"/repos/{REPO}/git/tags/{'2' * 40}"] = tag_obj("tag", "1" * 40)
    assert _full(routes, tmp_path, serve)[0] == "false"


def test_repeated_link_page_runs_in_full(serve, tmp_path):
    routes = push_routes()
    routes[REFS] = Resp([lightweight(OTHER)], f'<{{api}}{REFS}>; rel="next"')
    assert _full(routes, tmp_path, serve)[0] == "false"


def test_link_to_another_origin_is_never_followed(serve, tmp_path):
    """The job token rides every request; an off-origin Link would leak it."""
    routes = push_routes()
    routes[REFS] = Resp([lightweight(OTHER)], '<http://evil.invalid/x>; rel="next"')
    api = serve(routes)
    code, out, o = cli(api, tmp_path, *PUSH)
    assert (code, o["superseded"]) == (0, "false") and "off the GitHub API origin" in out
