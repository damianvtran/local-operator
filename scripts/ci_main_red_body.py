#!/usr/bin/env python3
"""Render the `ci-main-red` issue body with the diagnosis a triager needs.

WHY THIS EXISTS
---------------
`full-suite.yml`'s `report` job used to file that issue with nothing but the
SHA, the *job keys* that failed and a run link:

    Full suite red at <sha>.

    Failing jobs:
    - `test`

    Run: https://github.com/.../actions/runs/<id>

Answering the first question a triager asks — infra, flake, or regression? —
then cost an artifact/log download and several tool calls, on every red, from
an agent with no browser. Measured on run 42463603282 (issue #3617): the body
named `test` and the run; the actual cause was `curl: (22) ... 504` in the
`Install ripgrep` step of shard `test (3.12, 1)`, which never reached pytest.

So this script reads the run's own job metadata and logs and adds, per failing
job: its name, the `##[error]` lines, the last output before the failure, and
— the highest-value field — the `FAILED tests/...` node ids from a failing
shard. It states explicitly when a job died in a step BEFORE the tests ran
(install, checkout, setup), because that is the infra-vs-test-red distinction.

DESIGN RULES, AND WHY EACH ONE IS LOAD-BEARING
----------------------------------------------
BEST-EFFORT, ALWAYS. The filing step is the only thing that tells anyone main
is red. A body builder that raises leaves main red and unnoticed, which is
strictly worse than the thin body it replaced. Every enrichment is therefore
guarded: no job metadata, no log, unparseable log, a hung `gh` — each degrades
to the field it could not fill, and the caller gets a body with the SHA, the
failing keys and the run link no matter what. `render_body` does not raise for
any input; `main` catches even its own bugs and emits the thin body.

BOUNDED. A job log here is 118 KB-220 KB (measured, run 42463603282); pasting
one would not fit an issue and would not be read. Caps: 20 node ids per shard,
5 error lines and 5 tail lines per job, 6 job logs fetched, ~300 chars per
line, 12000 chars per body. Every cap that fires says so in the body, so a
reader never mistakes a truncated list for the whole story.

OFFLINE RENDERING IS THE FIRST-CLASS PATH. `--evidence-dir` reads a captured
`jobs.json` + `<job_id>.log` and touches no network. That is how the PR that
added this was verified against the real #3617 logs, and how the unit tests
run; the live `gh` path is the same renderer with a fetching front end.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

#: Caps. Each is a rendering bound, not a measurement of what the logs contain
#: (the logs are unbounded); they keep the body readable and inside the issue
#: body limit with room to spare.
MAX_NODES_PER_JOB = 20
MAX_ERROR_LINES = 5
MAX_TAIL_LINES = 5
MAX_JOBS_WITH_LOGS = 6
MAX_ERROR_CHARS = 200
MAX_TAIL_CHARS = 240
MAX_BODY_CHARS = 12000

#: A `##[error]` line as GitHub Actions writes it. Both spellings occur in
#: practice (`##[error]` in the raw log, `## [error]` when a human quotes one),
#: and the log this reads is the raw one, so both are accepted.
_ERROR_RE = re.compile(r"^##\s?\[?error\]\s?(.*)$")
#: Any other workflow command (`##[group]`, `##[warning]`, `##[debug]`, ...),
#: which is runner bookkeeping rather than output from the failing command.
_COMMAND_RE = re.compile(r"^##\[")
#: pytest's short-test-summary lines: `FAILED <nodeid> - <message>`.
_FAILED_RE = re.compile(r"^FAILED\s+(tests/\S+)(?:\s+-\s+(.*))?$")
#: `gh run view --job <id> --log` prefixes every line with the job and step
#: names, tab-separated; the raw API log does not. Strip it so one parser reads
#: both, and so the step name is available even without job metadata.
_LOG_PREFIX_RE = re.compile(r"^[^\t\n]*\t[^\t\n]*\t")
#: The leading ISO-8601 timestamp GitHub puts on every log line.
_TIMESTAMP_RE = re.compile(r"^\S{20,}Z ")
#: The runner echoing the command it is about to run (`Run bash scripts/...`),
#: which is setup noise rather than the reason the step failed.
_ECHO_RE = re.compile(r"^Run \S")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

#: A step whose name matches this executes the test suite. Used only to decide
#: whether a job died BEFORE its tests: a job whose test steps all ended
#: `skipped`/`cancelled` never ran a test, whatever its other steps said.
_TEST_STEP_RE = re.compile(
    r"(?i)(pytest|run test|test shard|run the tests|e2e test|end-to-end|suite run)",
)


def _clip(text: str, limit: int) -> str:
    """Truncate to `limit` characters, marking the cut so it cannot be missed."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _clean_ansi(line: str) -> str:
    # A UTF-8 BOM prefixes the first line of some steps' output; it would defeat
    # the timestamp strip and with it every match below.
    return _ANSI_RE.sub("", line).lstrip("\ufeff")


@dataclass
class LogLine:
    """One line of a job log, split into the parts that matter."""

    step: str
    message: str


@dataclass
class JobDiagnosis:
    """What can be said about ONE failing job from its metadata and log."""

    name: str
    url: str = ""
    log_available: bool = False
    failed_step: str = ""
    #: Names of test steps that never executed, when the job died before them.
    unrun_test_steps: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    tail: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    #: Node ids found beyond MAX_NODES_PER_JOB, so the cut is visible.
    nodes_omitted: int = 0
    #: True when the log was deliberately not fetched because the body caps how
    #: many job logs it carries. Distinct from "the API refused it".
    log_capped: bool = False
    #: True when the log names the test step as failing but reported no node id
    #: (a collection error or a crash), which is different from "no tests ran".
    test_step_failed_without_nodes: bool = False


def parse_log(text: str) -> list[LogLine]:
    """Split a job log into step-tagged lines, tolerating both raw and run-view forms.

    ORDER MATTERS: `gh run view --job --log` puts `<job>\t<step>\t` in front of
    the timestamp, so the prefix is stripped first and the timestamp second.
    The other way round leaves the timestamp in the message, which silently
    defeats the `FAILED tests/...` and `##[error]` patterns (measured: the
    first version of this file rendered a shard as "no FAILED node id" while
    the log carried one).
    """
    lines: list[LogLine] = []
    for raw in text.splitlines():
        stripped = _clean_ansi(raw)
        step = ""
        match = _LOG_PREFIX_RE.match(stripped)
        if match:
            step = stripped.split("\t")[1]
            stripped = stripped[match.end() :]
        stripped = _TIMESTAMP_RE.sub("", stripped)
        lines.append(LogLine(step=step, message=stripped.rstrip()))
    return lines


def _message_body(message: str) -> str:
    """The message without a workflow-command prefix (`##[error]`, `##[group]`)."""
    error = _ERROR_RE.match(message)
    if error:
        return error.group(1)
    return message


def diagnose(name: str, url: str, steps: list[dict], log_text: str | None) -> JobDiagnosis:
    """Build the diagnosis for one failing job. Never raises for odd input."""
    diagnosis = JobDiagnosis(name=name, url=url)

    # Step metadata is what turns "it failed" into "it failed BEFORE the tests",
    # which is the infra-vs-test-red distinction this body exists to carry.
    # The steps arrive in execution order, so the FIRST failure is the step the
    # job died in; everything after it is skipped or cleanup.
    failing_step = ""
    for step in steps or []:
        if isinstance(step, dict) and step.get("conclusion") == "failure":
            failing_step = str(step.get("name") or "")
            break
    diagnosis.failed_step = failing_step
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        step_name = str(step.get("name") or "")
        if _TEST_STEP_RE.search(step_name) and step.get("conclusion") in {
            "skipped",
            "cancelled",
        }:
            diagnosis.unrun_test_steps.append(step_name)

    if log_text is None:
        return diagnosis
    diagnosis.log_available = True

    lines = parse_log(log_text)
    # The failing step's lines, when the log carries step names; otherwise the
    # whole log, which is still better than nothing.
    scoped = [ln for ln in lines if ln.step == failing_step] if failing_step else lines
    if not scoped:
        scoped = lines

    seen_errors: set[str] = set()
    seen_nodes: set[str] = set()
    seen_tail: set[str] = set()
    #: Raw `FAILED ...` lines already carried by the node list, so the tail does
    #: not spend a fifth of its five-line budget repeating one.
    node_messages: set[str] = set()
    for line in lines:
        node = _FAILED_RE.match(line.message.strip())
        if node:
            node_id = node.group(1)
            node_messages.add(line.message.strip())
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            if len(diagnosis.nodes) < MAX_NODES_PER_JOB:
                reason = _clip(node.group(2) or "", MAX_ERROR_CHARS)
                diagnosis.nodes.append(f"{node_id} - {reason}" if reason else node_id)
            else:
                diagnosis.nodes_omitted += 1

    for line in scoped:
        error = _ERROR_RE.match(line.message)
        if error:
            body = _clip(error.group(1), MAX_ERROR_CHARS)
            # The runner's own "Process completed with exit code N" line says
            # nothing the log does not, but it is the only `##[error]` many
            # failing steps produce; keep it, drop exact duplicates.
            if body and body not in seen_errors and len(diagnosis.errors) < MAX_ERROR_LINES:
                seen_errors.add(body)
                diagnosis.errors.append(body)

    # Tail: the last distinct output lines of the failing step. This is where
    # the actual cause lives when it is not a `##[error]` line at all — the
    # #3617 case is four identical `curl: (22) ... 504` lines and a runner
    # `exit code 22`, and the curl line is the diagnosis.
    #
    # WHERE THE TAIL STARTS: the runner wraps each step in `##[group]Run ...`
    # and dumps the step's whole environment as `KEY: value` lines before
    # `##[endgroup]`; after that marker it shows the command's real output. So
    # the window opens at the LAST `##[endgroup]`, which is what keeps four
    # `RG_SHA256:`/`pythonLocation:` echo lines from crowding out the curl error.
    tail_start = 0
    for index, line in enumerate(scoped):
        if line.message.strip() == "##[endgroup]":
            tail_start = index + 1
    for line in reversed(scoped[tail_start:]):
        if len(diagnosis.tail) >= MAX_TAIL_LINES:
            break
        body = _message_body(line.message).strip()
        if not body or _COMMAND_RE.match(line.message) or _ECHO_RE.match(body):
            continue
        if line.message.strip() in node_messages:
            continue
        clipped = _clip(body, MAX_TAIL_CHARS)
        if clipped in seen_tail:
            continue
        seen_tail.add(clipped)
        diagnosis.tail.append(clipped)
    diagnosis.tail.reverse()

    if not diagnosis.nodes and _TEST_STEP_RE.search(diagnosis.failed_step):
        diagnosis.test_step_failed_without_nodes = True
    return diagnosis


def _covers(job_name: str, needs_key: str) -> bool:
    """Does `job_name` belong to the failed `needs` key?

    A matrix job is named `<key> (<leg>, <shard>)` and a plain job is the key
    itself, so the prefix form is what maps one `needs` entry to its shards.
    """
    return job_name == needs_key or job_name.startswith(f"{needs_key} (")


def _render_job(diagnosis: JobDiagnosis) -> list[str]:
    """One failing job as markdown bullets, bounded by the module caps."""
    link = f" — [log]({diagnosis.url})" if diagnosis.url else ""
    lines = [f"- `{diagnosis.name}`{link}"]

    if not diagnosis.failed_step and not diagnosis.log_available:
        reason = (
            f"no log fetched: the body carries at most {MAX_JOBS_WITH_LOGS} job logs"
            if diagnosis.log_capped
            else "no log or step detail available for this job"
        )
        lines.append(f"  - {reason}; see the run page")
        return lines

    if diagnosis.failed_step and diagnosis.unrun_test_steps:
        never = ", ".join(f"`{name}`" for name in diagnosis.unrun_test_steps[:4])
        lines.append(
            f"  - **failed in step `{diagnosis.failed_step}`, before the tests ran** "
            f"({never} never executed): a setup/infrastructure failure, not a test red"
        )
    elif diagnosis.failed_step:
        lines.append(f"  - **failed in step `{diagnosis.failed_step}`**")

    if diagnosis.nodes:
        count = len(diagnosis.nodes) + diagnosis.nodes_omitted
        more = f", first {MAX_NODES_PER_JOB} shown" if diagnosis.nodes_omitted else ""
        lines.append(f"  - failing test node ids ({count}{more}):")
        lines.extend(f"    - `{node}`" for node in diagnosis.nodes)
        if diagnosis.nodes_omitted:
            lines.append(f"    - … {diagnosis.nodes_omitted} more, see the log")
    elif diagnosis.test_step_failed_without_nodes:
        lines.append(
            "  - the test step failed without a `FAILED tests/...` node id "
            "(collection error or a crash before the summary); see the log"
        )
    elif diagnosis.log_capped:
        lines.append(
            f"  - no log fetched: the body carries at most {MAX_JOBS_WITH_LOGS} "
            "job logs; see the run page"
        )
    elif not diagnosis.log_available:
        lines.append("  - log unavailable for this job; see the run page")

    if diagnosis.errors:
        lines.append("  - `##[error]`: " + " | ".join(f"`{e}`" for e in diagnosis.errors))
    if diagnosis.tail:
        lines.append("  - last output before the failure:")
        lines.extend(f"    - `{line}`" for line in diagnosis.tail)
    return lines


def render_body(
    *,
    sha: str,
    run_url: str,
    failing: list[str],
    cancelled: list[str],
    diagnoses: list[JobDiagnosis],
) -> str:
    """Compose the issue body. Total by construction: no input makes it raise."""
    parts = [f"Full suite red at {sha}.", "", "Failing jobs:"]
    used: set[str] = set()
    for key in failing:
        matched = [d for d in diagnoses if _covers(d.name, key)]
        if not matched:
            # A failing needs key with no job metadata behind it still has to
            # appear: the key alone is what the old body had.
            parts.append(f"- `{key}` — no per-job detail available for this job")
            continue
        for diagnosis in matched:
            parts.extend(_render_job(diagnosis))
            used.add(diagnosis.name)
    # A job that was diagnosed but that no needs key named still has to show.
    for diagnosis in diagnoses:
        if diagnosis.name not in used:
            parts.extend(_render_job(diagnosis))
            used.add(diagnosis.name)

    if cancelled:
        parts.extend(["", "Cancelled jobs:"])
        parts.extend(f"- `{name}`" for name in cancelled)

    parts.extend(["", f"Run: {run_url}", ""])
    body = "\n".join(parts)

    if len(body) > MAX_BODY_CHARS:
        trimmed = body[:MAX_BODY_CHARS].rsplit("\n", 1)[0]
        body = (
            trimmed
            + f"\n\n… body truncated at {MAX_BODY_CHARS} characters; the rest is "
            + f"on the run page: {run_url}\n"
        )
    return body


def _thin_body(sha: str, run_url: str, failing: list[str], cancelled: list[str]) -> str:
    """The pre-enrichment body. The floor this change must never fall below."""
    body = f"Full suite red at {sha}.\n\nFailing jobs:\n"
    body += "".join(f"- `{name}`\n" for name in failing) or "- (none reported)\n"
    if cancelled:
        body += "\nCancelled jobs:\n" + "".join(f"- `{name}`\n" for name in cancelled)
    return body + f"\nRun: {run_url}\n"


def _load_needs(raw: str) -> dict:
    """Parse `--needs-json`, accepting a literal, `@path`, or a path that exists.

    The workflow passes `toJSON(needs)` literally; `@path` and a bare path
    exist for offline rendering against a captured payload.
    """
    text = raw
    if raw.startswith("@"):
        text = Path(raw[1:]).read_text(encoding="utf-8")
    elif raw and not raw.lstrip().startswith("{") and Path(raw).is_file():
        text = Path(raw).read_text(encoding="utf-8")
    data = json.loads(text or "{}")
    return data if isinstance(data, dict) else {}


def _jobs_from_api(payload: str) -> list[dict]:
    """Decode `gh api .../jobs` output, tolerating several concatenated pages."""
    decoder = json.JSONDecoder()
    jobs: list[dict] = []
    index = 0
    while index < len(payload):
        while index < len(payload) and payload[index] in " \n\r\t":
            index += 1
        if index >= len(payload):
            break
        value, index = decoder.raw_decode(payload, index)
        if isinstance(value, dict):
            jobs.extend(value.get("jobs") or [])
        elif isinstance(value, list):
            jobs.extend(item for item in value if isinstance(item, dict))
    return jobs


def _run(cmd: list[str], timeout: float) -> str | None:
    """Run a command, returning its stdout or None. Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _select_failing_jobs(jobs: list[dict], failing: list[str]) -> list[dict]:
    """Map each failed `needs` key to the concrete jobs that failed.

    ONE mapping, used by both the live and the offline path: a second copy is
    how the two drift apart and one of them silently starts naming the wrong
    shard.
    """
    selected: list[dict] = []
    for key in failing:
        matched = [job for job in jobs if _covers(str(job.get("name") or ""), key)]
        if not matched:
            # No job metadata for this key (the API refused or was never asked):
            # the key alone still has to appear.
            selected.append({"name": key})
            continue
        selected.extend(job for job in matched if job.get("conclusion") == "failure")
    return selected


def collect(
    repo: str, run_id: str, failing: list[str], *, deadline: float = 100.0
) -> list[JobDiagnosis]:
    """Fetch job metadata + logs for the failing jobs. Best-effort throughout."""
    payload = _run(
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}/jobs", "--paginate"],
        timeout=min(30.0, deadline),
    )
    jobs = _jobs_from_api(payload) if payload else []
    if not jobs:
        return [JobDiagnosis(name=key) for key in failing]

    selected = _select_failing_jobs(jobs, failing)
    with_logs = selected[:MAX_JOBS_WITH_LOGS]

    def _one(job: dict) -> tuple[dict, str | None]:
        job_id = job.get("databaseId") or job.get("id")
        if not job_id:
            return job, None
        return job, _run(
            ["gh", "run", "view", "--job", str(job_id), "--log", "-R", repo],
            timeout=min(45.0, deadline),
        )

    logs: dict[int, str | None] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_one, job): job for job in with_logs}
        try:
            for future in as_completed(futures, timeout=deadline):
                job = futures[future]
                job_id = job.get("databaseId") or job.get("id") or id(job)
                try:
                    logs[job_id] = future.result()[1]
                except Exception:  # noqa: BLE001 - a log fetch must never fail filing
                    logs[job_id] = None
        except Exception:  # noqa: BLE001 - deadline or pool failure: use what we have
            pass

    diagnoses: list[JobDiagnosis] = []
    for job in selected:
        job_id = job.get("databaseId") or job.get("id")
        diagnosis = diagnose(
            name=str(job.get("name") or ""),
            url=str(job.get("html_url") or ""),
            steps=job.get("steps") or [],
            log_text=logs.get(job_id),
        )
        if job_id not in logs and len(diagnoses) >= MAX_JOBS_WITH_LOGS:
            diagnosis.log_capped = True
        diagnoses.append(diagnosis)
    return diagnoses


def render_from_evidence(
    *, sha: str, run_url: str, failing: list[str], cancelled: list[str], evidence: Path
) -> str:
    """Render offline from a captured `jobs.json` and per-job `<id>.log` files."""
    jobs_path = evidence / "jobs.json"
    jobs = _jobs_from_api(jobs_path.read_text(encoding="utf-8")) if jobs_path.exists() else []
    selected = _select_failing_jobs(jobs, failing) if jobs else [{"name": key} for key in failing]

    diagnoses = []
    for index, job in enumerate(selected):
        job_id = job.get("databaseId") or job.get("id")
        # The cap is the SAME one the live path fetches under: reading past it
        # offline would make the evidence render prove a bound the workflow does
        # not have.
        capped = index >= MAX_JOBS_WITH_LOGS
        log_path = evidence / f"{job_id}.log" if job_id and not capped else None
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path is not None and log_path.exists()
            else None
        )
        diagnosis = diagnose(
            name=str(job.get("name") or ""),
            url=str(job.get("html_url") or ""),
            steps=job.get("steps") or [],
            log_text=log_text,
        )
        # Beyond the cap the log was never looked for, which is not the same as
        # the log being unavailable; a reader should not go hunting for it.
        diagnosis.log_capped = capped
        diagnoses.append(diagnosis)
    return render_body(
        sha=sha,
        run_url=run_url,
        failing=failing,
        cancelled=cancelled,
        diagnoses=diagnoses,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the ci-main-red issue body with per-job diagnosis.",
    )
    parser.add_argument("--sha", required=True, help="the red commit SHA")
    parser.add_argument("--run-url", required=True, help="the run's HTML URL")
    parser.add_argument(
        "--needs-json",
        required=True,
        help="`toJSON(needs)` — a literal mapping or `@/path/to/file`",
    )
    parser.add_argument("--repo", default="", help="owner/name (default $GITHUB_REPOSITORY)")
    parser.add_argument("--run-id", default="", help="the run's database id")
    parser.add_argument(
        "--evidence-dir",
        default="",
        help="offline: directory holding jobs.json and <job_id>.log; no network",
    )
    parser.add_argument("--out", default="", help="write the body here instead of stdout")
    args = parser.parse_args(argv)

    repo = args.repo or _env("GITHUB_REPOSITORY")
    run_id = args.run_id or _env("GITHUB_RUN_ID")
    try:
        needs = _load_needs(args.needs_json)
    except (OSError, ValueError):
        needs = {}
    failing = [
        key
        for key, value in needs.items()
        if isinstance(value, dict) and value.get("result") == "failure"
    ]
    cancelled = [
        key
        for key, value in needs.items()
        if isinstance(value, dict) and value.get("result") == "cancelled"
    ]

    try:
        if not failing:
            # Without the failing keys there is nothing to key a diagnosis off,
            # and a body that says "Failing jobs:" and then names none is worse
            # than the thin one: it reads as "nothing failed".
            print(
                "::warning title=ci-main-red body enrichment skipped::no failing job "
                "in the needs payload; filing the thin body",
                file=sys.stderr,
            )
            body = _thin_body(args.sha, args.run_url, failing, cancelled)
        elif args.evidence_dir:
            body = render_from_evidence(
                sha=args.sha,
                run_url=args.run_url,
                failing=failing,
                cancelled=cancelled,
                evidence=Path(args.evidence_dir),
            )
        elif repo and run_id:
            body = render_body(
                sha=args.sha,
                run_url=args.run_url,
                failing=failing,
                cancelled=cancelled,
                diagnoses=collect(repo, run_id, failing),
            )
        else:
            body = _thin_body(args.sha, args.run_url, failing, cancelled)
    except Exception as exc:  # noqa: BLE001 - filing must survive anything here
        print(
            f"::warning title=ci-main-red body enrichment failed::{exc}",
            file=sys.stderr,
        )
        body = _thin_body(args.sha, args.run_url, failing, cancelled)

    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body)
    return 0


def _env(name: str) -> str:
    import os

    return os.environ.get(name, "")


if __name__ == "__main__":
    raise SystemExit(main())
