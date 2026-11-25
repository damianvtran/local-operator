#!/usr/bin/env python3
"""Exercise the v0.113.0 team-resume regression across an actual wheel upgrade.

Use the old product to write the conversations. No test creates or modifies a
session sidecar: doing that could make a passing run depend on today's idea of
what a pre-binding session looks like, rather than what users had on disk.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import secrets
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path
from uuid import uuid4

from smoke_wheel import (
    INDEX_URL,
    SmokeError,
    _clean_env,
    _free_loopback_port,
    _python_paths,
    _reap,
    _run,
    _tail,
    _without_inherited_python_env,
)

# v0.112.9 predates local-team-binding.json; use the same published wheel
# on both CI and developer hosts. v0.113.0 was yanked for this regression.
BASELINE = (
    "https://files.pythonhosted.org/packages/39/2b/"
    "f944e32bcb50b362257571907505d1d3e26bdf777560480c28b5d29c18b5/"
    "local_operator-0.112.9-py3-none-any.whl"
    "#sha256=de9d5ccbf1a4e1b9e2acd94e5dafd4c7e49ab8ec83935d772756180894bcc0db"
)
# Pin the broken wheel's published bytes so yanked-index selection cannot
# change which build supplies the negative control.
BROKEN = (
    "https://files.pythonhosted.org/packages/8e/8d/"
    "5cc99bd61f238dcbe3b3971a4bbef94867e27a6e6a3109fd29c62297e632/"
    "local_operator-0.113.0-py3-none-any.whl"
    "#sha256=ad65e16da2312f1b5158c726dc43c6cafe5ff9af6d7f096a6e186152c9b843dc"
)
TEAM = "upgrade-smoke"
MOCK_REPLY = "Hello from the mock provider"
ATTACHMENT_CODE = "unresolved_attachment"
ATTACHMENT_MESSAGE = "The attached profile or team could not be restored."
# Same cold runtime/server startup allowance as the wheel startability smoke.
RUNTIME_TIMEOUT = 90


def require(condition, detail):
    if not condition:
        raise SmokeError(detail)


def until(label, check):
    deadline = time.monotonic() + RUNTIME_TIMEOUT
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.1)
    raise SmokeError(f"{label} did not settle")


class Device:
    def __init__(self, root, lop, *, relay_port=None):
        root.mkdir()
        self.root = root
        self.lop = lop
        self.env = _clean_env(root)
        self.config = Path(self.env["LOCAL_OPERATOR_CONFIG_DIR"])
        settings = "values:\n  hosting: test\n  model_name: mock-model\n"
        if relay_port is not None:
            settings += f"  network:\n    listen_address: 127.0.0.1\n    port: {relay_port}\n"
        (self.config / "config.yml").write_text(settings, encoding="utf-8")
        self.relay_port = relay_port
        self.token = secrets.token_urlsafe(32)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def cli(self, *args, extra_env=None, stdin=""):
        result = subprocess.run(
            [str(self.lop), *args],
            env={**self.env, **(extra_env or {})},
            cwd=self.root,
            timeout=RUNTIME_TIMEOUT,
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )
        require(
            result.returncode == 0,
            f"lop {' '.join(args[:2])}: exit {result.returncode}\n"
            f"{_tail(result.stdout)}\n{_tail(result.stderr)}",
        )
        return result.stdout

    @contextlib.contextmanager
    def process(self, label, *args, extra_env=None):
        log_path = self.root / f"{label}.log"
        with log_path.open("wb") as log:
            proc = subprocess.Popen(
                [str(self.lop), *args],
                env={**self.env, **(extra_env or {})},
                cwd=self.root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            yield proc
        except Exception:
            print(f"{label} log:\n{_tail(log_path.read_text(errors='replace'))}", flush=True)
            raise
        finally:
            _reap(proc)

    @contextlib.contextmanager
    def server(self, label):
        port = _free_loopback_port()
        with self.process(
            label,
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            extra_env={"LOCAL_OPERATOR_DESKTOP_TOKEN": self.token},
        ) as proc:

            def healthy():
                require(proc.poll() is None, f"lop serve exited: {proc.returncode}")
                try:
                    status, body = self.http(port, "/health")
                    return body if status == 200 else None
                except urllib.error.URLError:
                    return None

            until(f"{label} health", healthy)
            yield port

    @contextlib.contextmanager
    def relay(self):
        require(self.relay_port is not None, "relay port was not provisioned")
        with self.process("relay", "network", "serve", "--no-launchd") as proc:

            def answering():
                require(proc.poll() is None, f"relay exited: {proc.returncode}")
                status = json.loads(self.cli("network", "status", "--json"))
                return status.get("relay_answering")

            until("relay control socket", answering)
            yield

    def http(self, port, path, body=None):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
        )
        try:
            response = self.opener.open(request, timeout=RUNTIME_TIMEOUT)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.code, json.load(response)

    def session_dir(self, sid):
        return self.config / "sessions" / sid

    def stop_sessions(self):
        # The CLI scopes discovery to this synthetic config; owned runtimes
        # detach, so reaping the HTTP/relay process group alone is not enough.
        self.cli("stop", "--all", "--yes")


def send(device, port, sid, text):
    return device.http(
        port,
        f"/v1/desktop/sessions/{sid}/messages",
        {"request_id": str(uuid4()), "text": text},
    )


def transcript(device, sid):
    path = device.session_dir(sid) / "transcript.jsonl"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def wait_for_reply(device, sid, text, old_reply_count):
    def replied():
        body = transcript(device, sid)
        return text in body and body.count(MOCK_REPLY) > old_reply_count

    until(f"{sid} deterministic reply to {text!r}", replied)


def seed(device):
    device.cli("teams", "create", TEAM, "--manager", "manager")
    first = "First operator turn before upgrading"
    with device.server("legacy") as port:
        status, result = device.http(
            port,
            "/v1/desktop/sessions",
            {
                "request_id": str(uuid4()),
                "cwd": str(device.root),
                "target": {"kind": "team", "name": TEAM},
            },
        )
        require(status == 200, f"create operator chat: {status} {result}")
        operator_sid = result["result"]["session_id"]
        try:
            status, result = send(device, port, operator_sid, first)
            require(status == 200, f"legacy operator admission: {status} {result}")
            wait_for_reply(device, operator_sid, first, 0)
        finally:
            device.stop_sessions()

    work = "First workstream turn before upgrading"
    # A terminal owner's --workstream is not marked as delegated. Use the
    # documented real-CLI QA environment (docs/EXEC.md) for this birth only.
    device.cli(
        "exec",
        "--workstream",
        "--team",
        TEAM,
        "--hosting",
        "test",
        "--model",
        "mock-model",
        work,
        extra_env={"LOCAL_OPERATOR_AGENT_SHELL": "1", "LOCAL_OPERATOR_ALLOW_NESTED_SESSION": "1"},
    )
    dirs = [p for p in (device.config / "sessions").iterdir() if work in transcript(device, p.name)]
    require(len(dirs) == 1, f"expected one persisted workstream, got {dirs}")
    work_sid = dirs[0].name
    for kind, sid in (("operator", operator_sid), ("workstream", work_sid)):
        directory = device.session_dir(sid)
        attachment = json.loads((directory / "attachment.json").read_text())
        require(attachment["team"] == TEAM, f"{kind} team attachment missing: {attachment}")
        require(not (directory / "local-team-binding.json").exists(), f"{kind} already bound")
        if kind == "workstream":
            origin = json.loads((directory / "origin.json").read_text())
            require(origin["origin"] == "agent-workstream", f"not a workstream: {origin}")
        require(MOCK_REPLY in transcript(device, sid), f"{kind} did not complete on the baseline")
        print(f"BASELINE {kind}: {sid} attached to {TEAM}; reply persisted; no binding", flush=True)
    return {"operator": operator_sid, "workstream": work_sid}


def resume(device, port, cases, *, expect_broken):
    for kind, sid in cases.items():
        before = transcript(device, sid)
        follow_up = f"First {kind} follow-up after upgrading"
        status, result = send(device, port, sid, follow_up)
        if expect_broken:
            detail = result.get("detail", {})
            require(
                status == 409
                and isinstance(detail, dict)
                and detail.get("code") == ATTACHMENT_CODE
                and ATTACHMENT_MESSAGE in detail.get("text", ""),
                f"{kind}: expected AttachmentUnavailable, got HTTP {status} {result}",
            )
            require(follow_up not in transcript(device, sid), f"{kind}: refused turn persisted")
            print(f"NEGATIVE {kind}: HTTP {status} {ATTACHMENT_CODE}: {ATTACHMENT_MESSAGE}")
        else:
            require(
                status == 200 and result["result"]["status"] == "admitted",
                f"{kind}: follow-up failed: HTTP {status} {result}",
            )
            wait_for_reply(device, sid, follow_up, before.count(MOCK_REPLY))
            require(
                (device.session_dir(sid) / "local-team-binding.json").is_file(),
                f"{kind}: continued without restoring the team binding",
            )
            print(f"CANDIDATE {kind}: HTTP {status}; same session replied and restored its binding")
        device.cli("stop", sid, "--json")


def enroll(owner, viewer):
    network = json.loads(owner.cli("network", "init", "upgrade-smoke", "--no-start", "--json"))
    invite = json.loads(
        owner.cli("network", "invite", "--role", "drive", "--no-credential-offers", "--json")
    )
    # The shipped automated-join flow still waits for the inviter's comparison.
    # Supply only that synthetic human answer via the product's pairing-test seam.
    endpoint = f"127.0.0.1:{owner.relay_port}"
    with viewer.process(
        "join",
        "network",
        "join",
        f"@{invite['path']}",
        "--host",
        endpoint,
        "--automated",
        "--name",
        "upgrade-viewer",
        "--json",
    ) as join:

        def pending():
            require(join.poll() is None, f"join exited before confirmation: {join.returncode}")
            receipt = json.loads(owner.cli("network", "confirm", "--list", "--json"))
            return next(
                (r for r in receipt["pending"] if r["invite_id"] == invite["invite_id"]), None
            )

        row = until("paired device's comparison", pending)
        require(row["transcribed"] == row["sas"], "synthetic devices derived different codes")
        owner.cli(
            "network",
            "confirm",
            invite["invite_id"],
            "--sas-stdin",
            "--json",
            extra_env={"LOP_NETWORK_TEST_MODE": "1"},
            stdin="yes\n",
        )
        require(join.wait(timeout=RUNTIME_TIMEOUT) == 0, "join did not complete")
    print("MESH baseline: two isolated members paired on loopback; no credential offers")
    return network["device_id"]


def resume_over_mesh(owner, viewer, owner_id, cases):
    for kind, sid in cases.items():
        before = transcript(owner, sid)
        text = f"First authorized mesh follow-up for {kind}"
        receipt = json.loads(
            viewer.cli(
                "network", "sessions", "--json", "--peer", owner_id, "--send", sid, "--", text
            )
        )
        require(
            receipt["ok"] is True
            and receipt["outcome"] == "finished"
            and receipt["session_id"] == sid,
            f"mesh {kind}: unexpected send receipt {receipt}",
        )
        wait_for_reply(owner, sid, text, before.count(MOCK_REPLY))
        require(
            (owner.session_dir(sid) / "local-team-binding.json").is_file(),
            f"mesh {kind}: team was not restored",
        )
        print(f"MESH candidate {kind}: authorized send replied on owner; receipt={receipt}")
        owner.cli("stop", sid, "--json")


def install(python, device, spec, expected):
    result = _run(
        [
            str(python),
            "-m",
            "pip",
            "--isolated",
            "install",
            "--disable-pip-version-check",
            "--index-url",
            INDEX_URL,
            spec,
        ],
        env=device.env,
        cwd=device.root,
        timeout=RUNTIME_TIMEOUT,
        label=f"install {spec}",
    )
    require(
        result.returncode == 0,
        f"wheel install failed: {_tail(result.stdout)}\n{_tail(result.stderr)}",
    )
    version = device.cli("--version").strip()
    require(version == f"v{expected}", f"expected v{expected}, got {version!r}")
    imported = _run(
        [str(python), "-I", "-c", "import local_operator; print(local_operator.__file__)"],
        env=device.env,
        cwd=device.root,
        timeout=RUNTIME_TIMEOUT,
        label="wheel provenance",
    )
    purelib = _python_paths(python, device.env, device.root)
    require(
        imported.returncode == 0
        and purelib.resolve() in Path(imported.stdout.strip()).resolve().parents,
        f"runtime did not import from its installed wheel: {imported.stdout}\n{imported.stderr}",
    )
    print(version, flush=True)


def run(python, lop, root, wheel):
    negative = Device(root / "negative", lop)
    install(python, negative, BASELINE, "0.112.9")
    negative_cases = seed(negative)

    # Seed independently under the one baseline install. A failed restore must
    # never change the candidate's input. All baseline processes are stopped
    # before we swap the installed package, leaving each device's config intact.
    candidate = Device(root / "candidate", lop)
    candidate_cases = seed(candidate)
    owner = Device(root / "owner", lop, relay_port=_free_loopback_port())
    with owner.relay():
        viewer = Device(root / "viewer", lop, relay_port=_free_loopback_port())
        owner_id = enroll(owner, viewer)
        mesh_cases = seed(owner)

    install(python, negative, BROKEN, "0.113.0")
    with negative.server("negative") as port:
        try:
            resume(negative, port, negative_cases, expect_broken=True)
        finally:
            negative.stop_sessions()

    install(python, candidate, str(wheel), wheel.name.split("-")[1])
    with candidate.server("candidate") as port:
        try:
            resume(candidate, port, candidate_cases, expect_broken=False)
        finally:
            candidate.stop_sessions()
    with owner.relay(), viewer.relay():
        try:
            resume_over_mesh(owner, viewer, owner_id, mesh_cases)
        finally:
            owner.stop_sessions()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=lambda name: Path(name).resolve(), required=True)
    parser.add_argument(
        "--venv",
        type=lambda name: Path(name).resolve(),
        help="Reuse a disposable venv; its local-operator install will be replaced",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lu-") as directory:
        root = Path(directory)
        runtime = args.venv or root / "venv"
        if args.venv is None:
            with _without_inherited_python_env():
                venv.EnvBuilder(with_pip=True, symlinks=True).create(runtime)
        run(runtime / "bin/python", runtime / "bin/lop", root, args.wheel)


if __name__ == "__main__":
    main()
