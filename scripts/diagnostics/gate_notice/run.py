"""Manual-only Linux diagnostic orchestration; NEVER a product/release gate.

One dependency environment, two explicitly pinned source winners, one bounded
module execution each. Setup/observer invalidity stops the experiment; a valid
product test failure is retained and compared, never retried or reclassified.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import inspect
import json
import os
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from unittest.mock import patch

import observer

HERE = Path(__file__).resolve().parent
CARRIER = HERE.parents[2]
REFS = {
    "current": "318f23af220592c16fcdb66d8faa096b04c7c521",
    "base": "b42471d3935cb7e95d64428af27c32626665f4c0",
}
MODULE = "tests/unit/tui/test_gate_resurface_real_entry.py"
CELL = "test_an_answer_that_never_reached_the_owner_writes_no_receipt"


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parameters(function):
    args = function.args
    out = [(a.arg, "POSITIONAL_ONLY") for a in args.posonlyargs]
    out += [(a.arg, "POSITIONAL_OR_KEYWORD") for a in args.args]
    if args.vararg:
        out.append((args.vararg.arg, "VAR_POSITIONAL"))
    out += [(a.arg, "KEYWORD_ONLY") for a in args.kwonlyargs]
    if args.kwarg:
        out.append((args.kwarg.arg, "VAR_KEYWORD"))
    return out


def contract(root: Path):
    result = {}
    for filename, owner, names in (
        ("local_operator/tui/app.py", "OperatorApp", observer.APP),
        (
            "local_operator/session/attached.py",
            "AttachedSession",
            observer.ATTACHED + observer.PROPERTIES,
        ),
    ):
        tree = ast.parse((root / filename).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
        for name in names:
            function = next(
                (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name), None
            )
            if function is None and owner == "OperatorApp" and name == "call_later":
                # This one instrument is inherited from the single installed
                # Textual dependency, not declared in either product checkout.
                from textual.message_pump import MessagePump

                result[f"{owner}.{name}"] = [
                    (p.name, p.kind.name)
                    for p in inspect.signature(MessagePump.call_later).parameters.values()
                ]
                continue
            if function is None:
                raise observer.InstrumentationError(
                    f"missing synchronous source contract: {owner}.{name}"
                )
            result[f"{owner}.{name}"] = parameters(function)
    return result


def require_events(records, required):
    found = {record.get("event") for record in records}
    if not required <= found:
        raise observer.InstrumentationError("required observer event missing")


def calibration(work: Path):
    output = work / "artifacts" / "calibration"
    recorder = observer.Observer(output, CARRIER)
    recorder.emit("identity", calibration=True)
    marker = object()
    failure = ValueError("calibration sentinel")

    class Subject:
        def __init__(self):
            self.reads = 0

        @property
        def cold(self):
            self.reads += 1
            return False

        def bridge(self, pending=None):
            self._gate_task = asyncio.create_task(noop())
            if pending is failure:
                raise failure
            return marker

        async def asynchronous(self):
            return marker

    async def noop():
        return None

    signature = inspect.signature(Subject.bridge)
    recorder.wrap(Subject, "bridge", [(p.name, p.kind.name) for p in signature.parameters.values()])
    assert inspect.signature(Subject.bridge) == signature
    assert not inspect.iscoroutinefunction(Subject.bridge)

    recorder.wrap_property(Subject, "cold", [("self", "POSITIONAL_OR_KEYWORD")])

    async def exercise():
        subject = Subject()
        assert subject.cold is False and subject.reads == 1
        assert subject.bridge() is marker
        await subject._gate_task
        try:
            subject.bridge(failure)
        except ValueError as error:
            assert error is failure
        else:
            raise observer.InstrumentationError("observer swallowed original exception")
        await subject._gate_task

    asyncio.run(exercise())
    negatives = []
    for name, expected in (("missing", []), ("asynchronous", []), ("bridge", [])):
        try:
            recorder.wrap(Subject, name, expected)
        except observer.InstrumentationError:
            negatives.append(name)
        else:
            raise observer.InstrumentationError(
                "negative observer calibration accepted invalid hook"
            )
    recorder.finish(0)
    records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    require_events(records, {"identity", "enter", "leave", "raise"})
    try:
        require_events([r for r in records if r["event"] != "enter"], {"enter"})
    except observer.InstrumentationError:
        negatives.append("missing_trace_event")
    else:
        raise observer.InstrumentationError("negative event validation was not rejected")
    leave = next(r for r in records if r.get("event") == "leave")
    raised = next(r for r in records if r.get("event") == "raise")
    assert leave["state"]["_gate_task"]["task"] != raised["state"]["_gate_task"]["task"]
    # Underlying operations are stubs in BOTH positive/negative policy controls:
    # a broken guard can never accidentally signal a live process or use DNS.
    touched = []

    def stub(*args, **kwargs):
        touched.append(True)
        return marker

    fake: Any = type("Socket", (), {"family": socket.AF_INET})()
    with (
        patch.object(socket, "getaddrinfo", stub),
        patch.object(socket.socket, "connect", stub),
        patch.object(socket.socket, "connect_ex", stub),
        patch.multiple(socket, gethostbyname=stub, gethostbyname_ex=stub, gethostbyaddr=stub),
        patch.object(os, "kill", stub),
        patch.object(os, "killpg", stub),
    ):
        observer.install_policy(work)
        assert socket.getaddrinfo("127.0.0.1", 1) is marker
        assert socket.socket.connect(fake, ("127.0.0.1", 1)) is marker
        assert os.kill(42424242, 0) is marker
        assert os.killpg(42424242, 0) is marker
        for operation in (
            lambda: socket.getaddrinfo("fixture.invalid", 1),
            lambda: socket.socket.connect(fake, ("192.0.2.1", 1)),
            lambda: socket.socket.connect_ex(fake, ("192.0.2.1", 1)),
            lambda: socket.gethostbyname("fixture.invalid"),
            lambda: socket.gethostbyname_ex("fixture.invalid"),
            lambda: socket.gethostbyaddr("192.0.2.1"),
            lambda: os.kill(42424242, 15),
            lambda: os.killpg(42424242, 15),
        ):
            before = len(touched)
            try:
                operation()
            except observer.InstrumentationError:
                assert len(touched) == before
            else:
                raise observer.InstrumentationError("negative policy calibration failed")
    save(
        output / "result.json",
        {
            "positive": True,
            "negative_controls": negatives,
            "blocked_stub_operations": dict(observer.POLICY),
            "observer_sha256": observer.digest(HERE / "observer.py"),
            "runner_sha256": observer.digest(Path(__file__)),
        },
    )
    observer.POLICY.update({key: 0 for key in observer.POLICY})
    observer.POLICY_ROOT = None


def environment(iso: Path, root: Path, trace: Path, contract_path: Path):
    for name in ("home", "tmp", "runtime"):
        (iso / name).mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(iso / "home"),
        "TMPDIR": str(iso / "tmp"),
        "LOCAL_OPERATOR_CONFIG_DIR": str(iso / "home" / ".local-operator"),
        "LOCAL_OPERATOR_NO_NOTIFICATIONS": "1",
        "XDG_RUNTIME_DIR": str(iso / "runtime"),
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(HERE) + os.pathsep + str(root),
        "GATE_DIAG_SOURCE_ROOT": str(root),
        "GATE_DIAG_TRACE_DIR": str(trace),
        "GATE_DIAG_CONTRACT": str(contract_path),
        "GATE_DIAG_ISO": str(iso),
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        **{key: os.environ[key] for key in ("RUNNER_TRACKING_ID",) if key in os.environ},
    }


def probe(root: Path, output: Path, contract_path: Path):
    observer.install_policy(Path(os.environ["GATE_DIAG_ISO"]))
    winners = observer.source_probe(root, json.loads(contract_path.read_text()))
    if any(observer.POLICY.values()):
        raise observer.InstrumentationError("source setup attempted prohibited I/O")
    save(
        output,
        {
            "source": str(root),
            "modules": winners,
            "python": sys.version,
            "observer_sha256": observer.digest(HERE / "observer.py"),
            "policy": observer.POLICY,
        },
    )


def roots(args):
    return {
        "current": (args.current_root or args.work / "current").resolve(),
        "base": (args.base_root or args.work / "base").resolve(),
    }


def preflight(args):
    calibration(args.work)
    for name, root in roots(args).items():
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != REFS[name]:
            raise observer.InstrumentationError("source checkout is not pinned")
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--exit-code",
                "HEAD",
                "--",
                "local_operator",
                "tests",
            ],
            check=True,
        )
        path = args.work / "artifacts" / name / "contract.json"
        save(path, contract(root))
        save(
            path.parent / "source-manifest.json",
            {
                "git_sha": head,
                "test_module_sha256": observer.digest(root / MODULE),
                "contract_sha256": observer.digest(path),
                "bounded_runner_sha256": observer.digest(CARRIER / "scripts/run_bounded.py"),
            },
        )
        if not args.light_only:
            iso = Path(tempfile.mkdtemp(prefix="gate-probe-", dir=args.work))
            cmd = [
                sys.executable,
                str(Path(__file__)),
                "probe",
                "--source",
                str(root),
                "--output",
                str(path.parent / "source-probe.json"),
                "--contract",
                str(path),
            ]
            with (path.parent / "source-probe.log").open("w") as stream:
                subprocess.run(
                    cmd,
                    cwd=iso,
                    env=environment(iso, root, path.parent, path),
                    check=True,
                    timeout=90,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
    save(
        args.work / "artifacts" / "preflight.json",
        {
            "light_only": args.light_only,
            "refs": REFS,
            "python": sys.version,
            "controller_sha256": observer.digest(Path(__file__)),
            "observer_sha256": observer.digest(HERE / "observer.py"),
        },
    )


def validate_lane(directory: Path):
    summaries = list(directory.glob("summary-*.json"))
    if len(summaries) != 3:
        raise observer.InstrumentationError("expected controller and exactly two worker summaries")
    required = {"_maybe_start_gate", "_note_gate_reply_undelivered", "_retire_gate_reply_notice"}
    seen = set()
    for path in summaries:
        summary = json.loads(path.read_text())
        trace = directory / f"trace-{summary['pid']}.ndjson"
        if summary["instrumentation_errors"] or any(summary["policy"].values()):
            raise observer.InstrumentationError("observer/policy invalidated lane")
        if observer.digest(trace) != summary["trace_sha256"]:
            raise observer.InstrumentationError("trace hash mismatch")
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        require_events(records, {"identity"})
        seen.update(record.get("hook") for record in records)
    if not required <= seen:
        raise observer.InstrumentationError("required real instruments not exercised")
    xml = ET.parse(directory / "junit.xml")
    cells = [node for node in xml.iter("testcase") if node.get("name") == CELL]
    if len(cells) != 1:
        raise observer.InstrumentationError("exact target missing from terminal JUnit")
    if not (directory / "pytest.log").stat().st_size:
        raise observer.InstrumentationError("missing terminal log")


def run(args):
    ready = json.loads((args.work / "artifacts" / "preflight.json").read_text())
    if (
        ready["light_only"]
        or ready["controller_sha256"] != observer.digest(Path(__file__))
        or ready["observer_sha256"] != observer.digest(HERE / "observer.py")
    ):
        raise observer.InstrumentationError("full unchanged preflight required before dispatch")
    results = {}
    for name, root in roots(args).items():
        directory = args.work / "artifacts" / name
        contract_path = directory / "contract.json"
        if json.loads(contract_path.read_text()) != json.loads(json.dumps(contract(root))):
            raise observer.InstrumentationError("source contract changed after preflight")
        iso = Path(tempfile.mkdtemp(prefix=f"gate-{name}-", dir=args.work))
        command = [
            sys.executable,
            str(CARRIER / "scripts/run_bounded.py"),
            "--timeout",
            "400",
            "--memory-mb",
            "2048",
            "--",
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "observer",
            str(root / MODULE),
            "-n",
            "2",
            "--max-worker-restart=0",
            "-q",
            "--basetemp",
            str(iso / "pytest"),
            "--junitxml",
            str(directory / "junit.xml"),
        ]
        with (directory / "pytest.log").open("w") as stream:
            result = subprocess.run(
                command,
                cwd=iso,
                env=environment(iso, root, directory, contract_path),
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=420,
            )
        save(
            directory / "command.json",
            {"command": command, "exit": result.returncode, "ref": REFS[name]},
        )
        validate_lane(directory)
        if result.returncode not in (0, 1):
            raise observer.InstrumentationError("lane did not reach valid terminal test result")
        results[name] = result.returncode
    save(
        args.work / "artifacts" / "comparison.json",
        {
            "product_test_exit_codes": results,
            "diagnostic_only": True,
            "no_root_cause_classification": True,
        },
    )
    if any(results.values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("calibrate", "run", "probe"))
    parser.add_argument("--work", type=Path)
    parser.add_argument("--current-root", type=Path)
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--light-only", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args()
    if args.operation == "probe":
        probe(args.source, args.output, args.contract)
    else:
        args.work = args.work.resolve()
        args.work.mkdir(parents=True, exist_ok=True)
        (preflight if args.operation == "calibrate" else run)(args)


if __name__ == "__main__":
    main()
