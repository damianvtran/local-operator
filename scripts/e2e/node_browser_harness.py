"""Isolated node-browser E2E: a headless node's signed per-site approval, no human.

WHY THIS EXISTS. A headless mesh node (``lop browser setup --node``) runs the Local
Operator extension in a Chrome nobody can see, so a site is approved only by the
OPERATOR's signature, which the extension re-verifies against the anchor in Chrome
managed policy (``browser_bridge/site_approval.py``). Proving that end to end used
to need Damian at his Mac to answer a card. This harness proves it with a
SYNTHETIC operator key, so any agent can re-run it after a change to the node
browser, the extension, the ask door or the signed ``decide`` frame.

WHAT IS REAL. The installed extension CRX (the same one the live node runs, read
only), a real headless Chrome with a fresh profile, the real bridge daemon
(``python -m local_operator.browser_bridge.daemon``), policy-delivered pairing
(``managed-pairing.ts``), a real ``Session`` running the real browser tool, the
real ``RuntimeServer``/``ServingSessionHandle`` door, and a SEPARATE operator
process that attaches with the real ``AttachClient`` and answers through
``asks.decision.bridge_decision_target`` + ``AttachClient.bridge_decide`` — the
same two calls the desktop ``/answers`` route makes — so the challenge, the ES256
signature, the runtime's verification and the extension's re-verification are all
production code. What is NOT real: the model (a scripted stream, as in
``tests/e2e/harness.py``) and the root-owned anchor file, replaced in-process by
``tests/fixtures/anchor_fixture.install_fixture_anchor`` in both roles.

WHAT IT DOES NOT COVER (yet). The card's trip over two mesh relays into the
operator's inbox and the desktop ``/v1/desktop/sessions/{sid}/answers`` route
itself. The operator attaches to the node runtime directly over loopback
instead; both legs have their own in-repo coverage
(``tests/unit/network/test_remote_attention_relay.py``,
``tests/unit/network/test_remote_viewer.py``). See AGENTS.md "Testing a node
browser without the operator".

SAFETY ARGUMENT — why a run cannot touch the operator's real node:

* The inner run lives in private user, mount and network namespaces
  (``unshare -Urmn``). A scratch directory is bind-mounted over
  ``/etc/opt/chrome/policies/managed`` (so Chrome sees ONLY the harness policy,
  never the machine-wide force-install carrying Damian's key) and an empty one
  over ``/etc/local-operator/operators``. The inner run ABORTS if the real
  ``local-operator.json`` policy is still visible.
* The private network has only ``lo``: the real bridge daemon on
  ``127.0.0.1:4099`` is unreachable (asserted: connect is refused before the
  harness daemon starts), so the harness daemon may use the extension's default
  port and a policy pairing code can never reach the real daemon.
* Every config root (node, operator, wrong-key operator), the Chrome profile and
  the screenshots live under ``$LOCAL_OPERATOR_SCRATCHPAD``; the launcher refuses
  any root outside it or overlapping the real ``~/.local-operator`` config. The
  real asks/notifications stores are never opened.
* The anchor seam is test code (``tests/fixtures``), installed in-process only
  under ``--harness``; nothing in ``local_operator/`` reads a flag, an
  environment variable or a path override for it, and the wheel ships no
  ``scripts/``/``tests/``. ``--security-control`` checks that a normal process
  with the same isolated HOME never trusts the synthetic key.
* The outer launcher hashes the real policy and anchor, records the live node
  Chrome's PID and the paired-identity lines of the real ``lop browser status``
  before and after, and fails the run if any differ.
* The whole run is one ``systemd-run --user --scope`` with ``MemoryMax=1536M``
  and no swap; children are started in their own process groups and reaped by
  exact group id.

Run (from a checkout with a .venv; Chrome at /opt/google/chrome/chrome)::

    .venv/bin/python scripts/e2e/node_browser_harness.py --harness

Results land in ``$LOCAL_OPERATOR_SCRATCHPAD/node-browser-harness-<stamp>/``
(``results.json``, ``results.md``, ``light.png``, ``dark.png``).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parents[2]
# A standalone entrypoint must import THIS checkout (mirrors mesh_process_node).
sys.path.insert(0, str(REPO))

REAL_POLICY_DIR = Path("/etc/opt/chrome/policies/managed")
REAL_POLICY = REAL_POLICY_DIR / "local-operator.json"
REAL_ANCHOR_DIR = Path("/etc/local-operator/operators")
CHROME = Path("/opt/google/chrome/chrome")
LIVE_UNIT = "local-operator-node-browser.service"
REAL_BRIDGE_PORT = 4099
DEFAULT_STORYBOOK = Path.home() / "local-operator-ui/.worktrees/sb-pr1833/storybook-static"
STORY = "iframe.html?id=browser-approvals-dock--approved&args=theme:{theme}"
#: The ``lop browser status`` lines that identify the operator's REAL pairing.
IDENTITY_PREFIXES = ("paired:", "identities:", "- ", "extension connected:", "port:")

# Bounds measured on node2, 2026-11-16 (n=1 preflight each; generous by design —
# these are hang backstops, not latency assertions): daemon health <2 s,
# extension connect after Chrome start ~5-15 s, policy reload ~5-10 s.
CONNECT_BACKSTOP_S = 120.0
PAIR_BACKSTOP_S = 120.0
RUN_BACKSTOP_S = 900
# The answered ask started the agent's turn within ~1 s in run 5; 20 s is the
# window before the harness nudges it instead. A turn here is 2-4 browser calls
# against loopback pages; 180 s is a hang backstop, not a latency assertion.
ANSWER_DELIVERY_WINDOW_S = 20.0
TURN_BACKSTOP_S = 180.0


# ---------------------------------------------------------------------------
# Guards (unit-tested in tests/unit/scripts/test_node_browser_harness.py)
# ---------------------------------------------------------------------------


def refusal(
    *,
    harness_flag: bool,
    root: Path,
    scratchpad: str | None,
    real_config: Path,
) -> str | None:
    """Why this launch must not run, or ``None``. Pure, so it can be pinned.

    The harness writes keys, policy and Chrome profiles; pointed at the real
    config it could overwrite the operator's node. So it runs only when asked
    explicitly (``--harness``) and only inside the session scratchpad, never
    inside or above the real config root.
    """
    if not harness_flag:
        return "refused: pass --harness to run the isolated node-browser harness"
    if not scratchpad:
        return "refused: LOCAL_OPERATOR_SCRATCHPAD is not set; the harness writes only there"
    resolved = root.resolve()
    pad = Path(scratchpad).resolve()
    real = real_config.resolve()
    if resolved != pad and pad not in resolved.parents:
        return f"refused: root {resolved} is not inside the scratchpad {pad}"
    if resolved == real or resolved in real.parents:
        return f"refused: root {resolved} is (or contains) the real config root {real}"
    # The scratchpad itself lives under ~/.local-operator/sessions/<id>/; the
    # root must be a CHILD of it, never one of the real stores beside it.
    if resolved == pad:
        return "refused: use a fresh subdirectory of the scratchpad, not the scratchpad itself"
    return None


def namespace_refusal(policy_file: Path = REAL_POLICY) -> str | None:
    """Inside the namespaces the real managed policy must be masked."""
    if policy_file.exists():
        return (
            f"aborted: {policy_file} is visible, so this process is NOT inside the "
            "harness mount namespace; refusing to start Chrome or a bridge daemon"
        )
    return None


def port_refused(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as probe:
        probe.settimeout(1.0)
        return probe.connect_ex((host, port)) != 0


# ---------------------------------------------------------------------------
# Host evidence (outer, real HOME)
# ---------------------------------------------------------------------------


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        return f"unreadable:{error.__class__.__name__}"


def host_snapshot() -> dict[str, Any]:
    """Hashes of the real anchor/policy, the live Chrome PID, the real pairing."""
    pid = subprocess.run(
        ["systemctl", "--user", "show", "-p", "MainPID", LIVE_UNIT],
        capture_output=True,
        text=True,
    ).stdout.strip()
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    env.pop("LOCAL_OPERATOR_CONFIG_DIR", None)
    status = subprocess.run(
        ["lop", "browser", "status"], capture_output=True, text=True, env=env, timeout=60
    ).stdout
    identity = [
        line.rstrip() for line in status.splitlines() if line.strip().startswith(IDENTITY_PREFIXES)
    ]
    hashes = {str(REAL_POLICY): _sha(REAL_POLICY)}
    for anchor in sorted(REAL_ANCHOR_DIR.glob("*.json")):
        hashes[str(anchor)] = _sha(anchor)
    return {"hashes": hashes, "live_chrome": pid, "browser_status_identity": identity}


# ---------------------------------------------------------------------------
# Outer launcher
# ---------------------------------------------------------------------------


def _child_env(**extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CMUX_", "LOP_", "LOCAL_OPERATOR_CONFIG_DIR", "XPC_"))
    }
    env.update(extra)
    return env


def outer(args: argparse.Namespace) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    before = host_snapshot()
    policy_mask = root / "policy"
    anchor_mask = root / "anchor-mask"
    for path in (policy_mask, anchor_mask):
        path.mkdir(exist_ok=True)
    inner_argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--inner",
        "--harness",
        "--root",
        str(root),
        "--storybook-dir",
        str(args.storybook_dir),
    ] + (["--preflight"] if args.preflight else [])
    script = " && ".join(
        [
            f"mount --bind {shlex.quote(str(policy_mask))} {REAL_POLICY_DIR}",
            f"mount --bind {shlex.quote(str(anchor_mask))} {REAL_ANCHOR_DIR}",
            "ip link set lo up",
            # Map back to the real uid so Chrome is not "root" and keeps its sandbox.
            "exec unshare -U --map-user=1000 --map-group=1000 "
            + " ".join(shlex.quote(a) for a in inner_argv),
        ]
    )
    command = [
        "systemd-run",
        "--user",
        "--scope",
        "-q",
        "-p",
        "MemoryMax=1536M",
        "-p",
        "MemorySwapMax=0",
        "--",
        "timeout",
        str(RUN_BACKSTOP_S),
        "unshare",
        "-Urmn",
        "sh",
        "-c",
        script,
    ]
    log = (root / "inner.log").open("w")
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=_child_env(), start_new_session=True
    )
    try:
        rc = process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        log.close()
    after = host_snapshot()
    results_path = root / "results.json"
    report: dict[str, Any] = (
        json.loads(results_path.read_text()) if results_path.exists() else {"rows": []}
    )
    rows = report.setdefault("rows", [])
    for key, label in (
        ("hashes", "real /etc policy + anchor hashes unchanged"),
        ("live_chrome", "live node Chrome PID unchanged"),
        ("browser_status_identity", "real `lop browser status` paired identity unchanged"),
    ):
        rows.append(
            {
                "row": label,
                "expected": json.dumps(before[key]),
                "actual": json.dumps(after[key]),
                "verdict": "PASS" if before[key] == after[key] else "FAIL",
            }
        )
    report["host_before"] = before
    report["host_after"] = after
    report["inner_rc"] = rc
    results_path.write_text(json.dumps(report, indent=2))
    markdown = render_table(rows)
    (root / "results.md").write_text(markdown)
    print(markdown)
    print(f"inner rc={rc}; results: {results_path}")
    failed = [row for row in rows if row["verdict"] != "PASS"]
    return 0 if rc == 0 and not failed and rows else 1


def render_table(rows: list[dict[str, Any]]) -> str:
    def cell(text: Any) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ")[:400]

    lines = ["| row | expected | actual | verdict |", "|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {cell(row['row'])} | {cell(row['expected'])} | {cell(row['actual'])} "
            f"| {row['verdict']} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Inner run (inside the namespaces)
# ---------------------------------------------------------------------------


class Results:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"rows": [], "facts": {}}

    def row(self, row: str, expected: str, actual: Any, ok: bool, blocked: bool = False) -> bool:
        """``blocked``: the row's precondition did not hold, so it proves nothing."""
        self.data["rows"].append(
            {
                "row": row,
                "expected": expected,
                "actual": actual if isinstance(actual, str) else json.dumps(actual, default=str),
                "verdict": "BLOCKED" if blocked else ("PASS" if ok else "FAIL"),
            }
        )
        self.flush()
        return ok

    def fact(self, key: str, value: Any) -> None:
        self.data["facts"][key] = value
        self.flush()

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, default=str))


class Group:
    """Children in their own process groups, reaped by exact group id."""

    def __init__(self) -> None:
        self.children: list[subprocess.Popen[Any]] = []

    def spawn(self, argv: list[str], *, log: Path, env: dict[str, str]) -> subprocess.Popen[Any]:
        handle = log.open("w")
        child = subprocess.Popen(
            argv, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        handle.close()
        self.children.append(child)
        return child

    def reap(self) -> None:
        for child in reversed(self.children):
            if child.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        for child in self.children:
            try:
                child.wait(max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def _until(predicate: Any, timeout: float, step: float = 0.5) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(step)
    return None


def _make_operator_key(config_root: Path, label: str) -> Any:
    """A real file-only operator key, staged as ``lop operator init`` does."""
    from local_operator.operator.keychain import FILE_ONLY
    from local_operator.operator.sign import anchor_for_handle, create_key
    from local_operator.operator.trust import anchor_bytes, staging_path

    config_root.mkdir(parents=True, exist_ok=True)
    handle = create_key(config_root=config_root, preference=FILE_ONLY)
    anchor = anchor_for_handle(handle, label=label)
    staged = staging_path(config_root)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(anchor_bytes(anchor))
    return anchor


def _write_policy(policy_dir: Path, extension_id: str, update: Path, **extra: Any) -> None:
    from local_operator.browser_bridge import node_host

    policy = node_host.render_policy(extension_id, update, **extra)
    target = policy_dir / "harness.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(policy, indent=2))
    os.replace(tmp, target)


def _extension_source() -> tuple[str, Path, str]:
    """The extension id, CRX and version the live node runs (read only)."""
    # The passwd home, not ``$HOME``: the inner run points HOME at the node's
    # scratch root, and the installed CRX lives under the real account.
    import pwd
    import re

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    update = real_home / ".local-operator/browser/extension/update.xml"
    text = update.read_text()
    appid = re.search(r"appid='([a-p]{32})'", text)
    codebase = re.search(r"codebase='file://([^']+)'", text)
    version = re.search(r"<updatecheck[^>]*version='([0-9.]+)'", text)
    if not (appid and codebase and version):
        raise SystemExit(f"cannot read the installed extension from {update}")
    return appid.group(1), Path(codebase.group(1)), version.group(1)


def _call(call_id: str, args: dict[str, Any]) -> list[Any]:
    from tests.e2e.harness import tool_call_turn

    return tool_call_turn(
        text=f"step {call_id}", tool_name="browser", tool_call_id=call_id, arguments=args
    )


class Capture:
    """A screenshot step that waits for the story to RENDER before it counts.

    WHY. Run 6 captured Storybook's loading spinner for the dark story: ``open``
    returns at document load and Storybook renders afterwards, so the frame was
    taken too early and the two hashes differed for the wrong reason. The
    browser tool has no wait-for-selector action, so the tape polls the way a
    model would: ``snapshot`` (the story root's "Approvals" heading must be in
    the accessibility tree), then ``screenshot``, then check the PNG itself —
    enough non-background pixels (:func:`rendered_fraction`) and the theme's
    own page background (:func:`background_luminance`: dark < 100, light > 180).
    Anything short of that sleeps a second and tries again, up to
    ``RENDER_WAIT_S``; a timeout records ``rendered: False`` and the row FAILS
    with "not rendered" rather than citing a spinner.
    """

    def __init__(self, theme: str, path: str) -> None:
        self.theme = theme
        self.path = path
        self.outcome: dict[str, Any] = {"theme": theme, "rendered": False, "attempts": 0}
        self._started = 0.0
        self._phase = "snapshot"

    def _ready(self) -> bool:
        frame = Path(self.path)
        if not frame.exists():
            return False
        fraction = rendered_fraction(frame)
        luminance = background_luminance(frame)
        self.outcome.update(
            rendered_fraction=round(fraction, 4), background_luminance=round(luminance, 1)
        )
        themed = (
            luminance < DARK_MAX_LUMINANCE
            if self.theme == "dark"
            else (luminance > LIGHT_MIN_LUMINANCE)
        )
        return fraction >= RENDERED_MIN_FRACTION and themed

    async def next_turn(self, request: Any) -> list[Any] | None:
        """The next model turn for this step, or ``None`` when the step is done."""
        if not self._started:
            self._started = time.monotonic()
        attempt = self.outcome["attempts"]
        if self._phase == "shot":
            if self._ready():
                self.outcome["rendered"] = True
                return None
            self._phase = "snapshot"
        elif self._phase == "check":
            text = _last_tool_text(request)
            self.outcome["story_root_seen"] = "Approvals" in text
            if self.outcome["story_root_seen"]:
                self._phase = "shot"
                return _call(
                    f"shot_{self.theme}_{attempt}", {"action": "screenshot", "path": self.path}
                )
            self._phase = "snapshot"
        if time.monotonic() - self._started > RENDER_WAIT_S:
            self.outcome["error"] = f"not rendered within {RENDER_WAIT_S:.0f} s"
            return None
        if attempt:
            await asyncio.sleep(1.0)
        self.outcome["attempts"] = attempt + 1
        self._phase = "check"
        return _call(
            f"snap_{self.theme}_{attempt + 1}", {"action": "snapshot", "search": "Approvals"}
        )


def _last_tool_text(request: Any) -> str:
    for message in reversed(getattr(request, "messages", None) or ()):
        if getattr(message, "role", "") == "tool":
            return " ".join(str(getattr(block, "text", "") or "") for block in message.content)
    return ""


class PollingTape:
    """The scripted model: fixed turns plus :class:`Capture` steps.

    Same contract as ``tests.e2e.harness.ScriptedStream`` (one turn per model
    call, ``(request, signal) -> async iterator``), but a capture step decides
    its next call from the previous tool result, like a model reading its own
    screenshots. A call past the end raises, as ScriptedStream does.
    """

    def __init__(self, steps: list[Any]) -> None:
        self.steps = list(steps)
        self.captures = {step.theme: step for step in steps if isinstance(step, Capture)}

    def __call__(self, request: Any, signal: Any = None) -> Any:
        async def gen() -> Any:
            while self.steps:
                step = self.steps[0]
                if isinstance(step, Capture):
                    turn = await step.next_turn(request)
                    if turn is None:
                        self.steps.pop(0)
                        continue
                else:
                    turn = self.steps.pop(0)
                for event in turn:
                    yield event
                return
            raise RuntimeError("the harness tape ran out: an unexpected extra model call")

        return gen()


def _tape(urls: dict[str, str], shots: dict[str, str]) -> PollingTape:
    from tests.e2e.harness import text_turn

    return PollingTape(
        [
            # Prompt 1: the open is refused, then the card is raised.
            _call("call_1", {"action": "open", "url": urls["light"]}),
            _call("call_2", {"action": "request_access", "url": urls["light"]}),
            text_turn("asked the operator"),
            # The turn the APPROVE starts (answer delivery): open, capture both themes.
            _call("call_3", {"action": "open", "url": urls["light"]}),
            Capture("light", shots["light"]),
            _call("call_5", {"action": "open", "url": urls["dark"]}),
            Capture("dark", shots["dark"]),
            text_turn("captured both themes"),
            # Prompt 3: a second origin, which the operator will DENY.
            _call("call_7", {"action": "open", "url": urls["deny"]}),
            _call("call_8", {"action": "request_access", "url": urls["deny"]}),
            text_turn("asked about the second origin"),
            # The turn the DENY starts: the open must still be refused.
            _call("call_9", {"action": "open", "url": urls["deny"]}),
            text_turn("denied as expected"),
            # Prompt 5: a third origin whose request the AGENT withdraws with its
            # own identity (``cancel_access`` from this session's tool context,
            # which also withdraws the card) before the operator's late answer.
            _call("call_10", {"action": "open", "url": urls["expire"]}),
            _call("call_11", {"action": "request_access", "url": urls["expire"]}),
            _call("call_12", {"action": "cancel_access", "url": urls["expire"]}),
            text_turn("withdrew the third request"),
        ]
    )


def _tool_results(session: Any) -> list[dict[str, Any]]:
    """Tool results as the node's transcript recorded them (complete lines only).

    Read from disk, not memory: the journal is what every later reader trusts,
    and it carries the typed ``error_code`` (``provider_payload.details``) that
    the model-facing text deliberately words for a person.
    """
    path = Path(session.transcript.path)
    data = path.read_bytes() if path.exists() else b""
    out = []
    for line in data[: data.rfind(b"\n") + 1].decode("utf-8").split("\n"):
        if not line.strip():
            continue
        payload = json.loads(line).get("payload") or {}
        if payload.get("role") != "tool":
            continue
        out.append(
            {
                "id": payload.get("tool_call_id") or "",
                "error": bool(payload.get("is_error")),
                "text": " ".join(
                    str(block.get("text") or "")
                    for block in payload.get("content") or ()
                    if isinstance(block, dict)
                ),
                "details": (payload.get("provider_payload") or {}).get("details") or {},
            }
        )
    return out


def _last_result(session: Any, call_id: str) -> dict[str, Any]:
    for result in reversed(_tool_results(session)):
        if result["id"] == call_id:
            return result
    return {"id": call_id, "error": True, "text": "<no tool result recorded>", "details": None}


async def _continue_after_answer(
    session: Any, first_call: str, last_call: str, fallback_prompt: str
) -> str:
    """Let the node agent resume once its card is answered; how it resumed.

    An answered queued ask is DELIVERED to the agent, which starts its next turn
    by itself when the session is idle (run 5 measured it: prompting on top of it
    raised ``TurnInFlight``). So wait for that turn; only if none begins within
    the window is the agent nudged with a prompt, and the result says which.
    ``_turn_lock`` is private, read here only to know when the turn is over.
    """
    lock = session._turn_lock

    def started() -> bool:
        return lock.locked() or _has_result(session, first_call)

    how = "auto (answer delivery)"
    if not await _until(started, ANSWER_DELIVERY_WINDOW_S, step=0.2):
        how = "prompted (no turn began after the answer)"
        await session.prompt(fallback_prompt)
    await _until(
        lambda: _has_result(session, last_call) and not lock.locked(), TURN_BACKSTOP_S, step=0.2
    )
    while lock.locked():
        await asyncio.sleep(0.2)
    return how


def _has_result(session: Any, call_id: str) -> bool:
    return any(result["id"] == call_id for result in _tool_results(session))


def _open_site_ask(session: Any, origin: str) -> dict[str, Any] | None:
    queue = session.ask_queue()
    for record in reversed(queue.open_records() if queue else []):
        for question in record.get("questions") or ():
            if (question.get("site") or {}).get("origin") == origin:
                return record
    return None


def withdrawal_confirmed(result: dict[str, Any]) -> bool:
    """Whether the agent's ``cancel_access`` removed the extension's entry.

    The extension's own reply is the proof: ``state: cancelled`` only when it
    found and dropped this requester's pending entry (``approval-store.ts``
    ``cancelAccess``; ``none`` when nothing matched, which is what run 6's
    identity-less call got). Run 7 gated on the card's ``withdrawn_asks`` too
    and reported BLOCKED although the entry was gone: that list is filled only
    on a host ``node_host.headless_node_once()`` calls a node, i.e. a mesh
    member, and this rig's node root is not one (the relay leg is out of
    scope). So the card stays open here; the extension entry does not.
    """
    details = result.get("details") or {}
    return not result.get("error") and details.get("state") == "cancelled"


def _site_ask(session: Any, origin: str) -> dict[str, Any] | None:
    """The newest site card for ``origin`` in ANY status (a withdrawn one too)."""
    queue = session.ask_queue()
    for record in reversed(queue.records() if queue else []):
        for question in record.get("questions") or ():
            if (question.get("site") or {}).get("origin") == origin:
                return record
    return None


def _ask_status(session: Any, ask_id: str) -> str:
    queue = session.ask_queue()
    found = queue.find(ask_id) if queue else None
    return str((found or {}).get("status") or "missing")


def _sites(node_root: Path) -> str:
    env = _child_env(LOCAL_OPERATOR_CONFIG_DIR=str(node_root), HOME=str(node_root / "home"))
    done = subprocess.run(
        [sys.executable, "-m", "local_operator.cli", "browser", "sites"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return (done.stdout + done.stderr).strip()


def _site_lists(node_root: Path) -> dict[str, list[str]]:
    """``lop browser sites --json``: the extension's allowed and denied origins.

    The human listing prints BOTH (an "Allowed" and a "Denied" section), so a
    substring test on it cannot tell a signed Deny from an Approve — run 6 read
    a correctly denied origin as "listed". Decide on the typed lists instead.
    """
    env = _child_env(LOCAL_OPERATOR_CONFIG_DIR=str(node_root), HOME=str(node_root / "home"))
    done = subprocess.run(
        [sys.executable, "-m", "local_operator.cli", "browser", "sites", "--json"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    try:
        listing = json.loads(done.stdout)
    except ValueError:
        return {"allowed": [], "denied": [], "error": [done.stdout[-200:] + done.stderr[-200:]]}
    return {
        "allowed": [str(item) for item in listing.get("allowed") or ()],
        "denied": [str(item) for item in listing.get("denied") or ()],
    }


def _allowed(node_root: Path, origin: str) -> bool:
    return origin in _site_lists(node_root)["allowed"]


def rendered_fraction(path: Path) -> float:
    """Share of pixels that differ from the frame's corner (background) colour.

    WHY. Two different sha256s do not prove two rendered themes: run 6's dark
    frame was Storybook's loading spinner on white (0.06% non-background) while
    the light frame showed the story (6.2%), so the hashes differed for the
    wrong reason. A frame must show content before the pair counts. Sampled on
    a 2-px grid; Pillow is imported lazily (it ships with the images extra).
    """
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    background = pixels[width - 1, height - 1]
    total = differing = 0
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            total += 1
            if sum(abs(a - b) for a, b in zip(pixels[x, y], background)) > 30:
                differing += 1
    return differing / total if total else 0.0


#: Between the two frames measured in run 6 (n=1 each): the loading spinner at
#: 0.0006 and the rendered light story at 0.0616.
RENDERED_MIN_FRACTION = 0.01

#: Page-background luminance (Rec. 601) of the frame's corner. Run 6's light
#: story measured (242, 237, 227) -> ~238; the dark bound is set well below
#: mid-grey. Not yet measured on a rendered dark frame — run 7 records it.
DARK_MAX_LUMINANCE = 100.0
LIGHT_MIN_LUMINANCE = 180.0
#: How long a capture waits for the story to render. Run 6's light frame had
#: rendered by the time ``open`` returned; 20 s is a backstop for a slow first
#: paint, not an expected latency.
RENDER_WAIT_S = 20.0


def background_luminance(path: Path) -> float:
    """Luminance of the frame's bottom-right pixel: the theme's page background."""
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
    red, green, blue = rgb.getpixel((rgb.width - 1, rgb.height - 1))
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _operator(
    root: Path, key_root: Path, node_root: Path, sid: str, ask: dict[str, Any], word: str, mode: str
) -> dict[str, Any]:
    """Run ONE operator decision in a separate process with its own config root."""
    ask_file = root / f"ask-{uuid.uuid4().hex[:8]}.json"
    ask_file.write_text(json.dumps(ask))
    env = _child_env(
        LOCAL_OPERATOR_CONFIG_DIR=str(key_root),
        HOME=str(key_root / "home"),
        TMPDIR=str(key_root / "tmp"),
    )
    done = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--operator",
            "--harness",
            "--root",
            str(root),
            "--node-root",
            str(node_root),
            "--session",
            sid,
            "--ask-file",
            str(ask_file),
            "--word",
            word,
            "--mode",
            mode,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        start_new_session=True,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if not lines:
        return {
            "ok": False,
            "connected": False,
            "error": f"operator child rc={done.returncode}: {done.stderr[-800:]}",
        }
    return dict(json.loads(lines[-1]))


async def inner(args: argparse.Namespace) -> int:
    root = Path(args.root)
    results = Results(root / "results.json")
    aborted = namespace_refusal()
    if aborted:
        print(aborted, file=sys.stderr)
        results.row("namespace masks the real managed policy", "absent", aborted, False)
        return 2
    results.row(
        "namespace masks the real managed policy",
        f"{REAL_POLICY} absent",
        f"exists={REAL_POLICY.exists()}; visible={sorted(os.listdir(REAL_POLICY_DIR))}",
        True,
    )
    refused = port_refused(REAL_BRIDGE_PORT)
    results.row(
        "real bridge 127.0.0.1:4099 unreachable before harness daemon",
        "connect refused",
        f"refused={refused}",
        refused,
    )
    if not refused:
        return 2

    node_root, op_root, bad_root = root / "node", root / "operator", root / "wrong-key-operator"
    for base in (node_root, op_root, bad_root):
        for sub in ("home", "tmp"):
            (base / sub).mkdir(parents=True, exist_ok=True)
    shots = {"light": str(root / "light.png"), "dark": str(root / "dark.png")}

    # Synthetic keys: the operator's (trusted by the node), and a second one
    # nobody trusts (the wrong-key control). The node root holds NO private key,
    # or the browser tool would refuse to raise a card (the agent could sign).
    from local_operator.operator.trust import anchor_bytes

    anchor = _make_operator_key(op_root, "harness-operator")
    _make_operator_key(bad_root, "harness-wrong-key")
    for base in (node_root, op_root, bad_root):
        (base / "fixture-owned-anchor.json").write_bytes(anchor_bytes(anchor))
    results.fact("synthetic_operator_key_id", anchor.key_id)

    os.environ.update(
        LOCAL_OPERATOR_CONFIG_DIR=str(node_root),
        HOME=str(node_root / "home"),
        TMPDIR=str(node_root / "tmp"),
    )
    from tests.fixtures.anchor_fixture import install_fixture_anchor

    native_reads = install_fixture_anchor(node_root)
    from local_operator.network import identity

    device = identity.load_or_mint(node_root, name="harness-node")
    results.fact("node_device_id", device.device_id)

    group = Group()
    session_box: dict[str, Any] = {}
    try:
        return await _inner_flow(
            args,
            root,
            results,
            group,
            node_root,
            op_root,
            bad_root,
            shots,
            native_reads,
            session_box,
        )
    finally:
        runtime = session_box.get("runtime")
        if runtime is not None:
            with contextlib.suppress(Exception):
                runtime.close()
        group.reap()
        profile = root / "chrome-profile"
        shutil.rmtree(profile, ignore_errors=True)
        if session_box.get("chrome_tmp"):
            shutil.rmtree(session_box["chrome_tmp"], ignore_errors=True)
        results.fact("chrome_profile_removed", not profile.exists())
        results.fact("native_anchor_reads", native_reads["native_anchor_reads"])


async def _inner_flow(
    args: argparse.Namespace,
    root: Path,
    results: Results,
    group: Group,
    node_root: Path,
    op_root: Path,
    bad_root: Path,
    shots: dict[str, str],
    native_reads: dict[str, int],
    session_box: dict[str, Any],
) -> int:
    from local_operator.browser_bridge import install, site_approval

    env = _child_env(
        LOCAL_OPERATOR_CONFIG_DIR=str(node_root),
        HOME=str(node_root / "home"),
        TMPDIR=str(node_root / "tmp"),
    )
    storybook = Path(args.storybook_dir)
    if not (storybook / "iframe.html").exists():
        results.row("storybook-static present", "iframe.html", str(storybook), False)
        return 2
    ports = {name: _free_port() for name in ("light", "deny", "expire")}
    for name, port in ports.items():
        group.spawn(
            [
                sys.executable,
                "-m",
                "http.server",
                str(port),
                "--bind",
                "127.0.0.1",
                "--directory",
                str(storybook),
            ],
            log=root / f"http-{name}.log",
            env=env,
        )
    base = {name: f"http://127.0.0.1:{port}" for name, port in ports.items()}
    urls = {
        "light": f"{base['light']}/{STORY.format(theme='localOperatorLight')}",
        "dark": f"{base['light']}/{STORY.format(theme='localOperatorDark')}",
        "deny": f"{base['deny']}/{STORY.format(theme='localOperatorLight')}",
        "expire": f"{base['expire']}/{STORY.format(theme='localOperatorLight')}",
    }

    group.spawn(
        [
            sys.executable,
            "-m",
            "local_operator.browser_bridge.daemon",
            "--port",
            str(REAL_BRIDGE_PORT),
        ],
        log=root / "bridge-daemon.log",
        env=env,
    )
    healthy = await _until(lambda: install.health(REAL_BRIDGE_PORT, timeout=1.0), 30)
    results.row("harness bridge daemon healthy (inside netns)", "health ok", healthy, bool(healthy))
    if not healthy:
        return 2

    extension_id, crx, version = _extension_source()
    results.fact("extension", {"id": extension_id, "crx": str(crx), "version": version})
    extension_dir = root / "extension"
    extension_dir.mkdir(exist_ok=True)
    from local_operator.browser_bridge import node_host

    update = extension_dir / "update.xml"
    update.write_text(node_host.render_update_manifest(extension_id, crx, version))
    approval = site_approval.anchor_policy_fields(node_root)
    results.row(
        "policy carries the SYNTHETIC anchor + node id",
        "operatorKeyId == synthetic key id",
        {
            k: (v[:16] + "…" if isinstance(v, str) and len(v) > 40 else v)
            for k, v in approval.items()
        },
        approval.get(site_approval.POLICY_KEY_ID)
        == results.data["facts"]["synthetic_operator_key_id"],
    )
    policy_dir = REAL_POLICY_DIR  # the bind-mounted scratch dir, inside the namespace
    _write_policy(policy_dir, extension_id, update, approval=approval)

    profile = root / "chrome-profile"
    chrome_argv = [
        str(CHROME),
        "--headless=new",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--password-store=basic",
        "--use-mock-keychain",
        "about:blank",
    ]
    # Chrome's singleton socket is a unix socket under TMPDIR, whose path must
    # fit in 108 bytes; the scratchpad path does not (run 1: "Socket path too
    # long"). A short private dir under /tmp, removed with the profile.
    chrome_tmp = Path(tempfile.mkdtemp(prefix="nbh-", dir="/tmp"))
    session_box["chrome_tmp"] = chrome_tmp
    chrome_env = {**env, "TMPDIR": str(chrome_tmp)}
    sandbox = "sandboxed"
    chrome = group.spawn(chrome_argv, log=root / "chrome.log", env=chrome_env)
    await asyncio.sleep(6)
    if chrome.poll() is not None:
        sandbox = "--no-sandbox (sandboxed launch exited rc=%s)" % chrome.returncode
        chrome = group.spawn(
            chrome_argv[:1] + ["--no-sandbox"] + chrome_argv[1:],
            log=root / "chrome-nosandbox.log",
            env=chrome_env,
        )
    results.fact("chrome_sandbox", sandbox)

    def codes() -> Any:
        if chrome.poll() is not None:
            # Chrome died (see chrome*.log): stop waiting at once.
            return ["chrome-exited"]
        answer = install.request_pairing_codes(port=REAL_BRIDGE_PORT, root=node_root)
        return answer.get("codes") if answer.get("ok") else None

    pending = await _until(codes, CONNECT_BACKSTOP_S, step=1.0)
    results.row(
        "extension force-installed and connected unpaired",
        "a pairing code minted for it",
        {"codes": len(pending or []), "chrome_alive": chrome.poll() is None},
        bool(pending),
    )
    if not pending or pending == ["chrome-exited"]:
        results.data["rows"][-1]["verdict"] = "FAIL"
        results.flush()
        return 2
    code = str(pending[0].get("code") or "")
    _write_policy(policy_dir, extension_id, update, pairing_code=code, approval=approval)

    def paired() -> Any:
        answer = install.request_pairing_codes(port=REAL_BRIDGE_PORT, root=node_root)
        return answer.get("paired_connected_count", 0) >= 1 and answer

    done = await _until(paired, PAIR_BACKSTOP_S, step=1.0)
    # The code is removed as soon as it was used, as node_host setup does.
    _write_policy(policy_dir, extension_id, update, approval=approval)
    results.row(
        "policy pairing code pairs the extension", "paired_connected_count>=1", done, bool(done)
    )
    if not done:
        return 2
    if args.preflight:
        # Rig check only: namespace, daemon, CRX install and policy pairing.
        return 0

    # --- the node session --------------------------------------------------
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle
    from local_operator.tools.builtin import build_browser_tool
    from tests.e2e.harness import build_session

    tool = build_browser_tool(None)
    if tool is None:
        results.row("browser tool advertised on the node", "tool present", None, False)
        return 2
    sid = uuid.uuid4().hex[:12]
    session_dir = node_root / "sessions" / sid
    session_dir.mkdir(parents=True)
    stream = _tape(urls, shots)
    session = build_session(session_dir, stream, tools=[tool], cwd=node_root)
    loop = asyncio.get_running_loop()
    handle = ServingSessionHandle(session, loop, cwd=str(node_root))
    runtime = RuntimeServer(handle, kind="tui")
    session_box["runtime"] = runtime
    await runtime.start_in_process()
    live = await _until(lambda: [r for r, s in registry.scan(node_root) if s == "live"], 20, 0.1)
    results.fact("session_id", getattr(session, "session_id", sid))
    sid = str(getattr(session, "session_id", "") or sid)
    if not live:
        results.row("node runtime published", "live record", None, False)
        return 2

    origin = base["light"]
    await session.prompt("open the approvals dock story")
    refused_open = _last_result(session, "call_1")
    results.row(
        "first open refused: origin_not_allowed",
        "is_error, details.error_code == origin_not_allowed",
        {
            "error_code": refused_open["details"].get("error_code"),
            "text": refused_open["text"][:160],
        },
        refused_open["error"] and refused_open["details"].get("error_code") == "origin_not_allowed",
    )
    ask = _open_site_ask(session, origin)
    results.row(
        "request_access raises the signed site card (site:v1:)",
        "open ask with a site:v1: question for the origin",
        {
            "ask": (ask or {}).get("ask_id"),
            "question_id": ((ask or {}).get("questions") or [{}])[0].get("question_id", "")[:24],
            "result": _last_result(session, "call_2")["text"][:200],
        },
        bool(ask),
    )
    if not ask:
        return 2
    ask_id = str(ask["ask_id"])
    approve_word = "approve"

    # --- negative controls before the real decision -----------------------
    wrong = await asyncio.to_thread(
        _operator, root, bad_root, node_root, sid, ask, approve_word, "signed"
    )
    results.row(
        "wrong-key operator signature refused",
        "decide refused, ask still open, origin not granted",
        {
            "verdict": wrong,
            "ask": _ask_status(session, ask_id),
            "allowed": _allowed(node_root, origin),
        },
        wrong.get("connected")
        and (not wrong.get("ok"))
        and _ask_status(session, ask_id) == "open"
        and not _allowed(node_root, origin),
    )
    plain = await asyncio.to_thread(
        _operator, root, op_root, node_root, sid, ask, approve_word, "plain"
    )
    results.row(
        "plain ask_respond from the operator surface is not the decision",
        "refused, ask still open, origin not granted",
        {
            "verdict": plain,
            "ask": _ask_status(session, ask_id),
            "allowed": _allowed(node_root, origin),
        },
        plain.get("connected")
        and (not plain.get("ok"))
        and _ask_status(session, ask_id) == "open"
        and not _allowed(node_root, origin),
    )
    question = ask["questions"][0]
    agent = await session.site_decide(ask_id, str(question["question_id"]), approve_word)
    results.row(
        "agent's own unsigned decide (node-local, no operator signature) refused",
        "extension refuses (unsigned_decision), origin not granted",
        {
            "verdict": agent,
            "ask": _ask_status(session, ask_id),
            "allowed": _allowed(node_root, origin),
        },
        (not agent.get("ok")) and not _allowed(node_root, origin),
    )

    # --- the real decision ----------------------------------------------
    approved = await asyncio.to_thread(
        _operator, root, op_root, node_root, sid, ask, approve_word, "signed"
    )
    sites_after = _sites(node_root)
    allowed_after = _allowed(node_root, origin)
    results.row(
        "operator's signed Approve lands on the node",
        "decided; ask answered; `lop browser sites` lists the origin",
        {"verdict": approved, "ask": _ask_status(session, ask_id), "sites": sites_after[:300]},
        bool(approved.get("ok")) and allowed_after,
    )
    # ``call_5`` is the dark ``open``; the turn then polls the dark capture, so
    # "call_5 recorded and the turn lock free" means both captures settled.
    resumed = await _continue_after_answer(
        session, "call_3", "call_5", "now open it and capture both themes"
    )
    results.fact("resumed_after_approve", resumed)
    opened = _last_result(session, "call_3")
    results.row(
        "open succeeds after approval",
        "no error",
        {"resumed": resumed, "text": opened["text"][:200]},
        not opened["error"],
    )
    digests: dict[str, str] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    for theme in ("light", "dark"):
        path = Path(shots[theme])
        digests[theme] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        outcomes[theme] = dict(stream.captures[theme].outcome)
    results.fact(
        "screenshots",
        {k: {"path": shots[k], "sha256": v, **outcomes[k]} for k, v in digests.items()},
    )
    rendered = all(outcome["rendered"] for outcome in outcomes.values())
    results.row(
        "light and dark screenshots rendered in their themes and differ",
        "story root seen; rendered_fraction >= "
        f"{RENDERED_MIN_FRACTION}; background luminance dark < {DARK_MAX_LUMINANCE:.0f}, "
        f"light > {LIGHT_MIN_LUMINANCE:.0f}; different sha256",
        {
            "sha256": digests,
            "captures": outcomes,
            **({} if rendered else {"error": "not rendered"}),
        },
        rendered and all(digests.values()) and digests["light"] != digests["dark"],
    )

    # --- deny ---------------------------------------------------------------
    deny_origin = base["deny"]
    await session.prompt("try the second origin")
    deny_ask = _open_site_ask(session, deny_origin)
    if deny_ask is None:
        results.row("deny: card raised", "open ask", _last_result(session, "call_8"), False)
    else:
        denied = await asyncio.to_thread(
            _operator, root, op_root, node_root, sid, deny_ask, "deny", "signed"
        )
        resumed_deny = await _continue_after_answer(
            session, "call_9", "call_9", "open the second origin again"
        )
        again = _last_result(session, "call_9")
        again["resumed"] = resumed_deny
        lists = _site_lists(node_root)
        results.row(
            "deny: signed Deny keeps the origin out and the open refused",
            "decided; origin in sites' denied list, not allowed; open still refused",
            {"verdict": denied, "lists": lists, "open": again["details"], "resumed": resumed_deny},
            bool(denied.get("ok"))
            and deny_origin in lists["denied"]
            and deny_origin not in lists["allowed"]
            and again["error"]
            and again["details"].get("error_code") == "origin_not_allowed",
        )

    # --- expired card (entry withdrawn before the answer) -------------------
    # Run 6 withdrew with a bare harness ``cancel_access`` that carried no
    # requester identity, so the extension (which binds an entry to
    # ``session:<id>``) kept it and rightly honoured the late Approve. The AGENT
    # now withdraws through its own browser tool (``call_12``), the same
    # identity-carrying path that raised it; the extension answers
    # ``state: cancelled`` and the tool withdraws the card with it. Only then
    # does the operator answer, from the card it had been shown.
    expire_origin = base["expire"]
    await session.prompt("try the third origin")
    expire_ask = _site_ask(session, expire_origin)
    withdrawal = _last_result(session, "call_12")
    if expire_ask is None:
        results.row("expired: card raised", "an ask", _last_result(session, "call_11"), False)
    else:
        details = withdrawal["details"] or {}
        gone = withdrawal_confirmed(withdrawal)
        late = await asyncio.to_thread(
            _operator, root, op_root, node_root, sid, expire_ask, approve_word, "signed"
        )
        allowed_late = _allowed(node_root, expire_origin)
        results.row(
            "expired card: a signed Approve after the entry is withdrawn grants nothing",
            "agent cancel_access -> state cancelled + card withdrawn; "
            "then the signed Approve is refused; origin not allowed",
            {
                "withdrawal": {"error": withdrawal["error"], "details": details},
                "ask": _ask_status(session, str(expire_ask["ask_id"])),
                "verdict": late,
                "allowed": allowed_late,
            },
            bool(late.get("connected")) and (not late.get("ok")) and not allowed_late,
            blocked=not gone,
        )
    results.row(
        "no native /etc anchor read in the node process",
        "0",
        str(native_reads["native_anchor_reads"]),
        native_reads["native_anchor_reads"] == 0,
    )
    return 0 if all(r["verdict"] == "PASS" for r in results.data["rows"]) else 1


# ---------------------------------------------------------------------------
# Operator role (a separate process, its own config root holding the key)
# ---------------------------------------------------------------------------


def _emit(payload: dict[str, Any]) -> None:
    """One operator verdict line, stamped as having reached the node runtime."""
    print(json.dumps({"connected": True, **payload}, default=str), flush=True)


async def operator_role(args: argparse.Namespace) -> int:
    """Answer ONE card the way the desktop ``/answers`` route does.

    ``_verified_decision_target`` → ``bridge_decision_target`` (checks the site
    card and notes it verified) → ``bridge.remote.bridge_decide`` (the attach
    client asks the node runtime for a challenge and signs it with THIS root's
    operator key). ``--mode plain`` sends the ordinary ``ask_respond`` instead,
    which must never count as the decision.
    """
    key_root = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
    from tests.fixtures.anchor_fixture import install_fixture_anchor

    install_fixture_anchor(key_root)
    from local_operator.asks.decision import bridge_decision_target
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.session.runtime import registry

    ask = json.loads(Path(args.ask_file).read_text())
    question = ask["questions"][0]
    slot = str(question.get("id") or "")
    answers = {slot: [args.word]}
    records = [r for r, s in registry.scan(Path(args.node_root), reap=False) if s == "live"]
    if not records:
        print(json.dumps({"ok": False, "error": "no live node runtime"}))
        return 1
    client = AttachClient(lambda _p: None, lambda _r: None)
    await client.connect(records[0], args.session)
    # Every verdict from here on reached the runtime: a crash or a refused
    # connect is never mistaken for the door refusing the answer.
    try:
        if args.mode == "plain":
            try:
                detail = await client.ask_respond(ask["ask_id"], answers, by="desktop")
                _emit({"ok": True, "detail": str(detail)})
            except Exception as error:  # noqa: BLE001 — the refusal IS the result
                _emit({"ok": False, "error": f"{type(error).__name__}: {error}"})
            return 0
        row = SimpleNamespace(ask_id=ask["ask_id"], questions=ask["questions"])
        target = bridge_decision_target(row, answers)
        if target is None:
            _emit({"ok": False, "error": "no decision target"})
            return 0
        try:
            verdict = await client.bridge_decide(*target)
        except Exception as error:  # noqa: BLE001
            _emit({"ok": False, "error": f"{type(error).__name__}: {error}"})
            return 0
        if isinstance(verdict, dict):
            _emit({"ok": bool(verdict.get("ok", True)), **verdict})
        else:
            _emit({"ok": True, "detail": str(verdict)})
        return 0
    finally:
        with contextlib.suppress(Exception):
            await client.close()


# ---------------------------------------------------------------------------
# Security control
# ---------------------------------------------------------------------------


def security_control(args: argparse.Namespace) -> int:
    """A NORMAL launch with the same isolated HOME never trusts the synthetic key.

    Runs ``lop``'s own anchor reader in a fresh interpreter WITHOUT the fixture:
    it must report the host's real anchor (or none) and never the synthetic key
    id the harness minted. Also asserts no product module mentions the seam.
    """
    root = Path(args.root)
    report = json.loads((root / "results.json").read_text())
    synthetic = report["facts"]["synthetic_operator_key_id"]
    node_root = root / "node"
    env = _child_env(LOCAL_OPERATOR_CONFIG_DIR=str(node_root), HOME=str(node_root / "home"))
    probe = (
        "from local_operator.operator.trust import load_anchor;"
        "l=load_anchor();print(l.anchor.key_id if l.anchor else 'none')"
    )
    seen = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=60
    ).stdout.strip()
    grep = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            r"install_fixture_anchor\|_ANCHOR_ROOT_OVERRIDE =",
            "--",
            "local_operator/",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    ).stdout.strip()
    rows = [
        {
            "row": "security control: normal launch, same isolated HOME, no fixture",
            "expected": f"real anchor or none; never {synthetic[:8]}…",
            "actual": f"load_anchor() key id = {seen[:12]}…" if seen != "none" else "none",
            "verdict": "PASS" if seen and seen != synthetic else "FAIL",
        },
        {
            "row": "security control: no product module references the seam",
            "expected": "git grep empty",
            "actual": grep or "(empty)",
            "verdict": "PASS" if not grep else "FAIL",
        },
    ]
    report["rows"].extend(rows)
    (root / "results.json").write_text(json.dumps(report, indent=2))
    (root / "results.md").write_text(render_table(report["rows"]))
    print(render_table(rows))
    return 0 if all(row["verdict"] == "PASS" for row in rows) else 1


# ---------------------------------------------------------------------------


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness", action="store_true", help="required: confirm the run")
    parser.add_argument("--root", default="")
    parser.add_argument("--storybook-dir", default=str(DEFAULT_STORYBOOK))
    parser.add_argument("--security-control", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="stop after the rig is up (namespace, daemon, extension paired)",
    )
    parser.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--operator", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--node-root", default="", help=argparse.SUPPRESS)
    parser.add_argument("--session", default="", help=argparse.SUPPRESS)
    parser.add_argument("--ask-file", default="", help=argparse.SUPPRESS)
    parser.add_argument("--word", default="approve", help=argparse.SUPPRESS)
    parser.add_argument("--mode", default="signed", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    scratchpad = os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")
    if not args.root and scratchpad:
        args.root = str(Path(scratchpad) / f"node-browser-harness-{time.strftime('%Y%m%dT%H%M%S')}")
    real_config = Path(os.path.expanduser("~")) / ".local-operator"
    if args.operator:
        # Operator children run with their own config root as HOME/config; the
        # scratchpad guard was applied by the launcher that spawned them.
        return asyncio.run(operator_role(args))
    reason = refusal(
        harness_flag=args.harness,
        root=Path(args.root or "."),
        scratchpad=scratchpad,
        real_config=Path(os.environ.get("LOCAL_OPERATOR_HARNESS_REAL_CONFIG", real_config)),
    )
    if reason and not args.inner:
        print(reason, file=sys.stderr)
        return 2
    if args.inner:
        return asyncio.run(inner(args))
    if args.security_control:
        return security_control(args)
    return outer(args)


if __name__ == "__main__":
    sys.exit(main())
