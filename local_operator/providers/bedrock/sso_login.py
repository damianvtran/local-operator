"""Run ``aws sso login`` / ``aws login`` for ONE profile, on explicit user action.

Design §5.4. The AWS CLI owns the IAM Identity Center (and ``aws login``)
device/PKCE dance and writes the token caches botocore reads; re-implementing
that OIDC client here would duplicate a security-sensitive flow AWS already
ships. So this module only *drives* the CLI, under hard rules:

* EXPLICIT ACTION ONLY. Called from the TUI/CLI login chooser after the user
  picked "Sign in now", or from ``POST /v1/auth/providers/bedrock/sso-login``.
  Nothing calls it on an expiry, a census, or a turn: an expired session is
  reported with its command, never auto-refreshed by launching a browser.
* Never in ``lop exec`` / the SDK: those never run a login flow at all (they
  get no ``LoginCallbacks``), and :func:`refusal` additionally refuses when no
  person can reach a browser (``headless_display``) or ``aws`` is missing.
* ``stdin=DEVNULL`` so the CLI can never block on a prompt nobody sees; a new
  process group (``start_new_session``) so a cancel or the timeout kills the
  CLI AND anything it spawned, by exact group id — never ``pkill``.
* ``--no-browser --use-device-code`` for SSO: the CLI prints the device
  verification URL and the user code instead of opening a browser, so the
  HOST decides how to show them (the desktop renderer opens the URL itself;
  opening it here too would open the browser twice — contract delta 3). AWS
  CLI < 2.22 has no ``--use-device-code`` (device code was its only flow), so
  an "Unknown options" refusal retries once without it.
* ``aws login`` has no ``--no-browser``; its ``--remote`` mode wants a pasted
  code on stdin, which ``DEVNULL`` forbids. So ``aws login`` runs in its normal
  local-callback mode and opens the browser ITSELF; the URL it prints is
  reported as progress text only, never through ``on_auth_url``, so no host
  opens a second tab.

Output is scanned only for the URL and the code; the CLI never prints tokens
(they go straight to ``~/.aws/sso/cache``), and nothing else from its output
is surfaced beyond a short, bounded tail used for the debug log on failure.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from local_operator.procstate import terminate_process_tree
from local_operator.providers.bedrock import profiles as bprofiles

if TYPE_CHECKING:
    from local_operator.harness.types import AbortSignal
    from local_operator.providers.oauth.callback_server import LoginCallbacks

logger = logging.getLogger(__name__)

#: Hard ceiling on one sign-in. An IAM Identity Center device code lives 600 s
#: (AWS OIDC ``StartDeviceAuthorization`` returns ``expiresIn: 600``), so a
#: sign-in still running after that can no longer succeed.
SSO_LOGIN_TIMEOUT_S = 600.0
#: Grace between SIGTERM and SIGKILL for the CLI's process group.
_KILL_GRACE_S = 2.0

_URL_RE = re.compile(r"https://[^\s\"'<>]+")
#: IAM Identity Center user codes are two groups of four ("ABCD-EFGH").
_CODE_RE = re.compile(r"^\s*([A-Z0-9]{4}-[A-Z0-9]{4})\s*$")
_CODE_QUERY_RE = re.compile(r"[?&]user_code=([A-Z0-9]{4}-[A-Z0-9]{4})")

Refusal = Literal["aws_cli_missing", "headless", "not_sso"]


class SsoLoginRefused(Exception):
    """The sign-in may not run here; ``code`` says why, ``command`` what to run."""

    def __init__(self, code: Refusal, command: str | None) -> None:
        super().__init__(code)
        self.code: Refusal = code
        self.command = command


class SsoLoginFailed(Exception):
    """The CLI ran and exited non-zero (or timed out)."""

    def __init__(self, returncode: int | None, *, timed_out: bool = False) -> None:
        super().__init__(f"aws exited {returncode}")
        self.returncode = returncode
        self.timed_out = timed_out


@dataclass(frozen=True)
class SsoLoginResult:
    profile: str
    session: bprofiles.ProfileSession


def refusal(
    kind: bprofiles.ProfileKind,
    *,
    environ: Mapping[str, str] | None = None,
    aws_path: str | None = None,
) -> Refusal | None:
    """Why a sign-in may not start here, or ``None`` when it may."""
    from local_operator.providers.login_catalog import headless_display

    if kind not in ("sso", "login"):
        return "not_sso"
    if aws_path is None and bprofiles.aws_cli_path() is None:
        return "aws_cli_missing"
    if headless_display(dict(environ) if environ is not None else None):
        return "headless"
    return None


def _argv(aws: str, profile: str, kind: bprofiles.ProfileKind, *, device_code: bool) -> list[str]:
    if kind == "login":
        return [aws, "login", "--profile", profile]
    argv = [aws, "sso", "login", "--profile", profile, "--no-browser"]
    if device_code:
        argv.append("--use-device-code")
    return argv


def _kill_group(proc: asyncio.subprocess.Process, *, force: bool = False) -> None:
    """Stop the CLI and everything it spawned (it leads its group: start_new_session).

    ``procstate.terminate_process_tree`` owns the platform split — a
    leader-checked ``killpg`` on POSIX, ``taskkill /T`` on Windows — so this
    module never touches the POSIX-only ``os.killpg``/``signal.SIGKILL``
    attributes that do not exist on Windows (xplat probe
    ``static.posix_attributes``). Never raises.
    """
    if proc.returncode is not None:
        return
    terminate_process_tree(proc.pid, force=force)


async def _reap(proc: asyncio.subprocess.Process) -> None:
    _kill_group(proc)
    try:
        await asyncio.wait_for(proc.wait(), _KILL_GRACE_S)
    except (TimeoutError, asyncio.TimeoutError):
        _kill_group(proc, force=True)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), _KILL_GRACE_S)


async def _report(hook, *args) -> None:  # noqa: ANN001 - LoginCallbacks hook
    from local_operator.providers.oauth.callback_server import report_safely

    await report_safely(hook, *args)


async def _run_once(
    argv: list[str],
    *,
    profile: str,
    kind: bprofiles.ProfileKind,
    callbacks: "LoginCallbacks",
    deadline: float,
    abort: "AbortSignal | None",
) -> tuple[int | None, list[str]]:
    from local_operator.i18n.keys import wire_login_bedrock as k
    from local_operator.i18n.messages import render_resolved
    from local_operator.providers.oauth.callback_server import report_flow_details

    env = dict(os.environ)
    # No pager, no colour codes in what we scan.
    env["AWS_PAGER"] = ""
    env["NO_COLOR"] = "1"
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    tail: deque[str] = deque(maxlen=12)
    url: str | None = None
    code: str | None = None
    announced = False

    async def pump() -> None:
        nonlocal url, code, announced
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                return
            line = raw.decode("utf-8", "replace").rstrip()
            tail.append(line)
            if url is None:
                match = _URL_RE.search(line)
                if match:
                    url = match.group(0).rstrip(".,)")
                    query_code = _CODE_QUERY_RE.search(url)
                    if query_code:
                        code = query_code.group(1)
            if code is None:
                match = _CODE_RE.match(line)
                if match:
                    code = match.group(1)
            if announced or url is None:
                continue
            # SSO prints the URL, then the code a few lines later; wait for the
            # code before announcing so the host gets both in one report.
            if kind == "sso" and code is None:
                continue
            announced = True
            if kind == "sso":
                await report_flow_details(callbacks, user_code=code)
                if callbacks.on_auth_url is not None:
                    text = render_resolved(k.sso_enter_code(url=url, code=code or ""))
                    await _report(callbacks.on_auth_url, url, text)
            else:
                # ``aws login`` opened the browser itself: text only (see module doc).
                await _report(callbacks.on_progress, render_resolved(k.sso_browser_opened(url=url)))
            await _report(callbacks.on_progress, render_resolved(k.sso_waiting(profile=profile)))

    pump_task = asyncio.create_task(pump())
    waiters: set[asyncio.Task[None]] = {pump_task}
    abort_task = asyncio.create_task(abort.wait()) if abort is not None else None
    if abort_task is not None:
        waiters.add(abort_task)
    loop = asyncio.get_running_loop()
    try:
        remaining = deadline - loop.time()
        done, _pending = await asyncio.wait(
            waiters, timeout=max(0.0, remaining), return_when=asyncio.FIRST_COMPLETED
        )
        if abort_task is not None and abort_task in done:
            raise asyncio.CancelledError
        if pump_task not in done:
            raise TimeoutError
        await asyncio.wait_for(proc.wait(), max(0.1, deadline - loop.time()))
        return proc.returncode, list(tail)
    finally:
        if abort_task is not None:
            abort_task.cancel()
        pump_task.cancel()
        await _reap(proc)


async def run_sso_login(
    profile: str,
    *,
    kind: bprofiles.ProfileKind,
    callbacks: "LoginCallbacks",
    abort: "AbortSignal | None" = None,
    timeout: float = SSO_LOGIN_TIMEOUT_S,
    aws_path: str | None = None,
) -> SsoLoginResult:
    """Run the sign-in; return the profile's session state as re-read from cache.

    Raises :class:`SsoLoginRefused` before spawning anything, :class:`SsoLoginFailed`
    for a non-zero exit or the timeout, and ``CancelledError`` on cancel/abort —
    in every case the CLI's process group is gone before this returns.
    """
    command = bprofiles.login_command(profile, kind)
    reason = refusal(kind, aws_path=aws_path)
    if reason is not None:
        raise SsoLoginRefused(reason, command)
    aws = aws_path or bprofiles.aws_cli_path()
    if aws is None or not bprofiles.valid_profile_input(profile):
        raise SsoLoginRefused("aws_cli_missing" if aws is None else "not_sso", command)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    device_code = kind == "sso"
    try:
        returncode, tail = await _run_once(
            _argv(aws, profile, kind, device_code=device_code),
            profile=profile,
            kind=kind,
            callbacks=callbacks,
            deadline=deadline,
            abort=abort,
        )
        if returncode and device_code and any("--use-device-code" in line for line in tail):
            # AWS CLI < 2.22: no such flag, and device code is already its flow.
            returncode, tail = await _run_once(
                _argv(aws, profile, kind, device_code=False),
                profile=profile,
                kind=kind,
                callbacks=callbacks,
                deadline=deadline,
                abort=abort,
            )
    except TimeoutError:
        raise SsoLoginFailed(None, timed_out=True) from None
    if returncode != 0:
        logger.debug("aws sign-in for a Bedrock profile exited %s", returncode)
        raise SsoLoginFailed(returncode)
    return SsoLoginResult(profile, bprofiles.sso_status(profile))
