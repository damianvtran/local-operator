"""``aws sso login`` driver: refusals, URL/code capture, timeout and cancel reaping.

A fake ``aws`` executable stands in for the CLI (never the real one): it
prints what AWS CLI v2 prints for ``--no-browser --use-device-code`` and,
in the hang modes, spawns a grandchild so the tests prove the WHOLE process
group is killed, not just the direct child.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

from local_operator.harness.types import AbortSignal
from local_operator.providers.bedrock import sso_login
from local_operator.providers.oauth.callback_server import LoginCallbacks
from tests.unit.providers.bedrock.awsfixtures import isolate_aws, write_sso_token

pytestmark = pytest.mark.asyncio

FAKE_AWS = r"""#!{python}
import os, subprocess, sys, time
args = sys.argv[1:]
mode = os.environ.get("FAKE_AWS_MODE", "ok")
with open(os.environ["FAKE_AWS_LOG"], "a") as log:
    log.write(" ".join(args) + "\n")
if mode == "old" and "--use-device-code" in args:
    print("Unknown options: --use-device-code", flush=True)
    sys.exit(252)
print("Browser will not be automatically opened.", flush=True)
print("Please visit the following URL:", flush=True)
print("", flush=True)
print("https://device.sso.us-east-1.amazonaws.com/", flush=True)
print("", flush=True)
print("Then enter the code:", flush=True)
print("", flush=True)
print("WXYZ-ABCD", flush=True)
if mode in ("hang", "cancel"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(os.environ["FAKE_AWS_LOG"] + ".child", "w") as f:
        f.write(str(child.pid))
    time.sleep(120)
if mode == "fail":
    sys.exit(255)
if mode in ("ok", "old"):
    import hashlib, json
    from datetime import datetime, timedelta, timezone
    cache = os.path.join(os.environ["HOME"], ".aws", "sso", "cache")
    os.makedirs(cache, exist_ok=True)
    exp = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
    name = hashlib.sha1(b"corp").hexdigest()
    with open(os.path.join(cache, name + ".json"), "w") as out:
        json.dump({{"accessToken": "t", "expiresAt": exp}}, out)
    print("Successfully logged into Start URL: https://corp.awsapps.com/start", flush=True)
"""


@pytest.fixture
def fake_aws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    isolate_aws(tmp_path, monkeypatch)
    script = tmp_path / "bin" / "aws"
    script.parent.mkdir()
    script.write_text(FAKE_AWS.format(python=sys.executable))
    script.chmod(0o755)
    monkeypatch.setenv("FAKE_AWS_LOG", str(tmp_path / "aws.log"))
    monkeypatch.setattr(sso_login.bprofiles, "aws_cli_path", lambda: str(script))
    return script


class Recorder:
    def __init__(self) -> None:
        self.urls: list[tuple[str, str | None]] = []
        self.codes: list[str | None] = []
        self.progress: list[str] = []

    def callbacks(self) -> LoginCallbacks:
        def on_auth_url(url: str, instructions: str | None = None) -> None:
            self.urls.append((url, instructions))

        def on_flow_details(
            *, user_code=None, launch_url=None, expires_in=None
        ) -> None:  # noqa: ANN001
            self.codes.append(user_code)

        return LoginCallbacks(
            on_auth_url=on_auth_url,
            on_progress=self.progress.append,
            on_flow_details=on_flow_details,
        )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_device_url_and_code_reach_the_host_and_state_is_reread(fake_aws: Path) -> None:
    rec = Recorder()
    result = await sso_login.run_sso_login("corp-sso", kind="sso", callbacks=rec.callbacks())
    assert rec.urls[0][0] == "https://device.sso.us-east-1.amazonaws.com/"
    assert "WXYZ-ABCD" in (rec.urls[0][1] or "")
    assert rec.codes == ["WXYZ-ABCD"]
    assert result.session.state == "valid"
    argv = (fake_aws.parent.parent / "aws.log").read_text().split()
    assert argv == ["sso", "login", "--profile", "corp-sso", "--no-browser", "--use-device-code"]


async def test_old_cli_without_device_code_flag_is_retried_once(
    fake_aws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_AWS_MODE", "old")
    result = await sso_login.run_sso_login("corp-sso", kind="sso", callbacks=Recorder().callbacks())
    lines = (fake_aws.parent.parent / "aws.log").read_text().splitlines()
    assert len(lines) == 2 and "--use-device-code" not in lines[1]
    assert result.session.state == "valid"


async def test_nonzero_exit_is_a_failure(fake_aws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_AWS_MODE", "fail")
    with pytest.raises(sso_login.SsoLoginFailed) as raised:
        await sso_login.run_sso_login("corp-sso", kind="sso", callbacks=Recorder().callbacks())
    assert raised.value.returncode == 255 and not raised.value.timed_out


async def test_timeout_kills_the_whole_process_group(
    fake_aws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_AWS_MODE", "hang")
    with pytest.raises(sso_login.SsoLoginFailed) as raised:
        await sso_login.run_sso_login(
            "corp-sso", kind="sso", callbacks=Recorder().callbacks(), timeout=3.0
        )
    assert raised.value.timed_out
    child = int((fake_aws.parent.parent / "aws.log.child").read_text())
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert not _alive(child)


async def test_abort_signal_cancels_and_reaps(
    fake_aws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_AWS_MODE", "cancel")
    signal = AbortSignal()
    rec = Recorder()
    task = asyncio.create_task(
        sso_login.run_sso_login("corp-sso", kind="sso", callbacks=rec.callbacks(), abort=signal)
    )
    log_child = fake_aws.parent.parent / "aws.log.child"
    deadline = time.monotonic() + 10
    while not log_child.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    signal.abort("test")
    with pytest.raises(asyncio.CancelledError):
        await task
    child = int(log_child.read_text())
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert not _alive(child)


async def test_refusals_spawn_nothing(fake_aws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(sso_login.SsoLoginRefused) as raised:
        await sso_login.run_sso_login("dev", kind="static", callbacks=Recorder().callbacks())
    assert raised.value.code == "not_sso"
    monkeypatch.setenv("SSH_CONNECTION", "1 2 3 4")
    with pytest.raises(sso_login.SsoLoginRefused) as raised:
        await sso_login.run_sso_login("corp-sso", kind="sso", callbacks=Recorder().callbacks())
    assert raised.value.code == "headless"
    assert raised.value.command == "aws sso login --profile corp-sso"
    monkeypatch.delenv("SSH_CONNECTION")
    monkeypatch.setattr(sso_login.bprofiles, "aws_cli_path", lambda: None)
    with pytest.raises(sso_login.SsoLoginRefused) as raised:
        await sso_login.run_sso_login("corp-sso", kind="sso", callbacks=Recorder().callbacks())
    assert raised.value.code == "aws_cli_missing"
    assert not (fake_aws.parent.parent / "aws.log").exists()


@pytest.mark.filterwarnings("ignore")
def test_refusal_is_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_aws(tmp_path, monkeypatch)
    write_sso_token(tmp_path, "corp", expires_in=timedelta(hours=1), refresh=False)
    monkeypatch.setattr(sso_login.bprofiles, "aws_cli_path", lambda: "/usr/bin/true")
    assert sso_login.refusal("sso", environ={}) is None
    assert sso_login.refusal("login", environ={"SSH_TTY": "/dev/ttys001"}) == "headless"
