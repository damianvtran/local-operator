"""The actual helper process can explain a denied repo, never authorize it.

Credential stdout/exit status remain the pure helper's protocol. The only new
stderr surface requires matching, untrusted display context from the borrow.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials import github as github_mod
from tests.unit.network.test_credentials_real_link import _parser
from tests.unit.network.test_github_pushable_designation import _config
from tests.unit.network.test_github_pushable_designation import (  # noqa: F401
    env as _env_fixture,
)

# Export the imported pytest fixture under its canonical name.
env = _env_fixture

TOKEN = "synthetic-wrapper-credential-not-a-real-token"
ALLOWED = ["octo-org/app"]
REQUEST = "protocol=https\nhost=github.com\npath=octo-org/denied.git\n\n"
CONTEXT_ENV = "LOCAL_OPERATOR_GITHUB_DIAGNOSTIC_SCOPE"


def _context(token: str = TOKEN, repos: list[str] | None = None) -> str:
    return json.dumps(
        {
            "version": 1,
            "identity": hashlib.sha256(token.encode()).hexdigest(),
            "designation": github_mod.designation_digest(ALLOWED if repos is None else repos),
        }
    )


def _run(
    holder: Any,
    tmp_path: Path,
    request: str = REQUEST,
    *,
    token: str = TOKEN,
    repos: list[str] | None = None,
    context: str | None = None,
    operation: str = "get",
) -> subprocess.CompletedProcess[str]:
    _config(holder.root, repositories=ALLOWED if repos is None else repos)
    guard = tmp_path / "guard"
    guard.mkdir(exist_ok=True)
    proof = tmp_path / "child-proof.json"
    # Actual connect(2) negative control runs in the child before CLI import.
    (guard / "sitecustomize.py").write_text("""import atexit,ipaddress,json,os,pathlib,socket,sys
root=pathlib.Path(os.environ["PROBE_SOURCE_ROOT"])
def audit(event,args):
 if event=="socket.connect" and isinstance(args[1],tuple):
  if not ipaddress.ip_address(args[1][0]).is_loopback:
   raise RuntimeError("non-loopback forbidden")
sys.addaudithook(audit)
with socket.socket() as s:
 try:s.connect(("203.0.113.9",443))
 except RuntimeError:pass
 else:raise AssertionError("negative socket control failed")
@atexit.register
def done():
 from local_operator.network.credentials import github
 assert pathlib.Path(github.__file__).resolve().is_relative_to(root)
 proof={"pid":os.getpid(),"source":str(github.__file__),"guard":True}
 pathlib.Path(os.environ["PROBE_CHILD_PROOF"]).write_text(json.dumps(proof))
""")
    source = Path(__file__).resolve().parents[3]
    child = dict(
        os.environ,
        PYTHONPATH=str(guard) + os.pathsep + str(source),
        PROBE_SOURCE_ROOT=str(source),
        PROBE_CHILD_PROOF=str(proof),
    )
    child.pop("GH_TOKEN", None)
    child.pop("GITHUB_TOKEN", None)
    child.pop(CONTEXT_ENV, None)
    if token:
        child["GH_TOKEN"] = token
    if context is not None:
        child[CONTEXT_ENV] = context
    result = subprocess.run(
        [sys.executable, "-m", "local_operator.cli", "credential", "git-helper", operation],
        input=request,
        env=child,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    loaded = json.loads(proof.read_text())
    assert loaded["guard"] and not Path("/proc", str(loaded["pid"])).exists()
    assert result.returncode == 0
    # Context can affect ONLY human stderr. It never supplies the list/token
    # for credential selection, even if a caller forges every metadata field.
    assert result.stdout == github_mod.git_helper_reply(
        operation,
        request.splitlines(),
        token=token,
        repositories=tuple(ALLOWED if repos is None else repos),
    )
    assert TOKEN not in result.stderr
    assert hashlib.sha256(TOKEN.encode()).hexdigest() not in result.stderr
    if context:
        assert context not in result.stderr
    return result


def test_actual_wrapper_explains_a_valid_repo_denied_by_its_known_borrowed_scope(
    env: Any,
    tmp_path: Path,  # noqa: F811 — imported fixture
) -> None:
    result = _run(env, tmp_path, context=_context())
    assert result.stdout == ""
    assert "octo-org/denied" in result.stderr and "lending device" in result.stderr
    assert "opted out" in result.stderr and "not pushable" in result.stderr
    assert "lop network credential github repos" in result.stderr
    _parser().parse_args(["network", "credential", "github", "repos"])


@pytest.mark.parametrize(
    "case",
    [
        "allowed",
        "missing-token",
        "unknown-context",
        "unknown-list",
        "malformed-context",
        "wrong-token",
        "forged-list",
        "allowed-forged-list",
        "http",
        "other-host",
        "missing-host",
        "bad-path",
        "malformed-line",
        "duplicate-host",
        "store",
        "erase",
        "context-bool",
        "context-float",
        "context-extra",
        "spaced-key",
        "empty-key",
        "fingerprint-path",
        "fingerprint-path-uppercase",
    ],
)
def test_other_helper_outcomes_stay_silent_and_forged_context_cannot_select_credentials(
    env: Any,
    tmp_path: Path,
    case: str,  # noqa: F811
) -> None:
    request, token, repos, context, op = REQUEST, TOKEN, ALLOWED, _context(), "get"
    if case.startswith("allowed"):
        request = REQUEST.replace("denied", "app")
    if case == "missing-token":
        token = ""
    if case == "unknown-context":
        context = None
    if case == "unknown-list":
        repos = []
    if case == "malformed-context":
        context = "not-json"
    if case == "wrong-token":
        context = _context("different-synthetic-bearer")
    if case in {"forged-list", "allowed-forged-list"}:
        context = _context(repos=["octo-org/denied"])
    if case == "http":
        request = REQUEST.replace("https", "http")
    if case == "other-host":
        request = REQUEST.replace("github.com", "github.com.evil.invalid")
    if case == "missing-host":
        request = REQUEST.replace("host=github.com\n", "")
    if case == "bad-path":
        request = REQUEST.replace("octo-org/denied.git", "../credential")
    if case == "malformed-line":
        request = REQUEST + "not-a-field\n"
    if case == "duplicate-host":
        request = REQUEST + "host=github.com\n"
    if case in {"store", "erase"}:
        op = case
    if case.startswith("context-"):
        metadata = json.loads(context or "{}")
        if case == "context-bool":
            metadata["version"] = True
        if case == "context-float":
            metadata["version"] = 1.0
        if case == "context-extra":
            metadata["extra"] = "untrusted"
        context = json.dumps(metadata)
    if case == "spaced-key":
        request = REQUEST.replace("host=", "host =")
    if case == "empty-key":
        request = REQUEST.replace("host=github.com", "=github.com")
    if case.startswith("fingerprint-path"):
        identity = hashlib.sha256(TOKEN.encode()).hexdigest()
        if case.endswith("uppercase"):
            identity = identity.upper()
        request = REQUEST.replace("octo-org/denied.git", "octo-org/" + identity)
    result = _run(env, tmp_path, request, token=token, repos=repos, context=context, operation=op)
    assert result.stderr == ""


# QUARANTINED as the CULPRIT, not a victim: borrowed_git_env arms the daemon
# self-revoke timer, which later fires `DELETE /installation/token` during an
# unrelated test's teardown and trips the network guard there. REMOVE this
# marker when #4261 (test/netguard-self-revoke-belt) merges.
@pytest.mark.skip(
    reason=(
        "QUARANTINED 2026-11-18 flaky-sweep ledger lopdev-E01: arms the daemon "
        "self-revoke timer, whose `DELETE https://api.github.com/installation/token` "
        "trips the network guard in tests/unit/remote_compute/test_cli_remote.py "
        "teardown (#4235 gate jobs 135207196151, 135107436353); owner: #4261 author; "
        "real fix: #4261 test/netguard-self-revoke-belt — remove this skip when it merges"
    )
)
def test_borrow_attaches_only_matching_display_context_without_an_extra_request(
    tmp_path: Path,
) -> None:
    import time
    from types import SimpleNamespace

    from local_operator.network.credentials.types import (
        CredentialRef,
        Grant,
        GrantScope,
    )

    owner = "d_" + "a" * 32
    now = int(time.time() * 1000)
    grant = Grant(
        access_token=TOKEN,
        kind="bearer",
        token_expires_at_ms=now + 600_000,
        grant_expires_at_ms=now + 600_000,
        credential_ref=CredentialRef(
            owner_device=owner,
            owner_device_name="synthetic-owner",
            provider="github",
            kind=github_mod.GITHUB_KIND,
            credential_id=0,
        ),
        served_by=owner,
        scope=GrantScope(kind="device"),
        narrowing=github_mod.grant_narrowing(ALLOWED, generation_ms=now),
    )
    calls: list[str] = []

    def ask(key: str, **kwargs: Any) -> Grant:
        calls.append(key)
        return grant

    client: Any = SimpleNamespace(
        should_borrow=lambda key: True,
        request_grant_sync=ask,
        owner_of=lambda key: owner,
        self_device="d_" + "b" * 32,
        network_id="n_diagnostic",
    )
    env_map, token = github_mod.borrowed_git_env(root=tmp_path, client=client)
    assert calls == ["github"] and token == TOKEN
    assert env_map[CONTEXT_ENV] == github_mod._diagnostic_scope(TOKEN, ALLOWED)
    assert github_mod.repositories_for(tmp_path) == tuple(ALLOWED)


def test_optional_metadata_is_scrubbed_or_dropped_without_changing_the_borrowed_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    monkeypatch.setattr(
        github_mod,
        "borrowed_git_env",
        lambda **kwargs: ({"GH_TOKEN": TOKEN, CONTEXT_ENV: context}, TOKEN),
    )
    registered: list[str] = []
    env_map, token = github_mod.lent_github_env(
        session_id="synthetic-session", register=registered.append
    )
    assert token == TOKEN and env_map[CONTEXT_ENV] == context
    assert registered == [TOKEN, context, hashlib.sha256(TOKEN.encode()).hexdigest()]

    def only_token(value: str) -> None:
        if value != TOKEN:
            raise RuntimeError("synthetic optional-metadata registration refusal")

    env_map, token = github_mod.lent_github_env(session_id="synthetic-session", register=only_token)
    assert token == TOKEN and env_map == {"GH_TOKEN": TOKEN}
