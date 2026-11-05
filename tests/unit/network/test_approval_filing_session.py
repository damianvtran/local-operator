"""Approval requests filed from a tool child name the session that filed them.

``LOCAL_OPERATOR_SESSION_ID`` (``secrets.access.SESSION_ID_ENV``) is AUDIT-ONLY:
it attributes, it never authorizes. Before this, a `lop network approvals
request` run from the bash/eval tool filed ``requested_by.session_id == ""``, so
the operator's list could not say which conversation asked, and a later
`withdraw` from that same session needed a flag the model had no source for.

Real paths: the bash tool's actual child env -> this worktree's own `lop` ->
the approvals store -> the desktop list rows (``approval_rows``); the finite
share factory -> ``create_request`` digest -> ``exact_shape`` -> a real
file-only operator signature. Synthetic ids, isolated config, no keychain, no
live approval, no network.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.network import approvals as A
from local_operator.network import cli as net_cli
from local_operator.network.credentials import consent, finite_share
from local_operator.secrets.access import SESSION_ID_ENV
from local_operator.server.utils.desktop_approvals import approval_rows
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools
from tests.unit.network.test_approvals_cli import REQUEST_ARGS
from tests.unit.network.test_approvals_cli import root as approvals_root  # noqa: F401
from tests.unit.network.test_approvals_store import _make_key
from tests.unit.network.test_credentials_consent import N, P
from tests.unit.network.test_credentials_consent import rig as consent_rig  # noqa: F401
from tests.unit.network.test_finite_share import _signer

SID = "a1b2c3d4e5f6"
NESTED = "0f0e0d0c0b0a"
_LOP = str(Path(sys.executable).parent / "lop")

root = approvals_root
rig = consent_rig


def _env_probe() -> str:
    source = (
        f"import os; v = os.environ.get({SESSION_ID_ENV!r}); "
        "print('<ABSENT>' if v is None else repr(v))"
    )
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"


async def _bash(context: ToolContext, command: str) -> str:
    tools = {tool.name: tool for tool in create_tools(context)}
    result = await tools["bash"].execute("p1", {"command": command}, None, None, context)
    assert not result.is_error, result.text
    return result.text


def _stdout(text: str) -> str:
    return text.split("--- stdout ---", 1)[1].split("--- stderr ---", 1)[0].strip()


# ---------------------------------------------------------------------------
# The child environment: three arms, no store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_bash_child_carries_the_context_session_in_three_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    # SET from the context; a value the harness inherited loses to it (nested wins).
    monkeypatch.setenv(SESSION_ID_ENV, "stale-parent")
    assert _stdout(await _bash(ToolContext(cwd=str(ws), session_id=NESTED), _env_probe())) == repr(
        NESTED
    )
    # CLEARED: inherited, but this context has no session -> the child must not
    # file under its parent's id.
    assert _stdout(await _bash(ToolContext(cwd=str(ws)), _env_probe())) == repr("")
    # OMITTED: nothing held, nothing inherited.
    monkeypatch.delenv(SESSION_ID_ENV, raising=False)
    assert _stdout(await _bash(ToolContext(cwd=str(ws)), _env_probe())) == "<ABSENT>"


def test_building_the_env_reads_no_store_or_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.secrets import access, keys

    def forbidden(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("an env injection opened a secret store or keychain")

    monkeypatch.setattr(access, "resolve_master_key", forbidden)
    monkeypatch.setattr(keys, "load_master_key", forbidden)
    monkeypatch.setattr(access, "SecretStore", forbidden)
    monkeypatch.delenv(SESSION_ID_ENV, raising=False)
    assert builtin.session_id_env_injection(SID) == {SESSION_ID_ENV: SID}
    assert builtin.session_id_env_injection("") == {}
    assert builtin.session_id_env_injection(None) == {}


def test_the_eval_kernel_exports_the_real_session_id_never_its_registry_key() -> None:
    from local_operator.tools import eval as eval_tool

    assert eval_tool._session_id_of(ToolContext(cwd=".", session_id=SID)) == SID
    # ``_session_key`` falls back to ``ctx-<id>`` to name a KERNEL; that is not a
    # session and must not be exported as one.
    bare = ToolContext(cwd=".")
    assert eval_tool._session_key(bare).startswith("ctx-")
    assert eval_tool._session_id_of(bare) == ""
    assert eval_tool._session_id_of(None) == ""


# ---------------------------------------------------------------------------
# Tool child -> real `lop` -> request record -> desktop list rows
# ---------------------------------------------------------------------------


def _request_cmd() -> str:
    return (
        f"{shlex.quote(_LOP)} network approvals request --host 127.0.0.2 --user probe "
        "--name probe-node --host-key-fp SHA256:abc --json"
    )


@pytest.mark.asyncio
async def test_a_tool_child_files_its_request_under_its_session(
    root: Path, tmp_path: Path  # noqa: F811
) -> None:
    _make_key(root)
    ws = tmp_path / "ws"
    ws.mkdir()
    context = ToolContext(cwd=str(ws), session_id=SID)
    payload = json.loads(_stdout(await _bash(context, _request_cmd())))
    row = next(r for r in approval_rows(root) if r["approval_id"] == payload["approval_id"])
    assert row["requested_by"]["session_id"] == SID
    assert row["requested_by"]["surface"] == "cli"

    # The SAME session withdraws through the same env; nothing passes a flag.
    withdrawn = json.loads(
        _stdout(
            await _bash(
                context,
                f"{shlex.quote(_LOP)} network approvals withdraw {payload['approval_id']} --json",
            )
        )
    )
    assert withdrawn["state"] == "withdrawn"
    row = next(r for r in approval_rows(root) if r["approval_id"] == payload["approval_id"])
    assert row["withdrawn_by"]["session_id"] == SID


@pytest.mark.asyncio
async def test_another_session_cannot_withdraw_it(root: Path, tmp_path: Path) -> None:  # noqa: F811
    _make_key(root)
    ws = tmp_path / "ws"
    ws.mkdir()
    payload = json.loads(
        _stdout(await _bash(ToolContext(cwd=str(ws), session_id=SID), _request_cmd()))
    )
    other = ToolContext(cwd=str(ws), session_id=NESTED)
    tools = {tool.name: tool for tool in create_tools(other)}
    result = await tools["bash"].execute(
        "p1",
        {
            "command": f"{shlex.quote(_LOP)} network approvals withdraw "
            f"{payload['approval_id']} --json"
        },
        None,
        None,
        other,
    )
    refused = json.loads(_stdout(result.text))
    assert refused["code"] == "approval_requester_mismatch"
    assert A.load_record(payload["approval_id"], root=root)["state"] == "requested"


def _cli(command: str, **fields: Any) -> dict[str, Any]:
    return dict(network_command="approvals", approvals_command=command, **fields)


def test_an_explicit_flag_wins_and_no_context_stays_legacy(
    root: Path,  # noqa: F811
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_key(root)
    monkeypatch.setenv(SESSION_ID_ENV, SID)
    assert net_cli.main(Namespace(**_cli("request", **{**REQUEST_ARGS, "session": NESTED}))) == 0
    flagged = json.loads(capsys.readouterr().out)["approval_id"]
    monkeypatch.delenv(SESSION_ID_ENV)
    assert net_cli.main(Namespace(**_cli("request", **{**REQUEST_ARGS, "session": ""}))) == 0
    legacy = json.loads(capsys.readouterr().out)["approval_id"]
    rows = {r["approval_id"]: r["requested_by"]["session_id"] for r in approval_rows(root)}
    assert rows[flagged] == NESTED
    assert rows[legacy] == ""


# ---------------------------------------------------------------------------
# Finite share: attribution is stamped BEFORE the digest and is signed
# ---------------------------------------------------------------------------


def _finite(rig: Any, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> dict[str, Any]:  # noqa: F811
    _signer(rig, monkeypatch)
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "p1", rig.root
    )
    return finite_share.request(
        N, rig.own.device_id, rig.peer.device_id, P, "session", rig.root, **kw
    )


def test_a_finite_card_binds_its_filing_session_into_the_signed_digest(
    rig: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    stamped = _finite(rig, monkeypatch, filed_by_session=SID)
    assert stamped["requested_by"]["session_id"] == SID
    assert finite_share.exact_shape(stamped)
    assert finite_share.disclosure(stamped) is not None
    assert stamped["request_digest"] == A.request_digest(stamped)
    # Rewriting the attribution after filing is a different digest: what the
    # operator signs names who asked, and history cannot be re-attributed.
    forged = {**stamped, "requested_by": {**stamped["requested_by"], "session_id": NESTED}}
    assert A.request_digest(forged) != stamped["request_digest"]


def test_an_unattributed_legacy_finite_card_still_verifies(
    rig: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    legacy = _finite(rig, monkeypatch)
    assert legacy["requested_by"]["session_id"] == ""
    assert finite_share.exact_shape(legacy)
    assert finite_share.disclosure(legacy) is not None


@pytest.mark.parametrize("value", ["not-a-session", "A1B2C3D4E5F6", "a1b2c3d4e5f6\n", "x" * 400])
def test_a_finite_card_never_carries_free_bytes_as_attribution(
    rig: Any, monkeypatch: pytest.MonkeyPatch, value: str  # noqa: F811
) -> None:
    # The factory files anything that is not a session id as unattributed...
    filed = _finite(rig, monkeypatch, filed_by_session=value)
    assert filed["requested_by"]["session_id"] == ""
    # ...and a record carrying one anyway is not a shape this build will disclose.
    smuggled = {**filed, "requested_by": {**filed["requested_by"], "session_id": value}}
    assert not finite_share.exact_shape(smuggled)


def test_the_env_names_no_authority(
    rig: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """A forged ``LOCAL_OPERATOR_SESSION_ID`` changes attribution and nothing else:
    the finite card still needs the operator's real signature to run."""
    monkeypatch.setenv(SESSION_ID_ENV, SID)
    filed = _finite(rig, monkeypatch, filed_by_session=os.environ[SESSION_ID_ENV])
    with pytest.raises(Exception):  # noqa: B017 — any refusal; the point is "not applied"
        finite_share.apply(filed["approval_id"], {}, rig.root)
    assert A.load_record(filed["approval_id"], root=rig.root)["state"] == "requested"
