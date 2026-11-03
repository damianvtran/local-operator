"""The typed CLI reaches the sole refresh owner over both real broker legs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import store
from local_operator.providers.auth_store import AuthStore
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    _pull,
    _share,
    devices,
    idp,
    mesh,
    stub_provider,
)


def _refresh(rig: Any, home: Path) -> tuple[int, dict[str, Any]]:
    # The CLI must have the borrower's ambient root, while the two in-process
    # relays keep the owner's root for the legacy AuthStore construction seam.
    # A separate product process tests that boundary without replacing any hop.
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("CMUX_", "LOP_")) and name != "XPC_FLAGS"
    }
    env.update(
        HOME=str(home),
        LOCAL_OPERATOR_CONFIG_DIR=str(rig.b.root),
        GIT_CONFIG_SYSTEM="/dev/null",
        GIT_TERMINAL_PROMPT="0",
        GIT_ASKPASS="/bin/false",
    )
    command = [
        sys.executable,
        "-m",
        "local_operator.cli",
        "network",
        "credential",
        "refresh",
        STUB_PROVIDER,
        "--json",
    ]
    result = subprocess.run(command, env=env, cwd=home, capture_output=True, text=True, timeout=85)
    assert "access-" not in result.stdout + result.stderr
    assert "refresh-" not in result.stdout + result.stderr
    assert result.stdout.strip(), (result.returncode, result.stderr)
    body = json.loads(result.stdout)
    print(
        f"CLI network credential refresh {STUB_PROVIDER} --json: rc={result.returncode} {body}"
    )
    return result.returncode, body


def test_explicit_cli_refresh_keeps_owner_gate_and_rotates_once(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    rig = request.getfixturevalue("mesh")
    _share(rig)
    _pull(rig)
    # The existing fixture's login is expired and the borrower has DRIVE, not
    # ADMIN. Refusal must happen before any provider exchange or row rotation.
    rc, body = _refresh(rig, tmp_path)
    assert rc == 1 and body["code"] == "not_authorised"
    assert len(rig.idp.posts) == 0
    print("non-admin side effect: synthetic IdP posts=0")

    # Change only the synthetic receiver-side membership. The owner checks this
    # record on every ask, independently of CLI help or readiness guidance.
    with store.mutate(rig.network_id, rig.a.root) as record:
        borrower = record.member(rig.borrower)
        assert borrower is not None
        borrower.role = "admin"
        store.save(record, rig.a.root)
    rc, body = _refresh(rig, tmp_path)
    assert rc == 0 and body == {"ok": True, "credential_name": STUB_PROVIDER}
    assert len(rig.idp.posts) == 1
    auth = AuthStore.open_read_only(rig.a.root / "auth.db")
    try:
        rows = auth.list_credentials(STUB_PROVIDER)
        assert len(rows) == 1
        assert rows[0].data["refresh"] == rig.idp.current_refresh
        assert rows[0].data["access"] == "access-1"
    finally:
        auth.close()
    print("admin side effect: synthetic IdP posts=1; owner auth.db rotation verified")

    # This stub reports a 400 through a generic AuthStoreError, so the shipped
    # owner classifies it as refresh_failed (not grant_invalid). The CLI must
    # preserve that actual classification, not reinterpret the IdP response.
    rig.idp.invalid = True
    rc, body = _refresh(rig, tmp_path)
    assert rc == 1 and body["code"] == "refresh_failed"
    assert len(rig.idp.posts) == 2
    print("failed refresh side effect: synthetic IdP posts=2; no successful grant")
