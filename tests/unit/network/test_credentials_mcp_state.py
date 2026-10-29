"""The durable per-key auth evidence (``mcp_auth.json``).

WHY THIS FILE EXISTS, and what it refuses to become. The evidence document is the
answer to "why is this server not working" that survives a restart and a TTL — the
two things the refusal cache (``placement.state.json``) deliberately does not do.
These cells pin the properties a security reviewer asks about, not the JSON:

* a state outside the closed vocabulary is never written;
* no credential material can reach the file (the writer applies the same
  ``_assert_no_material`` guard as the placement document);
* the file is 0600, and a write that cannot happen never raises into a borrow;
* a device's evidence is forgotten by device, not by key prefix guessing.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from local_operator.network.credentials import mcp_state

NETWORK = "n_" + "a" * 24
KEY = "mcp:" + "https://example.test/mcp"
SECRET_KEY = "secret:" + "EXAMPLE_TOKEN"
OWNER = "d_" + "1" * 32
OTHER = "d_" + "2" * 32


def test_record_load_and_clear_round_trip(root: Path) -> None:
    mcp_state.record(
        NETWORK,
        KEY,
        state="expired",
        code="refresh_failed",
        owner_device=OWNER,
        owner_name="macbook",
        root=root,
    )
    entries = mcp_state.load(NETWORK, root)
    assert entries[KEY]["state"] == "expired"
    assert entries[KEY]["code"] == "refresh_failed"
    assert entries[KEY]["owner_device"] == OWNER
    assert entries[KEY]["owner_name"] == "macbook"
    assert entries[KEY]["at"] > 0

    # A later success REPLACES the failure rather than leaving both.
    mcp_state.record(NETWORK, KEY, state="synced", root=root)
    assert mcp_state.load(NETWORK, root)[KEY]["state"] == "synced"

    mcp_state.clear(NETWORK, KEY, root=root)
    assert mcp_state.load(NETWORK, root) == {}


def test_an_unknown_state_is_never_written(root: Path) -> None:
    mcp_state.record(NETWORK, KEY, state="signed-in-maybe", root=root)
    assert mcp_state.load(NETWORK, root) == {}


def test_the_file_is_private_and_has_no_material(root: Path) -> None:
    mcp_state.record(
        NETWORK,
        SECRET_KEY,
        state="revoked",
        code="not_a_holder",
        owner_device=OWNER,
        root=root,
    )
    path = mcp_state.mcp_auth_path(NETWORK, root)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert mcp_state.file_is_private(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rendered = json.dumps(payload)
    assert "access_token" not in rendered and "refresh_token" not in rendered
    # The schema itself is pinned: a reader must be able to tell which document
    # it opened, and a rename is a deliberate migration, not an accident.
    assert payload["schema"] == mcp_state.MCP_AUTH_SCHEMA
    assert payload["network_id"] == NETWORK


def test_the_writer_refuses_material(root: Path) -> None:
    """The guard is structural, not a promise in a docstring.

    ``record`` swallows every failure (a borrow must never fail over a diary), so
    the refusal is asserted against the private writer the public functions share.
    """
    with pytest.raises(ValueError):
        mcp_state._write(NETWORK, {KEY: {"state": "synced", "access_token": "oops"}}, root)


def test_forget_device_drops_only_that_devices_rows(root: Path) -> None:
    mcp_state.record(NETWORK, KEY, state="expired", owner_device=OWNER, root=root)
    mcp_state.record(NETWORK, SECRET_KEY, state="unavailable", owner_device=OTHER, root=root)
    mcp_state.forget_device(NETWORK, OWNER, root=root)
    entries = mcp_state.load(NETWORK, root)
    assert KEY not in entries
    assert entries[SECRET_KEY]["owner_device"] == OTHER


def test_a_write_that_cannot_land_never_raises(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only config dir")

    monkeypatch.setattr(mcp_state, "_write", refuse)
    mcp_state.record(NETWORK, KEY, state="expired", root=root)  # must not raise
    assert mcp_state.load(NETWORK, root) == {}


def test_an_unreadable_file_reads_as_no_evidence(root: Path) -> None:
    path = mcp_state.mcp_auth_path(NETWORK, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert mcp_state.load(NETWORK, root) == {}


def test_state_for_code_is_the_one_mapping() -> None:
    assert mcp_state.state_for_code("interactive_required") == "expired"
    assert mcp_state.state_for_code("refresh_failed") == "expired"
    assert mcp_state.state_for_code("not_a_holder") == "revoked"
    assert mcp_state.state_for_code("member_not_active") == "revoked"
    assert mcp_state.state_for_code("owner_offline") == "unavailable"
    assert mcp_state.state_for_code("internal") == ""
