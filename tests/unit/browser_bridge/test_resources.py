from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.browser_bridge import resources
from local_operator.browser_bridge.backend import BridgeUnreachable
from local_operator.browser_bridge.resources import (
    RESOURCE_NAME,
    BrowserResource,
    cleanup_disposition,
    cleanup_exact,
    read_inventory,
)
from local_operator.session_lease import SessionLeaseHeldError, acquire_session_lease


class BridgeFixture:
    """Disposable protocol peer; never discovers the operator's bridge."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_close = False
        self.retained = False
        #: Override the state ``owner_finish`` replies with, so the "extension
        #: could not remove the tab" answer can be exercised as itself rather
        #: than as a transport failure.
        self.finish_state = ""
        #: Raise the real transport error, whose message names the diagnosing
        #: command — the detail the operator is supposed to receive.
        self.unreachable = False

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "owner_recover":
            return {"ownership_version": 1, "state": "owned", "tab": "bridge:100:private"}
        if method == "owner_retain":
            self.retained = True
            return {"state": "retained"}
        if method == "owner_finish":
            if self.unreachable:
                raise BridgeUnreachable(
                    "browser bridge unreachable: no live daemon state. " "Run 'lop browser status'."
                )
            if self.fail_close:
                raise ConnectionError("disposable fixture unavailable")
            if self.finish_state:
                return {"state": self.finish_state}
            return {"state": "retained" if self.retained else "closed"}
        raise AssertionError(method)


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> BridgeFixture:
    fixture = BridgeFixture()
    monkeypatch.setattr(resources, "BridgeClient", lambda: fixture)
    return fixture


def test_private_sidecar_and_redacted_inventory(tmp_path: Path) -> None:
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    resource.remember("bridge:100:secret-capability")
    assert os.stat(resource.path).st_mode & 0o777 == 0o600
    raw = json.dumps(read_inventory(tmp_path))
    assert resource.record["proof"] not in raw
    assert "secret-capability" not in raw
    assert read_inventory(tmp_path)[0]["cleanup_candidate"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "cancelled", "failed", "disposed"])
async def test_terminal_cleanup_is_durable_and_idempotent(
    tmp_path: Path, bridge: BridgeFixture, outcome: str
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("bridge:100:private")
    result = await resource.finish(resource.generation, outcome)
    assert result.state == "closed"
    assert json.loads(resource.path.read_text())["terminal"] == outcome
    assert json.loads(resource.path.read_text())["surface_id"] == ""
    assert (await resource.finish(resource.generation, outcome)).state == "closed"


@pytest.mark.asyncio
async def test_failed_cleanup_survives_reconstruction(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("bridge:100:private")
    bridge.fail_close = True
    assert (await resource.finish(resource.generation, "failed")).state == "pending"
    record = json.loads(resource.path.read_text())
    assert record["terminal"] == "failed"
    assert record["surface_id"] == "bridge:100:private"
    # Reconstruction models a restarted host. The proof, the allocation AND the
    # generation all survive: on an unleased directory identity belongs to the
    # SESSION, not to the BrowserResource object, so a second instance adopts
    # the stored owner instead of minting a token that would fence out an
    # incumbent still holding a live tab (review round 1, B1).
    resumed = BrowserResource(tmp_path, tmp_path.name)
    resumed.initialize()
    assert resumed.params()["owner_proof"] == resource.record["proof"]
    assert resumed.generation == resource.generation
    bridge.fail_close = False
    await resumed.recover()
    assert resumed.record["surface_id"] == "bridge:100:private"
    # The incumbent is NOT fenced out by the reconstruction: same identity.
    resource.remember("bridge:100:private")


@pytest.mark.asyncio
async def test_pending_login_and_paused_retention_protect_terminal_scope(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.record["retention"] = "pending login"
    resource.remember("bridge:100:private")
    assert (await resource.finish(resource.generation, "completed")).state == "retained"
    assert resource.record["surface_id"] == "bridge:100:private"
    assert [call[0] for call in bridge.calls] == ["owner_recover", "owner_retain", "owner_finish"]


def test_execution_lease_fences_even_lazy_old_host(tmp_path: Path) -> None:
    lease = acquire_session_lease(tmp_path)
    old = BrowserResource(tmp_path, tmp_path.name)
    lease.release()
    successor = acquire_session_lease(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="lease changed"):
            old.initialize()
        current = BrowserResource(tmp_path, tmp_path.name)
        current.initialize()
        assert current.generation == successor.generation
    finally:
        successor.release()


@pytest.mark.asyncio
async def test_exact_cleanup_rejects_live_lease_and_stale_selection(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    lease = acquire_session_lease(tmp_path)
    try:
        resource = BrowserResource(tmp_path, tmp_path.name)
        resource.initialize()
        resource.record["terminal"] = "failed"
        resource.remember("bridge:100:private")
        with pytest.raises(SessionLeaseHeldError):
            await cleanup_exact(tmp_path, resource.generation)
        assert (await cleanup_exact(tmp_path, "wrong-generation")).state == "unresolved"
        assert not bridge.calls
    finally:
        lease.release()
    assert (await cleanup_exact(tmp_path, resource.generation)).state == "closed"


@pytest.mark.asyncio
async def test_dead_process_alone_never_authorizes_cleanup(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("bridge:100:private")
    assert (await cleanup_exact(tmp_path, resource.generation)).state == "unresolved"
    assert not bridge.calls


def test_unleased_second_instance_never_fences_out_the_live_owner(tmp_path: Path) -> None:
    """B1: identity is durable per SESSION, not per BrowserResource instance.

    In-process children take this path — they use ``claim_session``, not
    ``acquire_session_lease`` — so a per-instance token meant every subagent
    could have its live tab silently transferred to a newcomer, stranding a
    tab in the pool that nobody could close.
    """
    assert not (tmp_path / ".execution-lease").exists()
    incumbent = BrowserResource(tmp_path, tmp_path.name)
    incumbent.initialize()
    incumbent.remember("bridge:2:live")

    newcomer = BrowserResource(tmp_path, tmp_path.name)
    newcomer.initialize()

    assert newcomer.generation == incumbent.generation
    incumbent.remember("bridge:2:live")  # still authoritative over its own tab
    assert incumbent.record["surface_id"] == "bridge:2:live"


def test_identity_read_never_mints_or_rotates(tmp_path: Path) -> None:
    """execution_generation is a lookup; asking who owns must not renumber."""
    fresh = BrowserResource(tmp_path, tmp_path.name)
    assert fresh.execution_generation == ""
    fresh.initialize()
    stored = fresh.generation
    for _ in range(3):
        assert BrowserResource(tmp_path, tmp_path.name).execution_generation == stored
    assert json.loads((tmp_path / RESOURCE_NAME).read_text())["generation"] == stored


@pytest.mark.asyncio
async def test_failed_cleanup_does_not_rotate_the_copied_generation(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """D3/U2: the identifier the listing told the operator to copy stays valid."""
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.record["terminal"] = "failed"
    resource.remember("bridge:5:tok", state="cleanup_pending")
    copied = resource.generation
    bridge.fail_close = True

    for _ in range(3):
        result = await cleanup_exact(tmp_path, copied)
        assert result.state == "pending"
        assert json.loads((tmp_path / RESOURCE_NAME).read_text())["generation"] == copied

    bridge.fail_close = False
    assert (await cleanup_exact(tmp_path, copied)).state == "closed"


@pytest.mark.asyncio
async def test_listing_and_cleanup_agree_on_the_stranded_state(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """M1: a failed removal persists 'pending'; both surfaces must accept it."""
    # Own sessions root: tmp_path.parent is pytest's shared directory and would
    # scan sibling tests' records.
    sessions = tmp_path / "sessions"
    directory = sessions / "stranded"
    resource = BrowserResource(directory, "stranded")
    resource.initialize()
    resource.record["terminal"] = "completed"
    resource.remember("bridge:100:private")
    # The extension REPORTS pending when chrome.tabs.remove threw: the reply
    # overwrites the durable 'cleanup_pending' intent, which is how the two
    # surfaces came to disagree about the one genuinely stranded state.
    bridge.finish_state = "pending"
    assert (await resource.finish(resource.generation, "completed")).state == "pending"

    row = read_inventory(sessions)[0]
    assert row["state"] == "pending"
    assert row["cleanup_candidate"] is True, "the one stranded state must be listed as cleanable"

    bridge.finish_state = ""
    assert (await cleanup_exact(directory, resource.generation)).state == "closed"


@pytest.mark.asyncio
async def test_refusals_name_the_specific_cause_and_the_route_out(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """U1/U4: the crash shape must name resume, not read as a mistyped id."""
    crashed = tmp_path / "crashed"
    resource = BrowserResource(crashed, "crashed")
    resource.initialize()
    resource.remember("bridge:9:stranded")  # killed before finish: no terminal

    row = read_inventory(tmp_path)[0]
    assert row["cleanup_candidate"] is False
    assert "resume" in row["blocked_reason"] and "lop --resume" in row["blocked_reason"]

    refused = await cleanup_exact(crashed, resource.generation)
    assert refused.state == "unresolved"
    assert "lop --resume" in refused.detail
    assert "stale" not in refused.detail, "a live owner is not a stale selection"

    wrong = await cleanup_exact(crashed, "not-the-generation")
    assert "generation does not match" in wrong.detail
    missing = await cleanup_exact(tmp_path / "no-such-session", "any")
    assert "no ownership record" in missing.detail

    retained = tmp_path / "held"
    held = BrowserResource(retained, "held")
    held.initialize()
    held.record.update(terminal="completed", retention="pending login")
    held.remember("bridge:9:held")
    assert "pending login" in (await cleanup_exact(retained, held.generation)).detail
    assert bridge.calls == [], "no refusal may reach the bridge"


def test_finalized_unleased_child_can_browse_on_a_later_run(tmp_path: Path) -> None:
    """M3/Q2: a settled scope must not lock a resumed child out of the browser.

    In-process children use ``claim_session``, not ``acquire_session_lease``,
    so gating the retire branch on holding a lease made ``terminal`` permanent
    for EVERY subagent: ``allocate`` refuses on terminal, and ``hub
    op='resume'`` relaunches children on their own directory, so an ordinary
    finish-then-resume left a child that could never browse again. No round-1
    guard set ``terminal`` on an UNLEASED record, which is why the regression
    passed 15/15.
    """
    directory = tmp_path / "sessions" / "child"
    first = BrowserResource(directory, "child")
    first.initialize()
    first.record.update(state="retained", surface_id="bridge:9:tok", terminal="cancelled")
    first._save()
    assert not (directory / ".execution-lease").exists()

    resumed = BrowserResource(directory, "child")
    resumed.initialize()
    assert resumed.record.get("terminal") is None
    resumed.allocate()


def test_paused_unleased_child_consumes_its_release_on_resume(tmp_path: Path) -> None:
    """A pause routes through cancellation, so the unleased path must release it."""
    directory = tmp_path / "sessions" / "paused"
    first = BrowserResource(directory, "paused")
    first.initialize()
    first.record.update(state="retained", terminal="cancelled", retention="paused scope")
    first._save()

    resumed = BrowserResource(directory, "paused")
    resumed.initialize()
    assert resumed.record.get("release_pause") is True


def test_operator_cleanup_of_a_stranded_scope_is_never_blocked_by_a_resume(
    tmp_path: Path,
) -> None:
    """Retiring a stranded scope must not cost the operator their recovery route.

    A failed close leaves the tab out there, and the resumed owner is the party
    responsible for it — the route the crash-shape refusal names. That is safe
    for the operator's own path because ``cleanup_exact`` reaches a record
    through ``adopt``, never through ``initialize``, so an untouched stranded
    record stays a cleanup candidate with its terminal intact.
    """
    directory = tmp_path / "sessions" / "stranded"
    first = BrowserResource(directory, "stranded")
    first.initialize()
    first.record.update(state="cleanup_pending", surface_id="bridge:4:tok", terminal="completed")
    first._save()

    # Untouched by any new run, the row remains the operator's to clean up.
    assert cleanup_disposition(read_inventory(tmp_path / "sessions")[0])[0] is True

    # A resume takes responsibility for the tab and may browse again.
    resumed = BrowserResource(directory, "stranded")
    resumed.initialize()
    assert resumed.record.get("terminal") is None
    resumed.allocate()


def test_a_live_incumbent_is_never_retired_by_a_concurrent_instance(tmp_path: Path) -> None:
    """B1 must survive the M3 fix: no terminal means no retire, so no fencing."""
    directory = tmp_path / "sessions" / "live"
    incumbent = BrowserResource(directory, "live")
    incumbent.initialize()
    incumbent.remember("bridge:2:live")

    newcomer = BrowserResource(directory, "live")
    newcomer.initialize()
    assert newcomer.generation == incumbent.generation
    # The incumbent still owns its tab; the newcomer adopted, not seized.
    incumbent.remember("bridge:2:live")


@pytest.mark.asyncio
async def test_pending_cleanup_names_the_command_not_the_class(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """D10/U10: the bridge's own message names the fix; the class name does not."""
    directory = tmp_path / "sessions" / "unreachable"
    resource = BrowserResource(directory, "unreachable")
    resource.initialize()
    resource.record["terminal"] = "completed"
    resource.remember("bridge:100:private")
    bridge.unreachable = True

    result = await resource.finish(resource.generation, "completed")
    assert result.state == "pending"
    assert "BridgeUnreachable" != result.detail
    assert "lop browser status" in result.detail
