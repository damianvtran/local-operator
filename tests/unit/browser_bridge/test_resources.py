from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.browser_bridge import resources
from local_operator.browser_bridge.backend import BridgeError, BridgeUnreachable
from local_operator.browser_bridge.protocol import ErrorCode
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
        #: Override ``owner_recover``'s reply. The stranding defect is a host
        #: answering ``allocating`` with NO tab while the record still holds a
        #: live one (a tab paused behind a password-manager prompt did exactly
        #: this in sessions 704fdeafe9e3 and d6d5022bd45e).
        self.recover_reply: dict[str, Any] | None = None
        #: Per-surface error a plain ``close`` raises; absent means it closes.
        self.close_errors: dict[str, Exception] = {}

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "owner_recover":
            if self.recover_reply is not None:
                return {"ownership_version": 1, **self.recover_reply}
            return {"ownership_version": 1, "state": "owned", "tab": "bridge:100:private"}
        if method == "close":
            error = self.close_errors.get(str(params.get("tab")))
            if error is not None:
                raise error
            return {}
        if method == "owner_retain":
            self.retained = True
            return {"state": "retained"}
        if method == "owner_release":
            self.retained = False
            return {"state": "released", "terminal": ""}
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
async def test_cleanup_exact_refuses_a_session_a_move_is_handing_over(
    tmp_path: Path, bridge: BridgeFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ONE LEASE ACQUIRER THE MOVE'S GUARDS DID NOT COVER (review round 2, MINOR 1).

    Every product opener reaches a session through ``session_factory._prepare``, which
    refuses an id whose move is in flight. This path takes the execution lease DIRECTLY, so
    a cleanup landing between the move's lease re-check and its delete would be a second
    writer on a transcript another device is about to own. The holder writes only
    ``.browser-resource.json``, which is why the reviewer rated it a minor; the refusal
    costs one sentence.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path.parent))
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    # A resource whose OWNER finished and recorded the terminal intent — i.e. everything
    # this path needs to be willing to clean up — so the only thing that can refuse is the
    # handoff guard under test (without it, the next step takes the lease and cleans up).
    resource.record["terminal"] = "failed"
    resource.remember("bridge:100:private")
    directory, resource_generation = tmp_path, resource.generation

    from local_operator.session.placement import write_handoff_entry

    config_dir = directory.parent.parent
    write_handoff_entry(
        config_dir,
        directory.name,
        {
            "role": "source",
            "phase": "prepared",
            "to_device": "d_other",
            "to_name": "pixel",
            "instance_id": "i_probe",
            "at": 1.0,
        },
    )
    result = await cleanup_exact(directory, resource_generation, config_dir=config_dir)

    assert result.state == "unresolved", result
    assert "pixel" in result.detail, result
    assert not bridge.calls


@pytest.mark.asyncio
async def test_the_move_guard_is_consulted_through_a_requested_config_root(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """A caller that knows its config root gets the guard whatever the path looks like.

    MEASURED (review round 3, NIT 1): the root was ``directory.parent.parent``, so a caller
    passing anything but ``<config>/sessions/<id>`` got a root belonging to no config at all —
    no guard and no error, the one shape in which a safety check fails open. The product's one
    caller (``cli``) now passes the root it already resolved; this cell asks for the guard
    from a directory deliberately NOT in that layout, which is what the old derivation
    silently mis-handled.
    """
    from local_operator.session.placement import write_handoff_entry

    store = tmp_path / "store"
    store.mkdir()
    # ONE level below the store, deliberately: the derivation this replaces walked up TWO
    # (`directory.parent.parent`), so a fixture at the conventional depth cannot tell the two
    # implementations apart — and a cell that passes under both is not a cell for this fix.
    directory = store / "the-session"
    directory.mkdir()
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    resource.record["terminal"] = "failed"
    resource.remember("bridge:100:private")
    write_handoff_entry(
        store,
        directory.name,
        {
            "role": "source",
            "phase": "prepared",
            "to_device": "d_other",
            "to_name": "pixel",
            "instance_id": "i_probe",
            "at": 1.0,
        },
    )

    result = await cleanup_exact(directory, resource.generation, config_dir=store)

    assert result.state == "unresolved", result
    assert "pixel" in result.detail, result
    assert not bridge.calls


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


# --- a tab the session still owns is never forgotten ---------------------------
#
# Reproduces the 2026-11-17 stranding: a session held `bridge:8`, the host's
# `owner_recover` answered "allocating" with no tab (the tab was paused behind a
# password-manager prompt, so the host could not attribute it), and the record's
# ONLY handle slot was overwritten with "". The next `open` minted a new tab,
# teardown closed only that one, and the first tab stayed open forever, eating
# one of the host's eight slots per incident.


async def _lose_the_handle_through_recover(
    resource: BrowserResource, bridge: BridgeFixture
) -> None:
    resource.initialize()
    resource.remember("bridge:8:first")
    bridge.recover_reply = {"state": "allocating", "tab": ""}
    resource.recovered = False
    await resource.recover()


@pytest.mark.asyncio
async def test_recover_that_loses_the_tab_keeps_the_old_handle_owed(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    await _lose_the_handle_through_recover(resource, bridge)
    record = json.loads(resource.path.read_text())
    assert record["surface_id"] == ""
    assert [entry["surface"] for entry in record["stranded_surfaces"]] == ["bridge:8:first"]


@pytest.mark.asyncio
async def test_a_replacing_open_keeps_the_previous_live_handle_owed(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("bridge:8:first")
    resource.remember("bridge:9:second")
    assert resource.record["surface_id"] == "bridge:9:second"
    assert [entry["surface"] for entry in resource.record["stranded_surfaces"]] == [
        "bridge:8:first"
    ]
    # Re-remembering the SAME handle (every action does) is not a replacement,
    # and a close (empty handle) is a settlement, not a loss.
    resource.remember("bridge:9:second")
    resource.remember("")
    assert len(resource.record["stranded_surfaces"]) == 1


@pytest.mark.asyncio
async def test_teardown_closes_every_still_owned_surface(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    await _lose_the_handle_through_recover(resource, bridge)
    bridge.recover_reply = None
    resource.remember("bridge:9:second")
    result = await resource.finish(resource.generation, "completed")
    assert result.state == "closed"
    closes = [params for method, params in bridge.calls if method == "close"]
    assert [params["tab"] for params in closes] == ["bridge:8:first"]
    # Closed under this scope's owner proof, so an owner-aware host can
    # attribute it; the entry's allocation id is the one current when it was
    # stranded (the host's close accepts any allocation of the scope).
    assert closes[0]["owner_proof"] == resource.record["proof"]
    assert "stranded_surfaces" not in json.loads(resource.path.read_text())


@pytest.mark.asyncio
@pytest.mark.parametrize("ownership", [True, False], ids=["owner-aware", "legacy"])
@pytest.mark.parametrize("pause_after", [0, 1], ids=["before-first", "after-first"])
async def test_terminal_pause_leaves_unattempted_stranded_closes_owed(
    tmp_path: Path,
    bridge: BridgeFixture,
    monkeypatch: pytest.MonkeyPatch,
    ownership: bool,
    pause_after: int,
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    await _lose_the_handle_through_recover(resource, bridge)
    resource.remember("bridge:9:second")
    resource.remember("bridge:10:live")
    monkeypatch.setattr(resource, "ownership_mode", lambda: ownership)
    owed = ["bridge:8:first", "bridge:9:second"]

    # The fake peer records admission before it answers, so the callback sees
    # a pause that lands by the time the first close comes back.
    def closes() -> list[str]:
        return [params["tab"] for method, params in bridge.calls if method == "close"]

    skipped = await resource.finish(
        resource.generation,
        "completed",
        close_allowed=lambda: len(closes()) < pause_after,
    )
    assert skipped.state == "pending"
    assert skipped.admission_skipped
    assert closes() == owed[:pause_after]
    assert not any(method == "owner_finish" for method, _ in bridge.calls)
    saved = json.loads(resource.path.read_text())
    assert saved["state"] == "cleanup_pending"
    assert saved["surface_id"] == "bridge:10:live"
    assert [entry["surface"] for entry in saved["stranded_surfaces"]] == owed[pause_after:]

    settled = await resource.finish(resource.generation, "completed", close_allowed=lambda: True)
    assert settled.state == "closed"
    assert closes() == (owed if ownership else [*owed, "bridge:10:live"])
    assert "stranded_surfaces" not in json.loads(resource.path.read_text())


@pytest.mark.asyncio
async def test_a_stranded_tab_already_gone_is_settled(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    await _lose_the_handle_through_recover(resource, bridge)
    bridge.close_errors["bridge:8:first"] = BridgeError(ErrorCode.TAB_CLOSED, "tab is gone")
    assert (await resource.finish(resource.generation, "completed")).state == "closed"
    assert "stranded_surfaces" not in resource.record


@pytest.mark.asyncio
async def test_a_stranded_close_that_fails_stays_pending_and_cleanup_retries_it(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    # Own sessions root, so the inventory scan sees only this record.
    sessions = tmp_path / "sessions"
    directory = sessions / "synthetic"
    resource = BrowserResource(directory, directory.name)
    await _lose_the_handle_through_recover(resource, bridge)
    bridge.close_errors["bridge:8:first"] = BridgeError(
        ErrorCode.OWNER_REFUSED, "a password manager prompt in that tab is waiting"
    )
    result = await resource.finish(resource.generation, "completed")
    assert result.state == "pending"
    # The host's own words reach the caller instead of a generic sentence.
    assert "password manager prompt" in result.detail
    record = json.loads(resource.path.read_text())
    assert record["state"] == "pending"
    assert [entry["surface"] for entry in record["stranded_surfaces"]] == ["bridge:8:first"]
    row = read_inventory(sessions)[0]
    assert row["cleanup_candidate"] is True
    assert row["stranded_tabs"] == 1
    # Redacted: the count travels, the capability never does.
    assert "first" not in json.dumps(read_inventory(sessions))

    bridge.close_errors.clear()
    cleaned = await cleanup_exact(directory, str(record["generation"]))
    assert cleaned.state == "closed", cleaned.detail
    assert "stranded_surfaces" not in json.loads(resource.path.read_text())


def test_a_settled_record_still_owing_a_stranded_tab_is_a_cleanup_candidate() -> None:
    eligible, reason = cleanup_disposition(
        {
            "terminal": "completed",
            "state": "closed",
            "stranded_surfaces": [{"surface": "ui:8:x", "allocation_id": "a"}],
        }
    )
    assert eligible, reason


def test_a_reissued_capability_for_the_same_tab_is_not_owed(tmp_path: Path) -> None:
    """Owing `ui:100:old` after the host re-issued `ui:100:new` for the SAME tab
    would make teardown close the tab the session is actively using."""
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("bridge:100:old")
    resource.remember("bridge:100:new")
    assert "stranded_surfaces" not in resource.record


@pytest.mark.asyncio
async def test_an_unproven_ledger_never_receives_a_stranded_close(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    await _lose_the_handle_through_recover(resource, bridge)
    bridge.recover_reply = {"state": "unresolved", "tab": ""}
    resource.remember("bridge:9:second")
    resource.recovered = False
    result = await resource.finish(resource.generation, "completed")
    assert result.state == "unresolved"
    assert "close" not in [method for method, _ in bridge.calls]
    assert [entry["surface"] for entry in resource.record["stranded_surfaces"]] == [
        "bridge:8:first"
    ]


@pytest.mark.asyncio
async def test_a_reattributed_tab_is_not_owed_and_a_retained_scope_keeps_it(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """Review R1 (PR #4189): the incident's own recovery path. A tab paused
    behind a password-manager prompt loses attribution, the user dismisses the
    prompt and the host attributes the SAME handle again. Left owed, `finish`
    closed the live — here retained — tab before `owner_retain`."""
    resource = BrowserResource(tmp_path, tmp_path.name)
    await _lose_the_handle_through_recover(resource, bridge)
    assert resource.record["stranded_surfaces"][0]["surface"] == "bridge:8:first"
    bridge.recover_reply = {"state": "owned", "tab": "bridge:8:first"}
    resource.recovered = False
    await resource.recover()
    assert resource.record["surface_id"] == "bridge:8:first"
    assert "stranded_surfaces" not in json.loads(resource.path.read_text())
    resource.record["retention"] = "pending site approval"
    bridge.retained = True
    result = await resource.finish(resource.generation, "completed")
    assert result.state == "retained"
    assert [p["tab"] for m, p in bridge.calls if m == "close"] == []


@pytest.mark.asyncio
async def test_remember_of_an_owed_handle_makes_it_live_not_owed(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """The tool's `finally: remember(state.surface_id)` can put the handle
    recover just stranded back into the live slot."""
    resource = BrowserResource(tmp_path, tmp_path.name)
    await _lose_the_handle_through_recover(resource, bridge)
    resource.remember("bridge:8:first")
    assert "stranded_surfaces" not in resource.record


@pytest.mark.asyncio
async def test_close_stranded_never_closes_the_live_handle(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """Belt-and-braces: a record written before entries were dropped on
    re-adoption must not make the retry close the tab in use."""
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("bridge:8:first")
    resource.record["stranded_surfaces"] = [
        {"surface": "bridge:8:first", "allocation_id": "a"},
        {"surface": "bridge:9:old", "allocation_id": "a"},
    ]
    assert await resource.close_stranded() == []
    assert [p["tab"] for m, p in bridge.calls if m == "close"] == ["bridge:9:old"]
    assert resource.record["surface_id"] == "bridge:8:first"


def test_an_owed_handle_whose_tab_number_was_reused_stays_owed(tmp_path: Path) -> None:
    """Exact handle only: a live tab reusing an owed tab's NUMBER is a
    different tab, so the old capability stays owed (QA Q-e2's shape)."""
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.record["stranded_surfaces"] = [{"surface": "ui:8:A", "allocation_id": "a"}]
    resource.remember("ui:8:B")
    assert resource.record["stranded_surfaces"] == [{"surface": "ui:8:A", "allocation_id": "a"}]


# -- pending-ask handoff hold (session a64fec9430f3) -------------------------
#
# The incident: a sign-in tab was left open for ask a-e69c, the fleet update
# retired the runtime, and the retiring process's disposal closed the tab
# because the scope was unretained. The hold is keyed: only an ask bound to
# the exact live tab AND its origin (`bind_ask_handoff`, written by
# `ask browser_handoff=true`) keeps that tab, and only until THAT ask settles.

SIGNIN = "https://signin.example.test/login?next=%2Fapps"
TAB = "bridge:100:private"


def _queue_ask(directory: Path, ask_id: str, *, expires_in_ms: int = 3_600_000) -> int:
    from local_operator.asks import store

    now = store.now_ms()
    store.append_event(
        directory,
        {
            "v": 1,
            "kind": "queued",
            "ask_id": ask_id,
            "at": now,
            "created_at": now,
            "expires_at": now + expires_in_ms,
            "timeout_s": max(1, expires_in_ms // 1000),
            "questions": [{"id": "signin", "question": "Sign in?", "options": []}],
            "tool_call_id": "call_synthetic",
            "urgent": False,
        },
    )
    return now + expires_in_ms


def _settle_ask(directory: Path, ask_id: str, kind: str) -> None:
    from local_operator.asks import store

    event: dict[str, Any] = {
        "v": 1,
        "kind": kind,
        "ask_id": ask_id,
        "at": store.now_ms(),
        "tool_call_id": "call_synthetic",
    }
    if kind == "answered":
        event["answers"] = {"signin": ["Signed in"]}
        event["by"] = {"surface": "unverified"}
    store.append_event(directory, event)


def _methods(bridge: BridgeFixture) -> list[str]:
    return [method for method, _ in bridge.calls]


def _owned_tab(directory: Path, url: str = SIGNIN) -> BrowserResource:
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    resource.remember(TAB)
    resource.note_surface_origin(url)
    return resource


def _handoff(resource: BrowserResource, ask_id: str, **kw: Any) -> None:
    resource.bind_ask_handoff(ask_id, _queue_ask(resource.directory, ask_id, **kw))


@pytest.mark.asyncio
async def test_disposal_keeps_the_tab_bound_to_an_open_handoff_ask(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """The reproduction: before the hold this closed the tab."""
    resource = _owned_tab(tmp_path)
    _handoff(resource, "a-e69c")
    result = await resource.finish(resource.generation, "disposed")
    assert result.state == "retained"
    assert [p["reason"] for m, p in bridge.calls if m == "owner_retain"] == ["pending ask a-e69c"]
    record = json.loads(resource.path.read_text())
    assert record["surface_id"] == TAB
    assert record[resources.ASK_HANDOFFS_FIELD][0]["origin"] == "https://signin.example.test"
    assert "close" not in _methods(bridge)


@pytest.mark.asyncio
async def test_an_unrelated_open_ask_does_not_retain_an_active_tab(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """Over-retention control: a tab held for the agent's own work plus an
    unbound preference question closes at disposal exactly as before."""
    resource = _owned_tab(tmp_path)
    _queue_ask(tmp_path, "a-preference")
    assert (await resource.finish(resource.generation, "disposed")).state == "closed"
    assert "owner_retain" not in _methods(bridge)


@pytest.mark.asyncio
async def test_another_asks_settlement_cannot_release_the_handoff(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = _owned_tab(tmp_path)
    _queue_ask(tmp_path, "a-unrelated")
    _handoff(resource, "a-signin")
    _settle_ask(tmp_path, "a-unrelated", "answered")
    assert (await resource.finish(resource.generation, "disposed")).state == "retained"
    assert "owner_release" not in _methods(bridge)
    assert resource.record["retention"] == "pending ask a-signin"


@pytest.mark.asyncio
@pytest.mark.parametrize("settle", ["answered", "withdrawn", "timed_out"])
async def test_a_settled_handoff_closes_as_before(
    tmp_path: Path, bridge: BridgeFixture, settle: str
) -> None:
    resource = _owned_tab(tmp_path)
    if settle == "timed_out":
        _handoff(resource, "a-1", expires_in_ms=-1_000)
    else:
        _handoff(resource, "a-1")
        _settle_ask(tmp_path, "a-1", settle)
    assert (await resource.finish(resource.generation, "disposed")).state == "closed"
    assert "owner_retain" not in _methods(bridge)
    assert resources.ASK_HANDOFFS_FIELD not in json.loads(resource.path.read_text())


@pytest.mark.asyncio
async def test_origin_isolation_a_handoff_for_a_does_not_hold_the_tab_on_b(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = _owned_tab(tmp_path)
    _handoff(resource, "a-signin")
    # The agent then drove the same tab to another site.
    resource.note_surface_origin("https://other.example.test/dashboard")
    assert (await resource.finish(resource.generation, "disposed")).state == "closed"
    assert "owner_retain" not in _methods(bridge)


def test_binding_refuses_without_a_live_tab_or_a_known_origin(tmp_path: Path) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    with pytest.raises(resources.BrowserOwnershipError, match="no browser tab"):
        resource.bind_ask_handoff("a-1", None)
    resource.remember(TAB)
    with pytest.raises(resources.BrowserOwnershipError, match="not known"):
        resource.bind_ask_handoff("a-1", None)


@pytest.mark.asyncio
async def test_child_completion_with_a_handoff_keeps_the_old_rule(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = _owned_tab(tmp_path)
    _handoff(resource, "a-1")
    assert (await resource.finish(resource.generation, "completed")).state == "closed"
    assert "owner_retain" not in _methods(bridge)


async def _retire_with_handoff(directory: Path, bridge: BridgeFixture) -> None:
    first = _owned_tab(directory)
    _handoff(first, "a-1")
    assert (await first.finish(first.generation, "disposed")).state == "retained"
    bridge.calls.clear()


@pytest.mark.asyncio
async def test_successor_readdresses_the_held_tab_and_keeps_it_while_open(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    await _retire_with_handoff(tmp_path, bridge)
    successor = BrowserResource(tmp_path, tmp_path.name)
    successor.initialize()
    await successor.ensure_recovered()
    assert successor.record["surface_id"] == TAB
    assert "owner_release" not in _methods(bridge)
    assert (await successor.finish(successor.generation, "disposed")).state == "retained"


@pytest.mark.asyncio
@pytest.mark.parametrize("settle", ["answered", "withdrawn"])
async def test_successor_releases_when_the_ask_settled_while_it_was_down(
    tmp_path: Path, bridge: BridgeFixture, settle: str
) -> None:
    await _retire_with_handoff(tmp_path, bridge)
    _settle_ask(tmp_path, "a-1", settle)
    successor = BrowserResource(tmp_path, tmp_path.name)
    successor.initialize()
    await successor.ensure_recovered()
    assert successor.record["surface_id"] == TAB
    assert successor.record["retention"] == ""
    assert "owner_release" in _methods(bridge)
    assert (await successor.finish(successor.generation, "disposed")).state == "closed"


@pytest.mark.asyncio
async def test_successor_releases_when_the_ask_settles_after_it_took_over(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """Settlement AFTER an update: the new runtime recovered while the ask was
    still open (hold kept), then the answer arrives; its disposal releases."""
    await _retire_with_handoff(tmp_path, bridge)
    successor = BrowserResource(tmp_path, tmp_path.name)
    successor.initialize()
    await successor.ensure_recovered()
    assert successor.record["retention"] == "pending ask a-1"
    _settle_ask(tmp_path, "a-1", "answered")
    assert (await successor.finish(successor.generation, "disposed")).state == "closed"
    assert _methods(bridge)[-2:] == ["owner_release", "owner_finish"]


@pytest.mark.asyncio
async def test_owner_close_drops_the_handoff_and_lifts_the_hold(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    await _retire_with_handoff(tmp_path, bridge)
    successor = BrowserResource(tmp_path, tmp_path.name)
    successor.initialize()
    await successor.ensure_recovered()
    await successor.drop_ask_handoffs()
    assert "owner_release" in _methods(bridge)
    assert successor.record["retention"] == ""
    assert resources.ASK_HANDOFFS_FIELD not in successor.record


@pytest.mark.asyncio
async def test_hold_is_released_when_the_tab_was_closed_meanwhile(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    await _retire_with_handoff(tmp_path, bridge)
    successor = BrowserResource(tmp_path, tmp_path.name)
    successor.initialize()
    bridge.recover_reply = {"state": "closed", "tab": ""}
    await successor.ensure_recovered()
    await successor.finish(successor.generation, "disposed")
    assert "owner_release" in _methods(bridge)
    assert json.loads(successor.path.read_text())["retention"] == ""


@pytest.mark.asyncio
async def test_an_explicit_retain_is_never_rewritten_or_released(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = _owned_tab(tmp_path)
    resource.record["retention"] = "pending login"
    result = await resource.finish(resource.generation, "disposed")
    assert result.state == "retained"
    assert [p["reason"] for m, p in bridge.calls if m == "owner_retain"] == ["pending login"]
    assert "owner_release" not in _methods(bridge)


@pytest.mark.asyncio
async def test_cleanup_refuses_a_live_handoff_hold_and_closes_a_lapsed_one(
    tmp_path: Path, bridge: BridgeFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _retire_with_handoff(tmp_path, bridge)
    record = json.loads((tmp_path / RESOURCE_NAME).read_text())
    assert record["state"] == "retained"
    eligible, reason = cleanup_disposition(record)
    assert not eligible and "waiting on that answer" in reason
    assert not cleanup_disposition({**record, resources.ASK_HOLD_EXPIRES_FIELD: None})[0]
    deadline = int(record[resources.ASK_HOLD_EXPIRES_FIELD])
    monkeypatch.setattr(resources, "_now_ms", lambda: deadline + 1)
    assert cleanup_disposition(record) == (True, "")
    result = await cleanup_exact(tmp_path, record["generation"])
    assert result.state == "closed", result
    assert _methods(bridge)[-2:] == ["owner_release", "owner_finish"]


@pytest.mark.asyncio
async def test_the_hold_label_and_deadline_are_the_bound_asks_own(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """Review: an unrelated 2 h ask must not stretch a 10-minute sign-in hold,
    and the label names the ask being served, not the newest one."""
    resource = _owned_tab(tmp_path)
    _handoff(resource, "a-signin", expires_in_ms=600_000)
    deadline = resource.record[resources.ASK_HANDOFFS_FIELD][0]["expires_at"]
    _queue_ask(tmp_path, "a-later-preference", expires_in_ms=7_200_000)
    assert (await resource.finish(resource.generation, "disposed")).state == "retained"
    record = json.loads(resource.path.read_text())
    assert record["retention"] == "pending ask a-signin"
    assert record[resources.ASK_HOLD_EXPIRES_FIELD] == deadline


@pytest.mark.asyncio
async def test_an_unreadable_ask_log_never_reads_as_no_open_asks(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    """R1-2: a read failure keeps the bound tab (bounded by its own deadline)
    instead of closing it as if every ask had settled."""
    from local_operator.asks import store

    resource = _owned_tab(tmp_path)
    _handoff(resource, "a-signin")
    log = store.asks_log_path(tmp_path)
    log.unlink()
    log.mkdir()  # read_bytes() now raises IsADirectoryError, an OSError
    with pytest.raises(OSError):
        resources.open_ask_ids(tmp_path)
    assert (await resource.finish(resource.generation, "disposed")).state == "retained"
