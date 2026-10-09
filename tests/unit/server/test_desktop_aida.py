"""``/v1/desktop/aida`` — the frozen contract, exercised over the real app.

The route is the cross-repo boundary (design §4), so this file drives it the
way the renderer does — ASGI transport, bearer token, JSON bodies — against an
isolated config root, and pins each clause the UI was written against:
``GET`` never creates, ``open``/``greet`` ensure, ``greet`` is idempotent,
``pause``/``resume`` move the flag, a disabled install answers ``enabled: false``
on GET and 409 ``aida_disabled`` on POST, and the read payload carries her
configured ``name`` (the renameable-chief-of-staff contract, 2026-09-28).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from local_operator.config import ConfigManager
from local_operator.server.app import app
from tests.unit.aida.conftest import isolated_root_path, write_config

TOKEN = "aida-route-token"


@pytest.fixture()
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The SAME root every other aida test uses, built by the same function.

    A local fixture over the shared body rather than an imported fixture
    object: importing one and then naming it as a parameter is an F811
    redefinition, and a second hand-rolled root here is how the route tests
    and the engine tests would drift into testing two subtly different
    sandboxes.
    """
    return isolated_root_path(tmp_path, monkeypatch)


@pytest.fixture()
def client(isolated_root: Path, monkeypatch: pytest.MonkeyPatch):
    """The app over the isolated root, with the desktop plane open."""
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app.state.config_manager = ConfigManager(config_dir=isolated_root)
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    client = httpx.AsyncClient(transport=transport, base_url="http://test", headers=headers)
    return client


@pytest.mark.asyncio
async def test_get_never_creates_the_session(client, isolated_root: Path) -> None:
    async with client as http:
        response = await http.get("/v1/desktop/aida")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result == {
        "enabled": True,
        "session_id": None,
        "paused": False,
        "greeted": False,
        # The rename contract's field rides the read shape and defaults to the
        # packaged name; a renderer never needs a null branch for it.
        "name": "Aida",
        # First-run onboarding (Lane B), additive: the greeting ledger, the
        # first-run predicate the desktop's onboarding finish() reads, and the
        # sign-in identity (null until a Radient login decodes one).
        "greeting": {
            "state": "owed",
            "surface": None,
            "requested_at": None,
            "armed_at": None,
            "delivered_at": None,
        },
        # No provider in this rig, so the first-run experience is not yet
        # reachable (the greeting needs a turn that can run).
        "first_run_pending": False,
        "operator": None,
        # The same word the POST answers with, on the READ (Lane B round 1):
        # step 3 of the desktop's wizard promises "she will say hello first"
        # BEFORE the press, and ``greeted`` cannot say it — a pending greeting
        # and one that will never come are both false there.
        "greeting_state": "owed",
        # The live-owner flag, on the read for shape parity; a GET performs no
        # operation, so nobody else is carrying one out.
        "held": False,
    }


@pytest.mark.asyncio
async def test_open_creates_and_answers_the_frozen_shape(client, isolated_root: Path) -> None:
    async with client as http:
        response = await http.post("/v1/desktop/aida", json={"op": "open"})
    assert response.status_code == 200
    result = response.json()["result"]
    # THE OP SHAPE EXACTLY (freeze §4): `enabled` is GET's field — a POST only
    # reaches here when it is true. Two additive keys (first-run onboarding):
    # ``greeting_state`` tells the desktop whether she is about to speak (which
    # ``greeted``, now "delivered", cannot), and ``held`` says a live session on
    # this machine carries the effect out instead of this call.
    assert set(result) == {"session_id", "paused", "greeted", "greeting_state", "held"}
    assert result["greeting_state"] == "owed"
    assert result["held"] is False
    assert result["session_id"]
    assert result["paused"] is False
    assert (isolated_root / "sessions" / result["session_id"]).is_dir()


@pytest.mark.asyncio
async def test_pause_and_resume_move_the_flag(client, isolated_root: Path) -> None:
    async with client as http:
        await http.post("/v1/desktop/aida", json={"op": "open"})
        paused = await http.post("/v1/desktop/aida", json={"op": "pause"})
        assert paused.status_code == 200 and paused.json()["result"]["paused"] is True
        resumed = await http.post("/v1/desktop/aida", json={"op": "resume"})
        assert resumed.status_code == 200 and resumed.json()["result"]["paused"] is False
        status = await http.post("/v1/desktop/aida", json={"op": "status"})
        assert status.status_code == 200 and status.json()["result"]["paused"] is False


@pytest.mark.asyncio
async def test_a_resume_refused_by_a_held_store_lock_says_so(
    client, isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``busy`` word gets its own receipt (round 3d, N2).

    ``proactive.resume`` answers ``"busy"`` when a peer held the store lock for
    the whole wait — nothing was armed on that call — so the generic
    "the next check-in is armed." would tell the desktop's operator a check-in
    exists when it does not. The sentence also has to stay calm: the refusal is
    retryable, and the next boot or tick arms it.
    """
    from local_operator.aida import proactive

    async def busy(*_args: object, **_kwargs: object) -> str:
        return "busy"

    monkeypatch.setattr(proactive, "resume", busy)
    async with client as http:
        await http.post("/v1/desktop/aida", json={"op": "open"})
        resumed = await http.post("/v1/desktop/aida", json={"op": "resume"})

    assert resumed.status_code == 200
    message = resumed.json()["message"]
    assert "busy" in message and "arms" in message, message
    assert "the next check-in is armed" not in message, message


@pytest.mark.asyncio
async def test_greet_refuses_without_a_provider_and_stamps_nothing(
    client, isolated_root: Path
) -> None:
    async with client as http:
        await http.post("/v1/desktop/aida", json={"op": "open"})
        response = await http.post("/v1/desktop/aida", json={"op": "greet"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "aida_no_provider"
    # The GREETING is still owed, which is the invariant this test exists for
    # (a refusal must not spend the one-time greeting). Asserted on the
    # greeting's own ledger rather than on file absence since slice B: the
    # nudge window (R25) shares `onboarding.json` and the `open` above armed a
    # cadence row, whose message spends a nudge window — so the file now
    # exists for a ledger that has nothing to do with the greeting.
    from local_operator.aida import onboarding

    assert onboarding.greeted_at(isolated_root) is None


@pytest.mark.asyncio
async def test_the_read_payload_carries_the_configured_name(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``name`` is read LIVE from ``aida.name`` — the cross-repo contract.

    The UI slice labels her row with ``aida.data?.name ?? "Aida"``, so the
    field must exist, be a plain string, and change the moment the config
    key changes — with no restart and with no session of hers even existing
    (a rename made in a terminal or the desktop is visible to every other
    renderer on its next GET).
    """
    write_config(isolated_root, {"aida": {"name": "Sovereign"}})
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app.state.config_manager = ConfigManager(config_dir=isolated_root)
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=headers
    ) as http:
        response = await http.get("/v1/desktop/aida")
        opened = await http.post("/v1/desktop/aida", json={"op": "open"})
    assert response.status_code == 200
    assert response.json()["result"]["name"] == "Sovereign"
    # The read's ledger word tracks the ledger, not a constant: this rig has a
    # provider-less install, so it stays ``owed`` however many opens run.
    assert response.json()["result"]["greeting_state"] == "owed"
    # The receipts speak the configured name too, not the packaged string.
    assert opened.status_code == 200
    assert "Sovereign" in opened.json()["message"]


@pytest.mark.asyncio
async def test_unknown_op_is_refused_by_the_schema(client) -> None:
    async with client as http:
        response = await http.post("/v1/desktop/aida", json={"op": "banish"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_disabled_install_answers_get_and_refuses_post(client, isolated_root: Path) -> None:
    async with client as http:
        await http.post("/v1/desktop/aida", json={"op": "open"})
    write_config(isolated_root, {"aida": {"enabled": False}})
    app.state.config_manager = ConfigManager(config_dir=isolated_root)

    # A FRESH client: httpx refuses to re-enter one ("Cannot open a client
    # instance more than once"), and the fresh construction is also what the
    # desktop does after a settings change.
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=headers
    ) as http:
        got = await http.get("/v1/desktop/aida")
        posted = await http.post("/v1/desktop/aida", json={"op": "open"})
    assert got.status_code == 200
    assert got.json()["result"]["enabled"] is False
    assert posted.status_code == 409
    assert posted.json()["detail"]["code"] == "aida_disabled"


@pytest.mark.asyncio
async def test_capability_is_advertised(client) -> None:
    async with client as http:
        response = await http.get("/v1/capabilities")
    assert response.json()["result"]["features"].get("aida") == 1


@pytest.mark.asyncio
async def test_greet_is_the_attended_request_and_answers_the_ledger_state(
    client, isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``greet`` from the desktop moves owed → requested → armed, hidden.

    The contract the desktop's onboarding ``finish()`` is written against: a
    200 whose ``greeting_state`` is ``armed`` means "navigate to her session,
    her message is about to land"; ``greeted`` stays false until the fire.
    """
    from local_operator.aida import onboarding
    from local_operator.wakes import store as wake_store

    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    async with client as http:
        response = await http.post("/v1/desktop/aida", json={"op": "greet"})
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["greeting_state"] == "armed"
        assert result["greeted"] is False
        state = (await http.get("/v1/desktop/aida")).json()["result"]
        assert state["greeting"]["state"] == "armed"
        assert state["greeting"]["surface"] == "desktop"
        assert state["first_run_pending"] is True
        again = await http.post("/v1/desktop/aida", json={"op": "greet"})
        assert again.status_code == 200
        assert "already" in again.json()["message"]
    entry = wake_store.read_entry(isolated_root, result["session_id"]) or {}
    rows = [r for r in entry.get("schedules") or [] if r["id"] == onboarding.GREETING_WAKE_ID]
    assert len(rows) == 1 and rows[0]["hidden"] is True


@pytest.mark.asyncio
async def test_the_read_payload_carries_the_radient_identity(
    client, isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.aida import onboarding

    monkeypatch.setattr(
        onboarding, "radient_identity", lambda root: {"name": "Jane Doe", "email": "jane@x.com"}
    )
    async with client as http:
        result = (await http.get("/v1/desktop/aida")).json()["result"]
    assert result["operator"] == {"name": "Jane Doe", "email": "jane@x.com", "source": "radient"}


@pytest.mark.asyncio
async def test_greet_says_held_when_another_window_owns_her(
    client, isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The discriminator the desktop branches on instead of our prose.

    A live session owns her rows, so ``arm_wake`` refuses with a 503 and the
    greeting is armed by that owner moments later — still a 200, but with
    ``held: true`` so the window knows to say "she will greet you in the other
    window" rather than claiming she is about to speak here.
    """
    from local_operator.aida import onboarding
    from local_operator.wakes.arm import WakeWriteError

    async def _owner(*args, **kwargs):
        raise WakeWriteError("a live session owns her rows", status=503, code="wake_live_owner")

    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    monkeypatch.setattr("local_operator.wakes.arm.arm_wake", _owner)
    async with client as http:
        await http.post("/v1/desktop/aida", json={"op": "open"})
        response = await http.post("/v1/desktop/aida", json={"op": "greet"})
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["held"] is True
    # The request SURVIVED: the owner (or a later resume) still arms it.
    assert result["greeting_state"] == "requested"
    assert "another window" in response.json()["message"]


@pytest.mark.asyncio
async def test_a_skipped_greeting_does_not_claim_she_already_said_hello(
    client, isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1, R-5: ``already`` used to cover three different ledgers.

    An install whose greeting was marked ``skipped`` (it already has
    conversations) is never going to be greeted, so the desktop must not answer
    "she has already introduced herself" — that sentence reports a pending
    thing as a done one. The refusal is still a 200 with the same shape: the
    renderer reads ``greeting_state``, and nothing about the wire changed.
    """
    from local_operator.aida import onboarding

    monkeypatch.setattr(onboarding, "provider_configured", lambda root: True)
    # The engagement signal itself is covered by the onboarding tests; here it
    # only has to be true so the route reaches the skipped branch.
    monkeypatch.setattr(onboarding, "_her_conversation_had", lambda root: True)
    async with client as http:
        response = await http.post("/v1/desktop/aida", json={"op": "greet"})
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["greeting_state"] == "skipped"
        assert result["greeted"] is False
        message = response.json()["message"]
        assert "already introduced herself" not in message
        assert "already has conversations" in message
