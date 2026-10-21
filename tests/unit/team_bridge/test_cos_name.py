"""The chief of staff's bridge name: the resolver, the wire field, the re-sync.

Runs the REAL resolver over real config writes (``settings_io.write_setting``)
and the real Aida state file, and the REAL client/discipline over
``httpx.MockTransport``, so what is pinned is what ships: which name is read,
when ``agent_name`` is on the wire, and when it is not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator import settings_io
from local_operator.aida import naming as aida_naming
from local_operator.aida import state as aida_state
from local_operator.config import ConfigManager
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import Boundary, BridgeRefusal
from local_operator.team_bridge.cos_name import (
    AGENT_NAME_MAX_CHARS,
    bridge_agent_name,
    cos_display_name,
)
from tests.unit.team_bridge.test_client import _make_client, _ok
from tests.unit.team_bridge.test_discipline import (
    _committed_owner,
    _Counter,
    _discipline,
    _endpoint_payload,
)

AIDA_SESSION = "sess_aida"

# pytest-asyncio only marks coroutine tests, so the sync resolver cells above
# the wire section run unchanged under the module mark.
pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


def _designate(root: Path, session: str) -> None:
    settings_io.write_setting(
        ConfigManager(root), settings_io.BY_KEY["team_bridge.session"], session
    )


def _aida_lives_in(root: Path, session: str = AIDA_SESSION) -> None:
    aida_state.write_state(root, {"session_id": session})


# -- the resolver -----------------------------------------------------------


def test_the_designated_aida_reads_her_default_name(tmp_path: Path):
    _aida_lives_in(tmp_path)
    _designate(tmp_path, AIDA_SESSION)

    assert cos_display_name(tmp_path) == aida_naming.DEFAULT_NAME == "Aida"


def test_a_renamed_aida_reads_her_current_name(tmp_path: Path):
    _aida_lives_in(tmp_path)
    _designate(tmp_path, AIDA_SESSION)
    aida_naming.set_name(tmp_path, "  Friday   Jones ")

    assert cos_display_name(tmp_path) == "Friday Jones"


def test_no_designated_cos_reads_none(tmp_path: Path):
    """Badge-only (§13.6): no chief of staff on the bridge, nothing to publish."""

    _aida_lives_in(tmp_path)

    assert cos_display_name(tmp_path) is None
    assert cos_display_name(None) is None


def test_a_designated_non_aida_session_never_publishes_its_title(tmp_path: Path):
    """A generated conversation title is content, not a name the user chose."""

    _aida_lives_in(tmp_path)
    _designate(tmp_path, "sess_other")

    assert cos_display_name(tmp_path) is None


def test_a_designated_session_with_no_aida_state_reads_none(tmp_path: Path):
    _designate(tmp_path, AIDA_SESSION)

    assert cos_display_name(tmp_path) is None


def test_a_broken_aida_state_never_raises(tmp_path: Path):
    _designate(tmp_path, AIDA_SESSION)
    aida_state.state_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    aida_state.state_path(tmp_path).write_text("{not json", encoding="utf-8")

    assert cos_display_name(tmp_path) is None


@pytest.mark.parametrize(
    ("raw", "wire"),
    [
        ("Aida", "Aida"),
        ("Ai\u200bda", "Aida"),  # Cf zero-width space: the server refuses it as invisible
        ("Ai\u00a0da", "Ai da"),  # NBSP: the server refuses non-U+0020 spaces
        ("Ai\ufe0fda", "Aida"),  # variation selector
        ("\u2800", None),  # braille blank: blank-looking
        ("\u0301\u0301", None),  # combining marks only: blank-looking
        ("", None),
        (None, None),
        ("x" * 80, "x" * AGENT_NAME_MAX_CHARS),  # aida.name allows 80; the bridge 64
        ("Ai\ud800da", "Aida"),  # lone surrogate (Cs): invalid UTF-8, httpx cannot encode it
    ],
)
def test_the_wire_name_is_repaired_into_the_label_grammar(raw: Any, wire: str | None):
    assert bridge_agent_name(raw) == wire


# -- the wire field ---------------------------------------------------------


def _recording(sent: list[dict[str, Any]], *, refuse_agent_name: bool = False):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        if refuse_agent_name and "agent_name" in body:
            return httpx.Response(
                400,
                json={
                    "error": "refused",
                    "code": "invalid_request",
                    "details": {"field": "agent_name", "rule": "unknown_value"},
                },
            )
        return _ok({"endpoint": {"id": "bep_1", "state": "active"}, "settings": {}})

    return handler


async def test_agent_name_rides_an_active_write():
    sent: list[dict[str, Any]] = []
    client = _make_client(_recording(sent))

    await client.put_settings(
        state="active", consent_version="v1", agent_name="Aida", endpoint=None
    )

    assert sent == [{"state": "active", "consent_version": "v1", "agent_name": "Aida"}]


async def test_no_agent_name_keeps_the_pre_field_body():
    sent: list[dict[str, Any]] = []
    client = _make_client(_recording(sent))

    await client.put_settings(state="active", consent_version="v1", endpoint=None)

    assert sent == [{"state": "active", "consent_version": "v1"}]


@pytest.mark.parametrize("bundled", ["paused", "disabled"])
async def test_agent_name_is_never_bundled_into_a_pause(bundled: str):
    sent: list[dict[str, Any]] = []
    client = _make_client(_recording(sent))

    with pytest.raises(ValueError, match="sent alone"):
        await client.put_settings(state=bundled, agent_name="Aida")
    assert sent == []


async def test_a_refused_agent_name_is_raised_unchanged_and_never_stripped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """R1-1/R1-3: no strip-and-retry. One request, every other field intact.

    A silent re-send without the name would turn a rename re-sync into a bare
    ``state=active`` assertion; the refusal is the caller's to see instead, and
    only its closed-set code reaches the log — never the name.
    """

    sent: list[dict[str, Any]] = []
    client = _make_client(_recording(sent, refuse_agent_name=True), config_dir=tmp_path)

    with caplog.at_level("WARNING", logger="local_operator.team_bridge.client"):
        with pytest.raises(BridgeRefusal) as info:
            await client.put_settings(
                state="active",
                tenant_id="tnt_1",
                consent_version="bridge_consent_v1",
                label="Ben",
                agent_name="Aida-secret-sentinel",
                endpoint=None,
                expected_account_id="acc_1",
            )

    assert info.value.details == {"field": "agent_name", "rule": "unknown_value"}
    assert sent == [
        {
            "state": "active",
            "tenant_id": "tnt_1",
            "consent_version": "bridge_consent_v1",
            "label": "Ben",
            "agent_name": "Aida-secret-sentinel",
        }
    ]
    assert "invalid_request/unknown_value" in caplog.text
    assert "Aida-secret-sentinel" not in caplog.text
    from local_operator.team_bridge.store import Store

    # Nothing was written upstream, so nothing is committed locally either.
    assert Store(tmp_path).committed_owner_count() == 0


async def test_another_validation_refusal_is_not_retried():
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "error": "refused",
                "code": "invalid_request",
                "details": {"field": "label", "rule": "taken"},
            },
        )

    client = _make_client(handler)
    with pytest.raises(BridgeRefusal):
        await client.put_settings(
            state="active", consent_version="v1", label="Ben", agent_name="Aida", endpoint=None
        )
    assert len(sent) == 1


# -- the rename re-sync at the status boundary ------------------------------


def _echoing(agent_name: Any = None, *, echo: bool = True, state: str = "active"):
    """The paired hub's status endpoint: ``agent_name`` ALWAYS present, null when unset.

    ``echo=False`` is an OLDER hub, whose endpoint block is ``{id, label,
    state}`` only — the one shape in which the re-sync must stay silent.
    """

    payload = _endpoint_payload(state)
    if echo:
        payload["tenants"][0]["endpoint"]["agent_name"] = agent_name
    return payload


def _enrolled_aida(root: Path, name: str | None = None) -> None:
    _committed_owner(root, session=AIDA_SESSION)
    _aida_lives_in(root)
    _designate(root, AIDA_SESSION)
    if name is not None:
        aida_naming.set_name(root, name)


async def test_a_rename_is_re_sent_once_at_a_fresh_status_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _enrolled_aida(tmp_path, "Friday")
    counter = _Counter(lambda request: _ok(_echoing("Aida")))
    discipline = _discipline(counter, tmp_path)

    await discipline.check(Boundary.APP_START)

    puts = [r for r in counter.requests if r.method == "PUT"]
    assert len(puts) == 1
    # A bare rename: state asserted active (the endpoint IS active), no consent
    # echo, so no receipt is (re)committed by a status read.
    assert json.loads(puts[0].content) == {
        "state": "active",
        "tenant_id": "tnt_1",
        "agent_name": "Friday",
    }

    # The debounce: a server still echoing the old name is not re-PUT on the
    # next fresh read, nor from the cache.
    await discipline.check(Boundary.WINDOW_FOCUS)
    monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)
    await discipline.check(Boundary.SURFACE_OPEN)
    assert len([r for r in counter.requests if r.method == "PUT"]) == 1


async def test_an_in_step_name_sends_nothing(tmp_path: Path):
    _enrolled_aida(tmp_path)
    counter = _Counter(lambda request: _ok(_echoing("Aida")))

    await _discipline(counter, tmp_path).check(Boundary.APP_START)

    assert [r.method for r in counter.requests] == ["GET"]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_echoing(echo=False), id="server-without-the-field"),
        pytest.param(_echoing("Aida", state="paused"), id="paused-endpoint-never-resumed"),
        pytest.param(_echoing("Aida", state="disabled"), id="disabled-endpoint"),
    ],
)
async def test_the_re_sync_never_fires_where_it_could_change_state(
    tmp_path: Path, payload: dict[str, Any]
):
    _enrolled_aida(tmp_path, "Friday")
    counter = _Counter(lambda request: _ok(payload))

    await _discipline(counter, tmp_path).check(Boundary.APP_START)

    assert [r.method for r in counter.requests] == ["GET"]


async def test_the_re_sync_needs_this_machine_to_hold_the_enrollment(tmp_path: Path):
    """No committed owner here: another surface enrolled; a read never writes."""

    _aida_lives_in(tmp_path)
    _designate(tmp_path, AIDA_SESSION)
    aida_naming.set_name(tmp_path, "Friday")
    counter = _Counter(lambda request: _ok(_echoing("Aida")))

    await _discipline(counter, tmp_path).check(Boundary.APP_START)

    assert [r.method for r in counter.requests] == ["GET"]


async def test_no_designated_cos_never_sends_the_field(tmp_path: Path):
    _committed_owner(tmp_path)
    _aida_lives_in(tmp_path)
    counter = _Counter(lambda request: _ok(_echoing("Aida")))

    await _discipline(counter, tmp_path).check(Boundary.APP_START)

    assert [r.method for r in counter.requests] == ["GET"]


async def test_an_unset_name_echoed_as_null_is_published(tmp_path: Path):
    """The paired hub serves ``agent_name: null`` until one is set."""

    _enrolled_aida(tmp_path)
    counter = _Counter(lambda request: _ok(_echoing(None)))

    await _discipline(counter, tmp_path).check(Boundary.APP_START)

    puts = [json.loads(r.content) for r in counter.requests if r.method == "PUT"]
    assert puts == [{"state": "active", "tenant_id": "tnt_1", "agent_name": "Aida"}]


async def test_a_failed_re_sync_is_retried_at_the_next_fresh_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """R1-2: the debounce records only a SUCCESS; a transient failure retries."""

    _enrolled_aida(tmp_path, "Friday")
    attempts: list[int] = []

    def responder(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            attempts.append(1)
            if len(attempts) == 1:
                return httpx.Response(503, json={"error": "busy", "code": "service_unavailable"})
            return _ok({"endpoint": {"id": "bep_1", "state": "active"}, "settings": {}})
        return _ok(_echoing("Aida"))

    counter = _Counter(responder)
    discipline = _discipline(counter, tmp_path)
    monkeypatch.setattr(tb_client, "STATUS_CACHE_TTL_S", 0.0)

    await discipline.check(Boundary.APP_START)
    await discipline.check(Boundary.SURFACE_OPEN)
    await discipline.check(Boundary.SURFACE_OPEN)

    # Failed once, succeeded once, then debounced while the hub lags its echo.
    assert len(attempts) == 2


async def test_a_refused_rename_never_degrades_into_a_bare_state_assertion(tmp_path: Path):
    """R1-3: a refused name is one PUT, never a follow-up ``{state: active}``."""

    _enrolled_aida(tmp_path, "Friday")

    def responder(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            return httpx.Response(
                400,
                json={
                    "error": "refused",
                    "code": "invalid_request",
                    "details": {"field": "agent_name", "rule": "invisible_character"},
                },
            )
        return _ok(_echoing("Aida"))

    counter = _Counter(responder)
    check = await _discipline(counter, tmp_path).check(Boundary.APP_START)

    puts = [json.loads(r.content) for r in counter.requests if r.method == "PUT"]
    assert puts == [{"state": "active", "tenant_id": "tnt_1", "agent_name": "Friday"}]
    assert check.state == tb_client.STATE_READY


def test_a_malformed_config_is_never_moved_aside_by_the_resolver(tmp_path: Path):
    """R1-4: the resolver reads the watcher snapshot; a half-written edit stays put."""

    _aida_lives_in(tmp_path)
    _designate(tmp_path, AIDA_SESSION)
    aida_naming.set_name(tmp_path, "Friday")
    assert cos_display_name(tmp_path) == "Friday"

    config = tmp_path / "config.yml"
    torn = config.read_text(encoding="utf-8") + "\naida: {name: [unterminated\n"
    config.write_text(torn, encoding="utf-8")

    # The last-good snapshot still answers, and the user's file is untouched.
    assert cos_display_name(tmp_path) == "Friday"
    assert config.read_text(encoding="utf-8") == torn
    assert not list(tmp_path.glob("config.yml.bad.*"))
