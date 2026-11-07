"""``lop vault connect|status|disconnect`` with a stub transport and a fake clock."""

from __future__ import annotations

import io
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from local_operator.i18n.text import active_locale
from local_operator.vault_agent import cli, eligibility, keys, wire
from local_operator.vault_agent.eligibility import Observation

from .conftest import ENDPOINT, SEED, TENANT, Transport, make_client

GATE = eligibility.Gate(tenant_id=TENANT, endpoint_id=ENDPOINT, tenant_name="Acme")


@pytest.fixture(autouse=True)
def _english(monkeypatch):
    monkeypatch.setenv("LOP_I18N", "0")


@pytest.fixture
def wired(config_dir, monkeypatch):
    state = {"observation": Observation(state="open"), "gate": GATE}
    monkeypatch.setattr(cli, "_refresh", lambda _root: state["observation"])
    monkeypatch.setattr(eligibility, "gate", lambda _root: state["gate"])
    monkeypatch.setattr(keys, "load_or_create_seed", lambda endpoint, base=None: (SEED, True))
    return state


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _row(**overrides):
    row = {
        "delegation_id": "vdl_" + "c" * 32,
        "tenant_id": TENANT,
        "endpoint_id": ENDPOINT,
        "key_fingerprint": keys.fingerprint_for_seed(SEED),
        "expires_at": "2026-11-14T00:00:00Z",
        "state": "live",
    }
    row.update(overrides)
    return row


def _connect(root, transport, **kwargs):
    out = io.StringIO()
    clock = Clock()
    opened: list[str] = []
    rc = cli.connect(
        root,
        out=out,
        sleep=clock.sleep,
        clock=clock,
        opener=lambda url: opened.append(url) or True,
        client_factory=lambda _root: make_client(transport),
        **kwargs,
    )
    return rc, out.getvalue(), opened, clock


def test_connect_prints_the_4x4_grid_opens_the_link_and_caches_the_pickup(
    config_dir, wired
) -> None:
    transport = Transport(
        answers=[
            (200, {"delegations": [], "next_cursor": None}),  # not approved yet
            (200, {"delegations": [_row()], "next_cursor": None}),
        ]
    )
    rc, text, opened, clock = _connect(config_dir, transport)
    assert rc == cli.EXIT_OK
    fingerprint = keys.fingerprint_for_seed(SEED)
    for line in wire.fingerprint_grid(fingerprint):
        assert f"\n  {line}\n" in text
    assert "exactly these 16 groups" in text
    # The link: console origin, the delegate route, and identifiers + public key only.
    (url,) = opened
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == (
        "https://console.radienthq.com/dashboard/vaults/delegate"
    )
    assert parse_qs(parts.query) == {
        "tenant": [TENANT],
        "ep": [ENDPOINT],
        "pk": [wire.b64url(keys.public_key_from_seed(SEED))],
    }
    assert SEED.hex() not in url
    assert "Connected." in text
    assert clock.now == cli.POLL_SECONDS  # one wait between the two reads
    grant = eligibility.read_grant(
        config_dir, tenant_id=TENANT, endpoint_id=ENDPOINT, fingerprint=fingerprint
    )
    assert grant is not None and grant.delegation_id == "vdl_" + "c" * 32


def test_connect_stops_at_once_on_the_dark_503(config_dir, wired) -> None:
    transport = Transport(answers=[(503, {"code": "vault_unavailable"})])
    rc, text, _opened, clock = _connect(config_dir, transport)
    assert rc == cli.EXIT_NOT_CONNECTED
    assert text.rstrip().endswith("Agent access isn't available yet.")
    assert len(transport.calls) == 1 and clock.now == 0.0


def test_connect_times_out_with_a_next_step(config_dir, wired) -> None:
    transport = Transport(answers=[(200, {"delegations": [], "next_cursor": None})] * 3)
    rc, text, _opened, _clock = _connect(config_dir, transport, timeout=10)
    assert rc == cli.EXIT_NOT_CONNECTED
    assert "lop vault connect" in text.splitlines()[-1]


@pytest.mark.parametrize(
    ("state", "phrase"),
    [
        ("signed_out", "lop login radient"),
        ("not_team", "Radient Team plan"),
        ("no_endpoint", "lop team-bridge setup"),
        ("unavailable", "Agent access isn't available yet."),
        ("not_approved", "isn't approved yet"),
        ("frozen", "frozen"),
        ("error", "Try again"),
    ],
)
def test_a_closed_gate_prints_one_reason_and_touches_nothing(
    config_dir, wired, monkeypatch, state, phrase
) -> None:
    wired["observation"] = Observation(state=state, tenant_name="Acme")
    wired["gate"] = None
    created: list[str] = []
    monkeypatch.setattr(keys, "load_or_create_seed", lambda *a, **k: created.append("x"))
    transport = Transport()
    rc, text, opened, _clock = _connect(config_dir, transport)
    assert rc == cli.EXIT_NOT_CONNECTED
    assert phrase in text and len(text.strip().splitlines()) == 1
    assert created == [] and opened == [] and transport.calls == []


def test_status_shows_the_grid_and_the_grant(config_dir, wired, monkeypatch) -> None:
    monkeypatch.setattr(keys, "load_seed", lambda endpoint, base=None: SEED)
    fingerprint = keys.fingerprint_for_seed(SEED)
    eligibility.write_grant(
        config_dir,
        eligibility.Grant("vdl_x", "2999-01-01T00:00:00Z", TENANT, ENDPOINT, fingerprint),
    )
    out = io.StringIO()
    assert cli.status(config_dir, out=out) == cli.EXIT_OK
    text = out.getvalue()
    assert "available in Acme" in text
    assert wire.fingerprint_grid(fingerprint)[3] in text
    assert "Connected." in text


def test_disconnect_deletes_the_key_and_points_at_the_console(config_dir, monkeypatch) -> None:
    deleted: list[str] = []
    monkeypatch.setattr(
        keys, "delete_seed", lambda endpoint, base=None: deleted.append(endpoint) or True
    )
    monkeypatch.setattr(
        "local_operator.team_bridge.store.Store.active_owner_readonly",
        lambda self: SimpleNamespace(tenant_id=TENANT, endpoint_id=ENDPOINT),
    )
    eligibility.write_grant(
        config_dir, eligibility.Grant("vdl_x", "2999-01-01T00:00:00Z", TENANT, ENDPOINT, "f" * 64)
    )
    out = io.StringIO()
    assert cli.disconnect(config_dir, out=out) == cli.EXIT_OK
    assert deleted == [ENDPOINT]
    assert not (eligibility.vault_dir(config_dir) / eligibility.GRANT_NAME).exists()
    assert "https://console.radienthq.com/dashboard/vaults/agents" in out.getvalue()


def test_english_is_pinned_for_these_assertions() -> None:
    assert active_locale() == "en"
