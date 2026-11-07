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
    """``text`` is stdout then stderr: the order a person sees them in a terminal."""
    out = io.StringIO()
    err = io.StringIO()
    clock = Clock()
    opened: list[str] = []
    rc = cli.connect(
        root,
        out=out,
        err=err,
        sleep=clock.sleep,
        clock=clock,
        opener=lambda url: opened.append(url) or True,
        client_factory=lambda _root: make_client(transport),
        **kwargs,
    )
    _connect.err = err.getvalue()  # type: ignore[attr-defined]
    return rc, out.getvalue() + err.getvalue(), opened, clock


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
    lines = text.splitlines()
    grid = [f"  {line}" for line in wire.fingerprint_grid(fingerprint)]
    first = lines.index(grid[0])
    assert lines[first : first + 4] == grid
    # Set apart by blank lines, the instruction BEFORE the grid, console labels.
    assert lines[first - 1] == "" and lines[first + 4] == ""
    # The instructions sit ABOVE the label, so the label is directly over its value.
    assert lines[first - 2] == "Key fingerprint:"
    assert lines[first - 3] == "If they differ, stop."
    assert "exactly these 16 groups" in lines[first - 4]
    assert lines[first - 5] == f"Agent endpoint: {ENDPOINT}"
    assert lines[first - 6] == "Radient Vault agent access · Acme"
    # The URL alone on its line, unindented, plain (stdout is not a TTY here).
    assert lines[lines.index("Opening the approval page in your browser:") + 1] == opened[0]
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
    assert any(line.startswith("Connected until ") for line in lines)
    assert lines[-1] == "This agent can read the vaults you shared with it."
    assert _connect.err == ""  # type: ignore[attr-defined]
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
    assert _connect.err == "Agent access isn't available yet.\n"  # type: ignore[attr-defined]
    assert len(transport.calls) == 1 and clock.now == 0.0


def test_connect_maps_401_to_sign_in_and_403_to_membership(config_dir, wired) -> None:
    rc, text, _o, _c = _connect(config_dir, Transport(answers=[(401, {"code": "unauthorized"})]))
    assert rc == cli.EXIT_NOT_CONNECTED
    assert text.splitlines()[-1] == "Sign in to Radient first: run `lop login radient`."
    rc, text, _o, _c = _connect(config_dir, Transport(answers=[(403, {"code": "vault_no_access"})]))
    assert rc == cli.EXIT_NOT_CONNECTED
    assert "Ask an organization owner or admin" in text.splitlines()[-1]


@pytest.mark.parametrize(
    ("timeout", "phrase"),
    [
        (30, "up to 30 seconds"),
        (60, "up to 1 minute;"),
        (90, "up to 2 minutes"),
        (600, "10 minutes"),
    ],
)
def test_the_wait_is_rounded_up_never_down(timeout, phrase) -> None:
    assert phrase in cli._waiting(timeout)


def test_a_naive_expiry_is_read_as_utc() -> None:
    naive = eligibility.parse_instant("2026-11-14T00:00:00")
    aware = eligibility.parse_instant("2026-11-14T00:00:00Z")
    assert naive is not None and naive == aware
    assert eligibility.parse_instant("not a time") is None


def test_the_link_is_hyperlinked_only_on_a_capable_tty(monkeypatch) -> None:
    import local_operator.terminals as terminals

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    url = "https://console.radienthq.com/dashboard/vaults/agents"
    monkeypatch.setattr(terminals, "is_hyperlinked", lambda env=None: True)
    assert cli._link(url, io.StringIO()) == url
    assert cli._link(url, Tty()) == f"\x1b]8;;{url}\x1b\\{url}\x1b]8;;\x1b\\"
    monkeypatch.setattr(terminals, "is_hyperlinked", lambda env=None: False)
    assert cli._link(url, Tty()) == url


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


def _grant(config_dir) -> None:
    eligibility.write_grant(
        config_dir,
        eligibility.Grant(
            "vdl_x", "2999-01-01T00:00:00Z", TENANT, ENDPOINT, keys.fingerprint_for_seed(SEED)
        ),
    )


def _status(config_dir, **kwargs) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    rc = cli.status(config_dir, out=out, err=err, **kwargs)
    return rc, out.getvalue(), err.getvalue()


def test_status_shows_the_grid_and_the_grant(config_dir, wired, monkeypatch) -> None:
    monkeypatch.setattr(keys, "load_seed", lambda endpoint, base=None: SEED)
    _grant(config_dir)
    rc, text, _err = _status(config_dir)
    assert rc == cli.EXIT_OK
    lines = text.splitlines()
    assert lines[0] == "Radient Vault agent access · Acme"
    assert text.count("Acme") == 1  # the org is printed once
    assert f"  {wire.fingerprint_grid(keys.fingerprint_for_seed(SEED))[3]}" in lines
    assert lines[-2].startswith("Connected until ")


@pytest.mark.parametrize(
    ("state", "last"),
    [
        ("unavailable", "Agent access isn't available yet."),
        ("signed_out", "Sign in to Radient first: run `lop login radient`."),
    ],
)
def test_status_with_a_closed_gate_never_says_connected_nor_advises_connect(
    config_dir, wired, monkeypatch, state, last
) -> None:
    """A key and an unexpired cached grant, but the gate closed (design D2)."""
    wired["observation"] = Observation(state=state, tenant_name="Acme")
    wired["gate"] = None
    monkeypatch.setattr(keys, "load_seed", lambda endpoint, base=None: SEED)
    monkeypatch.setattr(
        "local_operator.team_bridge.store.Store.active_owner_readonly",
        lambda self: SimpleNamespace(tenant_id=TENANT, endpoint_id=ENDPOINT),
    )
    _grant(config_dir)
    rc, text, _err = _status(config_dir)
    assert rc == cli.EXIT_OK
    assert text.splitlines()[-1] == last  # the reason is the line people act on
    assert "Connected" not in text and "lop vault connect" not in text
    assert "this machine still holds this key" in text and "Acme" in text
    _rc, payload, _err = _status(config_dir, as_json=True)
    assert '"connected": false' in payload and '"available": false' in payload


def test_status_json_reports_connected(config_dir, wired, monkeypatch) -> None:
    monkeypatch.setattr(keys, "load_seed", lambda endpoint, base=None: SEED)
    _grant(config_dir)
    import json

    _rc, payload, _err = _status(config_dir, as_json=True)
    data = json.loads(payload)
    assert data["connected"] is True and data["delegation_id"] == "vdl_x"


def test_status_with_a_damaged_key_gives_a_remedy_not_a_traceback(
    config_dir, wired, monkeypatch
) -> None:
    from local_operator.secrets.errors import SecretCorrupt

    def damaged(endpoint, base=None):
        raise SecretCorrupt("damaged")

    monkeypatch.setattr(keys, "load_seed", damaged)
    rc, text, err = _status(config_dir)
    assert rc == cli.EXIT_OK and text == ""
    assert "lop vault disconnect" in err and "lop vault connect" in err


def test_disconnect_deletes_every_key_and_points_at_the_console(config_dir) -> None:
    """Every LOP_VAULT_DELEGATION_* row, with NO Team Bridge owner (security S-2):
    the current endpoint's and a stale endpoint's alike, in a real store."""
    keys.load_or_create_seed(ENDPOINT, config_dir)
    keys.load_or_create_seed("bep_old", config_dir)
    _grant(config_dir)
    out = io.StringIO()
    assert cli.disconnect(config_dir, out=out) == cli.EXIT_OK
    assert keys.load_seed(ENDPOINT, config_dir) is None
    assert keys.load_seed("bep_old", config_dir) is None
    assert not (eligibility.vault_dir(config_dir) / eligibility.GRANT_NAME).exists()
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("Removed 2 agent keys from this machine.")
    assert lines[-1] == "https://console.radienthq.com/dashboard/vaults/agents"
    # Nothing left: the truthful zero.
    out = io.StringIO()
    assert cli.disconnect(config_dir, out=out) == cli.EXIT_OK
    assert out.getvalue().splitlines()[0] == "This machine has no agent key to remove."


def test_disconnect_reports_a_store_failure_without_a_traceback(config_dir, monkeypatch) -> None:
    from local_operator.secrets.errors import SecretStoreError

    def broken(base=None):
        raise SecretStoreError("broker down")

    monkeypatch.setattr(keys, "delete_all_seeds", broken)
    out, err = io.StringIO(), io.StringIO()
    assert cli.disconnect(config_dir, out=out, err=err) == cli.EXIT_NOT_CONNECTED
    assert "no agent key was removed" in err.getvalue() and out.getvalue() == ""


def test_english_is_pinned_for_these_assertions() -> None:
    assert active_locale() == "en"


def test_a_title_without_an_organization_name_never_shows_the_tenant_id(
    config_dir, wired, monkeypatch
) -> None:
    """Design round 2, D9: signed out, a key on disk, no name known."""
    wired["observation"] = Observation(state="signed_out")
    wired["gate"] = None
    monkeypatch.setattr(keys, "load_seed", lambda endpoint, base=None: SEED)
    monkeypatch.setattr(
        "local_operator.team_bridge.store.Store.active_owner_readonly",
        lambda self: SimpleNamespace(tenant_id=TENANT, endpoint_id=ENDPOINT),
    )
    _rc, text, _err = _status(config_dir)
    assert text.splitlines()[0] == "Radient Vault agent access"
    assert TENANT not in text
