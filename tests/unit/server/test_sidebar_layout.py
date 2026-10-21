"""The sidebar-layout preference: the store, the route pair, and the gate.

WHAT IS PINNED HERE, and why each half earns a test:

* the STORE — a missing file answers the shipped arrangement; a full record
  round-trips byte-for-byte; a subset is stored as given; every corruption
  (bad JSON, a non-object, an unknown version, an unknown or duplicated
  destination, a missing region) degrades to the defaults rather than costing
  the sidebar its column. Reads fail OPEN by contract (design §4).
* the ROUTE — the response shape is ``{layout: {version, visible, more}}``,
  a read never creates the file, validation refusals are 422 and do not touch
  the stored record, and a write that raises ``OSError`` is a 500 with a
  machine code (``sidebar_layout_write_failed``), never a 200 the user was
  told succeeded.
* the GATE — the router carries ``require_desktop``: no environment token is
  a 503, a missing bearer is a 401, for BOTH methods.
* the NEGOTIATION — ``feature_flags()["sidebar_layout"] == 1``; a client that
  cannot see it must not offer customisation at all.

The route tests run against a REAL temporary config directory through the real
router, because the file writes are the behaviour under test. ``tests/
unit/server/test_desktop_controls.py`` separately walks every ``/v1/desktop``
route of the REAL app and proves this one is gated there too; the 401/503 here
prove the dependency reaches the handler in an isolated app.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from local_operator.config import ConfigManager
from local_operator.server import desktop as desktop_module
from local_operator.server.features import feature_flags
from local_operator.server.routes import desktop_preferences
from local_operator.server.sidebar_layout import (
    DEFAULT_SIDEBAR_LAYOUT,
    SIDEBAR_DESTINATION_IDS,
    SIDEBAR_LAYOUT_FILE,
    SidebarLayout,
    default_layout,
    read_layout,
    write_layout,
)
from local_operator.session.store_failures import display_root

TOKEN = "synthetic-sidebar-layout-token"

SHIPPED = {
    "version": 1,
    "visible": ["aida", "library", "projects", "agents", "schedules", "browser"],
    "more": ["team-bridge", "agent-hub", "mesh"],
}


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_the_shipped_arrangement_is_the_design_table() -> None:
    """The constants ARE the design's table (design §2), not a second memory."""
    assert SIDEBAR_DESTINATION_IDS == (
        "aida",
        "library",
        "projects",
        "agents",
        "schedules",
        "browser",
        "team-bridge",
        "agent-hub",
        "mesh",
    )
    assert DEFAULT_SIDEBAR_LAYOUT == {"visible": SHIPPED["visible"], "more": SHIPPED["more"]}
    assert read_layout(Path("/nonexistent-config-root")) == SHIPPED


def test_the_defaults_are_a_fresh_copy_each_call(tmp_path: Path) -> None:
    """One caller mutating a read result must not redefine later defaults."""
    first = read_layout(tmp_path)
    first["visible"].append("mesh")
    first["more"].clear()
    assert read_layout(tmp_path) == SHIPPED
    assert default_layout() == SHIPPED


def test_a_full_record_round_trips_through_the_file(tmp_path: Path) -> None:
    written = {"visible": ["library", "aida"], "more": ["mesh", "browser"]}
    stored = write_layout(tmp_path, written)
    assert stored == {"version": 1, **written}
    on_disk = json.loads((tmp_path / SIDEBAR_LAYOUT_FILE).read_text(encoding="utf-8"))
    assert on_disk == stored
    assert read_layout(tmp_path) == stored


def test_a_subset_is_stored_as_given(tmp_path: Path) -> None:
    """The client fills missing ids from the default (design §4), not the server."""
    stored = write_layout(tmp_path, {"visible": ["browser"], "more": []})
    assert stored == {"version": 1, "visible": ["browser"], "more": []}
    assert read_layout(tmp_path) == stored


@pytest.mark.parametrize(
    ("raw", "because"),
    [
        ("{not json", "malformed JSON"),
        ("[]", "not an object"),
        ('{"version": 2, "visible": [], "more": []}', "a newer schema version"),
        ('{"version": 1, "visible": ["nope"], "more": []}', "an unknown destination"),
        ('{"version": 1, "visible": ["aida", "aida"], "more": []}', "a duplicate"),
        ('{"version": 1, "visible": ["aida"], "more": ["aida"]}', "a cross-region duplicate"),
        ('{"version": 1, "visible": ["aida"]}', "a missing region"),
        ('{"version": 1, "visible": null, "more": []}', "a null region"),
        ('{"version": 1, "visible": "aida", "more": []}', "a scalar region"),
    ],
)
def test_a_bad_file_degrades_to_the_defaults(
    tmp_path: Path, raw: str, because: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Every corruption answers the shipped order, and says so in the log.

    A corrupt preference must not cost the column (design §4): the read is
    best-effort, and the worst outcome is today's arrangement. The warning is
    pinned because silent degradation hides a persistent corruption forever.
    """
    (tmp_path / SIDEBAR_LAYOUT_FILE).write_text(raw, encoding="utf-8")
    with caplog.at_level("WARNING", logger="local_operator.server.sidebar_layout"):
        assert read_layout(tmp_path) == SHIPPED, because
    assert SIDEBAR_LAYOUT_FILE in caplog.text, because


def test_an_unreadable_file_degrades_to_the_defaults(tmp_path: Path) -> None:
    """A directory where the file belongs is a read failure, not a 500."""
    (tmp_path / SIDEBAR_LAYOUT_FILE).mkdir()
    assert read_layout(tmp_path) == SHIPPED


def test_a_write_does_not_leave_a_temporary_file_behind(tmp_path: Path) -> None:
    write_layout(tmp_path, {"visible": ["aida"], "more": []})
    assert sorted(path.name for path in tmp_path.iterdir()) == [SIDEBAR_LAYOUT_FILE]


def test_a_failed_replace_removes_the_temporary_file(tmp_path: Path, monkeypatch) -> None:
    """The cleanup arm is the one that matters, and only a failure reaches it.

    The success test above cannot exercise ``write_layout``'s
    ``except BaseException: os.unlink`` branch. A surviving
    ``.sidebar-layout.json.*.tmp`` is litter in the config root and a file a
    later scan could mistake for state, so the branch is pinned directly
    (agent review round 1, R1-5).
    """

    def refuse(_source: str, _target: Path) -> None:
        raise OSError("replace refused")

    monkeypatch.setattr("local_operator.server.sidebar_layout.os.replace", refuse)
    with pytest.raises(OSError):
        write_layout(tmp_path, {"visible": ["aida"], "more": []})
    assert sorted(tmp_path.iterdir()) == []


def test_the_store_refuses_a_version_it_does_not_write(tmp_path: Path) -> None:
    """An echoed GET version is accepted; a future one is refused, not ignored."""
    with pytest.raises(ValidationError):
        write_layout(tmp_path, {"version": 2, "visible": ["aida"], "more": []})
    stored = write_layout(tmp_path, {"version": 1, "visible": ["aida"], "more": []})
    assert stored == {"version": 1, "visible": ["aida"], "more": []}


def test_an_invalid_layout_is_never_persisted(tmp_path: Path) -> None:
    """The store validates too, so no caller can persist what the reader rejects."""
    with pytest.raises(ValidationError):
        write_layout(tmp_path, {"visible": ["nope"], "more": []})
    assert not (tmp_path / SIDEBAR_LAYOUT_FILE).exists()


def test_the_model_refuses_more_ids_than_there_are_destinations() -> None:
    """Ten entries cannot be ten known ids; the bound is pinned anyway.

    The duplicate guard is what rejects this list — with nine known ids a tenth
    entry must repeat one. The explicit bound exists so the invariant does not
    depend on a later reader noticing that inference.
    """
    assert len(SIDEBAR_DESTINATION_IDS) == 9
    with pytest.raises(ValidationError):
        SidebarLayout(visible=list(SIDEBAR_DESTINATION_IDS) + ["aida"], more=[])


# ---------------------------------------------------------------------------
# The route pair
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def desktop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An isolated config root, the real router, and the desktop bearer.

    ``_CLAIMED`` is cleared because it is one-way process state: a sibling test
    that accepted a claim would otherwise keep this plane open with a token
    this test never set, and the "unmanaged" case below would be order- (and
    xdist-worker-) dependent.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setattr(desktop_module, "_CLAIMED", None)
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name, raising=False)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.include_router(desktop_preferences.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path


@pytest.mark.asyncio
async def test_the_read_answers_the_defaults_and_creates_nothing(desktop) -> None:
    client, root = desktop
    response = await client.get("/v1/desktop/sidebar-layout")
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"layout": SHIPPED}
    assert not (root / SIDEBAR_LAYOUT_FILE).exists()


@pytest.mark.asyncio
async def test_the_put_stores_the_record_and_the_read_returns_it(desktop) -> None:
    client, root = desktop
    wanted = {"visible": ["browser", "aida"], "more": ["mesh"]}
    response = await client.put("/v1/desktop/sidebar-layout", json={"layout": wanted})
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"layout": {"version": 1, **wanted}}
    on_disk = json.loads((root / SIDEBAR_LAYOUT_FILE).read_text(encoding="utf-8"))
    assert on_disk == {"version": 1, **wanted}
    assert (await client.get("/v1/desktop/sidebar-layout")).json()["result"] == {
        "layout": {"version": 1, **wanted}
    }


@pytest.mark.asyncio
async def test_an_empty_layout_is_a_valid_well_formed_preference(desktop) -> None:
    """Everything folded into More is a real arrangement (design §3), not an error."""
    client, _root = desktop
    response = await client.put(
        "/v1/desktop/sidebar-layout", json={"layout": {"visible": [], "more": ["aida"]}}
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["layout"]["visible"] == []


@pytest.mark.asyncio
async def test_an_unknown_destination_is_a_422_and_changes_nothing(desktop) -> None:
    client, root = desktop
    assert (
        await client.put(
            "/v1/desktop/sidebar-layout", json={"layout": {"visible": ["nope"], "more": []}}
        )
    ).status_code == 422
    assert not (root / SIDEBAR_LAYOUT_FILE).exists()


@pytest.mark.asyncio
async def test_a_duplicate_destination_is_a_422(desktop) -> None:
    client, _root = desktop
    body = {"layout": {"visible": ["aida"], "more": ["aida"]}}
    response = await client.put("/v1/desktop/sidebar-layout", json=body)
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_more_ids_than_destinations_is_a_422(desktop) -> None:
    client, _root = desktop
    body = {"layout": {"visible": list(SIDEBAR_DESTINATION_IDS), "more": ["aida"]}}
    response = await client.put("/v1/desktop/sidebar-layout", json=body)
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_a_missing_region_is_a_422(desktop) -> None:
    """Both regions are the contract; an omitted one must not mean "reset it"."""
    client, _root = desktop
    response = await client.put("/v1/desktop/sidebar-layout", json={"layout": {"visible": []}})
    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_the_read_answer_can_be_sent_back_as_the_write_body(desktop) -> None:
    """A read-modify-write client round-trips the document it was handed.

    ``GET`` answers ``{version, visible, more}``; ``PUT`` used to refuse that
    exact object because ``version`` was an unknown key, so the route's own
    answer was invalid input (agent review round 1, R1-3). The echoed version
    is validated and never stored from the body.
    """
    client, _root = desktop
    read = await client.get("/v1/desktop/sidebar-layout")
    assert read.status_code == 200, read.text
    layout = read.json()["result"]["layout"]
    assert layout["version"] == 1
    written = await client.put("/v1/desktop/sidebar-layout", json={"layout": layout})
    assert written.status_code == 200, written.text
    assert written.json()["result"]["layout"] == layout
    assert (await client.get("/v1/desktop/sidebar-layout")).json()["result"]["layout"] == layout


@pytest.mark.asyncio
async def test_a_future_version_in_the_body_is_a_422(desktop) -> None:
    """Accepting the read answer must not become accepting any schema."""
    client, _root = desktop
    body = {"layout": {"version": 2, "visible": ["aida"], "more": []}}
    assert (await client.put("/v1/desktop/sidebar-layout", json=body)).status_code == 422


@pytest.mark.asyncio
async def test_a_failed_write_is_a_500_with_a_vetted_sentence(desktop) -> None:
    """A save the user was told succeeded and did not is worse than an error.

    The failure is REAL — a directory sits where the file belongs, so
    ``os.replace`` raises — which is what makes the cleanup assertion
    meaningful: the staged temp file must be gone, and the answer must be the
    route's vetted sentence naming the config root, never the exception text
    (agent review round 1, R1-2/R1-5).
    """
    client, root = desktop
    target = root / SIDEBAR_LAYOUT_FILE
    target.mkdir()
    response = await client.put(
        "/v1/desktop/sidebar-layout", json={"layout": {"visible": ["aida"], "more": []}}
    )
    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "sidebar_layout_write_failed"
    assert detail["text"] == desktop_preferences.write_refusal(root)
    assert display_root(root) in detail["text"]
    assert "Errno" not in detail["text"]
    assert target.is_dir()
    assert not [path for path in root.iterdir() if path.name.endswith(".tmp")]


@pytest.mark.asyncio
async def test_a_failed_write_logs_the_real_error(desktop, monkeypatch, caplog) -> None:
    """The client's copy never carries the exception; the log must (R1-2)."""
    client, _root = desktop

    def refuse(_config_dir: Path, _layout: dict[str, Any]) -> dict[str, Any]:
        raise OSError("No space left on device")

    monkeypatch.setattr(desktop_preferences, "write_layout", refuse)
    with caplog.at_level("ERROR", logger="local_operator.server.routes.desktop_preferences"):
        response = await client.put(
            "/v1/desktop/sidebar-layout", json={"layout": {"visible": ["aida"], "more": []}}
        )
    assert response.status_code == 500, response.text
    assert "No space left on device" not in response.json()["detail"]["text"]
    assert "No space left on device" in caplog.text
    assert any(record.exc_info for record in caplog.records if record.levelname == "ERROR")


def test_the_refusal_names_the_root_home_relative() -> None:
    """The sentence uses the sibling ``display_root`` spelling, not an absolute path.

    A refusal that says "the volume holding /Users/someone/.local-operator" is
    addressable but noisy, which is why every sibling names the root through
    ``display_root`` (agent review round 2, R2-2). The failure tests above use
    a temporary root, which is not under the home directory and so is left
    alone by that helper; this pins the shortening itself.
    """
    rooted = Path.home() / ".local-operator"
    sentence = desktop_preferences.write_refusal(rooted)
    assert "~/.local-operator" in sentence
    assert str(rooted) not in sentence
    assert "Retrying will not help" not in sentence


# ---------------------------------------------------------------------------
# The gate and the negotiation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "body"),
    [("get", None), ("put", {"layout": {"visible": [], "more": []}})],
)
async def test_the_router_requires_the_desktop_bearer(desktop, method, body) -> None:
    client, _root = desktop
    extra = {"json": body} if body else {}
    response = await getattr(client, method)(
        "/v1/desktop/sidebar-layout", headers={"Authorization": ""}, **extra
    )
    assert response.status_code == 401, response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "body"),
    [("get", None), ("put", {"layout": {"visible": [], "more": []}})],
)
async def test_the_router_requires_the_desktop_environment(
    desktop, monkeypatch, method, body
) -> None:
    """A standalone daemon did not open this plane; the bearer it carries is noise."""
    client, _root = desktop
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_TOKEN")
    response = await getattr(client, method)(
        "/v1/desktop/sidebar-layout", **({"json": body} if body else {})
    )
    assert response.status_code == 503, response.text


def test_the_capability_is_advertised_at_version_one() -> None:
    """Absent means "do not offer customisation"; a future field is additive."""
    assert feature_flags()["sidebar_layout"] == 1
