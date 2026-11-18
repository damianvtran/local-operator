"""ADR 0008 slice 1: the phone reads and writes ``notifications.phone_away_after_s``.

The setting describes the COMPUTER (how long its desk may sit idle before a
completion goes to the phone), so it lives in core config, and the phone edits
it per computer through this relay route (slice 6 renders the row). Pinned here,
on the real authenticated relay app:

- unauthenticated reads and writes are refused;
- the four ADR values persist through the same ``settings_io`` writer every
  other surface uses, and the core's own reader sees them;
- off-list values (90, 0, true, "120") are a 422 and leave the file untouched;
- the capability key the phone feature-detects is advertised.
"""

from starlette.testclient import TestClient

from local_operator import settings_io
from local_operator.config import CONFIG_FILE_NAME, ConfigManager
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir
from local_operator.session.runtime.presence import configured_away_after_s

URL = "/api/settings/phone-away-after"
PASSWORD = "synthetic-away-password"


def client() -> TestClient:
    instance = TestClient(
        build_app(MobileDaemon(port=0, password=PASSWORD)), follow_redirects=False
    )
    assert instance.post("/login", data={"password": PASSWORD}).status_code in (200, 303)
    return instance


def test_unauthenticated_reads_and_writes_are_refused() -> None:
    anonymous = TestClient(build_app(MobileDaemon(port=0, password=PASSWORD)))
    assert anonymous.get(URL).status_code == 401
    assert anonymous.patch(URL, json={"value": 300}).status_code == 401


def test_default_then_each_choice_persists_and_the_core_reader_sees_it() -> None:
    api = client()
    initial = api.get(URL)
    assert initial.status_code == 200, initial.text
    assert initial.headers["cache-control"] == "no-store"
    body = initial.json()
    assert body["capability"] == "phone-away-after-v1"
    assert body["setting"]["key"] == "notifications.phone_away_after_s"
    assert body["setting"]["value"] == 120
    assert body["setting"]["is_default"] is True
    assert [row["value"] for row in body["setting"]["choices"]] == [60, 120, 300, 600]

    for value in (60, 300, 600, 120):
        saved = api.patch(URL, json={"value": value})
        assert saved.status_code == 200, saved.text
        assert saved.json()["setting"]["value"] == value
        values = ConfigManager(config_dir()).get_config().values
        assert configured_away_after_s(values) == value
    assert client().get(URL).json()["setting"]["value"] == 120, "survives a restart"


def test_refused_values_leave_the_config_bytes_unchanged() -> None:
    api = client()
    assert api.patch(URL, json={"value": 300}).status_code == 200
    path = config_dir() / CONFIG_FILE_NAME
    original = path.read_bytes()
    for patch in (
        {"value": 90},
        {"value": 0},
        {"value": True},
        {"value": "120"},
        {"value": 120, "key": "anything.else"},
        {},
    ):
        refused = api.patch(URL, json=patch)
        assert refused.status_code == 422, (patch, refused.text)
        assert refused.json()["code"] == "invalid_setting"
    not_json = api.patch(URL, content=b"{", headers={"content-type": "application/json"})
    assert not_json.status_code == 422
    assert path.read_bytes() == original


def test_the_relay_advertises_the_capability() -> None:
    api = client()
    response = api.get("/api/sessions")
    assert response.status_code == 200, response.text
    capabilities = response.json()["capabilities"]
    assert capabilities["phone_away_after"] == {"available": True}


def test_the_reader_refuses_off_list_values() -> None:
    assert configured_away_after_s({}) == 120
    for raw in (90, 0, True, "300", None, 3600):
        assert configured_away_after_s({"notifications": {"phone_away_after_s": raw}}) == 120
    assert configured_away_after_s({"notifications": {"phone_away_after_s": 600}}) == 600
    assert settings_io.BY_KEY["notifications.phone_away_after_s"].default == 120
