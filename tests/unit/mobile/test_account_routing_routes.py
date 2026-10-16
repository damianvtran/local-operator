"""Whitelisted account-policy API on the actual authenticated relay app."""

from starlette.testclient import TestClient

from local_operator.config import CONFIG_FILE_NAME, ConfigManager
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir
from local_operator.providers.account_policy import effective_account_selection_policy

URL = "/api/settings/account-routing"


def client():
    instance = TestClient(
        build_app(MobileDaemon(port=0, password="synthetic-routing-password")),
        follow_redirects=False,
    )
    assert instance.post("/login", data={"password": "synthetic-routing-password"}).status_code in (
        200,
        303,
    )
    return instance


def test_auth_negative_then_canonical_persisted_save_and_restart():
    anonymous = TestClient(build_app(MobileDaemon(port=0, password="synthetic-routing-password")))
    assert anonymous.get(URL).status_code == 401
    assert anonymous.patch(URL, json={"value": "fair"}).status_code == 401
    api = client()
    initial = api.get(URL)
    assert initial.status_code == 200
    assert initial.headers["cache-control"] == "no-store"
    assert initial.json()["capability"] == "account-routing-v1"
    assert initial.json()["setting"]["value"] == "reset-soonest"
    assert {row["value"] for row in initial.json()["setting"]["choices"]} == {
        "reset-soonest",
        "headroom",
        "fair",
    }
    saved = api.patch(URL, json={"value": "fair"})
    assert saved.status_code == 200
    assert saved.json()["setting"]["value"] == "fair"
    manager = ConfigManager(config_dir())
    assert effective_account_selection_policy(manager.get_config().values) == "fair"
    restarted = client()
    assert restarted.get(URL).json()["setting"]["value"] == "fair"


def test_refused_values_arbitrary_config_and_origin_leave_bytes_unchanged():
    api = client()
    assert api.patch(URL, json={"value": "headroom"}).status_code == 200
    path = config_dir() / CONFIG_FILE_NAME
    original = path.read_bytes()
    for patch in (
        {"value": True},
        {"value": "wrong"},
        {"value": None},
        {"value": "fair", "key": "approvals.mode"},
        {"retry": {}},
        {},
    ):
        response = api.patch(URL, json=patch)
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "invalid_setting"
        assert path.read_bytes() == original
    denied = api.patch(URL, json={"value": "fair"}, headers={"Origin": "https://evil.example"})
    assert denied.status_code == 403
    assert path.read_bytes() == original
    assert api.patch("/api/settings/approvals.mode", json={"value": "off"}).status_code == 404


def test_permission_rejection_keeps_prior_effective_policy_and_can_recover():
    api = client()
    assert api.patch(URL, json={"value": "headroom"}).status_code == 200
    root = config_dir()
    path = root / CONFIG_FILE_NAME
    original = path.read_bytes()
    root.chmod(0o500)
    try:
        response = api.patch(URL, json={"value": "fair"})
        assert response.status_code == 503
        assert response.json()["code"] == "setting_write_failed"
        assert str(root) not in response.text
        assert path.read_bytes() == original
        assert api.get(URL).json()["setting"]["value"] == "headroom"
    finally:
        root.chmod(0o700)
    assert api.patch(URL, json={"value": "fair"}).json()["setting"]["value"] == "fair"


def test_unreadable_config_is_not_quarantined_or_overwritten():
    api = client()
    assert api.get(URL).status_code == 200
    path = config_dir() / CONFIG_FILE_NAME
    path.write_text("retry: [invalid YAML")
    original = path.read_bytes()
    for response in (api.get(URL), api.patch(URL, json={"value": "fair"})):
        assert response.status_code == 409
        assert response.json()["code"] == "config_unreadable"
        assert path.read_bytes() == original
    assert not list(path.parent.glob("*.corrupt.*"))
