"""SSR login copy, its fail-closed extraction gate, and unchanged auth routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from local_operator.i18n import catalogues
from local_operator.i18n.messages import render
from local_operator.mobile import daemon
from scripts.i18n.login import check_login_source

ROOT = Path(__file__).resolve().parents[3]


def test_english_login_and_auth_contract() -> None:
    client = TestClient(
        daemon.build_app(daemon.MobileDaemon(port=0, password="fixture-password")),
        follow_redirects=False,
    )
    page = client.get("/login")
    assert page.status_code == 200
    assert '<html lang="en" dir="ltr">' in page.text
    assert "<title>local operator — sign in</title>" in page.text
    assert '<label for="password">password</label>' in page.text
    assert '<button type="submit">sign in</button>' in page.text
    assert "__TITLE__" not in page.text
    assert 'name="password"' in page.text and 'autocomplete="current-password"' in page.text
    assert "lo-mobile-command:" in page.text and "lo-mobile-draft:" in page.text
    assert client.get("/api/sessions").status_code == 401
    refusal = client.post("/login", data={"password": "wrong"})
    assert refusal.status_code == 401
    assert '<p class="error" role="alert">Wrong password.</p>' in refusal.text
    assert (
        client.post(
            "/login",
            data={"password": "fixture-password"},
            headers={"origin": "https://other.invalid"},
        ).status_code
        == 403
    )
    success = client.post("/login", data={"password": "fixture-password"})
    assert success.status_code == 303 and success.headers["location"] == "/"
    assert "HttpOnly" in success.headers["set-cookie"]
    assert "SameSite=lax" in success.headers["set-cookie"]
    assert client.get("/api/sessions").json()["sessions"] == []
    assert client.get("/login").status_code == 303
    assert client.get("/logout").headers["clear-site-data"] == '"storage"'
    assert client.get("/api/sessions").status_code == 401


def test_single_backend_locale_live_resolution_and_escaping(monkeypatch, tmp_path) -> None:
    """Synthetic unshipped translations exercise SSR mechanics, not shipment."""
    target = tmp_path / "ur"
    (tmp_path / "en").mkdir()
    (tmp_path / "en/relay.login.json").write_text(
        (ROOT / "local_operator/i18n/catalogues/en/relay.login.json").read_text()
    )
    target.mkdir()
    source = json.loads((ROOT / "local_operator/i18n/catalogues/en/relay.login.json").read_text())
    for key in source:
        source[key] = '<img src=x onerror="alert(1)"> & __SUBMIT__'
    (target / "relay.login.json").write_text(json.dumps(source))
    monkeypatch.setattr(catalogues, "catalogue_root", lambda: tmp_path)
    calls = []
    monkeypatch.setattr(daemon, "resolve_language", lambda: calls.append("ur") or "ur")
    rendered = daemon._render_login_html(error=True)
    assert calls == ["ur"]
    assert '<html lang="ur" dir="rtl">' in rendered
    assert "<img src=x" not in rendered
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; __SUBMIT__" in rendered
    monkeypatch.setattr(daemon, "resolve_language", lambda: "en")
    assert '<html lang="en" dir="ltr">' in daemon._render_login_html()


def test_source_context_and_generated_key_parity() -> None:
    path = ROOT / "local_operator/i18n/catalogues/en"
    messages = json.loads((path / "relay.login.json").read_text())
    context = json.loads((path / "relay.login.context.json").read_text())
    assert set(messages) == set(context)
    assert (
        render(daemon.relay_login.error_password().code, {}, locale="en")
        == messages["relay.login.error.password"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "<p>New hardcoded login copy</p>",
        '<input placeholder="New password hint">',
        '<button aria-label="New accessible name">__SUBMIT__</button>',
        '<script>document.title = "New script copy";</script>',
        "<button>local operator</button>",
    ],
)
def test_scanner_mutations_fail(mutation: str) -> None:
    source = (ROOT / "local_operator/mobile/daemon.py").read_text()
    assert check_login_source(source) == []
    mutant = source.replace("</form>", mutation + "</form>")
    findings = check_login_source(mutant)
    assert findings and all("relay.login: hardcoded" in finding for finding in findings)


def test_renderer_literal_mutation_fails() -> None:
    source = (ROOT / "local_operator/mobile/daemon.py").read_text()
    source = source.replace("text(relay_login.error_password())", '"Hardcoded refusal"')
    assert check_login_source(source)


def test_mutation_fails_normal_checker(i18n_check, monkeypatch, tmp_path) -> None:
    path = tmp_path / "local_operator/mobile/daemon.py"
    path.parent.mkdir(parents=True)
    source = (ROOT / "local_operator/mobile/daemon.py").read_text()
    path.write_text(source.replace("</form>", "<p>New login copy</p></form>"))
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    assert i18n_check.main([]) == 1


def test_scanner_missing_or_dynamic_template_fails() -> None:
    assert check_login_source("_LOGIN_HTML = build_template()")
    assert check_login_source("x = 'not the template'")
