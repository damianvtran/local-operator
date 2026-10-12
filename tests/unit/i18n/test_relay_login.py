"""SSR login copy, its fail-closed extraction gate, and unchanged auth routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from local_operator.i18n import catalogues, resolve
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


@pytest.mark.parametrize("locale", ["en", "fr", "es", "ru", "vi"])
def test_actual_shipped_missing_namespace_uses_english_metadata(
    monkeypatch, tmp_path, locale: str
) -> None:
    # Real resolver/catalogues, not a mocked language string: these are the
    # committed shipped tags while relay.login currently carries English only.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text(f"values:\n  language: {locale}\n")
    assert daemon.resolve_language() == locale
    client = TestClient(
        daemon.build_app(daemon.MobileDaemon(port=0, password="fixture-password")),
        follow_redirects=False,
    )
    for response in [client.get("/login"), client.post("/login", data={"password": "wrong"})]:
        assert '<html lang="en" dir="ltr">' in response.text
        assert "<title>local operator — sign in</title>" in response.text
        assert '<label for="password">password</label>' in response.text
        assert '<button type="submit">sign in</button>' in response.text
    assert client.get("/login").status_code == 200
    refused = client.post("/login", data={"password": "wrong"})
    assert refused.status_code == 401
    assert '<p class="error" role="alert">Wrong password.</p>' in refused.text


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
    # The production runtime now gates every explicit render locale through
    # shared eligibility. Make only this synthetic control eligible so it
    # exercises escaping/RTL rather than a forbidden held-catalogue preview.
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "ur"))
    calls = []
    monkeypatch.setattr(daemon, "resolve_language", lambda: calls.append("ur") or "ur")
    rendered = daemon._render_login_html(error=True)
    assert calls == ["ur"]
    assert '<html lang="ur" dir="rtl">' in rendered
    assert "<img src=x" not in rendered
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &amp; __SUBMIT__" in rendered
    monkeypatch.setattr(daemon, "resolve_language", lambda: "en")
    assert '<html lang="en" dir="ltr">' in daemon._render_login_html()


def test_present_namespace_honors_real_backend_locale(monkeypatch, tmp_path) -> None:
    # Synthetic catalogue root, not production translation/ledger shipment.
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "fr").mkdir()
    english = (ROOT / "local_operator/i18n/catalogues/en/relay.login.json").read_text()
    (root / "en/relay.login.json").write_text(english)
    translated = {
        "relay.login.page.title": "Connexion",
        "relay.login.password.label": "Mot de passe",
        "relay.login.submit.label": "Se connecter",
        "relay.login.error.password": "Mot de passe incorrect.",
    }
    (root / "fr/relay.login.json").write_text(json.dumps(translated))
    monkeypatch.setattr(catalogues, "catalogue_root", lambda: root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("values:\n  language: fr\n")
    assert daemon.resolve_language() == "fr"
    client = TestClient(
        daemon.build_app(daemon.MobileDaemon(port=0, password="fixture-password")),
        follow_redirects=False,
    )
    for response in [client.get("/login"), client.post("/login", data={"password": "wrong"})]:
        assert '<html lang="fr" dir="ltr">' in response.text
        assert "<title>Connexion</title>" in response.text
        assert '<label for="password">Mot de passe</label>' in response.text
        assert '<button type="submit">Se connecter</button>' in response.text
    assert "Mot de passe incorrect." in client.post("/login", data={"password": "wrong"}).text


@pytest.mark.parametrize(
    "shape",
    [
        "empty",
        "title_only",
        "missing_title",
        "missing_password",
        "missing_submit",
        "missing_error",
        "invalid",
    ],
)
def test_incomplete_namespace_falls_back_as_whole_login(monkeypatch, tmp_path, shape: str) -> None:
    # Generated identities include the refusal even on GET: otherwise a form
    # could switch language only after the first failed authentication attempt.
    title = daemon.relay_login.page_title().code
    codes = {
        "title": title,
        "password": daemon.relay_login.password_label().code,
        "submit": daemon.relay_login.submit_label().code,
        "error": daemon.relay_login.error_password().code,
    }
    candidate: dict[str, str | int] = {code: "Synthetic French fixture" for code in codes.values()}
    if shape == "empty":
        candidate = {}
    elif shape == "title_only":
        candidate = {title: "Synthetic French fixture"}
    elif shape == "invalid":
        candidate = {title: 42}
    else:
        candidate.pop(codes[shape.removeprefix("missing_")])
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "fr").mkdir()
    (root / "en/relay.login.json").write_text(
        (ROOT / "local_operator/i18n/catalogues/en/relay.login.json").read_text()
    )
    (root / "fr/relay.login.json").write_text(json.dumps(candidate))
    monkeypatch.setattr(catalogues, "catalogue_root", lambda: root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("values:\n  language: fr\n")
    assert daemon.resolve_language() == "fr"
    client = TestClient(
        daemon.build_app(daemon.MobileDaemon(port=0, password="fixture-password")),
        follow_redirects=False,
    )
    page = client.get("/login")
    refusal = client.post("/login", data={"password": "wrong"})
    assert page.status_code == 200 and refusal.status_code == 401
    for response in [page, refusal]:
        assert '<html lang="en" dir="ltr">' in response.text
        assert "<title>local operator — sign in</title>" in response.text
        assert '<label for="password">password</label>' in response.text
        assert '<button type="submit">sign in</button>' in response.text
        assert "Synthetic French fixture" not in response.text
    assert '<p class="error" role="alert">Wrong password.</p>' in refusal.text
    assert daemon.resolve_language() == "fr"


def test_held_namespace_file_cannot_override_backend_eligibility(monkeypatch, tmp_path) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "zh-CN").mkdir()
    source = (ROOT / "local_operator/i18n/catalogues/en/relay.login.json").read_text()
    (root / "en/relay.login.json").write_text(source)
    held = {key: "Held catalogue fixture" for key in json.loads(source)}
    (root / "zh-CN/relay.login.json").write_text(json.dumps(held))
    monkeypatch.setattr(catalogues, "catalogue_root", lambda: root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("values:\n  language: zh-CN\n")
    assert daemon.resolve_language() == "en"
    rendered = daemon._render_login_html(error=True)
    assert '<html lang="en" dir="ltr">' in rendered
    assert "Held catalogue fixture" not in rendered
    assert "Wrong password." in rendered


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


@pytest.mark.parametrize(
    "script",
    [
        'const help = "https://example.invalid"; document.title = "Sign in now";',
        'document.title = "Sign in // now";',
        "document.title = `// Sign in now`;",
        'document.title = "Sign in /* now */";',
        r'document.title = "Sign in \"// now";',
        r"document.title = 'Sign in \'// now';",
        r"document.title = `Sign in \`// now`;",
        '// comment with "quotes"\n document.title = "Sign in now";',
        '/* comment with "quotes" */ document.title = "Sign in now";',
    ],
)
def test_script_comment_escape_mutations_fail_normal_gate(
    script: str, i18n_check, monkeypatch, tmp_path
) -> None:
    # Repr preserves JS backslashes through the outer Python HTML constant;
    # an escaped quote must reach the lexer exactly as the browser receives it.
    template = daemon._LOGIN_HTML.replace("</form>", f"<script>{script}</script></form>")
    source = f"_LOGIN_HTML = {template!r}"
    findings = check_login_source(source)
    assert findings and all("hardcoded script literal" in finding for finding in findings)
    path = tmp_path / "local_operator/mobile/daemon.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    assert i18n_check.main([]) == 1


def test_script_comments_and_allowlisted_keys_still_pass_normal_gate(
    i18n_check, monkeypatch, tmp_path
) -> None:
    script = (
        '// "Developer comment"\n'
        "/* `Another developer comment` */ "
        'const keys = ["lo-mobile-command:", "lo-mobile-draft:"];'
    )
    template = daemon._LOGIN_HTML.replace("</form>", f"<script>{script}</script></form>")
    source = f"_LOGIN_HTML = {template!r}"
    assert check_login_source(source) == []
    path = tmp_path / "local_operator/mobile/daemon.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    assert i18n_check.main([]) == 0


def test_renderer_literal_mutation_fails() -> None:
    source = (ROOT / "local_operator/mobile/daemon.py").read_text()
    source = source.replace('text(messages["ERROR"])', '"Hardcoded refusal"')
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
