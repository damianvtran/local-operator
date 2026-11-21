"""Discovery receipts must describe observations, not manufacture freshness.

The subprocess rows use the supported ``python -m local_operator`` entrypoint;
only registry transport and the installed-version input are synthetic. Installer,
bootstrap and service tripwires make a false success distinguishable from a
probe that accidentally exercised a helper instead of the real CLI route.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from local_operator import update


def _wheel(version: str, *, mac: bool = False) -> dict[str, Any]:
    tag = "py3-none-macosx_11_0_universal2" if mac else "py3-none-any"
    return {
        "filename": f"local_operator-{version}-{tag}.whl",
        "packagetype": "bdist_wheel",
        "url": "https://files.pythonhosted.org/packages/fixture.whl",
        "digests": {"sha256": "a" * 64},
        "yanked": False,
    }


def _document(version: str, *, ready: bool = True, mac: bool = False) -> dict[str, Any]:
    return {"info": {"version": version}, "urls": [_wheel(version, mac=mac)] if ready else []}


def _client(project, simple=None, release=None, *, project_status=200):
    requests = []

    def handler(request):
        requests.append(str(request.url))
        if str(request.url) == update.PYPI_JSON_URL:
            return httpx.Response(project_status, json=project)
        if str(request.url) == update.PYPI_SIMPLE_URL:
            return httpx.Response(200, json=simple if simple is not None else {"files": []})
        assert str(request.url).startswith("https://pypi.org/pypi/local-operator/")
        return httpx.Response(404 if release is None else 200, json=release)

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


@pytest.mark.parametrize("age", [0, update.TTL_S + 1])
def test_forced_outage_keeps_background_cache_but_marks_it_non_authoritative(tmp_path, age):
    update._write_cache(update._cache_path(tmp_path), {"version": "0.111.6", "ready": True})
    path = update._cache_path(tmp_path)
    data = json.loads(path.read_text())
    data["fetched_at"] -= age
    path.write_text(json.dumps(data))
    client, requests = _client({}, project_status=503)
    with client, patch.object(update, "installed_version", return_value="0.111.6"):
        result = update.check_latest(force=True, cache_dir=tmp_path, client=client)
    assert result.latest == "0.111.6"  # Background API contract is retained.
    assert result.provenance.source == "cache-fallback"
    assert result.provenance.forced
    assert result.provenance.live_attempted
    assert result.provenance.cache_age_seconds >= age
    assert result.provenance.fallback == "project-unavailable"
    assert [v.outcome for v in result.provenance.views] == ["http-error", "skipped", "skipped"]
    assert requests == [update.PYPI_JSON_URL]


def test_ttl_cache_still_skips_all_requests(tmp_path):
    update._write_cache(update._cache_path(tmp_path), {"version": "0.111.6", "ready": True})
    with patch.object(update, "_fetch_pypi_read", side_effect=AssertionError("network")):
        result = update.check_latest(cache_dir=tmp_path)
    assert result.provenance.source == "cache-ttl"
    assert not result.provenance.live_attempted
    assert result.provenance.views == ()


@pytest.mark.parametrize("mac", [False, True])
@pytest.mark.parametrize("ready", [False, True])
def test_answered_readiness_and_wheel_receipt(tmp_path, mac, ready):
    update._write_cache(update._cache_path(tmp_path), {"version": "0.110.0", "ready": True})
    client, requests = _client(_document("0.111.7", ready=ready, mac=mac))
    with client, patch.object(update, "_HOST_IS_MACOS", mac):
        result = update.check_latest(force=True, cache_dir=tmp_path, client=client)
    assert result.latest == ("0.111.7" if ready else None)
    assert result.pending_version == (None if ready else "0.111.7")
    assert result.provenance.readiness == ("ready" if ready else "pending")
    assert result.provenance.wheel_policy == ("macos-universal2" if mac else "host-compatible")
    assert result.provenance.wheel_filename == (
        _wheel("0.111.7", mac=mac)["filename"] if ready else None
    )
    assert result.provenance.selected_view == "project-json"
    assert requests == [update.PYPI_JSON_URL, update.PYPI_SIMPLE_URL]
    assert update._cache_path(tmp_path).exists() is ready


def test_mac_pure_wheel_pending_then_universal_ready(tmp_path):
    with patch.object(update, "_HOST_IS_MACOS", True):
        for mac in (False, True):
            client, _ = _client(_document("0.111.7", mac=mac))
            with client:
                result = update.check_latest(force=True, cache_dir=tmp_path, client=client)
            assert result.provenance.readiness == ("ready" if mac else "pending")


@pytest.mark.parametrize(
    "release",
    [
        None,
        {"info": {"version": "0.111.8"}},
        {"info": {"version": "0.111.7"}, "urls": []},
        _document("0.111.7"),
    ],
)
def test_simple_newer_confirmation_and_fallback(tmp_path, release):
    client, requests = _client(
        _document("0.111.6"),
        {"versions": [], "files": [{"filename": _wheel("0.111.7")["filename"], "yanked": False}]},
        release,
    )
    with client:
        result = update.check_latest(force=True, cache_dir=tmp_path, client=client)
    confirmed = release is not None and release["info"]["version"] == "0.111.7"
    ready = confirmed and release is not None and bool(release.get("urls"))
    assert result.latest == ("0.111.7" if ready else None if confirmed else "0.111.6")
    assert result.provenance.selected_view == ("release-json" if confirmed else "project-json")
    assert result.provenance.fallback == ("none" if confirmed else "release-unavailable")
    assert result.provenance.views[1].named_version == "0.111.7"
    assert len(requests) == 3


def test_project_releases_candidate_is_identified(tmp_path):
    payload = _document("0.111.6")
    payload["releases"] = {"0.111.7": [_wheel("0.111.7")]}
    client, _ = _client(payload)
    with client:
        result = update.check_latest(force=True, cache_dir=tmp_path, client=client)
    assert result.latest == "0.111.7"
    assert result.provenance.selected_view == "project-releases"
    assert result.provenance.views[0].named_version == "0.111.6"
    assert result.provenance.wheel_filename == _wheel("0.111.7")["filename"]


@pytest.mark.parametrize("failure", ["transport", "json", "shape"])
def test_failures_are_categorical_and_redacted(tmp_path, failure):
    secret = "SECRET_DO_NOT_PRINT?token=private\nINJECTED_HEADER"

    def handler(request):
        if failure == "transport":
            raise httpx.ConnectError(secret, request=request)
        if failure == "json":
            return httpx.Response(200, content=secret, headers={"Set-Cookie": secret})
        return httpx.Response(200, json={"info": {"untrusted": secret}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = update.check_latest(force=True, cache_dir=tmp_path, client=client)
    receipt = json.dumps(asdict(result.provenance))
    assert "SECRET" not in receipt and "private" not in receipt and "Cookie" not in receipt
    assert result.provenance.views[0].outcome == (
        "transport-error" if failure == "transport" else "parse-error"
    )
    assert result.provenance.source == "unavailable"


# This child is deliberately self-contained: no fixture's monkeypatch survives
# into it, and environment inheritance is replaced rather than filtered by name.
_CLI_PROBE = r"""
import json, os, runpy, subprocess, sys
from pathlib import Path
import httpx
from local_operator import update, update_bootstrap
scenario = sys.argv[1]
def tripwire(*args, **kwargs):
    raise AssertionError("INSTALL_SERVICE_BOOTSTRAP_TRIPWIRE")
update.perform_upgrade = tripwire
update._run_services_tail = tripwire
update_bootstrap.controller = tripwire
update_bootstrap.controller_tail = tripwire
subprocess.run = tripwire
subprocess.Popen = tripwire
if scenario == "tripwire-control":
    update.perform_upgrade(target="0.111.7")
update.installed_version = lambda: "0.111.6"
if scenario in ("outage", "outage-debug", "outage-default"):
    update._write_cache(update._cache_path(None), {"version": "0.111.6", "ready": True})
requests = []
def get(url, **kwargs):
    requests.append(url)
    assert url in (update.PYPI_JSON_URL, update.PYPI_SIMPLE_URL), "UNAPPROVED_ORIGIN"
    if "outage" in scenario:
        assert url == update.PYPI_JSON_URL, "UNEXPECTED_SECOND_PROBE"
        return httpx.Response(503, request=httpx.Request("GET", url))
    version = "0.111.5" if scenario == "ahead" else "0.111.7" if scenario == "newer" else "0.111.6"
    payload = {"files": []} if url == update.PYPI_SIMPLE_URL else {
        "info": {"version": version}, "urls": [] if scenario == "pending" else [{
            "filename": f"local_operator-{version}-py3-none-any.whl",
            "packagetype": "bdist_wheel", "url": "https://files.pythonhosted.org/fixture.whl",
            "digests": {"sha256": "a" * 64}, "yanked": False
        }]
    }
    return httpx.Response(200, json=payload, request=httpx.Request("GET", url))
httpx.get = get
print("SOURCE " + update.__file__, file=sys.stderr)
sys.argv = ["lop", "update"] + ([] if scenario == "outage-default" else ["--check"])
if scenario != "outage":
    sys.argv.append("--debug")
try:
    runpy.run_module("local_operator", run_name="__main__")
finally:
    print("REQUESTS " + json.dumps(requests), file=sys.stderr)
"""


@pytest.mark.parametrize(
    "scenario,code",
    [
        ("outage", 1),
        ("outage-debug", 1),
        ("outage-default", 1),
        ("outage-empty", 1),
        ("equal", 0),
        ("ahead", 0),
        ("newer", 2),
        ("pending", 1),
    ],
)
def test_real_module_cli_receipts_and_exit_codes(tmp_path, scenario, code):
    root = Path(update.__file__).resolve().parents[1]
    environment = {
        "HOME": str(tmp_path),
        "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / "config"),
        "PATH": os.defpath,
        "TERM": "xterm-256color",
        "PYTHONPATH": str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    process = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_CLI_PROBE), scenario],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == code, process.stdout + process.stderr
    assert str(root / "local_operator" / "update.py") in process.stderr
    assert "TRIPWIRE" not in process.stderr
    assert "REQUESTS " in process.stderr
    receipt_lines = [
        line for line in process.stdout.splitlines() if line.startswith("update-discovery ")
    ]
    assert len(receipt_lines) == (0 if scenario == "outage" else 1)
    if receipt_lines:
        receipt = json.loads(receipt_lines[0].removeprefix("update-discovery "))
        assert receipt["route"] == "pypi-discovery"
        assert receipt["forced"] is True
        assert receipt["live_attempted"] is True
        assert len(receipt["views"]) == 3
    if "outage" in scenario:
        assert "is the latest" not in process.stdout
        if scenario == "outage-empty":
            assert "could not reach PyPI" in process.stderr
            assert "unavailable" in process.stdout
        else:
            assert "not authoritative" in process.stderr
            assert "cache-fallback" in process.stdout
        assert process.stdout.splitlines()[-1].startswith('{"update_report":')
        report = json.loads(process.stdout.splitlines()[-1])["update_report"]
        assert report["failure"]["kind"] == "network"
        assert report["failure"]["exit_code"] == 1
    elif scenario == "equal":
        assert "0.111.6 is the latest" in process.stdout
    elif scenario == "ahead":
        assert "ahead of the observed PyPI version 0.111.5" in process.stdout
        assert "is the latest" not in process.stdout
    elif scenario == "newer":
        assert "latest on PyPI: 0.111.7" in process.stdout
    else:
        assert "still publishing" in process.stderr
        assert "pending" in process.stdout


def test_cli_installer_tripwire_negative_control(tmp_path):
    """A broken probe must fail loudly rather than make no-install look green."""
    root = Path(update.__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_CLI_PROBE), "tripwire-control"],
        env={
            "HOME": str(tmp_path),
            "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / "config"),
            "PATH": os.defpath,
            "TERM": "xterm-256color",
            "PYTHONPATH": str(root),
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 1
    assert "INSTALL_SERVICE_BOOTSTRAP_TRIPWIRE" in process.stderr
    assert "is the latest" not in process.stdout


@pytest.mark.parametrize(
    "bad", ["TOKEN_SENTINEL", "1.2.3\\nTOKEN_SENTINEL", "1." + "2" * 100 + ".3"]
)
def test_receipt_rejects_non_version_and_unbounded_values(bad):
    assert update._receipt_version(bad) is None


def test_live_wins_over_ready_cache_even_when_older(tmp_path):
    update._write_cache(update._cache_path(tmp_path), {"version": "0.111.9", "ready": True})
    client, _ = _client(_document("0.111.6"))
    with client, patch.object(update, "installed_version", return_value="0.111.6"):
        result = update.check_latest(force=True, cache_dir=tmp_path, client=client)
    assert result.latest == "0.111.6"
    assert result.provenance.source == "live"
    assert result.provenance.selected_version == "0.111.6"


def test_newer_release_key_receipt_names_selected_view():
    payload = _document("0.111.6")
    payload["releases"] = {"0.111.7": [_wheel("0.111.7")]}
    client, requests = _client(payload)
    with client:
        result = update._fetch_pypi_read(client=client)
    assert result.version == "0.111.7"
    assert result.provenance.selected_view == "project-releases"
    assert result.provenance.views[0].named_version == "0.111.6"
    assert result.provenance.named_version == "0.111.7"
    assert result.provenance.wheel_filename == _wheel("0.111.7")["filename"]
    assert len(requests) == 2


@pytest.mark.parametrize("simple_case", ["older", "http-error", "malformed"])
def test_ready_project_is_not_suppressed_by_an_older_or_broken_simple_view(simple_case):
    requests = []

    def handler(request):
        requests.append(str(request.url))
        if str(request.url) == update.PYPI_JSON_URL:
            return httpx.Response(200, json=_document("0.111.7"))
        assert str(request.url) == update.PYPI_SIMPLE_URL
        if simple_case == "http-error":
            return httpx.Response(503)
        if simple_case == "malformed":
            return httpx.Response(200, content="SECRET_BAD_JSON")
        return httpx.Response(200, json={"files": [{"filename": _wheel("0.111.6")["filename"]}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = update._fetch_pypi_read(client=client)
    assert result.version == "0.111.7"
    assert result.provenance.selected_view == "project-json"
    assert result.provenance.fallback == (
        "none" if simple_case == "older" else "simple-unavailable"
    )
    assert result.provenance.views[2].outcome == "skipped"
    assert len(requests) == 2
