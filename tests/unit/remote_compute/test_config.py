"""The remote-compute API base: the one override, its rules, and its warning.

WHAT IS BEING PINNED, AND WHY IT IS A SECURITY TEST AS MUCH AS A CONFIG ONE.
The base this module resolves is where the operator's Radient BEARER is sent.
So the rules are not preferences: a cleartext hop to anywhere but this machine
would hand the token to the network path, and an unusable value has to become a
refusal rather than a quiet fall back to production — a run sent to the wrong
host is work nobody can account for.

The warning is pinned the same way: it must appear once, on stderr, naming the
host, because the destination is otherwise invisible in anything the command
prints.
"""

from __future__ import annotations

import pytest

from local_operator.i18n.keys import wire_remote_compute as _wr
from local_operator.i18n.text import tr
from local_operator.remote_compute import config as config_module
from local_operator.remote_compute.config import ENV_VAR, resolve_api_url
from local_operator.remote_compute.errors import InvalidApiOverride
from local_operator.tunnels.config import DEFAULT_API_URL

_t = tr("wire.remote_compute")


@pytest.fixture(autouse=True)
def fresh_announcements(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test owns the once-per-process memory, so counts are its own."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(config_module, "_announced", set())


def test_unset_is_production(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_api_url() == DEFAULT_API_URL


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://staging.example.invalid", "https://staging.example.invalid"),
        # A trailing slash is stripped, once and for every case below it.
        ("https://staging.example.invalid/", "https://staging.example.invalid"),
        # A path is the operator's gateway, so it survives byte-identical.
        ("https://staging.example.invalid:8443/api", "https://staging.example.invalid:8443/api"),
        ("https://127.0.0.1", "https://127.0.0.1"),
        ("http://127.0.0.1:8123", "http://127.0.0.1:8123"),
        ("http://[::1]:8123/", "http://[::1]:8123"),
        ("http://localhost:8123", "http://localhost:8123"),
    ],
)
def test_accepted_values(monkeypatch: pytest.MonkeyPatch, value: str, expected: str) -> None:
    monkeypatch.setenv(ENV_VAR, value)
    assert resolve_api_url() == expected


@pytest.mark.parametrize(
    ("value", "why"),
    [
        ("http://staging.example.invalid", "cleartext to a host that is not this machine"),
        ("http://10.0.0.5:8123", "a private address is still a network hop"),
        ("https://user:secret@staging.example.invalid", "userinfo is a credential"),
        ("https://user@staging.example.invalid", "userinfo without a password"),
        ("https://staging.example.invalid/?tenant=1", "a query has no place in a base"),
        ("https://staging.example.invalid/#frag", "nor a fragment"),
        ("ftp://staging.example.invalid", "not an HTTP scheme"),
        ("staging.example.invalid", "no scheme at all"),
        ("https:///no-host", "no host"),
        ("", "empty is a mistake, not a decision"),
        ("   ", "and so is whitespace"),
    ],
)
def test_refused_values(monkeypatch: pytest.MonkeyPatch, value: str, why: str) -> None:
    monkeypatch.setenv(ENV_VAR, value)
    with pytest.raises(InvalidApiOverride) as caught:
        resolve_api_url()
    # ONE sentence, and it names the variable rather than echoing the value.
    message = str(caught.value)
    assert ENV_VAR in message
    assert "\n" not in message and message.endswith(".")
    if value.strip():
        assert value.strip() not in message, why


def test_the_warning_is_one_line_on_stderr_and_names_the_host(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_VAR, "https://staging.example.invalid")
    resolve_api_url()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == _t(_wr.api_override(host="staging.example.invalid")) + "\n"


def test_the_warning_is_printed_once_per_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A command builds more than one client; the user gets one line."""
    monkeypatch.setenv(ENV_VAR, "https://staging.example.invalid")
    resolve_api_url()
    resolve_api_url()
    assert capsys.readouterr().err.count("\n") == 1


def test_no_warning_without_an_override(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    resolve_api_url()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
