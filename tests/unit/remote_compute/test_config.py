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

import httpx
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
        # Kept as written by the resolver; httpx is what collapses `..` later.
        ("https://staging.example.invalid/gw/../api", "https://staging.example.invalid/gw/../api"),
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
        ("https://staging.example.invalid#", "nor an empty fragment"),
        ("https://staging.example.invalid?", "nor an empty query"),
        ("ftp://staging.example.invalid", "not an HTTP scheme"),
        ("staging.example.invalid", "no scheme at all"),
        ("https:///no-host", "no host"),
        ("", "empty is a mistake, not a decision"),
        ("https://staging.example.invalid\n", "a newline is not invisible to a parser"),
        ("https://staging.example.invalid\t", "nor is a tab"),
        ("https://sta ging.example.invalid", "an interior space is not an escape"),
        ("https://staging.example.invalid/x y", "nor is one in the path"),
        ("https://host%41.example.invalid", "a percent-escape rewrites the authority"),
        ("https://staging.example.invalid/%2f", "and has no place in the path"),
        (
            "https://\u0445\u043e\u0441\u0442.example.invalid",
            "non-ASCII is not this base's vocabulary",
        ),
        ("https://staging.example.invalid.", "a trailing dot is not the host that was meant"),
        ("https://staging..example.invalid", "an empty label is not a hostname"),
        ("https://.staging.example.invalid", "nor is a leading dot"),
        ("https://staging.example.invalid:0", "port 0 asks for an ephemeral port"),
        ("https://staging.example.invalid:65536", "and 65536 is not a port"),
        ("https://staging.example.invalid:", "a port separator with no port"),
        ("https://[::1", "an unbalanced IPv6 bracket"),
        ("https://[ff02::1%eth0]", "an IPv6 zone is percent-encoded"),
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


@pytest.mark.parametrize(
    ("value", "address"),
    [
        # E4-3 — the line names EXACTLY the host:port that will be contacted.
        ("https://staging.example.invalid", "staging.example.invalid"),
        ("https://STAGING.Example.Invalid/", "staging.example.invalid"),
        ("https://staging.example.invalid:8443/api", "staging.example.invalid:8443"),
        ("http://localhost:8123", "localhost:8123"),
        ("http://[::1]:8123/", "[::1]:8123"),
    ],
)
def test_the_warning_names_exactly_the_address_contacted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str, address: str
) -> None:
    """Uppercase input announces the lowercased host; IPv6 is bracketed."""
    monkeypatch.setenv(ENV_VAR, value)
    resolve_api_url()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == _t(_wr.api_override(host=address)) + "\n"


#: One digit past CPython's bound on a decimal ``int()`` conversion (4300),
#: which is why the port is capped at five digits by the PATTERN rather than by
#: relying on the conversion to fail.
_TOO_LONG_PORT = "https://h.example:" + "9" * 4301


@pytest.mark.parametrize(
    ("value", "why"),
    [
        (_TOO_LONG_PORT, "a port int() refuses outright"),
        ("https://xn--a.example", "an xn-- label that decodes to nothing"),
        ("https://xn--0.example", "nor is this one a punycode encoding"),
        ("https://XN--a.example", "the A-label prefix is case-insensitive"),
    ],
)
def test_a_value_only_the_build_would_refuse_is_refused_where_it_was_parsed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str, why: str
) -> None:
    """E5-1: httpx must never be the one to discover an unusable base.

    A 4301-digit port used to reach a bare ``int()`` and a non-decoding
    ``xn--`` label used to reach ``httpx.URL``; both raise, neither is the
    typed refusal, and the CLI reports a raised exception as the generic
    "remote compute failed" with nothing the operator can act on. Each of these
    is refused here instead, with no line announced for a host nothing will
    contact.
    """
    monkeypatch.setenv(ENV_VAR, value)
    # A TRIPWIRE, not a second assertion: if the guard lets any of these
    # through to the ``httpx.URL`` backstop, the build is reached and the test
    # fails here rather than passing on the backstop's typed refusal.
    monkeypatch.setattr(
        config_module.httpx,
        "URL",
        lambda url: pytest.fail(f"httpx.URL was reached with {url!r}"),
    )
    with pytest.raises(InvalidApiOverride):
        resolve_api_url()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "", why


def test_a_decodable_alabel_is_still_a_hostname(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The rule is "decodes", not "has the prefix": refuse only what cannot."""
    monkeypatch.setenv(ENV_VAR, "https://xn--bcher-kva.example")
    assert resolve_api_url() == "https://xn--bcher-kva.example"
    assert capsys.readouterr().err == _t(_wr.api_override(host="xn--bcher-kva.example")) + "\n"


@pytest.mark.parametrize(
    ("value", "address"),
    [
        # E5-2 — one address, many spellings: ``ipaddress`` answers the
        # question, so a spelling this list does not happen to contain is not
        # refused for that reason.
        ("http://[0:0:0:0:0:0:0:1]:8123", "[0:0:0:0:0:0:0:1]:8123"),
        ("http://[::0001]:8123", "[::0001]:8123"),
        # ``127.0.0.0/8`` is loopback beyond the one literal a text list held.
        ("http://127.0.0.2:8123", "127.0.0.2:8123"),
        ("http://127.255.255.254", "127.255.255.254"),
        # The name is still allowed by name.
        ("http://localhost:8123", "localhost:8123"),
    ],
)
def test_every_loopback_spelling_may_be_used_over_cleartext(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str, address: str
) -> None:
    monkeypatch.setenv(ENV_VAR, value)
    assert resolve_api_url() == value.rstrip("/")
    assert capsys.readouterr().err == _t(_wr.api_override(host=address)) + "\n"


@pytest.mark.parametrize(
    ("value", "why"),
    [
        ("http://0.0.0.0:8123", "every interface is not this machine"),
        ("http://127.1:8123", "a short form the resolver would expand"),
        ("http://2130706433:8123", "nor is the integer form a literal here"),
        ("http://[::]:8123", "the unspecified address is not loopback"),
    ],
)
def test_a_host_that_is_not_loopback_is_still_refused_over_cleartext(
    monkeypatch: pytest.MonkeyPatch, value: str, why: str
) -> None:
    monkeypatch.setenv(ENV_VAR, value)
    with pytest.raises(InvalidApiOverride) as caught:
        resolve_api_url()
    assert ENV_VAR in str(caught.value), why


def test_an_httpx_refusal_never_escapes_as_anything_but_the_typed_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """E4-1 backstop: httpx.URL is constructed inside the resolver.

    Pinned by making it raise: a value this module accepted but httpx refuses
    must still leave as InvalidApiOverride, because the CLI maps that type and
    treats anything else as a generic failure.
    """

    def refuse(url: str) -> object:
        raise httpx.InvalidURL(f"Invalid URL {url!r}")

    monkeypatch.setattr(config_module.httpx, "URL", refuse)
    monkeypatch.setenv(ENV_VAR, "https://staging.example.invalid")
    with pytest.raises(InvalidApiOverride):
        resolve_api_url()
    # And nothing was announced for a base the client will not use.
    assert capsys.readouterr().err == ""


def test_the_backstop_does_not_swallow_a_failure_outside_the_build_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E5-3 — ``except Exception`` hid this module's bugs as the operator's.

    Only ``InvalidURL``/``ValueError``/``UnicodeError`` mean "this value cannot
    be made into a URL". Anything else raised by ``httpx.URL`` is a defect
    here, and reporting it as an unusable override would send the operator to
    check a variable that was never the problem.
    """

    def explode(url: str) -> object:
        raise RuntimeError(f"a bug in this module: {url!r}")

    monkeypatch.setattr(config_module.httpx, "URL", explode)
    monkeypatch.setenv(ENV_VAR, "https://staging.example.invalid")
    with pytest.raises(RuntimeError):
        resolve_api_url()


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
