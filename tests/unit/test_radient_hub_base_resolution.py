"""The CLI resolves one Radient Agent Hub API root, and only one.

``agents delete`` and ``agents push`` defaulted the ``radient_base_url`` config
key to the version-less ``https://api.radienthq.com`` and ``agents pull`` to the
non-existent ``https://api.radientlabs.ai`` — three call sites, three literals,
two of them wrong — while the client joins every hub path onto whatever base it
is handed. One configuration therefore resolved three different destinations,
and two of them could never work: the delete landed on the hub's unversioned
404 and the pull never left the machine.

These tests pin the resolution rule itself and the invariant that keeps it
single, because the failure mode is a fourth literal in a fourth place.
"""

import re
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from local_operator import cli
from local_operator.cli import agents_delete_command
from local_operator.env import (
    DEFAULT_RADIENT_API_BASE_URL,
    get_env_config,
    normalize_radient_api_base_url,
    resolve_radient_api_base_url,
)
from local_operator.providers import radient_credentials


@pytest.fixture
def isolated_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config dir no run of these tests can escape from.

    The CLI handlers below construct real ``ConfigManager``/``CredentialManager``
    instances from the directory they are handed, and both derive nothing from
    the ambient environment — but ``Path.home()`` is redirected anyway so that
    an accidental un-redirected read cannot reach the operator's live store.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path / ".local-operator"


def test_the_canonical_default_is_the_versioned_hub_root() -> None:
    """The default carries ``/v1`` because the client joins ``/agents/…`` onto it."""
    assert DEFAULT_RADIENT_API_BASE_URL.endswith("/v1")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The exact literal two CLI call sites used as their default: a host
        # root with no path at all is the one shape the version segment can go
        # missing from, and it addresses the hub's unversioned 404.
        ("https://api.radienthq.com", "https://api.radienthq.com/v1"),
        ("https://api.radienthq.com/", "https://api.radienthq.com/v1"),
        # The documented shape is returned byte-identical: the credential
        # resolver matches a destination on this path, and
        # tests/e2e/test_desktop_legacy_radient.py pins the same string
        # against a fake upstream.
        ("https://api.radienthq.com/v1", "https://api.radienthq.com/v1"),
        ("https://api.radienthq.com/v1/", "https://api.radienthq.com/v1"),
        ("http://127.0.0.1:41234/v1", "http://127.0.0.1:41234/v1"),
        # A base that names its own path is the operator's gateway; completing
        # it would retarget a proxy prefix rather than fix a mistake.
        ("https://gateway.internal/radient/v2", "https://gateway.internal/radient/v2"),
    ],
)
def test_a_host_root_gains_the_version_segment_and_a_path_is_left_alone(
    value: str, expected: str
) -> None:
    assert normalize_radient_api_base_url(value) == expected


def test_resolution_prefers_config_then_environment_then_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One rule, three tiers — the same one ``get_env_config`` already applies."""
    monkeypatch.setenv("RADIENT_API_BASE_URL", "https://env.example/v1")
    assert resolve_radient_api_base_url("https://config.example") == "https://config.example/v1"
    # An empty configured value is the key being absent, not a base of "".
    assert resolve_radient_api_base_url("") == "https://env.example/v1"
    assert resolve_radient_api_base_url(None) == "https://env.example/v1"
    monkeypatch.delenv("RADIENT_API_BASE_URL")
    assert resolve_radient_api_base_url(None) == DEFAULT_RADIENT_API_BASE_URL


def test_agents_delete_addresses_the_versioned_hub_path(
    isolated_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reported defect, at the seam the CLI actually builds.

    ``lop agents delete --id <hub id>`` used to send ``DELETE /agents/{id}`` —
    the hub answers that with ``{"error": "Not Found"}`` — because the config
    key had no value and the call site's own default had no version segment.
    """
    agent_id = "8f14e45f-ceea-467a-9c1c-1f0e2f0a1234"
    monkeypatch.setattr(
        radient_credentials,
        "resolve_radient_credential_sync",
        lambda *args, **kwargs: SecretStr("test-key"),
    )
    response = MagicMock()
    response.status_code = 204
    response.text = ""
    with patch("requests.delete", return_value=response) as mock_delete:
        result = agents_delete_command(
            Namespace(name=None, agent_id=agent_id), MagicMock(), isolated_config_dir
        )
    assert result == 0
    assert mock_delete.call_args[0][0] == f"{DEFAULT_RADIENT_API_BASE_URL}/agents/{agent_id}"


def test_the_environment_tier_is_completed_where_it_is_produced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One configuration, ONE destination — including for consumers that never
    call the resolver.

    The server routes, the desktop transport and the speech/models clients all
    read ``EnvConfig.radient_api_base_url`` directly, so a bare hand-set env value
    used to work in the CLI and 404 in the desktop transport. Normalizing at the
    producer is what makes the two agree; the bare host is the shape that has to
    agree, not the already-versioned one.
    """
    monkeypatch.setenv("RADIENT_API_BASE_URL", "http://hub.example")

    env_config = get_env_config()
    assert env_config.radient_api_base_url == "http://hub.example/v1"
    # The CLI's resolver and the raw value the other surfaces consume are the
    # same string, which is the whole claim.
    assert resolve_radient_api_base_url() == env_config.radient_api_base_url


def test_the_desktop_hub_transport_resolves_the_same_root() -> None:
    """The desktop transport reaches hub routes through the provider definition.

    That is a third consumer of this host, so the definition quotes the constant
    instead of spelling it again — checked through the transport's own
    ``base_url()``, which is what actually builds ``{base}/agents/{id}/…``.
    """
    from local_operator.providers.registry import get_provider_definition
    from local_operator.server.routes import desktop_radient

    definition = get_provider_definition("radient")
    assert definition is not None
    assert definition.base_url == DEFAULT_RADIENT_API_BASE_URL
    assert desktop_radient.base_url() == DEFAULT_RADIENT_API_BASE_URL


def test_the_cli_resolves_the_hub_base_in_exactly_one_place() -> None:
    """No call site may answer "which host?" for itself again.

    The three literals this replaced were not a reviewing oversight: each site
    was written where it was needed. What makes that safe is that the answer
    lives in one module and is quoted from there, so this asserts the shape
    rather than any single value.
    """
    source = Path(cli.__file__).read_text()
    # Any URL-shaped Radient host rather than two remembered spellings: the
    # failure mode is a fourth call site naming a host nobody had listed, which
    # a literal-by-literal assertion cannot see.
    assert not re.search(r"https?://[^\"']*radient", source)
    # One READ of the key, whichever accessor spells it — ``get_value``,
    # ``get_nested_value`` or a fresh ``get_config_value`` call all count.
    assert source.count('"radient_base_url"') == 1
