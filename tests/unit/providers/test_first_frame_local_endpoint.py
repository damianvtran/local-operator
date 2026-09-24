"""The first frame survives a hand-edited local endpoint (review round 2, R2-1).

WHY THIS IS ITS OWN FILE. Reading every provider through ``cached_available_models``
made this frame reach ``providers.local.resolve_base_url`` for the five
``local_setup`` providers -- a call that was UNREACHABLE from here before that
change. ``resolve_base_url`` normalises whatever the config holds, and the
settings editor validates through ``validate_endpoint_setting`` while the FILE
does not, so ``providers.lmstudio.base_url: http://localhost:notaport`` is a
reachable configuration. Measured against the shipped app, the exception emptied
the whole catalogue: 409 with 0 rows on the composer's ``command-entities``, 500
on the models route, 502 on the phone, where the previous head answered 200 with
120 rows on all three.

Two cases, because they arrive at the same answer through different paths: an
endpoint the normaliser REJECTS, and no endpoint at all (the generic gateway's
preset is empty).

The line that makes these tests real rather than decorative is the
``pytest.raises`` in each of them: it proves the config file IS in effect for the
provider under test, so the frame's rows are produced DESPITE the failing
resolution rather than because nothing was read.

The frame is asserted equal to ``static_catalogue()`` -- element-wise, in order --
because that is what it painted before this delta and what it must keep painting
in every degrade: no local provider ships any static rows at all
(``static_models`` is empty for all five), so the correct contribution from a
broken local endpoint is NOTHING, and the correct behaviour for the frame is
everything else, intact.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.model import discovery
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.controller import ProviderController
from local_operator.providers.local import resolve_base_url


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated config dir AND home, which is what both roots follow."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("LMSTUDIO_API_KEY", "OLLAMA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture
def controller(tmp_path: Path):
    store = AuthStore(tmp_path / "auth.db")
    instance = ProviderController(store)
    try:
        yield instance
    finally:
        instance.close()
        store.close()


def _hand_edit_config(root: Path, provider: str, base_url: str) -> None:
    """``provider``'s endpoint, written the way a person's editor would write it.

    The ``values`` wrapper is the real file shape: ``ConfigManager._load_config``
    reads ``config["values"]`` and back-fills missing top-level keys from the
    defaults, so a hand-edited file only has to carry the branch it changes. A
    file with ``providers`` at the TOP level is silently ignored -- which is how
    a first attempt at this test measured the cold path and proved nothing.
    """
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"providers": {provider: {"base_url": base_url}}}}),
        encoding="utf-8",
    )


def _row_keys(rows) -> list[dict[str, Any]]:
    return [dataclasses.asdict(row) for row in rows]


def test_a_rejected_local_endpoint_does_not_take_the_frame_down(
    isolated: Path, controller: ProviderController, tmp_path: Path
) -> None:
    _hand_edit_config(isolated, "lmstudio", "http://localhost:notaport")

    # The file IS in effect -- without this the test would pass on a frame that
    # simply never read the endpoint, which is the vacuous shape to avoid.
    with pytest.raises(ValueError):
        resolve_base_url("lmstudio")

    # And the reader this frame goes through is total: no exception, the shipped
    # answer, status "static".
    rows, status = discovery.cached_available_models("lmstudio", cache_dir=tmp_path / "cache")
    assert status == "static"
    assert rows == []

    # The frame paints every OTHER provider, byte for byte as before.
    frame = controller.initial_catalogue(cache_dir=tmp_path / "cache")
    assert _row_keys(frame) == _row_keys(controller.static_catalogue())
    assert any(row.provider == "anthropic" for row in frame)
    assert not [row for row in frame if row.provider == "lmstudio"]


def test_an_unconfigured_gateway_does_not_take_the_frame_down(
    isolated: Path, controller: ProviderController, tmp_path: Path
) -> None:
    """The other path in: no endpoint at all, so there is no document to name."""
    _hand_edit_config(isolated, "openai-compatible", "")

    assert resolve_base_url("openai-compatible") == ""

    frame = controller.initial_catalogue(cache_dir=tmp_path / "cache")
    assert _row_keys(frame) == _row_keys(controller.static_catalogue())


def test_the_frame_reads_the_config_at_most_once(
    isolated: Path, controller: ProviderController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A structural bound on the cost, not a wall-clock one (review R2-2).

    Every ``local_setup`` provider resolves its endpoint through
    ``provider_settings``, and a call with no values mapping of its own
    constructs a ``ConfigManager`` and parses config.yml. Five providers doing
    that on a route the composer calls per keystroke turned a 0.24 ms frame into
    14.51 ms (measured, A/B interleaved). The frame hands one snapshot down, so
    the count is 1 at most -- and this asserts the COUNT rather than a duration,
    which is the property that cannot drift with the host's load.
    """
    import local_operator.config as config_module
    from local_operator.config import ConfigManager

    constructions = {"n": 0}

    class _Counting(ConfigManager):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            constructions["n"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(config_module, "ConfigManager", _Counting)
    controller.initial_catalogue(cache_dir=tmp_path / "cache")

    assert constructions["n"] <= 1, (
        "the frame read the config once per local provider: "
        f"{constructions['n']} loads for one call"
    )
