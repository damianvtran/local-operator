"""The model catalogue: real windows for aggregators, and never a hard failure.

Aggregators (OpenRouter, Radient) route hundreds of models, so their static
registry entry is a placeholder carrying ``context_window = -1``. Before the
catalogue existed, ``configure_model`` silently fell back to a 128,000 window
and zero prices for every one of them, which is wrong in a way that changes
runtime behaviour rather than just a display: auto compaction derives its
threshold from the window, so a 1M-context model summarised its history at
~102k instead of ~800k.

These tests pin both halves of the contract: the numbers come from the
catalogue when it is reachable, and NOTHING in the path can stop a session
from starting when it is not.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest import mock

import httpx
import pytest

from local_operator.harness.types import ChatRequest, Message
from local_operator.model import catalogue
from local_operator.model import configure as configure_mod
from local_operator.model import effort
from local_operator.model.catalogue import cached_listing
from local_operator.providers.clients import OpenAICompatClient


def _payload(model_id: str = "vendor/model", window: int = 1_000_000) -> dict[str, Any]:
    return {
        "data": [
            {
                "id": model_id,
                "name": model_id,
                "description": "d",
                "context_length": window,
                "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            }
        ]
    }


# -- cache mechanics ---------------------------------------------------------


def test_a_fresh_cache_is_used_without_fetching(tmp_path) -> None:
    calls = []

    def fetch():
        calls.append(1)
        return _payload()

    first = cached_listing("openrouter", fetch, cache_dir=tmp_path)
    second = cached_listing("openrouter", fetch, cache_dir=tmp_path)
    assert first == second == _payload()
    assert len(calls) == 1, "the second call must be served from disk"


def test_an_expired_cache_refetches(tmp_path) -> None:
    cached_listing("openrouter", lambda: _payload(window=1), cache_dir=tmp_path)
    fresh = cached_listing("openrouter", lambda: _payload(window=2), ttl_s=-1, cache_dir=tmp_path)
    assert fresh is not None
    assert fresh["data"][0]["context_length"] == 2


def test_a_failed_fetch_prefers_a_stale_cache_over_nothing(tmp_path) -> None:
    """A window that is a week old beats one that is wrong by 8x."""
    cached_listing("openrouter", lambda: _payload(window=999), cache_dir=tmp_path)

    def boom():
        raise RuntimeError("no network")

    got = cached_listing("openrouter", boom, ttl_s=-1, cache_dir=tmp_path)
    assert got is not None
    assert got["data"][0]["context_length"] == 999


def test_a_failed_fetch_with_no_cache_returns_none(tmp_path) -> None:
    """The one case where the caller must keep its static fallback — and it
    must be a None, not an exception, or an offline start dies here."""

    def boom():
        raise RuntimeError("no network")

    assert cached_listing("openrouter", boom, cache_dir=tmp_path) is None


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",  # valid JSON, wrong shape
        json.dumps({"payload": {"data": []}}),  # no fetched_at
        json.dumps({"fetched_at": "nope", "payload": {}}),  # unparseable stamp
    ],
    ids=["truncated", "wrong-type", "no-timestamp", "bad-timestamp"],
)
def test_an_unusable_cache_is_treated_as_absent(tmp_path, content: str) -> None:
    """A half-written file from a killed process must not raise; this is an
    optimisation store, so the only correct response is to refetch."""
    path = tmp_path / "openrouter.json"
    path.write_text(content, encoding="utf-8")
    assert cached_listing("openrouter", lambda: _payload(), cache_dir=tmp_path) == _payload()


def test_the_cache_write_is_atomic(tmp_path) -> None:
    """Two sessions can start at once, so a reader must never observe a
    partial document. Assert the temp file is not left behind and the final
    document parses."""
    cached_listing("openrouter", lambda: _payload(), cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.tmp")), "temp file leaked"
    raw = json.loads((tmp_path / "openrouter.json").read_text(encoding="utf-8"))
    assert raw["payload"] == _payload()
    assert time.time() - raw["fetched_at"] < 60


def test_an_unwritable_cache_dir_still_returns_the_payload(tmp_path, monkeypatch) -> None:
    """Losing the cache is a performance problem; failing the call is not."""

    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(catalogue.Path, "mkdir", refuse)
    assert cached_listing("openrouter", lambda: _payload(), cache_dir=tmp_path) == _payload()


# -- integration with configure_model ---------------------------------------
#
# The enrichment seam is `local_operator.model.discovery.available_models`, which
# these tests stub. It replaced a per-provider client factory that only ever ran
# for the two aggregators, and that narrowness was a real hole: the model picker
# offers whatever a provider's live listing returns, so a user could select a real
# model absent from the shipped registry — `anthropic/claude-opus-5` — and the
# session would run with `context_window = -1`, which silently disables compaction.


def _row(model_id: str, **kwargs):
    """One `DiscoveredModel` shaped the way an AGGREGATOR lists a model.

    OpenRouter and Radient describe every model they route — window and both
    prices — so these defaults are the truth for them and for any OpenAI-compatible
    gateway that fills in `context_length`/`pricing`. They are NOT the truth for
    every wire, and defaulting them here once let two tests pin a fiction: a row
    carrying a PRICE is a shape `_fetch_anthropic` cannot emit, and a test built on
    one passes against an enrichment that cannot work in production. Use
    :func:`_anthropic_row` for that wire.
    """
    from local_operator.model.discovery import DiscoveredModel

    fields = {
        "id": model_id,
        "context_window": 1_000_000,
        "input_price": 0.1,
        "output_price": 0.2,
    }
    fields.update(kwargs)
    return DiscoveredModel(**fields)


def _anthropic_row(model_id: str, name: str = "", **kwargs):
    """One row exactly as `_fetch_anthropic` builds it from `GET /v1/models`.

    That endpoint reports `id`, `display_name`, `max_input_tokens`, `max_tokens`
    and a `capabilities` object — but never a price and never a prompt-caching
    flag, so those two stay at the "unknown" zero here no matter what a caller
    passes for the limits. Defaults are the 5-generation numbers verified live on
    2026-08-07 (1M in, 128k out); pass `context_window=0, max_tokens=0` for the
    terse answer an older API version or a stripping proxy still gives.
    """
    from local_operator.model.discovery import DiscoveredModel

    fields = {
        "id": model_id,
        "name": name or model_id,
        "context_window": 1_000_000,
        "max_tokens": 128_000,
        "supports_images": True,
    }
    fields.update(kwargs)
    return DiscoveredModel(**fields)


def _bare_info(model_id: str):
    """A registry entry that knows nothing — the state enrichment exists to fix."""
    from local_operator.model.registry import ModelInfo

    return ModelInfo(id=model_id, name=model_id, description="", context_window=-1)


def _stub_discovery(monkeypatch, rows, status: str = "ok"):
    """Point `_info_from_discovery` at a canned listing."""
    from local_operator.model import discovery as discovery_mod

    def fake(provider_id, **_kwargs):
        if callable(rows):
            return rows(provider_id), status
        return list(rows), status

    monkeypatch.setattr(discovery_mod, "available_models", fake)


def test_configure_model_takes_the_window_from_the_listing(monkeypatch, tmp_path) -> None:
    """The defect this module exists for: a model's real window reaching the
    ModelSpec instead of the 128k placeholder."""
    from local_operator.model import configure as configure_mod

    _stub_discovery(monkeypatch, [_row("vendor/big")])
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    config = configure_mod.configure_model(hosting="openrouter", model_name="vendor/big")
    assert config.spec.context_window == 1_000_000
    assert config.info.input_price == pytest.approx(0.1)
    assert config.info.output_price == pytest.approx(0.2)


def test_a_direct_provider_is_enriched_too_not_just_the_aggregators(monkeypatch, tmp_path) -> None:
    """The hole the model picker turned into a routine path, pinned against the
    row Anthropic's wire can ACTUALLY produce.

    The reported symptom was a status band reading `1.8%/200k` on
    `anthropic/claude-opus-5`, whose real window is 1M. Anthropic's listing carries
    `max_input_tokens` and `max_tokens` per model, so this is not a number that has
    to be guessed from a family floor — the enrichment path either reads it or the
    session compacts at 160k on a model with 1M of room, throwing away 84% of it.

    The id is an UNSHIPPED one rather than `claude-opus-5` itself, and the swap is
    the point rather than a convenience. Enrichment is gated on the registry being
    incomplete (`_needs_enrichment`: no real window, or no price), and now that the
    5-generation rows carry Anthropic's published prices the whole family is
    complete — so resolving `claude-opus-5` correctly does no listing at all and
    could no longer demonstrate anything about this path. What the picker actually
    offers that the registry cannot describe is the id released since this package
    was cut, which is what this now uses.

    Prices stay zero here because neither source has one: Anthropic's wire quotes
    none, and the stubbed aggregator catalogue has no row for this id either. An
    invented price renders in the band as fact.
    """
    from local_operator.model import configure as configure_mod
    from local_operator.model.registry import anthropic_models

    unshipped = "claude-nova-6"
    assert unshipped not in anthropic_models, "the id this test treats as unshipped now ships"
    _stub_discovery(monkeypatch, [_anthropic_row(unshipped, "Claude Nova 6")])
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    configure_mod.invalidate_model_info_cache()

    config = configure_mod.configure_model(hosting="anthropic", model_name=unshipped)
    assert config.spec.context_window == 1_000_000, "the listing's window never reached the spec"
    assert config.spec.max_output_tokens == 128_000, "64k silently truncates long answers"
    assert config.spec.supports_prompt_cache is True, "no cache_control on the priciest model"
    assert config.spec.supports_images is True
    assert config.info.input_price == 0.0
    assert config.info.output_price == 0.0
    assert config.info.name == "Claude Nova 6"


def test_an_undescribed_claude_snapshot_inherits_its_familys_window(monkeypatch, tmp_path) -> None:
    """A dated snapshot of a shipped model, listed by a wire that says nothing.

    Anthropic serves undated ids (`claude-opus-5`) and dated snapshots of the same
    model (`claude-opus-5-20260112`) and adds new ones between releases of this
    package, so an id absent from the registry is routine rather than exotic. With a
    terse listing — an older API version, a proxy that strips fields — nothing on
    the wire can supply the window, and the flat family template answers 200k for
    the whole vendor. That is the number the user saw. The FAMILY is what the id
    itself still says, so the snapshot resolves to Opus 5's real 1M.

    The PRICE is inherited by the same route and matters more than the window's
    sibling assertions suggest: a dated snapshot is the same model at the same
    published rate, so a snapshot that costed as unknown while its undated twin
    costed correctly would put a "$—" in the band for no reason a user could see.
    """
    from local_operator.model import configure as configure_mod
    from local_operator.model.registry import anthropic_models

    snapshot = "claude-opus-5-20260112"
    assert snapshot not in anthropic_models, "the id this test treats as unshipped now ships"
    _stub_discovery(
        monkeypatch,
        [_anthropic_row(snapshot, "Claude Opus 5", context_window=0, max_tokens=0)],
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    config = configure_mod.configure_model(hosting="anthropic", model_name=snapshot)
    assert config.spec.context_window == 1_000_000, "fell back to the 200k family floor"
    assert config.spec.max_output_tokens == 128_000
    assert config.spec.supports_prompt_cache is True
    assert (config.info.input_price, config.info.output_price) == (5.0, 25.0)
    assert (config.info.cache_writes_price, config.info.cache_reads_price) == (6.25, 0.50)


def test_an_unknown_claude_generation_does_not_inherit_downward(monkeypatch, tmp_path) -> None:
    """Inheritance runs FORWARD only, and the asymmetry is the reason.

    Windows have only ever grown within a tier, so a generation newer than anything
    shipped can safely take the newest known limits. Backwards it is unsafe: the
    default threshold is `min(0.8 * window, 600k)`, so handing a genuinely-200k
    model a 1M window puts the trigger at 600k — past the model's real limit — and
    every turn beyond 200k 400s instead of compacting. An older unshipped id
    therefore gets the conservative 200k floor, not its tier's newest numbers.
    """
    from local_operator.model import configure as configure_mod

    _stub_discovery(monkeypatch, [_anthropic_row("claude-opus-2", context_window=0, max_tokens=0)])
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    config = configure_mod.configure_model(hosting="anthropic", model_name="claude-opus-2")
    assert config.spec.context_window == 200_000
    assert config.spec.max_output_tokens == 64_000


@pytest.mark.parametrize(
    "listing_says, expected",
    [
        # The provider denies image input: that reaches the spec, which is what
        # gates the snapcompact VISION strategy for the session.
        (False, False),
        # The provider affirms it.
        (True, True),
        # The provider said nothing, so the registry's own answer stands.
        (None, True),
    ],
)
def test_the_providers_own_capability_answer_reaches_the_spec(
    monkeypatch, tmp_path, listing_says, expected
) -> None:
    """Review finding C-07, at the end of the chain it matters for.

    `ModelSpec.supports_images` selects the compaction strategy, so a wrong value
    is not cosmetic: a text-only model routed through the vision path builds an
    archive of image frames the provider will reject. Both the shipped Anthropic
    rows and the family template say `supports_images=True`, so under the previous
    OR a live `false` could never take effect and the precision of the capability
    read had no consumer.

    Asserted on a SHIPPED row, which is the only place the harm is real: an
    unshipped id has no vision claim of its own to be wrong about. That the row is
    reachable at all depends on `limits_from_listing` — the ten current-generation
    rows transcribed their limits out of this very listing, so resolution asks it
    again rather than pinning the session to the transcription date. Pricing those
    rows had briefly closed that door and made this case unreachable.
    """
    from local_operator.model import configure as configure_mod
    from local_operator.model.registry import anthropic_models

    assert anthropic_models["claude-opus-5"].supports_images is True, "fixture drifted"
    assert (
        anthropic_models["claude-opus-5"].limits_from_listing is True
    ), "a row that does not re-ask its listing cannot receive a live capability answer"
    _stub_discovery(
        monkeypatch,
        [_anthropic_row("claude-opus-5", "Claude Opus 5", supports_images=listing_says)],
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    # All three cases resolve the SAME id, and resolution is memoized per TTL
    # bucket for the life of the process — without this the second and third cases
    # would assert against the first case's answer and pass for the wrong reason.
    configure_mod.invalidate_model_info_cache()

    config = configure_mod.configure_model(hosting="anthropic", model_name="claude-opus-5")
    assert config.spec.supports_images is expected
    # A capability answer must not disturb the limits the same listing carried.
    assert config.spec.context_window == 1_000_000


@pytest.mark.parametrize("break_it", ["discovery-raises", "no-such-model", "empty-listing"])
def test_configure_model_never_fails_on_a_bad_listing(monkeypatch, tmp_path, break_it: str) -> None:
    """Every listing failure mode degrades to the static fallback. A session MUST
    start with no network, an unknown model id, or an empty answer."""
    from local_operator.model import configure as configure_mod
    from local_operator.model import discovery as discovery_mod
    from local_operator.model.configure import UNKNOWN_CONTEXT_WINDOW

    if break_it == "discovery-raises":

        def fake(provider_id, **_kwargs):
            raise RuntimeError("no network")

        monkeypatch.setattr(discovery_mod, "available_models", fake)
    elif break_it == "no-such-model":
        _stub_discovery(monkeypatch, [_row("other/model")])
    else:
        _stub_discovery(monkeypatch, [], status="empty")
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    config = configure_mod.configure_model(hosting="openrouter", model_name="vendor/asked-for")
    assert config.spec.context_window == UNKNOWN_CONTEXT_WINDOW


def test_a_model_with_a_real_static_entry_never_consults_a_listing(monkeypatch) -> None:
    """Enrichment is for gaps only. Anthropic's Sonnet window is in the registry, so
    touching the network for it would add latency to every session start for
    nothing."""
    from local_operator.model import configure as configure_mod
    from local_operator.model import discovery as discovery_mod

    def explode(*_args, **_kwargs):
        raise AssertionError("a known model must not trigger discovery")

    monkeypatch.setattr(discovery_mod, "available_models", explode)
    config = configure_mod.configure_model(
        hosting="anthropic", model_name="claude-sonnet-4-20250514"
    )
    assert config.spec.context_window == 200_000


@pytest.mark.parametrize("provider", ["openrouter", "radient", "anthropic", "ollama"])
def test_a_keyless_install_never_raises_resolving_metadata(monkeypatch, provider: str) -> None:
    """A metadata optimisation must never become "the CLI will not start".

    The first version of this got it wrong in the client constructors —
    `OpenRouterClient` raises on an empty key and `RadientClient` needs a base_url —
    and turned a missing key into a failed start. The transports moved into
    `discovery`, which is contractually non-raising, so what is pinned here is the
    END of the chain: no key anywhere, no exception, and a usable answer.
    """
    from local_operator.model import configure as configure_mod
    from local_operator.model.registry import ModelInfo

    for name in ("OPENROUTER_API_KEY", "RADIENT_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    fallback = ModelInfo(id="x", name="x", description="")
    info = configure_mod._info_from_discovery(provider, "no-such-model", fallback)
    assert info is fallback, "an absent model must hand back the caller's fallback"


@pytest.mark.parametrize(
    "hosting,model",
    [("openrouter", "google/gemini-2.0-flash-001"), ("radient", "auto")],
)
def test_configure_model_survives_a_keyless_install(monkeypatch, hosting: str, model: str) -> None:
    """End of the same defect: the whole call must still return a usable
    configuration with no credentials anywhere."""
    from local_operator.model.configure import UNKNOWN_CONTEXT_WINDOW, configure_model

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RADIENT_API_KEY", raising=False)

    config = configure_model(hosting=hosting, model_name=model)
    assert config.spec.context_window >= UNKNOWN_CONTEXT_WINDOW


# -- clock and concurrency hazards ------------------------------------------


def test_a_future_timestamp_is_stale_not_permanently_fresh(tmp_path) -> None:
    """Clamping a negative age to zero — the obvious move — makes an entry
    written under a skewed clock look fresh FOREVER, so one NTP correction or a
    file copied between machines pins the catalogue with no recovery short of
    deleting it by hand. Refetching once is the cheap direction to be wrong in.
    """
    path = tmp_path / "openrouter.json"
    path.write_text(
        json.dumps({"fetched_at": time.time() + 86_400 * 30, "payload": _payload()}),
        encoding="utf-8",
    )
    _, age = catalogue._read_cache(path)
    assert age == float("inf")

    calls: list[int] = []

    def fetch():
        calls.append(1)
        return _payload(window=7)

    got = cached_listing("openrouter", fetch, cache_dir=tmp_path)
    assert got is not None
    assert len(calls) == 1, "a future-stamped entry must be refetched"
    assert got["data"][0]["context_length"] == 7


def test_two_writers_never_share_a_temp_file_name(tmp_path, monkeypatch) -> None:
    """Two writers sharing one temp file interleave their bytes into it, and the
    atomic rename then delivers the interleaved document intact. It parses as
    JSON and passes every shape check, so the result is silently-wrong prices and
    windows served for the whole TTL.

    Pinned by NAME rather than by racing threads, deliberately: a race test cannot
    fail reliably, and it could not have caught the earlier PID-suffixed name at
    all, because threads in one process share a PID — every writer computed one
    identical name, so the fixed and unfixed paths were literally the same code.

    Observed at the RENAME, not at whatever call produces the name, so the test
    pins the property (each writer stages into a file of its own) rather than the
    mechanism that provides it.
    """
    path = tmp_path / "openrouter.json"
    staged: list[str] = []
    real_replace = catalogue.Path.replace

    def record(self, target):  # noqa: ANN001
        staged.append(self.name)
        return real_replace(self, target)

    monkeypatch.setattr(catalogue.Path, "replace", record)
    # One PID for both writes: the case the PID suffix could not cover, and the
    # one that actually occurs, since configure_model runs inside server request
    # handlers on a thread pool.
    monkeypatch.setattr(catalogue.os, "getpid", lambda: 4242)

    catalogue._write_cache(path, _payload())
    catalogue._write_cache(path, _payload(window=42))

    assert len(staged) == 2, "each write must stage through a temp file"
    assert staged[0] != staged[1], "two writers in one process must not share a temp file"
    assert not list(tmp_path.glob("*.tmp")), "temp files must be renamed away"
    assert (
        json.loads(path.read_text(encoding="utf-8"))["payload"]["data"][0]["context_length"] == 42
    )


def test_another_writers_temp_file_cannot_corrupt_this_write(tmp_path) -> None:
    """A stranded temp file from some other writer must not be adopted."""
    import os

    path = tmp_path / "openrouter.json"
    (tmp_path / f"{path.name}.{os.getpid()}.tmp").write_text("{partial", encoding="utf-8")
    (tmp_path / f"{path.name}.tmp").write_text("{partial", encoding="utf-8")
    catalogue._write_cache(path, _payload(window=42))
    assert (
        json.loads(path.read_text(encoding="utf-8"))["payload"]["data"][0]["context_length"] == 42
    )


def test_a_failed_write_does_not_strand_a_temp_file(tmp_path, monkeypatch) -> None:
    """Otherwise a persistently failing start accumulates one temp file per run."""
    path = tmp_path / "openrouter.json"
    real_replace = catalogue.Path.replace

    def fail_replace(self, target):  # noqa: ANN001
        raise OSError("cross-device link")

    monkeypatch.setattr(catalogue.Path, "replace", fail_replace)
    catalogue._write_cache(path, _payload())
    monkeypatch.setattr(catalogue.Path, "replace", real_replace)
    assert not list(tmp_path.glob("*.tmp"))


def test_the_in_process_memo_expires_with_the_wall_clock() -> None:
    """A bare ``lru_cache`` would pin whatever metadata a long-lived process saw
    at boot: the HTTP server and scheduler workers run for days, the disk cache
    would refresh underneath them, and nothing would ever read the new numbers.

    Driven entirely through the PUBLIC entry point, and by advancing a patched
    clock. Calling the memoized inner function with a hand-made bucket argument
    instead — the earlier form of this test — only demonstrated that
    ``lru_cache`` keys on its parameters, which is a language guarantee: it held
    for any third argument, so hardcoding the bucket to a constant (restoring the
    never-expiring memo exactly) left it passing. What needs pinning is that
    ``resolve_model_info`` derives the bucket from the CLOCK.
    """
    from local_operator.model.configure import (
        _resolve_model_info_cached,
        resolve_model_info,
    )

    model = "claude-sonnet-4-20250514"
    now = 1_700_000_000.0
    with mock.patch.object(configure_mod.time, "time", lambda: now):
        resolve_model_info("anthropic", model)
        hits = _resolve_model_info_cached.cache_info().hits
        # Same instant: a hit, or the memo is pointless.
        resolve_model_info("anthropic", model)
        assert _resolve_model_info_cached.cache_info().hits == hits + 1

    misses = _resolve_model_info_cached.cache_info().misses
    # One TTL later the bucket has rolled, so the same call must MISS and go back
    # to the disk cache. Under a fixed bucket this is a hit and the assertion
    # fails, which is the point.
    with mock.patch.object(configure_mod.time, "time", lambda: now + catalogue.DEFAULT_TTL_S):
        resolve_model_info("anthropic", model)
    assert _resolve_model_info_cached.cache_info().misses == misses + 1


# -- credential resolution (S1) ----------------------------------------------


#: ``(provider, env_var, credential_file_key)`` for both shapes of
#: ``ProviderDefinition.env_keys``. Every credential test runs over both on
#: purpose: ``env_keys`` is ``str | Callable[[], str | None] | None``, and a reader
#: written as ``isinstance(env_keys, str)`` passes every openrouter test while
#: silently resolving NOTHING for the one provider using the callable form. That
#: provider is Anthropic, whose catalogue 401s unauthenticated — so the shape that
#: was untested was also the one where failing to read a key costs the most.
_ENV_KEY_SHAPES = [
    pytest.param("openrouter", "OPENROUTER_API_KEY", "OPENROUTER_API_KEY", id="str-env-keys"),
    pytest.param("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY", id="callable-env-keys"),
]


@pytest.mark.parametrize("provider, env_var, file_key", _ENV_KEY_SHAPES)
def test_the_key_is_read_from_the_apps_own_credential_store(
    tmp_path, monkeypatch, provider: str, env_var: str, file_key: str
) -> None:
    """Reading only ``os.environ`` made the whole fix a no-op for the users who
    configured credentials the sanctioned way.

    ``local-operator credential update`` writes the CredentialManager file and
    the TUI's ``/login`` writes the AuthStore; neither touches the environment.
    Their sessions streamed fine (the stream-time cascade reads those stores)
    while the band showed a 128k window and no cost forever, with the failure
    recorded only at debug level.
    """
    from local_operator.credentials import CredentialManager
    from local_operator.model import configure as configure_mod

    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    CredentialManager(tmp_path).set_credential(file_key, "sk-store-value")

    assert configure_mod._catalogue_api_key(provider) == "sk-store-value"


@pytest.mark.parametrize("provider, env_var, file_key", _ENV_KEY_SHAPES)
def test_an_env_var_takes_precedence_over_the_stored_credential(
    tmp_path, monkeypatch, provider: str, env_var: str, file_key: str
) -> None:
    """An explicit env var is the operator overriding config for one run."""
    from local_operator.credentials import CredentialManager
    from local_operator.model import configure as configure_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    CredentialManager(tmp_path).set_credential(file_key, "sk-store-value")
    monkeypatch.setenv(env_var, "sk-env-value")

    assert configure_mod._catalogue_api_key(provider) == "sk-env-value"


def test_an_env_api_key_beats_a_stored_oauth_row_and_keeps_its_kind(monkeypatch) -> None:
    """Precedence AND credential kind, for the provider where both were wrong.

    With the callable ``env_keys`` form unread, ``ANTHROPIC_API_KEY`` resolved to
    nothing and the cascade fell through to the OAuth store. Two things broke at
    once: the documented order inverted, so a key exported for this one run lost to
    a months-old stored login; and the KIND flipped with it, so the request went
    out as ``Authorization: Bearer`` plus the OAuth beta header where ``x-api-key``
    was correct. Anthropic answers the wrong shape with a 401, which is exactly the
    "model cannot be described" outcome enrichment exists to prevent.
    """
    from local_operator.model import configure as configure_mod

    monkeypatch.setattr(
        configure_mod,
        "_oauth_listing_token",
        lambda _provider: ("oauth-access-token", True, None),
    )
    assert configure_mod._catalogue_credential("anthropic") == (
        "oauth-access-token",
        True,
        None,
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-explicit")
    assert configure_mod._catalogue_credential("anthropic") == (
        "sk-ant-explicit",
        False,
        None,
    )


def test_an_oauth_token_from_the_environment_is_not_sent_as_an_api_key(monkeypatch) -> None:
    """``_anthropic_env_key`` prefers ``ANTHROPIC_OAUTH_TOKEN`` and returns only the
    VALUE, so reporting whatever it hands back as an API key would send an OAuth
    token as ``x-api-key`` — a 401, and a model left undescribed."""
    from local_operator.model import configure as configure_mod

    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "sk-ant-oat-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-key")

    assert configure_mod._catalogue_credential("anthropic") == (
        "sk-ant-oat-token",
        True,
        None,
    )


def test_a_plain_api_key_is_not_read_as_oauth_because_a_value_collides(monkeypatch) -> None:
    """The kind is inferred from variable NAMES, and a value can sit under several.

    Any variable with OAUTH in its name holding the same string flipped the flag,
    and for OpenAI that is not a cosmetic misread: the OAuth route needs a ChatGPT
    account id an env key cannot supply, so the provider became unlistable
    outright — silently, and for as long as both variables stayed set. An OAuth
    token misread the other way costs one 401 and falls back to the registry, so
    an ambiguous value resolves to the cheaper mistake.
    """
    from local_operator.model import configure as configure_mod

    monkeypatch.setattr(configure_mod, "_oauth_listing_token", lambda _provider: ("", False, None))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-shared")
    monkeypatch.setenv("SOME_TOOL_OAUTH_TOKEN", "sk-proj-shared")

    assert configure_mod._catalogue_credential("openai") == ("sk-proj-shared", False, None)

    # And a value that appears ONLY under an OAuth-named variable is still OAuth,
    # which is the case the name test exists for.
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "sk-ant-oat-only")
    assert configure_mod._catalogue_credential("anthropic") == ("sk-ant-oat-only", True, None)


def test_stored_openai_oauth_account_scope_reaches_model_discovery(tmp_path, monkeypatch) -> None:
    from local_operator.model import configure as configure_mod
    from local_operator.providers.auth_store import AuthStore

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store = AuthStore()
    store.upsert_credential(
        "openai",
        {
            "access": "chatgpt-token",
            "refresh": "refresh-token",
            "account_id": "acct-42",
        },
    )
    store.close()

    assert configure_mod._catalogue_credential("openai") == (
        "chatgpt-token",
        True,
        "acct-42",
    )


def test_no_key_anywhere_still_resolves(tmp_path, monkeypatch) -> None:
    """The aggregators' listing endpoints are PUBLIC catalogue data — `GET
    /api/v1/models` answers 200 with no Authorization header at all — so a keyless
    or OAuth-only install must still be able to learn a real context window instead
    of silently keeping 128k."""
    from local_operator.model import configure as configure_mod

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    assert configure_mod._catalogue_api_key("openrouter") == ""
    _stub_discovery(monkeypatch, [_row("vendor/big", context_window=777_000)])
    info = configure_mod._info_from_discovery("openrouter", "vendor/big", _bare_info("vendor/big"))
    assert info.context_window == 777_000


# -- schema drift that validates but cannot be mapped (S2) -------------------


def _drifted_payload() -> dict[str, Any]:
    """Validates under ``extra="allow"``, then breaks ``float()``/``int()``."""
    return {
        "data": [
            {
                "id": "v/m",
                "name": "v/m",
                "description": "d",
                "context_length": {"max": 1000},
                "pricing": {
                    "prompt": "0.1",
                    "completion": "0.2",
                    "input_cache_read": {"usd": 1},
                },
            }
        ]
    }


@pytest.mark.parametrize(
    "bad_window, raised_by_python",
    [
        ({"max": 1000}, "TypeError"),
        ("not-a-number", "ValueError"),
        (float("nan"), "ValueError"),
        (float("inf"), "OverflowError"),
    ],
)
def test_a_payload_that_validates_but_cannot_map_is_reported_not_raised(
    bad_window, raised_by_python: str
) -> None:
    """Validation is NOT a guarantee that the mapping will succeed. The listing
    schemas set ``extra="allow"``, so a wrong-shaped extra validates cleanly and
    then raises inside the conversions.

    All four shapes are parametrized because they raise three DIFFERENT exceptions
    and each was a separate hole: catching ValueError alone let the dict through,
    and adding TypeError still left ``Infinity`` — which ``json.loads`` accepts as
    a bare literal and ``int()`` then rejects with OverflowError, a subclass of
    neither.

    Asserted against ``_info_from_listing`` itself, which is where the rule lives
    and which is still reached by the injected-client path
    (``get_model_info_from_openrouter``). It raises ``_UnmappableEntry`` — a
    ValueError subclass, so every existing caller's ``except ValueError`` still
    holds — rather than letting the raw conversion error escape.
    """
    from local_operator.clients.openrouter import OpenRouterListModelsResponse
    from local_operator.model.registry import ModelInfo

    payload = _drifted_payload()
    payload["data"][0]["context_length"] = bad_window
    # The premise of the finding: pydantic accepts it.
    listing = OpenRouterListModelsResponse.model_validate(payload)

    template = ModelInfo(id="v/m", name="v/m", description="")
    with pytest.raises(configure_mod._UnmappableEntry):
        configure_mod._info_from_listing(listing, "v/m", template, "openrouter")
    # A ValueError subclass on purpose: callers outside this module catch that.
    with pytest.raises(ValueError):
        configure_mod._info_from_listing(listing, "v/m", template, "openrouter")


def test_a_bad_entry_does_not_stop_a_good_one_being_mapped() -> None:
    """A bad entry is not a bad document. The conversions only ever run for the ONE
    model whose id matched, out of a listing that carries several hundred, so a
    single unreadable row must not cost the other 339 their metadata."""
    from local_operator.clients.openrouter import OpenRouterListModelsResponse
    from local_operator.model.registry import ModelInfo

    bad = _drifted_payload()["data"][0]
    good = _payload(model_id="good/model", window=999_000)["data"][0]
    listing = OpenRouterListModelsResponse.model_validate({"data": [good, bad]})
    template = ModelInfo(id="t", name="t", description="")

    resolved = configure_mod._info_from_listing(listing, "good/model", template, "openrouter")
    assert resolved.context_window == 999_000
    with pytest.raises(ValueError):
        configure_mod._info_from_listing(listing, bad["id"], template, "openrouter")


def test_invalidate_is_safe_when_there_is_no_cache(tmp_path) -> None:
    catalogue.invalidate("openrouter", cache_dir=tmp_path)  # must not raise


# -- credential-change invalidation ------------------------------------------


def test_every_scoped_document_of_one_credential_is_dropped_together(tmp_path) -> None:
    """A login cannot know WHICH document the new credential will resolve to.

    ``discovery._cache_key`` picks between a plain document, a host-scoped one
    and a per-account one using the credential it is handed -- a decision made
    after the login, not before it. Dropping only the plain key therefore leaves
    the document the next listing actually reads in place, which is the stale
    catalogue this exists to clear.
    """
    for name in (
        "openai.listing.json",
        "openai.oauth.listing.json",
        "openai.oauth.abc123def456.listing.json",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    dropped = catalogue.invalidate_documents("openai", cache_dir=tmp_path)

    assert dropped == 3
    assert list(tmp_path.glob("*.json")) == []


def test_invalidation_does_not_reach_a_provider_whose_id_shares_a_prefix(tmp_path) -> None:
    """``zai`` and ``zai-oauth`` are separate credential identities.

    The dot in the glob is what keeps them apart. Without it, logging in to the
    shorter id would silently drop the longer one's catalogue -- an invisible
    extra fetch for a provider the user never touched, and a cache stampede on
    a shared machine.
    """
    (tmp_path / "zai.listing.json").write_text("{}", encoding="utf-8")
    (tmp_path / "zai-oauth.listing.json").write_text("{}", encoding="utf-8")
    (tmp_path / "openrouter.listing.json").write_text("{}", encoding="utf-8")

    assert catalogue.invalidate_documents("zai", cache_dir=tmp_path) == 1

    survivors = {path.name for path in tmp_path.glob("*.json")}
    assert survivors == {"zai-oauth.listing.json", "openrouter.listing.json"}


def test_invalidating_an_unlisted_provider_is_zero_not_an_error(tmp_path) -> None:
    """The ordinary case for any provider that was never listed. A login must not
    fail because there was no catalogue to clear."""
    assert catalogue.invalidate_documents("anthropic", cache_dir=tmp_path) == 0


def test_the_skills_index_sharing_this_directory_is_never_touched(tmp_path) -> None:
    """The cache root is shared (see ``purge_legacy_documents``). A glob loose
    enough to take the skills index out would make every login re-embed it."""
    (tmp_path / "anthropic.listing.json").write_text("{}", encoding="utf-8")
    (tmp_path / "anthropic.skills.meta.json").write_text("{}", encoding="utf-8")

    assert catalogue.invalidate_documents("anthropic", cache_dir=tmp_path) == 1
    assert (tmp_path / "anthropic.skills.meta.json").exists()


# -- orphaned documents from earlier layouts (M-08) --------------------------


def test_purge_removes_only_the_documents_no_reader_can_reach(tmp_path) -> None:
    """~800 KB per install was measured sitting in the cache under dead names.

    Two generations are dead -- ``<provider>.models.json`` from the bare-provider
    key and ``<provider>.models.models.json`` from the doubled suffix -- while the
    current ``<provider>.listing.json`` and the skills index's files, which share
    this directory, must survive.
    """
    dead = [
        tmp_path / "openrouter.models.json",
        tmp_path / "radient.models.json",
        tmp_path / "openrouter.models.models.json",
        tmp_path / "anthropic.models.models.json",
    ]
    alive = [
        tmp_path / "openrouter.listing.json",
        tmp_path / "4b2e3b6dedc6f0c7.skills.meta.json",
        tmp_path / "4b2e3b6dedc6f0c7.skills.vec",
    ]
    for path in dead + alive:
        path.write_text("{}", encoding="utf-8")

    catalogue.purge_legacy_documents(tmp_path)
    catalogue.purge_legacy_documents(tmp_path)  # idempotent, no marker file needed

    assert [path.name for path in dead if path.exists()] == []
    assert all(path.exists() for path in alive)
    # No bookkeeping file either: state would be one more thing to get wrong.
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(p.name for p in alive)


def test_purge_is_safe_when_the_cache_dir_does_not_exist(tmp_path) -> None:
    # The common case on a first run, and it must not create the directory
    # either: nothing is cached yet, so there is nothing to hold.
    missing = tmp_path / "never-created"
    catalogue.purge_legacy_documents(missing)  # must not raise
    assert not missing.exists()


def test_a_cached_listing_call_sweeps_the_orphans(tmp_path) -> None:
    """The sweep has to be on the read path: there is no other moment that knows
    the cache directory, and a dead name has no reader left to notice it."""
    orphan = tmp_path / "openrouter.models.json"
    orphan.write_text(
        json.dumps({"fetched_at": time.time(), "payload": _payload()}), encoding="utf-8"
    )

    served = cached_listing("openrouter.listing", lambda: _payload(window=7), cache_dir=tmp_path)

    assert served is not None
    assert served["data"][0]["context_length"] == 7
    assert not orphan.exists()
    assert (tmp_path / "openrouter.listing.json").exists()


# -- real concurrency, not a happy-path stand-in (S4) ------------------------


def test_concurrent_writers_never_produce_a_corrupt_cache(tmp_path) -> None:
    """The original test was NAMED for this guarantee but wrote once, single
    threaded — so the shared-temp-file race passed it and was found by audit
    instead. Drive real threads with different payloads and assert every read
    lands on a complete document.
    """
    import threading

    errors: list[str] = []

    def writer(window: int) -> None:
        for _ in range(15):
            catalogue._write_cache(tmp_path / "openrouter.json", _payload(window=window))

    def reader() -> None:
        for _ in range(40):
            payload, _age = catalogue._read_cache(tmp_path / "openrouter.json")
            if payload is None:
                continue
            try:
                # A torn document shows up here: a truncated JSON body fails to
                # parse (already None), and an interleaved one loses its shape.
                assert payload["data"][0]["context_length"] in (1, 2, 3)
            except (KeyError, IndexError, TypeError, AssertionError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=writer, args=(n,)) for n in (1, 2, 3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"readers observed a corrupt document: {errors[:3]}"
    assert not list(tmp_path.glob("*.tmp")), "temp files leaked"


def test_the_spec_carries_the_enriched_window_not_the_placeholder(monkeypatch, tmp_path) -> None:
    """`build_model_spec` must resolve through the ENRICHED path.

    It used to call `get_model_info` directly, which returns the `-1` placeholder
    for any model it does not ship; the spec then normalised that to the 128k
    unknown default. So a 1M-context model ran as a 128k one even though the
    enrichment had already learned its real window — and the spec is what the
    session runs on, so compaction sized itself off the wrong number and threw away
    eight times the room it had.
    """
    from local_operator.model import configure as configure_mod

    _stub_discovery(monkeypatch, [_row("vendor/huge", context_window=1_000_000, max_tokens=65_536)])
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    spec = configure_mod.build_model_spec("openrouter", "vendor/huge")
    assert spec.context_window == 1_000_000, "the placeholder path was taken"
    assert spec.max_output_tokens == 65_536


def test_the_spec_still_prefers_a_real_registry_entry(monkeypatch) -> None:
    """A shipped model must not pay for a listing call, and must keep its own
    numbers rather than a listing's."""
    from local_operator.model import configure as configure_mod
    from local_operator.model import discovery as discovery_mod

    def explode(*_args, **_kwargs):
        raise AssertionError("a known model must not trigger discovery")

    monkeypatch.setattr(discovery_mod, "available_models", explode)
    spec = configure_mod.build_model_spec("anthropic", "claude-sonnet-4-20250514")
    assert spec.context_window == 200_000


@pytest.mark.parametrize(
    "provider, model_id, window, max_out",
    [
        # Google: the registry ships `gemini-2.5-pro-preview-05-06` and nothing
        # for the current flagships, and its wire is neither openai-compat nor
        # an aggregator - the shape most likely to be missed by an enrichment
        # gate written around routers.
        ("google", "gemini-3-pro", 2_097_152, 65_536),
        # Alibaba/DashScope: openai-compat, but a DIRECT provider, so it was on
        # the wrong side of the old `canonical in LISTING_PROVIDERS` gate too.
        ("alibaba", "qwen3-max", 262_144, 32_768),
    ],
)
def test_a_direct_provider_gets_its_real_window_from_its_own_listing(
    monkeypatch, tmp_path, provider: str, model_id: str, window: int, max_out: int
) -> None:
    """Enrichment must cover DIRECT providers, not only the aggregators.

    The original gate was `canonical in LISTING_PROVIDERS`, i.e. openrouter and
    radient. Every other provider fell through to the shipped registry, and the
    registry only knows the models that existed when it was written: a user who
    names `gemini-3-pro` or `qwen3-max` - both real, both current - got the 128k
    `UNKNOWN_CONTEXT_WINDOW` placeholder, and compaction sized itself off that
    instead of the 2M/262k the provider would have reported for free.

    Scope, stated honestly: `_stub_discovery` replaces `available_models`
    WHOLESALE, so nothing below that seam runs — this test passes unchanged if
    `_fetch_gemini` is deleted. What it pins is the resolution gate above the seam,
    which is provider-agnostic, and both parameters exercise the same code path.
    The wire-specific claim belongs to a test that speaks HTTP:
    :func:`test_the_gemini_wire_reaches_the_spec_through_a_real_transport` below,
    plus the per-transport tests in `test_discovery.py`.
    """
    from local_operator.model import configure as configure_mod

    _stub_discovery(monkeypatch, [_row(model_id, context_window=window, max_tokens=max_out)])
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    spec = configure_mod.build_model_spec(provider, model_id)
    assert spec.context_window == window, "a direct provider fell back to the placeholder"
    assert spec.max_output_tokens == max_out


# -- the wire, not the seam --------------------------------------------------


class _CannedHttp(httpx.Client):
    """The slice of `httpx.Client` the transports use, answering canned bodies.

    Same shape as `test_discovery.py`'s stub client, kept here rather than shared
    because this module needs it for exactly one boundary test and importing test
    helpers across modules is how a fixture ends up serving two contracts.
    """

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = list(pages)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(
        self,
        url: Any,
        *,
        params: Any = None,
        headers: Any = None,
        timeout: Any = None,
        **kwargs: Any,
    ) -> "_CannedResponse":
        self.calls.append((str(url), dict(params or {})))
        assert self._pages, f"unexpected extra request to {url}"
        return _CannedResponse(self._pages.pop(0))


class _CannedResponse(httpx.Response):
    def __init__(self, body: dict[str, Any]) -> None:
        self.status_code = 200
        self._body = body

    def json(self, **kwargs: Any) -> dict[str, Any]:
        return self._body


@pytest.mark.parametrize("spelling", ["gemini-2.5-pro", "models/gemini-2.5-pro", "Gemini-2.5-Pro"])
def test_the_gemini_wire_reaches_the_spec_through_a_real_transport(
    monkeypatch, tmp_path, spelling: str
) -> None:
    """End to end from Google's real envelope to the ModelSpec, with only the
    socket replaced.

    Everything between runs for real: `_fetch_gemini`'s query-parameter auth and
    pagination, `_row_from_gemini_entry`'s `models/` strip and `generateContent`
    filter, the merge, the cache, and the resolution gate. Deleting `_fetch_gemini`
    fails this test, which is the wire-specific claim the seam-level test above
    cannot make.

    Parametrized over spellings because discovery NORMALISES ids on ingest while
    the user types what Google's own docs show. `models/gemini-2.5-pro` matched
    nothing under an exact-only comparison and that session ran at 128k — a model
    with 1M of context compacting at ~102k.
    """
    from local_operator.model import configure as configure_mod
    from local_operator.model import discovery as discovery_mod

    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEY", "AIza-test-key")
    http = _CannedHttp(
        [
            {
                "models": [
                    {
                        "name": "models/gemini-2.5-pro",
                        "displayName": "Gemini 2.5 Pro",
                        "inputTokenLimit": 1_048_576,
                        "outputTokenLimit": 65_536,
                        "supportedGenerationMethods": ["generateContent", "countTokens"],
                    },
                    {
                        "name": "models/text-embedding-004",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                ]
            }
        ]
    )
    real_available_models = discovery_mod.available_models

    def wired(provider_id, **kwargs):
        # `_info_from_discovery` owns the call and passes neither a client nor a
        # cache dir, so the transport is injected here. Everything below this line
        # is production code.
        kwargs.setdefault("cache_dir", tmp_path)
        return real_available_models(provider_id, client=http, **kwargs)

    monkeypatch.setattr(discovery_mod, "available_models", wired)

    spec = configure_mod.build_model_spec("google", spelling)
    assert spec.context_window == 1_048_576, "the listing's window never reached the spec"
    assert spec.max_output_tokens == 65_536
    assert http.calls, "the transport was never exercised"
    assert http.calls[0][1]["key"] == "AIza-test-key"


# -- what the enrichment gate lets through -----------------------------------


def test_a_registry_row_with_a_real_window_but_no_price_still_gets_enriched(
    monkeypatch, tmp_path
) -> None:
    """A window-only gate leaves nine shipped rows priced at $0 forever.

    `google/gemini-2.0-flash-exp`, `google/gemini-2.0-pro-exp-02-05`, the
    `alibaba/qwen2.5-coder-*` pair and five more all ship with a real window and no
    prices. Under a gate that asks only about the window they can never enter
    enrichment, so their cost never resolves and the status band reads "cost
    unavailable" for the life of the install — while this module's contract names
    prices as one of the things enrichment fixes.
    """
    from local_operator.model import configure as configure_mod
    from local_operator.model.registry import get_model_info

    shipped = get_model_info("alibaba", "qwen2.5-coder-1.5b-instruct")
    assert shipped.context_window == 32_768 and not shipped.input_price, "fixture drifted"

    _stub_discovery(
        monkeypatch,
        [_row("qwen2.5-coder-1.5b-instruct", context_window=0, input_price=0.3, output_price=0.9)],
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    info = configure_mod.resolve_model_info("alibaba", "qwen2.5-coder-1.5b-instruct")
    assert info.input_price == pytest.approx(0.3)
    assert info.output_price == pytest.approx(0.9)
    # The listing had no window; the registry's own must survive the merge.
    assert info.context_window == 32_768


def test_a_fully_described_model_performs_zero_discovery(monkeypatch) -> None:
    """The second reason to enter must not become a reason to enter ALWAYS.

    Session start is on this path, so a listing call for a model the registry
    already describes completely would be latency paid by every user on every
    start, for nothing.
    """
    from local_operator.model import configure as configure_mod
    from local_operator.model import discovery as discovery_mod

    def explode(*_args, **_kwargs):
        raise AssertionError("a fully described model must not trigger discovery")

    monkeypatch.setattr(discovery_mod, "available_models", explode)

    info = configure_mod.resolve_model_info("anthropic", "claude-3-5-sonnet-20241022")
    assert info.context_window == 200_000
    assert info.input_price == pytest.approx(3.0)


def test_the_resolved_info_is_never_the_registrys_own_object(monkeypatch) -> None:
    """One session mutating its `ModelInfo` must not rewrite the process registry.

    `get_model_info` hands out module-level singletons and `ModelInfo` is a mutable
    pydantic model, so a memo that returns them shares one object across every
    session in the process — a server or a TUI resolving many models. Latent until
    someone writes to `config.info`, and by then the corruption is global and
    silent.
    """
    from local_operator.model import configure as configure_mod
    from local_operator.model.registry import anthropic_models

    shipped = anthropic_models["claude-3-5-sonnet-20241022"]
    first = configure_mod.resolve_model_info("anthropic", "claude-3-5-sonnet-20241022")
    assert first is not shipped

    first.context_window = 1
    assert shipped.context_window == 200_000, "a caller corrupted the shipped registry"
    second = configure_mod.resolve_model_info("anthropic", "claude-3-5-sonnet-20241022")
    assert second.context_window == 200_000, "the memo served the mutated object"


def test_the_memo_can_be_dropped_when_the_cause_of_a_bad_answer_is_fixed(monkeypatch) -> None:
    """A resolution that degraded for a FIXABLE reason must not be pinned for a
    full TTL bucket: the user who pastes a key mid-session has removed the cause,
    and 24h of stale numbers is not an acceptable answer to that.

    Uses a generation the registry cannot describe, because the point is a
    resolution with nothing but the fallback behind it: a shipped id would carry its
    own name and hide whether the memo ever re-resolved.
    """
    from local_operator.model import configure as configure_mod
    from local_operator.model.registry import anthropic_models

    future = "claude-opus-9"
    assert future not in anthropic_models, "the id this test treats as unshipped now ships"

    _stub_discovery(monkeypatch, [])
    configure_mod.invalidate_model_info_cache()
    degraded = configure_mod.resolve_model_info("anthropic", future)
    assert degraded.name == future, "the listing had nothing to add"

    _stub_discovery(monkeypatch, [_anthropic_row(future, "Claude Opus 9")])
    assert configure_mod.resolve_model_info("anthropic", future).name == future

    configure_mod.invalidate_model_info_cache()
    assert configure_mod.resolve_model_info("anthropic", future).name == "Claude Opus 9"


def test_an_unshipped_xai_id_does_not_keep_the_unknown_placeholder_name(monkeypatch) -> None:
    """The reported Grok 4.6 band.

    xAI's listing quotes a 500k window and no display name. Resolution used to
    start from the shared ``unknown_model_info`` singleton (``name="Unknown"``)
    and keep that word when the listing had nothing to overwrite it with, so
    the status band painted ``Unknown`` for a model that was running fine.
    The fallback must describe THIS id, and a nameless listing must not invent
    one.
    """
    from local_operator.model import configure as configure_mod
    from local_operator.model.discovery import DiscoveredModel
    from local_operator.model.registry import unknown_model_info, xai_models

    model_id = "grok-4.20"
    assert model_id not in xai_models, "the id this test treats as unshipped now ships"

    _stub_discovery(
        monkeypatch,
        [DiscoveredModel(id=model_id, name="", context_window=500_000)],
    )
    configure_mod.invalidate_model_info_cache()
    info = configure_mod.resolve_model_info("xai", model_id)
    spec = configure_mod.build_model_spec("xai", model_id, info)

    assert info is not unknown_model_info
    assert info.id == model_id
    assert info.name == model_id
    assert info.name != "Unknown"
    assert info.context_window == 500_000
    # The id wearing a name's clothes is refused by the band (see naming.py),
    # so the operator sees ``grok-4.20`` rather than the placeholder word.
    assert spec.display_name in ("", model_id)
    assert spec.display_name != "Unknown"
    assert spec.context_window == 500_000


@pytest.mark.parametrize(
    "provider, model_id, supported",
    [
        # Verified live against api.anthropic.com/v1/messages: these 400 on
        # either parameter and 200 with neither.
        ("anthropic", "claude-opus-5", False),
        ("anthropic", "claude-sonnet-5", False),
        # ...and these answer 200 WITH both, so the boundary is the generation
        # digit straight after the tier. A trailing "-5" match would have taken
        # the whole 4.5 family down with it.
        ("anthropic", "claude-opus-4-5", True),
        ("anthropic", "claude-sonnet-4-5", True),
        ("anthropic", "claude-haiku-4-5", True),
        ("anthropic", "claude-3-5-sonnet-latest", True),
        # A tier name no fixed list contains. `opus|sonnet|haiku` was written when
        # those were all there were, and `claude-fable-5` — a real tier — sailed
        # straight through it and sent the pair to an endpoint that rejects it.
        # The generation is what decides, not the tier's name.
        ("anthropic", "claude-fable-5", False),
        ("anthropic", "claude-fable-4", True),
        # The aggregator returns 200 either way because it strips the pair
        # before forwarding, so the model — not the wire — decides. Same
        # answer on both routes, or the flag would be untestable and the fix
        # would evaporate the moment OpenRouter stopped normalising.
        ("openrouter", "anthropic/claude-opus-5", False),
        ("openrouter", "anthropic/claude-sonnet-4.5", True),
        # OpenAI's o-series and gpt-5 reject the same pair.
        ("openai", "o1", False),
        ("openai", "o3-mini", False),
        ("openai", "gpt-5", False),
        ("openai", "gpt-4o", True),
        # `thinking`/`reasoner` set ModelSpec.reasoning but NOT this flag:
        # Gemini and DeepSeek accept temperature on those variants, and
        # silently dropping a working setting is worse than the 400 it avoids.
        ("google", "gemini-2.5-flash-thinking", True),
        ("deepseek", "deepseek-reasoner", True),
        # Kimi's coding-plan host (api.kimi.com/coding/v1) pins the pair
        # instead of rejecting the keys: HTTP 400 "invalid temperature: only 1
        # is allowed for this model" / "invalid top_p: only 0.95 is allowed",
        # while omitting both succeeds. Verified live against k3 and
        # k2-thinking on chat/completions. Scoped to the coding-plan ids —
        # the mainland moonshot host accepts the pair on its own models, so
        # kimi-k2-0711-preview must keep its sampling settings and must not
        # be caught by the k3 arm's "k<digit>" shape.
        ("kimi", "k3", False),
        ("kimi", "k3-256k", False),
        # Live-probed alongside k3: pins the pair the same way.
        ("kimi", "k2-thinking", False),
        ("kimi", "kimi-for-coding", False),
        ("kimi", "kimi-for-coding-highspeed", False),
        ("kimi", "kimi-k2-0711-preview", True),
        ("kimi", "moonshot-v1-128k", True),
        # The id shape is kimi's, not the world's: a k-something on another
        # provider keeps its parameters.
        ("openrouter", "moonshotai/kimi-k2", True),
        # Gemini stays True on BOTH generations, and that is the point of the
        # flag: this is the hard-REJECTION capability (a 400), not the
        # "should we assert a value" question. Google ignores an unwanted
        # temperature rather than rejecting it, so the pair is dropped by the
        # sampling POLICY (a `None` on the spec) instead — which is also what
        # keeps a deliberate user override sendable. See
        # `test_gemini_3_omits_sampling_on_every_route`.
        ("google", "gemini-3.8-flash", True),
        ("google", "gemini-2.5-pro", True),
    ],
)
def test_the_spec_knows_which_models_reject_sampling_parameters(
    provider, model_id, supported
) -> None:
    """`claude-opus-5` answered HTTP 400 "`temperature` is deprecated for this
    model." on every single turn because the wire clients sent the pair
    unconditionally. The capability is derived here, once, so the clients stay
    free of model-name knowledge."""
    from local_operator.model import configure as configure_mod

    spec = configure_mod.build_model_spec(provider, model_id)
    assert spec.supports_sampling_params is supported


@pytest.mark.parametrize(
    "provider, model_id, temperature, top_p",
    [
        # OMIT rows: send nothing and let the vendor's own default apply.
        #
        # Gemini 3+ — "we recommend removing this parameter", and 3.6+ ignores
        # a custom value outright (ai.google.dev/gemini-api/docs/gemini-3).
        ("google", "gemini-3.8-flash", None, None),
        ("google", "gemini-3-flash", None, None),
        ("google", "gemini-3.1-pro-preview", None, None),
        # Forward-reading over the generation digit.
        ("google", "gemini-4-pro", None, None),
        # Model-keyed, not provider-keyed: the same weights behind an
        # aggregator behave identically. A provider-keyed rule would have fixed
        # the direct route and left these two asserting 0.2.
        ("openrouter", "google/gemini-3.8-flash", None, None),
        ("radient", "google/gemini-3-pro", None, None),
        # Anthropic 4.7+ — "Models released after Claude Opus 4.6 do not
        # support setting temperature ... all other values will be rejected
        # with a 400 error." The app sent 0.2, so this was a live outage.
        ("anthropic", "claude-opus-4-7", None, None),
        ("anthropic", "claude-opus-4-8", None, None),
        ("anthropic", "claude-sonnet-4.7", None, None),
        # OpenAI GPT-6 — "Remove `temperature`, `top_p`, and `top_logprobs`."
        # The literal `gpt-5` arm in _NO_SAMPLING_PARAMS never matched this.
        ("openai", "gpt-6-astra", None, None),
        # DeepSeek V4 — thinking on by default, where the pair is inert.
        ("deepseek", "deepseek-v4-pro", None, None),
        # Moonshot/Kimi — pinned to fixed values across the current lineup.
        ("kimi", "kimi-k2-0711-preview", None, None),
        # Z.AI/GLM — per-SERIES defaults (1.0 vs 0.6), so no single seed is
        # right for all of them.
        ("zai", "glm-5.3", None, None),
        # Mistral — publishes no static default by design.
        ("mistral", "mistral-large-latest", None, None),
        # xAI — defaults are UNKNOWN in the vendor's own reference.
        ("xai", "grok-4", None, None),
        # Ollama — never overrule the publisher's Modelfile.
        ("ollama", "qwen3:32b", None, None),
        # The fallback, and the most consequential row: an unknown model gets
        # silence rather than an invented 0.2.
        ("openrouter", "some-vendor/model-nobody-has-characterised", None, None),
        # SEED rows: the vendor documents a value silence would not reproduce.
        ("google", "gemini-2.5-pro", 1.0, 0.95),
        ("google", "gemini-2.0-flash", 1.0, 0.95),
        # The `reasoning` markers must not divert a 2.x model into an OMIT row.
        ("google", "gemini-2.5-flash-thinking", 1.0, 0.95),
        # A snapshot date must not read as a generation number — the trap
        # `_EFFORT_TABLE` documents and the `\d{2,3}` bound closes.
        ("google", "gemini-2.5-pro-20250314", 1.0, 0.95),
        # Qwen documents temperature 0.7 with top_p 0.8 — the pair diverges,
        # which is why the knobs are decided independently.
        ("alibaba", "qwen3-coder-plus", 0.7, 0.8),
        # Anthropic 4.5/4.6 and gpt-4o still HONOUR the pair — they keep
        # `supports_sampling_params=True` and must not regress into the
        # hard-rejection arm — but they still stop being TOLD 0.2. Their
        # vendors document 1.0, so the app-wide constant was an assertion over
        # a documented default here exactly as it was everywhere else; these
        # fall through to the fallback and let the vendor decide.
        ("anthropic", "claude-sonnet-4-5", None, None),
        ("anthropic", "claude-3-5-sonnet-latest", None, None),
        ("openai", "gpt-4o", None, None),
    ],
)
def test_the_sampling_policy_answers_per_family(provider, model_id, temperature, top_p) -> None:
    """The app shipped temperature 0.2 / top_p 0.9 to every provider on every
    turn — an invented pair asserted over each vendor's own documented default,
    which no major vendor sets anywhere near. `None` here means the key is
    omitted so the vendor's default applies; a number means the vendor
    documents something silence would not reproduce."""
    from local_operator.model import configure as configure_mod

    spec = configure_mod.build_model_spec(provider, model_id)
    assert spec.temperature == temperature
    assert spec.top_p == top_p


@pytest.mark.parametrize(
    "dotted, hyphenated",
    [
        ("gemini-3.0-flash", "gemini-3-flash"),
        ("gemini-2.5-pro", "gemini-2-5-pro"),
    ],
)
class TestADotAndAHyphenNameTheSameGeneration:
    """Vendors ship both spellings, and a rule that reads only one silently
    misclassifies the other on exactly the models it exists to protect."""

    def test_both_spellings_get_the_same_sampling_policy(self, dotted, hyphenated) -> None:
        from local_operator.model import configure as configure_mod

        one = configure_mod.build_model_spec("google", dotted)
        other = configure_mod.build_model_spec("google", hyphenated)
        assert one.temperature == other.temperature
        assert one.top_p == other.top_p


def test_configure_model_does_not_stomp_the_models_own_sampling_policy() -> None:
    """Trap: ``configure_model`` used to copy ``temperature``/``top_p`` onto the
    spec unconditionally from parameters defaulting to the app-wide 0.2/0.9,
    overwriting the model's own policy on the MAIN construction path. Without
    this fix the whole table is dead on arrival."""
    from local_operator.model.configure import configure_model

    seeded = configure_model("google", "gemini-2.5-pro")
    assert seeded.spec.temperature == 1.0
    assert seeded.spec.top_p == 0.95
    # ...and the legacy scalar attributes report what will actually be sent.
    assert seeded.temperature == 1.0
    assert seeded.top_p == 0.95

    omitted = configure_model("google", "gemini-3.8-flash")
    assert omitted.spec.temperature is None
    assert omitted.spec.top_p is None


def test_an_explicit_override_still_wins_on_every_family() -> None:
    """The escape hatch. A user or agent record that deliberately sets a value
    — and the server's per-request ``options`` — must still win, including on
    families whose policy is to omit. Determinism-seeking users must not be
    stranded by a table written for the default case."""
    from local_operator.model.configure import configure_model

    for provider, model_id in (
        ("google", "gemini-3.8-flash"),
        ("anthropic", "claude-opus-4-7"),
        ("ollama", "qwen3:32b"),
        ("google", "gemini-2.5-pro"),
    ):
        overridden = configure_model(provider, model_id, temperature=0.0, top_p=0.5)
        assert overridden.spec.temperature == 0.0, model_id
        assert overridden.spec.top_p == 0.5, model_id

    # A partial override leaves the other knob on the model's own policy.
    partial = configure_model("google", "gemini-2.5-pro", temperature=0.3)
    assert partial.spec.temperature == 0.3
    assert partial.spec.top_p == 0.95


def test_a_providers_listing_may_narrow_the_sampling_policy_but_never_widen_it() -> None:
    """OpenRouter publishes a ``supported_parameters`` allowlist, and 81 of 425
    live models omit ``temperature`` from it — a per-model fact the curated
    table cannot keep up with, so an unknown model gets it right for free.

    But it may only NARROW. The listing answers "will the aggregator forward
    this", not "will the model honour it": `google/gemini-3.8-flash` DOES list
    temperature as supported, because OpenRouter forwards it and Google then
    ignores it. A listing must therefore never turn a curated OMIT back into a
    send.
    """
    from pydantic import BaseModel, ConfigDict

    from local_operator.model import configure as configure_mod

    class _Entry(BaseModel):
        # Mirrors the real listing schemas, which declare no sampling fields and
        # carry them as extras; built via `model_validate` so the undeclared
        # keys are the extras the production path actually reads.
        model_config = ConfigDict(extra="allow")

    configure_mod._sampling_support_memo.clear()
    try:
        # NARROWING: a family the table seeds, whose listing denies the key.
        seeded = configure_mod.build_model_spec("openrouter", "alibaba/qwen3-coder-plus")
        assert seeded.temperature == 0.7
        configure_mod._remember_listing_sampling_support(
            "openrouter",
            "alibaba/qwen3-coder-plus",
            _Entry.model_validate(
                {
                    "id": "alibaba/qwen3-coder-plus",
                    "supported_parameters": ["max_tokens", "tools"],
                }
            ),
        )
        narrowed = configure_mod.build_model_spec("openrouter", "alibaba/qwen3-coder-plus")
        assert narrowed.temperature is None
        assert narrowed.top_p is None

        # NO WIDENING: the listing says temperature is supported, but Google
        # ignores it, so the curated OMIT must survive.
        configure_mod._remember_listing_sampling_support(
            "openrouter",
            "google/gemini-3.8-flash",
            _Entry.model_validate(
                {
                    "id": "google/gemini-3.8-flash",
                    "supported_parameters": ["temperature", "top_p", "max_tokens"],
                }
            ),
        )
        still_omitted = configure_mod.build_model_spec("openrouter", "google/gemini-3.8-flash")
        assert still_omitted.temperature is None
        assert still_omitted.top_p is None

        # SILENCE defers to the table: a listing with no allowlist at all (the
        # Radient case, whose passthrough semantics are unverified) changes
        # nothing.
        configure_mod._remember_listing_sampling_support(
            "radient",
            "google/gemini-2.5-pro",
            _Entry.model_validate({"id": "google/gemini-2.5-pro"}),
        )
        untouched = configure_mod.build_model_spec("radient", "google/gemini-2.5-pro")
        assert untouched.temperature == 1.0
        assert untouched.top_p == 0.95
    finally:
        configure_mod._sampling_support_memo.clear()


@pytest.mark.parametrize(
    "provider, model_id",
    [
        # "all other values will be rejected with a 400 error"
        ("anthropic", "claude-opus-4-7"),
        ("anthropic", "claude-opus-4-8"),
        ("anthropic", "claude-sonnet-4.7"),
        # "Unsupported parameters: Remove `temperature`, `top_p`"
        ("openai", "gpt-6-astra"),
        # The aggregator routes reach the same weights and must agree.
        ("openrouter", "anthropic/claude-opus-4-7"),
        ("radient", "openai/gpt-6-astra"),
    ],
)
def test_a_reject_family_suppresses_an_override_instead_of_400ing(provider, model_id) -> None:
    """REGRESSION (R1): dropping only our own default is not enough on a family
    whose vendor REJECTS the parameter. A stored agent temperature — the shape
    the server's own API examples advertise — would otherwise ride to the wire
    and fail every single turn, which is the PR's headline outage relocated
    from the default path onto the override path.

    This is the distinction between "the vendor ignores it" (an override is
    harmless) and "the vendor rejects it" (an override is a failed turn).
    """
    from local_operator.model.configure import configure_model

    configured = configure_model(provider, model_id, None, temperature=0.2, top_p=0.9)
    assert configured.spec.supports_sampling_params is False

    request = ChatRequest(model=configured.spec, messages=[Message.user("hi")])
    body = OpenAICompatClient("https://x")._build_body(request)
    assert "temperature" not in body
    assert "top_p" not in body


@pytest.mark.parametrize(
    "provider, model_id",
    [
        # Ignores an unwanted value, so an override costs nothing.
        ("google", "gemini-3.8-flash"),
        # Genuinely honours the pair.
        ("anthropic", "claude-sonnet-4-5"),
        ("google", "gemini-2.5-pro"),
        # Legacy Kimi ids accept the pair; only the documented coding-host ids
        # are hard-suppressed, via _KIMI_PINNED_SAMPLING.
        ("kimi", "kimi-k2-0711-preview"),
    ],
)
def test_an_ignore_or_honour_family_still_forwards_an_override(provider, model_id) -> None:
    """The other half of R1: suppression must not spread. A user who asks for
    determinism on a family that ignores or honours the value still gets it."""
    from local_operator.model.configure import configure_model

    configured = configure_model(provider, model_id, None, temperature=0.2, top_p=0.9)
    request = ChatRequest(model=configured.spec, messages=[Message.user("hi")])
    body = OpenAICompatClient("https://x")._build_body(request)
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9


@pytest.mark.parametrize(
    "provider, model_id, expected",
    [
        # OMIT families: the errand must not re-assert a value the main turn
        # deliberately stopped sending.
        ("openai", "gpt-6-astra", None),
        ("google", "gemini-3.8-flash", None),
        ("deepseek", "deepseek-v4-pro", None),
        ("anthropic", "claude-opus-4-7", None),
        # A seeded family still gets its documented value on the errand.
        ("google", "gemini-2.5-pro", 1.0),
        ("alibaba", "qwen3-coder-plus", 0.7),
    ],
)
def test_the_title_errand_follows_the_same_sampling_policy_as_a_turn(
    provider, model_id, expected
) -> None:
    """REGRESSION (Q1): the conversation-titling errand built its own
    ``ChatRequest(temperature=0)``, and a request value OUTRANKS the spec in
    ``_sampling_params`` — so the errand re-asserted 0.0 on exactly the families
    the policy table exists for, while the main turn correctly sent nothing.

    The failure mode is quiet: the errand is ``isolated`` with one attempt and
    no fallback, so a rejected request surfaces only as a conversation that
    never gets a title.

    Mirrors the construction at ``session.py``'s ``_title_via_model``.
    """
    spec = configure_mod.build_model_spec(provider, model_id)
    request = ChatRequest(
        model=spec,
        messages=[Message.user("t")],
        tools=[],
        tool_choice="none",
        max_tokens=32,
        temperature=spec.temperature,
    )
    body = OpenAICompatClient("https://x")._build_body(request)
    if expected is None:
        assert "temperature" not in body
    else:
        assert body["temperature"] == expected


def test_a_hard_rejection_still_beats_an_explicit_override() -> None:
    """The one place an override does NOT win, and deliberately so: a family
    that answers HTTP 400 makes the turn fail rather than honour a preference.
    ``supports_sampling_params`` is that separate, stronger capability."""
    from local_operator.model.configure import configure_model

    rejected = configure_model("anthropic", "claude-opus-5", temperature=0.7)
    assert rejected.spec.supports_sampling_params is False


def test_concurrent_cold_misses_fetch_exactly_once(tmp_path, monkeypatch) -> None:
    """DA5: N processes cold-missing one document must not fan out N fetches.

    The lease's whole job: the winner fetches, losers wait briefly and serve
    the winner's document. A fan-out would hammer the same public listing
    endpoint (429 risk) for data a single request already fetched.
    """
    import threading
    import time as time_mod

    from local_operator.model.catalogue import cached_listing

    fetches: list[float] = []
    lock = threading.Lock()
    release = threading.Event()

    def slow_fetch() -> dict[str, Any]:
        with lock:
            fetches.append(time_mod.perf_counter())
        # Hold the fetch open long enough for every loser to arrive and be
        # told to wait; a real listing round trip is slower than this.
        release.wait(timeout=5.0)
        return {"capture": 1, "models": [{"id": "m1"}]}

    results: list[dict[str, Any] | None] = [None] * 5
    threads = []

    def worker(index: int) -> None:
        results[index] = cached_listing("test.listing", slow_fetch, ttl_s=3600, cache_dir=tmp_path)

    try:
        for index in range(5):
            thread = threading.Thread(target=worker, args=(index,))
            thread.start()
            threads.append(thread)
            # Stagger slightly so the first thread reliably wins the lease.
            time_mod.sleep(0.05)
        # All five have run; let the winner's fetch complete.
        time_mod.sleep(0.2)
        release.set()
        for thread in threads:
            thread.join(timeout=5.0)
    finally:
        release.set()

    assert len(fetches) == 1, f"expected exactly one live fetch, got {len(fetches)}"
    assert all(result is not None for result in results), results


def test_a_lease_from_a_dead_holder_is_stolen_after_expiry(tmp_path) -> None:
    """A crashed fetcher must not strand the document for the lease TTL."""
    import json
    import time as time_mod

    from local_operator.model.catalogue import _ListingFetchLease

    document = tmp_path / "test.listing.json"
    document.write_text(json.dumps({"payload": {"old": True}, "fetched_at": 0.0}))
    lease = _ListingFetchLease(document)
    # Simulate a dead peer's lease: already expired.
    stale = tmp_path / "test.listing.json.fetching"
    stale.write_text(json.dumps({"holder": "dead:0000", "expires_at": time_mod.time() - 1.0}))
    assert lease.acquire() is True
    assert lease._held is True
    lease.release()
    assert not stale.exists(), "release must remove this holder's lease file"


# -- stale-while-revalidate --------------------------------------------------
#
# The incident these guard: a 22h-old `anthropic.listing.json` — inside the 24h
# hard TTL, so never refetched — hid a model published that morning from
# `/model` and from the cost band for a whole working day. `read_listing` now
# serves a document past the SOFT TTL immediately and refreshes it off the
# calling path, so the NEXT call sees the new document at zero on-path cost.
# Threads are JOINED via `catalogue._revalidation_threads()`, never slept on
# (AGENTS.md "Timing").


def _plant(tmp_path, key: str, payload: dict[str, Any], *, age_s: float) -> None:
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"fetched_at": time.time() - age_s, "payload": payload}), encoding="utf-8"
    )


def _join_revalidations() -> None:
    for thread in catalogue._revalidation_threads():
        thread.join(timeout=5.0)


@pytest.fixture
def _no_backoff_state():
    """Each test starts with no in-flight or recently-attempted keys."""
    with catalogue._revalidate_lock:
        catalogue._revalidating.clear()
        catalogue._last_attempt.clear()
        catalogue._threads.clear()
    yield
    _join_revalidations()


def test_a_document_between_soft_and_hard_ttl_is_served_and_revalidated_in_the_background(
    tmp_path, _no_backoff_state
) -> None:
    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=2 * 3600)
    calls: list[int] = []

    def fetch():
        calls.append(1)
        return _payload(window=2)

    listing = catalogue.read_listing("anthropic.listing", fetch, cache_dir=tmp_path)

    # Served the OLD document, synchronously, with no fetch on this path...
    assert listing.payload == _payload(window=1)
    assert listing.fetched is False
    assert listing.refreshing is True
    assert 2 * 3600 <= listing.age_s < 3 * 3600
    # ...and the background thread rewrote it.
    _join_revalidations()
    assert calls == [1]
    raw = json.loads((tmp_path / "anthropic.listing.json").read_text(encoding="utf-8"))
    assert raw["payload"] == _payload(window=2)
    assert time.time() - raw["fetched_at"] < 60
    assert not list(tmp_path.glob("*.fetching")), "the background fetch must release its lease"


def test_a_document_under_the_soft_ttl_schedules_nothing(tmp_path, _no_backoff_state) -> None:
    _plant(tmp_path, "anthropic.listing", _payload(), age_s=30 * 60)
    listing = catalogue.read_listing(
        "anthropic.listing", lambda: pytest.fail("no fetch expected"), cache_dir=tmp_path
    )
    assert listing.refreshing is False
    assert listing.fetched is False
    assert catalogue._revalidation_threads() == []


def test_a_document_past_the_hard_ttl_still_fetches_synchronously(
    tmp_path, _no_backoff_state
) -> None:
    """The offline/cold contract: beyond the hard TTL the CALLER gets the fresh
    document, not a promise of one."""
    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=25 * 3600)
    listing = catalogue.read_listing(
        "anthropic.listing", lambda: _payload(window=2), cache_dir=tmp_path
    )
    assert listing.payload == _payload(window=2)
    assert listing.fetched is True
    assert listing.refreshing is False
    assert catalogue._revalidation_threads() == []


def test_a_background_revalidation_that_loses_the_lease_exits_without_fetching(
    tmp_path, _no_backoff_state
) -> None:
    """A peer process is already fetching; unlike the sync path there is nothing
    to wait for, because the caller already has its answer."""
    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=2 * 3600)
    (tmp_path / "anthropic.listing.json.fetching").write_text(
        json.dumps({"holder": "peer:1234", "expires_at": time.time() + 30.0})
    )
    calls: list[int] = []

    def fetch():
        calls.append(1)
        return _payload(window=2)

    listing = catalogue.read_listing("anthropic.listing", fetch, cache_dir=tmp_path)
    assert listing.refreshing is True
    _join_revalidations()
    assert calls == []
    raw = json.loads((tmp_path / "anthropic.listing.json").read_text(encoding="utf-8"))
    assert raw["payload"] == _payload(window=1)


def test_a_failed_background_revalidation_keeps_the_stale_document_and_backs_off(
    tmp_path, _no_backoff_state
) -> None:
    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=2 * 3600)
    calls: list[int] = []

    def boom():
        calls.append(1)
        raise RuntimeError("offline")

    first = catalogue.read_listing("anthropic.listing", boom, cache_dir=tmp_path)
    assert first.refreshing is True
    _join_revalidations()
    assert calls == [1]
    raw = json.loads((tmp_path / "anthropic.listing.json").read_text(encoding="utf-8"))
    assert raw["payload"] == _payload(window=1), "a failed refresh must not touch the document"
    assert not list(tmp_path.glob("*.fetching"))

    # Within the backoff: served again, no second thread — an offline machine
    # must not spawn a thread per repaint.
    second = catalogue.read_listing("anthropic.listing", boom, cache_dir=tmp_path)
    assert second.payload == _payload(window=1)
    assert second.refreshing is False
    assert catalogue._revalidation_threads() == []
    assert calls == [1]


def test_two_reads_in_one_process_spawn_one_revalidation(tmp_path, _no_backoff_state) -> None:
    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=2 * 3600)
    gate = __import__("threading").Event()
    calls: list[int] = []

    def slow_fetch():
        calls.append(1)
        gate.wait(timeout=5.0)
        return _payload(window=2)

    try:
        first = catalogue.read_listing("anthropic.listing", slow_fetch, cache_dir=tmp_path)
        second = catalogue.read_listing("anthropic.listing", slow_fetch, cache_dir=tmp_path)
    finally:
        gate.set()
    _join_revalidations()
    assert (first.refreshing, second.refreshing) == (True, False)
    assert calls == [1]


def test_cached_listing_never_schedules_a_thread(tmp_path, _no_backoff_state) -> None:
    """The compatibility wrapper pins soft TTL to hard TTL: three states, no threads."""
    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=2 * 3600)
    got = cached_listing(
        "anthropic.listing", lambda: pytest.fail("no fetch expected"), cache_dir=tmp_path
    )
    assert got == _payload(window=1)
    assert catalogue._revalidation_threads() == []


def test_peek_listing_reads_without_sweeping_or_fetching(tmp_path) -> None:
    orphan = tmp_path / "openrouter.models.json"
    orphan.write_text("{}", encoding="utf-8")
    _plant(tmp_path, "models-dev.listing", {"etag": '"abc"'}, age_s=10)

    peeked = catalogue.peek_listing("models-dev.listing", cache_dir=tmp_path)
    assert peeked.payload == {"etag": '"abc"'}
    assert 10 <= peeked.age_s < 60
    assert peeked.fetched is False and peeked.refreshing is False
    assert orphan.exists(), "peek must not sweep; only read_listing owns the purge"
    absent = catalogue.peek_listing("nothing.listing", cache_dir=tmp_path)
    assert absent.payload is None and absent.age_s == float("inf")


def test_a_background_thread_runs_off_the_calling_thread(tmp_path, _no_backoff_state) -> None:
    """Structural, not timed (AGENTS.md): the fetch runs on a thread that is not
    the caller's, which is the whole point of the soft TTL."""
    import threading

    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=2 * 3600)
    seen: list[int] = []

    def fetch():
        seen.append(threading.get_ident())
        return _payload(window=2)

    catalogue.read_listing("anthropic.listing", fetch, cache_dir=tmp_path)
    _join_revalidations()
    assert seen and seen[0] != threading.get_ident()


def test_a_readers_predicate_can_expire_a_document_inside_the_ttl(
    tmp_path, _no_backoff_state
) -> None:
    """`refetch_if` is how discovery's `want_id` miss becomes ONE synchronous fetch
    on the calling thread, rather than a race against the background refresh."""
    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=2 * 3600)
    asked: list[tuple[dict[str, Any], float]] = []

    def lacks(document, age_s):
        asked.append((document, age_s))
        return True

    listing = catalogue.read_listing(
        "anthropic.listing", lambda: _payload(window=2), cache_dir=tmp_path, refetch_if=lacks
    )
    assert listing.fetched is True
    assert listing.payload == _payload(window=2)
    assert listing.refreshing is False
    assert catalogue._revalidation_threads() == []
    assert asked[0][0] == _payload(window=1) and asked[0][1] >= 2 * 3600


def test_a_failed_sync_fetch_reports_failed_with_the_stale_document(tmp_path) -> None:
    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=25 * 3600)

    def boom():
        raise RuntimeError("offline")

    listing = catalogue.read_listing("anthropic.listing", boom, cache_dir=tmp_path)
    assert listing.payload == _payload(window=1)
    assert (listing.fetched, listing.failed) == (False, True)


def test_the_background_refresh_runs_the_revalidate_thunk_not_the_on_path_one(
    tmp_path, _no_backoff_state
) -> None:
    """The two thunks differ in BUDGET: `fetch` carries the caller's ceiling (2 s
    from a repaint), `revalidate` the provider's full default. A background
    refresh that inherited the on-path budget failed on every link slower than
    2 s, backed off, and left the document to the 24 h sync path."""
    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=2 * 3600)
    ran: list[str] = []

    def on_path():
        ran.append("fetch")
        return _payload(window=2)

    def off_path():
        ran.append("revalidate")
        return _payload(window=3)

    listing = catalogue.read_listing(
        "anthropic.listing", on_path, cache_dir=tmp_path, revalidate=off_path
    )
    assert listing.refreshing is True
    _join_revalidations()
    assert ran == ["revalidate"]
    raw = json.loads((tmp_path / "anthropic.listing.json").read_text(encoding="utf-8"))
    assert raw["payload"] == _payload(window=3)

    # The SYNC path still runs `fetch`: past the hard TTL the caller's budget
    # is the right one, because the caller is waiting on it.
    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=25 * 3600)
    ran.clear()
    listing = catalogue.read_listing(
        "anthropic.listing", on_path, cache_dir=tmp_path, revalidate=off_path
    )
    assert listing.fetched is True and ran == ["fetch"]


def test_a_sync_reader_that_loses_the_lease_to_its_own_thread_joins_it(
    tmp_path, _no_backoff_state
) -> None:
    """R1-5: call A serves the soft-window document and schedules a refresh; call
    B, declared expired by its predicate, finds the lease held by A's thread.
    It must join that thread — returning the moment the refresh lands, with
    the refreshed document — rather than poll the mtime for the whole window
    or serve the old document (which the caller's memo would pin for a day)."""
    import threading

    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=2 * 3600)
    gate = threading.Event()
    entered = threading.Event()  # set once A's thread HOLDS the lease
    sync_calls: list[int] = []

    def slow_refresh():
        entered.set()
        gate.wait(timeout=5.0)
        return _payload(window=2)

    def sync_fetch():
        sync_calls.append(1)
        return _payload(window=99)

    first = catalogue.read_listing("anthropic.listing", slow_refresh, cache_dir=tmp_path)
    assert first.refreshing is True
    thread = catalogue._revalidation_threads()[0]
    # Not a clock wait: B must start AFTER the thread took the lease, or B
    # would legitimately win it and fetch on its own.
    assert entered.wait(timeout=5.0)

    result: list[catalogue.Listing] = []

    def second_call():
        result.append(
            catalogue.read_listing(
                "anthropic.listing",
                sync_fetch,
                cache_dir=tmp_path,
                refetch_if=lambda _payload_, _age: True,
            )
        )

    joiner = threading.Thread(target=second_call)
    joiner.start()
    # B is parked on A's thread, not fetching on its own and not returning
    # the stale document: nothing has been served yet.
    joiner.join(timeout=0.3)
    assert joiner.is_alive() and result == []
    gate.set()
    joiner.join(timeout=5.0)
    thread.join(timeout=5.0)

    assert sync_calls == [], "B must not fetch behind its own process's refresh"
    assert result[0].payload == _payload(window=2), "B serves what its own thread wrote"
    assert result[0].fetched is False and result[0].failed is False


def test_a_sync_reader_joining_its_own_failed_thread_returns_at_once(
    tmp_path, _no_backoff_state
) -> None:
    """The case the mtime poll got wrong: a refresh that FAILS never renames a
    document, so a poller waits out the whole window. Joining returns as soon
    as the thread does, with the stale document."""
    import threading

    _plant(tmp_path, "anthropic.listing", _payload(window=1), age_s=2 * 3600)
    gate = threading.Event()
    entered = threading.Event()

    def failing_refresh():
        entered.set()
        gate.wait(timeout=5.0)
        raise RuntimeError("offline")

    first = catalogue.read_listing("anthropic.listing", failing_refresh, cache_dir=tmp_path)
    assert first.refreshing is True
    assert entered.wait(timeout=5.0)

    result: list[catalogue.Listing] = []
    sync_calls: list[int] = []

    def sync_fetch():
        sync_calls.append(1)
        return _payload(window=99)

    def second_call():
        result.append(
            catalogue.read_listing(
                "anthropic.listing",
                sync_fetch,
                cache_dir=tmp_path,
                refetch_if=lambda _payload_, _age: True,
            )
        )

    joiner = threading.Thread(target=second_call)
    joiner.start()
    joiner.join(timeout=0.3)
    assert joiner.is_alive() and result == [], "B is parked on A's thread"
    released = time.monotonic()
    gate.set()
    joiner.join(timeout=5.0)
    _join_revalidations()

    assert sync_calls == []
    assert result[0].payload == _payload(window=1)
    # The join ended when the thread did, not when the 2 s poll window ran out.
    assert time.monotonic() - released < catalogue._LISTING_LEASE_WAIT_S / 2


# -- stranded temp files ------------------------------------------------------


def test_purge_removes_stranded_temp_files_older_than_the_gate_only(tmp_path) -> None:
    """A daemon revalidation thread killed at interpreter exit mid-`json.dump`
    leaves `<key>.listing.json.<random>.tmp` that nothing else reclaims. Only
    OLD ones go: a young one is a peer's in-flight write, and deleting it would
    strand that peer's rename."""
    import os

    old = tmp_path / "anthropic.listing.json.abc123.tmp"
    young = tmp_path / "openrouter.listing.json.def456.tmp"
    foreign = tmp_path / "pypi-local-operator.json.ghi789.tmp"  # update.py's, not ours
    document = tmp_path / "anthropic.listing.json"
    for path in (old, young, foreign, document):
        path.write_text("{", encoding="utf-8")
    stale_mtime = time.time() - catalogue._STRANDED_TEMP_MIN_AGE_S - 60
    os.utime(old, (stale_mtime, stale_mtime))
    os.utime(foreign, (stale_mtime, stale_mtime))

    catalogue.purge_stranded_temp_files(tmp_path)
    catalogue.purge_stranded_temp_files(tmp_path)  # idempotent

    assert not old.exists()
    assert young.exists(), "an in-flight peer write must not be deleted"
    assert foreign.exists(), "another writer's temp files are its own business"
    assert document.exists()


def test_read_listing_sweeps_stranded_temp_files(tmp_path, _no_backoff_state) -> None:
    import os

    _plant(tmp_path, "anthropic.listing", _payload(), age_s=10)
    stranded = tmp_path / "anthropic.listing.json.zzz.tmp"
    stranded.write_text("{", encoding="utf-8")
    stale_mtime = time.time() - catalogue._STRANDED_TEMP_MIN_AGE_S - 60
    os.utime(stranded, (stale_mtime, stale_mtime))

    catalogue.read_listing(
        "anthropic.listing", lambda: pytest.fail("no fetch expected"), cache_dir=tmp_path
    )
    assert not stranded.exists()


def test_purge_stranded_is_safe_when_the_cache_dir_does_not_exist(tmp_path) -> None:
    missing = tmp_path / "never-created"
    catalogue.purge_stranded_temp_files(missing)  # must not raise
    assert not missing.exists()


def test_the_models_dev_projection_is_not_a_provider_document(tmp_path) -> None:
    """``models-dev.listing.json`` belongs to no credential, and no provider's
    storage id is a dotted prefix of it, so no login can drop it and cost the
    machine a 4.4 MB refetch."""
    from local_operator.providers.registry import (
        PROVIDER_REGISTRY,
        credential_provider_id,
    )

    (tmp_path / "models-dev.listing.json").write_text("{}", encoding="utf-8")
    for definition in PROVIDER_REGISTRY:
        catalogue.invalidate_documents(credential_provider_id(definition.id), cache_dir=tmp_path)
    assert (tmp_path / "models-dev.listing.json").exists()


# ---------------------------------------------------------------------------
# The reasoning-effort ladder: listing over table, and the boundary that keeps
# an aggregator's answer off a direct provider's model
# ---------------------------------------------------------------------------


def test_the_listing_ladder_beats_the_hand_transcribed_table(monkeypatch, tmp_path) -> None:
    """The reported bug and its fix, through the real spec builder.

    `google/gemini-3.8-flash` matches no arm of `model.effort`'s table, so the
    status band had no effort segment at all and `shift+tab` said "not
    adjustable" — while OpenRouter publishes the ladder for that exact model.
    """
    _stub_discovery(
        monkeypatch,
        [
            _row(
                "google/gemini-3.8-flash",
                reasoning_efforts=("low", "medium", "high"),
                reasoning_default_effort="medium",
            )
        ],
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    configure_mod.invalidate_model_info_cache()

    spec = configure_mod.build_model_spec("openrouter", "google/gemini-3.8-flash")

    assert spec.reasoning_efforts == ("low", "medium", "high")
    # NOT seeded from the listing's own `default_effort`, which is the measured
    # decision rather than caution: sending that default is not the same request
    # as omitting the key (`z-ai/glm-5.3` at `max` returned 2.16x the reasoning
    # tokens of omission, n=12, non-overlapping), and it is model-specific. So
    # `default_effort` describes neither what omission does nor a level in
    # force, and painting it on the band would assert a depth that is not.
    # The reported bug is fixed by the LADDER: the segment appears (`auto`),
    # `shift+tab` and `/effort` cycle it, and the wire is unchanged.
    assert spec.reasoning_effort is None
    assert spec.reasoning_default_effort is None
    assert spec.reasoning is True


def test_the_listing_may_narrow_a_ladder_the_table_over_granted(monkeypatch, tmp_path) -> None:
    """Precedence cuts both ways, and this direction is the evidence for it. The
    table applies one transcribed `gpt-5.4` ladder to every `gpt-[5-9]` id;
    OpenRouter says `gpt-5.4-pro` takes only medium/high/xhigh there. A rung the
    route rejects is not a rung, so losing `none`/`low` is correct."""
    _stub_discovery(
        monkeypatch,
        [
            _row(
                "openai/gpt-5.4-pro",
                reasoning_efforts=("medium", "high", "xhigh"),
                reasoning_default_effort="medium",
            )
        ],
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    configure_mod.invalidate_model_info_cache()

    spec = configure_mod.build_model_spec("openrouter", "openai/gpt-5.4-pro")

    assert spec.reasoning_efforts == ("medium", "high", "xhigh")
    assert "none" not in spec.reasoning_efforts
    assert "low" not in spec.reasoning_efforts


def test_a_silent_listing_still_defers_to_the_table(monkeypatch, tmp_path) -> None:
    """The case that must not regress, and it is the common one: 91 shipped
    registry rows never fetch a listing, and every direct provider but Anthropic
    publishes no reasoning field. Silence means the table answers for the
    LADDER, exactly as it did before this feature existed.

    The SEED is a separate question and this is an aggregator route, so nothing
    is seeded — see `test_an_aggregator_route_never_seeds_warm_or_cold`. The
    direct route still seeds the table's default."""
    _stub_discovery(monkeypatch, [_row("anthropic/claude-opus-5")])
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    configure_mod.invalidate_model_info_cache()

    spec = configure_mod.build_model_spec("openrouter", "anthropic/claude-opus-5")

    assert spec.reasoning_efforts == ("low", "medium", "high", "xhigh", "max")
    assert spec.reasoning_effort is None


def test_an_aggregator_ladder_never_reaches_a_direct_providers_model(monkeypatch, tmp_path) -> None:
    """The single most important boundary in this feature.

    `_fill_from_row` is the SECOND-HAND catalogue path: it is consulted for a
    DIRECT provider's model under `prices.OPENROUTER_NAMESPACE`, which is how an
    `anthropic/*` id gets priced from OpenRouter's document. It deliberately
    refuses to take capabilities, and the ladder is a capability — OpenRouter's
    claim is first-hand only about ITS OWN route. If the ladder travelled that
    path, an aggregator's answer would decide what `claude-opus-5` accepts on
    Anthropic's own API. It must not, and `ModelInfo` carrying no ladder field
    at all is what makes that unrepresentable rather than merely unwritten.
    """
    from local_operator.model.registry import ModelInfo

    row = _row(
        "claude-opus-5",
        reasoning_efforts=("low", "high"),
        reasoning_default_effort="low",
        input_price=3.0,
        output_price=15.0,
    )
    info = ModelInfo(id="claude-opus-5", name="Claude Opus 5", description="")

    filled = configure_mod._fill_from_row(info, row)

    assert not hasattr(filled, "reasoning_efforts"), (
        "ModelInfo must carry no ladder field: the second-hand path is exactly "
        "where an aggregator's route-specific answer could leak onto a direct "
        "provider's model"
    )

    # The leak path in full: the AGGREGATOR leg reads OpenRouter's public
    # document for a direct-provider id, and must record nothing about efforts
    # while doing so. Only `_info_from_discovery` — the provider's own listing,
    # read under the provider's own id — may write that memo.
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    configure_mod.invalidate_model_info_cache()

    configure_mod._from_aggregator_catalogue("openrouter", "claude-opus-5", info)

    assert configure_mod._listing_effort("anthropic", "claude-opus-5") is None
    assert configure_mod._listing_effort("openrouter", "claude-opus-5") is None


def test_the_ladder_memo_is_cleared_with_the_metadata_memo(monkeypatch, tmp_path) -> None:
    """A listing read before the credential arrived states nothing, and leaving
    that cached would keep the band on the table's answer for a full TTL bucket
    after the user pasted the key that corrects it."""
    _stub_discovery(
        monkeypatch,
        [_row("vendor/m", reasoning_efforts=("low", "high"), reasoning_default_effort="high")],
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    configure_mod.invalidate_model_info_cache()

    configure_mod.build_model_spec("openrouter", "vendor/m")
    assert configure_mod._listing_effort("openrouter", "vendor/m") == ("low", "high")

    configure_mod.invalidate_model_info_cache()
    assert configure_mod._listing_effort("openrouter", "vendor/m") is None


def test_the_ladder_lookup_never_raises() -> None:
    """It sits on the session-start path and `build_model_spec` is reachable
    from a TUI repaint, so a failure here must cost a fallback, never a frame.
    A cold memo is the same answer as a silent listing: the table answers."""
    assert configure_mod._listing_effort("openrouter", "never/resolved") is None


def _openrouter_listing(model_id: str, reasoning: dict[str, object] | None):
    """A real `OpenRouterListModelsResponse` for the `model_info_client` branch.

    The literal wire shape, not a stand-in: this test exists because the two
    constructors parse DIFFERENT payloads (this one reads `list_models()`
    directly, the other reads a `DiscoveredModel` discovery already built), and
    a fake that flattened that difference away could not catch them disagreeing.
    """
    from local_operator.clients.openrouter import OpenRouterListModelsResponse

    entry: dict[str, object] = {
        "id": model_id,
        "name": model_id,
        "description": "",
        "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
        "context_length": 1_000_000,
    }
    if reasoning is not None:
        entry["reasoning"] = reasoning
    return OpenRouterListModelsResponse.model_validate({"data": [entry]})


class _StubListingClient:
    """Minimal `ModelListingClient`: the one method the branch calls."""

    def __init__(self, listing) -> None:
        self._listing = listing

    def list_models(self):
        return self._listing


def test_both_constructors_resolve_the_same_ladder_for_the_same_model(
    monkeypatch, tmp_path
) -> None:
    """The ladder must not be a function of WHICH constructor the caller used.

    `configure_model` has two branches. The default one resolves through
    `resolve_model_info` -> `_info_from_discovery`, which warms the effort memo.
    Passing a `model_info_client` takes the other: `get_model_info_from_openrouter`
    -> `_info_from_listing`, which parses its own payload. That branch warmed
    nothing, so `build_model_spec` found a cold memo and silently fell back to
    `model.effort`'s hand-transcribed table — the same model, the same provider,
    the same process, two different ladders.

    It is live rather than theoretical: `server/routes/speech.py` builds a
    client whenever an OpenRouter key is set, so on that path the reported bug
    (`google/gemini-3.8-flash` with no ladder at all) was simply not fixed, and
    `openai/gpt-5.4-pro` kept offering the `none`/`low` rungs the route 400s.

    Pinned on a model the TABLE also answers for, so a regression cannot hide
    behind both paths happening to return nothing.
    """
    model_id = "openai/gpt-5.4-pro"
    ladder = ("medium", "high", "xhigh")
    reasoning = {"supported_efforts": ["xhigh", "high", "medium"], "default_effort": "medium"}
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    # Branch 1: the default path, through discovery.
    _stub_discovery(monkeypatch, [_row(model_id, reasoning_efforts=ladder)])
    configure_mod.invalidate_model_info_cache()
    via_discovery = configure_mod.configure_model(hosting="openrouter", model_name=model_id)

    # Branch 2: the `model_info_client` path, which never reaches discovery.
    # Discovery is stubbed EMPTY so a cold memo cannot be warmed behind its
    # back — whatever this branch resolves, it resolved on its own.
    _stub_discovery(monkeypatch, [])
    configure_mod.invalidate_model_info_cache()
    via_client = configure_mod.configure_model(
        hosting="openrouter",
        model_name=model_id,
        model_info_client=_StubListingClient(_openrouter_listing(model_id, reasoning)),
    )

    assert via_client.spec.reasoning_efforts == ladder
    assert via_discovery.spec.reasoning_efforts == ladder
    assert via_client.spec.reasoning_efforts == via_discovery.spec.reasoning_efforts
    # The rungs the route rejects stay off BOTH ladders, which is the
    # user-visible half: the table would have offered them here.
    assert "none" not in via_client.spec.reasoning_efforts
    assert "low" not in via_client.spec.reasoning_efforts
    # And neither branch seeds a level from the listing's `default_effort`.
    assert via_client.spec.reasoning_effort is None
    assert via_discovery.spec.reasoning_effort is None


def test_the_listing_client_branch_defers_to_the_table_when_the_row_is_silent(
    monkeypatch, tmp_path
) -> None:
    """The other half of the same boundary: warming the memo must not turn a
    silent row into a denial. A listing entry with no `reasoning` block leaves
    the memo cold, so the table answers — exactly as it does everywhere else."""
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    _stub_discovery(monkeypatch, [])
    configure_mod.invalidate_model_info_cache()

    config = configure_mod.configure_model(
        hosting="openrouter",
        model_name="anthropic/claude-opus-5",
        model_info_client=_StubListingClient(_openrouter_listing("anthropic/claude-opus-5", None)),
    )

    assert config.spec.reasoning_efforts == ("low", "medium", "high", "xhigh", "max")
    # The table answers the LADDER here. It does not seed, because this is an
    # aggregator route: the `high` ≡ omission equivalence is documented for
    # Anthropic's own API, and OpenRouter interposes its own reasoning gate
    # ahead of it (measured: `claude-opus-4.6` 0 reasoning tokens omitted vs 70
    # seeded). The direct route still seeds — see
    # `test_a_direct_route_still_seeds_the_tables_default`.
    assert config.spec.reasoning_effort is None


def test_an_aggregator_route_never_seeds_warm_or_cold(monkeypatch, tmp_path) -> None:
    """The seeding rule, pinned directly rather than as a side assertion.

    Nothing seeds on an aggregator route — not the listing's `default_effort`
    and not the table's either. The table's `high` rests on Anthropic
    documenting it as equivalent to omission, and that is a fact about
    ANTHROPIC'S OWN API; an aggregator interposes its own reasoning gate ahead
    of it. Measured through OpenRouter, n=10 per arm: `claude-opus-4.6` returned
    0 reasoning tokens with the key omitted and 70 with `high` sent, so there
    the seed does not restate a default, it switches reasoning on.

    It also closes the warm/cold split. An earlier revision stopped seeding only
    on the listing branch while the table branch kept its default, so the boot
    state of every model both sources answer for depended on whether an HTTP
    call landed: band `auto` and no key when the listing was reached, band
    `high` and `reasoning_effort: 'high'` when it was not. Because neither arm
    seeds now, both agree by construction.
    """
    model_id = "anthropic/claude-opus-4.6"
    listing_ladder = ("low", "medium", "high", "max")
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    # The table DOES answer for this id, so a regression cannot hide behind
    # both sources happening to be silent.
    assert effort.default_effort(model_id) == "high"

    # WARM: the listing states a ladder, and it wins the LADDER.
    _stub_discovery(
        monkeypatch,
        [
            _row(
                model_id,
                reasoning_efforts=listing_ladder,
                reasoning_default_effort="high",
            )
        ],
    )
    configure_mod.invalidate_model_info_cache()
    warm = configure_mod.build_model_spec("openrouter", model_id)

    # COLD: no listing row at all, so the table supplies the ladder.
    _stub_discovery(monkeypatch, [])
    configure_mod.invalidate_model_info_cache()
    cold = configure_mod.build_model_spec("openrouter", model_id)

    assert warm.reasoning_efforts == listing_ladder  # the listing wins the ladder
    assert cold.reasoning_efforts == effort.supported_efforts(model_id)
    # Neither arm seeds, so neither sends a key.
    assert warm.reasoning_effort is None
    assert cold.reasoning_effort is None
    assert warm.reasoning_default_effort is None
    assert cold.reasoning_default_effort is None
    # The property that failed: same model, same route, same answer either way.
    assert warm.reasoning_effort == cold.reasoning_effort


def test_a_direct_route_still_seeds_the_tables_default(monkeypatch, tmp_path) -> None:
    """The non-regression the aggregator rule must not cost: on the route whose
    documentation establishes the `high` ≡ omission equivalence, the table still
    seeds exactly as it always has. Measured on that route, omitting and sending
    `high` are the same request, so the seed only stops the band understating
    what is already in force."""
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    _stub_discovery(monkeypatch, [])
    configure_mod.invalidate_model_info_cache()

    spec = configure_mod.build_model_spec("anthropic", "claude-opus-5")

    assert spec.reasoning_efforts == ("low", "medium", "high", "xhigh", "max")
    assert spec.reasoning_effort == "high"
    assert spec.reasoning_default_effort == "high"


def test_the_dotted_repair_gains_a_ladder_without_gaining_a_wire_key(monkeypatch, tmp_path) -> None:
    """F7's claim, pinned. Widening the table's separator to `[.-]` gives dotted
    Anthropic ids the ladder they should always have had. On the aggregator
    route that must buy a LADDER and nothing else — no `reasoning_effort` on the
    wire — because those are exactly the rows measured to switch reasoning on
    when seeded (`claude-sonnet-4.6`: 0 tokens omitted, 164 seeded)."""
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    _stub_discovery(monkeypatch, [])
    configure_mod.invalidate_model_info_cache()

    for model_id in ("anthropic/claude-opus-4.6", "anthropic/claude-sonnet-4.6"):
        spec = configure_mod.build_model_spec("openrouter", model_id)
        # The ladder the repair restored: the reported bug is fixed —
        # `shift+tab` and `/effort` have rungs to offer.
        assert spec.reasoning_efforts
        assert spec.reasoning is True
        # And no key reaches the wire.
        assert spec.reasoning_effort is None


def test_a_listing_default_never_seeds_even_when_the_table_is_silent(monkeypatch, tmp_path) -> None:
    """The other half: a LISTING-derived default is never a seed, on any path.

    Measured, not cautious. Sending OpenRouter's stated `default_effort` is not
    the same request as omitting the key — `z-ai/glm-5.3` seeded at `max`
    returned 3.06x the reasoning tokens of omission (n=12 per arm), and
    `mistralai/mistral-small-2603` went from a median 0 reasoning tokens to
    167.5. So the listing default describes neither what omission does nor a
    level in force, and painting it on the band would assert a depth that is not.
    """
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    _stub_discovery(
        monkeypatch,
        [
            _row(
                "z-ai/glm-5.3",
                reasoning_efforts=("low", "high", "max"),
                reasoning_default_effort="max",
            )
        ],
    )
    configure_mod.invalidate_model_info_cache()

    spec = configure_mod.build_model_spec("openrouter", "z-ai/glm-5.3")

    # The table has no arm for this id, so nothing legitimises a seed.
    assert effort.default_effort("z-ai/glm-5.3") is None
    assert spec.reasoning_efforts == ("low", "high", "max")
    assert spec.reasoning_effort is None
    assert spec.reasoning_default_effort is None


def test_an_off_ladder_table_default_seeds_nothing_rather_than_clamping(
    monkeypatch, tmp_path
) -> None:
    """The membership guard on the direct route, for the day a listing NARROWS.

    Aggregator routes no longer seed at all, so this is reachable only where a
    direct provider's listing supplies a ladder narrower than the table's
    default. Seeding the default anyway would put a level on the wire the route
    rejects, and clamping to a neighbour would be this code inventing a depth no
    source stated. `None` — band `auto`, key omitted — is the honest answer, and
    the user can still pick a rung.
    """
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    assert effort.default_effort("claude-opus-5") == "high"

    # A listing that offers everything EXCEPT the table's default.
    _stub_discovery(
        monkeypatch,
        [
            _row(
                "claude-opus-5",
                reasoning_efforts=("low", "medium"),
                reasoning_default_effort="medium",
            )
        ],
    )
    configure_mod.invalidate_model_info_cache()
    configure_mod._store_listing_effort("anthropic", "claude-opus-5", ("low", "medium"))

    spec = configure_mod.build_model_spec("anthropic", "claude-opus-5")

    assert spec.reasoning_efforts == ("low", "medium")
    assert spec.reasoning_effort is None
    assert spec.reasoning_default_effort is None


def test_the_ladder_memo_enforces_its_bound_within_one_bucket(monkeypatch, tmp_path) -> None:
    """Bucket eviction alone does not bound the memo, and the comment used to
    claim it did. Many models resolved inside ONE TTL bucket evict nothing,
    because every key held is current — so a long-lived process (the server, a
    scheduler worker) grew the dict without limit. Entries are tiny, so this is
    a leak ceiling rather than a cache-efficiency knob."""
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    configure_mod.invalidate_model_info_cache()

    for index in range(300):
        configure_mod._store_listing_effort("openrouter", f"vendor/m{index}", ("low", "high"))

    assert len(configure_mod._effort_memo) <= configure_mod._EFFORT_MEMO_MAX
    # The most recent writes survive; the oldest are what gets dropped.
    assert configure_mod._listing_effort("openrouter", "vendor/m299") == ("low", "high")
    assert configure_mod._listing_effort("openrouter", "vendor/m0") is None
