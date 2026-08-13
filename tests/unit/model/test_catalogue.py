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

from local_operator.model import catalogue
from local_operator.model import configure as configure_mod
from local_operator.model.catalogue import cached_listing


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

    model_id = "grok-4.6"
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
    # so the operator sees ``grok-4.6`` rather than the placeholder word.
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
