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
from unittest import mock

import pytest

from local_operator.model import catalogue
from local_operator.model import configure as configure_mod
from local_operator.model.catalogue import cached_listing


def _payload(model_id: str = "vendor/model", window: int = 1_000_000) -> dict:
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
    path = tmp_path / "openrouter.models.json"
    path.write_text(content, encoding="utf-8")
    assert cached_listing("openrouter", lambda: _payload(), cache_dir=tmp_path) == _payload()


def test_the_cache_write_is_atomic(tmp_path) -> None:
    """Two sessions can start at once, so a reader must never observe a
    partial document. Assert the temp file is not left behind and the final
    document parses."""
    cached_listing("openrouter", lambda: _payload(), cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.tmp")), "temp file leaked"
    raw = json.loads((tmp_path / "openrouter.models.json").read_text(encoding="utf-8"))
    assert raw["payload"] == _payload()
    assert time.time() - raw["fetched_at"] < 60


def test_an_unwritable_cache_dir_still_returns_the_payload(tmp_path, monkeypatch) -> None:
    """Losing the cache is a performance problem; failing the call is not."""

    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(catalogue.Path, "mkdir", refuse)
    assert cached_listing("openrouter", lambda: _payload(), cache_dir=tmp_path) == _payload()


# -- integration with configure_model ---------------------------------------


def test_configure_model_takes_the_window_from_the_catalogue(monkeypatch, tmp_path) -> None:
    """The defect this module exists for: an aggregator's real window reaching
    the ModelSpec instead of the 128k placeholder."""
    from local_operator.clients.openrouter import OpenRouterListModelsResponse
    from local_operator.model import configure as configure_mod

    class FakeClient:
        def list_models(self):
            return OpenRouterListModelsResponse.model_validate(
                _payload("vendor/big", window=1_000_000)
            )

    monkeypatch.setattr(
        configure_mod,
        "_catalogue_source",
        lambda _p: (FakeClient(), OpenRouterListModelsResponse),
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    config = configure_mod.configure_model(hosting="openrouter", model_name="vendor/big")
    assert config.spec.context_window == 1_000_000
    assert config.info.input_price == pytest.approx(0.1)
    assert config.info.output_price == pytest.approx(0.2)


@pytest.mark.parametrize(
    "break_it",
    ["client-raises", "no-such-model", "payload-drift"],
)
def test_configure_model_never_fails_on_a_bad_catalogue(
    monkeypatch, tmp_path, break_it: str
) -> None:
    """Every catalogue failure mode degrades to the static fallback. A session
    MUST start with no network, an unknown model id, or a drifted schema."""
    from local_operator.clients.openrouter import OpenRouterListModelsResponse
    from local_operator.model import configure as configure_mod
    from local_operator.model.configure import UNKNOWN_CONTEXT_WINDOW

    class FakeClient:
        def list_models(self):
            if break_it == "client-raises":
                raise RuntimeError("no network")
            if break_it == "payload-drift":
                return OpenRouterListModelsResponse.model_validate({"data": []})
            return OpenRouterListModelsResponse.model_validate(_payload("other/model"))

    monkeypatch.setattr(
        configure_mod,
        "_catalogue_source",
        lambda _p: (FakeClient(), OpenRouterListModelsResponse),
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    config = configure_mod.configure_model(hosting="openrouter", model_name="vendor/asked-for")
    assert config.spec.context_window == UNKNOWN_CONTEXT_WINDOW


def test_a_provider_with_a_real_static_entry_never_consults_the_catalogue(monkeypatch) -> None:
    """The catalogue is for placeholder entries only. Anthropic's window is in
    the registry, so touching the network for it would add latency to every
    session start for nothing."""
    from local_operator.model import configure as configure_mod

    def explode(_provider):
        raise AssertionError("the catalogue must not be consulted for anthropic")

    monkeypatch.setattr(configure_mod, "_catalogue_source", explode)
    config = configure_mod.configure_model(
        hosting="anthropic", model_name="claude-sonnet-4-20250514"
    )
    assert config.spec.context_window == 200_000


@pytest.mark.parametrize("provider", ["openrouter", "radient"])
def test_a_keyless_install_does_not_raise_building_a_client(monkeypatch, provider: str) -> None:
    """`OpenRouterClient` raises RuntimeError on an empty key IN ITS
    CONSTRUCTOR, and `RadientClient` requires a base_url. The first version of
    the catalogue got both wrong and turned a metadata optimisation into "the
    CLI will not start without a key" — caught by the existing default-names
    tests, which is exactly the failure this pins.
    """
    from local_operator.model import configure as configure_mod

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RADIENT_API_KEY", raising=False)

    # Must answer, not raise: None (no client) or a usable (client, model) pair.
    source = configure_mod._catalogue_source(provider)
    assert source is None or len(source) == 2


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
    path = tmp_path / "openrouter.models.json"
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
    path = tmp_path / "openrouter.models.json"
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

    path = tmp_path / "openrouter.models.json"
    (tmp_path / f"{path.name}.{os.getpid()}.tmp").write_text("{partial", encoding="utf-8")
    (tmp_path / f"{path.name}.tmp").write_text("{partial", encoding="utf-8")
    catalogue._write_cache(path, _payload(window=42))
    assert (
        json.loads(path.read_text(encoding="utf-8"))["payload"]["data"][0]["context_length"] == 42
    )


def test_a_failed_write_does_not_strand_a_temp_file(tmp_path, monkeypatch) -> None:
    """Otherwise a persistently failing start accumulates one temp file per run."""
    path = tmp_path / "openrouter.models.json"
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
    from local_operator.model.configure import _resolve_model_info_cached, resolve_model_info

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


def test_the_key_is_read_from_the_apps_own_credential_store(tmp_path, monkeypatch) -> None:
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

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    CredentialManager(tmp_path).set_credential("OPENROUTER_API_KEY", "sk-or-store-value")

    assert configure_mod._catalogue_api_key("openrouter") == "sk-or-store-value"
    assert configure_mod._catalogue_source("openrouter") is not None


def test_an_env_var_takes_precedence_over_the_stored_credential(tmp_path, monkeypatch) -> None:
    """An explicit env var is the operator overriding config for one run."""
    from local_operator.credentials import CredentialManager
    from local_operator.model import configure as configure_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    CredentialManager(tmp_path).set_credential("OPENROUTER_API_KEY", "sk-or-store-value")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-value")

    assert configure_mod._catalogue_api_key("openrouter") == "sk-or-env-value"


def test_no_key_anywhere_still_builds_a_client(tmp_path, monkeypatch) -> None:
    """The listing endpoints are PUBLIC catalogue data — `GET /api/v1/models`
    answers 200 with no Authorization header at all. The clients still refuse an
    empty key, so a placeholder token is what lets a keyless or OAuth-only
    install learn its real context window instead of silently keeping 128k.
    """
    from local_operator.model import configure as configure_mod

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    assert configure_mod._catalogue_api_key("openrouter") == ""
    assert configure_mod._catalogue_source("openrouter") is not None


# -- schema drift that validates but cannot be mapped (S2) -------------------


def _drifted_payload() -> dict:
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
def test_a_payload_that_validates_but_cannot_map_does_not_raise(
    tmp_path, monkeypatch, bad_window, raised_by_python
) -> None:
    """Validation is NOT a guarantee that the mapping will succeed. The listing
    schemas set ``extra="allow"``, so a wrong-shaped extra validates cleanly and
    then raises inside the conversions, failing session start outright.

    All four shapes are parametrized because they raise three DIFFERENT
    exceptions and each was a separate hole: catching ValueError alone let the
    dict through, and adding TypeError still left ``Infinity`` — which
    ``json.loads`` accepts as a bare literal and ``int()`` then rejects with
    OverflowError, a subclass of neither.
    """
    from local_operator.clients.openrouter import OpenRouterListModelsResponse
    from local_operator.model.registry import ModelInfo

    payload = _drifted_payload()
    payload["data"][0]["context_length"] = bad_window
    # The premise of the finding: pydantic accepts it.
    assert OpenRouterListModelsResponse.model_validate(payload) is not None

    class FakeClient:
        def list_models(self):
            return OpenRouterListModelsResponse.model_validate(payload)

    monkeypatch.setattr(
        configure_mod,
        "_catalogue_source",
        lambda _p: (FakeClient(), OpenRouterListModelsResponse),
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)

    fallback = ModelInfo(id="v/m", name="v/m", description="")
    assert configure_mod._info_from_catalogue("openrouter", "v/m", fallback) is fallback
    # And the whole call still returns a usable configuration.
    config = configure_mod.configure_model(hosting="openrouter", model_name="v/m")
    assert config.spec.context_window == configure_mod.UNKNOWN_CONTEXT_WINDOW


def test_one_unmappable_entry_keeps_the_rest_of_the_catalogue(tmp_path, monkeypatch) -> None:
    """A bad entry is not a bad document.

    An earlier revision purged the whole cache file here, reasoning that a
    payload which cannot be mapped should not be served for the TTL. But the
    conversions only ever run for the ONE model whose id matched, out of a listing
    that carries several hundred: purging discarded metadata that was correct for
    all the others, and since the upstream document is unchanged the refetch
    re-poisons the cache immediately. The lasting effect was an extra HTTP call
    per start plus a refetch for every other model.
    """
    from local_operator.clients.openrouter import OpenRouterListModelsResponse
    from local_operator.model.registry import ModelInfo

    bad = _drifted_payload()["data"][0]
    good = _payload(model_id="good/model", window=999_000)["data"][0]
    fetches: list[int] = []

    class FakeClient:
        def list_models(self):
            fetches.append(1)
            return OpenRouterListModelsResponse.model_validate({"data": [good, bad]})

    monkeypatch.setattr(
        configure_mod,
        "_catalogue_source",
        lambda _p: (FakeClient(), OpenRouterListModelsResponse),
    )
    monkeypatch.setattr(catalogue, "default_cache_dir", lambda: tmp_path)
    template = ModelInfo(id="v", name="v", description="")

    # The bad entry falls back without taking the document with it.
    assert configure_mod._info_from_catalogue("openrouter", bad["id"], template) is template
    assert (tmp_path / "openrouter.models.json").exists(), "one bad entry purged the whole cache"

    # So the good entry still resolves from cache — no second fetch.
    resolved = configure_mod._info_from_catalogue("openrouter", "good/model", template)
    assert resolved.context_window == 999_000
    assert len(fetches) == 1, "the good entry had to be refetched"


def test_invalidate_is_safe_when_there_is_no_cache(tmp_path) -> None:
    catalogue.invalidate("openrouter", cache_dir=tmp_path)  # must not raise


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
            catalogue._write_cache(tmp_path / "openrouter.models.json", _payload(window=window))

    def reader() -> None:
        for _ in range(40):
            payload, _age = catalogue._read_cache(tmp_path / "openrouter.models.json")
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
