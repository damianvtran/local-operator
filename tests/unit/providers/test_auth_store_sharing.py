"""The process-level ``auth_store`` accessor: sharing, closing, threading.

WHAT THIS PINS. ``AuthStore`` used to be constructed once per read site, which
on a real boot meant seven SQLite connections to one ``auth.db`` (measured:
``classification/vendors.py`` x3, ``model/configure.py`` x3,
``session_factory.py`` x1). The sites whose use is a bounded read through the
cascade now share one process-level store instead.

These tests assert COUNTS and IDENTITY, never a duration: how many connections
were opened, whether two acquisitions are the same object, and whether a site
still closes what it owns. A timing assertion here would flake on a loaded host
and would not distinguish "one connection reused" from "seven connections that
happened to be fast".

The closing contract is the sharp edge of this change and is pinned from both
sides: a shared store must survive a stray close (the accessor reopens it), and
the callers that deliberately own a short-lived connection — the ``closing(
AuthStore())`` in ``tunnels/`` — must still close theirs.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from local_operator.providers import auth_store as auth_store_module
from local_operator.providers.auth_store import (
    AuthStore,
    close_shared_auth_stores,
    shared_auth_store,
)

#: Provider env vars the cascade would otherwise resolve from the developer's
#: shell. The suite must not read a real key while asserting on the store.
_LEAKY_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_OAUTH_TOKEN",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "KIMI_API_KEY",
    "XAI_API_KEY",
    "RADIENT_API_KEY",
    "TYPESAFE_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture()
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A hermetic credential root, plus an empty shared map around the test.

    ``LOCAL_OPERATOR_CONFIG_DIR`` is what makes ``config_dir()`` — and so the
    bare ``AuthStore()`` sites — resolve INSIDE the test rather than at the
    operator's live ``auth.db``. The operator's store holds real credentials and
    must never be touched by a test run.
    """
    config_root = tmp_path / "cfg"
    config_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_root))
    for var in _LEAKY_ENV:
        monkeypatch.delenv(var, raising=False)
    close_shared_auth_stores()
    yield config_root
    # The teardown is the production one, so the fixture also pins that
    # ``close_shared_auth_stores`` releases everything it handed out.
    close_shared_auth_stores()


@pytest.fixture()
def auth_connects(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every ``sqlite3.connect`` to an ``auth.db``, in order.

    The seam is ``sqlite3.connect`` itself rather than a hook inside
    ``AuthStore``: counting at the accessor would assume the thing under test,
    while counting at the driver records the resource the finding is about — a
    file descriptor and a connection — however it was reached.
    """
    seen: list[str] = []
    real_connect = sqlite3.connect

    def counting_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        target = str(args[0]) if args else ""
        if target.endswith("auth.db"):
            seen.append(target)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting_connect)
    return seen


def test_n_read_sites_open_one_connection(root: Path, auth_connects: list[str]) -> None:
    """THE regression: the cascade's read sites share ONE connection.

    The sites are called through their real entry points — the classification
    leg, the model-listing token probe, and both DeepSeek search probes — so the
    assertion is about what those call sites do, not about a helper in
    isolation. Before sharing this was six connections; the assertion is a
    single connect for all of them.
    """
    from local_operator.classification.vendors import auth_store_api_key
    from local_operator.model.configure import _oauth_listing_token
    from local_operator.web_search.providers import (
        _deepseek_login_present,
        _resolve_deepseek_key,
    )

    async def exercise() -> None:
        # Three classification legs (the measured x3 per boot), each of which
        # used to build and close its own store.
        for provider in ("typesafe", "openrouter", "radient"):
            await auth_store_api_key(root, provider)
        # Two model-listing probes.
        _oauth_listing_token("deepseek")
        _oauth_listing_token("openai")
        # Two search-key checks (the per-check constructions in the hot path).
        _deepseek_login_present()
        await _resolve_deepseek_key(root)

    asyncio.run(exercise())

    assert len(auth_connects) == 1, f"expected one shared connection, opened {auth_connects}"
    # And the one connection is the store's own, not a throwaway: acquiring
    # again hands back the same live object.
    assert shared_auth_store().list_credentials() == []
    assert len(auth_connects) == 1


def test_repeated_acquisition_is_the_same_object(root: Path, auth_connects: list[str]) -> None:
    """Identity, not just count: reuse means one live store, not N cheap ones."""
    first = shared_auth_store()
    for _ in range(5):
        assert shared_auth_store() is first
    assert len(auth_connects) == 1


def test_a_different_credential_root_gets_its_own_store(
    root: Path, tmp_path: Path, auth_connects: list[str]
) -> None:
    """The key is the RESOLVED root, which is what keeps an isolated run isolated.

    Two roots must never share a store even when both are reached from one
    process: ``shared_auth_store`` keys on the resolved ``auth.db`` path AND the
    canonical config root the env tier reads, so an isolated run cannot be
    served the default root's answers.
    """
    other = tmp_path / "other-root"
    other.mkdir()
    first = shared_auth_store()
    second = shared_auth_store(other / "auth.db", config_dir=other)
    assert second is not first
    assert second.db_path == other / "auth.db"
    assert len(auth_connects) == 2


def test_none_and_the_explicit_default_root_share_one_store(
    root: Path, auth_connects: list[str]
) -> None:
    """``config_dir=None`` IS ``config_dir()``, so it must not cost a connection.

    ``secrets_dir`` resolves ``base`` as ``(base if base is not None else
    config_dir())``, so the two spellings denote one directory. Keying them apart
    would open a second connection to the same file for no answer-level reason —
    which is exactly the duplication being removed.
    """
    from_root = shared_auth_store(root / "auth.db", config_dir=root)
    from_none = shared_auth_store()
    assert from_none is from_root
    assert len(auth_connects) == 1


def test_config_overrides_are_part_of_the_key(root: Path, auth_connects: list[str]) -> None:
    """A caller with override tiers must not be handed a store without them.

    ``config_overrides`` is tier 2 of the cascade, so a store built without them
    answers differently. Sharing one would change an answer, which this change
    may not do.
    """
    plain = shared_auth_store()
    overridden = shared_auth_store(config_overrides={"openai": "sk-override"})
    assert overridden is not plain
    assert overridden.override_keys("openai") == ["sk-override"]
    assert plain.override_keys("openai") == []
    assert len(auth_connects) == 2


def test_using_the_shared_store_after_a_close_is_safe(root: Path, auth_connects: list[str]) -> None:
    """A stray close must not poison the process — the chosen half of the contract.

    Nothing that receives a shared store may close it, but a future call site
    that does (or an early teardown) must not turn every later read into
    ``ProgrammingError: Cannot operate on a closed database``. The accessor
    detects the closed store and reopens it, so the cost of a stray close is one
    extra connection rather than a dead process-wide store.
    """
    first = shared_auth_store()
    first.close()
    assert first.closed

    # The raw failure mode this protects against, stated as the test's own
    # premise: a closed store still raises.
    with pytest.raises(sqlite3.ProgrammingError):
        first.list_credentials()

    reopened = shared_auth_store()
    assert reopened is not first
    assert not reopened.closed
    # Usable, and answering from the same database.
    assert reopened.list_credentials() == []
    reopened.upsert_credential("openai", {"key": "k", "type": "api_key", "source": "login"})
    assert [r.data["key"] for r in reopened.list_credentials("openai")] == ["k"]
    assert len(auth_connects) == 2


def test_close_records_the_flag_and_a_double_close_is_harmless(root: Path) -> None:
    """``closed`` is set by ``close`` and stays set; closing twice does not raise."""
    store = AuthStore(root / "auth.db", config_dir=root)
    assert not store.closed
    store.close()
    assert store.closed
    store.close()  # sqlite's own close is a no-op the second time
    assert store.closed


def test_close_shared_auth_stores_releases_what_it_handed_out(root: Path) -> None:
    """The teardown closes the shared store and empties the map.

    A closed handle is what the next acquisition must not be given, so this is
    also what makes the reopen path reachable after a teardown — the atexit
    handler and this call are the same function.
    """
    store = shared_auth_store()
    close_shared_auth_stores()
    assert store.closed
    assert auth_store_module._SHARED_STORES == {}
    fresh = shared_auth_store()
    assert not fresh.closed
    assert fresh is not store


def test_acquisition_from_many_threads_opens_one_connection(
    root: Path, auth_connects: list[str]
) -> None:
    """Sharing is thread-safe, and the first-connect race yields one connection.

    The store's connection is opened ``check_same_thread=False`` with every
    statement serialized by a reentrant mutex (see ``_SerializedConnection``),
    so a worker thread gets a correct answer rather than a ``ProgrammingError``.
    The accessor adds the one thing that needs its own lock: the map is guarded
    ACROSS construction, so N threads racing for the first store do not each open
    one.
    """
    barrier = threading.Barrier(8)

    def acquire() -> AuthStore:
        barrier.wait()
        return shared_auth_store()

    with ThreadPoolExecutor(max_workers=8) as pool:
        stores = [f.result() for f in [pool.submit(acquire) for _ in range(8)]]

    assert len({id(s) for s in stores}) == 1
    assert len(auth_connects) == 1

    # ...and the shared store answers from a worker thread, which is the D18
    # failure this flag and mutex exist to prevent.
    def read_off_thread() -> list[Any]:
        return shared_auth_store().list_credentials()

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(read_off_thread).result() == []


def test_the_map_is_bounded_and_never_closes_a_held_store(
    root: Path, tmp_path: Path, auth_connects: list[str]
) -> None:
    """More roots than the bound must not grow descriptors without limit.

    Eviction DROPS the map's reference rather than closing, so a store a caller
    still holds stays usable — the opposite choice would hand that caller a
    ``ProgrammingError`` to save a descriptor.
    """
    held = shared_auth_store()
    for index in range(auth_store_module._SHARED_STORES_MAX + 4):
        extra = tmp_path / f"root-{index}"
        extra.mkdir()
        shared_auth_store(extra / "auth.db", config_dir=extra)

    assert len(auth_store_module._SHARED_STORES) <= auth_store_module._SHARED_STORES_MAX
    # The held store survived eviction and is still live.
    assert not held.closed
    assert held.list_credentials() == []


def test_a_caller_that_owns_its_connection_still_closes_it(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A site deliberately left on ``closing(AuthStore())`` still closes.

    ``tunnels/api.py`` is NOT consolidated, and the reason is the contract at
    ``_release_detached_refresh``: its bounded caller hands a refresh exchange to
    a supervisor that outlives the call, so the connection must die with the call
    rather than being shared. This pins that those sites keep closing — sharing
    them would silently delete the guarantee they were written to provide.
    """
    from local_operator.tunnels import api as tunnels_api

    built: list[AuthStore] = []

    class Recording(AuthStore):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(tunnels_api, "AuthStore", Recording)

    # No radient login row in this root, so the call raises after reading --
    # and ``closing`` must still close the connection on that path.
    with pytest.raises(ValueError):
        tunnels_api.credential_id()

    assert len(built) == 1
    assert built[0].closed
    assert built[0] is not shared_auth_store()


def test_closing_a_private_store_leaves_the_shared_one_usable(
    root: Path, auth_connects: list[str]
) -> None:
    """The session-owned store and the shared store are independent handles.

    ``session_factory`` builds its own store and
    ``attach_auth_dispose`` closes it on ``session.dispose()``. That store keys
    identically to the shared one (same file, same root), so if it were the
    shared one, disposing a single session would close the process's store under
    every other reader — the concrete form of the hazard
    ``_release_detached_refresh`` records. This pins the independence the split
    relies on.
    """
    shared = shared_auth_store()
    private = AuthStore(root / "auth.db", config_dir=root)
    private.close()
    assert private.closed
    assert not shared.closed
    assert shared.list_credentials() == []
