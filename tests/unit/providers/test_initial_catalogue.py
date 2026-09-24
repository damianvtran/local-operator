"""The FIRST frame: every provider's rows, from the cache, with no network.

WHY THIS FILE EXISTS. ``ProviderController.initial_catalogue`` is the frame a
picker paints on the keystroke that opens it -- and for the desktop composer's
inline ``/model `` argument list it is the ONLY frame, because
``/v1/desktop/sessions/{id}/command-entities?command=model`` never goes live. It
used to read the shipped static registry for a direct provider, so a model that
exists only in the provider's own listing could not be offered there even after
the dialog had refetched it.

Two properties keep that change honest, and they are pinned here in the order
they matter:

1. THE COLD-CACHE INVARIANT (the hard one). With nothing usable on disk the
   first frame must be exactly the rows this method always painted -- same
   order, same count, element-wise identical fields. ``static_catalogue()`` is
   the reference because it is the shipped-registry builder that did not move.
2. THE GAP. With a cached listing carrying an id the registry does not have, the
   first frame must offer that id -- with zero fetches, because the whole point
   of a first frame is that it paints before the network answers.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import pytest

from local_operator.model import discovery
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.controller import ProviderController


@pytest.fixture
def controller(tmp_path: Path):
    """A real controller over a real (empty) credential store in ``tmp_path``."""
    store = AuthStore(tmp_path / "auth.db")
    instance = ProviderController(store)
    try:
        yield instance
    finally:
        instance.close()
        store.close()


def _plant(
    cache_dir: Path,
    provider: str,
    *,
    ids: list[str],
    routed: bool = False,
) -> None:
    """A cached listing document for ``provider``, carrying exactly ``ids``.

    Written through the module's own capture-version helper rather than a
    literal, so a capture bump makes this fixture read as unusable instead of
    silently passing on a shape the reader no longer accepts.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{provider}.listing.json").write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {
                    "capture": discovery.listing_capture_version(provider),
                    "models": [
                        {"id": model_id, "context_window": 1_000_000, "routed": routed}
                        for model_id in ids
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def _plant_prices(
    cache_dir: Path, provider: str, model_id: str, *, input_price: float, output_price: float
) -> None:
    """A models.dev projection carrying one priced id -- the price chain's source.

    Same capture constant the reader checks, so a bump here reads as an unusable
    document rather than as a silently unpriced one.
    """
    from local_operator.model.prices import PRICE_CATALOGUE_CAPTURE

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "models-dev.listing.json").write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {
                    "capture": PRICE_CATALOGUE_CAPTURE,
                    "providers": {
                        provider: {
                            model_id: {
                                "name": model_id,
                                "cost": {"input": input_price, "output": output_price},
                                "limit": {"context": 1_000_000, "output": 32_000},
                                "modalities": {"output": ["text"]},
                            }
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _no_network(*_args: object, **_kwargs: object) -> list[discovery.DiscoveredModel]:
    raise AssertionError("the first frame must only peek at the cache")


def test_the_first_frame_with_nothing_cached_is_the_shipped_rows(
    controller: ProviderController, tmp_path: Path
) -> None:
    """Cold cache: byte-identical to the builder that never moved, in order.

    ORDER IS PART OF THE INVARIANT, and it is named rather than assumed: within
    a provider the rows come in the registry dict's own order, and the providers
    themselves come in ``_chat_providers()`` registry order. The comparison is
    made on ``dataclasses.asdict`` and on the selector list, so a refactor that
    regrouped rows (say, every live row ahead of every static one across
    providers) fails here instead of moving a model out from under the user's
    cursor between the first frame and the live one.

    The two fields a naive port drops silently are asserted by VALUE, not by
    presence: ``time_of_use`` (read from the registry row) and ``routed``.
    """
    rows = controller.initial_catalogue(cache_dir=tmp_path / "empty-cache")
    shipped = controller.static_catalogue()

    assert rows, "the shipped registry must contribute rows on a cold cache"
    assert [row.provider for row in rows] == [row.provider for row in shipped]
    assert [row.selector for row in rows] == [row.selector for row in shipped]
    assert [dataclasses.asdict(row) for row in rows] == [dataclasses.asdict(row) for row in shipped]

    deepseek = next(
        row for row in rows if row.provider == "deepseek" and row.model_id == "deepseek-flash"
    )
    assert deepseek.time_of_use == "deepseek-tou", (
        "the schedule is read from the registry row; a port that reads only the "
        "merged DiscoveredModel must still carry it through unchanged"
    )
    # `routed` is carried from the merged row, and the load-bearing half of that
    # lives in `test_a_cached_listings_routed_row_carries_its_flag_to_the_frame`,
    # because no shipped id is a meta route and a cold frame therefore has no
    # `True` to offer. What this asserts is the other direction: the mapping must
    # not invent a `True` out of an id it has no listing for (review round 2,
    # R2-5).
    assert not any(row.routed for row in rows)


@pytest.mark.parametrize("provider", ["lmstudio", "openai-compatible"])
def test_a_local_provider_contributes_no_rows_because_it_ships_none(
    controller: ProviderController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """`static_models` is empty for all five local providers, so frame one gives none.

    Pinned as a test rather than a comment because it is the fact that decides
    what "degrades to the shipped rows" MEANS for these providers: nothing at all
    -- which is also what they contributed before every provider was read
    through the cache reader (agent review round 2, R2-1).
    """
    monkeypatch.setattr(discovery, "fetch_models", _no_network)

    frame = [
        row for row in controller.initial_catalogue(cache_dir=tmp_path) if row.provider == provider
    ]

    assert frame == []


def test_a_cached_listing_reaches_the_first_frame_for_a_direct_provider(
    controller: ProviderController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gap: a provider-listed id the registry does not carry, on frame one.

    Anthropic is the observed case (a model present in ``/v1/models`` and absent
    from the shipped registry), and it is a DIRECT provider -- the class the old
    first frame read from ``static_models`` alone. The listing names ONLY the id
    the registry lacks, which is what makes the union assertion below
    load-bearing: ``claude-opus-5`` can only be there because the registry's own
    rows rode along (review round 2, R2-5).
    """
    monkeypatch.setattr(discovery, "fetch_models", _no_network)
    _plant(tmp_path, "anthropic", ids=["claude-fable-6"])

    rows = [
        row
        for row in controller.initial_catalogue(cache_dir=tmp_path)
        if row.provider == "anthropic"
    ]
    ids = {row.model_id for row in rows}

    assert "claude-fable-6" in ids, "a cached id the registry lacks must paint on frame one"
    assert "claude-opus-5" in ids, (
        "a shipped id the listing never mentioned is gone: a cached listing is a "
        "UNION for a direct provider, not an authoritative set"
    )
    assert any(not row.connected for row in rows), "no credential here, so nothing is connected"


def test_a_cached_listings_routed_row_carries_its_flag_to_the_frame(
    controller: ProviderController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``routed`` survives the cache round trip and the row mapping.

    The flag is computed by the parser that read the wire and stored in the
    document, so a cached row is the only place a first frame can get a ``True``
    from -- the shipped registry has no meta route at all (asserted in the
    cold-frame test). The id is arbitrary; the CARRIAGE is what is under test.
    """
    monkeypatch.setattr(discovery, "fetch_models", _no_network)
    _plant(tmp_path, "anthropic", ids=["claude-router"], routed=True)

    row = next(
        row
        for row in controller.initial_catalogue(cache_dir=tmp_path)
        if row.model_id == "claude-router"
    )

    assert row.routed is True


def test_a_listing_only_rows_prices_reach_the_first_frame(
    controller: ProviderController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live path's keyless price chain, reused: this surface never goes live.

    A cached listing states whatever money the provider publishes, and for a
    model the registry has never heard of, a listing that publishes none leaves
    the row at the unknown sentinel -- a blank cell. That blank would be
    PERMANENT on the composer's inline list, which is painted from this frame and
    never refreshed from the network, so the same disk-only chain the live pass
    runs fills it here (agent review round 2, R2-3).

    The shipped rows are asserted UNTOUCHED beside it: frame one is the
    registry's answer for registry models, and that equality is what the
    cold-frame invariant rests on.
    """
    monkeypatch.setattr(discovery, "fetch_models", _no_network)
    _plant(tmp_path, "anthropic", ids=["claude-opus-5", "claude-fable-6"])
    _plant_prices(tmp_path, "anthropic", "claude-fable-6", input_price=10.0, output_price=50.0)

    rows = {
        row.model_id: row
        for row in controller.initial_catalogue(cache_dir=tmp_path)
        if row.provider == "anthropic"
    }

    filled = rows["claude-fable-6"]
    assert (filled.input_price, filled.output_price) == (10.0, 50.0)
    shipped = {
        row.model_id: row for row in controller.static_catalogue() if row.provider == "anthropic"
    }
    assert shipped, "the fixture depends on anthropic having shipped rows to compare"
    for model_id, row in shipped.items():
        assert rows[model_id] == row, f"the price chain touched a shipped row: {model_id}"


def test_a_listing_only_row_keeps_the_unknown_sentinel_with_no_price_document(
    controller: ProviderController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No document on disk means no price, never an invented one.

    The chain is disk-only, so this is also what proves the frame acquired no
    network leg: ``fetch_models`` is poisoned, and a fetch would raise.
    """
    monkeypatch.setattr(discovery, "fetch_models", _no_network)
    _plant(tmp_path, "anthropic", ids=["claude-fable-6"])

    row = next(
        row
        for row in controller.initial_catalogue(cache_dir=tmp_path)
        if row.model_id == "claude-fable-6"
    )

    assert (row.input_price, row.output_price) == (-1.0, -1.0)


def test_a_provider_whose_listing_owns_the_set_still_prunes(
    controller: ProviderController, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The policy hook did not move: deepseek's listing OWNS its selectable set.

    ``discovery._listing_replaces_static()`` says a native DeepSeek listing
    replaces the registry rather than unioning with it, and this frame has to
    keep honouring that -- the point of the rule is that a shipped-but-retired id
    must not flash in the picker. Reading every provider through the same
    reader preserves it; re-deriving the merge here would not.
    """
    monkeypatch.setattr(discovery, "fetch_models", _no_network)
    _plant(tmp_path, "deepseek", ids=["deepseek-flash"])

    ids = {
        row.model_id
        for row in controller.initial_catalogue(cache_dir=tmp_path)
        if row.provider == "deepseek"
    }

    assert ids == {"deepseek-flash"}
