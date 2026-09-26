"""The episode USD cap scales from a STATED price basis, and says so.

The defect these pin: a $3.00 per-episode cap that a cheap route never came
near truncated a ~24x-pricier route at 106 of its 500 protocol steps
(2026-09-26), so the sealed partial read as a capability result when it was a
budget artifact. The fix reads the figure as "an episode budget at the basis
route's prices" and rescales it by the routes' own published rates, one-way
(never downward). These tests exercise the arithmetic, the refusal paths, and
the manifest stamping that keep the change legible.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_episode

_BASIS = ("openrouter", "deepseek/deepseek-v4.1-flash")
_ROUTE = ("openrouter", "qwen/qwen3.8-max-0902")


def _rows(prices: dict[str, tuple[float, float]]) -> list[SimpleNamespace]:
    """Stand-in listing rows; the call site reads only id/input/output."""
    return [
        SimpleNamespace(id=model, input_price=inp, output_price=outp)
        for model, (inp, outp) in prices.items()
    ]


def _stub_listing(monkeypatch, prices: dict[str, tuple[float, float]]) -> dict[str, object]:
    """Replace the fresh price fetch; records how it was called."""
    captured: dict[str, object] = {}

    def fake_openrouter_rows(*, want_id=None, timeout=None, ttl_s=None, cache_dir=None):
        captured.update(want_id=want_id, timeout=timeout, ttl_s=ttl_s, cache_dir=cache_dir)
        return _rows(prices)

    monkeypatch.setattr("local_operator.model.prices.openrouter_rows", fake_openrouter_rows)
    return captured


def test_without_a_basis_the_figure_is_used_as_stated() -> None:
    """No flag, no scaling: the historical figure is exactly what runs."""
    cap = run_episode._scale_usd_cap(3_000_000, route=_ROUTE, basis=None)
    assert cap.effective_micros == cap.configured_micros == 3_000_000
    assert cap.factor == 1.0
    assert cap.basis is None
    stamped = run_episode._usd_cap_metadata(cap)
    assert stamped["usd_cap_effective_micros"] == 3_000_000
    assert stamped["usd_cap_price_factor_micros"] == 1_000_000
    assert stamped["usd_cap_at"] is None
    assert "usd_cap_route_price_micros_per_mtok" not in stamped


def test_a_basis_equal_to_the_route_is_recorded_and_not_rescaled(monkeypatch) -> None:
    """The basis may name the run's own route; the pairing is still stamped."""
    _stub_listing(monkeypatch, {})  # must not be consulted
    cap = run_episode._scale_usd_cap(3_000_000, route=_ROUTE, basis=_ROUTE)
    assert cap.effective_micros == 3_000_000 and cap.factor == 1.0
    assert cap.basis == "openrouter/qwen/qwen3.8-max-0902"


def test_scaling_uses_both_published_rates_and_never_tightens(monkeypatch) -> None:
    """The factor is the route/basis ratio of input+output rates, floored at 1.

    Prices here are the 2026-09-26 live values for the pair the campaign ran,
    pinned as fixtures so the arithmetic is reproducible offline: the figure
    scales 8.0/0.325 = ~24.6x upward for the pricier route, and a pricier basis
    with a cheaper run route floors at 1.0 — the scaling may only relax a cap.
    """
    prices = {
        "deepseek/deepseek-v4.1-flash": (0.035, 0.29),
        "qwen/qwen3.8-max-0902": (2.0, 6.0),
    }
    captured = _stub_listing(monkeypatch, prices)

    cap = run_episode._scale_usd_cap(3_000_000, route=_ROUTE, basis=_BASIS)
    assert cap.route_price_usd_per_mtok == pytest.approx(8.0)
    assert cap.basis_price_usd_per_mtok == pytest.approx(0.325)
    assert cap.factor == pytest.approx(8.0 / 0.325)
    assert cap.effective_micros == round(3_000_000 * 8.0 / 0.325)
    assert cap.effective_micros == 73_846_154
    assert cap.basis == "openrouter/deepseek/deepseek-v4.1-flash"

    stamped = run_episode._usd_cap_metadata(cap)
    # Integer encoding, per the portable-metadata contract: millionths for the
    # factor, micro-USD per Mtok for the two rates.
    assert stamped["usd_cap_price_factor_micros"] == 24_615_385
    assert stamped["usd_cap_route_price_micros_per_mtok"] == 8_000_000
    assert stamped["usd_cap_basis_price_micros_per_mtok"] == 325_000

    # One fresh fetch serves both ids, and it is deliberately NOT the shared
    # catalogue cache: an isolated, zero-TTL read of the provider's listing.
    assert captured["ttl_s"] == 0.0
    assert captured["timeout"] == run_episode._PRICE_BASIS_FETCH_TIMEOUT_S
    assert "lop-usd-basis-" in Path(str(captured["cache_dir"])).name

    # The reverse pairing may not shrink the operator's figure.
    reverse = run_episode._scale_usd_cap(3_000_000, route=_BASIS, basis=_ROUTE)
    assert reverse.factor == 1.0
    assert reverse.effective_micros == 3_000_000


def test_the_demo_pair_re_derives_the_campaign_caps(monkeypatch) -> None:
    """The evidence the fix was accepted on, as a regression pin.

    qwen3.8-max-0902 under a $3.00 figure stated at deepseek-v4.1-flash:
    $73.85 (was $3.00 — the cap that truncated task_016 at 106 steps).
    deepseek-v4.1-flash itself: $3.00, unchanged, because its ratio is 1.0.
    """
    prices = {
        "deepseek/deepseek-v4.1-flash": (0.035, 0.29),
        "qwen/qwen3.8-max-0902": (2.0, 6.0),
    }
    _stub_listing(monkeypatch, prices)
    qwen = run_episode._scale_usd_cap(3_000_000, route=_ROUTE, basis=_BASIS)
    deepseek = run_episode._scale_usd_cap(3_000_000, route=_BASIS, basis=_BASIS)
    assert qwen.effective_micros == 73_846_154
    assert deepseek.effective_micros == 3_000_000


def test_an_unpriceable_basis_refuses_at_preflight(monkeypatch) -> None:
    """A basis that cannot be priced is refused, never silently un-scaled."""
    _stub_listing(monkeypatch, {"qwen/qwen3.8-max-0902": (2.0, 6.0)})
    with pytest.raises(ValueError, match="does not price"):
        run_episode._scale_usd_cap(3_000_000, route=_ROUTE, basis=_BASIS)

    _stub_listing(
        monkeypatch,
        {"qwen/qwen3.8-max-0902": (2.0, 6.0), "deepseek/deepseek-v4.1-flash": (0.0, 0.0)},
    )
    with pytest.raises(ValueError, match="unpriced rates"):
        run_episode._scale_usd_cap(3_000_000, route=_ROUTE, basis=_BASIS)

    def broken_fetch(**_kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("local_operator.model.prices.openrouter_rows", broken_fetch)
    with pytest.raises(ValueError, match="could not read the live price listing"):
        run_episode._scale_usd_cap(3_000_000, route=_ROUTE, basis=_BASIS)


def test_providers_without_a_live_price_listing_refuse_rather_than_guess() -> None:
    """Out-of-set providers fail closed: an unverified source sized the caps wrong."""
    with pytest.raises(ValueError, match="no live public price listing"):
        run_episode._scale_usd_cap(
            3_000_000, route=("openai", "gpt-4o"), basis=("openai", "gpt-4o-mini")
        )
    with pytest.raises(ValueError, match="no live public price listing"):
        run_episode._scale_usd_cap(3_000_000, route=_ROUTE, basis=("anthropic", "claude-sonnet-4"))


def test_the_cli_wires_the_flag() -> None:
    args = run_episode.build_parser().parse_args(
        [
            "--selector",
            "s.json",
            "--task-id",
            "t",
            "--route",
            "openrouter/qwen/qwen3.8-max-0902",
            "--run-root",
            "r",
            "--max-usd-at",
            "openrouter/deepseek/deepseek-v4.1-flash",
        ]
    )
    assert args.max_usd_at == ("openrouter", "deepseek/deepseek-v4.1-flash")

    args = run_episode.build_parser().parse_args(
        [
            "--selector",
            "s.json",
            "--task-id",
            "t",
            "--route",
            "openrouter/qwen/qwen3.8-max-0902",
            "--run-root",
            "r",
        ]
    )
    assert args.max_usd_at is None


def test_the_derivation_rides_into_the_spec_metadata(tmp_path) -> None:
    """The stamped fields are part of the spec the bundle seals.

    ``run()`` merges ``_usd_cap_metadata`` into the same metadata mapping
    ``build_spec`` hashes and seals, so a reader of any bundle can tell which
    bound applied and re-derive why."""
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "task.py").write_text("TASK = {}\n")
    cap = run_episode._scale_usd_cap(3_000_000, route=_ROUTE, basis=None)
    spec = run_episode.build_spec(
        episode_id="ep-cap",
        selector=run_episode.AdapterSelector.model_validate(
            {
                "schema_version": "1.6",
                "adapter_id": "tiny",
                "distribution": "tiny-adapter",
                "version": "1.0",
                "entry_point": "tiny:create",
                "package_digest": "a" * 64,
                "release_digest": "b" * 64,
                "python_executable": str(tmp_path / "python"),
                "workspace": str(tmp_path),
                "workspace_digest": "c" * 64,
                "route_capability": "computer",
            }
        ),
        task_id="task",
        route=run_episode._route_identity(*_ROUTE),
        benchmark_id="bench",
        benchmark_release="release",
        secret_refs=(),
        infra_values=(),
        max_usd_micros=cap.effective_micros,
        max_wall_ms=1000,
        max_steps=1,
        metadata=run_episode._usd_cap_metadata(cap),
    )
    assert spec.metadata["usd_cap_effective_micros"] == 3_000_000
    assert spec.metadata["usd_cap_configured_micros"] == 3_000_000
    assert spec.metadata["usd_cap_price_factor_micros"] == 1_000_000
    assert spec.metadata["usd_cap_at"] is None


def test_main_refuses_an_unpriceable_basis_before_allocating(tmp_path, monkeypatch, capsys) -> None:
    """End to end at the CLI: the refusal is a preflight exit, not a traceback."""
    selector = tmp_path / "selector.json"
    selector.write_text(
        json.dumps(
            {
                "schema_version": "1.6",
                "adapter_id": "tiny",
                "distribution": "tiny-adapter",
                "version": "1.0",
                "entry_point": "tiny:create",
                "package_digest": "a" * 64,
                "release_digest": "b" * 64,
                "python_executable": str(tmp_path / "python"),
                "workspace": str(tmp_path),
                "workspace_digest": "c" * 64,
                "route_capability": "computer",
            }
        )
    )
    _stub_listing(monkeypatch, {})
    status = run_episode.main(
        [
            "--selector",
            str(selector),
            "--task-id",
            "task",
            "--route",
            "openrouter/qwen/qwen3.8-max-0902",
            "--run-root",
            str(tmp_path / "run"),
            "--max-usd",
            "3",
            "--max-usd-at",
            "openrouter/deepseek/deepseek-v4.1-flash",
        ]
    )
    assert status == run_episode.EXIT_PREFLIGHT
    err = capsys.readouterr().err
    assert "does not price" in err
    assert "drop --max-usd-at" in err
