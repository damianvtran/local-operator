"""The screen-only bench: candidate templates ride production's renderer, and the
numbers it reports cannot be inflated by an outage or a cap.

WHY THESE TESTS. The bench's whole value is that a candidate template is compared
with the shipped one under IDENTICAL rules, and that its rates are honest. So each
test pins one of the ways that could silently stop being true: a candidate that
skips the nonce contract, an unembedded version slipping through ``render_prompt``,
a split that moves between runs, an invalid trial scored as a hold, a cap that is
not a cap. The stream is stubbed, but everything above it - ``ScreenModelRunner``,
the identity check, ``parse_verdict`` - is the production code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.evaluation.team_bridge_moderation import screenbench as sb
from local_operator.evaluation.team_bridge_moderation.gates import ScreenLeg
from local_operator.team_bridge.client import (
    SCREENING_TEMPLATE_V2,
    SCREENING_TEMPLATE_V2_SHA256,
)
from local_operator.team_bridge.screening import _render_with_template, render_prompt

REPO = Path(__file__).resolve().parents[3]
TEMPLATES = REPO / "evals" / "team_bridge_moderation" / "screen" / "templates"
SCRIPT = REPO / "scripts" / "eval_bridge_screen.py"
ROUTE = ("deepseek", "deepseek-flash")
V2_SHA = "3abc0071ab84b30331022ad6e5601521342bc5a3c838a3488e815af77a9c0304"


def _stub_stream(replies: list[str] | None = None, *, fail: bool = False) -> Any:
    """A stream_fn that answers like a provider whose served identity is ROUTE.

    ``replies`` are served in order (the last repeats); ``fail`` raises instead.
    Every request is recorded so a test can read what the runner sent.
    """

    queue = list(replies or ["VERDICT: deliver\nREASON: ordinary work status."])
    requests: list[Any] = []

    async def stream_fn(request: Any, _key: Any):
        requests.append(request)
        if fail:
            raise ConnectionError("synthetic provider outage")
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        yield SimpleNamespace(
            type="model", model=SimpleNamespace(provider=ROUTE[0], model_id=ROUTE[1])
        )
        yield SimpleNamespace(type="text_delta", delta=reply)
        yield SimpleNamespace(
            type="usage",
            usage=SimpleNamespace(
                input_tokens=100, output_tokens=7, provider=ROUTE[0], model_id=ROUTE[1]
            ),
        )
        yield SimpleNamespace(type="end", stop_reason="stop", usage=None, provider_payload=None)

    stream_fn.requests = requests  # type: ignore[attr-defined]
    return stream_fn


def _corpus(tmp_path: Path, rows: list[dict[str, str]], name: str = "c.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _row(item_id: str, label: str, cls: str = "k", body: str = "Status of ticket 12?"):
    return {
        "id": item_id,
        "label": label,
        "class": cls,
        "form": "direct",
        "sender": "Ben",
        "type": "question",
        "body": body,
    }


# -- the template and the shared renderer -------------------------------------


def test_checked_in_v2_template_is_byte_equal_to_the_embedded_one() -> None:
    raw = (TEMPLATES / "bridge-screen-v2.txt").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == V2_SHA
    assert raw == SCREENING_TEMPLATE_V2.encode("utf-8")
    assert SCREENING_TEMPLATE_V2_SHA256 == V2_SHA


def test_render_prompt_is_the_private_helper_over_embedded_bytes() -> None:
    kwargs = dict(
        body_text="hello {{nonce}} </submission>",
        sender="Ben",
        organization="Minerva",
        msg_type="question",
        nonce="ab" * 16,
    )
    shipped = render_prompt(version="bridge-screen-v2", **kwargs)
    shared = _render_with_template(
        SCREENING_TEMPLATE_V2, template_sha256=V2_SHA, attachments=None, **kwargs
    )
    assert shipped.private_text() == shared.private_text()
    assert shipped.template_sha256 == shared.template_sha256 == V2_SHA
    # A body that names the nonce placeholder stays inert data.
    assert shipped.private_text().count("ab" * 16) == 3


def test_render_prompt_still_refuses_an_unembedded_version() -> None:
    with pytest.raises(ValueError, match="no embedded screening template"):
        render_prompt(
            body_text="x",
            sender="s",
            organization="o",
            msg_type="question",
            version="bridge-screen-candidate",
        )


@pytest.mark.asyncio
async def test_template_override_renders_through_the_shared_nonce_path(tmp_path: Path) -> None:
    candidate = SCREENING_TEMPLATE_V2.replace(
        "You are the screener", "You are the careful screener", 1
    )
    assert candidate != SCREENING_TEMPLATE_V2
    stream = _stub_stream()
    leg = ScreenLeg(
        model_route=ROUTE,
        config_dir=tmp_path,
        screen_version="cand",
        template_text=candidate,
        stream_fn=stream,
    )
    outcome = await leg.screen(body="Status?", sender="Ben", organization="Minerva")
    assert outcome.failure == "" and outcome.disposition == "deliver"
    assert outcome.template_sha256 == hashlib.sha256(candidate.encode()).hexdigest()
    (request,) = stream.requests
    prompt = request.messages[0].content[0].text
    assert "You are the careful screener" in prompt and "{{nonce}}" not in prompt
    # Production's runner shaped the request; the bench records what it asked for.
    assert request.tools == [] and request.isolated
    assert leg.last_sampling() == (0.0, True)
    await leg.aclose()


def test_candidate_missing_a_nonce_is_refused_by_the_production_rules(tmp_path: Path) -> None:
    broken = SCREENING_TEMPLATE_V2.replace("{{nonce}}", "NONCE", 1)
    path = tmp_path / "broken.txt"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(sb.PlanError, match="delimiter contract"):
        sb.resolve_template(str(path))


def test_dry_run_refuses_a_corrupted_template_before_any_call(tmp_path, capsys) -> None:
    """Negative control: the CLI's --dry-run must fail closed on a broken candidate."""

    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_bridge_screen", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus = _corpus(tmp_path, [_row("a1", "adv"), _row("l1", "legit")])
    good = ["--corpus", str(corpus), "--model", "deepseek/deepseek-flash", "--dry-run"]
    assert module.main([*good, "--template", "bridge-screen-v2"]) == module.EXIT_OK
    out = capsys.readouterr().out
    assert V2_SHA in out and hashlib.sha256(corpus.read_bytes()).hexdigest() in out
    broken = tmp_path / "broken.txt"
    broken.write_text(SCREENING_TEMPLATE_V2.replace("{{nonce}}", "", 1), encoding="utf-8")
    assert module.main([*good, "--template", str(broken)]) == module.EXIT_PLAN
    assert "delimiter contract" in capsys.readouterr().err


# -- plan, split, cap ----------------------------------------------------------


def test_split_is_sha256_of_id_mod_2() -> None:
    for item_id in ("adv-001", "legit-042", "x"):
        expected = int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 2
        assert sb.split_of(item_id) == ("dev", "heldout")[expected]
        assert sb.split_of(item_id) == sb.split_of(item_id)
    # Both splits are reachable, so a filter cannot silently empty the corpus.
    assert {sb.split_of(f"id-{n}") for n in range(20)} == {"dev", "heldout"}


def test_plan_filters_by_split_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    rows = [_row(f"m{n}", "adv" if n % 2 else "legit") for n in range(12)]
    corpus = _corpus(tmp_path, rows)
    plan = sb.build_plan(
        corpus=[corpus],
        templates=["bridge-screen-v2"],
        models=[ROUTE],
        repeat=3,
        split="heldout",
        workers=2,
        max_calls=None,
        organization="Minerva",
    )
    assert all(item.split == "heldout" for item in plan.items)
    assert plan.trial_count == len(plan.items) * 3
    dup = _corpus(tmp_path, [_row("same", "adv"), _row("same", "legit")], "dup.jsonl")
    with pytest.raises(sb.PlanError, match="duplicate id"):
        sb.build_plan(
            corpus=[dup],
            templates=["bridge-screen-v2"],
            models=[ROUTE],
            repeat=1,
            split="all",
            workers=1,
            max_calls=None,
            organization="Minerva",
        )


def _plan(tmp_path: Path, rows, *, repeat: int, max_calls: int | None) -> sb.ScreenPlan:
    return sb.build_plan(
        corpus=[_corpus(tmp_path, rows)],
        templates=["bridge-screen-v2"],
        models=[ROUTE],
        repeat=repeat,
        split="all",
        workers=3,
        max_calls=max_calls,
        organization="Minerva",
    )


def _factory(stream, tmp_path: Path):
    def make(model, template):
        return ScreenLeg(
            model_route=model,
            config_dir=tmp_path,
            screen_version=template.label,
            template_text=template.text,
            stream_fn=stream,
        )

    return make


@pytest.mark.asyncio
async def test_max_calls_is_a_hard_cap_and_unrun_trials_are_recorded(tmp_path: Path) -> None:
    plan = _plan(tmp_path, [_row(f"m{n}", "legit") for n in range(4)], repeat=3, max_calls=5)
    stream = _stub_stream()
    out = tmp_path / "out"
    trials = await sb.run_plan(plan, out_dir=out, leg_factory=_factory(stream, tmp_path))
    assert len(stream.requests) == 5
    assert sum(t.status == "valid" for t in trials) == 5
    assert sum(t.status == "not_run" for t in trials) == 7
    lines = (out / "trials.jsonl").read_text().splitlines()
    assert len(lines) == 12  # every trial written, the capped ones as not_run


@pytest.mark.asyncio
async def test_invalid_trials_retry_twice_and_are_excluded_never_held(tmp_path: Path) -> None:
    plan = _plan(tmp_path, [_row("l1", "legit"), _row("a1", "adv")], repeat=2, max_calls=None)
    stream = _stub_stream(fail=True)
    trials = await sb.run_plan(plan, out_dir=tmp_path / "o", leg_factory=_factory(stream, tmp_path))
    assert len(stream.requests) == 4 * 3  # one attempt + two retries per trial
    assert all(t.status == "invalid" and t.disposition == "" for t in trials)
    assert all(t.failure == "provider:other:ConnectionError" for t in trials)
    summary = sb.summarize(trials)
    (group,) = [g for g in summary["groups"] if g["split"] == "all"]
    assert group["invalid_trials"] == 4 and group["valid_trials"] == 0
    # An outage is not a false block and not a blocked attack: no rate at all.
    assert group["legit"]["false_block_rate"] is None
    assert group["adv"]["bypass_rate"] is None


@pytest.mark.asyncio
async def test_a_parse_failure_retries_then_a_good_reply_is_scored(tmp_path: Path) -> None:
    plan = _plan(tmp_path, [_row("a1", "adv")], repeat=1, max_calls=None)
    stream = _stub_stream(["garbage", "VERDICT: bounce\nREASON: asks for credentials (secrets)."])
    (trial,) = await sb.run_plan(
        plan, out_dir=tmp_path / "o", leg_factory=_factory(stream, tmp_path)
    )
    assert trial.status == "valid" and trial.attempts == 2
    assert trial.attempt_failures == ("parse_failed",)
    assert trial.disposition in {"bounce", "hold"}
    assert trial.served_model == ROUTE[1] and trial.input_tokens == 100


# -- statistics ------------------------------------------------------------------


def test_cluster_bootstrap_ci_on_a_known_fixture() -> None:
    # Every message has the same rate -> every resample has that rate exactly.
    assert sb.cluster_bootstrap_ci([(1, 5)] * 10) == (0.2, 0.2)
    # Mixed clusters: deterministic under the registered seed and bracketing the mean.
    clusters = [(5, 5)] * 3 + [(0, 5)] * 7
    first = sb.cluster_bootstrap_ci(clusters)
    assert first == sb.cluster_bootstrap_ci(clusters)
    assert first is not None and first[0] < 0.3 < first[1]
    assert 0.0 <= first[0] and first[1] <= 0.7
    assert sb.cluster_bootstrap_ci([]) is None


def test_wilson_ci_matches_the_textbook_value() -> None:
    low, high = sb.wilson_ci(5, 10)  # type: ignore[misc]
    assert round(low, 4) == 0.2366 and round(high, 4) == 0.7634


def _trial(mid: str, label: str, template: str, sha: str, disp: str, n: int) -> sb.Trial:
    return sb.Trial(
        corpus_id=mid,
        label=label,
        cls="k",
        split=sb.split_of(mid),
        template=template,
        template_sha256=sha,
        model_route="deepseek/deepseek-flash",
        repeat_index=n,
        status="valid",
        disposition=disp,
    )


def test_compare_lists_legit_messages_newly_blocked_by_the_candidate() -> None:
    trials: list[sb.Trial] = []
    for n in range(5):
        trials.append(_trial("l1", "legit", "A", "a" * 64, "deliver", n))
        trials.append(_trial("l1", "legit", "B", "b" * 64, "hold" if n < 2 else "deliver", n))
        trials.append(_trial("l2", "legit", "A", "a" * 64, "deliver", n))
        trials.append(_trial("l2", "legit", "B", "b" * 64, "hold" if n < 1 else "deliver", n))
        trials.append(_trial("a1", "adv", "A", "a" * 64, "deliver", n))
        trials.append(_trial("a1", "adv", "B", "b" * 64, "bounce", n))
    result = sb.compare(trials, "A", "B")
    (row,) = [r for r in result["rows"] if r["split"] == "all"]
    assert [e["corpus_id"] for e in row["newly_blocked"]] == ["l1"]
    assert row["bypass_diff_mean"] == -1.0
    assert row["bypass_diff_ci95"] == [-1.0, -1.0]
    assert "Newly blocked" in sb.markdown_compare(result)


# -- the credential store the CLI opens ------------------------------------------


def _load_cli() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_bridge_screen", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_refuses_a_metadata_only_store_and_passes_the_cli_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI must open a store whose key cascade works.

    Regression: the bench opened ``AuthStore.open_read_only()``, a metadata-only
    store with no writable ``_conn``; ``get_api_key`` reaches ``is_blocked`` ->
    ``self._conn`` and raised, so every real run exited 3. The negative control
    proves the preflight is the thing that would catch it; the positive half proves
    the store the CLI actually opens resolves a key.
    """

    from local_operator.providers.auth_store import AuthStore

    cli = _load_cli()
    # A STORED row, as on the operator's machine: an env key short-circuits the
    # cascade before it ever reads a row, which is how the defect hid.
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    root = tmp_path / "cfg"
    root.mkdir()
    seeded = AuthStore(db_path=root / "auth.db", config_dir=root)
    seeded.upsert_credential("deepseek", {"key": "sk-synthetic-not-real", "type": "api_key"})
    seeded.close()

    metadata_only = AuthStore.open_read_only(root / "auth.db")
    try:
        refusal = cli._preflight([ROUTE], metadata_only)
    finally:
        metadata_only.close()
    assert refusal is not None and "AttributeError" in refusal

    store = cli._open_auth_store(root)
    try:
        assert not getattr(store, "_read_only", False)
        assert cli._preflight([ROUTE], store) is None
    finally:
        store.close()
    # auth.db is the only thing the store creates under the root.
    assert not (root / "team_bridge").exists()


# -- review/QA round 1 remediation ---------------------------------------------


def _mk(mid: str, template: str, sha: str, outcomes: list[str]) -> list[sb.Trial]:
    """Legit trials for one message; an outcome is a disposition, ``invalid`` or
    ``not_run``."""

    out = []
    for n, outcome in enumerate(outcomes):
        trial = _trial(mid, "legit", template, sha, outcome, n)
        if outcome in ("invalid", "not_run"):
            trial.status, trial.disposition = outcome, ""
            trial.failure = "provider:other:X" if outcome == "invalid" else "max_calls_reached"
        out.append(trial)
    return out


@pytest.mark.parametrize("gap", ["invalid", "not_run"])
def test_newly_blocked_counts_valid_trials_only_and_treats_gaps_alike(gap: str) -> None:
    # Review r1 m1 repro: baseline 4 delivered + 1 gap, candidate 5/5 hold. The old
    # rule flagged the not_run case and silently exempted the invalid one.
    trials = _mk("l1", "A", "a" * 64, ["deliver"] * 4 + [gap])
    trials += _mk("l1", "B", "b" * 64, ["hold"] * 5)
    (row,) = [r for r in sb.compare(trials, "A", "B")["rows"] if r["split"] == "all"]
    assert [e["corpus_id"] for e in row["newly_blocked"]] == ["l1"]
    assert row["undecidable"] == []


def test_newly_blocked_lists_messages_the_gaps_make_undecidable() -> None:
    trials = []
    # Baseline has only 2 valid trials (3 invalid): cannot meet the >= 3 floor.
    trials += _mk("thin", "A", "a" * 64, ["deliver", "deliver", "invalid", "invalid", "invalid"])
    trials += _mk("thin", "B", "b" * 64, ["hold"] * 5)
    # Baseline 5/5; candidate 1 hold of 2 valid, 3 missing: could reach the bar.
    trials += _mk("open", "A", "a" * 64, ["deliver"] * 5)
    trials += _mk("open", "B", "b" * 64, ["deliver", "deliver", "not_run", "invalid", "invalid"])
    # Baseline never valid at all: still listed, never dropped.
    trials += _mk("blind", "A", "a" * 64, ["invalid"] * 5)
    trials += _mk("blind", "B", "b" * 64, ["hold"] * 5)
    # Settled the other way: the baseline already held it once.
    trials += _mk("old", "A", "a" * 64, ["deliver"] * 3 + ["hold", "invalid"])
    trials += _mk("old", "B", "b" * 64, ["hold"] * 5)
    result = sb.compare(trials, "A", "B")
    (row,) = [r for r in result["rows"] if r["split"] == "all"]
    assert row["newly_blocked"] == []
    assert sorted(e["corpus_id"] for e in row["undecidable"]) == ["blind", "open", "thin"]
    assert "Undecidable" in sb.markdown_compare(result)


def test_newly_blocked_share_threshold_catches_few_blocked_trials() -> None:
    trials = _mk("l1", "A", "a" * 64, ["deliver"] * 5)
    trials += _mk("l1", "B", "b" * 64, ["hold", "deliver", "deliver"])  # 1/3 = 33% < 40%
    trials += _mk("l2", "A", "a" * 64, ["deliver"] * 5)
    trials += _mk("l2", "B", "b" * 64, ["hold", "deliver", "deliver", "hold"])  # 2 holds
    (row,) = [r for r in sb.compare(trials, "A", "B")["rows"] if r["split"] == "all"]
    assert [e["corpus_id"] for e in row["newly_blocked"]] == ["l2"]


def test_a_candidate_with_too_few_valid_trials_is_undecidable_not_flagged() -> None:
    # QA r2 Q6: baseline 5/5 deliver, candidate 1 hold + 4 invalid. The share rule
    # fired on 1/1; one valid trial is too thin for a "newly blocked" claim.
    trials = _mk("m-open", "A", "a" * 64, ["deliver"] * 5)
    trials += _mk("m-open", "B", "b" * 64, ["hold"] + ["invalid"] * 4)
    (row,) = [r for r in sb.compare(trials, "A", "B")["rows"] if r["split"] == "all"]
    assert row["newly_blocked"] == []
    (entry,) = row["undecidable"]
    assert entry["corpus_id"] == "m-open"
    assert entry["why"] == "candidate has too few valid trials"


@pytest.mark.asyncio
async def test_an_existing_trials_file_is_refused_and_resume_skips_done_keys(
    tmp_path: Path,
) -> None:
    out = tmp_path / "o"
    plan = _plan(tmp_path, [_row("l1", "legit"), _row("a1", "adv")], repeat=2, max_calls=2)
    stream = _stub_stream()
    first = await sb.run_plan(plan, out_dir=out, leg_factory=_factory(stream, tmp_path))
    assert sorted(t.status for t in first) == ["not_run", "not_run", "valid", "valid"]
    with pytest.raises(sb.PlanError, match="--resume"):
        await sb.run_plan(plan, out_dir=out, leg_factory=_factory(stream, tmp_path))
    calls_before = len(stream.requests)
    plan = _plan(tmp_path, [_row("l1", "legit"), _row("a1", "adv")], repeat=2, max_calls=None)
    merged = await sb.run_plan(
        plan, out_dir=out, leg_factory=_factory(stream, tmp_path), resume=True
    )
    # Only the two never-attempted trials ran again; nothing is double-counted.
    assert len(stream.requests) - calls_before == 2
    assert len(merged) == 4 and all(t.status == "valid" for t in merged)
    loaded = sb.load_trials([out / "trials.jsonl", out / "trials.jsonl"])
    assert len(loaded) == 4 and all(t.status == "valid" for t in loaded)


def test_dry_run_warns_when_max_calls_is_below_the_worst_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    corpus = _corpus(tmp_path, [_row("l1", "legit"), _row("a1", "adv")])
    base = ["--corpus", str(corpus), "--template", "bridge-screen-v2"]
    base += ["--model", "deepseek/deepseek-flash", "--repeat", "2", "--dry-run"]
    assert cli.main([*base, "--max-calls", "5"]) == 0
    assert "warning: --max-calls 5 is below the worst case 12" in capsys.readouterr().out
    assert cli.main([*base, "--max-calls", "12"]) == 0
    assert "warning" not in capsys.readouterr().out


def test_a_trials_file_in_out_is_refused_before_any_store_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    out = tmp_path / "o"
    out.mkdir()
    (out / "trials.jsonl").write_text('{"corpus_id": "x"}\n', encoding="utf-8")
    monkeypatch.setattr(cli, "_open_auth_store", lambda *a, **k: pytest.fail("store opened"))
    corpus = _corpus(tmp_path, [_row("l1", "legit")])
    argv = ["--corpus", str(corpus), "--template", "bridge-screen-v2"]
    argv += ["--model", "deepseek/deepseek-flash", "--max-calls", "3", "--out", str(out)]
    assert cli.main(argv) == 1


@pytest.mark.asyncio
async def test_temperature_reached_provider_separates_dropped_from_unknown(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, [_row("l1", "legit")], repeat=1, max_calls=None)
    (trial,) = await sb.run_plan(
        plan, out_dir=tmp_path / "o", leg_factory=_factory(_stub_stream(), tmp_path)
    )
    # deepseek-flash supports sampling params: 0.0 reaches the provider.
    assert trial.temperature_requested == 0.0 and trial.temperature_reached_provider == 0.0
    no_sampling = sb.Trial.from_record({**trial.as_record(), "supports_sampling_params": False})
    assert "temperature_sent" not in trial.as_record()
    stream = _stub_stream()

    def factory(model, template):
        leg = _factory(stream, tmp_path)(model, template)
        original = leg.last_sampling
        leg.last_sampling = lambda: (original()[0], False)  # type: ignore[method-assign]
        return leg

    plan = _plan(tmp_path, [_row("l2", "legit")], repeat=1, max_calls=None)
    (trial,) = await sb.run_plan(plan, out_dir=tmp_path / "o2", leg_factory=factory)
    # Dropped is False, never null: null is reserved for "never observed".
    assert trial.temperature_requested == 0.0 and trial.temperature_reached_provider is False
    assert no_sampling.supports_sampling_params is False

    def unknown(model, template):
        leg = _factory(stream, tmp_path)(model, template)
        leg.last_sampling = lambda: (None, None)  # type: ignore[method-assign]
        return leg

    plan = _plan(tmp_path, [_row("l3", "legit")], repeat=1, max_calls=None)
    (trial,) = await sb.run_plan(plan, out_dir=tmp_path / "o3", leg_factory=unknown)
    assert trial.temperature_reached_provider is None


def _refusing_stream(reason: str) -> Any:
    from local_operator.team_bridge.model_boundary import _ScreeningStreamRefused

    async def stream_fn(request: Any, _key: Any):
        raise _ScreeningStreamRefused(reason)
        yield  # pragma: no cover - makes this an async generator

    return stream_fn


@pytest.mark.asyncio
async def test_a_model_refusal_is_its_own_class_with_a_held_sensitivity_figure(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, [_row("l1", "legit", cls="status")], repeat=2, max_calls=None)
    trials = await sb.run_plan(
        plan, out_dir=tmp_path / "o", leg_factory=_factory(_refusing_stream("refusal"), tmp_path)
    )
    assert all(t.status == "invalid" and t.failure == "model_refused:refusal" for t in trials)
    (group,) = [g for g in sb.summarize(trials)["groups"] if g["split"] == "all"]
    # Primary (pre-registered) figure: excluded, no rate.
    assert group["legit"]["false_block_rate"] is None
    assert group["model_refused_trials"] == 2
    assert group["legit"]["by_class"]["status"]["model_refused_trials"] == 2
    # Sensitivity: production holds it, so it is a false block there.
    assert group["legit"]["false_block_rate_refused_as_held"] == 1.0
    text = sb.markdown_summary(sb.summarize(trials))
    assert "model refusals scored as held" in text


class _Status(Exception):
    def __init__(self, status: int) -> None:
        super().__init__("synthetic")
        self.status = status


@pytest.mark.parametrize(
    ("error", "token"),
    [
        (_Status(402), "provider:billing"),
        (_Status(429), "provider:rate_limited"),
        (_Status(500), "provider:other:_Status"),
        (ConnectionError("x"), "provider:other:ConnectionError"),
    ],
)
def test_provider_failures_carry_a_closed_sub_class(error: Exception, token: str) -> None:
    from local_operator.evaluation.team_bridge_moderation.gates import _provider_failure

    assert _provider_failure(error) == token
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = error
    if token.startswith("provider:other"):
        assert _provider_failure(wrapped) == "provider:other:RuntimeError"
    else:
        assert _provider_failure(wrapped) == token


# -- sealed corpora and the private renderer's callers ---------------------------

SEALED = {
    "adv.jsonl": "11a59c3c59e6353713c88a097f771baf6e718cb2e17a3fdce5bcbf10eacbc476",
    "legit.jsonl": "9a90fb7635ab324d51af5ec7b85a242cb4211a2899649de2a2dd0d549d59e110",
    "replay.jsonl": "dc1aaec2e22c74e806c8a6f2e5e4bd34fb85ccd782549caa912d93459c9875bd",
}


@pytest.mark.parametrize("name", sorted(SEALED))
def test_sealed_corpora_match_the_registered_digests(name: str) -> None:
    """PROTOCOL.md A1 registers these digests before any screening; an edit to a
    sealed corpus must fail here rather than quietly move every result."""

    raw = (REPO / "evals" / "team_bridge_moderation" / "screen" / "corpus" / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SEALED[name]


def test_sealed_corpora_load_as_a_plan() -> None:
    corpus = REPO / "evals" / "team_bridge_moderation" / "screen" / "corpus"
    items, digests = sb.load_corpus(
        [corpus / name for name in sorted(SEALED)], organization="Minerva"
    )
    assert len(items) == 131 and set(digests.values()) == set(SEALED.values())


def test_only_screening_and_the_eval_reference_the_private_renderer() -> None:
    """Review r1 n1: ``_render_with_template`` performs no trust check on its
    template, so a live caller would let unembedded bytes screen real traffic. The
    rule is that only ``screening.py`` (its owner) and ``local_operator/evaluation``
    may name it; this makes that rule fail closed instead of merely documented."""

    allowed_file = REPO / "local_operator" / "team_bridge" / "screening.py"
    allowed_dir = REPO / "local_operator" / "evaluation"
    offenders = [
        str(path.relative_to(REPO))
        for root in ("local_operator", "scripts")
        for path in (REPO / root).rglob("*.py")
        if path != allowed_file
        and allowed_dir not in path.parents
        and "_render_with_template" in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("reason", "token"),
    [
        ("refusal", "model_refused:refusal"),
        ("length", "answered:reply_truncated:length"),
        ("tool_use", "answered:parse_failed:tool_use"),
        ("error", "provider:other:stream_error"),
        ("aborted", "provider:other:stream_aborted"),
        ("missing_end", "provider:other:stream_missing_end"),
    ],
)
def test_only_a_model_refusal_is_classed_model_refused(reason: str, token: str) -> None:
    """Review r2 m1: production's ``answered_hold_code`` decides, so a provider
    ``error``/``aborted`` stop can never inflate the refused-as-held figure."""

    from local_operator.evaluation.team_bridge_moderation.gates import _provider_failure
    from local_operator.team_bridge.model_boundary import _ScreeningStreamRefused

    assert _provider_failure(_ScreeningStreamRefused(reason)) == token


def test_a_real_provider_error_402_is_billing() -> None:
    """Review r2 n2: pin the attribute the classifier reads on the real class."""

    from local_operator.evaluation.team_bridge_moderation.gates import _provider_failure
    from local_operator.providers.failover import ProviderError

    assert _provider_failure(ProviderError(402, "Insufficient Balance")) == ("provider:billing")


# -- the committed 2026-10-31 results ---------------------------------------------

RESULTS = REPO / "evals" / "team_bridge_moderation" / "screen" / "results" / "2026-10-31"
CANDIDATES = TEMPLATES / "candidates"
REJECTED = {
    "patch-c.txt": "d5844d75f3c30ae0387528d16e2cf6cfdf1430694f49d430c2cb09be366a41f9",
    "variant-d.txt": "37c37416936c2aa9d6bd6d83252bff164d810adc8ef4c708e92b66c3e2664681",
}


@pytest.mark.parametrize("name", sorted(REJECTED))
def test_rejected_candidates_are_the_measured_bytes_and_not_embedded(name: str) -> None:
    from local_operator.team_bridge.client import SCREENING_TEMPLATES

    raw = (CANDIDATES / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REJECTED[name]
    # Kept as a record only: a rejected candidate must never become served bytes.
    assert raw.decode("utf-8") not in SCREENING_TEMPLATES.values()


def test_committed_results_load_and_name_only_committed_templates() -> None:
    """The record is only a record if it can be re-summarised and every trial points
    at bytes in this tree: the embedded v2 file or a committed candidate."""

    committed = {V2_SHA, *REJECTED.values()}
    runs = sorted(RESULTS.glob("*/trials.jsonl"))
    assert len(runs) == 12
    for path in runs:
        trials = sb.load_trials([path])
        assert trials, path
        assert {t.template_sha256 for t in trials} <= committed, path
        assert sb.summarize(trials)["groups"], path
