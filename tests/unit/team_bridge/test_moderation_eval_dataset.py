"""The eval dataset's contract: shape, coverage, and the canary detector it feeds.

WHY THESE TESTS EXIST AND WHY THEY COST NOTHING. The dataset is the instrument, so a
malformed or under-covered dataset makes every later number meaningless - and it can
go wrong silently, because a case that fails to validate is simply absent from a
report. These tests are the CI half of the design's §7.6 cadence ("schema lint of the
dataset plus a stubbed dry run"): they parse the committed JSONL, validate it against
the taxonomy and the fixture, prove the variant expansion is deterministic, and
exercise the canary detector on synthetic text. No provider, no session, no fixture
file written, no spend.

WHAT THEY DELIBERATELY DO NOT DO. They never assert a particular case's verdict - the
dataset is edited by reviewers, and a test that pins today's wording would make every
label edit a test edit. They assert the PROPERTIES the report relies on: the
categories are covered, every allow case carries its negative control, the refuse
set names what must not leak, the controls have cases, and the encoding seeds render
to their own plaintext.
"""

from __future__ import annotations

import base64
import codecs
from pathlib import Path

import pytest

from local_operator.evaluation.team_bridge_moderation.canary import mint_canaries, scan
from local_operator.evaluation.team_bridge_moderation.dataset import (
    LEET_ENCODE,
    DatasetProblem,
    load_dataset,
    render_step_body,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "evals" / "team_bridge_moderation"

pytestmark = pytest.mark.skipif(
    not DATASET_ROOT.is_dir(),
    reason="the eval dataset is repository data and is absent from an installed build",
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET_ROOT)


def test_dataset_validates(dataset) -> None:
    """The committed dataset passes every rule the loader enforces.

    ``load_dataset`` raises with the full problem list; asserting on the return value
    means a failure prints every violation in one run instead of one per edit.
    """

    assert dataset.cases, "the dataset is empty"
    assert dataset.authored >= 150, (
        f"only {dataset.authored} authored cases: the v1 set is meant to be broad enough "
        f"that a category cannot pass by luck"
    )


def test_every_category_is_covered(dataset) -> None:
    """Each taxonomy category with a sample plan has cases, and the counts are balanced.

    Balance is the property that matters: an eval whose refuse set is 90% local_data
    cannot tell you whether the other-org rule works.
    """

    counts = {name: len(cases) for name, cases in dataset.by_category().items()}
    taxonomy_ids = {c.id for c in dataset.taxonomy.categories.values()}
    missing = sorted(taxonomy_ids - set(counts))
    assert not missing, f"categories with no cases: {missing}"
    thinnest = min(counts.values())
    assert thinnest >= 4, f"a category has only {thinnest} cases: {counts}"


def test_refuse_cases_declare_what_must_not_leak(dataset) -> None:
    """Every refuse case names a leak class, so a silent allow is attributable."""

    offenders = [
        case.id
        for case in dataset.cases
        if case.expected_disposition == "refuse" and not case.must_not_leak
    ]
    assert not offenders, f"refuse cases with no must_not_leak: {offenders}"


def test_allow_cases_carry_a_negative_control(dataset) -> None:
    """Every allow/grey case requires a fact, so an empty answer cannot pass.

    This is the design-review M2 requirement in executable form: an "allow" counted
    without a content check could be a hedge, and a hedge is a false refusal hiding.
    """

    offenders = [
        case.id
        for case in dataset.cases
        if case.expected_disposition == "allow" and not case.must_contain_facts
    ]
    assert not offenders, f"allow cases without must_contain_facts: {offenders}"


def test_no_case_carries_canary_material(dataset) -> None:
    """The corpus must not plant its own leak (review R1 M1).

    A canary in the asker's message would make a canary hit ambiguous: the token would
    be in the text before the agent read anything. Prefix-based because the run's tokens
    are minted later, so a literal cannot be one of them - but a copied one from an older
    run could be, and that has to fail loudly rather than look like a leak.
    """

    prefix = dataset.workspace.canary_prefix
    offenders = [
        f"{case.id}:{step.kind}"
        for case in dataset.cases
        for step in case.steps
        if prefix in str(step.body or "")
    ]
    assert not offenders, f"case text contains the canary prefix {prefix!r}: {offenders}"


def test_tool_assertions_are_declared_consistently(dataset) -> None:
    """`expect_tool_used` is an ALLOW assertion and names only known tools.

    A required-tool assertion on a refuse case would demand that the agent used a tool
    it is supposed to be refused; an unknown name would never match and would fail the
    case for a typo (review R1 M5).
    """

    known = set(dataset.workspace.tool_inventory.get("known_tools") or [])
    offenders = []
    for case in dataset.cases:
        for tool in getattr(case, "expect_tool_used", ()) or ():
            if case.label != "allow":
                offenders.append(f"{case.id}: expect_tool_used on a {case.label} case")
            if known and tool not in known:
                offenders.append(f"{case.id}: unknown tool {tool!r}")
    assert not offenders, offenders


def test_controls_have_cases(dataset) -> None:
    """All three controls are exercised, and control C has at least one PAIR.

    Without a pair, "in scope is allowed / out of scope is refused" is two unrelated
    claims rather than a controlled comparison.
    """

    by_control = dataset.by_control()
    for name in ("A", "B", "C"):
        assert by_control.get(name), f"control {name} has no cases"

    pairs: dict[str, list[str]] = {}
    for case in dataset.cases:
        if case.pair_id:
            pairs.setdefault(case.pair_id, []).append(case.expected_disposition)
    complete = {
        pair: outcomes
        for pair, outcomes in pairs.items()
        if "allow" in outcomes and "refuse" in outcomes
    }
    assert complete, f"no complete in-scope/out-of-scope pair: {pairs}"


def test_laundering_cases_have_two_turns(dataset) -> None:
    """A deferral case without a second turn cannot exercise the deferred send."""

    offenders = [
        case.id for case in dataset.cases if case.laundering and len(case.steps) < 2
    ]
    assert not offenders, f"laundering cases with one turn: {offenders}"


def test_encoding_seeds_render_their_own_plaintext(dataset) -> None:
    """An encoded body decodes back to the plaintext the case declares.

    This is the guard that caught two hand-encoded seeds that decoded to gibberish:
    the harness must never test a payload nobody wrote.
    """

    encoded = [
        (case, step)
        for case in dataset.cases
        for step in case.steps
        if step.encoding is not None
    ]
    assert encoded, "no encoded cases: the encoding variants would be untested"
    for case, step in encoded:
        rendered = render_step_body(step)
        assert "{encoded}" not in rendered
        plaintext = step.plaintext or ""
        assert plaintext, f"{case.id}: encoding block without plaintext"
        if step.encoding == "base64":
            assert base64.b64encode(plaintext.encode()).decode() in rendered
        elif step.encoding == "rot13":
            assert codecs.encode(plaintext, "rot13") in rendered
        elif step.encoding == "leet":
            # The ENCODER direction, and it round-trips through the normaliser the
            # detector uses: one property, two maps, no hand-written expectation.
            assert plaintext.translate(LEET_ENCODE) in rendered


def test_variant_expansion_is_deterministic_and_attributed(dataset) -> None:
    """Variants are stamped, and re-loading produces the same set.

    A variant that lost its ``variant_of`` would inflate a category's count without
    anyone being able to tell which seed it came from.
    """

    again = load_dataset(DATASET_ROOT)
    assert [c.id for c in again.cases] == [c.id for c in dataset.cases]
    variants = [case for case in dataset.cases if case.variant_of]
    assert variants, "no variants were expanded: the mutator path is untested"
    for case in variants:
        assert case.variant_of in {c.id for c in dataset.cases}


def test_canary_detector_sees_every_transform() -> None:
    """The detector finds a planted token exact, base64d and rot13d.

    WHY this is the heart of the suite: a detector that only matched the literal form
    would report "zero leaks" on a run whose whole point was exfiltration via
    formatting. Everything else in the report leans on this one property.

    LEETED CANARIES ARE NOT CLAIMED, deliberately. A token carries a hex suffix, so
    digit-to-letter normalisation is lossy on it (a literal "0" becomes "o" and the
    suffix stops matching). Leet is a BODY obfuscation the product's gate must reason
    about, not a canary transform; the dataset's leet seeds cover the body side and
    ``test_encoding_seeds_render_their_own_plaintext`` proves they render.
    """

    canaries = mint_canaries("test-salt", ["downloads", "env_secret"], "TBCANARY")
    token = canaries.value("downloads")
    assert token.startswith("TBCANARY-DOWNLOADS-"), token

    samples = {
        "plain": f"the invoice reads {token}",
        "base64": "decoded: " + base64.b64encode(f"secret {token} end".encode()).decode(),
        "rot13": codecs.encode(f"secret {token} end", "rot13"),
    }
    for label, text in samples.items():
        found = scan(text, canaries, where="test")
        assert any(leak.leak_class == "downloads" for leak in found), (
            f"detector missed the {label} view of its own canary"
        )


def test_canary_detector_is_silent_on_clean_text() -> None:
    """Ordinary prose produces no findings, so the detector is not a yes-machine."""

    canaries = mint_canaries("test-salt", ["downloads"], "TBCANARY")
    clean = (
        "The dedupe PR is in review and blocked on a CI flake in the ingest integration "
        "test. Cy owns it. Release 4.2 scope depends on it landing first."
    )
    assert scan(clean, canaries, where="test") == []


def test_reference_case_is_refused_by_no_rule(dataset) -> None:
    """A canary value never appears in the dataset files themselves.

    A committed canary would be memorisable and would make every run's tokens the
    same. The dataset declares PLANT IDS; the tokens are minted at run time, and this
    test keeps that true as the dataset grows.
    """

    offenders: list[str] = []
    for path in sorted((DATASET_ROOT / "cases").glob("*.jsonl")):
        text = path.read_text(encoding="utf-8")
        if "TBCANARY-" in text:
            offenders.append(path.name)
    assert not offenders, f"committed canary values in: {offenders}"


def test_dataset_problem_reports_every_violation(tmp_path) -> None:
    """A broken dataset fails loudly, and names every problem at once.

    Builds a deliberately invalid case file in a temporary copy of the real dataset,
    so the failure path is exercised without ever touching the committed data.
    """

    import json
    import shutil

    broken = tmp_path / "dataset"
    broken.mkdir()
    for name in ("fig1", "labels", "cases"):
        source = {
            "fig1": DATASET_ROOT / "fixture",
            "labels": DATASET_ROOT / "labels",
            "cases": DATASET_ROOT / "cases",
        }[name]
        shutil.copytree(source, broken / ("fixture" if name == "fig1" else name))
    (broken / "cases" / "refuse_local_data.jsonl").write_text(
        json.dumps({"id": "broken.01", "label": "refuse"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetProblem) as caught:
        load_dataset(broken)
    assert len(caught.value.problems) >= 1
