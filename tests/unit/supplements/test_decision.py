"""The decision: the two questions' shapes, the featured-set rule, the overrules, fail-open,
and the egress scrub every vendor-bound string passes (memo §4; round-1 security S-R8)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.classification.types import Answer, Question
from local_operator.classification.vendors import questions_payload
from local_operator.redaction_shapes import REDACTION_MARKER
from local_operator.supplements import decision as dec
from local_operator.supplements.candidates import Candidate, prefilter
from local_operator.supplements.evidence import Dataset, Evidence
from tests.unit.supplements.support import call

from .conftest import FIXTURES


def _cand(
    name: str,
    *,
    tier: int = 1,
    kind: str = "markdown",
    order: int = 1,
    tool: str = "write",
    intent: str = "",
) -> Candidate:
    return Candidate(
        path=f"secret-client-dir/{name}",
        absolute=f"/Users/op/secret-client-dir/{name}",
        name=name,
        kind=kind,
        size_bytes=2048,
        mtime=1.0,
        tier=tier,
        tool=tool,
        order=order,
        intent=intent,
    )


STRUCTURED = Evidence(
    structured=True,
    datasets=(Dataset("Latency", "answer", ("r", "ms"), (("a", "1"),) * 3, 3, ("ms",)),),
    forms=("table",),
)


class FakeService:
    """The shared classification service's ``decide`` surface, recording what it was asked."""

    vendor_name = "radient"

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.asked: list[tuple[str, Question]] = []

    async def decide(self, *, state: str, question: Question) -> Answer | None:
        self.asked.append((state, question))
        got = self.answers.get(question.id)
        if isinstance(got, BaseException):
            raise got
        return got


def _files_answer(choice: str, probs: dict[str, float]) -> Answer:
    return Answer(id=dec.FILES_QUESTION_ID, kind="choice", value=choice, probabilities=probs)


def _graphics_answer(p: float) -> Answer:
    return Answer(id=dec.GRAPHICS_QUESTION_ID, kind="noul", value=p)


async def _decide(service, candidates, evidence=Evidence(structured=False), **kw):
    kw.setdefault("want_files", True)
    kw.setdefault("want_graphics", True)
    return await dec.decide(
        service,
        user_text=kw.pop("user_text", "make me a report"),
        answer_text=kw.pop("answer_text", "Wrote it."),
        candidates=candidates,
        evidence=evidence,
        max_featured=kw.pop("max_featured", 4),
        **kw,
    )


# -- the question shapes ------------------------------------------------------------------


def test_the_files_question_is_a_choice_over_f1_to_fn_plus_none_with_no_directory() -> None:
    options = dec.option_ids(
        [_cand("report.md", intent="Writing the Q3 report"), _cand("data.csv", kind="csv")]
    )
    question = dec.files_question(options)
    assert (question.id, question.kind) == ("supplement_files", "choice")
    assert list(question.criteria) == ["f1", "f2", "none"]
    first = question.criteria["f1"]
    assert first == "report.md (markdown, 2.0 KB) written by write: Writing the Q3 report"
    # The directory is the field that would carry a client's name to the vendor (S-R8).
    assert all("secret-client-dir" not in text for text in question.criteria.values())


def test_the_files_question_offers_at_most_twelve() -> None:
    options = dec.option_ids([_cand(f"f{i}.md") for i in range(30)])
    assert list(options) == [f"f{i}" for i in range(1, 13)]


def test_the_graphics_question_is_a_noul_with_the_rubric_in_the_options() -> None:
    question = dec.graphics_question()
    assert (question.id, question.kind) == ("supplement_graphics", "noul")
    assert set(question.criteria) == {"true", "false"}
    assert "version" in question.instructions and "talked about" in question.instructions


def test_the_questions_match_the_c0_fixture_grammar() -> None:
    """C0 pinned the shapes the grammar accepts; building through ``Question`` re-checks that
    the mapping/array rules hold for the two questions (a score or malformed mapping raises)."""
    assert isinstance(dec.files_question(dec.option_ids([_cand("a.md")])), Question)
    assert isinstance(dec.graphics_question(), Question)
    sample = json.loads((FIXTURES / "rows" / "done_files_only.json").read_text())
    assert set(sample["payload"]["details"]["decision"]) == {
        "vendor",
        "files_p",
        "graphics_p",
        "skipped",
    }


def test_state_is_bounded_and_carries_shapes_never_rows() -> None:
    state = dec.graphics_state("u" * 10_000, "a" * 10_000, STRUCTURED)
    assert state.count("…[truncated]") == 2
    assert "Latency: 3 rows x 2 cols (r, ms)" in state
    assert "a" * 10 in state and "('a', '1')" not in state


# -- featured set ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chosen", "probs", "expected"),
    [
        ("f1", {"f1": 0.9, "f2": 0.05, "none": 0.05}, ["f1"]),
        ("f1", {"f1": 0.5, "f2": 0.4, "f3": 0.1}, ["f1", "f2"]),  # f2 >= floor and >= half of top
        ("f1", {"f1": 0.6, "f2": 0.2, "f3": 0.2}, ["f1"]),  # 0.2 under the 0.25 floor
        ("f1", {"f1": 0.9, "f2": 0.3}, ["f1"]),  # 0.3 >= floor but < half of 0.9
        ("none", {"none": 0.8, "f1": 0.2}, []),  # none wins with p >= 0.5
        ("f2", {"f2": 0.4, "f1": 0.35, "none": 0.25}, ["f2", "f1"]),  # chosen first
        ("f9", {"f9": 0.9, "f1": 0.1}, []),  # an id that was never offered cannot be invented
        ("f1", {}, ["f1"]),  # a vendor with no distribution still names its choice
    ],
)
def test_qualifying_ids_follow_the_memo_rule(chosen, probs, expected) -> None:
    assert dec.qualifying_ids(chosen, probs, offered=["f1", "f2", "f3"]) == expected


def test_none_winning_below_half_does_not_silence_a_strong_option() -> None:
    assert dec.qualifying_ids(
        "none", {"none": 0.4, "f1": 0.39, "f2": 0.21}, offered=["f1", "f2"]
    ) == ["f1"]


@pytest.mark.asyncio
async def test_a_vendor_pick_is_featured_and_the_qualified_overflow_becomes_more() -> None:
    files = [_cand(f"r{i}.md", order=i) for i in range(1, 7)]
    probs = {f"f{i}": 0.3 for i in range(1, 7)}
    service = FakeService({"supplement_files": _files_answer("f1", probs)})
    out = await _decide(service, files, max_featured=4)
    assert [c.name for c in out.featured] == ["r1.md", "r2.md", "r3.md", "r4.md"]
    assert [c.name for c in out.more] == ["r5.md", "r6.md"]
    assert out.vendor == "radient"
    assert out.files_p["secret-client-dir/r1.md"] == 0.3


@pytest.mark.asyncio
async def test_files_the_vendor_rejected_are_not_more() -> None:
    """'N more' means qualified-but-capped. Rejected files in the disclosure would be spam."""
    files = [_cand("keep.md"), _cand("junk1.md"), _cand("junk2.md")]
    service = FakeService(
        {"supplement_files": _files_answer("f1", {"f1": 0.9, "f2": 0.05, "f3": 0.05})}
    )
    out = await _decide(service, files)
    assert [c.name for c in out.featured] == ["keep.md"] and out.more == ()


# -- graphics -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graphics_need_the_threshold_and_the_structured_signal() -> None:
    service = FakeService({"supplement_graphics": _graphics_answer(0.83)})
    assert (await _decide(service, [], STRUCTURED)).graphics is True
    service = FakeService({"supplement_graphics": _graphics_answer(0.69)})
    assert (await _decide(service, [], STRUCTURED)).graphics is False
    service = FakeService({"supplement_graphics": _graphics_answer(0.7)})
    assert (await _decide(service, [], STRUCTURED)).graphics is True, "the threshold is inclusive"


@pytest.mark.asyncio
async def test_a_model_yes_without_evidence_is_never_asked_let_alone_obeyed() -> None:
    service = FakeService({"supplement_graphics": _graphics_answer(0.99)})
    out = await _decide(service, [], Evidence(structured=False))
    assert out.graphics is False
    assert service.asked == [], "no structured data -> no paid call at all"
    # The other half of the same rule, driven directly: were a "yes" to arrive anyway (an
    # answer cached before the signal moved, a hand-built service), the signal still wins.
    p, go = dec.apply_graphics(_graphics_answer(0.99), Evidence(structured=False))
    assert (p, go) == (0.99, False), "a 'yes' without evidence is overruled, not merely unasked"
    p, go = dec.apply_graphics(_graphics_answer(0.99), STRUCTURED)
    assert go is True


@pytest.mark.asyncio
async def test_graphics_are_not_asked_without_a_generator() -> None:
    service = FakeService({"supplement_graphics": _graphics_answer(0.99)})
    out = await _decide(service, [], STRUCTURED, want_graphics=False)
    assert out.graphics is False and service.asked == []


@pytest.mark.asyncio
async def test_both_questions_ride_one_gather_and_a_turn_with_neither_asks_nothing() -> None:
    service = FakeService(
        {
            "supplement_files": _files_answer("f1", {"f1": 1.0}),
            "supplement_graphics": _graphics_answer(0.9),
        }
    )
    out = await _decide(service, [_cand("a.md")], STRUCTURED)
    assert {q.id for _s, q in service.asked} == {"supplement_files", "supplement_graphics"}
    assert out.graphics and len(out.featured) == 1
    quiet = FakeService({})
    await _decide(quiet, [])
    assert quiet.asked == []


# -- fail-open and the no-vendor heuristic -----------------------------------------------------


@pytest.mark.parametrize(
    "service", [None, FakeService({}), FakeService({"supplement_files": RuntimeError("boom")})]
)
@pytest.mark.asyncio
async def test_no_answer_degrades_to_the_heuristic_never_an_error(service) -> None:
    files = [
        _cand("old.md", order=1),
        _cand("new.csv", kind="csv", order=5),
        _cand("tool.py", kind="code", order=9),  # source: never a deliverable
        _cand("named.pdf", tier=3, kind="pdf", order=99),  # prose-only: not written this turn
        _cand("made.csv", tier=2, kind="csv", order=7),  # a shell flag is not "written by write"
    ]
    out = await _decide(service, files, STRUCTURED)
    assert [c.name for c in out.featured] == [
        "new.csv",
        "old.md",
    ], "tier-1 deliverables, newest first"
    assert out.graphics is False, "graphics never run without a vendor"
    assert (
        out.vendor == dec.HEURISTIC_VENDOR or service is None and out.vendor == dec.HEURISTIC_VENDOR
    )


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed() -> None:
    service = FakeService({"supplement_files": asyncio.CancelledError()})
    with pytest.raises(asyncio.CancelledError):
        await _decide(service, [_cand("a.md")])


@pytest.mark.asyncio
async def test_a_declined_vendor_with_nothing_featured_is_an_empty_decision() -> None:
    service = FakeService({"supplement_files": _files_answer("none", {"none": 0.9, "f1": 0.1})})
    out = await _decide(service, [_cand("scratch.md")])
    assert out.empty and out.more == ()


# -- the egress boundary (memo §4's egress statement; round-1 security S-R8) ---------------

#: The fixture's credential-shaped string: an issuer-prefixed token, the spelling the shape
#: table masks. Built from a fixed tail, so nothing here can resemble a live value.
LEAK_TOKEN = "glpat-" + "AbCdEf12GhIjKl34MnOpQr56"


def _vendor_payload(asked: list[tuple[str, Question]]) -> str:
    """Everything a vendor receives for these asks, as text.

    The state is joined verbatim -- the classification layer adds no scrub of its own, which
    is what R6 measured -- and the questions go through the same serializer the vendor body
    uses (``classification.vendors.questions_payload``). Asserting on this string is
    asserting on the request body's data.
    """
    return "\n".join(
        state + "\n" + json.dumps(questions_payload([question]), sort_keys=True)
        for state, question in asked
    )


@pytest.mark.asyncio
async def test_the_vendor_payload_is_scrubbed_before_the_call(tmp_path: Path) -> None:
    """A real turn through the pre-filter and the decision: what the vendor would receive
    carries none of the four leak classes -- an absolute path (the tmp root stands in for
    home), a client directory, a credential shape, an email-valued credential -- while the
    benign base name still arrives (docs review R6 on #2155; security S-R8)."""
    target = tmp_path / "clients" / "acme-corp" / "q3-summary.pdf"
    target.parent.mkdir(parents=True)
    target.write_text("the Q3 numbers\n" + "x" * 64)
    items = [call("write", path=str(target), i="Writing it to ~/clients/acme-corp/q3-summary.pdf")]
    answer_text = (
        f"Exported the Q3 report to {target}. "
        f"Rotated the deploy token {LEAK_TOKEN} yesterday. "
        "SMTP_PASSWORD=ops@example.com is now in the vault."
    )
    pre = prefilter(items, answer_text, cwd=str(tmp_path), home=str(tmp_path), want_graphics=False)
    assert pre.skipped is None and len(pre.candidates) == 1
    service = FakeService({"supplement_files": _files_answer("f1", {"f1": 1.0})})
    decision = await dec.decide(
        service,
        user_text="export the Q3 numbers",
        answer_text=answer_text,
        candidates=pre.candidates,
        evidence=pre.evidence,
        want_files=True,
        want_graphics=False,
        max_featured=4,
    )
    assert [c.name for c in decision.featured] == ["q3-summary.pdf"]
    payload = _vendor_payload(service.asked)
    for leak in (str(tmp_path), "acme-corp", LEAK_TOKEN, "ops@example.com"):
        assert leak not in payload, f"{leak!r} reached the vendor payload"
    assert "q3-summary.pdf" in payload, "the base name is what the decision judges"
    assert "Writing it to q3-summary.pdf" in payload, "the intent line is reduced too"
    assert REDACTION_MARKER in payload, "the credential was masked, not merely missing"


def test_option_text_reduces_a_path_in_the_intent_line() -> None:
    canned = _cand("q3-summary.pdf", intent="Writing it to /Users/damian/clients/acme-corp/q3.pdf")
    text = dec.option_text(canned)
    assert "/Users" not in text and "acme-corp" not in text
    assert text.endswith("Writing it to q3.pdf")


def test_both_states_reduce_file_uris_and_tilde_paths() -> None:
    state = dec.files_state(
        "see file:///Users/damian/work/notes.md", "and ~/clients/acme-corp/final.csv"
    )
    assert "notes.md" in state and "final.csv" in state
    assert "/Users" not in state and "acme-corp" not in state


def test_the_graphics_state_scrubs_the_dataset_titles_it_carries() -> None:
    dataset = Dataset(
        "exports to /Users/damian/clients/acme-corp/q3.csv",
        "answer",
        ("region", "ms"),
        (("a", "1"), ("b", "2"), ("c", "3")),
        3,
        ("ms",),
    )
    state = dec.graphics_state(
        "u", "a", Evidence(structured=True, datasets=(dataset,), forms=("table",))
    )
    assert "acme-corp" not in state and "/Users" not in state
    assert "q3.csv" in state


def test_benign_slashes_urls_and_addresses_survive_byte_identical() -> None:
    """The over-masking arm: ``and/or``, ``24/7`` and a URL's path must not be taken by
    the reduction, and a bare address in prose is not a credential -- the shape corpus pins
    the same negative (``https://user@example.com/profile`` must survive). An address in a
    credential POSITION is masked, and the payload test above covers that spelling."""
    benign = "ratio 24/7 and/or km/h; see https://example.com/reports/q3.html; ping ops@example.com"
    assert benign in dec.files_state(benign, benign)
