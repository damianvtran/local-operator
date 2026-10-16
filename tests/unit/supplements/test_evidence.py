"""The structured-data signal (memo §2.3): the three forms, and the shapes that must NOT fire."""

from __future__ import annotations

import pytest

from local_operator.supplements import evidence as ev
from tests.unit.supplements.support import result

TABLE = (
    "| region | p50 | p99 |\n|---|---|---|\n| us | 12 | 40 |\n| eu | 15 | 55 |\n| ap | 22 | 90 |\n"
)


def test_a_markdown_table_with_a_numeric_column_is_form_a() -> None:
    found = ev.extract([], f"Latency by region:\n\n{TABLE}")
    assert found.structured and found.forms == ("table",)
    assert found.datasets[0].shape() == "3 rows x 3 cols (region, p50, p99)"
    assert found.datasets[0].title == "Latency by region"


def test_a_tool_result_table_counts_not_just_the_answer() -> None:
    found = ev.extract([result("a\tb\n1\t2\n3\t4\n5\t6\n", tool="bash")], "Done.")
    assert found.structured and found.forms == ("delimited",)


def test_csv_in_tool_output_is_form_a() -> None:
    found = ev.extract([result("name,ms\nx,1.5\ny,2.5\nz,3.5\n")], "ok")
    assert found.structured


def test_a_json_array_of_objects_with_a_shared_numeric_key_is_form_b() -> None:
    body = '[{"r": "us", "ms": 12}, {"r": "eu", "ms": 15}, {"r": "ap", "ms": 22}]'
    assert ev.extract([result(body)], "ok").forms == ("json",)
    fenced = f"Here:\n```json\n{body}\n```\n"
    assert ev.extract([], fenced).forms == ("json",)
    wrapped = '{"results": ' + body + "}"
    assert ev.extract([result(wrapped)], "ok").structured


def test_four_numbers_sharing_a_unit_in_one_paragraph_is_form_c() -> None:
    found = ev.extract([], "Runs took 12 ms, 15 ms, 9 ms and 22 ms respectively.")
    assert found.structured and found.forms == ("units",)


@pytest.mark.parametrize(
    "text",
    [
        "Paris.",
        "Merged #123 into main.",
        "Upgraded to v1.2.3 on 2026-10-09.",
        "It improved across 3 regions.",
        "Latency was 12 ms and 15 ms.",  # two values: no figure helps
        "Runs took 12 ms, 15 s, 9 MB and 22 GB.",  # four numbers, four different units
        "Step 1, step 2, step 3, step 4 and step 5.",  # ordinals are not measurements
        "| name | note |\n|---|---|\n| a | x |\n| b | y |\n| c | z |\n",  # no numeric column
        "| n | name |\n|---|---|\n| 1 | a |\n| 2 | b |\n| 3 | c |\n",  # a row-number index
        "| a | b |\n|---|---|\n| x | 1 |\n| y | 2 |\n",  # only two data rows
    ],
)
def test_prose_and_near_misses_do_not_fire(text: str) -> None:
    assert not ev.extract([], text).structured, text


def test_numbers_split_across_paragraphs_do_not_pool() -> None:
    text = "Took 12 ms and 15 ms.\n\nThen 9 ms and 22 ms."
    assert not ev.extract([], text).structured


def test_an_errored_tool_result_is_not_data() -> None:
    assert not ev.extract([result("a,b\n1,2\n3,4\n5,6\n", error=True)], "failed").structured


def test_the_scan_is_bounded() -> None:
    huge = "x,y\n" + "1,2\n" * 5_000_000
    found = ev.extract([result(huge[: ev.SCAN_BUDGET_CHARS * 4])], "ok")
    assert found.structured
    assert len(found.datasets[0].rows) <= ev.MAX_ROWS_KEPT
    assert found.datasets[0].n_rows <= ev.SCAN_BUDGET_CHARS


def test_dataset_titles_are_sanitised_single_lines() -> None:
    text = "\x1b[31m## Evil\x1b[0m\u202etitle\n" + TABLE
    title = ev.extract([], text).datasets[0].title
    assert "\x1b" not in title and "\n" not in title and "\u202e" not in title


def test_units_only_signal_is_not_drawable_without_datasets() -> None:
    """Regression: 'units' alone (no dataset) used to start a fork whose every block was
    rejected as 'not in the evidence' -> a failed row -> "Couldn't prepare highlights"."""
    from local_operator.supplements.decision import apply_graphics, drawable

    found = ev.extract([], "Runs took 12 ms, 15 ms, 9 ms and 22 ms respectively.")
    assert found.structured and not found.datasets
    assert not drawable(found)

    class Yes:
        value = 0.95

    assert apply_graphics(Yes(), found) == (0.95, False)
    table = ev.extract([], "| r | ms |\n|---|---|\n| a | 12 |\n| b | 40 |\n| c | 7 |\n")
    assert drawable(table) and apply_graphics(Yes(), table) == (0.95, True)


@pytest.mark.parametrize(
    "answer",
    [
        "No changes were needed.",
        "- Added 296 comments.",
        "There are 296 comments, 225 profiles, 325 links and 3194 matches.",
        "- Count 1.2.3; 2026-10-14; version 325; ticket 3194.",
    ],
)
def test_labelled_summary_does_not_admit_unstructured_or_single_counts(answer):
    from local_operator.supplements.evidence import extract

    assert not extract([], answer).structured


def test_labelled_summary_generator_payload_requires_table_not_comparable_axes():
    from local_operator.supplements.evidence import extract
    from local_operator.supplements.generator import evidence_block

    answer = "- Added 296 comments to 225 profiles, covering 325 links and 3194 matches."
    evidence = extract([], answer)
    assert evidence.structured and evidence.datasets[0].presentation == "table-only"
    payload = evidence_block(evidence.datasets, user_text="summarise", answer_text=answer)
    assert "table-only" in payload and "distinct entity units" in payload
    assert all(value in payload for value in ("296", "225", "325", "3194"))


@pytest.mark.parametrize(
    "answer",
    [
        "- Delta: -10 comments, -20 profiles, -30 links, -40 matches.",
        "- Delta: +10 comments, +20 profiles, +30 links, +40 matches.",
        "- Delta: −10 comments, −20 profiles, −30 links, −40 matches.",
        "- Forecast: 10-20 comments, 30-40 profiles, 50-60 links and 70-80 matches.",
        "- Forecast: 10 - 20 comments, 30 – 40 profiles, 50 — 60 links, 70 to 80 matches.",
        "- Dates: 2026-10-14 comments, 2026-10-15 profiles, 2026-10-16 links, 2026-10-17 matches.",
        "- IDs: case-296 comments, case-225 profiles, case-325 links and case-3194 matches.",
        (
            "- Ticket 101 covers comments; ticket 202 covers profiles; ticket "
            "303 covers links; ticket 404 covers matches."
        ),
        (
            "- Ticket 101 Covers comments; ticket 202 Covers profiles; ticket "
            "303 Covers links; ticket 404 Covers matches."
        ),
        "- Delta: - 10 comments, - 20 profiles, - 30 links and - 40 matches.",
        "Historical 296 comments, 225 profiles, 325 links, 3194 matches.\n- No counts here.",
        "- Earlier 100 comments, 225 profiles, 325 links, 3194 matches.\n- Now 296 comments.",
    ],
)
def test_labelled_summary_rejects_ambiguous_numeric_semantics(answer):
    from local_operator.supplements.decision import drawable

    found = ev.extract([], answer)
    assert ev._labelled_summary(answer) is None
    assert found.datasets == ()
    # A pre-existing unit signal can fire on repeated 'covers'; no dataset means
    # no paid graphics call. Do not conflate the signal with authoritative rows.
    assert not drawable(found)


@pytest.mark.parametrize("qualifier", ["supplementary", "SUPPLEMENTARY", "ACME Production"])
def test_labelled_summary_keeps_explicit_vocabulary_and_real_uppercase_acronyms(qualifier):
    answer = (
        f"- Added 296 {qualifier} comments to 225 profiles, covering 325 links and 3,194 matches."
    )
    found = ev._labelled_summary(answer)
    assert found is not None
    assert {r[0]: r[1] for r in found.rows} == {
        "comments": "296",
        "profiles": "225",
        "links": "325",
        "matches": "3194",
    }


@pytest.mark.parametrize(
    "qualifier",
    [
        "between 10 and",
        "up to",
        "at least",
        "at most",
        "about",
        "roughly",
        "approximately",
        "around",
        "more than",
        "less than",
        "fewer than",
        "greater than",
        "nearly",
        "almost",
        "estimated",
        "from 10 to",
    ],
)
def test_summary_refuses_entire_qualified_count_bullet(qualifier):
    # Even with four OTHER exact counts, a qualified count bullet must not be
    # silently collapsed into the authoritative point-count table.
    answer = (
        "- Added 296 comments to 225 profiles, covering 325 links and 3194 matches.\n"
        f"- Expected {qualifier} 20 comments."
    )
    assert ev._labelled_summary(answer) is None
    assert not ev.extract([], answer).datasets


def test_between_ranges_do_not_become_upper_endpoint_counts():
    answer = (
        "- Forecast: between 10 and 20 comments, between 30 and 40 profiles, "
        "between 50 and 60 links, between 70 and 80 matches."
    )
    assert ev._labelled_summary(answer) is None
    assert not ev.extract([], answer).datasets
