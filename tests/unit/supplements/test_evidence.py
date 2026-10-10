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
