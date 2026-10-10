"""The honesty validator and its Python mirror of ``LO.fmt`` (memo §2.10, §4.2, App. A).

THE MUTATION PROOF LIVES HERE. Two tests hand the validator a component that lies and require
it to be REFUSED -- one fabricated series (a number no evidence block holds) and one
extrapolated trend (a fourth point the evidence never measured). They are the tests that fail
if the provenance check is ever loosened to "well-formed and plausible".

The ``LO.fmt`` mirror is pinned against the RUNNING prelude under node (``prelude_harness.mjs``
``{call: …}`` steps), on shared vectors, for the reason the mirror exists at all: a Python
copy of a JS formatting rule is exactly the kind of thing that is right until a tie rounds the
other way. Without node the mirror test skips locally and FAILS on CI -- see
``test_prelude.py``'s ``node_available`` for the reason a skip is never silently green.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator.supplements import fmt as fmt_mod
from local_operator.supplements.evidence import Dataset
from local_operator.supplements.prompt import (
    GENERATOR_SYSTEM_PROMPT,
    MAX_INSTRUCTION_CHARS,
    repair_message,
    steer_instruction,
)
from local_operator.supplements.validate import (
    MAX_COMPONENT_BYTES,
    _WellFormed,
    validate_output,
)

PRELUDE_DIR = Path(__file__).resolve().parents[3] / "local_operator/supplements/prelude"
HARNESS = Path(__file__).resolve().parent / "prelude_harness.mjs"
REPO_ROOT = Path(__file__).resolve().parents[3]
MEMO = REPO_ROOT / "docs/design/turn-supplements.md"

#: The evidence one turn's pre-filter would have produced: a small latency table.
EVIDENCE = (
    Dataset(
        title="Latency by region",
        source="bench.csv",
        columns=("region", "ms"),
        rows=(("us-east", "120"), ("us-west", "98.5"), ("eu-west", "143"), ("ap-south", "211.25")),
        n_rows=4,
        numeric_columns=("ms",),
    ),
)


def component(data: str, body: str, *, title: str = "Latency", source: str = "bench.csv") -> str:
    """One fork answer, in App. A's own grammar."""
    return (
        f'<component title="{title}" source="{source}">\n'
        f"<data>{data}</data>\n"
        f"<html>{body}</html>\n"
        "</component>"
    )


HONEST = component(
    json.dumps(
        {
            "lat": {
                "title": "Latency by region (ms)",
                "columns": ["region", "ms"],
                "rows": [
                    ["us-east", 120],
                    ["us-west", 98.5],
                    ["eu-west", 143],
                    ["ap-south", 211.25],
                ],
            }
        }
    ),
    '<div id="c"></div><script>LO.bar(document.getElementById("c"),"lat",'
    '{x:"region",y:"ms",unit:"ms"})</script>',
)


# --- the honest path ---------------------------------------------------------------------


def test_an_honest_component_is_accepted_and_stored_as_a_blob() -> None:
    result = validate_output(HONEST, EVIDENCE)
    assert not result.rejected
    (accepted,) = result.components
    assert (accepted.title, accepted.source) == ("Latency", "bench.csv")
    # The stored blob is the leading <data> block(s) then the body: what
    # ``document.split_component`` reads back, and what the digest covers.
    assert accepted.blob.startswith("<data>")
    assert "LO.bar(" in accepted.blob
    assert accepted.height_hint == 320


def test_NONE_is_a_valid_answer_with_no_components() -> None:
    result = validate_output("NONE", EVIDENCE)
    assert result.none and not result.components and not result.rejected


def test_a_component_without_a_source_is_refused() -> None:
    result = validate_output(HONEST.replace(' source="bench.csv"', ""), EVIDENCE)
    assert result.rejected and any("source" in error for error in result.repair_errors)


# --- the mutation proofs -----------------------------------------------------------------


def test_a_fabricated_series_is_refused() -> None:
    """MUTATION: an invented column. Nothing in the evidence holds 7,000 ms."""
    lying = component(
        json.dumps(
            {
                "lat": {
                    "title": "Latency",
                    "columns": ["region", "ms"],
                    "rows": [["us-east", 120], ["us-west", 7000], ["eu-west", 143]],
                }
            }
        ),
        '<div id="c"></div><script>LO.bar(document.getElementById("c"),"lat",'
        '{x:"region",y:"ms",unit:"ms"})</script>',
    )
    result = validate_output(lying, EVIDENCE)
    assert not result.components, "a fabricated value reached the accepted set"
    assert any("evidence" in error for error in result.rejected[0].errors)


def test_an_extrapolated_trend_is_refused() -> None:
    """MUTATION: a forecast point. The evidence measured four regions, not a fifth quarter."""
    extrapolated = component(
        json.dumps(
            {
                "lat": {
                    "title": "Latency",
                    "columns": ["region", "ms"],
                    "rows": [
                        ["us-east", 120],
                        ["us-west", 98.5],
                        ["eu-west", 143],
                        ["ap-south", 211.25],
                        ["q5-forecast", 230.4],
                    ],
                }
            }
        ),
        '<div id="c"></div><script>LO.line(document.getElementById("c"),"lat",'
        '{x:"region",y:"ms",unit:"ms"})</script>',
    )
    result = validate_output(extrapolated, EVIDENCE)
    assert not result.components, "an extrapolated point reached the accepted set"
    assert any("evidence" in error for error in result.rejected[0].errors)


def test_a_baked_in_rounded_label_is_refused() -> None:
    """Design round 1 (D4): the data is honest, the PRINTED value is not.

    "98.6" is a rounded literal the component wrote itself; ``LO.fmt`` at source precision
    prints "98.5", so the label claims a precision the source does not have. The value is
    chosen below 100 on purpose: a 3-digit one is refused earlier by the §4.2 literal rule,
    which would leave THIS rule untested.
    """
    rounded = component(
        json.dumps(
            {
                "lat": {
                    "title": "Latency",
                    "columns": ["region", "ms"],
                    "rows": [["us-west", 98.5]],
                }
            }
        ),
        '<div id="c">us-west p50: 98.6 ms</div>',
    )
    result = validate_output(rounded, EVIDENCE)
    assert not result.components, "a rounded literal reached the accepted set"
    assert any("<data> supports" in error for error in result.repair_errors), result.repair_errors
    # ...and the SOURCE-precision spelling of the same value passes, so the rule is about the
    # printed precision and not about printing a number at all.
    exact = rounded.replace("98.6 ms", "98.5 ms")
    assert validate_output(exact, EVIDENCE).components


def test_a_derived_column_is_recomputed_and_a_wrong_one_refused() -> None:
    """Memo §2.10: a computed column is allowed only where the validator can recompute it.

    Two numeric columns, one ratio and one percentage -- the two shapes the memo names -- and
    a third row whose declared value is wrong, which must be refused rather than trusted.
    """
    evidence = (
        Dataset(
            title="Requests",
            source="bench.csv",
            columns=("region", "p50", "p99"),
            rows=(("us-east", "100"), ("us-west", "200")),
            n_rows=2,
            numeric_columns=("p50", "p99"),
        ),
    )
    columns = ["region", "p50", "p99", "overhead (derived: p99/p50)", "load (derived: p50/p99*100)"]
    good_rows = [["us-east", 100, 200, 2.0, 50.0]]
    bad_rows = [["us-east", 100, 200, 3.0, 50.0]]
    body = '<div id="c"></div><script>LO.table(document.getElementById("c"),"req")</script>'
    good = component(
        json.dumps({"req": {"title": "Requests", "columns": columns, "rows": good_rows}}), body
    )
    bad = component(
        json.dumps({"req": {"title": "Requests", "columns": columns, "rows": bad_rows}}), body
    )
    assert validate_output(good, evidence).components, "a correct ratio was refused"
    refused = validate_output(bad, evidence)
    assert not refused.components
    assert any("derived" in error for error in refused.repair_errors), refused.repair_errors


# --- the §4.2 scan -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        '<div id="c"></div><script>fetch("http://x")</script>',
        '<div id="c"></div><script>parent.api.readFile("/etc/hosts")</script>',
        '<iframe srcdoc="x"></iframe>',
        '<div id="c" onclick="eval(1)"></div>',
        '<script>new Worker("w.js")</script>',
    ],
)
def test_the_scan_refuses_the_named_escapes(body: str) -> None:
    result = validate_output(component("{}", body), EVIDENCE)
    assert not result.components and result.rejected


def test_an_oversized_component_is_refused() -> None:
    big = component("{}", "<div>" + "x" * MAX_COMPONENT_BYTES + "</div>")
    result = validate_output(big, EVIDENCE)
    assert not result.components
    assert any("bytes" in error for error in result.rejected[0].errors)


def test_a_malformed_document_is_refused() -> None:
    checker = _WellFormed()
    checker.feed("<div><span></div>")
    checker.close()
    assert checker.problem, "unbalanced markup must be detected"


# --- the LO.fmt mirror, against the running prelude ---------------------------------------

VECTORS: tuple[tuple[Any, str, str, int | None, bool], ...] = (
    (120, "120", "", None, False),
    (98.5, "98.5", "ms", None, False),
    (211.25, "211.25", "ms", None, False),
    (0, "0", "", None, False),
    (-3.5, "-3.5", "", None, False),
    (1234567.5, "1234567.5", "", None, False),
    (0.0001, "0.0001", "", None, False),
    (None, "null", "", None, False),
    ("", '""', "", None, False),
    (211.25, "211.25", "ms", 1, False),
    (211.25, "211.25", "ms", 0, False),
    (12345, "12345", "", None, True),
    (1500000, "1500000", "", None, True),
    (10000, "10000", "", None, True),
    (9999, "9999", "", None, True),
    ("us-east", '"us-east"', "", None, False),
    (42, "42", "%", None, False),
    (12345678901, "12345678901", "", None, False),
    (0.5, "0.5", "%", None, False),
    (-1234.5, "-1234.5", "ms", None, False),
    (1e-7, "1e-7", "", None, False),
    (2.5e7, "25000000", "", None, True),
    ("3.2GB", '"3.2GB"', "", None, False),
)


def test_the_fmt_mirror_is_pinned_against_the_prelude_under_node(tmp_path: Path) -> None:
    if shutil.which("node") is None:
        if os.environ.get("CI"):
            pytest.fail("node is not on PATH: the LO.fmt mirror cannot be verified on CI")
        pytest.skip("node is not installed: the LO.fmt mirror was NOT verified against the JS")
    steps = []
    for _, js, unit, digits, compact in VECTORS:
        parts = []
        if unit:
            parts.append(f"unit:{json.dumps(unit)}")
        if digits is not None:
            parts.append(f"digits:{digits}")
        if compact:
            parts.append("compact:1")
        steps.append({"call": f"LO.fmt({js}, {{{','.join(parts)}}})"})
    scenario = tmp_path / "fmt.json"
    scenario.write_text(json.dumps({"data": {}, "steps": steps}), encoding="utf-8")
    done = subprocess.run(
        ["node", str(HARNESS), str(PRELUDE_DIR / "prelude.js"), str(scenario)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    theirs = [call["value"] for call in json.loads(done.stdout)["calls"]]
    mine = [
        fmt_mod.fmt(value, unit=unit, digits=digits, compact=compact)
        for value, _, unit, digits, compact in VECTORS
    ]
    assert theirs == mine, [
        (v[1], a, b) for v, a, b in zip(VECTORS, theirs, mine, strict=True) if a != b
    ]


def test_the_mirror_never_pads_decimals_the_source_does_not_have() -> None:
    assert fmt_mod.fmt(120, unit="ms") == "120 ms"
    assert fmt_mod.fmt(120.0, unit="ms") == "120 ms"
    assert fmt_mod.fmt(None) == fmt_mod.NOT_A_VALUE


# --- the prompt: verbatim, priced, and fork-only -------------------------------------------


def test_the_prompt_is_the_memos_appendix_a_fence_character_for_character() -> None:
    """The two halves cannot drift: the code is the memo's text or the test is red."""
    text = MEMO.read_text(encoding="utf-8")
    start = text.index("## Appendix A")
    opening = text.index("```text", start)
    closing = text.index("```", opening + len("```text"))
    fence = text[opening + len("```text") : closing].strip("\n")
    assert GENERATOR_SYSTEM_PROMPT == fence


def test_the_prompt_is_within_five_percent_of_the_measured_token_count() -> None:
    tiktoken = pytest.importorskip("tiktoken", reason="the token budget is measured with tiktoken")
    counted = len(tiktoken.get_encoding("o200k_base").encode(GENERATOR_SYSTEM_PROMPT))
    # The memo's figure (re-measured for the body-only amendment, 2026-10-10).
    assert abs(counted - 1140) <= 0.05 * 1140, counted


def test_a_steer_instruction_is_flat_and_bounded() -> None:
    packed = steer_instruction("line one\nline two")
    assert packed == "<instruction>line one line two</instruction>"
    long = steer_instruction("x" * (MAX_INSTRUCTION_CHARS * 2))
    assert len(long) == len("<instruction>") + MAX_INSTRUCTION_CHARS + len("</instruction>")


def test_the_repair_message_names_the_errors_and_stays_short() -> None:
    message = repair_message(["one", "two"])
    assert message.startswith("Fix only these components") and "one; two" in message
    assert len(repair_message(["x" * 4000])) < 600


# --- structural attributes are layout, not claims (round-1 review R3) --------------------


def test_a_raw_svg_components_geometry_is_not_read_as_displayed_text() -> None:
    """R3, the report's repro: the blessed raw-SVG path must not be refused for LAYOUT.

    ``viewBox``, path data, ``points``, ``transform``, coordinates and sizes are structure
    the component DRAWS with (App. A: "no fixed widths, use viewBox SVG"); the rendered-
    values check reads element TEXT and data-bearing attributes, not tag internals.
    """
    raw = component(
        json.dumps(
            {
                "lat": {
                    "title": "Latency (ms)",
                    "columns": ["region", "ms"],
                    "rows": [["us-west", 98.5]],
                }
            }
        ),
        '<svg viewBox="0 0 48 20" role="img">'
        '<rect x="0" y="0" width="48" height="20" fill="none"></rect>'
        '<polyline points="0,20 12,8 24,14 48,10"></polyline>'
        '<path d="M0 20 L48 2" transform="translate(2,3)"></path>'
        '<circle cx="24" cy="10" r="5" stroke-width="2"></circle>'
        '<text x="4" y="14">98.5 ms</text>'
        "</svg>",
    )
    result = validate_output(raw, EVIDENCE)
    assert not result.rejected, result.repair_errors


def test_a_fabricated_number_in_svg_text_or_an_aria_label_is_still_refused() -> None:
    """R3's guard: exempting geometry must not exempt CLAIMS -- text and readable attrs."""
    data = json.dumps(
        {
            "lat": {
                "title": "Latency (ms)",
                "columns": ["region", "ms"],
                "rows": [["us-west", 98.5]],
            }
        }
    )
    in_text = component(
        data,
        '<svg viewBox="0 0 48 20"><text x="4" y="14">99.9 ms</text></svg>',
    )
    result = validate_output(in_text, EVIDENCE)
    assert not result.components, "a baked-in text literal reached the accepted set"
    assert any("<data> supports" in error for error in result.repair_errors), result.repair_errors

    in_attribute = component(
        data,
        '<svg viewBox="0 0 48 20" role="img" aria-label="p50: 99.9 ms">'
        '<rect x="0" y="0" width="48" height="20"></rect></svg>',
    )
    result = validate_output(in_attribute, EVIDENCE)
    assert not result.components, "a fabricated aria-label reached the accepted set"
    assert any("<data> supports" in error for error in result.repair_errors), result.repair_errors


# --- structural geometry at chart scale (round-2 review R2-1/R2-2) -------------------------


def test_a_raw_svg_chart_scale_is_not_refused_for_its_geometry() -> None:
    r"""R2-1: the literal rule must exempt structural geometry, not only ``viewBox``/style.

    The round-1 acceptance case drew on a 48x20 canvas, so every geometry number stayed
    under the ``\d{3,}`` rule's reach and a real chart was still refused:
    ``points="0,480 320,240 640,60"`` tripped ``inline numeric literal '480'``, and a
    normalized 0-100 chart tripped on ``100``. ``x1``/``y1`` are exercised because the
    pair regex could not match them at all (its name class had no digits), leaving their
    values to the static-numeral scan as false positives too.
    """
    data = json.dumps(
        {
            "lat": {
                "title": "Latency (ms)",
                "columns": ["region", "ms"],
                "rows": [["us-east", 120], ["us-west", 98.5], ["eu-west", 143]],
            }
        }
    )
    raw = component(
        data,
        '<svg viewBox="0 0 640 480" role="img">'
        '<polyline points="0,480 320,240 640,60"></polyline>'
        '<line x1="0" y1="480" x2="640" y2="60" stroke="currentColor"></line>'
        "</svg>",
    )
    result = validate_output(raw, EVIDENCE)
    assert result.components and not result.rejected, result.repair_errors

    normalized = component(
        data,
        '<svg viewBox="0 0 100 100">'
        '<rect x="0" y="0" width="100" height="100" fill="none"></rect>'
        '<polyline points="0,100 50,60 100,10"></polyline>'
        "</svg>",
    )
    result = validate_output(normalized, EVIDENCE)
    assert result.components and not result.rejected, result.repair_errors


def test_a_fabricated_three_digit_numeral_in_text_or_aria_label_is_still_refused() -> None:
    """R2-1's guard, at the literal rule's own scale: blanking GEOMETRY must not exempt
    readable CLAIMS. A fabricated 3+ digit numeral in element text or ``aria-label`` is
    still refused -- including a text numeral that also appears as geometry, which is the
    shape the exemption could most plausibly hide.
    """
    data = json.dumps(
        {
            "lat": {
                "title": "Latency (ms)",
                "columns": ["region", "ms"],
                "rows": [["us-west", 98.5]],
            }
        }
    )
    in_text = component(data, '<svg viewBox="0 0 48 20"><text x="4" y="14">999 ms</text></svg>')
    result = validate_output(in_text, EVIDENCE)
    assert not result.components, "a 3-digit fabricated text literal reached the accepted set"
    assert any("999" in error for error in result.repair_errors), result.repair_errors

    in_attribute = component(
        data,
        '<svg viewBox="0 0 48 20" role="img" aria-label="p50: 999 ms">'
        '<rect x="0" y="0" width="48" height="20"></rect></svg>',
    )
    result = validate_output(in_attribute, EVIDENCE)
    assert not result.components, "a 3-digit fabricated aria-label reached the accepted set"
    assert any("999" in error for error in result.repair_errors), result.repair_errors

    beside_geometry = component(
        data,
        '<svg viewBox="0 0 640 480"><polyline points="0,480 320,240 640,60"></polyline>'
        '<text x="4" y="14">480 ms</text></svg>',
    )
    result = validate_output(beside_geometry, EVIDENCE)
    assert not result.components, "a fabricated text numeral beside geometry was accepted"
    assert any("<data> supports" in error for error in result.repair_errors), result.repair_errors


def test_unquoted_and_multi_line_geometry_is_exempt_too() -> None:
    r"""R3-2: the exemption's grammar covers unquoted values and multi-line quoted values.

    The pair regex required a QUOTED, SINGLE-LINE value, so ``<rect x=0 y=0 width=640
    height=480>`` still tripped ``inline numeric literal '640'`` and a wrapped ``points``
    list was refused for the same class of reason as R2-1 (real geometry refused, lower
    frequency). Unquoted values cannot contain whitespace (HTML's attribute grammar), so
    a spaced coordinate LIST has to stay quoted for the value to be blankable -- the
    boundary the regex comment records, in the direction that keeps scanning (refusal),
    not one that silently accepts.
    """
    data = json.dumps(
        {
            "lat": {
                "title": "Latency (ms)",
                "columns": ["region", "ms"],
                "rows": [["us-east", 120], ["us-west", 98.5], ["eu-west", 143]],
            }
        }
    )
    unquoted = component(
        data,
        '<svg viewBox="0 0 640 480" role="img">'
        "<rect x=0 y=0 width=640 height=480 fill=none></rect>"
        "</svg>",
    )
    result = validate_output(unquoted, EVIDENCE)
    assert result.components and not result.rejected, result.repair_errors

    wrapped = component(
        data,
        '<svg viewBox="0 0 640 480">'
        '<polyline points="0,480\n320,240\n640,60"></polyline>'
        "</svg>",
    )
    result = validate_output(wrapped, EVIDENCE)
    assert result.components and not result.rejected, result.repair_errors


def test_an_unquoted_value_outside_the_structural_table_is_still_refused() -> None:
    """R3-2's guard: widening the pair regex to unquoted values must not exempt readable
    claims. A 3+ digit numeral in an unquoted NON-structural attribute (``title=7000``)
    and a fabricated text numeral beside unquoted geometry are both still refused.
    """
    data = json.dumps(
        {
            "lat": {
                "title": "Latency (ms)",
                "columns": ["region", "ms"],
                "rows": [["us-west", 98.5]],
            }
        }
    )
    claimed_attribute = component(
        data,
        '<svg viewBox="0 0 48 20" title=7000><rect x=0 y=0 width=48 height=20></rect></svg>',
    )
    result = validate_output(claimed_attribute, EVIDENCE)
    assert not result.components, "an unquoted fabricated attribute reached the accepted set"
    assert any("7000" in error for error in result.repair_errors), result.repair_errors

    beside_unquoted_geometry = component(
        data,
        '<svg viewBox="0 0 640 480"><rect x=0 y=0 width=640 height=480></rect>'
        "<text x=4 y=14>480 ms</text></svg>",
    )
    result = validate_output(beside_unquoted_geometry, EVIDENCE)
    assert not result.components, "a fabricated text numeral beside unquoted geometry was accepted"
    assert any("<data> supports" in error for error in result.repair_errors), result.repair_errors
