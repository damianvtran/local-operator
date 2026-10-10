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
                "rows": [["us-east", 120], ["us-west", 98.5], ["eu-west", 143], ["ap-south", 211.25]],
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
    assert result.rejected and "source" in result.rejected[0].describe()


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

    "211.3" is a rounded literal the component wrote itself; ``LO.fmt`` at source precision
    prints "211.25", so the label claims a precision the source does not have.
    """
    rounded = component(
        json.dumps(
            {
                "lat": {
                    "title": "Latency",
                    "columns": ["region", "ms"],
                    "rows": [["ap-south", 211.25]],
                }
            }
        ),
        '<div id="c">ap-south p99: 211.3 ms</div>',
    )
    result = validate_output(rounded, EVIDENCE)
    assert not result.components, "a rounded literal reached the accepted set"
    assert any("not reproducible" in error for error in result.rejected[0].errors)


def test_a_derived_column_is_recomputed_and_a_wrong_one_refused() -> None:
    share = {"columns": ["region", "ms", "share (derived: ms/285.75)"], "rows": [["ap-south", 211.25, 0.7393]]}
    good = component(
        json.dumps({"lat": {"title": "Latency", **share, "rows": [["ap-south", 211.25, 0.7393]]}}),
        '<div id="c"></div><script>LO.table(document.getElementById("c"),"lat")</script>',
    )
    bad = component(
        json.dumps({"lat": {"title": "Latency", **share, "rows": [["ap-south", 211.25, 0.9]]}}),
        '<div id="c"></div><script>LO.table(document.getElementById("c"),"lat")</script>',
    )
    assert validate_output(good, EVIDENCE).components, "a correct ratio was refused"
    refused = validate_output(bad, EVIDENCE)
    assert not refused.components
    assert any("derived" in error for error in refused.rejected[0].errors)


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
