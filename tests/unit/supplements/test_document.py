"""The pure document assembler (lane C0): shape, injection order, data block, CSP."""

from __future__ import annotations

import json
import re

import pytest

from local_operator.supplements import document
from local_operator.supplements.document import (
    CONTENT_SECURITY_POLICY,
    PRELUDE_VERSION,
    assemble_document,
    assemble_stored_component,
    data_json,
    split_component,
)
from tests.unit.supplements.conftest import FIXTURES, raw

DOCS = ("populated", "empty", "error")


@pytest.mark.parametrize("name", DOCS)
def test_the_assembled_fixture_documents_are_byte_stable(name: str) -> None:
    blob = raw(f"components/{name}.html")
    assert assemble_stored_component(blob) == raw(f"documents/{name}.html")
    # pure: the same input is the same bytes, call after call
    assert assemble_stored_component(blob) == assemble_stored_component(blob)


@pytest.mark.parametrize("name", DOCS)
def test_the_document_has_the_memo_shape_in_the_memo_order(name: str) -> None:
    html = raw(f"documents/{name}.html")
    marks = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<meta http-equiv="Content-Security-Policy" content="',
        "<style>",
        "</style></head><body>",
        '<script type="application/json" id="lo-data">',
        "<script>",  # the prelude
        "</body></html>",
    ]
    positions = [html.index(mark) for mark in marks]
    assert positions == sorted(positions) and len(set(positions)) == len(marks)
    assert html.startswith(marks[0]) and html.endswith(marks[-1])


@pytest.mark.parametrize("name", ("populated", "error"))
def test_the_prelude_is_loaded_before_the_bodys_inline_scripts_run(name: str) -> None:
    """Memo §2.6 note: the §2.6 sketch prints the prelude last and then says the opposite;
    the note is the contract, because the component's script needs `LO` on its first line."""
    html = raw(f"documents/{name}.html")
    prelude_at = html.index(document.prelude_js()[:60])
    data_at = html.index('id="lo-data"')
    body_script_at = html.index("<script>LO.")
    assert data_at < prelude_at < body_script_at, "data block, then prelude, then the body"


def test_the_prelude_is_injected_verbatim_once_each() -> None:
    html = raw("documents/populated.html")
    assert html.count(document.prelude_css()) == 1
    assert html.count(document.prelude_js()) == 1


def test_the_csp_is_the_memos_policy_in_a_meta_not_a_header() -> None:
    html = raw("documents/populated.html")
    assert (
        f'<meta http-equiv="Content-Security-Policy" content="{CONTENT_SECURITY_POLICY}">' in html
    )
    for directive in (
        "default-src 'none'", "script-src 'unsafe-inline'", "style-src 'unsafe-inline'",
        "img-src data:", "font-src data:", "connect-src 'none'", "frame-src 'none'",
        "form-action 'none'", "base-uri 'none'",
    ):  # fmt: skip
        assert directive in CONTENT_SECURITY_POLICY
    # the policy sits before any script: it must govern them
    assert html.index("Content-Security-Policy") < html.index("<script")


def test_the_data_block_is_the_merged_json_the_prelude_reads() -> None:
    html = raw("documents/populated.html")
    block = re.search(r'<script type="application/json" id="lo-data">(.*?)</script>', html, re.S)
    assert block is not None
    data = json.loads(block.group(1))
    assert list(data) == ["lat"]
    assert data["lat"]["columns"] == ["region", "ms"]
    assert data["lat"]["rows"][1] == ["us-west", 98.5]  # source precision survives
    assert "<data>" not in html, "the stored <data> element must not be left in the body"


def test_an_empty_component_still_assembles_to_a_loadable_document() -> None:
    html = raw("documents/empty.html")
    assert 'id="lo-data">{}</script>' in html
    assert html.endswith("</script>\n\n</body></html>")


def test_data_cannot_break_out_of_its_script_block() -> None:
    hostile = {"d": {"title": "</script><script>alert(1)</script><!--", "rows": [["<b>"]]}}
    out = data_json(hostile)
    assert "<" not in out
    assert json.loads(out) == hostile, "the escape must read back as the same data"
    html = assemble_document("<div></div>", hostile)
    # exactly the structural script elements: data, prelude (no component script here)
    assert html.count("</script>") == 2


def test_assembly_is_concatenation_not_templating() -> None:
    """Braces and percent signs in the body must pass through untouched."""
    body = '<div>{0} {x} %s {{}} $1 \\1</div><script>var a={b:"%d"}</script>'
    html = assemble_document(body, {})
    assert body in html


def test_split_component_merges_blocks_and_rejects_conflicts() -> None:
    body, data = split_component('<data>{"a":{"x":1}}</data><data>{"b":{"y":2}}</data>\n<p>hi</p>')
    assert body == "<p>hi</p>" and data == {"a": {"x": 1}, "b": {"y": 2}}
    # the same dataset restated identically is fine
    assert split_component('<data>{"a":1}</data><data>{"a":1}</data>')[1] == {"a": 1}
    with pytest.raises(ValueError, match="twice"):
        split_component('<data>{"a":1}</data><data>{"a":2}</data>')
    with pytest.raises(ValueError, match="not valid JSON"):
        split_component("<data>{nope</data>")
    with pytest.raises(ValueError, match="object"):
        split_component("<data>[1]</data>")


def test_a_non_finite_number_is_refused_at_both_boundaries() -> None:
    """Agent review R5: ``NaN``/``Infinity`` would reach ``lo-data`` as bare tokens that
    ``JSON.parse`` refuses, killing the whole prelude (no ``LO``, no ``ready``)."""
    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValueError, match="non-finite"):
            split_component(f'<data>{{"a":[[1,{token}]]}}</data><p></p>')
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            data_json({"a": {"rows": [[value]]}})
        with pytest.raises(ValueError):
            assemble_document("<p></p>", {"a": value})


def test_a_dataset_string_containing_the_close_tag_stays_whole() -> None:
    """Agent review R6(a): the JSON is parsed in place, so ``</data>`` inside a value does
    not end the block."""
    body, data = split_component('<data>{"a":{"title":"x</data>y"}}</data>\n<p>b</p>')
    assert data == {"a": {"title": "x</data>y"}} and body == "<p>b</p>"


def test_attributes_on_the_data_tag_are_tolerated() -> None:
    """Agent review R6(b): ``<data value="…">`` is still a data block, not body text."""
    body, data = split_component('<data value="7">{"a":1}</data ><DATA>{"b":2}</DATA><p>b</p>')
    assert data == {"a": 1, "b": 2} and body == "<p>b</p>"
    # ...and <datalist> is not a <data> tag
    assert split_component("<datalist></datalist>") == ("<datalist></datalist>", {})


def test_only_the_leading_blocks_are_data_and_the_body_is_kept_verbatim() -> None:
    """Agent review R6(c): a ``<data>`` the body's own script mentions is the body's."""
    body_in = '<div id="c"></div><script>var s="<data>{\\"z\\":1}</data>"</script>'
    body, data = split_component('<data>{"a":1}</data>\n' + body_in)
    assert data == {"a": 1} and body == body_in
    assert split_component(body_in) == (body_in, {})


def test_a_block_with_trailing_text_after_its_json_is_refused() -> None:
    with pytest.raises(ValueError, match="exactly one JSON object"):
        split_component('<data>{"a":1} trailing</data>')


def test_the_prelude_files_are_package_data() -> None:
    """The wheel must carry what the assembler reads at runtime."""
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads((Path(document.__file__).parents[2] / "pyproject.toml").read_text())
    shipped = pyproject["tool"]["setuptools"]["package-data"]["local_operator"]
    assert "supplements/prelude/prelude.css" in shipped
    assert "supplements/prelude/prelude.js" in shipped
    # the sources are git-only: nothing at runtime reads them
    assert not any("src" in entry and "supplements" in entry for entry in shipped)


def test_prelude_version_is_an_integer_a_host_can_name() -> None:
    assert isinstance(PRELUDE_VERSION, int) and PRELUDE_VERSION >= 1


def test_the_fixture_tree_is_what_the_generator_emits() -> None:
    """Drift guard: a hand edit, or a contract change without regeneration, fails here."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("supplement_fixture_build", FIXTURES / "build.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.generate()
    on_disk = {
        p.relative_to(FIXTURES).as_posix(): p.read_bytes()
        for sub in ("rows", "events", "messages", "components", "documents", "geometry")
        for p in (FIXTURES / sub).glob("*")
    }
    assert on_disk == expected, "run: .venv/bin/python tests/fixtures/supplements/build.py"
