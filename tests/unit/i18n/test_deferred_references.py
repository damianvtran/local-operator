"""No deferred namespace may silently gain a render point.

``catalogues.DEFERRED_NAMESPACES`` takes a namespace OUT of the eager bundle:
its messages reach a client only if that client fetches the namespace's own
route. That is safe exactly while every producer renders the copy IN-PROCESS
and puts a finished string on the wire. The first desktop or relay screen that
starts sending one of these namespaces' keys as an ENVELOPE IDENTITY would
render English on every non-English client until someone noticed — with no
failing test to point at the cause, because the deferred set is what makes the
keys absent from the bundle.

So the reviewed census lives in the repository, not in an MR description:

* ``local_operator/i18n/deferred_wire_inventory.json`` records, per
  deferred namespace, its reason (asserted EQUAL to the dict's, so the two
  cannot drift), the render points a human verified as plain-string consumers
  (path, symbol, note), and the per-file reference count at review time.
* The counts are a RATCHET: they may fall when a render point moves out, and a
  rise or an unlisted file fails. The failure message names both legitimate
  responses — make the namespace eager again (its catalogue returns to the
  bundle), or move the new use to a listed render point — and the third case
  this policy needs: record it as one more verified plain-string consumer.

The counting rule itself lives in ``scripts/i18n/deferred_watch.py`` (one
implementation, so the tool that writes the census and the test that checks it
cannot disagree), and is stated in that module's docstring.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

from local_operator.i18n import catalogues

REPO = Path(__file__).resolve().parents[3]
INVENTORY = REPO / "local_operator" / "i18n" / "deferred_wire_inventory.json"


def _inventory() -> dict[str, Any]:
    """The parsed inventory: a ``note`` string beside the ``namespaces`` map.

    The two value shapes are why this is ``Any`` and not a TypedDict; it is a
    review artifact read by a dev tool, not a runtime contract.
    """
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def test_the_inventory_names_exactly_the_deferred_namespaces() -> None:
    """The inventory is the policy's other half: same keys, same reasons."""
    inventory = _inventory()["namespaces"]
    assert set(inventory) == set(catalogues.DEFERRED_NAMESPACES), (
        "Every deferred namespace needs a recorded reason and its verified "
        "render points, and the inventory may not keep a namespace that is no "
        "longer deferred: local_operator/i18n/deferred_wire_inventory.json"
    )
    for namespace, reason in catalogues.DEFERRED_NAMESPACES.items():
        assert inventory[namespace]["reason"] == reason, (
            f"{namespace}: the inventory's reason and DEFERRED_NAMESPACES' "
            "reason must be the same text, so neither can rot unnoticed"
        )
        points = inventory[namespace]["render_points"]
        assert points, f"{namespace} has no verified render point recorded"
        for point in points:
            assert (
                point["path"] and point["symbol"] and point["note"]
            ), f"{namespace}: a render point needs path, symbol and note: {point!r}"
            assert (REPO / point["path"]).is_file(), (
                f"{namespace}: render point {point['path']} no longer exists; "
                "re-verify what renders this namespace"
            )


def test_no_deferred_namespace_gained_a_reference(
    i18n_deferred_watch: ModuleType,
) -> None:
    """A new render point for deferred copy fails here, by file and by count."""
    recorded = i18n_deferred_watch.recorded_references(_inventory())
    live = i18n_deferred_watch.census()
    failures = i18n_deferred_watch.new_references(recorded, live)
    assert not failures, (
        "A deferred namespace gained a reference — most likely a new render "
        "point. Either the namespace has gained a render point, in which case "
        "remove it from DEFERRED_NAMESPACES in "
        "local_operator/i18n/catalogues.py (its catalogue returns to the eager "
        "set), or move the new use to an existing listed render point. If the "
        "use is a plain-string consumer like the ones recorded, re-verify it "
        "and refresh the census with "
        "`python scripts/i18n/deferred_watch.py --write`. Failures: " + "; ".join(failures)
    )


def test_the_census_catches_one_added_reference(
    i18n_deferred_watch: ModuleType, tmp_path: Path
) -> None:
    """Negative control: the guard fails on a reference the repo does not have.

    PROOF, not decoration: a ratchet that cannot name a violation on a known
    input is indistinguishable from a ratchet that never fires, and this file's
    other tests pass on today's tree either way. The scratch tree carries the
    two shapes a new render point takes — a fresh file importing the key
    module under an alias, and one more call site inside an already-listed
    file — and both must be reported.
    """
    package = tmp_path / "local_operator"
    (package / "harness").mkdir(parents=True)
    clean = package / "harness" / "bridge_copy.py"
    clean.write_text("def render() -> str:\n    return 'text'\n", encoding="utf-8")
    namespaces = ("wire.tool_rows.bridge",)

    baseline = i18n_deferred_watch.census(package, namespaces)
    assert baseline == {namespace: {} for namespace in namespaces}

    # (a) a NEW file renders one of the namespace's keys through an alias
    (package / "harness" / "bridge_extra.py").write_text(
        "from local_operator.i18n.keys import wire_tool_rows_bridge as _k\n\n"
        "def verb() -> str:\n    return _k.verb_running()\n",
        encoding="utf-8",
    )
    after_new_file = i18n_deferred_watch.census(package, namespaces)
    assert after_new_file["wire.tool_rows.bridge"] == {
        "harness/bridge_extra.py": 2
    }, "one import of the key module and one attribute call"
    assert i18n_deferred_watch.new_references(baseline, after_new_file) == [
        "wire.tool_rows.bridge: harness/bridge_extra.py (0 -> 2)"
    ]

    # (b) one more call inside an ALREADY-LISTED file raises that file's count
    clean.write_text(
        "from local_operator.i18n.keys import wire_tool_rows_bridge as _k\n\n"
        "def render() -> str:\n    return _k.verb_running()\n",
        encoding="utf-8",
    )
    after_rise = i18n_deferred_watch.census(package, namespaces)
    assert i18n_deferred_watch.new_references(after_new_file, after_rise) == [
        "wire.tool_rows.bridge: harness/bridge_copy.py (0 -> 2)"
    ]

    # A count that FALLS is not a failure: a render point may move out.
    assert i18n_deferred_watch.new_references(after_rise, baseline) == []


def test_every_deferred_key_call_is_rendered_through_a_wrapper(
    i18n_deferred_watch: ModuleType,
) -> None:
    """The census counts references; this checks what each one DOES.

    A reference that gains an envelope hop still counts exactly once, so the
    ratchet cannot see it. Every deferred key call today sits inside `_t(`,
    `_desc(` or `render(` — the in-process render seam — and that is the
    property to keep: an edit that hands the `Msg` (or its `code`) onward is
    the failure this test names.
    """
    live = i18n_deferred_watch.census()
    violations = i18n_deferred_watch.unwrapped_key_calls(live, i18n_deferred_watch.PACKAGE)
    assert not violations, (
        "A deferred key is called outside a render wrapper. These namespaces "
        "are not in the eager bundle, so a Msg or code leaving the process "
        "renders English on every non-English client: render it with "
        "_t()/_desc()/render(), or return the Msg for the caller to render. "
        "Offenders: " + "; ".join(violations)
    )


def test_the_parent_child_exclusion_counts_each_namespace_separately(
    i18n_deferred_watch: ModuleType, tmp_path: Path
) -> None:
    """`wire.slash` must not be charged for `wire.slash.library`'s keys.

    The two namespaces are both deferred and one is a string prefix of the
    other, so a plain substring count would move the parent's baseline when a
    child render point appears — the wrong file would fail, for the wrong
    reason. This pins the lookahead that keeps them apart.
    """
    package = tmp_path / "local_operator"
    package.mkdir()
    child = package / "child.py"
    parent = package / "parent.py"
    namespaces = ("wire.slash", "wire.slash.library")

    child.write_text(
        "from local_operator.i18n.keys import wire_slash_library as _lib\n\n"
        "def text() -> str:\n    return _lib.help()\n",
        encoding="utf-8",
    )
    parent.write_text(
        "from local_operator.i18n.keys import wire_slash as _slash\n\n"
        "def text() -> str:\n    return _slash.help()\n",
        encoding="utf-8",
    )
    counts = i18n_deferred_watch.census(package, namespaces)
    assert counts["wire.slash.library"] == {"child.py": 2}, "import + call"
    assert counts["wire.slash"] == {"parent.py": 2}, "child.py must not be counted here"


def test_the_wrapper_check_catches_an_envelope_edit(
    i18n_deferred_watch: ModuleType, tmp_path: Path
) -> None:
    """The teeth: a key call edited to hand the Msg onward is reported.

    Both files below compile and count identically as references; only the
    rendered one is acceptable. Without this control the wrapper assertion
    could rot into a test that cannot fail.
    """
    package = tmp_path / "local_operator"
    package.mkdir()
    namespaces = ("wire.tool_rows.bridge",)
    good = package / "good.py"
    bad = package / "bad.py"
    good.write_text(
        "from local_operator.i18n.keys import wire_tool_rows_bridge as _k\n"
        "from local_operator.i18n.text import tr\n\n"
        '_t = tr("wire.tool_rows.bridge")\n\n'
        "def text() -> str:\n    return _t(_k.verb_running())\n",
        encoding="utf-8",
    )
    bad.write_text(
        "from local_operator.i18n.keys import wire_tool_rows_bridge as _k\n\n"
        "def envelope() -> dict:\n    return _k.verb_running().envelope()\n",
        encoding="utf-8",
    )
    references = i18n_deferred_watch.census(package, namespaces)
    problems = i18n_deferred_watch.unwrapped_key_calls(references, package, namespaces)
    assert len(problems) == 1, problems
    assert problems[0].startswith("bad.py:"), problems
    assert "is not inside a render wrapper (envelope)" in problems[0], problems
