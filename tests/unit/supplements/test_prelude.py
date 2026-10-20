"""The vendored prelude: its size budget, its pin, and what its own JS does (lane C0).

The prelude is code that runs inside an opaque-origin frame on four surfaces, so a change
to it is a change to every surface at once. Three things are therefore pinned:

* the SIZE of the minified pair (memo §2.6: <= 11 KB raw / <= 4.5 KB gzip), measured on the
  VENDORED minified bytes, never on a rebuild -- esbuild output drifts by more than the
  gzip headroom between versions (the spike measured 1 and 2 byte differences between two
  builds), so a test that rebuilt would pass and fail on the toolchain, not on the budget;
* the BYTES, by digest, bound to ``PRELUDE_VERSION`` so an edit that forgets to bump the
  version fails;
* the BEHAVIOUR the contract promises of the frame -- the nonce echo (memo §4.1 S-R4) and
  the label geometry at the 220 px floor -- by executing the real ``prelude.js`` under node
  against a stub DOM (``prelude_harness.mjs``). Without node these FAIL on CI (``CI`` set)
  and skip with a reason locally -- never a silent pass.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator.supplements import contract, document
from tests.unit.supplements.conftest import FIXTURES, load

PRELUDE_DIR = Path(document.__file__).resolve().parent / "prelude"
HARNESS = Path(__file__).resolve().parent / "prelude_harness.mjs"

#: Memo §2.6 caps: 11 KB raw, 4.5 KB gzip.
RAW_CAP = 11 * 1024
GZIP_CAP = 4608

#: The pinned pair. Change these ONLY with a rebuilt pair, a bumped PRELUDE_VERSION and the
#: size table in prelude/BUILD.md. The sizes are the numbers the PR reports.
#:
#: PRELUDE_VERSION 2 (memo §8 F9): the `div.ttl` table title and its stylesheet rule were
#: stripped -- the component is body-only (§2.8, operator directive 2026-10-10) -- which is
#: what moved CSS 2,044 -> 2,008 B, JS 8,202 -> 8,047 B and gzip 4,607 -> 4,562 B.
#:
#: PRELUDE_VERSION 3: the ROOT element paints `--lo-canvas`, so a live theme flip keeps the
#: surface's token instead of the UA canvas. CSS 2,008 -> 2,056 B, gzip 4,562 -> 4,573 B; JS
#: untouched at 8,047 B. This left 35 B gzip of headroom against the 4,608 B cap.
#:
#: PRELUDE_VERSION 4: finite-only scales and labelled bar gaps. JS 8,170 B, pair
#: 10,226 B raw / 4,602 B gzip; 6 B gzip remained under the unchanged cap.
#:
#: PRELUDE_VERSION 6: horizontal bar height from the bar count, measured category gutter,
#: width-aware tick density, and dark-fallback ink. CSS 2,056 -> 1,878 B (the `h1` rule,
#: the reduced-motion block, `.dim` and the redundant light color-scheme rule were dead
#: weight under the no-chrome/no-motion contract and paid for the change); JS 8,077 ->
#: 8,114 B. Pair 9,992 B raw / 4,602 B gzip, 6 B under the unchanged cap.
PINNED_CSS_BYTES = 1878
PINNED_JS_BYTES = 8114
PINNED_GZIP_BYTES = 4602
PINNED_DIGEST = "15f90c388de1faef1afa05c9f7bde5fe3280acfdb5b710da362c7eaaf2275a64"
PINNED_VERSION = 6


@pytest.fixture
def node_available() -> None:
    """The behaviour tests below are the ONLY execution of the prelude's JS in the suite, so
    a missing ``node`` must never read as green: on CI it is a failure (the runner image
    ships node; losing it would silently drop the nonce and geometry coverage), locally a
    skip whose reason says what did not run."""
    if shutil.which("node") is not None:
        return
    if os.environ.get("CI"):
        pytest.fail("node is not on PATH: the prelude behaviour tests cannot run on CI")
    pytest.skip("node is not installed: the prelude's JS behaviour was NOT exercised")


needs_node = pytest.mark.usefixtures("node_available")


def _pair() -> tuple[bytes, bytes]:
    return (PRELUDE_DIR / "prelude.css").read_bytes(), (PRELUDE_DIR / "prelude.js").read_bytes()


def test_the_vendored_pair_fits_the_memo_budget() -> None:
    css, js = _pair()
    # `cat prelude.css prelude.js | gzip -9`: the ONE method every figure in the memo uses.
    gzipped = len(gzip.compress(css + js, compresslevel=9, mtime=0))
    assert (len(css), len(js)) == (PINNED_CSS_BYTES, PINNED_JS_BYTES)
    assert len(css) + len(js) <= RAW_CAP
    assert gzipped == PINNED_GZIP_BYTES
    assert gzipped <= GZIP_CAP, f"{gzipped} B gzip exceeds the {GZIP_CAP} B cap: trim, don't raise"


def test_the_gzip_figure_matches_the_shell_method_the_memo_quotes() -> None:
    if shutil.which("gzip") is None:
        pytest.skip("gzip not installed")
    css, js = _pair()
    shell = subprocess.run(["gzip", "-9", "-c"], input=css + js, capture_output=True, check=True)
    # Different gzip implementations differ by a few bytes of header; the cap, not the
    # exact figure, is what must hold under every one of them.
    assert len(shell.stdout) <= GZIP_CAP


def test_the_bytes_are_pinned_to_prelude_version() -> None:
    css, js = _pair()
    digest = hashlib.sha256(css + b"\0" + js).hexdigest()
    assert (document.PRELUDE_VERSION, digest) == (PINNED_VERSION, PINNED_DIGEST), (
        "prelude bytes changed: rebuild per prelude/BUILD.md, bump PRELUDE_VERSION, "
        "re-pin the digest and the sizes in this file"
    )


def test_the_prelude_cannot_break_out_of_its_script_element() -> None:
    css, js = _pair()
    for blob in (css, js):
        text = blob.decode("utf-8").lower()
        assert "</script" not in text and "<!--" not in text and "</style" not in text


def test_the_prelude_makes_no_network_or_storage_calls() -> None:
    """Memo App. C: no network, no storage, and `parent` only through the one `P()`."""
    js = (PRELUDE_DIR / "prelude.js").read_text(encoding="utf-8")
    for needle in (
        "fetch(", "XMLHttpRequest", "WebSocket", "localStorage", "sessionStorage",
        "indexedDB", "sendBeacon", "eval(", "importScripts", "document.cookie",
    ):  # fmt: skip
        assert needle not in js, needle
    assert js.count("parent.postMessage") == 1


#: The root element's rule and the dark block, as the minified build writes them.
_ROOT_RULE = re.compile(r"html\[data-ready\]\{([^}]*)\}")
_DARK_RULE = re.compile(r":root\[data-mode=dark\]\{([^}]*)\}")


def _declarations(rule: re.Pattern[str], css: str) -> dict[str, str]:
    match = rule.search(css)
    assert match is not None, f"the stylesheet has no rule matching {rule.pattern}"
    out: dict[str, str] = {}
    for declaration in match.group(1).split(";"):
        if ":" in declaration:
            name, _, value = declaration.partition(":")
            out[name.strip()] = value.strip()
    return out


def test_the_root_element_paints_the_canvas_token_as_the_frames_ground() -> None:
    """The frame's ground is `--lo-canvas` from the ROOT element (the live theme-flip defect).

    WHY THIS IS A STYLE PIN AND WHAT IT DOES NOT PROVE. The behaviour tests below run the
    prelude's JS against a stub DOM, which has no CSS engine, so nothing in this suite can
    read a computed or a painted colour: a stub that modelled "what a root with no background
    falls back to" would be this file asserting its own assumption. What is pinned is the
    style contract the defect came from -- the ROOT element declares the canvas token as its
    background (a root with no background is what let the UA canvas show through: #ffffff on
    a light flip, #121212 on a dark one, measured in local-operator-mobile PR #83), and the
    dark block declares a canvas token of its own, so the ground does not depend on which
    direction the theme moved. The rendered proof -- the real assembled document, mounted in
    a sandboxed `data:` frame, flipped both ways, ground read from the painted pixels and
    from the frame's own computed style -- is in the PR that added this test.

    WHAT THIS PIN COVERS, PRECISELY: the two directions are NOT asserted symmetrically here.
    The assertions below read the ROOT's declaration and the DARK block's own `--lo-canvas`;
    the light ground is the `:root` default `--lo-canvas:#fff`, left implicit because a light
    push overrides it inline. So the LIGHT arm rests on the rendered evidence named above, and
    this docstring should not be read as "the pin proves both directions".
    """
    css = (PRELUDE_DIR / "prelude.css").read_text(encoding="utf-8")
    root = _declarations(_ROOT_RULE, css)
    assert root.get("background") == "var(--lo-canvas)"
    assert root.get("visibility") == "visible"
    assert _declarations(_DARK_RULE, css).get("--lo-canvas") == "#121212"
    # Exactly one element paints the ground: `body` stays transparent by design, so the token
    # is the frame's canvas rather than a box painted inside it.
    assert re.search(r"body\{[^}]*background:transparent", css) is not None


# --------------------------------------------------------------------------------------
# Behaviour, under node
# --------------------------------------------------------------------------------------


def _run(scenario: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    done = subprocess.run(
        ["node", str(HARNESS), str(PRELUDE_DIR / "prelude.js"), str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def theme(nonce: str | None, mode: str = "light") -> dict[str, Any]:
    message: dict[str, Any] = {"lo": "supplement-host", "t": "theme", "mode": mode, "vars": {}}
    if nonce is not None:
        message["nonce"] = nonce
    return {"host": message}


PING = {"host": {"lo": "supplement-host", "t": "ping"}}


@needs_node
def test_ready_goes_out_without_a_nonce_because_none_exists_yet(tmp_path: Path) -> None:
    out = _run({"steps": []}, tmp_path)
    assert out["posts"] == [{"lo": "supplement", "v": 1, "t": "ready"}]


@needs_node
def test_resize_error_and_pong_echo_the_nonce_from_the_first_theme_push(tmp_path: Path) -> None:
    out = _run(
        {"steps": [theme("n-first"), PING, {"size": True}, {"error": "boom"}]},
        tmp_path,
    )
    by_type: dict[str, list[dict[str, Any]]] = {}
    for post in out["posts"]:
        by_type.setdefault(post["t"], []).append(post)
    assert "n" not in by_type["ready"][0]
    for kind in ("resize", "error", "pong"):
        assert by_type[kind], kind
        assert all(post["n"] == "n-first" for post in by_type[kind]), kind
        assert all(post["lo"] == "supplement" and post["v"] == 1 for post in by_type[kind])
    assert by_type["error"][0]["msg"] == "boom"


@needs_node
def test_a_second_theme_push_cannot_rebind_the_nonce(tmp_path: Path) -> None:
    """The binding is the FIRST push's: a later push (a navigated successor document, a
    forged host message) must not be able to move it."""
    out = _run(
        {"steps": [theme("n-first"), theme("n-second", mode="dark"), PING, {"error": "x"}]},
        tmp_path,
    )
    echoed = {post["n"] for post in out["posts"] if post["t"] in ("pong", "error", "resize")}
    assert echoed == {"n-first"}


@needs_node
def test_a_first_push_without_a_nonce_leaves_every_state_moving_post_unmarked(
    tmp_path: Path,
) -> None:
    """A host that never sends a nonce gets messages it must drop by rule (memo §4.1);
    the frame does not invent one, and a later push does not retrofit one."""
    out = _run({"steps": [theme(None), theme("late"), PING, {"error": "x"}]}, tmp_path)
    assert [p for p in out["posts"] if p["t"] in ("pong", "error")]
    assert all("n" not in post for post in out["posts"])


@needs_node
def test_messages_not_from_the_parent_are_ignored(tmp_path: Path) -> None:
    out = _run(
        {"steps": [{"other": {**theme("evil")["host"], "mode": "dark"}}, PING]},
        tmp_path,
    )  # fmt: skip
    # no theme was applied, so the ping is answered WITHOUT a nonce
    pongs = [post for post in out["posts"] if post["t"] == "pong"]
    assert pongs and all("n" not in post for post in pongs)


@needs_node
def test_content_inside_the_frame_can_read_the_nonce_by_design(tmp_path: Path) -> None:
    """Pins the TRUE property (agent review R1 / QA Q-1), so no later edit re-states the
    false one. The nonce authenticates the browsing context the host mounted -- a navigated
    successor never received the push -- and is NOT a secret from scripts in that context:
    ``LO.theme`` is the push as sent, and any script could add its own listener anyway.
    Keeping it from a successor is the host's navigation rule (memo §4.1)."""
    # h=7 marks the component's own post: the prelude's resize (posted on the same flush)
    # carries the nonce too, so picking "the last post" would pass without the forge.
    forge = "parent.postMessage({lo:'supplement',v:1,t:'resize',h:7,n:LO.theme.nonce},'*')"
    out = _run({"steps": [theme("bound"), {"draw": forge}]}, tmp_path)
    forged = [post for post in out["posts"] if post.get("h") == 7]
    assert len(forged) == 1, out["posts"]
    forged = forged[0]
    assert forged["n"] == "bound"
    # ...which is why a host clamps every accepted value: passing the nonce check says
    # where a post came from, not that it is benign.
    assert contract.accept_frame_message(forged, "bound") == forged


@needs_node
def test_an_error_thrown_before_the_theme_push_is_posted_with_the_nonce_once_bound(
    tmp_path: Path,
) -> None:
    """QA round 1, Q-2, on the real fixture: ``documents/error.html``'s inline script throws
    during parse, BEFORE the host's theme push at ``load``. The prelude must hold that error
    and post it -- with the nonce -- the moment the push binds it; before the fix it went
    out unmarked, every host dropped it, and the frame stayed blank forever."""
    import re

    html = (FIXTURES / "documents" / "error.html").read_text(encoding="utf-8")
    data_block = re.search(
        r'<script type="application/json" id="lo-data">(.*?)</script>', html, re.S
    )
    assert data_block is not None
    data = json.loads(data_block[1])
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    prelude, component = scripts[0], scripts[-1]
    assert prelude == (PRELUDE_DIR / "prelude.js").read_text(encoding="utf-8")
    assert "missing" in component
    out = _run({"data": data, "steps": [{"script": component}, theme("X"), PING]}, tmp_path)
    kinds = [post["t"] for post in out["posts"]]
    assert kinds[0] == "ready" and "n" not in out["posts"][0]
    errors = [post for post in out["posts"] if post["t"] == "error"]
    assert len(errors) == 1, out["posts"]  # held, not posted unmarked AND again
    assert errors[0]["n"] == "X" and "unknown dataset missing" in errors[0]["msg"]
    assert contract.accept_frame_message(errors[0], "X") == errors[0]
    # it is flushed at binding, ahead of everything else the push triggers
    assert kinds.index("error") < kinds.index("pong")


@needs_node
def test_only_the_first_pre_nonce_error_is_held(tmp_path: Path) -> None:
    """The first error is the cause; later ones are its fallout, and a host shows one line
    either way. One slot keeps the queue bounded whatever a component throws."""
    out = _run({"steps": [{"error": "first"}, {"error": "second"}, theme("X")]}, tmp_path)
    errors = [post for post in out["posts"] if post["t"] == "error"]
    assert [(e["msg"], e["n"]) for e in errors] == [("first", "X")]


@needs_node
@pytest.mark.parametrize(
    ("length", "echoed"),
    [(contract.NONCE_MAX_CHARS, True), (contract.NONCE_MAX_CHARS + 1, False)],
)
def test_the_frame_binds_a_nonce_up_to_the_contract_limit_and_refuses_a_longer_one(
    tmp_path: Path, length: int, echoed: bool
) -> None:
    """Agent review R4: the limit is the contract's ``NONCE_MAX_CHARS`` on both sides. A
    longer nonce is refused (bound as ""), never truncated: a truncated echo would fail
    every host comparison while looking almost right."""
    nonce = "a" * length
    out = _run({"steps": [theme(nonce), PING]}, tmp_path)
    pong = next(post for post in out["posts"] if post["t"] == "pong")
    assert (pong.get("n") == nonce) is echoed
    if not echoed:
        assert "n" not in pong
    assert (contract.accept_frame_message(pong, nonce) is not None) is echoed


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float]:
    width = min(a["right"], b["right"]) - max(a["left"], b["left"])
    height = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    return (width, height) if width > 0 and height > 0 else (0.0, 0.0)


@needs_node
def test_a_long_unit_at_the_220_px_floor_stays_inside_the_frame(tmp_path: Path) -> None:
    """Design round 3, D3-2: the 220 px long-unit fixture (``USD/day``).

    The unit is too long to reserve in the left margin at the floor width, so the top tick
    is drawn INSIDE the plot (the D2-1 fallback). The contract this pins is the one the
    memo claims: **never clipped**. Every label box lies inside the 220 px frame.
    """
    fixture = load("geometry/long_unit_220.json")
    out = _run(
        {
            "width": fixture["width"],
            "data": fixture["data"],
            "steps": [theme("n"), {"draw": fixture["draw"]}],
        },
        tmp_path,
    )
    texts = out["texts"]
    top = [t for t in texts if t["text"].endswith("USD/day")]
    assert (
        len(top) == 1 and top[0]["anchor"] == "start"
    ), "the top tick must use the inside fallback"
    for t in texts:
        assert t["left"] >= 0 and t["right"] <= fixture["width"], t


@needs_node
def test_the_known_top_tick_overprint_at_220_px_is_recorded_not_hidden(tmp_path: Path) -> None:
    """D3-2 (minor, accepted by design round 3 as a recorded C0 fixture, not a gate): the
    inside-the-plot top tick `1,000 USD/day` overprints the `910` value label (the browser
    measured 20.7 x 10.5 px; this stub-DOM model reads a similar intersection).

    Pinned as the EXACT set of overlapping pairs so the defect is visible in the suite: the
    fix (flip the fallback tick above the plot, or drop the colliding value label) must
    delete this test's expectation in the same PR, which is the point. A model of the
    browser's metrics: it proves WHICH pair collides, not the pixel count.
    """
    fixture = load("geometry/long_unit_220.json")
    out = _run(
        {
            "width": fixture["width"],
            "data": fixture["data"],
            "steps": [theme("n"), {"draw": fixture["draw"]}],
        },
        tmp_path,
    )
    texts = out["texts"]
    colliding = sorted(
        (a["text"], b["text"])
        for i, a in enumerate(texts)
        for b in texts[i + 1 :]
        if all(v > 0.5 for v in _overlap(a, b))
    )
    assert colliding == [("1,000 USD/day", "910")]


@needs_node
def test_the_script_inside_the_assembled_document_is_the_one_that_echoes_the_nonce(
    tmp_path: Path,
) -> None:
    """The nonce test above runs ``prelude.js`` from the package; this one runs the script
    a HOST actually receives -- extracted from the assembled populated document -- so a
    change to the assembler that altered or truncated the injected prelude fails here."""
    import re

    html = (FIXTURES / "documents" / "populated.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    injected = next(s for s in scripts if s.startswith('"use strict"'))
    assert injected == (PRELUDE_DIR / "prelude.js").read_text(encoding="utf-8")
    extracted = tmp_path / "injected.js"
    extracted.write_text(injected, encoding="utf-8")
    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        json.dumps({"steps": [theme("host-minted"), PING, {"size": True}, {"error": "boom"}]})
    )
    done = subprocess.run(
        ["node", str(HARNESS), str(extracted), str(scenario)],
        capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    posts = json.loads(done.stdout)["posts"]
    by_type = {p["t"]: p for p in posts}
    assert "n" not in by_type["ready"]
    assert [by_type[t]["n"] for t in ("pong", "resize", "error")] == ["host-minted"] * 3


@needs_node
def test_the_component_draws_no_title_anywhere(tmp_path: Path) -> None:
    """F9 (memo §8) + §2.8's no-chrome rule: a component is body-only, so the table must not
    draw its dataset title.

    This is the before/after discriminator for the `div.ttl` strip: `prelude.src.js` used to
    prepend `o.title || data.title` as a visible `div.ttl`, and the drawn nodes are inspected
    here rather than the source, so re-adding a title line by any route (helper, markup, a new
    class name that the CSS still styles) fails. The labelling the DATA needs stays: the
    column headers are asserted present, so the test cannot pass by drawing nothing.
    """
    title = "Latency by region (ms)"
    out = _run(
        {
            "data": {
                "lat": {
                    "title": title,
                    "columns": ["region", "ms"],
                    "rows": [["us-east", 120], ["eu-west", 143], ["ap-south", 211.25]],
                }
            },
            "steps": [
                theme("n"),
                {"draw": 'LO.table(document.getElementById("c"), "lat")'},
            ],
        },
        tmp_path,
    )
    nodes = out["nodes"]
    assert [n for n in nodes if n["cls"] == "ttl"] == [], "a title node class is drawn"
    assert title not in " ".join(n["text"] for n in nodes), "the dataset title is drawn"
    drawn = " ".join(n["text"] for n in nodes)
    for header in ("region", "ms"):
        assert header in drawn, f"column header {header!r} is not drawn"
    # The table is drawn INSIDE the figure; the only child the figure may hold is the scroll
    # wrap, never a title line before it.
    figure = next(n for n in nodes if n["tag"] == "figure")
    assert figure["text"] == ""


@needs_node
@pytest.mark.parametrize("values", [["$0.15", "$0.60", "$0.015"], [0.15, "N/A", 0.60]])
def test_assembled_prices_have_finite_bar_geometry(values: list[Any], tmp_path: Path) -> None:
    """Execute the shipped script with the exact data bytes a document serves."""
    source = {
        "d1": {
            "columns": ["name", "price"],
            "rows": [list(row) for row in zip(["Input", "Output", "Cached"], values)],
        }
    }
    html = document.assemble_document(
        "<script>LO.bar(document.getElementById('c'),'d1',{x:'name',y:'price'})</script>", source
    )
    match = re.search(r'<script type="application/json" id="lo-data">(.*?)</script>', html)
    assert match is not None
    out = _run(
        {
            "data": json.loads(match[1]),
            "steps": [
                {
                    "draw": (
                        "LO.bar(document.getElementById('c'),'d1',"
                        "{x:'name',y:'price',horizontal:true})"
                    )
                }
            ],
        },
        tmp_path,
    )
    bars = [n for n in out["nodes"] if n["tag"] == "rect"]
    assert len(bars) == (2 if "N/A" in values else 3)
    for bar in bars:
        for attr in ("x", "y", "width", "height"):
            assert float(bar["attrs"][attr]) >= 0
    assert all("NaN" not in str(n) for n in out["nodes"])
    assert any(n["cls"] == "grid" for n in out["nodes"])
    if "N/A" in values:
        assert any(n["text"] == "N/A" for n in out["nodes"])


@needs_node
def test_legacy_table_preserves_formatted_cells(tmp_path: Path) -> None:
    source = {"d1": {"columns": ["price", "share"], "rows": [["$0.15", "45%"]]}}
    script = "LO.table(document.getElementById('c'),'d1')"
    html = document.assemble_document(f'<div id="c"></div><script>{script}</script>', source)
    match = re.search(r'id="lo-data">(.*?)</script>', html)
    assert match is not None
    served = json.loads(match[1])
    assert served == source
    out = _run({"data": served, "steps": [{"draw": script}]}, tmp_path)
    cells = [n["text"] for n in out["nodes"] if n["tag"] == "td"]
    assert cells == ["$0.15", "45%"]


@needs_node
def test_null_bar_cell_still_paints_nothing(tmp_path: Path) -> None:
    out = _run(
        {
            "data": {"d1": {"columns": ["x", "y"], "rows": [["a", 1], ["b", None]]}},
            "steps": [{"draw": "LO.bar(document.getElementById('c'),'d1',{x:'x',y:'y'})"}],
        },
        tmp_path,
    )
    assert len([n for n in out["nodes"] if n["tag"] == "rect"]) == 1
    assert not any(n["text"] == "–" for n in out["nodes"])


@needs_node
@pytest.mark.parametrize("horizontal", [False, True])
def test_multiseries_gap_labels_share_bar_centres(horizontal: bool, tmp_path: Path) -> None:
    rows = [["a", 1, 2], ["b", "N/A", "P50"], ["c", 3, 4]]
    script = (
        "LO.bar(document.getElementById('c'),'d1',"
        + json.dumps({"x": "x", "y": ["y", "z"], "horizontal": horizontal})
        + ")"
    )
    out = _run(
        {"data": {"d1": {"columns": ["x", "y", "z"], "rows": rows}}, "steps": [{"draw": script}]},
        tmp_path,
    )
    bars = [n["attrs"] for n in out["nodes"] if n["tag"] == "rect"]
    gaps = [n["attrs"] for n in out["nodes"] if n["text"] in ("N/A", "P50")]
    axis, size = ("y", "height") if horizontal else ("x", "width")
    band = (float(bars[2][axis]) - float(bars[0][axis])) / 2
    for index, gap in enumerate(gaps):
        expected = float(bars[index][axis]) + band + (float(bars[index][size]) + 1) / 2
        if horizontal:
            expected += 4  # text baseline relative to the visual centre
        else:
            assert gap["text-anchor"] == "middle"
        assert float(gap[axis]) == pytest.approx(expected)


@needs_node
def test_all_string_line_column_is_labelled_but_null_is_not(tmp_path: Path) -> None:
    out = _run(
        {
            "data": {
                "d1": {"columns": ["x", "y"], "rows": [["a", "N/A"], ["b", "P50"], ["c", None]]}
            },
            "steps": [{"draw": "LO.line(document.getElementById('c'),'d1',{x:'x',y:'y'})"}],
        },
        tmp_path,
    )
    assert {"N/A", "P50"} <= {n["text"] for n in out["nodes"]}
    assert not any(n["text"] == "–" for n in out["nodes"])
    assert all(
        "NaN" not in str(n["attrs"]) and "Infinity" not in str(n["attrs"]) for n in out["nodes"]
    )


#: The stored component's own plot call: a horizontal bar of three prices, body-only.
_BAR = "LO.bar(document.getElementById('c'),'d1',{x:'name',y:'price',horizontal:true})"
_BAR_DATA = {
    "d1": {
        "columns": ["name", "price"],
        "rows": [["Input", 0.15], ["Output", 0.6], ["Cached input (prompt-cache read)", 0.015]],
    }
}


def _bar_scenario(data: dict[str, Any], width: int = 620, draw: str = _BAR) -> dict[str, Any]:
    return {"width": width, "data": data, "steps": [theme("n"), {"draw": draw}]}


@needs_node
def test_horizontal_bar_height_follows_the_bar_count_not_the_column_width(
    tmp_path: Path,
) -> None:
    """Operator 2026-10-19: a three-bar price chart at a wide column grew to 460 px because
    ``_frame`` derived the height from the width (``0.45 x W``), so the frame overflowed the
    UI's 480 px clamp and painted a scrollbar. Height is now a band per bar plus margins."""
    heights = {
        width: _run(_bar_scenario(_BAR_DATA, width), tmp_path)["svgH"] for width in (360, 620, 900)
    }
    assert len(set(heights.values())) == 1, heights
    assert heights[620] < 260, heights  # three bars, not a 0.45 aspect box

    six = {
        "d1": {
            "columns": ["name", "price"],
            "rows": [[f"row {i}", i / 6] for i in range(1, 7)],
        }
    }
    assert _run(_bar_scenario(six), tmp_path)["svgH"] > heights[620]


@needs_node
def test_horizontal_bar_height_is_capped_for_many_bars(tmp_path: Path) -> None:
    many = {
        "d1": {
            "columns": ["name", "price"],
            "rows": [[f"row {i}", i / 40] for i in range(1, 41)],
        }
    }
    out = _run(_bar_scenario(many, width=1024), tmp_path)
    assert 100 < out["svgH"] <= 380, out["svgH"]


@needs_node
def test_horizontal_bar_gutter_fits_the_longest_category_label(tmp_path: Path) -> None:
    """The fixed 120 px gutter truncated ``Cached input (prompt-cache read)`` (and a wider
    measured label could overflow it); the gutter is now the measured widest label."""
    out = _run(_bar_scenario(_BAR_DATA), tmp_path)
    nodes = [n for n in out["nodes"] if n["tag"] == "text"]
    assert [n["text"] for n in nodes if n["text"].startswith("Cached")] == [
        "Cached input (prompt-cache read)"
    ], "the label was truncated instead of fitting the measured gutter"
    drawn = [t for t in out["texts"] if t["text"].startswith("Cached")]
    assert drawn and all("…" not in t["text"] for t in drawn)
    assert all(t["left"] >= 0 for t in out["texts"]), "a label paint box starts outside the frame"


@needs_node
def test_value_labels_stay_off_the_fill_for_a_full_bar_and_a_tiny_one(tmp_path: Path) -> None:
    """Value labels are drawn OUTSIDE the bar end, in the canvas ink (`svg .lbl`), for the
    full bar and the 1.5%-of-max bar alike: a value never sits on a series fill, so it cannot
    become dark-on-blue. The operator's 2026-10-19 screenshot read dark-on-dark because the
    frame had no host theme push; the dark fallback below owns that case."""
    data = {
        "d1": {
            "columns": ["name", "price"],
            "rows": [["tiny", 0.015], ["full", 0.6]],
        }
    }
    out = _run(_bar_scenario(data), tmp_path)
    bars = [n["attrs"] for n in out["nodes"] if n["tag"] == "rect"]
    labels = {n["text"]: n["attrs"] for n in out["nodes"] if n["tag"] == "text"}
    for text, bar in (("0.015", bars[0]), ("0.6", bars[1])):
        assert labels[text].get("text-anchor", "start") == "start"
        assert float(labels[text]["x"]) >= float(bar["width"]) + 4, text


@needs_node
def test_the_wide_horizontal_plot_fits_the_480_px_frame_the_ui_clamps(tmp_path: Path) -> None:
    """The UI clamps a graphic's frame at 480 px. The document is the SVG scaled to the column
    (``svgH x column / svgW``) plus body padding; at a 1200 px column the old width-derived
    height overflowed, which is the scrollbar the operator reported."""
    width = 1200
    out = _run(_bar_scenario(_BAR_DATA, width), tmp_path)
    rendered = out["svgH"] * width / out["svgW"] + 16
    assert rendered <= 480, (rendered, out["svgH"], out["svgW"])
    assert out["svgH"] < 250


def test_the_dark_fallback_declares_ink_that_is_readable_on_its_own_ground() -> None:
    """A host that pushes no theme leaves `data-mode=dark` painting `--lo-canvas:#121212`.
    The light-theme ink defaults (``#1f1f1f`` / ``#4e4e4e`` / ``#666``) are then unreadable —
    the operator's 2026-10-19 screenshot showed the value and axis labels as dark-on-dark.
    The fallback must carry its own light ink; a real host push overrides these inline."""
    css = (PRELUDE_DIR / "prelude.css").read_text(encoding="utf-8")
    dark = _declarations(_DARK_RULE, css)
    assert dark["--lo-canvas"] == "#121212"
    assert dark["--lo-ink"] == "#e8e8e8"
    assert dark["--lo-ink-muted"] == "#b8b8b8"
