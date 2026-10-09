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
PINNED_CSS_BYTES = 2044
PINNED_JS_BYTES = 8202
PINNED_GZIP_BYTES = 4607
PINNED_DIGEST = "84545fe0244dc3ee4cb2b285bb55435d0f32926902769e448e3c4a6c3924aa36"
PINNED_VERSION = 1


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
