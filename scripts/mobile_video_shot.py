"""Capture the video-generation card out of headless Chrome over CDP.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_video_shot.py OUTDIR

Starts ``scripts/mobile_video_fixture.py`` (the REAL bundle over synthetic
projections carrying every video-card state), taps into each session the way a
thumb does, and photographs the card at 390x844 and 360x780 at dpr 2 — the two
phone viewports this surface is judged on — plus a geometry dump per frame: the
card's box, the tile's, the progress bar's, the CANCEL CONTROL's (which must be
ABSENT for video — ruling R4: the rail is post-turn, so the card withholds it),
the hold's, the state lines' text, the post-turn receipt's text when the session
carries one, and the body's scroll width. So "the control is withheld" and
"nothing overflows sideways" are measured statements rather than claims about a
PNG.

THE TAP IS ATTEMPTED, NOT ASSUMED. On the two live sessions the script presses
where the control WOULD be (the real gesture the imagegen rig uses) and records
the result — ``clicked: false`` is the absence, produced by a gesture rather
than by a selector that could be asking the wrong question.

ONE MOTION CHECK. On the running session a second frame ~450 ms later is taken
so the tile's shimmer is evidenced as MOTION (the two frames differ) rather
than asserted from one still.

``Chrome``/``Page`` are reused from ``scripts/mobile_overflow_capture.py``:
one throwaway ``--headless=new`` browser per run, its own profile,
``--use-mock-keychain``, and a teardown that asserts no helper survived — all
of which that module already got right, and a second copy here would be a
second thing to keep right. The taps are the REAL gesture for the same reason
the sibling rigs give: the app owns its routes, and a click is what a thumb
does.

THE PLAYER'S EVIDENCE IS ITS OWN TRAFFIC. On the two player rows the capture
holds the clip's request at the Fetch domain so the RESOLVING state is
photographed while genuinely unresolved, then releases it and lets the
browser's media stack fetch the bytes its own way — recording every
attachment request's Range header beside the status and Content-Range it got
back (206s are the point: phone Safari will not play a full-body 200). The
ready frame is taken only once ``readyState`` says metadata decoded and
``play()`` advanced the clock; the missing row is photographed only after
the element reported its error.
"""

from __future__ import annotations

import json
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from scripts.mobile_overflow_capture import Chrome, Page


class EventPage(Page):
    """The overflow rig's Page, with the CDP event stream KEPT.

    ``Page.send`` drains the websocket until the matching reply and discards
    everything else; for the player the EVENTS are the data — the browser's
    own media requests and the status each got — so this subclass stashes
    event messages instead of dropping them. Nothing else differs.
    """

    def __init__(self, ws_url: str) -> None:
        self.events: list[dict[str, Any]] = []
        super().__init__(ws_url)

    def send(self, method: str, **params: Any) -> dict[str, Any]:
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)


#: Both widths the card is judged at — the glance rig's own pair.
VIEWPORTS = ((390, 844), (360, 780))

#: The sessions to photograph, keyed by the row label the list shows. The value
#: is the prefix used for the frame's file name. Order is the reading order of
#: the state machine. ``queued-pos`` carries the queue datum AND the withheld
#: control; ``cancelling-wire`` is the feed-driven hold (``stage:
#: "cancelling"`` with no click — for video there is no control to press, so
#: the feed is the only source); ``receipt`` is the POST-TURN presence: the
#: ``job_result`` delivery folded to the notice row the phone renders.
TAP_SESSIONS: dict[str, str] = {
    "Video gen queued": "queued",
    "Video gen queue position": "queued-pos",
    "Video gen running": "running",
    "Video gen cancelling": "cancelling-wire",
    "Video gen done": "done",
    "Video gen failed": "failed",
    "Video gen already finished": "finished",
    "Video gen receipt": "receipt",
    "Video gen player": "player",
    "Video gen player missing": "player-missing",
}

#: The live sessions that would carry a cancel control if this were an image
#: card. The script presses where it would be and records ``clicked: false`` —
#: the withheld control as a gesture's own reading (ruling R4).
NO_CONTROL_ATTEMPTS = ("running", "queued-pos")

GEOMETRY_JS = r"""
(() => {
  const box = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {
      top: Math.round(r.top), bottom: Math.round(r.bottom),
      left: Math.round(r.left), right: Math.round(r.right),
      width: Math.round(r.width), height: Math.round(r.height),
    };
  };
  const text = (el) => (el && el.textContent ? el.textContent.trim() : null);
  const card = document.querySelector('[data-testid="image-gen-card"]');
  const buttons = Array.from(document.querySelectorAll('button'));
  const byText = (label) =>
    buttons.find((b) => (b.textContent || '').trim() === label) || null;
  const tile = document.querySelector('.lo-gen-tile');
  const bar = document.querySelector('[role="progressbar"]');
  const scroller = document.querySelector('.lo-scroll');
  const stateLines = card
    ? Array.from(card.querySelectorAll('p')).map((p) => text(p))
    : [];
  const hold = card ? card.querySelector('[data-testid="image-gen-hold"]') : null;
  if (hold) stateLines.push(text(hold));
  // The POST-TURN receipt row: the notice the job_result delivery folds to,
  // found by the frozen §5.1 sentence shape rather than by position.
  const receipt = Array.from(document.querySelectorAll('p'))
    .map((p) => text(p))
    .find((t) => t && /background job '.*' completed:/.test(t)) || null;
  return JSON.stringify({
    card: box(card),
    row: box(card ? card.querySelector('button') : null),
    tile: box(tile),
    bar: bar
      ? {
          box: box(bar),
          valueNow: bar.getAttribute('aria-valuenow'),
          label: bar.getAttribute('aria-label'),
        }
      : null,
    // MUST be null on every video frame: the control is withheld (R4).
    cancel: box(byText('cancel')),
    hold: box(hold),
    stateLines,
    receipt,
    // The player (the bounded follow-up, §3.2): null where no mount exists;
    // on the player rows the `data-state` attribute is the still's own
    // reading of resolving/ready/error, and readyState/currentTime are the
    // browser's own evidence that the clip DECODED and ADVANCED — the
    // bytes came through the real route, rung by rung.
    player: (() => {
      const el = document.querySelector('[data-testid="attachment-video"]');
      const vid = el ? el.querySelector('video') : null;
      return el
        ? {
            box: box(el),
            state: el.getAttribute('data-state'),
            readyState: vid ? vid.readyState : null,
            currentTime: vid ? Math.round((vid.currentTime || 0) * 100) / 100 : null,
            src: vid ? vid.getAttribute('src') : null,
          }
        : null;
    })(),
    hasPlayer: document.querySelector('video') !== null,
    imageCount: card ? card.querySelectorAll('img').length : 0,
    scroller: box(scroller),
    bodyScrollW: document.body.scrollWidth,
    innerW: window.innerWidth,
  });
})()
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _tap_js(label: str) -> str:
    """Tap the list row whose text the label names — the REAL gesture."""
    return (
        "(() => {"
        " const nodes = Array.from("
        'document.querySelectorAll(\'a,button,[role="link"],[role="button"]\'));'
        f" const hit = nodes.find((n) => (n.textContent || '').includes({label!r}));"
        " if (!hit) return JSON.stringify({ tapped: false });"
        " hit.click();"
        " return JSON.stringify({ tapped: true, tag: hit.tagName });"
        "})()"
    )


def _click_button_js(label: str) -> str:
    """Press the in-card control whose text the label names — the REAL gesture.

    On a video card there is no such control; the caller records
    ``clicked: false`` as the withheld-control evidence.
    """
    return (
        "(() => {"
        " const hit = Array.from(document.querySelectorAll('button'))"
        f".find((n) => (n.textContent || '').trim() === {label!r});"
        " if (!hit) return JSON.stringify({ clicked: false });"
        " hit.click();"
        " return JSON.stringify({ clicked: true });"
        "})()"
    )


def _wait_for_event(
    page: "EventPage", method: str, start: int, seconds: float = 10.0
) -> dict[str, Any] | None:
    """The first ``method`` event seen since index ``start``, consumed."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        for i in range(start, len(page.events)):
            event = page.events[i]
            if event.get("method") == method:
                page.events.pop(i)
                return event
        time.sleep(0.15)
    return None


def _response_header(headers: dict[str, Any], name: str) -> Any:
    """One response header's value, looked up case-insensitively.

    HTTP header names are case-insensitive by definition, and CDP preserves
    whatever case reached the wire — a fixed-case lookup reads a MISS as an
    absent header (round-1 review, N2: ``accept-ranges`` read as null on a
    206 that demonstrably carried it). The raw dict also rides the artifact
    so the question is answerable from the JSON rather than re-derived.
    """
    wanted = name.lower()
    for key, value in (headers or {}).items():
        if key.lower() == wanted:
            return value
    return None


def _attachment_requests(page: "EventPage", start: int) -> list[dict[str, Any]]:
    """The browser's OWN attachment requests and the status each got back.

    The player's whole point on a phone is Safari's byte-Range requirement,
    so the evidence is the media element's own traffic: each request's Range
    header beside the response's status and Content-Range. A 200-only
    reading would be the failure this rig exists to catch.
    """
    requests: dict[str, dict[str, Any]] = {}
    for event in page.events[start:]:
        method = event.get("method")
        params = event.get("params") or {}
        if method == "Network.requestWillBeSent":
            url = (params.get("request") or {}).get("url") or ""
            if "/attachment?" in url:
                requests[params.get("requestId") or url] = {
                    "url": url,
                    "range": ((params.get("request") or {}).get("headers") or {}).get("Range"),
                }
        elif method == "Network.responseReceived":
            url = (params.get("response") or {}).get("url") or ""
            if "/attachment?" in url:
                entry = requests.setdefault(params.get("requestId") or url, {"url": url})
                response = params.get("response") or {}
                headers = response.get("headers") or {}
                entry["status"] = response.get("status")
                entry["content_range"] = _response_header(headers, "Content-Range")
                entry["accept_ranges"] = _response_header(headers, "Accept-Ranges")
                entry["headers"] = headers
    return list(requests.values())


def _capture_player(
    page: "EventPage", outdir: Path, label: str, frame: str, *, missing: bool
) -> dict[str, Any]:
    """The player's states, captured through the real relay.

    READY: the clip's request is HELD at the Fetch domain, so the resolving
    frame (reserved box, no spinner) is genuinely unresolved when taken;
    releasing it lets the browser's media stack fetch the bytes its own way
    (the Range requests recorded below), and the ready frame waits for
    ``readyState`` to decode metadata and ``play()`` to advance.

    MISSING: nothing is seeded behind the ref, so the route's 404 must land
    as the designed ``video unavailable`` state — photographed only after
    the element reports its error.
    """
    start = len(page.events)
    if missing:
        entry: dict[str, Any] = {"tap": json.loads(page.js(_tap_js(label)))}
        deadline = time.time() + 10
        while time.time() < deadline:
            state = page.js(
                "(() => { const el = document.querySelector('[data-testid=\"attachment-video\"]');"
                " return el ? el.getAttribute('data-state') : null; })()"
            )
            if state == "error":
                break
            time.sleep(0.25)
        entry.update(json.loads(page.js(GEOMETRY_JS)))
        page.shot(outdir / f"{frame}.png")
        entry["media"] = _attachment_requests(page, start)
        return entry

    page.send("Fetch.enable", patterns=[{"urlPattern": "*kind=video*", "requestStage": "Request"}])
    entry = {"tap": json.loads(page.js(_tap_js(label)))}
    paused = _wait_for_event(page, "Fetch.requestPaused", start, seconds=12)
    time.sleep(0.4)
    entry.update(json.loads(page.js(GEOMETRY_JS)))
    entry["resolving_state"] = (entry.get("player") or {}).get("state")
    page.shot(outdir / f"{frame}.png")
    if paused is not None:
        page.send("Fetch.continueRequest", requestId=paused["params"]["requestId"])
    page.send("Fetch.disable")

    deadline = time.time() + 12
    while time.time() < deadline:
        ready = page.js(
            "(() => { const v = document.querySelector('video');"
            " return !!(v && v.readyState >= 1); })()"
        )
        if ready:
            break
        time.sleep(0.3)
    page.js(
        "(() => { const v = document.querySelector('video');"
        " if (v) { const p = v.play(); if (p && p.catch) p.catch(() => {}); } return true; })()"
    )
    time.sleep(1.2)
    entry["ready"] = json.loads(page.js(GEOMETRY_JS))
    page.shot(outdir / f"{frame}-ready.png")
    # PLAY IT WITH A HAND, not a scripted call: a synthetic click at the
    # element is REAL user activation, so the unmuted play the product ships
    # is exactly what starts — an autoplay-policy rejection of the JS play()
    # above is not the app failing, and the frame must not pretend otherwise.
    ready_box = ((entry.get("ready") or {}).get("player") or {}).get("box")
    if ready_box:
        cx = ready_box["left"] + ready_box["width"] / 2
        cy = ready_box["top"] + ready_box["height"] / 2
        page.send(
            "Input.dispatchMouseEvent",
            type="mousePressed",
            x=cx,
            y=cy,
            button="left",
            clickCount=1,
        )
        page.send(
            "Input.dispatchMouseEvent",
            type="mouseReleased",
            x=cx,
            y=cy,
            button="left",
            clickCount=1,
        )
    time.sleep(0.7)
    entry["playing"] = json.loads(page.js(GEOMETRY_JS))
    entry["media"] = _attachment_requests(page, start)
    return entry


def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    # A THROWAWAY PASSWORD, GENERATED PER RUN AND NEVER PRINTED — the same
    # contract the sibling rigs document: passed to the fixture as an argument
    # and typed into the login form below, so it appears in no file and in no
    # output.
    password = secrets.token_urlsafe(16)
    fixture = subprocess.Popen(
        [sys.executable, "scripts/mobile_video_fixture.py", str(port), password],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    chrome = Chrome()
    report: dict[str, Any] = {}
    try:
        # Wait for the fixture to answer before pointing a browser at it: an
        # immediate connect gives a refused socket, which reads like a broken
        # bundle rather than a race.
        deadline = time.time() + 45
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/login", timeout=2):
                    break
            except Exception:  # noqa: BLE001  -- not up yet
                time.sleep(0.4)
        else:
            raise RuntimeError("fixture never served /login")

        page = EventPage(chrome.target_ws())
        page.metrics(*VIEWPORTS[0])
        page.goto(f"{base}/login")
        page.js(
            "(() => { const f = document.querySelector('form');"
            f" f.password.value = {password!r}; f.submit(); return true; }})()"
        )
        time.sleep(2.0)
        # The media evidence is network traffic, so the event stream must be on
        # before the first player row renders.
        page.send("Network.enable")
        for vw, vh in VIEWPORTS:
            page.metrics(vw, vh)
            page.goto(f"{base}/#/")
            time.sleep(1.5)
            for label, prefix in TAP_SESSIONS.items():
                frame = f"{prefix}-{vw}x{vh}"
                if prefix in ("player", "player-missing"):
                    report[frame] = _capture_player(
                        page, outdir, label, frame, missing=prefix == "player-missing"
                    )
                    print(f"{frame}: {json.dumps(report[frame], indent=1)}", flush=True)
                    page.goto(f"{base}/#/")
                    time.sleep(1.5)
                    continue
                report[frame] = {"tap": json.loads(page.js(_tap_js(label)))}
                time.sleep(2.0)
                report[frame].update(json.loads(page.js(GEOMETRY_JS)))
                first = outdir / f"{frame}.png"
                page.shot(first)
                if prefix in NO_CONTROL_ATTEMPTS:
                    # THE WITHHELD CONTROL, AS A GESTURE'S OWN READING: press
                    # where the imagegen card's cancel would be. ``clicked:
                    # false`` is the evidence ruling R4 asks for — and the
                    # geometry dump beside it shows the tap landed nowhere
                    # because there is nothing there.
                    report[frame]["cancel_press"] = json.loads(page.js(_click_button_js("cancel")))
                if prefix == "running":
                    # Frame B exists to show the tile MOVES: a shimmer is
                    # motion, and one still can only show a phase. The flag
                    # below is the WHOLE-frame comparison — any animation in
                    # the viewport moves it.
                    time.sleep(0.45)
                    second = outdir / f"{frame}-b.png"
                    page.shot(second)
                    report[frame]["frames_differ"] = first.read_bytes() != second.read_bytes()
                print(f"{frame}: {json.dumps(report[frame], indent=1)}", flush=True)
                # Back to the list for the next session — the route the app spells.
                page.goto(f"{base}/#/")
                time.sleep(1.5)
        page.close()
    finally:
        # TEARDOWN IS BEST-EFFORT PER RESOURCE, and the first failure is
        # re-raised after both are attempted (the imagegen rig's review round
        # 1, F1). The Chrome teardown fails CLOSED — it raises when a helper
        # outlived the run — and a bare ``chrome.close()`` line skipped the
        # fixture's own teardown below it on exactly that alert path, stranding
        # a live daemon (loopback-only and isolated, but still a process nobody
        # collects). Each resource is reclaimed whatever the other did.
        failure: Exception | None = None
        try:
            chrome.close()
        except Exception as exc:  # noqa: BLE001 — re-raised after both claims
            failure = exc
        try:
            fixture.terminate()
            try:
                fixture.wait(timeout=10)
            except subprocess.TimeoutExpired:
                fixture.kill()
        except Exception as exc:  # noqa: BLE001
            if failure is None:
                failure = exc
            else:
                print(f"teardown: second failure: {exc!r}", file=sys.stderr)
        if failure is not None:
            raise failure
    (outdir / "video-mobile-geometry.json").write_text(json.dumps(report, indent=2))
    print("shots:", ", ".join(f"{key}.png" for key in report))


if __name__ == "__main__":
    main()
