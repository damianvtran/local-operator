"""Per-locale rendered evidence for the relay SPA on #3165, out of headless Chrome.

Measurement only: this file lives in the session scratchpad and the repo is not
edited. Run:

    TMPDIR="$SCRATCH" <worktree>/.venv/bin/python locale_frames_capture.py \
        --outdir "$SCRATCH/frames" --password-file "$SCRATCH/pw" --plan "$SCRATCH/plan.json"

WHAT IT RECORDS. For each case it drives the REAL built SPA served by the REAL
relay daemon (locale_fixture.py) and collects, for every animation frame from
document start:

  * the visible text of ``#root`` (first 240 chars),
  * ``document.documentElement.lang`` and ``dir`` at that instant,
  * the frame's timestamp from ``performance.now()``.

That is what makes "no English frame" falsifiable: the frames are the browser's
own paint opportunities, sampled from inside the page, not a poll taken after
the fact. ``firstContent`` is the first frame whose root text was non-empty.

INSTRUMENT DISCIPLINE (AGENTS.md §6, and the imported ``Chrome`` driver follows
it): one browser for the whole run, unique profile under TMPDIR,
``--headless=new``, ``--remote-debugging-port=0`` read from
``DevToolsActivePort``, viewport via ``Emulation.setDeviceMetricsOverride``
(never ``--window-size``), process-group teardown with an atexit sweep that
ASSERTS zero leftovers.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import os as _os
REPO = Path(_os.environ["LF_REPO"])
sys.path.insert(0, str(REPO))

from scripts.mobile_overflow_capture import Chrome  # noqa: E402

VIEWPORT = (390, 844)

#: Installed BEFORE any page script runs. One entry per animation frame, so the
#: first content frame is an observed paint and the language recorded beside it
#: is what the document declared at that instant.
SAMPLER = r"""
(() => {
  const LF = (window.__LF = { frames: [], firstContent: null, paints: null });
  // COST MATTERS: `innerText` forces a layout, so computing it on every frame
  // throttles the very page under measurement — a first version did exactly
  // that and starved the list's SSE frame past the settle window, which reads
  // as "the list is empty" rather than as a slow harness. `textContent` is a
  // cheap string read; the expensive snapshot is taken only on frames where the
  // text, the language or the direction actually changed, which are the frames
  // the evidence is about.
  let lastLen = -1;
  let lastLang = null;
  let lastDir = null;
  let lastText = '';
  let n = 0;
  const sample = () => {
    const t = Math.round(performance.now());
    const root = document.getElementById('root');
    const len = root ? (root.textContent || '').length : 0;
    const lang = document.documentElement.lang || null;
    const dir = document.documentElement.dir || null;
    const changed = len !== lastLen || lang !== lastLang || dir !== lastDir;
    if (changed) {
      lastText = root ? (root.innerText || '').replace(/\s+/g, ' ').trim() : '';
      lastLen = len;
      lastLang = lang;
      lastDir = dir;
      if (LF.firstContent === null && lastText.length > 0) {
        LF.firstContent = { t, lang, dir, text: lastText.slice(0, 240) };
      }
    }
    LF.frames.push({ t, lang, dir, n: len, changed, text: lastText.slice(0, 240) });
    if (++n < 1200) requestAnimationFrame(sample);
  };
  requestAnimationFrame(sample);
  window.addEventListener('load', () => {
    LF.paints = performance.getEntriesByType('paint')
      .map(e => ({ name: e.name, t: Math.round(e.startTime) }));
  });
})()
"""

POLL_JS = """(() => {
  const root = document.getElementById('root');
  return root ? (root.textContent || '').trim().length > 0 : false;
})()"""

POLL_TEXT_JS = """(() => {
  const root = document.getElementById('root');
  return root ? (root.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 200) : '';
})()"""

READ_JS = r"""
(() => {
  const root = document.getElementById('root');
  const text = root ? (root.innerText || '') : '';
  const lf = window.__LF || {};
  return JSON.stringify({
    lang: document.documentElement.lang || null,
    dir: document.documentElement.dir || null,
    text: text.replace(/\s+/g, ' ').trim(),
    frames: lf.frames || [],
    firstContent: lf.firstContent || null,
    paints: lf.paints || null,
    readyState: document.readyState,
  });
})()
"""


class Page:
    """CDP client with an event pump: requests are serialized, events buffered.

    A reader thread drains the socket into a queue so ``send`` can wait for its
    own reply while the events that arrive in between survive; without that, the
    Network events that carry the chunk evidence are swallowed by the very call
    that navigates.
    """

    def __init__(self, ws_url: str) -> None:
        from websockets.sync.client import connect

        self.ws = connect(ws_url, max_size=80 * 1024 * 1024)
        self.n = 0
        self.events: list[dict[str, Any]] = []
        self._inbox: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.send("Page.enable")
        self.send("Runtime.enable")
        self.send("Network.enable")

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                message = self.ws.recv()
            except Exception:
                self._inbox.put(None)
                return
            self._inbox.put(json.loads(message))

    def send(self, method: str, **params: Any) -> dict[str, Any]:
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                msg = self._inbox.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                break
            if msg is None:
                raise RuntimeError(f"{method}: socket closed")
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)
        raise RuntimeError(f"{method}: no reply")

    def pump(self, seconds: float) -> None:
        """Collect events for a while without issuing a request."""
        deadline = time.time() + seconds
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            try:
                msg = self._inbox.get(timeout=remaining)
            except queue.Empty:
                return
            if msg is None:
                return
            self.events.append(msg)

    def settle(
        self, *, minimum_s: float = 0.0, quiet_ms: int = 900, cap_s: float = 12.0
    ) -> dict[str, Any]:
        """Wait until the rendered text stops changing, then report how it went.

        Returns the number of consecutive quiet milliseconds observed and the
        total wait, so a reader can tell a genuinely settled screen from one that
        hit the cap while still moving.
        """
        time.sleep(minimum_s)
        started = time.time()
        last = None
        quiet_since = time.time()
        samples = 0
        while time.time() - started < cap_s:
            samples += 1
            current = self.js(
                "(() => { const r = document.getElementById('root');"
                " return JSON.stringify([r ? r.textContent.length : 0,"
                " document.documentElement.lang || null]); })()"
            )
            now = time.time()
            if current != last:
                last = current
                quiet_since = now
            elif (now - quiet_since) * 1000 >= quiet_ms:
                return {
                    "quietMs": round((now - quiet_since) * 1000),
                    "waitedMs": round((now - started) * 1000),
                    "samples": samples,
                    "capped": False,
                }
            time.sleep(0.2)
        return {
            "quietMs": 0,
            "waitedMs": round((time.time() - started) * 1000),
            "samples": samples,
            "capped": True,
        }

    def metrics(self, width: int, height: int) -> None:
        self.send(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=2,
            mobile=True,
        )

    def cold_goto(self, url: str) -> None:
        """A genuinely cold load: the document is replaced, the sampler re-runs.

        ``Page.navigate`` alone reuses the same document for a hash-only change,
        which would leave the previous run's frames in place and quietly turn a
        cold-paint claim into a warm one.
        """
        self.send("Page.navigate", url="about:blank")
        time.sleep(0.15)
        self.send("Page.navigate", url=url)

    def js(self, expression: str) -> Any:
        result = self.send(
            "Runtime.evaluate",
            expression=expression,
            awaitPromise=True,
            returnByValue=True,
        )
        return result.get("result", {}).get("value")

    def shot(self, path: Path) -> None:
        data = self.send("Page.captureScreenshot", format="png")["data"]
        path.write_bytes(base64.b64decode(data))

    def close(self) -> None:
        self._stop.set()
        try:
            self.ws.close()
        except Exception:
            pass


def new_tab(chrome: "Chrome") -> tuple["Page", str]:
    """A FRESH document for one capture, in a new tab of the same browser.

    WHY NOT REUSE THE TAB. Back-to-back navigations in one tab are not cold loads
    in practice: measured here, the third capture in a row painted its first
    frame in ENGLISH under ``lang="en"`` while the same screen as a first load
    painted the locale at 293 ms. The previous document's streams are still
    draining server-side when the next boot asks for its baseline config frame,
    which can push that frame past the 1500 ms budget — a harness-induced slow
    stream, not a product behaviour, and reporting it as one would be a false
    finding. One tab per capture removes the coupling; the BROWSER is still a
    single instance for the whole run, as rig hygiene requires.
    """
    request = urllib.request.Request(
        f"http://127.0.0.1:{chrome.port}/json/new?about:blank", method="PUT"
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        target = json.load(response)
    return Page(target["webSocketDebuggerUrl"]), target["id"]


def close_tab(chrome: "Chrome", target_id: str) -> None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{chrome.port}/json/close/{target_id}", timeout=20
        ) as response:
            response.read()
    except Exception:
        pass


class Fixture:
    """One relay daemon for one case, owned and reaped by this process.

    WHY IT LIVES HERE. Killing the pid of a `cd ... && python ... &` job kills the
    SUBSHELL and orphans uvicorn — measured here: four stale fixture servers
    stayed bound to their ports after their "teardown", and a later case then
    talked to a daemon from an earlier run whose caches were warm-empty. That
    reads exactly like a product defect (the list painting "no conversations"
    beside a working deep link) and cost a debugging round. Spawning with
    `start_new_session=True` makes the daemon its own process group; teardown by
    that group is exact, and it ASSERTS the port is closed before the next case
    starts.
    """

    def __init__(
        self,
        *,
        port: int,
        locale: str,
        mode: str,
        password_file: Path,
        log: Path,
        slow_chunk: str | None = None,
        slow_ms: int = 2500,
    ) -> None:
        self.port = port
        self.log = log
        command = [
            _os.environ.get("LF_PY", str(REPO / ".venv/bin/python")),
            str(Path(__file__).parent / "locale_fixture.py"),
            "--port", str(port),
            "--locale", locale,
            "--mode", mode,
            "--password-file", str(password_file),
        ]
        if slow_chunk:
            command += ["--slow-chunk", slow_chunk, "--slow-ms", str(slow_ms)]
        self.command = command
        self.proc: subprocess.Popen | None = None
        self._out = None

    def __enter__(self) -> "Fixture":
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self._out = self.log.open("wb")
        self.proc = subprocess.Popen(
            self.command,
            stdout=self._out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ},
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            if self._port_open():
                print(f"fixture up: port={self.port}", flush=True)
                return self
            if self.proc.poll() is not None:
                break
            time.sleep(0.25)
        detail = self.log.read_text(errors="replace")[-1500:] if self.log.exists() else ""
        self.__exit__(None, None, None)
        raise RuntimeError(f"fixture on {self.port} never served; log tail:\n{detail}")

    def _port_open(self) -> bool:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            return probe.connect_ex(("127.0.0.1", self.port)) == 0

    def __exit__(self, *exc: object) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=10)
        # The port must be CLOSED before the next case can be trusted: a survivor
        # is another daemon answering this case's requests.
        deadline = time.time() + 20
        while time.time() < deadline and self._port_open():
            time.sleep(0.2)
        if self._port_open():
            raise RuntimeError(f"fixture on {self.port} still listening after teardown")
        if self._out is not None:
            self._out.close()


def login(base: str, password: str) -> str:
    """The relay's own password gate; returns the session cookie header value."""
    body = urllib.parse.urlencode({"password": password}).encode()
    request = urllib.request.Request(
        f"{base}/login",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    # The redirect target is the SPA, which a urllib opener would then try to
    # parse; the cookie arrives on the 303 instead, so a no-redirect opener reads
    # exactly the header this needs and never fetches the bundle.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):  # noqa: D102
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        response = opener.open(request, timeout=10)
        headers = response.headers
    except urllib.error.HTTPError as exc:  # a 303 surfaces here under NoRedirect
        headers = exc.headers
    cookies = headers.get_all("Set-Cookie") or []
    if not cookies:
        raise RuntimeError(f"login at {base} set no cookie")
    return "; ".join(cookie.split(";")[0] for cookie in cookies)


def config_frame(base: str, cookie: str) -> dict[str, Any]:
    """The baseline `config` frame the daemon actually serves for this run.

    This is layer 1 of the evidence: the SERVER resolved this locale off the
    config file it was given. The frame's `language` is what the SPA renders.
    """
    request = urllib.request.Request(
        f"{base}/api/config/events",
        headers={"Accept": "text/event-stream", "Cookie": cookie},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if "text/event-stream" not in (response.headers.get("content-type") or ""):
            raise RuntimeError(f"config stream content-type was {response.headers.get('content-type')!r}")
        deadline = time.time() + 5
        buffer = ""
        while time.time() < deadline:
            chunk = response.read(1)
            if not chunk:
                break
            buffer += chunk.decode()
            if "\n\n" in buffer:
                for line in buffer.split("\n"):
                    if line.startswith("data:"):
                        return json.loads(line[5:].strip())
        raise RuntimeError("config stream produced no data line")


RESOURCE_JS = r"""
(() => {
  const out = [];
  for (const e of performance.getEntriesByType('resource')) {
    if (e.name.indexOf('bundled.') === -1 && e.name.indexOf('/api/') === -1) continue;
    out.push({
      path: e.name.split(location.origin)[1] || e.name,
      name: e.name.split('/').slice(-1)[0],
      startTime: Math.round(e.startTime),
      responseEnd: Math.round(e.responseEnd),
      transferSize: e.transferSize,
      encodedBodySize: e.encodedBodySize,
      decodedBodySize: e.decodedBodySize,
      status: e.responseStatus === undefined ? null : e.responseStatus,
    });
  }
  return JSON.stringify(out);
})()
"""


def chunk_requests(page: Page) -> list[dict[str, Any]]:
    """Every locale-chunk request this case made, with status and byte size."""
    started: dict[str, dict[str, Any]] = {}
    finished: dict[str, int] = {}
    for event in page.events:
        method = event.get("method")
        params = event.get("params", {})
        if method == "Network.responseReceived":
            response = params.get("response", {})
            url = response.get("url", "")
            if "bundled." not in url or not url.endswith(".js"):
                continue
            started[params.get("requestId", "")] = {
                "url": url.split("/")[-1],
                "status": response.get("status"),
                "mimeType": response.get("mimeType"),
                "fromDiskCache": response.get("fromDiskCache", False),
                "fromServiceWorker": response.get("fromServiceWorker", False),
            }
        elif method == "Network.loadingFinished":
            request_id = params.get("requestId")
            if request_id in started:
                finished[request_id] = params.get("encodedDataLength", -1)
    out = []
    for request_id, row in started.items():
        out.append({**row, "encodedDataLength": finished.get(request_id, -1)})
    return out


def capture(
    page: Page,
    label: str,
    url: str,
    outdir: Path,
    *,
    settle_s: float = 3.2,
    mid_ms: int | None = None,
    shot_first: int | None = None,
) -> dict[str, Any]:
    """One cold load, the frames in between, and a settled still.

    A FIXED SLEEP IS NOT A SETTLE. The list paints its loading chrome at ~0.7 s
    and takes the first `sessions` frame up to ~2.9 s later, so a 3.4 s sleep
    sometimes photographed the empty state and sometimes the list — measured
    here, twice, on the same build. The wait below therefore waits for the root
    to STOP CHANGING (900 ms of no change, 12 s cap) and the still is taken from
    that state; the frame timeline is still recorded from document start, so the
    first-paint claim is unaffected.
    """
    page.events.clear()
    marks: dict[str, Any] = {"label": label, "url": url}
    started = time.monotonic()
    page.cold_goto(url)
    elapsed = time.monotonic() - started
    marks["firstContentShot"] = None
    if shot_first is not None:
        # Photograph the FIRST frame that has any content, the instant it appears:
        # polling every 25 ms rather than sleeping to an average, because the
        # window between "nothing yet" and "the projection arrived" is a few
        # hundred milliseconds and a sleep lands outside it as often as inside.
        deadline = time.monotonic() + (shot_first / 1000)
        seen = page.js(POLL_JS)
        while seen is False and time.monotonic() < deadline:
            time.sleep(0.025)
            seen = page.js(POLL_JS)
        page.shot(outdir / f"{label}-first-content.png")
        marks["firstContentShot"] = {
            "seenAt": round((time.monotonic() - started) * 1000),
            "text": page.js(POLL_TEXT_JS),
            "lang": page.js("document.documentElement.lang || null"),
            "dir": page.js("document.documentElement.dir || null"),
        }
    mid = mid_ms if isinstance(mid_ms, list) else ([] if mid_ms is None else [mid_ms])
    marks["mids"] = []
    for at in mid:
        time.sleep(max(0.0, at / 1000 - elapsed))
        page.shot(outdir / f"{label}-at{at}.png")
        marks["mids"].append(
            {
                "at": at,
                "text": page.js(
                    "(() => { const r = document.getElementById('root');"
                    " return r ? (r.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 200) : ''; })()"
                ),
                "lang": page.js("document.documentElement.lang || null"),
            }
        )
    elapsed = time.monotonic() - started
    marks["settle"] = page.settle(minimum_s=max(0.0, settle_s - elapsed))
    page.pump(0.4)
    data = json.loads(page.js(READ_JS))
    page.shot(outdir / f"{label}-settled.png")
    marks.update(data)
    marks["chunks"] = chunk_requests(page)
    # The PAGE's own resource timeline, on the same clock as the frames: the
    # chunk's start/end relative to the first content frame is what makes the
    # budget claim checkable rather than asserted.
    marks["resources"] = json.loads(page.js(RESOURCE_JS))
    return marks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--password-file", required=True)
    parser.add_argument("--plan", required=True, help="JSON plan file: list of cases")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    password = Path(args.password_file).read_text().strip()
    plan = json.loads(Path(args.plan).read_text())

    chrome = Chrome()
    page = None
    try:
        page = Page(chrome.target_ws())
        page.metrics(*VIEWPORT)
        page.send("Network.setCacheDisabled", cacheDisabled=True)
        page.send("Page.addScriptToEvaluateOnNewDocument", source=SAMPLER)
        results = []
        for case in plan:
            with Fixture(
                port=case["port"],
                locale=case["locale"],
                mode=case["mode"],
                password_file=Path(args.password_file),
                log=outdir / f"fixture-{case['locale']}-{case['mode']}.log",
                slow_chunk=case.get("slow_chunk"),
                slow_ms=case.get("slow_ms", 2500),
            ):
                results.append(
                    _run_case(
                        page,
                        chrome,
                        case,
                        f"http://127.0.0.1:{case['port']}",
                        password,
                        outdir,
                    )
                )
                print(
                    f"captured {case['locale']}/{case['mode']}: "
                    f"{len(results[-1]['screens'])} screen(s)",
                    flush=True,
                )
                (outdir / "results.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=1)
                )
        (outdir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"wrote {outdir / 'results.json'}", flush=True)
    finally:
        if page is not None:
            page.close()
        chrome.close()


def _run_case(
    page: "Page",
    chrome: "Chrome",
    case: dict[str, Any],
    base: str,
    password: str,
    outdir: Path,
) -> dict[str, Any]:
    """Capture every screen of one case against a live fixture."""
    cookie = login(base, password)
    # The gate protects `/` as well as `/api/*`, so the cookie has to be in the
    # browser before the first navigation — otherwise the frames would be the
    # login page's, which is a different document.
    for pair in cookie.split("; "):
        name, value = pair.split("=", 1)
        page.send(
            "Network.setCookie",
            name=name,
            value=value,
            domain="127.0.0.1",
            path="/",
        )
    frame = config_frame(base, cookie)
    captured = []
    for screen in case["screens"]:
        tab, target_id = new_tab(chrome)
        try:
            tab.metrics(*VIEWPORT)
            tab.send("Network.setCacheDisabled", cacheDisabled=True)
            tab.send("Page.addScriptToEvaluateOnNewDocument", source=SAMPLER)
            captured.append(
                capture(
                    tab,
                    # A case may carry a `tag` so a CONTROL that reuses a locale and
                # mode (the held-chunk run) does not overwrite the evidence for
                # the ordinary one — measured: it did, on the first full pass.
                f"{case.get('tag') or case['locale']}-{case['mode']}-{screen['name']}",
                    f"{base}/#{screen['hash']}",
                    outdir,
                    settle_s=screen.get("settle_s", 5.0),
                    mid_ms=screen.get("mid_ms"),
                    shot_first=screen.get("shot_first"),
                )
            )
        finally:
            tab.close()
            close_tab(chrome, target_id)
    return {
        "locale": case["locale"],
        "mode": case["mode"],
        "port": case["port"],
        "slowChunk": case.get("slow_chunk"),
        "configFrame": frame,
        "screens": captured,
    }


if __name__ == "__main__":
    main()
