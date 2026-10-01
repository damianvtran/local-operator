import json, os, secrets, signal, subprocess, sys, time, urllib.request
from pathlib import Path
WT = os.environ["WT"]; SP = os.environ["SP"]; OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, WT)
from scripts.mobile_overflow_capture import Chrome, Page   # reuse the established rig
import socket
def free():
    with socket.socket() as s: s.bind(("127.0.0.1", 0)); return s.getsockname()[1]
env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
env["PYTHONPATH"] = ""
procs = []
def serve(tree):
    port, pw = free(), secrets.token_urlsafe(12)
    p = subprocess.Popen([f"{WT}/.venv/bin/python", f"{SP}/fixture.py", tree, str(port), pw], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=tree, start_new_session=True)
    procs.append(p)
    for _ in range(100):
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/login", timeout=2); break
        except Exception: time.sleep(0.4)
    else: raise SystemExit("fixture did not start: " + (p.stdout.read() if p.poll() is not None else ""))
    return port, pw
chrome = None
try:
    sides = {"before": serve(f"{SP}/before_tree"), "after": serve(WT)}
    chrome = Chrome(); page = Page(chrome.target_ws()); page.metrics(390, 844)
    page.send("Emulation.setTouchEmulationEnabled", enabled=True)
    report = {}
    def login(side):
        port, pw = sides[side]
        page.goto(f"http://127.0.0.1:{port}/login")
        page.js("localStorage.clear()")
        page.js(f"(()=>{{const i=document.querySelector('input[type=password]');i.value={pw!r};i.form.submit();}})()")
        time.sleep(2.0)
    def settle(s=1.2): time.sleep(s)
    for side in ("before", "after"):
        port, _ = sides[side]; base = f"http://127.0.0.1:{port}"; login(side)
        # --- #1869: the sheet
        page.goto(f"{base}/#/s/ev-live"); settle(2)
        page.js("""(()=>{const t=document.querySelector('textarea');const set=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set;set.call(t,'/');t.dispatchEvent(new Event('input',{bubbles:true}));})()""")
        settle(1.5)
        listing = page.js("JSON.stringify(Array.from(document.querySelectorAll('[role=dialog] button')).map(b=>b.textContent.trim().split(/\\s+/)[0]).filter(x=>x.startsWith('/')))")
        report[f"{side}_sheet"] = json.loads(listing or "[]")
        page.shot(OUT / f"{side}-1869-slash-sheet.png")
        # --- #1875: ended strip + failed send on an ended session
        page.goto(f"{base}/#/s/ev-ended"); settle(3)
        page.js("""(()=>{const t=document.querySelector('textarea');const set=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set;set.call(t,'are you there?');t.dispatchEvent(new Event('input',{bubbles:true}));})()""")
        settle(0.5)
        page.js("document.querySelector('button[aria-label=send]').click()"); settle(2)
        report[f"{side}_ended"] = {"strip": page.js("Array.from(document.querySelectorAll('p')).map(p=>p.textContent).filter(t=>/resume|ended/.test(t))") , "alert": page.js("(document.querySelector('[role=alert]')||{}).textContent")}
        page.shot(OUT / f"{side}-1875-ended.png")
    # --- #1870 measurements (after tree only: the toggle is new). Default meta is identical on both sides.
    if os.environ.get("ONLY1875"):
        (OUT / "report-1875.json").write_text(json.dumps({k: v for k, v in report.items()}, indent=1)); print(json.dumps(report, indent=1)); raise SystemExit(0)
    port, _ = sides["after"]; base = f"http://127.0.0.1:{port}"
    PROBE = "JSON.stringify({meta:document.querySelector('meta[name=viewport]').content,innerWidth:innerWidth,clientWidth:document.documentElement.clientWidth,vvScale:visualViewport.scale,vvWidth:visualViewport.width,vvHeight:visualViewport.height,dpr:devicePixelRatio,dataView:document.documentElement.dataset.view||null})"
    def pinch(scale):
        try:
            page.send("Input.synthesizePinchGesture", x=195, y=400, scaleFactor=scale, relativeSpeed=400, gestureSourceType="touch")
        except RuntimeError as e:
            print("pinch refused:", scale, e)
        settle(1.2)
    report["m1870"] = {}
    def fresh():
        # A new device-metrics override resets the page scale that a previous gesture left behind.
        page.send("Emulation.clearDeviceMetricsOverride"); page.goto("about:blank"); page.metrics(390, 844)
    for mode in ("default", "wide"):
        fresh()
        page.goto(f"{base}/#/")
        page.js(f"localStorage.{'setItem' if mode=='wide' else 'removeItem'}('lo-mobile-wide-view','1')")
        fresh(); page.goto(f"{base}/#/"); settle(2.5)
        rec = {"loaded": json.loads(page.js(PROBE))}
        page.shot(OUT / f"after-1870-list-{mode}.png")
        pinch(0.3); rec["after_pinch_out"] = json.loads(page.js(PROBE))
        page.shot(OUT / f"after-1870-list-{mode}-pinched-out.png")
        pinch(3); rec["after_pinch_in_back"] = json.loads(page.js(PROBE))
        report["m1870"][mode] = rec
        fresh(); page.goto(f"{base}/#/")
        page.js("Object.keys(localStorage).filter(k=>k.startsWith('lo-mobile-draft')).forEach(k=>localStorage.removeItem(k))")
        fresh(); page.goto(f"{base}/#/s/ev-live"); settle(2.5)
        rec["session_loaded"] = json.loads(page.js(PROBE))
        rec["column_box"] = page.js("(()=>{const c=document.querySelector('.h-dvh');const r=c.getBoundingClientRect();return {w:Math.round(r.width),h:Math.round(r.height),styleH:c.style.height,vvh:c.style.getPropertyValue('--lo-vvh')}})()")
        rec["composer_font_px"] = page.js("getComputedStyle(document.querySelector('textarea')).fontSize")
        page.shot(OUT / f"after-1870-session-{mode}.png")
        pinch(0.3); rec["session_after_pinch_out"] = json.loads(page.js(PROBE))
        rec["column_box_after_pinch"] = page.js("(()=>{const c=document.querySelector('.h-dvh');const r=c.getBoundingClientRect();return {w:Math.round(r.width),h:Math.round(r.height),vvh:c.style.getPropertyValue('--lo-vvh')}})()")
        page.shot(OUT / f"after-1870-session-{mode}-pinched-out.png")
    # toggle through the real control
    fresh(); page.goto(f"{base}/#/"); page.js("localStorage.clear()"); fresh(); page.goto(f"{base}/#/"); settle(2.5)
    page.js("document.querySelector('button[aria-label=\"wide view\"]').click()"); settle(1.5)
    report["toggle_click"] = {"probe": json.loads(page.js(PROBE)), "pressed": page.js("document.querySelector('button[aria-label=\"wide view\"]').getAttribute('aria-pressed')"), "stored": page.js("localStorage.getItem('lo-mobile-wide-view')")}
    page.shot(OUT / "after-1870-toggle-on.png")
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
finally:
    try: page.close()
    except Exception: pass
    if chrome: chrome.close()
    for p in procs:
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception: pass
        try: p.wait(timeout=10)
        except Exception: pass
