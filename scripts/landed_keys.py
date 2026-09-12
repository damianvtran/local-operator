"""On the LANDED page: do the hint's keys still do something the reader can see?

After one pagedown the cursor is off-screen, so ``enter`` (advertised in the hint)
acts on a row the reader cannot see. This records what the frame does when it is
pressed — whether the visible page changes, and whether the offset survives the
re-render — and the docstring states no claim beyond what the frames show.
"""
from __future__ import annotations
import asyncio, hashlib, os, sys
from pathlib import Path
for _k in tuple(os.environ):
    if _k.startswith("CMUX_"): os.environ.pop(_k)
REPO=Path(sys.argv[1]); OUT=Path(sys.argv[2]); sys.path.insert(0,str(REPO)); sys.path.insert(0,'/tmp/des994')
import scripts.probe_isolation  # noqa
from svgrows import svg_rows
from scripts.visual_capture import save_capture
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen
from tests.unit.tui.test_analytics_panel import _tall_report_agg
from tests.unit.tui.test_app_pilot import FakeSession, _factory

def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()

async def main():
    app=OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120,45)) as pilot:
        await pilot.pause()
        s=AnalyticsScreen(_tall_report_agg()); await app.push_screen(s)
        await pilot.pause(); await pilot.pause()
        b=s._scroll
        await pilot.press('pagedown'); await pilot.wait_for_scheduled_animations(); await pilot.pause()
        a=OUT/'landed-before-enter.svg'; save_capture(app,str(a))
        print('after pagedown: y=',b.scroll_offset.y,'cursor=',s._cursor,'expanded=',len(s._expanded))
        await pilot.press('enter'); await pilot.pause(); await pilot.wait_for_scheduled_animations(); await pilot.pause()
        c=OUT/'landed-after-enter.svg'; save_capture(app,str(c))
        print('after enter: y=',b.scroll_offset.y,'cursor=',s._cursor,'expanded=',len(s._expanded))
        print('frame changed:', md5(a)!=md5(c))
        ra,rc=svg_rows(str(a)),svg_rows(str(c))
        diff=[r for r in range(8,37) if ra.get(r)!=rc.get(r)]
        print('body rows that changed:',diff)
        for r in diff[:6]:
            print(f'  {r}: {ra.get(r,"").replace(chr(160)," ").rstrip()[:95]}')
            print(f'  {r}: {rc.get(r,"").replace(chr(160)," ").rstrip()[:95]}')
asyncio.run(main())
