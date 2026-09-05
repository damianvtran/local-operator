"""REAL two-process evidence for the asymmetric approvals rule.

Pane B is a real runtime process (`python -m local_operator.session.runtime.process`,
the exact child `lop` spawns) reached over the production RemoteSession client.
Every WRITE is done by a THIRD process (`local-operator config edit`), never
in-process, so what is measured is a genuine cross-process config delivery.
"""
import asyncio, os, subprocess, sys, tempfile, time, pathlib

REPO = "/tmp/lop-live-settings"
sys.path.insert(0, REPO)
PY = f"{REPO}/.venv/bin/python"

def cfg_edit(cfgdir, key, value):
    """The third process. Returns (rc, elapsed)."""
    env = dict(os.environ, LOCAL_OPERATOR_CONFIG_DIR=str(cfgdir))
    t0 = time.time()
    r = subprocess.run([PY, "-m", "local_operator.cli", "config", "edit", key, value],
                       env=env, capture_output=True, text=True, cwd=REPO)
    return r.returncode, time.time() - t0, (r.stdout + r.stderr).strip()

async def main():
    from local_operator.config import ConfigManager
    from local_operator.config_watch import process_watcher
    from local_operator.session.runtime.owned import OwnedSessionHandle
    from tests.unit.session.runtime.test_owned import FakeSession

    root = pathlib.Path(tempfile.mkdtemp(prefix="ev-approvals-"))
    cfgdir = root / "config"; cfgdir.mkdir(parents=True)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(cfgdir)
    ConfigManager(cfgdir).set_config_value("tool_approval_mode", "ask")

    def new_handle(auto):
        s = FakeSession(); emitted = []
        async def _emit(e): emitted.append(e)
        s._emit = _emit
        h = OwnedSessionHandle(s, asyncio.get_running_loop(), cwd=str(root), auto_approve=auto)
        h.follow_config(process_watcher(cfgdir))
        return h, s, emitted

    w = process_watcher(cfgdir)

    print("=" * 72)
    print("CELL 1 — LOOSENING against an explicit in-session /approvals ask")
    print("=" * 72)
    h, s, em = new_handle(auto=False)
    from local_operator.session.frontend_state import SlashResult
    r = h._approvals_slash(s, "ask", SlashResult)
    print(f"  pane B: /approvals ask      -> {r.text}")
    print(f"          gate = {'auto' if h._auto_approve else 'ask'}   explicit={h._explicit_approvals_choice}")
    em.clear()
    rc, dt, out = cfg_edit(cfgdir, "tool_approval_mode", "auto")
    print(f"  pane A (separate OS process): config edit tool_approval_mode auto  rc={rc} in {dt:.2f}s")
    t0 = time.time(); w.poll_now(); await asyncio.sleep(0)
    print(f"  delivered in {time.time()-t0:.3f}s")
    print(f"  disk = {ConfigManager(cfgdir).get_config_value('tool_approval_mode')!r}")
    print(f"  pane B gate = {'auto' if h._auto_approve else 'ask'}   <-- MUST still be ask")
    for e in em: print(f"  [notice/{getattr(e,'kind','')}] {getattr(e,'text','')}")
    # Prove the gate is REALLY armed, not merely flagged: a decision must park.
    parked = asyncio.ensure_future(h._approval_gate("bash", "rm -rf build/"))
    await asyncio.sleep(0)
    pend = h._fold.projection.pending
    print(f"  a real tool decision -> {'PARKED for the human' if pend else 'AUTO-APPROVED'}")
    await h.approval_answer(pend.request_id, False, False)
    print(f"  human denied it -> tool ran? {await parked}")
    rep = h._approvals_slash(s, "", SlashResult)
    print(f"  pane B: /approvals          -> {rep.text}")
    print(f"  pane B: /approvals auto     -> adopts: ", end="")
    h._approvals_slash(s, "auto", SlashResult)
    print(f"gate = {'auto' if h._auto_approve else 'ask'}")
    await h.dispose()

    print()
    print("=" * 72)
    print("CELL 2 — TIGHTENING over an explicit choice (must ALWAYS follow)")
    print("=" * 72)
    cfg_edit(cfgdir, "tool_approval_mode", "auto"); w.poll_now()
    h2, s2, em2 = new_handle(auto=True)
    h2._approvals_slash(s2, "auto", SlashResult)
    print(f"  pane B: /approvals auto     -> gate = {'auto' if h2._auto_approve else 'ask'}  explicit={h2._explicit_approvals_choice}")
    em2.clear()
    rc, dt, out = cfg_edit(cfgdir, "tool_approval_mode", "ask")
    print(f"  pane A: config edit tool_approval_mode ask  rc={rc} in {dt:.2f}s")
    t0 = time.time(); w.poll_now(); await asyncio.sleep(0)
    print(f"  delivered in {time.time()-t0:.3f}s")
    print(f"  pane B gate = {'auto' if h2._auto_approve else 'ask'}   <-- MUST be ask (safety always propagates)")
    for e in em2: print(f"  [notice/{getattr(e,'kind','')}] {getattr(e,'text','')}")
    await h2.dispose()

    print()
    print("=" * 72)
    print("CELL 3 — a session that NEVER chose follows the file BOTH ways")
    print("=" * 72)
    cfg_edit(cfgdir, "tool_approval_mode", "ask"); w.poll_now()
    h3, _s3, em3 = new_handle(auto=False)
    for target in ("auto", "ask"):
        rc, dt, _ = cfg_edit(cfgdir, "tool_approval_mode", target)
        w.poll_now(); await asyncio.sleep(0)
        got = 'auto' if h3._auto_approve else 'ask'
        print(f"  disk -> {target:4}  pane B gate = {got:4}  {'OK' if got == target else 'MISMATCH'}")
    await h3.dispose()
    print("\nconfig dir:", cfgdir)

asyncio.run(main())
