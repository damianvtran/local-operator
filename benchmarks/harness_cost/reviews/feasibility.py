"""Offline-feasibility probe for each task's checker (read-only on rig data).

WHY: v1 says each checker was validated (gold scores 1, untouched input
scores 0). That does not show the task can be passed with the toolchain the
prompt offers (python-docx / python-pptx / openpyxl, no LibreOffice). If the
gold file loses its pass after a load+save round trip through that library,
that diagnoses cache/format sensitivity requiring a reference-solution audit;
it does not prove the task impossible or invalidate every agent solution.
HOW: copy gold and input into a temporary home laid out as /home/user, then
score four variants with evaluate.py: gold, gold after a library round trip,
input (untouched), and missing. Output is one line per task.
Constraints: needs external OSWorld metric dependencies and private artifacts; not part of public analysis; writes only
under a temporary directory, which it deletes.
"""
import json, os, shutil, subprocess, sys, tempfile
HERE = os.environ.get('BENCH_REVIEW_FIXTURES')
if not HERE or not os.path.isfile(os.path.join(HERE, 'evaluate.py')):
    raise SystemExit('Historical artifact review requires external BENCH_REVIEW_FIXTURES with evaluate.py, complete task specs, cache/ and runs/. These private artifacts are NOT shipped; use regenerate.py for public numerical reproduction.')
PY = sys.executable
T = json.load(open(os.path.join(HERE, "v1-snapshot/tasks.json")))

def roundtrip(src, dst):
    ext = os.path.splitext(src)[1].lower()
    code = {".docx": "import docx,sys;docx.Document(sys.argv[1]).save(sys.argv[2])",
            ".pptx": "import pptx,sys;pptx.Presentation(sys.argv[1]).save(sys.argv[2])",
            ".xlsx": "import openpyxl,sys;openpyxl.load_workbook(sys.argv[1]).save(sys.argv[2])"}.get(ext)
    if not code:
        shutil.copy(src, dst); return "copy"
    r = subprocess.run([PY, "-c", code, src, dst], capture_output=True, text=True)
    return "ok" if r.returncode == 0 else "ERR " + r.stderr[-120:]

def score(tid, home):
    out = subprocess.run([PY, os.path.join(HERE, "evaluate.py"), tid, home], capture_output=True, text=True)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return {"score": None, "reason": out.stderr[-150:]}

for t in T:
    ev = t["evaluator"]; rel = ev["result"]["path"][len("/home/user/"):]
    exp = ev.get("expected") or {}
    if exp.get("type") != "cloud_file":
        print(t["id"][:8], t["_tier"], ev["func"], "rule-based checker; not probed"); continue
    gold = os.path.join(HERE, "cache", t["id"], "gold", exp["dest"])
    res = {}
    with tempfile.TemporaryDirectory() as d:
        dst = os.path.join(d, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(gold, dst); res["gold"] = score(t["id"], d)["score"]
        rt = roundtrip(gold, dst + ".rt" + os.path.splitext(dst)[1]); os.replace(dst + ".rt" + os.path.splitext(dst)[1], dst) if rt != "copy" else None
        res["gold_roundtrip"] = score(t["id"], d)["score"] if not rt.startswith("ERR") else rt
        inp = os.path.join(HERE, "cache", t["id"], "input", os.path.basename(rel))
        if os.path.exists(inp):
            shutil.copy(inp, dst); res["input"] = score(t["id"], d)["score"]
        os.remove(dst); res["missing"] = score(t["id"], d)["score"]
    print(t["id"][:8], t["_tier"], ev["func"], json.dumps(res))
