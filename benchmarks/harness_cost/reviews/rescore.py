"""Rescore the kept result files of v1 rep-0 runs (read-only on rig data).

WHY: confirms runs.jsonl scores can be reproduced from the kept artefacts,
and lets the review attach a checker reason to each all-fail task.
HOW: evaluate.py reads <dir>/<result rel path>; kept/ holds only the result
file's basename, so we stage it in a temporary home at the expected relative
path before scoring. Constraints: external BENCH_REVIEW_FIXTURES must be supplied;
private artifacts are not shipped. No model/network calls; temporary homes only,
never the invoking user's real HOME. This is historical methodology, not a
public checker rerun claimed by the analysis kit.
"""
import json, os, shutil, subprocess, tempfile, glob, sys
HERE = os.environ.get('BENCH_REVIEW_FIXTURES')
if not HERE or not os.path.isfile(os.path.join(HERE, 'evaluate.py')):
    raise SystemExit('Historical artifact review requires external BENCH_REVIEW_FIXTURES with evaluate.py, complete task specs, cache/ and runs/. These private artifacts are NOT shipped; use regenerate.py for public numerical reproduction.')
RUNS = os.environ.get("BENCH_RUNS", os.path.join(HERE, "runs"))
PY = sys.executable
T = {t["id"]: t for t in json.load(open(os.path.join(HERE, "v1-snapshot/tasks.json")))}
logged = {json.loads(l)["run_id"]: json.loads(l) for l in open(os.path.join(HERE, "v1-snapshot/runs.jsonl"))}
only = set(sys.argv[1:])
agree = disagree = missing = 0
for rid, r in sorted(logged.items()):
    if r["rep"] != 0 or (only and rid.split("__")[2] not in only):
        continue
    t = T[r["task_id"]]; rel = t["evaluator"]["result"]["path"][len("/home/user/"):]
    kept = os.path.join(RUNS, rid, "kept", os.path.basename(rel))
    with tempfile.TemporaryDirectory() as d:
        if os.path.exists(kept):
            os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True); shutil.copy(kept, os.path.join(d, rel))
        else:
            missing += 1
        out = subprocess.run([PY, os.path.join(HERE, "evaluate.py"), t["id"], d], capture_output=True, text=True)
        s = json.loads(out.stdout.strip().splitlines()[-1])
    same = s["score"] == r["score"]; agree += same; disagree += (not same)
    if only or not same:
        print(rid, "logged", r["score"], "rescored", s)
print(f"agree={agree} disagree={disagree} kept-missing={missing}")
