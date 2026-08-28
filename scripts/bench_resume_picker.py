"""Benchmark the ``/resume`` picker open path, stage by stage.

Usage:

    PYTHONPATH=. .venv/bin/python scripts/bench_resume_picker.py \
        <config_dir> <label> [--json out.json]

``PYTHONPATH=.`` matters for the reason ``bench_tui_lag.py`` documents: the
script must import THIS checkout, not whatever an editable install resolves to.

Reports each stage separately rather than one total, because the picker open is
a sum of independent costs (directory scan, per-row name read, digest index,
search tiers) and a single number cannot say which one regressed. Build a
store to run it against with ``scripts/bench_resume_picker_store.py``.
"""

import gc
import json
import statistics
import sys
import time
from pathlib import Path

cd = Path(sys.argv[1])
label = sys.argv[2]
out_json = None
if "--json" in sys.argv:
    out_json = sys.argv[sys.argv.index("--json") + 1]

from local_operator.resume import recent_session_rows, recent_sessions  # noqa: E402
from local_operator.session.search_index import (  # noqa: E402
    SoftSearchIndex,
    build_index,
    search_digests,
)

res: dict[str, float | int] = {}


def timed(fn, reps=3):
    ts = []
    for _ in range(reps):
        gc.collect()
        t = time.perf_counter()
        r = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return min(ts), r


# --- scan -------------------------------------------------------------------
ms, rows_all = timed(lambda: recent_sessions(cd, 10**9))
res["recent_sessions_uncapped_ms"] = round(ms, 1)
res["user_sessions"] = len(rows_all)

ms, _ = timed(lambda: recent_sessions(cd, 200))
res["recent_sessions_limit200_ms"] = round(ms, 1)

ms, rows = timed(lambda: recent_session_rows(cd, 10**9))
res["recent_session_rows_uncapped_ms"] = round(ms, 1)
ids = [r.id for r in rows]

# --- build_index ------------------------------------------------------------
build_index(cd, ids)  # prime the cache
ms, digests = timed(lambda: build_index(cd, ids))
res["build_index_warm_ms"] = round(ms, 1)
res["digests"] = len(digests)

# The daemon-thrash sequence: a limit=200 caller followed by the picker's full
# call. Before D2 the 200-id call prunes the cache and the picker pays a cold
# rebuild; after D2 it must cost the same as the warm call above.
build_index(cd, ids)
build_index(cd, ids[:200])
gc.collect()
t = time.perf_counter()
build_index(cd, ids)
res["build_index_after_daemon_thrash_ms"] = round((time.perf_counter() - t) * 1000, 1)

# --- search tiers -----------------------------------------------------------
t = time.perf_counter()
search_digests(digests, "minerva")
res["exact_search_first_ms"] = round((time.perf_counter() - t) * 1000, 2)
qs = ["migration", "pipeline", "watchlist", "dossier", "screening"]
ts = []
for q in qs:
    t = time.perf_counter()
    search_digests(digests, q)
    ts.append((time.perf_counter() - t) * 1000)
res["exact_search_steady_ms"] = round(statistics.median(ts), 2)

idx = SoftSearchIndex()
gc.collect()
t = time.perf_counter()
idx.search(digests, "m")
res["soft_first_keystroke_ms"] = round((time.perf_counter() - t) * 1000, 1)
ts = []
for q in ["mi", "mig", "migr", "migra"]:
    t = time.perf_counter()
    idx.search(digests, q)
    ts.append((time.perf_counter() - t) * 1000)
res["soft_steady_keystroke_ms"] = round(statistics.median(ts), 1)

try:
    import resource

    res["rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)
except Exception:
    pass

# --- total picker open ------------------------------------------------------
gc.collect()
t = time.perf_counter()
r2 = recent_session_rows(cd, 10**9)
build_index(cd, [r.id for r in r2])
res["picker_open_total_ms"] = round((time.perf_counter() - t) * 1000, 1)

print(f"=== {label} ===")
for k, v in res.items():
    print(f"{k:42s} {v}")
if out_json:
    Path(out_json).write_text(json.dumps({"label": label, **res}, indent=2))
