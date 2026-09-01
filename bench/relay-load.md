# Mobile relay session-load benchmarks

`scripts/bench_relay_load.py` measures the paths a phone pays when it opens
the session list and a session, against the REAL store (3,925 sessions,
1.5 GB, transcripts to 53 MB). Run:

```sh
PYTHONPATH=. .venv/bin/python scripts/bench_relay_load.py
```

`PYTHONPATH=.` matters: the script must import THIS checkout's
`local_operator`, and a plain run from another directory can resolve an
editable install pointing elsewhere.

## relay-load-before.json / relay-load-after.json

Captured on this machine against `origin/main` @ `aa8de03c` (before) and the
same tree with the relay performance fixes (after). The before column is the
manager's diagnosis script run on the baseline worktree; the after column is
the extended script (adds list summaries cold/warm) on the fixed tree.

| Path | Metric | Before | After |
| --- | --- | --- | --- |
| `GET /api/sessions` (list) | cold scan | 356 ms (300-4,900 ms under loop contention, per the diagnosis) | 324 ms (one scan, off the loop) |
| `GET /api/sessions` (list) | warm repaint | 356 ms every time (no cache; a live session repaints ~30x/s) | **1.4 ms** (TTL cache) |
| Durable projection (SSE seed) | warm, 53 MB transcript | 505 ms | **5 ms** |
| Durable projection (SSE seed) | warm, 31 MB transcript | 158 ms | **1 ms** |
| Durable projection (SSE seed) | warm, worst of 8 largest | 505 ms | **127 ms** (one session mid-append; median 8 ms) |
| History page (80 rows) | repeated, 53 MB transcript | 265 ms per page | **<1 ms** per page |
| History page (80 rows) | repeated, worst of 8 largest | 251 ms | **<1 ms** |

The durable projection's first (cold) call still pays one full fold — that is
the one-time cost the cache amortizes (1.3-2.8 s on the 53 MB transcripts,
dominated by the file parse the old code paid on EVERY request). Every later
open reads only the bytes appended since, and every history page folds from
the cached replay instead of re-parsing the file.

The warm durable-projection outliers (109-127 ms on two sessions) are
sessions a live process was appending to during the run: the incremental tail
fold processes the appended window. The median across the eight is 8 ms.

## End-to-end: old daemon (4098) vs new daemon (4198)

`relay-e2e-dual.json` is the adapted `e2e_relay_timing.py` run against BOTH
daemons at once: port 4098 is the production daemon (old binary), port 4198
is this tree's daemon started with `LO_MOBILE_NO_DIAL=1` — observer mode,
because a registrant admits at most one daemon connection and a second dial
would evict the production daemon's live bridge. The observer therefore
serves live sessions from the durable fold (its SSE column for LIVE rows is
the durable path, not a live projection push); the list and durable rows are
directly comparable.

| Metric | old-4098 | new-4198 |
| --- | --- | --- |
| `GET /api/sessions` first | 1,082 ms | 72 ms |
| `GET /api/sessions` warm | 624 ms | **1.0 ms** |
| Durable SSE first frame (4 sessions) | 2.0-69.8 ms | 1.9-43.7 ms |
| Durable history page | 5.2-25.9 ms | 1.9-4.2 ms |
| Live-session history page | 6.5-12.4 ms | 3.5-4.2 ms |

The list route is the headline: the old daemon blocked its event loop on the
durable scan for every repaint (624-1,082 ms here, up to 4.9 s in the
diagnosis), which is what starved every other request. The new daemon's warm
list is a cached in-memory merge. SSE first frames for durable sessions are
flat-to-faster; the wedge the diagnosis captured (4 live sessions whose SSE
first frame never arrived within 12 s) came from oversized-frame floods on
the loop, which the registrant-side frame cap removes at the source.
