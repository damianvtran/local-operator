"""Build a synthetic session store for ``scripts/bench_resume_picker.py``.

Usage:

    .venv/bin/python scripts/bench_resume_picker_store.py <root> <users> <subagents>

Mirrors the real store's shape: ~2700 user sessions and ~29000 subagent
sessions (the subagent population is what actually explodes, at ~10.6x the
user population on the operator's machine). Transcripts are realistic in size
so digest reads and title scans cost what they cost in production.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
n_user = int(sys.argv[2])
n_sub = int(sys.argv[3])
sessions = root / "sessions"
sessions.mkdir(parents=True, exist_ok=True)

random.seed(1234)

WORDS = (
    "minerva pergamon deploy pipeline migration watchlist enrichment dossier "
    "screening sanction adverse media entity resolve ingest catalogue release "
    "picker resume digest transcript session index vocabulary keystroke latency "
    "browser extension relay mobile daemon analytics ledger provider token "
    "review remediation blocker finding evidence gate merge tag publish"
).split()


def body(n_turns: int) -> str:
    """Transcript lines in the REAL persisted schema.

    ``digest_transcript`` only indexes ``type=="message"`` entries whose
    ``payload.role`` is indexed and whose ``content`` is a list of text parts;
    anything else digests to "" and the benchmark then measures an empty
    corpus rather than the search cost it is meant to measure.
    """
    out = []
    for i in range(n_turns):
        text = " ".join(random.choice(WORDS) for _ in range(random.randint(20, 120)))
        role = "user" if i % 2 == 0 else "assistant"
        out.append(
            json.dumps(
                {
                    "id": f"m{i}",
                    "ts": 0,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": role,
                        "content": [{"type": "text", "text": text}],
                    },
                }
            )
        )
    return "\n".join(out) + "\n"


now = time.time()
span = 90 * 86400  # three months

made = 0
for i in range(n_user + n_sub):
    is_sub = i >= n_user
    sid = f"{i:012x}"
    d = sessions / sid
    d.mkdir(exist_ok=True)
    (d / "transcript.jsonl").write_text(body(random.randint(4, 40)), encoding="utf-8")
    if is_sub:
        (d / "origin.json").write_text(json.dumps({"origin": "subagent"}), encoding="utf-8")
    else:
        (d / "title.json").write_text(
            json.dumps(
                {
                    "text": " ".join(random.choice(WORDS) for _ in range(random.randint(2, 6))),
                    "names": [],
                }
            ),
            encoding="utf-8",
        )
    mt = now - random.random() * span
    os.utime(d / "transcript.jsonl", (mt, mt))
    os.utime(d, (mt, mt))
    made += 1
    if made % 5000 == 0:
        print(f"  {made}...", flush=True)

print(f"built {made} dirs in {root}")
