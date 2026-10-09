#!/usr/bin/env python3
"""Generate (or ``--check``) the turn-supplement contract fixtures.

WHY GENERATED. These files are the frozen contract every other lane codes against
(``docs/design/turn-supplements.md`` §6: engine, routes, TUI, relay web, UI, native). Hand
edits drift; generating them from the contract's own types means a change to a shape, a
state word or the prelude shows up as a diff in exactly the fixtures it affects, and
``tests/unit/supplements/test_fixtures.py`` fails until the regeneration is committed.

    .venv/bin/python tests/fixtures/supplements/build.py          # rewrite
    .venv/bin/python tests/fixtures/supplements/build.py --check  # exit 1 on drift

Layout (see README.md): rows/ events/ messages/ components/ documents/ geometry/.
Everything is deterministic: fixed ids, fixed timestamps, digests computed from the real
component bytes the way ``AttachmentStore.put_bytes`` does (sha256, first 32 hex).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from local_operator.harness.types import SupplementProgressEvent
from local_operator.supplements.contract import (
    SUPPLEMENT_CUSTOM_TYPE,
    reader_disposition,
)
from local_operator.supplements.document import assemble_stored_component

HERE = Path(__file__).resolve().parent

# --- ids ---------------------------------------------------------------------------------
SESSION_ID = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
ANCHOR = "a1b2c3d4e5f60718293a4b5c6d7e8f90"  # the final assistant message id
ANCHOR_B = "b2c3d4e5f60718293a4b5c6d7e8f90a1"
ANCHOR_C = "c3d4e5f60718293a4b5c6d7e8f90a1b2"
JOB = "3f9c1a7e5b20"  # 12 hex, stable across versions of one anchor
JOB_B = "7d41c0aa9e13"
JOB_C = "e0b5236f8c47"
AT = 1791000000.0

# --- the three components (the STORED blob: <data> blocks, then the body) -----------------
COMPONENTS: dict[str, str] = {
    "populated": (
        '<data>{"lat":{"title":"Latency by region (ms)","columns":["region","ms"],'
        '"rows":[["us-east",120],["us-west",98.5],["eu-west",143],["ap-south",211.25]]}}</data>\n'
        '<div id="c"></div>\n'
        '<script>LO.bar(document.getElementById("c"),"lat",{x:"region",y:"ms",unit:"ms"})</script>'
    ),
    # A valid component with nothing in it: no datasets, no body. The document must still
    # assemble, load the prelude and post `ready` (a host mounts it, shows an empty frame).
    "empty": "",
    # The frame-level failure case: the inline script asks the prelude for a dataset that
    # does not exist, so it throws, the window `error` handler posts {t:"error"} and the
    # host replaces the frame with its one quiet "Couldn't render this graphic" line.
    "error": (
        '<data>{"lat":{"title":"Latency (ms)","columns":["region","ms"],"rows":[["us-east",120]]}}'
        "</data>\n"
        '<div id="c"></div>\n'
        '<script>LO.bar(document.getElementById("c"),"missing",{x:"region",y:"ms"})</script>'
    ),
}


def digest(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


DIGESTS = {name: digest(blob) for name, blob in COMPONENTS.items()}


def _file(path: str, name: str, kind: str, size: int, why: str) -> dict[str, Any]:
    return {
        "path": path,
        "name": name,
        "kind": kind,
        "size_bytes": size,
        "mtime": AT - 60.0,
        "why": why,
    }


FILES = [
    _file("reports/latency.md", "latency.md", "markdown", 4120, "written by write"),
    _file("data/bench.csv", "bench.csv", "csv", 913, "written by bash"),
]


def _row(**fields: Any) -> dict[str, Any]:
    """A full journal entry carrying a ``supplement_v1`` details payload."""
    details = {"anchor": ANCHOR, "job": JOB, "version": 1, "files": [], "components": []}
    details.update(fields)
    details.setdefault("at", AT)
    # key order = the contract's documented order, so the fixtures read like the memo
    order = [
        "anchor",
        "job",
        "version",
        "state",
        "files",
        "files_more",
        "more",
        "components",
        "images",
        "decision",
        "instruction",
        "model",
        "turns",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "error",
        "dismissed",
        "at",
    ]
    ordered = {k: details[k] for k in order if k in details}
    ordered.update({k: v for k, v in details.items() if k not in ordered})
    return {
        "id": hashlib.sha256(json.dumps(ordered, sort_keys=True).encode()).hexdigest()[:32],
        "ts": ordered["at"],
        "type": "custom",
        "payload": {"custom_type": SUPPLEMENT_CUSTOM_TYPE, "details": ordered},
    }


DECISION = {
    "vendor": "radient",
    "files_p": {"reports/latency.md": 0.97, "data/bench.csv": 0.91},
    "graphics_p": 0.83,
    "skipped": None,
}
COMPONENT_REF = {
    "attachment": DIGESTS["populated"],
    "title": "Latency by region (ms)",
    "source": "bench.csv rows 1-4 (tool result of bash at 10:41)",
    "mime": "text/html",
    "height_hint": 320,
}

ROWS: dict[str, dict[str, Any]] = {
    # v1 written right after the decision: file callouts appear without waiting for the generator
    "decided": _row(state="decided", files=FILES, decision=DECISION),
    # THE STALE-ROW FIXTURE every lane asserts against (memo §2.4): non-terminal, no live job
    "queued_stale": _row(state="queued", files=FILES, decision=DECISION),
    "done_populated": _row(
        version=2,
        state="done",
        files=FILES,
        files_more=3,
        more=["reports/q3.md", "reports/q4.md", "data/raw.csv"],
        components=[COMPONENT_REF],
        decision=DECISION,
        instruction="make it a table",
        model="anthropic/claude-haiku-4-5",
        turns=2,
        tokens_in=9120,
        tokens_out=2210,
        cost_usd=0.0123,
        error="",
        at=AT + 41.5,
    ),
    "done_files_only": _row(version=1, state="done", files=FILES, decision=DECISION, at=AT + 2),
    # a finish with nothing to show renders NOTHING (no header, no reserved frame)
    "done_empty": _row(anchor=ANCHOR_B, job=JOB_B, state="done", at=AT + 90),
    "failed": _row(
        anchor=ANCHOR_B,
        job=JOB_B,
        version=1,
        state="failed",
        error="generator timed out",
        at=AT + 120,
    ),
    "cancelled": _row(anchor=ANCHOR_B, job=JOB_B, version=2, state="cancelled", at=AT + 130),
    # cut by a newer user turn: renders NOTHING, no Retry under an answer the user moved past
    "superseded": _row(
        anchor=ANCHOR_C, job=JOB_C, state="cancelled", error="superseded", at=AT + 200
    ),
    # the operator's "not useful" signal (supplement_dismiss): surfaces hide the row
    "dismissed": _row(anchor=ANCHOR_C, job=JOB_C, version=2, state="skipped", dismissed=True),
}

# Two versions of ONE anchor, in journal order: the newest wins (a reader rule).
JOURNAL_VERSIONS = [ROWS["decided"], ROWS["done_populated"]]

# What every surface must paint for a row, with and without a live job. The expectation is
# DERIVED from the reference reader (contract.reader_disposition) and then pinned, so a
# change to the stale-row rule surfaces as a fixture diff in this lane's PR.
DISPOSITIONS = {
    name: {
        "live": reader_disposition(row["payload"]["details"], job_live=True),
        "cold": reader_disposition(row["payload"]["details"], job_live=False),
    }
    for name, row in ROWS.items()
}


def _event(**fields: Any) -> dict[str, Any]:
    base = {"anchor": ANCHOR, "job": JOB, "version": 1}
    base.update(fields)
    return SupplementProgressEvent(**base).model_dump(mode="json")


EVENTS: dict[str, dict[str, Any]] = {
    "decided": _event(state="decided", files=FILES),
    "queued": _event(state="queued"),
    "running_generating": _event(state="running", stage="generating", elapsed_s=6.4),
    "running_repairing": _event(state="running", stage="repairing", elapsed_s=31.0, version=2),
    "cancelling": _event(state="cancelling", elapsed_s=12.0),
    "done": _event(
        version=2,
        state="done",
        elapsed_s=41.5,
        files=FILES,
        components=[
            {k: COMPONENT_REF[k] for k in ("attachment", "title", "source", "height_hint")}
        ],
    ),
    "failed": _event(
        state="failed", error="generator timed out", error_type="timeout", elapsed_s=90.0
    ),
    "cancelled_superseded": _event(
        anchor=ANCHOR_C, job=JOB_C, state="cancelled", error="superseded", elapsed_s=3.0
    ),
    "skipped": _event(state="skipped"),
}

TOKEN = "n-4f2a9c1e07b3"  # a per-frame nonce, as the host would mint it
MESSAGES = {
    "host": {
        "theme": {
            "lo": "supplement-host",
            "t": "theme",
            "mode": "light",
            "vars": {"--lo-canvas": "#ffffff", "--lo-ink": "#1f1f1f", "--font-sans": "system-ui"},
            "nonce": TOKEN,
        },
        "ping": {"lo": "supplement-host", "t": "ping"},
    },
    "frame_accepted": {
        "ready": {"lo": "supplement", "v": 1, "t": "ready"},
        "resize": {"lo": "supplement", "v": 1, "t": "resize", "h": 312, "n": TOKEN},
        "error": {
            "lo": "supplement",
            "v": 1,
            "t": "error",
            "msg": "unknown dataset missing",
            "n": TOKEN,
        },
        "pong": {"lo": "supplement", "v": 1, "t": "pong", "n": TOKEN},
    },
    # every one of these must be DROPPED by a host
    "frame_rejected": {
        "resize_without_nonce": {"lo": "supplement", "v": 1, "t": "resize", "h": 312},
        "resize_stale_nonce": {"lo": "supplement", "v": 1, "t": "resize", "h": 312, "n": "n-old"},
        "pong_without_nonce": {"lo": "supplement", "v": 1, "t": "pong"},
        "error_without_nonce": {"lo": "supplement", "v": 1, "t": "error", "msg": "x"},
        "resize_negative_height": {"lo": "supplement", "v": 1, "t": "resize", "h": -1, "n": TOKEN},
        "resize_boolean_height": {"lo": "supplement", "v": 1, "t": "resize", "h": True, "n": TOKEN},
        "unknown_type": {"lo": "supplement", "v": 1, "t": "navigate", "n": TOKEN},
        "wrong_version": {"lo": "supplement", "v": 2, "t": "pong", "n": TOKEN},
        "wrong_tag": {"lo": "supplement-host", "v": 1, "t": "pong", "n": TOKEN},
    },
}

# The design round's D3-2 case: a unit long enough that the top tick falls back INSIDE the
# plot at the 220 px floor. The third bar (910) sits under the fallback label.
GEOMETRY_LONG_UNIT = {
    "width": 220,
    "title": "Spend (USD/day)",
    "data": {
        "d": {
            "title": "Spend (USD/day)",
            "columns": ["service", "usd"],
            "rows": [
                ["Search", 420],
                ["Billing", 610],
                ["Notify", 910],
                ["Gateway", 380],
                ["Media", 250],
                ["Archive", 1000],
            ],
        }
    },
    "draw": "LO.bar(document.getElementById('c'),'d',{x:'service',y:'usd',unit:'USD/day'})",
}


def generate() -> dict[str, bytes]:
    """Every fixture file as ``{relative path: bytes}``."""
    out: dict[str, bytes] = {}

    def put_json(path: str, obj: Any) -> None:
        out[path] = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    for name, row in ROWS.items():
        put_json(f"rows/{name}.json", row)
    put_json("rows/journal_versions.json", JOURNAL_VERSIONS)
    put_json("rows/dispositions.json", DISPOSITIONS)
    for name, event in EVENTS.items():
        put_json(f"events/{name}.json", event)
    put_json("messages/messages.json", MESSAGES)
    put_json("geometry/long_unit_220.json", GEOMETRY_LONG_UNIT)
    for name, blob in COMPONENTS.items():
        out[f"components/{name}.html"] = blob.encode("utf-8")
        out[f"documents/{name}.html"] = assemble_stored_component(blob).encode("utf-8")
    put_json("components/digests.json", DIGESTS)
    return out


def main(argv: list[str]) -> int:
    files = generate()
    if "--check" in argv:
        stale = [
            rel
            for rel, data in files.items()
            if not (HERE / rel).is_file() or (HERE / rel).read_bytes() != data
        ]
        known = {p.relative_to(HERE).as_posix() for d in files for p in [HERE / d]}
        extra = sorted(
            p.relative_to(HERE).as_posix()
            for sub in ("rows", "events", "messages", "components", "documents", "geometry")
            for p in (HERE / sub).glob("*")
            if p.relative_to(HERE).as_posix() not in known
        )
        for rel in stale:
            print(f"stale: {rel}")
        for rel in extra:
            print(f"orphan: {rel}")
        return 1 if stale or extra else 0
    for rel, data in files.items():
        target = HERE / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f"wrote {len(files)} fixtures under {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
