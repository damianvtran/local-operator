"""Verify minimal historical cache counters without paid calls or private bodies.

Why: every numeric claim in CACHE-QUALIFICATION must derive from supplied rows.
Constraints: hashes attest withheld inputs only; this is not a live cache rerun
or independent verification of provider telemetry. No network/imported harness.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def verify(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parent
    rows = json.loads((root / "evidence/cache-controls.json").read_text())
    expected = {
        "early-tool-nonstream": [0, 18081, 0, 18081],
        "negative-system-only": [0, 17991, 8287, 17991],
    }
    if len(rows) != 8:
        raise ValueError("Expected exactly eight minimal evidence rows")
    result = {}
    for control, reads in expected.items():
        group = sorted((r for r in rows if r["control"] == control), key=lambda r: r["ordinal"])
        if [r["ordinal"] for r in group] != [0, 1, 2, 3]:
            raise ValueError("Missing or duplicate ordinal")
        if [r["namespace"] for r in group] != ["A", "A", "B", "A"]:
            raise ValueError("Expected A/A/B/A schedule")
        actual = [r["cache_read_input_tokens"] for r in group]
        if actual != reads:
            raise ValueError("Counter mismatch")
        if len({r["before_sha256"] for r in group}) != 1:
            raise ValueError("Control changed more than namespace")
        a_hashes = {r["after_sha256"] for r in group if r["namespace"] == "A"}
        if len(a_hashes) != 1 or group[2]["after_sha256"] in a_hashes:
            raise ValueError("Namespace lineage hash mismatch")
        for row in group:
            if row["request_sha256"] != row["after_sha256"]:
                raise ValueError("Final request hash differs from post-transformation hash")
            if row["ended"] < row["started"]:
                raise ValueError("Reversed timing")
            for key in ("request_sha256", "before_sha256", "after_sha256"):
                if not re.fullmatch(r"[a-f0-9]{64}", row[key]):
                    raise ValueError("Invalid digest")
            for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                if not isinstance(row[key], int) or row[key] < 0:
                    raise ValueError("Invalid token counter")
        result[control] = {"read_tokens": actual, "rows": len(group)}
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
