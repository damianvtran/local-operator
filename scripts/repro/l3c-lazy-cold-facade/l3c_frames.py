"""Capture the /events frames a real serve process emitted, and diff two runs."""

from __future__ import annotations

import json
import sys

#: Frame fields two independent processes cannot make equal, and why: the bridge
#: mints a fresh ``epoch`` per attachment, the serve clocks are wall time, and
#: cursors/tokens are positions minted per read. Nothing here is history.
VOLATILE = {
    "epoch",
    "served_at",
    "now",
    "built_at",
    "ts",
    "ts_source",
    "cursor",
    "before_token",
    "snapshot_token",
    "subscription_id",
}


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def extract(path: str, out: str) -> None:
    frames = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("data: "):
                continue
            try:
                frames.append(json.loads(line[6:].strip()))
            except ValueError:
                continue
    snapshot = next((f for frame in frames if (f := frame).get("type") == "snapshot"), None)
    payload = {
        "frame_kinds": [frame.get("type") for frame in frames[:4]],
        "snapshot": canonical(snapshot["payload"]) if snapshot else None,
        "raw_snapshot": snapshot,
    }
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps({"frame_kinds": payload["frame_kinds"], "has_snapshot": snapshot is not None}))


def compare(left: str, right: str) -> int:
    base = json.load(open(left, encoding="utf-8"))
    other = json.load(open(right, encoding="utf-8"))
    if base["snapshot"] == other["snapshot"] and base["snapshot"] is not None:
        print("IDENTICAL: the canonical snapshot frame matches field for field")
        return 0
    problems: list[str] = []

    def diff(a, b, path="$") -> None:
        if type(a) is not type(b):
            problems.append(f"{path}: {type(a).__name__} != {type(b).__name__}")
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    problems.append(f"{path}.{key}: present on one side only")
                else:
                    diff(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, list):
            if len(a) != len(b):
                problems.append(f"{path}: len {len(a)} != {len(b)}")
            for index, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{index}]")
        elif a != b:
            problems.append(f"{path}: {a!r} != {b!r}")

    diff(base["snapshot"], other["snapshot"])
    print(f"DIFFERS in {len(problems)} place(s):")
    for problem in problems[:40]:
        print("  " + problem)
    return 1


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(compare(sys.argv[2], sys.argv[3]))
