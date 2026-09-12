"""Read back the frames seamshot.py wrote: painted rows, seam maths, hashes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/des994")
from svgrows import svg_rows  # noqa: E402


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load(tag: str) -> dict:
    record = json.loads(Path(f"/tmp/des994/{tag}/{tag}-record.json").read_text())
    geo = record["geometry_start"]
    body_y, body_h = geo["body_region"][1], geo["body_region"][3]
    for frame in record["frames"]:
        rows = svg_rows(frame["svg"])
        frame["painted_body_rows"] = [rows.get(body_y + offset, "").rstrip() for offset in range(body_h)]
        frame["svg_md5"] = md5(Path(frame["svg"]))
        if "geometry_resize" in record and frame["label"].startswith("resize"):
            ry, rh = record["geometry_resize"]["body_region"][1], record["geometry_resize"]["body_region"][3]
            frame["painted_body_rows"] = [rows.get(ry + offset, "").rstrip() for offset in range(rh)]
    return record


def seam(record: dict, before_label: str, after_label: str) -> dict:
    frames = {f["label"]: f for f in record["frames"]}
    a, b = frames[before_label], frames[after_label]
    vh = record["geometry_start"]["viewport_height"]
    a_rows, b_rows = a["painted_body_rows"], b["painted_body_rows"]
    a_content = list(range(int(a["scroll_y"]), int(a["scroll_y"]) + len(a_rows)))
    b_content = list(range(int(b["scroll_y"]), int(b["scroll_y"]) + len(b_rows)))
    return {
        "before": before_label,
        "after": after_label,
        "scroll_y_before": a["scroll_y"],
        "scroll_y_after": b["scroll_y"],
        "viewport_height": vh,
        "step_rows": b["scroll_y"] - a["scroll_y"],
        "content_rows_before": (a_content[0], a_content[-1]),
        "content_rows_after": (b_content[0], b_content[-1]),
        "content_rows_shared": sorted(set(a_content) & set(b_content)),
        "content_rows_skipped": sorted(set(range(a_content[0], b_content[-1] + 1)) - set(a_content) - set(b_content)),
        "painted_lines_repeated": sorted({r for r in a_rows if r.strip()} & {r for r in b_rows if r.strip()}),
        "before_bottom_row": a_rows[-1],
        "after_top_row": b_rows[0],
        "frames_md5": {before_label: a["svg_md5"], after_label: b["svg_md5"]},
        "byte_identical": a["svg_md5"] == b["svg_md5"],
    }


def show(tag: str, label: str) -> None:
    record = load(tag)
    frames = {f["label"]: f for f in record["frames"]}
    f = frames[label]
    print(f"\n===== {tag} / {label}  scroll_y={f['scroll_y']} md5={f['svg_md5']}")
    for i, row in enumerate(f["painted_body_rows"]):
        print(f"  {i:2d} |{row}")


if __name__ == "__main__":
    tag = sys.argv[1]
    record = load(tag)
    print(f"### {tag}  grid={record['geometry_start']['grid']}")
    for f in record["frames"]:
        print(f"  {f['label']:24s} scroll_y={str(f['scroll_y']):6s} md5={f['svg_md5']}")
    print("\n--- seam (start -> first frame after pagedown)")
    print(json.dumps(seam(record, "start", "pagedown-first"), indent=2))
    print("\n--- settled identity: pagedown-first vs pagedown-settled")
    print(json.dumps(seam(record, "pagedown-first", "pagedown-settled"), indent=2)[:600])
    print("\n--- home vs start (top of report)")
    print(json.dumps(seam(record, "home", "start"), indent=2)[:600])
    (Path(f"/tmp/des994/{tag}/analysis.json")).write_text(json.dumps(record, indent=2) + "\n")
