"""Run the finite capture census, rasterize at native size, and write a manifest.

Each script remains the owner of its scenario. Sequential subprocesses preserve
its CLI and isolate module caches without multiplying app workers on this host.
The HTML index is a contact sheet, not evidence that its thumbnails were viewed.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cases() -> list[dict[str, Any]]:
    inventory = json.loads((ROOT / "scripts/visual_inventory.json").read_text())
    result = []
    for script, variants in inventory["capture_scripts"].items():
        for variant in variants:
            args = ["{output}"]
            if script == "eager_boot_shot.py":
                args += ["--isolated"]
            elif script in {"ask_long_shot.py", "ask_user_repro.py"}:
                args += ["150x40", variant[3], *(["reveal"] if "reveal" in variant else [])]
            elif script == "ask_scroll_shot.py":
                args += [
                    "100x30",
                    {"top": "0", "bottom-down": "12", "pagedown": "2", "reveal": "0"}[variant],
                    "pagedown" if variant == "pagedown" else "down",
                ]
                if variant == "reveal":
                    args += ["reveal"]
            elif script == "approval_shot.py":
                args += ["100x30", *(["focus"] if variant == "focus" else [])]
            elif script in {
                "copy_shot.py",
                "fork_shot.py",
                "settings_shot.py",
                "settings_suggest_shot.py",
                "sibling_shot.py",
            }:
                args += ["100x30", variant]
            elif script in {"fallback_shot.py", "nerd_glyph_shot.py"}:
                args += [variant]
            elif script == "stop_ladder_shot.py":
                args += [variant, "100x30"]
            elif script == "org_chart_shot.py":
                scenario = "nested" if variant.startswith("nested-") else variant
                tier = (
                    "0"
                    if variant.endswith("tier0")
                    else ("2" if variant.endswith("tier2") else "1")
                )
                args += [scenario, "100x30", tier]
                if variant.endswith("expanded"):
                    args += ["expand"]
                if variant.endswith("legend"):
                    args += ["legend"]
            elif script == "shot_login.py":
                args += [
                    "alibaba",
                    {
                        "prompt": "",
                        "empty-submit": "enter",
                        "cancel": "escape",
                        "synthetic-key-submit": "s,k,minus,t,e,s,t,enter",
                    }[variant],
                ]
            elif script == "steer_receipt_probe.py":
                args = [variant, "{output}"]
            elif script == "theme_preview.py":
                args = ["{directory}"]
            result.append({"id": f"{Path(script).stem}-{variant}", "script": script, "args": args})
    # The new page fixture deliberately fills families absent from the existing
    # scripts, rather than creating a second settings/ask implementation.
    from scripts.pages_shot import PAGES

    for page in PAGES:
        result.append(
            {
                "id": f"page-{page}",
                "script": "pages_shot.py",
                "args": ["{output}", page, "100x30", "dark"],
            }
        )
    for size in ("158x44", "208x54"):
        result.append(
            {
                "id": f"reference-welcome-{size}",
                "script": "pages_shot.py",
                "args": ["{output}", "welcome", size, "radient"],
                "reference": "Estimated viewport, not measured terminal settings",
            }
        )
    for size in ("80x24", "100x30"):
        for palette in ("dark", "light"):
            result.append(
                {
                    "id": f"transcript-{size}-{palette}",
                    "script": "pages_shot.py",
                    "args": ["{output}", "transcript", size, palette],
                }
            )
        for script in ("ask_shot.py", "settings_shot.py"):
            result.append(
                {"id": f"{Path(script).stem}-{size}", "script": script, "args": ["{output}", size]}
            )
    result.append(
        {
            "id": "stop-short",
            "script": "stop_ladder_shot.py",
            "args": ["{output}", "offer", "80x24"],
        }
    )
    return result


def rasterize(svg: Path, converter: str) -> dict[str, Any]:
    png = svg.with_suffix(".png")
    subprocess.run([converter, str(svg), "-o", str(png)], check=True, capture_output=True)
    metadata = json.loads(svg.with_suffix(".geometry.json").read_text())
    # A reference comparison is uniformly fit, never a new layout or a forced
    # width/height pair. Native output remains alongside it for glyph inspection.
    return {
        "svg": svg.name,
        "png": png.name,
        "native_pixels": metadata["native_pixels"],
        "inspection": "NOT_INSPECTED",
        "screen": metadata["screen"],
    }


def run(outdir: Path, selected: list[dict[str, Any]], converter: str | None) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "preset": "8x17 cells / 13px local font; reproducible, not emulator defaults",
        "rasterizer": converter,
        "cases": [],
    }
    failures = 0
    for case in selected:
        directory = outdir / case["id"]
        directory.mkdir(exist_ok=True)
        output = directory / "capture.svg"
        args = [a.format(output=output, directory=directory) for a in case["args"]]
        command = [sys.executable, str(ROOT / "scripts" / case["script"]), *args]
        entry = {**case, "command": command, "artifacts": [], "status": "FAIL"}
        try:
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, timeout=120
            )
            (directory / "run.log").write_text(completed.stdout + completed.stderr)
            entry["returncode"] = completed.returncode
            svgs = sorted(directory.glob("*.svg"))
            if completed.returncode or not svgs:
                raise RuntimeError(
                    f"script exit={completed.returncode}, SVG count={len(svgs)}; see run.log"
                )
            for svg in svgs:
                if converter:
                    artifact = rasterize(svg, converter)
                else:
                    artifact = {"svg": svg.name, "inspection": "BLOCKED_NO_RASTERIZER"}
                entry["artifacts"].append(artifact)
                if converter and case.get("reference"):
                    subprocess.run(
                        [
                            converter,
                            "-w",
                            "1024",
                            str(svg),
                            "-o",
                            str(svg.with_name(svg.stem + "-fit1024.png")),
                        ],
                        check=True,
                        capture_output=True,
                    )
            entry["status"] = "PASS" if converter else "SVG_ONLY"
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, KeyError) as exc:
            entry["error"] = str(exc)
            failures += 1
        manifest["cases"].append(entry)
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{entry['status']}: {case['id']}", flush=True)
    cards = []
    for entry in manifest["cases"]:
        for artifact in entry["artifacts"]:
            if "png" in artifact:
                src = f"{entry['id']}/{artifact['png']}"
                cards.append(
                    f'<figure><a href="{html.escape(src)}">'
                    f'<img src="{html.escape(src)}" width="400"></a>'
                    f'<figcaption>{html.escape(entry["id"])} / '
                    f'{html.escape(artifact["png"])}</figcaption></figure>'
                )
    (outdir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Capture gallery</title>'
        "<h1>Native captures</h1><p>Thumbnails are navigation, not native-size evidence. "
        "Click for the original. Inspection status is in manifest.json.</p>" + "\n".join(cards)
    )
    return int(bool(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--svg-only", action="store_true", help="Explicitly skip optional native rasterization"
    )
    args = parser.parse_args()
    all_cases = cases()
    if args.list:
        print("\n".join(case["id"] for case in all_cases))
        return
    known = {case["id"] for case in all_cases}
    if set(args.case) - known:
        parser.error(f"unknown case: {', '.join(sorted(set(args.case) - known))}")
    if args.output is None:
        parser.error("output directory is required")
    converter = None if args.svg_only else shutil.which("rsvg-convert")
    if not args.svg_only and converter is None:
        parser.error("rsvg-convert not found; install librsvg or explicitly pass --svg-only")
    selected = [c for c in all_cases if not args.case or c["id"] in args.case]
    raise SystemExit(run(args.output.resolve(), selected, converter))


if __name__ == "__main__":
    main()
