"""Score saved files using narrowly selected, pinned public OSWorld metrics.

WHY: importing desktop_env normally initializes VM dependencies. Namespace-only
packages let us call the upstream file metrics without starting a VM, replacing
GUI save postconfig with the runner's requirement to save outputs in place.
This is an offline adaptation, NOT the original GUI benchmark or oracle proof.
Source: https://github.com/xlang-ai/OSWorld/tree/8903ebeb0efdf74331bd7dc37ed91ccbbff22317
Metric directory within that pinned tree: desktop_env/evaluators/metrics
Apache-2.0 attribution: licenses/OSWorld-LICENSE. OCR is deliberately unavailable:
any metric path requesting it fails explicitly rather than downloading a model.
"""

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

from fetch_task import PIN, assets, check_pin, git, sha256, vm_relative

MODULES = {
    "compare_table": "table",
    "compare_pptx_files": "slides",
    "compare_docx_files": "docs",
    "compare_font_names": "docs",
    "check_json_settings": "vscode",
    "check_json_keybindings": "vscode",
    "compare_docx_tables": "docs",
    "compare_docx_files_and_ignore_new_lines": "docs",
    "compare_text_file": "vscode",
    "compare_zip_files": "vscode",
}


class UnsupportedMetricPath(RuntimeError):
    """A required online/model operation is outside this offline adapter."""


def no_ocr(*args, **kwargs):
    raise UnsupportedMetricPath("OCR/model-backed metric path is not supported offline")


def checked_path(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("path escapes configured root")
    return path


def load_task(task_file, cache, task_id, root):
    """Verify generated metadata against git objects, not just a self-reported pin."""
    manifest = json.loads((task_file.parent / "task-manifest.json").read_text())
    check_pin(root, manifest["source_pin"])
    raw = task_file.read_bytes()
    if manifest["tasks_sha256"] != sha256(raw):
        raise ValueError("tasks.json SHA256 mismatch")
    matches = [task for task in json.loads(raw) if task["id"] == task_id]
    if len(matches) != 1:
        raise ValueError("task ID must occur exactly once")
    task = matches[0]
    record = manifest["tasks"][task_id]
    rel = f"evaluation_examples/examples/{task['_domain']}/{task_id}.json"
    original = git(root, "show", f"{PIN}:{rel}")
    if record["path"] != rel or record["sha256"] != sha256(original):
        raise ValueError("upstream task provenance mismatch")
    public = json.loads(original)
    if {
        k: v for k, v in task.items() if k not in ("_domain", "_tier", "_development_only")
    } != public:
        raise ValueError("task definition differs from pinned public source")
    expected_assets = assets(task)
    for rel, item in expected_assets.items():
        asset = manifest["assets"].get(rel, {})
        if any(asset.get(k) != v for k, v in item.items()):
            raise ValueError("fixture source map mismatch")
        # Cache configuration points to the cache/ directory, not the data root.
        path = checked_path(cache, rel.removeprefix("cache/"))
        if not asset.get("sha256") or not path.is_file():
            raise ValueError("fixture missing/unhashed; run fetch_task.py with --download")
        import hashlib

        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != asset["sha256"]:
            raise ValueError(f"fixture SHA256 mismatch: {rel}")
    return task


def metric_function(root, function):
    if function not in MODULES:
        raise ValueError(f"unsupported metric: {function!r}")
    # Do not execute package __init__ files (gym/VM controller side effects).
    for name, rel in [
        ("desktop_env", "desktop_env"),
        ("desktop_env.evaluators", "desktop_env/evaluators"),
        ("desktop_env.evaluators.metrics", "desktop_env/evaluators/metrics"),
    ]:
        module = types.ModuleType(name)
        module.__path__ = [str(root / rel)]
        sys.modules[name] = module
    ocr = types.ModuleType("easyocr")
    setattr(ocr, "Reader", no_ocr)
    sys.modules["easyocr"] = ocr
    module = importlib.import_module("desktop_env.evaluators.metrics." + MODULES[function])
    return getattr(module, function)


def configured(args, attr, env):
    value = getattr(args, attr) or os.environ.get(env)
    data_dir = os.environ.get("BENCH_TASKS_DIR")
    if not value and data_dir and attr in ("tasks_file", "task_cache"):
        value = str(Path(data_dir) / {"tasks_file": "tasks.json", "task_cache": "cache"}[attr])
    if not value:
        raise ValueError(f"set {env} or --{attr.replace('_', '-')}")
    return Path(value).expanduser().resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--osworld-root")
    parser.add_argument("--tasks-file")
    parser.add_argument("--task-cache")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="verify pin, assets and metric imports; do not score",
    )
    args = parser.parse_args()
    root = configured(args, "osworld_root", "BENCH_OSWORLD_ROOT")
    task_file = configured(args, "tasks_file", "BENCH_TASKS_FILE")
    cache = configured(args, "task_cache", "BENCH_TASK_CACHE")
    task = load_task(task_file, cache, args.task_id, root)
    ev = task["evaluator"]
    if ev["result"].get("type") != "vm_file":
        raise ValueError("only vm_file result getters are supported")
    function = metric_function(root, ev["func"])
    if args.preflight:
        print(
            json.dumps(
                {
                    "preflight": "ok",
                    "task_id": args.task_id,
                    "source_pin": PIN,
                    "metric": ev["func"],
                    "oracles_validated": False,
                }
            )
        )
        return
    result = checked_path(args.workdir.resolve(), str(vm_relative(ev["result"]["path"])))
    if not result.is_file():
        print(json.dumps({"score": 0.0, "reason": "result file missing"}))
        return
    exp, options = ev.get("expected") or {}, ev.get("options") or {}
    if exp["type"] == "cloud_file":
        expected = checked_path(cache, f"{args.task_id}/gold/{exp['dest']}")
        expected = str(expected)
    else:
        expected = exp["rules"]
    try:
        score = float(function(str(result), expected, **options))
    except UnsupportedMetricPath:
        raise
    except Exception as exc:
        # Preserve source-adapter behavior for malformed artifacts, but do not
        # turn missing dependencies or setup failures into ordinary task failures.
        if isinstance(exc, (ImportError, FileNotFoundError)):
            raise
        print(
            json.dumps(
                {"score": 0.0, "reason": f"checker error: {type(exc).__name__}: {exc}"[:400]}
            )
        )
        return
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("upstream metric returned an invalid score")
    print(json.dumps({"score": score}))


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        OSError,
        KeyError,
        ImportError,
        UnsupportedMetricPath,
        subprocess.CalledProcessError,
    ) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:600]}))
        sys.exit(2)
