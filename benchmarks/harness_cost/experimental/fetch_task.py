"""Materialize explicitly selected public OSWorld tasks outside this source tree.

WHY: task text and fixtures are not redistributed with the benchmark. Read git
objects at a full audited pin, preserve setup/evaluator maps, and record hashes
so mutable upstream fixture URLs cannot silently define a reproducible run.
Source: https://github.com/xlang-ai/OSWorld/tree/8903ebeb0efdf74331bd7dc37ed91ccbbff22317
OSWorld is Apache-2.0; see licenses/OSWorld-LICENSE. No setup commands execute here.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

PIN = "8903ebeb0efdf74331bd7dc37ed91ccbbff22317"
SMOKE = "37608790-6147-45d0-9f20-1137bb35703d"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check_pin(root, pin):
    if pin != PIN:
        raise ValueError(f"only the reviewed full source pin {PIN} is supported")
    if git(root, "rev-parse", "HEAD").decode().strip() != pin:
        raise ValueError("OSWorld checkout HEAD does not match --pin")
    if git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "desktop_env",
        "evaluation_examples",
    ):
        raise ValueError("OSWorld task/metric tree must be clean, including untracked files")


def vm_relative(value):
    if not isinstance(value, str) or not value.startswith("/home/user/"):
        raise ValueError(f"unsupported VM path: {value!r}")
    rel = PurePosixPath(value[len("/home/user/") :])
    if not rel.parts or ".." in rel.parts or rel.is_absolute():
        raise ValueError("unsafe VM path")
    return rel


def assets(task):
    """Keep runner_v2's input-basename map; reject collisions instead of flattening silently."""
    found = {}
    for step in task.get("config", []):
        if step["type"] == "download":
            for item in step["parameters"]["files"]:
                rel = vm_relative(item["path"])
                key = f"cache/{task['id']}/input/{rel.name}"
                value = {"url": item["url"], "vm_path": item["path"]}
                if key in found and found[key] != value:
                    raise ValueError(f"input basename collision: {key}")
                found[key] = value
    ev = task["evaluator"]
    vm_relative(ev["result"]["path"])
    exp = ev.get("expected") or {}
    if exp.get("type") == "cloud_file":
        dest = exp["dest"]
        if not isinstance(dest, str) or PurePosixPath(dest).name != dest or dest in (".", ".."):
            raise ValueError("gold dest must be a basename")
        found[f"cache/{task['id']}/gold/{dest}"] = {"url": exp["path"]}
    elif exp.get("type") != "rule":
        raise ValueError("only cloud_file and rule expectations are supported")
    return found


def public_url(url):
    parsed = urllib.parse.urlparse(url)
    # Restrict downloads to the public fixture collection, not task-supplied arbitrary hosts.
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or not parsed.path.startswith("/datasets/xlangai/ubuntu_osworld_file_cache/resolve/")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"unsupported public fixture URL: {url!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osworld-root", required=True, type=Path)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        metavar="DOMAIN:UUID:TIER",
        help="repeat explicitly; tier is short or long",
    )
    parser.add_argument(
        "--download", action="store_true", help="explicitly permit HTTPS fixture downloads"
    )
    args = parser.parse_args()
    root, out = args.osworld_root.resolve(), args.output_dir.resolve()
    check_pin(root, args.pin)
    source = Path(__file__).resolve().parents[3]
    if out == source or source in out.parents or out == root or root in out.parents:
        raise ValueError("output must be external to the benchmark and OSWorld source directories")
    tasks, records, planned, seen = [], {}, {}, set()
    for selection in args.task:
        domain, task_id, tier = selection.split(":")
        if not re.fullmatch(r"[a-z_]+", domain) or not re.fullmatch(
            r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", task_id
        ):
            raise ValueError("invalid public task domain/UUID")
        if tier not in ("short", "long") or task_id in seen:
            raise ValueError("tier must be short/long and task IDs unique")
        seen.add(task_id)
        rel = f"evaluation_examples/examples/{domain}/{task_id}.json"
        raw = git(root, "show", f"{args.pin}:{rel}")
        task = json.loads(raw)
        if task["id"] != task_id:
            raise ValueError("upstream task ID mismatch")
        task.update(_domain=domain, _tier=tier)
        if task_id == SMOKE:
            task["_development_only"] = True
        planned.update(assets(task))
        records[task_id] = {"path": rel, "sha256": sha256(raw), "domain": domain, "tier": tier}
        tasks.append(task)
    payload = (json.dumps(tasks, indent=2, ensure_ascii=False) + "\n").encode()
    # Never overwrite a previous materialization; differing tasks need their own directory.
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "source_url": "https://github.com/xlang-ai/OSWorld",
        "source_pin": args.pin,
        "tasks_sha256": sha256(payload),
        "tasks": records,
        "downloads_complete": False,
        "assets": {},
    }
    for rel, item in sorted(planned.items()):
        public_url(item["url"])
        record: dict[str, str | None] = dict(item, sha256=None)
        if args.download:
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            # The redirect is necessary for Hugging Face's public CDN. No auth headers are sent.
            with (
                urllib.request.urlopen(item["url"], timeout=60) as response,
                target.open("xb") as dest,
            ):
                digest = hashlib.sha256()
                while block := response.read(1024 * 1024):
                    dest.write(block)
                    digest.update(block)
            record["sha256"] = digest.hexdigest()
        manifest["assets"][rel] = record
    manifest["downloads_complete"] = args.download or not planned
    (out / "tasks.json").write_bytes(payload)
    (out / "task-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "assets": len(planned),
                "output_dir": str(out),
                "downloads_complete": manifest["downloads_complete"],
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        print(f"fetch_task: {exc}", file=sys.stderr)
        sys.exit(2)
