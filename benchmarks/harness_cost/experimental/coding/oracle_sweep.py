"""Build the Terminal-Bench 2.1 task subset and validate it with Harbor's oracle agent.

WHY: the coding arm of the harness cost benchmark runs a fixed, seeded subset of
Terminal-Bench 2.1 rather than all 89 tasks (budget). A subset task is only
useful if it is solvable in THIS environment (Docker Desktop on Apple Silicon,
amd64 images under emulation), so each task's reference solution (`-a oracle`,
no model calls, free) must score 1.0 here before any harness is measured on it.

Subset rule (decided before any harness ran, so it cannot be tuned to results):
  1. all 89 tasks of terminal-bench/terminal-bench-2-1;
  2. drop qemu-* and install-windows-3.11 (a VM inside an emulated container);
  3. drop tasks whose Docker Hub repository storage exceeds 1 GB (disk is
     shared and has a hard floor; this biases toward lighter tasks -- disclose);
  4. sort, shuffle with random.Random(20261103), walk the order;
  5. take a task if its oracle scores 1.0, otherwise skip it and record the
     replacement; stop at N=30 accepted tasks.
The shuffle is deterministic for a given CPython major version (3.x random's
shuffle algorithm); the Python version is recorded in the output.

CONSTRAINTS: one task image on disk at a time. Before each pull we check free
space on the data volume and stop if it would fall below FLOOR_MB; after each
task only that image is removed (never a global prune). Stop at 17 GiB to
protect the shared 15 GiB emergency floor. Storage checks use portable disk_usage.

Usage: oracle_sweep.py [--n 30]
Writes subset.json (accepted tasks with image digests) and oracle_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path
from typing import TypedDict

# Every docker call goes through the rig's pull guard (R6 F17); egress.py is a
# sibling, importable because this script runs from its own directory.
from egress import run_docker

HERE = Path(__file__).resolve().parent
TASKS = Path(os.environ.get("CODING_TASK_ROOT", "tasks/terminal-bench-2-1")).resolve()
OUTPUT = Path(os.environ.get("CODING_ORACLE_OUTPUT", "oracle-output")).resolve()
SEED = 20261103
SIZE_CAP = 1_000_000_000
FLOOR_MB = 17 * 1024  # stop above the shared 15 GiB hard floor, with headroom
START_MB = 21 * 1024  # leave ~4 GiB for verifier installs / unpacked image growth
DATA_VOL = os.environ.get("CODING_DISK_VOLUME", str(Path.cwd()))


def free_mb() -> int:
    return shutil.disk_usage(DATA_VOL).free // (1024 * 1024)


class _HubMeta(TypedDict):
    repo_storage: int | None
    tag_full_size: int
    archs: list[str]
    hub_digest: str | None


class _TaskMeta(_HubMeta):
    image: str


class _OracleResult(TypedDict):
    exit: int
    reward: float | None
    wall_s: float
    disk_stop: bool
    min_free_mb: int
    tail: str


def hub_meta(image: str) -> _HubMeta:
    repo, tag = image.split(":")
    r = json.load(
        urllib.request.urlopen(f"https://hub.docker.com/v2/repositories/{repo}/", timeout=30)
    )
    t = json.load(
        urllib.request.urlopen(
            f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}", timeout=30
        )
    )
    return {
        "repo_storage": r.get("storage_size"),
        "tag_full_size": t.get("full_size"),
        "archs": sorted({i.get("architecture") for i in t.get("images", [])}),
        "hub_digest": t.get("digest"),
    }


def seeded_order() -> tuple[list[str], dict[str, _TaskMeta]]:
    tasks = {
        d.name: tomllib.loads((d / "task.toml").read_text())
        for d in sorted(TASKS.iterdir())
        if (d / "task.toml").exists()
    }
    meta: dict[str, _TaskMeta] = {}
    for n, c in tasks.items():
        meta[n] = {
            "image": c["environment"]["docker_image"],
            **hub_meta(c["environment"]["docker_image"]),
        }
    cand = [
        n
        for n in tasks
        if not n.startswith("qemu-")
        and n != "install-windows-3.11"
        and (meta[n]["repo_storage"] or 0) <= SIZE_CAP
    ]
    order = sorted(cand)
    random.Random(SEED).shuffle(order)
    return order, meta


def run_oracle(name: str) -> _OracleResult:
    jobs = OUTPUT / "jobs-oracle"
    job = f"oracle-{name}"
    retry = 0
    while (jobs / job).exists():
        retry += 1
        job = f"oracle-{name}-retry{retry}"
    t0 = time.time()
    # The filesystem is shared: a pre-pull check alone cannot protect the
    # floor while a verifier installs dependencies or another session writes.
    output = OUTPUT / ("oracle-" + name + ".out")
    with output.open("w") as fh:
        p = subprocess.Popen(
            [
                "harbor",
                "run",
                "-p",
                str(TASKS / name),
                "-a",
                "oracle",
                "-o",
                str(jobs),
                "--job-name",
                job,
                "-n",
                "1",
                "-q",
                "-y",
                "--timeout-multiplier",
                "3",
            ],
            stdout=fh,
            stderr=fh,
            start_new_session=True,
        )
        minimum, disk_stop = free_mb(), False
        while p.poll() is None:
            minimum = min(minimum, free_mb())
            if minimum < FLOOR_MB:
                disk_stop = True
                os.killpg(p.pid, signal.SIGINT)
                break
            time.sleep(5)
        try:
            code = p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            code = p.wait()
    rewards = [float(f.read_text().strip()) for f in (jobs / job).glob("*/verifier/reward.txt")]
    return {
        "exit": code,
        "reward": rewards[0] if rewards else None,
        "wall_s": round(time.time() - t0, 1),
        "disk_stop": disk_stop,
        "min_free_mb": minimum,
        "tail": output.read_text()[-400:] if code else "",
    }


def image_digest(image: str) -> str | None:
    p = run_docker(
        ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"],
        capture_output=True,
        text=True,
    )
    try:
        ds = json.loads(p.stdout)
        return ds[0].split("@", 1)[1] if ds else None
    except Exception:
        return None


def drop_image(image: str) -> None:
    # Remove only this task image. A global prune could remove another lane's
    # unused-but-needed image; Docker Desktop trims the freed blocks itself.
    run_docker(["docker", "rmi", image], capture_output=True, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    if not a.execute:
        ap.error("oracle dispatch requires --execute; no network or Docker preflight was performed")
    if not 1 <= a.n <= 30:
        ap.error("n must be 1..30")
    if "CODING_TASK_ROOT" not in os.environ or "CODING_ORACLE_OUTPUT" not in os.environ:
        ap.error("explicit CODING_TASK_ROOT and CODING_ORACLE_OUTPUT needed")
    if OUTPUT == HERE or HERE in OUTPUT.parents:
        ap.error("output must be outside published source")
    if free_mb() < START_MB:
        ap.error("need >=21 GiB free before any network or Docker work")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    order, meta = seeded_order()
    log = OUTPUT / "oracle_log.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            done[r["task"]] = r
    previous_min = free_mb()
    if (OUTPUT / "subset.json").exists():
        previous_min = min(
            previous_min,
            json.loads((OUTPUT / "subset.json").read_text()).get(
                "min_free_mb_during_sweep", previous_min
            ),
        )
    accepted, skipped, hw = [], [], {"min_free_mb": previous_min}
    for name in order:
        if len(accepted) >= a.n:
            break
        if name in done and done[name].get("reward") is not None:
            r = done[name]
        else:
            m = meta[name]
            need = (
                int(m["tag_full_size"] / (1024 * 1024) * 3) + 512
            )  # compressed -> unpacked + VM growth, conservative
            if free_mb() < START_MB or free_mb() - need < FLOOR_MB:
                print(
                    f"STOP: free {free_mb()} MB; need >= {START_MB} MB before work "
                    f"and estimated post-pull >= {FLOOR_MB} MB",
                    file=sys.stderr,
                )
                break
            res = run_oracle(name)
            hw["min_free_mb"] = min(hw["min_free_mb"], free_mb(), res.get("min_free_mb", free_mb()))
            dig = image_digest(m["image"])
            drop_image(m["image"])
            r = {
                "task": name,
                **res,
                "image": m["image"],
                "image_digest": dig or m["hub_digest"],
                "hub_tag_full_size": m["tag_full_size"],
                "repo_storage": m["repo_storage"],
                "ts": time.time(),
            }
            with log.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(json.dumps({k: r[k] for k in ("task", "reward", "wall_s")}), flush=True)
        if r.get("reward") is None or r.get("disk_stop"):
            print(
                f"STOP: infrastructure failure for {name}; retain candidate order for retry",
                file=sys.stderr,
            )
            break
        (accepted if r.get("reward") == 1.0 else skipped).append(r)
    out = {
        "bench": "terminal-bench-2.1",
        "dataset": "terminal-bench/terminal-bench-2-1",
        "seed": SEED,
        "python": sys.version.split()[0],
        "size_cap_bytes": SIZE_CAP,
        "candidates": len(order),
        "seeded_order": order,
        "tasks": [
            {
                "name": r["task"],
                "task8": r["task"][:8],
                "image": r["image"],
                "image_digest": r["image_digest"],
                "hub_tag_full_size_bytes": r.get("hub_tag_full_size"),
                "oracle_reward": r["reward"],
            }
            for r in accepted
        ],
        "replaced": [
            {"name": r["task"], "oracle_reward": r.get("reward"), "exit": r.get("exit")}
            for r in skipped
        ],
        "complete": len(accepted) == a.n,
        "target_count": a.n,
        "min_free_mb_during_sweep": hw["min_free_mb"],
    }
    (OUTPUT / "subset.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"accepted {len(accepted)} skipped {len(skipped)} min_free {hw['min_free_mb']} MB")


if __name__ == "__main__":
    main()
