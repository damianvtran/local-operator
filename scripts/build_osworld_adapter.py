#!/usr/bin/env python3
"""Materialise the OSWorld V2 adapter workspace and compute its digests.

This is the build-time step that needs the gated Hugging Face task corpus
(`xlangai/osworld_v2_tasks`, gated: "auto") and so is a human/operator command,
NOT a CI job — CI has no HF_TOKEN and no gate acceptance.

What it produces, and why each piece exists:

- ``adapter-release.json`` — exactly ``{"release_digest": "<64 hex>"}`` and
  nothing else. ``discovery.verify_release_manifest`` requires that canonical
  shape, and the worker refuses to launch without it.
- ``tasks/task_*.py`` — the gated task classes, materialised INTO the
  workspace so the episode's ``workspace_digest`` pins the exact task bytes
  that were run (that is what ``EpisodeSpec.task_digest`` binds to).
- ``benchmark_release.json`` — a copy of the V2 release manifest, which
  supplies the benchmark_release / environment_digest pins.
- ``task_hashes.json`` — the 108-task sha256 manifest, verified against the
  downloaded bytes so a corrupt or swapped task fails the build, not an
  episode.

The workspace must contain no symlinks and no hardlinks
(``discovery.workspace_digest`` rejects both), which is why the tasks are
COPIED in, never linked, and why the source tree (with its .git) is never the
workspace.

Usage:
    python scripts/build_osworld_adapter.py \
        --benchmark-release osworld-v2-2026.08.08 \
        --out /opt/lop-adapters/osworld-v2/<version>/workspace
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import sys
from pathlib import Path


def _release_digest(
    *, version: str, package_digest: str, benchmark_release: str, task_manifest_sha256: str
) -> str:
    """Our attestation of the build, tying the harness build to the benchmark
    release — the claim a leaderboard number must carry."""

    payload = (
        f"lop-osworld-v2-adapter|{version}|{package_digest}|"
        f"{benchmark_release}|{task_manifest_sha256}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_readonly(path: Path, data: bytes) -> None:
    path.write_bytes(data)
    # The workspace is immutable once built: a task byte that changed after
    # the digest was computed would make the evidence pin a lie.
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-release", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--package-digest", default="0" * 64)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "HF_TOKEN is not set. The OSWorld V2 task corpus is gated "
            "(xlangai/osworld_v2_tasks); accept the terms once and export a token.",
            file=sys.stderr,
        )
        return 2

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks").mkdir(exist_ok=True)

    # NOTE: the actual gated download lands with PR 2's AWS provider, which is
    # when a real workspace is first materialised. PR 1's tests build their
    # own minimal workspaces from fixture tasks, so this script's download body
    # is intentionally a stub that fails loudly rather than fabricate a
    # workspace that would produce unattested evidence.
    print(
        "build_osworld_adapter: the gated HF download is wired in PR 2 alongside\n"
        "the AWS provider. PR 1 tests materialise fixture workspaces directly.\n"
        f"Requested release: {args.benchmark_release} -> {out}",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
