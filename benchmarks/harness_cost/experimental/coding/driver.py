"""Portable opt-in coding dispatcher; publication is NOT a runnable preregistration.

WHY: the source lane stopped before a paid coding pilot. An example must not
silently turn an unfinished subset or an installed release into frozen evidence.
Paid dispatch is unconditionally blocked until executable/runtime identity and
retained tool-wire capture admission are implemented; self-asserted hashes and
verification booleans are not evidence. The remaining preflight is source-only.
CONSTRAINTS: external task/toolchain/output paths, one image at a time, no host
profiles/auth DB, no payloads here. An external meter owns OPENROUTER_API_KEY.

Budget authority (R6 F19): the external ``meter_proxy`` ledger is the SINGLE
spend authority for this lane, exactly as for the desktop lane -- every
container route (meter port, LoP edge, Codex bridge -> translator -> meter)
ends at it, and it enforces trial and campaign caps on durable reservations.
The former lane-local reservation guard sat on no container route once the gate
existed, so it was removed rather than kept as a silent second layer; the
``--*-budget-usd`` arguments must equal the meter's BENCH_BUDGET_USD and
BENCH_TRIAL_BUDGET_USD: preflight reads them from its /health and refuses on any
difference (R7 F22), so a meter started with other caps cannot silently govern.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC_LOADOUT = (HERE / "../../public_loadout").resolve()
SOURCE_REVISION = "500883ef56ca504325ffd80c5b90bf2435d00e3f"  # released v0.94.22
BRIDGE_PORT = 8789  # lane-owned Codex Responses bridge, loopback only
HARNESS = {
    "claude-code": ("FrozenClaudeCode", "2.1.274"),
    "codex-best": ("FrozenCodex", "0.147.0"),
    "opencode": ("OpenCodeV2", "2.0.27"),
    "local-operator": ("LocalOperator", "500883ef-v0.94.22"),
    "omp": ("Omp", "18.11.2"),
    "hermes": ("FrozenHermes", "0.22.6"),
    "dsh": ("Dsh", "0.2.0-rc.2"),
}
# Hermes' CLI says "vunknown" and its dist metadata 0.0.0, so its version is
# derived (hermes_version), never read from the binary: the built source is
# this commit, the peeled target of both rc.3-v0.22.6 and the final v0.22.6
# (an automated promotion of rc.3), per the frozen ls-remote capture.
HERMES_SOURCE_REVISION = "f0d94882e89ab8d4098b5bb2f166adfb9ee0a0ab"
HERMES_TAG_EVIDENCE = "hermes-tag-evidence.json"
# Tracked files of HERMES_SOURCE_REVISION (git ls-tree), all hashed by the host
# source check. Exact, not "> 0": a partial tree must not verify (R13 M1).
HERMES_CHECKED_FILES = 22066


def hermes_version(source_verification, evidence=None):
    """Hermes' version from its verified source tree, or raise.

    ``source_verification`` is the host source check (source-verification.json):
    every tracked file of HERMES_SOURCE_REVISION hashed against the toolchain
    copy. The version is the release tag the frozen ls-remote lines peel to
    that commit; HARNESS["hermes"] must equal it. Runtime builders call this
    instead of parsing ``hermes --version``.

    The real record lists offending paths, so missing/mismatched/
    unsupported_git_entries must be EMPTY LISTS (S9e: comparing them with 0
    refused every real record); any other type refuses.
    """
    tree = source_verification.get("hermes_tracked_source") or {}
    if (tree.get("commit") != HERMES_SOURCE_REVISION or tree.get("all_checked_match") is not True
            or type(tree.get("checked_files")) is not int
            or tree["checked_files"] != HERMES_CHECKED_FILES
            or any(tree.get(k) != [] or not isinstance(tree.get(k), list)
                   for k in ("missing", "mismatched", "unsupported_git_entries"))):
        raise ValueError("Hermes source tree is not the verified pinned commit")
    if evidence is None:
        evidence = json.loads((HERE / HERMES_TAG_EVIDENCE).read_text())
    lines, pin = set(evidence.get("ls_remote_lines", [])), HARNESS["hermes"][1]
    for tag in (f"v{pin}", f"rc.3-v{pin}"):
        if f"{HERMES_SOURCE_REVISION}\trefs/tags/{tag}^{{}}" not in lines:
            raise ValueError(f"Hermes tag evidence does not peel {tag} to the pinned commit")
    return pin


def check_source_revision(harness, record):
    """Source-pinned harnesses: the runtime record must name the pinned commit."""
    if harness == "local-operator" and record.get("source_revision") != SOURCE_REVISION:
        raise ValueError("Local Operator must be released v0.94.22 source 500883ef")
    if harness == "hermes" and record.get("source_revision") != HERMES_SOURCE_REVISION:
        raise ValueError("Hermes must be verified source f0d94882 (v0.22.6)")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen(path, expected):
    if len(expected) != 64 or digest(path) != expected:
        raise ValueError("frozen artifact SHA-256 mismatch")
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("frozen artifact must be an object")
    return value


def free_mb(volume):
    return shutil.disk_usage(volume).free // (1024 * 1024)


def check_space(volume, compressed):
    available = free_mb(volume)
    estimate = compressed * 3 / (1024 * 1024) + 512
    if available < 21 * 1024 or available - estimate < 17 * 1024:
        raise ValueError(
            "disk gate: need >=21 GiB free and estimated >=17 GiB after pull; hard floor 15 GiB"
        )


# The container metering closure: the shared desktop meter/edge/classifier
# bytes PLUS every coding-lane file on the paid container path. The zero-paid
# container S4 (egress denial, gold isolation, oracle=edge parity) is valid only
# for these exact bytes; any change invalidates its record.
CONTAINER_CLOSURE = (
    "../meter_proxy.py",
    "../lop_edge.py",
    "../radient_meter/classifier.py",
    "../radient_meter/json_types.py",
    "../radient_meter/model.py",
    "../radient_meter/sse_stream.py",
    "../package_egress.py",
    "adapters.py",
    "bench_verifier.py",
    "driver.py",
    "egress.py",
    "gate_relay.py",
    "gold_probe.py",
    "public-provenance.json",
    "post-agent-classes.json",
    # Re-derives the frozen COPY pairs at trial start (driver.public_provenance).
    "provenance_derive.py",
    "lop_launch.py",
    "package-egress-allowlist.json",
    "public-roots.pem",
)
# Agent-phase package egress (egress.py, ../package_egress.py): the policy code
# and its frozen exact-host allowlist. Both are closure members AND a named
# preregistration input, so widening the allowlist voids the container S4
# record and any frozen preregistration.
PACKAGE_ALLOWLIST = HERE / "package-egress-allowlist.json"
# Frozen public roots (certifi 2026.07.22, Mozilla; MPL-2.0 notice in
# ../licenses) appended to LoP's run CA so its tool children can verify the
# registries the edge tunnels (R8 F24). Frozen, not the host's store, so every
# run trusts the same bytes and the S4 record binds them.
PUBLIC_ROOTS = HERE / "public-roots.pem"
# Must equal lop_launch.PARITY_KEYS / lop_edge.CA_PARITY_KEYS (tested).
LOP_PARITY_KEYS = ("PIP_CERT", "CURL_CA_BUNDLE", "NODE_EXTRA_CA_CERTS")


def container_closure_hashes():
    return {name: digest(HERE / name) for name in CONTAINER_CLOSURE}


def package_egress_hashes():
    return {"policy": digest(HERE.parent / "package_egress.py"),
            "allowlist": digest(PACKAGE_ALLOWLIST),
            "public_roots": digest(PUBLIC_ROOTS)}


def package_agent_env():
    """Frozen tool-child env from the allowlist file, identical for all seven arms.

    Only ``UV_PYTHON_INSTALL_MIRROR`` exists (S9): it moves uv's managed-Python
    downloads off github.com, which stays denied host-wide as a contamination
    route. The mirror host must itself be allowlisted, so this file cannot name
    a route the policy would not already permit.
    """
    from urllib.parse import urlsplit

    data = json.loads(PACKAGE_ALLOWLIST.read_text())
    env = data.get("agent_env", {})
    if set(env) - {"UV_PYTHON_INSTALL_MIRROR"}:
        raise ValueError("package agent_env names an unreviewed variable")
    for value in env.values():
        url = urlsplit(value)
        if url.scheme != "https" or url.hostname not in data["allow"]:
            raise ValueError("package agent_env must name an allowlisted https mirror")
    return dict(env)


def package_egress_summary(logs):
    """Every package-egress connection of one trial, for its runs.jsonl record.

    ``logs`` are the gate's and (LoP) the edge's JSON-lines logs; a missing
    file means that side saw no connection. Disclosed, never priced: the bytes
    are registry traffic, not model traffic.
    """
    rows = []
    for log in logs:
        if Path(log).is_file():
            rows += [json.loads(line) for line in Path(log).read_text().splitlines() if line]
    return {
        "policy": package_egress_hashes(),
        "allowed": sum(1 for r in rows if r.get("allowed") is True),
        "denied": sum(1 for r in rows if r.get("allowed") is False),
        "bytes_up": sum(int(r.get("bytes_up", 0)) for r in rows),
        "bytes_down": sum(int(r.get("bytes_down", 0)) for r in rows),
        "connections": rows,
    }


def container_evidence(args):
    """Return the record digest iff a container S4 record matches these bytes.

    The record is an explicit CLI argument (``--coding-edge-evidence``), like
    every other frozen artifact: the hold is decided before ANY environment
    lookup, so no credential is touched on a refused run.

    Same shape as runner_v2.lop_edge_preflight (R4 F11): the record names an
    evidence artifact in its own directory whose bytes must still hash to the
    recorded digest, so the record is not self-attesting.
    """
    raw = getattr(args, "coding_edge_evidence", None)
    if not raw:
        return None
    path = Path(raw)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    rel = record.get("evidence_path")
    base = path.resolve().parent
    artifact = (base / rel).resolve() if isinstance(rel, str) and rel else None
    if (
        record.get("container_s4_passed") is not True
        or record.get("lop_head") != SOURCE_REVISION
        or record.get("files") != container_closure_hashes()
        or artifact is None
        or not artifact.is_relative_to(base)
        or not artifact.is_file()
        or digest(artifact) != record.get("evidence_sha256")
    ):
        return None
    return record["evidence_sha256"]


def block_paid_coding(args):
    # Hash-matching JSON proves byte identity, not executable or wire behavior.
    # Keep this hold before even auth-presence checks, including direct callers.
    # It lifts only for a container S4 record bound to the exact closure bytes.
    if container_evidence(args) is None:
        raise ValueError(
            "HARDBLOCK PAID CODING: no zero-paid container S4 record "
            "(--coding-edge-evidence) matches the current metering closure"
        )


# SHA-256 of zero bytes. An empty gold file (llm-inference-batching-scheduler's
# ``tests/__init__.py``) carries no information an agent could use, yet its
# digest matches every empty file in any image (/.dockerenv, /etc/subgid, pip's
# py.typed: 49 in the a18da995 sweep), refusing the task for every arm. ONLY
# this digest is excluded, and the exclusion is recorded in the manifest; no
# small-file threshold, because a short non-empty file can still be the answer.
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _gold_files(task_dir):
    task_dir = Path(task_dir)
    return {str(p.relative_to(task_dir)): digest(p) for sub in ("solution", "tests")
            for p in sorted((task_dir / sub).rglob("*")) if p.is_file()}


def gold_digests(task_dir):
    """SHA-256 of every non-empty host-side solution/ and tests/ file (hashes only)."""
    digests = sorted(set(_gold_files(task_dir).values()) - {EMPTY_SHA256})
    if not digests:
        raise ValueError("task has no solution/tests files to guard")
    return digests


def gold_exclusions(task_dir):
    """The disclosed exclusion record: which guarded files were zero-length."""
    empty = sorted(p for p, sha in _gold_files(task_dir).items() if sha == EMPTY_SHA256)
    return [{"sha256": EMPTY_SHA256, "files": empty,
             "reason": "zero-length file carries no information; its digest matches "
                       "every empty file in the image"}] if empty else []


def task_payload(task_dir):
    """SHA-256 over the whole task tree's ``relpath -> sha256`` map."""
    task_dir = Path(task_dir)
    files = {str(p.relative_to(task_dir)): digest(p)
             for p in sorted(task_dir.rglob("*")) if p.is_file()}
    if any(not p.resolve().is_relative_to(task_dir.resolve())
           for p in task_dir.rglob("*") if p.is_file()):
        raise ValueError("escaping task payload symlink")
    return hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), files


def public_provenance(task_dir, image_digest):
    """Freeze host-reviewed COPY pairs, never discoveries from a live image.

    Only the checked-in exact inventory can relax a strict gold match. Task
    payload and image changes require renewed review, not a learned baseline.
    """
    import provenance_derive

    task_dir = Path(task_dir)
    payload, files = task_payload(task_dir)
    policy_path = HERE / "public-provenance.json"
    policy = json.loads(policy_path.read_text())
    approved = policy["tasks"].get(task_dir.name)
    public = []
    if approved is not None:
        if (approved["task_payload_sha256"] != payload
                or approved["image_digest"] != image_digest
                or approved["dockerfile_sha256"] != files.get("environment/Dockerfile")):
            raise ValueError("stale public provenance; task/image requires review")
        # COPY pairs are re-derived from the task's own Dockerfile and must equal
        # the frozen review exactly, so a hand-edited or drifted pair is refused.
        try:
            derived, _runs = provenance_derive._copy_pairs(task_dir, set(gold_digests(task_dir)))
        except provenance_derive.Unsupported as exc:
            raise ValueError(f"stale public COPY input: {exc}") from exc
        frozen_copies = [e for e in approved["public_files"]
                         if e.get("origin") == "dockerfile-copy"]
        if sorted(derived, key=lambda e: e["path"]) != sorted(
                frozen_copies, key=lambda e: e["path"]):
            raise ValueError("stale public COPY input")
        for entry in approved["public_files"]:
            if entry.get("origin") == "dockerfile-copy":
                if files.get(entry["source"]) != entry["sha256"]:
                    raise ValueError("stale public COPY input")
            elif str(entry.get("origin", "")).startswith("build-time clone of "):
                # The clone's bytes are bound by the pinned commit, its script
                # by the task payload hash; the script must still be present.
                if entry.get("build_input") not in files or not re.fullmatch(
                        r"build-time clone of https://\S+@[0-9a-f]{40}", entry["origin"]):
                    raise ValueError("stale public clone input")
            else:
                raise ValueError("public pair without a reviewed origin")
            public.append({"path": entry["path"], "sha256": entry["sha256"]})
    solutions = sorted({digest(p) for p in (task_dir / "solution").rglob("*")
                        if p.is_file()} - {EMPTY_SHA256})
    manifest = {"version": 1, "task_payload_sha256": payload,
                "image_digest": image_digest, "policy_sha256": digest(policy_path),
                "public_files": public, "solution_digests": solutions,
                "gold_digests": gold_digests(task_dir),
                "gold_exclusions": gold_exclusions(task_dir)}
    classes = digest_classes(task_dir, payload, solutions)
    if classes is not None:
        manifest["digest_classes"] = classes
        manifest["classes_policy_sha256"] = digest(HERE / "post-agent-classes.json")
    return manifest


def digest_classes(task_dir, payload, solutions):
    """Frozen ``gold digest -> expected-output-fixture | verifier-hidden | solution``.

    S9c (R11 F27/F30): refuse is the default. Only a fixture a reviewer listed
    in post-agent-classes.json as an expected output -- with citations of the
    verifier lines that compare against it, re-validated here byte for byte
    -- may be disclosed and graded when it appears after the agent phase.
    Every other hidden tests/ digest is ``verifier-hidden`` and every
    solution-tree digest is ``solution`` (both refuse). A digest shared with
    any unlisted tests/ file, or with the solution tree, refuses. A task
    without a payload-bound entry gets ``None``: the old strict refusal.
    """
    import provenance_derive

    path = HERE / "post-agent-classes.json"
    frozen = json.loads(path.read_text()) if path.is_file() else {"tasks": {}}
    entry = frozen["tasks"].get(Path(task_dir).name)
    if entry is None:
        return None
    if entry.get("task_payload_sha256") != payload:
        raise ValueError("stale post-agent classes; task requires review")
    try:
        listed = provenance_derive.expected_output_fixtures(task_dir, entry)
    except provenance_derive.Unsupported as exc:
        raise ValueError(f"stale post-agent classes: {exc}") from exc
    tests = {str(p.relative_to(Path(task_dir) / "tests")): digest(p)
             for p in (Path(task_dir) / "tests").rglob("*") if p.is_file()}
    unlisted = {sha for rel, sha in tests.items() if rel not in listed}
    return {sha: "solution" if sha in solutions
            else "expected-output-fixture" if sha in listed.values() and sha not in unlisted
            else "verifier-hidden" for sha in gold_digests(task_dir)}


DISCLOSED_CLASS = "expected-output-fixture"


def post_agent_flags(job):
    """S9b/S9c disclosure for one trial, from BenchVerifier's host-side record.

    Rows are (path, digest, class); hashes and paths only, never bytes.
    ``graded`` says whether this trial's disclosures were actually graded
    (R11 F29). ``None`` when the verifier record is missing.
    """
    records = list(Path(job).glob("*/verifier/bench-isolation.json"))
    if len(records) != 1:
        return None
    record = json.loads(records[0].read_text())
    rows = record.get("post_agent_disclosed") or []
    return {"counts": {DISCLOSED_CLASS: sum(1 for r in rows if r.get("class") == DISCLOSED_CLASS)},
            "matches": rows, "graded": record.get("graded") is True,
            "fence_ok": (record.get("fence") or {}).get("ok"),
            "preflight_ok": (record.get("preflight_receipt") or {}).get("ok")}


def flag_table(output):
    """Publish ``task -> harness -> {graded, refused}`` disclosure counts (S9c).

    R11 F29: only graded trials count as disclosed-and-graded; trials whose
    late matches were refused (fence, receipt or other refusing matches) are
    counted separately, never folded into the graded cell.
    """
    table = {}
    for line in (Path(output) / "runs.jsonl").read_text().splitlines():
        row = json.loads(line)
        flags = row.get("post_agent_flags") or {}
        cell = table.setdefault(row.get("task", "?"), {}).setdefault(row["harness"], {
            "trials": 0, "graded": {"trials": 0, DISCLOSED_CLASS: 0},
            "refused": {"trials": 0, DISCLOSED_CLASS: 0}})
        cell["trials"] += 1
        count = (flags.get("counts") or {}).get(DISCLOSED_CLASS, 0)
        if count:
            side = cell["graded" if flags.get("graded") is True else "refused"]
            side["trials"] += 1
            side[DISCLOSED_CLASS] += count
    (Path(output) / "post-agent-flags.json").write_text(
        json.dumps(table, indent=1, sort_keys=True) + "\n")
    return table


@contextmanager
def egress_canary():
    """A run-owned, OPEN loopback listener the gate never forwards (R6 F16).

    The in-container egress probe dials it via the host-gateway name: because
    the port is known open, only a missing route can make that dial fail, so
    "refused" is not mistaken for "fenced". Never accepted or read; the kernel
    backlog completes a handshake, which is all a route test needs.
    """
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(8)
        yield sock.getsockname()[1]


def current_run_file(args):
    """The meter's ``BENCH_CURRENT_RUN_FILE``, named explicitly (R6 F19).

    Codex reaches the meter through a translator with a fixed base URL, so the
    meter tags its rows from this file. A mismatch is fail-closed, not silent:
    the meter would tag another id, ``wire_rows`` finds no rows for this run
    and the trial is recorded unverified and stops the lane.
    """
    path = getattr(args, "meter_current_run_file", None)
    if path is None or not Path(path).is_absolute():
        raise ValueError("--meter-current-run-file must be the meter's absolute path")
    return Path(path).resolve()


def check_meter_caps(args):
    """Refuse unless the running meter enforces exactly the CLI caps (R7 F22).

    The meter is the lane's single budget authority, so its caps -- not the
    CLI's -- are what bound spend. Exact float equality on purpose: both sides
    parse the same decimal text, and any drift means a different budget. A
    meter without a ``caps`` report (an older build) is refused, not trusted.
    """
    from urllib.request import urlopen

    with urlopen(args.meter_url.rstrip("/") + "/health", timeout=5) as response:
        caps = json.loads(response.read()).get("caps")
    if not isinstance(caps, dict) or (caps.get("total_usd"), caps.get("trial_usd")) != (
        args.total_budget_usd,
        args.trial_budget_usd,
    ):
        raise ValueError(
            "meter caps differ from --total-budget-usd/--trial-budget-usd "
            f"(meter reports {caps!r}); restart the meter with matching "
            "BENCH_BUDGET_USD/BENCH_TRIAL_BUDGET_USD"
        )


@contextmanager
def codex_current_run(args, harness, run_id):
    """Point the meter's current-run file at this Codex run, then restore it.

    R7 F21: restored in ``finally`` (success, error, timeout, interrupt) to its
    prior bytes, or removed if absent, so no later request through the bridge
    is tagged with a finished run's id. Non-Codex trials never touch it.
    """
    if harness != "codex-best":
        yield
        return
    path = current_run_file(args)
    prior = path.read_bytes() if path.exists() else None
    path.write_text(run_id)  # the meter tags Codex rows from it
    try:
        yield
    finally:
        if prior is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(prior)


def container_plan(run_id, harness, image, *, meter_port, bridge_port, edge=None):
    """Harbor arguments that put one paid trial behind the gate (egress.py).

    ``edge`` (LoP only) is ``{"port": int, "env": dict, "ca_file": Path}`` from
    the host lop_edge; its loopback proxy URL is rewritten to the gate host on
    the SAME port, so the gate relays exactly the edge. Returns ``ports`` (the
    gate map), ``env`` (``--ae`` entries) and ``urls`` for the adapters.
    """
    import egress
    from urllib.parse import urlsplit

    # The bridge reaches the meter as /r/@current, tagged from the meter's
    # current-run file, which names a Codex run only during that run. Any other
    # harness that could reach it would bill under a stale Codex id and bypass
    # its own trial cap (R7 F21), so only Codex trials get the bridge port.
    codex = harness == "codex-best"
    ports = [meter_port] + ([bridge_port] if codex else []) + ([edge["port"]] if edge else [])
    urls = {"anthropic": egress.gate_url(meter_port, f"/r/{run_id}")}
    if codex:
        urls["codex"] = egress.gate_url(bridge_port, "/v1")
    env = {}
    if harness != "local-operator":
        # Package route for tool children (apt reads http(s)_proxy from env;
        # pip, uv, curl and git honour the same). NO_PROXY exempts the gate
        # host: every model URL above is http://bench-gate:<port>, so model
        # traffic goes straight to the meter/bridge, never into the proxy
        # (which would deny it anyway). Both spellings, identical values.
        proxy = f"http://{egress.GATE_SERVICE}:{egress.PACKAGE_PROXY_PORT}"
        no_proxy = f"{egress.GATE_SERVICE},127.0.0.1,localhost,::1"
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            env[key] = proxy
        env.update(NO_PROXY=no_proxy, no_proxy=no_proxy)
    if harness == "local-operator":
        if not edge:
            raise ValueError("LoP runs only behind its host edge")
        child = dict(edge["env"])
        proxy = urlsplit(child["HTTPS_PROXY"])
        if proxy.hostname != "127.0.0.1" or proxy.port != edge["port"]:
            raise ValueError("edge announced an unexpected proxy address")
        gated = child["HTTPS_PROXY"].replace(
            f"@127.0.0.1:{edge['port']}", f"@{egress.GATE_SERVICE}:{edge['port']}"
        )
        for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            child[key] = gated
        # The mounted trust bundle (run CA + public roots, R8 F24) for every
        # trust variable the launcher checks; the edge pins its CA fingerprint.
        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", *LOP_PARITY_KEYS):
            child[key] = egress.CA_TARGET
        env.update(child)
    elif edge:
        raise ValueError("only LoP is metered through the edge")
    # Same mirror pin for every arm, LoP included (adapters.EDGE_ENV_KEYS).
    env.update(package_agent_env())
    return {"ports": ports, "env": env, "urls": urls, "image": image}


def preflight(args):
    """Fail closed before credentials, artifact reads or dispatch side effects."""
    block_paid_coding(args)
    for value in (
        args.total_budget_usd,
        args.trial_budget_usd,
        args.input_usd_per_m,
        args.output_usd_per_m,
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("explicit positive finite caps and price bounds required")
    if args.trial_budget_usd > args.total_budget_usd:
        raise ValueError("trial cap exceeds total cap")
    if not args.billing_tier.strip():
        raise ValueError("explicit billing tier required")
    prereg = frozen(args.preregistration, args.preregistration_sha256)
    runtime = frozen(args.runtime_attestation, args.runtime_sha256)
    wire = frozen(args.toolwire_attestation, args.toolwire_sha256)
    oracle = frozen(args.oracle_attestation, args.oracle_sha256)
    if not args.tasks.strip() or not args.harnesses.strip() or args.rep < 0:
        raise ValueError("nonempty task/harness lists and nonnegative rep needed")
    current_run_file(args)
    for url in (args.meter_url, args.translator_url):
        from urllib.parse import urlsplit

        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("transport URL must be credential-free HTTP(S)")
    check_meter_caps(args)
    if args.output.resolve() == HERE or HERE in args.output.resolve().parents:
        raise ValueError("outputs must be outside published source")
    if (
        prereg.get("frozen") is not True
        or runtime.get("frozen") is not True
        or wire.get("verified") is not True
    ):
        raise ValueError("frozen prereg/runtime and verified toolwire attestations required")
    # S9d: the toolwire attestation must cover exactly the preregistered arms;
    # a subset (an arm excluded as RED) or superset attests a different run.
    attested, planned = wire.get("harnesses"), prereg.get("harnesses")
    if (not isinstance(attested, list) or not isinstance(planned, list) or not attested
            or len(set(attested)) != len(attested) or set(attested) != set(planned)):
        raise ValueError("toolwire harness set differs from the preregistration")
    expected = {
        "billing_tier": args.billing_tier,
        "model": args.model,
        "harnesses": args.harnesses.split(","),
        "tasks": args.tasks.split(","),
        "rep": args.rep,
        "total_budget_usd": args.total_budget_usd,
        "trial_budget_usd": args.trial_budget_usd,
        "input_usd_per_m": args.input_usd_per_m,
        "output_usd_per_m": args.output_usd_per_m,
        "runtime_sha256": args.runtime_sha256,
        "toolwire_sha256": args.toolwire_sha256,
        "subset_sha256": digest(args.subset),
        "oracle_sha256": args.oracle_sha256,
        "source_sha256": {p.name: digest(p) for p in sorted(HERE.glob("*.py"))},
        "catalog_sha256": digest(HERE / "codex-model-catalog.json"),
        "package_egress_sha256": package_egress_hashes(),
    }
    if any(prereg.get(k) != v for k, v in expected.items()):
        raise ValueError("requested run differs from frozen preregistration")
    schemas = PUBLIC_LOADOUT / "public_mcp_schemas.json"
    if (
        wire.get("schema_sha256") != digest(schemas)
        or wire.get("billing_tier") != args.billing_tier
    ):
        raise ValueError("toolwire schema or billing tier mismatch")
    if wire.get("runtime_sha256") != args.runtime_sha256 or wire.get("model") != args.model:
        raise ValueError("toolwire runtime or model mismatch")
    if (
        wire.get("preregistered_tools_observed") is not True
        or wire.get("output_limit_enforced") is not True
    ):
        raise ValueError("observed tool surface and enforced output limits required")
    if (
        wire.get("billing_tier_enforced") is not True
        or wire.get("guard_route_verified") is not True
    ):
        raise ValueError("explicit tier enforcement and guarded route evidence missing")
    if wire.get("text_only") is not True:
        raise ValueError("only text billing is conservatively priced")
    subset = json.loads(args.subset.read_text())
    if subset.get("complete") is not True or len(subset["tasks"]) != subset.get("target_count"):
        raise ValueError("subset incomplete: published snapshot is 21/30, not a frozen full run")
    if (
        oracle.get("frozen") is not True
        or oracle.get("verified") is not True
        or oracle.get("subset_sha256") != digest(args.subset)
    ):
        raise ValueError("external frozen oracle evidence missing or mismatched")
    tasks = {t["name"]: t for t in subset["tasks"]}
    if len(tasks) != 30:
        raise ValueError("frozen subset must have 30 unique task identifiers")
    names, harnesses = expected["tasks"], expected["harnesses"]
    if len(set(names)) != len(names) or len(set(harnesses)) != len(harnesses):
        raise ValueError("duplicate task/harness would reuse trial ID")
    for name in names:
        if Path(name).name != name or name in {".", ".."} or name not in tasks:
            raise ValueError("invalid selected task")
        proof = oracle.get("tasks", {}).get(name, {})
        if (
            proof.get("reward") != 1
            or proof.get("image_digest") != tasks[name].get("image_digest")
            or proof.get("task_checksum") != tasks[name].get("task_checksum")
        ):
            raise ValueError("selected task lacks matching external oracle evidence")
        if not (args.task_root / name / "task.toml").is_file():
            raise ValueError("external task payload missing")
        payload_hashes = prereg.get("task_files", {}).get(name, {})
        if not payload_hashes:
            raise ValueError("frozen external task file hashes missing")
        task_dir = (args.task_root / name).resolve()
        actual = {
            str(p.relative_to(task_dir)): digest(p) for p in task_dir.rglob("*") if p.is_file()
        }
        if actual != payload_hashes or any(
            not p.resolve().is_relative_to(task_dir) for p in task_dir.rglob("*")
        ):
            raise ValueError("external task payload hash mismatch or escaping symlink")
        image_digest = tasks[name].get("image_digest", "")
        if not image_digest.startswith("sha256:") or len(image_digest) != 71:
            raise ValueError("task image digest missing")
        if not isinstance(tasks[name].get("hub_tag_full_size_bytes"), int):
            raise ValueError("image compressed size missing")
    if len({tasks[n]["task8"] for n in names}) != len(names):
        raise ValueError("task8 collision would reuse run IDs")
    # The model guard cannot account for Radient classifier traffic. An
    # attestation cannot conjure an absent reservation/accounting path, and
    # disabling Radient would change the treatment. Hold paid LOP dispatch
    # until that separate gate has executable, independently validated code.
    # The edge meters classifier traffic on the shared ledger; that path is
    # admitted only by the container S4 record (block_paid_coding above).
    if "local-operator" in harnesses and container_evidence(args) is None:
        raise ValueError("paid Local Operator blocked: no container edge evidence")
    for harness in harnesses:
        if harness not in HARNESS:
            raise ValueError("unknown harness")
        record = runtime.get("runtimes", {}).get(harness, {})
        if record.get("version") != HARNESS[harness][1] or record.get("linux_verified") is not True:
            raise ValueError("runtime version or Linux verification missing")
        files = record.get("files", {})
        if not files:
            raise ValueError("frozen runtime file hashes missing")
        for relative, sha in files.items():
            path = (args.toolchain / relative).resolve()
            if not path.is_relative_to(args.toolchain.resolve()) or digest(path) != sha:
                raise ValueError("runtime file hash mismatch")
        check_source_revision(harness, record)
    if not args.request_log.is_file():
        raise ValueError("existing external metering log needed")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY environment credential required")
    check_space(args.disk_volume, max(tasks[n]["hub_tag_full_size_bytes"] for n in names))
    return [tasks[n] for n in names], harnesses


STOP_GRACE_SECONDS = 15
_OWNED_GROUPS = {}


def owned_process(*args, **kwargs):
    # Only our fresh session grants group-wide cleanup authority. Never infer
    # ownership from an arbitrary Popen PID (especially an already-exited one).
    if kwargs.get("start_new_session") is not True:
        raise ValueError("owned processes require a fresh session")
    proc = subprocess.Popen(*args, **kwargs)
    _OWNED_GROUPS[proc] = (os.getpid(), proc.pid)
    return proc


def _signal_owned_group(group, sig):
    try:
        os.killpg(group, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Darwin reports EPERM, not ESRCH, for an orphan group containing only
        # zombies or members already exiting (ps stat "Z" / "E"). Do not swallow
        # real permission failures: prove no live member remains, using a
        # read-only process table (no ambient process signals).
        rows = subprocess.run(
            ["ps", "-axo", "pgid=,stat="], capture_output=True, text=True, check=True, timeout=5
        ).stdout.splitlines()
        members = [
            row.split()[1] for row in rows if len(row.split()) == 2 and row.split()[0] == str(group)
        ]
        if any(not status.startswith(("Z", "E")) for status in members):
            raise
        return False


def stop(proc):
    ownership = _OWNED_GROUPS.get(proc)
    if ownership is None:
        return  # Includes completed cleanup: never signal a potentially reused ID.
    owner, group = ownership
    if owner != os.getpid() or group <= 1 or group == os.getpgrp() or group != proc.pid:
        raise ValueError("refusing cleanup outside this process's owned group")
    try:
        if not _signal_owned_group(group, signal.SIGTERM):
            return
        # Waiting only for the leader misses orphans. Allow the whole group a
        # bounded grace period, then sweep even when the leader exited first.
        deadline = time.monotonic() + STOP_GRACE_SECONDS
        while time.monotonic() < deadline:
            proc.poll()  # Reap our direct child, not someone else's descendants.
            try:
                if not _signal_owned_group(group, 0):
                    return
            except PermissionError:
                # S9e (d0b2b32c capture, pass A): a group torn down by its own
                # exit can report EPERM while a member is still leaving, and
                # raising here lost the trial's result.json. Within the grace
                # period keep polling; the SIGKILL sweep below re-raises if a
                # member is still live and unsignallable after it.
                pass
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        _signal_owned_group(group, signal.SIGKILL)
        proc.wait(timeout=STOP_GRACE_SECONDS)
    finally:
        _OWNED_GROUPS.pop(proc, None)
        proc.poll()


@contextmanager
def lane_services(args):
    """Own only the lane's Codex bridge; external meter and translator are never killed.

    No lane budget process: the meter is the single budget authority (module
    docstring, R6 F19).
    """
    block_paid_coding(args)
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", BRIDGE_PORT))
    args.output.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "BRIDGE_UPSTREAM": args.translator_url,
        "BRIDGE_LOG": str(args.output / "bridge.jsonl"),
    }
    processes, streams = [], []
    try:
        for name, extra in (("codex_bridge", [str(BRIDGE_PORT)]),):
            stream = (args.output / (name + ".log")).open("a")
            streams.append(stream)
            processes.append(
                owned_process(
                    [sys.executable, str(HERE / (name + ".py")), *extra],
                    cwd=HERE,
                    env=env,
                    stdout=stream,
                    stderr=stream,
                    start_new_session=True,
                )
            )
        for _ in range(40):
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("lane service failed setup; inspect local service logs")
            # The bridge has no health route; a listening socket is readiness.
            try:
                with socket.create_connection(("127.0.0.1", BRIDGE_PORT), timeout=1):
                    break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError("bridge startup failed")
        yield
    finally:
        for proc in processes:
            stop(proc)
        for stream in streams:
            stream.close()


def wire_rows(log, run_id):
    rows = []
    for line in log.read_text().splitlines():
        row = json.loads(line)
        if row.get("run_id") == run_id:
            cost = (row.get("usage") or {}).get("cost")
            if cost is None or not math.isfinite(float(cost)) or float(cost) < 0:
                raise ValueError("missing/invalid exact-run billing evidence")
            rows.append(row)
    return rows


def run_trial(args, harness, task, provenance=None):
    block_paid_coding(args)
    if provenance is None:
        provenance = public_provenance(args.task_root / task["name"], task["image_digest"])
    run_id = f"coding__{harness}__{task['task8']}__r{args.rep}"
    job = args.output / "jobs" / run_id
    if job.exists() or (args.request_log.exists() and wire_rows(args.request_log, run_id)):
        raise ValueError("refusing reused run ID")
    cls, version = HARNESS[harness]
    from urllib.parse import urlsplit

    # Inside the container every host service is reached ONLY via the gate,
    # on the same port number it listens on at the host (egress.py).
    meter_port = urlsplit(args.meter_url).port
    edge = start_edge(args, run_id) if harness == "local-operator" else None
    try:
        plan = container_plan(
            run_id, harness, args.image, meter_port=meter_port, bridge_port=BRIDGE_PORT,
            edge=edge and {"port": edge["port"], "env": edge["env"], "ca_file": edge["ca_file"]},
        )
        # Held open for the whole Harbor run: the pre-inference probe dials it.
        with egress_canary() as canary:
            if canary in plan["ports"]:
                raise ValueError("egress canary collides with a gate port")
            with codex_current_run(args, harness, run_id):
                return _run_gated_trial(
                    args, harness, task, run_id, job, cls, version, plan, edge, canary, provenance
                )
    finally:
        stop_edge(edge)


def start_edge(args, run_id):
    """Host-side lop_edge for one LoP trial; the bearer goes in on stdin only.

    The real Radient bearer comes from the driver's own environment and is
    never forwarded: run_trial's Harbor env is an allowlist without it.
    """
    token = os.environ.get("BENCH_RADIENT_BEARER", "")
    if not token:
        raise ValueError("BENCH_RADIENT_BEARER required on the host for the LoP edge")
    run_dir = args.output / "edge" / run_id
    proc = owned_process(
        [sys.executable, str(HERE.parent / "lop_edge.py")],
        cwd=HERE.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, start_new_session=True,
    )
    # The edge serves LoP's package route itself (lop_edge docstring): LoP's
    # proxy variables must keep naming the edge, so its tool children reach
    # registries through it under the same policy as the gate's proxy.
    config = {"run_id": run_id, "run_dir": str(run_dir), "meter_base": args.meter_url.rstrip("/"),
              "package_allowlist": str(PACKAGE_ALLOWLIST), "public_roots": str(PUBLIC_ROOTS)}
    if os.environ.get("BENCH_DECISIONS_URL"):  # zero-paid rig fixture only
        config["decisions_url"] = os.environ["BENCH_DECISIONS_URL"]
    proc.stdin.write(json.dumps({**config, "radient_token": token}) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        stop(proc)
        raise ValueError("lop_edge exited during startup")
    announced = json.loads(line)
    # The CA FILE only (R6 N6): run_dir also holds edge-requests.jsonl.
    return {"proc": proc, "port": announced["port"], "env": announced["env"],
            "ca_file": run_dir / "edge-ca.pem"}


def stop_edge(edge):
    if not edge:
        return
    proc = edge["proc"]
    try:
        proc.stdin.close()  # edge settles or latches in-flight work, then exits
        proc.wait(timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        pass
    stop(proc)


def codex_config(base_url):
    """codex-best's base ``config.toml`` (Harbor ``--ak config=``), as a dict.

    ``web_search = "disabled"`` (S9e): Codex 0.147 otherwise sends its
    Responses ``web_search`` tool, which LiteLLM 1.106.6 translates into
    Anthropic's server tool ``web_search_20250305``. The meter refuses every
    unpriced server tool (402, d0b2b32c DEV capture), and no other arm sends
    one on this path, so equal treatment is no server-side web search for any
    arm. The key is Codex's own top-level setting (its binary documents
    ``"live"``/``"indexed"``/``"cached"``/``"disabled"``; Harbor's own
    ``web_search`` option writes the same key). The bridge refuses the tool
    as a backstop; the meter's refusal is unchanged.
    """
    return {
        "model_provider": "bench",
        "model_catalog_json": "/tmp/bench-support/codex-model-catalog.json",
        "web_search": "disabled",
        "model_providers": {
            "bench": {
                "name": "bench",
                "base_url": base_url,
                "env_key": "OPENAI_API_KEY",
                "wire_api": "responses",
            }
        },
    }


def _run_gated_trial(args, harness, task, run_id, job, cls, version, plan, edge, canary,
                     provenance):
    import egress

    support = args.output / "support" / run_id
    support.mkdir(parents=True)
    for name in egress.SUPPORT_FILES:
        shutil.copy(HERE / name, support / Path(name).name)
    egress_log = args.output / "egress" / run_id
    egress_log.mkdir(parents=True)
    override = egress.write_override(
        args.output / "support" / (run_id + ".compose.json"),
        egress.compose_override(run_id, plan["ports"], plan["image"], support, args.toolchain,
                                egress_log),
    )
    url = plan["urls"]["anthropic"]
    mounts = [
        {
            "type": "bind",
            "source": str(args.toolchain.resolve()),
            "target": "/opt/bench",
            "read_only": True,
        },
        *egress.support_mounts(support, edge and edge["ca_file"]),
    ]
    argv = [
        args.harbor,
        "run",
        "-p",
        str(args.task_root / task["name"]),
        "-a",
        "adapters:" + cls,
        "-m",
        "anthropic/" + args.model,
        "--ak",
        "version=" + version,
        "-o",
        str(args.output / "jobs"),
        "--job-name",
        run_id,
        "-n",
        "1",
        "-k",
        "1",
        "-r",
        "0",
        "-q",
        "-y",
        "--skip-credential-checks",
        "--mounts",
        json.dumps(mounts),
        "--extra-docker-compose",
        str(override),
        "--verifier-import-path",
        "bench_verifier:BenchVerifier",
        "--ae",
        "PATH=/opt/bench/node/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin",
        "--ae",
        "BENCH_ANTHROPIC_URL=" + url,
    ]
    if harness in ("claude-code", "opencode", "hermes"):
        argv += [
            "--base-url",
            url + ("/v1" if harness == "opencode" else ""),
            "--api-key-name",
            "ANTHROPIC_API_KEY",
            "--ae",
            "ANTHROPIC_API_KEY=benchmark-dummy",
        ]
    for key, value in plan["env"].items():  # LoP edge env (dummy key, gate proxy)
        argv += ["--ae", f"{key}={value}"]
    if harness == "claude-code":
        argv += ["--ae", "ENABLE_TOOL_SEARCH=true", "--effort", "high"]
    if harness == "codex-best":
        config = codex_config(plan["urls"]["codex"])
        argv += [
            "--base-url",
            plan["urls"]["codex"],
            "--api-key-name",
            "OPENAI_API_KEY",
            "--api-format",
            "openai_responses",
            "--ae",
            "OPENAI_API_KEY=benchmark-dummy",
            "--ak",
            "config=" + json.dumps(config),
            "--effort",
            "high",
        ]
    env = {
        k: v
        for k, v in os.environ.items()
        if k in {"PATH", "HOME", "LANG", "TMPDIR", "DOCKER_HOST", "DOCKER_CONTEXT"}
    }
    # Harbor-process-only inputs for the pre-inference and verifier checks:
    # hashes and port numbers, never task bytes or credentials.
    package_logs = [egress_log / "connections.jsonl"] + (
        [args.output / "edge" / run_id / "package-egress.jsonl"] if edge else [])
    # R11 F31: every arm launches the gate's package proxy (LoP also its edge
    # policy), so every listed log must exist at grading; the host creates
    # them empty now and BenchVerifier treats a missing one as a fence
    # anomaly. BENCH_PACKAGE_PROXY_LAUNCHED=none (no arm today) is the only
    # case in which no log is expected.
    for log in package_logs:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.touch()
    provenance_json = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    provenance_sha = hashlib.sha256(provenance_json.encode()).hexdigest()
    (args.output / (run_id + ".public-provenance.json")).write_text(provenance_json + "\n")
    env.update(
        BENCH_PUBLIC_PROVENANCE=provenance_json,
        BENCH_PUBLIC_PROVENANCE_SHA256=provenance_sha,
        BENCH_TASK_PAYLOAD_SHA256=provenance["task_payload_sha256"],
        BENCH_IMAGE_DIGEST=task["image_digest"],
        PYTHONPATH=str(HERE),
        CODING_MODEL=args.model,
        BENCH_GOLD_DIGESTS=",".join(gold_digests(args.task_root / task["name"])),
        BENCH_GATE_PORTS=",".join(str(p) for p in plan["ports"]),
        BENCH_EGRESS_CANARY_PORT=str(canary),
        # Host-side fence logs BenchVerifier reads before grading a late hidden
        # match (S9b); the same files package_egress_summary records below.
        BENCH_PACKAGE_EGRESS_LOGS=",".join(str(p) for p in package_logs),
    )
    started, disk_stop = time.time(), False
    with (args.output / (run_id + ".log")).open("x") as stream:
        proc = owned_process(
            argv, cwd=HERE, env=env, stdout=stream, stderr=stream, start_new_session=True
        )
        try:
            while proc.poll() is None:
                if free_mb(args.disk_volume) < 17 * 1024:
                    disk_stop = True
                    stop(proc)  # soft stop protects the 15 GiB emergency floor
                    break
                time.sleep(2)
            code = proc.wait()
        finally:
            stop(proc)
    rows = wire_rows(args.request_log, run_id)
    results = list(job.glob("*/result.json"))
    result = json.loads(results[0].read_text()) if len(results) == 1 else {}
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    record = {
        "run_id": run_id,
        "harness": harness,
        "version": version,
        "score": reward,
        "exit_code": code,
        "disk_stop": disk_stop,
        "wall_s": time.time() - started,
        "billing_tier": args.billing_tier,
        "cost_usd": sum(float(r["usage"]["cost"]) for r in rows),
        "requests": len(rows),
        "verified": bool(rows) and reward is not None and not disk_stop and code == 0,
        "public_provenance_sha256": provenance_sha,
        "public_provenance_policy_sha256": provenance["policy_sha256"],
        "package_egress": package_egress_summary(package_logs),
        "task": task["name"],
        "post_agent_flags": post_agent_flags(job),
    }
    with (args.output / "runs.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    flag_table(args.output)
    print(json.dumps(record))
    if code or disk_stop or not record["verified"]:
        raise RuntimeError(
            "trial incomplete; stopping rather than counting infrastructure as quality"
        )


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in (
        "toolchain",
        "task-root",
        "output",
        "request-log",
        "preregistration",
        "runtime-attestation",
        "toolwire-attestation",
        "oracle-attestation",
    ):
        ap.add_argument("--" + name, type=Path, required=True)
    for name in (
        "preregistration-sha256",
        "runtime-sha256",
        "toolwire-sha256",
        "oracle-sha256",
        "billing-tier",
        "model",
        "tasks",
        "harnesses",
        "meter-url",
        "translator-url",
    ):
        ap.add_argument("--" + name, required=True)
    for name in ("total-budget-usd", "trial-budget-usd", "input-usd-per-m", "output-usd-per-m"):
        ap.add_argument("--" + name, type=float, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument(
        "--meter-current-run-file", type=Path, required=True,
        help="absolute path of the meter's BENCH_CURRENT_RUN_FILE (Codex run tagging)",
    )
    ap.add_argument("--subset", type=Path, default=HERE / "subset.json")
    ap.add_argument("--disk-volume", type=Path, default=Path.cwd())
    ap.add_argument("--harbor", default="harbor")
    ap.add_argument("--coding-edge-evidence", type=Path)
    ap.add_argument("--execute", action="store_true")
    return ap


def main():
    import egress

    ap = parser()
    args = ap.parse_args()
    try:
        tasks, harnesses = preflight(args)
        if not args.execute:
            print("Preflight passed; dispatch requires --execute")
            return
        with lane_services(args):
            for task in tasks:
                check_space(args.disk_volume, task["hub_tag_full_size_bytes"])
                image = task["image"].split(":")[0] + "@" + task["image_digest"]
                # Bind the original task before the existing temporary image-pin
                # rewrite; hashing that rewrite would make valid provenance stale.
                provenance = public_provenance(args.task_root / task["name"], task["image_digest"])
                config = args.task_root / task["name"] / "task.toml"
                original = config.read_text()
                source_image = tomllib.loads(original)["environment"]["docker_image"]
                # Task payloads are external scratch copies. Restore even when
                # a trial fails; never edit the upstream dataset checkout.
                replacement = original.replace(
                    'docker_image = "' + source_image + '"', 'docker_image = "' + image + '"'
                )
                if tomllib.loads(replacement)["environment"]["docker_image"] != image:
                    raise ValueError("task image cannot be pinned safely")
                args.image = image  # the gate reuses it: no second image pull
                # The ONLY pull on this path: explicit, digest-pinned, behind a
                # fresh disk gate. Compose services carry pull_policy: never.
                check_space(args.disk_volume, task["hub_tag_full_size_bytes"])
                egress.run_docker(["docker", "pull", "--platform", "linux/amd64", image],
                                  explicit_pull=True, check=True)
                try:
                    config.write_text(replacement)
                    for harness in harnesses:
                        check_space(args.disk_volume, 0)
                        run_trial(args, harness, task, provenance)
                finally:
                    config.write_text(original)
                    # Only this lane's exact image, never global prune.
                    egress.run_docker(["docker", "rmi", image], check=True, capture_output=True)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
