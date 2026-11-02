"""Opt-in native QA, never ordinary pytest collection or an outer supervisor.

Runs the ACTUAL wrapper on this thread; its real capability grants every signal.
The observation hook only releases a command barrier after real child enrolment.
No numeric-PG fallback or cleanup helper; uncertainty stops dispatch. Run only
under a separately approved safe bootstrap, not the historic unsafe wrapper.
"""

from __future__ import annotations

import json
import os
import runpy
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from local_operator import owned_process_group as owned  # noqa: E402

CHILD = r"""
import os,signal,sys,time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
p=Path(sys.argv[1]); (p/'child-ready').write_text(str(os.getpid()))
deadline=time.monotonic()+8
while time.monotonic()<deadline: time.sleep(.01)
"""
LEADER = r"""
import os,signal,subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1]); mode=sys.argv[2]
if mode=='timeout': signal.signal(signal.SIGTERM,signal.SIG_IGN)
child=subprocess.Popen([sys.executable,'-S','-c',sys.argv[3],str(p)])
(p/'child-pid').write_text(str(child.pid))
deadline=time.monotonic()+8
while time.monotonic()<deadline:
    if mode=='leader-exit' and (p/'enrolled').exists(): sys.exit(0)
    time.sleep(.01)
sys.exit(2)
"""


def persist(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    wrapper = runpy.run_path(str(ROOT / "scripts/run_bounded.py"))
    # Detached groups are not priced by an outer bash PG. Require the actual
    # allocation and let the production resolver preserve/tighten it, never zero.
    grant_key = wrapper["memory_guard"].RESOLVED_MEMORY_MB_ENV
    raw = os.environ.get(grant_key, "")
    if not raw.isdigit() or int(raw) <= 0:
        raise SystemExit("positive central memory grant required before dispatch")
    budget, budget_source = wrapper["_memory_budget"](None)
    if budget.ceiling_mb <= 0 or budget.ceiling_mb > int(raw):
        raise SystemExit("effective child ceiling must not disable/widen the grant")
    original = owned.OwnedGroup.refresh
    for mode in ("leader-exit", "timeout", "forward"):
        case = root / mode
        case.mkdir(mode=0o700)
        observations, errors = {}, []
        # The CURRENT record location: every spawned run writes <run_id>.spawn.json
        # and <run_id>.exit.json here, not the pre-A2 refusal-only directory.
        sink = Path(os.environ["LOCAL_OPERATOR_SCRATCHPAD"]) / "run-bounded-runs"
        previous_records = set(sink.glob("*.json"))
        released = False

        def observed(self, **kwargs):
            nonlocal released
            result = original(self, **kwargs)
            # Successful enrolment is a protocol barrier, not a guessed sleep.
            if (
                not released
                and result.status == "VERIFIED"
                and (case / "child-ready").exists()
                and (case / "child-pid").exists()
            ):
                try:
                    child = int((case / "child-pid").read_text())
                    if child in self.receipts:
                        observations.update(self.receipts)
                        from dataclasses import asdict

                        persist(
                            case / "enrolled-identities.json",
                            [asdict(r) for r in observations.values()],
                        )
                        (case / "enrolled").write_text("ready")
                        released = True
                        if mode == "forward":
                            os.kill(os.getpid(), signal.SIGTERM)
                except (OSError, ValueError) as exc:
                    errors.append(type(exc).__name__)
            return result  # observer I/O can never replace real authority

        owned.OwnedGroup.refresh = observed
        rc = None
        try:
            rc = wrapper["run"](
                [sys.executable, "-S", "-c", LEADER, str(case), mode, CHILD],
                timeout=3,
                grace=1,
                memory_mb=None,
            )
        except Exception as exc:
            errors.append("wrapper:" + type(exc).__name__)
        finally:
            owned.OwnedGroup.refresh = original
        expected = {"leader-exit": 0, "timeout": 124, "forward": 143}[mode]
        # Read-only exact-birth census; missing samples remain UNKNOWN unless
        # strict group census proves empty. No retry/signalling in this postlude.
        groups = {r.pgid for r in observations.values()}
        census = {}
        for pg in groups:
            try:
                census[str(pg)] = owned.strict_census(pg, time.monotonic() + 1)
            except Exception as exc:
                census[str(pg)] = None
                errors.append("census:" + type(exc).__name__)
        try:
            samples = owned.bounded_samples(observations, time.monotonic() + 1)
        except Exception as exc:
            samples = {}
            errors.append("samples:" + type(exc).__name__)
        facts = {
            str(pid): {
                "sample_present": pid in samples,
                "same_birth": samples[pid].is_birth(r.scheme, r.birth) if pid in samples else None,
                "zombie": samples[pid].zombie if pid in samples else None,
                "exists": owned.resources._pid_exists(pid),
            }
            for pid, r in observations.items()
        }
        outcome = {
            "mode": mode,
            "rc": rc,
            "expected": expected,
            "released": released,
            "effective_memory_mb": budget.ceiling_mb,
            "budget_source": budget_source,
            "observer_errors": errors,
            "groups": census,
            "identity_samples": facts,
            "records": [str(p) for p in set(sink.glob("*.json")) - previous_records],
        }
        persist(case / "outcome.json", outcome)
        print(json.dumps(outcome), flush=True)
        if not outcome["records"]:
            # Loudly, not silently: a canary whose glob finds nothing is proving
            # less than it reports, so this case fails instead of passing quietly.
            print(json.dumps({"mode": mode, "missing_record_under": str(sink)}), flush=True)
            return 1
        if (
            rc != expected
            or not released
            or errors
            or any(census.values())
            or any(f["exists"] is not False for f in facts.values())
        ):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
