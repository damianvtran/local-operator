"""Closed Python composition for the detached pipeline fixture, not production.

The owner must relaunch THIS injected composition as well as the frontend:
relaunching the production script correctly refuses a detached lease and never
exercises the SDK/MCP/worker path the test asserts. Nothing reads argv/env or
provider/path names to decide authority. Production main has no such injection;
its strong loaded-supervisor admission remains the immutable CLI default.
"""

from __future__ import annotations

import argparse
import sys

from local_operator.owned_process import OwnershipGuarantee
from scripts import run_episode


def _admit_fixture_owner(args: argparse.Namespace) -> str | None:
    # Keep the real lease/source/PID/birth verification. Only this explicitly
    # injected test composition requests the weaker generic identity guarantee.
    return run_episode._owned_host_refusal(
        args, required_guarantee=OwnershipGuarantee.IDENTIFIED_PROCESS
    )


def main() -> int:
    return run_episode.main(
        internal_test_support=True,
        _test_owner_entry=(
            sys.executable,
            "-m",
            "tests.unit.evaluation.adapters.osworld.managed_owner_entry",
        ),
        _test_owner_admission=_admit_fixture_owner,
    )


if __name__ == "__main__":
    sys.exit(main())
