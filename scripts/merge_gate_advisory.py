#!/usr/bin/env python3
"""Bound one advisory gh lookup, without turning availability into gate authority.

The caller shares a monotonic deadline across status lookups. Only recognised
transient failures retry; auth and unusable responses return unchanged for the
Makefile's fail-closed/manual-override handling. The 17-second operator-wait
budget is policy (3 requests × 5 seconds + 2 waits × 1 second), not a claim about
GitHub latency. This helper neither matches runs nor writes labels.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time

ATTEMPTS = 3
REQUEST_SECONDS = 5
WAIT_SECONDS = 1
BUDGET_SECONDS = ATTEMPTS * REQUEST_SECONDS + (ATTEMPTS - 1) * WAIT_SECONDS


def main() -> int:
    deadline = float(sys.argv[1])
    command = sys.argv[2:]
    output = "request timed out: total advisory budget exhausted"
    for attempt in range(1, ATTEMPTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=min(REQUEST_SECONDS, remaining),
                check=False,
            )
            output = result.stdout + result.stderr
            if result.returncode == 0:
                print(result.stdout, end="")
                return 0
        except subprocess.TimeoutExpired:
            output = "request timed out: advisory request deadline exceeded"
        # Authentication must never be retried or silently treated as an outage.
        auth = re.search(
            r"HTTP 401|bad credentials|unauthori[sz]ed|authentication failed", output, re.I
        ) or (re.search(r"HTTP 403", output, re.I) and not re.search(r"rate.?limit", output, re.I))
        transient = not output.strip() or re.search(
            r"rate.?limit|HTTP 429|HTTP 5[0-9][0-9]|timed? out|timeout|network|connection|"
            r"no response|API unavailable|error connecting|no such host|TLS handshake|EOF",
            output,
            re.I,
        )
        if auth or not transient:
            print(output, end="")
            return 1
        if attempt == ATTEMPTS:
            break
        print(f"advisory lookup transient failure: retry {attempt}/{ATTEMPTS - 1}", file=sys.stderr)
        time.sleep(min(WAIT_SECONDS, max(0, deadline - time.monotonic())))
    print(output, end="")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
