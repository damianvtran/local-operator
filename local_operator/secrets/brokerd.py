"""``python -m local_operator.secrets.brokerd`` — the broker's process entry.

Its own module, rather than a ``__main__`` guard inside
:mod:`local_operator.secrets.broker`, so the daemon can be spawned by module
name without importing the CLI. It is started automatically by
:func:`local_operator.secrets.client.ensure_broker`; running it by hand is
useful for watching it in the foreground while debugging.
"""

from __future__ import annotations

import argparse
import signal
import sys
from types import FrameType

from local_operator import procname
from local_operator.secrets.broker import IDLE_SHUTDOWN_S, run_broker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="local-operator-secretd")
    parser.add_argument(
        "--idle-shutdown",
        type=float,
        default=IDLE_SHUTDOWN_S,
        help="Exit after this many seconds with no sessions and no requests (0 disables)",
    )
    args = parser.parse_args(argv)

    def _terminate(signum: int, frame: FrameType | None) -> None:
        """Turn SIGTERM into the same clean path as Ctrl-C.

        Without this the default disposition kills the process outright, the
        socket file survives as a corpse, and the next broker has to reap it.
        Reaping is implemented anyway (a SIGKILL cannot be handled), but the
        ordinary shutdown should not depend on it.
        """
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)

    # Linux only, and it does NOT replace the argv label the parent set in
    # `client._spawn_broker` — the two are independent name axes (see
    # `procname`). `prctl(PR_SET_NAME)` does not survive an exec, so a child
    # that wants a `comm` must set its own; without this the broker's `comm` is
    # whatever image it was exec'd through, which for a branded launch is the
    # bare product name. `comm` truncates at 15 bytes so only BRAND fits there,
    # which is exactly why the argv label carries the detail and this call
    # carries only the family name.
    procname.brand_this_process()

    return run_broker(idle_shutdown_s=args.idle_shutdown)


if __name__ == "__main__":
    sys.exit(main())
