"""Refuse to launch Local Operator on the Harbor coding path (fail closed).

WHY: the benchmark measures released, unmodified Local Operator v0.94.22
(``500883ef56ca504325ffd80c5b90bf2435d00e3f``). That release talks only to its
first-party URLs, so it can be metered solely through the per-run TLS edge
(``../lop_edge.py``) and its kernel egress sandbox. Neither exists inside a
Harbor container yet. The former in-process endpoint rebinding (and the
unreleased thinking-replay candidate it served) is removed rather than kept as
a second, unmetered way to run the product.

CONSTRAINT: until the coding path gains an edge with S4/S5 evidence of its own,
this launcher exits non-zero before importing ``local_operator`` -- no path can
dispatch LoP model or classifier traffic unmetered.
"""

import sys

RELEASED_REVISION = "500883ef56ca504325ffd80c5b90bf2435d00e3f"


def main() -> None:
    raise SystemExit(
        "Local Operator " + RELEASED_REVISION[:8] + " is metered only through lop_edge; "
        "the Harbor coding path has no edge, so it refuses to launch"
    )


if __name__ == "__main__":
    main()
    sys.exit(1)
