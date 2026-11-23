"""The defaults a remote run takes when the caller states none, in one place.

WHY A MODULE AND NOT CONSTANTS IN ``cli.py``. The desktop GUI composes the same
run body the CLI does, and it must compose it with the same defaults — a panel
that offered "m" while the command line sent "l" would make two runs of the same
task cost different amounts. Importing them from ``cli.py`` would drag argparse
into a server process, so they live here and ``cli.py`` imports them back.

``config.yml`` HAS NO SECTION FOR THESE, and that is deliberate rather than
overlooked: ``remote_compute.enabled`` is the only key the section carries
(``local_operator.remote_compute.is_enabled``), and a size or an hours default is
a settings-registry change with its own review — a registry row, a `/settings`
surface row and the consumer-default test beside them. Until that change, the
contract is "send the server's own default explicitly", which is what these two
values are: ``m`` and two hours, both stated in the run-create body.
"""

from __future__ import annotations

import dataclasses

#: The size a run takes when the surface offers no choice; the server's own tier.
DEFAULT_SIZE = "m"

#: The wall-clock bound a run takes when the surface offers no choice, in hours.
DEFAULT_MAX_HOURS = 2.0


@dataclasses.dataclass(frozen=True)
class RunDefaults:
    """What a run body gets for the two fields a caller may leave unset."""

    size: str = DEFAULT_SIZE
    max_hours: float = DEFAULT_MAX_HOURS


def run_defaults() -> RunDefaults:
    """The defaults, as one object a surface can render or pass straight on.

    A function rather than the constants themselves so a surface cannot grow a
    second copy of them, and so a future config-backed default has one place to
    be read from without every caller changing.
    """
    return RunDefaults()


__all__ = ["DEFAULT_MAX_HOURS", "DEFAULT_SIZE", "RunDefaults", "run_defaults"]
