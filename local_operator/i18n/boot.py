"""Process-level locale activation for the surfaces that boot outside the TUI.

WHY THIS MODULE EXISTS. ``i18n.text`` splits the DECISION from the bookkeeping
on purpose: ``set_active_locale`` owns the move and the re-materialisation of
module-level constants ("this module owns only the bookkeeping around a move"),
and the owning SURFACE decides when a process moves. The TUI keeps that
decision in ``OperatorApp._apply_language``. A headless entry point needs the
identical decision, made of exactly the same three pieces:

* resolve through the shared chain — ``i18n.resolve.resolve_language``, i.e.
  config ``values.language`` → ``LOP_LANG`` → the OS → ``en``, filtered to the
  shipped set — so the CLI and the TUI cannot answer "what language is this?"
  differently;
* hand the answer to the mechanism (:func:`set_active_locale`), which is also
  what runs the registered refreshers, so a module whose copy lives in
  import-time constants re-materialises instead of keeping English;
* never let copy decide whether work happens: a locale lookup that fails is
  worth rendering English over, never worth refusing a command.

Spelling those three beside each entry point is how two surfaces drift apart,
so they live here once and the entry points call this.

ONE ACTIVATION PER PROCESS, and the call sites partition the paths rather than
overlapping them:

* ``remote_compute.cli.remote_command`` — the ``--remote`` dispatch, which
  ``cli.main`` returns into BEFORE its own dispatch step, so it is the only
  activation that path gets;
* ``cli.main`` — every other path that reaches dispatch: the whole
  ``lop <command>`` surface, the headless REPL and the local ``exec`` run.

WHAT IS DELIBERATELY NOT ACTIVATED, and why:

* argparse's ``--help``/``--version`` — they exit inside ``parse_args``,
  before a command exists to activate for, and rendering help must not
  construct a ``ConfigManager`` (that constructor can move a malformed
  ``config.yml`` aside). Help resolves its own copy through
  ``i18n.cli_help.help_locale``, which reads the same chain through the
  side-effect-free ``read_config_values``;
* machine-readable invocations (``--json``, and ``exec``'s ``--json`` /
  ``json_mode``) — a document a script parses must not vary with the
  operator's language;
* the full-screen TUI — ``OperatorApp.on_mount`` resolves and applies the
  language itself (and owns the repaint that follows a move), so this is not
  its boot path.
"""

from __future__ import annotations

from collections.abc import Callable

from .resolve import resolve_language
from .text import set_active_locale


def activate_resolved_locale(resolve: Callable[[], str] | None = None) -> None:
    """Point this process at the language the shared chain resolves.

    Call it ONCE, before the first line of copy the surface renders and after
    config is readable, never per command invocation: a same-locale move is a
    no-op by ``set_active_locale``'s contract, so a second call costs a config
    read and nothing else.

    ``resolve`` is the caller's own resolver binding, for the module whose
    module-level ``resolve_language`` import is the seam its tests patch
    (``remote_compute.cli``). The default reads THIS module's binding at call
    time rather than capturing the function object, so patching
    ``boot.resolve_language`` works too.

    Never raises. A language lookup is copy, not work: a config read that
    fails leaves the process on ``en`` and the command still runs, and a
    refresher that throws inside the move must not take the command down with
    it.
    """
    resolver = resolve if resolve is not None else resolve_language
    locale = "en"
    try:
        locale = resolver()
    except Exception:  # noqa: BLE001 — a locale is copy; it never blocks the command
        locale = "en"
    try:
        set_active_locale(locale)
    except Exception:  # noqa: BLE001 — same reason: the command continues on English
        pass
