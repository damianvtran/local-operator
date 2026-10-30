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
* machine-readable invocations on the LOCAL dispatch (``--json``, and
  ``exec``'s ``--json`` / ``json_mode``) — a document a script parses must not
  vary with the operator's language. The invariant is scoped to the local
  dispatch on purpose: ``exec --remote --json`` is still a remote run whose
  refusals are prose printed by ``remote_command``, and that path activates
  before the CLI's dispatch can say anything about it;
* the full-screen TUI — ``OperatorApp.on_mount`` resolves and applies the
  language itself (and owns the repaint that follows a move), so this is not
  its boot path.
"""

from __future__ import annotations

from collections.abc import Callable

from .resolve import CONFIG_KEY, DEFAULT_LANGUAGE, resolve_language
from .text import set_active_locale


def configured_language() -> str:
    """The ``language`` value on disk, or ``DEFAULT_LANGUAGE`` when unset.

    Deliberately NOT a bare ``resolve_language()``: that falls back to building
    a ``ConfigManager`` when the process has no config watcher (the CLI case),
    and construction CREATES the config directory — measured: with no
    ``~/.local-operator`` on disk, ``read_config_values`` leaves it absent and
    ``ConfigManager(directory)`` creates it. Boot-time copy must create nothing:
    a run that is about to be refused (the agent-shell guard asserts a refused
    ``lop`` leaves the home untouched) and an argparse ``--help`` must not
    touch a store they are not going to use, and a resolution read has no
    business renaming an unparsable file either.

    ``read_config_values`` is that reader — it never writes, moves or reports
    anything, and it reads through the same parse memo as the manager. Returns
    ``DEFAULT_LANGUAGE`` (``auto``) rather than ``None`` for an unreadable file
    for the reason ``i18n.cli_help`` documents: a ``None`` would send the
    resolver back to the manager this function exists to avoid.
    """
    from local_operator.config import read_config_values
    from local_operator.paths import config_dir

    try:
        values = read_config_values(config_dir())
    except UnicodeDecodeError:
        # Outside read_config_values' own contract, and equally unavailable here.
        values = None
    value = values.get(CONFIG_KEY) if isinstance(values, dict) else None
    return value if isinstance(value, str) else DEFAULT_LANGUAGE


def _resolve_for_boot() -> str:
    """The chain's answer over the read-only config value (see above)."""
    return resolve_language(configured_language())


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
    resolver = resolve if resolve is not None else _resolve_for_boot
    locale = "en"
    try:
        locale = resolver()
    except Exception:  # noqa: BLE001 — a locale is copy; it never blocks the command
        locale = "en"
    try:
        set_active_locale(locale)
    except Exception:  # noqa: BLE001 — same reason: the command continues on English
        pass
