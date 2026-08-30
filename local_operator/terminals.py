"""Which terminal emulator this process is running in, asked once.

Three places already needed this question answered and each had grown its own
copy of the marker tests: ``tui/notify.detect_protocol`` (which notification
protocol does this terminal speak), ``tui/glyphs._nerd_capable_terminal`` (does
it ship a Nerd symbol font), and now the spawn registry (can it open a window
for a fork). The markers are the same facts in all three; only the CONCLUSION
drawn from them differs, which is exactly the shape that belongs in one leaf
module with the conclusions left to the callers.

Deliberately stdlib-only and importing nothing from this package, so the leaf
modules that need it (``glyphs`` is one, and its docstring is explicit about not
wanting an edge into the notification stack) can read it without acquiring each
other's dependencies.

DETECTION IS FROM ENVIRONMENT MARKERS, NEVER FROM A QUERY. A capability query
needs a reply read off stdin, and stdin belongs to Textual's input loop while
the app is running. The markers are injected by the emulator at spawn and
cannot change under a running process.

THE MARKERS ARE INHERITED, WHICH IS A TRAP. Every one of these variables is
passed down to descendants, including ones that crossed into a container or an
ssh hop where that emulator does not exist. A predicate here answers "was this
process tree started by X", which is the right question for a notification
escape (harmless if wrong) and NOT sufficient on its own for spawning a window
(which shells out). Spawn backends therefore gate on a resolvable binary as
well; see ``spawn/registry.py``.
"""

from __future__ import annotations

import os
from typing import Mapping

#: The environment a predicate reads. Passed in rather than read from
#: ``os.environ`` inside each function so detection is testable without
#: mutating process state — a test that has to ``monkeypatch.setenv`` five
#: variables to exercise one branch tends not to exercise the other four.
EnvMap = Mapping[str, str]

#: kitty injects a window id per window; its ``TERM`` is its own terminfo entry.
KITTY_WINDOW_ENV = "KITTY_WINDOW_ID"
KITTY_TERM_PREFIX = "xterm-kitty"

#: ghostty exports its resource root, and its packaged builds also export the
#: binary path. cmux embeds ghostty and sets these too — which is why cmux is
#: detected FIRST everywhere this module's predicates are used in order.
GHOSTTY_RESOURCES_ENV = "GHOSTTY_RESOURCES_DIR"
GHOSTTY_BIN_ENV = "GHOSTTY_BIN"

#: WezTerm exports the pane id in every pane, and the executable path.
WEZTERM_PANE_ENV = "WEZTERM_PANE"
WEZTERM_EXECUTABLE_ENV = "WEZTERM_EXECUTABLE"

#: iTerm2's per-session identifier.
ITERM_SESSION_ENV = "ITERM_SESSION_ID"

#: ``TERM_PROGRAM`` values, lowercased. Apple_Terminal is the ONLY marker
#: Terminal.app sets, which is why it has no dedicated variable above.
TERM_PROGRAM_ENV = "TERM_PROGRAM"
TERM_PROGRAM_GHOSTTY = "ghostty"
TERM_PROGRAM_ITERM = "iterm.app"
TERM_PROGRAM_WEZTERM = "wezterm"
TERM_PROGRAM_WARP = "warpterminal"
TERM_PROGRAM_APPLE = "apple_terminal"

#: cmux injects both per surface. BOTH matter: the workspace names a workspace
#: of many surfaces, the surface names the pane actually holding this session.
CMUX_SURFACE_ENV = "CMUX_SURFACE_ID"
CMUX_WORKSPACE_ENV = "CMUX_WORKSPACE_ID"

#: Set by sshd in a remote session. Used to say "no window server here" in a
#: fork's fallback receipt, which reads as competent rather than broken.
SSH_CONNECTION_ENV = "SSH_CONNECTION"
SSH_TTY_ENV = "SSH_TTY"


def _source(env: EnvMap | None) -> EnvMap:
    """``env`` when given, else the live process environment."""
    return os.environ if env is None else env


def term_program(env: EnvMap | None = None) -> str:
    """``TERM_PROGRAM``, lowercased for comparison against the constants."""
    return _source(env).get(TERM_PROGRAM_ENV, "").lower()


def is_kitty(env: EnvMap | None = None) -> bool:
    """True when this process was started by kitty.

    ``startswith`` on ``TERM`` rather than equality so variants
    (``xterm-kitty-direct``) still match.
    """
    source = _source(env)
    return bool(
        source.get(KITTY_WINDOW_ENV) or source.get("TERM", "").startswith(KITTY_TERM_PREFIX)
    )


def is_ghostty(env: EnvMap | None = None) -> bool:
    """True when this process was started by ghostty — INCLUDING under cmux.

    cmux embeds ghostty, so this is true inside a cmux surface as well. Callers
    that must tell them apart ask :func:`is_cmux` first; that ordering is why
    a ghostty-first spawn registry would open a stray OS window out of a cmux
    surface.
    """
    source = _source(env)
    return bool(
        source.get(GHOSTTY_RESOURCES_ENV)
        or source.get(GHOSTTY_BIN_ENV)
        or term_program(env) == TERM_PROGRAM_GHOSTTY
    )


def is_wezterm(env: EnvMap | None = None) -> bool:
    """True when this process was started by WezTerm."""
    source = _source(env)
    return bool(
        source.get(WEZTERM_PANE_ENV)
        or source.get(WEZTERM_EXECUTABLE_ENV)
        or term_program(env) == TERM_PROGRAM_WEZTERM
    )


def is_iterm(env: EnvMap | None = None) -> bool:
    """True when this process was started by iTerm2."""
    source = _source(env)
    return bool(source.get(ITERM_SESSION_ENV) or term_program(env) == TERM_PROGRAM_ITERM)


def is_apple_terminal(env: EnvMap | None = None) -> bool:
    """True when this process was started by macOS Terminal.app.

    ``TERM_PROGRAM`` is the only marker it sets, so there is nothing else to
    check and a bare ``TERM=xterm-256color`` cannot be distinguished from it by
    any other means.
    """
    return term_program(env) == TERM_PROGRAM_APPLE


def is_cmux(env: EnvMap | None = None) -> bool:
    """True when this process is a cmux surface, by MARKERS ALONE.

    Not sufficient for spawning: the markers are inherited across an ssh hop
    into a host with no cmux CLI. ``multiplexer.cmux`` gates on a resolvable
    binary for that reason and the spawn backend defers to it.
    """
    source = _source(env)
    return bool(source.get(CMUX_SURFACE_ENV) and source.get(CMUX_WORKSPACE_ENV))


def is_ssh(env: EnvMap | None = None) -> bool:
    """True when this process is inside an ssh session."""
    source = _source(env)
    return bool(source.get(SSH_CONNECTION_ENV) or source.get(SSH_TTY_ENV))
