"""``/mcp add`` and ``/mcp remove``, shared by the terminal and the runtime.

These two verbs write the GLOBAL ``~/.local-operator/mcp.json`` and reconnect
the session's MCP manager, so they are machine-and-session work rather than
terminal work — which means a detached runtime must be able to run them too.
Before round 5 only `OperatorApp` had them: the routed handler recognised the
three grant verbs and let ``add``/``remove`` fall through to a bare server
LISTING, so a user typing ``/mcp add notion …`` got a plausible-looking table,
believed the server was added, and found days later that nothing was written
(round 5, U15).

Extracted verbatim rather than reimplemented. The refusal rules here are the
substance of the commands — ``load_all_mcp_configs`` merges eight sources and
local-operator writes two of them, so most of this code is about declining to
touch a config that belongs to another tool — and a second copy beside them
would be a second source of truth for which files we may write.

The reconnect is injected as a callback because that is the only part that
genuinely differs: the app posts a Textual worker, the runtime awaits on its
own loop. Everything above it is identical.
"""

from __future__ import annotations

import os
from typing import Callable, Literal

NoticeKind = Literal["info", "success", "warning", "error"]


def _outranks_global_mcp_scope(source: str | os.PathLike[str], cwd: str) -> bool:
    """Whether ``source`` beats the GLOBAL mcp.json in the merge order.

    ``/mcp add`` always writes the global file, and ``load_all_mcp_configs``
    resolves a name first-source-wins in a fixed priority order. Anything ahead
    of the global file therefore keeps defining the server after our write, so
    the add is a no-op the user cannot see. This answers "would the write be
    observable", which is a different question from "do we own the file" — a
    project ``.local-operator/mcp.json`` is ours to write AND outranks us.

    Resolved paths on both sides, for the reason ``owned_scope_for_source``
    resolves: a symlinked home or a ``/private/var`` prefix must not decide it.
    """
    from pathlib import Path

    root = Path(cwd).expanduser()
    ahead_of_global = (
        root / ".local-operator" / "mcp.json",
        root / ".mcp.json",
    )
    try:
        resolved = Path(source).expanduser().resolve()
        return any(candidate.expanduser().resolve() == resolved for candidate in ahead_of_global)
    except OSError:
        return False


#: Which TOOL owns each foreign MCP config file, keyed by the trailing path
#: fragment ``load_all_mcp_configs`` reads it from. A refusal that names only
#: the file leaves the user hunting for what writes it; naming the tool makes
#: "remove it there" an instruction rather than a dead end. The Codex entry is
#: read-only for a reason that will not change soon: ``tomllib`` parses TOML
#: (``load``/``loads``) and cannot emit it, and ``tomli_w`` is not a
#: dependency, so refusing is the only correct answer for a Codex-imported
#: server rather than a policy we could relax (issue #367).
_FOREIGN_MCP_CONFIGS: tuple[tuple[tuple[str, ...], str], ...] = (
    ((".claude.json",), "imported from Claude Code"),
    ((".claude", ".mcp.json"), "imported from Claude Code"),
    ((".cursor", "mcp.json"), "imported from Cursor"),
    ((".vscode", "mcp.json"), "imported from VS Code"),
    ((".codex", "config.toml"), "imported from Codex CLI"),
    # Read by the loader, never written by ``_scope_path`` — foreign to the
    # writer despite living in the project the user is sitting in.
    ((".mcp.json",), "a project .mcp.json local-operator does not write"),
)


def _foreign_config_origin(source: str | None) -> str:
    """Name the tool that owns ``source``, for the ``/mcp remove`` refusal."""
    # Function-local import: this module keeps `pathlib` off its import path
    # (every other use here is local too), and this runs once per refusal.
    from pathlib import Path

    if source:
        parts = Path(source).parts
        for fragment, origin in _FOREIGN_MCP_CONFIGS:
            if len(parts) >= len(fragment) and tuple(parts[-len(fragment) :]) == fragment:
                return origin
    return "not written by local-operator"


def _home_abbreviated(text: str) -> str:
    """Abbreviate the home prefix wherever it appears INSIDE a message.

    :func:`_home_relative` abbreviates a string that IS a path; this one is for
    prose that merely contains one — the config writers embed the file they
    refused to write in their error text ("server 'x' already exists in
    /Users/…/mcp.json"), and a receipt that abbreviates its success path while
    spelling the whole thing out on failure reads as two different commands.
    """
    home = os.path.expanduser("~")
    if home in ("", "/"):
        return text
    return text.replace(home + os.sep, "~" + os.sep)


def _home_relative(path: str) -> str:
    """``~/.local-operator/config.yml`` rather than the full ``/Users/…`` form.

    The prefix is the same on every machine and costs a third of the line the
    confirmation has to spend saying WHERE it wrote. A path outside the home
    tree — the ``LOCAL_OPERATOR_CONFIG_DIR`` override, a test's tmp dir — is
    left absolute, because there is no shorter honest rendering of it.
    """
    home = os.path.expanduser("~")
    if home in ("", "/"):
        return path
    if path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    # A path that came from `Path.resolve()` can disagree with `$HOME` purely
    # by symlink — macOS hands out `/private/var/…` for a `/var/…` home — so a
    # prefix test on the raw strings leaves an under-home path rendered in
    # full. Retry against the resolved home before giving up; still absolute
    # for anything genuinely outside the home tree.
    try:
        resolved_home = os.path.realpath(home)
    except OSError:
        return path
    if resolved_home not in ("", "/") and path.startswith(resolved_home + os.sep):
        return "~" + path[len(resolved_home) :]
    return path


def mcp_add_result(tokens: list[str], reconnect: Callable[[], None]) -> tuple[str, NoticeKind]:
    """Do one ``/mcp add`` and return its receipt as ``(text, kind)``.

    Grammar, the smallest unambiguous thing that covers both transports::

        /mcp add <name> <url>              -> http server
        /mcp add <name> <command> [args…]  -> stdio server

    The discriminator is whether the third token parses as an http(s) URL.
    There is no scope token: the write always lands in the GLOBAL
    ``~/.local-operator/mcp.json``, matching ``lop mcp add``'s default, and
    the receipt NAMES the file so an invisible default becomes a visible
    fact the user can go and check.

    OAuth is deliberately NOT inferred from a URL. Real configs carry
    non-OAuth http servers (an internal gateway, a header-authenticated
    endpoint), so inferring ``auth: oauth`` from the scheme would silently
    change how a server authenticates and produce a browser prompt for a
    server that never needed one. Added without auth; ``/mcp login <name>``
    is the documented next step for a server that does use OAuth.

    Env vars are out of scope on purpose — a ``KEY=VALUE`` token in this
    grammar cannot be told apart from a command argument, and the CLI's
    explicit ``--env`` flag already covers it.
    """
    from local_operator.mcp.config import (
        MCPConfigWriteError,
        add_server,
        load_all_mcp_configs,
        owned_scope_for_source,
    )

    if len(tokens) < 2:
        return (
            "usage: /mcp add <name> <url>  |  /mcp add <name> <command> [args…]",
            "warning",
        )
    name, target, *rest = tokens
    is_url = target.startswith(("http://", "https://"))
    cwd = os.getcwd()
    # ``remove`` refuses to touch a server it does not own; ``add`` is the
    # mirror operation and has to answer the same question, or the two
    # verbs disagree about one invariant. What the answer IS depends on
    # priority, so the two cases are reported differently rather than
    # collapsed into one refusal:
    #
    #   * The existing definition OUTRANKS the global file we write (a
    #     project ``.local-operator/mcp.json`` or ``.mcp.json``). The write
    #     lands and changes nothing the user can observe — they keep
    #     getting the old server. Refused, because a success receipt for a
    #     write with no effect is a receipt that lies, and that is worse
    #     than the shadowing case below.
    #   * The existing definition ranks BELOW it (an imported foreign
    #     config). Our entry would win and silently repoint a server the
    #     user still maintains in Claude Code or Cursor — exactly what
    #     ``_mcp_remove_result`` refuses to cause from the other side.
    #     Refused too, naming the file and the tool, so the user can
    #     change it where it lives or pick another name.
    try:
        existing = load_all_mcp_configs(cwd)[1].get(name)
    except Exception:  # noqa: BLE001 — an unreadable config is reported by the write
        existing = None
    if existing is not None:
        # Priority FIRST: `<cwd>/.mcp.json` is both unowned and
        # higher-priority, and "your write would not take effect" is the
        # more useful thing to say about it than "you would shadow it" —
        # which would also be false.
        if _outranks_global_mcp_scope(existing, cwd):
            return (
                f"{name!r} is already defined in {_home_relative(str(existing))}, "
                f"which takes priority over the global config /mcp add writes.\n"
                f"Adding it here would have no effect. Edit that file, or remove "
                f"the entry there first.",
                "warning",
            )
        if owned_scope_for_source(existing, cwd) is None:
            return (
                f"{name!r} is already defined in {_home_relative(str(existing))} "
                f"({_foreign_config_origin(existing)}).\nAdding it here would "
                f"shadow that entry rather than update it. Change it there, or "
                f"pick another name.",
                "warning",
            )
    try:
        if is_url:
            if rest:
                # An http server takes exactly one target; trailing tokens
                # are a stdio-shaped mistake and dropping them silently
                # would configure something the user did not describe.
                return (
                    f"/mcp add {name} <url> takes no extra arguments — " f"got {' '.join(rest)!r}",
                    "warning",
                )
            path = add_server(name, url=target, cwd=os.getcwd())
        else:
            path = add_server(name, command=target, args=list(rest) or None, cwd=os.getcwd())
    except MCPConfigWriteError as exc:
        return (f"could not add MCP server {name!r}: {_home_abbreviated(str(exc))}", "warning")
    except Exception as exc:  # noqa: BLE001 — a failed write is a notice, not a crash
        return (f"could not add MCP server {name!r}: {exc}", "error")
    reconnect()
    # An http server added without auth is the common half-done case, so
    # the receipt still points at the next step — but it must point at one
    # that WORKS. `/mcp login` was wrong: this command deliberately writes
    # no `auth` block, and `_resolve_mcp_server` refuses any server whose
    # `auth.type` is not `oauth`, so the suggestion failed for every server
    # this command can create. The CLI's `--oauth` flag is the only path
    # that writes the block, so name that instead. Inferring OAuth from the
    # URL remains ruled out: real configs carry non-OAuth http servers, and
    # guessing would silently change how a server authenticates.
    hint = (
        f" — needs OAuth? re-add it with: lop mcp add {name} --url {target} --oauth"
        if is_url
        else ""
    )
    return (f"added MCP server {name!r} to {_home_relative(str(path))}{hint}", "success")


def mcp_remove_result(name: str, reconnect: Callable[[], None]) -> tuple[str, NoticeKind]:
    """Do one ``/mcp remove`` and return its receipt as ``(text, kind)``.

    The refusal is the point of this command. ``load_all_mcp_configs``
    merges EIGHT sources but local-operator only writes two of them, so a
    server can be perfectly visible in ``/mcp`` and still be none of our
    business to delete. Removing an entry defined by ``~/.claude.json``
    would either fail or, worse, write a local-operator file that shadows a
    config the user still maintains in Claude Code — a silent divergence
    between two tools that both claim to own the server.

    ``<cwd>/.mcp.json`` is refused for a subtler reason: it is READ by
    ``load_all_mcp_configs`` but never written by ``_scope_path``, so it is
    foreign to the writer even though it is a local-operator-ish path.

    A Codex TOML source (``~/.codex/config.toml``, see issue #367) is the
    permanent case: ``tomllib`` is read-only (``load``/``loads`` only, and
    ``tomli_w`` is not a dependency), so an imported Codex server can never
    be removed in place. Refusal there is the ONLY correct behaviour rather
    than a policy choice, and stays correct until a TOML writer is added.
    """
    from local_operator.mcp.config import (
        MCPConfigWriteError,
        load_all_mcp_configs,
        owned_scope_for_source,
        remove_server,
    )

    cwd = os.getcwd()
    try:
        configs, sources = load_all_mcp_configs(cwd)
    except Exception as exc:  # noqa: BLE001 — an unreadable config is a notice
        return (f"could not read the MCP configuration: {exc}", "error")
    if name not in configs:
        # Same shape as ``_resolve_mcp_server``'s unknown-name refusal,
        # minus its OAuth check: removal is not an OAuth operation, so a
        # stdio server must reach the real answer rather than "does not use
        # OAuth login".
        return (f"MCP server {name!r} is not configured — see /mcp", "warning")
    source = sources.get(name)
    scope = owned_scope_for_source(source, cwd)
    if scope is None:
        return (
            f"{name!r} is defined in {_home_relative(str(source))} "
            f"({_foreign_config_origin(source)}), not by local-operator.\n"
            "Remove it there.",
            "warning",
        )
    try:
        path = remove_server(name, scope=scope, cwd=cwd)
    except MCPConfigWriteError as exc:
        return (
            f"could not remove MCP server {name!r}: {_home_abbreviated(str(exc))}",
            "warning",
        )
    except Exception as exc:  # noqa: BLE001 — a failed write is a notice, not a crash
        return (f"could not remove MCP server {name!r}: {exc}", "error")
    reconnect()
    return (f"removed MCP server {name!r} from {_home_relative(str(path))}", "success")
