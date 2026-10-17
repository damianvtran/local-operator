"""Help tranche 3: `lop wake` + `lop mcp` — argparse help to keys, two namespaces.

Same ruled classes as help tranches 1-2, applied per site: **help prose is
copy -> keys, flag and command tokens stay IN SITU inside the values;
deprecated/hidden flags may take ignore-with-reason.** These two blocks have
38 `help=` literals (19 + 19), no descriptions, no deprecated/hidden flags —
so **zero ignores**, pinned by test (`# i18n: ignore` must not appear).

Three `wake --json` sites share the `json` key because their text is identical
(the sessions precedent); `mcp`'s `--json` does not exist in this block.
One site is parameterised: `wake create`'s `message` help carried the shared
`MAX_WAKE_MESSAGE_CHARS` cap in an f-string; it renders `{max}` — passed as
`str()` so EN output is byte-identical to the f-string ("2000", not "2,000";
locale digit grouping would apply to an int, and the cap is a value token
that stays in situ per the class ruling).

Rendering is "the pattern": `resolve_help_locale()` once, then
`_w.<key>().envelope(locale=help_locale)["text"]` — never the global renderer;
an EN-only namespace must not un-ship audited locales.

Fidelity: every key renders to the byte its base call carried (table below,
anchored to the base file's AST), and the REAL argparse output (`lop wake
--help` plus its subcommands; `lop mcp --help` plus its subcommands) is
byte-identical base vs branch.

The ceiling assertions DERIVE the expected count from the scanner (equality =
no slack; a `<= 630` bound = may only fall) — the stale-pin chain is closed
structurally; this test re-arms nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.harness.wake_types import MAX_WAKE_MESSAGE_CHARS
from local_operator.i18n import catalogues
from local_operator.i18n.keys import cli_mcp_help as _mh
from local_operator.i18n.keys import cli_wake_help as _wh

REPO = Path(__file__).resolve().parents[3]
CLI = "local_operator/cli.py"
CEILING = 630  # the count after help tranche 3; may only fall


def _live_cli_count(i18n_check) -> int:
    entries = i18n_check.load_allowlist(i18n_check.ALLOWLIST)
    scanned = i18n_check.scan_tree(i18n_check.REPO)
    return i18n_check._counts(scanned, entries).get(CLI, 0)


def test_every_key_renders_the_byte_its_base_call_carried() -> None:
    for rendered, expected in CASES:
        assert rendered == expected, rendered


def test_both_catalogues_are_the_golden_fixtures() -> None:
    for namespace, fixture, size in (
        ("cli.wake_help", "cli_wake_help_en_golden.json", 17),
        ("cli.mcp_help", "cli_mcp_help_en_golden.json", 19),
    ):
        live = catalogues.load_catalogue("en", namespace)
        golden = json.loads(
            (REPO / "tests/unit/i18n/fixtures" / fixture).read_text(encoding="utf-8")
        )
        assert live == golden, f"{namespace}: shipped catalogue and fixture drifted"
        assert len(golden) == size, (namespace, len(golden))
        sidecar = catalogues.context("en", namespace)
        for key, entry in sidecar.items():
            assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_the_wake_block_renders_through_the_help_bootstrap() -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    start = source.index("wake_parser = subparsers.add_parser(")
    after = start + len("wake_parser = subparsers.add_parser(")
    nxt = after + source[after:].index(" = subparsers.add_parser(")
    block = source[start:nxt]
    assert block.count("_wh.") == 19, block.count("_wh.")
    assert block.count('.envelope(locale=help_locale)["text"]') == 19
    assert "# i18n: ignore" not in block


def test_the_mcp_block_renders_through_the_help_bootstrap() -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    start = source.index("mcp_parser = subparsers.add_parser(")
    nxt = source.index("add_search_subparser(subparsers, parent_parser)")
    block = source[start:nxt]
    assert block.count("_mh.") == 19, block.count("_mh.")
    assert block.count('.envelope(locale=help_locale)["text"]') == 19
    assert "# i18n: ignore" not in block


def test_the_ceiling_has_no_slack_and_stays_unactivated(i18n_check) -> None:
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    live = _live_cli_count(i18n_check)
    assert baseline["files"].get(CLI) == live, (baseline["files"].get(CLI), live)
    assert live <= CEILING, live
    assert CLI not in baseline["enforced"], "cli.py is activated without the file being clean"


def test_a_planted_literal_fails_the_checker(tmp_path, monkeypatch, i18n_check) -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    target = tmp_path / CLI
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    baseline_dir = tmp_path / "i18n"
    baseline_dir.mkdir()
    baseline = baseline_dir / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "note": "slice fixture",
                "enforced": [CLI],
                "files": {CLI: _live_cli_count(i18n_check)},
            }
        )
    )
    allowlist = baseline_dir / "allowlist.toml"
    allowlist.write_text("entries = []\n")
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    monkeypatch.setattr(i18n_check, "BASELINE", baseline)
    monkeypatch.setattr(i18n_check, "ALLOWLIST", allowlist)
    monkeypatch.setattr(i18n_check, "check_catalogues", lambda: [])
    assert i18n_check.main([]) == 0
    target.write_text(source + '\nprint("planted wake prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


CASES = [
    (
        _wh.wake().envelope(locale="en")["text"],
        "Inspect scheduled wakes and the supervisor that fires them",
    ),  # wake
    (
        _wh.status().envelope(locale="en")["text"],
        "whether a supervisor is installed, and what it would fire next",
    ),  # status
    (
        _wh.json().envelope(locale="en")["text"],
        "machine-readable output",
    ),  # json
    (
        _wh.status_install().envelope(locale="en")["text"],
        "install the supervisor now rather than waiting for the next schedule",
    ),  # status_install
    (
        _wh.status_uninstall().envelope(locale="en")["text"],
        "remove the supervisor; scheduled wakes then fire only when a session is open",
    ),  # status_uninstall
    (
        _wh.install().envelope(locale="en")["text"],
        "install or repair the supervisor now (restarts a stale or stopped one)",
    ),  # install
    (
        _wh.list().envelope(locale="en")["text"],
        "every scheduled wake on this machine, soonest first",
    ),  # list
    (
        _wh.create().envelope(locale="en")["text"],
        "schedule a wake for a session (installs the supervisor on demand)",
    ),  # create
    (
        _wh.create_session().envelope(locale="en")["text"],
        "session id to wake",
    ),  # create_session
    (
        _wh.create_when().envelope(locale="en")["text"],
        'when to fire: a duration ("in 2m", "45s") or a clock time ("at 09:30")',
    ),  # create_when
    (
        _wh.create_message(max=str(MAX_WAKE_MESSAGE_CHARS)).envelope(locale="en")["text"],
        "the self-prompt delivered when it fires (at most 2000 characters)",
    ),  # create_message
    (
        _wh.create_every().envelope(locale="en")["text"],
        'repeat every DURATION after the first fire ("5m", "1h", "8h30m")',
    ),  # create_every
    (
        _wh.create_until().envelope(locale="en")["text"],
        'stop repeating after this time ("in 7d", "at 09:30")',
    ),  # create_until
    (
        _wh.create_limit().envelope(locale="en")["text"],
        "stop repeating after N fires",
    ),  # create_limit
    (
        _wh.create_notify().envelope(locale="en")["text"],
        (
            "notify on completion of the wake's turn (default quiet: scripted arms "
            "match the contract; errored turns notify regardless)"
        ),
    ),  # create_notify
    (
        _wh.serve().envelope(locale="en")["text"],
        "run the supervisor in the foreground (what the LaunchAgent runs)",
    ),  # serve
    (
        _wh.serve_once().envelope(locale="en")["text"],
        "fire whatever is due right now, then exit",
    ),  # serve_once
    (
        _mh.mcp().envelope(locale="en")["text"],
        "Manage MCP servers",
    ),  # mcp
    (
        _mh.list().envelope(locale="en")["text"],
        "List configured MCP servers (all sources merged)",
    ),  # list
    (
        _mh.add().envelope(locale="en")["text"],
        "Add an MCP server (stdio command or http/sse URL)",
    ),  # add
    (
        _mh.add_name().envelope(locale="en")["text"],
        "Server name",
    ),  # add_name
    (
        _mh.add_command().envelope(locale="en")["text"],
        "Stdio command to launch the server",
    ),  # add_command
    (
        _mh.add_arg().envelope(locale="en")["text"],
        "Stdio command argument (repeatable)",
    ),  # add_arg
    (
        _mh.add_env().envelope(locale="en")["text"],
        "Environment variable KEY=VALUE for the stdio server (repeatable)",
    ),  # add_env
    (
        _mh.add_url().envelope(locale="en")["text"],
        "HTTP/SSE server URL (alternative to --command)",
    ),  # add_url
    (
        _mh.add_scope().envelope(locale="en")["text"],
        "Config scope to write (default: global ~/.local-operator/mcp.json)",
    ),  # add_scope
    (
        _mh.add_oauth().envelope(locale="en")["text"],
        "Enable OAuth for a remote HTTP server",
    ),  # add_oauth
    (
        _mh.remove_server().envelope(locale="en")["text"],
        "Remove an MCP server from a config scope",
    ),  # remove_server
    (
        _mh.remove_name().envelope(locale="en")["text"],
        "Server name to remove",
    ),  # remove_name
    (
        _mh.remove_scope().envelope(locale="en")["text"],
        "Config scope to remove from (default: global)",
    ),  # remove_scope
    (
        _mh.login().envelope(locale="en")["text"],
        "Authenticate one OAuth-enabled MCP server",
    ),  # login
    (
        _mh.login_name().envelope(locale="en")["text"],
        "Server name to authenticate",
    ),  # login_name
    (
        _mh.logout().envelope(locale="en")["text"],
        "Remove the stored OAuth credential for one MCP server",
    ),  # logout
    (
        _mh.logout_name().envelope(locale="en")["text"],
        "Server name to log out",
    ),  # logout_name
    (
        _mh.reauth().envelope(locale="en")["text"],
        "Log out of one OAuth MCP server and run a fresh authorization",
    ),  # reauth
    (
        _mh.reauth_name().envelope(locale="en")["text"],
        "Server name to re-authenticate",
    ),  # reauth_name
]
