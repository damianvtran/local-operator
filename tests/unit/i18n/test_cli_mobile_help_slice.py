"""Help tranche 4: `lop mobile` — argparse help to `cli.mobile_help`.

Same ruled classes as help tranches 1-3, applied per site: **help prose is
copy -> keys, flag and command tokens stay IN SITU inside the values;
deprecated/hidden flags may take ignore-with-reason.** This block has 15
`help=` findings, no descriptions, no deprecated/hidden flags — so **zero
ignores**, pinned by test (`# i18n: ignore` must not appear).

One site is the programme's first `select`: the `for action in ("start",
"stop", "restart")` loop rendered `f"{action.capitalize()} the daemon"`.
The varying part is LANGUAGE (a verb), so the key carries the three words as
select branches — translators localize them — and the site passes the raw
loop value; a pre-capitalized `{action}` param would have shipped an
English-only transformation as data. EN renders byte-identically for all
three iterations (three table entries below).

Rendering is "the pattern": `resolve_help_locale()` once, then
`_mb.<key>().envelope(locale=help_locale)["text"]` — never the global renderer.

Fidelity: every key renders to the byte its base call carried (table below,
anchored to the base file's AST), and the REAL argparse output (`lop mobile
--help` plus each of its eleven subcommands, bare `mobile`, unknown
subcommand) is byte-identical base vs branch.

The ceiling assertions DERIVE the expected count from the scanner (equality =
no slack; a `<= 615` bound = may only fall) — the stale-pin chain stays
closed structurally.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import cli_mobile_help as _mb

REPO = Path(__file__).resolve().parents[3]
CLI = "local_operator/cli.py"
CEILING = 615  # the count after help tranche 4; may only fall


def _live_cli_count(i18n_check) -> int:
    entries = i18n_check.load_allowlist(i18n_check.ALLOWLIST)
    scanned = i18n_check.scan_tree(i18n_check.REPO)
    return i18n_check._counts(scanned, entries).get(CLI, 0)


def test_every_key_renders_the_byte_its_base_call_carried() -> None:
    for rendered, expected in CASES:
        assert rendered == expected, rendered


def test_the_catalogue_is_the_golden_fixture() -> None:
    live = catalogues.load_catalogue("en", "cli.mobile_help")
    golden = json.loads(
        (REPO / "tests/unit/i18n/fixtures/cli_mobile_help_en_golden.json").read_text(
            encoding="utf-8"
        )
    )
    assert live == golden, "the shipped catalogue and the fixture golden drifted"
    assert len(golden) == 15, len(golden)
    sidecar = catalogues.context("en", "cli.mobile_help")
    for key, entry in sidecar.items():
        assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_the_mobile_block_renders_through_the_help_bootstrap() -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    start = source.index("mobile_parser = subparsers.add_parser(")
    after = start + len("mobile_parser = subparsers.add_parser(")
    nxt = after + source[after:].index(" = subparsers.add_parser(")
    block = source[start:nxt]
    assert block.count("_mb.") == 15, block.count("_mb.")
    assert block.count('.envelope(locale=help_locale)["text"]') == 15
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
    target.write_text(source + '\nprint("planted mobile prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


CASES = [
    (
        _mb.mobile().envelope(locale="en")["text"],
        "Phone access: control this machine's lop sessions",
    ),  # mobile
    (
        _mb.install().envelope(locale="en")["text"],
        "Install the daemon, password and LaunchAgent",
    ),  # install
    (
        _mb.status().envelope(locale="en")["text"],
        "Daemon health, gate and live sessions",
    ),  # status
    (
        _mb.daemon_action(action="start").envelope(locale="en")["text"],
        "Start the daemon",
    ),  # daemon_action[start]
    (
        _mb.daemon_action(action="stop").envelope(locale="en")["text"],
        "Stop the daemon",
    ),  # daemon_action[stop]
    (
        _mb.daemon_action(action="restart").envelope(locale="en")["text"],
        "Restart the daemon",
    ),  # daemon_action[restart]
    (
        _mb.logs().envelope(locale="en")["text"],
        "Tail the daemon's and the runtimes' logs",
    ),  # logs
    (
        _mb.logs_lines().envelope(locale="en")["text"],
        "Lines per file (up to two are read)",
    ),  # logs_lines
    (
        _mb.logs_follow().envelope(locale="en")["text"],
        (
            "Follow by name, so a log created or rotated while you watch is seen (a"
            " descriptor-bound follow misses both)"
        ),
    ),  # logs_follow
    (
        _mb.password().envelope(locale="en")["text"],
        "Show or rotate the portal password",
    ),  # password
    (
        _mb.devices().envelope(locale="en")["text"],
        "List this computer's push devices, or revoke/unrevoke one",
    ),  # devices
    (
        _mb.devices_command().envelope(locale="en")["text"],
        "list (default), revoke <device_id>, or unrevoke <device_id>",
    ),  # devices_command
    (
        _mb.devices_device_id().envelope(locale="en")["text"],
        "the device_id `lop mobile devices` lists; required by revoke/unrevoke",
    ),  # devices_device_id
    (
        _mb.devices_port().envelope(locale="en")["text"],
        "the mobile daemon's port (default: the daemon's own default)",
    ),  # devices_port
    (
        _mb.uninstall().envelope(locale="en")["text"],
        "Remove the LaunchAgent",
    ),  # uninstall
    (
        _mb.uninstall_purge().envelope(locale="en")["text"],
        "Also delete the password",
    ),  # uninstall_purge
    (
        _mb.serve().envelope(locale="en")["text"],
        "Run the daemon (foreground)",
    ),  # serve
]
