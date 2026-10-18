"""Help tranche 7: `lop browser` help + `lop send` + `lop model`.

Three blocks, 30 flagged `help=` findings (browser 18, send 7, model 5) -> three
namespaces. Same ruled classes: prose -> keys; tokens in situ; zero ignores.

Two shapes recur here:
- browser's start/stop/restart loop is the programme's SECOND `select` site
  (the mobile daemon_action shape): the verb is language, so the key carries
  the three branches and the site passes the raw loop value; EN byte-identical
  for all three iterations (three rows below).
- the tabs/reconcile loop consumes literals from a TUPLE via a loop variable
  (`for action, blurb in (...): add_parser(action, help=blurb)`) -- the second
  variable-indirection blind-spot instance. Both blurbs are extracted anyway
  (count-neutral); the loop's `help=blurb` stays valid once the tuple values
  are envelope renders.

The send epilog (assembled from SEND_EXIT_UNCONFIRMED) stays as code: `epilog=`
is not a scanner sink and text built from constants keeps the constants as the
single source (the T6 `mobile devices` precedent, reviewer-verified).

Per-SITE shape pins (the T6 precedent) + derived ceiling pins below.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import cli_browser_help as _bh
from local_operator.i18n.keys import cli_model_help as _mo
from local_operator.i18n.keys import cli_send_help as _se

REPO = Path(__file__).resolve().parents[3]
CLI = "local_operator/cli.py"
CEILING = 519  # the count after help tranche 7; may only fall

PINS = [
    ("_bh", "browser"),
    ("_bh", "install"),
    ("_bh", "status"),
    ("_bh", "status_repair"),
    ("_bh", "json"),
    ("_bh", "tabs"),
    ("_bh", "reconcile"),
    ("_bh", "cleanup"),
    ("_bh", "cleanup_yes"),
    ("_bh", "daemon_action"),
    ("_bh", "pair"),
    ("_bh", "pair_reset"),
    ("_bh", "pair_list"),
    ("_bh", "pair_revoke"),
    ("_bh", "drive"),
    ("_bh", "drive_target"),
    ("_bh", "logs"),
    ("_bh", "uninstall"),
    ("_bh", "uninstall_purge"),
    ("_bh", "serve"),
    ("_se", "send"),
    ("_se", "target"),
    ("_se", "message"),
    ("_se", "pid"),
    ("_se", "session"),
    ("_se", "steer"),
    ("_se", "wake"),
    ("_mo", "model"),
    ("_mo", "target"),
    ("_mo", "selector"),
    ("_mo", "pid"),
    ("_mo", "session"),
]
assert len(PINS) == 32


def _live_cli_count(i18n_check) -> int:
    entries = i18n_check.load_allowlist(i18n_check.ALLOWLIST)
    scanned = i18n_check.scan_tree(i18n_check.REPO)
    return i18n_check._counts(scanned, entries).get(CLI, 0)


def test_every_key_renders_the_byte_its_base_call_carried() -> None:
    for rendered, expected in CASES:
        assert rendered == expected, rendered


def test_the_catalogues_are_the_golden_fixtures() -> None:
    for namespace, fixture, size in (
        ("cli.browser_help", "cli_browser_help_en_golden.json", 20),
        ("cli.send_help", "cli_send_help_en_golden.json", 7),
        ("cli.model_help", "cli_model_help_en_golden.json", 5),
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


def test_every_block_renders_through_its_envelope_and_no_ignore_exists() -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    for alias, key in PINS:
        if key == "daemon_action":
            # the select site passes the raw loop value; its shape is param'd
            shape = f'{alias}.{key}(action=action).envelope(locale=help_locale)["text"]'
        else:
            shape = f'{alias}.{key}().envelope(locale=help_locale)["text"]'
        assert shape in source, shape
    for line in source.splitlines():
        if "# i18n: ignore" in line:
            assert not any(alias in line for alias in ("_bh.", "_se.", "_mo.")), line


def test_the_ceiling_has_no_slack_and_stays_unactivated(i18n_check) -> None:
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    live = _live_cli_count(i18n_check)
    assert baseline["files"].get(CLI) == live, (baseline["files"].get(CLI), live)
    assert live <= CEILING, live
    assert not i18n_check.in_enforced_scope(
        CLI, baseline["enforced"]
    ), "cli.py is activated without the file being clean"


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
    target.write_text(source + '\nprint("planted tranche7 prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


CASES = [
    (
        _bh.browser().envelope(locale="en")["text"],
        "Connect the browser tool to a Chromium extension",
    ),  # browser_help.browser
    (
        _bh.install().envelope(locale="en")["text"],
        "Install the bridge daemon",
    ),  # browser_help.install
    (
        _bh.status().envelope(locale="en")["text"],
        "Show daemon, extension and pairing status",
    ),  # browser_help.status
    (
        _bh.status_repair().envelope(locale="en")["text"],
        (
            "Reconcile advertised state against reality: drop tabs that no longer e"
            "xist and republish a fresh heartbeat. Safe while sessions are live."
        ),
    ),  # browser_help.status_repair
    (
        _bh.json().envelope(locale="en")["text"],
        "machine-readable output",
    ),  # browser_help.json
    (
        _bh.tabs().envelope(locale="en")["text"],
        "List durable ownership records and live tabs (read-only)",
    ),  # browser_help.tabs
    (
        _bh.reconcile().envelope(locale="en")["text"],
        "Alias for 'tabs': inspect ownership without closing anything",
    ),  # browser_help.reconcile
    (
        _bh.cleanup().envelope(locale="en")["text"],
        "Close one proven-terminal browser owner after revalidation",
    ),  # browser_help.cleanup
    (
        _bh.cleanup_yes().envelope(locale="en")["text"],
        "Approve this exact cleanup",
    ),  # browser_help.cleanup_yes
    (
        _bh.daemon_action(action="start").envelope(locale="en")["text"],
        "Start the daemon",
    ),  # browser_help.daemon_action[start]
    (
        _bh.daemon_action(action="stop").envelope(locale="en")["text"],
        "Stop the daemon",
    ),  # browser_help.daemon_action[stop]
    (
        _bh.daemon_action(action="restart").envelope(locale="en")["text"],
        "Restart the daemon",
    ),  # browser_help.daemon_action[restart]
    (
        _bh.pair().envelope(locale="en")["text"],
        "Show the extension pairing code",
    ),  # browser_help.pair
    (
        _bh.pair_reset().envelope(locale="en")["text"],
        "Revoke the paired browser first",
    ),  # browser_help.pair_reset
    (
        _bh.pair_list().envelope(locale="en")["text"],
        "List the authorised browser extensions, and which one is driving",
    ),  # browser_help.pair_list
    (
        _bh.pair_revoke().envelope(locale="en")["text"],
        (
            "Revoke ONE authorised extension (its id, an unambiguous id prefix, or "
            "part of its label), leaving every other one paired"
        ),
    ),  # browser_help.pair_revoke
    (
        _bh.drive().envelope(locale="en")["text"],
        "Choose which authorised extension drives the browser",
    ),  # browser_help.drive
    (
        _bh.drive_target().envelope(locale="en")["text"],
        "Extension id (or an unambiguous prefix), or part of its label",
    ),  # browser_help.drive_target
    (
        _bh.logs().envelope(locale="en")["text"],
        "Tail the daemon log",
    ),  # browser_help.logs
    (
        _bh.uninstall().envelope(locale="en")["text"],
        "Remove the bridge daemon",
    ),  # browser_help.uninstall
    (
        _bh.uninstall_purge().envelope(locale="en")["text"],
        "Also delete pairing state",
    ),  # browser_help.uninstall_purge
    (
        _bh.serve().envelope(locale="en")["text"],
        "Run the bridge daemon (foreground)",
    ),  # browser_help.serve
    (
        _se.send().envelope(locale="en")["text"],
        "Send a message to another local lop session (no cmux needed)",
    ),  # send_help.send
    (
        _se.target().envelope(locale="en")["text"],
        (
            "conversation-name / session-id / cwd substring (case-insensitive). Omi"
            "t when addressing with --pid/--session; passing both is refused."
        ),
    ),  # send_help.target
    (
        _se.message().envelope(locale="en")["text"],
        (
            "message text; omit to read the body from stdin. With --pid/--session a"
            " single positional IS the message, and typing one while also piping a "
            "body is refused."
        ),
    ),  # send_help.message
    (
        _se.pid().envelope(locale="en")["text"],
        "target by exact pid",
    ),  # send_help.pid
    (
        _se.session().envelope(locale="en")["text"],
        "target by exact session id",
    ),  # send_help.session
    (
        _se.steer().envelope(locale="en")["text"],
        "inject mid-turn (steer) instead of the default mailbox",
    ),  # send_help.steer
    (
        _se.wake().envelope(locale="en")["text"],
        "if the target is idle, drive a turn now (mailbox mode only)",
    ),  # send_help.wake
    (
        _mo.model().envelope(locale="en")["text"],
        "Switch another running lop session's model (like /model there)",
    ),  # model_help.model
    (
        _mo.target().envelope(locale="en")["text"],
        (
            "conversation-name / session-id / cwd substring (case-insensitive). Omi"
            "t when addressing with --pid/--session."
        ),
    ),  # model_help.target
    (
        _mo.selector().envelope(locale="en")["text"],
        "the model to switch to, e.g. deepseek/deepseek-flash",
    ),  # model_help.selector
    (
        _mo.pid().envelope(locale="en")["text"],
        "target by exact pid",
    ),  # model_help.pid
    (
        _mo.session().envelope(locale="en")["text"],
        "target by exact session id",
    ),  # model_help.session
]
