"""Ratchet slice 12: `lop browser` — the first CLI tranche, AND its CLASS DECISION.

cli.py carries 798 findings in a 16,080-line file: it is a PROGRAMME, not a
slice. Root accepted the decomposition by sub-app ("a PR no reviewer can hold
in their head" is a gate-quality criterion, not a convenience one) and required
the print/echo class decision as an artifact of the first tranche, PER SITE,
never by scanner heuristics. This is it.

| class | sites | treatment |
|---|---|---|
| (a) human status/progress copy | 54 | **extracted** -> `cli.browser.*` |
| (b) token line (contract) | 1 | **ignored per site**: `  generation={...}` — the token |
| | | names the `--generation` flag |
| (c) ANSI-styled passthrough | 4 | **ignored per site**: the text is localized at |
| | | its own layer; the escapes are formatting |
| (d) command hints | in (a) | whole line is copy; command tokens stay in |
| | | situ (the `--help` ruling) |

**The ignores must not mask live English.** Two class-(c) sites carry local
English in their neighbourhood, and it was extracted in the same slice rather
than waved through: the install failure DEFAULT (`install failed`, previously
buried inside `result.get('error', 'install failed')`) and the drive `message`
assembly (`no connected extension matches '...'` / `no single connected
extension matches '...'`) — the assignment-blind class, invisible to the
collector, exactly like the sidebar's assignments in slice 11.

**Byte fidelity is proven by rendering, not by reading:** every extracted key
renders, with sentinel params, to the bytes the base file's own expressions
produced (the table below — the values were extracted from the base file's AST,
not retyped).

**What this slice does NOT extract — and the list is NOT exhaustive:** the
`build_cli_parser()` help surface (276 findings: its own programme, split per
subparser group), every other command family in the file (`main` 35, `mobile`
42, `sessions` 32+, `qwencloud` 36, `agents` 17, and the long tail), and
assignment-class English the collector cannot see (`_waiting_label(...)`'s
labels, and whatever else a file-wide sweep finds). **cli.py is NOT
activated**: activation happens when the FILE reaches zero findings; this
tranche is one region, so it shrinks the recorded ceiling by exactly the 60 it
cleared — a map, not a certificate.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import cli_browser as _b
from local_operator.i18n.text import tr

REPO = Path(__file__).resolve().parents[3]
_bw = tr("cli.browser")

CLI = "local_operator/cli.py"
CEILING = 747


def test_every_key_renders_the_bytes_the_base_expressions_produced() -> None:
    """Byte fidelity, driven with sentinel params: `(call, expected)` pairs.

    Each expected string is the base file's own literal (extracted by AST),
    with sentinels where the base interpolated a value.
    """
    for rendered, expected in CASES:
        assert rendered == expected, rendered


def test_the_catalogue_is_the_golden_fixture() -> None:
    live = catalogues.load_catalogue("en", "cli.browser")
    golden = json.loads(
        (REPO / "tests/unit/i18n/fixtures/cli_browser_en_golden.json").read_text(encoding="utf-8")
    )
    assert live == golden, "the shipped catalogue and the fixture golden drifted"
    assert len(golden) == 56, len(golden)
    sidecar = catalogues.context("en", "cli.browser")
    for key, entry in sidecar.items():
        assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_the_class_decision_is_on_the_sites() -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    pragmas = [line for line in source.splitlines() if "# i18n: ignore" in line]
    browser_pragmas = [line for line in pragmas if "generation" in line or "ANSI-styled" in line]
    assert len(browser_pragmas) == 6, browser_pragmas
    assert sum("generation" in line for line in browser_pragmas) == 1
    assert sum("ANSI-styled" in line for line in browser_pragmas) == 5


def test_the_ignores_do_not_mask_live_english() -> None:
    """The two neighbourhoods the ignores could have hidden were extracted."""
    source = (REPO / CLI).read_text(encoding="utf-8")
    assert "result.get('error', _bt(_b.install_failed()))" in source
    assert "_bt(_b.drive_no_connected(target=args.target))" in source
    assert "_bt(_b.drive_no_single(target=args.target))" in source
    assert _bw(_b.install_failed()) == "install failed"
    assert _bw(_b.drive_no_connected(target="tok")) == "no connected extension matches 'tok'."
    assert _bw(_b.drive_no_single(target="tok")) == "no single connected extension matches 'tok'."


def test_the_usage_line_renders_its_literal_braces() -> None:
    """The one value carrying literal braces: encoded as a quoted ICU run, and
    verified by rendering — the engine's per-brace form does NOT work here."""
    assert (
        _bw(_b.usage())
        == "usage: lop browser {install|status|start|stop|restart|pair|drive|logs|uninstall|serve}"
    )


def test_the_ceiling_shrank_by_exactly_this_slice() -> None:
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["files"].get(CLI) == CEILING, baseline["files"].get(CLI)
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
        json.dumps({"note": "slice fixture", "enforced": [CLI], "files": {CLI: CEILING}})
    )
    allowlist = baseline_dir / "allowlist.toml"
    allowlist.write_text("entries = []\n")
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    monkeypatch.setattr(i18n_check, "BASELINE", baseline)
    monkeypatch.setattr(i18n_check, "ALLOWLIST", allowlist)
    monkeypatch.setattr(i18n_check, "check_catalogues", lambda: [])
    assert i18n_check.main([]) == 0
    target.write_text(source + '\nprint("planted slice prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


CASES = [
    (
        _bw(_b.no_changes()),
        "No changes. Repeat with --yes to approve this exact session and generation.",
    ),  # no_changes
    (_bw(_b.invalid_session()), "Invalid session id; no action taken."),  # invalid_session
    (
        _bw(_b.cleanup_blocked(detail="«detail»")),
        "Cleanup blocked: «detail» Nothing was closed.",
    ),  # cleanup_blocked
    (
        _bw(_b.live_tabs(count="«count»")),
        "\nlive tabs («count», not selectable for cleanup):",
    ),  # live_tabs
    (
        _bw(_b.handles_redacted()),
        (
            "  Handles are redacted, so these cannot be attributed to a record above.\n  Clo"
            "se an unwanted tab by hand; provenance is unproven."
        ),
    ),  # handles_redacted
    (
        _bw(_b.read_only()),
        "\nRead-only. PID absence, age, and localhost URLs never authorize cleanup.",
    ),  # read_only
    (
        _bw(_b.exact_cleanup()),
        "Exact cleanup: lop browser cleanup SESSION --generation GENERATION --yes",
    ),  # exact_cleanup
    (_bw(_b.bridge_installed()), "\nbrowser bridge installed and healthy."),  # bridge_installed
    (
        _bw(_b.load_extension()),
        "  load the Local Operator extension, then run `lop browser pair`.",
    ),  # load_extension
    (_bw(_b.bridge_reconciled()), "\nbrowser bridge state reconciled."),  # bridge_reconciled
    (_bw(_b.installed_line(value="«value»")), "installed:           «value»"),  # installed_line
    (_bw(_b.daemon_healthy(value="«value»")), "daemon healthy:      «value»"),  # daemon_healthy
    (
        _bw(_b.bridge_predates()),
        ("bridge:              predates the file-transfer actions — run 'lop browser res" "tart'"),
    ),  # bridge_predates
    (
        _bw(_b.extension_connected(value="«value»")),
        "extension connected: «value»",
    ),  # extension_connected
    (
        _bw(_b.extension_update_note(note="«note»")),
        "                     «note» (fixes and security patches)",
    ),  # extension_update_note
    (_bw(_b.paired_line(value="«value»")), "paired:              «value»"),  # paired_line
    (_bw(_b.driving_tabs(label="«label»")), "driving:             «label»"),  # driving_tabs
    (_bw(_b.driving_none()), "driving:             no tabs driven"),  # driving_none
    (_bw(_b.driving_url(url="«url»")), "driving:             «url»"),  # driving_url
    (
        _bw(_b.heartbeat_stale(age="«age»")),
        (
            "\n\x1b[1;33mheartbeat:           STALE by «age»s (discovery file is not being refr"
            "eshed)\x1b[0m"
        ),
    ),  # heartbeat_stale
    (
        _bw(_b.heartbeat_fallback()),
        (
            "                     sessions will fall back to cmux despite the daemon being "
            "healthy."
        ),
    ),  # heartbeat_fallback
    (
        _bw(_b.heartbeat_repair()),
        "                     run 'lop browser status --repair' to reconcile.",
    ),  # heartbeat_repair
    (_bw(_b.port_line(port="«port»")), "port:                «port»"),  # port_line
    (_bw(_b.log_line(log="«log»")), "log:                 «log»"),  # log_line
    (_bw(_b.downloads_line(path="«path»")), "downloads:           «path»"),  # downloads_line
    (
        _bw(_b.downloads_detail(size="«size»", audit="«audit»")),
        "                     «size» bytes, one audit row per decision in «audit»",
    ),  # downloads_detail
    (_bw(_b.supervisor_line(name="«name»")), "supervisor:          «name»"),  # supervisor_line
    (_bw(_b.config_root_line(path="«path»")), "config root:         «path»"),  # config_root_line
    (_bw(_b.legacy_install(path="«path»")), "legacy install:      «path»"),  # legacy_install
    (
        _bw(_b.legacy_note()),
        (
            "                     (written by an older build under the shared name;\n       "
            "               'lop browser uninstall' removes it)"
        ),
    ),  # legacy_note
    (
        _bw(_b.unclaimed_registration(detail="«detail»")),
        "\n\x1b[1;33munclaimed registration:\x1b[0m «detail»",
    ),  # unclaimed_registration
    (
        _bw(_b.pair_no_extension()),
        "no browser extension is paired. Run 'lop browser pair' to pair one.",
    ),  # pair_no_extension
    (
        _bw(_b.waiting_header()),
        "waiting:             these installs have asked to pair",
    ),  # waiting_header
    (
        _bw(_b.revoke_no_match(target="«target»")),
        "\x1b[1;31mno single authorised extension matches '«target»'.\x1b[0m",
    ),  # revoke_no_match
    (_bw(_b.nothing_paired()), "  (nothing is paired)"),  # nothing_paired
    (
        _bw(_b.revoked(label="«label»")),
        "revoked «label»; any live connection for it is dropped within a few seconds.",
    ),  # revoked
    (
        _bw(_b.revoked_browser()),
        "revoked the paired browser; any live connection is dropped within a few seconds.",
    ),  # revoked_browser
    (_bw(_b.pairing_code(code="«code»")), "pairing code: «code»"),  # pairing_code
    (
        _bw(_b.enter_code()),
        "enter this 6-digit code in the Local Operator extension popup.",
    ),  # enter_code
    (_bw(_b.open_popup()), "open the extension popup to pair a browser again."),  # open_popup
    (
        _bw(_b.pairing_code_labeled(code="«code»", label="«label»")),
        "pairing code: «code»   «label»",
    ),  # pairing_code_labeled
    (
        _bw(_b.enter_each_code()),
        "enter each code in the matching Local Operator extension popup.",
    ),  # enter_each_code
    (
        _bw(_b.already_paired()),
        (
            "a browser is already paired. To pair another, open ITS popup and enter the cod"
            "e that appears here (the code appears once that install's extension connects)."
            " 'lop browser pair --list' lists every authorised install; '--reset' revokes t"
            "hem all."
        ),
    ),  # already_paired
    (
        _bw(_b.no_extension_waiting()),
        "no extension is waiting to pair. Open the extension popup, then retry.",
    ),  # no_extension_waiting
    (_bw(_b.authorised_installs()), "authorised installs:"),  # authorised_installs
    (_bw(_b.now_driving(id="«id»")), "now driving: «id»"),  # now_driving
    (_bw(_b.service_ok(command="«command»")), "browser bridge «command» ok"),  # service_ok
    (
        _bw(_b.logs_missing(path="«path»")),
        (
            "no daemon log at «path».\nThe bridge has not run under a service supervisor on "
            "this machine. Run `lop browser install`, or `lop browser serve` to run it in t"
            "he foreground."
        ),
    ),  # logs_missing
    (
        _bw(_b.cannot_run(command="«command»", location="«location»")),
        (
            "\x1b[1;31mcannot run `«command»`: not installed on this system.\x1b[0m\nThe daemon's "
            "output is at «location»."
        ),
    ),  # cannot_run
    (_bw(_b.note_prefix(warning="«warning»")), "\x1b[1;33mnote:\x1b[0m «warning»"),  # note_prefix
    (
        _bw(_b.usage()),
        (
            "usage: lop browser {install|status|start|stop|restart|pair|drive|logs|uninstal"
            "l|serve}"
        ),
    ),  # usage
    (
        _bw(_b.drive_no_connected(target="«target»")),
        "no connected extension matches '«target»'.",
    ),  # drive_no_connected
    (
        _bw(_b.drive_no_single(target="«target»")),
        "no single connected extension matches '«target»'.",
    ),  # drive_no_single
    (_bw(_b.install_failed()), "install failed"),  # install_failed
    (_bw(_b.status_yes()), "yes"),  # status_yes
    (_bw(_b.status_no()), "no"),  # status_no
]
