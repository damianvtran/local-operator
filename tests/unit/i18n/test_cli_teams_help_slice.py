"""Help tranche 2: `lop teams` — the argparse help, extracted to `cli.teams_help`.

Same ruled classes as help tranche 1, applied per site: **help prose is copy ->
keys, flag and command tokens stay IN SITU inside the values; deprecated/hidden
flags may take ignore-with-reason.** This block has 34 `help=` literals, no
descriptions, and no deprecated/hidden flags — so **zero ignores**, pinned by
test (`# i18n: ignore` must not appear in the block).

Rendering is "the pattern" (proven by the exec slice and help tranche 1):
`resolve_help_locale()` once, then `_th.<key>().envelope(locale=help_locale)["text"]`
— never the global renderer; an EN-only namespace must not un-ship audited locales.

Fidelity: every key renders to the byte its base call carried (table below,
anchored to the base file's AST), and the REAL argparse output (`lop teams
--help` plus each of its nine subcommands) is byte-identical base vs branch.

The ceiling assertions DERIVE the expected count from the scanner (equality =
no slack; a `<= 668` bound = may only fall) — the stale-pin chain that bit
twice (#2552 Q-1, #2558 M1/Q1) is closed structurally, not re-armed here.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import cli_teams_help as _th

REPO = Path(__file__).resolve().parents[3]
CLI = "local_operator/cli.py"
CEILING = 668  # the count after help tranche 2; may only fall


def _live_cli_count(i18n_check) -> int:
    entries = i18n_check.load_allowlist(i18n_check.ALLOWLIST)
    scanned = i18n_check.scan_tree(i18n_check.REPO)
    return i18n_check._counts(scanned, entries).get(CLI, 0)


def test_every_key_renders_the_byte_its_base_call_carried() -> None:
    for rendered, expected in CASES:
        assert rendered == expected, rendered


def test_the_catalogue_is_the_golden_fixture() -> None:
    live = catalogues.load_catalogue("en", "cli.teams_help")
    golden = json.loads(
        (REPO / "tests/unit/i18n/fixtures/cli_teams_help_en_golden.json").read_text(
            encoding="utf-8"
        )
    )
    assert live == golden, "the shipped catalogue and the fixture golden drifted"
    assert len(golden) == 34, len(golden)
    sidecar = catalogues.context("en", "cli.teams_help")
    for key, entry in sidecar.items():
        assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_the_teams_block_renders_through_the_help_bootstrap() -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    start = source.index("teams_parser = subparsers.add_parser(")
    # skip PAST this call's own prefix before looking for the next top-level one
    after = start + len("teams_parser = subparsers.add_parser(")
    nxt = after + source[after:].index(" = subparsers.add_parser(")
    block = source[start:nxt]
    assert block.count("_th.") == 34, block.count("_th.")
    assert block.count('.envelope(locale=help_locale)["text"]') == 34
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
    target.write_text(source + '\nprint("planted teams prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


CASES = [
    (_th.teams().envelope(locale="en")["text"], "Manage teams"),  # teams
    (_th.list().envelope(locale="en")["text"], "List all teams"),  # list
    (_th.create().envelope(locale="en")["text"], "Create a new team"),  # create
    (_th.create_name().envelope(locale="en")["text"], "Name of the team to create"),  # create_name
    (
        _th.create_manager().envelope(locale="en")["text"],
        "Role or specialist who orchestrates (default: manager)",
    ),  # create_manager
    (
        _th.create_member().envelope(locale="en")["text"],
        "Roster slot as role or role:count (repeatable)",
    ),  # create_member
    (
        _th.create_description().envelope(locale="en")["text"],
        "One-line description",
    ),  # create_description
    (
        _th.create_label().envelope(locale="en")["text"],
        (
            "Display label shown in listings, the chart and the status band (free text; a"
            "ddressing still uses the name)"
        ),
    ),  # create_label
    (
        _th.create_alias().envelope(locale="en")["text"],
        (
            "Extra addressing key for this team (repeatable; letters, digits, dot, unders"
            "core, hyphen)"
        ),
    ),  # create_alias
    (_th.show().envelope(locale="en")["text"], "Show a team's roster and briefs"),  # show
    (_th.show_name().envelope(locale="en")["text"], "Name of the team to show"),  # show_name
    (_th.delete().envelope(locale="en")["text"], "Delete a team by name"),  # delete
    (_th.delete_name().envelope(locale="en")["text"], "Name of the team to delete"),  # delete_name
    (
        _th.push().envelope(locale="en")["text"],
        (
            "Push a team to an organization or the public hub (Agent Hub); exit codes: 0 "
            "published, 1 error, 2 preview-only with changes pending, 3 declined, 130 can"
            "celled"
        ),
    ),  # push
    (_th.push_name().envelope(locale="en")["text"], "Name of the local team to push"),  # push_name
    (
        _th.push_org().envelope(locale="en")["text"],
        (
            "Organization tenant to publish into (`lop teams push --org <tenant_id> <team"
            ">`; requires `lop login radient`)"
        ),
    ),  # push_org
    (
        _th.push_public().envelope(locale="en")["text"],
        "Publish to the public Agent Hub instead of an organization",
    ),  # push_public
    (
        _th.push_id().envelope(locale="en")["text"],
        (
            'Hub team id to overwrite (republish; what `lop teams push` printed as "Team '
            'ID"); needs --org or --public'
        ),
    ),  # push_id
    (
        _th.pull().envelope(locale="en")["text"],
        "Pull a team from the hub by id, or by name from the public hub (Agent Hub)",
    ),  # pull
    (
        _th.pull_team_id().envelope(locale="en")["text"],
        (
            "ID of the published team to pull, or a public team's name (`lop teams pull a"
            "-team`; names resolve over the public listing)"
        ),
    ),  # pull_team_id
    (
        _th.pull_org().envelope(locale="en")["text"],
        (
            "Organization tenant the team belongs to (`lop teams pull --org <tenant_id> <"
            "team-id>`; requires `lop login radient`)"
        ),
    ),  # pull_org
    (
        _th.sync().envelope(locale="en")["text"],
        "Check pulled teams against the hub and merge their updates",
    ),  # sync
    (
        _th.sync_name().envelope(locale="en")["text"],
        "Sync only the team with this name",
    ),  # sync_name
    (
        _th.sync_all().envelope(locale="en")["text"],
        "Sync every linked team (default when --name is absent)",
    ),  # sync_all
    (
        _th.link().envelope(locale="en")["text"],
        "Link a local team to a hub team so it can receive updates",
    ),  # link
    (_th.link_name().envelope(locale="en")["text"], "Name of the local team"),  # link_name
    (
        _th.link_hub_team_id().envelope(locale="en")["text"],
        "ID of the hub team to link it to",
    ),  # link_hub_team_id
    (
        _th.link_org().envelope(locale="en")["text"],
        "Organization tenant that owns the hub team",
    ),  # link_org
    (
        _th.link_accept_unknown_baseline().envelope(locale="en")["text"],
        "Link even though the local copy differs from the hub's (nothing is deleted)",
    ),  # link_accept_unknown_baseline
    (_th.search().envelope(locale="en")["text"], "Search the public Agent Hub for teams"),  # search
    (
        _th.search_query().envelope(locale="en")["text"],
        (
            "Text to match against a public team's name or description (omit to list the "
            "newest teams)"
        ),
    ),  # search_query
    (
        _th.search_page().envelope(locale="en")["text"],
        "Page to list when no query is given (default: 1)",
    ),  # search_page
    (
        _th.search_perpage().envelope(locale="en")["text"],
        "Teams per page for a listing (default: 20; the hub maxes at 100)",
    ),  # search_perpage
    (
        _th.search_json().envelope(locale="en")["text"],
        "Print the raw listing or matches as JSON",
    ),  # search_json,
]
