"""Help tranche 5: `lop agents` + `lop hub` — argparse help to keys.

Same ruled classes as help tranches 1-4, applied per site: **help prose is
copy -> keys, flag and command tokens stay IN SITU inside the values;
deprecated/hidden flags may take ignore-with-reason.** These two blocks have
33 `help=` findings (21 agents + 12 hub), no descriptions. The deprecated
`--force` alias on `agents sync` carries `help=argparse.SUPPRESS` -- not a
string and not a scanner sink, so it produces no finding; **zero ignores**,
pinned by test (`# i18n: ignore` must not appear in either block).

Out of scope here (tail's module-level families): the shared
`_add_hub_publish_flags` / `_add_hub_sync_flags` helpers, which serve teams
AND agents and carry their own help literals at module level.

Rendering is "the pattern": `resolve_help_locale()` once, then
`_ag.<key>()` / `_hu.<key>()` `.envelope(locale=help_locale)["text"]` --
never the global renderer.

Fidelity: every key renders to the byte its base call carried (table below,
anchored to the base file's AST), and the REAL argparse output (`lop agents
--help` plus its six subcommands; `lop hub --help` plus its subcommands) is
byte-identical base vs branch.

The ceiling assertions DERIVE the expected count from the scanner (equality =
no slack; a `<= 582` bound = may only fall) -- the stale-pin chain stays
closed structurally.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import cli_agents_help as _ag
from local_operator.i18n.keys import cli_hub_help as _hu

REPO = Path(__file__).resolve().parents[3]
CLI = "local_operator/cli.py"
CEILING = 582  # the count after help tranche 5; may only fall


def _live_cli_count(i18n_check) -> int:
    entries = i18n_check.load_allowlist(i18n_check.ALLOWLIST)
    scanned = i18n_check.scan_tree(i18n_check.REPO)
    return i18n_check._counts(scanned, entries).get(CLI, 0)


def test_every_key_renders_the_byte_its_base_call_carried() -> None:
    for rendered, expected in CASES:
        assert rendered == expected, rendered


def test_the_catalogues_are_the_golden_fixtures() -> None:
    for namespace, fixture, size in (
        ("cli.agents_help", "cli_agents_help_en_golden.json", 21),
        ("cli.hub_help", "cli_hub_help_en_golden.json", 12),
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


def test_the_agents_block_renders_through_the_help_bootstrap() -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    start = source.index("agents_parser = subparsers.add_parser(")
    after = start + len("agents_parser = subparsers.add_parser(")
    nxt = after + source[after:].index(" = subparsers.add_parser(")
    block = source[start:nxt]
    assert block.count("_ag.") == 21, block.count("_ag.")
    assert block.count('.envelope(locale=help_locale)["text"]') == 21
    assert "# i18n: ignore" not in block


def test_the_hub_block_renders_through_the_help_bootstrap() -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    start = source.index("hub_parser = subparsers.add_parser(")
    after = start + len("hub_parser = subparsers.add_parser(")
    nxt = after + source[after:].index(" = subparsers.add_parser(")
    block = source[start:nxt]
    assert block.count("_hu.") == 12, block.count("_hu.")
    assert block.count('.envelope(locale=help_locale)["text"]') == 12
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
    target.write_text(source + '\nprint("planted agents prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


CASES = [
    (
        _ag.agents().envelope(locale="en")["text"],
        "Manage agents",
    ),  # agents
    (
        _ag.list().envelope(locale="en")["text"],
        "List all agents",
    ),  # list
    (
        _ag.list_page().envelope(locale="en")["text"],
        "Page number to display (default: 1)",
    ),  # list_page
    (
        _ag.list_perpage().envelope(locale="en")["text"],
        "Number of agents per page (default: 10)",
    ),  # list_perpage
    (
        _ag.create().envelope(locale="en")["text"],
        "Create a new agent",
    ),  # create
    (
        _ag.create_name().envelope(locale="en")["text"],
        "Name of the agent to create",
    ),  # create_name
    (
        _ag.delete().envelope(locale="en")["text"],
        "Delete an agent (local by name or Radient by ID)",
    ),  # delete
    (
        _ag.delete_name().envelope(locale="en")["text"],
        "Name of the agent to delete locally",
    ),  # delete_name
    (
        _ag.delete_id().envelope(locale="en")["text"],
        "ID of the agent to delete from Radient",
    ),  # delete_id
    (
        _ag.delete_org().envelope(locale="en")["text"],
        (
            "Delete an agent from this organization's workspace instead of the publ"
            "ic hub (Agent Hub org sharing; requires `lop login radient`)"
        ),
    ),  # delete_org
    (
        _ag.push().envelope(locale="en")["text"],
        (
            "Push (upload) an agent to Radient; `--org` and `--hub-id` push through"
            " the preview/confirm flow (exit codes: 0 published, 1 error, 2 preview"
            "-only with changes pending, 3 declined, 130 cancelled)"
        ),
    ),  # push
    (
        _ag.push_name().envelope(locale="en")["text"],
        "Name of the agent to push to Radient",
    ),  # push_name
    (
        _ag.push_id().envelope(locale="en")["text"],
        "ID of the agent to push to Radient (explicit overwrite)",
    ),  # push_id
    (
        _ag.push_org().envelope(locale="en")["text"],
        (
            "Publish into this organization's workspace instead of the public hub ("
            "Agent Hub org sharing; requires `lop login radient`)"
        ),
    ),  # push_org
    (
        _ag.push_hub_id().envelope(locale="en")["text"],
        (
            "Republish (overwrite) this hub listing from the local agent (checks fo"
            "r personal references first); needs --name (--id is a local id)"
        ),
    ),  # push_hub_id
    (
        _ag.pull().envelope(locale="en")["text"],
        "Pull (download) an agent from Radient",
    ),  # pull
    (
        _ag.pull_id().envelope(locale="en")["text"],
        "ID of the agent to pull from Radient",
    ),  # pull_id
    (
        _ag.pull_org().envelope(locale="en")["text"],
        (
            "Pull from this organization's workspace instead of the public hub (Age"
            "nt Hub org sharing; requires `lop login radient`)"
        ),
    ),  # pull_org
    (
        _ag.sync().envelope(locale="en")["text"],
        "Update installed starters and hub-pulled agents to their latest text",
    ),  # sync
    (
        _ag.sync_name().envelope(locale="en")["text"],
        "Sync only the installed profile with this name",
    ),  # sync_name
    (
        _ag.sync_all().envelope(locale="en")["text"],
        "Sync every installed starter and hub-pulled agent (default when --name is absent)",
    ),  # sync_all
    (
        _hu.hub().envelope(locale="en")["text"],
        "Agent Hub update status",
    ),  # hub_hub
    (
        _hu.status().envelope(locale="en")["text"],
        "Show which pulled agents and teams have hub updates",
    ),  # hub_status
    (
        _hu.status_json().envelope(locale="en")["text"],
        "Emit the desktop route payload",
    ),  # hub_status_json
    (
        _hu.links().envelope(locale="en")["text"],
        "List every hub link (healthy or not) with its source, scope and policy",
    ),  # hub_links
    (
        _hu.links_json().envelope(locale="en")["text"],
        "Emit the desktop links payload",
    ),  # hub_links_json
    (
        _hu.link().envelope(locale="en")["text"],
        "Edit one hub link's independent policy",
    ),  # hub_link
    (
        _hu.link_set().envelope(locale="en")["text"],
        "Set this link's subscribe and/or auto_push policy",
    ),  # hub_link_set
    (
        _hu.link_set_link_id().envelope(locale="en")["text"],
        "The link id from `lop hub links`",
    ),  # hub_link_set_link_id
    (
        _hu.link_set_subscribe().envelope(locale="en")["text"],
        "Unattended inbound synchronization for this link",
    ),  # hub_link_set_subscribe
    (
        _hu.link_set_auto_push().envelope(locale="en")["text"],
        "Unattended outbound publication for this link",
    ),  # hub_link_set_auto_push
    (
        _hu.link_set_expected_revision().envelope(locale="en")["text"],
        "The policy revision you last saw (CAS); omitted reads it fresh first",
    ),  # hub_link_set_expected_revision
    (
        _hu.link_set_confirm_public_egress().envelope(locale="en")["text"],
        "Explicit consent to publish authored changes to a PUBLIC origin",
    ),  # hub_link_set_confirm_public_egress
]
