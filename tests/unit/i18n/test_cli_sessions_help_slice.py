"""Help tranche 1: `lop sessions` — the argparse help, extracted to `cli.sessions_help`.

The class decision for help text was ruled once and applies per site (never by
scanner heuristics): **help prose is copy -> keys, with flag and command tokens
staying IN SITU inside the values; deprecated/hidden flags may take
ignore-with-reason.** In this block there are no deprecated or hidden flags, so
there are no ignores: **51 literals across 45 flagged calls -> 47 keys** (45
help= + 6 description=; the five `--json` sites share one key because the text
is identical). The block's `argparse.SUPPRESS` positional carries no literal.

Rendering follows the pattern the exec slice proved: help exits before startup
owns config repair or locale activation, so each site renders through the
read-only bootstrap — `resolve_help_locale()` once, then
`_sh.<key>().envelope(locale=help_locale)["text"]` — never the global renderer,
because an EN-only namespace must not un-ship audited locales.

Fidelity is proven two ways: every key renders to the byte its base call
carried (the table below, anchored to the base file's AST), and the REAL
argparse output — `lop --help` plus `sessions` and each of its five subcommands
— is byte-identical between the base revision and this branch (captured in the
PR evidence; the tracked version of that check lives in the generated-table
test, which CI can run without a subprocess).

**What this does NOT cover:** the other command families (teams, wake, mcp,
mobile, …), the global `parser`/`parent_parser` flags, and the long tail — the
programme's plan-of-record splits those per subparser group; `sessions` is
tranche one.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import cli_sessions_help as _sh

REPO = Path(__file__).resolve().parents[3]
CLI = "local_operator/cli.py"
CEILING = 702  # the count after help tranche 1; may only fall
BLOCK = (
    "sessions_parser",
    "cleanup_parser",
    "reclaim_parser",
    "resume_parser",
    "move_parser",
    "sync_parser",
)


def _live_cli_count(i18n_check) -> int:
    """Derived, not hard-pinned: every later extraction tranche moves cli.py's
    ceiling, and a hard pin turns each tranche into a stale-pin break (round 1
    of this very PR was that failure, twice over). Equality keeps "no slack";
    the bound keeps "may only fall"."""
    entries = i18n_check.load_allowlist(i18n_check.ALLOWLIST)
    scanned = i18n_check.scan_tree(i18n_check.REPO)
    return i18n_check._counts(scanned, entries).get(CLI, 0)


def test_every_key_renders_the_byte_its_base_call_carried() -> None:
    """47/47 against values taken from the base file's AST (the table below)."""
    for rendered, expected in CASES:
        assert rendered == expected, rendered


def test_the_catalogue_is_the_golden_fixture() -> None:
    live = catalogues.load_catalogue("en", "cli.sessions_help")
    golden = json.loads(
        (REPO / "tests/unit/i18n/fixtures/cli_sessions_help_en_golden.json").read_text(
            encoding="utf-8"
        )
    )
    assert live == golden, "the shipped catalogue and the fixture golden drifted"
    assert len(golden) == 47, len(golden)
    sidecar = catalogues.context("en", "cli.sessions_help")
    for key, entry in sidecar.items():
        assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_the_sessions_block_renders_through_the_help_bootstrap() -> None:
    """The block's 45 calls + 6 descriptions all use the envelope pattern; and
    no site inside the block carries an ignore (there is none to carry)."""
    source = (REPO / CLI).read_text(encoding="utf-8")
    start = source.index("sessions_parser = subparsers.add_parser(")
    # end at the NEXT top-level section (resume-click), NOT at a named later
    # parser: any block extracted between `sessions` and that name would land
    # inside this span and inflate the count — the exact stale-span class this
    # file's own tranche remediated elsewhere (help tranche 3 hit it, 50->69).
    after = start + len("sessions_parser = subparsers.add_parser(")
    nxt = after + source[after:].index(" = subparsers.add_parser(")
    block = source[start:nxt]
    calls = block.count("_sh.")
    assert calls == 51, calls
    assert block.count('.envelope(locale=help_locale)["text"]') == 51
    assert "# i18n: ignore" not in block
    assert "_sh." in block


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
    target.write_text(source + '\nprint("planted help prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


CASES = [
    (
        _sh.sessions().envelope(locale="en")["text"],
        (
            "List active lop sessions and their resource usage; `sessions resume` reopens"
            " a set of stored sessions (paused/failed/all) in one bounded batch; `session"
            "s cleanup` previews or runs the session cleanup policy; `sessions reclaim` p"
            "reviews or ends runtimes nothing can reach"
        ),
    ),  # sessions
    (
        _sh.sessions_description().envelope(locale="en")["text"],
        (
            "List active lop sessions and their resource usage; `sessions resume` reopens"
            " a set of stored sessions in one command; `sessions cleanup` previews or run"
            "s the session cleanup policy; `sessions reclaim` previews or ends runtimes n"
            "othing can reach."
        ),
    ),  # sessions_description
    (_sh.json().envelope(locale="en")["text"], "machine-readable output"),  # json
    (
        _sh.all().envelope(locale="en")["text"],
        (
            "also list stored sessions that are not running (RSS/UPTIME/etc. are shown as"
            " — for these; sorted newest-first by last activity)"
        ),
    ),  # all
    (
        _sh.limit().envelope(locale="en")["text"],
        "with --all (or a set filter), cap the stored rows listed; positive (default: 50)",
    ),  # limit
    (
        _sh.paused().envelope(locale="en")["text"],
        (
            "list only stored sessions whose last turn was STOPPED rather than completed "
            "or lost — a deliberate stop or a build retirement (the set the bulk-resume p"
            "ath selects; implies --all)"
        ),
    ),  # paused
    (
        _sh.failed().envelope(locale="en")["text"],
        (
            "list only stored sessions whose last recorded outcome was a failure (an invo"
            "luntary death with no deliberate-stop evidence; implies --all). A death whos"
            "e outcome was never imported is in neither set"
        ),
    ),  # failed
    (
        _sh.peer().envelope(locale="en")["text"],
        "list the sessions held by one peer device (its own catalogue, over the mesh)",
    ),  # peer
    (
        _sh.all_peers().envelope(locale="en")["text"],
        "also list the sessions other devices in this network are holding",
    ),  # all_peers
    (
        _sh.cleanup().envelope(locale="en")["text"],
        "Run the session cleanup policy now (use --dry-run to preview)",
    ),  # cleanup
    (
        _sh.cleanup_description().envelope(locale="en")["text"],
        (
            "Apply session.cleanup.* from config (each limit overridable below) to the se"
            "ssion store, in two classes: your conversations (needs session.cleanup.enabl"
            "ed: true unless --force) and delegated subagent/background sessions (session"
            ".cleanup.delegated.*, on by default). Every removal is recorded in <config>/"
            "sessions/.cleanup-log.jsonl."
        ),
    ),  # cleanup_description
    (
        _sh.cleanup_scratchpads_only().envelope(locale="en")["text"],
        (
            "Audit pads with --dry-run; otherwise reclaim only scratchpad build/cache "
            "content (never transcripts, never your conversations)."
        ),
    ),  # cleanup_scratchpads_only
    (
        _sh.cleanup_dry_run().envelope(locale="en")["text"],
        "list what would be removed and why, without removing anything",
    ),  # cleanup_dry_run
    (
        _sh.cleanup_force().envelope(locale="en")["text"],
        "run even when session.cleanup.enabled is false (lists first, then asks)",
    ),  # cleanup_force
    (
        _sh.cleanup_yes().envelope(locale="en")["text"],
        "do not ask for confirmation before removing",
    ),  # cleanup_yes
    (
        _sh.cleanup_max_sessions().envelope(locale="en")["text"],
        "keep only the N most recently active sessions (overrides config)",
    ),  # cleanup_max_sessions
    (
        _sh.cleanup_max_inactive_days().envelope(locale="en")["text"],
        "remove sessions idle longer than N days (overrides config)",
    ),  # cleanup_max_inactive_days
    (
        _sh.cleanup_max_total_bytes().envelope(locale="en")["text"],
        "trim the least recently active sessions past this store size (overrides config)",
    ),  # cleanup_max_total_bytes
    (
        _sh.cleanup_remove_empty().envelope(locale="en")["text"],
        "remove directories that never got a transcript (overrides config)",
    ),  # cleanup_remove_empty
    (
        _sh.cleanup_delegated_max_age_hours().envelope(locale="en")["text"],
        "remove subagent/background sessions idle over N hours, 2-720 (overrides config)",
    ),  # cleanup_delegated_max_age_hours
    (
        _sh.reclaim().envelope(locale="en")["text"],
        "End session runtimes nothing can reach (dry run; --yes to act)",
    ),  # reclaim
    (
        _sh.reclaim_description().envelope(locale="en")["text"],
        (
            "Find live session runtimes that no discovery record, no viewer, no attach an"
            "d no existing config root can reach, and ask them to leave with SIGTERM. The"
            " runtime finishes any turn in flight first (its own signal drain, bounded by"
            " SIGNAL_DRAIN_S) — this command never sends SIGKILL. Any runtime with a reco"
            "rd, an attached interface, a live root it does not own, or CPU spent inside "
            "the confirm window is refused."
        ),
    ),  # reclaim_description
    (
        _sh.reclaim_dry_run().envelope(locale="en")["text"],
        "list what would be reclaimed, without signalling anything",
    ),  # reclaim_dry_run
    (
        _sh.reclaim_yes().envelope(locale="en")["text"],
        "do not ask for confirmation before signalling",
    ),  # reclaim_yes
    (
        _sh.reclaim_confirm_s().envelope(locale="en")["text"],
        (
            "how long to watch before signalling (default: 60, minimum: 50). The window i"
            "s the safety property: a runtime that gains a record, an attach or CPU insid"
            "e it is dropped from the pass, and a window too short to measure CPU would d"
            "rop one of the four refusals"
        ),
    ),  # reclaim_confirm_s
    (
        _sh.resume().envelope(locale="en")["text"],
        "Resume a set of stored sessions in one command (bulk)",
    ),  # resume
    (
        _sh.resume_description().envelope(locale="en")["text"],
        (
            "Resume stored sessions without a shell loop: `--paused` (the interrupted/ret"
            "ired pair), `--failed` (error), or `--all` (every stored, non-live session —"
            " the widest reading, so given with the other selectors it wins). Newest firs"
            "t, capped at 20 unless --limit is given. Each session is reopened headlessly"
            " as `lop exec --resume <id> --background` with --message (default: a continu"
            "ation). Children run with bounded concurrency (6 at a time) and print one li"
            "ne per session as it resolves. Exit codes: 0 = every selected session resume"
            "d; 1 = at least one failed or was unresolved at the bound; 2 = misuse (no se"
            "t selected)."
        ),
    ),  # resume_description
    (
        _sh.resume_paused().envelope(locale="en")["text"],
        (
            "select stored sessions whose last outcome was interrupted/retired (the 'Unse"
            "en interruption' pair; implies the store scope)"
        ),
    ),  # resume_paused
    (
        _sh.resume_failed().envelope(locale="en")["text"],
        "select stored sessions whose last outcome was error (implies the store scope)",
    ),  # resume_failed
    (
        _sh.resume_all().envelope(locale="en")["text"],
        (
            "select every stored, non-live session (newest first, capped); the widest rea"
            "ding — given with --paused/--failed it wins"
        ),
    ),  # resume_all
    (
        _sh.resume_dry_run().envelope(locale="en")["text"],
        "print the selected set (newest first) and resume nothing",
    ),  # resume_dry_run
    (
        _sh.resume_limit().envelope(locale="en")["text"],
        "cap the set at the N newest matches (default: 20)",
    ),  # resume_limit
    (
        _sh.resume_message().envelope(locale="en")["text"],
        (
            "the message each resumed session runs (default: a continuation — 'Continue t"
            "he task from where it left off.')"
        ),
    ),  # resume_message
    (
        _sh.move().envelope(locale="en")["text"],
        "Move a conversation to another device, or bring one home",
    ),  # move
    (
        _sh.move_description().envelope(locale="en")["text"],
        (
            "Hand a conversation to another device (`--to build-box`), or bring one that "
            "lives elsewhere back here (`--to local`). The conversation keeps its id and "
            "the copy on the device it left is deleted, unless `--keep` is given, which l"
            "eaves the original running and mints a NEW id for the copy. A session with a"
            " turn in flight is refused, not interrupted."
        ),
    ),  # move_description
    (
        _sh.move_session().envelope(locale="en")["text"],
        "the conversation id to move (with --cancel-queued, the id is that flag's value)",
    ),  # move_session
    (
        _sh.move_to().envelope(locale="en")["text"],
        "a peer device (id or name), or `local` to bring the conversation here",
    ),  # move_to
    (
        _sh.move_keep().envelope(locale="en")["text"],
        "copy and leave the original running; the copy gets a new id and a fork marker",
    ),  # move_keep
    (
        _sh.move_wait().envelope(locale="en")["text"],
        "re-check a busy conversation every 5s, up to SECONDS (bare: up to 30 min)",
    ),  # move_wait
    (
        _sh.move_queue().envelope(locale="en")["text"],
        (
            "wait for the next safe point instead of refusing a busy conversation: the mo"
            "ve is queued durably on the conversation's device and runs when its turn end"
            "s (attached clients there are announced and given a window to follow or deta"
            "ch). Ignored with --keep"
        ),
    ),  # move_queue
    (
        _sh.move_cancel_queued().envelope(locale="en")["text"],
        (
            "cancel a queued move that has not started yet; takes the conversation id and"
            " nothing else"
        ),
    ),  # move_cancel_queued
    (
        _sh.move_from_replica().envelope(locale="en")["text"],
        (
            "recover the last copy this device synced as a NEW session, for when the devi"
            "ce that held it is gone"
        ),
    ),  # move_from_replica
    (
        _sh.move_engage_on_arrival().envelope(locale="en")["text"],
        (
            "start the conversation on the destination as soon as it lands, so it is live"
            " there without a first prompt; a conversation whose turn is in flight is sti"
            "ll refused rather than drained, and a failure to start it does not affect th"
            "e move — so the exit status is the MOVE's: a script reads `engagement.engage"
            "d` in --json, and on a move to a peer that device's own log is where the out"
            "come shows"
        ),
    ),  # move_engage_on_arrival
    (
        _sh.sync().envelope(locale="en")["text"],
        "Pull a copy of a conversation another device owns",
    ),  # sync
    (
        _sh.sync_description().envelope(locale="en")["text"],
        (
            "Bring this device's copy of a conversation it does not own up to date. The c"
            "opy is stored outside the session store and is never opened as a session; `l"
            "op sessions move <id> --to local --from-replica` recovers it as a new one."
        ),
    ),  # sync_description
    (
        _sh.sync_session().envelope(locale="en")["text"],
        "the conversation id to pull",
    ),  # sync_session
    (
        _sh.sync_owner().envelope(locale="en")["text"],
        "the peer that holds it (default: ask)",
    ),  # sync_owner
]
