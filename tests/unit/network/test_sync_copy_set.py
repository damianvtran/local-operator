"""WHAT A SESSION DIRECTORY MAY HOLD, enumerated from the modules that write it.

THE TEST THE LAST DATA-LOSS BUG ARGUED FOR (review round 1, B-M2). ``scratchpad/``
and ``created_at.json`` were in neither the copy set nor the exclusion list, so a
move did not carry them and then deleted the source directory they were in: on the
operator's workstation that is 3,017 scratchpads and 10,840 birth-time sidecars, and
neither was recoverable. A copy set written as a list of files cannot notice a file
type nobody told it about, so the two guards here invert the question:

* :func:`test_every_path_shaped_constant_is_classified_or_declared` DERIVES the
  candidate set from the source (every module-level path-shaped name constant under
  ``local_operator/``, as ``test_reason_surfaces.py`` derives its renderers), so a new
  sidecar lands in this test the moment its CONSTANT exists, wherever it is declared. The
  claim that the earlier version did this was false — it imported a hand-written list, and
  the reviewer's new constant in ``resume.py`` left it green (review round 2, MAJOR 1);
* :func:`test_every_entry_type_this_product_writes_is_copied_or_excluded` IMPORTS
  each name from the module that owns it, so a TYPO in a literal is caught, and the
  failure says "add it to one list or the other, with a reason";
* :func:`test_a_directory_of_every_entry_type_round_trips` builds a directory
  holding ONE of every such entry and copies it for real, then asserts every
  classified entry either arrived byte-identical or did NOT arrive and is named in
  ``EXCLUDED_ENTRIES`` — which is what makes ``scratchpad/`` (a tree) and
  ``created_at.json`` (a file) both covered by one property rather than by two
  hand-written assertions;
* and ``sync.assert_complete`` is what makes the answer FAIL CLOSED in the product
  when the copy set is behind: a move refuses rather than deleting what it did not
  copy, so the next unlisted entry type is a refusal with a sentence, not another
  silent loss.

NOTHING HERE IS A GUESS ABOUT THE MISSING NAMES: the list below is measured, and
the comment on each entry says where the measurement comes from.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import sync

# ---------------------------------------------------------------------------
# The entry types, imported from their owners
# ---------------------------------------------------------------------------


def _entry_owners() -> dict[str, str]:
    """``name -> the module that owns it`` for every entry a session may hold.

    Imported rather than spelled, deliberately: this is the list that has to grow
    when the product grows, and an import makes that automatic where a literal
    would not. The trees are included as directory names.
    """
    from local_operator.browser_bridge.resources import RESOURCE_NAME
    from local_operator.fork import FORK_BOUNDARY_NAME
    from local_operator.resume import (
        ATTACHMENT_SIDECAR_NAME,
        ORIGIN_CACHE_NAME,
        ORIGIN_NAME,
        TITLE_SIDECAR_NAME,
    )
    from local_operator.scratchpad import SCRATCHPAD_DIRNAME
    from local_operator.session.creation import CREATED_AT_NAME
    from local_operator.session.placement import MESH_STAMP_NAME
    from local_operator.session.retention import (
        ATTACHMENT_SIDECAR_FILENAME,
        DESKTOP_MARKER_NAME,
        LIVE_MARKER_NAME,
        TRANSCRIPT_FILENAME,
    )
    from local_operator.session.runtime.inbox import INBOX_NAME
    from local_operator.session.runtime.registry import (
        STOP_MARKER_NAME,
        TURN_JOURNAL_NAME,
    )
    from local_operator.session_lease import LEASE_NAME, MIRROR_NAME, RECOVERY_LOCK_NAME
    from local_operator.wakes.lock import WAKE_LOCK_NAME

    return {
        TRANSCRIPT_FILENAME: "session.transcript",
        TITLE_SIDECAR_NAME: "resume",
        ATTACHMENT_SIDECAR_NAME: "resume",
        ATTACHMENT_SIDECAR_FILENAME: "resume",
        ORIGIN_NAME: "resume",
        ORIGIN_CACHE_NAME: "resume",
        CREATED_AT_NAME: "session.creation",
        TURN_JOURNAL_NAME: "session.runtime.registry",
        STOP_MARKER_NAME: "session.runtime.registry",
        INBOX_NAME: "session.runtime.inbox",
        FORK_BOUNDARY_NAME: "fork",
        DESKTOP_MARKER_NAME: "session.retention",
        MESH_STAMP_NAME: "session.placement",
        LIVE_MARKER_NAME: "session.retention",
        LEASE_NAME: "session_lease",
        MIRROR_NAME: "session_lease",
        RECOVERY_LOCK_NAME: "session_lease",
        WAKE_LOCK_NAME: "wakes.lock",
        RESOURCE_NAME: "browser_bridge.resources",
        # The copy machinery's own two: a replica's cursor (never inside a session)
        # and the move's boot marker, which lives in a staging directory and is
        # deleted by the promote (a crash can leave one inside a promoted session,
        # which is why it is classified rather than ignored).
        sync.REPLICA_CURSOR_NAME: "network.sync",
        "ready.json": "network.mobility",
        # THE TREES. A directory, so it is classified as one.
        SCRATCHPAD_DIRNAME: "scratchpad",
    }


#: The names MEASURED in the operator's live store on 2026-09-24 (10,841 session
#: directories, ``ls | sort | uniq -c``, counts in the report). Kept separate from
#: the imported list because it is the EVIDENCE that the imported list is complete:
#: a name here that no constant owns is a writer nobody has found yet, and the test
#: below fails rather than letting it be dropped.
MEASURED_IN_THE_STORE: tuple[str, ...] = (
    "created_at.json",
    "transcript.jsonl",
    "title-scan.json",
    "origin.json",
    "scratchpad",
    ".browser-resource.json",
    "origin-scan.json",
    "title.json",
    "subagent-roster.v1.json",
    "attachment.json",
    "desktop.json",
    ".session.pid",
    "turn-journal.json",
    "inbox.jsonl",
    ".execution-lease.recovery",
    ".execution-lease",
    "runtime-stop.json",
    ".wake-write.lock",
    "mesh.json",
)


def _classified(name: str) -> bool:
    return name in sync.COPY_SET_NAMES or name in sync.COPY_SET_TREES or name in sync.NEVER_COPIED


# ---------------------------------------------------------------------------
# The two lists together have to cover everything
# ---------------------------------------------------------------------------


def test_every_entry_type_this_product_writes_is_copied_or_excluded() -> None:
    """Every entry type is copied, or excluded WITH ITS REASON. No third answer."""
    unclassified = sorted(name for name in _entry_owners() if not _classified(name))
    assert unclassified == [], (
        f"a session directory can hold {unclassified}, and the copy set neither "
        "carries it nor excludes it: a move would delete it. Add each name to "
        "sync.COPY_SET_NAMES (it travels), sync.COPY_SET_TREES (its files travel) "
        "or sync.EXCLUDED_ENTRIES (with the reason it must not)"
    )
    # Every excluded name states why, and the two exclusion views agree.
    assert set(sync.EXCLUDED_ENTRIES) == set(sync.NEVER_COPIED)
    for name in sorted(sync.NEVER_COPIED):
        assert sync.EXCLUDED_ENTRIES[name].strip(), f"{name} is excluded with no reason"
    # A name cannot be in two answers at once.
    assert not (set(sync.COPY_SET_NAMES) & set(sync.NEVER_COPIED))
    assert not (set(sync.COPY_SET_TREES) & set(sync.NEVER_COPIED))


def test_the_measured_store_entries_are_all_classified() -> None:
    """Every name the operator's real store holds is classified by this build.

    THE EVIDENCE HALF. The imported list above is only as complete as the constants
    it can find; this one is a census of the machine the feature runs on, so a
    writer that spells its name inline instead of exporting a constant still shows
    up instead of being lost in a move.
    """
    unclassified = sorted(name for name in MEASURED_IN_THE_STORE if not _classified(name))
    assert unclassified == [], (
        f"the live store holds {unclassified}, which this build's copy set does not "
        "classify; measure the writer and add it to sync.COPY_SET_NAMES, "
        "sync.COPY_SET_TREES or sync.EXCLUDED_ENTRIES"
    )


# ---------------------------------------------------------------------------
# The DERIVED half: every path-shaped name constant in the product
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS, AND WHY THE LIST ABOVE IS NOT ENOUGH. The guard this replaces
# imported a HAND-WRITTEN list of constants and asserted each was classified. It read as
# automatic ("a new sidecar lands in this test the moment its constant exists") and was
# not: the reviewer added ``REVIEW_PROBE_SIDECAR_NAME = "review-probe.json"`` to
# ``resume.py`` and the test stayed green, because a list of imports cannot see a
# constant nobody added to it. ``origin/main`` then did exactly that with
# ``GOAL_SIDECAR_NAME``, and every session with a judged goal became unmovable while the
# guard passed (review round 2, MAJOR 1).
#
# SO THE SET IS DERIVED FROM THE SOURCE, the way ``test_reason_surfaces.py`` derives the
# renderers it guards: every module-level string constant under ``local_operator/`` whose
# NAME ends in a path-shaped suffix (``NAME``, ``FILENAME``, ``DIRNAME``, ``FILE``,
# ``DIR``, ``SIDECAR``, ``MARKER``, ``RECORD``, ``STEM``, ``PATH``, ``BASENAME``) and whose
# VALUE is a relative path of one or more segments is a candidate. Each one must either be
# classified by the copy set or be DECLARED below, per module, WITH THE COUNT. A new
# constant anywhere in the product therefore fails this test until somebody classifies
# it or writes down why that module's paths can never be an entry of a session directory.
#
# WHAT THE DERIVATION CANNOT DO, STATED PLAINLY. It sees constants, not files: a name
# built at runtime (an f-string, a ``.with_suffix``, a concatenation) is invisible to it,
# and so is a name spelled inline at its write site. It also cannot tell a file from an
# identifier — ``ACTION_TOOL_NAME = "apply_actions"`` is a tool's name, not a path — which
# is why the declarations exist rather than a longer analysis. And it is a SPELLING
# rule: a constant it does not recognise by name, or whose value is not a relative path,
# is not a candidate at all. Review round 3 (MINOR 1) measured two spellings that escaped
# the round-2 lists — ``REVIEW_PROBE_MARKER = "judge.json"`` (a suffix the name list did
# not carry) and ``REVIEW_PROBE_NESTED_PATH = "judge/index.json"`` (a value with a
# separator in it) — and both are candidates now, with a cell each. A spelling OUTSIDE
# this list is still invisible, which is why the hand-written classification list and the
# store census stay BESIDE this derivation rather than being replaced by it. What it DOES
# cover is the exact failure that happened: a new constant with a literal value in a
# module that owns session state.
_PATH_SUFFIX = re.compile(
    r"(NAME|FILENAME|DIRNAME|FILE|DIR|SIDECAR|MARKER|RECORD|STEM|PATH|BASENAME)$"
)
_UPPER_NAME = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
#: A path-shaped VALUE: one or more RELATIVE segments — no leading ``/``, and no segment
#: that is ``.`` or ``..``. One segment was the round-2 rule; widening it is what catches
#: ``judge/index.json``. The product fails closed on such a name anyway (an unknown entry
#: at a session root is refused before a byte is copied), so this was a gap in the guard's
#: COVERAGE rather than in behaviour — and a guard with a known spelling that escapes it is
#: a guard that will be trusted for exactly the case it misses.
_PATH_SHAPED = re.compile(r"^(?!\.\.?(?:/|$))[A-Za-z0-9._-]+(?:/(?!\.\.?(?:/|$))[A-Za-z0-9._-]+)*$")

#: ``module -> (how many of its constants are NOT classified, why none of them is an entry
#: of a session directory)``. The count is part of the declaration on purpose: a new
#: constant in a declared module changes it, so the module cannot become a blind surface
#: the way the import list was. Every module with at least one candidate appears here, so
#: "a module I did not think about" is itself a failing state.
_DERIVED_DECLARATIONS: dict[str, tuple[int, str]] = {
    # ---- this install's CONFIG, LOG and STORE surface (outside ``sessions/``) ----
    "local_operator/paths.py": (
        5,
        "the store's own roots and log file (``~/.local-operator``, ``logs``, "
        "``runtime.log``, the agent home): the directories a session lives UNDER",
    ),
    "local_operator/config.py": (1, "``config.yml``, the store's configuration file"),
    "local_operator/config_watch.py": (1, "the same file, watched"),
    "local_operator/config_migrations.py": (1, "the marker recording which migrations ran"),
    "local_operator/logger.py": (1, "the process log at the store root"),
    "local_operator/update.py": (
        2,
        "the distribution name and the PyPI version cache: names in the update "
        "channel, never a file in a session",
    ),
    "local_operator/harness/reply_channel.py": (1, "the structured-reply TOOL's name"),
    "local_operator/evaluation/runner/action_tool.py": (1, "the action tool's name"),
    "local_operator/agents.py": (
        1,
        "``_DEFAULT_EXPORT_STEM``: the stem an ``lop agents export`` file is written "
        "under, chosen by that command rather than by a session",
    ),
    "local_operator/info/collect.py": (
        1,
        "``_ENV_CMUX_SOCKET_PATH``: an ENVIRONMENT VARIABLE's name — the identifier-not-a-"
        "path case this file's own docstring names, which a value-shaped rule cannot tell "
        "from a filename",
    ),
    "local_operator/references.py": (1, "a THREAD name for reference reads"),
    "local_operator/providers/oauth/zai.py": (1, "a keychain entry's name, not a path"),
    "local_operator/providers/oauth/kimi.py": (
        1,
        "the device-id file under the agent home, beside the provider's own config",
    ),
    "local_operator/providers/qwencloud_console.py": (1, "a SECRET's name"),
    "local_operator/mobile/auth.py": (1, "the mobile session COOKIE's name"),
    "local_operator/mobile/seen.py": (1, "the mobile seen-store under the store root"),
    "local_operator/secrets/keys.py": (
        2,
        "``secrets/`` and its registration ticket: the credential store at the config root",
    ),
    "local_operator/secrets/legacy_env.py": (1, "the legacy credentials file at the store root"),
    "local_operator/secrets/protocol.py": (2, "the secret broker's socket and lock"),
    "local_operator/session/cleanup.py": (
        3,
        "the cleanup log and last-cleanup record (sessions root) and the store marker "
        "that gates removal: none of them is a file a session directory holds",
    ),
    "local_operator/session_factory.py": (
        2,
        "the store-maintenance lock and stamp at the config root",
    ),
    "local_operator/session/archived.py": (1, "the archived-ids list, a store-level record"),
    "local_operator/session/search_index.py": (1, "the search index, a store-level record"),
    "local_operator/session/retention.py": (
        1,
        "``sessions`` itself: the directory the entries live IN",
    ),
    "local_operator/session/attachments.py": (
        1,
        "``attachments``: the content-addressed store at the config root. A copy set "
        "carries the blobs a transcript REFERENCES; a directory of this name inside a "
        "session is refused (``sync.unlisted_entries``), because staging it would land "
        "content the store never reads",
    ),
    "local_operator/network/store.py": (
        5,
        "the mesh's own store under ``<config>/network``: catalogue, audit, outbox, "
        "pending and the networks directory",
    ),
    "local_operator/network/projection.py": (1, "the tombstone list under ``network/``"),
    "local_operator/network/audit.py": (
        1,
        "``_ROTATION_MARKER`` (``.gz``): the suffix a rotated audit file carries under "
        "``network/audit`` — a store file, never a session entry",
    ),
    "local_operator/network/types.py": (
        1,
        "``PEERS_RUN_DIRNAME`` (``run/peers``): the mesh's run directory under the store root",
    ),
    "local_operator/network/sync.py": (
        4,
        "``network``, ``replicas`` and ``staging`` under the config root, plus "
        "``attachments`` (see above): the sync plane's own directories",
    ),
    "local_operator/network/credentials/placement.py": (
        3,
        "``placement.json``, ``placement.state.json`` and ``.placement.lock`` under "
        "``network/credentials/<network_id>/``: the broker's per-network record of who owns "
        "a credential and who may borrow it — a store file beside ``sessions/``, never an "
        "entry a session directory holds",
    ),
    "local_operator/session/placement.py": (
        2,
        "``network`` again and the handoff journal inside it: both outside ``sessions/``",
    ),
    "local_operator/session/runtime/presence.py": (
        3,
        "the delivery record and its directory under ``run/desktop``: per-device runtime "
        "machinery",
    ),
    "local_operator/session/runtime/registry.py": (1, "``reaped`` under the runtime's run dir"),
    "local_operator/session/runtime/types.py": (
        3,
        "``run``'s per-runtime subdirectories (``run/mobile``, ``run/host``, ``run/serve``): "
        "the runtime plane's own run directories under the store root",
    ),
    "local_operator/session/runtime/viewers.py": (
        1,
        "``run/viewers``: where a viewer's attachment record lands, under the store root",
    ),
    "local_operator/tools/group_reaper.py": (1, "``proc-groups`` under the store root"),
    "local_operator/tools/spill.py": (1, "``spill``, where elided tool output is parked"),
    "local_operator/tools/builtin.py": (1, "the ripgrep excludes file under the agent home"),
    "local_operator/wakes/store.py": (1, "``wakes`` under the store root"),
    "local_operator/wakes/deliveries.py": (1, "``deliveries`` under ``wakes/``"),
    "local_operator/wakes/spooled.py": (1, "``spooled`` under ``wakes/``"),
    "local_operator/web_fetch/service.py": (1, "the fetched-page cache under the store root"),
    "local_operator/exec_mode.py": (1, "the exec job journal under the store root"),
    "local_operator/mcp/tool_cache.py": (1, "the MCP tool cache database"),
    # ---- the BROWSER surfaces (their own run directories) ----
    "local_operator/browser_bridge/state.py": (
        2,
        "``run/browser`` and its bridge record ``bridge.json``",
    ),
    "local_operator/browser_bridge/gen_ts.py": (1, "the generated protocol bundle's name"),
    "local_operator/browser_bridge/daemon.py": (
        2,
        "the daemon's pairing record (``browser/pairing.json``) and the pending file beside "
        "it (``run/browser/pairing-pending.json``): both under the store root",
    ),
    "local_operator/browser_bridge/install.py": (
        1,
        "the default config dirname, as ``paths`` has it",
    ),
    "local_operator/browser_files.py": (
        3,
        "the browser download audit log, the downloads directory beside it, and the "
        "fallback filename stem for a download with no name of its own",
    ),
    "local_operator/ui_browser/state.py": (
        2,
        "``run/ui-browser`` and its host record ``host.json``",
    ),
    # ---- the desktop app's own feeds and the TUI's per-user state ----
    "local_operator/server/utils/desktop_feed.py": (
        2,
        "the agent and team directories the desktop app reads: the profile store",
    ),
    "local_operator/tui/sidebar_pins.py": (1, "the sidebar's pinned-session list"),
    "local_operator/tui/move_targets.py": (1, "the move dialog's recent-targets list"),
    "local_operator/tui/resume_click.py": (1, "the desktop app's bundle/binary name"),
    "local_operator/tui/input_decode.py": (
        1,
        "``_MARKER``: a sentinel STRING embedded in a wrapper module's source, not a path",
    ),
    "local_operator/tui/terminal_modes.py": (1, "``_GATE_MARKER``: a terminal-mode latch's name"),
    "local_operator/tui/notifier_app/__init__.py": (
        1,
        "``_MARKER`` (``.build-stamp``): the notifier app's own build stamp, written into its "
        "package directory",
    ),
    # ---- an adapter's own scratch ----
    "local_operator/evaluation/adapters/supervisor.py": (
        1,
        "an evaluation supervisor's rescue file",
    ),
}


def _derived_constants(source: str) -> dict[str, str]:
    """``constant -> value`` for every module-level path-shaped name constant in ``source``.

    Deliberately syntactic: a module-level assignment of a STRING LITERAL. Aliases
    (``X = OTHER_NAME``) are not followed, because a name that is another name changes
    nothing about whether the value is a session entry — the constant it points at is
    itself a candidate wherever it is defined.
    """
    found: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if not _UPPER_NAME.match(target.id) or not _PATH_SUFFIX.search(target.id):
                continue
            if _PATH_SHAPED.match(node.value.value):
                found[target.id] = node.value.value
    return found


def _unclassified_in_module(source: str, module: str) -> list[str]:
    """Names in ``source`` that are neither classified nor declared for ``module``.

    The whole rule, in one place, so the test below and the probe that mutates a source
    string exercise the SAME code rather than two spellings of it.
    """
    declared = _DERIVED_DECLARATIONS.get(module)
    if declared and declared[0] == 0:
        declared = None
    out: list[str] = []
    for name, value in sorted(_derived_constants(source).items()):
        if _classified(value):
            continue
        if declared:
            continue
        out.append(f"{name}={value!r}")
    return out


def _unclassified_everywhere() -> dict[str, list[str]]:
    """``module -> unclassified constant spellings`` for the whole product."""
    root = Path(__file__).resolve().parents[3]
    found: dict[str, list[str]] = {}
    for path in sorted(root.joinpath("local_operator").rglob("*.py")):
        module = str(path.relative_to(root))
        names = _unclassified_in_module(path.read_text(encoding="utf-8"), module)
        if names:
            found[module] = names
    return found


def test_every_path_shaped_constant_is_classified_or_declared() -> None:
    """THE GUARD WITH TEETH: derived from the source, per module, with counts.

    A constant added ANYWHERE in the product makes this fail until it is classified (it
    is an entry of a session directory, so the copy set has to carry it or exclude it with
    a reason) or declared (its module's paths live somewhere else, and the count says how
    many are accounted for). The count is what makes a declaration a statement rather than
    a hole: adding a second constant to a declared module fails until the row is updated,
    and the reason for the new one has to be written down with it.
    """
    offenders: list[str] = []
    for module, names in sorted(_unclassified_everywhere().items()):
        if module in _DERIVED_DECLARATIONS:
            continue
        offenders.append(f"{module}: {', '.join(names)}")
    assert offenders == [], (
        "these modules hold path-shaped constants the copy set does not classify and no "
        "declaration accounts for:\n  "
        + "\n  ".join(offenders)
        + "\n\nAdd each name to sync.COPY_SET_NAMES (it travels with a session), "
        "sync.COPY_SET_TREES (its files travel) or sync.EXCLUDED_ENTRIES (with the reason "
        "it must not), or add the module to _DERIVED_DECLARATIONS with the count and the "
        "reason none of its constants is an entry of a session directory."
    )
    # AND THE OTHER DIRECTION: a declaration whose count has drifted, or which describes a
    # module that no longer has candidates, is a stale row rather than a statement.
    stale: list[str] = []
    for module, (count, reason) in sorted(_DERIVED_DECLARATIONS.items()):
        assert reason.strip(), module
        path = Path(__file__).resolve().parents[3] / module
        if not path.is_file():
            stale.append(f"{module} (no such module)")
            continue
        names = _unclassified_in_module(path.read_text(encoding="utf-8"), "")
        if len(names) != count:
            stale.append(f"{module} (declares {count}, holds {len(names)}: {', '.join(names)})")
    assert stale == [], f"declarations that no longer describe the source: {stale}"


def test_the_derivation_reports_a_new_constant_in_an_owning_module() -> None:
    """THE MUTATION THE OLD GUARD MISSED, run as a test rather than by hand.

    This is the reviewer's probe verbatim: a brand-new constant appended to the module
    that owns ``goal.json``. The old guard stayed green (7 passed) because it imported a
    hand-written list; the derivation above reports it, and the test asserts BOTH halves —
    that the new name is reported, and that it stops being reported once it is classified.
    A guard whose teeth are only demonstrated in a review round is a guard that will lose
    them again.
    """
    module = "local_operator/resume.py"
    pristine = (Path(__file__).resolve().parents[3] / module).read_text(encoding="utf-8")
    assert _unclassified_in_module(pristine, module) == [], "the real module is already clean"

    mutated = pristine + '\nREVIEW_PROBE_SIDECAR_NAME = "review-probe.json"\n'
    assert _unclassified_in_module(mutated, module) == [
        "REVIEW_PROBE_SIDECAR_NAME='review-probe.json'"
    ]
    # And a constant that IS classified is not noise on this path: the same shape, in the
    # same module, pointing at a name the copy set already carries.
    classified = pristine + '\nREVIEW_PROBE_ALIAS_NAME = "goal.json"\n'
    assert _unclassified_in_module(classified, module) == []


def test_the_derivation_sees_the_spellings_round_3_measured_escaping() -> None:
    """TWO SPELLINGS THE ROUND-2 GUARD LET THROUGH, as cells rather than as a review note.

    MEASURED (review round 3, MINOR 1): ``REVIEW_PROBE_MARKER = "judge.json"`` and
    ``REVIEW_PROBE_NESTED_PATH = "judge/index.json"`` both left the derived guard green (3
    passed) — the first because the name suffix was not in the list, the second because the
    value had to be a single segment. Nothing was lost (the product refuses an unknown
    ``judge.json`` at a session root before a byte is copied), but MAJOR 1's failure mode was
    a session that could not be MOVED, and a constant using a spelling outside the list
    arrives there with the guard quiet. Both are reported now.
    """
    module = "local_operator/resume.py"
    pristine = (Path(__file__).resolve().parents[3] / module).read_text(encoding="utf-8")

    marker = pristine + '\nREVIEW_PROBE_MARKER = "judge.json"\n'
    assert _unclassified_in_module(marker, module) == ["REVIEW_PROBE_MARKER='judge.json'"]

    nested = pristine + '\nREVIEW_PROBE_NESTED_PATH = "judge/index.json"\n'
    assert _unclassified_in_module(nested, module) == [
        "REVIEW_PROBE_NESTED_PATH='judge/index.json'"
    ]

    # THE BOUNDARY OF THE VALUE RULE, pinned so it is a decision rather than a surprise: an
    # ABSOLUTE path and a ``..`` step are not entries of a session directory, and a name with
    # none of the suffixes above is not a candidate (``ACTION_TOOL_NAME`` is a tool).
    for ignored in ('REVIEW_PROBE_ABS_PATH = "/etc/hosts"', 'REVIEW_PROBE_UP_PATH = "../x.json"'):
        assert _unclassified_in_module(pristine + "\n" + ignored + "\n", module) == []


# ---------------------------------------------------------------------------
# The round trip: one of every entry type, copied for real
# ---------------------------------------------------------------------------


def _ask(root: Path) -> Any:
    """The owner's half, in process — the same shape ``test_sync.py`` uses."""

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        phase = str(frame.get("phase") or "")
        session_id = str(frame.get("session_id") or "")
        if phase == "plan":
            return sync.build_manifest(
                root, session_id, have=frame.get("have") if isinstance(frame, dict) else {}
            )
        if phase == "fetch":
            return sync.serve_fetch(
                root,
                session_id,
                plan=str(frame.get("plan_id") or ""),
                name=str(frame.get("name") or ""),
                offset=int(frame.get("offset") or 0),
                limit=int(frame.get("limit") or sync.SYNC_CHUNK_BYTES),
            )
        if phase == "verify":
            return sync.serve_verify(
                root,
                session_id,
                plan=str(frame.get("plan_id") or ""),
                name=str(frame.get("name") or ""),
                prefix_bytes=int(frame.get("prefix_bytes") or 0),
                prefix_digest=str(frame.get("prefix_digest") or ""),
            )
        raise AssertionError(f"unexpected phase {phase!r}")

    return ask


def _every_entry_directory(root: Path, session_id: str) -> Path:
    """One session directory holding ONE of every entry type, each with its own bytes.

    Distinct content per entry so "did it arrive" is a byte comparison rather than a
    presence check, and a tree with a nested directory so the tree half is real.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(_entry_owners()) | set(MEASURED_IN_THE_STORE)):
        if name in sync.COPY_SET_TREES:
            continue
        (directory / name).write_text(f"content of {name}\n", encoding="utf-8")
    # The transcript is JSONL and (in the real product) references attachments, so it
    # gets a real row plus a referenced blob and its sidecar in the shared store.
    ref = "a" * 32
    (directory / "transcript.jsonl").write_text(
        json.dumps({"id": "e1", "type": "image", "attachment": ref}) + "\n", encoding="utf-8"
    )
    store = root / "attachments"
    store.mkdir(parents=True, exist_ok=True)
    (store / f"{ref}.bin").write_bytes(b"blob bytes")
    (store / f"{ref}.json").write_text(json.dumps({"mime_type": "image/png"}), encoding="utf-8")
    tree = directory / "scratchpad"
    (tree / "nested").mkdir(parents=True, exist_ok=True)
    (tree / "notes.md").write_text("notes\n", encoding="utf-8")
    (tree / "nested" / "run.txt").write_text("nested content\n", encoding="utf-8")
    return directory


def test_a_directory_of_every_entry_type_round_trips(tmp_path: Path) -> None:
    """Copy a directory holding every entry type: each one arrives or is excluded.

    THIS IS THE PROPERTY THAT REPLACES A LIST OF SPECIAL CASES. For every classified
    entry the outcome is asserted in the direction the classification promises —
    byte-identical at the destination for a copied name or a tree file, ABSENT for
    an excluded one — and a name that is neither would fail this test on its
    presence assertion (``copied`` is derived from the two lists, so a name added to
    one of them is exercised without editing this test).
    """
    source = tmp_path / "owner"
    source.mkdir()
    session_id = "b7d1c0ffee42"
    directory = _every_entry_directory(source, session_id)
    dest = tmp_path / "holder"

    sync.sync_from(source, session_id, ask=_ask(source), into=dest)

    for name in sorted(set(_entry_owners()) | set(MEASURED_IN_THE_STORE)):
        arrives = name in sync.COPY_SET_NAMES or name in sync.COPY_SET_TREES
        if name == "transcript.jsonl":
            assert (dest / name).read_bytes() == (directory / name).read_bytes()
            continue
        if name in sync.COPY_SET_TREES:
            # A tree is classified as a DIRECTORY: its own assertions are below, and
            # this is the one entry whose files (not itself) are the copy's members.
            assert (dest / name).is_dir(), f"{name} is a copy-set tree but did not arrive"
            continue
        if arrives:
            assert (dest / name).is_file(), f"{name} is in the copy set but did not arrive"
            assert (dest / name).read_bytes() == (directory / name).read_bytes(), name
        else:
            assert name in sync.EXCLUDED_ENTRIES, name
            assert not (
                dest / name
            ).exists(), f"{name} is excluded ({sync.EXCLUDED_ENTRIES[name]}) but arrived anyway"
    # The tree, including its nested directory, and the referenced blob's sidecar.
    assert (dest / "scratchpad" / "notes.md").read_text(encoding="utf-8") == "notes\n"
    assert (dest / "scratchpad" / "nested" / "run.txt").read_text(encoding="utf-8") == (
        "nested content\n"
    )
    assert (dest / "attachments" / f"{'a' * 32}.bin").is_file()
    assert (dest / "attachments" / f"{'a' * 32}.json").is_file()


# ---------------------------------------------------------------------------
# Fail closed: an entry nobody classified refuses a DELETING move
# ---------------------------------------------------------------------------


def test_an_unlisted_entry_refuses_a_deleting_move(tmp_path: Path) -> None:
    """``assert_complete`` names it, and the source is untouched.

    The product's answer to "the copy set is behind" has to be a refusal rather than
    a copy-it-and-delete-the-rest, because the alternative is the bug this file
    exists for. The sentence must name the entry, since the person reading it is the
    one who can move or remove it.
    """
    directory = tmp_path / "sessions" / "abc123"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    (directory / "a-file-from-the-future.dat").write_bytes(b"\x00\x01")

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.assert_complete(directory)
    assert refusal.value.code == "unlisted_content"
    assert "a-file-from-the-future.dat" in refusal.value.message
    assert (directory / "a-file-from-the-future.dat").exists()


def test_an_irregular_entry_in_a_tree_refuses_a_deleting_move(tmp_path: Path) -> None:
    """A symlink inside ``scratchpad/`` is not portable data, so a move refuses.

    A symlink's target names a path on the SOURCE device; writing the same text on
    the destination points an agent at a different file (or at nothing). Copying it
    would be a lie and skipping it silently would be a deletion, so the move refuses
    with the path named. Measured on the operator's workstation: 100 of 3,019
    scratchpads hold one.
    """
    directory = tmp_path / "sessions" / "abc123"
    (directory / "scratchpad").mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("elsewhere\n", encoding="utf-8")
    (directory / "scratchpad" / "link.txt").symlink_to(outside)

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.assert_complete(directory)
    assert refusal.value.code == "unlisted_content"
    assert "scratchpad/link.txt" in refusal.value.message

    # The plan reports it too, without failing: a --keep copy carries what it can and
    # leaves the source alone, so it has nothing to lose by skipping it.
    from tests.unit.network.test_sync import seed  # noqa: PLC0415 — the shared fixture

    seed(tmp_path, "abc123")
    plan = sync.build_manifest(tmp_path, "abc123")
    assert "scratchpad/link.txt" in plan["trees_skipped"]
    assert "scratchpad/link.txt" not in plan["trees"]


def test_the_tree_walk_is_bounded_to_the_tree(tmp_path: Path) -> None:
    """A tree entry name cannot escape the tree, and a peer cannot ask for one.

    ``_tree_entry_path`` is the peer-facing guard: names arrive in fetch frames, so
    ``scratchpad/../../etc/passwd`` has to be refused rather than resolved. The
    attachment branch gets the same treatment (``_blob_file_name``), because a
    destination WRITES to the path that name resolves to.
    """
    from local_operator.network.sync import _item_path

    tmp_path.mkdir(parents=True, exist_ok=True)
    session = tmp_path / "sessions" / "abc123"
    session.mkdir(parents=True)
    (session / "scratchpad").mkdir()
    (session / "scratchpad" / "notes.md").write_text("ok\n", encoding="utf-8")

    assert _item_path(tmp_path, "abc123", "scratchpad/notes.md", None) == (
        session / "scratchpad" / "notes.md"
    )
    for escape in (
        "scratchpad/../transcript.jsonl",
        "scratchpad/../../outside.txt",
        "/etc/passwd",
        "scratchpad/",
        "scratchpad/./notes.md",
    ):
        assert _item_path(tmp_path, "abc123", escape, None) is None, escape
    for escape in ("attachments/../../etc/passwd", "attachments/", "attachments/a/b.bin"):
        assert _item_path(tmp_path, "abc123", escape, None) is None, escape


def test_a_copy_does_not_carry_the_store_of_an_unreferenced_blob(tmp_path: Path) -> None:
    """Only blobs the transcript references travel, with their sidecars.

    The store is never pruned, so it accumulates blobs no session uses. Copying the
    whole store would make every move cost the install's entire attachment history
    (and would put another session's content in this one's replica), so the copy set
    is the transcript's own references — and the sidecar travels with its blob, or
    the destination cannot resolve the mime type (review round 1, M-1).
    """
    source = tmp_path / "owner"
    source.mkdir()
    session_id = "c0ffee123456"
    directory = source / "sessions" / session_id
    directory.mkdir(parents=True)
    used, unused = "1" * 32, "2" * 32
    (directory / "transcript.jsonl").write_text(
        json.dumps({"id": "e1", "type": "image", "attachment": used}) + "\n", encoding="utf-8"
    )
    store = source / "attachments"
    store.mkdir()
    for ref in (used, unused):
        (store / f"{ref}.bin").write_bytes(b"bytes")
        (store / f"{ref}.json").write_text("{}", encoding="utf-8")

    dest = tmp_path / "holder"
    sync.sync_from(source, session_id, ask=_ask(source), into=dest)

    assert (dest / "attachments" / f"{used}.bin").is_file()
    assert (dest / "attachments" / f"{used}.json").is_file()
    assert not (dest / "attachments" / f"{unused}.bin").exists()


# ---------------------------------------------------------------------------
# The shapes the REAL store has, which a census of constants cannot show
# ---------------------------------------------------------------------------
#
# Measured on the operator's store (11,115 session directories, 2026-09-24): 137 sessions
# refused to move. 121 of them held a symlink inside ``scratchpad/``, 17 held a directory
# an agent created at the session root, and 2 held a ``*.tmp`` file left by a writer that
# died mid-rename. Every one of those is a session a person works in, so "correct to
# refuse" is not an answer: each shape gets a rule, and the rules are these four tests.


def _seeded(root: Path, session_id: str = "abc123", *, rows: int = 3) -> Path:
    """A session holding the whole copy set, from the shared fixture in ``test_sync``."""
    from tests.unit.network.test_sync import seed  # noqa: PLC0415 — the shared fixture

    return seed(root, session_id, rows=rows)


def _move_manifest(root: Path, session_id: str) -> dict[str, Any]:
    """The plan a DELETING move gets: its transcript whole, like the product's."""
    return sync.build_manifest(root, session_id, whole_transcript=True)


def test_a_directory_at_the_session_root_travels(tmp_path: Path) -> None:
    """An agent-made directory at the root is CONTENT, classified by what it is.

    The census names them: ``notes``, ``drafts``, ``workspace``, ``qa-1443``, ``bench``,
    ``pr1334``, … — 17 sessions. A copy set that refuses them is a copy set that cannot
    move the sessions people actually work in, so a directory is carried as a tree, its
    regular files and its carried links with it. A FILE of an unknown name is still
    refused (``test_an_unlisted_entry_refuses_a_deleting_move``): a name is something a
    reader keys on, a container is not.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = _seeded(source)
    (directory / "notes").mkdir()
    (directory / "notes" / "todo.md").write_text("finish the move\n", encoding="utf-8")
    (directory / "notes" / "nested").mkdir()
    (directory / "notes" / "nested" / "deep.txt").write_text("deep\n", encoding="utf-8")

    sync.assert_complete(directory)  # a DELETING move is allowed to proceed
    plan = _move_manifest(source, "abc123")
    assert "notes/todo.md" in plan["trees"]
    assert "notes/nested/deep.txt" in plan["trees"]

    dest = tmp_path / "holder"
    sync.sync_from(source, "abc123", ask=_ask(source), into=dest)
    assert (dest / "notes" / "todo.md").read_text(encoding="utf-8") == "finish the move\n"
    assert (dest / "notes" / "nested" / "deep.txt").read_text(encoding="utf-8") == "deep\n"


def test_a_cache_directory_is_excluded_and_reported(tmp_path: Path) -> None:
    """A bytecode or test-runner cache is derived state: not copied, not refused.

    Copied it would be pure cost (the real store has a ``.pytest_cache`` of hundreds of MB
    in a session's own tree), refused it would make a session unmovable because an agent
    once ran ``pytest`` in it. It is REPORTED in the plan instead, so "why did that not
    come across?" is answered by the document.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = _seeded(source)
    (directory / ".pytest_cache" / "v" / "cache").mkdir(parents=True)
    (directory / ".pytest_cache" / "v" / "cache" / "lastfailed").write_text("{}\n")

    sync.assert_complete(directory)
    plan = _move_manifest(source, "abc123")
    assert ".pytest_cache/v/cache/lastfailed" not in plan["trees"]
    assert plan["trees_excluded"] == [".pytest_cache"]

    dest = tmp_path / "holder"
    sync.sync_from(source, "abc123", ask=_ask(source), into=dest)
    assert not (dest / ".pytest_cache").exists()


def test_a_tmp_leftover_does_not_make_a_session_unmovable(tmp_path: Path) -> None:
    """A crash artifact is not content, and it must not block a move forever.

    Every atomic writer here writes ``<name>.<pid>.tmp`` beside its target and renames it
    (``session/runtime/registry._staged_write``, ``resume``'s sidecars). A writer killed
    mid-rename therefore leaves one behind — measured in two of the operator's sessions
    (``.subagent-roster.v1.json.<rand>.tmp`` and ``title-scan.<pid>.tmp``) — and with no
    rule for it, the session could never be moved again without the user finding a hidden
    file. The leftovers are named in the plan and left behind, not copied: they are a
    partial write of something the destination will write itself.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = _seeded(source)
    (directory / ".subagent-roster.v1.json.n2f_mr71.tmp").write_text("{}", encoding="utf-8")
    (directory / "title-scan.79786.tmp").write_text("{}", encoding="utf-8")
    (directory / "goal.json.4242.tmp").write_text("{}", encoding="utf-8")

    sync.assert_complete(directory)
    plan = _move_manifest(source, "abc123")
    assert plan["transients"] == [
        ".subagent-roster.v1.json.n2f_mr71.tmp",
        "goal.json.4242.tmp",
        "title-scan.79786.tmp",
    ]

    dest = tmp_path / "holder"
    sync.sync_from(source, "abc123", ask=_ask(source), into=dest)
    assert not (dest / "goal.json.4242.tmp").exists()


def test_a_symlink_inside_the_session_travels_as_a_relative_link(tmp_path: Path) -> None:
    """6,629 of the operator's scratchpad links point inside their own session.

    pytest's ``*-current`` links, a venv's ``bin/python3``: all of them are the session's
    own files, so refusing them refused 121 sessions for a link that means the same thing
    anywhere. CARRIED, as a link — never as a copy of what it points at (that would
    duplicate a whole subtree under a second name) — with an absolute spelling rewritten
    RELATIVE, because the store root on the destination is a different path, and on
    another device a different machine's home.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = _seeded(source)
    (directory / "scratchpad" / "link.txt").symlink_to("notes.md")
    (directory / "scratchpad" / "absolute.txt").symlink_to(directory / "scratchpad" / "notes.md")
    (directory / "dangling-out.txt")  # not a session member at all: see the sibling test

    sync.assert_complete(directory)
    plan = _move_manifest(source, "abc123")
    carried = dict(plan["links"])
    assert carried["scratchpad/link.txt"] == "notes.md"
    assert (
        carried["scratchpad/absolute.txt"] == "notes.md"
    ), "an absolute inside link is made relative"
    assert "scratchpad/link.txt" not in plan["trees"], "a link is not a file member"

    dest = tmp_path / "holder"
    sync.sync_from(source, "abc123", ask=_ask(source), into=dest)
    link = dest / "scratchpad" / "link.txt"
    assert link.is_symlink(), "the destination holds a LINK, not a copy of its target"
    assert os.readlink(link) == "notes.md"
    assert (dest / "scratchpad" / "absolute.txt").is_symlink()
    assert os.readlink(dest / "scratchpad" / "absolute.txt") == "notes.md"
    assert link.read_text(encoding="utf-8") == "the operator's notes", "and it still resolves"


def test_a_symlink_out_of_the_session_refuses_a_deleting_move(tmp_path: Path) -> None:
    """341 links point outside their session, and those still refuse — with the link named.

    This is the shape the module's own docstring has always argued about: the text names a
    path on THIS device (a scratch checkout, ``/var/folders``), so on the destination it
    resolves to a different file or to nothing, and an agent following it reads something
    the owner never referenced. Refusing names the link and what to do; the 84 sessions
    affected are the price of not writing a lie into the copy.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = _seeded(source)
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("not this session's\n", encoding="utf-8")
    (directory / "scratchpad" / "out.txt").symlink_to(outside)

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.assert_complete(directory)
    assert refusal.value.code == "unlisted_content"
    assert "scratchpad/out.txt" in refusal.value.message

    # A --keep copy is not a delete, so it carries what it can and leaves the rest.
    plan = sync.build_manifest(source, "abc123")
    assert "scratchpad/out.txt" in plan["trees_skipped"]
    assert "scratchpad/out.txt" not in plan["trees"]


def test_a_socket_or_fifo_in_a_tree_is_reported_and_never_refused(tmp_path: Path) -> None:
    """A fifo or socket holds no data: the payload is in the process that made it.

    A tmux socket left by a killed rig (measured in five of the operator's sessions, plus
    two fifos) is a crash artifact of the same family as a ``*.tmp``: nothing is lost by
    leaving it, and refusing made those sessions permanently unmovable. Reported in the
    plan so the decision is visible.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = _seeded(source)
    os.mkfifo(directory / "scratchpad" / "probe.fifo")

    sync.assert_complete(directory)
    plan = _move_manifest(source, "abc123")
    assert plan["trees_unportable"] == ["scratchpad/probe.fifo"]

    dest = tmp_path / "holder"
    sync.sync_from(source, "abc123", ask=_ask(source), into=dest)
    assert not (dest / "scratchpad" / "probe.fifo").exists()


def test_a_store_directory_inside_a_session_is_refused(tmp_path: Path) -> None:
    """``attachments/`` at a session root is not a blob store and not content.

    The store lives at the config root, so this name inside a session is something else —
    and it is the one NAME (not a directory in general) a copy must not guess about, since
    the adopter resolves blobs through the store and would carry content nothing reads.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = _seeded(source)
    (directory / "attachments").mkdir()
    (directory / "attachments" / "stray.bin").write_bytes(b"bytes")

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.assert_complete(directory)
    assert "attachments" in refusal.value.message


def test_a_symlinked_content_tree_is_refused_and_reported(tmp_path: Path) -> None:
    """A TREE NAME THAT IS A LINK IS REFUSED, and the plan says so.

    MEASURED BEFORE THE FIX (review round 3, MAJOR). ``unlisted_entries`` decided by NAME
    first, so a session whose ``scratchpad`` was a symlink was skipped there (the name is in
    ``COPY_SET_TREES``) and never walked (``content_trees`` does not walk a link, correctly):
    the plan reported ``trees: []`` and ``trees_skipped: []``, the deleting move committed,
    and the source directory went WITH THE LINK INSIDE IT. Nothing needs to be live for that
    to be loss — the link is the session's own pointer to the user's tree — and it is
    reachable by pointing ``scratchpad`` at another volume, which is what somebody with a
    1 GB scratchpad does.

    REFUSED RATHER THAN CARRIED, deliberately: a copy carries a session's own trees, and this
    link's target is by construction outside the session (a link INSIDE one is carried by the
    tree walk, as its sibling cells measure). Carrying it would either write a link that
    resolves to nothing on the destination, or copy a tree in under a name the user did not
    choose — both worse than a sentence naming the link.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = _seeded(source)
    elsewhere = tmp_path / "other-volume"
    (elsewhere / "scratchpad").mkdir(parents=True)
    (elsewhere / "scratchpad" / "notes.md").write_text("the operator's notes\n", encoding="utf-8")
    shutil.rmtree(directory / "scratchpad")
    (directory / "scratchpad").symlink_to(elsewhere / "scratchpad")

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.assert_complete(directory)
    assert refusal.value.code == "unlisted_content", refusal.value
    assert "scratchpad" in refusal.value.message
    assert "symlink" in refusal.value.message, (
        "the sentence must name the SHAPE: the copy set DOES carry scratchpad, so 'this "
        "build does not carry it' sends a person to the wrong place"
    )

    # THE PLAN REPORTS IT, which is what made the hole invisible: a plan saying ``trees: []``
    # and ``trees_skipped: []`` about a scratchpad cannot be read to find out why a session
    # will not move.
    plan = sync.build_manifest(source, "abc123")
    assert plan["trees_skipped"] == ["scratchpad"], plan["trees_skipped"]
    assert [name for name in plan["trees"] if name.startswith("scratchpad/")] == []

    # The target is untouched, and so is the session's own pointer to it.
    assert (elsewhere / "scratchpad" / "notes.md").is_file()
    assert (directory / "scratchpad").is_symlink()


def test_a_symlinked_store_file_keeps_its_exclusion(tmp_path: Path) -> None:
    """The shape rule is about what a copy CARRIES, and an excluded name is not carried.

    The narrow half of the fix, pinned so the next round cannot widen it by accident:
    ``mesh.json`` is excluded because the destination writes its own (the ownership stamp),
    so a session whose stamp happens to be a link is not refused — it is excluded for the
    reason the exclusion already states. Refusing it would newly block a session for a name
    whose bytes this build never copies.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = _seeded(source)
    outside = tmp_path / "stamp.json"
    outside.write_text("{}\n", encoding="utf-8")
    stamp = directory / "mesh.json"
    if stamp.exists():
        stamp.unlink()
    stamp.symlink_to(outside)

    sync.assert_complete(directory)
    plan = sync.build_manifest(source, "abc123")
    assert "mesh.json" not in plan["trees_skipped"]


def test_a_fetch_does_not_re_hash_the_copy_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE COST FIX, as a property rather than a stopwatch reading.

    Every fetch used to re-derive the whole copy set's digest — including ``scratchpad/``,
    which is where a session's big files live — so a move's cost was O(bytes x chunks): an
    8 MB scratchpad took 2.6 s and a 32 MB one 38.7 s (review round 2, MAJOR 2). Now the
    digests are taken once when the plan is built and each chunk is validated with a
    ``stat``. Measured here by counting hashes: the count at the end of a multi-chunk pull
    is the count the plan itself produced, however many chunks crossed the wire.
    """
    source = tmp_path / "owner"
    source.mkdir()
    directory = _seeded(source, rows=3)
    # Three chunks' worth, so the count below is a real span and not one call.
    (directory / "scratchpad" / "big.bin").write_bytes(os.urandom(3 * sync.SYNC_CHUNK_BYTES + 7))

    hashes: list[str] = []
    real = sync._stream_digest

    def counting(path: Path) -> str:
        hashes.append(str(path))
        return real(path)

    monkeypatch.setattr(sync, "_stream_digest", counting)
    dest = tmp_path / "holder"
    result = sync.sync_from(source, "abc123", ask=_ask(source), into=dest)

    chunks = (3 * sync.SYNC_CHUNK_BYTES + 7) // sync.SYNC_CHUNK_BYTES + 1
    assert chunks >= 4, "the file must cross several chunks for this to mean anything"
    assert len(hashes) <= len(set(hashes)) + 1, (
        f"{len(hashes)} hashes for {len(set(hashes))} distinct members: the fetch path is "
        "re-hashing members instead of validating them with one stat"
    )
    assert (dest / "scratchpad" / "big.bin").stat().st_size == 3 * sync.SYNC_CHUNK_BYTES + 7
    assert result["ok"] is True
