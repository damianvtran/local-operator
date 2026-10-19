"""S2 (wire.slash): slash-command descriptions resolve from the en catalogue.

The extraction is a NO-OP (RFC §5): every ``SlashCommand.description`` carries
the byte-identical English it carried before the slice, and it now comes from
``local_operator/i18n/catalogues/en/wire.slash.json`` through the i18n runtime
(one ``wire.slash.<command>`` key per registry entry, RFC §2.2).

The table below is the PRE-SLICE English, captured from the registry on
``origin/main`` @ ``22498f1a59`` (the M0 merge) before the rewrite — the
English-pinned shape of RFC §7: an edit that changes wording must change the
catalogue and this table together, under the pinned ``en`` locale.

The C slice (live language access) later registered ONE new command,
``/language``, through the same runtime path — an ADDITION rather than a
rewrite, pinned here beside the migrated rows so a wording edit to it meets
the same discipline.
"""

from __future__ import annotations

from local_operator.i18n import catalogues
from local_operator.i18n.keys import wire_slash, wire_slash_library
from local_operator.i18n.messages import render
from local_operator.slash_commands import SLASH_COMMANDS

#: ``command name -> description as it read before S2`` — captured by dumping
#: the pre-slice registry (name, description, aliases, desktop_destination);
#: only the description column is pinned here, the machine fields are pinned
#: by the registry's own tests.
#: Not in wire.slash on purpose: any edit to that catalogue voids the audited locale ledger hashes.
MEMORY_COMMAND = "memory"
#: ``/import`` follows the ``/memory`` PRECEDENT (keep a new command out of the
#: audited wire.slash catalogue), but its copy lives in the English-only
#: ``tui.import`` namespace because the page is TUI-owned — not in
#: ``wire.memory``'s namespace class. This test exempts it from the key walk.
IMPORT_COMMAND = "import"
#: ``/review`` is the same shape as ``/import``: a TUI-owned page whose copy
#: lives in the English-only ``tui.review`` namespace, deliberately out of the
#: audited ``wire.slash`` catalogue.
REVIEW_COMMAND = "review"
#: ``/schedule`` repeats the shape once more: a TUI-owned page whose copy lives
#: in the English-only ``tui.schedule`` namespace.
SCHEDULE_COMMAND = "schedule"
#: ``/library`` and ``/librarian`` are the first commands whose copy moved out
#: of ``wire.slash`` into a TRANSLATED CHILD namespace, ``wire.slash.library``
#: (keys ``help`` and ``librarian_help``). They stay out of the audited
#: ``wire.slash`` catalogue so its ledger hash remains valid, and are pinned
#: against that child namespace by
#: ``test_library_commands_resolve_through_their_child_namespace`` instead —
#: the exemption carries coverage rather than hiding a gap.
LIBRARY_COMMAND = "library"
LIBRARIAN_COMMAND = "librarian"
CHILD_NAMESPACE_COMMANDS = (LIBRARY_COMMAND, LIBRARIAN_COMMAND)

PREVIOUS_ENGLISH: dict[str, str] = {
    "help": "List all commands",
    "keys": "Show the keyboard legend",
    "exit": "Quit the app",
    "clear": "Clear the transcript (history is untouched)",
    "copy": "Copy an agent message or code block",
    "links": "Open a link from this conversation in a browser",
    "new": "Start a new conversation",
    "reload": "Relaunch this conversation on the current install",
    "update": "Install the latest version from PyPI and relaunch",
    "resume": "Pick a past conversation to resume, or resume one (id)",
    "aida": "Open your chief of staff, or send her a request",
    "move": "Change this session's working directory",
    "rename": "Name this conversation, or /title --refresh",
    "fork": "Branch this chat; --switch here, --window elsewhere; <message> starts work",
    "archive": "Hide from the lists, still resumable by id",
    "unarchive": "Show this conversation in the lists again",
    "delete": "Delete this conversation for good; asks to confirm",
    "model": "Switch model; /model default saves it for new sessions",
    "effort": "Show or set reasoning effort (shift+tab cycles)",
    "fast": "Toggle faster output at premium pricing",
    "theme": "Switch color theme; arrows preview live",
    # Registered by the C slice, not migrated by S2: `/language`'s description
    # is new English by design (the same runtime path, one new key). Round-1
    # D3: "the language" → "the UI language" — the old reading left it open
    # whether the agent's output language was in play (it is a different
    # thing, SYS1's §2.9).
    "language": "Show or set the UI language; auto follows the system",
    "provider": "List providers and their login/usage state",
    "settings": "Change every setting on one page",
    "sidebar": "Show or hide conversations; 'focus' keys the list",
    "search": "Configure web search providers and load balancing",
    "accounts": "List stored credentials",
    "failovers": "Show the model failover cascade and what is serving",
    "usage": "Show provider usage quota",
    "context": "Show prompt, tool-schema and message token usage",
    "session": "Usage, cost, diagnostics; --copy copies the session ID",
    "info": "Install, version, and running sessions on this machine",
    "analytics": "Aggregated token-consumption analytics across all sessions",
    "goal": "Set the goal and start work; /goal --clear clears it",
    "loop": "Loop toward a goal: /loop <goal>, <n>; --stop cancels",
    "btw": "Ask a side question off the record (esc closes it)",
    "compact": "Compact the context now",
    "stop": "End this session, another by name/pid, or all — /resume reopens it",
    "approvals": "Show or set tool approval mode for this session",
    "skills": "List loaded skills",
    "mcp": "List MCP servers; add/remove one, or manage an OAuth grant",
    "login": "Authenticate a provider",
    "logout": "Remove stored provider credentials",
    "notifications": "Unread completions; read marks them read",
    "mobile": "Radient phone access: status, enable, stop, billing",
    "network": "Networks, peers, and this device's mesh state",
    "credential": "Type or paste a secret after a space; masked",
    "team": "List teams, chart a team's org, or send a request to a team's manager",
    "agent": "List agents, switch an agent's class, or speak to this session as one",
    "project": "Track workstreams: list, show, new, delete, link, unlink",
    # Added AFTER the S2 slice (turn supplements, TUI lane): there is no
    # pre-slice English for a command that did not exist, so this row pins the
    # text the catalogue shipped with — the same edit-together rule applies.
    "highlights": "Cancel, retry or steer the highlights under an answer",
    # The hub-streams core slice's `/hub` row: the catalogue key landed first
    # (shipped ahead), and this row folds it in as the command registers — the
    # edit-together rule the SHIPPED_AHEAD allowance existed to keep honest.
    "hub": "Open Hub links — per-link subscribe and auto-push policy",
}


def test_every_registry_description_is_the_pre_slice_english() -> None:
    # The migrated SITES: the registry field the picker, /help, the desktop
    # catalogue and the phone sheet all read. A missing/extra command raises a
    # KeyError here rather than passing silently.
    # `/memory`, `/import` and `/review` are deliberately absent from
    # `wire.slash`: their copy lives in English-only namespaces so the audited
    # wire.slash hash stays valid, and a missing or extra command raises a
    # KeyError here either way rather than passing silently.
    # 52 pre-slice rows + the four TUI-owned pages (memory/import/review/
    # schedule) + the two child-namespace commands (library/librarian).
    assert len(SLASH_COMMANDS) == len(PREVIOUS_ENGLISH) + 4 + 2 == 58
    for command in SLASH_COMMANDS:
        if command.name in (
            MEMORY_COMMAND,
            IMPORT_COMMAND,
            REVIEW_COMMAND,
            SCHEDULE_COMMAND,
            *CHILD_NAMESPACE_COMMANDS,
        ):
            continue
        assert command.description == PREVIOUS_ENGLISH[command.name], command.name


def test_the_catalogue_carries_exactly_the_command_descriptions() -> None:
    # Both directions: no key for a command is missing, and the slice added no
    # key that is not a registry description.
    messages = catalogues.load_catalogue("en", "wire.slash")
    expected = {f"wire.slash.{name}": text for name, text in PREVIOUS_ENGLISH.items()}
    assert messages == expected


def test_library_commands_resolve_through_their_child_namespace() -> None:
    # `/library` and `/librarian` are the first commands whose copy lives in a
    # TRANSLATED CHILD namespace instead of the audited `wire.slash` catalogue
    # (whose hash must stay valid). They are exempt from the `wire.slash` walk
    # above, so this test is what keeps the exemption honest: the child
    # namespace must carry exactly their two descriptions, and the generated
    # per-key functions must resolve to the registry text under `en`.
    #
    # Both directions are asserted, so deleting a key or adding a third one
    # fails here rather than passing silently as "exempt".
    child = catalogues.load_catalogue("en", "wire.slash.library")
    assert child == {
        "wire.slash.library.help": "Browse the global asset library",
        "wire.slash.library.librarian_help": "Open, pause or run the Librarian",
    }
    by_name = {command.name: command for command in SLASH_COMMANDS}
    assert by_name[LIBRARY_COMMAND].description == child["wire.slash.library.help"]
    assert (
        by_name[LIBRARIAN_COMMAND].description
        == child["wire.slash.library.librarian_help"]
    )
    for key, function in (
        ("wire.slash.library.help", wire_slash_library.help),
        ("wire.slash.library.librarian_help", wire_slash_library.librarian_help),
    ):
        message = function()
        assert message.code == key
        assert render(message.code, message.params, locale="en") == child[key]


def test_each_generated_key_resolves_to_the_previous_english_under_en() -> None:
    # The resolution path the registry itself uses — the generated per-key
    # function, rendered through the runtime — asserted under the pinned `en`
    # locale (RFC §7).
    for command in SLASH_COMMANDS:
        if command.name in (
            MEMORY_COMMAND,
            IMPORT_COMMAND,
            REVIEW_COMMAND,
            SCHEDULE_COMMAND,
            *CHILD_NAMESPACE_COMMANDS,
        ):
            continue
        function = getattr(wire_slash, command.name)
        message = function()
        assert message.code == f"wire.slash.{command.name}"
        assert render(message.code, message.params, locale="en") == PREVIOUS_ENGLISH[command.name]
