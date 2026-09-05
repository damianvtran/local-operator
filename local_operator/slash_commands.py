"""Host-neutral slash metadata shared by terminal and desktop front ends.

This is the original terminal registry, not a desktop copy. Argument and echo
semantics must stay identical across hosts; only each host's action destination
and presentation differ. Importing this module never loads the Textual app.
"""

from local_operator.tui.autocomplete import ArgumentMode, SlashCommand

#: ONE route named here, not both. The second route (the `/settings` model rows)
#: does not fit: this string is sized by the picker footer, whose budget is 43
#: cells at 50 columns (card width minus `_GUTTER_CELLS` + `_EDGE_MARGIN`), and
#: this clause is 42. Any "; /settings too" tail measures 57 and truncates
#: mid-word at the one width where the instruction most needs to survive whole.
#: So the footer gets the command and the roomier surfaces get the pair: the
#: bare-`/model` notice names `/settings` in `_persist_hint_notice`, which wraps
#: instead of truncating, and the `/help` row is reachable at any width.
PERSIST_HINT = "/model default saves this for new sessions"

SLASH_COMMANDS: list[SlashCommand] = [
    # The help table is the receipt.
    SlashCommand("help", "List all commands", desktop_destination="commands"),
    # The app is gone; there is no ledger left to read.
    SlashCommand("exit", "Quit the app", aliases=("quit",), desktop_destination="window.close"),
    # Empties the surface the echo would land on — it was wiped a line later.
    SlashCommand(
        "clear",
        "Clear the transcript (history is untouched)",
        desktop_destination="transcript.clear",
    ),
    # Beside `/clear` because they are the two commands that act on the
    # TRANSCRIPT AS A DOCUMENT rather than on the conversation: one empties the
    # surface, the other takes a message out of it. Deliberately NOT beside
    # `/compact`, which shares its first three letters and nothing else —
    # compaction rewrites history for the model, this reads the frame for the
    # human.
    #
    # NOT an echo. The clipboard receipt names how much landed there, which is
    # strictly more than the typed word, and nothing here reaches the model —
    # `/approvals`' rule exactly.
    #
    # The description names WHAT CAN BE PICKED, not one message, because the
    # command opens a chooser: a whole answer, or a single code block or quote
    # out of it. "the last agent message" described the pre-picker behaviour and
    # would now send a user looking for the one thing the command no longer
    # does. 35 cells, inside the ~55 the description column wraps past (see
    # `/model` and `/theme`, where a wrapping row renders a phantom command name
    # in `/help`).
    SlashCommand(
        "copy", "Copy an agent message or code block", desktop_destination="transcript.copy"
    ),
    # Replaces the transcript; a row describing the old one would not survive.
    SlashCommand("new", "Start a new conversation", desktop_destination="sessions.new"),
    # In-process reboot cannot load a replaced wheel; this command exists so
    # ``/update`` is not the only way to pick up new code. Same relaunch
    # helper as ``/update`` — the conversation comes back via ``--resume``.
    SlashCommand(
        "reload",
        "Relaunch this conversation on the current install",
        desktop_destination="sessions.reload",
    ),
    # The notice (or the relaunch) is the receipt. echo=False is the default;
    # pin it in ECHO_POLICY so a later flip cannot sneak a user row onto an
    # empty splash that ``/update`` is required to leave standing.
    SlashCommand(
        "update", "Install the latest version from PyPI and relaunch", desktop_destination="updates"
    ),
    # The picker (or "resuming session <id>…") is the receipt, and a resume
    # replaces the transcript anyway.
    SlashCommand(
        "resume",
        "Pick a past conversation to resume, or resume one (id)",
        aliases=("recall",),
        desktop_destination="sessions.resume",
    ),
    # Beside `/resume` because it names the thing the picker lists. NOT an echo:
    # the argument is the conversation's own label — it goes on the band and the
    # terminal tab, never into anything the model is told — and the receipt
    # quotes the title that ended up in force, which is strictly more than the
    # typed words (the store trims and caps them).
    SlashCommand(
        "rename",
        "Rename this conversation; auto-naming never overrides it",
        desktop_destination="session.rename",
    ),
    # Beside the session-transition family because it is one: /fork is the entry
    # the table was missing, the one that carries history INTO a fresh session
    # (/new discards it, /resume moves to an existing one).
    #
    # echo=True, and it is the case the registry's echo rule was written for: the
    # argument becomes a user turn the MODEL is given — in the FORK. The receipt
    # names both session ids, but only the echo shows what the fork was asked to
    # do, and that text is not visible anywhere in this window otherwise.
    #
    # consumes_prompt=True because the argument is free text destined for a
    # model, so an inline /fork reassembles to the front of the composer rather
    # than splicing into the middle of a sentence.
    SlashCommand(
        "fork",
        # Terse for the reason `/model` and `/theme` record above: the
        # description column wraps past ~55 cells. The long form was 76
        # characters — the longest of all 32 commands — and it was the ONLY row
        # in `/help` that wrapped at 100 columns, hanging its orphan word back
        # in the COMMAND column so the listing rendered a phantom command named
        # `message`. The picker truncated it before the argument clause at every
        # common width, cutting exactly the half that says the argument exists.
        #
        # `<message>` is front-loaded rather than trailing so it survives that
        # truncation: at 60 columns a user still sees that the argument is a
        # message the branch STARTS ON, which is what stops them typing a title
        # and being billed for a turn in another window. `docs/fork.md` carries
        # the rest.
        "Branch this conversation; /fork <message> starts it on that",
        echo=True,
        consumes_prompt=True,
        desktop_destination="session.fork",
    ),
    # The switch receipt names the old AND new label — strictly more than the
    # typed selector, which may have been elided to `default`.
    SlashCommand(
        "model",
        # Terse by necessity — the description column wraps past ~55 cells at 80
        # columns, and `Switch model; ` + the 42-cell hint measured 56 and
        # orphaned "sessions" on its own line (design review D2). "Switch" alone
        # keeps the row whole at 80 (49 cells) and still carries PERSIST_HINT
        # verbatim rather than a fifth paraphrase; the command name beside it
        # already says what is being switched. The `<provider>/<id>` shape it
        # used to show moved to the tip pool, which has the room (`welcome.TIPS`).
        # `/model saved` is not here for the same reason: the notice a bare
        # `/model` prints is the surface with room for the third command.
        f"Switch; {PERSIST_HINT}",
        aliases=("models",),
        desktop_destination="session.model",
    ),
    # Next to `/model` because it is the same question one level down: which
    # model, and then how hard it thinks.
    #
    # NOT an echo. The argument is a setting, not words the model is given, and
    # the receipt names the resulting level — the durable fact — where the typed
    # word is only how it was reached. Exactly `/approvals`' rule.
    SlashCommand(
        "effort",
        "Show or set reasoning effort (shift+tab cycles)",
        # OPTIONAL: the space offers this model's rungs, and a bare `/effort`
        # still prints the ladder with the current one marked. The list is what
        # the printed ladder could never be — the rungs are OFFERED rather than
        # transcribed by hand from a line of prose.
        arguments=ArgumentMode.OPTIONAL,
        desktop_destination="session.effort",
    ),
    # Beside `/effort` because they are the two dials on the SAME request, and a
    # user comparing "make it quicker" against "make it think less" should find
    # them adjacent. They are not the same axis: effort changes how hard the
    # model thinks, fast mode buys the identical answer sooner at a premium
    # price (`model.speed` opens with the distinction).
    #
    # NOT an echo, the same rule `/effort` and `/approvals` follow: the argument
    # is a setting rather than words the model is given, and the receipt names
    # the resulting state — the durable fact — where the typed word is only how
    # it was reached.
    SlashCommand(
        "fast",
        # Names the TRADE, not just the effect. This is the only dial in the app
        # that costs meaningfully more money, and a description promising speed
        # while omitting the premium would sell half the bargain. 47 cells, in
        # under the ~55 at which the description column wraps.
        "Toggle faster output at premium pricing",
        # OPTIONAL: bare `/fast` toggles, and the space offers on/off/status for
        # a user who wants to name the resulting state rather than flip into it.
        arguments=ArgumentMode.OPTIONAL,
        desktop_destination="session.fast",
    ),
    # NOT an echo, same rule as `/approvals`: the argument is a setting, and
    # the receipt names the theme that ended up in force — strictly more than
    # the typed word, which may have been an abbreviation the matcher resolved.
    SlashCommand(
        "theme",
        # Terse like `/model`'s: the description column wraps past ~55 cells.
        # "live preview" is the half the list cannot teach on its own — a user
        # has to know arrowing is safe before they will browse with it.
        "Switch color theme; arrows preview live",
        aliases=("themes",),
        # OPTIONAL: a bare `/theme` reports the active theme, and the space
        # offers every registered ramp with the current one marked.
        arguments=ArgumentMode.OPTIONAL,
        desktop_destination="appearance",
    ),
    # The listing is the receipt.
    SlashCommand(
        "provider", "List providers and their login/usage state", desktop_destination="providers"
    ),
    # The PAGE is the receipt, the same rule `/usage` and `/analytics` follow:
    # it replaces the transcript region, so a notice printed behind it would
    # only be readable after leaving. Beside `/theme` and `/search` because it
    # is the surface that contains both of them.
    SlashCommand(
        "settings",
        "Change every setting on one page",
        aliases=("config",),
        desktop_destination="settings",
    ),
    SlashCommand(
        "search",
        "Configure web search providers and load balancing",
        desktop_destination="settings.search",
    ),
    # The listing is the receipt.
    SlashCommand("accounts", "List stored credentials", desktop_destination="accounts"),
    # The listing is the receipt — the cascade tree IS the whole answer, and
    # the command takes no argument to restate.
    #
    # NO `failover` singular alias, despite it being an equally natural spelling:
    # the picker sizes its name column on the widest `/name  /alias` pair, and
    # `/failovers  /failover` (21 cells) is 3 wider than the current widest, so
    # the alias permanently narrows the DESCRIPTION column for every command at
    # every width (it truncated `List all commands` on the 41-cell frame that
    # `test_descriptions_come_back_above_the_collapse_width` pins). The singular
    # still reaches this command through the picker's prefix match, which is the
    # cheap half of what an alias would buy.
    SlashCommand(
        "failovers",
        "Show the model failover cascade and what is serving",
        desktop_destination="session.failovers",
    ),
    # The panel is the receipt — the row the owner reported as noise.
    SlashCommand("usage", "Show provider usage quota", desktop_destination="usage"),
    SlashCommand(
        "context",
        "Show prompt, tool-schema and message token usage",
        desktop_destination="session.context",
    ),
    # The screen it opens IS the receipt (same rule as `/usage`). The argument
    # names WHICH analytics view; today only `usage` exists, so the list is an
    # OFFER — a bare `/analytics` opens the usage view rather than doing
    # nothing, which is what makes the single-view case feel like one command
    # while leaving room for `/analytics cost`, `/analytics latency`, ... later.
    SlashCommand(
        "analytics",
        "Aggregated token-consumption analytics across all sessions",
        arguments=ArgumentMode.OPTIONAL,
        desktop_destination="analytics",
    ),
    # THE exception. `/goal <text>` is the one command whose argument reaches
    # the model: the goal rides the system prompt's volatile tail on every later
    # turn (`Session.set_goal`). Words the model is given are the transcript's
    # subject matter, and they belong to the user, so they get a user row rather
    # than being paraphrased into a system notice. `_cmd_goal` writes that row
    # itself, only on the branch that actually stored something — the flag is
    # the permission, not the trigger.
    SlashCommand(
        "goal",
        "Show, set, or clear the session goal",
        echo=True,
        consumes_prompt=True,
        desktop_destination="session.goal",
    ),
    # Not an exception: LOOP_PROMPT is app-authored, not the user's words, and
    # `_loop_worker` already labels every iteration it starts (`· loop 1/3`), so
    # no agent output here is left unattributed. `echo=False` suppresses the
    # command's own slash-echo row; the live path additionally registers
    # LOOP_PROMPT in `_pending_user_echoes` (in `_loop_worker`) so the
    # session's user MessageStartEvent is consumed silently rather than
    # painted — two different receipts for two different events (the typed
    # command, and the prompt the turn later announces).
    SlashCommand(
        "loop",
        # Advertises BOTH forms so the goal mode is discoverable from the palette
        # without reading the source: free text is a goal a judge decides is met,
        # a number is a bounded iteration count.
        "Loop toward a goal: /loop <goal text>, or /loop <n> for n turns",
        consumes_prompt=True,
        desktop_destination="session.loop",
    ),
    # NOT an exception, and the reason IS the feature. The question does reach
    # the model, but only for one off-the-record request that never joins the
    # conversation (`SessionProtocol.complete_aside`) — so a user row in the
    # ledger would be the one trace the aside promises not to leave, and would
    # still be sitting there after Esc claimed to have thrown the exchange
    # away. The card is the receipt; `^f` inside it is how an exchange gets a
    # row, as a real turn rather than an echo.
    SlashCommand(
        "btw",
        "Ask a side question off the record (esc closes it)",
        consumes_prompt=True,
        desktop_destination="session.aside",
    ),
    # NOT an echo, and the receipt is the reason. The pass narrates itself
    # through the same `compacting context…` / `context compacted · 128.4k →
    # 21.9k tokens` notices the automatic one emits, and a refusal says why it
    # did not run — nothing typed here reaches the model, so a user row above
    # that would only restate the word.
    SlashCommand("compact", "Compact the context now", desktop_destination="session.compact"),
    # The kill switch (design §12): bare stops THIS session, `/stop <target>`
    # stops another one (the `send` target vocabulary: name / session id /
    # pid / substring), `/stop all` arms a 10 s window and a repeat executes.
    # The receipt is the stop line itself, so no echo: nothing here reaches
    # the model, and the receipt names what was stopped — strictly more than
    # the typed word.
    SlashCommand(
        "stop",
        "End this session, another by name/pid, or all — /resume reopens it",
        arguments=ArgumentMode.OPTIONAL,
        desktop_destination="sessions.stop",
    ),
    # The receipt states the resulting mode, which is the durable fact; the
    # typed argument is only how it was reached.
    SlashCommand(
        "approvals",
        # Names the SCOPE word, not the modes: the modes are rows in the list a
        # space opens, where they can carry which one is live and which one the
        # next launch will use. `default` is the half a list cannot teach on its
        # own, because a user has to suspect it exists to go looking for it —
        # the same job `PERSIST_HINT` does on `/model`.
        "Show or set tool approval mode; add default to keep it",
        arguments=ArgumentMode.OPTIONAL,
        desktop_destination="session.approvals",
    ),
    # The listing is the receipt.
    SlashCommand("skills", "List loaded skills", desktop_destination="skills"),
    # The listing is the receipt; the subcommands configure servers or manage
    # the OAuth grants startup never opens a browser for. OPTIONAL: bare
    # `/mcp` answers something (the listing), so Enter still sends it and the
    # subcommand list is an offer for the next keystroke, matching
    # `/approvals`. The description names the SHAPE rather than all six verbs —
    # the argument picker enumerates them with a line of help each, which is
    # more than this one truncating row can carry.
    SlashCommand(
        "mcp",
        "List MCP servers; add/remove one, or manage an OAuth grant",
        arguments=ArgumentMode.OPTIONAL,
        desktop_destination="mcp",
    ),
    # The flow narrates itself: URL block, progress notices, then success.
    # REQUIRED for both: bare, neither has anything to run — the provider list
    # IS the command, which is why completing the word opens it instead of
    # submitting a no-op over the list it just drew.
    SlashCommand(
        "login",
        "Authenticate a provider",
        arguments=ArgumentMode.REQUIRED,
        desktop_destination="auth.login",
    ),
    # The worker reports the removal, naming the provider.
    SlashCommand(
        "logout",
        "Remove stored provider credentials",
        arguments=ArgumentMode.REQUIRED,
        desktop_destination="auth.logout",
    ),
    # Uses this computer's Radient login and user service. The final setup or
    # status notice is its receipt, so the command has no model-facing echo.
    #
    # NOT offered on the desktop: `desktop_destination` is deliberately unset,
    # which keeps it out of `command_catalogue()` and therefore out of the
    # command palette and the slash popup. It previously advertised
    # `radient.mobile`, a destination the renderer has no adapter for, so the
    # command was fully discoverable and then dead-ended in an error naming an
    # internal id (code review 8, design D4, UX U8).
    #
    # Offered-but-broken is the worst of the three options. The remaining two
    # are to build it or to withhold it, and building it is not a remediation:
    # phone provisioning has no proxy behind `/v1/desktop/radient` (which
    # serves account, billing, usage and agent catalogue only), so a desktop
    # host would have to invent an upstream contract. The terminal command is
    # untouched and still does the whole job.
    SlashCommand(
        "mobile",
        "Radient phone access: status, enable, stop, billing",
    ),
    # The listing (or the masked paste prompt) is the receipt. The argument is
    # a KEY NAME, never the secret, so echoing it would only restate the
    # notice that already names what was stored or forgotten.
    SlashCommand(
        "credential",
        "Hand the agent a secret it can use but never read; paste is masked",
        aliases=("cred",),
        arguments=ArgumentMode.OPTIONAL,
        desktop_destination="session.credential",
    ),
    # NOT an echo. `/team <name> <request>` does reach the model, but as
    # the request text itself via `_submit_prompt`, which already writes
    # the user row. Echoing the slash line would duplicate it. Bare
    # `/team` is a listing and the listing is the receipt.
    SlashCommand(
        "team",
        "List teams, chart a team's org, or send a request to a team's manager",
        aliases=("teams",),
        arguments=ArgumentMode.OPTIONAL,
        # The request AFTER the team name is a prompt the manager is given, so an
        # inline `/team` reassembles to the front (name from the autofill, the
        # draft as the request) rather than eating the draft as the name.
        consumes_prompt=True,
        desktop_destination="session.team",
    ),
    # Same echo reasoning as `/team`, which this command mirrors surface for
    # surface: bare `/agent` is a listing (the listing is the receipt), a
    # named attach prints a notice, and `/agent <name> <message>` reaches the
    # model as the MESSAGE via `_submit_prompt`, which writes the user row.
    # This is the USER-driven way to adopt a role/specialist mid-session; the
    # `agent` TOOL is the model-driven way to author and inspect them — two
    # surfaces over one registry, not a collision.
    SlashCommand(
        "agent",
        # D4: "agents", standardizing the noun with the listing header and the
        # attach/detach notices rather than saying "agent profiles" here.
        "List agents, or speak to this session as one",
        aliases=("agents",),
        arguments=ArgumentMode.OPTIONAL,
        # The message AFTER the agent name is a prompt the persona is given, so
        # an inline `/agent` reassembles to the front like `/team`.
        consumes_prompt=True,
        desktop_destination="session.agent",
    ),
]


def slash_command_for(text: str) -> SlashCommand | None:
    """The registry entry a typed line invokes, or ``None`` if nothing matches.

    Resolves through :attr:`SlashCommand.names`, so an alias answers with the
    same entry as its primary name — ``/quit`` must not get a different echo
    policy from ``/exit`` just because it was spelled the other way.

    Matching is case-insensitive because registry names are lowercase and this
    is the ONE resolver both the echo permission and
    :meth:`OperatorApp._run_slash_command`'s dispatch read. Only one function
    ever decides what a typed word means, so ``/Usage`` cannot echo as one
    command and run as another.
    """
    token = text.split(maxsplit=1)[0].lower() if text.strip() else ""
    if not token.startswith("/"):
        return None
    name = token[1:]
    return next((entry for entry in SLASH_COMMANDS if name in entry.names), None)
