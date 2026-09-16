"""Canonical slash metadata shared by terminal and headless frontends.

Keep this registry free of widget imports: creating a runtime's first canonical
snapshot must not import Textual just to learn command names and capabilities.
The TUI reexports these objects for existing callers.

Argument and echo semantics are the SAME on every host; only the action
destination and its presentation differ. ``desktop_destination`` carries that
one host-specific fact per entry so a desktop surface never needs a second
command registry to diverge from — an entry without one is not offered on the
desktop at all (see ``/mobile``).
"""

from local_operator.tui.autocomplete import ArgumentMode, ArgumentShape, SlashCommand

#: ONE sentence for ONE instruction, carried verbatim by every surface that
#: mentions ``/model default``: the bare-``/model`` notice, the switch receipt,
#: the model picker's footer and the ``/help`` row. They used to say it four
#: different ways within two keystrokes of each other — "saves this provider and
#: model as the boot default", "to make it the boot default", "saves the boot
#: default", "persists it" — so a user met a new phrasing on every surface
#: instead of learning one string.
#:
#: Sized by its TIGHTEST site. The picker footer truncates at the card's width
#: and this clause sits after the access note, so it has to be complete and short.
#: "Saves provider and model" named the payload but not why it mattered; "saves
#: this for new sessions" names the consequence, fits the same slot, and includes
#: the article the clipped phrase lacked. The budget arithmetic is in the last
#: paragraph below — one set of figures, so nobody "fixes" the string toward a
#: stale number.
#:
#: REPOINTED BACK at `/model default`, after #369 briefly pointed it at a `d`
#: key on a picker row. That key is gone: inside a filter every printable
#: character belongs to the query, so a `d` that saved config on an empty query
#: and narrowed the list otherwise was a mode with nothing on screen to mark it.
#: The ambiguity #369 reported is closed by the COMMAND being unambiguous — a
#: bare `/model default` writes the model the session is already on and switches
#: nothing — not by moving the write onto a keystroke.
#:
#: ONE route named here, not both. The second route (the `/settings` model rows)
#: does not fit: this string is sized by the picker footer, whose budget is 43
#: cells at 50 columns (card width minus `_GUTTER_CELLS` + `_EDGE_MARGIN`), and
#: this clause is 42. Any "; /settings too" tail measures 57 and truncates
#: mid-word at the one width where the instruction most needs to survive whole.
#: So the footer gets the command and the roomier surfaces get the pair: the
#: bare-`/model` notice names `/settings` in `_persist_hint_notice`, which wraps
#: instead of truncating, and the `/help` row is reachable at any width.
PERSIST_HINT = "/model default saves this for new sessions"


#: Slash commands handled synchronously before any prompt is sent. One
#: registry entry per command; aliases live on the entry (TUI-014).
#:
#: ``echo`` says whether running the command leaves a user row in the visible
#: ledger. It USED to be unconditional, on the reasoning that typing a command
#: is the same visible commitment as sending a prompt. That reasoning had the
#: wrong subject: a prompt is echoed because the transcript is the only record
#: of what the user said, whereas every handler below already reports what it
#: did — ``/usage`` opens the panel that IS the answer, ``/provider`` prints the
#: list, ``/model p/id`` names both labels — so the echo was a row restating a
#: row underneath it. The reading record kept the keystrokes and gained nothing.
#:
#: So the test is not "did the user commit to something" but "would the receipt
#: be missing something without it", and exactly one thing qualifies: an
#: argument that becomes part of what the MODEL is told. Comment per entry
#: below; the table is pinned in ``tests/unit/tui/test_slash_echo.py``.
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
    SlashCommand(
        "new",
        "Start a new conversation",
        # The word is the new-session picker's SELECTION (`selected=args`), so a
        # whole-draft `/new foo` is that picker's gesture; a sentence after the word
        # is prose.
        argument_shape=ArgumentShape.WORD,
        desktop_destination="sessions.new",
    ),
    # In-process reboot cannot load a replaced wheel; this command exists so
    # ``/update`` is not the only way to pick up new code. Same relaunch
    # helper as ``/update`` — the conversation comes back via ``--resume``.
    SlashCommand(
        "reload",
        "Relaunch this conversation on the current install",
        # The word is the reload picker's SELECTION (`selected=args`).
        argument_shape=ArgumentShape.WORD,
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
        # The word is the resume picker's SELECTION (`selected=args`) — a session id
        # or name, one token; a sentence after the word is prose.
        argument_shape=ArgumentShape.WORD,
        desktop_destination="sessions.resume",
    ),
    # Beside the session-transition family because it changes the same session
    # rather than replacing it: `/new` discards the conversation, `/resume`
    # moves to another, and this one keeps the conversation and changes WHERE
    # it works. Placed after `/resume` and before `/rename` because those two
    # are the other commands that alter a session in place.
    #
    # NOT an echo, the rule `/approvals` and `/rename` follow: the argument is
    # a setting rather than words the model is given, and the receipt names the
    # directory that ended up in force — strictly more than the typed words,
    # which may have been `~/x` or a relative path the app resolved.
    SlashCommand(
        "move",
        # Names the two ways to answer it, because the argument form is the
        # half a picker cannot teach: a user who has only ever seen the list
        # will not guess that a path can be typed straight in. 44 cells, inside
        # the ~55 at which the description column wraps (see `/model`, where a
        # wrapping row renders a phantom command name in `/help`).
        "Change this session's working directory",
        # OPTIONAL: a bare `/move` opens the picker, which is the discoverable
        # route, and the space offers the same suggestions for a user who would
        # rather type. Enter on the bare command still does something useful,
        # which is exactly the distinction `/approvals` and `/effort` draw
        # against `/login`'s REQUIRED.
        arguments=ArgumentMode.OPTIONAL,
        # The desktop EXECUTES the path (`argsBehavior: "execute"`), and a path is
        # arbitrary text — two words included — so every whole-draft `/move …` is the
        # command.
        argument_shape=ArgumentShape.ANY,
        desktop_destination="session.move",
    ),
    # Beside `/resume` because it names the thing the picker lists. NOT an echo:
    # the argument is the conversation's own label — it goes on the band and the
    # terminal tab, never into anything the model is told — and the receipt
    # quotes the title that ended up in force, which is strictly more than the
    # typed words (the store trims and caps them).
    #
    # `/title` is an ALIAS, not a second entry, and that is the whole design of
    # the refresh word. The two things a user wants to do to a conversation's
    # name — say what it is, or ask for it to be worked out again — are one
    # subject, and splitting them across `/rename` and a sibling `/retitle`
    # would put two nearly-identical rows in `/help` whose difference is four
    # letters in the middle of the word. One entry with `refresh` as its
    # argument states the relationship instead: `/title <words>` is the
    # imperative, `/title refresh` is the request, and a bare `/title` reports.
    #
    # OPTIONAL rather than NONE now that an argument has a value list: the space
    # offers `--refresh` to a user who does not know the capability exists, and
    # Enter on the bare command still reports the current name — the exact
    # distinction `/approvals` and `/effort` draw against `/login`'s REQUIRED.
    # Free typing is unaffected, so an arbitrary title still submits.
    SlashCommand(
        "rename",
        # 43 cells, inside the ~55 at which the description column wraps and
        # renders a phantom command name in `/help` (see `/model`, `/theme`);
        # composed with the 20-cell name column that is a 63-cell row, one line
        # at 80 columns.
        # The `--refresh` flag has to be HERE because the help table is where a
        # user learns the command exists at all, and the capability it names is
        # the reason this entry changed. The INVOCATION is spelled out rather
        # than the bare flag alone because this is the only surface that teaches
        # the words to TYPE: "or --refresh the name" reads as though the flag
        # takes "the name" as a value — and `parse_title_arg("--refresh the
        # name")` really does store that literal as the conversation's title,
        # so the misreading is reachable, not pedantic. The picker needs no
        # such help: it supplies the argument itself, so its row teaches the
        # flag alone.
        "Name this conversation, or /title --refresh",
        aliases=("title",),
        arguments=ArgumentMode.OPTIONAL,
        # The title is arbitrary text: `native_action` pre-fills the form's text field
        # with it and the OWNER dispatch hands it to the runtime.
        argument_shape=ArgumentShape.ANY,
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
        # and being billed for a turn in the fork. `docs/fork.md` carries
        # the rest.
        "Branch this chat; --switch here, --window elsewhere; <message> starts work",
        echo=True,
        consumes_prompt=True,
        # The trailing text is the branch's starting instruction, so a message
        # that merely OPENS with `/fork` is still prose — the desktop half of the
        # same fact `consumes_prompt` carries for the TUI (see the field's
        # docstring).
        prefixes_text=True,
        # ANY, not the NONE default: the command owns its trailing text
        # whatever it says (see ArgumentShape's precedence note).
        argument_shape=ArgumentShape.ANY,
        desktop_destination="session.fork",
    ),
    # The switch receipt names the old AND new label — strictly more than the
    # typed selector, which may have been elided to `default`.
    SlashCommand(
        "model",
        # A `/help`-specific carrier, NOT `PERSIST_HINT` verbatim: the footer
        # hint is sized by the picker (42 cells), and every ≤12-cell lead that
        # kept it whole here left the row a fragment (`Switch;` was 49 cells
        # and stayed whole, but read as a broken sentence beside the
        # `Switch color theme; …` neighbour — design review round 2, D7).
        # `Switch model; ` + the hint measured 56 cells at 80 columns and
        # orphaned "sessions" on its own line (D2), so the answer is a shorter
        # sentence of its own: name the thing being switched and drop the
        # "saves this" carrier the footer needs for its 43-cell budget. The
        # notice a bare `/model` prints is the surface with room for the full
        # three-route sentence; `/help` needs only the persist command's name.
        "Switch model; /model default saves it for new sessions",
        aliases=("models",),
        # The trailing selector is a value this command owns, not the start of a
        # message: `/model gpt-5` is the model command, while
        # `/mcp logout seems to cause a crash` is prose. See the field docstring.
        prefixes_text=True,
        # ANY, not the NONE default: the command owns its trailing text
        # whatever it says (see ArgumentShape's precedence note).
        argument_shape=ArgumentShape.ANY,
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
        # Trailing text is the level name, a value this command owns.
        prefixes_text=True,
        # ANY, not the NONE default: the command owns its trailing text
        # whatever it says (see ArgumentShape's precedence note).
        argument_shape=ArgumentShape.ANY,
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
        # WORD: ONE whitespace-free token is the command, and the picker's
        # on/off list is PRESENTATION (`choices` on the field), not the admission
        # rule — so `/fast maybe` is refused too, because the desktop runs the
        # whole-draft form whatever the token says. The vocabulary this shape may
        # carry (`command_argument_words`) is empty here and honoured by both
        # arms, so declaring one later is a one-line change rather than a second
        # rule.
        argument_shape=ArgumentShape.WORD,
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
        # Trailing text is the theme name, a value this command owns — the
        # desktop's `inline` list, which is the second half of the union the
        # field docstring describes.
        prefixes_text=True,
        # ANY, not the NONE default: the command owns its trailing text
        # whatever it says (see ArgumentShape's precedence note).
        argument_shape=ArgumentShape.ANY,
        desktop_destination="appearance",
    ),
    # The listing is the receipt.
    SlashCommand(
        "provider",
        "List providers and their login/usage state",
        # The word is the provider the panel opens on (`selection=args`).
        argument_shape=ArgumentShape.WORD,
        desktop_destination="providers",
    ),
    # The PAGE is the receipt, the same rule `/usage` and `/analytics` follow:
    # it replaces the transcript region, so a notice printed behind it would
    # only be readable after leaving. Beside `/theme` and `/search` because it
    # is the surface that contains both of them.
    SlashCommand(
        "settings",
        "Change every setting on one page",
        aliases=("config",),
        # The word is the settings FILTER (`filter=args`).
        argument_shape=ArgumentShape.WORD,
        desktop_destination="settings",
    ),
    SlashCommand(
        "sidebar",
        # `focus` is named because it is the only way into the list's keyboard
        # mode now that a pointer press no longer takes it (design round D1),
        # and because the panel's own footer advertises it: a description that
        # omitted the argument told a user the command had none. 49 cells — the
        # description column wraps past ~55, which would render a phantom
        # command name in `/help` (see the `/copy` note above).
        "Show or hide conversations; 'focus' keys the list",
        desktop_destination="sessions.sidebar",
    ),
    SlashCommand(
        "search",
        "Configure web search providers and load balancing",
        # `filter=args`: the route's presentation payload carries the word. The
        # desktop's own `/search` fixes that filter to `web-search` and reads no
        # word, so this is one of the two DELIBERATE EXCEPTIONS to the criterion
        # (see `ArgumentShape`) — kept because a whole-draft `/search <word>` is a
        # control the composer runs, while a sentence after the word is prose.
        argument_shape=ArgumentShape.WORD,
        desktop_destination="settings.search",
    ),
    # The listing is the receipt.
    SlashCommand(
        "accounts",
        "List stored credentials",
        # `selection=args` — the account/provider the panel opens on.
        argument_shape=ArgumentShape.WORD,
        desktop_destination="accounts",
    ),
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
    SlashCommand(
        "usage",
        "Show provider usage quota",
        # The word is the view the panel opens on (`selection=args`).
        argument_shape=ArgumentShape.WORD,
        desktop_destination="usage",
    ),
    SlashCommand(
        "context",
        "Show prompt, tool-schema and message token usage",
        # NONE, and the criterion is why: a shape is published only where the
        # desktop's command path USES the trailing text. This one DROPS it —
        # `_slash_result` calls `_context_slash_result(SlashResult)` with no args
        # (`app.py`) and `native_action` has no branch, so `/context x` would run
        # the command and silently discard `x` exactly as `/compact hello` does.
        # That is the operator's own complaint, so the text is prose.
        desktop_destination="session.context",
    ),
    # `session.diagnostics` rather than a reuse of `analytics`: both read the
    # same ledger, but this one is scoped to the CURRENT session id and the
    # analytics view is explicitly all-sessions (`daily_scope`), so pointing
    # them at one destination would make the desktop render a whole-install
    # total for a command documented as current-session only. The host already
    # has the session-scoped read it needs (`/v1/desktop/analytics?session_id=`);
    # like `/context`, it is a read-only view with no owner execution, so it
    # needs no `native_action` branch or `OWNER_COMMANDS` entry.
    SlashCommand(
        "session",
        "Current-session usage, cost and request diagnostics",
        desktop_destination="session.diagnostics",
    ),
    # Beside `/session` because it is the same family — a read-only diagnostic
    # screen with no owner execution — but a WIDER scope: `/session` describes
    # one conversation's spend, `/info` describes the INSTALL and every runtime
    # on the machine. It is the screen a user is asked to paste into an issue,
    # which is why it takes no argument at all: there is one answer, and making
    # someone name it would be a gate in front of a command that has one.
    #
    # No `desktop_destination` used to live here, because every field it
    # renders is a fact about THIS process and THIS host (install prefix, pids,
    # RSS, the in-memory subagent graph), so a desktop surface pointed at it
    # would describe the machine the host runs on rather than the one the user
    # is asking about. The destination now EXISTS and that caveat survives as
    # the PANEL'S HOST LABEL rather than as a reason to withhold the row: the
    # desktop normally talks to a loopback backend on the same machine, but the
    # transport is a URL (`LOCAL_OPERATOR_DESKTOP_BACKEND_URL`), so the panel
    # names what it is showing as "the machine this app is connected to". The
    # destination is `info`, served by `GET /v1/desktop/info`.
    #
    # Like `/context` it needs no `native_action` branch or `OWNER_COMMANDS`
    # entry: it is a read-only view with no owner execution.
    SlashCommand(
        "info",
        # "running sessions", not "sessions": `/analytics` describes past
        # CONVERSATIONS and this describes live PROCESSES, and both descriptions
        # sit in one picker where the shared word read as the same thing (UX
        # round 1, U9). One word buys the distinction.
        "Install, version, and running sessions on this machine",
        desktop_destination="info",
        # NO ALIASES, deliberately — `version`/`about` were added for UX round 1
        # U8 and reverted the same round. The claim that alias rows "cost no
        # space" is false here: the picker measures ONE name column across every
        # row, so `/info  /version  /about` at 23 cells became the widest entry
        # (previous max `/settings  /config`, 18) and took those 5 cells from
        # every other command's description. Measured consequence at 80 columns:
        # `/help`'s description collapsed to `List all co…`, and the `/model`
        # and `ctrl+v` rows in `/help` wrapped. Discoverability for one command
        # is not worth truncating the descriptions of all of them, and `/help`
        # already lists this one. See `test_descriptions_come_back_above_the_
        # collapse_width`, which is the guard that caught it.
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
        # The word is the view to open (`selection=args`).
        argument_shape=ArgumentShape.WORD,
        desktop_destination="analytics",
    ),
    # The argument becomes both a standing objective and an ordinary user
    # message. Submission owns its user row; status/clear never start a turn.
    SlashCommand(
        "goal",
        "Set the goal and start work; show or clear it",
        echo=True,
        consumes_prompt=True,
        # The trailing text is the objective, this command's own argument.
        prefixes_text=True,
        # ANY, not the NONE default: the command owns its trailing text
        # whatever it says (see ArgumentShape's precedence note).
        argument_shape=ArgumentShape.ANY,
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
        # Advertises all THREE forms so each is discoverable from the palette
        # without reading the source: free text is a goal a judge decides is met,
        # a number is a bounded iteration count, and `stop` is the escape hatch —
        # which used to appear only in a launch notice or an already-running
        # refusal, i.e. after the user needed it (UX round 1, U6).
        "Loop toward a goal: /loop <goal text>, /loop <n>, or /loop stop to cancel",
        consumes_prompt=True,
        # The trailing text is the loop instruction or count, this command's own
        # argument.
        prefixes_text=True,
        # ANY, not the NONE default: the command owns its trailing text
        # whatever it says (see ArgumentShape's precedence note).
        argument_shape=ArgumentShape.ANY,
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
        # The trailing text is the aside question, this command's own argument.
        prefixes_text=True,
        # ANY, not the NONE default: the command owns its trailing text
        # whatever it says (see ArgumentShape's precedence note).
        argument_shape=ArgumentShape.ANY,
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
        # The word is the same TARGET vocabulary the TUI's `/stop <target>` takes,
        # and the second of the two DELIBERATE EXCEPTIONS to the criterion (see
        # `ArgumentShape`): the desktop's picker owns `targets=[session_id]` and
        # drops the word, but a whole-draft `/stop <word>` is a control the
        # composer runs while a sentence after the word is prose.
        argument_shape=ArgumentShape.WORD,
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
        # Trailing text is the mode name, a value this command owns.
        prefixes_text=True,
        # ANY, not the NONE default: the command owns its trailing text
        # whatever it says (see ArgumentShape's precedence note).
        argument_shape=ArgumentShape.ANY,
        desktop_destination="session.approvals",
    ),
    # The listing is the receipt.
    SlashCommand(
        "skills",
        "List loaded skills",
        # `selection=args`.
        argument_shape=ArgumentShape.WORD,
        desktop_destination="skills",
    ),
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
        # The route parses `<subcommand> [name]` against `MCP_SUBCOMMANDS` and
        # `SERVER_NAME_RE`, so `/mcp logout` is the command and `/mcp logout seems to
        # cause a crash` is a message.
        argument_shape=ArgumentShape.SUBCOMMAND,
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
        # The route resolves the word against the provider registry — a known id is
        # the command, an unknown one is prose.
        argument_shape=ArgumentShape.PROVIDER,
        desktop_destination="auth.login",
    ),
    # The worker reports the removal, naming the provider.
    SlashCommand(
        "logout",
        "Remove stored provider credentials",
        arguments=ArgumentMode.REQUIRED,
        # Same lookup as `/login`: a known provider id is the command, an unknown one
        # is prose.
        argument_shape=ArgumentShape.PROVIDER,
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
    SlashCommand("mobile", "Radient phone access: status, enable, stop, billing"),
    # The listing (or the masked paste prompt) is the receipt. The argument is
    # a KEY NAME, never the secret, so echoing it would only restate the
    # notice that already names what was stored or forgotten.
    SlashCommand(
        "credential",
        # Describes the GESTURE, and describes it FIRST. The old copy ("paste is
        # masked") named only the clipboard route, so the operator who TYPED the
        # secret — the obvious human gesture — had no reason to expect it to
        # work, and in fact it did not: the line fell through to this command
        # with the secret as its argument.
        #
        # THE LEAD IS LOAD-BEARING, not a style choice. The picker ellipsizes the
        # description's TAIL, and measured on the real app it keeps ~47
        # characters at 100 columns and fewer at 80 — the previous copy rendered
        # as "Hand the agent a secret it can use but never re…", dropping its
        # own verb. So the two words that tell the operator the mode exists
        # ("Type or") have to come before the guarantee, because the guarantee is
        # the half that survives cropping either way.
        #
        # AND IT IS SHORT ENOUGH TO SURVIVE. The first attempt at this copy still
        # cropped before its own promise at EVERY width measured, 120 included
        # ("…can use but never rea…" — a truncated reassurance dangling mid-word,
        # which an operator completes wrongly). Measured, not estimated: the row
        # keeps ~31 cells at 60 columns and ~47 at 100, and the budget is not
        # monotonic in width because the transcript gutter indents it more as the
        # terminal grows (design round 1, D5; QA round 1, Q2). At 44 cells this
        # one paints whole from 80 columns up and still leads with the gesture
        # everywhere below that. The SPACE is named because it is what arms the
        # mode and nothing else on screen says so (UX round 1, U5).
        "Type or paste a secret after a space; masked",
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
        # Name AND request are this command's own argument: `/team ops fix this`
        # is the team command, while a draft that merely opens with the word is
        # prose. See the field docstring.
        prefixes_text=True,
        # ANY, not the NONE default: the command owns its trailing text
        # whatever it says (see ArgumentShape's precedence note).
        argument_shape=ArgumentShape.ANY,
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
        # Name AND message are this command's own argument, exactly as `/team`.
        prefixes_text=True,
        # ANY, not the NONE default: the command owns its trailing text
        # whatever it says (see ArgumentShape's precedence note).
        argument_shape=ArgumentShape.ANY,
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


def command_prefixes_text(spec: SlashCommand) -> bool:
    """Whether text typed after ``spec``'s word is an argument the command owns.

    The union of the two vocabularies a host can follow the word with: free text
    destined for a model (``consumes_prompt``) and a value chosen from a list
    (``prefixes_text``, which the desktop carries for exactly that reason). One
    predicate, read by the messages endpoint's admission test and by the desktop
    composer's planner.

    It exists so the route and the planner cannot answer "is this trailing text
    the command's argument" differently. That second decision is a named defect
    class here: the composer decided with the caret in hand while the endpoint
    decided on the leading word, so a prose draft could be planned as prose and
    then refused by the route forever — a refusal no resend clears.

    The TWO halves only. A command's text can also be one the desktop VALIDATES
    or FORWARDS without either a prompt or a list — ``/mcp logout`` is a
    subcommand, ``/login openai`` a provider id — and that third source is
    :attr:`SlashCommand.argument_shape`, read by
    :func:`command_argument_is_used`. This predicate stays the composer's
    vocabulary on its own: a caller asking "can I complete text here inline"
    wants exactly these two.
    """
    return spec.consumes_prompt or spec.prefixes_text


def _is_single_word(args: str, words: tuple[str, ...]) -> bool:
    """One whitespace-free token, optionally drawn from ``words``.

    ``words`` empty means "any single token" — the published reading of an empty
    ``argument_words``. The vocabulary is a parameter rather than a second lookup
    so this arm and the two below cannot disagree with the catalogue about which
    words a shape accepts: both read :func:`command_argument_words`.
    """
    parts = args.split()
    if len(parts) != 1:
        return False
    return not words or parts[0] in words


def command_argument_words(spec: SlashCommand) -> tuple[str, ...]:
    """The vocabulary a shape's first token must come from; empty means any word.

    THE ONE derivation, read by the validators below AND by the desktop
    catalogue's ``argument_words``, so the word a renderer accepts and the word
    this endpoint accepts cannot drift. A shape whose vocabulary is live or
    elsewhere — the provider registry, the MCP subcommands — is resolved HERE,
    lazily, for the same reason ``slash_commands`` imports no provider SDK at
    module scope: this is the module every host imports first.
    """
    if spec.argument_shape is ArgumentShape.PROVIDER:
        from local_operator.providers.registry import known_provider_ids

        return known_provider_ids()
    if spec.argument_shape is ArgumentShape.SUBCOMMAND:
        from local_operator.session.frontend_state import MCP_SUBCOMMANDS

        return tuple(sorted(MCP_SUBCOMMANDS))
    return ()


def _is_provider(args: str) -> bool:
    """Whether ``args`` is one token naming a provider THIS INSTALL knows.

    The command route's own lookup (``get_provider_definition``, which resolves
    legacy aliases too) reached through the vocabulary the catalogue publishes,
    so ``/login openai`` and ``/login zzz`` get one answer on both paths —
    including for an alias, which a hand-listed vocabulary would have missed.
    """
    from local_operator.providers.registry import get_provider_definition

    return get_provider_definition(args.strip()) is not None


def _is_mcp_invocation(args: str, words: tuple[str, ...]) -> bool:
    """``<subcommand> [name]`` — the shape the MCP setup form exists for.

    The command route's own parse, moved here so the route and the admission
    test cannot drift: at most two tokens, the first a real subcommand (from the
    published vocabulary), the second a server name. Anything else is prose, which
    is what keeps ``/mcp logout seems to cause a crash`` (the operator's own
    draft) a message.
    """
    from local_operator.mcp.config import SERVER_NAME_RE

    parts = args.split()
    if not parts or len(parts) > 2 or (words and parts[0] not in words):
        return False
    return len(parts) == 1 or bool(SERVER_NAME_RE.fullmatch(parts[1]))


def command_argument_is_used(spec: SlashCommand, args: str) -> bool:
    """Whether the desktop USES ``args`` as ``spec``'s argument — the ONE answer.

    Read by the messages endpoint's admission test (``whole_draft_command``) and,
    for the shapes the command route validates, by that route itself
    (:func:`command_argument_refusal`), so "is this trailing text this command's
    argument" has one derivation rather than one per call site.

    Three sources, in the order they are asked: free text destined for a model
    (``consumes_prompt``), a value chosen from a list (``prefixes_text``), and the
    shape the entry declares (``argument_shape``) for text the desktop validates
    or forwards. A blank ``args`` is the command itself, which is why the first
    test is emptiness rather than a shape.

    WHY THE SHAPE IS A THIRD SOURCE RATHER THAN A WIDENING OF THE FIRST TWO, and
    it is the defect this predicate was narrowed to avoid: the two booleans
    answer "can the composer COMPLETE this text inline", and the desktop's own
    command route consumes more than that. ``/login openai``, ``/mcp logout``,
    ``/rename my thing``, ``/move ~/x`` and ``/usage on`` are all whole-draft
    controls the desktop runs today, and refusing only the completable two would
    leave each of them accepted here and planned as a command by the composer —
    a paid model turn for a control, which the guard exists to prevent.

    The SHAPE is what keeps a sentence a message: ``/usage on`` is a selector
    word while ``/usage more prose`` is a sentence, ``/mcp logout`` is a valid
    subcommand while ``/mcp logout seems to cause a crash`` is not.
    """
    if not args.strip():
        return True
    # The BOOLEANS FIRST, and they are the whole answer when either is true: the
    # shape is the third source, asked only for text neither can describe. The
    # boolean-carried rows declare ``ANY`` as well, so a consumer that reads this
    # field alone (or ORs the three facts) reaches the same answer — and ``NONE``
    # can never be misread as "the booleans decide".
    if command_prefixes_text(spec):
        return True
    shape = spec.argument_shape
    words = command_argument_words(spec)
    if shape is ArgumentShape.ANY:
        return True
    if shape is ArgumentShape.WORD:
        return _is_single_word(args, words)
    if shape is ArgumentShape.PROVIDER:
        return _is_provider(args)
    if shape is ArgumentShape.SUBCOMMAND:
        return _is_mcp_invocation(args, words)
    return False


def command_argument_refusal(spec: SlashCommand, args: str) -> str | None:
    """The command ROUTE's own 422 sentence for ``args``, or ``None`` when it forwards them.

    The route asks a narrower question than :func:`command_argument_is_used`:
    "is this a WELL-FORMED argument", not "would this text have been a control".
    Only the two shapes the route has ever validated answer, each with the
    sentence it has always used, and both decisions come from the same per-shape
    validators — so the route cannot start refusing a text the admission rule
    accepts, nor the reverse.

    The shapes it does NOT cover are deliberate rather than forgotten: for a
    SELECTOR the route forwards whatever it is given (``selection=args``), and
    tightening that here would make ``/usage more prose`` a 422 on the command
    endpoint, where today it opens the panel. That asymmetry is stated once, in
    ``ArgumentShape``, rather than restated per call site.
    """
    if not args.strip():
        return None
    if spec.argument_shape is ArgumentShape.PROVIDER and not _is_provider(args):
        return "Choose a provider in the authentication panel"
    if spec.argument_shape is ArgumentShape.SUBCOMMAND and not _is_mcp_invocation(
        args, command_argument_words(spec)
    ):
        return "Use the MCP setup form for configuration and secret references"
    return None


def whole_draft_command(text: str) -> tuple[SlashCommand, str] | None:
    """The command a WHOLE draft invokes, or ``None`` when the draft is PROSE.

    The ONE answer to "would this text have been a control rather than a
    message", stated as the whole-draft branch of the composer's own rule
    (``slash-submit.ts``): a command claims a draft only when the draft IS the
    command — its word, plus (for a command that takes one) the argument that
    follows on the same line.

    A draft containing a NEWLINE is never a whole-draft command, and that is
    load-bearing rather than incidental. The composer decides per LINE with the
    CARET in hand: a body that opens with a command word is prose there whenever
    the caret is off the command line (QA round 2 Q4's shape), so this endpoint,
    which has no caret, reads any newline as prose rather than trying to guess a
    line.

    WHY THAT IS THE RIGHT RELAXATION even though a trailing newline can be a
    `whole` plan on the current consumer: a `whole` plan runs the command IN the
    composer and never posts the draft here, so accepting it is unreachable in
    practice and cannot turn a control into paid chat. Measured both ways —
    round 2's review drove the composer head (`fix/slash-prose-and-highlight`,
    which decides from this wire field) and measured `whole` for a command word
    followed only by a newline (`/usage`, `/compact`, `/login openai`, `/mcp
    logout`) and for the whitespace-only tails (spaces, a blank line, a CRLF); the
    older planner on `main` plans `send` for those same drafts at an end-of-draft
    caret, which is exactly the refusal this rule removes. Both hosts are served
    by accepting them, and neither is served by the earlier "interior newline
    only" test, which sent them in opposite directions.

    Whitespace that is not a newline still does not make prose: leading and
    trailing spaces are stripped, so `  /compact` and `/compact   ` are the
    command, because neither turns the draft into two lines.

    Anything that is not a whole-draft command is accepted as a message.
    """
    if "\n" in text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    # The SAME boundary the tokenizer uses. `slash_command_for` splits a line on
    # arbitrary whitespace, so cutting on a LITERAL space here read
    # `/usage\rfix it` as `(usage, "it")` — one WORD-shaped token, therefore a
    # 422 — while both composers read it as `/usage` plus the two-token argument
    # `fix it` and plan `send`. Planned prose and refused here is the
    # permanent-refusal class. `split(None, 1)` leaves `/goal\tship it` and
    # `/usage\ton` deciding exactly as they did (one token either way).
    parts = stripped.split(None, 1)
    word = parts[0]
    rest = parts[1] if len(parts) == 2 else ""
    if not word.startswith("/"):
        return None
    spec = slash_command_for(word)
    if spec is None:
        return None
    if not command_argument_is_used(spec, rest):
        return None
    return spec, rest


def primary_slash_name(command: str) -> str:
    """``command`` as its registry PRIMARY name; unchanged when nothing matches.

    The bare-word counterpart to :func:`slash_command_for`, for the dispatchers
    that receive a command NAME off the wire rather than a typed line. Both
    routed dispatchers (``OperatorApp._slash_result`` and the detached
    runtime's) match string literals, so without this an ALIAS — ``/title``,
    ``/models``, ``/recall`` — falls past every branch and is answered with an
    unsupported-command refusal for a command the owner in fact implements.
    That is the same bug ``slash_command_for`` was introduced to fix on the
    LOCAL path, which had already shipped once: the registry advertises the
    alias, the picker completes it, and running it says "unknown command".

    Unknown words pass through untouched so the caller's own fallback still
    sees what was actually asked for.
    """
    entry = slash_command_for(f"/{command}")
    return entry.name if entry is not None else command
