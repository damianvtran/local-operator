"""Which slash commands OWN the text typed after their word, and which do not.

Why this pin exists, and why it is not derived from ``ArgumentMode``: the
messages endpoint refuses a draft only when the draft IS a command, and the
"is the trailing text this command's argument" half of that rule has to be the
SAME fact the composer plans with. Two derivations is the second-decision defect
class this repo has already paid for once (the QA round 2 Q4 ``/usage\\nfix it``
bug): the composer planned a draft as prose while the endpoint refused it on the
leading word, and no resend could clear the refusal.

So the tables are pinned entry by entry, the way ``test_slash_echo.py`` pins
``echo``. A command added to ``SLASH_COMMANDS`` without an opinion fails here,
which is the only way "state your choice" can be enforced on a field that must
keep a default for the picker fixtures that have no opinion.

THREE tables, one per source of "text after this word is the command's argument":

* ``PREFIXES_TEXT_POLICY`` — the composer can COMPLETE this text inline (a
  free-text prompt, or a value chosen from a list). This is the pair of booleans
  the composer derives ``consumesText`` from.
* ``ARGUMENT_SHAPE_POLICY`` — the desktop VALIDATES or FORWARDS the text without
  either a prompt or a list (a provider id, an MCP subcommand, a selector, a
  path, a title), or REFUSES it because another surface owns it (``/credential``,
  whose typed text the command route answers with the masked-form sentence).
  Adding only the first two was this PR's round-1 MAJOR: 17
  whole-draft controls the desktop runs (``/mcp logout``, ``/login openai``,
  ``/rename x``, ``/usage on`` …) were accepted as messages, so a client whose
  command surface is off would have spent a paid turn on each.

``ArgumentMode`` is deliberately NOT the source of either. It answers "does a
space open a VALUE LIST in this terminal", which is true for ``/login``,
``/move``, ``/stop``, ``/mcp`` and ``/rename``, and false for ``/usage`` — yet
the desktop consumes text after all six. Widening to ``arguments is not
ArgumentMode.NONE`` was probed and turns the operator's own row back into a
command, which is the exact refusal this change removes.
"""

from __future__ import annotations

from local_operator.slash_commands import SLASH_COMMANDS
from local_operator.tui.autocomplete import ArgumentMode, ArgumentShape

#: The composer's inline-completable vocabulary, one row per registry entry.
#:
#: True means: a draft whose word is this command and whose trailing text is on
#: the same line IS this command, so the messages endpoint refuses it as a
#: message (a control must never become paid model chat) and the composer runs
#: it as a command.
PREFIXES_TEXT_POLICY = {
    # --- commands whose trailing text is FREE TEXT destined for a model -------
    # (these already carried `consumes_prompt`; the new field is the desktop's
    # half of the same fact, and the test below asserts the implication)
    # The trailing text becomes both the standing objective and a user message.
    "goal": True,
    # A loop instruction, an iteration bound, or `stop`.
    "loop": True,
    # The aside question, off the record.
    "btw": True,
    # The instruction the forked branch starts on.
    "fork": True,
    # The name AND the request the manager is given.
    "team": True,
    # The name AND the message the persona is given.
    "agent": True,
    # --- commands whose trailing text is a VALUE chosen from a list -----------
    # The desktop presents these as a picker, so the typed text narrows the list
    # rather than starting a message. `consumes_prompt` is False for all four
    # (a selector is not a prompt), which is exactly why the union predicate
    # needs this second vocabulary as well.
    # The model id or `default`.
    "model": True,
    # The reasoning-effort rung.
    "effort": True,
    # The approval mode, or `default` to persist it.
    "approvals": True,
    # The colour theme name.
    "theme": True,
    # --- everything else: the trailing text is prose, never an argument -------
    "help": False,
    "exit": False,
    "clear": False,
    # WHICH message is chosen in the picker the command opens, so there is
    # nothing typed for the command to own.
    "copy": False,
    # Takes no argument at all, for `/copy`'s reason: WHICH url is chosen in the
    # picker the command opens.
    "links": False,
    "new": False,
    "reload": False,
    "update": False,
    "resume": False,
    # A DIRECTORY, not a prompt or a value from a list: `/move ~/x` names where
    # the session works, and the desktop refuses hand-typed arguments here.
    "move": False,
    # A LABEL, not a value from a list: the desktop opens a form for it.
    "rename": False,
    # A toggle with an optional on/off/status word — but the word is a plain
    # setting, not a value the desktop offers as an inline list, so a trailing
    # sentence after it stays prose. Kept False deliberately: narrowing the
    # refusal is the safe direction, and `/fast` "means" the toggle.
    "fast": False,
    "provider": False,
    # The PAGE is the receipt; these take no argument at all, so any trailing
    # text is by definition not theirs.
    "settings": False,
    "sidebar": False,
    "search": False,
    "accounts": False,
    "failovers": False,
    "usage": False,
    # `/analytics [view]` names a screen, not a value from an inline list.
    "analytics": False,
    "session": False,
    "info": False,
    "compact": False,
    # The stop target vocabulary, typed rather than picked from an inline list.
    "stop": False,
    "context": False,
    "skills": False,
    # `/mcp <subcommand> <name>` is validated by the ROUTE against a fixed
    # subcommand list. That is a claim about `argument_shape` (`subcommand`
    # below), NOT about this field: `/mcp logout` IS a command the desktop runs,
    # and it is not completable inline only because no list offers its
    # subcommands in the composer.
    "mcp": False,
    # A provider name chosen in the authentication panel, not typed after the
    # word; the route refuses args outright.
    "login": False,
    "logout": False,
    # A key NAME, with the secret entered in the masked form.
    "credential": False,
    "mobile": False,
}


#: The THIRD source: the shape trailing text must have for the desktop to OWN it
#: as this command's argument, where neither a prompt nor an inline list applies.
#:
#: The rule these entries encode: text after the word is the command's argument
#: when the desktop's own path USES it — a handler that reads it, a form field it
#: pre-fills, a selection/filter it forwards — OR REFUSES it because another
#: surface owns it (`/credential` below: the route answers typed text with the
#: masked-form sentence, so the text is not prose either). A sentence is never one
#: of those, so `/usage on` (a view selector) is the command while `/usage more
#: prose` is a message, and `/mcp logout` is the command while `/mcp logout seems
#: to cause a crash` is a message.
ARGUMENT_SHAPE_POLICY = {
    # --- shapes: text the desktop validates or forwards -----------------------
    # ONE selector token, forwarded as the picker's `selected`/`selection`/
    # `filter`. A sentence after the word is prose, which is what keeps
    # `/usage more prose` a message.
    "new": ArgumentShape.WORD,
    "reload": ArgumentShape.WORD,
    "resume": ArgumentShape.WORD,
    # The on/off choice the desktop's picker offers.
    "fast": ArgumentShape.WORD,
    "provider": ArgumentShape.WORD,
    "settings": ArgumentShape.WORD,
    "search": ArgumentShape.WORD,
    "accounts": ArgumentShape.WORD,
    "usage": ArgumentShape.WORD,
    "skills": ArgumentShape.WORD,
    "analytics": ArgumentShape.WORD,
    # The same target vocabulary `/stop <target>` takes.
    "stop": ArgumentShape.WORD,
    # NONE, and the criterion is why: a shape is published only where the
    # desktop's command PATH uses the trailing text, and this one DROPS it —
    # `_slash_result` calls `_context_slash_result(SlashResult)` with no args and
    # `native_action` has no branch, so `/context x` runs the command and throws
    # `x` away. Same class as `/compact hello`, which is the operator's report.
    "context": ArgumentShape.NONE,
    # ONE token naming a provider THIS INSTALL knows — the route's own lookup, so
    # `/login openai` is the command and `/login zzz` is prose.
    "login": ArgumentShape.PROVIDER,
    "logout": ArgumentShape.PROVIDER,
    # `<subcommand> [name]`, the MCP shape the route validates against
    # MCP_SUBCOMMANDS and SERVER_NAME_RE. At most two tokens, so the operator's
    # own three-line draft stays a message.
    "mcp": ArgumentShape.SUBCOMMAND,
    # ANY text: a handler or a form field takes it, so every whole-draft form is
    # the command. `/rename <title>`'s title is arbitrary text, and `/move <path>`
    # executes the path directly, spaces included.
    "rename": ArgumentShape.ANY,
    "move": ArgumentShape.ANY,
    # --- none: text after the word is PROSE -----------------------------------
    # The page or panel IS the receipt and takes no argument at all; these open a
    # surface rather than consuming text, so a sentence after them is a message.
    "help": ArgumentShape.NONE,
    "exit": ArgumentShape.NONE,
    "clear": ArgumentShape.NONE,
    "copy": ArgumentShape.NONE,
    "links": ArgumentShape.NONE,
    "update": ArgumentShape.NONE,
    "sidebar": ArgumentShape.NONE,
    "failovers": ArgumentShape.NONE,
    "session": ArgumentShape.NONE,
    "info": ArgumentShape.NONE,
    "mobile": ArgumentShape.NONE,
    # The operator's own row, and the reason it is NONE rather than ANY: the
    # desktop runs `/compact` and SILENTLY DROPS what follows (the report was
    # that `/compact hello` "used to run and silently eat `hello`"). A command
    # whose handler does not read its text does not own it, so a sentence here is
    # a message — which is what the composer now plans for it too.
    "compact": ArgumentShape.NONE,
    # --- ANY: text the command owns -------------------------------------------
    # `consumes_prompt` says the text is a prompt; `prefixes_text` says it is a
    # value from a list. Both decide FIRST, and these rows say `ANY` here rather
    # than inheriting `none` — the command owns its trailing text whatever it
    # says — so a consumer that reads this field ALONE does not plan prose for a
    # control this endpoint refuses (round 2's MAJOR-1: the field published
    # `none` for these rows while every published definition of `none` said
    # "never"). Reading the shape first, or OR-ing all three sources, is now
    # correct either way.
    "goal": ArgumentShape.ANY,
    "loop": ArgumentShape.ANY,
    "btw": ArgumentShape.ANY,
    "fork": ArgumentShape.ANY,
    "team": ArgumentShape.ANY,
    "agent": ArgumentShape.ANY,
    "model": ArgumentShape.ANY,
    "effort": ArgumentShape.ANY,
    "approvals": ArgumentShape.ANY,
    "theme": ArgumentShape.ANY,
    # ...and ONE row neither boolean carries, because the text it owns is REFUSED
    # rather than consumed. The command route answers typed text with "Enter
    # credentials in the masked credential form, not command text", so the text
    # belongs to that form and must never be planned as prose: a consumer reading
    # NONE plans a whole draft as a message, which is how `/credential <secret>`
    # reached the model as a paid turn. ANY rather than WORD because a secret is
    # arbitrary text — `/credential my pass phrase` must not be admitted either.
    "credential": ArgumentShape.ANY,
}


def test_every_registered_command_states_a_prefixes_text_policy() -> None:
    """The forcing function: a new command must state this choice, not inherit it."""
    assert {command.name: command.prefixes_text for command in SLASH_COMMANDS} == (
        PREFIXES_TEXT_POLICY
    )


def test_a_command_that_consumes_a_prompt_also_prefixes_text() -> None:
    """``consumes_prompt ⇒ prefixes_text``, and the converse is deliberately unused.

    The two answer different halves of one question: ``consumes_prompt`` says the
    argument reaches a model (so an inline engage reassembles rather than
    splices), ``prefixes_text`` says the desktop treats the trailing text as the
    command's own. A prompt-consuming command's text is always its own, so the
    implication must hold; ``/model`` is the counterexample that keeps the
    converse out — its text is its own without ever becoming a prompt.
    """
    offenders = [
        command.name
        for command in SLASH_COMMANDS
        if command.consumes_prompt and not command.prefixes_text
    ]
    assert offenders == []


def test_the_vocabulary_is_exactly_the_desktop_argument_set() -> None:
    """The set itself, stated positively so a widening has to be argued for here.

    Both words of the union, so this is what the messages endpoint reads: the
    free-text prompts plus the list-valued settings. Any addition is a claim that
    text after that word belongs to the command — the direction that can turn a
    user's prose into a paid turn, so it is pinned as a literal.
    """
    flagged = {command.name for command in SLASH_COMMANDS if command.prefixes_text}
    assert flagged == {
        # free text destined for a model
        "goal",
        "loop",
        "btw",
        "fork",
        "team",
        "agent",
        # a value chosen from an inline list
        "model",
        "effort",
        "approvals",
        "theme",
    }


def test_every_registered_command_states_an_argument_shape() -> None:
    """The forcing function for the third source, as the two booleans have theirs."""
    assert {command.name: command.argument_shape for command in SLASH_COMMANDS} == (
        ARGUMENT_SHAPE_POLICY
    )


def test_the_boolean_carried_rows_declare_any_not_none() -> None:
    """`any` is published explicitly on every row a boolean carries.

    The reverse of this test used to pin the opposite (a boolean-carried row must
    NOT declare a shape), which was right only while `none` meant "ask the
    booleans". Every published definition of `none` says "never", so a consumer
    reading the field first planned PROSE for 37 whole-draft controls this
    endpoint refuses — round 2's MAJOR-1. The spelling is now: the booleans decide
    first, AND a row they carry declares `any`, so `none` can only ever mean "no
    source at all".
    """
    boolean_carried = {
        command.name
        for command in SLASH_COMMANDS
        if command.consumes_prompt or command.prefixes_text
    }
    # No ``boolean_carried == declared`` line here: the second set was the first one
    # written a different way, so the assert could not fail and proved nothing
    # (round 3's NIT). The set that carries the claim is the one below, checked
    # against the shapes actually published. The CONVERSE is not claimed —
    # `/credential` publishes `any` with both booleans false, because its text is
    # refused rather than consumed (the policy table's last entry).
    shapes = {command.name: command.argument_shape for command in SLASH_COMMANDS}
    assert {name for name in boolean_carried if shapes[name] is not ArgumentShape.ANY} == set()


def test_none_is_reserved_for_rows_neither_boolean_carries() -> None:
    """The other half, stated so `none` cannot drift back into a total answer.

    A `none` row must have both booleans false: if one were true, the published
    shape would contradict what the endpoint does with the row (it WOULD own the
    text), which is the contradiction round 2 found in the other direction.
    """
    offenders = [
        command.name
        for command in SLASH_COMMANDS
        if command.argument_shape is ArgumentShape.NONE
        and (command.consumes_prompt or command.prefixes_text)
    ]
    assert offenders == []


def test_a_row_that_offers_a_value_list_never_publishes_none() -> None:
    """The SECOND contradiction `none` can carry, and the one that leaked.

    `ArgumentMode` and the shape are two independent facts about the same text,
    and this is the invariant between them: a row whose `arguments` offers a
    value list says a space here opens a list, so the command takes text and the
    shape must name where that text GOES. Publishing `none` beside it says the
    text is prose instead, which is what the messages endpoint's admission rule
    read for `/credential`: a whole-draft `/credential <secret>` was admitted as a
    MESSAGE while the command route refused the identical text with its masked-
    form sentence, so a client planning prose from this field posted a raw
    credential to the model.

    Registry-wide rather than per-row because it is the cheaper pin for every
    FUTURE row: `none` and a value list are mutually exclusive claims, and a new
    command that makes both is wrong whatever its handler does.
    """
    offenders = [
        command.name
        for command in SLASH_COMMANDS
        if command.argument_shape is ArgumentShape.NONE
        and command.arguments is not ArgumentMode.NONE
    ]
    assert offenders == []
