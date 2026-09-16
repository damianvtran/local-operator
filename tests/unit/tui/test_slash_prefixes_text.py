"""Which slash commands OWN the text typed after their word, and which do not.

Why this pin exists, and why it is not derived from ``ArgumentMode``: the
messages endpoint refuses a draft only when the draft IS a command, and the
"is the trailing text this command's argument" half of that rule has to be the
SAME fact the composer plans with. Two derivations is the second-decision defect
class this repo has already paid for once (the QA round 2 Q4 ``/usage\\nfix it``
bug): the composer planned a draft as prose while the endpoint refused it on the
leading word, and no resend could clear the refusal.

So the table is pinned entry by entry, the way ``test_slash_echo.py`` pins
``echo`` and ``consumes_prompt``. A command added to ``SLASH_COMMANDS`` without
an opinion fails here, which is the only way "state your choice" can be enforced
on a field that must keep a default for the picker fixtures that have no opinion.

``ArgumentMode`` is deliberately NOT the source. It answers "does a space open a
VALUE LIST in this terminal", which is true for ``/login``, ``/move``,
``/stop``, ``/mcp`` and ``/rename`` — all presented on the desktop as a picker or
a form that refuses hand-typed arguments, so for THOSE the trailing text is
prose. Widening to ``arguments is not ArgumentMode.NONE`` was probed and turns
the operator's own row back into a command, which is the exact refusal this
change removes.
"""

from __future__ import annotations

from local_operator.slash_commands import SLASH_COMMANDS

#: The desktop's argument vocabulary, one row per registry entry.
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
    # subcommand list, and the desktop's own answer for hand-typed text here is
    # "prose" — which is what makes `/mcp logout seems to cause a crash` a
    # message instead of a malformed MCP invocation (the operator's report).
    "mcp": False,
    # A provider name chosen in the authentication panel, not typed after the
    # word; the route refuses args outright.
    "login": False,
    "logout": False,
    # A key NAME, with the secret entered in the masked form.
    "credential": False,
    "mobile": False,
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
