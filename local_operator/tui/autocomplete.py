"""Slash-command matching and ranking for the input editor.

Purely sync and I/O-free by design: matching runs on every keystroke, feeds
the picker that draws under the editor, and must resolve deterministically
before Enter is dispatched. File/path completion is async work and lives
elsewhere (later); only commands rank here.

This module ranks, it does not decide: :mod:`local_operator.tui.widgets.
command_picker` owns which match is highlighted and what gets inserted. Two
places computing "the" match is how the highlighted row and the applied
command drift apart, so there is exactly one — :func:`match_commands`.

Scoring contract:

- exact match: 1000
- prefix match: 900, flat — registry order breaks ties
- fuzzy subsequence: 1..40, denser matches score higher
- otherwise 0 (no match), which includes the empty prefix: "nothing typed"
  is not a match, and the bare-``/`` menu is the picker's call to make
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, TypeVar, runtime_checkable


class ArgumentMode(Enum):
    """What a command's ARGUMENT is, from the completion's point of view.

    The registry's one statement about argument completion, read by the editor
    to decide two different things with one fact:

    * whether a space after the command word opens the value list at all
      (:attr:`NONE` never does); and
    * whether Enter on the command ROW may also SEND it. ``/login`` with no
      provider does nothing, so Enter there opens the list and stops —
      submitting as well would run a no-op and clear the buffer the list was
      just drawn over. ``/approvals`` and ``/effort`` answer "what am I on"
      when bare, so Enter still sends them and the list is an OFFER for the
      next keystroke rather than a gate in front of a command that works.

    Deliberately NOT a tuple of :class:`ArgumentChoice` on the registry entry.
    Every list this app offers carries live state — which provider holds a
    credential, which mode is in force and which one is saved, which rungs THIS
    model accepts — so a frozen tuple beside the description would be a second
    copy of state with no way to refresh, and the first thing it would get
    wrong is the marker saying where the user already is. The registry declares
    that a list exists; the app fills it at the moment it opens.
    """

    #: No value list. The command takes free text or nothing.
    NONE = "none"
    #: The list is an offer; the bare command does something useful too.
    OPTIONAL = "optional"
    #: The list IS the command; bare, there is nothing to run.
    REQUIRED = "required"


class ArgumentShape(Enum):
    """What the TEXT AFTER a command's word IS, for the desktop.

    The registry's declaration for the third source of "this trailing text is
    the command's argument" — the two the composer derives from ``consumes_prompt``
    (free text) and ``prefixes_text`` (a value chosen from a list) are narrower
    than the field they approximate. What is left over is text the desktop
    VALIDATES or FORWARDS: the MCP subcommand and server name, a provider id, a
    selection, a filter, a path, a title.

    PRECEDENCE, and it is published so a consumer may OR the three facts rather
    than nest them: ``consumes_prompt`` and ``prefixes_text`` are read FIRST and
    are the whole answer when either is true. The shape is asked only after them,
    and a row that one of the booleans already carries declares ``ANY`` anyway —
    the command owns its trailing text, whatever it says — so a reader who takes
    the shape first, or who ORs the three, still reaches the same answer. ``NONE``
    is therefore reserved for rows where the booleans are false AND no shape
    applies; it is emitted explicitly rather than omitted, so no consumer has to
    infer a default.

    Read on the desktop's ADMISSION rule (``slash_commands.
    command_argument_is_used``) and by the command route's own validator, so
    "is this trailing text this command's argument" has exactly one answer per
    shape rather than one per call site. That is the second-decision defect
    class this repo has already paid for: a host planning a draft as prose while
    the other refuses it as a command leaves a message that no resend can clear.

    NOT :class:`ArgumentMode`, and the difference is the whole reason both exist.
    ``ArgumentMode`` answers "does a space open a VALUE LIST in THIS TERMINAL, and
    does Enter on the bare word also send it" — a TUI completion question. This
    answers "would the DESKTOP use text typed here", which is true for commands
    whose trailing text never opens a TUI list at all (``/mcp logout`` is a
    subcommand, ``/move ~/x`` is a path). Widening one into the other was probed
    and is wrong in both directions: ``/login`` is ``REQUIRED`` yet its word is a
    provider id the route validates, while ``/usage`` is no ``ArgumentMode`` list
    at all yet ``/usage on`` is a control the composer runs.

    THE CRITERION, stated because a shape is a decision per command and a reader
    has to be able to check one: a shape is published only where the desktop's
    command PATH uses the trailing text — a handler that reads it (a title, a
    path, a session selection), a presentation payload that carries it
    (``selection``, ``filter``, ``selected``, a choice field), or a validator that
    judges it (the provider id, the MCP subcommand). Where the path DROPS the text
    the shape is ``NONE`` and the text is prose: ``/compact hello`` and
    ``/context x`` both run a command that silently discards what followed, which
    is the operator's own report against ``/compact``.

    THREE DELIBERATE EXCEPTIONS, named here so the criterion is not read as a law
    the table breaks silently. ``/search <word>`` (the desktop fixes its filter to
    ``web-search`` and reads no word) and ``/stop <word>`` (the picker owns
    ``targets=[session_id]``) keep ``WORD`` even though the path drops the text:
    both are whole-draft controls the composer RUNS, so a sentence after the word
    is prose while a single word is not — and refusing the one-token form is the
    safe direction for a guard whose whole purpose is that a control never becomes
    paid model chat.

    ``/credential <anything>`` is the third, and it is a REFUSAL rather than a use:
    the command route answers typed text with "Enter credentials in the masked
    credential form, not command text", so the text is neither consumed by a
    handler nor prose — another surface owns it. It therefore publishes ``ANY``,
    which is the direction that CANNOT leak: a consumer reading ``NONE`` plans a
    whole draft as prose, so ``/credential <secret>`` typed into a message reached
    the model as a paid turn (measured, and the reason this exception is stated).
    The shape is asked here and not left to the route's own check, because the
    messages endpoint's admission rule reads only this vocabulary.
    """

    #: No argument AT ALL from any source: the two booleans are false for this
    #: row and no shape applies, so text after the word is prose, whatever it
    #: says. Read AFTER the booleans — see the precedence note above; a row they
    #: carry never says this (`/goal` is ``ANY``, not ``NONE``).
    #:
    #: NOR IS IT A ROW THAT OFFERS A VALUE LIST. ``ArgumentMode``
    #: (``/credential``'s ``optional``) says a space here opens a list, so the
    #: command does take text; publishing ``NONE`` beside it is the contradiction
    #: that let a whole-draft ``/credential <secret>`` reach the model as prose
    #: while the route refused the same text, and
    #: ``test_a_row_that_offers_a_value_list_never_publishes_none`` pins it for
    #: every row.
    NONE = "none"
    #: ONE whitespace-free token — a selector the desktop forwards as a
    #: ``selection``/``filter``/``selected`` value (a view, a mode, a session id).
    #: A sentence here is prose, which is what keeps `/usage more prose` a message.
    #: The vocabulary is ``slash_commands.command_argument_words``, empty for this
    #: shape's rows and honoured by BOTH arms (the validator and the catalogue),
    #: so a row needing one has a single place to declare it.
    WORD = "word"
    #: One token naming a provider THIS INSTALL knows — the route's own lookup
    #: (``get_provider_definition``), so ``/login openai`` is the command and
    #: ``/login zzz`` is prose. A ``WORD`` whose vocabulary is the provider ids.
    PROVIDER = "provider"
    #: ``<subcommand> [name]`` — the MCP shape the command route validates
    #: against ``MCP_SUBCOMMANDS`` and ``SERVER_NAME_RE``. At most two tokens, so
    #: ``/mcp logout`` is the command while ``/mcp logout seems to cause a crash``
    #: is prose.
    #:
    #: A row may declare its OWN vocabulary instead (:attr:`SlashCommand.
    #: subcommands`), which is what makes ``/network ls`` this shape rather than a
    #: second enum member: the two readers — the route's validator and the
    #: desktop catalogue's ``argument_words`` — go through
    #: :func:`slash_commands.command_argument_words`, so the vocabulary a row
    #: declares is the one both accept. The two-token cap is the SHAPE's and does
    #: not change with the vocabulary (``/network disconnect please`` is prose,
    #: which is the safe direction for a guard whose job is to keep a control from
    #: becoming paid chat); a row needing a longer argument names it in a shape of
    #: its own rather than widening this one.
    SUBCOMMAND = "subcommand"
    #: ``remote <peer>`` — a peer device named by id or by the name this device
    #: knows it under, optionally holding the peer's id in the same token
    #: (``remote devon#9f2c``). One token is ALSO accepted, and that is the
    #: legacy half rather than a concession: ``/new <word>`` is the desktop's
    #: new-session picker selection (``selected=args``), so a shape that demanded
    #: the two-token form would plan ``/new foo`` as PROSE and spend a paid turn
    #: on a control the user typed deliberately.
    #:
    #: The vocabulary is LIVE (the peer catalogue, or the offline member lists
    #: when the relay is down — see ``network/peers.py``), so this is the one shape
    #: whose validator resolves a name against something outside the registry, the
    #: way :attr:`PROVIDER` resolves against the provider registry.
    REMOTE_PEER = "remote_peer"
    #: The command owns its trailing text, whatever it says — a handler or a form
    #: field takes it (a session title, a working directory, a path), one of the
    #: two booleans above already carries it, or the route REFUSES it because
    #: another surface owns it (``/credential``, whose text must reach the masked
    #: form or nothing). Declared explicitly on those rows rather than left to the
    #: default: ``NONE`` means "no source at all", and a consumer that reads this
    #: field alone must not plan prose for a control the endpoint refuses.
    ANY = "any"


#: Exact / prefix tiers, with registry-order tie-break.
SCORE_EXACT = 1000
SCORE_PREFIX = 900
SCORE_FUZZY_MAX = 40


@dataclass(frozen=True)
class SlashCommand:
    """A user-facing slash command known to the app."""

    # Navigation is not owner execution. This field keeps desktop destinations
    # beside command identity instead of introducing another command registry.
    desktop_destination: str = field(default="", kw_only=True)
    name: str
    description: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    #: Whether running the command may write what was typed into the visible
    #: ledger as a user row.
    #:
    #: Keyword-only and defaulting to FALSE because the transcript is a
    #: reading record, not a keystroke log: every handler in
    #: ``local_operator.tui.app`` already reports what it did — a panel, a
    #: listing, a notice naming the new state — so an echo above that receipt
    #: is a second row saying the same thing. ``True`` is for the one case the
    #: receipt cannot cover: an argument that becomes something the MODEL is
    #: told (``/goal <text>`` rides the system prompt's volatile tail), where
    #: the ledger's job is to show what the model was given, attributed to the
    #: user who gave it.
    #:
    #: The registry decides WHETHER; the handler decides WHEN, by calling
    #: ``OperatorApp._echo_user_command`` at the point its effect has actually
    #: landed. Splitting it that way is what keeps the row honest: written
    #: before dispatch, ``/goal`` claimed the model had been given words for
    #: its read-only form, for ``/goal clear``, and for a set REJECTED because
    #: the session had not started yet.
    #:
    #: The flag lives on the registry entry, not in the submit handler, so the
    #: policy is read next to the command it governs; ``SLASH_COMMANDS`` is
    #: pinned entry-by-entry in ``tests/unit/tui/test_slash_echo.py`` so a new
    #: command cannot be added without stating its choice.
    echo: bool = field(default=False, kw_only=True)
    #: Whether this command's ARGUMENT is offered as a list, and how hard the
    #: offer is. See :class:`ArgumentMode`; the app fills the rows when the list
    #: opens (``OperatorApp.on_argument_query_opened``).
    #:
    #: Keyword-only and defaulting to :attr:`ArgumentMode.NONE` for the reason
    #: ``echo`` is: a command that has not thought about it gets the behaviour
    #: that changes nothing. Free typing is unaffected in every mode — the list
    #: ranks what is typed, it never filters what may be submitted.
    arguments: ArgumentMode = field(default=ArgumentMode.NONE, kw_only=True)
    #: Whether this command's argument is FREE-TEXT that becomes a prompt the
    #: model is given — the goal text, a loop instruction, a team/agent request,
    #: an aside question. These are the "start of the composer" commands: their
    #: whole point is to prefix a message, so engaging one INLINE (mid-draft)
    #: does not splice-and-run — it REASSEMBLES the command to the front of the
    #: composer with the rest of the draft as its argument, staged for the user
    #: to review and submit. That is what makes "type the message, then remember
    #: to route it" safe: nothing the user typed is ever consumed as a name or
    #: silently dropped (the D1 data-loss the naive end-of-line argument caused).
    #:
    #: Keyword-only, defaulting to FALSE for the same reason ``echo`` and
    #: ``arguments`` do: a command that has not opted in keeps the simple
    #: splice-and-run inline behaviour (``/usage``, ``/model``), which is right
    #: for every command whose argument is a SELECTOR rather than a message.
    #: ``SLASH_COMMANDS`` is pinned entry-by-entry in ``test_slash_echo.py`` so a
    #: new command must state this choice.
    consumes_prompt: bool = field(default=False, kw_only=True)
    #: Whether text typed AFTER this command's word is an ARGUMENT the command
    #: owns, on the DESKTOP — so ``/model gpt-5`` is the model command with its
    #: value rather than a message, while ``/mcp logout seems to be broken`` is a
    #: message that happens to open with a command word.
    #:
    #: Keyword-only and defaulting to FALSE for the same reason ``echo`` and
    #: ``consumes_prompt`` do. It is the UNION of two things the composer can
    #: complete: the free-text prompt (``consumes_prompt``) and the value chosen
    #: from a list (``inline`` in the renderer's ``picker-registry``, whose
    #: contents are a host presentation fact and deliberately not mirrored here).
    #:
    #: NOT the same question as ``arguments is not ArgumentMode.NONE``, and this is
    #: the trap. ArgumentMode answers "does a space open a VALUE LIST in this
    #: terminal", which is TRUE for ``/login``, ``/move``, ``/stop``, ``/mcp`` and
    #: ``/rename`` — all of which the DESKTOP presents as a picker or a form and
    #: deliberately refuses hand-typed arguments for
    #: (``desktop_sessions.py:657-678``). The desktop's answer for those is "that
    #: trailing text is prose", which is what makes ``/mcp logout seems to cause a
    #: crash`` a message instead of a malformed MCP invocation.
    #:
    #: It is the desktop's half of the ONE rule the messages endpoint and the
    #: composer both read (``slash_commands.command_prefixes_text``). Two
    #: derivations of "is this trailing text the command's argument" is the
    #: second-decision defect class this repo has already paid for: the route and
    #: the planner answering differently turns a prose draft into a permanent
    #: refusal no resend can clear.
    prefixes_text: bool = field(default=False, kw_only=True)
    #: What TEXT AFTER this command's word IS, for the desktop — the third source
    #: of "this trailing text is the command's argument", beside
    #: ``consumes_prompt`` (free text) and ``prefixes_text`` (a value chosen from
    #: a list). See :class:`ArgumentShape` for the values and why neither of the
    #: other two can stand in for it.
    #:
    #: PRECEDENCE: the two booleans are read FIRST and are the whole answer when
    #: either is true; this field is asked after them. A row the booleans already
    #: carry declares :attr:`ArgumentShape.ANY` explicitly — "the command owns its
    #: trailing text, whatever it says" — because the default below means NO
    #: source at all, and the field is published so a consumer may read it alone
    #: or OR the three facts and still be right.
    #:
    #: Keyword-only and defaulting to :attr:`ArgumentShape.NONE` for the reason
    #: ``echo`` and ``consumes_prompt`` do: a command that has not stated a shape
    #: gets the behaviour that changes nothing — text after it is PROSE, which is
    #: the direction that cannot turn a user's message into a command.
    #:
    #: ``SLASH_COMMANDS`` is pinned entry-by-entry in
    #: ``tests/unit/tui/test_slash_prefixes_text.py`` so a new command cannot be
    #: added without stating this choice.
    argument_shape: ArgumentShape = field(default=ArgumentShape.NONE, kw_only=True)
    #: Whether this command's argument list is a NAME slot: the first token is a
    #: name drawn from a roster (`/team <name> <request>`, `/agent <name> <message>`),
    #: and free text follows it.
    #:
    #: A SEPARATE fact from ``arguments``, and it has to be. Two readers need
    #: "is there a name slot" and both used to ask "is there a value list"
    #: instead, because for every command that existed the two answers coincided:
    #: the `$skill` floor (a `$` inside a prompt command's argument is ordinary
    #: text until the NAME SLOT has been passed, so the picker has to know which
    #: commands have one) and the composer's inline reassembly in
    #: ``Editor._apply_command`` (a bare inline `/team` deliberately does NOT
    #: reassemble its draft — the name is picked from the list first — while a
    #: bare `/goal` does). Giving a value list to a command with NO name slot is
    #: what splits them: ``/goal --clear``'s first token is not a name, so the
    #: proxy would have swallowed the ``$skill`` claim inside `/goal ` and
    #: stopped a bare inline `/goal` from reassembling — both documented, both
    #: pinned by tests (``test_skill_in_command_argument``).
    #:
    #: Keyword-only and defaulting to FALSE for the reason the fields above do: a
    #: command that has not stated a name slot does not get one.
    #: ``test_slash_goal_loop_flags`` pins the flag against the registry, so a
    #: third party has to state its choice the way the other fields' pins do.
    name_argument: bool = field(default=False, kw_only=True)
    #: This command's own SUBCOMMAND vocabulary, for
    #: :attr:`ArgumentShape.SUBCOMMAND` rows that are not MCP's.
    #:
    #: A field rather than a second ``ArgumentShape`` member, because the shape
    #: answers "what IS the text after the word" (a subcommand and its argument)
    #: while this answers "which words are subcommands of THIS command" — and the
    #: two questions have one answer each, not a cross product. The alternative was
    #: a member per command family, which grows the enum every time a grouped
    #: command lands and makes every reader switch on the command's NAME instead.
    #:
    #: Read by :func:`slash_commands.command_argument_words`, which is the ONE
    #: derivation both the route's validator and the desktop catalogue consult —
    #: so the word a handler accepts and the word a picker offers cannot drift.
    #: Empty means "not declared": an MCP row, whose vocabulary is
    #: ``session.frontend_state.MCP_SUBCOMMANDS``, is unchanged by this field's
    #: existence (one list, resolved lazily at the same place it always was).
    subcommands: tuple[str, ...] = field(default_factory=tuple, kw_only=True)

    @property
    def names(self) -> tuple[str, ...]:
        """Primary name first, then aliases — order is the tie-break order."""
        return (self.name, *self.aliases)


@runtime_checkable
class Completable(Protocol):
    """What the picker needs of anything it can offer and complete.

    Exists so the ONE picker widget can present two different kinds of list —
    the command word (:class:`SlashCommand`) and a command's argument
    (:class:`ArgumentChoice`) — without a second widget, a second matcher or a
    second set of key bindings growing beside the first. The alternative was a
    provider-specific picker, which is how a codebase ends up with two lists
    that drift apart in look and in behaviour.
    """

    # Declared as read-only PROPERTIES, not annotated attributes. A bare
    # ``name: str`` in a Protocol demands a *settable* attribute, which no
    # frozen dataclass can satisfy — and both implementers here are frozen
    # (:class:`SlashCommand` and :class:`ArgumentChoice`), deliberately, because
    # a suggestion the picker is holding must not mutate under it between the
    # keystroke that ranked it and the Enter that acts on it.
    @property
    def name(self) -> str:
        """The value a completion inserts."""
        ...

    @property
    def description(self) -> str:
        """One line explaining what this is."""
        ...

    @property
    def names(self) -> tuple[str, ...]:
        """Primary name first, then aliases."""
        ...


#: Bound to :class:`Completable` so a matcher hands back the SAME concrete type
#: it was given: ranking a list of commands returns commands, ranking a list of
#: argument choices returns argument choices. Without it every caller would have
#: to narrow the result back at runtime and silently drop anything it failed to
#: recognise.
ChoiceT = TypeVar("ChoiceT", bound=Completable)


@dataclass(frozen=True)
class ArgumentChoice:
    """One value offered for a slash command's ARGUMENT.

    ``detail`` is state rather than explanation — "logged in", "needs login" —
    and is rendered in a COLUMN pinned to the row's trailing edge, away from
    ``description``, because the two answer different questions: what this thing
    IS versus where it stands right now. The column is what makes the states
    scannable, so the strings inside it are left-aligned against one shared
    edge; right-aligning each string to its own row started three states at
    three different columns and left nothing to scan.
    """

    name: str
    description: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""
    #: Paints ``detail`` in the danger tint when the state is a problem the user
    #: should notice (a server that failed, a credential that cannot be read).
    alert: bool = False

    @property
    def names(self) -> tuple[str, ...]:
        """Primary name first, then aliases — order is the tie-break order."""
        return (self.name, *self.aliases)


def score_command_text_match(prefix: str, target: str) -> int:
    """Score how well a typed ``prefix`` matches a command ``target``.

    Case-insensitive. Exact 1000 > prefix 900 (flat, so registration order
    breaks ties) > fuzzy subsequence 1..40 > no match 0. The fuzzy band
    rewards density: consecutive matched characters and early matches push
    the score toward 40.
    """
    lower_prefix = prefix.lower()
    lower_target = target.lower()
    if not lower_prefix:
        return 0
    if lower_prefix == lower_target:
        return SCORE_EXACT
    if lower_target.startswith(lower_prefix):
        return SCORE_PREFIX
    return _subsequence_score(lower_prefix, lower_target)


def _subsequence_score(prefix: str, target: str) -> int:
    """Score ``prefix`` as an in-order subsequence of ``target``, 1..40 or 0."""
    score = 0
    prev_index = -2
    target_index = 0
    for char in prefix:
        found = target.find(char, target_index)
        if found < 0:
            return 0
        if found == prev_index + 1:
            score += 2  # consecutive run: dense match
        else:
            score += 1
        prev_index = found
        target_index = found + 1
    if score <= 0:
        return 0
    return max(1, min(SCORE_FUZZY_MAX, score))


def match_commands(
    text_before_cursor: str, commands: list[SlashCommand]
) -> list[tuple[str, SlashCommand]]:
    """Return ``(display_name, command)`` matches for slash text, best first.

    ``text_before_cursor`` is the editor text up to the caret; matching only
    applies to a single token starting with ``/``. Ties keep registration
    order (the prefix tier is deliberately flat, so registration order breaks ties).
    """
    token = text_before_cursor.strip()
    if not token.startswith("/"):
        return []
    typed = token[1:]
    scored: list[tuple[int, int, str, SlashCommand]] = []
    for registry_index, command in enumerate(commands):
        best = 0
        best_name = command.name
        for alias_index, alias in enumerate(command.names):
            score = score_command_text_match(typed, alias)
            if score > best:
                best = score
                best_name = alias
        if best > 0:
            scored.append((-best, registry_index, best_name, command))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [(name, command) for _, _, name, command in scored]


def match_choices(query: str, choices: Sequence[ChoiceT]) -> list[tuple[str, ChoiceT]]:
    """Rank ``choices`` against a bare ``query`` token, best first.

    The argument-side counterpart to :func:`match_commands`, sharing
    :func:`score_command_text_match` so a provider and a command are ranked by
    exactly the same rules — a user who has learned that ``/lgt`` finds
    ``logout`` should find ``anthrpc`` finds ``anthropic`` without learning a
    second behaviour.

    Two deliberate differences from :func:`match_commands`. There is no leading
    ``/`` to strip, because an argument is a bare word. And the returned display
    name is ALWAYS ``choice.name``, never the alias that happened to match: a
    command's aliases are themselves typeable commands (``/models`` really runs),
    whereas an argument's aliases are only a way to FIND it — ``claude`` finds
    the ``anthropic`` provider but ``/login claude`` is not a thing. Returning
    the alias would put a word into the buffer that the command then rejects as
    unknown.

    An empty ``query`` returns everything in the given order, because "I typed
    the command and stopped" is a request to see the whole set, not a failed
    match.
    """
    if not query:
        return [(choice.name, choice) for choice in choices]
    scored: list[tuple[int, int, ChoiceT]] = []
    for order, choice in enumerate(choices):
        best = max(
            (score_command_text_match(query, alias) for alias in choice.names),
            default=0,
        )
        if best > 0:
            scored.append((-best, order, choice))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [(choice.name, choice) for _, _, choice in scored]
