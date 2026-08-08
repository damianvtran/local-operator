"""Input editor — TextArea inverted to chat semantics.

Textual's TextArea defaults to newline-on-Enter; this product wants
submit-on-Enter. The subclass inverts that and takes the terminal key idioms:

- ``Enter`` submits (posts :class:`EditorSubmitted`); with the command picker
  open it first completes the highlighted command, THEN submits
- ``Shift+Enter`` inserts a newline
- ``Ctrl+C`` posts :class:`InterruptRequested` (abort the turn) — never exits
- ``Ctrl+D`` on an EMPTY buffer quits; otherwise it falls through to delete
- ``Up``/``Down`` move the picker's highlight while it is open; otherwise they
  cycle prompt history when the caret sits at the top/bottom edge of the
  buffer, and inside the text they keep their cursor-move meaning
- ``Tab`` completes the highlighted command WITHOUT submitting
- ``Esc`` dismisses the picker, leaving the typed text alone

Key interception happens in :meth:`_on_key`, which runs BEFORE TextArea's
document-insert path, so a handled key never reaches the buffer. Unhandled
keys fall through to the stock editor behavior.

The caret is SOLID and never blinks; see ``cursor_blink`` in :meth:`__init__`.

The editor OWNS its :class:`CommandPicker` (built in ``__init__``, mounted by
the app as a sibling below the input row, since the picker must draw outside
the chevron+editor row). One picker always exists, so every completion path —
Tab, Enter, mouse click — runs through the same code with no "no picker
attached" variant to keep in step.

The picker under the field serves two lists. While the command WORD is open it
offers commands; once the word is terminated by a space, ``/model`` hands the
model picker its query and ``/login``/``/logout`` put the SAME command picker
into argument mode over the provider list. Which one is live is decided by
parsing the buffer (``slash_context`` versus ``slash_argument``), so two lists
can never be open at once.
"""

from __future__ import annotations

from typing import Callable

from textual import events
from textual.message import Message
from textual.widgets import TextArea
from textual.widgets.text_area import Edit, EditResult

from local_operator.tui.autocomplete import SlashCommand
from local_operator.tui.widgets.command_picker import (
    CommandPicker,
    PickerMode,
    slash_argument,
    slash_context,
)
from local_operator.tui.widgets.model_picker import ModelPicker, ModelRow


class EditorSubmitted(Message):
    """Posted when the user submits the editor (Enter without Shift)."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class InterruptRequested(Message):
    """Posted on Ctrl+C: abort the running turn, never exit the app."""

    def __init__(self) -> None:
        super().__init__()


class StopRequested(Message):
    """Posted on Esc with no picker open: stop whatever the agent is doing.

    Separate from :class:`InterruptRequested` (Ctrl+C) because the app answers
    them differently: Esc first refuses a pending tool-approval prompt, and only
    aborts when there is a turn to abort, while Ctrl+C always interrupts.
    """

    def __init__(self) -> None:
        super().__init__()


class EditorQuit(Message):
    """Posted on Ctrl+D with an empty buffer."""

    def __init__(self) -> None:
        super().__init__()


class ModelQueryOpened(Message):
    """Posted when the buffer enters ``/model …`` and the model list appears.

    The editor knows WHEN the list opens (it parses the buffer) but nothing about
    what belongs in it, and the app knows the catalogue but not the keystrokes.
    This message is the seam. It fires on the closed→open TRANSITION only, so
    typing a query does not re-trigger a provider fetch per character.
    """

    def __init__(self) -> None:
        super().__init__()


class ProviderQueryOpened(Message):
    """Posted when the buffer enters ``/login …`` or ``/logout …``.

    Carries the command WORD because the two lists are different sets — every
    loginable provider versus only the ones holding a stored credential — and the
    app cannot recover which one from the buffer once the message is queued.

    Like :class:`ModelQueryOpened` it fires on the TRANSITION only, so the app
    reads the credential store once per opening rather than once per keystroke.
    """

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command


class Editor(TextArea):
    """Multiline prompt editor with submit-on-Enter, history, slash-completion."""

    #: Maximum remembered prompts.
    HISTORY_LIMIT = 200

    #: Command words whose ARGUMENT is a model selector. Both spellings, because
    #: `/models` is a registered alias and a user who typed the alias should get
    #: the same list.
    MODEL_COMMANDS = ("model", "models")

    #: Command words whose ARGUMENT is a provider id. These drive the SAME picker
    #: the command word uses, in its argument mode.
    PROVIDER_COMMANDS = ("login", "logout")

    #: Provider commands that DESTROY something. `/logout` removes a credential
    #: and an OAuth one costs another browser round trip to get back, so its rows
    #: are gated harder than the shared ambiguity rule gates the rest: see
    #: :meth:`_picker_choice_is_unambiguous` and :meth:`_apply_command`.
    DESTRUCTIVE_COMMANDS = ("logout",)

    #: Every command whose argument drives a list. Completing one of these opens
    #: that list, which IS the outcome of the keystroke — running the bare command
    #: as well would echo a no-op into the transcript and clear the buffer the
    #: list was just opened over.
    LIST_COMMANDS = MODEL_COMMANDS + PROVIDER_COMMANDS

    def __init__(
        self,
        placeholder: str = "Message Local Operator…",
        commands: list[SlashCommand] | None = None,
    ) -> None:
        # Built BEFORE super().__init__: TextArea's constructor loads its
        # initial document, which funnels through load_text() and therefore
        # through _sync_picker().
        self._picker = CommandPicker(self._apply_command)
        self._model_picker = ModelPicker(self._apply_model)
        # Which provider command the argument list is currently open for, or None
        # when the buffer is not in one. This is the transition edge the
        # ProviderQueryOpened message rides: without it the app would rebuild the
        # credential list on every character typed into the query. Assigned here
        # for the same reason as the pickers — _sync_picker() reads it during
        # super().__init__().
        self._provider_command: str | None = None
        # tab_behavior="indent": Tab NEVER moves focus (TUI-013). Command
        # completion consumes the key first; otherwise it indents.
        super().__init__(placeholder=placeholder, soft_wrap=True, tab_behavior="indent")
        # A SOLID caret, ALWAYS — not "solid while the buffer is empty".
        #
        # Textual draws the caret onto the PLACEHOLDER: with no text, the
        # placeholder branch of `TextArea.render_line` inverts cell 0 of
        # `Message Local Operator…` whenever `_draw_cursor` is true. With blink
        # on, `_draw_cursor` flips twice a second, so the first letter of the
        # placeholder strobes — and on the boot splash, where nothing else
        # moves, that 2 Hz invert next to a static logo WAS the startup
        # animation users read as obnoxious. Nothing else was repainting.
        #
        # Gating blink on "buffer is empty" or "boot layout is active" would fix
        # exactly that frame and leave the identical 2 Hz invert running for the
        # rest of the session, now with the caret silently changing behaviour at
        # the first keystroke for a reason the user cannot see. Discoverability
        # points the same way: a blinking caret is INVISIBLE half the time,
        # while this one — inverted `fg` on `surface` via the
        # `text-area--cursor` component class — is a full-contrast block that is
        # never absent. Blink earns its keep in a dense form where the caret has
        # to be FOUND; this is one always-focused field whose caret sits exactly
        # where the user last typed.
        #
        # It also retires a timer: stock blink runs `refresh_lines` every 500 ms
        # for the life of the app, so a completely idle session was repainting a
        # row forever — twice a second of terminal writes down an ssh pipe for a
        # caret that had not moved.
        self.cursor_blink = False
        self._history: list[str] = []
        self._history_index: int | None = None  # None = not navigating
        self._draft: str = ""  # buffer text saved when history nav starts
        self._on_model_chosen: Callable[[ModelRow], None] | None = None
        self.set_commands(commands or [])

    # -- public API ---------------------------------------------------------
    @property
    def picker(self) -> CommandPicker:
        """The slash-command picker. The app mounts it below the input row."""
        return self._picker

    @property
    def model_picker(self) -> ModelPicker:
        """The model list shown while the buffer holds ``/model <query>``."""
        return self._model_picker

    @property
    def provider_command(self) -> str | None:
        """Which provider command the buffer's argument list is open for, if any.

        Exposed so the app can check that a ``ProviderQueryOpened`` it is handling
        still describes the buffer. The message is one message-loop tick old, and
        a tick is enough for the user to have typed over the command — answering a
        stale one attaches a notice to a command that no longer exists. The buffer
        stays the single authority; this is how a reader outside the widget asks
        it, rather than parsing the text a second time.
        """
        return self._provider_command

    def set_model_handler(self, handler: Callable[[ModelRow], None] | None) -> None:
        """Install what a chosen model DOES.

        A callback rather than a message, because the two outcomes are different
        kinds of thing — switching the session model, or starting a login for a
        provider that has no credential — and only the app knows how to do either.
        The editor's job ends at "the user chose this row".
        """
        self._on_model_chosen = handler

    def prompt_history(self) -> list[str]:
        """Recorded prompts, oldest first.

        Named ``prompt_history`` rather than ``history``: ``TextArea`` already
        owns a ``history`` attribute for its undo stack, and shadowing it with
        a method of an unrelated type is a live footgun for anything that
        reaches for the base class's own edit history.
        """
        return list(self._history)

    def set_commands(self, commands: list[SlashCommand]) -> None:
        """Slash commands offered to the picker (sync, no I/O)."""
        self._picker.set_commands(commands)

    def clear_content(self) -> None:
        """Empty the buffer and leave history navigation."""
        self.text = ""
        self._history_index = None

    def begin_model_query(self) -> None:
        """Put the buffer into the state that shows the model list.

        Exists so the app can reopen the list after ``/model`` has been submitted
        (the command picker completes the word, which terminates it, which submits
        it and clears the buffer) without reaching into the editor's private cursor
        helpers. Writing the text rather than poking the widget is deliberate: the
        buffer is what decides which picker is open, so anything that opens one has
        to go through the buffer or be undone by the next resync.
        """
        self.text = "/model "
        self.move_cursor(self._end_of_buffer())
        self._history_index = None

    # -- key interception ---------------------------------------------------
    async def _on_key(self, event: events.Key) -> None:
        """Handle chat keys before TextArea's insert path sees them."""
        key = event.key
        # Re-derive the picker state from the buffer BEFORE routing. The
        # buffer is the only authority on whether a command word is open, and
        # syncing here means routing never depends on a queued Changed
        # message having been drained first.
        self._sync_picker()
        if self._model_picker.is_open():
            # Checked BEFORE the command picker, though the two are mutually
            # exclusive by construction (`slash_context` closes on the space that
            # `slash_argument` opens on). Ordering it first makes that invariant
            # cheap to keep: if the two ever did overlap, the picker the user is
            # looking at is the one holding a query they just typed into.
            if key == "escape":
                # Esc closes the LIST, not the command: the text survives so a
                # user who wanted to type the id by hand can carry on.
                self._model_picker.close()
                event.stop()
                event.prevent_default()
                return
            if key in ("up", "down"):
                self._model_picker.move(-1 if key == "up" else +1)
                event.stop()
                event.prevent_default()
                return
            if key in ("pageup", "pagedown"):
                self._model_picker.page(-1 if key == "pageup" else +1)
                event.stop()
                event.prevent_default()
                return
            if key in ("home", "end"):
                self._model_picker.jump(to_end=key == "end")
                event.stop()
                event.prevent_default()
                return
            if key in ("tab", "enter"):
                row = self._model_picker.highlighted()
                if row is not None:
                    # Tab COMPLETES the selector into the buffer, Enter ACTS on
                    # it. Unlike the command picker there is no ambiguity gate:
                    # every row here names one concrete model, and the worst case
                    # of a wrong pick is a switch the user reverses with another
                    # `/model` — not autonomous work or a deleted credential.
                    if key == "tab":
                        self._complete_model(row)
                    else:
                        self._model_picker.choose(self._model_picker.selected_index)
                    event.stop()
                    event.prevent_default()
                    return
        if key == "escape" and self._picker.is_pending():
            # The rows are one message-loop tick behind the keystroke that opened
            # the list — the app answers ProviderQueryOpened — and for that tick
            # the picker is in argument mode holding nothing, so an `is_open()`
            # gate DROPPED the Esc: the user dismissed the list and then watched
            # it appear anyway. Dismissing an empty argument list records the
            # query, which is what the arriving rows are checked against.
            self._picker.dismiss()
            event.stop()
            event.prevent_default()
            return
        if self._picker.is_open():
            if key == "escape":
                self._picker.dismiss()
                event.stop()
                event.prevent_default()
                return
            if key in ("up", "down"):
                # The picker owns the arrows while it is open: history nav and
                # caret movement both stay reachable one Esc away, and an open
                # list that ignored Up/Down would look broken.
                self._picker.move(-1 if key == "up" else +1)
                event.stop()
                event.prevent_default()
                return
            if key in ("tab", "enter"):
                # Both keys insert the SAME completion, so the highlighted row
                # can never mean two different commands depending on the key.
                #
                # Enter only SENDS when the choice is unambiguous: one match, or
                # a name the user typed in full. The registry's blast radius is
                # not uniform — `/loop` starts autonomous work and `/logout`
                # removes credentials — and with a fuzzy matcher `/lo` highlights
                # `loop` while `login` and `logout` also match, so "Enter runs
                # whatever the matcher picked" could start a loop for a user
                # reaching for login, rewriting their text and running it in one
                # keystroke. When it is ambiguous, Enter completes (Tab's
                # behaviour) and a second Enter sends — the extra keystroke only
                # appears where the intent genuinely is not clear (D16).
                name = self._picker.highlighted_name()
                if name is not None:
                    # Decided BEFORE the completion is applied: completing
                    # rewrites the text and re-syncs the picker, so asking
                    # afterwards would measure the completed word (always one
                    # exact match) and submit unconditionally.
                    unambiguous = self._picker_choice_is_unambiguous(name)
                    if self._picker.mode is PickerMode.ARGUMENT:
                        self._resolve_argument(name, key, unambiguous)
                    elif unambiguous:
                        self._apply_command(name)
                        # A command whose ARGUMENT drives its own list is not run
                        # by completing it — completing it IS opening the list. The
                        # trailing space `_apply_command` adds is what opens the
                        # model picker, so submitting as well echoed `/model` into
                        # the transcript, cleared the buffer, and made the app put
                        # the query back to reopen a list the keystroke had already
                        # opened. One keystroke, one outcome.
                        if key == "enter" and name.lower() not in self.LIST_COMMANDS:
                            self._submit()
                    elif key == "tab":
                        # Tab never sends, so it can safely take the highlighted
                        # row: that is the whole point of a completion key.
                        self._apply_command(name)
                    else:
                        # Ambiguous Enter. Grow the word to the matches' longest
                        # COMMON prefix and leave the list open, rather than
                        # completing to the highlighted row.
                        #
                        # Completing to the row put the highest-blast-radius
                        # candidate into the buffer ready to run — `/lo`
                        # highlights `loop`, so a reflex double-Enter started
                        # autonomous work for a user reaching for `/login`. The
                        # common prefix cannot be the wrong command by
                        # construction: it is the part every candidate agrees on.
                        # This is also the shell idiom, which means the two-Enter
                        # rule stops being a special case users have to learn.
                        self._extend_to_common_prefix()
                    event.stop()
                    event.prevent_default()
                    return
        if key == "escape":
            # LAST escape branch on purpose: every picker case above has already
            # returned, so this is Esc with nothing on screen to dismiss.
            #
            # It has to be handled rather than left to bubble, because
            # ``TextArea`` binds Escape to ``blur``. That made Esc a silent trap:
            # the first press moved focus out of the composer (so the user's next
            # keystrokes went nowhere) and only a LATER press, once focus had
            # already left, reached the app's stop. Consuming the key here keeps
            # focus put and gives Esc one meaning — stop what the agent is doing.
            self.post_message(StopRequested())
            event.stop()
            event.prevent_default()
            return
        if key == "enter":
            self._submit()
            event.stop()
            event.prevent_default()
            return
        if key == "shift+enter":
            # Explicit newline; TextArea's stock path would also submit here,
            # so insert the newline ourselves and consume the key.
            self.insert("\n")
            event.stop()
            event.prevent_default()
            return
        if key == "ctrl+c":
            self.post_message(InterruptRequested())
            event.stop()
            event.prevent_default()
            return
        if key == "ctrl+d" and not self.text:
            self.post_message(EditorQuit())
            event.stop()
            event.prevent_default()
            return
        if key == "up" and self._caret_at_top_edge() and self._history:
            self._navigate_history(-1)
            event.stop()
            event.prevent_default()
            return
        if (
            key == "down"
            and self._caret_at_bottom_edge()
            and (self._history_index is not None or self._history)
        ):
            self._navigate_history(+1)
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)

    # -- submit -------------------------------------------------------------
    def _submit(self) -> None:
        text = self.text
        if text.strip():
            self._record_history(text)
        self._picker.close()
        self.post_message(EditorSubmitted(text))
        self.clear_content()

    # -- command completion -------------------------------------------------
    def edit(self, edit: Edit) -> EditResult:
        """Every buffer mutation funnels through here — resync the picker.

        Hooking the two mutation funnels (:meth:`edit` for inserts/deletes/
        undo, :meth:`load_text` for whole-buffer replacement) keeps the picker
        exact for keystrokes, pastes, and history navigation alike, and does
        it synchronously. Listening for ``TextArea.Changed`` instead would put
        the picker one message-loop tick behind the buffer, which is the tick
        in which Enter decides whether to complete.
        """
        result = super().edit(edit)
        self._sync_picker()
        return result

    def load_text(self, text: str) -> None:
        super().load_text(text)
        self._sync_picker()

    def _sync_picker(self) -> None:
        """Re-derive EVERY list from the buffer.

        The buffer is the single authority, so no picker holds state another could
        contradict: `slash_context` is live while the command word is open and
        `slash_argument` takes over on the terminating space, which makes "exactly
        one list is showing" a property of the parse rather than a rule the widgets
        have to cooperate on. The command picker serves both the word and a
        provider argument, so the branch below is which LIST it derives, not which
        widget is visible.
        """
        provider_argument = slash_argument(self.text, self.PROVIDER_COMMANDS)
        if provider_argument is None:
            self._provider_command = None
            self._picker.sync(self.text)
        else:
            command = self._command_word()
            if command != self._provider_command:
                self._provider_command = command
                # Drop the previous command's rows before asking for this one's.
                # `/login` offers every provider and `/logout` only the ones with a
                # credential, so carrying them across would briefly offer a logout
                # from an account the user never had.
                self._picker.set_choices([])
                self.post_message(ProviderQueryOpened(command or ""))
            self._picker.sync_argument(provider_argument)
        argument = slash_argument(self.text, self.MODEL_COMMANDS)
        if argument is None:
            if self._model_picker.is_open():
                self._model_picker.close()
            return
        if self._model_picker.is_open():
            self._model_picker.set_query(argument)
        else:
            self._model_picker.open(argument)
            # Transition only. Posting per keystroke would re-fetch every provider
            # for each character the user types into the query.
            self.post_message(ModelQueryOpened())

    def _command_word(self) -> str | None:
        """The lower-cased command word on the buffer's first non-blank line."""
        line = next((line for line in self.text.split("\n") if line.strip()), "")
        stripped = line.lstrip()
        if not stripped.startswith("/"):
            return None
        return stripped[1:].partition(" ")[0].lower()

    def _complete_model(self, row: ModelRow) -> None:
        """Put ``row``'s selector in the buffer without acting on it.

        No trailing space, unlike a command completion: the selector IS the whole
        argument, and a trailing space would terminate it and close the list — so
        Tab would appear to both fill the field and give up on it.
        """
        self.text = f"/model {row.selector}"
        self.move_cursor(self._end_of_buffer())

    def _apply_model(self, row: ModelRow) -> None:
        """Hand a chosen row to the app and clear the buffer.

        The buffer is cleared HERE rather than by the handler because the command
        never reaches the submit path: choosing from the list is the whole
        interaction, so leaving `/model anthropic/claude-opus-5` behind would
        invite a second Enter that ran the switch again.
        """
        self._model_picker.close()
        handler = self._on_model_chosen
        self.clear_content()
        if handler is not None:
            handler(row)

    def _picker_choice_is_unambiguous(self, name: str) -> bool:
        """Whether Enter may SEND the highlighted command, not just insert it.

        Unambiguous means one of three things:

        * the user ARROWED onto this row. The gate exists because the matcher may
          have picked the row on the user's behalf, and an explicit move is the
          direct answer to that — the user read the list and chose. It is also the
          muscle memory every comparable picker (fzf, an editor command palette)
          has already taught: move, Enter, done;
        * the user typed the name in full, so they named it rather than letting
          the matcher choose; or
        * there is exactly one match, so the highlighted row is the only command
          the query could mean — EXCEPT on a destructive list, below.

        Everything else completes and waits for a second Enter. The point is the
        registry's uneven blast radius: `/usage` is harmless and `/loop` and
        `/logout` are not, so a fuzzy pick must not be *run* on one keystroke.

        "Exactly one match" is not evidence on a DESTRUCTIVE list, and that is the
        one place the distinction pays. The matcher is a subsequence matcher, so a
        query that spells nothing can still leave a single survivor: `/logout oer`
        reached openrouter, `/logout dpsk` deepseek and `/logout xoh` xai-oauth —
        each one Enter away from deleting a credential the user never named. A
        typo one letter off a real id lands in exactly this shape. So `/logout`
        asks for the id in full or an explicit move, and nothing else counts.
        """
        query = self._picker_query()
        if query is None:
            return False
        if self._picker.chosen_by_hand:
            return True
        if query.strip().lower() == name.strip().lower():
            return True
        return not self._argument_is_destructive() and len(self._picker.suggestions()) <= 1

    def _argument_is_destructive(self) -> bool:
        """Whether the OPEN argument list removes something when a row is chosen.

        Read off the buffer's command word rather than the rows: what a keystroke
        destroys is a property of the command, and a per-row flag would make the
        gate depend on data the app happened to fill in.
        """
        return (
            self._picker.mode is PickerMode.ARGUMENT
            and self._provider_command in self.DESTRUCTIVE_COMMANDS
        )

    def _picker_query(self) -> str | None:
        """The text the open list is matching against, or ``None`` when closed.

        One gate, two lists: the command word and a provider argument are parsed
        by different functions but judged by the same rule, so the destructive
        case each protects (`/lo` reaching `loop`, `/logout an` reaching a stored
        credential) cannot drift apart.
        """
        if self._picker.mode is PickerMode.ARGUMENT:
            return slash_argument(self.text, self.PROVIDER_COMMANDS)
        context = slash_context(self.text)
        return None if context is None else context.query

    def _apply_command(self, name: str) -> None:
        """What CHOOSING a row does — the picker's ``on_choose`` callback.

        A command word is completed into the buffer and left there. A NON-
        destructive argument row is RUN, matching the model picker: a click names
        one exact row with a pointer, which is not the guess the keyboard
        ambiguity gate protects against, and a click that only filled the field
        would leave the user reaching for Enter to finish a choice they already
        made.

        A `/logout` row is filled in instead. The picker sits directly above the
        input row — the row a user clicks to place the caret — so "one click, one
        credential gone" is a misclick away, and an OAuth credential is not
        recoverable without another browser login. Confirming with Enter is the
        same two-step the keyboard gate already requires of the same command, so
        the mouse is not learning a separate rule.

        For a command the trailing space is load-bearing, not cosmetic: most
        commands take an argument (``/model provider/id``), and it is also what
        closes the picker — the word is now whitespace-terminated, so the list
        drops away on the same keystroke that chose from it.
        """
        if self._picker.mode is PickerMode.ARGUMENT:
            if self._argument_is_destructive():
                self._complete_argument(name)
                return
            self._run_argument(name)
            return
        context = slash_context(self.text)
        if context is None:
            return
        # Everything from the slash onward IS the command word (that is what
        # makes it a command word), so the prefix is all that must survive.
        self.text = f"{self.text[: context.start]}/{name} "
        self.move_cursor(self._end_of_buffer())

    def _resolve_argument(self, name: str, key: str, unambiguous: bool) -> None:
        """Tab/Enter on an argument row: complete it, or run it.

        Enter runs only what the same gate calls unambiguous, for the reason the
        gate exists at all: `/logout` DELETES a credential, so acting on a row the
        fuzzy matcher guessed would make one mis-keystroke destructive. An
        ambiguous Enter completes instead — the buffer then holds the exact id,
        which is one match, so the second Enter runs it.
        """
        if key == "tab" or not unambiguous:
            self._complete_argument(name)
            return
        self._run_argument(name)

    def _complete_argument(self, name: str) -> None:
        """Put ``name`` in the argument slot without acting on it.

        No trailing space, for the same reason as :meth:`_complete_model`: the
        space terminates the argument, so the matcher would stop matching and Tab
        would appear to fill the field and abandon it in one keystroke.
        """
        argument = slash_argument(self.text, self.PROVIDER_COMMANDS)
        if argument is None:
            return
        # The argument is by construction the TAIL of the buffer (everything after
        # the command word's space, on the only non-blank line), so trimming its
        # length off the end leaves exactly `…/logout ` to append onto.
        self.text = f"{self.text[: len(self.text) - len(argument)]}{name}"
        self.move_cursor(self._end_of_buffer())

    def _run_argument(self, name: str) -> None:
        """Complete ``name`` and submit, so the command's own handler runs it.

        Submitting the finished text rather than calling a callback keeps ONE
        implementation of what `/login anthropic` means: the app's slash dispatch.
        A second path would be a second place for the login flow to drift.
        """
        self._complete_argument(name)
        self._submit()

    def _extend_to_common_prefix(self) -> None:
        """Grow the typed word to the matches' longest common prefix, no further.

        No trailing space and NO close, unlike :meth:`_apply_command`: the word is
        still a prefix of several commands, so the list has to stay up for the
        user to keep narrowing. Returns having changed nothing when the query is
        already the common prefix — the honest outcome, since there is no
        keystroke-free way to tell the candidates apart at that point.
        """
        context = slash_context(self.text)
        if context is None:
            return
        names = [name for name, _ in self._picker.suggestions()]
        if not names:
            return
        # Case-insensitively, since the query is matched that way. The surviving
        # prefix is taken from the FIRST name so the inserted text keeps the
        # registry's own casing rather than the user's.
        shared = names[0]
        for name in names[1:]:
            while shared and not name.lower().startswith(shared.lower()):
                shared = shared[:-1]
            if not shared:
                return
        if len(shared) <= len(context.query):
            return
        self.text = f"{self.text[: context.start]}/{shared}"
        self.move_cursor(self._end_of_buffer())
        self._sync_picker()

    # -- history ------------------------------------------------------------
    def _caret_row(self) -> int:
        return self.selection.end[0]

    def _caret_at_top_edge(self) -> bool:
        return self._caret_row() == 0

    def _caret_at_bottom_edge(self) -> bool:
        return self._caret_row() == self.document.line_count - 1

    def _record_history(self, text: str) -> None:
        stripped = text.strip()
        if stripped and (not self._history or self._history[-1] != stripped):
            self._history.append(stripped)
            if len(self._history) > self.HISTORY_LIMIT:
                self._history.pop(0)
        self._history_index = None

    def _navigate_history(self, direction: int) -> None:
        if not self._history:
            return
        if self._history_index is None:
            if direction < 0:
                self._draft = self.text
                self._history_index = len(self._history) - 1
            else:
                return  # Down with no navigation active: nothing to restore
        else:
            self._history_index += direction
        if self._history_index >= len(self._history):
            # Past the newest entry: restore the draft and exit navigation.
            self._history_index = None
            self.text = self._draft
            self.move_cursor(self._end_of_buffer())
            return
        self._history_index = max(0, self._history_index)
        self.text = self._history[self._history_index]
        self.move_cursor(self._end_of_buffer())

    def _end_of_buffer(self) -> tuple[int, int]:
        last_row = self.document.line_count - 1
        return last_row, len(self.document.get_line(last_row))
