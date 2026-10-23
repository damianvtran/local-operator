"""App-specific entries for the inherited ``ctrl+p`` command palette.

Textual gives ``OperatorApp`` a command palette for free: ``App.__init__``
injects a priority binding on ``ctrl+p`` and the palette gathers providers from
``App.COMMANDS``. Until this module existed the app registered none of its own,
so the palette opened onto the framework's three system actions (quit,
screenshot, theme) and nothing the operator actually came for.

WHAT THIS IS NOT: a second command registry. Every entry below is DERIVED from
the surface that already owns it — ``SLASH_COMMANDS`` for commands,
``settings_io.SECTIONS`` for settings, the session store for conversations and
the live session's agent/team registries for names. Selecting an entry hands a
line to the SAME dispatcher a typed command uses (``_run_slash_command``) or
opens the SAME settings page ``/settings`` opens, so a palette hit and the typed
spelling cannot drift into two behaviours.

ONE PROVIDER, DELIBERATELY. ``App.COMMANDS`` is a *set*, and Textual sorts
discovery hits by score — all of which are 0 — so several providers would put
their rows in an order that changes between launches (Python's per-process hash
seed decides set iteration). One provider therefore yields everything in a
fixed order: Textual's own system actions (quit, screenshot, theme) first,
exactly where they were before this slice, then commands (the thing a palette is
for), settings, agents and teams, then recent conversations. The framework's
provider is DELEGATED to rather than reimplemented — its rows and callbacks are
still Textual's, only their position in the sequence is ours. Typed queries
still rank by fuzzy score across the whole catalogue.

ENTRIES ARE PURE DATA. ``build_palette_entries`` takes already-resolved inputs
and returns ``PaletteEntry`` rows, so ranking, filtering, duplicate labels and
the empty catalogue are unit-testable without a Screen. The app method
``_palette_entries`` is the one seam that reads live state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable, Sequence

from rich.markup import escape
from rich.text import Text
from textual.command import CommandPalette, DiscoveryHit, Hit, Hits, Provider

from local_operator.i18n.keys import tui_palette as _p
from local_operator.i18n.text import tr

_t = tr("tui.palette")

#: What a palette row IS. The value is also the i18n key suffix for the dim tag
#: drawn beside the label, so the two cannot disagree about the vocabulary.
KIND_COMMAND = "command"
KIND_SETTING = "setting"
KIND_SESSION = "session"
KIND_AGENT = "agent"
KIND_TEAM = "team"

_KIND_MESSAGES = {
    KIND_COMMAND: _p.kind_command,
    KIND_SETTING: _p.kind_setting,
    KIND_SESSION: _p.kind_session,
    KIND_AGENT: _p.kind_agent,
    KIND_TEAM: _p.kind_team,
}


@dataclass(frozen=True)
class PaletteEntry:
    """One palette row: what it reads as, and what selecting it does.

    ``slash`` is the line handed to ``OperatorApp._run_slash_command`` — the
    same entry point a typed command uses. ``section`` is set only for a
    settings row and names the ``settings_io`` section to land on; those entries
    have no ``slash`` because the page is opened directly by the handler
    ``/settings`` already uses.
    """

    kind: str
    label: str
    description: str = ""
    slash: str = ""
    section: str = ""
    #: Space-separated alias words a query may match without them being drawn.
    aliases: str = ""
    #: Whether this row is offered before the user has typed anything. A row
    #: that is False is still searchable; it only stays out of the discovery
    #: list, which is how conversations past the recent cap are reachable
    #: without burying everything else on an empty query.
    discovery: bool = True

    @property
    def kind_label(self) -> str:
        """The localized one-word type tag shown beside the label."""
        return _t(_KIND_MESSAGES[self.kind]())

    @property
    def match_text(self) -> str:
        """The string a query is matched against (see `rank_entries`).

        Aliases are part of a command's identity to the user — `/quit` must be
        reachable by typing ``quit`` — but they are not worth a second line in
        the list, so they match without being displayed.
        """
        return f"{self.label} {self.aliases}".strip() if self.aliases else self.label

    @property
    def help_text(self) -> str:
        """The dim suffix: the type tag, then the source's own description.

        A settings section's or command's description is written for its own
        surface; it is included because "which of these two `/project`-ish
        rows did I mean" is answered by the description, not the name. Plain
        text is escaped before Textual renders it as markup: slash descriptions
        carry bracketed tokens such as ``[Paste #N]``, which ``Content`` would
        otherwise read as a style tag.
        """
        tag = self.kind_label
        if not self.description:
            return escape(tag)
        return escape(f"{tag} · {self.description}")


def build_palette_entries(
    *,
    slash_commands: Iterable[Any],
    sections: Sequence[Any],
    session_rows: Iterable[Any] = (),
    agent_rows: Iterable[Sequence[str]] = (),
    team_names: Iterable[str] = (),
    session_limit: int | None = 12,
) -> list[PaletteEntry]:
    """The palette catalogue, in its discovery order.

    ``slash_commands`` are ``SlashCommand`` rows (name, aliases, description);
    ``sections`` are ``settings_io.Section`` rows; ``session_rows`` are
    ``resume.SessionRow`` rows, most recent first; ``agent_rows`` are the
    ``(name, label, facts, summary)`` tuples ``/agent`` lists; ``team_names``
    are the names ``/team`` accepts.

    ``session_limit`` caps only the DISCOVERY list, never the search catalogue:
    a few hundred conversations would bury every other kind before the user has
    typed anything, while a query still has to be able to reach all of them
    (`/resume` made the same uncapped-scan decision for its picker). ``None``
    means no cap.
    """
    entries: list[PaletteEntry] = []

    for command in slash_commands:
        entries.append(
            PaletteEntry(
                kind=KIND_COMMAND,
                label=f"/{command.name}",
                description=command.description,
                slash=f"/{command.name}",
                aliases=" ".join(command.aliases),
            )
        )

    for section in sections:
        # The section NAME is the stable identity the settings page navigates
        # by; the title is localized and may change with the language.
        entries.append(
            PaletteEntry(
                kind=KIND_SETTING,
                label=section.title,
                description=section.description,
                section=section.name,
            )
        )

    for row in agent_rows:
        name, label, _facts, summary = row
        entries.append(
            PaletteEntry(
                kind=KIND_AGENT,
                label=label or name,
                description=summary,
                # Address by the registry KEY, never the display label: `/agent`
                # resolves the first token as the name. The key also MATCHES,
                # even when a chosen label is what is drawn — the key is what
                # the user types to reach this agent.
                slash=f"/agent ={name}",
                aliases=name,
            )
        )

    # ONE ROW PER NAME, first wins, compared on the same stripped casefold
    # `_team_choices` uses — the composite registry can hold a session team and
    # an account team under one spelling, and two identically-labelled rows
    # that route to the same name are noise.
    seen_teams: set[str] = set()
    for name in team_names:
        key = name.strip().casefold()
        if key in seen_teams:
            continue
        seen_teams.add(key)
        entries.append(
            PaletteEntry(
                kind=KIND_TEAM,
                label=name,
                # `=` is `/team`'s literal-name escape: without it a team named
                # `chart`, `new` or `edit` would open the chart/authoring
                # subcommand instead of attaching (`/agent` takes the same
                # escape, for its own reserved words).
                slash=f"/team ={name}",
            )
        )

    for index, row in enumerate(session_rows):
        entries.append(
            PaletteEntry(
                kind=KIND_SESSION,
                label=row.name or row.id,
                # The id is what the resume handler resolves, and two
                # conversations can legitimately share an opening line; showing
                # the id distinguishes the duplicate rows the user is choosing
                # between without making every row a hex address.
                description=row.id,
                slash=f"/resume {row.id}",
                discovery=session_limit is None or index < session_limit,
            )
        )

    return entries


def rank_entries(entries: Sequence[PaletteEntry], matcher: Any) -> list[tuple[PaletteEntry, float]]:
    """Entries a query matches, best score first, ties in catalogue order.

    Matching is on the LABEL only — for a command that means its name and
    aliases (``/exit`` is reachable by typing ``quit``), never its description.
    A description is prose an operator has not read yet; matching it would put
    rows in the list for words the visible line does not contain, which reads
    as the palette answering a question nobody asked.

    The sort is stable, so equal scores keep the catalogue's order — commands
    before settings before names before conversations — instead of whatever
    order the fuzzy matcher happened to visit them in.
    """
    scored: list[tuple[PaletteEntry, float]] = []
    for entry in entries:
        score = matcher.match(entry.match_text)
        if score > 0:
            scored.append((entry, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def palette_is_open(app: Any) -> bool:
    """Whether ``app``'s current screen is a command palette.

    A one-line seam around ``CommandPalette.is_open`` because that frame is
    typed ``App[object]`` and this app is ``App[None]`` — invariant, so pyright
    refuses the direct call at every site. The ignore lives here once instead
    of at each caller; the predicate itself stays the framework's.
    """
    return OperatorCommandPalette.is_open(app)  # type: ignore[arg-type]


class OperatorCommandProvider(Provider):
    """The app's catalogue, as the palette's one provider.

    ``startup`` runs in the provider's own task before any search, so the
    session-store scan happens off the input path. The catalogue is read ONCE
    per palette open: the palette is a momentary surface, and re-scanning the
    store on every keystroke would be a file walk per character typed.
    """

    def __init__(self, screen: Any, match_style: Any = None) -> None:
        super().__init__(screen, match_style)
        self._entries: list[PaletteEntry] = []
        #: Textual's system-command provider, delegated to for its rows. Held
        #: rather than reimplemented: the actions are the framework's and this
        #: app must not grow a second copy of "quit"/"screenshot" any more than
        #: it may grow a second copy of ``/help``.
        self._system: Any = None

    async def startup(self) -> None:
        from textual.system_commands import SystemCommandsProvider

        self._system = SystemCommandsProvider(self.screen, self.match_style)
        await self._system.startup()
        # `OperatorApp` is the only screen stack this provider is registered
        # on; the cast keeps the import out of this module (app.py imports us).
        app = self.app
        # The session-store scan is uncapped (see `_palette_session_rows`) and
        # cold it costs seconds. It runs in a thread so the palette paints and
        # accepts keys while the rows are gathered; `Provider.startup` is
        # already awaited off the search path.
        rows = await asyncio.to_thread(app._palette_session_rows)  # type: ignore[attr-defined]
        self._entries = app._palette_entries(session_rows=rows)  # type: ignore[attr-defined]

    async def shutdown(self) -> None:
        if self._system is not None:
            await self._system.shutdown()
            self._system = None

    def _hit_command(self, entry: PaletteEntry) -> Any:
        return partial(self.app._run_palette_entry, entry)  # type: ignore[attr-defined]

    async def discover(self) -> Hits:
        if self._system is not None:
            async for hit in self._system.discover():
                yield hit
        for entry in self._entries:
            if not entry.discovery:
                continue
            yield DiscoveryHit(
                # A `Text`, never a bare `str`: the framework turns a string
                # display through `Content.from_markup`, so a session title
                # such as ``fix [/] thing`` raised MarkupError inside the
                # gather worker (a query that never finishes) and ``[WIP]``
                # silently lost its brackets. Names are user data, not markup.
                display=Text(entry.label),
                command=self._hit_command(entry),
                text=entry.label,
                help=entry.help_text,
            )

    async def search(self, query: str) -> Hits:
        if self._system is not None:
            async for hit in self._system.search(query):
                yield hit
        matcher = self.matcher(query)
        for entry, score in rank_entries(self._entries, matcher):
            yield Hit(
                score,
                matcher.highlight(entry.label),
                self._hit_command(entry),
                text=entry.label,
                help=entry.help_text,
            )


class OperatorCommandPalette(CommandPalette):
    """Textual's palette with the app's two user-visible strings localized.

    Everything else — the input, the list, the busy indicator, the key
    handling — is the framework's, unchanged: this class exists because
    ``CommandPalette`` hardcodes its placeholder and its no-match line in
    English, and the app's copy ships in six locales.

    The no-match override copies ``CommandPalette._start_no_matches_countdown``
    (textual 8.2.8) rather than wrapping it, because the string is built inside
    the timer closure where no subclass seam exists. ``test_command_palette``
    pins the two private names it depends on (``_NO_MATCHES_COUNTDOWN``,
    ``_NO_MATCHES``) so a framework upgrade that moves them fails there instead
    of silently reverting the palette to English.
    """

    def __init__(
        self,
        providers: Any = None,
        *,
        placeholder: str | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(
            providers=providers,
            placeholder=placeholder if placeholder is not None else _t(_p.placeholder()),
            name=name,
            id=id,
            classes=classes,
        )

    def _start_no_matches_countdown(self, search_value: str) -> None:
        from rich.align import Align
        from rich.text import Text
        from textual.command import CommandList
        from textual.widgets.option_list import Option

        self._stop_no_matches_countdown()

        def _show_no_matches() -> None:
            if search_value:
                command_list = self.query_one(CommandList)
                command_list.add_option(
                    Option(
                        Align.center(
                            Text(_t(_p.no_matches()), style="not bold"),
                        ),
                        disabled=True,
                        id=self._NO_MATCHES,
                    )
                )
                self._list_visible = True
            else:
                # Discover mode with nothing to discover is left unmarked, as
                # the framework does: there is no query to have missed.
                self._list_visible = False

        self._no_matches_timer = self.set_timer(
            self._NO_MATCHES_COUNTDOWN,
            _show_no_matches,
        )
