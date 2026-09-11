"""QA round 1 — adversarial probes against ``/title refresh`` (PR #942).

Not a second copy of ``test_conversation_naming.py``'s coverage. Everything
here is an ATTACK: the sequences that should NOT release the ``user_set``
latch, the races the two round-fix commits exist for, the provider shapes that
are neither a title nor a clean failure, and the regressions sitting beside the
change.

The load-bearing invariant under test throughout: ``user_set`` is a one-way
latch that ONLY ``release_user_set`` reopens, only an explicit refresh calls
it, and only once a replacement title is actually in hand. Every probe below
asserts the LATCH STATE, not merely the visible title — the two diverge
exactly where the defect would be.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.session import naming
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _transcript_text
from tests.unit.tui.test_conversation_naming import (
    _GatedSession,
    _named,
    _ready,
    _settle,
)


def _turns(*texts: str) -> list[Any]:
    return [
        SimpleNamespace(role="user" if index % 2 == 0 else "assistant", text=text)
        for index, text in enumerate(texts)
    ]


async def _boot(title: str = "") -> tuple[OperatorApp, _GatedSession]:
    session = _GatedSession()
    session.title = title
    return OperatorApp(lambda: _factory(session)), session


# -- 1. argument parsing: can any input LOSE a title or fake a refresh? -------


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        # Whitespace-only arguments are the REPORT branch, never a store.
        ("", (False, "")),
        ("   ", (False, "")),
        ("\t\n ", (False, "")),
        # The bare verbs, and the case-insensitivity the casefold buys.
        ("refresh", (True, "")),
        ("REFRESH", (True, "")),
        ("ReFrEsH", (True, "")),
        ("  refresh\t", (True, "")),
        ("update", (True, "")),
        ("retitle", (True, "")),
        # Flag spellings, same casefold.
        ("--refresh", (True, "")),
        ("--REFRESH", (True, "")),
        ("--auto", (True, "")),
        ("--update", (True, "")),
        ("--retitle", (True, "")),
        ("--refresh\n", (True, "")),
        # The terminator. `--` alone yields no text, which every handler's
        # `not title` branch answers with the REPORT rather than a blank store.
        ("--", (False, "")),
        ("-- --refresh", (False, "--refresh")),
        ("-- refresh", (False, "refresh")),
        ("-- -- refresh", (False, "-- refresh")),
        # Near-misses: a flag with prose after it is a TITLE, not a verb.
        ("--refresh the billing importer", (False, "--refresh the billing importer")),
        ("--auto scaling notes", (False, "--auto scaling notes")),
        # A single dash was never an option marker.
        ("-refresh", (False, "-refresh")),
        # Verbs merely CONTAINED in a title stay titles.
        ("refresh the importer", (False, "refresh the importer")),
        ("update", (True, "")),
        ("updates", (False, "updates")),
        # Unicode look-alikes must not reach the verb set.
        ("\uff32\uff25\uff26\uff32\uff25\uff33\uff28", (False, "ＲＥＦＲＥＳＨ")),
        ("r\u0435fresh", (False, "r\u0435fresh")),
    ],
)
def test_parse_title_arg_classifies_every_edge(arg: str, expected: tuple[bool, str]) -> None:
    """No input may silently turn a title into a refresh, or a refresh into a title."""
    assert naming.parse_title_arg(arg) == expected


@pytest.mark.parametrize(
    "arg", ["---refresh", "--refres", "--nope", "--=refresh", "--refresh=1", "--" + "x" * 100]
)
def test_an_unknown_bare_flag_raises_rather_than_becoming_a_title(arg: str) -> None:
    """Storing a ``--``-leading token as a title is the failure this vocabulary exists
    to prevent: nobody types a real title starting with ``--``."""
    with pytest.raises(ValueError, match="unknown title option"):
        naming.parse_title_arg(arg)


def test_the_terminator_reaches_a_title_that_is_exactly_the_verb() -> None:
    """``/title -- refresh`` must STORE ``refresh``, not refresh. The escape hatch
    is the only thing standing between a user who wants that title and a command
    that quietly does something else."""
    is_refresh, title = naming.parse_title_arg("-- refresh")
    assert (is_refresh, title) == (False, "refresh")
    name = naming.ConversationName()
    assert name.set(title, user_set=True) == "refresh"
    assert name.user_set


def test_a_newline_bearing_title_is_collapsed_not_truncated() -> None:
    """A pasted multi-line title must reach the band as one row with all of its
    words, not cut at the first newline."""
    _, title = naming.parse_title_arg("billing\nimporter rewrite")
    name = naming.ConversationName()
    assert name.set(title, user_set=True) == "billing importer rewrite"


def test_a_500_character_title_is_cut_on_a_word_and_never_rejected() -> None:
    """Over-long is the user's words, so it is TRIMMED; only a model's answer is
    rejected outright by ``parse_title``."""
    _, title = naming.parse_title_arg(" ".join(["reconciliation"] * 60))
    name = naming.ConversationName()
    stored = name.set(title, user_set=True)
    assert len(stored) <= naming.MAX_TITLE_CHARS
    assert name.user_set
    assert not stored.endswith("reconcil")  # cut on a word, not mid-word


# -- 2. the user_set latch: every sequence that must NOT release it -----------


@pytest.mark.parametrize(
    ("label", "answer"),
    [
        ("the model declined", "<title/>"),
        ("the model returned nothing", ""),
        ("the answer was over-long", "<title>" + " ".join(["word"] * 40) + "</title>"),
        ("the answer restated the name", "<title>Ledger reconciliation</title>"),
    ],
)
@pytest.mark.asyncio
async def test_a_refresh_that_lands_no_title_keeps_the_latch(label: str, answer: str) -> None:
    """THE load-bearing rule. Released on any of these, automatic naming would
    silently re-arm over a name the user typed and is still keeping."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        session.grow_transcript(4)
        session.set_conversation_name("Ledger reconciliation", user_set=True)

        session.title = answer
        app._run_slash_command("/title refresh")
        await _settle()

        assert session.conversation_name == "Ledger reconciliation", label
        assert session.conversation_name_state.user_set, f"the rename was revoked: {label}"
        # And the latch still bites: a generated title cannot displace it.
        assert session.set_conversation_name("Generated", user_set=False) == "Ledger reconciliation"


@pytest.mark.asyncio
async def test_a_refresh_whose_provider_is_dead_keeps_the_latch_and_says_so() -> None:
    """A wedged provider reported as "unchanged" is the one outcome that actively
    misinforms — the user is handed a judgement that never happened."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        session.grow_transcript(4)
        session.set_conversation_name("Ledger reconciliation", user_set=True)

        async def dead(system: str, prompt: str) -> str:
            raise RuntimeError("429 rate limited")

        session.complete_once = dead  # type: ignore[method-assign]
        app._run_slash_command("/title refresh")
        await _settle()

        assert session.conversation_name == "Ledger reconciliation"
        assert session.conversation_name_state.user_set, "a dead provider revoked the rename"
        text = _transcript_text(app)
        assert "could not reach the model" in text
        assert "title unchanged: Ledger" not in text, "a dead provider reported as a judgement"


@pytest.mark.asyncio
async def test_a_refresh_on_an_empty_transcript_keeps_the_latch() -> None:
    """Nothing-yet spends no call, so there is certainly no replacement in hand."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        session._history = []  # nothing titleable
        session.set_conversation_name("Ledger reconciliation", user_set=True)
        before = len(session.completions)

        app._run_slash_command("/title refresh")
        await _settle()

        assert len(session.completions) == before, "a call was spent on an empty transcript"
        assert session.conversation_name == "Ledger reconciliation"
        assert session.conversation_name_state.user_set


@pytest.mark.asyncio
async def test_automatic_naming_resumes_after_a_release_then_yields_to_a_new_rename() -> None:
    """The half nobody writes: the latch must re-LATCH. A refresh hands the
    conversation back to automatic naming, and the very next ``/rename`` must
    take it away again — otherwise the release is a one-way door in the other
    direction and a generated title can stomp a fresh rename."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        session.grow_transcript(4)
        session.set_conversation_name("Ledger reconciliation", user_set=True)

        session.title = "<title>Billing importer rewrite</title>"
        app._run_slash_command("/title refresh")
        await _settle()
        assert session.conversation_name == "Billing importer rewrite"
        assert not session.conversation_name_state.user_set, "the release did not happen"

        # Auto-naming genuinely resumed.
        assert session.set_conversation_name("Auto chosen", user_set=False) == "Auto chosen"

        # …and a rename immediately afterwards re-latches and cannot be stomped.
        app._run_slash_command("/title Ledger reconciliation again")
        await _settle()
        assert session.conversation_name == "Ledger reconciliation again"
        assert session.conversation_name_state.user_set, "the rename did not re-latch"
        assert (
            session.set_conversation_name("Auto stomp", user_set=False)
            == "Ledger reconciliation again"
        )


# -- 3. concurrency and the wrong-conversation class of bug ------------------


@pytest.mark.asyncio
async def test_a_rename_landing_mid_call_outranks_the_refresh_and_keeps_its_latch() -> None:
    """The race the round-fix commits exist for, driven at the seam rather than
    inferred. The refresh is parked on its gate; a ``/rename`` lands; the answer
    then arrives deciding against a title no longer in force.

    Storing it would revoke the user's most recent instruction in favour of
    their previous one — and strip the latch protecting it on the way past.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        session.grow_transcript(4)

        gate = asyncio.Event()
        session.name_gate = gate
        session.name_started = asyncio.Event()
        session.title = "<title>Billing importer rewrite</title>"

        app._run_slash_command("/title refresh")
        await asyncio.wait_for(session.name_started.wait(), timeout=5)

        # The rename lands while the provider call is still in flight.
        app._run_slash_command("/title Ledger reconciliation")
        await _settle()
        assert session.conversation_name == "Ledger reconciliation"

        gate.set()
        await _settle()

        assert (
            session.conversation_name == "Ledger reconciliation"
        ), "the refresh overwrote a rename"
        assert session.conversation_name_state.user_set, "the refresh stripped the rename's latch"
        text = _transcript_text(app)
        assert "title refreshed" not in text, "credited itself with a name it did not choose"


@pytest.mark.asyncio
async def test_two_refreshes_racing_do_not_double_release_or_cross_paint() -> None:
    """Running the same operation twice, interleaved. The first is superseded by
    the second's generation bump and must store nothing."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        session.grow_transcript(4)
        session.set_conversation_name("Ledger reconciliation", user_set=True)

        gate = asyncio.Event()
        session.name_gate = gate
        session.name_started = asyncio.Event()
        session.title = "<title>First answer</title>"

        app._run_slash_command("/title refresh")
        await asyncio.wait_for(session.name_started.wait(), timeout=5)
        app._run_slash_command("/title refresh")
        await _settle()

        session.title = "<title>Second answer</title>"
        gate.set()
        await _settle()

        # Exactly one of them may win; neither may leave a half-applied state
        # (a released latch with the old title still in force).
        name = session.conversation_name
        latched = session.conversation_name_state.user_set
        assert name in {"First answer", "Second answer", "Ledger reconciliation"}
        if name == "Ledger reconciliation":
            assert latched, "the latch was released without a replacement landing"
        else:
            assert not latched, "a landed refresh left the latch on"


@pytest.mark.asyncio
async def test_a_refresh_dispatched_before_a_new_does_not_paint_the_replacement() -> None:
    """The bug two commits were spent on: a refresh must never repaint the
    conversation the user navigated TO."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        session.grow_transcript(4)

        gate = asyncio.Event()
        session.name_gate = gate
        session.name_started = asyncio.Event()
        session.title = "<title>Billing importer rewrite</title>"

        app._run_slash_command("/title refresh")
        await asyncio.wait_for(session.name_started.wait(), timeout=5)

        # The conversation's identity is replaced while the call is in flight.
        app._name_generation += 1
        await _settle()

        gate.set()
        await _settle()

        assert session.conversation_name == "Fix the login flow", "a superseded refresh painted"
        assert "title refreshed: Billing importer rewrite" not in _transcript_text(app)


@pytest.mark.asyncio
async def test_a_refresh_whose_source_retired_mid_call_paints_nothing() -> None:
    """A sidebar switch away from this conversation retires its source. The
    answer arriving afterwards must touch neither the title nor the transcript:
    this is the ``source.retired`` half of the guard, distinct from the
    generation bump, and it is the shape the two round-fix commits address.
    """
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        session.grow_transcript(4)
        session.set_conversation_name("Ledger reconciliation", user_set=True)

        gate = asyncio.Event()
        session.name_gate = gate
        session.name_started = asyncio.Event()
        session.title = "<title>Billing importer rewrite</title>"

        app._run_slash_command("/title refresh")
        await asyncio.wait_for(session.name_started.wait(), timeout=5)

        # The user navigates away; this conversation's source is retired.
        app._interaction.retired = True
        gate.set()
        await _settle()

        assert session.conversation_name == "Ledger reconciliation", "a retired source painted"
        assert session.conversation_name_state.user_set, "a retired refresh stripped the latch"
        assert "title refreshed" not in _transcript_text(app)


@pytest.mark.asyncio
async def test_the_refresh_stamps_the_schedule_on_its_OWN_source_not_the_current_one() -> None:
    """``source`` is captured at DISPATCH precisely so a switch during the call
    cannot stamp an innocent conversation's naming schedule. Here the app's
    ``_interaction`` is moved out from under the in-flight worker; the counters
    that move must be the dispatching source's, and the bystander's must not.
    """
    from local_operator.tui.session_interaction import SessionInteraction

    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        session.grow_transcript(6)
        mine = app._interaction
        mine.naming.refresh_count = 3

        gate = asyncio.Event()
        session.name_gate = gate
        session.name_started = asyncio.Event()
        session.title = "<title>Billing importer rewrite</title>"

        app._run_slash_command("/title refresh")
        await asyncio.wait_for(session.name_started.wait(), timeout=5)

        # A bystander conversation becomes current while the call is in flight.
        bystander = SessionInteraction(session=None)
        bystander.naming.refresh_count = 7
        bystander.naming.last_titled_turn_count = 99
        app._interaction = bystander

        gate.set()
        await _settle()

        assert bystander.naming.refresh_count == 7, "stamped an innocent conversation"
        assert bystander.naming.last_titled_turn_count == 99, "stamped an innocent conversation"
        assert mine.naming.refresh_count == 0, "the dispatching source was not re-seeded"


@pytest.mark.asyncio
async def test_a_landed_refresh_reseeds_the_schedule_rather_than_spending_the_budget() -> None:
    """``/title refresh`` re-decides the conversation's identity, so the drift
    clock restarts: the baseline is re-seeded and the refresh budget is reset,
    on THIS interaction's counters."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        session.grow_transcript(8)
        app._interaction.naming.refresh_count = 3
        app._interaction.naming.last_titled_turn_count = 1

        session.title = "<title>Billing importer rewrite</title>"
        app._run_slash_command("/title refresh")
        await _settle()

        assert session.conversation_name == "Billing importer rewrite"
        assert app._interaction.naming.refresh_count == 0, "a user-asked refresh was charged"
        assert app._interaction.naming.last_titled_turn_count >= 1


@pytest.mark.asyncio
async def test_a_refresh_that_lands_nothing_rearms_naming_on_an_unnamed_session() -> None:
    """The compounding failure the ``finally`` exists for: a refresh dispatched
    while the opening naming call was in flight supersedes it, and without the
    re-arm the conversation is left unnamed with naming permanently disarmed."""
    app, session = await _boot(title="")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._submit_prompt("fix the login redirect loop")
        session.gate.set()
        await _settle()
        assert not session.conversation_name
        session.grow_transcript(4)

        # The precondition that makes this test mean anything: naming's one
        # attempt must be SPENT when the refresh runs. Without this the assert
        # below passes vacuously against a flag that was already False, which is
        # exactly how a missing re-arm survives its own regression test
        # (verified by mutation: reverting the `finally` did not fail this test
        # until this line was added).
        app._interaction.naming.requested = True

        session.title = ""  # the refresh lands nothing either
        app._run_slash_command("/title refresh")
        await _settle()

        assert not session.conversation_name
        assert not app._interaction.naming.requested, "naming was left permanently disarmed"

    # And the converse: a refresh that DID land a title leaves the latch spent,
    # because the conversation now has the name naming exists to give it.
    app, session = await _boot(title="")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        app._submit_prompt("fix the login redirect loop")
        session.gate.set()
        await _settle()
        session.grow_transcript(4)
        app._interaction.naming.requested = True

        session.title = "<title>Login redirect loop</title>"
        app._run_slash_command("/title refresh")
        await _settle()

        assert session.conversation_name == "Login redirect loop"
        assert app._interaction.naming.requested, "a landed title must leave the latch spent"


# -- 4. provider failure injection: the receipt must stay honest -------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "answer", "outcome"),
    [
        ("declined", "<title/>", naming.TITLE_UNCHANGED),
        ("empty", "", naming.TITLE_UNCHANGED),
        ("whitespace only", "<title>   </title>", naming.TITLE_UNCHANGED),
        (
            "over MAX_TITLE_WORDS",
            "<title>" + " ".join(["w"] * 40) + "</title>",
            naming.TITLE_UNCHANGED,
        ),
        ("over MAX_TITLE_CHARS", "<title>" + "x" * 200 + "</title>", naming.TITLE_UNCHANGED),
        ("restates the anchor", "<title>Fix the login flow</title>", naming.TITLE_UNCHANGED),
        ("a real move", "<title>Billing importer rewrite</title>", naming.TITLE_REFRESHED),
    ],
)
async def test_refresh_title_classifies_every_provider_shape(
    label: str, answer: str, outcome: str
) -> None:
    """An over-long or empty answer is REJECTED by ``parse_title`` and must come
    back as "the name still fits", never stored raw."""

    async def answer_fn(system: str, prompt: str) -> str:
        return answer

    result = await naming.refresh_title(
        "Fix the login flow", answer_fn, turns=_turns("fix the login redirect loop", "done")
    )
    assert result.outcome == outcome, label
    if result.changed:
        assert len(result.title) <= naming.MAX_TITLE_CHARS
        assert len(result.title.split()) <= naming.MAX_TITLE_WORDS


@pytest.mark.asyncio
async def test_a_hung_provider_is_bounded_by_the_timeout_and_reported_as_unavailable() -> None:
    """The timeout is the entire budget — there is no retry here or underneath."""
    started = asyncio.Event()

    async def hang(system: str, prompt: str) -> str:
        started.set()
        await asyncio.sleep(30)
        return "<title>never</title>"

    loop = asyncio.get_running_loop()
    began = loop.time()
    result = await naming.refresh_title(
        "Fix the login flow", hang, turns=_turns("fix the login redirect loop"), timeout=0.2
    )
    assert result.outcome == naming.TITLE_UNAVAILABLE
    assert loop.time() - began < 5, "the timeout did not bound the call"
    assert naming.refresh_receipt(result, "Fix the login flow") == (
        "could not reach the model — the title is unchanged"
    )


@pytest.mark.asyncio
async def test_a_naming_failure_never_reaches_the_turn_beside_it() -> None:
    """Naming is decoration: no provider failure may fail the command or the turn."""
    for boom in (
        RuntimeError("429 rate limited"),
        asyncio.TimeoutError(),
        ValueError("malformed response"),
        KeyError("model"),
    ):

        async def raiser(system: str, prompt: str, _boom: BaseException = boom) -> str:
            raise _boom

        result = await naming.refresh_title(
            "Fix the login flow", raiser, turns=_turns("fix the login redirect loop")
        )
        assert result.outcome == naming.TITLE_UNAVAILABLE, boom


# -- 5. the routed / detached path ------------------------------------------


@pytest.mark.asyncio
async def test_the_routed_path_uses_the_tighter_deadline() -> None:
    """``ROUTED_TITLE_TIMEOUT_S`` is sized against the CLIENT's ack deadline, not
    the model's: a routed refresh landing after the socket closed would store a
    title and still report a lost connection."""
    assert naming.ROUTED_TITLE_TIMEOUT_S < naming.TITLE_TIMEOUT_S

    from local_operator.session.runtime.serving import ServingSessionHandle

    seen: dict[str, float] = {}
    real = naming.refresh_title

    async def spy(current, fn, **kwargs):
        seen["timeout"] = kwargs.get("timeout", naming.TITLE_TIMEOUT_S)
        return await real(current, fn, **kwargs)

    session = FakeSession()
    session.set_conversation_name("Fix the login flow", user_set=True)

    async def answer(system: str, prompt: str) -> str:
        return "<title>Billing importer rewrite</title>"

    session.complete_once = answer  # type: ignore[method-assign]
    session._history = _turns("fix the login redirect loop", "done")

    runtime = ServingSessionHandle.__new__(ServingSessionHandle)
    runtime._publish_name = lambda: None  # type: ignore[method-assign]

    naming.refresh_title = spy  # type: ignore[assignment]
    try:
        result = await runtime._title_refresh_slash(session, _SlashResult)
    finally:
        naming.refresh_title = real  # type: ignore[assignment]

    assert seen["timeout"] == naming.ROUTED_TITLE_TIMEOUT_S
    assert "title refreshed: Billing importer rewrite" in result.text
    assert not session.conversation_name_state.user_set, "the routed path did not release"


class _SlashResult:
    """Stand-in for the runtime's injected ``SlashResult`` constructor."""

    def __init__(
        self, *, kind: str = "", text: str = "", style: str = "", data: Any = None
    ) -> None:
        self.kind, self.text, self.style, self.data = kind, text, style, data


@pytest.mark.asyncio
async def test_the_routed_path_agrees_with_the_tui_on_what_a_refresh_does() -> None:
    """Two surfaces, one release rule: a phone and a terminal cannot drift on
    whether a refresh reopens the latch."""
    from local_operator.session.runtime.serving import ServingSessionHandle

    runtime = ServingSessionHandle.__new__(ServingSessionHandle)
    runtime._publish_name = lambda: None  # type: ignore[method-assign]

    # Unchanged: the latch stays.
    session = FakeSession()
    session.set_conversation_name("Ledger reconciliation", user_set=True)
    session._history = _turns("fix the login redirect loop", "done")

    async def declined(system: str, prompt: str) -> str:
        return "<title/>"

    session.complete_once = declined  # type: ignore[method-assign]
    result = await runtime._title_refresh_slash(session, _SlashResult)
    assert session.conversation_name == "Ledger reconciliation"
    assert session.conversation_name_state.user_set, "the routed path revoked a rename"
    assert "title unchanged: Ledger reconciliation" in result.text

    # Dead provider: the latch stays and the receipt is honest.
    async def dead(system: str, prompt: str) -> str:
        raise RuntimeError("429")

    session.complete_once = dead  # type: ignore[method-assign]
    result = await runtime._title_refresh_slash(session, _SlashResult)
    assert session.conversation_name_state.user_set
    assert "could not reach the model" in result.text


@pytest.mark.asyncio
async def test_a_rename_landing_mid_call_outranks_the_ROUTED_refresh_too() -> None:
    """The routed twin of the rename-during-the-call race, driven at the seam.

    Worth its own probe because the routed path has no worker and no generation
    stamp — the standing-title re-read is its ONLY defence. A ``/rename`` from
    the owning terminal while a phone's refresh is in flight must not be
    overwritten, and its latch must survive.
    """
    from local_operator.session.runtime.serving import ServingSessionHandle

    runtime = ServingSessionHandle.__new__(ServingSessionHandle)
    runtime._publish_name = lambda: None  # type: ignore[method-assign]

    session = FakeSession()
    session.set_conversation_name("Fix the login flow", user_set=True)
    session._history = _turns("fix the login redirect loop", "done")

    started = asyncio.Event()
    release = asyncio.Event()

    async def gated(system: str, prompt: str) -> str:
        started.set()
        await release.wait()
        return "<title>Billing importer rewrite</title>"

    session.complete_once = gated  # type: ignore[method-assign]
    task = asyncio.ensure_future(runtime._title_refresh_slash(session, _SlashResult))
    await asyncio.wait_for(started.wait(), timeout=5)

    # The rename lands from the terminal while the phone's call is in flight.
    session.set_conversation_name("Ledger reconciliation", user_set=True)
    release.set()
    result = await asyncio.wait_for(task, timeout=5)

    assert session.conversation_name == "Ledger reconciliation", "the routed refresh overwrote"
    assert session.conversation_name_state.user_set, "the routed refresh stripped the latch"
    assert "title unchanged: Ledger reconciliation" in result.text
    assert "refreshed" not in result.text, "credited itself with a name it did not choose"


@pytest.mark.asyncio
async def test_the_routed_rename_branch_refuses_an_unknown_flag() -> None:
    """The raising parser reaches the detached runtime too — an unknown flag must
    be refused there rather than stored as a title on a phone."""
    from local_operator.session.runtime.serving import ServingSessionHandle

    runtime = ServingSessionHandle.__new__(ServingSessionHandle)
    session = FakeSession()
    result = await runtime._rename_slash(session, "---refresh", _SlashResult)
    assert result.style == "warning"
    assert "unknown title option" in result.text
    assert session.conversation_name == "", "a typo became a title"


# -- 6. regression: the neighbours the alias change touches ------------------


def test_every_alias_resolves_to_its_registry_primary_name() -> None:
    """``primary_slash_name`` is what stops an ALIAS falling past every branch of
    the routed dispatchers and being refused as unsupported."""
    from local_operator.slash_commands import SLASH_COMMANDS, primary_slash_name

    assert primary_slash_name("title") == "rename"
    assert primary_slash_name("rename") == "rename"
    # Unknown words pass through so the caller's own fallback still sees them.
    assert primary_slash_name("definitely-not-a-command") == "definitely-not-a-command"
    # And the property holds for the WHOLE registry, not just the one alias.
    for entry in SLASH_COMMANDS:
        for alias in entry.names:
            assert primary_slash_name(alias) == entry.name, alias


@pytest.mark.asyncio
async def test_plain_rename_still_behaves_exactly_as_before() -> None:
    """The regression sitting immediately beside the change: ``/rename <words>``
    spends no provider call, latches, and reports the stored title."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        before = len(session.completions)

        app._run_slash_command("/rename Ledger reconciliation")
        await _settle()

        assert session.conversation_name == "Ledger reconciliation"
        assert session.conversation_name_state.user_set
        assert len(session.completions) == before, "/rename spent a provider call"
        assert session.set_conversation_name("Generated", user_set=False) == "Ledger reconciliation"


@pytest.mark.asyncio
async def test_a_bare_title_reports_and_stores_nothing() -> None:
    """Bare ``/title`` is the REPORT branch on a named and an unnamed session
    alike — and ``/title --`` reaches the same branch rather than storing ''."""
    app, session = await _boot(title="<title>Fix the login flow</title>")
    async with app.run_test(size=(100, 30)) as pilot:
        await _ready(pilot, app)
        await _named(app, session, "fix the login redirect loop")
        before = len(session.completions)

        for arg in ("", " ", "--"):
            app._run_slash_command(f"/title {arg}".rstrip())
            await _settle()
            assert session.conversation_name == "Fix the login flow", arg
            assert len(session.completions) == before, f"a report spent a call: {arg!r}"
        assert "conversation: Fix the login flow" in _transcript_text(app)
