"""Conversation auto-naming — a short title derived from the opening message.

A conversation with no name is a row of timestamps in a picker, so a message
buys a cheap title. Five properties govern the design, and each one exists
because the obvious implementation gets it wrong:

- **It must never cost a turn.** The title is a nicety; the turn is the
  product. Both generators here swallow every exception and bound themselves
  with a timeout, so a provider that raises, stalls, rate-limits, or returns
  nonsense yields ``None`` and the band keeps whatever it had. The call also
  runs alongside the turn rather than in front of it, and carries
  ``ChatRequest.isolated`` so a failure cannot move the turn's model, its
  credential, or its effort — see that field's docstring for the six pieces
  of session-wide state that shuts off.
- **Most messages do not deserve a call.** "hi", "thanks", "test" carry no
  topic; asking a model to title them spends money to produce noise. The
  deterministic :func:`is_low_signal` filter answers those without any
  network at all, which is also what makes the behaviour testable offline.
- **A bad title is worse than no title.** The model is asked for 3-7 words
  inside ``<title></title>``; anything longer than the caps is REJECTED
  rather than truncated, because a title cut mid-word reads like a bug while
  an absent title reads like a conversation that has not been named yet.
- **A late title is nearly no title.** The generated title used to wait for
  the turn to settle, and a first turn on this product runs for minutes, so
  the tab wore `lo › <cwd>` for the whole time anyone was looking.
  :func:`provisional_title` names the conversation from the opener the instant
  it is submitted — no network at all — and the model's title now lands a
  second or two later, concurrently with the turn, rather than after it.
- **A conversation drifts, so a title has to be allowed to.**
  :func:`generate_retitle` gives the model the current title and a new message
  and lets IT judge whether the subject or the goals have materially moved.
  That judgement is not expressible as a keyword rule, and a title still
  naming the first of five subjects misidentifies the session rather than
  merely under-describing it.

The holder (:class:`ConversationName`) mirrors ``GoalState``: a small mutable
object the session and its host share, so a name that lands asynchronously is
visible to the next reader without rebuilding anything. ``user_set`` is the
precedence flag — an explicit rename outranks a generated title forever,
including one still in flight when the rename happens, and including every
later re-title.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

#: Hard caps on a stored title. Both are enforced as REJECTION, not
#: truncation: see the module docstring.
MAX_TITLE_CHARS = 80
MAX_TITLE_WORDS = 12

#: The custom transcript entry a conversation's title is journalled under, so a
#: RESUMED session wears the name it earned instead of booting nameless. The
#: constant lives here beside the holder it describes, exactly as
#: ``WAKE_SCHEDULES_CUSTOM_TYPE`` lives beside the scheduler: writer and reader
#: sit in different modules, and a literal spelled twice is one rename away
#: from a session that quietly stops restoring its own name.
CONVERSATION_NAME_CUSTOM_TYPE = "conversation_name"

#: How long a title generation may run before it is abandoned. Single attempt,
#: so this is the WHOLE budget — there is no retry behind it, which is what
#: makes the number worth measuring rather than guessing: seven naming and
#: re-titling calls against anthropic/claude-opus-5 at its lowest effort came
#: back in 5.4–5.8 s each. This was 20.0 while the call still waited for the
#: turn to settle, which is three and a half times that tail — room only a
#: wedged connection could ever use. Tightened to 15 s by the same change that
#: put the call ALONGSIDE the turn: the ceiling now bounds a task running
#: beside the user's work, and a title that has not arrived in two and a half
#: times the measured latency has nothing left to win — the failure is
#: swallowed and the provisional excerpt is already on the band. Not tighter
#: than that, because 2.5x the slowest call measured is the headroom that keeps
#: a call which was going to answer from being cut off. The bound exists at all
#: so a wedged connection cannot leave a task alive for the life of the session.
TITLE_TIMEOUT_S = 15.0

#: Caps on a PROVISIONAL title — the opener-derived label worn until the
#: model's title lands. Tighter than the caps above on purpose: those bound an
#: ANSWER ("what is this about"), while this quotes a REQUEST, and the first
#: twelve words of a request are usually still mid-sentence. Eight words at 48
#: characters is the point past which the band's trailing segment stops being
#: a label and starts being a sentence.
MAX_PROVISIONAL_WORDS = 8
MAX_PROVISIONAL_CHARS = 48

#: The system block for the naming call. Deliberately terse: it rides on EVERY
#: naming call and it is the half of the request we control, so every clause
#: has to earn its tokens. Measured against anthropic/claude-opus-5 on a short
#: opener, trimming the wordier first draft took the call from 177 to 120
#: input tokens for the same title. The sentinel form survives the trim
#: because "this input has no topic" must be expressible as an ANSWER rather
#: than as a malformed one — without it, models invent a title for "hi".
TITLE_SYSTEM_PROMPT = (
    "Name this conversation from the user's message.\n"
    "Reply with only <title>3 to 7 words</title>.\n"
    "No topic (a greeting or pleasantry): reply exactly <title/>.\n"
    "No quotes, no trailing punctuation."
)

#: The system block for a RE-titling call. It carries the current title and
#: asks the model — not a keyword heuristic here — whether the subject or the
#: goals have materially moved. Keeping the sentinel as "no change" makes the
#: common answer the cheapest one to produce and the cheapest one to parse:
#: `parse_title` already reads `<title/>` as "no title from this call".
RETITLE_SYSTEM_PROMPT = (
    "A conversation is titled: {current}\n"
    "Given the user's new message, decide if the SUBJECT or GOALS have "
    "materially changed.\n"
    "Unchanged, or a follow-up on the same subject: reply exactly <title/>.\n"
    "Materially changed: reply with only <title>3 to 7 words</title> naming "
    "the new subject.\n"
    "No quotes, no trailing punctuation."
)

#: How much of a message the naming call sees. Was 2000, which made the cheap
#: call the expensive one on exactly the input most likely to be pasted: a log.
#: Measured against anthropic/claude-opus-5 on a 3 KB traceback paste, with the
#: prefix cache defeated by a nonce — 2000 chars billed 899 input tokens (2
#: uncached plus an 897-token cache WRITE that nothing will ever read back,
#: since every naming prompt is different), 320 chars billed 237 and stayed
#: under the provider's cache-write floor entirely. Both produced the same
#: title. A title needs the ASK, and the ask is at the top of the message — a
#: request whose subject first appears 300 characters in is a request whose
#: first sentence would have named it anyway.
MAX_PROMPT_CHARS = 320

#: Messages that are content-free on their own. Matched with punctuation and
#: case stripped, so "Hi!" and "hi" are one entry.
#:
#: Two groups, and the second one is newer than the feature. The filter used
#: to see only OPENERS, where the content-free case is a greeting. Re-titling
#: (:func:`generate_retitle`) runs this over every FOLLOW-UP too, and the
#: content-free case there is an acknowledgement — "looks good", "lgtm",
#: "carry on". Those are most of the messages in a long session and not one of
#: them can have moved a subject, so recognising them here is what keeps the
#: re-title check off the common path entirely rather than merely throttled.
_LOW_SIGNAL_PHRASES = frozenset(
    {
        # Openers with no topic.
        "hi",
        "hii",
        "hey",
        "hello",
        "yo",
        "sup",
        "hiya",
        "howdy",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "hi there",
        "hey there",
        "hello there",
        "test",
        "testing",
        "ping",
        "hello world",
        "are you there",
        "you there",
        "help",
        # Follow-ups that acknowledge rather than ask.
        "thanks",
        "thanks again",
        "thank you",
        "thx",
        "ty",
        "cheers",
        "ok",
        "okay",
        "k",
        "cool",
        "nice",
        "nice one",
        "nice work",
        "good work",
        "great",
        "great thanks",
        "awesome",
        "excellent",
        "perfect",
        "beautiful",
        "looks good",
        "that looks good",
        "looks great",
        "lgtm",
        "works",
        "it works",
        "that works",
        "all good",
        "sounds good",
        "yes",
        "no",
        "yep",
        "yup",
        "nope",
        "sure",
        "done",
        "go ahead",
        "go on",
        "carry on",
        "keep going",
        "continue",
        "proceed",
    }
)

#: Words that stay lower-case when they are not the first word. Title casing
#: every word turns "Fix The Login Redirect Loop" into a headline; the brand
#: voice is sentence case, and the model's own casing is preserved for
#: everything else so real names ("GitLab", "macOS") survive untouched.
_TRAILING_PUNCTUATION = ".,;:!?-–—"

_TITLE_TAG_RE = re.compile(r"<title\s*>(.*?)</title\s*>", re.IGNORECASE | re.DOTALL)
_EMPTY_TITLE_RE = re.compile(r"<title\s*/\s*>", re.IGNORECASE)
_QUOTE_CHARS = "\"'`“”‘’«»"


def is_low_signal(text: str) -> bool:
    """True when ``text`` is not worth spending a naming call on.

    Deliberately conservative in one direction only: a false "low signal"
    costs a missing title, while a false "substantive" costs a provider call
    and a title like "Friendly Greeting Exchange". Anything with more than a
    handful of words is treated as substantive regardless of its wording,
    because the phrase list can only ever recognise openers it has seen.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return True
    # Strip surrounding punctuation and case so "Hi!!" collapses onto "hi".
    folded = cleaned.lower().strip(_TRAILING_PUNCTUATION + " " + _QUOTE_CHARS)
    if not folded:
        return True
    if folded in _LOW_SIGNAL_PHRASES:
        return True
    # A greeting with a tail ("hi, can you fix the parser?") is substantive;
    # only a bare greeting is not. Single tokens that are not words (a lone
    # emoji, "???") carry no topic either.
    if len(folded) <= 2 and not folded.isalnum():
        return True
    return False


def parse_title(raw: str) -> str | None:
    """Extract and normalise a title from a naming call's raw reply.

    Returns ``None`` for the ``<title/>`` sentinel, for a reply with no tag
    at all (a model that ignored the format is a model whose output we cannot
    trust to be a title), and for an answer that breaks either cap.
    """
    if not raw:
        return None
    if _EMPTY_TITLE_RE.search(raw) and not _TITLE_TAG_RE.search(raw):
        return None
    match = _TITLE_TAG_RE.search(raw)
    if match is None:
        return None
    body = match.group(1)
    # First line only: a model that appends a rationale must not smuggle it
    # into a one-row status band.
    first_line = next((line for line in body.splitlines() if line.strip()), "")
    cleaned = " ".join(first_line.split()).strip(_QUOTE_CHARS + " ")
    cleaned = cleaned.rstrip(_TRAILING_PUNCTUATION).strip()
    # Strip once more: a quoted title with a trailing period ("Fix login".)
    # leaves a stray quote after the punctuation pass.
    cleaned = cleaned.strip(_QUOTE_CHARS + " ")
    if not cleaned:
        return None
    if len(cleaned) > MAX_TITLE_CHARS:
        return None
    words = cleaned.split()
    if len(words) > MAX_TITLE_WORDS:
        return None
    return _sentence_case(words)


def cut_on_a_word(text: str, max_chars: int) -> str:
    """``text`` shortened to ``max_chars``, on a word boundary, with an ellipsis.

    The one definition of "shorten a title" in the product, so the band, the
    terminal tab and the stored name all cut the same string the same way. Three
    places grew their own version of this and two of them disagreed; a title
    that reads `…reconcile the ledge` on the tab and `…reconcile the…` on the
    band is a bug report waiting to be filed.

    The boundary is only taken when it costs less than a third of the budget: a
    single enormous token (a URL, a base64 blob) is cut mid-token instead,
    because returning almost nothing for a string that plainly had content is
    the worse failure. The ellipsis is counted, so the result never exceeds
    ``max_chars``.
    """
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    spaced = cut.rsplit(" ", 1)[0]
    if len(spaced) >= (max_chars - 1) * 2 // 3:
        cut = spaced
    return cut.rstrip(" " + _TRAILING_PUNCTUATION) + "…"


#: Module-private alias, so the store below reads as one operation.
_cut_on_a_word = cut_on_a_word


def _sentence_case(words: list[str]) -> str:
    """Capitalise the first word, leave every other word's casing alone.

    ``str.title()`` would destroy "macOS" and "gRPC"; lower-casing would
    destroy proper nouns. The model already emits names with their own
    casing, so the only correction needed is the leading word.
    """
    first = words[0]
    # Only lift an all-lower-case first word: "gRPC startup crash" must keep
    # its lower-case g, and an already-capitalised word needs no help.
    if first[:1].islower() and first.islower():
        first = first[:1].upper() + first[1:]
    return " ".join([first, *words[1:]])


def provisional_title(text: str) -> str:
    """An opener-derived label to wear until the generated title lands.

    Returns ``""`` for anything :func:`is_low_signal` rejects, so the caller
    gets the same "no title" answer from both halves of this module.

    Why this exists at all. A model call cannot be instant. The generated title
    goes out WITH the turn now — it carries ``ChatRequest.isolated``, so it is
    not in the turn's way and nothing has to wait for it — but it is still a
    round trip, and the seven calls measured for :data:`TITLE_TIMEOUT_S` came
    back in 5.4–5.8 s. Without a stand-in the band and the terminal tab wear
    the `lo › <cwd>` fallback for those seconds, on the exact frame the user is
    looking at: the one right after they pressed Enter.

    The wait used to be minutes rather than seconds, which is what made a
    stand-in worth writing: the title waited for the turn to settle, and
    a first turn on this product routinely runs for minutes, so the fallback
    was worn for the whole turn and became a title only once the work was
    already on screen. Measured on a real provider on that version: a
    29.7-second opening turn, the title stored 31.5 seconds after the prompt
    was submitted. Concurrency took the minutes out; this takes the seconds.

    The opener fixes those seconds for free. It is in hand the moment it is
    submitted, it costs no provider call at all, and it is the SAME text
    ``/resume``'s picker derives its row labels from (``resume.session_name``)
    — so the tab, the band and the picker agree on what a conversation is
    called by construction rather than by coincidence. The model's title, being
    an answer rather than a quote, is still better, and displaces this one the
    moment it lands (``OperatorApp._store_title``).

    Truncation, not rejection, and that is the one deliberate disagreement
    with :func:`parse_title`. An over-long answer from the model is evidence
    the model ignored the format, so it is thrown away; an over-long opener is
    just a long request, and the excerpt is the point. The cut is on a word
    boundary with an ellipsis so it reads as a quotation rather than as a
    string that ran out of buffer.
    """
    if is_low_signal(text):
        return ""
    words = " ".join((text or "").split()).split()
    kept: list[str] = []
    used = 0
    for word in words[:MAX_PROVISIONAL_WORDS]:
        width = used + len(word) + (1 if kept else 0)
        if kept and width > MAX_PROVISIONAL_CHARS:
            break
        kept.append(word)
        used = width
    if not kept:
        return ""
    # `_sentence_case` lifts an all-lower-case leading word, which is right for
    # the model's prose and wrong for a quote of whatever the user pasted: an
    # opener starting with a URL or a path came out as `Https://example.com/…`
    # and `Src/main.py …`, which reads as a rendering bug rather than as a
    # quotation. So the lift applies only to something that is actually a word —
    # letters, with an apostrophe or hyphen allowed ("don't", "well-known").
    if all(char.isalpha() or char in "'-" for char in kept[0]):
        label = _sentence_case(kept)
    else:
        label = " ".join(kept)
    # A single word longer than the whole cap (a URL, a stack frame, a base64
    # blob) is cut mid-token rather than dropped. Dropping it would return ""
    # for an opener that plainly had content, which puts the cwd fallback back
    # on screen for exactly the paste-heavy openers this feature is aimed at.
    if len(label) > MAX_PROVISIONAL_CHARS:
        return label[:MAX_PROVISIONAL_CHARS].rstrip() + "…"
    if len(kept) < len(words):
        # Strip the punctuation the cut landed on first: "fix the parser," +
        # "…" reads as a typo, while "fix the parser…" reads as an excerpt.
        return label.rstrip(_TRAILING_PUNCTUATION) + "…"
    return label


async def _ask_for_title(system: str, prompt: str, complete_fn, timeout: float) -> str | None:
    """One bounded call, every failure resolving to ``None``.

    Shared by :func:`generate_title` and :func:`generate_retitle` so the two
    have exactly one error policy between them. There is no retry here and
    none underneath (the session marks the request single-attempt), so the
    timeout is the entire budget.
    """
    try:
        raw = await asyncio.wait_for(complete_fn(system, prompt), timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        # CancelledError is caught deliberately: the naming task is detached
        # and routinely cancelled at shutdown, and letting that propagate
        # would surface a teardown traceback for a feature nobody waited on.
        return None
    except Exception:
        # EVERY provider failure, 429 included. The turn is running alongside
        # this call and must never learn it happened; the request is `isolated`
        # so the failure cannot have moved the turn's route or credential
        # either. See ``ChatRequest.isolated``.
        return None
    return parse_title(str(raw or ""))


def _errand_prompt(text: str) -> str:
    """``text`` collapsed to one line and trimmed to the prompt budget."""
    return " ".join((text or "").split())[:MAX_PROMPT_CHARS]


async def generate_title(
    text: str,
    complete_fn,
    *,
    timeout: float = TITLE_TIMEOUT_S,
) -> str | None:
    """One cheap naming call for ``text``; ``None`` when there is no title.

    ``complete_fn(system, prompt)`` is any awaitable one-shot completion (the
    session's :meth:`complete_once`). Every failure mode — low-signal input,
    a raising callable, a hanging callable, a reply that ignores the format,
    an over-long answer — resolves to ``None``. The caller therefore needs no
    error handling at all, which is the point: naming is decoration, and
    decoration that can break a turn is a defect.
    """
    if is_low_signal(text):
        return None
    return await _ask_for_title(TITLE_SYSTEM_PROMPT, _errand_prompt(text), complete_fn, timeout)


async def generate_retitle(
    current: str,
    text: str,
    complete_fn,
    *,
    timeout: float = TITLE_TIMEOUT_S,
) -> str | None:
    """A REPLACEMENT title when ``text`` has moved the subject; else ``None``.

    A long conversation drifts. It opens as "Fix the login redirect loop" and
    four messages later it is about the billing importer, and a title that
    still names the first thing is worse than one that names nothing — it
    actively misidentifies the session in a tab bar of five.

    The DECISION belongs to the model, and that is deliberate rather than
    convenient. "The subject has materially changed" is a judgement about
    meaning: any keyword rule written here would fire on "actually, forget the
    parser" and miss "right, and now the same thing for invoices". So the model
    is given the current title and the new message and answers with the
    sentinel to keep what it has. ``None`` therefore means BOTH "no change" and
    "the call failed", which are the same instruction to the caller: leave the
    title alone.

    Same cost and the same isolation as :func:`generate_title` — one bounded,
    single-attempt, tools-free call. What keeps it cheap in aggregate is the
    CALLER's throttle, not this function: see the TUI's re-title policy.
    """
    if not current or is_low_signal(text):
        return None
    system = RETITLE_SYSTEM_PROMPT.format(current=current)
    title = await _ask_for_title(system, _errand_prompt(text), complete_fn, timeout)
    if title is None:
        return None
    # A model that "changes" the title to the one it already has has answered
    # "no change" in the expensive spelling. Treat it as the sentinel so the
    # caller never repaints, never journals, and never resets its throttle on a
    # non-event.
    if title.casefold() == current.casefold():
        return None
    return title


@dataclass
class ConversationName:
    """Mutable holder for a conversation's title (empty = unnamed).

    Shared between the session and its host exactly as ``GoalState`` is, so a
    title arriving on a detached task is visible to the next reader without
    any callback plumbing.
    """

    text: str = ""
    #: True once a human named this conversation. A generated title must
    #: never overwrite that, including one already in flight when the rename
    #: lands — the flag is checked at STORE time, not at request time.
    user_set: bool = False
    #: True once a naming call has been requested for this conversation.
    #: Naming fires once per conversation; without this the second message
    #: would rename a conversation the user is already reading.
    requested: bool = False

    def set(self, text: str, *, user_set: bool = True) -> str:
        """Store a title; a generated one never displaces a user-set one.

        Returns what is stored afterwards (which may be the previous value
        when a generated title lost to a user-set one).

        An over-long title is cut on a WORD boundary with an ellipsis rather
        than sliced mid-word. Only ``/rename`` can reach this — a model's
        over-long answer is REJECTED by :func:`parse_title` rather than
        truncated — and a name the user typed is worth keeping legible: sliced,
        an 88-character rename ended `…and reconcile the ledge` on both the band
        and the terminal tab, which reads as a string that ran out of buffer.

        The cut lives HERE rather than in either display surface because this is
        where the length is actually decided: ``MAX_TITLE_CHARS`` and the tab's
        ``MAX_LABEL_CHARS`` are both 80, so a tab-side cut could never fire for a
        conversation name — every title reaching it had already been sliced by
        this line (design review round 2, D6).
        """
        cleaned = _cut_on_a_word(" ".join((text or "").split()), MAX_TITLE_CHARS)
        if not user_set and self.user_set:
            return self.text
        self.text = cleaned
        if user_set:
            self.user_set = True
        return self.text

    def claim_request(self) -> bool:
        """Reserve the one naming attempt; False when it is already spent."""
        if self.requested or self.user_set:
            return False
        self.requested = True
        return True
