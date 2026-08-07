"""Conversation auto-naming — a short title derived from the opening message.

A conversation with no name is a row of timestamps in a picker, so the first
substantive user message buys one cheap title. Three properties govern the
design, and each one exists because the obvious implementation gets it wrong:

- **It must never cost a turn.** The title is a nicety; the turn is the
  product. ``generate_title`` therefore swallows every exception and bounds
  itself with a timeout, so a provider that raises, stalls, or returns
  nonsense yields ``None`` and the band simply stays nameless. Callers run it
  detached from the turn (see the TUI's naming worker) — awaiting it inline
  would put a second provider round trip in front of the user's first reply.
- **Most openers do not deserve a call.** "hi", "thanks", "test" carry no
  topic; asking a model to title them spends money to produce noise. The
  deterministic :func:`is_low_signal` filter answers those without any
  network at all, which is also what makes the behaviour testable offline.
- **A bad title is worse than no title.** The model is asked for 3-7 words
  inside ``<title></title>``; anything longer than the caps is REJECTED
  rather than truncated, because a title cut mid-word reads like a bug while
  an absent title reads like a conversation that has not been named yet.

The holder (:class:`ConversationName`) mirrors ``GoalState``: a small mutable
object the session and its host share, so a name that lands asynchronously is
visible to the next reader without rebuilding anything. ``user_set`` is the
precedence flag — an explicit rename outranks a generated title forever,
including one still in flight when the rename happens.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

#: Hard caps on a stored title. Both are enforced as REJECTION, not
#: truncation: see the module docstring.
MAX_TITLE_CHARS = 80
MAX_TITLE_WORDS = 12

#: How long a title generation may run before it is abandoned. The call is
#: detached from the turn, so this bound is not about latency the user feels
#: — it is about not leaking a task that hangs for the life of the session
#: against a wedged provider connection.
TITLE_TIMEOUT_S = 20.0

#: The system block for the naming call. It asks for the sentinel form
#: explicitly so "this input has no topic" is expressible as an ANSWER rather
#: than as a malformed one — without it, models invent a title for "hi".
TITLE_SYSTEM_PROMPT = (
    "You name conversations. Given the user's opening message, reply with a "
    "title of 3 to 7 words that names what the conversation is about.\n"
    "Reply with nothing but the title wrapped in <title></title>.\n"
    "If the message is a greeting, a pleasantry, or carries no topic, reply "
    "with exactly <title/> and nothing else.\n"
    "Do not use quotation marks, trailing punctuation, or the word "
    "'conversation'."
)

#: How much of the opening message the naming call sees. A title needs the
#: topic, not the whole essay, and an unbounded prompt would make the cheap
#: call the expensive one on a pasted log.
MAX_PROMPT_CHARS = 2000

#: Openers that are content-free on their own. Matched against the message
#: with punctuation and case stripped, so "Hi!" and "hi" are one entry.
_LOW_SIGNAL_PHRASES = frozenset(
    {
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
        "thanks",
        "thank you",
        "thx",
        "ty",
        "cheers",
        "ok",
        "okay",
        "k",
        "cool",
        "nice",
        "great",
        "yes",
        "no",
        "yep",
        "nope",
        "sure",
        "test",
        "testing",
        "ping",
        "hello world",
        "are you there",
        "you there",
        "help",
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
    prompt = " ".join((text or "").split())[:MAX_PROMPT_CHARS]
    try:
        raw = await asyncio.wait_for(complete_fn(TITLE_SYSTEM_PROMPT, prompt), timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        # CancelledError is caught deliberately: the naming task is detached
        # and routinely cancelled at shutdown, and letting that propagate
        # would surface a teardown traceback for a feature nobody waited on.
        return None
    except Exception:
        return None
    return parse_title(str(raw or ""))


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
        """
        cleaned = " ".join((text or "").split())[:MAX_TITLE_CHARS]
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
