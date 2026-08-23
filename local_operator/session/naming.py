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
- **A conversation drifts, so a title has to be allowed to — toward the THEME,
  not the newest message.** :func:`generate_retitle` gives the model a sampled
  ``<chat>`` of the whole trajectory (the opening turns, which state the
  subject, plus a recent tail) anchored on the CURRENT title, and lets IT judge
  whether the body of work has genuinely moved on. That judgement is not
  expressible as a keyword rule. The earlier design showed the model only the
  single newest message, so an IN-GOAL step read as a brand-new subject: a
  session building a web-fetch tool got renamed to "Find Port Credit
  restaurants" the moment the user exercised the tool, then again on the next
  follow-up, each pivot compounding because the next check anchored on the
  already-drifted title. Titling the whole trajectory against the current-title
  anchor is what keeps a drifting session named after what it is actually
  about. The three parts that make this work — whole-trajectory context, the
  current-title anchor, and the caller's growth-gated refresh schedule — each
  fail alone; see :func:`build_theme_context`, :data:`THEME_SYSTEM_PROMPT`,
  and :func:`should_refresh_theme`.

The holder (:class:`ConversationName`) mirrors ``GoalState``: a small mutable
object the session and its host share, so a name that lands asynchronously is
visible to the next reader without rebuilding anything. ``user_set`` is the
precedence flag — an explicit rename outranks a generated title forever,
including one still in flight when the rename happens, and including every
later re-title.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

#: A history entry as the theme sampler reads it. Deliberately ``Any`` and
#: duck-typed via ``getattr`` rather than a ``Protocol`` or an import of the
#: harness ``AgentMessage`` union: the sampler only ever needs ``.role`` and
#: ``.text``, and a ``CustomMessage`` in that union legitimately carries no
#: ``role`` at all (it is filtered out here, not type-excluded). Typing this as
#: the real union would either drag a harness dependency into what is meant to
#: be a leaf module or fail to describe the role-less custom entries; reading
#: the two attributes defensively is both honest and keeps naming standalone.
_Turn = Any


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

#: The system block for a RE-titling call, ported from the omp coding-agent's
#: proven ``title-theme-system`` prompt. It titles the WHOLE body of work rather
#: than the newest message, which is the fix for the drift the old prompt
#: caused: shown only the latest message, the model read every in-goal step as a
#: new subject. The user message that rides beside this block is a sampled
#: ``<chat>`` trajectory (see :func:`build_theme_context`) that already carries a
#: ``<current-title>`` anchor, so the anchor lives in the DATA, not here — the
#: system block only has to teach the model what the scaffolding means and to
#: repeat the anchor verbatim unless the subject genuinely moved. Keeping the
#: ``<title/>`` sentinel as "no change" makes the common answer the cheapest one
#: to produce and to parse: `parse_title` already reads it as "no title from
#: this call", and `generate_retitle` also folds a verbatim restatement of the
#: current title back onto that sentinel.
THEME_SYSTEM_PROMPT = (
    "Write a 3 to 7 word title for the overall theme of the conversation in "
    "<chat>. Title the whole body of work, not the most recent message.\n"
    "<current-title> is the name this conversation already has. Repeat it "
    "verbatim unless the work has moved to a different subject. A new step, "
    "file, question, or tool call inside the same body of work is NOT a "
    "different subject.\n"
    "The earliest turns establish the subject; later turns only refine it. "
    "<elided/> marks turns left out.\n"
    "Never title one file, error, or tool call the conversation happened to "
    "touch.\n"
    "Reply with only <title>3 to 7 words</title>. No task, just small talk: "
    "reply exactly <title/>.\n"
    "Capitalize only the first word and names. No quotes, no trailing "
    "punctuation."
)

#: How many turns to sample from each end of the trajectory for the theme
#: context, ported from omp's ``THEME_CONTEXT_HEAD_TURNS`` / ``_TAIL_TURNS``.
#: The HEAD is what states the subject — the opening request is the only turn
#: that says what the session is FOR rather than a step inside it — so it is
#: kept larger relative to the tail, which merely refines the theme or shows a
#: genuine change of subject. A tail-only window is precisely why the old design
#: chased the newest message: by turn 40 the opener had scrolled out entirely,
#: leaving the model to name whatever the last message touched.
THEME_HEAD_TURNS = 3
THEME_TAIL_TURNS = 4

#: Per-turn character budget inside the sampled ``<chat>``. The whole point of
#: the retitle call staying cheap is that it never grows with the conversation:
#: a handful of turns, each trimmed to a sentence or two, is enough to state a
#: theme, and an unbounded sample would put a pasted log back into the call
#: `MAX_PROMPT_CHARS` exists to keep out. Bounded PER TURN, not on the assembled
#: envelope, so the head turns the sampler exists to preserve are never the ones
#: cut (they are usually the long ones).
THEME_TURN_CHARS = 240

#: Longest ``<current-title>`` embedded in the context. A stored title is capped
#: at `MAX_TITLE_CHARS` (80) but a `user_set` rename can reach it, so the anchor
#: is bounded like any other body rather than trusted to be short.
THEME_CURRENT_TITLE_CHARS = 120

#: The self-closing marker standing in for the turns dropped between the head
#: and the tail. Two disjoint fragments presented as adjacent read as an abrupt
#: topic switch and invite exactly the drift this sampler exists to prevent, so
#: the gap is always marked when anything falls between the two halves.
_ELIDED_MARKER = "<elided/>"

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
#: Stray / unclosed ``<title>`` fragments. A truncated reply often starts
#: ``<title>the login redirect loop`` and never closes; treating that as a
#: tagged title stored the markup. Ported from omp's
#: ``.replace(/<\/?title>/gi, "")`` on the untagged path.
_STRAY_TITLE_TAG_RE = re.compile(r"</?title\s*/?>", re.IGNORECASE)
_QUOTE_CHARS = "\"'`“”‘’«»"

#: Thinking envelopes some non-Anthropic models leak into the visible stream
#: (xAI, DeepSeek, Kimi, local Qwen). A ``<title>`` inside one of these is the
#: model talking to itself, not the answer; the last *visible* marked title
#: wins. Ported from omp's ``extractVisibleMarkedTitle`` — we do not invent a
#: second parser beside :func:`parse_title`.
_THINKING_TAG_RE = re.compile(
    r"<(think|thinking|reasoning)>\s*.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_THINKING_FENCE_RE = re.compile(
    r"```(?:thinking|reasoning)\b.*?```",
    re.IGNORECASE | re.DOTALL,
)
#: Unclosed thinking. A 1024-token naming reply that ran out of budget mid-
#: ``<think>`` or mid-`` ```thinking `` has no visible answer yet; the rest of
#: the string is still inside the envelope. Accepting it as a title is how
#: truncated reasoning became the session name. Closed envelopes still match
#: the pair regexes above and strip as they do today. Line-start only so a
#: title that *mentions* ``<think>`` or `` ```thinking `` is not treated as
#: an envelope (omp keeps "Fix <think> tag parsing").
_UNCLOSED_THINKING_TAG_RE = re.compile(
    r"^[ \t]*<(think|thinking|reasoning)\b[^>]*>",
    re.IGNORECASE | re.MULTILINE,
)
_UNCLOSED_THINKING_FENCE_RE = re.compile(
    r"^[ \t]*```(?:thinking|reasoning)\b",
    re.IGNORECASE | re.MULTILINE,
)
#: Untagged path only. A later marked title remains authoritative; a bare
#: "Thinking process:" reply is the model answering the user, not naming.
_THINKING_PREAMBLE_RE = re.compile(
    r"^[ \t]*(?:(?:here(?:['’]s| is)[ \t]+(?:a|the|my)[ \t]+)|my[ \t]+)?"
    r"(?:thinking|thought|reasoning)(?:[ \t]+process)?[ \t]*:?[ \t]*(?:\r?\n|$)",
    re.IGNORECASE,
)
#: Conversational first line on a multi-line untagged reply. "Sure, I'll name
#: this." is the model talking, not the title; the short line after it usually
#: is. Prefer that later line over keeping the preamble (review m1). A single
#: chatty line still goes through :func:`_normalise_title_body` and is
#: rejected by the caps, not truncated.
_CHATTY_PREAMBLE_RE = re.compile(
    r"^(?:sure|okay|ok|alright|right|got it|understood|of course|certainly|"
    r"absolutely|yeah|yep|yes)(?:[,!.\s].*)?$",
    re.IGNORECASE,
)
#: A later untagged line that already looks like a title: short, no trailing
#: sentence period. Used only after a chatty first line so we do not invent a
#: classifier for every multi-line reply.
_UNTAGGED_TITLE_LINE_RE = re.compile(r"^[^\n.]{1,80}$")
_FENCED_JSON_RE = re.compile(
    r"^```[^\n]*\s*(.*?)\s*```$",
    re.IGNORECASE | re.DOTALL,
)


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


def _strip_thinking_envelopes(text: str) -> str:
    """Drop leaked ``<think>`` / fenced-reasoning blocks from ``text``.

    Applied before any title lookup so a tag the model wrote *inside* a
    thinking envelope cannot win, and so an untagged short title sitting
    after a leaked preamble is still visible.
    """
    stripped = _THINKING_TAG_RE.sub("", text)
    return _THINKING_FENCE_RE.sub("", stripped)


def _cut_unclosed_thinking(text: str) -> str:
    """Drop the tail that still sits inside an unclosed thinking envelope.

    Closed pairs are already gone after :func:`_strip_thinking_envelopes`.
    An opener that remains is the truncated-reasoning case: everything
    from that line onward is still inside thinking, including a closed
    ``<title>`` the model drafted in there. Text *before* the opener is
    visible (a finished answer, then more thinking that never closed).
    """
    starts = [
        match.start()
        for pattern in (_UNCLOSED_THINKING_TAG_RE, _UNCLOSED_THINKING_FENCE_RE)
        for match in [pattern.search(text)]
        if match is not None
    ]
    if not starts:
        return text
    return text[: min(starts)]


def _unwrap_json_title(candidate: str) -> str:
    """``{"title": "..."}`` (optionally fenced) → the inner string.

    Some models emit the structured shape they were trained on for title
    tasks instead of the ``<title>`` tag. Without this the raw JSON became
    the session name — or, more often, was rejected as over-long / not a
    tag, which is how those sessions kept the opener excerpt. Truncated
    JSON is salvaged the same way omp's ``unwrapJsonTitle`` does: pull the
    quoted ``title`` value if the object itself will not parse.

    The fence language is ignored: `` ```python\\n{"title": "…"}\\n``` ``
    is still JSON, and treating the language tag as the title named
    sessions "Python". Only unwrap when the fence body starts with ``{``
    so a fenced prose title is left alone.
    """
    text = candidate.strip()
    fenced = _FENCED_JSON_RE.match(text)
    if fenced is not None:
        body = fenced.group(1).strip()
        if body.startswith("{"):
            text = body
    if not text.startswith("{"):
        return candidate
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        quoted = re.search(r'"title"\s*:\s*("(?:[^"\\]|\\.)*")', text)
        if quoted is None:
            return candidate
        try:
            salvaged = json.loads(quoted.group(1))
        except json.JSONDecodeError:
            return candidate
        return salvaged.strip() if isinstance(salvaged, str) else candidate
    if isinstance(parsed, dict) and isinstance(parsed.get("title"), str):
        return parsed["title"].strip()
    return candidate


def _looks_like_title_line(line: str) -> bool:
    """Cheap title-shaped check for a later untagged line (review m1).

    3–7 words, no trailing sentence period. Not a classifier: used only
    after a chatty first line so we prefer a later short line over keeping
    the preamble. Rejection, not truncation, still applies via
    :func:`_normalise_title_body`.
    """
    cleaned = " ".join(line.split()).strip(_QUOTE_CHARS + " ")
    if not cleaned or cleaned.endswith("."):
        return False
    words = cleaned.split()
    if not (3 <= len(words) <= 7):
        return False
    return _UNTAGGED_TITLE_LINE_RE.match(cleaned) is not None


def _untagged_candidate(visible: str) -> str:
    """Pick the untagged body: skip a chatty first line when a later one titles.

    First-line-only parse used to keep ``Sure, I'll name this.`` and drop
    the real title on the next line. Prefer a later short title-shaped
    line over that preamble; if nothing later looks like a title, leave
    the first line for :func:`_normalise_title_body` to reject.
    """
    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    if len(lines) >= 2 and _CHATTY_PREAMBLE_RE.match(lines[0]):
        for line in reversed(lines[1:]):
            if _looks_like_title_line(line):
                return line
        # Chatty preamble with no later title-shaped line: reject rather
        # than store the preamble. First-line-only parse used to keep it.
        return ""
    return visible


def _normalise_title_body(body: str) -> str | None:
    """Shared quote / punct / cap rejection for a candidate title body."""
    # First line only: a model that appends a rationale must not smuggle it
    # into a one-row status band. Multi-line untagged replies with a chatty
    # first line are already reduced by :func:`_untagged_candidate`.
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


def parse_title(raw: str) -> str | None:
    """Extract and normalise a title from a naming call's raw reply.

    Tagged output is preferred: a well-formed ``<title>...</title>`` that
    sits *outside* a leaked thinking envelope is the answer we asked for.
    Untagged short text is what non-Anthropic models actually emit (Grok,
    DeepSeek, Kimi, most OpenAI-compat local servers), so rejecting those
    replies is how Grok sessions silently kept the opener excerpt forever.
    ``<title/>``, empty, quotes-only, and anything over the caps still
    return ``None`` — over-long answers are rejected, not truncated.

    Unclosed markup is not a title. An unclosed ``<title>`` is stripped
    as a fragment and the remainder is parsed untagged (omp's policy).
    An unclosed ``<think>`` / `` ```thinking `` envelope means the rest
    of the string is still inside thinking, so the reply is discarded
    unless a CLOSED visible ``<title>`` already won.
    """
    if not raw:
        return None
    visible = _cut_unclosed_thinking(_strip_thinking_envelopes(raw))
    if _EMPTY_TITLE_RE.search(visible) and not _TITLE_TAG_RE.search(visible):
        return None
    matches = list(_TITLE_TAG_RE.finditer(visible))
    if matches:
        # Last visible marked title wins: a draft tag the model wrote
        # before the real one is common, and a tag inside thinking was
        # already stripped (closed) or cut (unclosed) so it cannot win.
        return _normalise_title_body(_unwrap_json_title(matches[-1].group(1)))
    # Strip stray / unclosed ``<title>`` fragments so the leftover short
    # phrase can still name the session, never with the tag characters.
    visible = _STRAY_TITLE_TAG_RE.sub("", visible)
    if _THINKING_PREAMBLE_RE.search(visible.lstrip()):
        return None
    unwrapped = _unwrap_json_title(_untagged_candidate(visible).strip())
    return _normalise_title_body(unwrapped)


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


#: Growth-gated refresh schedule, ported from omp's ``session-titling``. These
#: three numbers together produce a geometric spacing: with the transcript
#: turn-count stamped at each titling, a session is titled at turn 1, then
#: eligible to refresh at >=6, >=16, >=36, >=76, >=156, then never. The spacing
#: is deliberate — an early session is still deciding what it is about and
#: should re-title cheaply, while a long-running one has an established identity
#: and its name must stop tracking the cursor. See :func:`should_refresh_theme`.
#:
#: - MAX caps refreshes per session: past it the name is final, because the
#:   growth gate alone would still permit a rename after a long enough run.
#: - GROWTH_FACTOR requires the transcript to be a multiple of its length at the
#:   last titling. Growth, not elapsed turns, is the signal: doubling the
#:   conversation is roughly the point at which the earlier sample can no longer
#:   represent it, which is exactly when a fresh sample is worth paying for.
#: - MIN_TURNS is an absolute floor added to the growth requirement; it also
#:   carries the never-titled case (``last_titled_turn_count`` 0), where the
#:   gate reduces to four turns — enough for a request plus a reply to have
#:   established a subject.
THEME_REFRESH_MAX = 5
THEME_REFRESH_GROWTH_FACTOR = 2
THEME_REFRESH_MIN_TURNS = 4


def should_refresh_theme(turn_count: int, last_titled_turn_count: int, refresh_count: int) -> bool:
    """Whether the conversation's auto title may be regenerated now.

    Pure and side-effect free so the caller stays dumb: it counts turns and
    asks, rather than carrying its own idea of "enough has changed". This is
    the PRIMARY gate that replaced the wall-time-only throttle — a long session
    used to keep re-titling indefinitely because the only bound was 120 seconds
    between checks, so every in-goal follow-up an hour in was still eligible to
    pivot the name. Gating on transcript GROWTH instead bounds re-titles to a
    handful over the life of a session and concentrates them early, when the
    subject is still settling. A small time floor may still sit in front of this
    in the caller as a churn guard, but growth is what makes the schedule finite.
    """
    if refresh_count >= THEME_REFRESH_MAX:
        return False
    return (
        turn_count >= last_titled_turn_count * THEME_REFRESH_GROWTH_FACTOR + THEME_REFRESH_MIN_TURNS
    )


def _theme_turns(turns: Sequence[_Turn], newest: str) -> list[tuple[str, str]]:
    """Collect ``(role, text)`` pairs the theme sampler titles from.

    Only ``user``/``assistant`` turns with rendered text count: tool results
    and host-authored custom entries (which carry no ``role``) are noise for a
    THEME judgement, and a blank turn contributes nothing but scaffolding. The
    newest message is appended as a trailing user turn when it is not already
    the last one, because the retitle call fires at SUBMIT — before the turn has
    run — so ``session.history()`` does not yet contain it, and the tail is
    exactly where a genuine change of subject shows up.
    """
    collected: list[tuple[str, str]] = []
    for turn in turns:
        role = getattr(turn, "role", "")
        if role not in ("user", "assistant"):
            continue
        text = " ".join((getattr(turn, "text", "") or "").split())
        if not text:
            continue
        collected.append((role, text))
    newest_clean = " ".join((newest or "").split())
    if newest_clean and (not collected or collected[-1] != ("user", newest_clean)):
        collected.append(("user", newest_clean))
    return collected


def build_theme_context(
    turns: Sequence[_Turn], newest: str = "", *, current_title: str = ""
) -> str:
    """A sampled ``<chat>`` of the whole trajectory for the theme titling call.

    Samples the head and the tail instead of the last N turns, which is the
    heart of the drift fix: a tail-only window is precisely why the old design
    chased the newest message, because by turn 40 the opening request — the one
    turn that states what the session is FOR — had scrolled out of the window
    entirely, leaving the model to name whatever the last message touched. The
    head states the subject; the tail refines it or shows a genuine pivot.

    The ``<current-title>`` anchor leads the envelope so the model reads the
    name it is being asked to keep before it reads the turns. When turns fall
    between the head and the tail the gap is marked with ``<elided/>``: two
    disjoint fragments presented as adjacent read as an abrupt topic switch and
    invite exactly the drift this sampler exists to prevent.

    Returns ``""`` when there is nothing titleable, which the caller reads the
    same way it reads a low-signal message: spend no call.
    """
    collected = _theme_turns(turns, newest)
    if not collected:
        return ""
    head_end = min(THEME_HEAD_TURNS, len(collected))
    tail_start = max(head_end, len(collected) - THEME_TAIL_TURNS)
    sampled = collected[:head_end] + collected[tail_start:]
    # Index into ``sampled`` where the tail begins; the marker is emitted once,
    # immediately before it, and only when turns were actually dropped between
    # the two halves. Placed before the following turn rather than after the
    # preceding one so it never lands in the trailing position, where it would
    # read as "the conversation continues" instead of "turns were skipped here".
    elided_before = head_end if tail_start > head_end else None

    parts: list[str] = []
    header = ""
    title_clean = " ".join((current_title or "").split())
    if title_clean:
        header = (
            f"<current-title>\n{cut_on_a_word(title_clean, THEME_CURRENT_TITLE_CHARS)}\n"
            "</current-title>\n\n"
        )
    for index, (role, text) in enumerate(sampled):
        if elided_before is not None and index == elided_before:
            parts.append(_ELIDED_MARKER)
        # Bounded PER TURN, not on the assembled envelope: budgeting the whole
        # string would spend its allowance on the long head turns the sampler
        # exists to preserve and cut them mid-tag. `cut_on_a_word` keeps the cut
        # legible, and the tag scaffolding it sits inside stays intact.
        body = cut_on_a_word(text, THEME_TURN_CHARS)
        parts.append(f"<{role}>\n{body}\n</{role}>")
    return f"<chat>\n{header}" + "\n\n".join(parts) + "\n</chat>"


async def generate_title(
    text: str,
    complete_fn,
    *,
    timeout: float = TITLE_TIMEOUT_S,
) -> str | None:
    """One cheap naming call for ``text``; ``None`` when there is no title.

    ``complete_fn(system, prompt)`` is any awaitable one-shot completion (the
    session's :meth:`complete_once`). Every failure mode — low-signal input,
    a raising callable, a hanging callable, a thinking preamble, or an
    over-long answer — resolves to ``None``. The caller therefore needs no
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
    turns: Sequence[_Turn] | None = None,
    timeout: float = TITLE_TIMEOUT_S,
) -> str | None:
    """A REPLACEMENT title when the THEME has moved; else ``None``.

    A long conversation drifts. It opens as "Fix the login redirect loop" and
    four messages later it is about the billing importer, and a title that
    still names the first thing is worse than one that names nothing — it
    actively misidentifies the session in a tab bar of five.

    But the earlier design over-corrected: given only ``current`` and the single
    newest ``text``, the model read every IN-GOAL step as a new subject. A
    session building a web-fetch tool got renamed the instant the user exercised
    it ("find Port Credit restaurants"), then again on the next follow-up. The
    fix is to title the WHOLE body of work: ``turns`` is the session's history
    (``session.history()``), from which :func:`build_theme_context` samples a
    ``<chat>`` of the opening turns (which state the subject) plus a recent tail,
    anchored on ``current``. The model keeps the anchor verbatim unless the
    subject genuinely moved — which a new step inside the same work is not.

    The DECISION still belongs to the model, and that is deliberate rather than
    convenient. "The subject has materially changed" is a judgement about
    meaning: any keyword rule written here would fire on "actually, forget the
    parser" and miss "right, and now the same thing for invoices". The model
    answers with the ``<title/>`` sentinel to keep what it has, so ``None`` means
    BOTH "no change" and "the call failed" — the same instruction to the caller:
    leave the title alone.

    Same cost and the same isolation as :func:`generate_title` — one bounded,
    single-attempt, tools-free call. What keeps it cheap in aggregate is the
    CALLER's growth-gated schedule (:func:`should_refresh_theme`), not this
    function. ``turns`` defaults to ``None`` for callers with no history handy;
    the newest ``text`` alone is then the whole trajectory, which is the old
    behaviour and still correct for a two-message session.
    """
    if not current or is_low_signal(text):
        return None
    context = build_theme_context(turns or (), text, current_title=current)
    if not context:
        return None
    title = await _ask_for_title(THEME_SYSTEM_PROMPT, context, complete_fn, timeout)
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
        cleaned = cut_on_a_word(" ".join((text or "").split()), MAX_TITLE_CHARS)
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
