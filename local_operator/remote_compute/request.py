"""The run-create body: a network is REQUIRED, and it is the tenant field.

WHY A MODEL INSTEAD OF A DICT. The server keys a remote run by the network it
was submitted through; a request built without one would have to be refused at
the far end, after the work was already addressed to a host. Making
``binding`` a required field means a later slice cannot construct the body at
all without deciding the tenant — the type checker catches the omission, not a
remote 4xx.

WHY A BINDING AND NOT AN ID. ``NetworkBinding`` is issued only by
``resolve_network``, which answers from this device's local mesh store. A bare
``str`` would accept ANY well-formed id, so the membership guarantee would hold
only while every caller happened to resolve; requiring the binding makes it a
type fact. The id on the wire is read from the binding.

THE REMAINING FIELDS ARE FROZEN IN :class:`RunSpec`, NOT IN THE REQUEST. The
request still takes a mapping (:attr:`RunCreateRequest.body`) so its one
invariant stays legible, but the tier-1 fields the wire contract names are
built by ``RunSpec`` and cross into the request through ``RunSpec.bind``. The
split is deliberate: adding typed fields BESIDE the mapping would give the same
body two spellings, and a body that disagrees with itself is exactly what the
``network_id`` rule below already refuses. A body may never carry its own
``network_id``, because that is the second spelling of the tenant field.
"""

from __future__ import annotations

import dataclasses
import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from local_operator.remote_compute.resolve import (
    CODE_DUPLICATE_NETWORK_ID,
    CODE_INVALID_NETWORK_ID,
    NETWORK_ID_PATTERN,
    NetworkBinding,
)


@dataclasses.dataclass(frozen=True)
class RunCreateRequest:
    """A run-create body bound to one network, ready to serialise.

    Build it from a :class:`~local_operator.remote_compute.resolve.NetworkBinding`
    — the only value that proves membership — and the id travels from there.
    The shape check is repeated here with ``fullmatch`` because this object is
    the last place before the id goes on the wire; it is the same constant, not
    a second rule.
    """

    binding: NetworkBinding
    #: Everything else the run-create body carries, unchanged into the JSON.
    body: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        # IMPORTED HERE so this module's import stays stdlib + resolve (which is
        # itself light); a refusal is raised out of the same typed family.
        from local_operator.network.types import MeshRefusal

        # ``isinstance`` rather than a type annotation alone: annotations are
        # not enforced at runtime, and a bare id is the exact mistake this model
        # exists to make unreachable.
        if not isinstance(self.binding, NetworkBinding):
            raise TypeError(
                "RunCreateRequest takes a NetworkBinding issued by "
                "remote_compute.resolve_network(); a bare network id cannot "
                "prove this device is a member of the network it names"
            )
        if not NETWORK_ID_PATTERN.fullmatch(self.binding.network_id):
            raise MeshRefusal(
                CODE_INVALID_NETWORK_ID,
                f"{self.binding.network_id!r} is not a minted mesh network id; "
                "resolve the network first (remote_compute.resolve_network) and "
                "use its binding",
            )
        if "network_id" in self.body:
            raise MeshRefusal(
                CODE_DUPLICATE_NETWORK_ID,
                "the run body may not carry its own `network_id`: the tenant is a "
                "field of this model, and two spellings of it could disagree",
            )

    @property
    def network_id(self) -> str:
        """The tenant id, read from the binding — never from a caller's string."""
        return self.binding.network_id

    def to_json(self) -> dict[str, Any]:
        """The wire object: the caller's fields, then the tenant field on top."""
        return {**self.body, "network_id": self.binding.network_id}


#: The compute sizes ``POST /v1/remote-compute/runs`` accepts.
RUN_SIZES: tuple[str, ...] = ("s", "m", "l")

#: A full commit SHA: exactly 40 hexadecimal characters.
#:
#: ``--sha`` is a COMMIT, not a ref. A branch name or an abbreviated hash reads
#: the same in a body and is resolved by the far end at a moment this caller
#: cannot see, so the commit that ran might not be the one that was asked for.
COMMIT_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}")

#: The URL forms a remote run can check out: ``https://``, ``ssh://``, or the
#: scp-like ``git@host:path`` spelling. Anything else (a local path, ``git://``,
#: ``file://``) either cannot be reached from the far end or names this disk.
REPO_URL_PATTERN = re.compile(r"(?:https://|ssh://)[^\s]+|git@[^\s:]+:[^\s]+")


def is_commit_sha(value: object) -> bool:
    """Whether ``value`` is a full 40-hex commit SHA."""
    return isinstance(value, str) and COMMIT_SHA_PATTERN.fullmatch(value) is not None


def is_repo_url(value: object) -> bool:
    """Whether ``value`` is a repository URL the far end can clone."""
    return isinstance(value, str) and REPO_URL_PATTERN.fullmatch(value) is not None


#: The server's bound on a run's ``name``: the value is TRIMMED and then measured
#: in UTF-8 BYTES, not characters, and a body outside the bound is a 400.
MAX_RUN_NAME_BYTES = 100

#: Marks a shortened default name, so a label cut at the bound does not read as
#: the whole prompt. It is reserved from the budget rather than added on top.
RUN_NAME_TRUNCATION_MARK = "…"

#: Interior characters an explicit run name may not carry (see
#: ``resolve_run_name``): controls, and the line and paragraph separators.
_REFUSED_NAME_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})

#: The bidirectional embedding, override, pop and isolate characters:
#: LRE/RLE/PDF/LRO/RLO (U+202A-U+202E) and LRI/RLI/FSI/PDI (U+2066-U+2069).
#: Other ``Cf`` characters (ZWJ in an emoji sequence, for one) are legitimate
#: in a label and stay allowed.
_BIDI_CONTROLS = frozenset(
    chr(code_point) for code_point in (*range(0x202A, 0x202F), *range(0x2066, 0x206A))
)

#: Unicode White_Space, which is exactly the set Go's ``strings.TrimSpace`` —
#: the server's own trim — removes. PYTHON'S ``str.strip`` IS A WIDER SET: it
#: follows ``str.isspace``, whose definition adds the file/group/record/unit
#: separators U+001C-U+001F (bidirectional class ``B``), so it trims characters
#: the far end keeps.
_SERVER_WHITESPACE = (
    frozenset("\t\n\v\f\r \x85\xa0\u1680\u2028\u2029\u202f\u205f\u3000")
    # EN QUAD .. HAIR SPACE: the eleven Zs code points U+2000-U+200A.
    | frozenset(chr(code_point) for code_point in range(0x2000, 0x200B))
)

#: Categories that ATTACH to a preceding character rather than standing on their
#: own: non-spacing (Mn) and enclosing (Me) marks. A first line made only of
#: these paints nothing, so it names a run as badly as a lone NUL does.
#:
#: NOT ``Cf``/``Cc``: :func:`_printable_run_name` has already removed those, and
#: NOT ``Mc`` either, because a spacing combining mark carries its own advance
#: width and stays visible.
_NON_SPACING_CATEGORIES = frozenset({"Mn", "Me"})


#: The refusal for a name the server would not accept, as ONE message for
#: every condition — blank, over the byte bound, more than one line — because
#: they have one remedy: leave ``--name`` out.
#:
#: WHY NOT A KEYED SENTENCE OF ITS OWN. New English copy needs the five
#: carried locales translated in the same change, and an unaudited
#: translation stales their ledger rows and un-ships them (docs/I18N.md §6;
#: measured here: adding one key to ``wire.remote_compute`` collapsed the
#: generated ``data/shipped.json`` to ``en``). So this is reported through
#: the generic ``remote compute failed:`` frame that every other field rule
#: in this class already uses, and the dedicated key belongs to the next
#: translation wave.
_BLANK_OR_LONG_NAME = (
    "a run name must be non-blank, at most "
    f"{MAX_RUN_NAME_BYTES} bytes of UTF-8, and one line with no control or "
    "bidirectional formatting characters inside it; "
    "leave --name out and one is taken from the prompt"
)


def _server_trim(text: str) -> str:
    """``text`` with the whitespace the SERVER trims removed from both ends.

    THE MEASUREMENT HAS TO BE THE SERVER'S, or a name is accepted here and
    refused there as a 400, after an idempotency key was minted. ``str.strip``
    is wider than Go's ``strings.TrimSpace``: ``"\x1c" + "a"*100`` strips to
    100 bytes client-side and reaches the far end as 101.
    """
    start, end = 0, len(text)
    while start < end and text[start] in _SERVER_WHITESPACE:
        start += 1
    while end > start and text[end - 1] in _SERVER_WHITESPACE:
        end -= 1
    return text[start:end]


def _has_standing_character(text: str) -> bool:
    """Whether ``text`` has a character that paints on its own.

    WHITESPACE DOES NOT COUNT, or the single space the collapse leaves between
    two marks would read as printable content and the fallback would never fire.
    """
    return any(
        unicodedata.category(character) not in _NON_SPACING_CATEGORIES and not character.isspace()
        for character in text
    )


def _printable_run_name(text: str) -> str:
    """``text`` with control and non-printable characters taken out of it.

    A CONTROL CHARACTER BECOMES A SPACE, so ``"a\tb"`` stays two words instead
    of being welded into ``"ab"``. FORMAT characters (zero-width joiners, byte-
    order marks) are DROPPED instead: they are invisible on the far end's
    surfaces, and turning the joiner in an emoji sequence into a space would
    split one glyph into two.
    """
    pieces: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category == "Cc":
            pieces.append(" ")
        elif category.startswith("C"):
            continue
        else:
            pieces.append(character)
    return "".join(pieces)


def _shorten_run_name(text: str) -> str:
    """``text`` cut on a CHARACTER boundary so its UTF-8 form fits the bound.

    Measured in bytes because that is what the server counts: a 40-character
    line of CJK is already over 100 bytes, and ``text[:100]`` would both overrun
    the bound and could not cut a character in half anyway.
    """
    if len(text.encode("utf-8")) <= MAX_RUN_NAME_BYTES:
        return text
    budget = MAX_RUN_NAME_BYTES - len(RUN_NAME_TRUNCATION_MARK.encode("utf-8"))
    kept: list[str] = []
    used = 0
    for character in text:
        width = len(character.encode("utf-8"))
        if used + width > budget:
            break
        kept.append(character)
        used += width
    # The mark goes on AFTER the trailing space is dropped, so the result never
    # ends in "… " and is still at most ``MAX_RUN_NAME_BYTES``.
    return "".join(kept).rstrip() + RUN_NAME_TRUNCATION_MARK


def short_repo_name(repo_url: str) -> str:
    """The repository's own name, from a URL the far end can clone.

    One rule covers all three spellings :func:`is_repo_url` accepts
    (``https://``, ``ssh://`` and the scp-like ``git@host:path``): drop a
    trailing ``.git`` and take the last path segment, which is the
    colon-separated one only in the scp-like form.
    """
    trimmed = repo_url.strip().rstrip("/")
    if trimmed.endswith(".git"):
        trimmed = trimmed[: -len(".git")]
    return trimmed.rsplit("/", 1)[-1].rsplit(":", 1)[-1].strip() or "repo"


def default_run_name(prompt: str, *, repo_url: str, repo_sha: str) -> str:
    """The name a run gets when ``--name`` was ABSENT.

    A name that was supplied but blank never reaches here — :func:`resolve_run_name`
    refuses it — so the only way to this fallback is leaving the flag out.

    THE PROMPT'S FIRST LINE, because that is the part of a task an operator
    recognises in a list of runs: the first line is the request, what follows is
    its detail. Whitespace is collapsed and control characters are removed so
    the label is one legible line, and the result is shortened on a character
    boundary so the bytes the server counts still fit.

    THE FALLBACK NAMES THE CHECKOUT, not the operator: a first line with nothing
    printable in it — a lone NUL, or nothing but combining marks that would
    attach to no base character — still needs a label, and
    ``lop exec <repo>@<sha>`` is the one thing every run of this command has.
    """
    lines = prompt.splitlines()
    first_line = lines[0] if lines else ""
    collapsed = " ".join(_printable_run_name(first_line).split())
    if collapsed and _has_standing_character(collapsed):
        return _shorten_run_name(collapsed)
    return _shorten_run_name(f"lop exec {short_repo_name(repo_url)}@{repo_sha[:7]}")


def resolve_run_name(
    name: str | None,
    *,
    prompt: str,
    repo_url: str,
    repo_sha: str,
) -> str:
    """The one place a run's ``name`` is decided, for every ``RunSpec`` caller.

    AN EXPLICIT NAME IS SENT UNCHANGED. The server trims it, and a client that
    quietly trimmed or shortened it would be carrying a label the operator did
    not ask for. What it may not be is a name the server refuses, which arrives
    as a 400 after an idempotency key was minted and the work addressed to a
    host: empty once trimmed, or over ``MAX_RUN_NAME_BYTES`` of UTF-8. Both are
    refused here instead. So is a name the server WOULD accept but that is not
    a usable one-line label — one carrying, inside it, a control character, a
    line or paragraph separator, or a bidirectional formatting character (which
    can make one run's label render as another's). All share ONE sentence at
    the surface because they share one remedy — leave ``--name`` out.

    A DERIVED name never reaches those checks and does not need to:
    ``default_run_name`` takes the first of ``splitlines()`` (which already cuts
    on Zl/Zp) and drops every ``C*`` character, which covers every refused
    class. ``test_a_derived_name_never_carries_a_refused_character`` pins it.

    AN ABSENT NAME IS DERIVED, A BLANK ONE IS REFUSED. Those are different
    requests: leaving the flag out says "name it for me", while
    ``--name "$VAR"`` with an empty ``VAR`` says "name it this" and got a
    substitute. The refusal is how the operator learns the variable was empty.
    """
    if name is None:
        return default_run_name(prompt, repo_url=repo_url, repo_sha=repo_sha)
    if not isinstance(name, str):
        raise ValueError(_BLANK_OR_LONG_NAME)
    trimmed = _server_trim(name)
    if not trimmed or len(trimmed.encode("utf-8")) > MAX_RUN_NAME_BYTES:
        raise ValueError(_BLANK_OR_LONG_NAME)
    # A RUN NAME IS A LABEL ON A LIST, so an interior control character is not a
    # formatting choice: it breaks the row, or hides the rest of the label
    # wherever the name is shown. Leading and trailing WHITESPACE control
    # characters (\t, \r, \n) are fine because the server trims them; what is
    # left after ``_server_trim`` is interior, and refused. The same holds for
    # the Unicode line and paragraph separators (Zl, Zp), which break the row
    # just as \n does, and for the bidirectional embedding/override/isolate
    # controls, which can make one run's label render as another's.
    if any(
        unicodedata.category(character) in _REFUSED_NAME_CATEGORIES or character in _BIDI_CONTROLS
        for character in trimmed
    ):
        raise ValueError(_BLANK_OR_LONG_NAME)
    return name


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """The tier-1 run body: what a short-lived remote run is made of.

    WHY A SECOND MODEL. ``RunCreateRequest`` holds the tenant invariant and
    nothing else; this holds the fields the public contract names, with their
    shapes checked ONCE, here, before a binding exists. ``bind`` is the only
    way from here into a request, so a caller cannot assemble a body whose
    ``network_id`` came from anywhere but ``resolve_network``.

    ``hosting`` AND ``model`` ARE REQUIRED AND DECLARED. A remote run does not
    inherit this device's provider keys, so the far end must be told which
    model to run; the CLI resolves them through the same ladder a local exec
    uses (agent > flag > team > config) and sends what the operator resolved.

    ``budget_usd`` is optional because the server applies its own admission
    default when absent. ``name`` is optional HERE AND REQUIRED THERE: the
    server trims the label and refuses an empty one or one over
    :data:`MAX_RUN_NAME_BYTES` of UTF-8, so a spec never leaves this class
    without one — an explicit name is carried exactly as given, and an absent
    one is derived from the prompt by :func:`resolve_run_name`. A name that was
    SUPPLIED but blank is refused, never replaced by a derived one.
    """

    prompt: str
    repo_url: str
    repo_sha: str
    hosting: str
    model: str
    size: str = "m"
    max_hours: float = 2.0
    budget_usd: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        # REFUSED HERE, ONCE: every one of these would otherwise travel to the
        # far end and come back as a 400 the operator cannot act on, after the
        # request had already been given an idempotency key.
        for field_name in ("prompt", "repo_url", "repo_sha", "hosting", "model"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"a remote run needs a non-empty {field_name}")
        if self.size not in RUN_SIZES:
            raise ValueError(
                f"unknown remote run size {self.size!r}; the sizes are {', '.join(RUN_SIZES)}"
            )
        # A BOOL IS AN INT to ``isinstance`` and to ``math.isfinite``, so
        # ``max_hours=True`` used to pass as one hour and ``budget_usd=False`` as
        # a zero budget. Both are type errors in the body, not choices.
        if isinstance(self.max_hours, bool) or not isinstance(self.max_hours, (int, float)):
            raise ValueError(f"max_hours must be a positive number, got {self.max_hours!r}")
        if not math.isfinite(self.max_hours) or self.max_hours <= 0:
            raise ValueError(f"max_hours must be a positive number, got {self.max_hours!r}")
        if self.budget_usd is not None and (
            isinstance(self.budget_usd, bool)
            or not isinstance(self.budget_usd, (int, float))
            or not math.isfinite(self.budget_usd)
            or self.budget_usd < 0
        ):
            raise ValueError(f"budget_usd must be zero or more, got {self.budget_usd!r}")
        # THE LAST THING CHECKED, because the default is derived FROM the prompt
        # and names the checkout. ``frozen`` means the resolved value is set
        # through ``object.__setattr__``; every caller — the CLI, a test, a
        # later slice — therefore composes the same body without having to know
        # the rule.
        object.__setattr__(
            self,
            "name",
            resolve_run_name(
                self.name,
                prompt=self.prompt,
                repo_url=self.repo_url,
                repo_sha=self.repo_sha,
            ),
        )

    def to_body(self) -> dict[str, Any]:
        """The wire fields, with the one optional field OMITTED when unset.

        Omitted rather than sent as ``null``: an explicit null for ``budget_usd``
        reads as a choice (no budget) where absence reads as "use the server's
        admission default", and only one of those is what the caller meant.

        ``name`` has no guard because it cannot be unset by the time a body
        exists: ``__post_init__`` resolves it to a non-empty string, so the
        conditional that used to sit here was unreachable.
        """
        body: dict[str, Any] = {
            "repo_url": self.repo_url,
            "repo_sha": self.repo_sha,
            "prompt": self.prompt,
            "hosting": self.hosting,
            "model": self.model,
            "size": self.size,
            "max_hours": self.max_hours,
        }
        body["name"] = self.name
        if self.budget_usd is not None:
            body["budget_usd"] = self.budget_usd
        return body

    def bind(self, binding: NetworkBinding) -> RunCreateRequest:
        """Pair this body with a resolved network; the only crossing into a request."""
        return RunCreateRequest(binding, self.to_body())
