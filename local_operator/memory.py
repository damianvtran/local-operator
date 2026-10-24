"""Bounded local retain/recall, with a transport-neutral causal operation log.

SQLite/FTS5 is deliberately the entire runtime dependency. Hindsight's structured
retain/recall lifecycle inspired provenance, freshness and bounded candidate pools
(https://github.com/vectorize-io/hindsight), not its inference/database stack or
benchmark claims. Scopes are visibility filters, not a sandbox against shell access.

Validate *before* opening a store: a rejected payload must not create a database,
FTS entry, history row or diagnostic containing the value. The heuristic shape/ID
checks cannot recognise every opaque free-text secret; never ingest transcripts or
environments wholesale. Secret names, rather than values, belong in memory.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import sqlite3
import tempfile
import time
import unicodedata
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlsplit

from local_operator.compaction.tokens import count_text_tokens
from local_operator.redaction_shapes import match_shape_names
from local_operator.supplements.denylist import is_sensitive

SESSION_EXPORT_NAME = "memory-session.jsonl"
SCOPES = ("global", "session", "team", "agent")
DEFAULT_EXCLUDED_DEVICES: tuple[str, ...] = ()
# One protocol definition shared by publication, execution and pre-journal retention.
# Valid JSON is insufficient: a rejected scalar/list/unknown key can itself contain
# an opaque credential that no named text-field guard ever inspects (R6 residual).
MEMORY_TOOL_OPS = ("store", "recall")
MEMORY_TOOL_REQUIRED = ("op", "text")
MEMORY_TOOL_CONTRACT_ERROR = (
    "Invalid memory contract. Use op=store|recall and text only for durable personal/global "
    "memory; scoped work uses read/write/edit memory://. No scope argument is accepted."
)


def valid_memory_tool_arguments(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(MEMORY_TOOL_REQUIRED).issubset(value)
        and not set(value) - {*MEMORY_TOOL_REQUIRED, "i"}
        and isinstance(value["op"], str)
        and value["op"] in MEMORY_TOOL_OPS
        and isinstance(value["text"], str)
        and ("i" not in value or isinstance(value["i"], str))
    )


ITEM_TOKENS = 256
ITEM_BYTES = 8192
#: Token caps per scope. One authority: the write gate (``_quota``), the move
#: destination check, the warning state and inventory/status all read these, so
#: a reported headroom cannot disagree with the number a write is refused on.
#: Global is the shared bank everyone reads; every other scope is per-owner.
GLOBAL_TOKENS = 16000
OWNER_TOKENS = 50000
#: Percent thresholds that warn ONCE on the write that crosses them, highest
#: first: a write jumping straight past both reports the endangering one.
WARNING_THRESHOLDS = (90, 80)
RECALL_TOKENS = 1200
RECALL_HARD_TOKENS = 2000
#: Hard cap on the main database file, enforced as ``max_page_count`` in
#: :meth:`MemoryStore.connection`. Reaching it makes SQLite refuse a write with
#: SQLITE_FULL instead of growing the file. This is a SEPARATE budget from the
#: token quotas above: a store can hold token headroom and still be out of bytes,
#: which is why `lop memory status` reports both and why the refusal names the
#: byte cap. Raising this (or auto-compacting) is a product decision, not a
#: repair for a full store.
MAX_DB_BYTES = 64 * 1024 * 1024
#: Page size the cap is expressed against in :meth:`MemoryStore.connection`.
#: ``PRAGMA max_page_count`` takes pages, so the enforced byte cap is
#: ``(MAX_DB_BYTES // 4096) * page_size`` — for a default 4096-byte store that
#: is just MAX_DB_BYTES floored to a page. Status and the refusal report this
#: enforced number, never the unfloored constant, so 100% cannot sit beside a
#: cap SQLite does not apply.
CAP_PAGE_SIZE = 4096
#: Percent of the byte cap at which `lop memory status` says the store is near
#: full. The token thresholds warn ONCE on the write that crosses them; this one
#: is a status-surface warning, because a store at the byte cap refuses the next
#: write outright — there is no later write to carry the warning.
STORAGE_WARNING_PERCENT = 90
MAX_OWNER_OPS = 20000
MAX_REVISIONS = 128
#: One confirmed filter-based forget accepts this many facts. The human preview
#: must refuse ABOVE it rather than print a confirmation the store will reject:
#: a command that cannot run is worse than no command. Reachable within the caps
#: (measured in review round 1: 205 minimal facts fit the global 16,000-token
#: bank), so the preview checks it before offering anything.
MAX_BULK_FORGET = 200
MAX_TOMBSTONES = 5000
MAX_BATCH_BYTES = 256 * 1024
INVENTORY_HELP = (
    "Inventory is not the bank. Search ?q=words, read an ID, consolidate explicitly; "
    "retrieval does not refresh verification. Read guide://memory for structured "
    "write/edit, scopes and forgetting."
)
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_NAME = re.compile(r"[A-Z][A-Z0-9_]{2,63}")
# Contextual ID checks complement the shared credential shape table, which does
# not claim to recognise identity documents. Names such as secret PASSPORT_NO
# remain useful and are not values; numeric assignments/documents are refused.
#
# Every quantifier here is BOUNDED. guard() runs on up to 8 KiB per field on the
# agent's tool path, and an unbounded `[A-Z0-9 -]+` after a repeated keyword is
# the quadratic shape that froze the capture scanner (see memory_capture._LOCATOR).
_SSN = re.compile(r"(?<![\w-])\d{3}[ \t-]{1,3}\d{2}[ \t-]{1,3}\d{4}(?![\w-])")
# Card-like digit runs are only refused when they ALSO pass Luhn and start like an
# issuer range: bare 13-digit unix-ms timestamps and order numbers are ordinary
# data (QA Q2), and refusing them made them impossible to store or even search.
_CARDISH = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_ANCHORED = re.compile(
    r"\b(?:passport|ssn|social.security|driver.?s?.?licen[cs]e|licen[cs]e|national.id"
    r"|card.number|tax.?(?:id|number|no)|tin|ein|itin|nino|national.insurance|routing|sort.code)"
    r"(?:\s{0,3}(?:no\.?|number|num|\#)){0,1}(?:\s{1,3}(?:is|was|are)){0,1}"
    r"\s{0,3}[:=\#]{0,1}\s{0,3}"
    r"(?:[A-Z0-9]{0,24}\d[A-Z0-9-]{0,24}|[A-Z]{1,3}[ -]\d[A-Z0-9-]{0,24})"
    r"(?:[ -][A-Z0-9]{1,24}){0,3}",
    re.I,
)
# A bare "account" is ordinary prose; only an explicit number marker makes it a value.
_ACCOUNT = re.compile(
    r"\b(?:(?:bank.)?account|acct)\s{0,3}(?:no\.?|number|num|\#)\s{0,3}[:=]{0,1}\s{0,3}"
    r"[A-Z0-9][A-Z0-9 -]{4,24}",
    re.I,
)
_IBAN = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}(?![A-Z0-9])", re.I)
# `password is X` prose. The verb list and the mixed-token requirement below keep
# "password policy: rotate quarterly" storable while "password is hunter2" is not.
_PROSE_SECRET = re.compile(
    r"\b(?:pass(?:word|phrase|code|wd)|pwd|login|credentials?)\b"
    r"(?:\s{1,3}for\s{1,3}[\w.-]{1,32}){0,1}\s{0,3}"
    r"(?:\bis\b|\bare\b|\bwas\b|=|:|->)\s{0,3}([^\n]{1,80})",
    re.I,
)
_PASSPHRASE_WORDS = re.compile(
    r"\bpassphrase\b\s{1,3}(?:(?:is|are|=|:)\s{1,3}){0,1}"
    r"(?!(?:policy|rules?|rotation|requirements?|length|manager|is|are|was|for|in|on|and|or)\b)"
    r"[a-z]{3,16}(?:\s{1,3}[a-z]{3,16}){3,}"
)
_HEX_RUN = re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{32,}(?![0-9A-Za-z])")
_B64_RUN = re.compile(r"(?<![\w+/=-])[A-Za-z0-9+/_-]{40,}={0,2}(?![\w+/=-])")
# A long hex digest is legitimate when it is NAMED as one (commit/sha/checksum);
# a bare one is indistinguishable from an API key.
_DIGEST_CONTEXT = re.compile(
    r"(?:commit|sha(?:1|256|512)?|rev(?:ision)?|merge|cherry-pick|checksum|digest|hash|tag)"
    r"\W{0,4}\Z",
    re.I,
)
_SYMBOLS = set("!@$%^&*")
# Cyrillic/Greek look-alikes of Latin letters, so a keyword like "pаssport" (Cyrillic
# a) still matches. Matching copy only; the stored text is never rewritten.
_CONFUSABLES = {
    ord(a): b
    for a, b in zip(
        "аеорсхуіјѕАВЕКМНОРСТХΑΒΕΖΗΙΚΜΝΟΡΤΥΧοαρ", "aeopcxyijsABEKMHOPCTXABEZHIKMNOPTYXoap"
    )
}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _fold(text: str) -> str:
    """Matching copy of ``text``: NFKC, invisible/format chars removed, digits ASCII.

    ``SSN 1<ZWSP>23-45-6789`` stored raw in the DB, session journal and transcript
    before this existed (QA Q1): the shape tables are ASCII-literal. The folded
    copy is only ever compared, never stored or echoed.
    """
    if text.isascii() and not _CONTROL.search(text):
        return text
    out: list[str] = []
    for ch in unicodedata.normalize("NFKC", text).translate(_CONFUSABLES):
        category = unicodedata.category(ch)
        if category in ("Cf", "Mn", "Me", "Co", "Cs") or (category == "Cc" and ch not in "\n\t\r"):
            continue
        if category == "Nd":
            ch = str(unicodedata.decimal(ch))
        elif category == "Pd" or ch == "\u2212":
            ch = "-"
        elif category in ("Zs", "Zl", "Zp"):
            ch = " "
        out.append(ch)
    return "".join(out)


# Work/lane/progress locators and status verbs. GLOBAL memory is for durable user facts;
# MR/CI/release status and lane next-steps belong in a session bank. These patterns are
# deliberately narrow: they require WORK CONTEXT next to the locator, and are only ever
# matched against the FOLDED copy (never stored or echoed).
#
# QA round 1 (M-1): a bare `#N` is how people write chapters, order/invoice/room/seat
# and extension numbers (`Office #4`, `order #1123`), and `next step`/`HOLD` occur in
# ordinary prose (`next step is to fold in the flour`, `a HOLD on the cheque`). A locator
# alone is therefore NOT a marker: it must sit next to a work word.
#
# Code review rounds 1/3 (M-1/F-3): `mrs?` under re.I matched the honorific Mr/Mrs
# (`Mr Okafor lives at #12`, `mr smith lives at #12`). The merge-request abbreviation
# is therefore UPPERCASE-ONLY here (`MR`, `MRs`); the lowercase form is never a tracker
# word. GitLab work uses `!N`, which is matched absolutely.
#
# KNOWN BOUNDARY (documented, not debt): this is a bounded SHAPE rule, so prose work
# text with no locator (`Fleet scratchpad purge 2026-10-15/16: …`) and hyphenated keys
# (`LOH-13`, `PR34`) still pass. Catching those needs a classifier, which the design
# deliberately rejects; the guide states this bound rather than overclaiming.
_WORK_WORD = (
    r"pr|pull request|merge request|(?-i:MRs?)|issue|ticket|ci|pipeline|build|commit|"
    r"sha|branch|rebase|squash|deploy|release|rollout|lane|backlog|sprint|milestone|"
    r"tracker|jira|linear|holding|blocked|pending|awaiting|drivable|wip"
)
_WORK_REF = re.compile(
    r"\b(?:" + _WORK_WORD + r")\b[^.\n]{0,24}(?<![\w#])#\d{1,6}\b"
    r"|(?<![\w#])#\d{1,6}\b[^.\n]{0,24}\b(?:" + _WORK_WORD + r")\b"
    r"|(?<![\w!])!\d{1,6}\b",
    re.I,
)
# A named commit needs a LETTER in the hash: `commit 5000000` is ordinary prose, an
# all-letter abbreviation (`cafebabe`) is not.
_WORK_NOUN = r"pr|(?-i:MRs?)|release|merge|ci|branch|tag|deploy|lane|ticket|issue"
_WORK_SHA = re.compile(r"\b(?:commit|sha|head)\s+(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b", re.I)
_WORK_STATUS = re.compile(
    r"\b(?:ci|pipeline|build|checks)\b[^.\n]{0,24}\b(?:green|red|pass(?:ed|ing)?|fail(?:ed|ing)?)"
    r"\b"
    r"|\b(?:merged|merge[sd]?|rebase[sd]?|squash(?:ed)?)\b[^.\n]{0,20}"
    r"\b(?:pr|(?-i:MRs?)|#\d|into main)\b"
    # Noun-then-verb needs TIGHT adjacency, or only a copula/auxiliary between:
    # `PR`/`MR` are also permanent residency, public relations and a person
    # (`My PR application merged into the family one`, `The PR agency merged with a rival`),
    # while `the PR was merged` / `MRs rebased onto main` are real work text.
    r"|\b(?:pr|(?-i:MRs?))\b(?:\s+(?:was|is|are|were|been|has|had|gets?|got|now))?\s{0,4}"
    r"(?:merged|merge[sd]?|rebase[sd]?|squash(?:ed)?)\b"
    r"|\b(?:cut|land|ship|tag)\b[^.\n]{0,20}\b(?:(?:the )?release|v\d)"
    r"|\bnext (?:step|lane)\b[^.\n]{0,40}\b(?:" + _WORK_NOUN + r")\b"
    r"|\b(?:the )?(?:" + _WORK_NOUN + r")\b[^.\n]{0,24}\bnext (?:step|lane)\b"
    r"|\bremaining work\b",
    re.I,
)
# A work-context HOLD: the token must concern a PR/MR/release/lane/merge/CI, not a
# bank/parcel/dry-cleaner hold an ordinary person records.
_WORK_HOLD = re.compile(
    r"(?<![\w-])HOLD(?![\w-])[^.\n]{0,30}\b(?:" + _WORK_NOUN + r")\b"
    r"|\b(?:" + _WORK_NOUN + r")\b[^.\n]{0,30}(?<![\w-])HOLD(?![\w-])",
    re.I,
)


def work_progress_markers(text: str) -> list[str]:
    """Names of work/lane/progress markers in ``text`` — value-free, never the match."""
    folded = _fold(text)
    found: list[str] = []
    for name, pattern in (
        ("tracker-ref", _WORK_REF),
        ("named-commit", _WORK_SHA),
        ("ci-release-status", _WORK_STATUS),
        ("hold-token", _WORK_HOLD),
    ):
        if pattern.search(folded):
            found.append(name)
    return found


def scope_fit(scope: str, fact: dict[str, Any]) -> None:
    """Refuse work/lane/progress content in the GLOBAL bank, before any write.

    A bounded SHAPE rule, not a classifier: it catches a tracker ref, a named commit,
    a CI/release status verb or a `next step` phrase. Global is for durable personal
    facts, contacts, preferences, environment; work state belongs in a session bank.
    Value-free refusal (no offending text echoed); explicit session/team/agent writes
    and every wire/import path are untouched, and forget is never reached here.
    """
    if scope != "global" or not isinstance(fact, dict):
        return
    haystack = "\n".join(str(fact.get(key) or "") for key in ("title", "body", "subject"))
    for key in ("tags", "aliases"):
        value = fact.get(key) or []
        if isinstance(value, list):
            haystack += "\n" + "\n".join(str(item) for item in value)
    if work_progress_markers(haystack):
        raise MemoryError(
            "This looks like session-scoped work (a tracker reference, CI/release "
            "status or a next step). Store work state in memory://session; global is "
            "for durable personal facts, contacts, preferences and environment."
        )


def _card_issuer_range(digits: str) -> bool:
    """Known payment-card prefix AND length. Random 13-digit ids (unix-ms, order numbers)
    pass Luhn about one time in ten, so Luhn alone refused ordinary data (QA Q2)."""
    n = len(digits)
    two = int(digits[:2])
    return (
        (digits[0] == "4" and n in (13, 16, 19))
        or (n == 16 and (51 <= two <= 55 or 2221 <= int(digits[:4]) <= 2720))
        or (n == 15 and two in (34, 37))
        or (16 <= n <= 19 and (digits[:4] == "6011" or two in (65, 35, 62) or digits[0] == "6"))
        or (14 <= n <= 19 and (two in (36, 38, 39) or 300 <= int(digits[:3]) <= 305))
        or (12 <= n <= 19 and two in (50, 56, 57, 58))
    )


def _luhn(digits: str) -> bool:
    total = 0
    for index, ch in enumerate(reversed(digits)):
        n = int(ch)
        if index % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _mixed_token(chunk: str) -> bool:
    """A password-looking token: letters AND digits/symbols, not a path, URL or NAME."""
    for raw in chunk.split():
        token = raw.strip("\"'`.,;)(")
        # SHOUTY_SNAKE is the NAME-reference form we ask agents to use; it is never a value.
        if (
            len(token) < 4
            or "/" in token
            or token.startswith(("http", "~"))
            or _NAME.fullmatch(token)
        ):
            continue
        if any(c.isalpha() for c in token) and any(c.isdigit() or c in _SYMBOLS for c in token):
            return True
    return False


def _looks_sensitive(text: str, *, opaque: bool) -> bool:
    """Heuristic pattern guard over the FOLDED text. Not proof of absence.

    ``opaque`` ENABLES the long-hex / base64 rule. It is on only for free-text fact fields
    (title/body/tags/aliases/subject): structural fields (op ids, body hashes, entity
    ids) are legitimately 64-hex, and a search for a commit SHA is ordinary, while storing
    an unnamed 40-hex string is how a leaked API key would be kept.
    """
    if _SSN.search(text):
        return True
    for match in _CARDISH.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _card_issuer_range(digits) and _luhn(digits):
            return True
    for match in _ANCHORED.finditer(text):
        if sum(c.isdigit() for c in match.group()) >= 5:
            return True
    for match in _ACCOUNT.finditer(text):
        if sum(c.isdigit() for c in match.group()) >= 5:
            return True
    for match in _IBAN.finditer(text):
        compact = match.group().replace(" ", "")
        if 15 <= len(compact) <= 34:
            rearranged = compact[4:] + compact[:4]
            number = "".join(str(int(c, 36)) for c in rearranged.upper())
            if int(number) % 97 == 1:
                return True
    for match in _PROSE_SECRET.finditer(text):
        if _mixed_token(match.group(1)):
            return True
    if _PASSPHRASE_WORDS.search(text):
        return True
    if opaque:
        for match in _HEX_RUN.finditer(text):
            if not _DIGEST_CONTEXT.search(text[max(0, match.start() - 16) : match.start()]):
                return True
        for match in _B64_RUN.finditer(text):
            token = match.group()
            if (
                any(c.isdigit() for c in token)
                and any(c.isupper() for c in token)
                and any(c.islower() for c in token)
                and token.count("/") <= 3
                and sum(token.count(c) for c in "-_") <= 3
            ):
                return True
    return False


#: Words that carry no discriminating evidence. Public because the read-only
#: duplicate preview in ``memory_duplicates`` must weigh the SAME vocabulary the
#: search ranker drops; a second private copy would drift and score differently.
STOP_WORDS = frozenset(
    (
        "a an and are as at be by can could did do does for from have how i in is it me my of on "
        "or our please should that the their this to use was we what when where which who will "
        "with would you your create new show find tell get save remember memory work help want "
        "need about"
    ).split()
)


class MemoryError(ValueError):
    """Safe, value-free actionable refusal suitable for user/tool surfaces."""


@functools.lru_cache(maxsize=1)
def fts5_available() -> bool:
    """Whether this interpreter's SQLite can build the porter FTS5 index we require."""
    probe = sqlite3.connect(":memory:")
    try:
        probe.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='porter unicode61')")
        return True
    except sqlite3.Error:
        return False
    finally:
        probe.close()


def safe_segment(value: Any, *, structural: bool = False) -> str:
    if not isinstance(value, str) or not _SAFE.fullmatch(value) or value in (".", ".."):
        raise MemoryError("Memory identity must be one safe segment (1–128 characters).")
    # Structural identifiers are minted, not user prose: device ids are
    # ``d_`` + hex and op/entity ids are UUID/hex, so a free-text sensitive
    # shape (an IBAN/mod-97 or card/Luhn match on the hex run) can occur by
    # chance in a legitimate id. Running guard() over them therefore refused
    # whole installs permanently — the id persists in store meta — while
    # adding no protection: the _SAFE shape above is the actual constraint
    # on an identifier. Callers validating fact *content* keep guard().
    if not structural:
        guard(value)
    return value


#: The shapes a value may legitimately BE as a structural REFERENCE, each one cited at
#: the source that mints it. A reference is not prose, so the free-text guard is the
#: wrong question to ask of one: its hex runs can satisfy the IBAN/mod-97 or card/Luhn
#: shape BY CHANCE — that collision refused every put on an install whose device id
#: happened to pass mod-97 (the write path fixed it through ``safe_segment``), and its
#: read-side twin refused every recall whose QUERY was such an id.
_REFERENCE_SHAPES = (
    # network.identity.device_id_for: "d_" + sha256(DEVICE_ID_DOMAIN + key).hexdigest()[:32].
    re.compile(r"d_[0-9a-f]{32}"),
    # memory._id(): the canonical UUID string minted for entity and operation ids —
    # the SCOPE/ID the CLI prints and the version=REV a read accepts. The store mints
    # UUIDv7, but this guard is shape-based and pins no version/variant nibble, so any
    # canonical UUID spelling matches.
    re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"),
)


def structural_reference(value: Any) -> bool:
    """True iff ``value`` IS exactly one minted structural reference, nothing else.

    A recall query is a READ and may legitimately BE an identifier this codebase mints
    and hands back (a device id, or an entity/operation id) rather than user prose, so
    ``guard()``'s store-oriented refusal — keep only the NAME — is the wrong verdict for
    it. Callers skip the free-text guard for a value this admits, and for that value
    alone: the match is a FULL match, so a secret-shaped token inside prose, a labelled
    value and any multi-token string are not references and keep the guard.

    The admitted set is exactly the two shapes above, each justified by the function
    that mints it. A shape with no such source is not a reference here: a bare 64-hex
    ``body_sha`` digest is spelled ``[0-9a-f]{64}`` at its validation, but it is an
    integrity value over the operation payload the store compares, never a *lookup*
    handle — it does ride op records (``memory_capture.py`` computes one per captured
    op) and no code path searches by it, since ``search()`` takes no body-hash input —
    and the free-text guard already admits it. This is a READ-side classification only;
    a WRITE keeps the guard, because storing a value is not the same act as looking one
    up.
    """
    return isinstance(value, str) and any(shape.fullmatch(value) for shape in _REFERENCE_SHAPES)


def guard(value: Any, *, opaque: bool = False) -> None:
    """Inspect every string, including metadata and wire-only fields, without echo.

    This is a PATTERN guard, not proof: a secret with no recognisable shape or label
    can still be stored (and journaled). Lop secrets hold values; memory holds NAMEs.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            guard(key, opaque=opaque)
            guard(child, opaque=opaque)
    elif isinstance(value, (list, tuple)):
        for child in value:
            guard(child, opaque=opaque)
    elif isinstance(value, str):
        if len(value.encode("utf-8")) > ITEM_BYTES:
            raise MemoryError("Memory field exceeds the 8192-byte cap; split into atomic facts.")
        folded = _fold(value)
        if (
            match_shape_names(value)
            or (folded != value and match_shape_names(folded))
            or _looks_sensitive(folded, opaque=opaque)
        ):
            raise MemoryError(
                "Sensitive value refused. Store it with lop secret and keep only its NAME here "
                "(e.g. 'uses secret GITHUB_TOKEN'); never the value."
            )
        # Check path-like tokens, not the complete prose (which can contain a
        # harmless reference beside a sensitive path). Never persist the latter.
        # A drive letter starts a token, not the suffix of a URL scheme: matching
        # `s://example...` in https:// made benign URLs resolve relative to the
        # session cwd and fail whenever that cwd was under a protected root.
        # The boundary belongs ONLY to the drive-letter arm. Unix separators
        # inside relative locators must still expose sensitive components such
        # as docs/.env or deploy/kubeconfig-prod (R7).
        for word in re.findall(r"(?:~?/|(?<![\w])[A-Za-z]:[/\\])[^\s<>\"']{1,512}", folded):
            if is_sensitive(word):
                raise MemoryError(
                    "Sensitive path refused; retain a safe locator or secret NAME instead."
                )


def origin_label(fact: dict[str, Any]) -> str:
    """``captured`` = automatic compaction capture still awaiting a human or an explicit
    store; ``stated`` = anything a user or agent deliberately wrote or corrected.
    Shown everywhere a fact is listed so unreviewed candidates are never mistaken for
    verified knowledge (UX round 1: U1)."""
    return "captured" if fact.get("status") == "candidate" else "stated"


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def format_bytes(value: int) -> str:
    """Human-readable binary size for the byte cap and its usage.

    Binary units because the cap is a power of two; one decimal so the round
    64.0 MiB cap reads as the same number in `status`, the refusal and the
    documentation. Byte-sized values stay integers so a nearly-empty store does
    not claim to hold "0.0 KiB".
    """
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{int(size)}B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


#: Total wire budget for the suggestion tail: the WHOLE block, wrapper
#: included. One number, shared by the renderer and the admission rule in
#: :meth:`MemoryStore.search` — when the two disagreed the documented
#: ``limit=3`` silently rendered 2.
SUGGESTION_TOKENS = 180

#: The suggestion block's fixed wrapper. ``memory_capture._session_tail``
#: renders these around the admitted lines; both it and the admission budget
#: charge them, because "180 tokens" is the block on the wire, not its lines.
#: Measured 19 tokens (14 header + 5 closing): charging only the lines let three
#: 156-char titles be admitted at 166 and then dropped by the tail's exact
#: check, so a documented ``limit=3`` rendered 2 — the same defect class the
#: per-line ruler fixed, confined to long titles.
SUGGESTION_BLOCK_HEADER = (
    "<memory_suggestions>\nUntrusted recall candidates; read before relying.\n"
)
SUGGESTION_BLOCK_CLOSING = "</memory_suggestions>"


def suggestion_block(lines: str) -> str:
    """The exact text ``lines`` renders as: header, admitted lines, closing.

    ONE authority for the wrapper, for the same reason :func:`suggest_line` is
    one for the lines: the tail builds the block with this and the admission
    check in :meth:`MemoryStore.search` measures the block this returns, so the
    two cannot drift apart on what a candidate costs.
    """
    return SUGGESTION_BLOCK_HEADER + lines + SUGGESTION_BLOCK_CLOSING


def suggest_line(item: Mapping[str, Any]) -> str:
    """The single rendered line a suggestion actually costs on the wire.

    ONE authority for this string, for two readers that must agree: the
    prompt tail (``memory_capture._session_tail``) renders it, and the
    suggestion budget in :meth:`MemoryStore.search` measures it. Before this
    existed the budget measured ``_dump(item)`` — the whole candidate record,
    ~87-90 tokens against the 32-35 the rendered line costs — so the
    documented ``tokens=180, limit=3`` admitted only TWO candidates and
    silently crowded out the third relevant fact in every measured fixture.
    """
    captured = " (captured)" if item["status"] == "candidate" else ""
    return f"- memory://{item['scope']}/{item['id']}: {json.dumps(item['title'])}{captured}\n"


def _suggestion_key(fact: Mapping[str, Any]) -> str:
    """Identity of a suggestion by its FACT, not its record.

    Distinct rows often carry one fact: capture writes the same body under
    slightly different titles ("Mercury deployment", "... corrected", "...
    again"). Collapsing on the normalized body makes them cost one slot; the
    title is only the fallback for a body-less row. Normalization is
    whitespace/case only — a fuzzy match would merge genuinely distinct facts,
    which is worse than showing two. Explicit ``search`` does NOT collapse
    (a reader may legitimately want every revision), so this key is only
    consulted on the suggestion path.
    """
    text = str(fact.get("body") or fact.get("title") or "")
    # ``casefold`` is the normalization that means case-insensitive: it folds
    # non-ASCII pairs ``lower`` misses (ß vs SS, final sigma). Identical for
    # ASCII, so this is the correctness of the collapse rule, not a behaviour
    # change for existing rows.
    return " ".join(text.casefold().split())


def _id() -> str:
    # UUIDv7 layout without depending on Python 3.14 (the supported floor is 3.12).
    bits = (int(time.time() * 1000) << 80) | (7 << 76) | (uuid.uuid4().int & ((1 << 76) - 1))
    bits = (bits & ~(3 << 62)) | (2 << 62)
    return str(uuid.UUID(int=bits))


@dataclass(frozen=True)
class MemoryIdentity:
    session_id: str = ""
    team_id: str = ""
    agent_id: str = ""

    def __post_init__(self) -> None:
        for value in (self.session_id, self.team_id, self.agent_id):
            if value:
                safe_segment(value)

    def owner(self, scope: str) -> str:
        if scope == "global":
            return "global"
        if scope not in SCOPES:
            raise MemoryError("Unknown scope; use session, team, agent or global.")
        owner = getattr(self, scope + "_id")
        if scope == "agent" and owner in ("", "main"):
            raise MemoryError(
                "Anonymous agent 'main' has no persistent agent scope. "
                "Use session or global memory."
            )
        if not owner:
            raise MemoryError(f"No authorized {scope} identity; use session or global memory.")
        return safe_segment(owner)

    def scopes(self) -> list[tuple[str, str]]:
        result = [("global", "global")]
        for scope in SCOPES[1:]:
            try:
                result.append((scope, self.owner(scope)))
            except MemoryError:
                continue
        return result


#: The status vocabulary a stored fact may carry. Exported so a READER
#: (``memory_inspect``) gates the same closed set the writer enforces instead of
#: restating it: an imported row can hold any string in this column, and the
#: reader echoes it to the screen. Two spellings of "valid status" is how one
#: surface ends up showing a value the other refuses.
FACT_STATUSES = ("active", "candidate", "disputed", "superseded", "expired")

#: The inclusive upper bound on any stored freshness timestamp (year 3000).
#: Named and exported for the same reason as ``FACT_STATUSES``: the read-only
#: inspector gates fields with the writer's rule, and a second literal in that
#: reader is how the two drift apart silently.
MAX_FACT_TIMESTAMP = 32503680000


def validate_fact(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise MemoryError("Memory content must be a JSON object; read guide://memory.")
    guard(raw)
    guard(_dump(raw))
    for key in ("title", "body", "subject", "tags", "aliases"):
        if key in raw:
            guard(raw[key], opaque=True)
    allowed = {
        "title",
        "body",
        "tags",
        "aliases",
        "source",
        "status",
        "verified_at",
        "expires_at",
        "pinned",
        "subject",
        "created_at",
        "source_time",
        "replaces",
    }
    if set(raw) - allowed:
        raise MemoryError("Unknown memory field; read guide://memory for the narrow protocol.")
    fact = {
        "title": "",
        "body": "",
        "tags": [],
        "aliases": [],
        "source": {},
        "status": "active",
        "verified_at": None,
        "expires_at": None,
        "pinned": False,
        "subject": "",
        "created_at": None,
        "source_time": None,
        "replaces": "",
        **raw,
    }
    for key in ("title", "body", "subject"):
        if not isinstance(fact[key], str):
            raise MemoryError("Memory title/body/subject must be text.")
    if (
        not fact["title"].strip()
        or not fact["body"].strip()
        or len(fact["title"]) > 160
        or len(fact["subject"]) > 128
    ):
        raise MemoryError(
            "Supply an atomic body and a title (1–160 characters); subject cap is 128."
        )
    for key in ("tags", "aliases"):
        if (
            not isinstance(fact[key], list)
            or len(fact[key]) > 12
            or any(not isinstance(x, str) or not x.strip() or len(x) > 64 for x in fact[key])
        ):
            raise MemoryError(
                "Tags/aliases must contain at most 12 short strings (64 characters each)."
            )
    if fact["status"] not in FACT_STATUSES or type(fact["pinned"]) is not bool:
        raise MemoryError("Invalid memory status or pinned flag.")
    for key in ("verified_at", "expires_at", "created_at", "source_time"):
        x = fact[key]
        if x is not None and (type(x) not in (int, float) or not 0 <= x <= MAX_FACT_TIMESTAMP):
            raise MemoryError("Freshness timestamps must be finite Unix seconds or null.")
    source = fact["source"]
    if (
        not isinstance(source, dict)
        or len(source) > 6
        or set(source) - {"session", "entry", "tool", "kind", "uri", "author"}
        or any(not isinstance(x, str) or len(x) > 256 for x in source.values())
    ):
        raise MemoryError(
            "Source allows session/entry/tool/kind/uri/author, each at most 256 characters."
        )
    if source.get("kind") not in ("reported", "observed", "inferred", "compaction", "summary"):
        raise MemoryError(
            "Source kind must be reported, observed, inferred, compaction or summary."
        )
    if source.get("session"):
        safe_segment(source["session"])
    if fact["replaces"]:
        safe_segment(fact["replaces"])
    wire = _dump(fact)
    if len(wire.encode()) > ITEM_BYTES or count_text_tokens(wire) > ITEM_TOKENS:
        raise MemoryError(
            "Memory item exceeds 256 tokens/8192 bytes including metadata. Split or consolidate it."
        )
    return fact


def validate_op(op: Any, *, allow_redacted: bool = False) -> dict[str, Any]:
    if not isinstance(op, dict) or len(_dump(op).encode()) > 16384:
        raise MemoryError("Invalid or oversized memory operation.")
    required = {
        "v",
        "op_id",
        "entity_id",
        "origin_device",
        "seq",
        "hlc",
        "parents",
        "kind",
        "scope",
        "owner_key",
        "fact",
        "body_sha",
    }
    if set(op) != required or type(op["v"]) is not int or op["v"] != 1:
        raise MemoryError("Unsupported memory operation version/fields; update both peers.")
    # Do NOT guard() the whole op: its structural identifiers (op/entity ids,
    # origin_device, owner_key, parents, body_sha) are minted hex/UUID shapes
    # that legitimately collide with the free-text IBAN/card patterns — a
    # ``d_`` + hex device id whose tail passes mod-97 made every put on that
    # install fail as "sensitive". Those fields get the structural _SAFE /
    # hex checks here and below instead. The fact payload is still free text
    # and is guarded as such by validate_fact() on the put path below, so a
    # real IBAN or card number in a title/body remains refused.
    for key in ("op_id", "entity_id", "origin_device", "owner_key"):
        safe_segment(op[key], structural=True)
    if type(op["seq"]) is not int or not 1 <= op["seq"] <= 2**63 - 1:
        raise MemoryError("Invalid origin sequence.")
    if (
        not isinstance(op["hlc"], list)
        or len(op["hlc"]) != 3
        or any(type(x) is not int or not 0 <= x <= 2**63 - 1 for x in op["hlc"][:2])
        or op["hlc"][2] != op["origin_device"]
    ):
        raise MemoryError("Invalid memory logical clock.")
    if not isinstance(op["parents"], list) or len(op["parents"]) > 16:
        raise MemoryError("Memory parents must be a bounded list.")
    for parent in op["parents"]:
        safe_segment(parent, structural=True)
    if (
        op["scope"] not in SCOPES
        or (op["scope"] == "global" and op["owner_key"] != "global")
        or (op["scope"] == "agent" and op["owner_key"] == "main")
    ):
        raise MemoryError("Invalid memory scope/owner.")
    if not isinstance(op["body_sha"], str) or not re.fullmatch(r"[0-9a-f]{64}", op["body_sha"]):
        raise MemoryError("Invalid memory content hash.")
    if op["kind"] == "put" and op["fact"] is None and allow_redacted:
        return op
    if op["kind"] == "put":
        fact = validate_fact(op["fact"])
        if hashlib.sha256(_dump(fact).encode()).hexdigest() != op["body_sha"]:
            raise MemoryError("Memory content hash mismatch.")
    elif op["kind"] != "tombstone" or op["fact"] is not None:
        raise MemoryError("Tombstones contain no memory body.")
    return op


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ops (
 op_id TEXT PRIMARY KEY, entity_id TEXT NOT NULL, origin TEXT NOT NULL,
 seq INTEGER NOT NULL, scope TEXT NOT NULL, owner TEXT NOT NULL,
 kind TEXT NOT NULL, data TEXT NOT NULL, UNIQUE(origin,seq));
CREATE INDEX IF NOT EXISTS ops_entity ON ops(entity_id);
CREATE INDEX IF NOT EXISTS ops_owner ON ops(scope,owner);
CREATE TABLE IF NOT EXISTS current (
 entity_id TEXT PRIMARY KEY, scope TEXT NOT NULL, owner TEXT NOT NULL,
 op_id TEXT NOT NULL, fact TEXT NOT NULL, tokens INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS current_owner ON current(scope,owner);
CREATE VIRTUAL TABLE IF NOT EXISTS recall USING fts5(
 entity_id UNINDEXED,title,tags,body,tokenize='porter unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(recall,'row');
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS associations(entity TEXT,scope TEXT,owner TEXT,locator TEXT,
 PRIMARY KEY(entity,locator));
CREATE INDEX IF NOT EXISTS associations_locator ON associations(scope,owner,locator);
CREATE TABLE IF NOT EXISTS causal_edges(op_id TEXT,parent TEXT,PRIMARY KEY(op_id,parent));
CREATE INDEX IF NOT EXISTS causal_edges_parent ON causal_edges(parent);
CREATE TABLE IF NOT EXISTS captures(session TEXT,compaction TEXT,status TEXT,n INTEGER,
 PRIMARY KEY(session,compaction));
CREATE TABLE IF NOT EXISTS journal_prune(at REAL NOT NULL,origin TEXT NOT NULL,
 ops INTEGER NOT NULL,bytes INTEGER NOT NULL,reason TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS journal_prune_at ON journal_prune(at);
"""


#: The relief sentence and its command hint for a near-cap scope, shared by the
#: write warning and `lop memory status`. ONE function because the two surfaces
#: drifted: the write warning learned that a session->session move is refused,
#: while status kept prescribing `--to session` at session scope — an impossible
#: instruction at the exact moment the operator is out of space.
def quota_relief(scope: str) -> tuple[str, str]:
    """``(advice, command hint)`` for relieving ``scope`` at 80% or more.

    Every branch returns both halves, and the hint is the command that actually
    works at that scope: global moves work state to session, session forgets
    (a session->session move is refused), and team/agent move only facts that
    belong elsewhere.
    """
    if scope == "global":
        return (
            "Consolidate stale facts with forget, or move work-state facts to session " "scope.",
            "`lop memory move ID --to session`",
        )
    if scope == "session":
        return (
            "Consolidate this session's stale facts with forget — completed steps, "
            "superseded values and next-step notes from earlier turns are the stale ones; "
            "move only facts that belong in another scope.",
            "`lop memory forget SCOPE/ID --revision REV --yes`",
        )
    return (
        "Consolidate stale facts with forget; move only facts that belong in another " "scope.",
        "`lop memory move ID --to SCOPE`",
    )


def _contains_unbounded_repeat(text: str) -> bool:
    """Whether ``text`` contains an unbounded repeat outside a character class."""
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "[":
            # A quantifier inside a character class is a literal, not a repeat.
            index += 1
            if index < len(text) and text[index] == "^":
                index += 1
            if index < len(text) and text[index] == "]":
                index += 1
            while index < len(text) and text[index] != "]":
                index += 2 if text[index] == "\\" else 1
            index += 1
            continue
        if char in "*+":
            return True
        if char == "{" and re.match(r"\{\d+,\}", text[index:]):
            return True
        index += 1
    return False


def _nested_unbounded_repeat(pattern: str) -> bool:
    """Whether a repeated group contains an unbounded repeat, e.g. ``(a+)+``.

    WHY refuse instead of compile: that shape backtracks exponentially in the
    subject (measured in review round 1: ``(a+)+$`` on 29 characters ran longer
    than three seconds), and Python's ``re`` has no timeout. This is a lexical
    HEURISTIC, not a proof — an overlapping alternation such as ``(a|aa)+`` is not
    detected — so the contract is "recognised catastrophic shapes are refused",
    not "every accepted pattern runs in bounded time". Filter subjects are
    bounded by the store (title <= 160 chars, item <= ITEM_BYTES).
    """
    stack: list[int] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 2
            continue
        if char == "[":
            index += 1
            if index < len(pattern) and pattern[index] == "^":
                index += 1
            if index < len(pattern) and pattern[index] == "]":
                index += 1
            while index < len(pattern) and pattern[index] != "]":
                index += 2 if pattern[index] == "\\" else 1
            index += 1
            continue
        if char == "(":
            stack.append(index)
        elif char == ")" and stack:
            start = stack.pop()
            following = pattern[index + 1 : index + 2]
            repeated = following in ("*", "+") or bool(re.match(r"\{\d+,\}", pattern[index + 1 :]))
            if repeated and _contains_unbounded_repeat(pattern[start + 1 : index]):
                return True
        index += 1
    return False


#: SQL predicate that honours the documented ``replaces`` edge: the outer
#: ``current`` row must not be the TARGET of an explicit replacement from an
#: authorised sibling. Shared by every read that hides a replaced fact
#: (``search`` and ``tagged``) so two readers cannot develop two definitions of
#: "replaced"; the outer table must be aliased ``c``.
_NOT_REPLACED_SQL = (
    "NOT EXISTS (SELECT 1 FROM ops p "
    "JOIN causal_edges e ON e.parent=p.op_id JOIN ops child ON child.op_id=e.op_id "
    "WHERE p.entity_id=c.entity_id AND child.entity_id<>c.entity_id "
    "AND p.scope=c.scope AND p.owner=c.owner "
    "AND child.scope=c.scope AND child.owner=c.owner)"
)


class MemoryStore:
    def __init__(
        self,
        config: Path,
        identity: MemoryIdentity,
        session_dir: Path | None = None,
        *,
        origin: str | None = None,
    ):
        self.root = Path(config) / "memory"
        self.path = self.root / "memory.db"
        self.identity = identity
        self.session_dir = session_dir
        # Origin is a minted device id (``d_`` + hex; see _origin), i.e.
        # structural — never free-text guard it (see safe_segment).
        self.origin = safe_segment(origin, structural=True) if origin else None

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        # Probe BEFORE touching disk: a SQLite build without FTS5 used to leave an
        # empty memory.db behind and then fail every search with an OperationalError.
        if not fts5_available():
            raise MemoryError(
                "Memory is unavailable: this Python's SQLite was built without FTS5. "
                "Use a Python whose sqlite3 supports FTS5, or set memory.enabled to false."
            )
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        created = not self.path.exists()
        db = sqlite3.connect(self.path, timeout=5)
        try:
            os.chmod(self.path, 0o600)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=5000")
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA secure_delete=ON")
            db.execute("PRAGMA journal_size_limit=1048576")
            db.execute("PRAGMA wal_autocheckpoint=64")
            db.execute(f"PRAGMA max_page_count={MAX_DB_BYTES // CAP_PAGE_SIZE}")
            db.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            db.close()
            if created:
                # A store whose schema never completed must not look initialised.
                for suffix in ("", "-wal", "-shm"):
                    Path(str(self.path) + suffix).unlink(missing_ok=True)
            raise MemoryError(
                "Memory store unavailable (SQLite " + type(exc).__name__ + "); "
                "no memory was read or written."
            ) from exc
        try:
            yield db
        except sqlite3.OperationalError as exc:
            # SQLITE_FULL has two causes and only one is this store's own cap:
            # the host disk can be full too. Translate it only when the cap is
            # implicated (see _page_cap_refusal); every other failure keeps the
            # caller's generic store-unavailable path.
            budget = self._page_cap_refusal(db, exc)
            if budget is not None:
                raise MemoryError(self._store_full_message(*budget)) from exc
            raise
        finally:
            db.close()

    @staticmethod
    def _page_budget(db: sqlite3.Connection) -> tuple[int, int, int, int]:
        """``(page_count, page_size, max_page_count, freelist_count)`` as enforced.

        Read through PRAGMAs rather than derived from ``MAX_DB_BYTES`` so an
        overridden cap (tests) and a non-default page size are measured as
        SQLite enforces them.
        """
        return (
            int(db.execute("PRAGMA page_count").fetchone()[0]),
            int(db.execute("PRAGMA page_size").fetchone()[0]),
            int(db.execute("PRAGMA max_page_count").fetchone()[0]),
            int(db.execute("PRAGMA freelist_count").fetchone()[0]),
        )

    @staticmethod
    def _at_page_cap(pages: int, page_size: int, cap: int, free: int, *, batch: bool) -> bool:
        """The ONE definition of "the byte cap is what refuses writes here".

        Free pages always make a store not-full: SQLite reuses the freelist for
        any table, so a store at ``max_page_count`` with freed pages still
        accepts writes. This is why a successful remedy (forget/VACUUM) must
        not leave status claiming FULL.

        ``batch`` widens the headroom horizon for the REFUSAL path only. One
        transaction can need several pages at once, so SQLite returns
        SQLITE_FULL while a page or two are still unallocated (measured: the
        failed commit left 30 of 32 pages allocated). Status uses
        ``batch=False``: that transient state still accepts smaller writes and
        is reported by the near-cap warning instead of a false FULL.

        Cost of the ``free == 0`` rule on the refusal path: a SQLITE_FULL while
        free pages remain reports the generic refusal, even at the cap where a
        transaction may have exhausted a small freelist mid-flight. That is
        accepted: translating it would put a cap diagnosis on a state that can
        still accept writes. The ``free == 0, pages < cap`` case IS translated,
        with the near-cap arm of the refusal message that states the remaining
        headroom; only ``free > 0`` stays generic.
        """
        slack = MAX_BATCH_BYTES // page_size if batch else 0
        return free == 0 and cap - pages <= slack

    def _page_cap_refusal(
        self, db: sqlite3.Connection, exc: sqlite3.OperationalError
    ) -> tuple[int, int, int] | None:
        """``(page_count, page_size, max_page_count)`` when this FULL is the cap.

        The tuple (rather than a bare cap figure) is what lets the message say
        which of the two states it fired in: at the cap, or below it with the
        batch too large for the remaining headroom.

        When the host filesystem reports less free space than this store's own
        cap, the disk itself may be the cause; that case deliberately keeps the
        generic refusal, because telling an operator whose disk is full to forget
        facts would be advice for the wrong problem. QA confirmed that guard is
        load-bearing: on a genuine disk-full image the page test would otherwise
        pass.
        """
        code = getattr(exc, "sqlite_errorcode", None)
        if code != sqlite3.SQLITE_FULL and "database or disk is full" not in str(exc):
            return None
        try:
            pages, page_size, cap, free = self._page_budget(db)
            cap_bytes = cap * page_size
            disk_free = shutil.disk_usage(self.root).free
        except (sqlite3.Error, OSError):
            return None
        if disk_free <= cap_bytes:
            return None
        if not self._at_page_cap(pages, page_size, cap, free, batch=True):
            return None
        return pages, page_size, cap

    def _store_full_message(self, pages: int, page_size: int, cap: int) -> str:
        """The refusal for a proven cap failure, worded for the state it fired in.

        TWO arms, because one sentence is false in one of them (round-3 review):

        * ``pages >= cap``: the store has reached the cap. Writes that need a new
          page are refused until space is freed; small writes that fit the last
          page can still succeed (round 2 proved this).
        * ``pages < cap``: the batch needed more pages than remain. Saying "has
          reached its cap" here would be false — writes needing new pages were
          measured succeeding immediately afterwards — so this arm states the
          remaining headroom and that a smaller write may still succeed.

        Generic on purpose for everything else: an unavailable store can be a
        permissions, disk or FTS problem, and telling that operator to forget
        facts would be wrong. These messages are only reached once the cap is
        proven, so they can be prescriptive.
        """
        cap_bytes = cap * page_size
        remedies = (
            "Inspect usage with `lop memory status`, reclaim space with "
            "`lop memory forget SCOPE/ID --revision REV --yes`, and compact the file "
            f"with `sqlite3 {shlex.quote(str(self.path))} 'VACUUM'`. Run that VACUUM "
            "first if the forget is also refused. See guide://memory."
        )
        if pages >= cap:
            return (
                f"Memory store is full: {self.path} has reached its "
                f"{format_bytes(cap_bytes)} cap, so writes that need a new page are "
                f"refused until space is freed. {remedies}"
            )
        headroom = (cap - pages) * page_size
        return (
            f"Memory store is near its cap: this request needs more pages than the "
            f"{format_bytes(headroom)} remaining under the {format_bytes(cap_bytes)} "
            "cap, so it was refused and nothing was saved; a smaller write may still "
            f"succeed. {remedies}"
        )

    def _storage_view(self, db: sqlite3.Connection) -> dict[str, Any]:
        """Bytes the store holds against the byte cap, measured from its pages.

        WHY ``page_count * page_size`` and not the ``memory.db*`` file sizes: the
        cap is enforced on the database's page count, while the WAL and SHM
        siblings have their own journal bound. Summing the files could report
        more than the cap while a write still succeeds (or miss the state that
        refuses it); page_count is the number SQLITE_FULL is decided from.
        The cap is ``max_page_count * page_size`` — the enforced number, which
        is page-floored when ``MAX_DB_BYTES`` is not page-aligned.
        """
        pages, page_size, cap, free = self._page_budget(db)
        used = pages * page_size
        cap_bytes = cap * page_size
        return {
            "storage_bytes": used,
            "storage_cap_bytes": cap_bytes,
            "storage_percent": round(100.0 * used / cap_bytes, 1) if cap_bytes else 0.0,
            "storage_free_bytes": free * page_size,
            "storage_full": self._at_page_cap(pages, page_size, cap, free, batch=False),
            "database_path": str(self.path),
        }

    def _origin(self, db: sqlite3.Connection) -> str:
        if self.origin:
            return self.origin
        row = db.execute("SELECT value FROM meta WHERE key='origin'").fetchone()
        if row:
            return str(row[0])
        # Reuse the install's existing stable mesh identity; minting an identity
        # is not enrollment and grants nobody access. A separate memory UUID
        # would become an unauthorised origin as soon as this store was paired.
        from local_operator.network.identity import load_or_mint

        value = load_or_mint(self.root.parent).device_id
        db.execute("INSERT INTO meta VALUES ('origin',?)", (value,))
        return value

    @staticmethod
    def _ops(db: sqlite3.Connection, entity: str) -> list[dict[str, Any]]:
        return [
            json.loads(row[0])
            for row in db.execute("SELECT data FROM ops WHERE entity_id=?", (entity,))
        ]

    @staticmethod
    def _heads(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parents = {p for op in ops for p in op["parents"]}
        return sorted(
            (op for op in ops if op["op_id"] not in parents),
            key=lambda op: (op["hlc"], op["op_id"]),
            reverse=True,
        )

    def _reindex(self, db: sqlite3.Connection, entity: str) -> None:
        ops = self._ops(db, entity)
        # Foreign-entity parents encode explicit replacements. Edges survive
        # body redaction and are indexed independently of arrival order; a
        # replacement/tombstone that arrives before its target cannot resurrect
        # the old fact when that target finally catches up.
        for operation in ops:
            for parent in operation["parents"]:
                db.execute(
                    "INSERT OR IGNORE INTO causal_edges VALUES (?,?)", (operation["op_id"], parent)
                )
        db.execute("DELETE FROM recall WHERE entity_id=?", (entity,))
        db.execute("DELETE FROM current WHERE entity_id=?", (entity,))
        db.execute("DELETE FROM associations WHERE entity=?", (entity,))
        if any(op["kind"] == "tombstone" for op in ops):
            # Forgotten IDs are permanently dead. Bodies in old puts are removed
            # too; keeping their hash and ancestry still permits convergence.
            for op in ops:
                if op["fact"] is not None:
                    op["fact"] = None
                    db.execute("UPDATE ops SET data=? WHERE op_id=?", (_dump(op), op["op_id"]))
            return
        heads = self._heads(ops)
        if not heads:
            return
        op = heads[0]
        fact = op["fact"]
        from local_operator.memory_capture import _LOCATOR

        locators = {match.group().rstrip(".,;:)]}") for match in _LOCATOR.finditer(fact["body"])}
        if fact["subject"]:
            locators.add("subject:" + fact["subject"])
        if fact["source"].get("uri"):
            locators.add(fact["source"]["uri"])
        for locator in sorted(locators)[:16]:
            db.execute(
                "INSERT OR IGNORE INTO associations VALUES (?,?,?,?)",
                (entity, op["scope"], op["owner_key"], locator),
            )
        db.execute(
            "INSERT INTO current VALUES (?,?,?,?,?,?)",
            (
                entity,
                op["scope"],
                op["owner_key"],
                op["op_id"],
                _dump(fact),
                count_text_tokens(_dump(fact)),
            ),
        )
        if fact["status"] in ("active", "candidate", "disputed"):
            db.execute(
                "INSERT INTO recall VALUES (?,?,?,?)",
                (entity, fact["title"], " ".join(fact["tags"] + fact["aliases"]), fact["body"]),
            )

    @staticmethod
    def token_cap(scope: str) -> int:
        return GLOBAL_TOKENS if scope == "global" else OWNER_TOKENS

    @classmethod
    def _quota_view(cls, db: sqlite3.Connection, scope: str, owner: str) -> dict[str, Any]:
        """Used/percent/headroom for one owner, from the same ``current`` rows ``_quota`` counts."""
        used = db.execute(
            "SELECT coalesce(sum(tokens),0) FROM current WHERE scope=? AND owner=?",
            (scope, owner),
        ).fetchone()[0]
        cap = cls.token_cap(scope)
        return {
            "scope": scope,
            "owner": owner,
            "used_tokens": used,
            "cap_tokens": cap,
            "percent": round(100.0 * used / cap, 1) if cap else 0.0,
            "headroom_tokens": cap - used,
        }

    @classmethod
    def _quota_notice(
        cls, db: sqlite3.Connection, scope: str, owner: str
    ) -> tuple[dict[str, Any], str | None]:
        """Return the quota view plus a warning only on a NEW 80/90% crossing.

        The state is the highest threshold this owner has been WARNED about, kept
        beside the store so it survives restarts: without it every subsequent
        write over the line nags, and with it a drop back below 80% (forget/move
        out) clears the state so a later recrossing warns again. Writes are the
        only caller — a read path must never mutate meta or carry a warning.
        """
        view = cls._quota_view(db, scope, owner)
        key = "quota-warn:" + scope + ":" + owner
        row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        stored = int(row[0]) if row else 0
        reached = next((line for line in WARNING_THRESHOLDS if view["percent"] >= line), 0)
        if not reached:
            if stored:
                db.execute("DELETE FROM meta WHERE key=?", (key,))
            return view, None
        if reached <= stored:
            return view, None
        db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, str(reached)))
        # The relief must be actionable WHERE it is offered: a session->session move
        # is refused, so telling a session owner to "move to session" is noise; for
        # team/agent only a fact that belongs elsewhere can move. The sentences live
        # in `quota_relief` so the status page cannot drift from this warning.
        advice, hint = quota_relief(scope)
        relief = advice + (f" ({hint})" if hint else "")
        if reached >= 90:
            guidance = "Consolidate now: writes are refused once the cap is reached. " + relief
        else:
            guidance = relief
        shortfall = "" if view["headroom_tokens"] >= 0 else " (OVER CAP)"
        return view, (
            f"Memory quota warning: {scope} crossed {reached}% "
            f"({view['used_tokens']}/{view['cap_tokens']} tokens, "
            f"{view['headroom_tokens']} left{shortfall}, {view['percent']}% used). " + guidance
        )

    @staticmethod
    def _quota(db: sqlite3.Connection, scope: str, owner: str, *, relief: bool = False) -> None:
        """Refuse growth past the caps. ``relief`` (a forget) skips the OP/TOMBSTONE counts:
        a tombstone is the only way to remove a sensitive body, so refusing it at the cap
        would make the cap a trap. The token quota still applies, and a forget only shrinks it."""
        count = db.execute(
            "SELECT count(*) FROM ops WHERE scope=? AND owner=?", (scope, owner)
        ).fetchone()[0]
        tombs = db.execute(
            "SELECT count(*) FROM ops WHERE scope=? AND owner=? AND kind='tombstone'",
            (scope, owner),
        ).fetchone()[0]
        tokens = db.execute(
            "SELECT coalesce(sum(tokens),0) FROM current WHERE scope=? AND owner=?", (scope, owner)
        ).fetchone()[0]
        if (not relief and (count > MAX_OWNER_OPS or tombs > MAX_TOMBSTONES)) or tokens > (
            GLOBAL_TOKENS if scope == "global" else OWNER_TOKENS
        ):
            raise MemoryError(
                "Memory quota reached (including history/tombstones). Inspect memory://, "
                "consolidate explicitly, or export history; no facts were evicted."
            )

    def prepare_fact(self, fact: Any) -> dict[str, Any]:
        """Pure host stamping/validation, usable before an index-repair write."""
        if not isinstance(fact, dict):
            raise MemoryError("Memory content must be a JSON object.")
        guard(fact)
        source = fact.get("source", {})
        if not isinstance(source, dict):
            raise MemoryError("Memory source must be an object.")
        source = {
            "session": self.identity.session_id,
            "kind": "reported",
            "author": self.identity.agent_id or "anonymous",
            **source,
        }
        return validate_fact(
            dict(
                fact,
                source=source,
                created_at=(time.time() if fact.get("created_at") is None else fact["created_at"]),
            )
        )

    def put(
        self,
        scope: str,
        fact: dict[str, Any],
        *,
        entity: str | None = None,
        expected_version: str | None = None,
        forget: bool = False,
        merge: bool = False,
    ) -> dict[str, Any]:
        """Create, correct (``expected_version``) or forget one fact.

        ``merge`` is the ONLY way to resolve concurrent edits: the new revision parents
        every current head, so the siblings stop being conflicts. An ordinary correction
        deliberately leaves unseen siblings alone, so a human who has READ the conflict
        list (``show``) opts in explicitly.
        """
        owner = self.identity.owner(scope)
        if forget and merge:
            # A merge forget builds a multi-parent tombstone, which is
            # indistinguishable on the wire from a move's source tombstone — the
            # peer-side move guard would refuse a legitimate forget. No shipping
            # surface passes both (the CLI's --resolve is correction-only), so
            # refuse the shape instead of keeping a latent false positive: resolve
            # the conflict with an edit, then forget the revision that was read.
            raise MemoryError(
                "Forget cannot merge concurrent revisions. Resolve them with an edit "
                "(--resolve), then forget the revision you read."
            )
        if entity is not None:
            safe_segment(entity)
            if not expected_version:
                raise MemoryError(
                    "Correction/forget needs expected_version from read; no blind overwrite."
                )
            safe_segment(expected_version)
        elif forget:
            raise MemoryError("Forget needs an existing memory ID and expected_version.")
        # Retention creation is not verification; unknown external source time
        # stays null. Every writer uses the same pure, value-free validator.
        clean = None if forget else self.prepare_fact(fact)
        if clean is not None:
            # Scope-fit refusal BEFORE any DB/index/oplog write (and before the
            # session export): global is for durable personal facts, not work state.
            scope_fit(scope, clean)
        if scope == "session" and self.session_dir:
            self.import_session()
        if entity is not None and not self.path.exists():
            raise MemoryError("Memory not found in the authorized scope.")
        creating = entity is None
        entity = entity or _id()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            ops = self._ops(db, entity)
            if creating and clean and clean["status"] == "active":
                # Exact provenance-aware idempotence, never fuzzy merging of
                # near-duplicates/contradictions. Freshness/status changes differ.
                duplicate = db.execute(
                    "SELECT entity_id,op_id FROM current WHERE scope=? AND owner=? "
                    "AND json_remove(fact,'$.created_at')=json_remove(?,'$.created_at') LIMIT 1",
                    (scope, owner, _dump(clean)),
                ).fetchone()
                if duplicate:
                    return {
                        "id": duplicate[0],
                        "version": duplicate[1],
                        "scope": scope,
                        "status": "active",
                        "deduplicated": True,
                        "quota": self._quota_view(db, scope, owner),
                    }
            heads: list[dict[str, Any]] = []
            if ops:
                if any(op["scope"] != scope or op["owner_key"] != owner for op in ops):
                    raise MemoryError("Memory not found in the authorized scope.")
                if any(op["kind"] == "tombstone" for op in ops):
                    raise MemoryError(
                        "Memory was forgotten. Store a new entity instead; resurrection is refused."
                    )
                heads = self._heads(ops)
                if expected_version not in {head["op_id"] for head in heads}:
                    raise MemoryError(
                        "Memory changed concurrently. Read its current revision and retry."
                    )
                if len(ops) >= MAX_REVISIONS and not forget:
                    raise MemoryError(
                        "Memory revision cap reached; consolidate into a new "
                        "explicitly linked fact."
                    )
                if clean and "created_at" not in fact:
                    selected = next(head for head in heads if head["op_id"] == expected_version)
                    clean["created_at"] = selected["fact"]["created_at"]
                # Editing the displayed head does not silently resolve unseen
                # concurrent siblings. Only that explicitly read revision parents
                # the correction; all other heads stay inspectable conflicts.
                if forget and heads[0]["op_id"] != expected_version:
                    raise MemoryError(
                        "Memory changed concurrently. Read its current revision and retry."
                    )
                parents = [expected_version]
                if merge:
                    parents = list(
                        dict.fromkeys([expected_version, *(head["op_id"] for head in heads)])
                    )[:16]
            elif expected_version:
                raise MemoryError("Memory not found in the authorized scope.")
            else:
                parents = []
            if clean and clean["replaces"]:
                target = self._ops(db, clean["replaces"])
                if not target or any(
                    x["scope"] != scope or x["owner_key"] != owner for x in target
                ):
                    raise MemoryError("Replacement target must exist in the same authorized scope.")
                if clean["replaces"] == entity:
                    raise MemoryError(
                        "Replacement names a separate fact; use an ordinary edit for this ID."
                    )
                parents = list(dict.fromkeys([*parents, self._heads(target)[0]["op_id"]]))
            origin = self._origin(db)
            seq = db.execute(
                "SELECT coalesce(max(seq),0)+1 FROM ops WHERE origin=?", (origin,)
            ).fetchone()[0]
            last = db.execute("SELECT value FROM meta WHERE key='clock'").fetchone()
            now = int(time.time() * 1000)
            clock = max(now, int(last[0]) + 1 if last else 0)
            op = {
                "v": 1,
                "op_id": _id(),
                "entity_id": entity,
                "origin_device": origin,
                "seq": seq,
                "hlc": [clock, 0, origin],
                "parents": parents,
                "kind": "tombstone" if forget else "put",
                "scope": scope,
                "owner_key": owner,
                "fact": clean,
                "body_sha": (
                    heads[0]["body_sha"]
                    if forget and heads
                    else hashlib.sha256(_dump(clean).encode()).hexdigest()
                ),
            }
            validate_op(op)
            db.execute(
                "INSERT INTO ops VALUES (?,?,?,?,?,?,?,?)",
                (op["op_id"], entity, origin, seq, scope, owner, op["kind"], _dump(op)),
            )
            db.execute("INSERT OR REPLACE INTO meta VALUES ('clock',?)", (str(clock),))
            self._reindex(db, entity)
            self._quota(db, scope, owner, relief=forget)
            # Headroom/percent every write, but a warning only on a NEW crossing;
            # the state lives in meta, committed with the write it describes.
            quota, warning = self._quota_notice(db, scope, owner)
            # A session's portable log is fsync'd before the index commit. If
            # that commit crashes, lazy import reconstructs it idempotently.
            if scope == "session" and self.session_dir:
                self._export(db, owner)
            db.commit()
            if forget:
                db.execute("INSERT INTO recall(recall) VALUES ('optimize')")
                db.commit()
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        result: dict[str, Any] = {
            "id": entity,
            "version": op["op_id"],
            "scope": scope,
            "status": "forgotten" if clean is None else clean["status"],
            "quota": quota,
        }
        if warning:
            result["warning"] = warning
        return result

    def forget_many(self, scope: str, expected: dict[str, str]) -> dict[str, Any]:
        """Forget a set of facts in ONE transaction, refusing if any revision moved.

        WHY this exists: the human bulk-forget preview is only honest if the
        confirmed run cannot leave half the set deleted. ``put(forget=True)`` is
        one fact per call and commits each, so looping it would make a partial
        failure permanent. This method re-reads every head inside a single
        ``BEGIN IMMEDIATE`` transaction and inserts every tombstone there: a
        mismatch on any entry refuses the WHOLE set with nothing written.

        ``expected`` maps entity id to the head revision the caller previewed —
        the same token ``put`` takes. The write lock is taken before the first
        re-read, which closes the window between a caller's SELECT and commit; a
        concurrent correction therefore surfaces as a refusal, never as a
        silently forgotten fact under a filter that merely still matched it.
        """
        if not isinstance(expected, dict) or not expected:
            raise MemoryError("Bulk forget needs at least one memory ID and revision.")
        if len(expected) > MAX_BULK_FORGET:
            raise MemoryError(f"A bulk forget accepts at most {MAX_BULK_FORGET} memories at once.")
        owner = self.identity.owner(scope)
        for entity, revision in expected.items():
            # Neither value is free text; refusing a malformed one here keeps the
            # transaction from opening on a request that cannot succeed.
            safe_segment(entity)
            safe_segment(revision)
        if scope == "session" and self.session_dir:
            self.import_session()
        if not self.path.exists():
            raise MemoryError("Memory not found in the authorized scope.")
        forgotten: list[dict[str, Any]] = []
        tokens = 0
        with self.connection() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                origin = self._origin(db)
                last = db.execute("SELECT value FROM meta WHERE key='clock'").fetchone()
                clock = max(int(time.time() * 1000), int(last[0]) + 1 if last else 0)
                for entity in sorted(expected):
                    ops = self._ops(db, entity)
                    if not ops or any(
                        op["scope"] != scope or op["owner_key"] != owner for op in ops
                    ):
                        raise MemoryError("Memory not found in the authorized scope.")
                    if any(op["kind"] == "tombstone" for op in ops):
                        raise MemoryError(
                            "Memory was forgotten concurrently. Read the current set and "
                            "retry; no fact in this bulk forget was changed."
                        )
                    heads = self._heads(ops)
                    if not heads or heads[0]["op_id"] != expected[entity]:
                        raise MemoryError(
                            "Memory changed concurrently. Read its current revision and retry; "
                            "no fact in this bulk forget was changed."
                        )
                    seq = db.execute(
                        "SELECT coalesce(max(seq),0)+1 FROM ops WHERE origin=?", (origin,)
                    ).fetchone()[0]
                    # One strictly increasing HLC per tombstone in this transaction:
                    # two ops sharing a clock would make head order depend on op_id.
                    clock = max(clock + 1, int(time.time() * 1000))
                    op = {
                        "v": 1,
                        "op_id": _id(),
                        "entity_id": entity,
                        "origin_device": origin,
                        "seq": seq,
                        "hlc": [clock, 0, origin],
                        "parents": [expected[entity]],
                        "kind": "tombstone",
                        "scope": scope,
                        "owner_key": owner,
                        "fact": None,
                        "body_sha": heads[0]["body_sha"],
                    }
                    validate_op(op)
                    db.execute(
                        "INSERT INTO ops VALUES (?,?,?,?,?,?,?,?)",
                        (op["op_id"], entity, origin, seq, scope, owner, op["kind"], _dump(op)),
                    )
                    db.execute("INSERT OR REPLACE INTO meta VALUES ('clock',?)", (str(clock),))
                    row = db.execute(
                        "SELECT tokens FROM current WHERE entity_id=?", (entity,)
                    ).fetchone()
                    tokens += int(row[0]) if row else 0
                    self._reindex(db, entity)
                    forgotten.append(
                        {
                            "id": entity,
                            "version": op["op_id"],
                            "scope": scope,
                            "status": "forgotten",
                        }
                    )
                self._quota(db, scope, owner, relief=True)
                quota, warning = self._quota_notice(db, scope, owner)
                if scope == "session" and self.session_dir:
                    self._export(db, owner)
                db.commit()
            except BaseException:
                # sqlite would roll back on close as well; being explicit keeps a
                # caller's error handler from seeing a half-applied batch if the
                # connection is ever reused or pooled.
                db.rollback()
                raise
            db.execute("INSERT INTO recall(recall) VALUES ('optimize')")
            db.commit()
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        result: dict[str, Any] = {"forgotten": forgotten, "tokens": tokens, "quota": quota}
        if warning:
            result["warning"] = warning
        return result

    # -------------------------------------------------------------- compaction
    #
    # The journal carries STATE, not history. ``_reindex`` rebuilds ``current``,
    # ``recall`` and ``associations`` from an entity's ops and materialises
    # ``_heads(ops)[0]``, so a non-head revision is unreadable by every read
    # path. Retaining one op per entity therefore reproduces the store exactly;
    # measured on a 65,929,216 B store: 50,377 ops -> 25,188, ``current`` rebuilt
    # 4,084/4,084 with 0 missing/extra/mismatched, 0 of 25,188 entities changed
    # state, ``causal_edges`` 7.7 MB -> 3 rows, file 32,411,648 B.
    #
    # WHY this needs no peer acknowledgement: the rule is a function of the local
    # op set alone, which is the property that matters here — the failure it
    # fixes is a full store that cannot accept a peer's push, so any rule that
    # waits for an ack cannot run. A rule phrased as "newer than the oldest peer
    # cursor" is dead on arrival: the cursor this store holds for a peer
    # describes what it RECEIVED from that peer (``network/memory.py``), so the
    # minimum over our own origin is always 0 and such a rule frees zero bytes.
    #
    # Protocol safety: a compacted journal is a legal stream. ``batch()`` emits
    # content-free skip ranges for absent sequence numbers and ``receive()``
    # only requires ops+skips to be contiguous, never an op per sequence — so
    # this pass needs no generation bump and no wire change.

    def _retained(self, ops: list[dict[str, Any]]) -> set[str]:
        """The op ids ONE entity keeps when its journal is compacted.

        See the compaction note above for why one op per entity is state
        preserving. TWO RULES MUST NOT BE SIMPLIFIED:

        * ALL heads, never ``self._heads(ops)[0]``. One head reproduces
          ``current`` and silently resolves visible conflicts: ``show()``'s
          conflict list and ``put(merge=...)`` both read the head SET, so
          keeping one would turn a concurrent edit into an invisible one and
          make a later merge resolve against the wrong parent.
        * A forgotten entity keeps EVERY tombstone and no puts. ``_reindex``
          clears an entity only on seeing a tombstone, so a peer that received
          the put and not the tombstone would hold a fact that nothing will
          ever clear. Puts under a tombstone are unreachable (``_reindex``
          returns before writing ``current``) and were 44% of this journal.
        """
        if any(op["kind"] == "tombstone" for op in ops):
            return {op["op_id"] for op in ops if op["kind"] == "tombstone"}
        return {op["op_id"] for op in self._heads(ops)}

    @staticmethod
    def _pins(db: sqlite3.Connection) -> set[str]:
        """Op ids that are endpoints of a CROSS-entity causal edge.

        ``_NOT_REPLACED_SQL`` (``search``/``tagged``) and ``show()``'s
        ``moved_from`` read a replacement/move edge by joining BOTH endpoints,
        so an op that looks like a superseded revision can be the parent half of
        a live replacement. ``causal_edges`` is the authority rather than the op
        bodies because an edge survives body redaction, and it is the one table
        that still knows a parent whose own row is already gone.
        """
        pinned: set[str] = set()
        for child, parent, child_entity, parent_entity in db.execute(
            "SELECT e.op_id, e.parent, c.entity_id, p.entity_id FROM causal_edges e "
            "JOIN ops c ON c.op_id=e.op_id JOIN ops p ON p.op_id=e.parent"
        ):
            if child_entity != parent_entity:
                pinned.update((child, parent))
        return pinned

    @staticmethod
    def _journal_fingerprint(victims: list[str]) -> str:
        """Short digest of the exact op set a preview would delete.

        The confirmed run recomputes it inside its write transaction: any
        concurrent write that would change what is deleted (a new revision, a
        forget, a received op) changes the victim set and is refused. Hashing
        victims only is deliberate — a write that only ADDS retained state does
        not change the outcome and must not make the operator's preview stale.
        """
        digest = hashlib.sha256()
        for op_id in sorted(victims):
            digest.update(op_id.encode() + b"\n")
        return digest.hexdigest()[:12]

    @staticmethod
    def _empty_journal_plan(path: Path) -> dict[str, Any]:
        """The plan shape for a store that does not exist (nothing to compact)."""
        return {
            "exists": False,
            "database_path": str(path),
            "ops": {"before": 0, "kept": 0, "dropped": 0},
            "entities": {"live": 0, "forgotten": 0},
            "retained": {"ops": 0, "payload_bytes": 0, "tombstones": 0, "pinned": 0},
            "dropped": {
                "superseded": 0,
                "under_forgotten": 0,
                "payload_bytes": 0,
            },
            "causal_edges": {"rows": 0, "kept": 0, "dropped": 0},
            "storage": {
                "bytes_before": 0,
                "pages_before": 0,
                "page_size": CAP_PAGE_SIZE,
                "freelist_pages": 0,
                "page_cap": MAX_DB_BYTES // CAP_PAGE_SIZE,
                "headroom_before": (MAX_DB_BYTES // CAP_PAGE_SIZE) * CAP_PAGE_SIZE,
                "bytes_after_estimate": 0,
                "pages_after_estimate": 0,
                "reclaim_estimate": 0,
                "estimate_basis": "none",
            },
            "last_pass": None,
            "fingerprint": MemoryStore._journal_fingerprint([]),
        }

    def _journal_estimate(
        self,
        db: sqlite3.Connection,
        keep: set[str],
        sizes: dict[str, int],
        surviving_edges: int,
        total_edges: int,
        pages: int,
        page_size: int,
    ) -> tuple[int, str]:
        """``(predicted bytes after the pass, which model produced it)``.

        ``dbstat`` is the only view of how many pages each b-tree actually
        holds, and the ops table is where a payload-only model is badly wrong:
        deleted rows leave free space INSIDE pages, which only ``VACUUM`` packs
        away, so "before - dropped payload" understates the reclaim by
        megabytes. Scaling every object by its own retained share is an upper
        bound (packing can only make the result smaller) and measured 13% high
        on the real store: 36,564,992 predicted vs 32,411,648 attained.

        ``dbstat`` is a compile option (``SQLITE_ENABLE_DBSTAT_VTAB``), not a
        SQLite version guarantee. Without it the payload-only lower bound is
        returned with a different ``basis`` so the preview can say which number
        the operator is reading rather than pass an estimate off as a
        measurement.
        """
        retained_bytes = sum(sizes[op_id] for op_id in keep)
        total_bytes = sum(sizes.values())
        try:
            stats = db.execute("SELECT name, sum(pgsize) FROM dbstat GROUP BY name").fetchall()
        except sqlite3.OperationalError:
            return max(0, pages * page_size - (total_bytes - retained_bytes)), "payload"
        if not total_bytes:
            return pages * page_size, "dbstat"
        index_table = {
            name: table
            for name, table in db.execute(
                "SELECT name, tbl_name FROM sqlite_schema WHERE type='index'"
            )
        }
        kept_rows = len(keep)
        rows = len(sizes)
        predicted_pages = 0
        for name, size in stats:
            if not size:
                continue
            held = -(-int(size) // page_size)
            table = "ops" if name == "ops" else index_table.get(name)
            if name == "ops":
                scaled = -(-held * retained_bytes // total_bytes)
            elif table == "ops":
                scaled = -(-held * kept_rows // rows)
            elif name == "causal_edges" or table == "causal_edges":
                scaled = -(-held * surviving_edges // total_edges) if total_edges else held
            else:
                scaled = held
            predicted_pages += max(1, scaled)
        return predicted_pages * page_size, "dbstat"

    def _journal_plan(self, db: sqlite3.Connection) -> tuple[dict[str, Any], list[str], set[str]]:
        """``(preview payload, victim op ids, owners whose export must be rewritten)``.

        ONE scan over ``ops`` rather than ``_ops`` per entity: the measured
        store has 25,188 entities, and the per-entity reader would be 25,188
        round trips for the same bytes. ``_retained``/``_heads`` — the only
        definition of a head — still decides; this only supplies their input,
        plus the per-op accounting (bytes, entity, scope/owner) a receipt and
        the export rewrite need.
        """
        rows = db.execute(
            "SELECT op_id, entity_id, owner, scope, data, length(CAST(data AS BLOB)) FROM ops"
        ).fetchall()
        by_entity: dict[str, list[dict[str, Any]]] = {}
        entity_of: dict[str, str] = {}
        size_of: dict[str, int] = {}
        place_of: dict[str, tuple[str, str]] = {}
        for op_id, entity, owner, scope, data, size in rows:
            by_entity.setdefault(entity, []).append(json.loads(data))
            entity_of[op_id] = entity
            size_of[op_id] = int(size)
            place_of[op_id] = (scope, owner)
        pinned = self._pins(db)
        keep = set(pinned)
        for ops in by_entity.values():
            keep |= self._retained(ops)
        dead = {
            entity
            for entity, ops in by_entity.items()
            if any(op["kind"] == "tombstone" for op in ops)
        }
        victims = [op_id for op_id in size_of if op_id not in keep]
        superseded = sum(1 for op_id in victims if entity_of[op_id] not in dead)
        tombstones = sum(
            1 for entity in dead for op in by_entity[entity] if op["kind"] == "tombstone"
        )
        edges = db.execute("SELECT op_id, parent FROM causal_edges").fetchall()
        surviving_edges = sum(1 for child, parent in edges if child in keep and parent in keep)
        pages, page_size, cap, free = self._page_budget(db)
        after_estimate, basis = self._journal_estimate(
            db, keep, size_of, surviving_edges, len(edges), pages, page_size
        )
        last = db.execute(
            "SELECT at, ops, bytes FROM journal_prune ORDER BY at DESC LIMIT 1"
        ).fetchone()
        dropped_bytes = sum(size_of[op_id] for op_id in victims)
        before = pages * page_size
        plan = {
            "exists": True,
            "database_path": str(self.path),
            "ops": {"before": len(size_of), "kept": len(keep), "dropped": len(victims)},
            "entities": {"live": len(by_entity) - len(dead), "forgotten": len(dead)},
            "retained": {
                "ops": len(keep),
                "payload_bytes": sum(size_of[op_id] for op_id in keep),
                "tombstones": tombstones,
                "pinned": len(pinned),
            },
            "dropped": {
                "superseded": superseded,
                "under_forgotten": len(victims) - superseded,
                "payload_bytes": dropped_bytes,
            },
            "causal_edges": {
                "rows": len(edges),
                "kept": surviving_edges,
                "dropped": len(edges) - surviving_edges,
            },
            "storage": {
                "bytes_before": before,
                "pages_before": pages,
                "page_size": page_size,
                "freelist_pages": free,
                "page_cap": cap,
                "headroom_before": cap * page_size - before,
                "bytes_after_estimate": after_estimate,
                "pages_after_estimate": -(-after_estimate // page_size),
                "reclaim_estimate": max(0, before - after_estimate),
                "estimate_basis": basis,
            },
            "last_pass": ({"at": last[0], "ops": last[1], "bytes": last[2]} if last else None),
            "fingerprint": self._journal_fingerprint(victims),
        }
        owners = {place_of[op_id][1] for op_id in victims if place_of[op_id][0] == "session"}
        return plan, victims, owners

    def journal_plan(self) -> dict[str, Any]:
        """What :meth:`compact` would delete and roughly what it would free.

        Read-only by construction: no INSERT/UPDATE/DELETE, and a store that
        does not exist is not created (``connection()`` would create it). This
        is the operator's preview — the pass refuses to run without the
        fingerprint it prints.
        """
        if not self.path.exists():
            return {**self._empty_journal_plan(self.path), "read_only": True}
        with self.connection() as db:
            plan, _victims, _owners = self._journal_plan(db)
        return {**plan, "read_only": True}

    def compact(self, *, expect: str) -> dict[str, Any]:
        """Apply the previewed journal compaction in one pass.

        ``expect`` is the ``fingerprint`` of the preview the operator read; the
        plan is recomputed inside the write transaction and a mismatch refuses
        the WHOLE pass with nothing deleted (the same optimistic-confirmation
        shape as a bulk forget's ``--expect``). Requiring it is deliberate: old
        revisions are unrecoverable afterwards, and a call without a preview is
        a call whose operator has not seen what it deletes.

        Serialisation: ``BEGIN IMMEDIATE`` takes the write lock BEFORE the plan
        is read, so ``apply_ops`` (also IMMEDIATE) lands wholly before or wholly
        after; ``busy_timeout=5000`` from ``connection()`` is what waits. A push
        that loses that race and read a pre-compaction sequence state gets one
        ``Memory sequence gap/overlap`` refusal on the receiver and succeeds on
        its next tick against the compacted stream — the journal announces the
        absent numbers as skip ranges, so the stream stays legal.

        Export ordering: the deletes commit first, then every affected session's
        portable ``memory-session.jsonl`` is rewritten. A crash between the two
        leaves an export that still holds dropped ops; ``import_session``
        re-applies them and the next pass drops them again, so the residue is
        bytes, never wrong state.

        ``VACUUM`` cannot run inside a transaction, and is skipped when the pass
        dropped nothing and the freelist is already empty — that is what makes a
        second run a no-op instead of a full file rewrite. It is also what makes
        the reclaim real: the DELETE only returns pages to the freelist.
        """
        if not self.path.exists():
            raise MemoryError("Memory not found: there is no journal to compact.")
        if not expect:
            raise MemoryError(
                "Compaction needs the fingerprint from its preview. Nothing was deleted. "
                "Run the preview first."
            )
        with self.connection() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                plan, victims, owners = self._journal_plan(db)
                if plan["fingerprint"] != expect:
                    raise MemoryError(
                        "The journal changed since the preview you confirmed (fingerprint "
                        "mismatch). Nothing was deleted; run the preview again."
                    )
                if victims:
                    db.executemany("DELETE FROM ops WHERE op_id=?", [(op,) for op in victims])
                    # An edge row is inert iff either endpoint is gone: the only
                    # reader (``_NOT_REPLACED_SQL``) INNER JOINs both ops, so
                    # deleting these cannot change a query result. They are also
                    # the second-largest object in the store, so they are pruned
                    # rather than left to rot.
                    db.execute(
                        "DELETE FROM causal_edges WHERE op_id NOT IN (SELECT op_id FROM ops) "
                        "OR parent NOT IN (SELECT op_id FROM ops)"
                    )
                    db.execute(
                        "INSERT INTO journal_prune VALUES (?,?,?,?,?)",
                        (
                            time.time(),
                            self._origin(db),
                            len(victims),
                            plan["dropped"]["payload_bytes"],
                            "operator",
                        ),
                    )
                db.commit()
                if victims:
                    for owner in sorted(owners):
                        directory = self._session_directory(owner, db)
                        if directory is None:
                            continue
                        try:
                            self._export(db, owner, directory)
                        except (MemoryError, OSError) as exc:
                            raise MemoryError(
                                "The journal was compacted, but rewriting session "
                                f"{owner[:8]}'s portable export failed ({exc}). The store is "
                                "correct; that export still holds the dropped operations and "
                                "the next pass rewrites it."
                            ) from exc
                    db.commit()
            except BaseException:
                db.rollback()
                raise
            free_before = plan["storage"]["freelist_pages"]
        vacuumed = False
        if plan["ops"]["dropped"] or free_before:
            try:
                with self.connection() as db:
                    db.execute("VACUUM")
                vacuumed = True
            except sqlite3.OperationalError as exc:
                raise MemoryError(
                    "The journal was compacted, but the file could not be rebuilt (VACUUM: "
                    f"{exc}). The freed pages are already reusable, so writes are not "
                    "blocked; free disk space and run the pass again to shrink the file."
                ) from exc
        with self.connection() as db:
            pages, page_size, cap, free = self._page_budget(db)
        after = pages * page_size
        return {
            **plan,
            "action": "compact",
            "read_only": False,
            "applied": True,
            "vacuumed": vacuumed,
            "storage": {
                **plan["storage"],
                "bytes_after": after,
                "pages_after": pages,
                "freelist_pages": free,
                "reclaimed_bytes": max(0, plan["storage"]["bytes_before"] - after),
                "headroom_after": cap * page_size - after,
            },
            "confirmation": expect,
        }

    def move(
        self,
        targets: str | list[str],
        destination: str,
        *,
        expected_version: str | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Move one or more facts to ``destination`` atomically, preserving the fact.

        WHY this exists: a fact written to the wrong scope could previously only be
        forgotten (destroys it) or edited (scope is immutable). The global bank filled
        to its cap with work state while session scope sat empty, and an agent could
        not act on the advice to "use session scope instead".

        DESIGN CONSTRAINT: ``apply_ops`` refuses an entity whose ops change scope, so
        a move cannot rewrite an entity in place. It is represented as two ordinary
        ops that OLD PEERS ALREADY UNDERSTAND (no protocol bump): a ``put`` at the
        destination whose first revision PARENTS the source head, and a ``tombstone``
        at the source. The destination fact is byte-identical (text, provenance,
        created_at and every other timestamp), ``show`` reports ``moved_from`` from
        the cross-entity root parent, and the ordinary op stream replicates it. A new
        op kind was rejected deliberately: a peer running an older release would
        refuse the whole batch and then sit behind a permanent sequence gap.

        The source tombstone is destructive exactly like a forget — it ends that
        entity. A move therefore refuses a source that already has unmerged
        concurrent revisions, and ``apply_ops`` refuses a move tombstone that would
        destroy a revision the moving device never saw (the tombstone names the
        destination op as a second parent, which is how a peer tells it apart from a
        forget without a protocol change).

        All-or-nothing: every target is resolved and validated, the destination
        quota and the source's history caps are checked, before the single
        ``BEGIN IMMEDIATE`` transaction inserts any op; any failure before commit
        leaves the store untouched.
        """
        if isinstance(targets, str):
            targets = [targets]
        if not targets:
            raise MemoryError("A move needs at least one memory ID.")
        if len(targets) > 64:
            raise MemoryError("A move accepts at most 64 memories at once.")
        if destination not in SCOPES:
            raise MemoryError("Unknown move destination; use session, team, agent or global.")
        dest_owner = self.identity.owner(destination)
        if expected_version is not None:
            safe_segment(expected_version)
        resolved: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for target in targets:
            if not isinstance(target, str) or not target.strip():
                raise MemoryError("Every move target must be an explicit memory ID.")
            source_scope, entity = self.resolve(target, scope)
            if (source_scope, entity) in seen:
                raise MemoryError(
                    f"{source_scope}/{entity} is listed twice; a move takes each fact once."
                )
            seen.add((source_scope, entity))
            if source_scope == destination:
                raise MemoryError(
                    f"{source_scope}/{entity} is already in {destination}; "
                    "moving it there would do nothing."
                )
            resolved.append((source_scope, entity))
        if expected_version is not None and len(resolved) != 1:
            raise MemoryError("A revision guards one fact; omit it when moving a batch.")
        if destination == "session" or any(source == "session" for source, _ in resolved):
            # A moved/opened session's portable log may hold rows the local index
            # has not rebuilt yet; moving against a stale index could miss a
            # conflict or resurrect a forgotten fact on the next import.
            self.import_session()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            origin = self._origin(db)
            # Validate EVERY target before writing anything, so a refusal later in
            # the batch cannot leave an earlier move half-applied.
            prepared: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
            needed = 0
            batch_facts: set[str] = set()
            source_counts: dict[tuple[str, str], int] = {}
            for source_scope, entity in resolved:
                source_owner = self.identity.owner(source_scope)
                ops = self._ops(db, entity)
                if not ops or any(
                    op["scope"] != source_scope or op["owner_key"] != source_owner for op in ops
                ):
                    raise MemoryError("Memory not found in the authorized scope.")
                if any(op["kind"] == "tombstone" for op in ops):
                    raise MemoryError(
                        f"{source_scope}/{entity} was forgotten; it cannot be moved. Store it "
                        "again in the destination scope instead."
                    )
                heads = self._heads(ops)
                if len(heads) > 1:
                    # Concurrent sibling revisions are a decision a human has to
                    # make. Moving the newest and tombstoning the entity would
                    # silently discard the others, so refuse while it is ambiguous.
                    raise MemoryError(
                        f"{source_scope}/{entity} has {len(heads)} concurrent revisions; "
                        "resolve the conflict (edit with --resolve) before moving it."
                    )
                head = heads[0]
                if expected_version is not None and head["op_id"] != expected_version:
                    raise MemoryError(
                        "Memory changed concurrently. Read its current revision and retry."
                    )
                fact = dict(head["fact"])
                # A move must never be a route around the guards ``put`` enforces.
                # The strongest of them is scope-fit: work/lane/progress content is
                # refused in the global bank, and moving it there would defeat it.
                scope_fit(destination, fact)
                if fact["replaces"]:
                    # ``replaces`` is a same-scope lineage link; carrying it across
                    # scopes is refused by ``put``'s replacement check. Say so plainly
                    # rather than fail mid-transaction.
                    raise MemoryError(
                        f"{source_scope}/{entity} supersedes another fact in the same "
                        "scope, so a cross-scope move would break that link. Resolve the "
                        "supersession first (forget the superseded fact, or keep this "
                        "one in place)."
                    )
                duplicate = db.execute(
                    "SELECT entity_id FROM current WHERE scope=? AND owner=? "
                    "AND json_remove(fact,'$.created_at')=json_remove(?,'$.created_at') LIMIT 1",
                    (destination, dest_owner, _dump(fact)),
                ).fetchone()
                if duplicate:
                    raise MemoryError(
                        f"{source_scope}/{entity} already has an identical fact at "
                        f"{destination}/{duplicate[0]}; a move would silently merge them. "
                        "Resolve one of the two first."
                    )
                needed += count_text_tokens(_dump(fact))
                # Two byte-identical facts in one batch would create two identical
                # destination rows: ``put``'s idempotence would have refused the
                # second, and ``move`` bypasses ``put``, so refuse it here.
                signature = _dump(
                    {key: value for key, value in fact.items() if key != "created_at"}
                )
                if signature in batch_facts:
                    raise MemoryError(
                        "Two targets in this batch hold the same fact; moving both would "
                        "duplicate it at the destination. Keep one."
                    )
                batch_facts.add(signature)
                source_counts[(source_scope, source_owner)] = (
                    source_counts.get((source_scope, source_owner), 0) + 1
                )
                prepared.append((source_scope, entity, head, fact))
            cap = self.token_cap(destination)
            current_tokens = db.execute(
                "SELECT coalesce(sum(tokens),0) FROM current WHERE scope=? AND owner=?",
                (destination, dest_owner),
            ).fetchone()[0]
            if current_tokens + needed > cap:
                raise MemoryError(
                    f"Memory move refused: {destination} would exceed its {cap}-token cap "
                    f"(current {current_tokens} tokens, needs {needed} tokens, cap {cap}). "
                    "Consolidate the destination or move fewer facts; no fact was truncated "
                    "and nothing was changed."
                )
            destination_ops = db.execute(
                "SELECT count(*) FROM ops WHERE scope=? AND owner=?",
                (destination, dest_owner),
            ).fetchone()[0]
            if destination_ops + len(prepared) > MAX_OWNER_OPS:
                raise MemoryError(
                    f"Memory move refused: {destination} would exceed its operation history "
                    "cap. Consolidate or export the destination history first."
                )
            # The source gains one tombstone per moved fact. Refuse rather than push
            # a scope past the history caps: the next write there would then be
            # refused, and tombstones are never collected (review MINOR-5).
            for (source_scope, source_owner), count in source_counts.items():
                source_ops = db.execute(
                    "SELECT count(*) FROM ops WHERE scope=? AND owner=?",
                    (source_scope, source_owner),
                ).fetchone()[0]
                source_tombs = db.execute(
                    "SELECT count(*) FROM ops WHERE scope=? AND owner=? AND kind='tombstone'",
                    (source_scope, source_owner),
                ).fetchone()[0]
                if source_ops + count > MAX_OWNER_OPS or source_tombs + count > MAX_TOMBSTONES:
                    raise MemoryError(
                        f"Memory move refused: {source_scope} would exceed its operation/"
                        "tombstone history cap. Consolidate or export that history first; "
                        "nothing was changed."
                    )
            seq = db.execute(
                "SELECT coalesce(max(seq),0)+1 FROM ops WHERE origin=?", (origin,)
            ).fetchone()[0]
            last = db.execute("SELECT value FROM meta WHERE key='clock'").fetchone()
            clock = max(int(time.time() * 1000), int(last[0]) + 1 if last else 0)
            moved: list[dict[str, Any]] = []
            affected_sessions: set[str] = set()
            for source_scope, entity, head, fact in prepared:
                new_id = _id()
                operation = {
                    "v": 1,
                    "op_id": _id(),
                    "entity_id": new_id,
                    "origin_device": origin,
                    "seq": seq,
                    "hlc": [clock, 0, origin],
                    "parents": [head["op_id"]],
                    "kind": "put",
                    "scope": destination,
                    "owner_key": dest_owner,
                    "fact": fact,
                    "body_sha": hashlib.sha256(_dump(fact).encode()).hexdigest(),
                }
                validate_op(operation)
                db.execute(
                    "INSERT INTO ops VALUES (?,?,?,?,?,?,?,?)",
                    (
                        operation["op_id"],
                        new_id,
                        origin,
                        seq,
                        destination,
                        dest_owner,
                        "put",
                        _dump(operation),
                    ),
                )
                self._reindex(db, new_id)
                moved.append(
                    {
                        "from": f"{source_scope}/{entity}",
                        "to": f"{destination}/{new_id}",
                        "version": operation["op_id"],
                        "title": fact["title"],
                        "tokens": count_text_tokens(_dump(fact)),
                    }
                )
                seq += 1
                clock += 1
                tombstone = {
                    "v": 1,
                    "op_id": _id(),
                    "entity_id": entity,
                    "origin_device": origin,
                    "seq": seq,
                    "hlc": [clock, 0, origin],
                    "parents": [head["op_id"], operation["op_id"]],
                    "kind": "tombstone",
                    "scope": source_scope,
                    "owner_key": self.identity.owner(source_scope),
                    "fact": None,
                    "body_sha": head["body_sha"],
                }
                validate_op(tombstone)
                db.execute(
                    "INSERT INTO ops VALUES (?,?,?,?,?,?,?,?)",
                    (
                        tombstone["op_id"],
                        entity,
                        origin,
                        seq,
                        source_scope,
                        tombstone["owner_key"],
                        "tombstone",
                        _dump(tombstone),
                    ),
                )
                self._reindex(db, entity)
                seq += 1
                clock += 1
                if source_scope == "session":
                    affected_sessions.add(tombstone["owner_key"])
                if destination == "session":
                    affected_sessions.add(dest_owner)
            db.execute("INSERT OR REPLACE INTO meta VALUES ('clock',?)", (str(clock),))
            self._quota(db, destination, dest_owner)
            quota, warning = self._quota_notice(db, destination, dest_owner)
            # Move-out relief must reset the SOURCE's crossing state exactly like a
            # forget does, or a later recross into that scope warns nothing.
            for (source_scope, source_owner), _count in source_counts.items():
                self._quota_notice(db, source_scope, source_owner)
            # Mirror ``put``/``apply_ops``: a session's portable log is written
            # before the index commit, so a crash is repaired by lazy import.
            for owner in sorted(affected_sessions):
                directory = (
                    self.session_dir
                    if owner == self.identity.session_id
                    else self._session_directory(owner, db)
                )
                if directory is not None:
                    if directory.is_dir() or owner == self.identity.session_id:
                        self._export(db, owner, directory)
            db.commit()
        result: dict[str, Any] = {
            "action": "moved",
            "destination": destination,
            "moved": moved,
            "quota": quota,
        }
        if warning:
            result["warning"] = warning
        return result

    def _session_directory(self, owner: str, db: sqlite3.Connection | None = None) -> Path | None:
        if owner == self.identity.session_id and self.session_dir:
            return self.session_dir
        root = self.root.parent / "sessions"
        row = None
        if db is not None:
            row = db.execute(
                "SELECT value FROM meta WHERE key=?", ("sessiondir:" + owner,)
            ).fetchone()
        elif self.path.exists():
            with self.connection() as reader:
                row = reader.execute(
                    "SELECT value FROM meta WHERE key=?", ("sessiondir:" + owner,)
                ).fetchone()
        candidate = root / (row[0] if row else owner)
        try:
            if candidate.resolve().is_relative_to(root.resolve()) and candidate.is_dir():
                return candidate
        except (OSError, RuntimeError):
            pass
        return None

    def _export(self, db: sqlite3.Connection, owner: str, directory: Path | None = None) -> None:
        directory = directory or self.session_dir
        assert directory is not None
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / SESSION_EXPORT_NAME
        if path.is_symlink():
            raise MemoryError("Portable memory must not be a symlink.")
        rows = db.execute(
            "SELECT data FROM ops WHERE scope='session' AND owner=? ORDER BY origin,seq", (owner,)
        ).fetchall()
        data = "".join(row[0] + "\n" for row in rows)
        encoded = data.encode("utf-8")
        if len(encoded) > 16 * 1024 * 1024:
            raise MemoryError("Session memory export cap reached; consolidate explicitly first.")
        # Unique 0600 staging, never O_TRUNC through an unexpected symlink. Like
        # network.store's staged writer this survives a PROCESS crash, not a
        # power cut (no directory fsync); the full old/new snapshot is the truth.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=".memory-session.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        try:
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        db.execute(
            "INSERT OR REPLACE INTO meta VALUES (?,?)",
            ("exporthash:" + owner, hashlib.sha256(encoded).hexdigest()),
        )
        sessions = (self.root.parent / "sessions").resolve()
        if directory.resolve().is_relative_to(sessions):
            # Local control-plane address, never a fact or a wire payload. This
            # also finds subagent namespaces before their holder opens again.
            relative = directory.resolve().relative_to(sessions).as_posix()
            db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?,?)", ("sessiondir:" + owner, relative)
            )

    def apply_ops(
        self,
        batch: list[dict[str, Any]],
        *,
        origins: set[str] | None = None,
        contiguous: bool = True,
        portable: bool = False,
        promote: bool = False,
    ) -> int:
        """Atomic idempotent merge. Authenticate origins before calling from mesh.

        A redacted put is legal ONLY beside an irreversible tombstone for its
        entity. Keeping its op/hash preserves causal/vector identity without
        reintroducing the forgotten content in export or later replication.
        """
        row_cap = MAX_OWNER_OPS if portable else 256
        byte_cap = 16 * 1024 * 1024 if portable else MAX_BATCH_BYTES
        if (
            not isinstance(batch, list)
            or len(batch) > row_cap
            or len(_dump(batch).encode()) > byte_cap
        ):
            raise MemoryError("Memory batch exceeds 256 operations/256 KiB.")
        tombs = {
            op.get("entity_id")
            for op in batch
            if isinstance(op, dict) and op.get("kind") == "tombstone"
        }
        for op in batch:
            validate_op(op, allow_redacted=True)
            if origins is not None and op["origin_device"] not in origins:
                raise MemoryError("Memory origin is not an active trusted network member.")
        if not batch:
            return 0
        if not portable:
            # A remote op may arrive before a moved session is opened. Index
            # its complete portable truth before exporting the merged namespace.
            for owner in {op["owner_key"] for op in batch if op["scope"] == "session"}:
                directory = self._session_directory(owner)
                if directory and (directory / SESSION_EXPORT_NAME).exists():
                    MemoryStore(self.root.parent, MemoryIdentity(owner), directory).import_session()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if promote:
                origin = self._origin(db)
                seq = db.execute(
                    "SELECT coalesce(max(seq),0) FROM ops WHERE origin=?", (origin,)
                ).fetchone()[0]
                clock_row = db.execute("SELECT value FROM meta WHERE key='clock'").fetchone()
                clock = max(int(time.time() * 1000), int(clock_row[0]) if clock_row else 0)
                prefix = self.identity.session_id + "|"

                def remap(value: str) -> str:
                    return hashlib.sha256((prefix + value).encode()).hexdigest()

                promoted = []
                for operation in batch:
                    mapped_id = remap(operation["op_id"])
                    existing = db.execute(
                        "SELECT data FROM ops WHERE op_id=?", (mapped_id,)
                    ).fetchone()
                    if existing:
                        promoted.append(json.loads(existing[0]))
                        continue
                    parents = [remap(parent) for parent in operation["parents"]] + [
                        operation["op_id"]
                    ]
                    if len(parents) > 16:
                        raise MemoryError("Promotion parent cap reached; no history was discarded.")
                    fact = operation["fact"]
                    if fact and fact["replaces"]:
                        fact = dict(fact, replaces=remap(fact["replaces"]))
                    seq += 1
                    clock += 1
                    rewritten = dict(
                        operation,
                        op_id=mapped_id,
                        entity_id=remap(operation["entity_id"]),
                        owner_key=self.identity.session_id,
                        origin_device=origin,
                        seq=seq,
                        hlc=[clock, 0, origin],
                        parents=parents,
                        fact=fact,
                        body_sha=(
                            hashlib.sha256(_dump(fact).encode()).hexdigest()
                            if fact
                            else operation["body_sha"]
                        ),
                    )
                    validate_op(rewritten, allow_redacted=True)
                    promoted.append(rewritten)
                batch = promoted
                tombs = {op["entity_id"] for op in batch if op["kind"] == "tombstone"}
            changed: set[str] = set()
            filled_gap = False
            owners: set[tuple[str, str]] = (
                {("session", self.identity.session_id)} if portable else set()
            )
            growing = set(owners)
            for op in sorted(batch, key=lambda x: (x["origin_device"], x["seq"])):
                entity = op["entity_id"]
                old_ops = self._ops(db, entity)
                if any(
                    x["scope"] != op["scope"] or x["owner_key"] != op["owner_key"] for x in old_ops
                ):
                    raise MemoryError("Memory entity scope/owner is immutable.")
                dead = entity in tombs or any(x["kind"] == "tombstone" for x in old_ops)
                if op["kind"] == "put" and op["fact"] is None and not dead:
                    raise MemoryError("Redacted memory put requires its tombstone.")
                old = db.execute("SELECT data FROM ops WHERE op_id=?", (op["op_id"],)).fetchone()
                if old:
                    previous = json.loads(old[0])
                    if dead:
                        previous["fact"] = None
                        comparison = dict(op, fact=None)
                    else:
                        comparison = op
                    if previous != comparison:
                        raise MemoryError(
                            "Memory operation identity cannot be reused with different content."
                        )
                    continue
                if op["kind"] == "tombstone":
                    # A move's tombstone must not erase a revision the mover never
                    # saw; a forget's own guarantee is untouched (see the helper).
                    conflict = self._move_tombstone_conflict(db, op, old_ops)
                    if conflict:
                        raise MemoryError(conflict)
                if contiguous:
                    seq = db.execute(
                        "SELECT coalesce(max(seq),0)+1 FROM ops WHERE origin=?",
                        (op["origin_device"],),
                    ).fetchone()[0]
                    if op["seq"] != seq:
                        raise MemoryError("Memory sequence gap; refresh peer state and retry.")
                if len(old_ops) >= MAX_REVISIONS and op["kind"] != "tombstone":
                    raise MemoryError("Memory revision cap reached; no conflicts discarded.")
                previous_max = db.execute(
                    "SELECT coalesce(max(seq),0) FROM ops WHERE origin=?", (op["origin_device"],)
                ).fetchone()[0]
                filled_gap |= op["seq"] < previous_max
                stored = dict(op, fact=None) if dead and op["kind"] == "put" else op
                try:
                    db.execute(
                        "INSERT INTO ops VALUES (?,?,?,?,?,?,?,?)",
                        (
                            op["op_id"],
                            entity,
                            op["origin_device"],
                            op["seq"],
                            op["scope"],
                            op["owner_key"],
                            op["kind"],
                            _dump(stored),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise MemoryError("Memory origin sequence reused; refresh peer state.") from exc
                changed.add(entity)
                owners.add((op["scope"], op["owner_key"]))
                if op["kind"] != "tombstone":
                    growing.add((op["scope"], op["owner_key"]))
            if filled_gap:
                previous = db.execute(
                    "SELECT value FROM meta WHERE key='stream-generation'"
                ).fetchone()
                db.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('stream-generation',?)",
                    (str(int(previous[0]) + 1 if previous else 1),),
                )
            if changed:
                clock = max(op["hlc"][0] for op in batch)
                last = db.execute("SELECT value FROM meta WHERE key='clock'").fetchone()
                db.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('clock',?)",
                    (str(max(clock, int(last[0]) if last else 0)),),
                )
            for entity in changed:
                self._reindex(db, entity)
            for scope, owner in owners:
                # A batch of only tombstones is a forget replicating from a peer; refusing
                # it at the op cap would leave the body alive here (see _quota).
                self._quota(db, scope, owner, relief=(scope, owner) not in growing)
                if scope == "session":
                    directory = self._session_directory(owner, db)
                    if directory and directory.is_dir():
                        self._export(db, owner, directory)
            db.commit()
            if tombs:
                db.execute("INSERT INTO recall(recall) VALUES ('optimize')")
                db.commit()
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return len(changed)

    def import_session(self) -> None:
        if not self.session_dir or not self.identity.session_id:
            return
        path = self.session_dir / SESSION_EXPORT_NAME
        if not path.exists():
            return
        if path.is_symlink() or not path.is_file():
            raise MemoryError("Portable memory must be a regular file, not a symlink.")
        with path.open("rb") as stream:
            data = stream.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise MemoryError("Portable session memory exceeds its 16 MiB cap.")
        digest = hashlib.sha256(data).hexdigest()
        key = "exporthash:" + self.identity.session_id
        if self.path.exists():
            with self.connection() as db:
                known = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
                if known and known[0] == digest:
                    return
        ops = []
        owners = set()
        for line in data.decode("utf-8").splitlines():
            op = json.loads(line)
            validate_op(op, allow_redacted=True)
            if op["scope"] != "session":
                raise MemoryError("Session memory export contains a non-session operation.")
            owners.add(op["owner_key"])
            ops.append(op)
        if len(owners) > 1:
            raise MemoryError("Portable memory must belong to one session owner.")
        if not ops:
            return
        # Never commit/export a partial batch over the full truth file: a crash
        # between chunks would lose its remaining rows. One bounded portable
        # transaction also re-keys a promotion under this device's REAL origin,
        # then atomically exports canonical rows before the SQLite index commits.
        self.apply_ops(
            ops,
            contiguous=False,
            portable=True,
            promote=next(iter(owners)) != self.identity.session_id,
        )

    def _where(self, scope: str | None) -> tuple[str, list[str]]:
        pairs = [(scope, self.identity.owner(scope))] if scope else self.identity.scopes()
        return "(" + " OR ".join("(scope=? AND owner=?)" for _ in pairs) + ")", [
            v for pair in pairs for v in pair
        ]

    def resolve(self, target: str, scope: str | None = None) -> tuple[str, str]:
        """Map what a human copied (``scope/id``, a bare id or a unique prefix) to (scope, id).

        list/search print ``scope/id``, so every command must accept that form (design
        and UX round 1: D1/U2). Only AUTHORIZED owners are consulted, so a prefix can
        never reveal another session's or team's facts. A prefix needs >= 8 characters:
        shorter ones match too much to be a deliberate reference.
        """
        if not isinstance(target, str) or not target:
            raise MemoryError("An explicit memory ID is required.")
        if "/" in target:
            named, _, target = target.partition("/")
            if named not in SCOPES:
                raise MemoryError("Memory references look like SCOPE/ID, e.g. global/01a1…")
            if scope not in (None, named):
                raise MemoryError("The reference's scope differs from --scope.")
            scope = named
        safe_segment(target)
        if not self.path.exists():
            raise MemoryError("Memory not found in the authorized scope.")
        where, values = self._where(scope)
        with self.connection() as db:
            for table in ("current", "ops"):
                exact = db.execute(
                    f"SELECT DISTINCT scope FROM {table} WHERE {where} AND entity_id=?",
                    [*values, target],
                ).fetchall()
                if exact:
                    return exact[0][0], target
            if len(target) < 8:
                raise MemoryError("Memory not found in the authorized scope.")
            rows = db.execute(
                f"SELECT DISTINCT entity_id,scope FROM current WHERE {where} "
                "AND substr(entity_id,1,?)=? LIMIT 4",
                [*values, len(target), target],
            ).fetchall()
        if len(rows) > 1:
            # The query is capped, so a full page means "at least this many" — a
            # bare count would understate an ambiguity the caller must widen.
            count = "at least 4" if len(rows) == 4 else str(len(rows))
            raise MemoryError(
                f"'{target}' matches {count} memories; type more of the ID "
                "(or use the full SCOPE/ID shown by list)."
            )
        if not rows:
            raise MemoryError("Memory not found in the authorized scope.")
        return rows[0][1], rows[0][0]

    def related(self, scope: str, entity: str) -> list[dict[str, Any]]:
        owner = self.identity.owner(scope)
        self.show(scope, entity)
        with self.connection() as db:
            rows = db.execute(
                "SELECT DISTINCT c.entity_id,c.op_id,c.fact FROM associations a "
                "JOIN associations b ON a.locator=b.locator "
                "AND a.scope=b.scope AND a.owner=b.owner "
                "JOIN current c ON c.entity_id=b.entity "
                "WHERE a.entity=? AND a.scope=? AND a.owner=? AND b.entity<>? "
                "ORDER BY json_extract(c.fact,'$.created_at') DESC,c.entity_id DESC LIMIT 3",
                (entity, scope, owner, entity),
            ).fetchall()
            return [
                {
                    "id": row[0],
                    "version": row[1],
                    "title": json.loads(row[2])["title"],
                    "scope": scope,
                    "untrusted": True,
                }
                for row in rows
            ]

    def show(self, scope: str, entity: str, *, version: str | None = None) -> dict[str, Any]:
        owner = self.identity.owner(scope)
        safe_segment(entity)
        if not self.path.exists():
            raise MemoryError("Memory not found in the authorized scope.")
        with self.connection() as db:
            ops = self._ops(db, entity)
            if not ops or any(op["scope"] != scope or op["owner_key"] != owner for op in ops):
                raise MemoryError("Memory not found in the authorized scope.")
            tombs = [op for op in ops if op["kind"] == "tombstone"]
            heads = self._heads(tombs or ops)
            if version:
                safe_segment(version)
                selected = [op for op in ops if op["op_id"] == version]
                if not selected or tombs:
                    raise MemoryError("Revision unavailable or forgotten in this scope.")
                heads = selected + [op for op in heads if op["op_id"] != version]
            return {
                "id": entity,
                "version": heads[0]["op_id"],
                "scope": scope,
                "owner": owner,
                "status": "forgotten" if tombs else heads[0]["fact"]["status"],
                "fact": None if tombs else heads[0]["fact"],
                "conflicts": (
                    []
                    if tombs
                    else [
                        {
                            "version": op["op_id"],
                            "title": op["fact"]["title"],
                            "status": op["fact"]["status"],
                            # Origin + wall time (HLC ms) let a human decide which
                            # sibling to keep; the revision is what edit/forget need.
                            "origin": op["origin_device"],
                            "at": op["hlc"][0] / 1000.0,
                        }
                        for op in heads[1:4]
                    ]
                ),
                "revisions": len(ops),
                "moved_from": None if tombs else self._moved_from(db, entity, ops),
                "untrusted": True,
            }

    def _moved_from(
        self, db: sqlite3.Connection, entity: str, ops: list[dict[str, Any]]
    ) -> dict[str, str] | None:
        """The source of a moved fact, from the cross-entity parent ``move`` writes.

        Derived from the entity's ROOT revision, not its head: an ordinary
        correction parents only the same entity, so after the first edit the head
        no longer names the move and the receipt's promise would silently vanish.
        An ordinary creation has no parents, and a ``replaces`` creation's parent
        is a supersession link (not a move), so this cannot mislabel either.
        """
        revisions = [op for op in ops if op["kind"] == "put"]
        ids = {op["op_id"] for op in revisions}
        # A root revision has no SAME-ENTITY parent; a correction does.
        roots = sorted(
            (op for op in revisions if not any(parent in ids for parent in op["parents"])),
            key=lambda op: (op["hlc"], op["op_id"]),
        )
        if not roots:
            return None
        root_fact = roots[0]["fact"] or {}
        replaced = root_fact.get("replaces") or ""
        for parent in roots[0]["parents"]:
            row = db.execute("SELECT scope,entity_id FROM ops WHERE op_id=?", (parent,)).fetchone()
            if row and row[1] != entity and row[1] != replaced:
                return {"scope": row[0], "id": row[1]}
        return None

    def _move_tombstone_conflict(
        self,
        db: sqlite3.Connection,
        op: dict[str, Any],
        old_ops: list[dict[str, Any]],
    ) -> str | None:
        """Refuse a MOVE's source tombstone when it would erase an unseen revision.

        ``move`` names its destination ``put`` as a second parent of the source
        tombstone; a forget tombstone's parents are all same-entity revisions. That
        is the protocol-safe marker: an old peer still reads the op as an ordinary
        tombstone, while a peer that holds a concurrent correction can refuse
        instead of redacting it. A move is advertised as a relocation, so losing an
        edit the mover never saw is not a documented forget.

        The destination op may be absent here when this peer's ``network.memory.scopes``
        filters out the destination scope; the second, unresolvable parent still
        identifies the op as move-shaped, and coverage comes from the same-entity
        parents the mover DID send — an unseen sibling is exactly a local head that
        is not among them.

        Deliberately returns a refusal only for that marker: a real forget must
        still replicate and redact unconditionally (a tombstone is the only way to
        remove a sensitive body). An entity already tombstoned locally is allowed
        through — there is no live revision left to protect.
        """
        if any(existing["kind"] == "tombstone" for existing in old_ops):
            return None
        local = {existing["op_id"] for existing in old_ops}
        covered = {parent for parent in op["parents"] if parent in local}
        unresolved = [parent for parent in op["parents"] if parent not in local]
        if not unresolved:
            return None  # an ordinary forget: every parent is a local revision
        move = False
        for parent in unresolved:
            row = db.execute("SELECT data FROM ops WHERE op_id=?", (parent,)).fetchone()
            if not row:
                continue
            foreign = json.loads(row[0])
            if foreign["entity_id"] != op["entity_id"] and foreign["kind"] == "put":
                move = True
        if not move and len(op["parents"]) > 1 and covered:
            # The destination scope was filtered out of this peer's sync config, so
            # the destination op never arrived. A forget's parents are ALL local by
            # construction; a second, unresolvable parent is move-shaped.
            move = True
        if not move:
            return None
        unseen = [head for head in self._heads(old_ops) if head["op_id"] not in covered]
        if not unseen:
            return None
        return (
            "Memory move refused: a move from another device would replace a revision "
            "this device holds but the moving device never saw. Read that revision and "
            "keep anything it adds (or forget the source fact). Memory sync retries "
            "automatically after about 30 minutes; restarting this device picks the retry up "
            "sooner."
        )

    def inventory(
        self, scope: str | None = None, *, limit: int = 20, cursor: str | None = None
    ) -> dict[str, Any]:
        where, values = self._where(scope)
        if cursor:
            safe_segment(cursor)
        if not 1 <= limit <= 50:
            raise MemoryError("Inventory limit must be 1–50.")
        if not self.path.exists():
            return {
                "items": [],
                "help": INVENTORY_HELP,
                "available_scopes": [name for name, _ in self.identity.scopes()],
                "active_tokens": 0,
                "ops": 0,
                "quotas": [],
                "storage_bytes": 0,
                "storage_cap_bytes": (MAX_DB_BYTES // CAP_PAGE_SIZE) * CAP_PAGE_SIZE,
                "storage_percent": 0.0,
                "storage_free_bytes": 0,
                "storage_full": False,
                "database_path": str(self.path),
                "limits": {
                    "global_tokens": GLOBAL_TOKENS,
                    "owner_tokens": OWNER_TOKENS,
                    "item_tokens": ITEM_TOKENS,
                },
            }
        with self.connection() as db:
            page_where = where + (" AND entity_id<?" if cursor else "")
            page_values = [*values, cursor] if cursor else values
            rows = db.execute(
                f"SELECT entity_id,scope,op_id,fact FROM current WHERE {page_where} "
                "ORDER BY entity_id DESC LIMIT ?",
                [*page_values, limit],
            ).fetchall()
            usage = db.execute(
                f"SELECT coalesce(sum(tokens),0) FROM current WHERE {where}", values
            ).fetchone()[0]
            count = db.execute(f"SELECT count(*) FROM ops WHERE {where}", values).fetchone()[0]
            pairs = [(scope, self.identity.owner(scope))] if scope else self.identity.scopes()
            quotas = []
            for name, owner in pairs:
                used = db.execute(
                    "SELECT coalesce(sum(tokens),0) FROM current WHERE scope=? AND owner=?",
                    (name, owner),
                ).fetchone()[0]
                cap = self.token_cap(name)
                quotas.append(
                    {
                        "scope": name,
                        "owner": owner,
                        "used_tokens": used,
                        "cap_tokens": cap,
                        "percent": round(100.0 * used / cap, 1) if cap else 0.0,
                        "headroom_tokens": cap - used,
                    }
                )
            return {
                "items": [
                    {
                        "id": row[0],
                        "scope": row[1],
                        "version": row[2],
                        "title": json.loads(row[3])["title"],
                        "status": json.loads(row[3])["status"],
                        "origin": origin_label(json.loads(row[3])),
                    }
                    for row in rows
                ],
                "active_tokens": usage,
                "ops": count,
                "quotas": quotas,
                "next_cursor": rows[-1][0] if len(rows) == limit else None,
                **self._storage_view(db),
                "limits": {
                    "global_tokens": GLOBAL_TOKENS,
                    "owner_tokens": OWNER_TOKENS,
                    "item_tokens": ITEM_TOKENS,
                    "owner_ops": MAX_OWNER_OPS,
                    "revisions": MAX_REVISIONS,
                    "storage_bytes": MAX_DB_BYTES,
                },
                "help": INVENTORY_HELP,
                "available_scopes": [name for name, _ in self.identity.scopes()],
            }

    def scan(self, scope: str | None = None) -> list[dict[str, Any]]:
        """Every current fact in the authorised scope(s), flattened for a preview.

        WHY not ``search()``: a maintenance preview must see the WHOLE scope —
        including facts the FTS ranker would not return together and rows whose
        status is candidate/disputed/expired, which are precisely the ones a
        cleanup is asked to retire. Read-only by construction: it checks for the
        file before ``connection()`` (which would create an empty store) and runs
        one SELECT, so ``consolidate`` can promise it changed nothing.
        """
        where, values = self._where(scope)
        if not self.path.exists():
            return []
        with self.connection() as db:
            rows = db.execute(
                f"SELECT entity_id,scope,op_id,fact,tokens FROM current WHERE {where} "
                "ORDER BY entity_id DESC",
                values,
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            fact = json.loads(row[3])
            items.append(
                {
                    "id": row[0],
                    "scope": row[1],
                    "version": row[2],
                    "title": fact["title"],
                    "body": fact["body"],
                    "tags": fact["tags"],
                    "status": fact["status"],
                    "created_at": fact["created_at"],
                    "verified_at": fact["verified_at"],
                    "tokens": row[4],
                }
            )
        return items

    def select(
        self,
        scope: str,
        *,
        older_than: float | None = None,
        unverified: bool = False,
        tags: list[str] | None = None,
        title_matches: str | None = None,
        body_matches: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Predicate selection over one scope for a bulk-forget preview.

        Composable: predicate KINDS are ANDed, repeated ``tags`` are ANY-of.
        Regexes are case-insensitive and length-capped; a group repeated around an
        unbounded repeat (``(a+)+``) is refused before compilation because Python's
        ``re`` has no timeout and that shape backtracks exponentially. The check is
        a lexical heuristic, not a proof (see ``_nested_unbounded_repeat``), so the
        contract is "recognised catastrophic shapes are refused", never "every
        accepted pattern is bounded". Read-only: the caller prints the result and
        hands the returned ``(id, version)`` pairs to :meth:`forget_many`, which
        re-checks every revision at commit time.
        """
        patterns: dict[str, re.Pattern[str]] = {}
        for name, value in (("title", title_matches), ("body", body_matches)):
            if value:
                if len(value) > 500:
                    raise MemoryError(f"--{name}-matches is limited to 500 characters.")
                if _nested_unbounded_repeat(value):
                    raise MemoryError(
                        f"--{name}-matches repeats a group that itself contains an "
                        "unbounded repeat (for example '(a+)+'); that shape can backtrack "
                        "exponentially, so it is refused rather than run. Rewrite it "
                        "without nesting repetitions."
                    )
                try:
                    patterns[name] = re.compile(value, re.IGNORECASE)
                except re.error:
                    # Never echo the pattern: a half-typed value may be a secret.
                    raise MemoryError(
                        f"--{name}-matches is not a valid regular expression."
                    ) from None
        current = time.time() if now is None else now
        cutoff = None
        if older_than is not None:
            if older_than <= 0:
                raise MemoryError("--older-than needs a positive duration.")
            cutoff = current - older_than
        wanted = {tag.lower() for tag in (tags or [])}
        rows = self.scan(scope)
        matched: list[dict[str, Any]] = []
        for row in rows:
            created = row["created_at"]
            # A fact with no retention timestamp cannot be shown to be old; a
            # time filter must not quietly treat it as ancient.
            if cutoff is not None and not (isinstance(created, (int, float)) and created <= cutoff):
                continue
            if unverified and row["verified_at"] is not None:
                continue
            if wanted and not wanted & {tag.lower() for tag in row["tags"]}:
                continue
            if "title" in patterns and not patterns["title"].search(row["title"]):
                continue
            if "body" in patterns and not patterns["body"].search(row["body"]):
                continue
            matched.append(row)
        return {
            "scope": scope,
            "items": matched,
            "total": len(rows),
            "tokens": sum(row["tokens"] for row in matched),
            "total_tokens": sum(row["tokens"] for row in rows),
        }

    def tagged(self, tag: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Current facts carrying ``tag``, NEWEST CREATED first, bounded by ``limit``.

        A read pure and simple: no write, no index, no second store. It exists
        because recall ranks by relevance, which is the wrong order for a caller
        that wants the most recent decisions, and because ``scan`` has no date
        order at all.

        **EXPLICIT replacement is honoured; content-based supersession is not.**
        A fact replaced through the documented ``replaces`` edge is hidden,
        exactly as :meth:`search` hides it — both use
        :data:`_NOT_REPLACED_SQL`, so the two readers cannot develop two
        definitions of "replaced". What the store does NOT do is DERIVE that
        two facts say the same thing: content is never compared, so a
        contradicted decision with no explicit edge may still appear. Callers
        must label the rest "most recently written", never "in force".

        Also excluded, because the store already treats them as settled: a past
        ``expires_at``, a stored ``expired`` status, and a stored
        ``superseded`` status. That is status the store was TOLD, not an
        inference about content.
        """
        if not isinstance(tag, str) or not tag or len(tag) > 64:
            raise MemoryError("A memory tag is a short non-empty string.")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise MemoryError("Tagged reads accept 1-100 results.")
        if not self.path.exists():
            return []
        where, values = self._where(None)
        with self.connection() as db:
            rows = db.execute(
                f"SELECT c.entity_id,c.scope,c.fact FROM current c WHERE {where} "
                "AND EXISTS (SELECT 1 FROM json_each(json_extract(c.fact,'$.tags')) "
                "AS tag_row WHERE tag_row.value=?) "
                f"AND {_NOT_REPLACED_SQL} "
                "AND (json_extract(c.fact,'$.expires_at') IS NULL "
                "OR json_extract(c.fact,'$.expires_at')>?) "
                "AND json_extract(c.fact,'$.status') NOT IN ('expired','superseded') "
                "ORDER BY json_extract(c.fact,'$.created_at') DESC, c.entity_id DESC "
                "LIMIT ?",
                [*values, tag, time.time(), limit],
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            fact = json.loads(row[2])
            out.append(
                {
                    "id": row[0],
                    "scope": row[1],
                    "body": fact.get("body", ""),
                    "tags": list(fact.get("tags") or ()),
                    "created_at": fact.get("created_at"),
                    "status": fact.get("status"),
                }
            )
        return out

    def search(
        self,
        query: str,
        scope: str | None = None,
        *,
        tokens: int = RECALL_TOKENS,
        limit: int = 10,
        suggest: bool = False,
    ) -> list[dict[str, Any]]:
        # ``guard`` is the free-text STORE guard: its own message asks for the value to
        # be kept as a secret NAME. A recall query is a READ and may legitimately BE a
        # minted structural reference (see structural_reference), so running the store
        # guard over it refused a benign identifier lookup with store-oriented advice.
        # Prose takes the guard below, exactly as before. Nothing is weakened off this
        # call site: search() persists nothing, and the journal/provider-payload copy of
        # the same arguments is guarded independently by memory_privacy.retained_arguments,
        # which fails closed — an admitted query spelling is still never retained.
        if not structural_reference(query):
            guard(query)
        if (
            not isinstance(query, str)
            or len(query) > 2048
            or type(tokens) is not int
            or not 1 <= tokens <= RECALL_HARD_TOKENS
            or not 1 <= limit <= 20
        ):
            raise MemoryError("Search accepts ≤2048 characters, 1–20 results and 1–2000 tokens.")
        where, values = self._where(scope)
        words = [t.lower() for t in re.findall(r"[\w]+", query) if len(t) >= 2]
        terms = list(dict.fromkeys(t for t in words if t not in STOP_WORDS))[:24]
        if not self.path.exists():
            return []
        if words and not suggest:
            # A stop-listed word is dropped from the query, which made a 2-letter alias such
            # as "IS" unfindable (QA Q3). Keep stop words only when they exactly match an
            # indexed tag/alias in an authorised scope; explicit search only, so the
            # suggestion path keeps its generic-word suppression.
            where_alias, alias_values = self._where(scope)
            with self.connection() as db:
                for word in dict.fromkeys(w for w in words if w in STOP_WORDS):
                    hit = db.execute(
                        "SELECT 1 FROM recall JOIN current c ON c.entity_id=recall.entity_id "
                        "WHERE recall MATCH ? AND "
                        + where_alias.replace("scope", "c.scope").replace("owner", "c.owner")
                        + " LIMIT 1",
                        ["tags:" + '"' + word + '"', *alias_values],
                    ).fetchone()
                    if hit and word not in terms and len(terms) < 24:
                        terms.append(word)
        if not terms:
            return []
        with self.connection() as db:
            match = " OR ".join('"' + term + '"' for term in terms)
            rows = db.execute(
                "SELECT c.entity_id,c.scope,c.op_id,c.fact FROM recall JOIN current c "
                "ON c.entity_id=recall.entity_id WHERE recall MATCH ? AND "
                f"{where} AND {_NOT_REPLACED_SQL} "
                "AND (json_extract(c.fact,'$.expires_at') IS NULL "
                "OR json_extract(c.fact,'$.expires_at')>?) "
                "AND (?=0 OR json_extract(c.fact,'$.status')<>'disputed') "
                "ORDER BY (json_extract(c.fact,'$.status')='candidate'), "
                "bm25(recall,0,4,2,1),json_extract(c.fact,'$.created_at') DESC, "
                "c.entity_id DESC LIMIT 10",
                [match, *values, time.time(), int(suggest)],
            ).fetchall()
            result: list[dict[str, Any]] = []
            used = 0
            # The suggestion budget is the WHOLE emitted block, matching
            # ``suggestion_block`` byte for byte: header and closing are fixed
            # costs the renderer pays, so admitting lines up to the full budget
            # over-commits by the 19-token wrapper and the tail's own exact
            # check then drops the last admitted line.
            block = ""
            # One slot per distinct FACT on the suggestion path, keyed by body.
            # Two rows carrying the same body are one fact and must charge the
            # budget once, or they crowd out a distinct matching fact (measured:
            # 2 entries / 1 fact, the third relevant fact never rendered).
            seen: set[str] = set()
            # Document frequency is indexed, not an O(bank) Python pass. Ask FTS
            # itself for per-term matches so porter stemming cannot disagree.
            authorized_count = db.execute(
                f"SELECT count(*) FROM current WHERE {where}", values
            ).fetchone()[0]
            for row in rows:
                fact = json.loads(row[3])
                if fact["expires_at"] is not None and fact["expires_at"] <= time.time():
                    continue
                if suggest and fact["status"] == "disputed":
                    continue
                if suggest:
                    hits = []
                    specific = False
                    for term in terms:
                        hit = db.execute(
                            "SELECT 1 FROM recall WHERE entity_id=? AND recall MATCH ?",
                            (row[0], '"' + term + '"'),
                        ).fetchone()
                        if hit:
                            hits.append(term)
                            freq = db.execute(
                                "SELECT count(*) FROM recall JOIN current c "
                                "ON c.entity_id=recall.entity_id WHERE recall MATCH ? "
                                f"AND {where}",
                                ['"' + term + '"', *values],
                            ).fetchone()[0]
                            specific |= len(term) >= 4 and freq <= max(2, authorized_count * 0.01)
                    coverage = len(hits) / len(terms)
                    if not ((len(hits) >= 2 and coverage >= 0.4) or (specific and coverage >= 0.5)):
                        continue
                    key = _suggestion_key(fact)
                    if key in seen:
                        continue
                    seen.add(key)
                item = {
                    "id": row[0],
                    "scope": row[1],
                    "version": row[2],
                    "title": fact["title"],
                    "status": fact["status"],
                    "origin": origin_label(fact),
                    "untrusted": True,
                }
                if not suggest:
                    item.update(
                        body=fact["body"],
                        source=fact["source"],
                        verified_at=fact["verified_at"],
                        expires_at=fact["expires_at"],
                    )
                # The suggestion budget is a WIRE budget: charge the block the
                # tail actually renders, not the full candidate record.
                # ``_dump(item)`` is ~3x the rendered line, which silently made
                # ``tokens=180, limit=3`` a 2-entry cap on every fixture; and a
                # per-line charge without the wrapper did the same to long
                # titles. An explicit search keeps the record-shaped charge,
                # since that is what a caller receives there.
                if suggest:
                    line = suggest_line(item)
                    if count_text_tokens(suggestion_block(block + line)) <= tokens:
                        result.append(item)
                        block += line
                else:
                    cost = count_text_tokens(_dump(item))
                    if used + cost <= tokens:
                        result.append(item)
                        used += cost
                if len(result) >= min(limit, 3 if suggest else limit):
                    break
            return result


def parse_url(url: str) -> tuple[str | None, str | None, dict[str, str]]:
    parts = urlsplit(url)
    if (
        not url[:9].lower() == "memory://"
        or parts.scheme != "memory"
        or parts.fragment
        or parts.username
        or parts.password
        or parts.port
    ):
        raise MemoryError(
            "Use memory://, memory://session, or memory://session/ID; owners are never addresses."
        )
    scope = parts.netloc or None
    entity = parts.path.lstrip("/") or None
    if scope is not None and scope not in SCOPES:
        raise MemoryError("Unknown memory scope; never guess another session/team owner.")
    if entity:
        safe_segment(entity)
    query = parse_qs(parts.query, strict_parsing=True)
    if set(query) - {"q", "tokens", "limit", "related", "version", "cursor"} or any(
        len(v) != 1 for v in query.values()
    ):
        raise MemoryError("Memory read supports q, tokens and limit only.")
    return scope, entity, {k: v[0] for k, v in query.items()}
