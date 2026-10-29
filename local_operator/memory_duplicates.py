"""Read-only lexical duplicate detection for the human ``lop memory consolidate`` preview.

WHY lexical only: consolidation is a review aid, and an embedding/model call would
make a read-only maintenance command depend on a provider, a network and a model
version — none of which this store can promise at maintenance time. The evidence a
reviewer needs is already in the text, so the whole measure runs locally over token
sets: no new dependency, no model import, no store access, and nothing here writes.

Two facts are candidates when their normalised titles are identical, or when their
title+body token sets overlap by at least ``SIMILARITY_THRESHOLD`` (Jaccard). Only
SHARED text is evidence; the pairs are closed transitively into groups so two exact
copies plus a near copy read as one decision, and every pair carries the measure and
a sample of the tokens that differ so a reviewer can judge without opening both facts.

Bounds are deliberate. Candidate generation uses an inverted token index restricted
to tokens present in at most ``MAX_BLOCK_DF`` facts: a word shared by half the store
would emit a quadratic number of pairs while separating nothing, and a genuine
near-duplicate also shares rarer words. Facts are already capped at ``ITEM_TOKENS``
by the store, so no per-fact token cap is needed.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from local_operator.memory import STOP_WORDS

#: Combined-title/body token overlap at which two facts are called near-duplicates.
#: 0.7 is deliberately on the conservative side: a false positive costs a reviewer
#: one `show`, while a threshold low enough to group merely-related facts would
#: flood the preview and teach people to ignore it.
SIMILARITY_THRESHOLD = 0.7

#: Highest document frequency a token may have and still GENERATE candidate pairs.
#: Not a similarity input: the Jaccard measure still counts every shared token.
MAX_BLOCK_DF = 100

#: Evidence is bounded so one long fact cannot push the measure out of the frame.
SHARED_SAMPLE = 8
DIFF_SAMPLE = 6
MAX_PAIRS_PER_GROUP = 4


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normal(text: str) -> str:
    """Whitespace/punctuation-insensitive form used only for exact-title equality."""
    return " ".join(_words(text))


def _title_key(title: str) -> str:
    """Normalised title, but only when it carries some alphanumeric content.

    A punctuation-only title normalises to empty, so two of them compared equal —
    QA round 1 measured a 254-member "identical" group of exactly that shape. An
    empty key is not evidence that two facts name the same subject.
    """
    key = _normal(title)
    return key if any(char.isalnum() for char in key) else ""


def _tokens(row: dict[str, Any]) -> set[str]:
    return {
        word
        for word in _words(row["title"] + " " + row["body"])
        if len(word) >= 2 and word not in STOP_WORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _pair(left: dict[str, Any], right: dict[str, Any], a: set[str], b: set[str]) -> dict[str, Any]:
    shared = a & b
    a_only = a - b
    b_only = b - a
    left_title = _title_key(left["title"])
    same_title = bool(left_title) and left_title == _title_key(right["title"])
    exact = same_title and _normal(left["body"]) == _normal(right["body"])
    return {
        "a": left["id"],
        "b": right["id"],
        "title_equal": same_title,
        "exact": exact,
        "similarity": 1.0 if exact else round(_jaccard(a, b), 2),
        "shared": sorted(shared)[:SHARED_SAMPLE],
        "a_only": sorted(a_only)[:DIFF_SAMPLE],
        "b_only": sorted(b_only)[:DIFF_SAMPLE],
        "a_only_count": len(a_only),
        "b_only_count": len(b_only),
    }


def _connected(count: int, pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Transitive closure over candidate pairs: near copies of one fact form ONE group."""
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in pairs:
        root, other = find(left), find(right)
        if root != other:
            parent[other] = root
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return [members for members in groups.values() if len(members) > 1]


def _scope_groups(scope: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens = [_tokens(row) for row in rows]
    edges: list[tuple[int, int]] = []
    evidence: dict[tuple[int, int], dict[str, Any]] = {}

    def connect(left: int, right: int) -> None:
        key = (left, right) if left < right else (right, left)
        if key in evidence:
            return
        pair = _pair(rows[key[0]], rows[key[1]], tokens[key[0]], tokens[key[1]])
        # Identical titles are a candidate on their own: bodies often drift in one
        # line while the fact still names the same subject, and a combined-text
        # Jaccard would hide that behind a long body.
        if not tokens[key[0]] and not tokens[key[1]]:
            # Two facts with no alphanumeric token have no lexical evidence to
            # compare (two empty sets score Jaccard 1.0); that was the other half
            # of the punctuation-only false group.
            return
        if (
            not pair["exact"]
            and not pair["title_equal"]
            and pair["similarity"] < SIMILARITY_THRESHOLD
        ):
            return
        evidence[key] = pair
        edges.append(key)

    # Exact duplicates first: their identical text is the strongest evidence and
    # costs one dictionary pass.
    exact_buckets: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (_normal(row["title"]), _normal(row["body"]))
        exact_buckets.setdefault(key, []).append(index)
    for bucket in exact_buckets.values():
        for index in bucket[1:]:
            connect(bucket[0], index)

    # Near duplicates via an inverted index. A pair is compared once, and only
    # when it shares at least one BLOCKING token (see MAX_BLOCK_DF).
    postings: dict[str, list[int]] = {}
    for index, words in enumerate(tokens):
        for word in words:
            postings.setdefault(word, []).append(index)
    compared: set[tuple[int, int]] = set()
    for word, indexes in postings.items():
        if len(indexes) > MAX_BLOCK_DF:
            continue
        for position, left in enumerate(indexes):
            for right in indexes[position + 1 :]:
                key = (left, right) if left < right else (right, left)
                if key in compared:
                    continue
                compared.add(key)
                connect(*key)

    groups: list[dict[str, Any]] = []
    for members in _connected(len(rows), edges):
        member_set = set(members)
        pairs = [pair for key, pair in evidence.items() if key[0] in member_set]
        pairs.sort(key=lambda pair: (-pair["similarity"], pair["a"], pair["b"]))
        groups.append(
            {
                "scope": scope,
                "members": [
                    {
                        "id": rows[index]["id"],
                        "version": rows[index]["version"],
                        "title": rows[index]["title"],
                        "tokens": rows[index]["tokens"],
                        "status": rows[index]["status"],
                    }
                    for index in sorted(members, key=lambda i: (rows[i]["title"], rows[i]["id"]))
                ],
                "pairs": pairs[:MAX_PAIRS_PER_GROUP],
                "pair_count": len(pairs),
                "exact": all(pair["exact"] for pair in pairs),
                "similarity": max(pair["similarity"] for pair in pairs),
            }
        )
    # A large group first, then the strongest measure, then a stable tiebreak so
    # repeated runs on an unchanged store print the same page.
    groups.sort(
        key=lambda group: (
            -len(group["members"]),
            -group["similarity"],
            group["members"][0]["id"],
        )
    )
    return groups


def duplicates(facts: Iterable[dict[str, Any]], *, limit: int = 10) -> dict[str, Any]:
    """Group near-duplicates within each scope; never mutates ``facts`` or the store."""
    if not 1 <= limit <= 50:
        raise ValueError("Duplicate preview limit must be 1–50.")
    rows = list(facts)
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scope.setdefault(row["scope"], []).append(row)
    groups: list[dict[str, Any]] = []
    for scope in sorted(by_scope):
        groups.extend(_scope_groups(scope, by_scope[scope]))
    groups.sort(
        key=lambda group: (
            -len(group["members"]),
            -group["similarity"],
            group["members"][0]["id"],
        )
    )
    return {
        "groups": groups[:limit],
        "group_count": len(groups),
        "limited": len(groups) > limit,
        "fact_count": len(rows),
    }
