"""Residue admission: knowledge in EVERY script becomes a fact, never ``ok``/0.

Why this file exists
--------------------
``_substantive`` decides whether a capture candidate still says anything after
its locator tokens are removed. It first shipped with an ASCII-only word rule
(``[A-Za-z]{3,}``), so a CJK, Japanese or Cyrillic decision left a non-empty
residue the rule could not match: the section was dropped as an admission skip
while the pass was still recorded as a success with ZERO facts. Nothing failed
loudly, and no test asserted that non-ASCII knowledge produces facts, so the
defect survived four review/QA rounds.

These tests pin the observable behaviour rather than the pattern: real
knowledge in each script MUST yield a fact row, while locator-only and
number-only bodies MUST still yield none. Restoring the ASCII-only rule makes
the CJK, pure-Cyrillic, Japanese and mixed-script cases below fail (the other
Cyrillic line contains the ASCII word ``SQLite``, so it passes either rule and
is kept only as coverage); the negative cases stop the suite from passing by
admitting everything.
"""

import pytest

from local_operator.memory_capture import _substantive, extract

# Real decisions in the scripts the ASCII-only rule silently dropped. The
# mixed-script case deliberately has NO ASCII word to fall back on (the version
# is Cyrillic + digits), so it cannot pass unless non-Latin letters are matched.
KNOWLEDGE = {
    "cjk": "我们决定采用方案二",
    "cyrillic": "Мы решили использовать SQLite",
    # Pure Cyrillic: the line above contains the ASCII word "SQLite", which an
    # ASCII-only rule still matches. This one has no Latin fallback, so it is
    # the case that pins the Cyrillic range specifically.
    "cyrillic-only": "Мы решили перейти на другую базу данных",
    "japanese": "決定しました",
    "mixed-script": "我们决定采用方案二，версия 2",
}

# Only locators or numbers: still not knowledge, in any script. A CJK filename
# is a locator like any other and must not become a fact.
NOT_KNOWLEDGE = {
    "ascii-path": "~/tools/widget/docs/proof.md",
    "cjk-path": "~/文档/方案二.md",
    "uri": "https://example.invalid/方案二",
    "version": "3.2.1",
    "numbers": "1234 5678",
}


def _section(body: str) -> str:
    """A decision heading with one body: the shape capture actually reads."""
    return f"## Key Decisions\n- {body}\n"


@pytest.mark.parametrize("body", KNOWLEDGE.values(), ids=KNOWLEDGE.keys())
def test_knowledge_in_any_script_is_substantive(body: str) -> None:
    assert _substantive(body) is True


@pytest.mark.parametrize("body", KNOWLEDGE.values(), ids=KNOWLEDGE.keys())
def test_knowledge_in_any_script_produces_facts(body: str) -> None:
    rows = extract(_section(body))
    assert [row["body"] for row in rows] == [body]
    # The defect's shape: the section was SKIPPED and the pass still reported ok.
    assert rows.skipped == 0


@pytest.mark.parametrize("body", NOT_KNOWLEDGE.values(), ids=NOT_KNOWLEDGE.keys())
def test_locator_or_number_bodies_produce_no_facts(body: str) -> None:
    assert _substantive(body) is False
    rows = extract(_section(body))
    assert rows == []
    assert rows.skipped == 1
