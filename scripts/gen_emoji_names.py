#!/usr/bin/env python3
"""Regenerate ``local_operator/team_bridge/emoji_names.json``.

WHY A CHECKED-IN, GENERATED TABLE. Agents write Slack/GitHub-style shortcodes
(``white_check_mark``, ``+1``, ``:tada:``) for a reaction, but the hub stores a
reaction key only as ONE fully-qualified RGI emoji sequence (agent-server
``internal/bridge/emoji``) and refuses anything else as ``not_an_emoji``. The
runtime therefore resolves a shortcode to that canonical sequence before the
request leaves, so an agent's ✅ and a human's ✅ are the same key and stack.
The table is generated, never edited, so a source upgrade is one reviewed diff
whose pinned count ``tests/unit/team_bridge/test_emoji_names.py`` forces the
author to restate. No PyPI ``emoji`` dependency: the alias layer is small and
the hub stays the only authority on "is this RGI".

SOURCES (both pinned; inputs are fetched by hand, the build never touches the
network, and a sha256 mismatch refuses to generate):

* GitHub gemoji ``db/emoji.json`` at tag v4.1.0 (MIT). Its ``aliases`` are the
  GitHub/Slack shortcodes. They WIN over CLDR names on any collision.
  https://raw.githubusercontent.com/github/gemoji/v4.1.0/db/emoji.json
* Unicode ``emoji-test.txt`` 18.0 — the SAME file agent-server's generator
  pins (``rgi_emoji.txt`` header sha256 below). It supplies the CLDR short
  names (snake_cased: ``check mark button`` -> ``check_mark_button``), the
  fully-qualified set every value must belong to, and the unqualified /
  minimally-qualified -> fully-qualified map (``❤`` -> ``❤️``).
  https://www.unicode.org/Public/18.0.0/emoji/emoji-test.txt

Usage (from the repo root)::

    curl -sSfLo gemoji.json https://raw.githubusercontent.com/github/gemoji/v4.1.0/db/emoji.json
    curl -sSfLo emoji-test.txt https://www.unicode.org/Public/18.0.0/emoji/emoji-test.txt
    python scripts/gen_emoji_names.py --gemoji gemoji.json --emoji-test emoji-test.txt

KEY GRAMMAR (must match ``emoji_names.shortcode_key``): lower-case, and ``-``
and space fold to ``_``. The generator refuses to write if that fold makes two
DIFFERENT emoji share a key within gemoji; a CLDR name whose folded key is
ambiguous across CLDR rows is dropped rather than guessed. Skin-tone variants
(any sequence holding U+1F3FB..U+1F3FF) are skipped: Slack's ``::skin-tone-N:``
suffix grammar is a non-goal, and the rows would triple the table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

GEMOJI_TAG = "v4.1.0"
GEMOJI_URL = f"https://raw.githubusercontent.com/github/gemoji/{GEMOJI_TAG}/db/emoji.json"
GEMOJI_SHA256 = "b174ae2aeb321b52f64adb9ff412f966a7f338839d780784dd15dcad702c2dd6"
EMOJI_TEST_VERSION = "18.0"
EMOJI_TEST_URL = "https://www.unicode.org/Public/18.0.0/emoji/emoji-test.txt"
EMOJI_TEST_SHA256 = "8f3735cda1f92a779d78af67cf86066bb1f07143dc22f2ac29394d9bc57ab21a"

OUT = Path(__file__).resolve().parents[1] / "local_operator/team_bridge/emoji_names.json"
_SKIN_TONES = {chr(c) for c in range(0x1F3FB, 0x1F400)}
_ROW = re.compile(r"^([0-9A-F ]+?)\s*;\s*([a-z-]+)\s*#\s*\S+\s+E\d+\.\d+\s+(.+)$")


def shortcode_key(raw: str) -> str:
    # Mirrors local_operator.team_bridge.emoji_names.shortcode_key; duplicated
    # here only so the generator runs without importing the package.
    return re.sub(r"[-\s]+", "_", raw.strip().strip(":").strip().lower())


def cldr_key(name: str) -> str:
    # "flag: Côte d’Ivoire" -> "flag_cote_d_ivoire"; "keycap: #" -> "keycap" is
    # rejected below because the punctuation carried the meaning.
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    ascii_name = ascii_name.replace("#", " number sign ").replace("*", " asterisk ")
    return re.sub(r"[^a-z0-9+]+", "_", ascii_name.lower()).strip("_")


def _sha(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != expected:
        sys.exit(f"{path}: sha256 {got} != pinned {expected}")
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemoji", type=Path, required=True)
    ap.add_argument("--emoji-test", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    test_text = _sha(args.emoji_test, EMOJI_TEST_SHA256).decode("utf-8")
    gemoji = json.loads(_sha(args.gemoji, GEMOJI_SHA256))

    fully: set[str] = set()
    qualify: dict[str, str] = {}
    cldr: list[tuple[str, str]] = []
    last_fully_by_name: dict[str, str] = {}
    for line in test_text.splitlines():
        m = _ROW.match(line)
        if m is None:
            continue
        seq = "".join(chr(int(cp, 16)) for cp in m.group(1).split())
        status, name = m.group(2), m.group(3).strip()
        if status == "fully-qualified":
            fully.add(seq)
            last_fully_by_name[name] = seq
            cldr.append((name, seq))
        elif status in {"unqualified", "minimally-qualified"}:
            # emoji-test lists these right after their fully-qualified row,
            # under the same CLDR name.
            qualify[seq] = last_fully_by_name[name]
    if len(fully) < 3000:
        sys.exit(f"emoji-test parsed only {len(fully)} fully-qualified rows")

    names: dict[str, str] = {}
    for entry in gemoji:
        seq = entry["emoji"]
        seq = qualify.get(seq, seq)
        if seq not in fully:
            sys.exit(f"gemoji {entry['aliases']} -> {seq!r} is not fully-qualified RGI")
        for alias in entry["aliases"]:
            key = shortcode_key(alias)
            if names.get(key, seq) != seq:
                sys.exit(f"gemoji alias key collision on {key!r}")
            names[key] = seq

    cldr_keys: dict[str, set[str]] = {}
    for name, seq in cldr:
        if _SKIN_TONES & set(seq):
            continue
        key = cldr_key(name)
        if key:
            cldr_keys.setdefault(key, set()).add(seq)
    dropped = 0
    for key, seqs in cldr_keys.items():
        if key in names:
            continue  # gemoji wins
        if len(seqs) != 1:
            dropped += 1
            continue
        names[key] = next(iter(seqs))

    doc = {
        "_generated": "scripts/gen_emoji_names.py — DO NOT EDIT",
        "_sources": {
            "gemoji": {"url": GEMOJI_URL, "tag": GEMOJI_TAG, "sha256": GEMOJI_SHA256},
            "emoji_test": {
                "url": EMOJI_TEST_URL,
                "version": EMOJI_TEST_VERSION,
                "sha256": EMOJI_TEST_SHA256,
            },
        },
        "names": dict(sorted(names.items())),
        "qualify": dict(sorted(qualify.items())),
    }
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(
        f"wrote {len(names)} names, {len(qualify)} qualify rows "
        f"({dropped} ambiguous CLDR keys dropped) to {args.out}"
    )


if __name__ == "__main__":
    main()
