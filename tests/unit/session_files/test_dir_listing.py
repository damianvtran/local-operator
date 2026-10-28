"""Assistant directory listings: the deliverables idiom an agent closes a turn with.

The two positive texts are the operator's own (session 519d73620f7b, 2026-10-27): the agent
named a directory and then bare filenames, and neither file reached the canvas Files list
because the full path was never spelled out. Every negative is a shape that must stay
unresolved, because admitting it would be a guess about which directory was meant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.session_files import scan as scan_module
from local_operator.session_files.scan import FileScan
from local_operator.supplements import listing as listing_module
from local_operator.supplements.listing import directory_listing

from ._fixtures import line

PDF = "Radient-IP-Scope-for-Counsel-2026-10-27.pdf"
DOCX = "Radient-IP-Scope-for-Counsel-2026-10-27 (editable).docx"

#: Row 023a2908141445e0bff1d0a08932e33d's tail, verbatim.
FINAL_TEXT = f"Files are in `~/Documents/Radient/`:\n- {PDF}\n- {DOCX}"
#: Row 7243b9ec0d554dc8b01519a9c8b77117's tail (the 12:27 screenshot), verbatim.
EARLIER_TEXT = (
    "I also deleted my unstyled first draft from the same folder.\n\n"
    f"Files are in `~/Documents/Radient/`:\n- {PDF}\n- {DOCX}"
    " — the same text without the Radient styling, so the lawyers can mark it up"
)


def _reply(text: str, row_id: str = "a1") -> dict[str, Any]:
    """An assistant text reply in the real persisted shape (no ``tool_calls`` key)."""
    return {
        "id": row_id,
        "ts": 1.0,
        "type": "message",
        "payload": {
            "kind": "message",
            "role": "assistant",
            "content": [{"text": text}],
            "stop_reason": "stop",
        },
    }


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home" / "Documents" / "Radient"
    folder.mkdir(parents=True)
    (folder / PDF).write_bytes(b"%PDF-1.7 " + b"x" * 91)
    (folder / DOCX).write_bytes(b"PK" + b"y" * 198)
    return tmp_path / "home"


def _paths(home: Path, *texts: str) -> dict[str, tuple[str, int | None]]:
    scan = FileScan(cwd=str(home), home=str(home))
    scan.feed_all([line(_reply(text, f"a{i}")) for i, text in enumerate(texts)])
    return {rec.path or "": (rec.origin, rec.size) for rec in scan.records.values()}


@pytest.mark.parametrize("text", [FINAL_TEXT, EARLIER_TEXT], ids=["final", "with-description"])
def test_operator_listings_admit_both_files(home: Path, text: str) -> None:
    folder = home / "Documents" / "Radient"
    assert _paths(home, text) == {
        str(folder / PDF): ("mentioned", 100),
        str(folder / DOCX): ("mentioned", 200),
    }


def test_absolute_and_file_url_introducers_with_every_marker_and_separator(home: Path) -> None:
    folder = home / "Documents" / "Radient"
    (folder / "notes.md").write_text("n")
    text = (
        f"Saved to {folder}:\n"
        f"* `{PDF}` – the memo\n"
        f"* notes.md: scratch notes\n"
        f"\nSee file://{folder}/ :\n"
        f"1. {DOCX} - editable copy"
    )
    assert set(_paths(home, text)) == {str(folder / n) for n in (PDF, DOCX, "notes.md")}


@pytest.mark.parametrize(
    "text",
    [
        # No introducer at all: a bare name is a guess about the directory.
        f"Here they are:\n- {PDF}\n- {DOCX}",
        # A directory, but the line does not end with ":".
        f"Files are in `~/Documents/Radient/`\n- {PDF}",
        # A sentence item, not a filename.
        f"Files are in `~/Documents/Radient/`:\n- I wrote the {PDF} for counsel",
        # The head before the separator has no known extension.
        "Results in ~/Documents/Radient:\n- Run the tests - they pass",
        # An annotated head the disk refuses is NOT retried as a whole-item name.
        f"Files are in `~/Documents/Radient/`:\n- I wrote {PDF} - it is final",
        # A filename that does not exist on disk.
        "Files are in `~/Documents/Radient/`:\n- missing.pdf",
        # A non-list line ends the run; a later list does not inherit the introducer.
        f"Files are in `~/Documents/Radient/`:\n- nope.pdf\n\nThat is all.\n- {PDF}",
        # A continuation line straight under an item makes it prose, so the item is dropped.
        f"Files are in `~/Documents/Radient/`:\n- {PDF}\n  which I wrote for counsel",
    ],
    ids=[
        "no-introducer",
        "no-colon",
        "sentence-item",
        "no-extension-head",
        "refused-head-not-retried",
        "missing-on-disk",
        "prose-line-ends-run",
        "continuation-drops-item",
    ],
)
def test_shapes_that_stay_unresolved(home: Path, text: str) -> None:
    assert _paths(home, text) == {}


def test_an_assistant_row_without_the_listing_seam_is_never_parsed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cost invariant for this pass: only the listing seam reaches ``json.loads``."""
    parses: list[int] = []
    real = scan_module._row
    monkeypatch.setattr(scan_module, "_row", lambda raw: parses.append(1) or real(raw))
    # Path-heavy prose, a mid-line colon, and a list whose preceding line ends in no ":".
    _paths(
        home,
        f"I saved ~/Documents/Radient/{PDF} for you. Note: done.\n\nNext steps.\n- tidy up",
    )
    assert parses == []
    _paths(home, FINAL_TEXT)
    assert len(parses) == 1


# -- renderer parity ------------------------------------------------------------------------
# Cases copied from local-operator-ui scripts/link-targets.test.mjs (commits 938fb5da58 and
# 6dfe529fbd), whose rule this module mirrors. These pin the GRAMMAR (no disk): an annotated
# head is a candidate here and the stat in `_note_mention` is the renderer's evidence oracle.
# Expected values use the renderer's spelling (`~/out/...`), which is what `directory_listing`
# returns before `resolve_path`.
SCREENSHOT_1227 = f"Files are in `~/Documents/Radient/`:\n- {PDF}\n- {DOCX}" + (
    " — the same text without the Radient styling, so the lawyers can mark it up"
)
LISTED = [f"~/Documents/Radient/{PDF}", f"~/Documents/Radient/{DOCX}"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (FINAL_TEXT, LISTED),
        (SCREENSHOT_1227, LISTED),
        ("Saved to ~/out/:\n* a.pdf\n* b.csv", ["~/out/a.pdf", "~/out/b.csv"]),
        ("Saved to ~/out/:\n+ a.pdf\n+ [x] b.csv", ["~/out/a.pdf", "~/out/b.csv"]),
        # A CHANGED bullet is a second list, which follows a list, not the introducer.
        ("Saved to ~/out/:\n* a.pdf\n+ b.csv", ["~/out/a.pdf"]),
        ("Saved to /tmp/out:\n1. a.pdf\n2. `b c.md`", ["/tmp/out/a.pdf", "/tmp/out/b c.md"]),
        ("Wrote file:///tmp/My%20Out/:\n- a.pdf", ["/tmp/My Out/a.pdf"]),
        # A loose list is still the same list.
        ("Files in ~/out:\n\n- a.pdf\n\n- b.pdf", ["~/out/a.pdf", "~/out/b.pdf"]),
        # Only the IMMEDIATE list, and not its nested items.
        (
            "Files in ~/out:\n- a.pdf\n  - nested.pdf\n- b.pdf\n\nAnd later:\n- c.pdf",
            ["~/out/a.pdf", "~/out/b.pdf"],
        ),
        # The panel's known-extension rule on a listed name.
        ("Files in ~/out:\n- build.xyz123\n- report.pdf", ["~/out/report.pdf"]),
        ("Files in ~/out:\n- my build.xyz123", []),
        # Every annotation separator, and a code-wrapped head.
        ("Files in ~/out:\n- a.pdf — the summary", ["~/out/a.pdf"]),
        ("Files in ~/out:\n- a.pdf – the summary", ["~/out/a.pdf"]),
        ("Files in ~/out:\n- my-2026-10-27.pdf - the summary", ["~/out/my-2026-10-27.pdf"]),
        ("Files in ~/out:\n- a b.docx: the editable copy", ["~/out/a b.docx"]),
        ("Files in ~/out:\n- `a b.docx` — the editable copy", ["~/out/a b.docx"]),
        # Annotated negatives: no extension, unknown extension, no separator.
        ("Files in ~/out:\n- Run the tests - they pass", []),
        ("Files in ~/out:\n- build.xyz123 — the binary", []),
        ("Files in ~/out:\n- Open report.pdf for the summary", []),
        # Shape passes, so it is a candidate; the disk (stat) is what refuses it.
        ("Files in ~/out:\n- I wrote report.pdf - it is final", ["~/out/I wrote report.pdf"]),
        # The renderer's negative controls.
        ("Here they are:\n- report.pdf\n- notes.md", []),
        ("Files are in ~/Documents/Radient/\n- report.pdf", []),
        ("Files are in ~/Documents/Radient/.\n- report.pdf", []),
        (
            "Files are in ~/Documents/Radient/:\n- Open report.pdf for the summary\n"
            "- I updated the contract draft.",
            [],
        ),
        ("I looked in ~/Documents/Radient/ and found these:\n- report.pdf", []),
        ("Read ~/x/report.pdf:\n- notes.md", []),
        ("Files in ~/out:\n- sub/report.pdf", []),
        ("Files in ~/out:\n- <name>.pdf\n- *.pdf", []),
        # A one-segment `/word` is as likely a slash command as a folder.
        ("Saved to /tmp:\n- a.pdf", []),
        ("Run /new:\n- a.pdf", []),
    ],
)
def test_grammar_matches_the_renderer(text: str, expected: list[str]) -> None:
    assert directory_listing(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Saved to ~/out/:\n* a.pdf", True),
        ("Files in ~/out:\n\n- a.pdf", True),
        ("Files:\n  1. a.pdf", True),
        ("Files in ~/out:\n\t- a.pdf", True),
        # F3: every whitespace JS `\s` admits, written as json.dumps escapes it.
        ("Files in ~/out:\u00a0\n- a.pdf", True),
        ("Files in ~/out:\u3000\n- a.pdf", True),
        ("Files in ~/out:\u2003\n\u2003- a.pdf", True),
        ("Files in ~/out:\f\n\f- a.pdf", True),
        ("Files in ~/out:\v\n\v- a.pdf", True),
        ("Files in ~/out:\ufeff\n\u202f- a.pdf", True),
        ("Files in ~/out: done\n- a.pdf", False),
        ("no colon here\n- a.pdf", False),
    ],
)
def test_byte_probe_sees_every_listing_seam(text: str, expected: bool) -> None:
    """The probe must never be NARROWER than the grammar, or a listing is silently skipped."""
    assert bool(scan_module._DIR_LISTING_PROBE.search(line(_reply(text)))) is expected


# -- review round 1 -------------------------------------------------------------------------
# F1: the byte probes can fire inside a NESTED value, so only the decoded row's own top-level
# role may route a row to the listing pass; every other row keeps its usual handler.


def test_a_tool_result_with_a_nested_assistant_role_keeps_its_command_origin(
    tmp_path: Path,
) -> None:
    from ._fixtures import assistant, call, tool

    home = tmp_path / "home"
    (home / "rp").mkdir(parents=True)
    (home / "rp" / "out.pdf").write_bytes(b"%PDF")
    scan = FileScan(cwd=str(home), home=str(home))
    scan.feed_all(
        [
            line(assistant("a", 1, [call("c1", "bash", command="render > ~/rp/out.pdf")])),
            line(
                tool(
                    "t1",
                    2,
                    "c1",
                    "bash",
                    "exit code: 0\n--- stdout ---\nFiles are in `~/rp/`:\n- out.pdf",
                    {"rows": [{"role": "assistant", "text": "quoted"}]},
                )
            ),
        ]
    )
    assert {rec.path: rec.origin for rec in scan.records.values()} == {
        str(home / "rp" / "out.pdf"): "command"
    }


def test_a_sessions_result_quoting_another_sessions_listing_adds_nothing(home: Path) -> None:
    from ._fixtures import assistant, call, tool

    (home / "Documents" / "Radient" / "other.pdf").write_bytes(b"%PDF")
    scan = FileScan(cwd=str(home), home=str(home))
    scan.feed_all(
        [
            line(assistant("a", 1, [call("c1", "sessions", op="peek", session="0123456789ab")])),
            line(
                tool(
                    "t1",
                    2,
                    "c1",
                    "sessions",
                    "Files are in `~/Documents/Radient/`:\n- other.pdf",
                    {"mesh": {"rows": [{"role": "assistant", "text": "Files are in ..."}]}},
                )
            ),
        ]
    )
    assert scan.records == {}


# F2 + renderer round 1 (local-operator-ui c1798b33b0). Expected values are what the renderer's
# `listedTargetsIn` returned for each input under MENTION_POLICY with an oracle answering
# "exists" (bundled and run with node against c1798b33b0; review round 1 ran the same harness
# against a895cf571b and found these 13 shapes differing). Grammar only: the stat decides later.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Emphasis around the directory: the token stops at `*`, `_` is a closer in the tail.
        ("**~/Documents/Radient/**:\n- a.pdf", ["~/Documents/Radient/a.pdf"]),
        ("*~/out/*:\n- a.pdf", ["~/out/a.pdf"]),
        ("**~/Documents/Radient**:\n- a.pdf", ["~/Documents/Radient/a.pdf"]),
        ("_~/out/_:\n- a.pdf", []),
        # Directory shapes the renderer refuses.
        ("Files in /v1/out/:\n- a.pdf", []),
        ("Files in /api/out/:\n- a.pdf", []),
        ("Files in //host/share/:\n- a.pdf", []),
        ("Files in /a/b/:59:\n- a.pdf", []),
        ("Files in /a/b/#frag:\n- a.pdf", []),
        ("Files in /a/.../b/:\n- a.pdf", []),
        ("Files in /a|b/c/:\n- a.pdf", []),
        ("Files in /a/{b}/c/:\n- a.pdf", []),
        # A `,` or `*` ends the token; the LAST token on the line is the directory.
        ("Files in /a/b,/c/d/:\n- a.pdf", ["/c/d/a.pdf"]),
        ("Files in /a/*/b/:\n- a.pdf", ["/b/a.pdf"]),
        # JS `.` stops at U+2028, so that line is not an item.
        ("Files in ~/out/:\n- a\u2028b.pdf", []),
    ],
)
def test_directory_token_matches_the_renderer(text: str, expected: list[str]) -> None:
    assert directory_listing(text) == expected


# The renderer's round-1 PARITY table and directory-shape cases (scripts/link-targets.test.mjs
# at c1798b33b0), plus the two #1451 round-2 rules mirrored ahead of the renderer: an ordered
# list straight under the introducer starts with a LITERAL `1` (`01.` is refused), and an item
# inside a blockquote is never an item.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```\nSaved to ~/out/:\n- a.pdf\n- b.pdf\n```", []),
        ("~~~\nSaved to ~/out/:\n- a.pdf\n~~~", []),
        ("Intro.\n\n    Saved to ~/out/:\n    - a.pdf\n    - b.pdf", []),
        ("## Files in ~/out/:\n- a.pdf", []),
        ("> Files in ~/out/:\n- a.pdf", []),
        ("> Files in ~/out/:\n> - a.pdf", []),
        ("Saved to /tmp/out:\n2. a.pdf\n3. b.pdf", []),
        ("Saved to /tmp/out:\n01. a.pdf", []),
        ("Saved to ~/out/:\n```\n- a.pdf\n```", []),
        ("Saved to ~/out/:\n- a.pdf\n- b.pdf", ["~/out/a.pdf", "~/out/b.pdf"]),
        ("Saved to /tmp/out:\n1. a.pdf\n2. b.pdf", ["/tmp/out/a.pdf", "/tmp/out/b.pdf"]),
        ("Saved to /tmp/out:\n\n2. a.pdf", ["/tmp/out/a.pdf"]),
        ("```\ncode\n```\nSaved to ~/out/:\n- a.pdf", ["~/out/a.pdf"]),
        ("Saved to ~/out/:\n- a.pdf\n\n- b.pdf", ["~/out/a.pdf", "~/out/b.pdf"]),
        # Directory shape: a trailing slash, or no dot past the last segment's first char.
        ("Bumped ~/repo/v1.2.3:\n- a.pdf", []),
        ("See ~/x/Next.js:\n- a.pdf", []),
        ("/etc/nginx/conf.d:\n- a.conf", []),
        ("Files in /etc/nginx/conf.d/:\n- a.conf", ["/etc/nginx/conf.d/a.conf"]),
        ("Wrote ~/.config/app:\n- settings.json", ["~/.config/app/settings.json"]),
        # `Makefile:` is shape-admitted as a root; `.o` is not a known extension.
        ("Edit ~/repo/Makefile:\n- target.o", []),
        # F4: the renderer's known-extension set.
        (
            "Files in ~/out/:\n- export.zip\n- deck.key\n- notes.odt\n- data.xml\n- r.pdf",
            [f"~/out/{n}" for n in ("export.zip", "deck.key", "notes.odt", "data.xml", "r.pdf")],
        ),
    ],
)
def test_block_shapes_match_the_renderer(text: str, expected: list[str]) -> None:
    assert directory_listing(text) == expected


def test_a_heading_introducer_admits_nothing_even_when_the_file_exists(home: Path) -> None:
    assert _paths(home, f"## Files are in `~/Documents/Radient/`:\n- {PDF}\n- {DOCX}") == {}


def test_bold_introducer_resolves_on_disk(home: Path) -> None:
    folder = home / "Documents" / "Radient"
    assert set(_paths(home, f"**~/Documents/Radient/**:\n- {PDF}")) == {str(folder / PDF)}


#: The renderer's KNOWN_EXTENSIONS at c1798b33b0 (file-kind.ts), sorted. Pinned here so the
#: check runs without the UI checkout; the test below re-reads the source when one is present.
RENDERER_KNOWN_EXTENSIONS_C1798B33B0 = (
    "3g2 3gp 7z aac astro avi avif bash bmp c cfg cjs clj conf cpp cs css csv dart doc docx env"
    " ex exs fish flac flv gif go gradle gz h heic heif htm html ico ini java jfif jpeg jpg js"
    " json jsonc jsx key kt less log lua m4a m4v markdown md mdown mdx mjs mkd mkdn mkv mov mp3"
    " mp4 odp ods odt oga ogg pages pdf php pjp pjpeg pl png ppt pptx py r rar rb rs rtf scala"
    " scss sh sql svelte svg swift tar text tif tiff toml ts tsv tsx txt vue wav webm webp wmv"
    " xhtml xls xlsx xml yaml yml zip zsh"
).split()


def test_known_extensions_mirror_the_renderer_snapshot() -> None:
    assert sorted(listing_module.KNOWN_EXTENSIONS) == RENDERER_KNOWN_EXTENSIONS_C1798B33B0


def test_known_extensions_match_the_renderer_source_when_checked_out() -> None:
    """Re-read file-kind.ts from a local-operator-ui checkout (``LOCAL_OPERATOR_UI_REPO``, or
    the sibling ``../local-operator-ui``); skipped where none exists, e.g. on CI."""
    import os
    import re as regex

    root = Path(os.environ.get("LOCAL_OPERATOR_UI_REPO") or Path(__file__).parents[3].parent)
    if "LOCAL_OPERATOR_UI_REPO" not in os.environ:
        root = root / "local-operator-ui"
    source = root / "src/renderer/src/features/chat/utils/file-kind.ts"
    if not source.is_file():
        pytest.skip("no local-operator-ui checkout")
    text = source.read_text(encoding="utf-8")
    union = regex.search(r"KNOWN_EXTENSIONS[^=]*=\s*new Set\(\[(.*?)\]\)", text, regex.DOTALL)
    assert union is not None
    found: set[str] = set()
    for group in regex.findall(r"\.\.\.([A-Z]+)", union.group(1)):
        lower = group.lower()
        block = regex.search(rf"export const {lower}Extensions = \[(.*?)\]", text, regex.DOTALL)
        assert block is not None, group
        found.update(regex.findall(r'"([^"]+)"', block.group(1)))
    assert found == set(listing_module.KNOWN_EXTENSIONS)


def test_byte_probe_covers_every_whitespace_the_grammar_admits() -> None:
    """F3, exhaustively: no whitespace placement may make the probe narrower than the grammar.

    The grammar indents with JS ``\\s`` and trims item text with JS ``trim``, while the probe
    reads the row's JSON bytes, so the two agree only if every spelling ``json.dumps`` writes
    is in the probe's gap class. Sweeps the whole JS whitespace set around the colon, on blank
    lines, as item indent and after the marker, and asserts: grammar finds a candidate => the
    probe fired.
    """
    whitespace = [c for c in listing_module._JS_SPACE_CHARS if c != "\n"]
    shapes: list[str] = []
    for char in whitespace:
        shapes += [
            f"Files in ~/out:{char}\n- a.pdf",
            f"Files in ~/out:\n{char}- a.pdf",
            f"Files in ~/out:\n\n{char}- a.pdf",
            f"Files in ~/out:\n{char}\n- a.pdf",
            f"Files in ~/out:\n-{char}a.pdf",
            f"Files in ~/out:\n- a.pdf{char}",
            f"Files in ~/out/{char}:\n- a.pdf",
        ]
        for other in whitespace:
            shapes.append(f"Files in ~/out:{char}\n{other}- a.pdf")
            shapes.append(f"Files in ~/out:\n{char}{other}- a.pdf")
    for text in shapes:
        if directory_listing(text):
            assert scan_module._DIR_LISTING_PROBE.search(line(_reply(text))), repr(text)
