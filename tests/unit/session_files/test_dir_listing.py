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
from local_operator.session_files.scan import FileScan, _directory_listing

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
# Expected values use the renderer's spelling (`~/out/...`), which is what `_directory_listing`
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
    assert _directory_listing(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Saved to ~/out/:\n* a.pdf", True),
        ("Files in ~/out:\n\n- a.pdf", True),
        ("Files:\n  1. a.pdf", True),
        ("Files in ~/out:\n\t- a.pdf", True),
        ("Files in ~/out: done\n- a.pdf", False),
        ("no colon here\n- a.pdf", False),
    ],
)
def test_byte_probe_sees_every_listing_seam(text: str, expected: bool) -> None:
    """The probe must never be NARROWER than the grammar, or a listing is silently skipped."""
    assert bool(scan_module._DIR_LISTING_PROBE.search(line(_reply(text)))) is expected
