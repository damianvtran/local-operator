"""Render the measured table into the README, from frames.json — never by hand.

Every number in the published table comes from the captured run: the frame
timestamps, the English-frame counts and the chunk request are read out of
`frames.json`. Hand-typing them is how an evidence table ends up disagreeing
with its own data.
"""

from __future__ import annotations

import json
import pathlib

BASE = pathlib.Path(
    "/Users/damian/local-operator/.worktrees/locale-frames-evidence/docs/evidence/locale-frames"
)
README = BASE / "README.md"
BEGIN = "<!-- measurements:begin -->"
END = "<!-- measurements:end -->"

ORDER = [
    ("ru-populated-deeplink", "ru", "session deep link (cold deep link `#/s/<id>`)"),
    ("ru-populated-list", "ru", "populated list `#/`"),
    ("ru-populated-settings", "ru", "settings surface"),
    ("ru-empty-list", "ru", "empty list"),
    ("ru-deadapi-list", "ru", "list, session API answering 503"),
    ("ru-deadapi-settings", "ru", "settings, API 503"),
    ("vi-populated-deeplink", "vi", "session deep link (cold)"),
    ("vi-populated-list", "vi", "populated list `#/`"),
    ("vi-populated-settings", "vi", "settings surface"),
    ("vi-empty-list", "vi", "empty list"),
    ("vi-deadapi-list", "vi", "list, session API answering 503"),
    ("vi-deadapi-settings", "vi", "settings, API 503"),
    ("fr-populated-deeplink", "fr", "session deep link (cold)"),
    ("fr-populated-list", "fr", "populated list `#/`"),
    ("fr-populated-settings", "fr", "settings surface"),
    ("fr-empty-list", "fr", "empty list"),
    ("fr-deadapi-list", "fr", "list, session API answering 503"),
    ("fr-deadapi-settings", "fr", "settings, API 503"),
    ("hi-control-populated-deeplink", "hi", "CONTROL — deep link, `hi` named in config"),
    ("hi-control-populated-list", "hi", "CONTROL — list, `hi` named in config"),
    ("fr-slowchunk-populated-deeplink", "fr", "CONTROL — deep link, `fr` chunk held past 1 500 ms"),
]


def main() -> None:
    data = json.loads((BASE / "frames.json").read_text())
    by_screen = {row["screen"]: row for row in data["captures"]}

    lines = [
        "| locale | state | `lang` at first content | first content frame | English content frames | locale chunk (status · decoded bytes · response end) |",
        "|---|---|---|---|---|---|",
    ]
    for screen, locale, description in ORDER:
        row = by_screen[screen]
        chunks = row["localeChunkRequests"]
        if chunks:
            chunk = chunks[0]
            chunk_text = (
                f"`{chunk['file']}` · {chunk['status']} · "
                f"{chunk['decodedBytes']:,} B · ends {chunk['endMs']:,} ms"
            )
        else:
            chunk_text = "none requested (no chunk for this locale)"

        lang = row["documentLang"]
        first = row["firstContentFrameMs"]
        english = row["englishContentFrames"]

        if screen == "fr-slowchunk-populated-deeplink":
            lang = f"`en` then `{row['documentLang']}`"
            first = f"{first:,} ms — ENGLISH `{row['firstContentText']}`"
            english = (
                f"{english}, last one at {row['lastEnglishFrameMs']:,} ms; "
                "French takes over when the chunk lands"
            )
        else:
            first = f"{first:,} ms — `{row['firstContentText']}`"
            lang = f"`{lang}`"
            english = str(english)
        lines.append(
            f"| {locale} | {description} | {lang} | {first} | {english} | {chunk_text} |"
        )

    table = "\n".join(lines)
    readme = README.read_text()
    head, rest = readme.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    README.write_text(f"{head}{BEGIN}\n{table}\n{END}{tail}")
    print(f"spliced {len(ORDER)} rows")


if __name__ == "__main__":
    main()
