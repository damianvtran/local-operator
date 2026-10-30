"""Assemble the evidence bundle for the PR comment from the final run.

Writes docs/evidence/locale-frames/ into the evidence worktree: the PNGs under
readable names, a README with the method and the frame table, and the raw
numbers as JSON so a reader can re-derive every claim in the comment.
"""

from __future__ import annotations

import json
import pathlib
import shutil

S = pathlib.Path("/Users/damian/.local-operator/sessions/d9aea2853586/scratchpad")
RUN = S / "evidence_run"
OUT = pathlib.Path(
    "/Users/damian/local-operator/.worktrees/locale-frames-evidence/docs/evidence/locale-frames"
)

#: Which captured stills go into the bundle, and under what name. The mapping is
#: explicit rather than a glob so a missing capture fails loudly instead of
#: publishing a bundle with a hole in it.
STILLS = {
    "ru-populated-deeplink-first-content.png": "ru-1-first-content-session-deeplink.png",
    "ru-populated-list-settled.png": "ru-2-populated-list.png",
    "ru-populated-settings-settled.png": "ru-3-populated-settings.png",
    "ru-empty-list-settled.png": "ru-4-empty-list.png",
    "ru-deadapi-list-settled.png": "ru-5-list-unavailable.png",
    "ru-deadapi-settings-settled.png": "ru-6-settings-load-error.png",
    "vi-populated-deeplink-first-content.png": "vi-1-first-content-session-deeplink.png",
    "vi-populated-list-settled.png": "vi-2-populated-list.png",
    "vi-populated-settings-settled.png": "vi-3-populated-settings.png",
    "vi-empty-list-settled.png": "vi-4-empty-list.png",
    "vi-deadapi-list-settled.png": "vi-5-list-unavailable.png",
    "vi-deadapi-settings-settled.png": "vi-6-settings-load-error.png",
    "fr-populated-deeplink-first-content.png": "fr-1-first-content-session-deeplink.png",
    "fr-populated-deeplink-settled.png": "fr-2-deeplink-settled.png",
    "fr-populated-list-settled.png": "fr-3-populated-list.png",
    "fr-populated-settings-settled.png": "fr-4-populated-settings.png",
    "fr-empty-list-settled.png": "fr-5-empty-list.png",
    "fr-deadapi-list-settled.png": "fr-6-list-unavailable.png",
    "fr-deadapi-settings-settled.png": "fr-7-settings-load-error.png",
    "hi-control-populated-deeplink-first-content.png": "control-hi-1-first-content-English.png",
    "hi-control-populated-list-settled.png": "control-hi-2-list-English.png",
    "fr-slowchunk-populated-deeplink-at2400.png": "control-slowchunk-1-English-fallback.png",
    "fr-slowchunk-populated-deeplink-at7000.png": "control-slowchunk-2-after-flip.png",
}


def main() -> None:
    results = json.loads((RUN / "results.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- stills -----------------------------------------------------------
    # The label -> file stem is the capture's own label, so a renamed case cannot
    # pick up another case's image.
    produced: dict[str, pathlib.Path] = {}
    for case in results:
        for screen in case["screens"]:
            label = screen["label"]
            produced[f"{label}-settled.png"] = RUN / f"{label}-settled.png"
            produced[f"{label}-first-content.png"] = RUN / f"{label}-first-content.png"
            for at in (2400, 7000):
                produced[f"{label}-at{at}.png"] = RUN / f"{label}-at{at}.png"

    missing = [name for name in STILLS if name not in produced or not produced[name].exists()]
    if missing:
        raise SystemExit(f"missing captures: {missing}")

    manifest = []
    for source_name, target_name in STILLS.items():
        shutil.copyfile(produced[source_name], OUT / target_name)
        manifest.append({"file": target_name, "from": source_name})

    # ---- numbers ----------------------------------------------------------
    # ONE classifier for the comment and the bundle: imported from the analysis
    # script so the stills a reader sees and the counts beside them cannot be
    # produced by two different opinions about what "English" means.
    from analyse import classify

    table = []
    for case in results:
        for screen in case["screens"]:
            frames = screen["frames"]
            first = screen["firstContent"]
            screen_name = screen["label"].rsplit("-", 1)[-1]
            verdicts = [
                (frame, classify(frame["text"], case["locale"], screen_name))
                for frame in frames
                if frame["text"]
            ]
            english = [f for f, v in verdicts if v in ("english", "both")]
            content = [f for f, _ in verdicts]
            chunks = [r for r in screen.get("resources", []) if "bundled." in r["path"]]
            table.append(
                {
                    "case": f"{case['locale']}/{case['mode']}",
                    "screen": screen["label"],
                    "configLanguageServer": case["configFrame"]["language"],
                    "documentLang": screen["lang"],
                    "documentDir": screen["dir"],
                    "framesSampled": len(frames),
                    "contentFrames": len(content),
                    "firstContentFrameMs": first["t"] if first else None,
                    "firstContentLang": first["lang"] if first else None,
                    "firstContentVerdict": classify(first["text"], case["locale"], screen_name)
                    if first
                    else None,
                    "firstContentText": first["text"][:160] if first else None,
                    "framesBeforeFirstContent": sum(
                        1 for f in frames if first and f["t"] < first["t"]
                    ),
                    "englishContentFrames": len(english),
                    "firstEnglishFrameMs": english[0]["t"] if english else None,
                    "lastEnglishFrameMs": english[-1]["t"] if english else None,
                    "settledText": screen["text"][:400],
                    "settle": screen.get("settle"),
                    "localeChunkRequests": [
                        {
                            "file": c["name"],
                            "status": c["status"],
                            "startMs": c["startTime"],
                            "endMs": c["responseEnd"],
                            "transferBytes": c["transferSize"],
                            "decodedBytes": c["decodedBodySize"],
                        }
                        for c in chunks
                    ],
                }
            )

    (OUT / "frames.json").write_text(
        json.dumps({"captures": table, "stills": manifest}, ensure_ascii=False, indent=1)
    )
    print(f"wrote {len(manifest)} stills and {len(table)} rows to {OUT}")

    # A guard the comment's central claim rests on: no capture loaded a second
    # locale's chunk, and no locale capture painted English copy.
    for row in table:
        if row["case"].startswith("hi") or row["screen"].startswith("fr-slowchunk"):
            continue
        if row["englishContentFrames"]:
            raise SystemExit(f"ENGLISH FRAMES in {row['screen']}: {row['englishContentFrames']}")
        if len(row["localeChunkRequests"]) != 1:
            raise SystemExit(f"{row['screen']}: {len(row['localeChunkRequests'])} chunk requests")
    print("guard ok: every locale capture has 0 English frames and exactly 1 chunk")


if __name__ == "__main__":
    main()
