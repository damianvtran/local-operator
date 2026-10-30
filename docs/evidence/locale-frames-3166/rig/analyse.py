"""Classify the captured frames per locale/state and print the QA matrix rows."""

from __future__ import annotations

import json
from pathlib import Path

import os, sys
S = Path(os.environ["LF_OUT"])

#: The English and per-locale strings each screen's first content frame can carry.
#: Taken from the generated bundles (`src/i18n/bundled*.gen.ts`), so a marker is
#: the product's own copy rather than a guess about wording.
MARKERS = {
    "deeplink": {
        "en": ["connecting to session", "Loading conversation", "No messages yet"],
        "ru": ["подключение к сессии", "Загрузка беседы", "Сообщений пока нет"],
        "vi": ["đang kết nối tới phiên", "Đang tải cuộc trò chuyện", "Chưa có tin nhắn nào"],
        "fr": ["connexion à la session", "Chargement de la conversation", "Aucun message pour l"],
        "hi": ["सत्र से कनेक्ट हो रहा है", "बातचीत लोड हो रही है", "अभी कोई संदेश नहीं"],
        "es": ["conectando con la sesión", "Cargando la conversación", "Aún no hay mensajes"],
    },
    # The list's OWN chrome, quoted from the rendered frames: the filter chips and
    # the New button are on screen in every list state, so they are what a
    # populated screen can be classified by (the empty-state sentence is not on a
    # populated screen at all). Each marker is unique to its locale — "Needs you"
    # or "À traiter" cannot collide.
    "list": {
        "en": ["Needs you", "Unread ·", "＋ New"],
        "ru": ["Ждут вас", "Непрочитано", "Новая"],
        "vi": ["Cần bạn", "Chưa đọc", "Mới"],
        "fr": ["À traiter", "Non lues", "Nouvelle"],
        "hi": ["आपका ध्यान चाहिए", "अपठित", "नया"],
        "es": ["Te necesita", "Sin leer", "Nuevo"],
    },
    "settings": {
        "en": ["Computer settings", "Changes apply only to this computer", "Loading account routing"],
        "ru": ["Настройки компьютера", "Изменения применяются только", "Загрузка маршрутизации"],
        "vi": ["Cài đặt máy tính", "thay đổi chỉ áp dụng", "Đang tải định tuyến"],
        "fr": ["Paramètres de l’ordinateur", "Les changements ne s’appliquent", "Chargement du routage"],
        "hi": ["कंप्यूटर सेटिंग", "बदलाव सिर्फ़ इस कंप्यूटर", "खाता रूटिंग लोड"],
        "es": ["Ajustes de la máquina", "Los cambios solo se aplican", "Cargando el enrutamiento"],
    },
}


def classify(text: str, locale: str, screen: str) -> str:
    """'locale', 'english', 'both' or 'neither' for one frame's visible copy."""
    if not text:
        return "empty"
    group = MARKERS[screen]
    # `hi` is the negative control: no chunk ships, so there is no locale copy
    # to look for and every content frame must classify as English.
    hits_locale = os.environ.get("LF_NO_LOCALE_MARKERS") != "1" and bool(group.get(locale)) and any(marker in text for marker in group[locale])
    hits_english = any(marker in text for marker in group["en"])
    if hits_locale and hits_english:
        return "both"
    if hits_locale:
        return "locale"
    if hits_english:
        return "english"
    return "neither"


def main() -> None:
    results = json.loads((S / "results.json").read_text())
    rows = []
    for case in results:
        locale = case["locale"]
        mode = case["mode"]
        for screen in case["screens"]:
            name = screen["label"].rsplit("-", 1)[-1]
            locale_for_markers = locale
            frames = screen["frames"]
            first = screen["firstContent"]
            content = [f for f in frames if f["text"]]
            english = [
                f for f in content if classify(f["text"], locale, name) in ("english", "both")
            ]
            before = [f for f in frames if first and f["t"] < first["t"]]
            english_before = [f for f in before if f["text"]]
            chunk = None
            for resource in screen.get("resources", []):
                if "bundled." in resource["path"]:
                    chunk = resource
            rows.append(
                {
                    "label": screen["label"],
                    "locale": locale,
                    "mode": mode,
                    "screen": name,
                    "configLanguage": case["configFrame"].get("language"),
                    "lang": screen["lang"],
                    "dir": screen["dir"],
                    "firstContent": first,
                    "firstVerdict": classify(first["text"], locale, name) if first else None,
                    "frames": len(frames),
                    "framesBeforeContent": len(before),
                    "contentFrames": len(content),
                    "englishFrames": len(english),
                    "firstEnglishAt": english[0]["t"] if english else None,
                    "englishBeforeContent": len(english_before),
                    "settledText": screen["text"],
                    "chunks": [c for c in screen.get("chunks", [])],
                    "chunkResource": chunk,
                    "settle": screen.get("settle"),
                }
            )
    (S / "analysis.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    print(
        f"{'label':34s} {'cfg':6s} {'lang':6s} {'first@':7s} {'verdict':8s} "
        f"{'EN':3s} {'chunk':38s} {'bytes':7s}"
    )
    for row in rows:
        chunk = row["chunkResource"]
        chunk_name = chunk["name"] if chunk else "-"
        chunk_status = f"{chunk['status']}" if chunk else "-"
        print(
            f"{row['label']:34s} {str(row['configLanguage']):6s} {str(row['lang']):6s} "
            f"{row['firstContent']['t'] if row['firstContent'] else -1:7d} "
            f"{str(row['firstVerdict']):8s} {row['englishFrames']:3d} "
            f"{(chunk_name + ' ' + chunk_status):38s} "
            f"{chunk['decodedBodySize'] if chunk else -1:7d}"
        )


if __name__ == "__main__":
    main()
