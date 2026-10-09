"""Locale resolution: config -> ``LOP_LANG`` -> OS -> ``en`` (RFC §2.5, §6).

One function answers "what language is this process in", and every surface that
reads it — the capabilities route now; the TUI, `lop exec` and the SDK next —
must call THIS function rather than re-spelling the chain. The chain, highest
wins:

1. an explicit ``language`` config tag (``auto`` means "keep looking");
2. ``LOP_LANG`` — the per-call override for tests and `exec`;
3. the operating system;
4. ``en``.

Two filters apply to whatever the chain finds. A tag is first normalised to the
supported set (`fr-CA` -> `fr`; `zh*` -> `zh-CN`; `zh-TW`/`Hant` -> unsupported,
i.e. falls back to `en` per the RFC; anything else -> unsupported), and then
checked against the SHIPPED set — the locales the translation ledger has passed
(§6). A locale that exists but is not audited must never key a live surface, so
an unshipped candidate resolves to ``en``. Today the shipped set is ``en``
alone: the mechanism ships dark and a user cannot surprise themselves.

The OS readers are injectable so the truth table is testable without a
subprocess or a real preferences file: ``system``/``environ``/``home`` select
the branch inputs, and the macOS reader falls back to ``defaults read -g`` only
when the preferences plist itself is absent (a GUI-login read via `plistlib`
measures 1.8 ms against 290 ms for the subprocess — RFC §12 probe 13).
"""

from __future__ import annotations

import json
import os
import platform
import plistlib
import subprocess
from pathlib import Path
from typing import Any, Mapping, cast

#: The consumer default for the ``language`` config key. Lives HERE, beside the
#: resolver that reads the key, and is what `_consumer_defaults` in
#: `tests/unit/test_settings_io.py` compares the registry row against.
DEFAULT_LANGUAGE = "auto"

#: The config key itself (top-level scalar in `config.yml`).
CONFIG_KEY = "language"

#: The full supported value space, in the RFC's order. Membership here is
#: NECESSARY for a tag to survive normalisation; SHIPMENT (the ledger) decides
#: whether it can be selected — see :func:`shipped_locales`.
SUPPORTED_LOCALES = ("en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi")


def _data_path(name: str) -> Path:
    return Path(__file__).with_name("data") / name


def shipped_locales() -> tuple[str, ...]:
    """Locales the ledger has passed, from the generated ``data/shipped.json``.

    `en` is always shipped: it is the source locale and every fallback lands on
    it, so a missing or unreadable file degrades to ``("en",)`` rather than to
    an empty set that would make every resolution fail.
    """
    try:
        data: Any = json.loads(_data_path("shipped.json").read_text(encoding="utf-8"))
        listed = data.get("locales", []) if isinstance(data, dict) else []
        picked = {loc for loc in listed if loc in SUPPORTED_LOCALES}
    except (OSError, ValueError):
        picked = set()
    picked.add("en")
    return tuple(locale for locale in SUPPORTED_LOCALES if locale in picked)


def normalise_locale(tag: str | None) -> str | None:
    """Map a locale tag to a supported locale, or None when unsupported.

    Region, script, encoding and modifier are stripped to the language
    (`en_US.UTF-8` -> `en`, `fr-CA` -> `fr`). Chinese maps to `zh-CN` except
    Traditional script and Taiwan, which v1 does not ship and which fall back
    to `en` by the RFC. (Hong Kong/Macau preferences ride the `zh*` branch
    under that literal rule; the wave review revisits them with the style
    guides.)
    """
    if not tag or not isinstance(tag, str):
        return None
    cleaned = tag.strip().split(".", 1)[0].split("@", 1)[0].replace("_", "-")
    if not cleaned:
        return None
    parts = cleaned.split("-")
    language = parts[0].lower()
    if language in ("c", "posix"):
        return None
    if language == "zh":
        modifiers = {part.lower() for part in parts[1:]}
        if "hant" in modifiers or "tw" in modifiers:
            return None
        return "zh-CN"
    return language if language in SUPPORTED_LOCALES else None


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def config_value() -> str | None:
    """Best-effort read of the ``language`` key, without destructiveness.

    Prefers the config WATCHER's already-validated snapshot (dict access) over
    constructing a ``ConfigManager``, mirroring ``tui.settings._load``: a
    manager's constructor can move a malformed ``config.yml`` aside, and a
    resolution read must never be what renames a user's file. Constructing one
    is the fallback only for processes with no watcher (CLI, tests), where
    there is no cross-process invalidation to race with.
    """
    try:
        from local_operator.config_watch import existing_watcher
        from local_operator.paths import config_dir

        directory = config_dir()
        watcher = existing_watcher(directory)
        if watcher is not None:
            value = watcher.values.get(CONFIG_KEY)
            return value if isinstance(value, str) else None
        from local_operator.config import ConfigManager

        value = ConfigManager(directory).get_config().values.get(CONFIG_KEY)
        return value if isinstance(value, str) else None
    except Exception:
        return None


def env_value(environ: Mapping[str, str] | None = None) -> str | None:
    """``LOP_LANG`` normalised, or None. Read per call so tests can flip it."""
    environ = os.environ if environ is None else environ
    return normalise_locale(environ.get("LOP_LANG"))


def os_language(
    *,
    system: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> str | None:
    """The OS language, or None when unsupported/undetectable.

    Injectable inputs keep the per-platform truth table testable: `system`
    selects the branch, `environ`/`home` feed it. The Windows reader is
    documented but unmeasured on this fleet (RFC §12's "uncertainties").
    """
    system = system if system is not None else platform.system()
    if system == "Darwin":
        return _macos_language(home if home is not None else Path.home())
    if system == "Windows":
        return _windows_language()
    return _posix_language(environ if environ is not None else os.environ)


def _posix_language(environ: Mapping[str, str]) -> str | None:
    """Linux/POSIX: ``LC_ALL`` > ``LC_MESSAGES`` > ``LANG`` > ``LANGUAGE`` list.

    A variable that does not normalise (e.g. ``LANG=de_DE``) falls THROUGH to
    the next source rather than stopping the chain — an unsupported preference
    should not silence a supported one behind it.
    """
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        tag = normalise_locale(environ.get(var))
        if tag is not None:
            return tag
    language = environ.get("LANGUAGE")
    if language:
        for entry in language.split(":"):
            tag = normalise_locale(entry)
            if tag is not None:
                return tag
    return None


def _macos_language(home: Path) -> str | None:
    """macOS: ``AppleLanguages`` from ``.GlobalPreferences.plist``.

    The plist is read with `plistlib` (stdlib); only its ABSENCE falls back to
    the ``defaults read -g`` subprocess. Entries are a preference LIST — the
    first supported one wins, so a user whose first language is unsupported
    (``de-DE``, then ``en-US``) still resolves to ``en``.
    """
    plist = home / "Library" / "Preferences" / ".GlobalPreferences.plist"
    try:
        with plist.open("rb") as handle:
            data = plistlib.load(handle)
        entries = data.get("AppleLanguages") if isinstance(data, dict) else None
        if isinstance(entries, list):
            for entry in entries:
                tag = normalise_locale(entry if isinstance(entry, str) else None)
                if tag is not None:
                    return tag
            return None
    except (OSError, plistlib.InvalidFileException, ValueError):
        pass
    return _macos_defaults_language()


def _macos_defaults_language() -> str | None:
    """``defaults read -g AppleLanguages`` fallback: parse the quoted list."""
    try:
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleLanguages"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        candidate = line.strip().rstrip(",").strip()
        if candidate.startswith('"') and candidate.endswith('"'):
            tag = normalise_locale(candidate[1:-1])
            if tag is not None:
                return tag
    return None


def _windows_language() -> str | None:
    """Windows: ``GetUserPreferredUILanguages``, then ``locale.getlocale()``.

    ctypes in the function body (never at import) so this module imports on
    every platform; the reader is documented but not exercised on our CI (RFC
    §12), which is exactly why the fallback chain is kept simple.
    """
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        flags = ctypes.c_ulong(0x8)  # MUI_LANGUAGE_NAME
        count = ctypes.c_ulong(0)
        size = ctypes.c_ulong(0)
        # First call measures the UTF-16, NUL-separated buffer (pcch in WCHARs
        # including terminators); the second fills it. BOOL false means no
        # preferred languages are configured, which falls to getlocale().
        if kernel32.GetUserPreferredUILanguages(
            flags, ctypes.byref(count), None, ctypes.byref(size)
        ):
            buffer = ctypes.create_unicode_buffer(size.value)
            if kernel32.GetUserPreferredUILanguages(
                flags, ctypes.byref(count), buffer, ctypes.byref(size)
            ):
                # ctypes' stubs type a buffer slice as a list; at runtime it is
                # the str the API wrote, so the cast is the honest annotation.
                for entry in cast(str, buffer[: size.value]).split("\x00"):
                    tag = normalise_locale(entry)
                    if tag is not None:
                        return tag
    except Exception:
        pass
    try:
        import locale as _locale

        tag = normalise_locale(_locale.getlocale()[0])
        if tag is not None:
            return tag
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def _kill_switch(environ: Mapping[str, str] | None) -> bool:
    """``LOP_I18N=0`` forces ``en`` — the RFC §9 operational control.

    Checked BEFORE every other source so a deployment can pin English without
    editing config, and independent of `language:` so it wins over it. The
    catalogue route keeps serving whatever it serves; this governs resolution.
    """
    source = environ if environ is not None else os.environ
    return source.get("LOP_I18N") == "0"


def resolve_language(
    config: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    system: str | None = None,
    home: Path | None = None,
) -> str:
    """Resolve the active locale for this process/call (never raises).

    ``config=None`` reads the config file; pass ``"auto"`` to exercise the
    chain without a config read (tests, `exec`). The result is always a
    SHIPPED locale — today that means ``"en"`` — so every consumer can treat
    it as renderable.
    """
    if _kill_switch(environ):
        return "en"
    if config is None:
        config = config_value()
    candidate: str | None = None
    if isinstance(config, str) and config != DEFAULT_LANGUAGE:
        # An explicit tag wins over everything below it. `auto` (and anything
        # that does not normalise) falls through.
        candidate = normalise_locale(config)
    if candidate is None:
        candidate = env_value(environ)
    if candidate is None:
        candidate = os_language(system=system, environ=environ, home=home)
    if candidate is None:
        return "en"
    return candidate if candidate in shipped_locales() else "en"


def language_options() -> tuple[tuple[str, str], ...]:
    """``(value, label)`` pairs for the `/settings` row: ``auto`` + shipped.

    The picker lists only shipped locales (RFC §3.1: unaudited locales are
    not offered), so this tuple grows automatically when a wave's ledger
    entries pass and ``data/shipped.json`` is regenerated — no code change.
    """
    options: list[tuple[str, str]] = [(DEFAULT_LANGUAGE, DEFAULT_LANGUAGE)]
    options.extend((locale, locale) for locale in shipped_locales())
    return tuple(options)
