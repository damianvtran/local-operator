"""The tree oracle stays meaningful with zero, partial, or complete coverage.

Absence must be manufactured here, not required of the shipping corpus: the last
translation landing must not retire fallback coverage or turn a healthy tree red.
"""

import json

import pytest

from local_operator.i18n import catalogues, messages, resolve
from tests.unit.i18n import locale_coverage


@pytest.mark.parametrize("coverage", ["absent", "partial", "complete", "english-copy"])
def test_per_key_oracle_and_runtime(tmp_path, monkeypatch, coverage):
    namespace = "fallback.probe"
    english = {
        "fallback.probe.first": "English first",
        "fallback.probe.second": "{count, plural, one {EN ONE} other {EN OTHER}}",
    }
    (tmp_path / "en").mkdir()
    (tmp_path / "en" / f"{namespace}.json").write_text(json.dumps(english), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    targets = locale_coverage.supported_targets()
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", *targets))
    monkeypatch.delenv("LOP_I18N", raising=False)
    for locale in targets:
        if coverage == "absent":
            continue
        carried = {"fallback.probe.first": f"{locale} first"}
        if coverage == "complete":
            carried["fallback.probe.second"] = f"{locale} second"
        elif coverage == "english-copy":
            carried["fallback.probe.first"] = english["fallback.probe.first"]
        (tmp_path / locale).mkdir()
        (tmp_path / locale / f"{namespace}.json").write_text(json.dumps(carried), encoding="utf-8")
    for locale in targets:
        for key in english:
            if coverage == "english-copy" and key.endswith("first"):
                with pytest.raises(AssertionError, match="English-identical shipped copy"):
                    locale_coverage.expected_message(locale, namespace, key, {})
                continue
            expected = locale_coverage.expected_message(locale, namespace, key, {"count": 21})
            if coverage == "absent" or key.endswith("second") and coverage != "complete":
                assert expected == ("English first" if key.endswith("first") else "EN OTHER")
            else:
                assert expected == f"{locale} {'first' if key.endswith('first') else 'second'}"
            assert messages.render(key, {"count": 21}, locale=locale) == expected
