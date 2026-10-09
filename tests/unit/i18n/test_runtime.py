"""The ICU-subset runtime: rendering semantics, escapes, and loud errors.

The positive cases are the RFC's subset (§2.3) exercised as messages actually
look; the error cases matter as much — the parity check depends on "outside
the subset" being a parse error, not a quietly different rendering.
"""

from __future__ import annotations

import pytest

from local_operator.i18n import runtime


def render(source: str, params: dict | None = None, locale: str = "en") -> str:
    return runtime.render_message(source, params or {}, locale)


class TestInterpolation:
    def test_plain_text_passes_through(self) -> None:
        assert render("Nothing to substitute.") == "Nothing to substitute."

    def test_named_argument(self) -> None:
        assert render("Hello, {name}!", {"name": "Damian"}) == "Hello, Damian!"

    def test_numbers_are_formatted_per_locale(self) -> None:
        assert render("{n} files", {"n": 1234}, "en") == "1,234 files"
        assert render("{n} files", {"n": 1234}, "hi") == "1,234 files"
        assert render("{n}", {"n": 12345678}, "hi") == "1,23,45,678"

    def test_datetime_uses_medium_date_and_short_time(self) -> None:
        from datetime import datetime

        got = render("{when}", {"when": datetime(2025, 7, 19, 14, 5)}, "en")
        assert got == "Jul 19, 2025, 2:05 PM"


class TestPlural:
    def test_categories_pick_their_branches(self) -> None:
        src = "{count, plural, one {# session} other {# sessions}} to resume"
        assert render(src, {"count": 1}) == "1 session to resume"
        assert render(src, {"count": 3}) == "3 sessions to resume"
        assert render(src, {"count": 1234}) == "1,234 sessions to resume"

    def test_exact_selectors_win_over_categories(self) -> None:
        src = "{n, plural, =0 {no items} one {# item} other {# items}}"
        assert render(src, {"n": 0}) == "no items"
        assert render(src, {"n": 1}) == "1 item"
        assert render(src, {"n": 7}) == "7 items"

    def test_decimal_value_selects_the_locale_category(self) -> None:
        src = "{n, plural, one {# файл} few {# файла} many {# файлов} other {# файла}}"
        assert render(src, {"n": 3}, "ru") == "3 файла"
        assert render(src, {"n": 11}, "ru") == "11 файлов"
        # Every decimal is `other` in ru — and the NUMBER itself formats with
        # the locale's comma: "1,5" is the ru rendering of 1.5.
        assert render(src, {"n": 1.5}, "ru") == "1,5 файла"

    def test_hash_inside_a_nested_select_still_names_the_plural(self) -> None:
        src = "{n, plural, other {{slot, select, a {A} other {B}} + #}}"
        assert render(src, {"n": 4, "slot": "a"}) == "A + 4"

    def test_missing_branch_is_a_format_error_naming_the_category(self) -> None:
        with pytest.raises(runtime.MessageFormatError) as err:
            render("{n, plural, one {x}}", {"n": 5})
        assert "other" in str(err.value) or "category" in str(err.value)


class TestSelect:
    def test_matching_key_and_fallback(self) -> None:
        src = "{gender, select, male {He} female {She} other {They}} replied"
        assert render(src, {"gender": "female"}) == "She replied"
        assert render(src, {"gender": "anything-else"}) == "They replied"


class TestEscaping:
    def test_doubled_quote_is_an_apostrophe(self) -> None:
        assert render("It''s {n} o''clock", {"n": 5}) == "It's 5 o'clock"

    def test_quoted_run_keeps_braces_literal(self) -> None:
        assert render("'{literal}' and '{'braces'}'") == "{literal} and {braces}"

    def test_quoted_hash_is_not_a_plural_number(self) -> None:
        assert render("{n, plural, other {'#' is literal}}", {"n": 3}) == "# is literal"


class TestSyntaxErrors:
    @pytest.mark.parametrize(
        "source",
        [
            "{unclosed",
            "{}",
            "plain } brace",
            "{n, weird, one {x}}",
            "{n, plural, one {x} one {y}}",
            "{n, plural,}",
            "{n, plural, one {x} offset:1}",
        ],
    )
    def test_outside_the_subset_is_a_parse_error(self, source: str) -> None:
        with pytest.raises(runtime.MessageSyntaxError):
            runtime.parse_message(source)

    def test_unmatched_brace_error_names_an_offset(self) -> None:
        with pytest.raises(runtime.MessageSyntaxError) as err:
            runtime.parse_message("a } b")
        assert "offset" in str(err.value)


class TestIntrospection:
    def test_message_arguments_keeps_first_appearance_and_kinds(self) -> None:
        got = runtime.message_arguments("{n, plural, one {# f} other {# fs}} for {name}")
        assert got == (("n", "plural"), ("name", "simple"))

    def test_plural_selectors_lists_selector_tuples(self) -> None:
        got = runtime.plural_selectors("{n, plural, =0 {z} one {# f} other {# fs}}")
        assert got == (("=0", "one", "other"),)

    def test_parse_is_cached_per_source(self) -> None:
        first = runtime.parse_message("cached {x}")
        second = runtime.parse_message("cached {x}")
        assert first is second
