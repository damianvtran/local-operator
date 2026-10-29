"""The ratchet and parity checker: kinds, pragmas, allowlist, refusals, parity.

Also the two REAL-TREE runs (the ratchet as a unit test, and the catalogue
parity of the shipped tree) — that is the "wire as a unit test" half of RFC
§2.7; the CI job is the other half.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _activate(tree: Path, prefixes: list[str]) -> None:
    """Point a fixture's enforcement scope at its mini-tree (test helper).

    The default scope (the i18n package + tooling) matches nothing under the
    mini trees, so every fixture finding would be advisory; tests that want a
    BLOCKING finding activate the fixture path first — the same one-line
    activation an extraction slice performs on the real baseline.
    """
    path = tree / "i18n" / "baseline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["enforced"] = prefixes
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture()
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, i18n_check) -> Path:
    (tmp_path / "local_operator").mkdir()
    i18n_dir = tmp_path / "i18n"
    i18n_dir.mkdir()
    allowlist = i18n_dir / "allowlist.toml"
    allowlist.write_text("entries = []\n", encoding="utf-8")
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    monkeypatch.setattr(i18n_check, "BASELINE", i18n_dir / "baseline.json")
    monkeypatch.setattr(i18n_check, "ALLOWLIST", allowlist)
    return tmp_path


class TestScanning:
    SOURCE = """\
print("printed prose")
console.print("console prose")
rows.append("appended prose")
body.update("updated prose")
parser.add_argument("--flag", help="help prose")
raise HTTPException(404, "http positional prose")
raise HTTPException(404, detail="http detail prose")
return CRUDResponse(status=200, message="crud prose")
yield NoticeEvent(text="notice prose", kind="info")
setting = Setting(key="k", path=("k",), section="s", label="label prose", help="h prose")
cmd = SlashCommand("name", "slash description prose")
choice = Choice("v", "choice label prose", "choice detail prose")
text = Text("constructed prose")
static = Static("static constructor prose")
label = Label("label constructor prose")
input = Input("input initial prose", placeholder="input placeholder prose")
custom = Custom(placeholder="custom placeholder prose")
raise SessionSpecError("invalid session spec prose")
raise sdk.SessionIsolationError("unsafe session isolation prose")
"""

    def test_every_sink_kind_counts(self, i18n_check) -> None:
        found = i18n_check.scan_source("local_operator/x.py", self.SOURCE)
        kinds = {v.kind for v in found}
        assert {
            "print",
            "textual.append",
            "textual.update",
            "help=",
            "HTTPException",
            "CRUDResponse",
            "NoticeEvent",
            "Setting copy",
            "SlashCommand copy",
            "Choice copy",
            "Text",
            "textual.Static",
            "textual.Label",
            "textual.Input",
            "placeholder=",
            "SDK session error",
        } <= kinds
        # `Setting(help=...)` is both the `help=` rule and the registry-copy
        # rule; it must be counted ONCE (per literal, not per rule).
        assert len([v for v in found if v.text == "h prose"]) == 1

    def test_sdk_error_sink_is_narrow(self, i18n_check) -> None:
        source = """\
raise SessionSpecError(copy.invalid_spec())
raise sdk.SessionIsolationError(copy.isolation_error(reason=reason))
raise SessionSpecError("{}")
raise ValueError("developer only exception prose")
"""
        assert i18n_check.scan_source("local_operator/sdk.py", source) == []

    def test_f_strings_count_and_tokens_do_not(self, i18n_check) -> None:
        source = 'print(f"Session {name} failed")\nprint(f"{a}-{b}")\n'
        found = i18n_check.scan_source("local_operator/y.py", source)
        assert [v.kind for v in found] == ["print"]

    def test_assembled_copy_counts_in_argument_position(self, i18n_check) -> None:
        """`+` and the ternary fold: both reach a sink as copy.

        These are the two constructs the pre-fold scanner could not see — the
        documented blind spot that let `"allowed " if allowed else "denied "`
        and `"effort " + value` sit outside the ratchet in already-activated
        files. One finding per CALL, not per operand: the expression is one
        piece of copy however many constants it is built from.
        """
        source = (
            'bits.append("no matching models" if query else "no models available")\n'
            'bits.append("effort " + (join(xs) or "none"))\n'
        )
        found = i18n_check.scan_source("local_operator/assembled.py", source)
        assert [v.kind for v in found] == ["textual.append", "textual.append"]

    def test_a_name_reference_stays_invisible(self, i18n_check) -> None:
        """The REMAINING blind spot, pinned so it is a fact and not a hope.

        Copy assigned or returned before it reaches a sink is not counted; nor
        is a value selected by `dict.get`. Widening to those needs data-flow
        analysis rather than another syntactic fold. This test is the honest
        statement of the limit: if a later change starts seeing them, this
        test fails and whoever widened it updates the docstring too.
        """
        source = (
            'msg = "copied to clipboard" + suffix\n'
            "out.append(msg)\n"
            'out.append({"darwin": "cmd"}.get(plat, "super"))\n'
        )
        assert i18n_check.scan_source("local_operator/refs.py", source) == []

    def test_a_rule_string_alone_is_not_prose(self, i18n_check) -> None:
        """A drawn rule carries no letters, so it never counts on its own."""
        source = 'text = Text("\u2500" * width)\nprint("\u2500" * width)\n'
        assert i18n_check.scan_source("local_operator/rule.py", source) == []

    def test_pragma_exempts_its_line_only(self, i18n_check) -> None:
        source = 'print("kept")  # i18n: ignore dev log\nprint("counted")\n'
        found = i18n_check.scan_source("local_operator/z.py", source)
        assert [v.text for v in found] == ["counted"]

    def test_reason_less_pragma_does_not_exempt(self, i18n_check) -> None:
        source = 'print("no reason here")  # i18n: ignore\n'
        found = i18n_check.scan_source("local_operator/z2.py", source)
        assert len(found) == 1


class TestRatchet:
    @pytest.mark.parametrize("constructor", ["SessionSpecError", "sdk.SessionIsolationError"])
    def test_sdk_error_ratchet(self, tree, i18n_check, monkeypatch, constructor):
        path = "local_operator/sdk.py"
        clean = f"raise {constructor}(copy.invalid_spec())\n"
        _write(tree, {path: clean})
        monkeypatch.setattr(i18n_check, "check_catalogues", lambda: [])
        assert i18n_check.main(["--init"]) == 0
        _activate(tree, [path])
        _write(tree, {path: f'raise {constructor}("planted SDK error prose")\n'})
        assert i18n_check.main([]) == 1
        _write(tree, {path: clean})
        assert i18n_check.main([]) == 0

    @pytest.mark.parametrize(
        "literal",
        ['Static("planted picker prose")', 'Input(placeholder="planted picker prose")'],
    )
    def test_projects_picker_constructor_ratchet(self, tree, i18n_check, monkeypatch, literal):
        """Activate only the picker; constructor/placeholder copy cannot bypass it."""
        path = "local_operator/tui/widgets/projects_send.py"
        _write(tree, {path: "Static(copy.send_title())\n"})
        monkeypatch.setattr(i18n_check, "check_catalogues", lambda: [])
        assert i18n_check.main(["--init"]) == 0
        _activate(tree, [path])
        _write(tree, {path: literal + "\n"})
        assert i18n_check.main([]) == 1
        _write(tree, {path: "Static(copy.send_title())\n"})
        assert i18n_check.main([]) == 0

    def test_new_file_outside_the_enforced_scope_is_advisory(
        self, tree, i18n_check, capsys
    ) -> None:
        # Round-3 C1: outside the enforced scope a NEW file's literals are
        # REPORTED but never fail — what keeps unrelated PRs in a concurrent
        # fleet green before their extraction slice exists.
        _write(tree, {"local_operator/old.py": 'print("existing")\n'})
        assert i18n_check.main(["--init"]) == 0
        assert i18n_check.main([]) == 0
        _write(tree, {"local_operator/new.py": 'print("brand new prose")\n'})
        assert i18n_check.main([]) == 0
        out = capsys.readouterr().out
        assert "advisory: local_operator/new.py: 1 literal(s)" in out

    def test_new_file_inside_the_enforced_scope_fails(self, tree, i18n_check) -> None:
        _write(tree, {"local_operator/old.py": 'print("existing")\n'})
        assert i18n_check.main(["--init"]) == 0
        _activate(tree, ["local_operator/"])
        assert i18n_check.main([]) == 0
        # Inside the scope a NEW file with a literal is a failure: new files
        # start at 0.
        _write(tree, {"local_operator/new.py": 'print("brand new prose")\n'})
        assert i18n_check.main([]) == 1
        # And its ceiling cannot be recorded.
        assert i18n_check.main(["--update", "local_operator/new.py"]) == 1

    def test_update_may_only_shrink(self, tree, i18n_check) -> None:
        _write(tree, {"local_operator/old.py": 'print("existing")\n'})
        assert i18n_check.main(["--init"]) == 0
        _activate(tree, ["local_operator/"])
        _write(tree, {"local_operator/old.py": 'print("existing")\nprint("added")\n'})
        assert i18n_check.main([]) == 1
        assert i18n_check.main(["--update", "local_operator/old.py"]) == 1
        # Extraction lowers it: pragma, then the update is accepted (== old).
        _write(
            tree,
            {"local_operator/old.py": ('print("existing")\nprint("added")  # i18n: ignore x\n')},
        )
        assert i18n_check.main(["--update", "local_operator/old.py"]) == 0
        assert i18n_check.main([]) == 0

    def test_init_refuses_to_rebaseline_silently(self, tree, i18n_check) -> None:
        _write(tree, {"local_operator/old.py": 'print("existing")\n'})
        assert i18n_check.main(["--init"]) == 0
        assert i18n_check.main(["--init"]) == 2
        assert i18n_check.main(["--init", "--force"]) == 0
        # round-1 n3: an existing baseline that is EMPTY is still an existing
        # baseline — a truthiness test used to treat it as absent and
        # overwrite it without --force.
        (tree / "i18n" / "baseline.json").write_text("{}\n", encoding="utf-8")
        assert i18n_check.main(["--init"]) == 2
        assert i18n_check.main(["--init", "--force"]) == 0
        # Round-3 C1: a forced re-snapshot preserves the REVIEW STATE — an
        # activated scope must survive it (only counts are re-captured).
        _activate(tree, ["local_operator/old.py"])
        assert i18n_check.main(["--init", "--force"]) == 0
        saved = json.loads((tree / "i18n" / "baseline.json").read_text(encoding="utf-8"))
        assert saved["enforced"] == ["local_operator/old.py"]

    def test_malformed_enforced_fails_closed(self, tree, i18n_check) -> None:
        # Round-4 m1: a hand-edited, present-but-NON-list `enforced` used to
        # become [] (silently un-enforcing everything); it must fall back to
        # DEFAULT_ENFORCED, while a deliberate [] stays a real state.
        assert i18n_check.main(["--init"]) == 0
        _write(tree, {"local_operator/i18n/x.py": 'print("prose")\n'})
        path = tree / "i18n" / "baseline.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["enforced"] = []  # deliberate: nothing enforced -> advisory only
        path.write_text(json.dumps(data), encoding="utf-8")
        assert i18n_check.main([]) == 0
        data["enforced"] = "local_operator/i18n/"  # a hand-edit mistake
        path.write_text(json.dumps(data), encoding="utf-8")
        assert i18n_check.main([]) == 1

    def test_allowlisted_file_is_exempt(self, tree, i18n_check) -> None:
        _write(
            tree,
            {
                "local_operator/devlog.py": 'print("pure dev logging")\n',
                "i18n/allowlist.toml": 'entries = [["local_operator/devlog.py", "dev logs"]]\n',
            },
        )
        assert i18n_check.main(["--init"]) == 0
        assert i18n_check.main([]) == 0
        # The reason is REQUIRED.
        (tree / "i18n" / "allowlist.toml").write_text(
            'entries = [["local_operator/devlog.py", ""]]\n', encoding="utf-8"
        )
        with pytest.raises(SystemExit):
            i18n_check.load_allowlist(tree / "i18n" / "allowlist.toml")
        # A directory prefix covers a subtree; there are no globs by design.
        (tree / "i18n" / "allowlist.toml").write_text(
            'entries = [["local_operator/", "whole dev tree"]]\n', encoding="utf-8"
        )
        assert i18n_check.main([]) == 0

    def test_enforced_scope_matching(self, i18n_check) -> None:
        assert i18n_check.in_enforced_scope(
            "local_operator/i18n/runtime.py", ["local_operator/i18n/"]
        )
        assert not i18n_check.in_enforced_scope(
            "local_operator/tui/app.py", ["local_operator/i18n/"]
        )
        assert i18n_check.in_enforced_scope("i18n/x.py", ["i18n/x.py"])
        assert not i18n_check.in_enforced_scope("i18n/xy.py", ["i18n/x.py"])

    def test_the_real_tree_passes_the_ratchet(self, i18n_check) -> None:
        # The committed baseline matches the committed tree — the unit-test
        # half of RFC §2.7's "run as a unit test and as a CI job". Findings
        # OUTSIDE the enforced scope never fail (round-3 C1), so this also
        # pins that a fleet-grown tree cannot break the suite.
        assert i18n_check.main([]) == 0

    def test_the_enforced_scope_itself_is_finding_free(self, i18n_check) -> None:
        # The scope M0 activates (the i18n package + its tooling) must carry no
        # RATCHET FINDINGS — every file at or below its ceiling. Not "zero
        # literals": the frozen developer-diagnostic prints in scripts/i18n/
        # are legitimate ceilings; what fails is growth, and this pins none.
        baseline = i18n_check.load_baseline(i18n_check.BASELINE)
        entries = i18n_check.load_allowlist(i18n_check.ALLOWLIST)
        scanned = i18n_check.scan_tree(i18n_check.REPO)
        counts = i18n_check._counts(scanned, entries)
        scoped = [
            finding.path
            for finding in i18n_check.findings(counts, baseline["files"])
            if i18n_check.in_enforced_scope(finding.path, baseline["enforced"])
        ]
        assert scoped == []


class TestCatalogueParity:
    def _fixture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, dict[str, str]]
    ) -> None:
        root = tmp_path / "catalogues"
        for rel, data in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr(catalogues, "_CATALOGUES", root)

    def test_consistent_catalogues_pass(self, tmp_path, monkeypatch, i18n_check) -> None:
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{n, plural, one {# item} other {# items}}"},
                "fr/demo.json": {
                    "demo.x": (
                        "{n, plural, one {# article} many {# articles} " "other {# articles}}"
                    )
                },
            },
        )
        assert i18n_check.check_catalogues() == []

    def test_locale_missing_a_namespace_passes(self, tmp_path, monkeypatch, i18n_check) -> None:
        """Per-namespace coverage (2026-10-11 amendment, docs/design/i18n.md §6):
        a namespace born after a locale's last translation wave is not an error
        — it rides en-fallback until its own wave, and the per-namespace ship
        gate decides whether the locale is offered."""
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "x"},
                "en/newborn.json": {"newborn.y": "y"},
                "fr/demo.json": {"demo.x": "t"},
            },
        )
        assert i18n_check.check_catalogues() == []

    def test_present_file_with_drift_still_fails(self, tmp_path, monkeypatch, i18n_check) -> None:
        # The flip side: scope narrowed, judgement unchanged — a file the
        # locale PRESENTS is validated against en exactly as before.
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "x"},
                "en/newborn.json": {"newborn.y": "y"},
                "fr/demo.json": {"demo.x": "t"},
                "fr/newborn.json": {},
            },
        )
        problems = i18n_check.check_catalogues()
        assert any("missing key 'newborn.y'" in p for p in problems)

    def test_key_set_mismatch(self, tmp_path, monkeypatch, i18n_check) -> None:
        self._fixture(
            tmp_path,
            monkeypatch,
            {"en/demo.json": {"demo.x": "x"}, "fr/demo.json": {}},
        )
        problems = i18n_check.check_catalogues()
        assert any("missing key 'demo.x'" in p for p in problems)

    def test_argument_mismatch(self, tmp_path, monkeypatch, i18n_check) -> None:
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{count} things"},
                "fr/demo.json": {"demo.x": "{count} {extra} choses"},
            },
        )
        problems = i18n_check.check_catalogues()
        assert any("arguments" in p for p in problems)

    def test_number_may_inflect_as_a_plural(self, tmp_path, monkeypatch, i18n_check) -> None:
        """A count en writes as a bare number may become an ICU plural.

        Urdu `گھنٹہ`/`گھنٹے` and Hindi `घंटा`/`घंटे` have no invariant form, so
        `{count, number} h` cannot render "3 hours" correctly — the upgrade
        keeps the same argument, position and number value."""
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{count, number} h"},
                "ur/demo.json": {"demo.x": "{count, plural, one {# گھنٹہ} other {# گھنٹے}}"},
            },
        )
        assert i18n_check.check_catalogues() == []

    @pytest.mark.parametrize(
        ("locale", "translation"),
        [
            # ur's CLDR set is one/other: `other` alone cannot render count 1.
            ("ur", "{count, plural, other {# گھنٹے}}"),
            # ru's set is one/few/many/other: `one` is missing even though the
            # other three are present, so this pins the FULL-set rule rather
            # than merely "more than one branch".
            ("ru", "{count, plural, few {# часа} many {# часов} other {# часа}}"),
        ],
    )
    def test_number_upgrade_requires_every_locale_category(
        self, tmp_path, monkeypatch, i18n_check, locale: str, translation: str
    ) -> None:
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{count, number} h"},
                f"{locale}/demo.json": {"demo.x": translation},
            },
        )
        problems = i18n_check.check_catalogues()
        assert any("missing locale categories" in p and "'one'" in p for p in problems)

    def test_number_upgrade_accepts_the_full_locale_category_set(
        self, tmp_path, monkeypatch, i18n_check
    ) -> None:
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{count, number} h"},
                "ru/demo.json": {
                    "demo.x": (
                        "{count, plural, one {# час} few {# часа} " "many {# часов} other {# часа}}"
                    )
                },
            },
        )
        assert i18n_check.check_catalogues() == []

    def test_number_upgrade_requires_the_value_in_every_branch(
        self, tmp_path, monkeypatch, i18n_check
    ) -> None:
        # `one` says "hour" but never renders the count, so count 1 loses its
        # number from the sentence — accepted before this guard existed.
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{count, number} h"},
                "ur/demo.json": {"demo.x": "{count, plural, one {گھنٹہ} other {# گھنٹے}}"},
            },
        )
        problems = i18n_check.check_catalogues()
        assert any("does not render the value" in p for p in problems)

    def test_number_upgrade_rejects_an_empty_branch(
        self, tmp_path, monkeypatch, i18n_check
    ) -> None:
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{count, number} h"},
                "ur/demo.json": {"demo.x": "{count, plural, one {# گھنٹہ} other {}}"},
            },
        )
        problems = i18n_check.check_catalogues()
        assert any("branch 'other' is empty" in p for p in problems)

    def test_reordered_arguments_fail(self, tmp_path, monkeypatch, i18n_check) -> None:
        # Order is part of the contract: `{second} then {first}` renders the
        # wrong value in each slot even though both names survive.
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{first} then {second}"},
                "fr/demo.json": {"demo.x": "{second} then {first}"},
            },
        )
        problems = i18n_check.check_catalogues()
        assert any("renamed or reordered" in p for p in problems)

    def test_number_may_not_change_to_date(self, tmp_path, monkeypatch, i18n_check) -> None:
        # A translator must not change what a value IS. `number -> plural` is
        # the sole permitted kind change; `number -> date` would render a
        # count as a calendar date.
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{count, number} h"},
                "fr/demo.json": {"demo.x": "{count, date} h"},
            },
        )
        problems = i18n_check.check_catalogues()
        assert any("changed kind 'number' -> 'date'" in p for p in problems)

    def test_a_plural_may_not_collapse_to_a_bare_number(
        self, tmp_path, monkeypatch, i18n_check
    ) -> None:
        """The reverse is still a failure: dropping branches would lose the
        grammar the en source already expresses."""
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{n, plural, one {# item} other {# items}}"},
                "fr/demo.json": {"demo.x": "{n, number} articles"},
            },
        )
        problems = i18n_check.check_catalogues()
        assert any("arguments" in p for p in problems)

    def test_number_upgrade_keeps_the_argument_name(
        self, tmp_path, monkeypatch, i18n_check
    ) -> None:
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{count, number} h"},
                "ur/demo.json": {"demo.x": "{n, plural, one {# گھنٹہ} other {# گھنٹے}}"},
            },
        )
        problems = i18n_check.check_catalogues()
        assert any("arguments" in p for p in problems)

    def test_invalid_plural_category_for_the_locale(
        self, tmp_path, monkeypatch, i18n_check
    ) -> None:
        # `few` exists in ru, not in fr.
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "{n, plural, one {# a} other {# b}}"},
                "fr/demo.json": {"demo.x": "{n, plural, one {# a} few {# c} other {# b}}"},
            },
        )
        problems = i18n_check.check_catalogues()
        assert any("'few' not in" in p for p in problems)

    def test_missing_other_branch(self, tmp_path, monkeypatch, i18n_check) -> None:
        self._fixture(tmp_path, monkeypatch, {"en/demo.json": {"demo.x": "{n, plural, one {# a}}"}})
        problems = i18n_check.check_catalogues()
        assert any("no `other` branch" in p for p in problems)

    def test_key_naming_and_namespace_prefix(self, tmp_path, monkeypatch, i18n_check) -> None:
        self._fixture(
            tmp_path,
            monkeypatch,
            {"en/demo.json": {"BadKey": "x", "other.namespace.y": "y"}},
        )
        problems = i18n_check.check_catalogues()
        assert any("not lower snake_case" in p for p in problems)
        assert any("namespace prefix" in p for p in problems)

    def test_context_sidecar_validation(self, tmp_path, monkeypatch, i18n_check) -> None:
        self._fixture(tmp_path, monkeypatch, {"en/demo.json": {"demo.x": "x"}})
        sidecar = tmp_path / "catalogues" / "en" / "demo.context.json"
        sidecar.write_text(
            json.dumps({"demo.x": {"maxCells": "nope"}, "demo.gone": {"role": "r"}}),
            encoding="utf-8",
        )
        problems = i18n_check.check_catalogues()
        assert any("unknown key" in p for p in problems)
        assert any("maxCells" in p for p in problems)

    def test_the_shipped_catalogues_pass(self, i18n_check) -> None:
        assert i18n_check.check_catalogues() == []

    def test_cross_file_duplicate_key_is_refused(self, tmp_path, monkeypatch, i18n_check) -> None:
        # NIT-1 (#2166): a key defined in both a namespace and its child
        # namespace passes the per-file prefix checks; resolution is
        # deepest-first, so without this check the duplicate would silently
        # shadow instead of failing.
        self._fixture(
            tmp_path,
            monkeypatch,
            {
                "en/demo.json": {"demo.x": "shallow", "demo.deep.y": "child"},
                "en/demo.deep.json": {"demo.deep.y": "also here"},
            },
        )
        problems = i18n_check.check_catalogues()
        duplicates = [p for p in problems if "defined by both" in p]
        assert duplicates, problems
        assert any(
            "'demo.deep.y'" in p and "'demo'" in p and "'demo.deep'" in p for p in duplicates
        )
