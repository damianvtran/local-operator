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

    def test_cell_fit_helpers_are_sinks(self, i18n_check) -> None:
        """Copy handed to a cell-fitting helper is painted, so it is counted.

        `result.append(truncate_cells("Sessions", w).ljust(w))` used to read
        clean: the literal sat inside a call inside the sink, never in the
        sink's own argument position. Only the TEXT argument counts — the
        width does not, and `fit_cells` (pixels) is not a text helper.
        """
        source = (
            'result.append(truncate_cells("Sessions", width).ljust(width))\n'
            'row = truncate_name("Untitled chat", 12)\n'
            'line = projects.truncate_row(f"{n} projects listed", cap=40)\n'
            "cols, rows = fit_cells(10, 10, 4, 4)\n"
            "plain = truncate_cells(name, width)\n"
        )
        found = i18n_check.scan_source("local_operator/fit.py", source)
        assert sorted((v.kind, v.text) for v in found) == [
            ("cell-fit truncate_cells", "Sessions"),
            ("cell-fit truncate_name", "Untitled chat"),
            ("cell-fit truncate_row", "<f-string>"),
        ]

    def test_names_resolve_to_local_and_module_bindings(self, i18n_check) -> None:
        """Copy bound to a name before the sink is followed back to its literal.

        The measured miss was `tui/widgets/session_sidebar.py`: a module
        `WEDGED_REMEDY = "..."` reached a sink through an f-string field and a
        local `label = ...` reached `truncate_cells`, so an enforced file at
        ceiling 0 rendered English. Each literal counts ONCE however many
        sinks reach it, at the literal's own line.
        """
        source = (
            'REMEDY = "lop stop if it stays silent"\n'
            "def paint(out, status, width):\n"
            '    label = "Active Sessions"\n'
            "    out.append(truncate_cells(label, width))\n"
            '    out.append(f"{status} · {REMEDY}")\n'
            "    out.append(REMEDY)\n"
            '    hint = "older rows hidden" if width < 20 else "scroll for more"\n'
            "    out.append(hint)\n"
            '    msg = "copied " + status\n'
            "    out.append(msg)\n"
        )
        found = i18n_check.scan_source("local_operator/names.py", source)
        assert sorted((v.line, v.text) for v in found) == [
            (1, "lop stop if it stays silent"),
            (3, "Active Sessions"),
            (7, "<f-string>"),
            (9, "<f-string>"),
        ]

    def test_a_local_shadow_hides_the_module_binding(self, i18n_check) -> None:
        """Parameters, loop targets and nested-scope locals bind nothing here.

        A parameter that shares a module constant's name is a different
        value; an inner function's `label = "..."` is not the outer one's.
        Following either would invent findings the reader cannot fix.
        """
        source = (
            'title = "Module heading"\n'
            "def a(out, title):\n"
            "    out.append(title)\n"
            "def b(out, rows):\n"
            "    for title in rows:\n"
            "        out.append(title)\n"
            "def c(out):\n"
            "    def inner():\n"
            '        label = "Inner only"\n'
            "        return label\n"
            "    out.append(label)\n"
        )
        assert i18n_check.scan_source("local_operator/shadow.py", source) == []

    def test_a_binding_pragma_exempts_what_it_binds(self, i18n_check) -> None:
        """The pragma may sit on the binding line — where a reader annotates it."""
        source = (
            'KEY = "ctrl+g"  # i18n: ignore key binding token\n'
            'SHOWN = "still copy"\n'
            "def f(out):\n"
            "    out.append(KEY)\n"
            "    out.append(SHOWN)\n"
            "    out.append(SHOWN)  # i18n: ignore call-span pragma still works\n"
        )
        found = i18n_check.scan_source("local_operator/pragma.py", source)
        # SHOWN is reached twice; the second call's pragma exempts only THAT
        # reach, and the first still counts the literal once.
        assert [(v.line, v.text) for v in found] == [(2, "still copy")]

    def test_dict_values_count_in_enforced_files_only(self, i18n_check) -> None:
        """Copy selected out of a dict literal counts where the file is enforced.

        `{"header:active": "Active Sessions"}[kind]` is the sidebar's header
        shape. The rule is scoped to enforced files because machine maps share
        the shape; advisory files are reported exactly as before.
        """
        source = (
            '_NAMES = {"pinned": "Pinned", "active": "Active Sessions"}\n'
            "def paint(out, kind, plat):\n"
            '    label = {"header:previous": "Previous Sessions", "header:x": "─"}[kind]\n'
            "    out.append(truncate_cells(label, 20))\n"
            "    out.append(_NAMES[kind])\n"
            '    out.append({"darwin": "cmd"}.get(plat, "the super key"))\n'
            "    out.append({'a': 1})\n"
        )
        # The default scope is the live baseline's, where the sidebar — the
        # file this rule was written for — is enforced; the literal-free stub
        # source stands in for it, so no real copy is asserted on.
        path = "local_operator/tui/widgets/session_sidebar.py"
        live = i18n_check.load_baseline(i18n_check.BASELINE)["enforced"]
        assert i18n_check.in_enforced_scope(path, live)
        enforced = i18n_check.scan_source(path, source)
        assert sorted((v.line, v.text) for v in enforced) == [
            (1, "Active Sessions"),
            (1, "Pinned"),
            (3, "Previous Sessions"),
            (6, "cmd"),
            (6, "the super key"),
        ]
        assert i18n_check.scan_source(path, source, enforced=[]) == []

    def test_what_stays_invisible(self, i18n_check) -> None:
        """The REMAINING blind spot, pinned so it is a fact and not a hope.

        Values returned from another function, attributes, imported constants
        and closures over an enclosing function's locals need inter-procedural
        or type-aware analysis. If a later change starts seeing them, this test
        fails and whoever widened it updates `_Resolver`'s docstring too.
        """
        source = (
            "from local_operator.copy import HEADING\n"
            "def title():\n"
            '    return "Returned copy"\n'
            "def outer(out):\n"
            '    label = "Closed over"\n'
            "    def inner():\n"
            "        out.append(label)\n"
            "    out.append(title())\n"
            "    out.append(self.caption)\n"
            "    out.append(HEADING)\n"
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
        #
        # A unit runner owns neither a pull-request base nor a trustworthy
        # origin/main: a depth-1 checked-out merge has no remote-tracking ref,
        # and a developer's ref can lag the audit migration. Pin both
        # base-relative gates to the checkout so this asserts the ratchet and
        # catalogue parity, and let test_audit_history's real Git bases plus
        # the dedicated i18n job (which runs this script unchanged) own the
        # delta proofs.
        assert i18n_check.main(["--audit-base", "HEAD", "--allowance-base", "HEAD"]) == 0

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

    def test_context_sidecar_shared_schema(self, tmp_path, monkeypatch, i18n_check) -> None:
        # read_public_copy refuses a whole sidecar over one bad entry, so the
        # writer must reject the same shapes at authoring time, naming the
        # field but never echoing a value (catalogues.CONTEXT_FIELDS).
        keys = ["demo.a", "demo.b", "demo.c", "demo.d", "demo.e", "demo.ok"]
        self._fixture(tmp_path, monkeypatch, {"en/demo.json": {k: "x" for k in keys}})
        sidecar = tmp_path / "catalogues" / "en" / "demo.context.json"
        sidecar.write_text(
            json.dumps(
                {
                    "demo.a": {"length_hint": "sentinel-value"},
                    "demo.b": {"placeholders": {"n": {"deep": "sentinel-value"}}},
                    "demo.c": {"width": 3},
                    "demo.d": "sentinel-value",
                    "demo.e": {"maxCells": True},
                    "demo.ok": {"width": "w", "length": "l", "placeholders": {}},
                }
            ),
            encoding="utf-8",
        )
        problems = i18n_check.check_catalogues()
        assert any("'demo.a' has unsupported field 'length_hint'" in p for p in problems)
        assert any("'demo.b' field 'placeholders' has an invalid type" in p for p in problems)
        assert any("'demo.c' field 'width' has an invalid type" in p for p in problems)
        assert any("'demo.d' must be an object" in p for p in problems)
        assert any("'demo.e' maxCells must be an int" in p for p in problems)
        assert not any("demo.ok" in p for p in problems)
        assert not any("sentinel-value" in p for p in problems)

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
