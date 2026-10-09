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
        } <= kinds
        # `Setting(help=...)` is both the `help=` rule and the registry-copy
        # rule; it must be counted ONCE (per literal, not per rule).
        assert len([v for v in found if v.text == "h prose"]) == 1

    def test_f_strings_count_and_tokens_do_not(self, i18n_check) -> None:
        source = 'print(f"Session {name} failed")\nprint(f"{a}-{b}")\n'
        found = i18n_check.scan_source("local_operator/y.py", source)
        assert [v.kind for v in found] == ["print"]

    def test_pragma_exempts_its_line_only(self, i18n_check) -> None:
        source = 'print("kept")  # i18n: ignore dev log\nprint("counted")\n'
        found = i18n_check.scan_source("local_operator/z.py", source)
        assert [v.text for v in found] == ["counted"]

    def test_reason_less_pragma_does_not_exempt(self, i18n_check) -> None:
        source = 'print("no reason here")  # i18n: ignore\n'
        found = i18n_check.scan_source("local_operator/z2.py", source)
        assert len(found) == 1


class TestRatchet:
    def test_init_then_green_then_new_file_fails(self, tree, i18n_check) -> None:
        _write(
            tree,
            {
                "local_operator/old.py": 'print("existing")\n',
            },
        )
        assert i18n_check.main(["--init"]) == 0
        assert i18n_check.main([]) == 0
        # A NEW file with a literal is a failure: new files start at 0.
        _write(tree, {"local_operator/new.py": 'print("brand new prose")\n'})
        assert i18n_check.main([]) == 1
        # And its ceiling cannot be recorded.
        assert i18n_check.main(["--update", "local_operator/new.py"]) == 1

    def test_update_may_only_shrink(self, tree, i18n_check) -> None:
        _write(tree, {"local_operator/old.py": 'print("existing")\n'})
        assert i18n_check.main(["--init"]) == 0
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

    def test_the_real_tree_passes_the_ratchet(self, i18n_check) -> None:
        # The committed baseline matches the committed tree — the unit-test
        # half of RFC §2.7's "run as a unit test and as a CI job".
        assert i18n_check.main([]) == 0


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
