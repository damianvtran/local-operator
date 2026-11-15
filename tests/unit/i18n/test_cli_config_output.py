"""``lop config`` command output renders through ``cli.config`` in the active locale.

The commands used to print English f-strings whatever language was selected.
These tests drive the REAL command functions against a real config directory and
assert (a) the catalogue sentence for a non-English locale reaches stdout and (b)
the old English literal is gone. The French rows are a synthetic fixture (see
``synthetic_locale``): the audited translations land in a later wave.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

import pytest
import yaml

from local_operator import cli
from local_operator.i18n import catalogues, messages, text
from tests.unit.i18n import synthetic_locale

NAMESPACE = "cli.config"

_FR = {
    "cli.config.created": "Nouveau fichier de configuration créé : {path}",
    "cli.config.opened": "Fichier de configuration ouvert : {path}",
    "cli.config.cleared": "{key} réinitialisé : il reprend sa valeur par défaut",
    "cli.config.updated": "{key} mis à jour : {value}",
    "cli.config.running_sessions_unchanged": (
        "Les sessions en cours sont inchangées ; tapez /approvals auto dans chacune."
    ),
    "cli.config.list_title": "Options de configuration",
    "cli.config.list_description": "Description : {description}",
    "cli.config.list_unrecognised": "clé non reconnue ; rien ne la lit",
    "cli.config.list_desc_hosting": "Plateforme du fournisseur d'IA",
    "cli.config.list_desc_model_name": "Le modèle précis à utiliser",
    "cli.config.list_desc_conversation_length": "[OBSOLÈTE] Longueur maximale de l'historique",
    "cli.config.list_desc_detail_length": "[OBSOLÈTE] Messages récents conservés intacts",
    "cli.config.list_desc_max_learnings_history": "[OBSOLÈTE] Nombre maximal d'acquis",
    "cli.config.list_desc_classification_notice": "[OBSOLÈTE] Les indices sont silencieux",
    "cli.config.list_desc_auto_save_conversation": "Enregistrer les conversations",
}

_ENGLISH_LITERALS = (
    "Created new configuration file",
    "Opened configuration file",
    "it reads as its default again",
    "Successfully updated",
    "Running sessions are unchanged",
    "Configuration Options",
    "Description:",
    "not a recognised key",
)


@pytest.fixture
def french(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An isolated config root plus an active French locale for ``cli.config``."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "unused-home")
    synthetic_locale.install(monkeypatch, "fr", NAMESPACE, _FR)
    # The autouse ``restore_active_locale`` fixture puts English back afterwards.
    text.set_active_locale("fr")
    assert text.active_locale() == "fr"
    return tmp_path / "cfg"


def _assert_no_english(output: str) -> None:
    for literal in _ENGLISH_LITERALS:
        assert literal not in output, literal


def test_the_english_source_is_the_old_literal_set() -> None:
    """Control: with English active the same calls print the pre-extraction bytes."""
    source = catalogues.load_catalogue("en", NAMESPACE)
    assert messages.render("cli.config.created", {"path": "/x"}, locale="en") == (
        "Created new configuration file at /x"
    )
    assert source["cli.config.list_title"] == "Configuration Options"
    assert source["cli.config.updated"] == "Successfully updated {key} to {value}"


def test_create_reports_the_path_in_the_active_locale(
    french: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.config_create_command() == 0
    out = capsys.readouterr().out
    assert out.strip() == f"Nouveau fichier de configuration créé : {french / 'config.yml'}"
    _assert_no_english(out)


@pytest.mark.parametrize("opener", ["gui", "editor"])
def test_open_reports_in_the_active_locale(
    opener: str,
    french: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.config_create_command() == 0
    capsys.readouterr()
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):  # noqa: ANN001
        calls.append(list(argv))
        if opener == "editor" and len(calls) == 1:
            raise OSError("no GUI opener on this host")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setenv("EDITOR", "true")
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert cli.config_open_command() == 0
    out = capsys.readouterr().out
    assert out.strip() == f"Fichier de configuration ouvert : {french / 'config.yml'}"
    _assert_no_english(out)


def test_edit_receipts_render_in_the_active_locale(
    french: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.config_edit_command(argparse.Namespace(key="hosting", value="openai")) == 0
    out = capsys.readouterr().out
    assert out.strip() == "hosting mis à jour : openai"
    _assert_no_english(out)
    # The stored value is what landed: only the sentence moved.
    assert yaml.safe_load((french / "config.yml").read_text())["values"]["hosting"] == "openai"

    # An empty argument UNSETS a key flagged ``empty_unsets``.
    key = "network.memory.exclude_devices"
    assert cli.config_edit_command(argparse.Namespace(key=key, value="")) == 0
    out = capsys.readouterr().out
    assert out.strip() == f"{key} réinitialisé : il reprend sa valeur par défaut"
    _assert_no_english(out)


def test_loosening_note_renders_in_the_active_locale(
    french: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.config_edit_command(argparse.Namespace(key="tool_approval_mode", value="auto")) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == _FR["cli.config.running_sessions_unchanged"]
    _assert_no_english("\n".join(lines))


def test_list_title_label_and_unrecognised_row_render_in_the_active_locale(
    french: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    french.mkdir(parents=True, exist_ok=True)
    (french / "config.yml").write_text("values:\n  zzz_unknown_key: 1\n", encoding="utf-8")
    assert cli.config_list_command() == 0
    out = capsys.readouterr().out
    assert "Options de configuration" in out
    assert "Description : clé non reconnue ; rien ne la lit" in out
    # Every schema row carries the localized label too, not only the unknown row.
    assert out.count("Description : ") > 10
    _assert_no_english(out)


def test_legacy_row_descriptions_are_keyed_and_the_rest_stay_schema_help(
    french: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The seven restated descriptions follow the locale; other rows keep ``setting.help``.

    These rows used a module-level English dict that was consulted FIRST, so
    ``hosting`` stayed English under any language even though its schema help is
    keyed. The retired keys must also keep the scannable bracket tag (CL-16).
    """
    from local_operator import settings_io

    assert cli.config_list_command() == 0
    out = capsys.readouterr().out
    expected = {
        "hosting": "Plateforme du fournisseur d'IA",
        "model_name": "Le modèle précis à utiliser",
        "conversation_length": "[OBSOLÈTE] Longueur maximale de l'historique",
        "detail_length": "[OBSOLÈTE] Messages récents conservés intacts",
        "max_learnings_history": "[OBSOLÈTE] Nombre maximal d'acquis",
        "classification.notice": "[OBSOLÈTE] Les indices sont silencieux",
        "auto_save_conversation": "Enregistrer les conversations",
    }
    for key, sentence in expected.items():
        block = out[out.index(f"│ {key}:") :].split("\n")
        assert f"│   Description : {sentence}" in block[1], key
    for english in (
        "AI provider platform",
        "The specific model to use",
        "DEPRECATED",
        "Whether to automatically save",
    ):
        assert english not in out, english
    # A key outside the legacy table still shows its keyed schema help.
    other = next(s for s in settings_io.SETTINGS if s.key not in expected)
    assert f"Description : {other.help}" in out


def test_english_list_keeps_the_pre_extraction_legacy_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Control: English output of the seven restated rows is byte-identical to before."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "unused-home")
    assert text.active_locale() == "en"
    assert cli.config_list_command() == 0
    out = capsys.readouterr().out
    for line in (
        "Description: AI provider platform (e.g., radient, openai, deepseek, anthropic, "
        "openrouter)",
        "Description: The specific model to use for interactions",
        "Description: [DEPRECATED — superseded by compaction] Maximum number of messages to "
        "keep in conversation history",
        "Description: [DEPRECATED — superseded by compaction] Number of recent messages to "
        "leave unsummarized in conversation history",
        "Description: [DEPRECATED — superseded by compaction] Maximum number of learning "
        "entries to retain",
        "Description: [DEPRECATED] Smart hints is silent in the chat; the call and its cost "
        "are in the session log",
        "Description: Whether to automatically save conversations",
    ):
        assert f"│   {line}" in out, line
