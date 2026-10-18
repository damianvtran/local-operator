"""Help tranche 6: the small families — nine blocks + the auth-status pair.

`memory`, `credential`, `config`, `aida`, `serve`, `qwencloud-ticket`,
`resume-click`, `login`, `logout`, and the `login-status`/`status` alias
pair — 33 flagged `help=` findings -> 10 namespaces.

Same ruled classes: prose -> keys; flag/command tokens in situ. **Zero
ignores**, pinned below. Three unflagged extra sites (two kinds), extracted under
the same per-site class decision and COUNT-NEUTRAL by construction:
- `credential_key_help` — a local variable holding shared help text used by
  two `help=` args; the variable-indirection blind spot of the sink scanner.
  The value is now the envelope render; the two references stay untouched.
- the `description=` texts on `config instructions` and `aida note` —
  `description=` is not a scanner sink (the `help=` rule is call-level; the
  description rule covers Setting/SlashCommand/Choice only). The sessions
  tranche extracted its descriptions the same way.

`resume-click` is the hidden parser whose omission of `help=` is the hide
idiom; its only copy is the positional's help.

The ceiling assertions DERIVE the count (equality = no slack; `<= 549`).
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.i18n import catalogues
from local_operator.i18n.keys import cli_aida_help as _ad
from local_operator.i18n.keys import cli_config_help as _cf
from local_operator.i18n.keys import cli_credential_help as _cr
from local_operator.i18n.keys import cli_login_help as _li
from local_operator.i18n.keys import cli_login_status_help as _ls
from local_operator.i18n.keys import cli_logout_help as _lo
from local_operator.i18n.keys import cli_memory_help as _me
from local_operator.i18n.keys import cli_qwencloud_help as _qc
from local_operator.i18n.keys import cli_resume_click_help as _rc
from local_operator.i18n.keys import cli_serve_help as _sv

REPO = Path(__file__).resolve().parents[3]
CLI = "local_operator/cli.py"
CEILING = 549  # the count after help tranche 6; may only fall


def _live_cli_count(i18n_check) -> int:
    entries = i18n_check.load_allowlist(i18n_check.ALLOWLIST)
    scanned = i18n_check.scan_tree(i18n_check.REPO)
    return i18n_check._counts(scanned, entries).get(CLI, 0)


def test_every_key_renders_the_byte_its_base_call_carried() -> None:
    for rendered, expected in CASES:
        assert rendered == expected, rendered


def test_the_catalogues_are_the_golden_fixtures() -> None:
    for namespace, fixture, size in (
        ("cli.memory_help", "cli_memory_help_en_golden.json", 1),
        ("cli.credential_help", "cli_credential_help_en_golden.json", 5),
        ("cli.config_help", "cli_config_help_en_golden.json", 9),
        ("cli.aida_help", "cli_aida_help_en_golden.json", 4),
        ("cli.serve_help", "cli_serve_help_en_golden.json", 5),
        ("cli.qwencloud_help", "cli_qwencloud_help_en_golden.json", 5),
        ("cli.resume_click_help", "cli_resume_click_help_en_golden.json", 1),
        ("cli.login_help", "cli_login_help_en_golden.json", 2),
        ("cli.logout_help", "cli_logout_help_en_golden.json", 2),
        ("cli.login_status_help", "cli_login_status_help_en_golden.json", 2),
    ):
        live = catalogues.load_catalogue("en", namespace)
        golden = json.loads(
            (REPO / "tests/unit/i18n/fixtures" / fixture).read_text(encoding="utf-8")
        )
        assert live == golden, f"{namespace}: shipped catalogue and fixture drifted"
        assert len(golden) == size, (namespace, len(golden))
        sidecar = catalogues.context("en", namespace)
        for key, entry in sidecar.items():
            assert all(entry.get(field) for field in ("role", "tone", "note")), key


def test_every_block_renders_through_its_envelope_and_no_ignore_exists() -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    # Per-SITE shape pins (the teams-slice precedent): every accessor call must
    # carry the exact bootstrap form, not merely the right count.
    for alias, key in PINS:
        shape = f'{alias}.{key}().envelope(locale=help_locale)["text"]'
        assert shape in source, shape
    # Tranche 6 adds zero ignores: every `# i18n: ignore` pragma in the file
    # (the pre-existing browser-era set, elsewhere in cli.py) must stay away
    # from every T6 accessor -- the absence is pinned per-site, not globally.
    for line in source.splitlines():
        if "# i18n: ignore" in line:
            assert not any(
                alias in line
                for alias in (
                    "_me.",
                    "_cr.",
                    "_cf.",
                    "_ad.",
                    "_sv.",
                    "_qc.",
                    "_rc.",
                    "_li.",
                    "_lo.",
                    "_ls.",
                )
            ), line


def test_the_ceiling_has_no_slack_and_stays_unactivated(i18n_check) -> None:
    baseline = json.loads((REPO / "i18n" / "baseline.json").read_text(encoding="utf-8"))
    live = _live_cli_count(i18n_check)
    assert baseline["files"].get(CLI) == live, (baseline["files"].get(CLI), live)
    assert live <= CEILING, live
    assert not i18n_check.in_enforced_scope(
        CLI, baseline["enforced"]
    ), "cli.py is activated without the file being clean"


def test_a_planted_literal_fails_the_checker(tmp_path, monkeypatch, i18n_check) -> None:
    source = (REPO / CLI).read_text(encoding="utf-8")
    target = tmp_path / CLI
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    baseline_dir = tmp_path / "i18n"
    baseline_dir.mkdir()
    baseline = baseline_dir / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "note": "slice fixture",
                "enforced": [CLI],
                "files": {CLI: _live_cli_count(i18n_check)},
            }
        )
    )
    allowlist = baseline_dir / "allowlist.toml"
    allowlist.write_text("entries = []\n")
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    monkeypatch.setattr(i18n_check, "BASELINE", baseline)
    monkeypatch.setattr(i18n_check, "ALLOWLIST", allowlist)
    monkeypatch.setattr(i18n_check, "check_catalogues", lambda: [])
    assert i18n_check.main([]) == 0
    target.write_text(source + '\nprint("planted small-family prose")\n', encoding="utf-8")
    assert i18n_check.main([]) == 1
    target.write_text(source, encoding="utf-8")
    assert i18n_check.main([]) == 0


# (accessor alias, key) for every T6 site -- the per-site shape pins above.
PINS = [
    ("_me", "memory"),
    ("_cr", "credential"),
    ("_cr", "update"),
    ("_cr", "delete"),
    ("_cr", "git_helper"),
    ("_cr", "key"),
    ("_cf", "config"),
    ("_cf", "open"),
    ("_cf", "edit"),
    ("_cf", "edit_key"),
    ("_cf", "edit_value"),
    ("_cf", "list"),
    ("_cf", "create"),
    ("_cf", "instructions"),
    ("_cf", "instructions_description"),
    ("_ad", "aida"),
    ("_ad", "note"),
    ("_ad", "note_description"),
    ("_ad", "note_text"),
    ("_sv", "serve"),
    ("_sv", "host"),
    ("_sv", "port"),
    ("_sv", "reload"),
    ("_sv", "listener_fd"),
    ("_qc", "qwencloud_ticket"),
    ("_qc", "set"),
    ("_qc", "status"),
    ("_qc", "rm"),
    ("_qc", "migrate"),
    ("_rc", "session"),
    ("_li", "login"),
    ("_li", "provider"),
    ("_lo", "logout"),
    ("_lo", "provider"),
    ("_ls", "login_status"),
    ("_ls", "status_alias"),
]
assert len(PINS) == 36


CASES = [
    (
        _me.memory().envelope(locale="en")["text"],
        "Inspect and correct scoped memory",
    ),  # memory_help.memory
    (
        _cr.credential().envelope(locale="en")["text"],
        "Manage API keys and credentials for different hosting platforms",
    ),  # credential_help.credential
    (
        _cr.update().envelope(locale="en")["text"],
        "Update a credential",
    ),  # credential_help.update
    (
        _cr.delete().envelope(locale="en")["text"],
        "Delete a credential",
    ),  # credential_help.delete
    (
        _cr.git_helper().envelope(locale="en")["text"],
        (
            "Serve the git credential protocol from the brokered environment (invok"
            "ed by git; not for direct use)"
        ),
    ),  # credential_help.git_helper
    (
        _cr.key().envelope(locale="en")["text"],
        (
            "Credential key to manage (e.g., RADIENT_API_KEY,DEEPSEEK_API_KEY, OPEN"
            "AI_API_KEY, ANTHROPIC_API_KEY, KIMI_API_KEY, ALIBABA_CLOUD_API_KEY, GO"
            "OGLE_AI_STUDIO_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY, XAI_API_K"
            "EY)"
        ),
    ),  # credential_help.key
    (
        _cf.config().envelope(locale="en")["text"],
        "Manage configuration settings",
    ),  # config_help.config
    (
        _cf.open().envelope(locale="en")["text"],
        "Open the configuration file in the default editor",
    ),  # config_help.open
    (
        _cf.edit().envelope(locale="en")["text"],
        "Edit a specific configuration value in the config file",
    ),  # config_help.edit
    (
        _cf.edit_key().envelope(locale="en")["text"],
        (
            "Configuration key to update (e.g., hosting, model_name, conversation_l"
            "ength, detail_length, max_learnings_history, auto_save_conversation)"
        ),
    ),  # config_help.edit_key
    (
        _cf.edit_value().envelope(locale="en")["text"],
        "New value for the configuration key (type is automatically converted based on the key)",
    ),  # config_help.edit_value
    (
        _cf.list().envelope(locale="en")["text"],
        "List available configuration options and their descriptions",
    ),  # config_help.list
    (
        _cf.create().envelope(locale="en")["text"],
        "Create a new configuration file",
    ),  # config_help.create
    (
        _cf.instructions().envelope(locale="en")["text"],
        (
            "Show which custom-instruction files a session assembles, in order (pat"
            "hs and sizes only, never their contents)"
        ),
    ),  # config_help.instructions
    (
        _cf.instructions_description().envelope(locale="en")["text"],
        (
            "Show which custom-instruction files a session assembles, in order, wit"
            "h the size each contributed and whether it was collapsed as a duplicat"
            "e, truncated, or overlaps an earlier source. Paths and sizes only, nev"
            "er the contents."
        ),
    ),  # config_help.instructions_description
    (
        _ad.aida().envelope(locale="en")["text"],
        "Aida's profile notes: record what she should remember",
    ),  # aida_help.aida
    (
        _ad.note().envelope(locale="en")["text"],
        "Append a note under 'About the operator' in system_prompt.md",
    ),  # aida_help.note
    (
        _ad.note_description().envelope(locale="en")["text"],
        (
            "Record a note into <config root>/system_prompt.md, inside the marked '"
            "About the operator' section, so every future session and subagent sees"
            " it. The file is resolved from the config root and is never an argumen"
            "t; the note must not contain the section markers. Omit TEXT to read th"
            "e note from STDIN (multi-line notes need no quoting)."
        ),
    ),  # aida_help.note_description
    (
        _ad.note_text().envelope(locale="en")["text"],
        "The note to record; omit to read it from STDIN",
    ),  # aida_help.note_text
    (
        _sv.serve().envelope(locale="en")["text"],
        "Start the FastAPI server",
    ),  # serve_help.serve
    (
        _sv.host().envelope(locale="en")["text"],
        (
            "Host address (default: 127.0.0.1). Loopback-only by policy: every non-"
            "loopback bind is refused. File serving is unclamped on loopback connec"
            "tions; reach this API remotely through an explicit tunnel or an SSH po"
            "rt-forward."
        ),
    ),  # serve_help.host
    (
        _sv.port().envelope(locale="en")["text"],
        "Port for the server (default: 1111)",
    ),  # serve_help.port
    (
        _sv.reload().envelope(locale="en")["text"],
        "Enable hot reload for the server",
    ),  # serve_help.reload
    (
        _sv.listener_fd().envelope(locale="en")["text"],
        (
            "Adopt an already-bound listening socket instead of binding one. Set by"
            " a daemon replacing its own process image onto a new build (`server/re"
            "load`), so the port never has a moment with nothing behind it. Not a u"
            "ser-facing flag."
        ),
    ),  # serve_help.listener_fd
    (
        _qc.qwencloud_ticket().envelope(locale="en")["text"],
        "Store the QwenCloud console session cookie that /usage reads",
    ),  # qwencloud_help.qwencloud_ticket
    (
        _qc.set().envelope(locale="en")["text"],
        "Store the cookie; the value is read from STDIN, never argv",
    ),  # qwencloud_help.set
    (
        _qc.status().envelope(locale="en")["text"],
        "Whether a cookie is stored and how old it is; never the value",
    ),  # qwencloud_help.status
    (
        _qc.rm().envelope(locale="en")["text"],
        "Remove the stored cookie",
    ),  # qwencloud_help.rm
    (
        _qc.migrate().envelope(locale="en")["text"],
        "Move a plaintext ticket into the encrypted secret store",
    ),  # qwencloud_help.migrate
    (
        _rc.session().envelope(locale="en")["text"],
        "session id to reopen",
    ),  # resume_click_help.session
    (
        _li.login().envelope(locale="en")["text"],
        "Log in to a provider (OAuth or API key)",
    ),  # login_help.login
    (
        _li.provider().envelope(locale="en")["text"],
        (
            "Provider to log in to (e.g., openai, anthropic, kimi, xai). Omit to li"
            "st login-capable providers."
        ),
    ),  # login_help.provider
    (
        _lo.logout().envelope(locale="en")["text"],
        "Log out of a provider (removes stored credentials)",
    ),  # logout_help.logout
    (
        _lo.provider().envelope(locale="en")["text"],
        "Provider to log out of",
    ),  # logout_help.provider
    (
        _ls.login_status().envelope(locale="en")["text"],
        "List stored provider credentials and their status",
    ),  # login_status_help.login_status
    (
        _ls.status_alias().envelope(locale="en")["text"],
        "Alias for login-status: list stored provider credentials",
    ),  # login_status_help.status_alias
]
