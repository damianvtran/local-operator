"""The §4.3 sensitive denylist: the S14 table as data, the live-accessor roots, symlinks.

A denylist gap is only visible at the data, so every row names the path AND the rule id that
must produce it. Every denied row has an ALLOWED twin elsewhere in this file or the
candidates tests, so a rule that over-matches (denies every ``config.json``) fails too.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import pytest

from local_operator import browser_files, references
from local_operator.supplements import denylist
from local_operator.supplements.denylist import is_sensitive

HOME = Path.home()

#: (path as written, expected rule id). The memo's S14 names, then the classes round 1 added.
S14_TABLE = [
    (".env.local", denylist.RULE_NAME_PREFIX),
    ("id_ed25519.pub", denylist.RULE_CREDENTIAL_PATTERN),
    ("~/.aws/credentials", denylist.RULE_NAME),
    ("secrets/x.json", denylist.RULE_COMPONENT),
    ("PROD.ENV", denylist.RULE_SUFFIX),
    ("~/.config/gh/hosts.yml", denylist.RULE_GH_HOSTS),
    ("~/.docker/config.json", denylist.RULE_DOCKER_CONFIG),
    ("application_default_credentials.json", denylist.RULE_GCLOUD),
    ("~/.config/gcloud/properties", denylist.RULE_GCLOUD),
    ("deploy_credentials.json", denylist.RULE_GCLOUD),
    ("~/.terraform.d/credentials.tfrc.json", denylist.RULE_COMPONENT),
    ("~/.terraformrc", denylist.RULE_TERRAFORM),
    ("~/.config/rclone/rclone.conf", denylist.RULE_TOOL_CONFIG),
    ("~/.s3cfg", denylist.RULE_TOOL_CONFIG),
    ("~/.htpasswd", denylist.RULE_TOOL_CONFIG),
    ("~/.netrc", denylist.RULE_NAME),
    ("_netrc", denylist.RULE_CREDENTIAL_PATTERN),
    ("~/.aws/sso/cache/abc.json", denylist.RULE_COMPONENT),
    ("~/.azure/accessTokens.json", denylist.RULE_COMPONENT),
    ("~/Library/Application Support/Google/Chrome/Default/Cookies", denylist.RULE_BROWSER_STORE),
    ("profile/cookies.sqlite", denylist.RULE_DATABASE),
    ("backup.sqlite", denylist.RULE_DATABASE),
    ("app.db", denylist.RULE_DATABASE),
    ("api_token.txt", denylist.RULE_TOKEN_SECRET),
    ("my-Secret-notes.md", denylist.RULE_TOKEN_SECRET),
    ("docker-compose.prod.yml", denylist.RULE_COMPOSE),
    # Round 1 (R5): a kubeconfig outside ``.kube`` is the same cluster credential store --
    # ``deploy/kubeconfig`` was the reproduced slip; the two spellings trees write are here
    # so the ``kubeconfig*`` / ``*.kubeconfig`` patterns cannot silently regress.
    ("deploy/kubeconfig", denylist.RULE_TOOL_CONFIG),
    ("kubeconfig-prod", denylist.RULE_TOOL_CONFIG),
    ("prod.kubeconfig", denylist.RULE_TOOL_CONFIG),
    ("server.PEM", denylist.RULE_SUFFIX),
    ("~/.ssh/config", denylist.RULE_COMPONENT),
    ("keys/signing.key", denylist.RULE_SUFFIX),
    ("service-account-prod.json", denylist.RULE_CREDENTIAL_PATTERN),
]

#: Ordinary deliverables and look-alikes that MUST pass. ``config.json`` outside ``.docker``,
#: ``hosts.yml`` outside ``gh``, a ``.config`` that is not gh/gcloud/rclone.
ALLOWED = [
    "report.md",
    "data/bench.csv",
    "config.json",
    "~/.config/starship.toml",
    "~/projects/hosts.yml",
    "docs/secrets-policy-overview.pdf.txt".replace("secrets", "policy"),
    "notes/tokenizer-design.md".replace("tokenizer", "design"),
    "chart.png",
    "~/Documents/analysis.xlsx",
    # QA round 1 (Q-2): an NFKC-STABLE non-ASCII name, and a compatibility spelling whose
    # fold is BENIGN, must survive -- the fold denies equivalence to a denied name, never
    # non-ASCII itself, or it would start refusing real deliverables.
    "na\u00efve-notes.md",  # naïve-notes.md
    # ｍｅｅｔｉｎｇ-ｎｏｔｅｓ.md folds to a benign ASCII name; it must survive.
    "\uff4d\uff45\uff45\uff54\uff49\uff4e\uff47-\uff4e\uff4f\uff54\uff45\uff53.md",
]

#: QA round 1 (Q-2): NFKC-equivalent spellings of denied names must land on the SAME rule
#: the plain spelling produces. Written as escapes because the codepoint IS the test -- the
#: casefold-only comparison let ``.ｅｎｖ`` (U+FF45 each) reach the vendor payload as an
#: offered basename; the fix folds the class (full-width, ligature, mathematical-bold,
#: canonical decompositions), not that spelling. The must-survive twins are in ``ALLOWED``.
COMPATIBILITY_TABLE = [
    (".\uff45\uff4e\uff56", denylist.RULE_NAME),  # .ｅｎｖ -- the QA reproduction
    ("\uff0e\uff45\uff4e\uff56", denylist.RULE_NAME),  # ．ｅｎｖ -- the dot is full-width too
    (".\uff53\uff53\uff48/config", denylist.RULE_COMPONENT),  # .ｓｓｈ/
    (".\uff4b\uff55\uff42\uff45/config", denylist.RULE_COMPONENT),  # .ｋｕｂｅ/
    (".con\ufb01g/gcloud/properties", denylist.RULE_GCLOUD),  # conﬁg: the fi ligature
    (".\U0001d41e\U0001d427\U0001d42f", denylist.RULE_NAME),  # .𝐞𝐧𝐯 -- mathematical bold
    # Full-width NFKC-equivalent of the S14 row ``service-account-prod.json``.
    (
        "\uff53\uff45\uff52\uff56\uff49\uff43\uff45"
        "-\uff41\uff43\uff43\uff4f\uff55\uff4e\uff54-\uff50\uff52\uff4f\uff44.json",
        denylist.RULE_CREDENTIAL_PATTERN,
    ),  # ｓｅｒｖｉｃｅ-ａｃｃｏｕｎｔ-ｐｒｏｄ.json
]


@pytest.mark.parametrize(("path", "rule"), S14_TABLE)
def test_the_s14_table_names_the_rule_that_denies_each_path(path: str, rule: str, tmp_path) -> None:
    assert is_sensitive(path, cwd=str(tmp_path)) == rule


@pytest.mark.parametrize("path", ALLOWED)
def test_ordinary_deliverables_and_lookalikes_pass(path: str, tmp_path) -> None:
    assert is_sensitive(path, cwd=str(tmp_path)) == ""


def test_matching_is_case_insensitive_like_the_filesystem(tmp_path) -> None:
    # APFS: ``.ENV`` and ``.env`` are the same inode (references._sensitive_name measured it).
    assert is_sensitive(".ENV", cwd=str(tmp_path)) != ""
    assert is_sensitive("~/.SSH/id_rsa", cwd=str(tmp_path)) != ""
    assert is_sensitive("Docker-Compose.yaml", cwd=str(tmp_path)) == denylist.RULE_COMPOSE


@pytest.mark.parametrize(("path", "rule"), COMPATIBILITY_TABLE)
def test_nfkc_equivalent_spellings_of_denied_names_are_denied(
    path: str, rule: str, tmp_path
) -> None:
    """QA round 1 (Q-2): folding is NFKC + casefold, so compatibility spellings that compared
    unequal before reach the same rule as the plain spelling they fold onto -- the class, not
    the one spelling the round reproduced."""
    assert is_sensitive(path, cwd=str(tmp_path)) == rule


def test_a_symlink_to_a_secret_is_denied_by_its_target(tmp_path) -> None:
    """``report.md -> ~/.ssh/id_rsa``: the link is benign by name, the target is not."""
    target_dir = tmp_path / ".ssh"
    target_dir.mkdir()
    secret = target_dir / "id_rsa"
    secret.write_text("PRIVATE")
    link = tmp_path / "report.md"
    link.symlink_to(secret)
    assert is_sensitive(link, cwd=str(tmp_path)) != ""
    # ...and the control: the same name pointing at an ordinary file is allowed.
    plain = tmp_path / "plain.txt"
    plain.write_text("hello")
    ok = tmp_path / "summary.md"
    ok.symlink_to(plain)
    assert is_sensitive(ok, cwd=str(tmp_path)) == ""


def test_a_sensitive_link_name_is_denied_even_when_the_target_is_benign(tmp_path) -> None:
    plain = tmp_path / "plain.txt"
    plain.write_text("hello")
    link = tmp_path / ".env"
    link.symlink_to(plain)
    assert is_sensitive(link, cwd=str(tmp_path)) == denylist.RULE_NAME


def test_the_relocated_config_dir_is_denied_not_a_literal_home_path(tmp_path, monkeypatch) -> None:
    """S-R2-4: every root is computed from the LIVE accessor. A hard-coded ``~/.local-operator``
    passes every non-relocated fixture and still leaks the store under a relocation."""
    relocated = tmp_path / "elsewhere" / "lop-config"
    (relocated / "secrets").mkdir(parents=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(relocated))
    for name in ("secrets/store.db", "sessions/abc/transcript.jsonl", "config.yml", "notes.md"):
        path = relocated / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        assert is_sensitive(path, cwd=str(tmp_path)) == denylist.RULE_CONFIG_DIR, name
    # The control: a sibling that is NOT under it is allowed.
    sibling = tmp_path / "elsewhere" / "report.md"
    sibling.write_text("ok")
    assert is_sensitive(sibling, cwd=str(tmp_path)) == ""


def test_the_scratchpad_root_is_denied_from_the_environment(tmp_path, monkeypatch) -> None:
    pad = tmp_path / "pad"
    pad.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(pad))
    out = pad / "draft.md"
    out.write_text("x")
    assert is_sensitive(out, cwd=str(tmp_path)) == denylist.RULE_SCRATCHPAD
    monkeypatch.delenv("LOCAL_OPERATOR_SCRATCHPAD")
    assert is_sensitive(out, cwd=str(tmp_path)) == ""


def test_a_folded_spelling_of_a_root_is_denied_like_the_root(tmp_path, monkeypatch) -> None:
    """QA round 1 (Q-2), the settings/env half: the roots compared in ``_under`` fold with the
    same function as the rules, so a canonically-equivalent (NFD) or case-variant spelling of
    a relocated config dir -- which APFS resolves to the same file -- cannot slip; the fold is
    not split across the two comparisons. Control: a sibling the fold does NOT map onto it."""
    cfg = tmp_path / "caf\u00e9-cfg"  # NFC on disk
    cfg.mkdir()
    (cfg / "notes.md").write_text("x")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    decomposed = unicodedata.normalize("NFD", str(cfg))
    assert decomposed != str(cfg), "NFD and NFC spellings must differ as strings"
    assert (
        is_sensitive(os.path.join(decomposed, "notes.md"), cwd=str(tmp_path))
        == denylist.RULE_CONFIG_DIR
    )
    case_variant = os.path.join(os.path.dirname(str(cfg)), "CAF\u00c9-CFG", "notes.md")
    assert is_sensitive(case_variant, cwd=str(tmp_path)) == denylist.RULE_CONFIG_DIR
    # Control: a sibling directory the fold does NOT map onto the root stays allowed.
    sibling = tmp_path / "caf\u00e9-other"
    sibling.mkdir()
    (sibling / "notes.md").write_text("x")
    assert is_sensitive(sibling / "notes.md", cwd=str(tmp_path)) == ""


def test_the_denylist_is_a_superset_of_both_existing_gates() -> None:
    """Composition, not a third list: a name added to either upstream gate is denied here."""
    for name in references.SENSITIVE_NAMES:
        assert is_sensitive(name, cwd="/work"), name
    for suffix in references.SENSITIVE_SUFFIXES:
        assert is_sensitive(f"x{suffix}", cwd="/work"), suffix
    for part in references.SENSITIVE_DIR_PARTS | browser_files.CREDENTIAL_COMPONENTS:
        assert is_sensitive(f"{part}/file.txt", cwd="/work"), part
    for pattern in browser_files.CREDENTIAL_NAME_PATTERNS:
        sample = pattern.replace("*", "x")
        assert is_sensitive(sample, cwd="/work"), pattern


def test_every_rule_id_is_reachable_from_the_table() -> None:
    """The table is exhaustive over the module's rule ids (a rule with no row is untested)."""
    declared = {
        value
        for name, value in vars(denylist).items()
        if name.startswith("RULE_") and isinstance(value, str)
    }
    covered = {rule for _path, rule in S14_TABLE} | {
        denylist.RULE_CONFIG_DIR,
        denylist.RULE_SCRATCHPAD,
    }
    assert not declared - covered, declared - covered


def test_an_unresolvable_path_is_judged_on_its_written_form(tmp_path) -> None:
    loop = tmp_path / "loop"
    os.symlink(loop, loop)  # a symlink loop: realpath must not raise
    assert is_sensitive(loop, cwd=str(tmp_path)) == ""
    assert is_sensitive(tmp_path / "a" / ".env", cwd=str(tmp_path)) == denylist.RULE_NAME


#: (path as written, expected rule) for the R6-1 class: a FOLD-PRODUCED separator. ``／``
#: (U+FF0F) is the one codepoint NFKC maps onto ``/``, so a path spelled with it is a single
#: component to ``PurePosixPath`` and becomes two only after the fold -- which is exactly why
#: the fold has to run before the split (`denylist._rule_for`).
FOLDED_SEPARATOR_TABLE = [
    ("．ｓｓｈ／config", denylist.RULE_COMPONENT),
    ("ｒｅｐｏ／．ｅｎｖ", denylist.RULE_NAME),
    (".config／gh／hosts.yml", denylist.RULE_GH_HOSTS),
    ("deploy／secrets／x.json", denylist.RULE_COMPONENT),
]

#: Must survive: the same spelling, a benign path. Without these a rule that denied every
#: path containing a full-width solidus would pass the table above.
FOLDED_SEPARATOR_ALLOWED = [
    "ｒｅｐｏｒｔ／ｄｒａｆｔ．ｍｄ",
    "notes／2026／summary.csv",
    "．ｓｓｈ-notes／report.md",
]


@pytest.mark.parametrize(("path", "rule"), FOLDED_SEPARATOR_TABLE)
def test_a_fold_produced_separator_splits_like_a_real_one(path: str, rule: str, tmp_path) -> None:
    """Agent review round 6 (R6-1): NFKC folds ``／`` onto ``/``, so a multi-component path
    spelled with full-width separators must reach the same structural rule as its plain
    spelling. Spelled ASCII-first (no literal U+FF0F in the source) so a future edit of this
    table cannot be defeated by an editor's own normalisation."""
    folded = unicodedata.normalize("NFKC", path)
    assert "／" not in folded and "/" in folded, "the fixture no longer tests R6-1"
    assert is_sensitive(path, cwd=str(tmp_path)) == rule


@pytest.mark.parametrize("path", FOLDED_SEPARATOR_ALLOWED)
def test_a_fold_produced_separator_does_not_deny_benign_paths(path: str, tmp_path) -> None:
    """The must-survive controls for the class above."""
    assert is_sensitive(path, cwd=str(tmp_path)) == ""
