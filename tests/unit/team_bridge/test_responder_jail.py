"""Slice C1 of GATE R3-HASH + S-41: the responder jail, pinned credentials,
repository materialization and the isolation attestation -- all UNWIRED.

WHAT IS PINNED, and the finding each block answers:

* **Profile shape (every platform; S-3, S-15, A-1).** Deny-default reads with
  an allowlist; ``/Library`` granted by children with ``Keychains`` OMITTED;
  ``/private/var/folders`` narrowed to the run's TMPDIR; nothing else under
  ``$HOME``; a filtered 443 egress allow followed by the loopback deny; no
  bare network or file-read allow anywhere.
* **Read-root refusal (every platform; S-15a).** An interpreter tree that
  resolves to ``/``, HOME, an ancestor of HOME, or a tree holding the config
  dir refuses to build -- the jail is never built wider.
* **In-process T0 tools (every platform; S-14).** The real ``read``/``grep``/
  ``glob`` tools under :meth:`ResponderJail.tool_confinement` refuse a planted
  symlink and a search rooted at HOME; ``web_fetch``'s SSRF guard is forced on
  whatever the owner configured.
* **Pinned credentials (every platform; S-2a, S-13).** The role ARN and the
  STS endpoint cannot be moved by ``~/.aws/config`` or ``AWS_ENDPOINT_URL_STS``;
  a wrong account, wrong role or long-lived key refuses; the module never
  imports the lop credential store.
* **Materialization (every platform with git; S-12).** A clone with a planted
  smudge driver materializes with the canary ABSENT, with ``git archive`` as
  the negative control proving the driver is live.
* **The kernel boundary (macOS only).** Real ``sandbox-exec`` probes, each
  with an unjailed control. CI's PR legs are Linux, so these skip there with
  the reason named; the PR records their run on the build host.
"""

from __future__ import annotations

import ast
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

import pytest

from local_operator.harness.types import ToolContext
from local_operator.team_bridge.responders import jail as jail_module
from local_operator.team_bridge.responders import pinned_credentials as creds_module
from local_operator.team_bridge.responders.isolation import (
    IsolationAttestation,
    attest,
    attestation_refusal,
)
from local_operator.team_bridge.responders.jail import (
    ENV_ALLOWLIST,
    JailUnavailable,
    ResponderJail,
    responder_web_fetch_settings,
)
from local_operator.team_bridge.responders.materialize import (
    MaterializeError,
    RepoNotAllowed,
    materialize_repo,
    pinned_repo_name,
)
from local_operator.team_bridge.responders.pinned_credentials import (
    RESPONDER_ROLE_ARN,
    STS_ENDPOINT,
    CredentialPinError,
    PinnedCredentials,
    resolve_responder_credentials,
)
from local_operator.tools import builtin
from local_operator.tools import confinement as confinement_module

DARWIN_ONLY = pytest.mark.skipif(
    sys.platform != "darwin" or not confinement_module.SANDBOX_EXEC.exists(),
    reason=(
        "the responder jail is a macOS seatbelt profile; its shape and refusals are pinned above"
    ),
)

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "/bin/false",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


@pytest.fixture
def fake_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A HOME + config dir under ``tmp_path`` laid out like the operator's.

    The jail computes HOME and the config dir itself; pointing both at a
    fixture keeps every probe off the real operator's files.
    """

    home = (tmp_path / "home").resolve()
    config = home / ".local-operator"
    run_home = config / "sessions" / "me" / "responder-home"
    other_session = config / "sessions" / "other"
    # The run home itself is NOT created: prepare_home refuses an existing
    # one (review R1-2, one fresh home per run).
    for directory in (
        home / "workspace",
        home / "radient-ml" / "secret-repo",
        other_session,
        run_home.parent,
    ):
        directory.mkdir(parents=True)
    (home / "workspace" / "export.csv").write_text("customer,row\n")
    (home / ".zsh_history").write_text(": 1:0;echo HISTORY-CANARY\n")
    (home / "radient-ml" / "secret-repo" / "README.md").write_text("not allowlisted\n")
    (other_session / "transcript.jsonl").write_text('{"role":"user"}\n')
    (config / "responders.yaml").write_text("channels: {}\n")
    (home / ".zshrc").write_text("# rc\n")
    if not jail_module._SYSTEM_CA_BUNDLE.is_file():
        # Linux CI has no /private/etc/ssl: stand a public-looking bundle in so
        # the platform-neutral tests can build a home.
        bundle = tmp_path / "cert.pem"
        bundle.write_text("-----BEGIN CERTIFICATE-----\nfixture\n-----END CERTIFICATE-----\n")
        monkeypatch.setattr(jail_module, "_SYSTEM_CA_BUNDLE", bundle)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    return {"home": home, "config": config, "run_home": run_home, "other": other_session}


def _jail(run_home: Path) -> ResponderJail:
    jail = ResponderJail.at(run_home, machinery=())
    jail.prepare_home()
    return jail


# --------------------------------------------------------------------------
# Profile shape (every platform)
# --------------------------------------------------------------------------


class TestProfileShape:
    def test_reads_are_deny_default_with_an_allowlist(self, fake_host: dict[str, Path]) -> None:
        jail = _jail(fake_host["run_home"])
        profile = jail.profile()
        assert "(deny default)" in profile
        # The scout shell's shape -- every read allowed -- must never appear (S-3).
        assert "(allow file-read*)\n" not in profile
        assert f'(subpath "{jail.home}")' in profile
        # Nothing else under HOME: the run home is the only HOME-side grant.
        home = str(fake_host["home"])
        granted = [p for p in jail.read_subpaths() if p == home or p.startswith(home + "/")]
        assert granted == [str(jail.home)]

    def test_library_is_granted_by_children_with_keychains_omitted(
        self, fake_host: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Path-aware: ``read_subpaths`` also lists /opt/homebrew and /usr/local.
        listings = {"/Library": ["Developer", "Keychains", "Preferences"]}
        monkeypatch.setattr(jail_module.os, "listdir", lambda path: listings.get(path, []))
        reads = _jail(fake_host["run_home"]).read_subpaths()
        assert "/Library" not in reads
        assert "/Library/Developer" in reads and "/Library/Preferences" in reads
        assert not any("Keychains" in path for path in reads)

    def test_var_folders_is_narrowed_to_the_runs_own_tmpdir(
        self, fake_host: dict[str, Path]
    ) -> None:
        jail = _jail(fake_host["run_home"])
        assert "/private/var/folders" not in jail.read_subpaths()
        assert jail.child_env()["TMPDIR"] == str(jail.tmpdir)
        assert jail.tmpdir.is_relative_to(jail.home)

    def test_egress_is_a_filtered_443_allow_followed_by_the_loopback_deny(
        self, fake_host: dict[str, Path]
    ) -> None:
        profile = _jail(fake_host["run_home"]).profile()
        lines = profile.splitlines()
        # A-1: a bare network allow defeats any network deny, in either order.
        assert "(allow network-outbound)" not in lines
        assert "(allow network*)" not in lines
        allow = lines.index('(allow network-outbound (remote tcp "*:443"))')
        deny = lines.index('(deny network-outbound (remote ip "localhost:*"))')
        # S-15b: ``*`` includes localhost; the deny must come AFTER the allow.
        assert deny > allow
        assert not any("remote tcp" in line and ":443" not in line for line in lines)

    def test_materialized_repos_are_write_denied_after_the_home_write_allow(
        self, fake_host: dict[str, Path]
    ) -> None:
        jail = _jail(fake_host["run_home"])
        lines = jail.profile().splitlines()
        write_allow = next(
            i for i, line in enumerate(lines) if line.startswith("(allow file-write*")
        )
        write_deny = lines.index(f'(deny file-write* (subpath "{jail.repos_dir}"))')
        assert write_deny > write_allow
        assert "/dev/tty" not in lines[write_allow]

    def test_mach_adds_only_the_dns_service(self, fake_host: dict[str, Path]) -> None:
        profile = _jail(fake_host["run_home"]).profile()
        mach = next(line for line in profile.splitlines() if line.startswith("(allow mach-lookup"))
        names = sorted(part.split('"')[1] for part in mach.split("(global-name ")[1:])
        assert names == sorted(
            (*confinement_module._MACH_LOOKUP_GLOBAL_NAMES, "com.apple.mDNSResponder")
        )

    def test_credential_denials_stay_last_as_belt_and_braces(
        self, fake_host: dict[str, Path]
    ) -> None:
        last = _jail(fake_host["run_home"]).profile().rstrip("\n").splitlines()[-1]
        assert last.startswith("(deny file-read* ")

    def test_a_missing_ca_bundle_refuses_to_prepare(
        self, fake_host: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(jail_module, "_SYSTEM_CA_BUNDLE", tmp_path / "absent.pem")
        with pytest.raises(JailUnavailable, match="CA bundle"):
            ResponderJail.at(fake_host["run_home"], machinery=()).prepare_home()

    def test_no_jail_means_no_spawn(
        self, fake_host: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        jail = _jail(fake_host["run_home"])
        monkeypatch.setattr(
            confinement_module, "shell_sandbox_unavailable_reason", lambda: "no sandbox here"
        )
        with pytest.raises(JailUnavailable, match="no sandbox here"):
            jail.wrap(["/bin/true"])


class TestReadRootRefusal:
    """S-15a: the interpreter subtree must never re-grant HOME or config."""

    @pytest.mark.parametrize("which", ["root", "home", "home-parent", "config", "config-parent"])
    def test_a_widening_read_root_refuses_to_build(
        self, fake_host: dict[str, Path], which: str
    ) -> None:
        roots = {
            "root": Path("/"),
            "home": fake_host["home"],
            "home-parent": fake_host["home"].parent,
            "config": fake_host["config"],
            "config-parent": fake_host["config"].parent,
        }
        with pytest.raises(JailUnavailable):
            ResponderJail.at(fake_host["run_home"], machinery=[roots[which]])

    def test_a_faked_interpreter_in_home_refuses(
        self, fake_host: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``~/bin/python`` -> its "subtree" is HOME itself.
        monkeypatch.setattr(confinement_module, "_interpreter_subtree", lambda: fake_host["home"])
        with pytest.raises(JailUnavailable, match="covers HOME"):
            ResponderJail.at(fake_host["run_home"])

    def test_a_benign_interpreter_tree_builds(
        self, fake_host: dict[str, Path], tmp_path: Path
    ) -> None:
        venv = tmp_path / "venv"
        venv.mkdir()
        jail = ResponderJail.at(fake_host["run_home"], machinery=[venv])
        assert str(venv.resolve()) in jail.read_subpaths()

    def test_a_run_home_covering_home_refuses(self, fake_host: dict[str, Path]) -> None:
        with pytest.raises(JailUnavailable):
            ResponderJail.at(fake_host["home"], machinery=())

    def test_an_interpreter_under_home_outside_uv_refuses(self, fake_host: dict[str, Path]) -> None:
        """S-21: ``~/workspace/x/.venv`` would grant ``~/workspace/x``."""

        venv = fake_host["home"] / "workspace" / "x" / ".venv"
        venv.mkdir(parents=True)
        with pytest.raises(JailUnavailable, match="outside uv"):
            ResponderJail.at(fake_host["run_home"], machinery=[venv])
        uv_python = fake_host["home"] / ".local" / "share" / "uv" / "python" / "cpython-3.12"
        uv_python.mkdir(parents=True)
        ResponderJail.at(fake_host["run_home"], machinery=[uv_python])  # control

    def test_a_symlinked_run_home_refuses(self, fake_host: dict[str, Path]) -> None:
        """R1-3: the leaf is never resolved through a link to another tree."""

        fake_host["run_home"].symlink_to(fake_host["home"] / "workspace")
        with pytest.raises(JailUnavailable, match="symlink"):
            ResponderJail.at(fake_host["run_home"], machinery=())

    def test_a_reused_run_home_refuses_to_prepare(self, fake_host: dict[str, Path]) -> None:
        """R1-2: a planted ``ca-bundle.crt -> ~/.zshrc`` is never written through."""

        jail = _jail(fake_host["run_home"])
        with pytest.raises(JailUnavailable, match="not fresh"):
            jail.prepare_home()
        zshrc = fake_host["home"] / ".zshrc"
        before = zshrc.read_bytes()
        planted = ResponderJail.at(fake_host["run_home"].parent / "second-home", machinery=())
        planted.home.mkdir()
        (planted.home / jail_module.CA_BUNDLE_NAME).symlink_to(zshrc)
        with pytest.raises(JailUnavailable, match="not fresh"):
            planted.prepare_home()
        assert zshrc.read_bytes() == before


def _covered(path: str, reads: tuple[str, ...]) -> list[str]:
    """The grants that cover ``path`` (equal or ancestor): ``subpath`` semantics."""

    return [r for r in reads if path == r or path.startswith(r.rstrip("/") + "/")]


class TestPackagePrefixes:
    """S-16: package prefixes are granted child by child, minus their state."""

    _LISTINGS = {
        "/Library": ["Developer", "Keychains", "Logs", ".localized"],
        "/usr": ["bin", "lib", "libexec", "local", "sbin", "share"],
        "/opt/homebrew": ["Cellar", "bin", "etc", "lib", "opt", "share", "var", ".git"],
        "/usr/local": ["bin", "etc", "lib", "var"],
    }

    def test_no_grant_covers_an_omitted_path(
        self, fake_host: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Coverage, not string membership (review R2-1): ``/usr`` once re-granted
        ``/usr/local/var`` while ``"/usr/local/var" not in reads`` still passed."""

        monkeypatch.setattr(jail_module.os, "listdir", lambda path: self._LISTINGS[path])
        reads = _jail(fake_host["run_home"]).read_subpaths()
        for omitted in (
            "/usr/local/var/x",
            "/usr/local/etc/x",
            "/opt/homebrew/var/x",
            "/opt/homebrew/etc/x",
            "/Library/Logs/x",
            "/Library/Keychains/x",
            "/private/var/folders/x",
        ):
            assert _covered(omitted, reads) == [], omitted
        # Controls: the toolchain trees ARE covered.
        for needed in ("/usr/bin/git", "/usr/lib/x", "/opt/homebrew/bin/aws", "/usr/local/bin/x"):
            assert _covered(needed, reads), needed
        assert "/opt/homebrew/.git" not in reads  # R2-3

    def test_an_ancestor_grant_fails_closed(
        self, fake_host: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the system list ever re-adds an ancestor, the jail refuses to build."""

        monkeypatch.setattr(jail_module.os, "listdir", lambda path: self._LISTINGS[path])
        widened = tuple(p for p in jail_module._REPLACED_SYSTEM_ROOTS if p != "/usr")
        monkeypatch.setattr(jail_module, "_REPLACED_SYSTEM_ROOTS", frozenset(widened))
        with pytest.raises(JailUnavailable, match="covers omitted /usr/local"):
            _jail(fake_host["run_home"]).read_subpaths()


class TestChildEnvironment:
    def test_the_environment_is_rebuilt_not_inherited(
        self, fake_host: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for key in ("AWS_PROFILE", "GH_TOKEN", "LOCAL_OPERATOR_HUB_TOKEN", "AWS_ENDPOINT_URL_STS"):
            monkeypatch.setenv(key, "planted")
        env = _jail(fake_host["run_home"]).child_env()
        assert set(env) <= set(ENV_ALLOWLIST)
        assert not any(key.startswith("LOCAL_OPERATOR_") for key in env)
        assert "planted" not in env.values()
        assert env["HOME"] == str(fake_host["run_home"])

    def test_a_key_outside_the_allowlist_is_a_programming_error(
        self, fake_host: dict[str, Path]
    ) -> None:
        with pytest.raises(ValueError, match="AWS_PROFILE"):
            _jail(fake_host["run_home"]).child_env({"AWS_PROFILE": "radient-prod-admin"})

    @pytest.mark.parametrize(
        "key", ["HOME", "PATH", "TMPDIR", "AWS_CA_BUNDLE", "SSL_CERT_FILE", "KUBECONFIG"]
    )
    def test_extra_cannot_override_the_keys_the_jail_sets(
        self, fake_host: dict[str, Path], key: str
    ) -> None:
        """S-23: ``extra`` is the credential keys and LC_ALL only."""

        with pytest.raises(ValueError, match=key):
            _jail(fake_host["run_home"]).child_env({key: "/planted"})

    def test_extra_accepts_the_pinned_credential_keys(self, fake_host: dict[str, Path]) -> None:
        creds = PinnedCredentials("ASIA" + "X" * 16, "s", "t", "", _GOOD_ARN)
        env = _jail(fake_host["run_home"]).child_env(creds.env())
        assert env["AWS_SESSION_TOKEN"] == "t" and env["HOME"] == str(fake_host["run_home"])

    def test_the_ca_bundle_is_copied_under_a_name_the_pem_denial_does_not_match(
        self, fake_host: dict[str, Path]
    ) -> None:
        jail = _jail(fake_host["run_home"])
        assert jail.ca_bundle.suffix != ".pem"
        assert jail.ca_bundle.read_bytes() == jail_module._SYSTEM_CA_BUNDLE.read_bytes()
        assert jail.child_env()["SSL_CERT_FILE"] == str(jail.ca_bundle)


# --------------------------------------------------------------------------
# S-14: the in-process T0 tools
# --------------------------------------------------------------------------


def _tool_context(jail: ResponderJail) -> ToolContext:
    confinement = jail.tool_confinement()
    return ToolContext(
        cwd=str(confinement.root),
        session_id="responder-jail-test",
        confinement_root=str(confinement.root),
    )


class TestInProcessTools:
    @pytest.mark.asyncio
    async def test_read_refuses_a_symlink_planted_in_the_run_home(
        self, fake_host: dict[str, Path]
    ) -> None:
        jail = _jail(fake_host["run_home"])
        target = fake_host["home"] / "workspace" / "export.csv"
        (jail.home / "x").symlink_to(target)
        context = _tool_context(jail)

        refused = await builtin.execute_read(
            "r", {"path": str(jail.home / "x")}, None, None, context
        )
        assert refused.is_error and "confined" in refused.text
        assert "customer,row" not in refused.text
        # Unjailed control: the same tool, no confinement, reads the file.
        free = ToolContext(cwd=str(jail.home), session_id="free")
        control = await builtin.execute_read("r", {"path": str(jail.home / "x")}, None, None, free)
        assert "customer,row" in control.text

    @pytest.mark.asyncio
    async def test_grep_and_glob_rooted_at_home_are_refused(
        self, fake_host: dict[str, Path]
    ) -> None:
        context = _tool_context(_jail(fake_host["run_home"]))
        grep = await builtin.execute_grep(
            "g", {"pattern": "HISTORY-CANARY", "path": str(fake_host["home"])}, None, None, context
        )
        assert grep.is_error and "confined" in grep.text
        assert "HISTORY-CANARY" not in grep.text
        glob = await builtin.execute_glob(
            "gl", {"pattern": "**/*.csv", "path": str(fake_host["home"])}, None, None, context
        )
        assert "export.csv" not in glob.text

    def test_web_fetch_ssrf_guard_is_forced_on_whatever_the_owner_set(self) -> None:
        from local_operator.web_fetch.models import WebFetchSettings
        from local_operator.web_fetch.service import PolicyRefusal, validate_public_url

        owner = WebFetchSettings(allow_private=True, max_bytes=1234)
        forced = responder_web_fetch_settings(owner)
        assert forced.allow_private is False
        assert forced.max_bytes == 1234  # only the guard is overridden
        assert owner.allow_private is True  # the owner's object is untouched
        url = "http://127.0.0.1:4097/"
        # The guard refuses by raising (it returns the pinned IP when it admits).
        with pytest.raises(PolicyRefusal) as refused:
            validate_public_url(url, allow_private=forced.allow_private)
        assert refused.value.reason == "private"
        # Control: the owner's own setting would have let it through.
        assert validate_public_url(url, allow_private=owner.allow_private) is None


# --------------------------------------------------------------------------
# S-2a / S-13: pinned credentials
# --------------------------------------------------------------------------

_GOOD_ARN = "arn:aws:sts::951603962783:assumed-role/radient-responder-readonly/alert-1"


def _recorder(identity: Mapping[str, Any], *, key: str = "ASIAEXAMPLEKEY123456"):
    calls: list[tuple[Any, str, Mapping[str, Any]]] = []

    def call(client: Any, operation: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append((client, operation, dict(params)))
        if operation == "assume_role":
            return {
                "Credentials": {
                    "AccessKeyId": key,
                    "SecretAccessKey": "s3cr3t-value",
                    "SessionToken": "t0ken-value",
                    "Expiration": "2026-11-24T12:00:00Z",
                },
                "AssumedRoleUser": {"Arn": identity.get("Arn")},
            }
        return identity

    return call, calls


@pytest.fixture
def tampered_aws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The owner's AWS config as a planter would leave it (S-13)."""

    config = tmp_path / "aws-config"
    config.write_text(
        "[profile radient-prod-admin]\n"
        "sso_session = radient\n"
        "sso_account_id = 951603962783\n"
        "sso_role_name = AdministratorAccess\n"
        "region = us-east-1\n"
        "endpoint_url = https://evil.example.invalid\n"
        "ca_bundle = /nonexistent/evil-ca.crt\n"
        "[sso-session radient]\n"
        "sso_start_url = https://example.invalid/start\n"
        "sso_region = ca-central-1\n"
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", os.devnull)
    monkeypatch.setenv("AWS_ENDPOINT_URL_STS", "https://evil-sts.example.invalid")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://evil-all.example.invalid")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")


def _aws_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile_body: str) -> None:
    config = tmp_path / "aws-config-variant"
    config.write_text(profile_body)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", os.devnull)


class TestPinnedCredentials:
    @pytest.mark.parametrize(
        "planted",
        [
            "credential_process = /usr/bin/touch {canary}\n",
            "source_profile = other\nrole_arn = arn:aws:iam::951603962783:role/admin\n",
            "credential_source = Environment\nrole_arn = arn:aws:iam::951603962783:role/x\n",
        ],
    )
    def test_a_source_profile_that_would_run_or_chain_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planted: str
    ) -> None:
        """S-19: the profile is refused BEFORE botocore could execute it."""

        canary = tmp_path / "CRED-PROCESS-CANARY"
        _aws_config(
            tmp_path,
            monkeypatch,
            "[profile radient-prod-admin]\nsso_session = radient\n"
            + planted.format(canary=canary)
            + "[sso-session radient]\nsso_start_url = https://example.invalid/start\n",
        )
        call, calls = _recorder({"Account": "951603962783", "Arn": _GOOD_ARN})
        with pytest.raises(CredentialPinError, match="not a plain SSO profile"):
            resolve_responder_credentials(session_name="alert-1", call=call)
        assert calls == [] and not canary.exists()

    def test_a_static_key_profile_is_not_an_sso_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _aws_config(tmp_path, monkeypatch, "[profile radient-prod-admin]\nregion = ca-central-1\n")
        call, calls = _recorder({"Account": "951603962783", "Arn": _GOOD_ARN})
        with pytest.raises(CredentialPinError, match="not an SSO profile"):
            resolve_responder_credentials(session_name="alert-1", call=call)
        assert calls == []

    def test_a_missing_profile_is_a_pin_error_not_a_raw_botocore_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R1-1: every failure reaches the spawner as CredentialPinError."""

        _aws_config(tmp_path, monkeypatch, "[profile someone-else]\nregion = ca-central-1\n")
        call, calls = _recorder({"Account": "951603962783", "Arn": _GOOD_ARN})
        with pytest.raises(CredentialPinError):
            resolve_responder_credentials(session_name="alert-1", call=call)
        assert calls == []

    def test_endpoint_overrides_in_config_and_env_do_not_move_sts(self, tampered_aws: None) -> None:
        call, calls = _recorder({"Account": "951603962783", "Arn": _GOOD_ARN})
        result = resolve_responder_credentials(session_name="alert-1", call=call)
        assert [op for _, op, _ in calls] == ["assume_role", "get_caller_identity"]
        for client, _, _ in calls:
            assert client.meta.endpoint_url == STS_ENDPOINT
            assert client.meta.region_name == "ca-central-1"
        assert result.assumed_role_arn == _GOOD_ARN

    def test_the_role_is_the_code_constant(self, tampered_aws: None) -> None:
        call, calls = _recorder({"Account": "951603962783", "Arn": _GOOD_ARN})
        resolve_responder_credentials(session_name="alert-1", call=call)
        params = calls[0][2]
        assert params["RoleArn"] == RESPONDER_ROLE_ARN
        assert params["DurationSeconds"] == 3600
        assert '"secretsmanager:*"' in params["Policy"]

    def test_the_identity_check_runs_on_the_returned_triple_only(self, tampered_aws: None) -> None:
        call, calls = _recorder({"Account": "951603962783", "Arn": _GOOD_ARN})
        resolve_responder_credentials(session_name="alert-1", call=call)
        identity_client = calls[1][0]
        frozen = identity_client._request_signer._credentials.get_frozen_credentials()
        assert frozen.access_key == "ASIAEXAMPLEKEY123456"
        assert frozen.token == "t0ken-value"

    @pytest.mark.parametrize(
        ("identity", "key", "match"),
        [
            (
                {
                    "Account": "111111111111",
                    "Arn": "arn:aws:sts::111111111111:assumed-role/radient-responder-readonly/x",
                },
                "ASIAEXAMPLEKEY123456",
                "account",
            ),
            (
                {
                    "Account": "951603962783",
                    "Arn": "arn:aws:sts::951603962783:assumed-role/admin/x",
                },
                "ASIAEXAMPLEKEY123456",
                "pinned responder role",
            ),
            (
                {
                    "Account": "951603962783",
                    "Arn": (
                        "arn:aws:sts::951603962783:assumed-role/radient-responder-readonly-wide/x"
                    ),
                },
                "ASIAEXAMPLEKEY123456",
                "pinned responder role",
            ),
            (
                {"Account": "951603962783", "Arn": _GOOD_ARN},
                "AKIALONGLIVEDKEY0000",
                "non-temporary",
            ),
        ],
    )
    def test_a_wrong_identity_refuses(
        self, tampered_aws: None, identity: Mapping[str, Any], key: str, match: str
    ) -> None:
        call, _ = _recorder(identity, key=key)
        with pytest.raises(CredentialPinError, match=match):
            resolve_responder_credentials(session_name="alert-1", call=call)

    def test_secrets_never_print(self) -> None:
        pinned = PinnedCredentials(
            "ASIAEXAMPLEKEY123456", "s3cr3t-value", "t0ken-value", "", _GOOD_ARN
        )
        assert "s3cr3t-value" not in repr(pinned) and "t0ken-value" not in repr(pinned)

    def test_the_lop_credential_store_is_never_imported(self) -> None:
        tree = ast.parse(Path(creds_module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add("." * node.level + (node.module or ""))
        assert not any(
            name.startswith("local_operator") or name.startswith(".") for name in imported
        ), imported
        assert {"json", "os", "re"} <= imported


# --------------------------------------------------------------------------
# S-12: materialization
# --------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=_GIT_ENV
    ).stdout.strip()


def _promisor_clone(clone_root: Path, canary: Path) -> Path:
    """A clone whose config makes a missing blob trigger a planted transport (S-18)."""

    repo = clone_root / "agent-server"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("readme\n")
    (repo / "missing.txt").write_text("this blob is removed after commit\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    missing = _git(repo, "rev-parse", "HEAD:missing.txt")
    (repo / ".git" / "objects" / missing[:2] / missing[2:]).unlink()
    # The plant: a promisor remote whose transport is an ext:: command.
    _git(repo, "config", "core.repositoryformatversion", "1")
    _git(repo, "config", "extensions.partialClone", "origin")
    _git(repo, "config", "remote.origin.promisor", "true")
    _git(repo, "config", "remote.origin.url", f"ext::sh -c touch% {canary}")
    _git(repo, "config", "protocol.ext.allow", "always")
    return repo


def _planted_clone(clone_root: Path, canary: Path) -> Path:
    repo = clone_root / "agent-server"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("agent-server readme\n")
    (repo / "bin").mkdir()
    (repo / "bin" / "run.sh").write_text("#!/bin/sh\n")
    (repo / "bin" / "run.sh").chmod(0o755)
    (repo / "link").symlink_to("/etc/hosts")
    (repo / ".gitattributes").write_text("* filter=x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    # The S-12 plant: a smudge driver in repo config AND in info/attributes.
    _git(repo, "config", "filter.x.smudge", f"touch {canary}; cat")
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text("* filter=x\n")
    return repo


class TestMaterialize:
    def test_a_planted_smudge_driver_never_runs(self, tmp_path: Path) -> None:
        canary = tmp_path / "CANARY"
        clone_root = tmp_path / "radient-ml"
        repo = _planted_clone(clone_root, canary)

        # Negative control: the driver IS live -- git archive runs it.
        subprocess.run(
            ["git", "-C", str(repo), "archive", "-o", str(tmp_path / "a.tar"), "HEAD"],
            check=True,
            capture_output=True,
            env=_GIT_ENV,
        )
        assert canary.exists(), "fixture broken: git archive did not run the planted driver"
        canary.unlink()

        dest_root = tmp_path / "run-home" / "repos"
        dest_root.mkdir(parents=True)
        result = materialize_repo("radient-ml/agent-server", dest_root, clone_root=clone_root)

        assert not canary.exists()
        assert result.commit == _git(repo, "rev-parse", "HEAD")
        assert (result.path / "README.md").read_text() == "agent-server readme\n"
        assert not (result.path / "link").exists() and result.skipped == 1
        assert os.access(result.path / "bin" / "run.sh", os.X_OK)
        assert not os.access(result.path / "README.md", os.W_OK)
        assert not os.access(result.path, os.W_OK)

    @pytest.mark.parametrize(
        "entry",
        ["minervaai/core", "radient-ml/../x", "radient-ml/a/b", "radient-ml/.git", "radient-ml/"],
    )
    def test_only_radient_ml_repositories_are_pinned(self, entry: str) -> None:
        with pytest.raises(RepoNotAllowed):
            pinned_repo_name(entry)

    def test_a_pinned_name_resolves(self) -> None:
        assert pinned_repo_name("radient-ml/agent-server") == "agent-server"

    def test_a_missing_clone_refuses(self, tmp_path: Path) -> None:
        with pytest.raises(MaterializeError, match="no local clone"):
            materialize_repo("radient-ml/absent", tmp_path, clone_root=tmp_path)

    def test_a_promisor_transport_never_runs(self, tmp_path: Path) -> None:
        """S-18: a planted lazy-fetch transport fails closed, canary absent."""

        canary = tmp_path / "LAZY-CANARY"
        clone_root = tmp_path / "radient-ml"
        repo = _promisor_clone(clone_root, canary)
        missing = _git(repo, "rev-parse", "HEAD:missing.txt")

        # Negative control: with the owner's ordinary git, reading the absent
        # blob DOES run the planted transport.
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-p", missing],
            capture_output=True,
            env=_GIT_ENV,
            timeout=60,
            check=False,
        )
        assert canary.exists(), "fixture broken: the lazy fetch did not run the transport"
        canary.unlink()

        dest_root = tmp_path / "run-home" / "repos"
        dest_root.mkdir(parents=True)
        with pytest.raises(MaterializeError):
            materialize_repo("radient-ml/agent-server", dest_root, clone_root=clone_root)
        assert not canary.exists()

    def test_a_symlinked_clone_is_refused(self, tmp_path: Path) -> None:
        """R1-4: ``~/radient-ml/<name>`` -> another tree is not materialized."""

        elsewhere = tmp_path / "minervaai"
        _planted_clone(elsewhere, tmp_path / "unused-canary")
        clone_root = tmp_path / "radient-ml"
        clone_root.mkdir()
        (clone_root / "agent-server").symlink_to(elsewhere / "agent-server")
        with pytest.raises(MaterializeError, match="symlink"):
            materialize_repo("radient-ml/agent-server", tmp_path / "dest", clone_root=clone_root)

    @pytest.mark.parametrize("redirect", ["commondir", "objects/info/alternates"])
    def test_a_redirecting_git_dir_is_refused(self, tmp_path: Path, redirect: str) -> None:
        """R2-2: a real ``.git`` that borrows another repository is not followed."""

        elsewhere = _planted_clone(tmp_path / "minervaai", tmp_path / "unused-canary")
        clone_root = tmp_path / "radient-ml"
        _planted_clone(clone_root, tmp_path / "unused-canary-2")
        # _planted_clone names its repo agent-server under each root.
        target = clone_root / "agent-server" / ".git" / redirect
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{elsewhere / '.git'}\n")
        with pytest.raises(MaterializeError, match="redirects"):
            materialize_repo("radient-ml/agent-server", tmp_path / "dest", clone_root=clone_root)

    def test_a_gitfile_clone_is_refused(self, tmp_path: Path) -> None:
        """R1-4: a ``.git`` FILE redirecting to another repository is not followed."""

        elsewhere = tmp_path / "minervaai"
        real = _planted_clone(elsewhere, tmp_path / "unused-canary")
        clone = tmp_path / "radient-ml" / "agent-server"
        clone.mkdir(parents=True)
        (clone / ".git").write_text(f"gitdir: {real / '.git'}\n")
        with pytest.raises(MaterializeError, match="no .git directory"):
            materialize_repo(
                "radient-ml/agent-server", tmp_path / "dest", clone_root=tmp_path / "radient-ml"
            )


# --------------------------------------------------------------------------
# Isolation attestation
# --------------------------------------------------------------------------


class TestAttestation:
    def _creds(self, arn: str = _GOOD_ARN) -> PinnedCredentials:
        return PinnedCredentials("ASIAEXAMPLEKEY123456", "s", "t", "", arn)

    def test_a_missing_attestation_refuses(self) -> None:
        refusal = attestation_refusal(None)
        assert refusal is not None and refusal.startswith("isolation_missing")

    def test_a_complete_attestation_passes_and_holds_no_secret(
        self, fake_host: dict[str, Path]
    ) -> None:
        jail = _jail(fake_host["run_home"])
        env = jail.child_env(self._creds().env())
        attestation = attest(jail, env, self._creds())
        assert attestation_refusal(attestation) is None
        assert "s" not in attestation.env_keys and "AWS_SECRET_ACCESS_KEY" in attestation.env_keys
        assert attestation.jail_profile_sha256 == jail.profile_sha256()

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"jail_profile_sha256": ""}, "digest"),
            ({"env_keys": ("PATH", "AWS_PROFILE")}, "AWS_PROFILE"),
            ({"assumed_role_arn": "arn:aws:sts::951603962783:assumed-role/admin/x"}, "identity"),
            ({"repos": (("minervaai/core", "a" * 40),)}, "not pinned"),
            ({"repos": (("radient-ml/agent-server", ""),)}, "no commit"),
        ],
    )
    def test_a_widened_attestation_refuses(self, change: Mapping[str, Any], match: str) -> None:
        base = {
            "jail_profile_sha256": "0" * 64,
            "env_keys": ("HOME", "PATH"),
            "assumed_role_arn": _GOOD_ARN,
            "repos": (),
        }
        refusal = attestation_refusal(IsolationAttestation(**{**base, **change}))
        assert refusal is not None and match in refusal


# --------------------------------------------------------------------------
# The kernel boundary (macOS only): each probe with an unjailed control
# --------------------------------------------------------------------------


def _run(
    argv: list[str], *, env: Mapping[str, str], cwd: Path
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv, capture_output=True, env=dict(env), cwd=cwd, timeout=60, check=False
    )


@DARWIN_ONLY
class TestKernelBoundary:
    @pytest.mark.parametrize(
        "relpath",
        [
            "workspace/export.csv",
            ".zsh_history",
            "radient-ml/secret-repo/README.md",
            ".local-operator/sessions/other/transcript.jsonl",
        ],
    )
    def test_host_data_reads_are_eperm_with_an_unjailed_control(
        self, fake_host: dict[str, Path], relpath: str
    ) -> None:
        jail = _jail(fake_host["run_home"])
        target = fake_host["home"] / relpath
        control = _run(["/bin/cat", str(target)], env=os.environ, cwd=jail.home)
        assert control.returncode == 0 and control.stdout
        jailed = _run(jail.wrap(["/bin/cat", str(target)]), env=jail.child_env(), cwd=jail.home)
        assert jailed.returncode != 0
        assert b"Operation not permitted" in jailed.stderr
        assert not jailed.stdout

    def test_a_materialized_repo_reads_but_does_not_write(self, fake_host: dict[str, Path]) -> None:
        jail = _jail(fake_host["run_home"])
        repo = jail.repos_dir / "agent-server"
        repo.mkdir()
        (repo / "README.md").write_text("ok\n")
        read = _run(
            jail.wrap(["/bin/cat", str(repo / "README.md")]), env=jail.child_env(), cwd=jail.home
        )
        assert read.returncode == 0 and read.stdout == b"ok\n"
        write = _run(
            jail.wrap(["/usr/bin/touch", str(repo / "new")]), env=jail.child_env(), cwd=jail.home
        )
        assert write.returncode != 0 and not (repo / "new").exists()

    @pytest.mark.parametrize("relpath", [".local-operator/responders.yaml", ".zshrc"])
    def test_writes_outside_the_home_are_eperm(
        self, fake_host: dict[str, Path], relpath: str
    ) -> None:
        jail = _jail(fake_host["run_home"])
        target = fake_host["home"] / relpath
        before = target.read_bytes()
        script = f"open({str(target)!r}, 'a').write('planted')"
        jailed = _run(
            jail.wrap(["/usr/bin/python3", "-c", script]), env=jail.child_env(), cwd=jail.home
        )
        assert jailed.returncode != 0
        assert target.read_bytes() == before

    def test_loopback_is_refused_and_the_deny_is_load_bearing(
        self, fake_host: dict[str, Path]
    ) -> None:
        """S-15b on an ephemeral port: 443 cannot be bound unprivileged here."""

        jail = _jail(fake_host["run_home"])
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        port = listener.getsockname()[1]
        stop = threading.Event()

        def accept() -> None:
            listener.settimeout(0.2)
            while not stop.is_set():
                try:
                    conn, _ = listener.accept()
                    conn.close()
                except OSError:
                    continue

        thread = threading.Thread(target=accept, daemon=True)
        thread.start()
        try:
            probe = [
                "/usr/bin/python3",
                "-c",
                "import socket;"
                f"socket.create_connection(('127.0.0.1',{port}),timeout=5);print('CONNECTED')",
            ]
            profile = jail._render(egress_port=port)
            jailed = _run(
                ["/usr/bin/sandbox-exec", "-p", profile, *probe],
                env=jail.child_env(),
                cwd=jail.home,
            )
            assert jailed.returncode != 0 and b"CONNECTED" not in jailed.stdout
            # Negative control: the same profile WITHOUT the loopback deny connects,
            # so the deny line -- not some other rule -- is what refuses.
            no_deny = profile.replace('(deny network-outbound (remote ip "localhost:*"))\n', "")
            assert no_deny != profile
            open_run = _run(
                ["/usr/bin/sandbox-exec", "-p", no_deny, *probe],
                env=jail.child_env(),
                cwd=jail.home,
            )
            assert open_run.returncode == 0 and b"CONNECTED" in open_run.stdout
        finally:
            stop.set()
            thread.join(timeout=5)
            listener.close()

    def test_af_unix_connect_is_refused(self, fake_host: dict[str, Path], tmp_path: Path) -> None:
        jail = _jail(fake_host["run_home"])
        path = Path("/tmp") / f"lop-c1-{os.getpid()}.sock"
        server = socket.socket(socket.AF_UNIX)
        server.bind(str(path))
        server.listen(1)
        try:
            probe = [
                "/usr/bin/python3",
                "-c",
                "import socket;s=socket.socket(socket.AF_UNIX);"
                f"s.connect({str(path)!r});print('CONNECTED')",
            ]
            control = _run(probe, env=os.environ, cwd=jail.home)
            assert b"CONNECTED" in control.stdout
            jailed = _run(jail.wrap(probe), env=jail.child_env(), cwd=jail.home)
            assert jailed.returncode != 0 and b"CONNECTED" not in jailed.stdout
        finally:
            server.close()
            path.unlink(missing_ok=True)

    def test_keychain_is_unreachable(self, fake_host: dict[str, Path]) -> None:
        jail = _jail(fake_host["run_home"])
        jailed = _run(
            jail.wrap(["/usr/bin/security", "list-keychains", "-d", "user"]),
            env=jail.child_env(),
            cwd=jail.home,
        )
        assert b"keychain" not in jailed.stdout.lower()

    def test_homebrew_state_is_eperm_with_an_unjailed_control(
        self, fake_host: dict[str, Path]
    ) -> None:
        """S-16: the owner's local data stores under ``/opt/homebrew/var``."""

        var = Path("/opt/homebrew/var")
        target = (
            next((p for p in sorted(var.rglob("*")) if p.is_file()), None) if var.is_dir() else None
        )
        if target is None or not os.access(target, os.R_OK):
            pytest.skip("no readable file under /opt/homebrew/var on this host")
        jail = _jail(fake_host["run_home"])
        probe = ["/usr/bin/head", "-c", "1", str(target)]
        control = _run(probe, env=os.environ, cwd=jail.home)
        assert control.returncode == 0
        jailed = _run(jail.wrap(probe), env=jail.child_env(), cwd=jail.home)
        assert jailed.returncode != 0 and not jailed.stdout

    def test_a_hardlink_to_host_data_cannot_be_made(self, fake_host: dict[str, Path]) -> None:
        """S-20a: a link(2) into the home would pass the in-process path check."""

        jail = _jail(fake_host["run_home"])
        source = fake_host["home"] / "workspace" / "export.csv"
        link = jail.home / "h"
        jailed = _run(
            jail.wrap(["/bin/ln", str(source), str(link)]), env=jail.child_env(), cwd=jail.home
        )
        assert jailed.returncode != 0 and not link.exists()

    def test_udp_is_refused(self, fake_host: dict[str, Path]) -> None:
        """S-20d: only filtered TCP :443 and the DNS socket leave the jail."""

        jail = _jail(fake_host["run_home"])
        probe = [
            "/usr/bin/python3",
            "-c",
            "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);"
            "s.sendto(b'x',('192.0.2.1',9));print('SENT')",
        ]
        control = _run(probe, env=os.environ, cwd=jail.home)
        assert b"SENT" in control.stdout
        jailed = _run(jail.wrap(probe), env=jail.child_env(), cwd=jail.home)
        assert jailed.returncode != 0 and b"SENT" not in jailed.stdout

    def test_usr_local_is_eperm_while_the_toolchain_runs(self, fake_host: dict[str, Path]) -> None:
        """S-16 / R2-1: ``/usr`` no longer covers ``/usr/local``."""

        if not Path("/usr/local").is_dir():
            pytest.skip("no /usr/local on this host")
        jail = _jail(fake_host["run_home"])
        probe = ["/bin/ls", "/usr/local"]
        control = _run(probe, env=os.environ, cwd=jail.home)
        assert control.returncode == 0
        jailed = _run(jail.wrap(probe), env=jail.child_env(), cwd=jail.home)
        assert jailed.returncode != 0 and not jailed.stdout
        for tool in (["/usr/bin/git", "--version"], ["/usr/bin/python3", "-V"]):
            ran = _run(jail.wrap(tool), env=jail.child_env(), cwd=jail.home)
            assert ran.returncode == 0, (tool, ran.stderr[-200:])
