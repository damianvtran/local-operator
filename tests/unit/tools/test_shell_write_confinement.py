"""The scout's write-confined shell: ``ShellWriteConfinement`` and its bash path.

WHAT IS PINNED, and why each half exists:

* **Profile shape (every platform).** The guarantee is the ABSENCE of certain
  allows -- no network allow anywhere, no bare mach-lookup, no write allow
  beyond the root and the device literals -- plus the credential denials
  sitting AFTER the broad read allow. A future edit that reintroduces any of
  these fails here before it fails a probe run.
* **Fail closed (every platform).** A host that cannot apply the sandbox gets
  the refusal and NO spawn at all -- asserted with a spawn that would fail the
  test if it were ever reached.
* **What the child is handed (every platform).** No session credential, no
  borrowed GitHub token, the strict environment policy, ``TMPDIR`` inside the
  root, and ``sandbox-exec`` as argv[0].
* **The boundary itself (macOS only).** Real commands through the real
  ``execute_bash`` under the real ``sandbox-exec``: the work a scout needs
  succeeds, and every mutation outside the root is refused WITH the outside
  file checked intact afterwards -- an exit code alone would also pass for a
  command that wrote and then failed. CI's macOS job does not run this file
  (it runs ``tui-e2e`` only), so these are run locally and their output is
  recorded on the PR.
"""

from __future__ import annotations

import os
import signal as signal_module
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.tools import builtin
from local_operator.tools import confinement as confinement_module
from local_operator.tools import shell_env
from local_operator.tools.confinement import (
    ShellWriteConfinement,
    shell_write_confinement_of,
)

DARWIN_ONLY = pytest.mark.skipif(
    sys.platform != "darwin" or not confinement_module.SANDBOX_EXEC.exists(),
    reason="the write boundary is a macOS seatbelt profile; the refusal path is covered below",
)

#: A git environment that can never reach the operator's keychain or global
#: config while the test BUILDS its fixture repository (outside the sandbox):
#: the system gitconfig on this fleet carries ``credential.helper=osxkeychain``.
_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "/bin/false",
}


def _context(root: Path, *, cwd: Path | None = None, variables: Any = None) -> ToolContext:
    """The context a scout child's bash call carries: write root == scratchpad."""

    return ToolContext(
        cwd=str(cwd or root),
        session_id="shell-write-test",
        shell_write_root=str(root),
        scratchpad_dir=str(root),
        variables=variables,
    )


async def _run(root: Path, command: str, *, cwd: Path | None = None) -> Any:
    return await builtin.execute_bash(
        "scout-bash", {"command": command}, AbortSignal(), None, _context(root, cwd=cwd)
    )


@pytest.fixture
def scout_dirs(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    """``(root, outside, home)``: the scout's scratch, a sibling it must not
    touch, and an isolated HOME / config dir whose credential stores the
    profile must deny. Resolved, because seatbelt matches real paths and
    macOS's temp dir sits behind the ``/var`` -> ``/private/var`` link."""

    base = tmp_path.resolve()
    root = base / "scratch"
    outside = base / "outside"
    home = base / "home"
    for directory in (root, outside, home):
        directory.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home / ".local-operator"))
    return root, outside, home


# ---------------------------------------------------------------------------
# Profile shape and plumbing (every platform)
# ---------------------------------------------------------------------------


class TestProfileShape:
    def test_default_deny_no_network_allow_and_writes_only_to_the_root(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, _outside, _home = scout_dirs
        profile = ShellWriteConfinement.at(root).profile()
        assert "(deny default)" in profile
        assert "(deny network*)" in profile
        # The actual guarantee: no network allow in ANY spelling (a network
        # allow defeats the deny wherever it sits -- see ToolConfinement).
        assert "(allow network" not in profile
        assert "(allow default" not in profile
        write_lines = [line for line in profile.splitlines() if "file-write" in line]
        assert len(write_lines) == 1, write_lines
        assert write_lines[0].startswith(f'(allow file-write* (subpath "{root}")')
        assert write_lines[0].count("(subpath ") == 1
        for literal in confinement_module._SYSTEM_WRITE_LITERALS:
            assert f'(literal "{literal}")' in write_lines[0]

    def test_mach_lookup_is_the_same_exact_name_allowlist(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, _outside, _home = scout_dirs
        profile = ShellWriteConfinement.at(root).profile()
        mach_lines = [line for line in profile.splitlines() if "mach-lookup" in line]
        assert len(mach_lines) == 1
        assert mach_lines[0].strip() != "(allow mach-lookup)"
        for name in confinement_module._MACH_LOOKUP_GLOBAL_NAMES:
            assert f'(global-name "{name}")' in mach_lines[0]
        assert "pasteboard" not in profile and "Security" not in profile

    def test_credential_denials_come_after_the_broad_read_allow(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        """Rule order is load-bearing (see ``kernel_exposed_read_subtrees``):
        a deny placed BEFORE a filtered allow is overridden on this host. Last
        is correct under every measured behaviour, so last is pinned."""

        root, _outside, home = scout_dirs
        lines = ShellWriteConfinement.at(root).profile().splitlines()
        read_allow = lines.index("(allow file-read*)")
        deny = next(i for i, line in enumerate(lines) if line.startswith("(deny file-read*"))
        assert deny > read_allow
        assert deny == len(lines) - 1
        denied = lines[deny]
        config = home / ".local-operator"
        assert f'(subpath "{config / "secrets"}")' in denied
        for rel in (".ssh", "Library/Keychains", ".aws", ".config/gh", ".netrc"):
            assert f'(subpath "{home / rel}")' in denied, rel
        assert "credentials|auth\\.db" in denied
        assert confinement_module._DOTENV_REGEX in denied

    def test_the_operators_real_config_dir_is_denied_even_when_isolated(
        self, scout_dirs: tuple[Path, Path, Path], monkeypatch, tmp_path: Path
    ) -> None:
        """An isolated ``LOCAL_OPERATOR_CONFIG_DIR`` must not leave the default
        ``~/.local-operator`` store readable: both roots are denied."""

        root, _outside, home = scout_dirs
        elsewhere = (tmp_path / "isolated-config").resolve()
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(elsewhere))
        denied = ShellWriteConfinement.at(root).profile().splitlines()[-1]
        assert f'(subpath "{elsewhere / "secrets"}")' in denied
        assert f'(subpath "{home / ".local-operator" / "secrets"}")' in denied

    def test_context_plumbing_and_temp_dir(self, tmp_path: Path) -> None:
        assert shell_write_confinement_of(None) is None
        assert shell_write_confinement_of(ToolContext(cwd=str(tmp_path))) is None
        confinement = shell_write_confinement_of(_context(tmp_path))
        assert confinement is not None and confinement.root == tmp_path.resolve()
        temp = confinement.temp_dir()
        assert temp == tmp_path.resolve() / "tmp" and temp.is_dir()


class TestBashToolSurface:
    def test_tier_is_read_only_when_write_confined(self, tmp_path: Path) -> None:
        ordinary = builtin.build_bash_tool()
        confined = builtin.build_bash_tool(_context(tmp_path))
        assert ordinary.approval_tier == "exec"
        assert confined.approval_tier == "read"
        assert "SANDBOXED" in confined.description
        assert "Operation not permitted" in confined.description

    def test_the_default_sessions_schema_is_unchanged(self, tmp_path: Path) -> None:
        """The sandbox note is paid only by scout children: an ordinary
        context builds the exact description it built before, so the
        start-of-session context budget does not move."""

        plain = builtin.build_bash_tool(ToolContext(cwd=str(tmp_path)))
        assert plain.description == builtin.build_bash_tool().description
        assert "SANDBOXED" not in plain.description
        assert plain.approval_tier == "exec"


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_no_sandbox_means_refusal_and_no_spawn(
        self, scout_dirs: tuple[Path, Path, Path], monkeypatch, tmp_path: Path
    ) -> None:
        root, _outside, _home = scout_dirs
        monkeypatch.setattr(confinement_module, "SANDBOX_EXEC", tmp_path / "no-sandbox-exec")

        async def must_not_spawn(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"spawned unwrapped: {args[:3]}")

        monkeypatch.setattr(builtin.asyncio, "create_subprocess_exec", must_not_spawn)
        result = await _run(root, "echo hi")
        assert result.is_error
        assert "refuses to run commands it cannot confine" in result.text
        assert str(root) in result.text

    @pytest.mark.asyncio
    async def test_a_failed_apply_probe_also_refuses(
        self, scout_dirs: tuple[Path, Path, Path], monkeypatch
    ) -> None:
        """``sandbox-exec`` present but unable to apply a profile (lop itself
        sandboxed) is the case the cached probe exists for."""

        root, _outside, _home = scout_dirs
        monkeypatch.setattr(confinement_module, "sandbox_unavailable_reason", lambda: None)
        monkeypatch.setattr(
            confinement_module, "_sandbox_apply_probe_reason", lambda: "sandbox_apply: EPERM"
        )

        async def must_not_spawn(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("spawned unwrapped")

        monkeypatch.setattr(builtin.asyncio, "create_subprocess_exec", must_not_spawn)
        result = await _run(root, "echo hi")
        assert result.is_error and "sandbox_apply: EPERM" in result.text


class TestWhatTheChildIsHanded:
    @pytest.mark.asyncio
    async def test_no_credentials_no_github_token_strict_env_temp_in_root(
        self, scout_dirs: tuple[Path, Path, Path], monkeypatch
    ) -> None:
        root, _outside, _home = scout_dirs
        monkeypatch.setattr(confinement_module, "shell_sandbox_unavailable_reason", lambda: None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient-provider-key")

        # The REAL store, with a credential stored the way ``lop`` stores one:
        # what an ordinary session's bash would inject into the child.
        from local_operator.variables import VariableStore

        store = VariableStore(cwd=".")
        store.store_credential("SESSION_SECRET_TOKEN", "s3cr3t-value")

        async def no_github(_context: Any) -> dict[str, str]:
            raise AssertionError("a write-confined shell must not borrow a GitHub token")

        monkeypatch.setattr(builtin, "_github_git_injections", no_github)
        seen_policy: list[Any] = []
        real_child_environment = shell_env.child_environment

        def spy_child_environment(policy: Any = None, **kwargs: Any) -> dict[str, str]:
            seen_policy.append(policy)
            return real_child_environment(policy, **kwargs)

        monkeypatch.setattr(shell_env, "child_environment", spy_child_environment)
        captured: dict[str, Any] = {}

        async def capture(*argv: Any, **kwargs: Any) -> Any:
            captured["argv"] = argv
            captured["env"] = kwargs["env"]
            raise OSError("captured; not spawning")

        monkeypatch.setattr(builtin.asyncio, "create_subprocess_exec", capture)
        await builtin.execute_bash(
            "scout-bash",
            {"command": "env"},
            AbortSignal(),
            None,
            _context(root, variables=store),
        )
        argv, env = captured["argv"], captured["env"]
        assert argv[0] == str(confinement_module.SANDBOX_EXEC) and argv[1] == "-p"
        assert "SESSION_SECRET_TOKEN" not in env
        assert "OPENAI_API_KEY" not in env
        assert not {"GH_TOKEN", "GITHUB_TOKEN"} & set(env)
        assert seen_policy == [shell_env.CONFINED_SHELL_POLICY]
        assert env["TMPDIR"] == str(root / "tmp") and env["TMP"] == str(root / "tmp")


# ---------------------------------------------------------------------------
# The real boundary (macOS): real commands, real sandbox-exec
# ---------------------------------------------------------------------------


@DARWIN_ONLY
class TestWhatAScoutCanDo:
    @pytest.mark.asyncio
    async def test_writes_inside_the_root(self, scout_dirs: tuple[Path, Path, Path]) -> None:
        root, _outside, _home = scout_dirs
        result = await _run(root, 'echo hello > "$LOCAL_OPERATOR_SCRATCHPAD/f.txt" && cat f.txt')
        assert not result.is_error, result.text
        assert (root / "f.txt").read_text().strip() == "hello"

    @pytest.mark.asyncio
    async def test_binary_inspection_tools_work(self, scout_dirs: tuple[Path, Path, Path]) -> None:
        root, _outside, _home = scout_dirs
        result = await _run(
            root, "strings /bin/ls | head -3 && otool -L /bin/ls | head -2 && file /bin/ls"
        )
        assert not result.is_error, result.text
        assert "/bin/ls:" in result.text and "Mach-O" in result.text

    @pytest.mark.asyncio
    async def test_python_with_tempfile_runs(self, scout_dirs: tuple[Path, Path, Path]) -> None:
        """``TMPDIR`` in the root is what lets ``tempfile`` work at all here."""

        root, outside, _home = scout_dirs
        (outside / "data.json").write_text('{"k": [1, 2, 3]}')
        script = (
            "import json, sys, tempfile; "
            f"d = json.load(open({str(outside / 'data.json')!r})); "
            "f = tempfile.NamedTemporaryFile(delete=False); f.write(b'x'); f.close(); "
            "print('sum', sum(d['k']), f.name)"
        )
        result = await _run(root, f'python3 -c "{script}"')
        assert not result.is_error, result.text
        assert "sum 6" in result.text and str(root / "tmp") in result.text

    @pytest.mark.asyncio
    async def test_tar_extracts_into_the_root(self, scout_dirs: tuple[Path, Path, Path]) -> None:
        root, outside, _home = scout_dirs
        archive = _make_archive(outside)
        result = await _run(root, f"mkdir -p x && tar -xf {archive} -C x && cat x/payload/a.txt")
        assert not result.is_error, result.text
        assert (root / "x" / "payload" / "a.txt").read_text() == "archived\n"

    @pytest.mark.asyncio
    async def test_git_reads_work_outside_the_root(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, outside, _home = scout_dirs
        repo = _make_repo(outside)
        result = await _run(root, "git log --oneline && git status --short", cwd=repo)
        assert not result.is_error, result.text
        assert "initial" in result.text


@DARWIN_ONLY
class TestWhatAScoutCannotDo:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "command",
        [
            "echo pwned > {target}",
            "echo pwned >> {target}",
            "rm -f {target}",
            "mv {target} {root}/stolen",
            "chmod 000 {target}",
            "touch -t 200001010000 {target}",
            "python3 -c \"open('{target}', 'w').write('pwned')\"",
            "cp /etc/hosts {outside}/new-file",
            "mkdir {outside}/new-dir",
        ],
    )
    async def test_mutations_outside_the_root_are_refused_and_change_nothing(
        self, scout_dirs: tuple[Path, Path, Path], command: str
    ) -> None:
        root, outside, _home = scout_dirs
        target = outside / "target.txt"
        target.write_text("original\n")
        before = target.stat()
        result = await _run(
            root, command.format(target=target, root=root, outside=outside) + " 2>&1"
        )
        assert "Operation not permitted" in result.text, result.text
        after = target.stat()
        assert target.read_text() == "original\n"
        assert (after.st_mode, after.st_mtime_ns, after.st_size) == (
            before.st_mode,
            before.st_mtime_ns,
            before.st_size,
        )
        assert sorted(p.name for p in outside.iterdir()) == ["target.txt"]
        assert not (root / "stolen").exists()

    @pytest.mark.asyncio
    async def test_a_symlink_from_the_root_does_not_carry_a_write_out(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        """The vnode is what the kernel checks, so a link planted INSIDE the
        root and written through is the same refused write."""

        root, outside, _home = scout_dirs
        target = outside / "target.txt"
        target.write_text("original\n")
        result = await _run(root, f"ln -s {target} link && echo pwned > link 2>&1; echo rc=$?")
        assert "Operation not permitted" in result.text, result.text
        assert target.read_text() == "original\n"

    @pytest.mark.asyncio
    async def test_a_hard_link_into_the_root_is_refused(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, outside, _home = scout_dirs
        target = outside / "target.txt"
        target.write_text("original\n")
        result = await _run(root, f"ln {target} hard 2>&1; echo pwned >> hard 2>&1; true")
        assert target.read_text() == "original\n", result.text
        assert target.stat().st_nlink == 1, result.text

    @pytest.mark.asyncio
    async def test_tar_cannot_extract_outside_the_root(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, outside, _home = scout_dirs
        archive = _make_archive(outside)
        dest = outside / "dest"
        dest.mkdir()
        result = await _run(root, f"tar -xf {archive} -C {dest} 2>&1")
        assert "Operation not permitted" in result.text or "Cannot" in result.text, result.text
        assert list(dest.iterdir()) == []

    @pytest.mark.asyncio
    async def test_git_add_and_commit_are_refused_and_head_is_unchanged(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, outside, _home = scout_dirs
        repo = _make_repo(outside)
        head = _git(repo, "rev-parse", "HEAD")
        (repo / "new.txt").write_text("new\n")
        result = await _run(
            root,
            "git add new.txt 2>&1; "
            "git -c user.name=s -c user.email=s@x commit -qam scout-commit 2>&1; "
            "git log --oneline | head -1",
            cwd=repo,
        )
        assert (
            "Operation not permitted" in result.text or "Unable to create" in result.text
        ), result.text
        assert _git(repo, "rev-parse", "HEAD") == head
        assert "new.txt" not in _git(repo, "diff", "--cached", "--name-only")
        assert "initial" in result.text  # the read half still answered

    @pytest.mark.asyncio
    async def test_loopback_connect_is_refused_and_the_listener_sees_nothing(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, _outside, _home = scout_dirs
        accepted: list[Any] = []
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(0.2)
        port = server.getsockname()[1]
        stop = threading.Event()

        def accept_loop() -> None:
            while not stop.is_set():
                try:
                    accepted.append(server.accept())
                except OSError:
                    continue

        thread = threading.Thread(target=accept_loop, daemon=True)
        thread.start()
        try:
            script = (
                "import socket; s = socket.socket(); "
                f"s.connect(('127.0.0.1', {port})); print('CONN' + 'ECTED')"
            )
            result = await _run(root, f'python3 -c "{script}" 2>&1')
        finally:
            stop.set()
            thread.join(timeout=2)
            server.close()
        assert "CONNECTED" not in result.text
        assert "Operation not permitted" in result.text, result.text
        assert accepted == []

    @pytest.mark.asyncio
    async def test_unix_socket_and_curl_are_refused(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, outside, _home = scout_dirs
        # AF_UNIX paths are capped at 104 bytes on macOS and pytest's temp
        # dir is longer than that; a short name under /private/tmp keeps the
        # listener OUTSIDE the scout root, which is all this test needs.
        sock_path = Path(f"/private/tmp/lo-scout-{os.getpid()}.sock")
        sock_path.unlink(missing_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(sock_path))
        server.listen(1)
        try:
            script = (
                "import socket; s = socket.socket(socket.AF_UNIX); "
                f"s.connect({str(sock_path)!r}); print('CONN' + 'ECTED')"
            )
            result = await _run(
                root,
                f'python3 -c "{script}" 2>&1; curl -sS -m 5 https://example.com 2>&1; echo rc=$?',
            )
        finally:
            server.close()
            sock_path.unlink(missing_ok=True)
        assert "CONNECTED" not in result.text
        assert "Operation not permitted" in result.text, result.text
        assert "rc=0" not in result.text

    @pytest.mark.asyncio
    async def test_kill_of_a_process_outside_the_sandbox_is_refused(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, _outside, _home = scout_dirs
        victim = subprocess.Popen(["/bin/sleep", "30"])
        try:
            result = await _run(root, f"kill -TERM {victim.pid} 2>&1; echo rc=$?")
            time.sleep(0.2)
            assert victim.poll() is None, result.text
            assert "Operation not permitted" in result.text, result.text
        finally:
            victim.send_signal(signal_module.SIGKILL)
            victim.wait(timeout=5)

    @pytest.mark.asyncio
    async def test_a_nested_permissive_sandbox_cannot_escape(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, outside, _home = scout_dirs
        result = await _run(
            root,
            f"sandbox-exec -p '(version 1)(allow default)' /usr/bin/touch {outside}/escaped 2>&1",
        )
        assert not (outside / "escaped").exists(), result.text
        assert "Operation not permitted" in result.text, result.text


@DARWIN_ONLY
class TestCredentialStoresAreUnreadable:
    @pytest.mark.asyncio
    async def test_config_secrets_are_unreadable_directly_and_through_a_symlink(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, _outside, home = scout_dirs
        config = home / ".local-operator"
        (config / "secrets").mkdir(parents=True)
        (config / "secrets" / "master.key").write_text("FAKE-MASTER-KEY\n")
        (config / "credentials.env").write_text("FAKE_CRED=1\n")
        (config / "auth.db").write_text("FAKE-AUTH\n")
        (config / "config.yml").write_text("sibling: readable\n")
        result = await _run(
            root,
            f"cat {config}/secrets/master.key 2>&1; "
            f"cat {config}/credentials.env 2>&1; "
            f"cat {config}/auth.db 2>&1; "
            f"ln -s {config}/secrets/master.key via-link && cat via-link 2>&1; "
            f"cat {config}/config.yml",
        )
        for leaked in ("FAKE-MASTER-KEY", "FAKE_CRED", "FAKE-AUTH"):
            assert leaked not in result.text, result.text
        assert result.text.count("Operation not permitted") >= 4, result.text
        assert "sibling: readable" in result.text

    @pytest.mark.asyncio
    async def test_home_credential_stores_and_dotenv_files_are_unreadable(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        root, outside, home = scout_dirs
        (home / ".ssh").mkdir()
        (home / ".ssh" / "id_ed25519").write_text("FAKE-SSH-KEY\n")
        (home / ".netrc").write_text("FAKE-NETRC\n")
        (outside / ".env").write_text("FAKE_DOTENV=1\n")
        (outside / "prod.env.local").write_text("FAKE_PROD=1\n")
        (outside / "envy.txt").write_text("not-a-dotenv\n")
        result = await _run(
            root,
            f"cat {home}/.ssh/id_ed25519 {home}/.netrc {outside}/.env "
            f"{outside}/prod.env.local {outside}/envy.txt 2>&1",
        )
        for leaked in ("FAKE-SSH-KEY", "FAKE-NETRC", "FAKE_DOTENV", "FAKE_PROD"):
            assert leaked not in result.text, result.text
        assert "not-a-dotenv" in result.text


@DARWIN_ONLY
class TestSeatbeltRuleOrder:
    """Pins the precedence the profile's rule order relies on (and that the
    corrected ``kernel_exposed_read_subtrees`` docstring states)."""

    @staticmethod
    def _read(profile: str, path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(confinement_module.SANDBOX_EXEC), "-p", profile, "/bin/cat", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_later_rule_wins_for_a_filtered_allow(self, tmp_path: Path) -> None:
        base = tmp_path.resolve()
        (base / "d").mkdir()
        secret = base / "d" / "s.txt"
        secret.write_text("hi\n")
        head = (
            "(version 1)(deny default)(allow process-fork)(allow process-exec)(allow sysctl-read)"
        )
        allow = '(allow file-read* (subpath "/"))'
        deny = f'(deny file-read* (subpath "{base / "d"}"))'
        assert self._read(head + allow + deny, secret).returncode != 0
        assert self._read(head + deny + allow, secret).stdout == "hi\n"

    def test_a_filtered_deny_beats_the_unfiltered_allow_in_either_order(
        self, tmp_path: Path
    ) -> None:
        base = tmp_path.resolve()
        (base / "d").mkdir()
        secret = base / "d" / "s.txt"
        secret.write_text("hi\n")
        head = (
            "(version 1)(deny default)(allow process-fork)(allow process-exec)(allow sysctl-read)"
        )
        deny = f'(deny file-read* (subpath "{base / "d"}"))'
        assert self._read(head + "(allow file-read*)" + deny, secret).returncode != 0
        assert self._read(head + deny + "(allow file-read*)", secret).returncode != 0


# ---------------------------------------------------------------------------
# Round 1 of PR #2987: derived denials (R2/Q1), unsafe roots (R3), and the
# non-blocking availability probe (R4)
# ---------------------------------------------------------------------------


class TestDerivedDenials:
    """The kernel deny list is DERIVED from ``supplements.denylist`` so the
    two cannot drift (R2). These pin the derivation, every platform."""

    def test_every_vocabulary_class_reaches_the_profile(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        from local_operator.supplements import denylist

        root, _outside, _home = scout_dirs
        denied = ShellWriteConfinement.at(root).profile().splitlines()[-1]
        names, suffixes, prefixes, parts, patterns = denylist._composed()
        for component in {*parts, *denylist._EXTRA_COMPONENTS}:
            if component in confinement_module._KERNEL_EXCLUDED_COMPONENTS:
                continue
            literal = confinement_module._seatbelt_regex_literal(component)
            assert f'(regex #"/{literal}(/|$)")' in denied, component
        for suffix in suffixes:
            literal = confinement_module._seatbelt_regex_literal(suffix)
            assert f"/[^/]*{literal}$" in denied, suffix
        leaf_globs = {*names, *denylist._EXTRA_NAMES, *patterns}
        leaf_globs |= {pattern for pattern, _rule in denylist._EXTRA_PATTERNS}
        for glob in leaf_globs - confinement_module._KERNEL_EXCLUDED_PATTERNS:
            regex = confinement_module._glob_to_seatbelt_regex(glob)
            assert f'(regex #"{regex}")' in denied, glob
        for sequence, _rule in denylist._COMPONENT_SEQUENCES:
            assert "/".join(sequence).replace(".", "\\.") in denied, sequence
        for directory, filename, _rule in denylist._DIR_FILE_PAIRS:
            assert f"{directory}/{filename}".replace(".", "\\.") in denied

    def test_exclusions_still_name_live_upstream_rules(self) -> None:
        """An exclusion that no longer matches an upstream rule would silently
        exclude nothing; and the glob translation handles only ``*``/``?``."""
        from local_operator.supplements import denylist

        _names, _suffixes, _prefixes, parts, patterns = denylist._composed()
        upstream_patterns = {*patterns, *(p for p, _rule in denylist._EXTRA_PATTERNS)}
        assert confinement_module._KERNEL_EXCLUDED_PATTERNS <= upstream_patterns
        assert confinement_module._KERNEL_EXCLUDED_COMPONENTS <= {
            *parts,
            *denylist._EXTRA_COMPONENTS,
        }
        for glob in upstream_patterns | set(denylist._EXTRA_NAMES):
            assert not set(glob) & set("[]{}!"), glob

    def test_lop_config_stores_are_denied_in_both_config_roots(
        self, scout_dirs: tuple[Path, Path, Path], monkeypatch, tmp_path: Path
    ) -> None:
        root, _outside, home = scout_dirs
        elsewhere = (tmp_path / "isolated-config").resolve()
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(elsewhere))
        denied = ShellWriteConfinement.at(root).profile().splitlines()[-1]
        for config in (elsewhere, home / ".local-operator"):
            for rel in confinement_module._CONFIG_SECRET_SUBPATHS:
                assert f'(subpath "{config / rel}")' in denied, (config, rel)
            assert "network/networks/[^/]*\\.secrets\\.json$" in denied

    def test_the_model_facing_note_does_not_overclaim(self, tmp_path: Path) -> None:
        description = builtin.build_bash_tool(_context(tmp_path)).description
        assert "known credential stores unreadable" in description
        assert "not every secret" in description


class TestUnsafeRoots:
    """R3: a path that could change the profile's meaning is refused, never
    interpolated. The reviewer's repro root is the first case."""

    @pytest.mark.parametrize(
        "name", ['"x) (allow default) (y', "back\\slash", "new\nline"], ids=["quote", "bs", "nl"]
    )
    @pytest.mark.asyncio
    async def test_an_unsafe_root_refuses_with_no_spawn(
        self, scout_dirs: tuple[Path, Path, Path], monkeypatch, name: str
    ) -> None:
        root, outside, _home = scout_dirs
        unsafe = root / name
        unsafe.mkdir()
        confinement = ShellWriteConfinement.at(unsafe)
        assert confinement_module.profile_path_refusal(confinement.root) is not None
        with pytest.raises(ValueError, match="sandbox profile cannot carry"):
            confinement.profile()
        assert confinement.wrap(["/bin/true"]) is None

        async def must_not_spawn(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"spawned: {args[:3]}")

        monkeypatch.setattr(builtin.asyncio, "create_subprocess_exec", must_not_spawn)
        result = await _run(unsafe, f"echo PWNED > {outside}/escaped")
        assert result.is_error
        assert "sandbox profile cannot carry" in result.text
        assert "refuses to run commands it cannot confine" in result.text
        assert not (outside / "escaped").exists()

    def test_an_unsafe_home_refuses_too(
        self, scout_dirs: tuple[Path, Path, Path], monkeypatch
    ) -> None:
        """The credential denials interpolate HOME: a HOME the profile cannot
        carry means they cannot be written either, so the shell refuses."""
        root, _outside, home = scout_dirs
        odd = home / 'h"ome'
        odd.mkdir()
        monkeypatch.setenv("HOME", str(odd))
        confinement = ShellWriteConfinement.at(root)
        assert confinement.wrap(["/bin/true"]) is None
        assert "sandbox profile cannot carry" in confinement.spawn_refusal()

    def test_an_ordinary_root_is_accepted(self, scout_dirs: tuple[Path, Path, Path]) -> None:
        root, _outside, _home = scout_dirs
        spaced = root / "with space (and parens)"
        spaced.mkdir()
        assert confinement_module.profile_path_refusal(spaced) is None
        assert f'(subpath "{spaced.resolve()}")' in ShellWriteConfinement.at(spaced).profile()


class TestProbeOffTheLoop:
    """R4: the first apply probe is a spawn; from a coroutine it runs on a
    worker thread, is cached, and stays fail closed."""

    @pytest.mark.asyncio
    async def test_first_probe_runs_in_a_thread_then_hits_the_cache(self, monkeypatch) -> None:
        calls: list[str] = []

        def fake_run(*args: Any, **kwargs: Any) -> Any:
            calls.append(threading.current_thread().name)
            return subprocess.CompletedProcess(args[0], 0, b"", b"")

        monkeypatch.setattr(confinement_module, "sandbox_unavailable_reason", lambda: None)
        monkeypatch.setattr(confinement_module.subprocess, "run", fake_run)
        confinement_module._sandbox_apply_probe_reason.cache_clear()
        try:
            assert confinement_module.shell_sandbox_known_unavailable_reason() is None
            assert not confinement_module._probe_is_cached()
            assert await confinement_module.shell_sandbox_unavailable_reason_async() is None
            assert await confinement_module.shell_sandbox_unavailable_reason_async() is None
            assert confinement_module.shell_sandbox_unavailable_reason() is None
            assert len(calls) == 1
            assert calls[0] != threading.main_thread().name
        finally:
            confinement_module._sandbox_apply_probe_reason.cache_clear()

    @pytest.mark.asyncio
    async def test_a_failing_probe_is_still_a_reason(self, monkeypatch) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> Any:
            return subprocess.CompletedProcess(args[0], 1, b"", b"sandbox_apply: EPERM")

        monkeypatch.setattr(confinement_module, "sandbox_unavailable_reason", lambda: None)
        monkeypatch.setattr(confinement_module.subprocess, "run", fake_run)
        confinement_module._sandbox_apply_probe_reason.cache_clear()
        try:
            reason = await confinement_module.shell_sandbox_unavailable_reason_async()
            assert reason is not None and "EPERM" in reason
            assert confinement_module.shell_sandbox_known_unavailable_reason() == reason
        finally:
            confinement_module._sandbox_apply_probe_reason.cache_clear()


#: The stores round 1 measured READABLE from this shell (R2 and QA Q1), each
#: with a fake marker. Relative to HOME; ``{lop}`` is the config dir.
_ROUND1_STORES: tuple[str, ...] = (
    "{lop}/network/identity/device.json",
    "{lop}/network/networks/n-1.secrets.json",
    "{lop}/network/credentials/github.json",
    "{lop}/operator/operator-key.pem",
    "{lop}/tunnel/cloudflared.token",
    "{lop}/push-handle.key",
    "{lop}/mcp.json",
    ".codex/auth.json",
    ".kube/config",
    ".config/gcloud/credentials.db",
    ".docker/config.json",
    ".gnupg/private-keys-v1.d/k.key",
    ".config/op/config",
    ".terraform.d/credentials.tfrc.json",
    ".azure/msal_token_cache.json",
    # #3022 r2 M3: the desktop app's desktop-plane bearer.
    "Library/Application Support/Local Operator/desktop-token",
    "Library/Application Support/local-operator-ui/desktop-token",
)


@DARWIN_ONLY
class TestRoundOneStoresAreUnreadable:
    """Real ``sandbox-exec``: every store round 1 read is refused, and the
    ordinary neighbours a research shell needs still read."""

    @pytest.mark.asyncio
    async def test_each_store_is_refused(self, scout_dirs: tuple[Path, Path, Path]) -> None:
        root, _outside, home = scout_dirs
        paths = []
        for index, rel in enumerate(_ROUND1_STORES):
            path = home / rel.replace("{lop}", ".local-operator")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"FAKE-STORE-{index}\n")
            paths.append(path)
        script = "; ".join(f"cat '{path}' 2>&1" for path in paths)
        result = await _run(root, script)
        for index, path in enumerate(paths):
            assert f"FAKE-STORE-{index}" not in result.text, (path, result.text)
        assert result.text.count("Operation not permitted") >= len(paths), result.text

    @pytest.mark.asyncio
    async def test_neighbours_and_source_trees_still_read(
        self, scout_dirs: tuple[Path, Path, Path]
    ) -> None:
        """No over-deny: the config file, a network's public record, and the
        SOURCE directories named like credential classes (lop's own
        ``network/credentials/`` and ``secrets/`` packages, a tokenizer)."""
        root, outside, home = scout_dirs
        config = home / ".local-operator"
        (config / "network" / "networks").mkdir(parents=True)
        # Valid YAML: ``execute_bash`` reads config.yml and quarantines a
        # malformed one, which would make this a test of the quarantine.
        (config / "config.yml").write_text("marker: READ-CONFIG\n")
        (config / "network" / "networks" / "n-1.json").write_text("READ-NETWORK\n")
        source = outside / "repo" / "local_operator"
        (source / "network" / "credentials").mkdir(parents=True)
        (source / "network" / "credentials" / "github.py").write_text("READ-CRED-SRC\n")
        (source / "secrets").mkdir()
        (source / "secrets" / "store.py").write_text("READ-SECRETS-SRC\n")
        (source / "tokenizer.py").write_text("READ-TOKENIZER\n")
        result = await _run(
            root,
            f"cat {config}/config.yml {config}/network/networks/n-1.json; "
            f"ls {source}/network/credentials; cat {source}/network/credentials/github.py; "
            f"cat {source}/secrets/store.py {source}/tokenizer.py; "
            f"grep -rl READ- {source} | wc -l",
        )
        for marker in (
            "READ-CONFIG",
            "READ-NETWORK",
            "github.py",
            "READ-CRED-SRC",
            "READ-SECRETS-SRC",
            "READ-TOKENIZER",
        ):
            assert marker in result.text, (marker, result.text)
        assert "Operation not permitted" not in result.text, result.text


def _make_archive(where: Path) -> Path:
    payload = where / "payload"
    payload.mkdir()
    (payload / "a.txt").write_text("archived\n")
    archive = where / "bundle.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname="payload")
    return archive


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, env=_GIT_ENV, capture_output=True, text=True, check=True
    ).stdout.strip()


def _make_repo(where: Path) -> Path:
    repo = where / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "initial")
    return repo
