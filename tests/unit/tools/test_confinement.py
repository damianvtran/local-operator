"""The conversation confinement: what a confined session's tools may reach.

WHY THIS FILE EXISTS. A session's local tools run on the HOST, in the
operator's account. The benchmark's session engagement measured what that
means for an episode nobody is watching: the model ran

    find /Users/damian/worktrees/osworld ... -name prize-items.pdf
    pdftotext -layout /Users/damian/worktrees/osworld/gated/assets/task_013/...

and read a task's input out of the campaign apparatus's gated assets -- a tree
that also holds the adapter build and other tasks' material. The fix is a
kernel boundary: a confined session's shell children run inside a seatbelt
sandbox scoped to the session's scratch (macOS), the path tools refuse paths
that resolve outside it, and a host with no mechanism REFUSES the shell rather
than run it unwrapped.

The tests split by platform on purpose:

* the PATH decisions (which paths a confined session may touch, what refusal
  it renders, which tool call sites consult it) are pure logic and run
  everywhere -- CI included;
* the SHELL enforcement is macOS-only by construction and is exercised with
  the real ``/usr/bin/sandbox-exec``, skipped elsewhere. A skip here means the
  platform has no boundary to test, not that the boundary is untested: the
  fail-closed refusal for such hosts is covered below and runs everywhere.

The NETWORK half (landed 2026-09-28, completion of the same boundary) is
pinned in the same two registers: the profile shape and the in-process tool
refusals are pure logic, while ``TestShellBoundary.test_a_jailed_shell_cannot
_reach_the_network`` proves the denial with the real kernel sandbox -- including
a PRE-FIX CONTROL that patches the profile back to main's
``(allow network-outbound)`` so the test shows the leak reproduces when only
that line differs, rather than merely passing.
"""

from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.tools import builtin
from local_operator.tools import confinement as confinement_module
from local_operator.tools.confinement import ToolConfinement, confinement_of

DARWIN_ONLY = pytest.mark.skipif(
    sys.platform != "darwin" or not confinement_module.SANDBOX_EXEC.exists(),
    reason="the shell boundary is a macOS seatbelt profile; the refusal path is covered below",
)


def _context(root: Path, *, cwd: Path | None = None) -> ToolContext:
    return ToolContext(
        cwd=str(cwd or root),
        session_id="confinement-test",
        confinement_root=str(root),
    )


class TestConfinementDecisions:
    """The pure-logic half: profile shape, path verdicts, context plumbing."""

    def test_the_profile_denies_by_default_and_allowlists_the_root(self, tmp_path: Path) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        profile = ToolConfinement.at(root).profile()
        assert "(deny default)" in profile
        # Read+write of the root, write of the null devices, and the read
        # machinery -- and the SAME root string the path checks compare
        # against (resolution is the whole reason `at` resolves).
        assert f'(subpath "{root}")' in profile
        assert "(allow file-write* (subpath " in profile
        assert '(allow file-read-metadata (subpath "/"))' in profile
        assert '(literal "/dev/null")' in profile
        # The network line: no allowance survives, and the explicit deny is
        # present for the reader. Measured while building this: a network
        # ALLOW defeats a network deny wherever either line sits (both orders
        # reached a live loopback listener; only the allow-free profile
        # refused), so the correctness is the absent allow and the deny is
        # documentation -- both are pinned so a future edit that reintroduces
        # the allow (however spelled) fails here before it fails a probe run.
        assert "network-outbound" not in profile
        assert "(deny network*)" in profile

    def test_a_relative_root_resolves_before_it_is_compared(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        monkeypatch.chdir(tmp_path)
        confinement = ToolConfinement.at("jail")
        assert confinement.root == root.resolve()
        assert confinement.contains(root / "inside.txt")

    def test_contains_is_a_resolved_prefix_test(self, tmp_path: Path) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        confinement = ToolConfinement.at(root)
        assert confinement.contains(root)
        assert confinement.contains(root / "deep" / "file.txt")
        assert not confinement.contains(tmp_path / "outside.txt")
        # A sibling whose name shares the prefix is NOT inside (relative_to,
        # not string prefix): the classic /jail-evil mistake.
        assert not confinement.contains(tmp_path / "jail-evil" / "file.txt")

    def test_path_denial_refuses_outside_and_unresolvable(self, tmp_path: Path) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        confinement = ToolConfinement.at(root)
        assert confinement.path_denial(root / "a.txt") is None
        outside = confinement.path_denial(tmp_path / "outside.txt")
        assert outside is not None and "confined" in outside
        unresolvable = confinement.path_denial(tmp_path / "x", resolvable=False)
        assert unresolvable is not None and "confined" in unresolvable

    def test_cwd_denial_refuses_a_forced_working_directory(self, tmp_path: Path) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        confinement = ToolConfinement.at(root)
        assert confinement.cwd_denial(str(root)) is None
        assert confinement.cwd_denial(None) is None
        denial = confinement.cwd_denial(str(tmp_path))
        assert denial is not None and "confined" in denial

    def test_from_context_reads_the_declared_field_and_tolerates_fakes(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        assert confinement_of(None) is None
        assert confinement_of(ToolContext()) is None
        # A duck-typed context without the field reads as unconfined, exactly
        # like the bash tool's delegation marker does.
        assert confinement_of(object()) is None  # type: ignore[arg-type]
        confining = confinement_of(_context(root))
        assert confining is not None and confining.root == root.resolve()

    def test_wrap_needs_a_mechanism_and_says_why_when_there_is_none(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        confinement = ToolConfinement.at(root)
        if sys.platform == "darwin":
            wrapped = confinement.wrap(["echo", "hi"])
            assert wrapped is not None
            assert wrapped[:3] == [
                str(confinement_module.SANDBOX_EXEC),
                "-p",
                confinement.profile(),
            ]
            assert wrapped[3:] == ["echo", "hi"]
        # No mechanism: wrap refuses and the refusal names the reason.
        monkeypatch.setattr(confinement_module, "SANDBOX_EXEC", tmp_path / "no-sandbox-exec")
        assert confinement.wrap(["echo", "hi"]) is None
        refusal = confinement.spawn_refusal()
        assert "refuses to run commands it cannot confine" in refusal
        assert str(root) in refusal

    def test_temp_dir_lives_inside_the_root(self, tmp_path: Path) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        temp = ToolConfinement.at(root).temp_dir()
        assert temp == root / "tmp" and temp.is_dir()


class TestShellBoundary:
    """The real bash tool against a real seatbelt profile (macOS only)."""

    @DARWIN_ONLY
    @pytest.mark.asyncio
    async def test_in_scratch_commands_still_run(self, tmp_path: Path) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        result = await builtin.execute_bash(
            "bash-in-jail",
            {"command": "echo hello > f.txt && cat f.txt"},
            AbortSignal(),
            None,
            _context(root),
        )
        assert not result.is_error, result.text
        assert "hello" in result.text
        assert (root / "f.txt").read_text().strip() == "hello"

    @DARWIN_ONLY
    @pytest.mark.asyncio
    async def test_the_operator_tree_is_unreachable(self, tmp_path: Path) -> None:
        """The measured leak shape, generalized: walks outside the root fail.

        The target is ``/Users`` because it is the macOS home tree the leak
        walked -- and because a pytest run may have HOME redirected, so the
        test may not lean on ``Path.home()`` to name it.
        """
        root = tmp_path / "jail"
        root.mkdir()
        result = await builtin.execute_bash(
            "bash-users",
            {"command": "find /Users -maxdepth 4 -name '*.pdf' 2>&1 | head -5"},
            AbortSignal(),
            None,
            _context(root),
        )
        assert "Operation not permitted" in result.text
        assert "/Users/" not in result.text.split("Operation not permitted")[0]

    @DARWIN_ONLY
    @pytest.mark.asyncio
    async def test_a_write_outside_the_root_is_denied_and_leaves_nothing(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        # /private/tmp is outside the allowlist; HOME may be redirected under
        # pytest, and /private/var/folders (where that frozen temp lives) is
        # READABLE by design -- the write deny is what this test must hit.
        leak = Path("/private/tmp") / "lo-confinement-probe.txt"
        assert not leak.exists()
        result = await builtin.execute_bash(
            "bash-write-out",
            {"command": f"echo probe > {leak}"},
            AbortSignal(),
            None,
            _context(root),
        )
        assert "Operation not permitted" in result.text
        assert not leak.exists()

    @DARWIN_ONLY
    @pytest.mark.asyncio
    async def test_a_symlink_inside_the_root_cannot_escape_it(self, tmp_path: Path) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        # A link to a location the allowlist does NOT cover: the kernel checks
        # the RESOLVED vnode, so the symlink is the same case as the
        # absolute path (a link to an allowed path would resolve to an allowed
        # path -- also correct, and not what this test is about).
        (root / "link-out").symlink_to("/Users")
        result = await builtin.execute_bash(
            "bash-symlink",
            {"command": "ls link-out 2>&1 | head -2"},
            AbortSignal(),
            None,
            _context(root),
        )
        assert "Operation not permitted" in result.text

    @pytest.mark.asyncio
    async def test_a_host_without_a_mechanism_refuses_to_run_the_command(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Fail closed: no sandbox means no shell, not an unwrapped shell."""
        root = tmp_path / "jail"
        root.mkdir()
        marker = root / "must-not-exist.txt"
        monkeypatch.setattr(confinement_module, "SANDBOX_EXEC", tmp_path / "no-sandbox-exec")
        result = await builtin.execute_bash(
            "bash-no-sandbox",
            {"command": f"touch {marker}"},
            AbortSignal(),
            None,
            _context(root),
        )
        assert result.is_error
        assert "cannot enforce" in result.text
        assert not marker.exists()

    @pytest.mark.asyncio
    async def test_an_unconfined_session_is_untouched(self, tmp_path: Path) -> None:
        """The default: no confinement on the context, no wrapper on the spawn."""
        result = await builtin.execute_bash(
            "bash-free",
            {"command": "echo free"},
            AbortSignal(),
            None,
            ToolContext(cwd=str(tmp_path), session_id="control"),
        )
        assert not result.is_error
        assert "free" in result.text

    @DARWIN_ONLY
    @pytest.mark.asyncio
    async def test_a_jailed_shell_cannot_reach_the_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The network completion, proved against the real kernel sandbox.

        The leak this closes was recorded on a real episode run: the model's
        shell (and the in-process ``web_search``/``web_fetch`` tools) reached
        the campaign host's network. This test refuses to take the fix on
        faith -- a LOCAL listener stands in for "the host's network" (so the
        test needs no internet and cannot flake on one), and the SAME command
        is then run with main's profile patched back in (one line: the
        ``allow network-outbound`` this branch replaced). The control leg is
        what makes the test discriminate: if the product profile ever regains
        a network allowance, the refused leg returns HTTP=200 and fails here.
        """

        class AliveHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib spelling
                body = b"host-service-alive"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        root = tmp_path / "jail"
        root.mkdir()
        context = _context(root)
        server = ThreadingHTTPServer(("127.0.0.1", 0), AliveHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            curl = (
                f"curl -sS -m 5 -o /dev/null -w 'HTTP=%{{http_code}}' "
                f"http://127.0.0.1:{server.server_address[1]}/ ; true"
            )
            refused = await builtin.execute_bash(
                "net-jailed", {"command": curl}, AbortSignal(), None, context
            )
            assert "HTTP=000" in refused.text, refused.text
            assert "HTTP=200" not in refused.text, refused.text

            # The pre-fix control: main's profile, one string swap, same tool
            # call path -- the listener IS reachable when only that line
            # differs, so the leg above is testing the line rather than the
            # listener.
            real_profile = ToolConfinement.profile

            def pre_fix(self: ToolConfinement) -> str:
                return real_profile(self).replace("(deny network*)", "(allow network-outbound)")

            monkeypatch.setattr(ToolConfinement, "profile", pre_fix)
            allowed = await builtin.execute_bash(
                "net-prefix-control", {"command": curl}, AbortSignal(), None, context
            )
            assert "HTTP=200" in allowed.text, allowed.text
            # Undo before the python legs: they must run under THIS branch's
            # profile, not the control's (monkeypatch would restore only at
            # test end).
            monkeypatch.undo()
        finally:
            server.shutdown()

        # The python shapes, which is what the recorded run's `bash` calls
        # would also have hit: a TCP connect fails with EPERM (even though the
        # seatbelt profile is not a firewall -- the syscall itself is denied),
        # and name resolution fails FAST rather than hanging on a resolver.
        connect = (
            'python3 -c "import socket\n'
            "s=socket.socket();s.settimeout(3)\n"
            "try:\n s.connect(('1.1.1.1',443));print('CONNECTED')\n"
            "except Exception as e:print('FAILED',type(e).__name__,str(e)[:80])\""
        )
        result = await builtin.execute_bash(
            "net-connect", {"command": connect}, AbortSignal(), None, context
        )
        assert "FAILED PermissionError" in result.text, result.text
        assert "Operation not permitted" in result.text, result.text

        resolve = (
            'python3 -c "import socket\n'
            "try:\n print('RESOLVED',socket.getaddrinfo('example.com',443)[0][4])\n"
            "except Exception as e:print('DNS-FAILED',type(e).__name__,str(e)[:80])\""
        )
        result = await builtin.execute_bash(
            "net-dns", {"command": resolve}, AbortSignal(), None, context
        )
        assert "DNS-FAILED gaierror" in result.text, result.text


class TestPathTools:
    """The file tools' refusal, which is pure logic and runs everywhere."""

    @pytest.mark.asyncio
    async def test_read_refuses_a_path_outside_the_root(self, tmp_path: Path) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        inside = root / "notes.txt"
        inside.write_text("inside")
        outside = tmp_path / "leak.txt"
        outside.write_text("TOPSECRET")
        context = _context(root)

        ok = await builtin.execute_read("read-in", {"path": str(inside)}, None, None, context)
        assert not ok.is_error, ok.text
        assert "inside" in ok.text

        refused = await builtin.execute_read(
            "read-out", {"path": str(outside)}, None, None, context
        )
        assert refused.is_error
        assert "confined" in refused.text
        assert "TOPSECRET" not in refused.text

    @pytest.mark.asyncio
    async def test_write_refuses_a_path_outside_the_root(self, tmp_path: Path) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        outside = tmp_path / "leak.txt"
        context = _context(root)

        refused = await builtin.execute_write(
            "write-out",
            {"path": str(outside), "content": "leak"},
            None,
            None,
            context,
        )
        assert refused.is_error
        assert "confined" in refused.text
        assert not outside.exists()

        ok = await builtin.execute_write(
            "write-in", {"path": str(root / "ok.txt"), "content": "fine"}, None, None, context
        )
        assert not ok.is_error, ok.text
        assert (root / "ok.txt").read_text() == "fine"

    @pytest.mark.asyncio
    async def test_grep_refuses_a_path_outside_the_root(self, tmp_path: Path) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")
        context = _context(root)
        refused = await builtin.execute_grep(
            "grep-out", {"pattern": "secret", "path": str(outside)}, None, None, context
        )
        assert refused.is_error
        assert "confined" in refused.text

    @pytest.mark.asyncio
    async def test_edit_refuses_a_path_outside_the_root(self, tmp_path: Path) -> None:
        root = tmp_path / "jail"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("old")
        context = _context(root)
        refused = await builtin.execute_edit(
            "edit-out",
            {"path": str(outside), "old_text": "old", "new_text": "new"},
            None,
            None,
            context,
        )
        assert refused.is_error
        assert "confined" in refused.text
        assert outside.read_text() == "old"

    @pytest.mark.asyncio
    async def test_glob_denies_a_symlinked_tree_outside_the_root(self, tmp_path: Path) -> None:
        """The review round 1 blocker, pinned (R-1).

        ``bash`` can create a symlink INSIDE the root (a legal write). Before
        this fix ``glob "linkapp/*"`` listed the outside tree's top
        directories and ``linkapp/**/*`` returned up to the 500-match cap of
        its contents -- the one path reader without the resolution rule its
        siblings (read/write/edit/grep/the shell) already had. This test pins
        both the exact repro shapes and the controls that keep ordinary work
        intact.
        """

        root = tmp_path / "jail"
        root.mkdir()
        outside = tmp_path / "outside-tree"
        (outside / "assets" / "deep" / "a" / "b").mkdir(parents=True)
        (outside / "assets" / "deep" / "a" / "b" / "leaf.txt").write_text("x")
        (outside / "assets" / "secret.txt").write_text("TOPSECRET")
        (outside / "manifests").mkdir()
        (root / "inner" / "deep").mkdir(parents=True)
        (root / "inner" / "deep" / "ok.txt").write_text("x")
        (root / "linkapp").symlink_to(outside)
        (root / "linkin").symlink_to(root / "inner")

        context = _context(root)
        for pattern in ("linkapp/*", "linkapp/**/*", "linkapp/**/secret.txt", "*.txt"):
            result = await builtin.execute_glob(
                "glob-out", {"pattern": pattern}, None, None, context
            )
            assert not result.is_error, result.text
            assert "No paths matched" in result.text, (pattern, result.text)
            assert "assets" not in result.text and "TOPSECRET" not in result.text

        # An in-root symlink resolves INSIDE and keeps working: the rule is
        # resolution, not a blanket ban on links.
        inside_link = await builtin.execute_glob(
            "glob-in-link", {"pattern": "linkin/**/*"}, None, None, context
        )
        assert "linkin/deep/ok.txt" in inside_link.text
        # The ordinary in-root control.
        control = await builtin.execute_glob(
            "glob-in", {"pattern": "inner/**/*"}, None, None, context
        )
        assert "inner/deep/ok.txt" in control.text
        # The free-session default is unchanged: no confinement, no filter.
        free = ToolContext(cwd=str(root), session_id="free")
        unconfined = await builtin.execute_glob(
            "glob-free", {"pattern": "linkapp/*"}, None, None, free
        )
        assert "assets" in unconfined.text

    @pytest.mark.asyncio
    async def test_every_path_taking_tool_denies_outside_the_root(self, tmp_path: Path) -> None:
        """The completeness table the round-1 standard asks for.

        The blocker was ONE reader missing the rule; this table is the class
        statement, so a future filesystem-reaching tool added without
        ``_confinement_denial`` fails here rather than in a probe run. Each
        case exercises the real executor against a target outside the root --
        directly and, where the tool resolves a path, through a symlink that
        resolves outside. ``bash`` is enforced at the kernel (darwin-only
        tests above), ``eval``/``lsp`` refuse outright (test below), and the
        browser tool's two write-destination sites share the same helper.
        """

        root = tmp_path / "jail"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("TOPSECRET")
        (root / "link-out").symlink_to(outside / "secret.txt")
        context = _context(root)

        cases = [
            ("read", builtin.execute_read, {"path": str(outside / "secret.txt")}),
            ("read symlink", builtin.execute_read, {"path": str(root / "link-out")}),
            (
                "write",
                builtin.execute_write,
                {"path": str(outside / "new.txt"), "content": "x"},
            ),
            (
                "edit",
                builtin.execute_edit,
                {"path": str(outside / "secret.txt"), "old_text": "TOPSECRET", "new_text": "x"},
            ),
            ("grep", builtin.execute_grep, {"pattern": "TOPSECRET", "path": str(outside)}),
            ("glob symlinked dir", builtin.execute_glob, {"pattern": "link-out/*"}),
        ]
        for name, execute, args in cases:
            result = await execute(f"{name}-case", args, None, None, context)
            assert result.is_error or "No paths matched" in result.text, (name, result.text)
            assert "confined" in result.text or "No paths matched" in result.text, (
                name,
                result.text,
            )
            assert "TOPSECRET" not in result.text, name
        assert not (outside / "new.txt").exists()
        assert (outside / "secret.txt").read_text() == "TOPSECRET"

    @pytest.mark.asyncio
    async def test_eval_and_lsp_refuse_when_confined(self, tmp_path: Path) -> None:
        """The two tools whose reach cannot be vouched for are refused outright."""
        root = tmp_path / "jail"
        root.mkdir()
        context = _context(root)

        from local_operator.tools import eval as eval_module

        eval_refusal = await eval_module.execute_eval(
            "eval-confined", {"code": "print(1)"}, None, None, context
        )
        assert eval_refusal.is_error
        assert "confined" in eval_refusal.text

        from local_operator.tools import lsp as lsp_module

        lsp_refusal = await lsp_module.execute_lsp(
            "lsp-confined",
            {"action": "symbols", "path": str(root / "file.py")},
            None,
            None,
            context,
        )
        assert lsp_refusal.is_error
        assert "confined" in lsp_refusal.text

    @pytest.mark.asyncio
    async def test_the_in_process_network_tools_refuse_when_confined(self, tmp_path: Path) -> None:
        """The five host-network spellings, each refused with the ONE sentence.

        These tools are in-process clients -- there is no child for the kernel
        to hold -- so the boundary is a refusal, and the assertion is the
        EXACT text (``network_refusal``), because the refusal a trajectory
        shows is the deferred finding the session-arm write-up carries: two
        spellings of the same reach must not produce two sentences. Each
        executor is exercised at its real entry point, and every one of them
        must answer BEFORE any config, service or host probe runs (the
        refusals are placed ahead of those, so this test is offline).
        """

        from local_operator.web_fetch.tool import execute_web_fetch
        from local_operator.web_search.read_tool import execute_web_read
        from local_operator.web_search.tool import execute_web_search

        root = tmp_path / "jail"
        root.mkdir()
        context = _context(root)
        confinement = ToolConfinement.at(root)

        cases = [
            ("web_search", execute_web_search, {"query": "NeurIPS best paper awards"}),
            (
                "web_fetch",
                execute_web_fetch,
                {"url": "https://neurips.cc/virtual/awards"},
            ),
            ("read", builtin.execute_read, {"path": "https://icml.cc/virtual/awards"}),
            ("web_read", execute_web_read, {"question": "who won the award"}),
            ("browser", builtin.execute_browser, {"action": "open", "url": "https://iclr.cc"}),
        ]
        for tool_name, execute, args in cases:
            result = await execute(f"refuse-{tool_name}", args, None, None, context)
            assert result.is_error, (tool_name, result.text)
            assert result.text == confinement.network_refusal(tool_name), (tool_name, result.text)

        # The same sentences name the confinement root, and the refusal is not
        # something a free session can reach: that default is pinned by
        # test_an_unconfined_context_is_untouched below.
        assert str(root.resolve()) in confinement.network_refusal("web_fetch")

    @pytest.mark.asyncio
    async def test_scratchpad_scheme_still_works_inside_a_confined_session(
        self, tmp_path: Path
    ) -> None:
        """The session's own scratchpad:// is inside the root by construction."""
        root = tmp_path / "jail"
        scratch = root / "scratchpad"
        scratch.mkdir(parents=True)
        context = ToolContext(
            cwd=str(root),
            session_id="confinement-test",
            confinement_root=str(root),
            scratchpad_dir=str(scratch),
        )
        written = await builtin.execute_write(
            "write-scratch",
            {"path": "scratchpad://note.txt", "content": "scratch"},
            None,
            None,
            context,
        )
        assert not written.is_error, written.text
        assert (scratch / "note.txt").read_text() == "scratch"

    def test_an_unconfined_context_is_untouched(self) -> None:
        """No field on the context: no refusal, no behaviour change at all."""
        assert confinement_of(ToolContext()) is None
